# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from contextlib import nullcontext
from functools import partial
from megatron.core import parallel_state

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms

from cosmos_predict1.utils.lazy_config import instantiate
from cosmos_predict1.utils import distributed, log, misc
from cosmos_predict1.utils.easy_io import easy_io
from cosmos_predict1.utils.visualize.video import save_img_or_video

from cosmos_predict1.diffusion.training.datasets.lidar_dataset.configs import COMMONDATA_CONFIG
from cosmos_predict1.diffusion.inference.inference_utils import load_model_by_config, load_network_model_and_cast
from cosmos_predict1.diffusion.training.models.imagetolidar_model import ImageToLidarModel
from cosmos_predict1.diffusion.training.module.pretrained_vae import VideoTokenizerInterface


from cosmos_predict1.utils.visualize.point_cloud import render_range_map_to_point_cloud
from cosmos_predict1.utils.lidar_rangemap import undo_row_col_temporal_repeat, unnormalize_and_reduce_channels, range_map_to_point_cloud, colorcode_depth_maps, project_lidar_to_rgb_impl, apply_motion_compensation_impl

def merge_images_vertically(image_paths, output_path):
    """
    Merge multiple JPG images vertically into one image.
    
    Args:
        image_paths (list): List of paths to JPG images
        output_path (str): Path where the merged image will be saved
    
    Returns:
        str: Path to the saved merged image
    """
    # Open all images
    images = [Image.open(path) for path in image_paths]
    
    # Convert all images to RGB mode if they aren't already
    images = [img.convert('RGB') for img in images]
    
    # Get the widths of all images
    widths = [img.width for img in images]
    
    # Find the maximum width
    max_width = max(widths)
    
    # Resize all images to have the same width while maintaining aspect ratio
    def resize_image(img, target_width):
        ratio = target_width / img.width
        new_height = int(img.height * ratio)
        return img.resize((target_width, new_height), Image.Resampling.LANCZOS)
    
    resized_images = [resize_image(img, max_width) for img in images]
    
    # Calculate total height
    total_height = sum(img.height for img in resized_images)
    
    # Create a new image with the max width and combined height
    merged_image = Image.new('RGB', (max_width, total_height))
    
    # Paste the images vertically
    y_offset = 0
    for img in resized_images:
        merged_image.paste(img, (0, y_offset))
        y_offset += img.height
    
    # Save the merged image
    merged_image.save(output_path)
    
    for img in image_paths:
        os.remove(img)
    return output_path


_UINT8_MAX_F = float(np.iinfo(np.uint8).max)
torch.enable_grad(False)
misc.set_random_seed(0)  # always use the same random seed


def render_range_image(range_map_input, range_map_recon):
    # range_map_input: [N, H, W]
    # range_map_recon: [N, H, W]

    device = range_map_input.device
    vis = render_range_map_to_point_cloud(
        range_map_input.float().cpu().numpy(), range_map_recon.float().cpu().numpy(), filter_outlier=True
    )  # shape: (bs, H, W, 3)

    vis_images = torch.from_numpy(vis).permute(0, 3, 1, 2).to(device)  # shape: (bs, 3, H, W)

    image_grid = torchvision.utils.make_grid(vis_images.clamp_(0, 1), nrow=1, pad_value=0, normalize=False)
    image_grid = image_grid.float().permute(1, 2, 0).cpu().detach().numpy()
    pil_image = Image.fromarray((image_grid * _UINT8_MAX_F).astype("uint8"))
    return pil_image


def colorcode_data(results, dataset_config, cmap="Spectral"):
    """
    Input: N,H, W
    Output: N, 3, H x W, [0, 1]
    """

    # colorcode the depth map
    color_depth = colorcode_depth_maps(
        results, near=np.log(dataset_config["min_range"]), far=np.log(dataset_config["max_range"]), cmap=cmap
    )  # N, 3, H, W

    return color_depth


class RGB2LidarInference:
    def __init__(self, args, config):
        self.args = args
        self.config = config

        rgb_resize_size = [720, 1280]
        rgb_crop_size = [704, 1280]
        self.img_transform = transforms.Compose(
            [
                transforms.Resize(
                    rgb_resize_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
                ),
                transforms.CenterCrop(rgb_crop_size),
            ]
        )

        self.skip_tensor_name = ["video", "lidar_hdmap", "timestamp_list", "pose_sensor_start_end"]
        self.n_views = args.n_views

        for view_idx in range(self.n_views):
            self.skip_tensor_name.append(f"camera_intrinsics_{view_idx}")
            self.skip_tensor_name.append(f"camera_c2lidar_transform_{view_idx}")

        self.precision_type = torch.bfloat16

    def clear_cuda_cache(self):
        """Clear CUDA cache to free up memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if distributed.is_initialized():
                distributed.barrier()


    def convert_dtype(self, data_batch):
        device = "cuda"
        is_cpu = (isinstance(device, str) and device == "cpu") or (
            isinstance(device, torch.device) and device.type == "cpu"
        )
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
                if k not in self.skip_tensor_name:
                    v = v.to(
                        device=device,
                        dtype=self.precision_type,
                        non_blocking=(not is_cpu),
                    )
                else:
                    v = v.to(device=device, non_blocking=(not is_cpu))
            elif isinstance(v, torch.Tensor):
                v = v.to(device=device, non_blocking=(not is_cpu))
            else:
                v = v
            data_batch[k] = v

        return data_batch


    def setup_environment(self):
        """Initialize environment and distributed settings"""
        self.rank = distributed.get_rank()
        from megatron.core import parallel_state
        
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=self.args.context_parallel_size)
        
        self.process_group = parallel_state.get_context_parallel_group()
        torch.manual_seed(0)

        self.data_parallel_id = parallel_state.get_data_parallel_rank()

    def load_model(self):
        """Load model and VAE"""
        log.info(f"Loading model from {self.args.checkpoint}")
        self.model = load_model_by_config(
            config_job_name=self.args.experiment,
            config_file=self.args.config,
            model_class=ImageToLidarModel,
        )
        load_network_model_and_cast(self.model, self.args.checkpoint)
        
        vae: VideoTokenizerInterface = instantiate(self.config.model.vae).cuda()
        self.model.vae = vae

        if self.args.context_parallel_size > 1:
            self.model.net.enable_context_parallel(self.process_group)

    def get_context(self):
        """Get model context"""
        return nullcontext

    def parse_data_batch_for_projection(self, data_batch, rgb_view_idx, batch_index=0):
        timestamp_list = data_batch["timestamp_list"][batch_index]  # t, 2
        camera_intrinsics = data_batch[f"camera_intrinsics_{rgb_view_idx}"][batch_index]  # 11
        camera_2_lidar = data_batch[f"camera_c2lidar_transform_{rgb_view_idx}"][batch_index]  # t, 4, 4
        t_sensor_start_sensor_end = data_batch["pose_sensor_start_end"][batch_index]  # t, 4, 4
        return timestamp_list, camera_intrinsics, camera_2_lidar, t_sensor_start_sensor_end
    
    
    def get_motion_compensated_points(self, data_batch,lidar_to_show, dset_name, batch_index=0, frame_index=0, n_cols=1800, column_shift=4):
        dataset_config = COMMONDATA_CONFIG[dset_name]
        repeat_row = dataset_config["repeat_row"]
        repeat_temporal = dataset_config["repeat_temporal"]
        repeat_col = dataset_config["repeat_col"]
        lidar_to_show = undo_row_col_temporal_repeat(lidar_to_show, repeat_row, repeat_col, repeat_temporal)  # [b, c, t, h, w]
        lidar_to_show, _ = unnormalize_and_reduce_channels(
            lidar_to_show,
            dataset_config["max_range"],
            dataset_config["min_range"],
            dataset_config["min_value"],
            dataset_config["inverse_depth"],
            dataset_config["to_three_channels"],
            self.args.near_buffer,
            self.args.far_buffer,
            dataset_config["inv_depth_threshold"],
        )  # b, t, h, w

        H = lidar_to_show.shape[2]

        range_map_input = lidar_to_show[:, :, : H // 2, :]
        range_map_recon = lidar_to_show[:, :, H // 2 :, :]

        range_map_input = range_map_input[batch_index][frame_index].numpy()  # [h,w]
        range_map_recon = range_map_recon[batch_index][frame_index].numpy()  # [h,w]
        input_points = range_map_to_point_cloud(range_map_input)  # N, 3
        recon_points = range_map_to_point_cloud(range_map_recon)  # N, 3
        timestamp_list = data_batch["timestamp_list"][batch_index] # t, 2
        t_sensor_start_sensor_end = data_batch["pose_sensor_start_end"][batch_index] # t, 4, 4
        
        timestamp_start = timestamp_list[frame_index][0]
        timestamp_end = timestamp_list[frame_index][1]
        timestamps_startend_us = [timestamp_start, timestamp_end]
        timestamps = np.linspace(timestamp_start, timestamp_end, n_cols)
        
        input_column_indices = np.where(range_map_input > 0)[1]  # this is from 0 - 1791, center cropped, we need to shift it to 4 -> 1795
        input_timestamp_us = timestamps[input_column_indices + column_shift]
        recon_column_indices = np.where(range_map_recon > 0)[1]
        recon_timestamp_us = timestamps[recon_column_indices + column_shift]
            
        input_points_compensated = apply_motion_compensation_impl(input_points, t_sensor_start_sensor_end[frame_index], timestamps_startend_us, input_timestamp_us)
        recon_points_compensated = apply_motion_compensation_impl(recon_points, t_sensor_start_sensor_end[frame_index], timestamps_startend_us, recon_timestamp_us)
        
        return input_points_compensated, recon_points_compensated
        

    def project_lidar_to_rgb(
        self, data_batch, save_folder, dset_name, base_fp_wo_ext, lidar_to_show, rgb_to_show, batch_index=0, frame_index=0
    ):
        input_points_compensated, recon_points_compensated = self.get_motion_compensated_points(data_batch, lidar_to_show, dset_name, batch_index, frame_index)
        overlaid_rgb_list = []
        rgb_to_show = torch.clamp(rgb_to_show + 1, 0, 2) / 2  # B, 3, V*T, H, W
        for rgb_view_idx in [1, 0, 2]:
            _, camera_intrinsics, camera_2_lidar, _ = self.parse_data_batch_for_projection(data_batch, rgb_view_idx)
            visual_lidar_input, visual_lidar_recon = project_lidar_to_rgb_impl(camera_intrinsics, camera_2_lidar[frame_index], input_points_compensated, recon_points_compensated)

            # apply the same image transformation
            visual_lidar = np.stack([visual_lidar_input, visual_lidar_recon], axis=0)  # 2, H, W, 3
            visual_lidar = visual_lidar.transpose(0, 3, 1, 2)  # N, 3, H, W
            visual_lidar = self.img_transform(torch.from_numpy(visual_lidar)).numpy()[0]  # N, 3, H, W

            # overlap rgb
            rgb = rgb_to_show[frame_index, :, rgb_view_idx].cpu().numpy()  # 3, H, W
            rgbs = rgb

            overlaid_rgbs = (visual_lidar * 0.6 + rgbs).clip(0, 1)  # 3, H, W
            overlaid_rgbs = overlaid_rgbs.transpose(1, 2, 0)  # H, W, 3
            overlaid_rgb_list.append(overlaid_rgbs)

        overlaid_rgb = np.concatenate(overlaid_rgb_list, axis=1)  # H, W*n_views, 3

        clip_name = data_batch["__key__"][batch_index]
        frame_indice = data_batch['rgb_frame_indices'][batch_index][frame_index]
        base_fp_wo_ext = f"{base_fp_wo_ext}_{clip_name}_{frame_indice}"

        save_path = f"{save_folder}/{base_fp_wo_ext}_overlay.jpg"
        media = Image.fromarray((overlaid_rgb * 255).astype(np.uint8))
        media.save(save_path)

        return save_path

    def save_vis(
        self, data_batch, to_show, save_folder, base_fp_wo_ext, dset_name, save_video_fps=10, data_type="lidar", batch_index=0, frame_index=0
    ):
        """
        Save visualization of lidar data
        to_show: b,c,t,h,w, [-1, 1]
        """
        dataset_config = COMMONDATA_CONFIG[dset_name]
        repeat_row = dataset_config["repeat_row"]
        repeat_temporal = dataset_config["repeat_temporal"]
        repeat_col = dataset_config["repeat_col"]

        clip_name = data_batch["__key__"][batch_index]
        frame_indice = data_batch['rgb_frame_indices'][batch_index][frame_index]

        file_base_fp = f"{base_fp_wo_ext}_{clip_name}_{frame_indice}.jpg"
        local_path = f"{save_folder}/{file_base_fp}"

        save_list = []        
        # remove row, col and temporal repeat
        if data_type == "lidar":
            to_show = undo_row_col_temporal_repeat(to_show, repeat_row, repeat_col, repeat_temporal)  # [b, c, t, h, w]
            to_show, _ = unnormalize_and_reduce_channels(
                to_show,
                dataset_config["max_range"],
                dataset_config["min_range"],
                dataset_config["min_value"],
                dataset_config["inverse_depth"],
                dataset_config["to_three_channels"],
                self.args.near_buffer,
                self.args.far_buffer,
                dataset_config["inv_depth_threshold"],
            )  # b, t, h, w

        ###################################
        # save the video
        vid_save_path = local_path
        if data_type == "lidar":
            to_show_grid = colorcode_data(to_show[batch_index], dataset_config)  # t, 3, h, w
            to_show_grid = to_show_grid.permute(1, 0, 2, 3)  # [c, t, Hx2, W]
        else:
            to_show = (to_show + 1).clamp(0, 2) / 2
            to_show_grid = rearrange(to_show, "b c t h w -> c t h (b w)")  # [c, t, Hx2, W]

        save_img_or_video(
            to_show_grid,
            vid_save_path.split(".jpg")[0],  # remove .mp4
            fps=save_video_fps,
        )
        log.info(f"save video to {vid_save_path}")
        save_list.append(vid_save_path)

        ###################################
        # sample the middle frame for visualization
        _T = to_show.shape[1]
        if _T > 1:
            to_show = to_show[:, [0, _T // 2, _T - 1]]  # [b, 3, h, w]

        if data_type == "lidar":
            H = to_show.shape[2]

            range_map_input = to_show[:, :, : H // 2, :]
            range_map_recon = to_show[:, :, H // 2 :, :]

            range_map_input = range_map_input[batch_index]  # [t,h,w]
            range_map_recon = range_map_recon[batch_index]  # [t,h,w]

            pil_media = render_range_image(range_map_input, range_map_recon)
            pil_media.save(local_path.replace(".jpg", "_pointcloud.jpg"))
            save_list.append(local_path.replace(".jpg", "_pointcloud.jpg"))

        return save_list
    
    def save_compensated_lidar(self, data_batch, dset_name, base_fp_wo_ext, save_folder, lidar_to_show, n_cols=1800, column_shift=4, batch_index=0, frame_index=0):
        input_points_compensated, recon_points_compensated = self.get_motion_compensated_points(data_batch, lidar_to_show, dset_name, batch_index, frame_index, n_cols, column_shift)
        # save them to numpy, with correct file name
        clip_name = data_batch["__key__"][batch_index]
        frame_indice = data_batch['rgb_frame_indices'][batch_index][frame_index]
        save_path = os.path.join(save_folder, f"{base_fp_wo_ext}_{clip_name}_{frame_indice}")
        np.savez(save_path, input_points_compensated=input_points_compensated, recon_points_compensated=recon_points_compensated)

    def sample(self, dset_name, save_folder, data_batch, c_iter_idx):
        """Generate samples from the model"""

        raw_data, x0, condition = self.model.get_data_and_condition(data_batch)

        seed = 1 if self.args.guidance > 0 else np.random.randint(0, 2**30 - 1)
        sample = self.model.generate_samples_from_batch(
            data_batch,
            guidance=self.args.guidance,
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
            is_negative_prompt=False,
            num_steps=self.args.ddim_iters,
            sigma_min=self.args.sigma_min,
            seed=seed,
            condition_latent=x0
        )

        lidar_latent_gen, rgb_latent_gen = self.model.split_latent_state(sample)
        lidar_latent_input, rgb_latent_input = self.model.split_latent_state(x0)

        if hasattr(self.model, "decode"):
            lidar_gen = self.model.vae.decode_lidar(lidar_latent_gen)  # [b, c, t, h, w]
            lidar_input = self.model.vae.decode_lidar(lidar_latent_input)

            rgb_latent_gen = rearrange(rgb_latent_gen, "b c (t v) h w -> (b v) c t h w", v=self.n_views)
            rgb_gen = self.model.vae.decode_image(rgb_latent_gen)
            rgb_gen = rearrange(rgb_gen, "(b v) c t h w -> b c (t v) h w", v=self.n_views)


            rgb_latent_input = rearrange(rgb_latent_input, "b c (t v) h w -> (b v) c t h w", v=self.n_views)
            rgb_input = self.model.vae.decode_image(rgb_latent_input)
            rgb_input = rearrange(rgb_input, "(b v) c t h w -> b c (t v) h w", v=self.n_views)

            lidar_to_show = torch.cat([lidar_input, lidar_gen], dim=-2).float().cpu()  # [b, c, t, h, w]

        base_fp_wo_ext = f"DataParallelID{self.data_parallel_id:04d}_Sample_Iter{c_iter_idx:09d}"

        # render lidar image
        lidar_vis = self.save_vis(data_batch, lidar_to_show, save_folder, base_fp_wo_ext+"_lidar", dset_name, False, data_type="lidar")

        # project lidar to rgb
        overlay_path = self.project_lidar_to_rgb(data_batch, save_folder, dset_name, base_fp_wo_ext, lidar_to_show, rgb_gen.float(), batch_index=0)
        input_files = [overlay_path, lidar_vis[0], lidar_vis[1]]
        
        # save the point cloud as well
        self.save_compensated_lidar(data_batch, dset_name, base_fp_wo_ext, save_folder, lidar_to_show, batch_index=0, frame_index=0)

        batch_index = 0
        clip_name = data_batch["__key__"][batch_index]
        frame_indice = data_batch['rgb_frame_indices'][batch_index][0]
        base_fp_wo_ext = f"{base_fp_wo_ext}_{clip_name}_{frame_indice}"

        output_path = f"{save_folder}/{base_fp_wo_ext}.jpg"


        merge_images_vertically(input_files, output_path)


        distributed.barrier()
        return output_path

    def run_inference(self):
        self.setup_environment()
        self.load_model()
        """Run inference pipeline"""
        dataloader_train = instantiate(self.config.dataloader_train)
        dset_name = "images_to_lidar"
        os.makedirs(self.args.save_dir, exist_ok=True)

        with self.get_context()():
            for c_iter_idx, data_batch in enumerate(dataloader_train):
                log.info(f"n_iter: {c_iter_idx}")
                data_batch = self.convert_dtype(data_batch)

                self.sample(dset_name, self.args.save_dir, data_batch, c_iter_idx)

                del data_batch
                torch.cuda.empty_cache()
                
        # clean up properly
        if self.args.context_parallel_size > 1:
            parallel_state.destroy_model_parallel()
            import torch.distributed as dist
            dist.destroy_process_group()