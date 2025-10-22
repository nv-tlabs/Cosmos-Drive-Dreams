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


import argparse
import os
import tarfile
from glob import glob
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import torch

# import open3d as o3d
from tqdm import tqdm

from cosmos_predict1.tokenizer.inference.lidar_lib import LidarProcessor
from cosmos_predict1.utils.lidar_rangemap import load_range_map
from cosmos_predict1.utils.visualize.point_cloud import vis_point_cloud
from cosmos_predict1.utils.visualize.video import save_images_to_video, stack_videos_vertically_with_ffmpeg
from cosmos_predict1.utils.lidar_rangemap import save_depth_maps_to_video
from cosmos_predict1.utils.misc import natural_key

def render_each_lidar(args):
    frame_idx, original_pts, recon_pts, save_folder = args
    n_pts = original_pts.shape[0]
    original_colors = np.array([[1, 0.706, 0]] * n_pts, dtype=np.float32)
    recon_colors = np.array([[0, 0.651, 0.929]] * n_pts, dtype=np.float32)

    save_file = os.path.join(save_folder, f"original_{frame_idx}.png")
    vis_point_cloud(original_pts, original_colors, save_file)
    save_file = os.path.join(save_folder, f"recon_{frame_idx}.png")
    vis_point_cloud(recon_pts, recon_colors, save_file)


def _args_parser():
    parser = argparse.ArgumentParser(description="Process range images and create videos.")
    parser.add_argument("--sample_path", type=str, help="Path to the sample range image.")
    parser.add_argument("--enc_path", type=str, help="Path to the encoder checkpoint.")
    parser.add_argument("--dec_path", type=str, help="Path to the decoder checkpoint.")
    parser.add_argument("--output_folder", type=str, help="Path to the dump folder.")
    parser.add_argument("--vis_pcd", type=int, default=1, help="Visualize the point cloud.")
    parser.add_argument("--tokenizer_dtype", type=str, default="bfloat16", help="tokenizer dtype")
    parser.add_argument("--n_rows_repeat", type=int, default=4, help="Number of times to repeat each row in the image.")
    parser.add_argument(
        "--n_cols_repeat", type=int, default=1, help="Number of times to repeat each column in the image."
    )
    parser.add_argument("--max_frames", type=int, default=20, help="Number of frames to use.")
    parser.add_argument("--fps", type=int, default=10, help="FPS of the video.")
    parser.add_argument("--downsample_factor_row", type=int, default=1, help="Downsample factor.")
    parser.add_argument("--downsample_factor_col", type=int, default=2, help="Downsample factor.")
    parser.add_argument("--downsample_method", type=str, default="scatter_min", help="Downsample method.")
    parser.add_argument("--max_range", type=float, default=100, help="max range")
    parser.add_argument("--min_range", type=float, default=5, help="min range")
    parser.add_argument("--colormap", type=str, default="Spectral", help="Colormap to apply to depth maps.")
    args = parser.parse_args()
    return args


def eval_sample(args):
    dump_dir = os.path.join("dump_results/lidar_tokenizer", args.output_folder)
    os.makedirs(dump_dir, exist_ok=True)
  
    lidar_processor = LidarProcessor(
        checkpoint_enc=args.enc_path,
        checkpoint_dec=args.dec_path,
        n_rows_repeat=args.n_rows_repeat,
        n_cols_repeat=args.n_cols_repeat,
        downsample_factor_row=args.downsample_factor_row,
        downsample_factor_col=args.downsample_factor_col,
        downsample_method=args.downsample_method,
        max_range=args.max_range,
        min_range=args.min_range,
        dtype=args.tokenizer_dtype,
    )

    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(dump_dir + "/histogram", exist_ok=True)
    os.makedirs(dump_dir + "/range_map_video", exist_ok=True)
    os.makedirs(dump_dir + "/point_cloud", exist_ok=True)


    # load the range map
    scene_name = args.sample_path.split("/")[-1].split(".")[0]
    tar_handle = tarfile.open(args.sample_path, "r")
    range_map = load_range_map(tar_handle)  # N, H, W
    n_frame = range_map.shape[0] if args.max_frames < 0 else min(args.max_frames, range_map.shape[0])
    range_map = range_map[:n_frame]
    
    original_range, recon_range, valid_mask, ray_directions = lidar_processor.process_video(range_map)
    original_range = np.where(valid_mask, original_range, 0)
    recon_range = np.where(valid_mask, recon_range, 0)

    # save the original and reconstructed range maps to videos
    save_path = f"{dump_dir}/range_map_video/{scene_name}.mp4"
    depth_maps = np.concatenate([original_range, recon_range], axis=1)  # N x (2 x H) x W
    save_depth_maps_to_video(torch.from_numpy(depth_maps), save_path, cmap=args.colormap, fps=args.fps)

    # plot the errors
    range_diff = np.abs(original_range - recon_range)[valid_mask]
    rmse = np.sqrt(np.mean(range_diff ** 2))
    mae = np.mean(np.abs(range_diff))
    relative_error = np.mean(np.abs(range_diff) / (original_range[valid_mask] + 1e-6))
    print(f"RMSE: {rmse:.2f} m, MAE: {mae:.2f} m, Rel error: {relative_error:.2f}")

    plt.hist(range_diff, bins=100, range=(0, 10))
    plt.xlabel("Range Difference (m)")
    plt.ylabel("Frequency")
    plt.title(f"Range Difference Histogram\nRMSE: {rmse:.2f} m, MAE: {mae:.2f} m, Rel error: {relative_error:.2f}")
    plt.savefig(f"{dump_dir}/histogram/{scene_name}.png")
    plt.close()

    # ----------render the point cloud and save as a video----------
    if args.vis_pcd == 1:
        original_pcd = original_range[..., np.newaxis] * ray_directions  # N x H x W x 3
        recon_pcd = recon_range[..., np.newaxis] * ray_directions
        save_folder = f"{dump_dir}/point_cloud/{scene_name}"
        os.makedirs(save_folder, exist_ok=True)

        max_workers = 8
        with Pool(processes=max_workers) as pool:
            # Prepare arguments for each frame
            process_args = [
                (
                    frame_idx,
                    original_pcd[frame_idx][valid_mask[frame_idx]],
                    recon_pcd[frame_idx][valid_mask[frame_idx]],
                    save_folder,
                )
                for frame_idx in range(n_frame)
            ]

            # Process frames in parallel with progress bar
            list(tqdm(pool.imap(render_each_lidar, process_args), total=n_frame, desc="Processing frames"))

        original_video_path = f"{save_folder}/original.mp4"
        recon_video_path = f"{save_folder}/recon.mp4"

        # create the original video
        original_images = glob(os.path.join(save_folder, "original_*.png"))
        original_images.sort(key=natural_key)

        save_images_to_video(original_images, original_video_path, fps=args.fps)
        # delete the original images
        for image in original_images:
            os.remove(image)

        # create the reconstructed video
        recon_images = glob(os.path.join(save_folder, "recon_*.png"))
        recon_images.sort(key=natural_key)
        save_images_to_video(recon_images, recon_video_path, fps=args.fps)
        # delete the reconstructed images
        for image in recon_images:
            os.remove(image)

        # vertically stack the original and reconstructed videos
        save_path = f"{save_folder}/point_cloud.mp4"
        stack_videos_vertically_with_ffmpeg(
            [original_video_path, recon_video_path], ["Original", "Reconstructed"], save_path
        )

        # remove the original and reconstructed videos
        os.remove(original_video_path)
        os.remove(recon_video_path)

        # move the point cloud video to the dump folder
        os.rename(save_path, f"{dump_dir}/point_cloud/{scene_name}.mp4")
        # delete the point cloud folder
        os.rmdir(save_folder)

if __name__ == "__main__":
    args = _args_parser()
    eval_sample(args)    