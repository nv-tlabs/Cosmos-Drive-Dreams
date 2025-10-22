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

import gc

import numpy as np
import torch

import os
import pickle

import einops
import omegaconf
import torchvision.transforms as transforms

from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils import log

try:
    from megatron.core import parallel_state
    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from cosmos_predict1.tokenizer.training.datasets.lidar_datasets.datasets import RadymLidar
from cosmos_predict1.tokenizer.training.datasets.lidar_datasets.data_fileds import DataField
from cosmos_predict1.utils.lidar_rangemap import RangeMapDownsampler, normalize_range_map

from cosmos_predict1.diffusion.training.datasets.lidar_dataset.configs import COMMONDATA_CONFIG

def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True, ignore_keys=None):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """

    batched = {key: [] for key in samples[0]}
    # assert isinstance(samples[0][first_key], (list, tuple)), type(samples[first_key])

    for s in samples:
        [batched[key].append(s[key]) for key in batched if not (ignore_keys is not None and key in ignore_keys)]

    result = {}
    for key in batched:
        if ignore_keys and key in ignore_keys:
            continue
        try:
            if isinstance(batched[key][0], bool):
                assert key == "is_preprocessed"
                result[key] = batched[key][0]  # this is a hack to align with cosmos data
            elif isinstance(batched[key][0], (int, float)):
                if combine_scalars:
                    result[key] = torch.from_numpy(np.array(list(batched[key])))
            elif isinstance(batched[key][0], torch.Tensor):
                if combine_tensors:
                    result[key] = torch.stack(list(batched[key]))
            elif isinstance(batched[key][0], np.ndarray):
                if combine_tensors:
                    result[key] = np.array(list(batched[key]))
            elif isinstance(batched[key][0], list) and isinstance(batched[key][0][0], int):
                result[key] = [torch.Tensor(elems).long() for elems in zip(*batched[key])]
            else:
                result[key] = list(batched[key])
        except Exception as e:
            print(key)
            raise e
        # result.append(b)
    del batched
    return result


class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size: int = 1, *args, **kw):
        dataset_obj = dataset.build_dataset()
        print("dl", kw.get("dataloaders"))
        if "dataloaders" in kw:
            kw.pop("dataloaders")  # HACK?
        print("dl2", kw.get("dataloaders"))
        super().__init__(dataset_obj, batch_size, collate_fn=dict_collation_fn, *args, **kw)


class LidarRangeMapDataset:
    def __init__(self, dataset_name, filter_list_path=None, **kwargs):
        self.dataset_name = dataset_name
        self.dataset_config = OmegaConf.create(COMMONDATA_CONFIG[dataset_name])
        self.extra_config = OmegaConf.create(kwargs)
        
        if filter_list_path is not None:
            self.dataset_config["filter_list_path"] = filter_list_path

    def build_dataset(self):
        config_dict = OmegaConf.to_container(self.dataset_config, resolve=True)
        return LidarRangeMapSampler(**config_dict)
    
    
def get_lidar_dataset(dataset_name, filter_list_path=None):
    dataset_config = OmegaConf.create(COMMONDATA_CONFIG[dataset_name])
    
    if filter_list_path is not None:
        dataset_config["filter_list_path"] = filter_list_path
    
    config_dict = OmegaConf.to_container(dataset_config, resolve=True)
    return LidarRangeMapSampler(**config_dict)


class LidarRangeMapSampler(Dataset):
    def __init__(
        self,
        data_name,
        root_path,
        filter_list_path,
        t5_embedding_path,
        custom_folders,
        custom_fields,
        metadata_folder,
        max_range,
        min_range,
        downsample_factor_row,
        downsample_factor_col,
        repeat_row,
        repeat_col,
        repeat_temporal,
        lidar_crop_size,
        rgb_crop_size,
        rgb_resize_size,
        lidar_size,
        downsample_method,  # for range map
        min_value,
        num_views,  # the number of RGB cameras
        sample_n_frames,
        lidar_length,
        frame_sampling_method,
        frame_sampling_every_n_frames,
        sample_lidar_prob,
        is_video_tokenizer,
        load_lidar_only=True,
        load_rgb_only=False,
        load_lidar_and_rgb=False,  # load both lidar and rgb
        load_lidar_or_rgb=False,
        RGB_FPS=30,
        LIDAR_FPS=10,
        load_t5_embeddings=False,
        inverse_depth=False,
        to_three_channels="repeat",  # "repeat" or "concat"
        inv_depth_threshold=20,
        view_indices=None,
    ):
        super().__init__()
        self.data_name = data_name
        # build the dataset
        self.dataset = RadymLidar(
            root_path=root_path,
            filter_list_path=filter_list_path,
            num_views=num_views,
            metadata_folder=metadata_folder,
            custom_folders=custom_folders,
            custom_fields=custom_fields,
            view_indices=view_indices,
        )
        self.plus = 1 if "hdmap" not in self.dataset.custom_fields else 2
        self.num_views = num_views

        self.LIDAR_FPS = LIDAR_FPS
        self.RGB_FPS = RGB_FPS

        self.sample_n_frames = sample_n_frames  # the "actual" target sequence length
        self.lidar_length = lidar_length
        self.rgb_length = lidar_length * (RGB_FPS // LIDAR_FPS)
        self.frame_sampling_method = frame_sampling_method
        assert self.frame_sampling_method in ["random", "sequential", "sequential_from_zero"]
        self.frame_sampling_every_n_frames = frame_sampling_every_n_frames

        self.inverse_depth = inverse_depth
        self.min_distance = min_range
        self.max_distance = max_range
        self.max_range = max_range if not inverse_depth else 1 / min_range
        self.min_range = min_range if not inverse_depth else 1 / max_range

        self.min_value = min_value  # min value after normalisation, if 0 then [0, 1],if -1, then [-1, 1]

        self.repeat_row = repeat_row
        self.repeat_col = repeat_col
        self.repeat_temporal = repeat_temporal
        self.to_three_channels = to_three_channels
        assert self.to_three_channels in ["repeat", "concat"]

        self.lidar_size = lidar_size
        self.lidar_crop_size = lidar_crop_size
        self.rgb_crop_size = rgb_crop_size
        self.rgb_resize_size = rgb_resize_size

        self.sample_lidar_prob = sample_lidar_prob if not load_rgb_only else -1
        self.load_lidar_only = load_lidar_only
        self.load_rgb_only = load_rgb_only
        self.load_lidar_and_rgb = load_lidar_and_rgb
        self.load_lidar_or_rgb = load_lidar_or_rgb

        assert sum([self.load_lidar_only, self.load_rgb_only, self.load_lidar_and_rgb, self.load_lidar_or_rgb]) == 1

        # t5 text embedding
        self.load_t5_embeddings = load_t5_embeddings
        self.t5_embedding_path = t5_embedding_path

        self.length = self.dataset.size_of_filter_set()
        log.critical(f"load_lidar_only: {self.load_lidar_only}")
        log.critical(f"dataset size: {self.length}")

        self.is_video_tokenizer = is_video_tokenizer  # if true, then the frame sampling method is sequential, also we need to upsample the lidar to the rgb fps
        if self.is_video_tokenizer:
            assert self.frame_sampling_method == "sequential"

        self.range_map_downsampler = RangeMapDownsampler(row_factor = downsample_factor_row, col_factor = downsample_factor_col, method = downsample_method)

        # image transform
        self.img_transform = transforms.Compose(
            [
                transforms.Resize(
                    self.rgb_resize_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
                ),
                transforms.CenterCrop(self.rgb_crop_size),
            ]
        )
        self.norm_image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

        # Prepare infinite dataset logic
        self.n_data = self.length
        if parallel_state.is_initialized():
            dp_group_id = parallel_state.get_data_parallel_rank()
            dp_world_size = parallel_state.get_data_parallel_world_size()
            log.critical(
                f"Using parallelism size CP :{parallel_state.get_context_parallel_world_size()}, "
                + f"TP :{parallel_state.get_tensor_model_parallel_world_size()} for video dataset, "
                + f"DP: {dp_group_id}, DP World size: {dp_world_size}"
            )
        else:
            dp_world_size = 1
            dp_group_id = 0

        # Data partitioning
        self.n_data_per_node = self.n_data // dp_world_size
        self.data_start_idx = dp_group_id * self.n_data_per_node
        self.dp_group_id = dp_group_id

        # Infinite loop multiplier
        maximum_iter = 2000000  # hack to create infinite loop
        self.multiplier = maximum_iter // self.n_data_per_node
        # self.multiplier = 1

    def __len__(self):
        """
        Return the artificially multiplied length
        """
        return self.multiplier * self.n_data_per_node

    def sample_random_frames(self, n_frames, total_frames):
        # sample self.sample_n_frames frames from the video
        # return the frame indices
        frame_indices = np.random.randint(0, total_frames, size=n_frames)
        return frame_indices

    def sample_sequential_frames(self, n_frames, total_frames):
        # sample every self.frame_sampling_every_n_frames frames
        if self.frame_sampling_method == "sequential_from_zero":
            start_idx = 0
        else:
            start_idx = np.random.randint(0, total_frames - n_frames * self.frame_sampling_every_n_frames)
        frame_indices = np.arange(start_idx, start_idx + n_frames * self.frame_sampling_every_n_frames, self.frame_sampling_every_n_frames)
        return frame_indices
    

    def sample_frame_indices(self, load_lidar):
        if load_lidar:
            if self.frame_sampling_method == "random":
                frame_indices = self.sample_random_frames(self.sample_n_frames, self.lidar_length)
            else:
                frame_indices = self.sample_sequential_frames(
                    self.sample_n_frames // self.repeat_temporal, self.lidar_length
                )

        else:  # load rgb
            if self.frame_sampling_method == "random":
                frame_indices = self.sample_random_frames(self.sample_n_frames, self.rgb_length)
            else:
                frame_indices = self.sample_sequential_frames(
                    self.sample_n_frames + 1, self.rgb_length
                )  # + 1 because the video tokenizer will need one more additional frame
        return frame_indices

    def normalise_and_convert_range_map_to_three_channels(self, range_maps):
        # Input: [N, H, W]
        # Output: [N, 3, H * repeat_row, W * repeat_col]
        # repeat along the channel dimension, and row and column
        if self.to_three_channels == "repeat":
            range_maps = normalize_range_map(
                range_maps, self.max_distance, self.min_distance, self.min_value, self.inverse_depth
            )  # shape: [N, H, W]
            range_maps = range_maps[:, None, :, :].repeat(3, axis=1)  # shape: [N, 3, 128, 1800]
        elif self.to_three_channels == "concat":
            inverse_depth = normalize_range_map(
                range_maps, self.max_distance, self.min_distance, self.min_value, True
            )  # shape: [N, H, W]
            depth = normalize_range_map(
                range_maps, self.max_distance, self.min_distance, self.min_value, False
            )  # shape: [N, H, W]
            range_maps = np.concatenate(
                [inverse_depth[:, None, :, :], depth[:, None, :, :], depth[:, None, :, :]], axis=1
            )  # shape: [N, 3, 128, 1800]

        return range_maps

    def process_lidar(self, range_maps):
        # downsample the range map
        range_maps = self.range_map_downsampler.downsample(
            range_maps
        )  # shape: [N, H, W]

        range_maps = self.normalise_and_convert_range_map_to_three_channels(
            range_maps
        )  # shape: [N, 3, 128 * repeat_row, 1800 * repeat_col]
        range_maps = range_maps.repeat(self.repeat_row, axis=2)  # shape: [N, 3, 128 * repeat_row, 1800]
        range_maps = range_maps.repeat(self.repeat_col, axis=3)  # shape: [N, 3, 128 * repeat_row, 1800 * repeat_col]

        # only crop along the column dimension
        if self.lidar_crop_size[1] != -1:
            if self.lidar_crop_size[1] > 1600: # we do center crop
                start_col_idx = (range_maps.shape[-1] - self.lidar_crop_size[1]) // 2
                range_maps = range_maps[:, :, :, start_col_idx : start_col_idx + self.lidar_crop_size[1]]
            else:
                start_col_idx = np.random.randint(0, range_maps.shape[-1] - self.lidar_crop_size[1])
                range_maps = range_maps[:, :, :, start_col_idx : start_col_idx + self.lidar_crop_size[1]]

        # if it comes from sequential sampling, then we need to repeat the range maps for  sequence
        if self.frame_sampling_method == "sequential" and self.is_video_tokenizer:
            first_range_map = range_maps[
                0
            ]  # let's just repeat the first frame to make task easier. This is the same practice as when I evaluate the tokenizer
            range_maps = range_maps.repeat(self.repeat_temporal, axis=0)
            range_maps = np.concatenate(
                [first_range_map[None], range_maps], axis=0
            )  # this one will be repeated later on in the encoder

        # to torch tensor
        range_maps = torch.from_numpy(range_maps).float()  # shape: [N, 3, 128 * repeat_row, 1800 * repeat_col]

        return range_maps

    def process_rgb(self, rgb_images):
        rgb_images = self.img_transform(rgb_images)
        rgb_images = self.norm_image(rgb_images)  # N C H W
        return rgb_images

    def prepare_dummy_data(self):
        data = {}
        dummy_text_embedding = torch.zeros(512, 1024)
        dummy_text_mask = torch.zeros(512)
        dummy_text_mask[0] = 1
        data["t5_text_embeddings"] = dummy_text_embedding
        data["t5_text_mask"] = dummy_text_mask

        return data

    def prepare_t5_embeddings(self, example, frame_indices):
        # determine which chunk to use based on the frame indices
        mid_indice = frame_indices[len(frame_indices) // 2]
        chunk_id = mid_indice // int(self.lidar_length / 5)

        if self.load_t5_embeddings:
            # t5_embed_path = os.path.join(self.t5_embedding_path, example["__key__"] + f"_{chunk_id}.pkl")
            t5_embed_path = os.path.join(self.t5_embedding_path, example["__key__"] + ".pkl")
            if os.path.exists(t5_embed_path):
                t5_embed_pickle = pickle.load(open(t5_embed_path, "rb"))
                t5_embed = {}
                text_embedding = t5_embed_pickle["pickle"]["ground_truth"]["embeddings"]["t5_xxl"]
                dummy_text_embedding = torch.zeros(512, 1024)
                dummy_text_mask = torch.zeros(512)
                n_text = text_embedding.shape[0]  # 512
                dummy_text_embedding[:n_text] = torch.from_numpy(text_embedding)
                dummy_text_mask[:n_text] = 1
                t5_embed["t5_text_embeddings"] = dummy_text_embedding
                t5_embed["t5_text_mask"] = dummy_text_mask
                del t5_embed_pickle
            else:
                log.info(f"t5 embedding not found for {example['__key__']}", rank0_only=False)
                t5_embed = self.prepare_dummy_data()
        else:
            t5_embed = self.prepare_dummy_data()
            
        if self.num_views > 1:
            t5_embed["t5_text_embeddings"] = torch.cat([t5_embed["t5_text_embeddings"]] * (self.num_views+self.plus), dim=0)
            t5_embed["t5_text_mask"] = torch.cat([t5_embed["t5_text_mask"]] * (self.num_views+self.plus), dim=0)

        return t5_embed

    def get_lidar(self, example, lidar_key = "lidar"):
        if lidar_key in example["custom"]:
            range_maps = example["custom"][lidar_key]  # shape: [N, 128, 3600]
            range_maps = self.process_lidar(range_maps)
            range_maps = range_maps.permute(1, 0, 2, 3).contiguous()  # shape: [C, N, H, W]
        else:
            range_maps = None

        return range_maps

    def get_lidar_hdmap(self, example, hdmap_key = "hdmap"):
        if hdmap_key in example["custom"]:
            hdmap = example["custom"][hdmap_key]  # shape: [N, 128, 3600]
            hdmap = self.process_lidar(hdmap)
            hdmap = hdmap.permute(1, 0, 2, 3).contiguous()  # shape: [C, N, H, W]
        else:
            hdmap = None
        return hdmap

    def get_rgb(self, example):
        key = DataField.IMAGE_RGB
        if key in example.keys():
            rgb_images_dict = dict()
            view_keys = example[key].keys()
            for view_key in view_keys:
                rgb_images = example[key][view_key]
                rgb_images = self.process_rgb(rgb_images)
                rgb_images = rgb_images.permute(1, 0, 2, 3).contiguous()  # shape: [C, N, H, W]
                rgb_images_dict[key.value + "_" + str(view_key)] = rgb_images
        else:
            rgb_images_dict = None
        return rgb_images_dict

    def get_rgb_hdmap(self, example):
        key = DataField.IMAGE_HDMAP
        if key in example.keys():
            rgb_hdmap_dict = dict()
            view_keys = example[key].keys()
            for view_key in view_keys:
                rgb_hdmap = example[key][view_key]
                rgb_hdmap = self.process_rgb(rgb_hdmap)
                rgb_hdmap = rgb_hdmap.permute(1, 0, 2, 3).contiguous()  # shape: [C, N, H, W]
                rgb_hdmap_dict[key.value + "_" + str(view_key)] = rgb_hdmap
        else:
            rgb_hdmap_dict = None
        return rgb_hdmap_dict
    
    def get_camera_c2lidar_transform(self, example):
        key = DataField.CAMERA_C2LIDAR_TRANSFORM
        if key in example.keys():
            camera_c2lidar_transform_dict = dict()
            view_keys = example[key].keys()
            for view_key in view_keys:
                camera_c2lidar_transform = example[key][view_key]
                camera_c2lidar_transform_dict[key.value + "_" + str(view_key)] = camera_c2lidar_transform
        else:
            camera_c2lidar_transform_dict = None
        return camera_c2lidar_transform_dict
    
    def get_camera_intrinsics(self, example):
        key = DataField.CAMERA_INTRINSICS
        if key in example.keys():
            view_keys = example[key].keys()
            camera_intrinsics_dict = dict()
            for view_key in view_keys:
                camera_intrinsics = example[key][view_key]
                camera_intrinsics_dict[key.value + "_" + str(view_key)] = camera_intrinsics
        else:
            camera_intrinsics_dict = None
        return camera_intrinsics_dict
    
    

    def __getitem__(self, idx):
        if self.load_rgb_only:
            raise NotImplementedError
        if self.load_lidar_or_rgb:
            raise NotImplementedError

        # sample the video index
        lidar_data_idx = (idx % self.n_data_per_node) + self.data_start_idx  # frame idx
        lidar_frame_indices = self.sample_frame_indices(load_lidar=True)

        # load the data
        try:
            if self.load_lidar_and_rgb:  # we also load the camera intrinsics and camera-to-lidar transformation matrix as well as the timestamp for motion compensation
                # example = self.dataset._read_rgb_and_lidar(lidar_data_idx, lidar_frame_indices, ["custom", DataField.CAMERA_C2LIDAR_TRANSFORM, DataField.CAMERA_INTRINSICS, DataField.IMAGE_HDMAP, DataField.IMAGE_RGB])
                example = self.dataset._read_rgb_and_lidar(lidar_data_idx, lidar_frame_indices, ["custom", DataField.CAMERA_C2LIDAR_TRANSFORM, DataField.CAMERA_INTRINSICS, DataField.IMAGE_RGB])
            else:
                example = self.dataset._read_rgb_and_lidar(lidar_data_idx, lidar_frame_indices, ["custom"])
        except Exception as e:
            log.info(f"Error loading data for index {idx}: {e}", rank0_only=False)
            return self.__getitem__(np.random.randint(0, self.__len__()))

        lidar_rangemap = self.get_lidar(example)
        lidar_hdmap = self.get_lidar_hdmap(example)
        rgb_images = self.get_rgb(example)
        rgb_hdmap = self.get_rgb_hdmap(example)
        t5_embed = self.prepare_t5_embeddings(example, lidar_frame_indices)
        
        
        cam_intrinsics = self.get_camera_intrinsics(example)
        cam_c2lidar_transform = self.get_camera_c2lidar_transform(example)

        key = "video"

        sample = {}
        sample[key] = lidar_rangemap
        sample["__key__"] = example["__key__"]
        sample["clip_name"] = "%s-%s-%d" % (self.data_name, example["__key__"], idx)
        sample["data_type"] = "lidar"
        sample["num_frames"] = self.sample_n_frames + 1 if self.is_video_tokenizer else self.sample_n_frames
        sample["is_preprocessed"] = True
        sample["image_size"] = torch.tensor(self.lidar_size)
        sample["padding_mask"] = torch.zeros(1, self.lidar_size[0], self.lidar_size[1])
        sample["fps"] = 24  # unsure why this value
        sample['num_views'] = self.num_views
        sample["view_indices"] = torch.arange(self.num_views + self.plus) # +1 for the lidar view
        sample["timestamp_list"] = example["timestamp_list"]
        sample["pose_sensor_start_end"] = example["pose_sensor_start_end"]
        sample["rgb_frame_indices"]=example["rgb_frame_indices"]
        sample.update(t5_embed)

        if lidar_hdmap is not None:
            sample["lidar_hdmap"] = lidar_hdmap
        if rgb_images is not None:
            sample.update(rgb_images)
        if rgb_hdmap is not None:
            sample.update(rgb_hdmap)
        if self.load_lidar_and_rgb:
            
            sample.update(cam_intrinsics)
            sample.update(cam_c2lidar_transform)
            
        
        del example
        gc.collect()
        # from imaginaire.utils import distributed, log
        # rank = distributed.get_rank()
        # print(f"RANK {rank}, {data_idx}: Working on clip {sample['clip_name']}.")
        return sample