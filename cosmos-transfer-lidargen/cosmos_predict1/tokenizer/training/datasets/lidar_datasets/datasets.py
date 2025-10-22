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


import os, torch, tarfile
from pathlib import Path
from typing import Any, List, Optional
import numpy as np
import torch.utils.data
from decord import VideoReader
from lru import LRU
from abc import ABC
from webdataset import WebDataset, non_empty
from cosmos_predict1.tokenizer.training.datasets.lidar_datasets.data_fileds import DataField
from cosmos_predict1.utils.lidar_rangemap import load_each_frame_from_tar_data

def get_sample(url):
    """Get a sample from a URL with basic auto-decoding."""
    if isinstance(url, Path):
        url = url.as_posix()
        
    dataset = WebDataset(url, nodesplitter=non_empty, shardshuffle=False, workersplitter=None).decode()
    return next(iter(dataset))



view_idx_to_camera_mapping = {
    0: "camera_front_wide_120fov",
    1: "camera_cross_left_120fov",
    2: "camera_cross_right_120fov",
    3: "camera_rear_left_70fov",
    4: "camera_rear_right_70fov",
    5: "camera_rear_tele_30fov",
}


class RadymLidar(ABC):
    MAX_ZIP_DESCRIPTORS = 10
    MAX_MP4_READERS = 1
    MAX_TAR_READERS = 1

    def __init__(
        self, root_path, filter_list_path: Optional[str] = None, num_views: int = -1, metadata_folder: Optional[str] = None,
        custom_folders: Optional[List[str]] = None, custom_fields: Optional[List[str]] = None, view_indices: Optional[List[int]] = None
    ):
        # For multi-view datasets, root_path is the path to camera idx 0.
        self.root_path = root_path

        # filter_list_path is a text file containing the list of mp4 files to load.
        # Each line in the file should contain the name of the mp4 file with or without the extension.
        if filter_list_path is None:
            self.filter_set = None
        else:
            self.filter_list_path = filter_list_path
            with open(self.filter_list_path, "r") as f:
                self.filter_set = [line.strip() for line in f.readlines()]
            # self.filter_set = set([x.split(".")[0] for x in self.filter_set])
            self.filter_set = [x.split(".")[0] for x in self.filter_set]
        self.n_views = num_views
        self.view_indices = view_indices
        # Process-dependent LRU cache for file handles of the tar files.
        self.worker_id = None
        self.zip_descriptors = LRU(
            self.MAX_ZIP_DESCRIPTORS, callback=self._evict_zip_handle
        )
        self.mp4_readers = LRU(self.MAX_MP4_READERS, callback=self._evict_mp4_reader)
        self.tar_readers = LRU(self.MAX_TAR_READERS, callback=self._evict_tar_reader)
        
        self.custom_folders = custom_folders
        self.custom_fields = custom_fields
        self.metadata_folder = metadata_folder

    @staticmethod
    def _evict_zip_handle(_, zip_handle):
        zip_handle.close()

    @staticmethod
    def _evict_mp4_reader(_, mp4_reader: VideoReader):
        # This is no-op, just a placeholder.
        del mp4_reader
        
    @staticmethod
    def _evict_tar_reader(_, tar_reader):
        tar_reader.close()

    def _check_worker_id(self):
        # Protect handle boundary:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            if self.worker_id is not None:
                assert self.worker_id == worker_info.id, "Worker id mismatch"
            else:
                self.worker_id = worker_info.id
    
    
    def _get_tar_handle(self, custom_folder, custom_fields, data_key):
        """
        here we read the tar file
        """
        self._check_worker_id()
        tar_path = os.path.join(custom_folder, custom_fields, f"{data_key}.tar")
        if tar_path in self.tar_readers:
            return self.tar_readers[tar_path]
        tar_handle = tarfile.open(tar_path, "r")
        self.tar_readers[tar_path] = tar_handle
        return tar_handle
        
        
    def _get_mp4_reader(self, custom_folder, custom_fields, data_key):
        self._check_worker_id()
        mp4_path = os.path.join(custom_folder, custom_fields, f"{data_key}.mp4")
        mp4_reader = VideoReader(mp4_path, num_threads=4)
        self.mp4_readers[mp4_path] = mp4_reader
        return mp4_reader

    def available_data_fields(self) -> list[DataField]:
        return [
            DataField.IMAGE_RGB,
            DataField.CAMERA_C2W_TRANSFORM,
            DataField.CAMERA_INTRINSICS,
            DataField.METRIC_DEPTH,
            DataField.DYNAMIC_INSTANCE_MASK,
            DataField.BACKWARD_FLOW,
            DataField.OBJECT_BBOX,
            DataField.CAPTION,
        ]
        
    def size_of_filter_set(self) -> int:
        return len(self.filter_set)

    def num_videos(self) -> int:
        return len(self.mp4_file_paths)

    def num_views(self, video_idx: int) -> int:
        return 1 if self.n_views == -1 else self.n_views

    def num_frames(self, video_idx: int, view_idx: int = 0) -> int:
        return len(self._get_mp4_reader(video_idx, "rgb", view_idx))
    
    
    def parse_range_map_tar(self, tar_file, frame_idx, n_rows=128, n_cols=3600):
        frame_idx_list = np.array([x.strip('.lidar_row.npz') for x in tar_file.getnames() if 'lidar_row' in x])
        
        if frame_idx.max() > frame_idx_list.shape[0]-1: # replace those values with random integer within the range
            frame_idx[frame_idx > frame_idx_list.shape[0]-1] = np.random.randint(0, frame_idx_list.shape[0]-1, size=np.sum(frame_idx > frame_idx_list.shape[0]-1))
        
        frame_idx_list = frame_idx_list[frame_idx]
        range_map_list = []
        
        
        for frame_idx in frame_idx_list:
            range_map = load_each_frame_from_tar_data(tar_file, frame_idx, n_rows, n_cols)
            range_map_list.append(range_map)
        range_map = np.stack(range_map_list, axis=0)
        return range_map    
    
    
    def parse_hdmap_tar(self, tar_file, frame_idx, n_rows=128, n_cols=3600):
        frame_idx_list = np.array([x.strip('.lidar_row.npz') for x in tar_file.getnames() if 'lidar_row' in x])
        
        if frame_idx.max() > frame_idx_list.shape[0]-1: # replace those values with random integer within the range
            frame_idx[frame_idx > frame_idx_list.shape[0]-1] = np.random.randint(0, frame_idx_list.shape[0]-1, size=np.sum(frame_idx > frame_idx_list.shape[0]-1))
            
        frame_idx_list = frame_idx_list[frame_idx]
        hdmap_list = []

        for frame_idx in frame_idx_list:
            hdmap = load_each_frame_from_tar_data(tar_file, frame_idx, n_rows, n_cols)
            hdmap_list.append(hdmap)
        hdmap = np.stack(hdmap_list, axis=0)
        return hdmap    
    
    
    def _parse_rgb_entries(self,lidar_idx, custom_folder, custom_fields, rgb_frame_indices, chunk_size=128):
        rgb_frame_indices = [int(ele) for ele in rgb_frame_indices]
        if chunk_size is not None:
            rgb_video_chunk_id = rgb_frame_indices[0] // chunk_size
            c_data_key = self.filter_set[lidar_idx] + f"_{rgb_video_chunk_id}" 
            
            local_rgb_frame_indices = [ele % chunk_size for ele in rgb_frame_indices]
            local_rgb_frame_indices = np.asarray(local_rgb_frame_indices).astype(np.int64)
            rgb_reader = self._get_mp4_reader(custom_folder, custom_fields, c_data_key)
            rgb_read = rgb_reader.get_batch(local_rgb_frame_indices)
        else: # we read the raw videos
            c_data_key = self.filter_set[lidar_idx]
            rgb_reader = self._get_mp4_reader(custom_folder, custom_fields, c_data_key)
            rgb_read = rgb_reader.get_batch(rgb_frame_indices)
            
        try:
            rgb_np = rgb_read.asnumpy()
        except AttributeError:
            rgb_np = rgb_read.numpy()
        rgb_np = rgb_np.astype(np.float32) / 255.0
        rgb_torch = torch.from_numpy(rgb_np).moveaxis(-1, 1).contiguous()
        return rgb_torch
    
    def _read_rgb_and_lidar(self, lidar_idx, lidar_frame_idx, data_fields):
        """
        We just use the custom fields to read both lidar and rgb related data
        """
        data_key = self.filter_set[lidar_idx]
        lidar_frame_idx = np.asarray(lidar_frame_idx).astype(np.int64)  
        
        # load the metadata 
        metadata_path = os.path.join(self.metadata_folder, f"{data_key}.npz")
        metadata = np.load(metadata_path)
        lidar_pose_list = np.array(metadata['pose_list'])[lidar_frame_idx]  # list
        lidar_timestamp_list = np.array(metadata['timestamps_list'])[lidar_frame_idx] 
        rgb_frame_indices = np.array(metadata['frame_indices'])[lidar_frame_idx].tolist()  # this can be different from lidar_frame_idx
        lidar_to_world = lidar_pose_list[:,0]  # N x 4 x 4
        lidar_to_world_next = lidar_pose_list[:,1]  # N x 4 x 4
        sensor_pose_start_end = np.linalg.inv(lidar_to_world_next) @ lidar_to_world
        
        output_dict: dict[str | DataField, Any] = {"__key__": data_key}
        camera_type = 'ftheta'
        output_dict['timestamp_list'] = lidar_timestamp_list
        output_dict['pose_sensor_start_end'] = sensor_pose_start_end
        output_dict['rgb_frame_indices'] = rgb_frame_indices
        view_indices = self.view_indices if self.view_indices is not None else range(self.n_views)
        
        for data_field in data_fields:
            if data_field == "custom":
                if self.custom_folders is not None:
                    output_dict[data_field] = {}
                    assert len(self.custom_folders) == len(self.custom_fields), "Custom folders and types must have the same length"
                    for custom_folder, custom_fields in zip(self.custom_folders, self.custom_fields):
                        if custom_fields in ['lidar']:  # read the lidar data
                            tar_handle = self._get_tar_handle(custom_folder, custom_fields, data_key)
                            range_maps = self.parse_range_map_tar(tar_handle, lidar_frame_idx) # this is numpy
                            output_dict[data_field][custom_fields] = range_maps
                        elif custom_fields in ['hdmap']:
                            tar_handle = self._get_tar_handle(custom_folder, custom_fields, data_key)
                            hdmap = self.parse_hdmap_tar(tar_handle, lidar_frame_idx)
                            output_dict[data_field][custom_fields] = hdmap
                        else:
                            raise NotImplementedError(f"Can't handle custom data field {data_field}")
            elif data_field == DataField.CAMERA_INTRINSICS:
                output_dict[data_field] = {}

                for c_view_idx, view_idx in enumerate(view_indices):
                    camera_name = view_idx_to_camera_mapping[view_idx]
                    intrinsic_tar = os.path.join(self.root_path, f"{camera_type}_intrinsic", f"{data_key}.tar")
                    intrinsic_data = get_sample(intrinsic_tar)[f'{camera_type}_intrinsic.{camera_name}.npy']
                    intrinsic_data = torch.from_numpy(intrinsic_data).contiguous()
                    output_dict[data_field][c_view_idx] = intrinsic_data
            elif data_field == DataField.CAMERA_C2LIDAR_TRANSFORM:
                output_dict[data_field] = {}
                for c_view_idx, view_idx in enumerate(view_indices):
                    camera_name = view_idx_to_camera_mapping[view_idx]
                    camera_pose_path = os.path.join(self.root_path, "pose", f"{data_key}.tar")
                    camera_pose_data = get_sample(camera_pose_path)
                    camera_to_world = np.array([camera_pose_data[f'{frame_idx}.pose.{camera_name}.npy'] for frame_idx in rgb_frame_indices])
                    camera_to_lidar = np.linalg.inv(lidar_to_world) @ camera_to_world
                    camera_to_lidar = torch.from_numpy(camera_to_lidar).contiguous()
                    output_dict[data_field][c_view_idx] = camera_to_lidar
            elif data_field == DataField.IMAGE_RGB: # here we directly read the raw videos
                output_dict[data_field] = {}
                for c_view_idx, view_idx in enumerate(view_indices):
                    custom_folder = self.root_path
                    custom_fields = f"{camera_type}_{view_idx_to_camera_mapping[view_idx]}"
                    rgb_torch = self._parse_rgb_entries(lidar_idx, custom_folder, custom_fields, rgb_frame_indices, chunk_size=None)
                    output_dict[data_field][c_view_idx] = rgb_torch
            else:
                raise NotImplementedError(f"Can't handle data field {data_field}")
            
        return output_dict