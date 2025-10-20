# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all
# intellectual property and proprietary rights in and to this software,
# related documentation and any modifications thereto. Any use,
# reproduction, disclosure or distribution of this software and related
# documentation without an express license agreement from NVIDIA
# CORPORATION & AFFILIATES is strictly prohibited.

import os
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.utils import natural_key
from utils.wds_utils import get_sample
from utils.bbox_utils import fix_static_objects
from utils.minimap_utils import simplify_minimap
from utils.minimap_utils_for_lidar import (
    get_minimap_points,
    get_bbox_points,
)
from utils.lidar_rangemap_utils import (
    undo_motion_compensation_impl,
)


class MadsLidarDataset(Dataset):
    """Dataset for loading and preparing HD map data for range-map projection.

    Performs CPU-side preprocessing, including reading LiDAR and map elements,
    preparing poses and timestamps, and organizing per-frame items.
    """
    
    def __init__(
        self,
        input_root: str,
        settings: Dict,
        split_filename: str,
        n_rows: int = 128,
        n_cols: int = 3600,
        density: float = 0.02,
        radius: float = 0.05,
        max_frames: int = -1
    ):
        """Initialize dataset configuration and load split.

        Args:
            input_root: Root directory containing input data.
            settings: Configuration dictionary.
            split_filename: File with one clip id per line.
            n_rows: Number of range-map rows.
            n_cols: Number of range-map columns.
            density: Sampling density for map elements.
            radius: Sampling radius for map elements.
            max_frames: Max frames to process per clip (-1 for all).
        """
        self.input_root = input_root
        self.settings = settings
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.density = density
        self.radius = radius
        self.max_frames = max_frames
        
        # Load clip list from split file
        with open(split_filename, 'r') as f:
            self.clip_list = [line.strip().replace('.tar', '') for line in f]
            
        print(f"Total clips from split file: {len(self.clip_list)}")
        
    def _load_clip_data(self, clip_id: str) -> Dict:
        """Load and preprocess data for a single clip."""
        # check if the clips exist
        lidar_path_tar = os.path.join(self.input_root, 'lidar_raw', f"{clip_id}.tar")
        if not os.path.exists(lidar_path_tar):
            raise FileNotFoundError(f"Lidar file not found: {lidar_path_tar}")
            
        lidar_data = get_sample(lidar_path_tar)
        if lidar_data is None:
            raise ValueError(f"Could not load data from {lidar_path_tar}")
        
        # load raw lidar points, here we undo the motion compensation to the points. 
        lidar_points_list, frame_indices, pose_list, timestamps_list, intensity_list = self._load_mads_lidar_without_motion_compensation(lidar_data, self.n_rows, self.n_cols)
        lidar_points_list = [torch.from_numpy(lidar_points).float() for lidar_points in lidar_points_list]
        intensity_list = [torch.from_numpy(intensity).float() for intensity in intensity_list]
        
        # Load object info
        all_object_info_file = os.path.join(self.input_root, 'all_object_info', f"{clip_id}.tar")
        all_object_info = get_sample(all_object_info_file)
        if all_object_info is None:
            raise ValueError(f"Could not load object info from {all_object_info_file}")
            
        all_object_info = fix_static_objects(all_object_info)
        
        # Load HD map elements, here the points are in the world frame, so are motion-compensated. 
        static_points_dict = self._load_hd_map_elements(clip_id)
        static_points = [torch.from_numpy(v).float() for v in static_points_dict.values()]
        static_points = torch.cat(static_points, dim=0)
        road_boundaries = static_points_dict['road_boundaries']
        lane_points = static_points_dict['lanelines']
        
        # Load bounding box elements
        bbox_points_list = self._load_bbox_elements(all_object_info, frame_indices)

        return {
            'road_boundaries': torch.from_numpy(road_boundaries).float(),
            'lane_points': torch.from_numpy(lane_points).float(),
            'frame_indices': frame_indices,
            'pose_list': pose_list,
            'bbox_points_list': bbox_points_list,
            'static_points': static_points,
            'lidar_points_list': lidar_points_list,
            # a list of [current_timestamp, next_timestamp]
            'timestamps_list': timestamps_list,
            'intensity_list': intensity_list
        }
        
    @staticmethod
    def _load_mads_lidar_without_motion_compensation(lidar_data, n_rows=128, n_cols=3600):
        """Prepare per-frame LiDAR points and metadata without motion compensation.
        Note that the raw data has been motion compensated, so we need to undo it.
        """
        lidar_keys = [ele for ele in lidar_data.keys() if 'lidar_raw' in ele]
        frame_indices = [ele.split('.')[0] for ele in lidar_keys]
        frame_indices.sort(key=natural_key)
        
        lidar_points_list = []
        usable_frame_indices = []
        pose_list = []
        timestamps_list = []
        intensity_list = []
        for idx, frame_idx in enumerate(frame_indices):
            lidar_raw_key = f"{frame_idx}.lidar_raw.npz"
            lidar_raw = lidar_data[lidar_raw_key]

            lidar_points = lidar_raw['xyz'].astype(np.float32)  # (N, 3)
            lidar_intensity = lidar_raw['intensity'].astype(np.float32)  # (N,)
            lidar_row = lidar_raw['row'].astype(np.uint8)  # (N,)
            lidar_col = lidar_raw['column'].astype(np.uint16)  # (N,)
            ego_pose = lidar_raw['lidar_to_world'].astype(np.float32)  # (4, 4)

            sel = (
                (lidar_col < n_cols) & (lidar_col >= 0) &
                (lidar_row < n_rows) & (lidar_row >= 0)
            )
            lidar_col = lidar_col[sel]
            lidar_row = lidar_row[sel]
            lidar_points = lidar_points[sel]
            lidar_intensity = lidar_intensity[sel]
            
            # undo motion compensation
            # get the pose and timestamp of the current frame and next frame 
            current_pose = ego_pose
            current_timestamp = int(lidar_raw['starting_timestamp'])
            if idx < len(frame_indices) - 1:
                next_key = f"{frame_indices[idx+1]}.lidar_raw.npz"
                next_raw = lidar_data[next_key]
                next_pose = next_raw['lidar_to_world'].astype(np.float32)  # (4, 4)
                next_timestamp = int(next_raw['starting_timestamp'])
                if next_timestamp <= current_timestamp:
                    continue
            else:
                continue
                    
            T_sensor_start_sensor_end = np.linalg.inv(next_pose) @ current_pose
            
            timestamps_startend_us = [current_timestamp, next_timestamp]
            
            # linearly interpolate the timestamps to the point timestamps
            timestamp_us = np.linspace(current_timestamp, next_timestamp, n_cols)[lidar_col]
            
            # undo motion compensation
            lidar_points = undo_motion_compensation_impl(
                lidar_points,
                T_sensor_start_sensor_end,
                timestamps_startend_us,
                timestamp_us,
            )
            lidar_points_list.append(lidar_points)
            usable_frame_indices.append(frame_idx)
            pose_list.append([ego_pose, next_pose])
            timestamps_list.append([current_timestamp, next_timestamp])
            intensity_list.append(lidar_intensity)
            
        return (lidar_points_list, usable_frame_indices, pose_list, timestamps_list, intensity_list)
        
    def _load_hd_map_elements(self, clip_id: str) -> torch.Tensor:
        """Load and process HD map elements for a clip."""
        minimap_types = self.settings['MINIMAP_TYPES']
        minimap_wds_files = [os.path.join(self.input_root, f"3d_{minimap_type}", f"{clip_id}.tar") for minimap_type in minimap_types]
        
        static_points = dict()
        for minimap_wds_file in minimap_wds_files:
            minimap_data_wo_meta_info, minimap_name = simplify_minimap(minimap_wds_file)
            static_points[minimap_name] = get_minimap_points(minimap_name, minimap_data_wo_meta_info, density=self.density, radius=self.radius)
        return static_points
        
    def _load_bbox_elements(
        self, all_object_info: Dict, frame_indices: List[str]
    ) -> List[torch.Tensor]:
        """Load and process bounding box elements."""
        bbox_points_list = get_bbox_points(all_object_info, frame_indices, self.density, self.radius)
        return [torch.from_numpy(bbox_points).float() for bbox_points in bbox_points_list]
        
    def __len__(self) -> int:
        return len(self.clip_list)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load one clip and return preprocessed tensors and metadata.

        Returns a dict with keys such as 'clip_id', 'frame_indices',
        'bbox_points_list', 'pose_list', 'static_points'.
        """
        clip_id = self.clip_list[idx]
        clip_data = self._load_clip_data(clip_id)
        clip_data.update({'clip_id': clip_id})
        
        return clip_data



def collate_fn(batch):
    """Collate heterogeneous batch items for the HD map dataset.

    - Lists: kept as lists of lists
    - Tensors: stacked along dim 0
    - Strings: collected into lists
    """
    if not batch:
        return None
        
    # Get all keys from first item
    keys = batch[0].keys()
    collated_batch = {}
    
    for key in keys:
        # Get all values for this key across batch
        values = [item[key] for item in batch]
        
        if isinstance(values[0], list):
            # For lists, keep as list of lists
            collated_batch[key] = values
        elif isinstance(values[0], torch.Tensor):
            # For tensors, stack along first dimension
            collated_batch[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], str):
            # For strings, keep as list
            collated_batch[key] = values
        else:
            raise TypeError(f"Unsupported type {type(values[0])} for key {key}")
            
    return collated_batch

def get_mads_dataloader(
    input_root: str,
    settings: Dict,
    split_filename: str,
    batch_size: int = 1,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, int, int]:
    """Create a DataLoader for HD map clips and return map dimensions.

    Args include common DataLoader params; returns (loader, n_rows, n_cols).
    """
    dataset = MadsLidarDataset(
        input_root=input_root,
        settings=settings,
        split_filename=split_filename,
    )
    
    dloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, **kwargs)
    
    return (dloader, dataset.n_rows, dataset.n_cols)
