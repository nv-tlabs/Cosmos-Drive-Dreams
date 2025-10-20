# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
Utilities for LiDAR range maps: sensor model init, point projection, ray directions,
motion-compensation helpers, and efficient tar saving.
"""
import json
import gc
import yaml
import torch
import numpy as np
from tqdm import tqdm
import torch_scatter
from utils.wds_utils import write_to_tar
from utils.utils import to_numpy_array, transform_points_torch
import ncore.impl.common.common as ncore_common
import ncore.impl.common.transformations as ncore_transformations
from typing import Tuple, List, Optional
from ncore.impl.sensors.lidar import RowOffsetStructuredSpinningLidarModel
from ncore.impl.data.types import RowOffsetStructuredSpinningLidarModelParameters

def load_default_lidar_model_parameters(param_path: str = "assets/row-offset-spinning-lidar-model-parameters.json"):
    """
    Load the default LiDAR model parameters from the JSON file.
    """
    
    param = json.load(open(param_path, "r"))
    model_parameters = RowOffsetStructuredSpinningLidarModelParameters.from_dict(param)
    model_parameters.fov_horiz_min_rad = param['fov_horiz_min_rad']
    model_parameters.fov_horiz_max_rad = param['fov_horiz_max_rad']
    model_parameters.fov_vert_min_rad = param['fov_vert_min_rad']
    model_parameters.fov_vert_max_rad = param['fov_vert_max_rad']
    model_parameters.column_azimuths_rad = np.linspace(np.pi, -np.pi, 3600, endpoint=False)
    
    return model_parameters

def init_ncore_default_sensor_model(device: str = "cuda"):
    model_parameters = load_default_lidar_model_parameters()
    return RowOffsetStructuredSpinningLidarModel(
        parameters=model_parameters,
        device=device,
        dtype=torch.float32,
    )
    
    
def init_ncore_sensor_model(sensor_config_path, device: str = "cuda"):
    """Init an NCORE row-offset spinning LiDAR model from calibration JSON."""
    model_parameters = load_default_lidar_model_parameters()
    lidar_calib_params = json.load(open(sensor_config_path, "r"))
    
    
    # update the model parameters according to my sensor model
    model_parameters.row_azimuth_offsets_rad = np.array(lidar_calib_params['row_azimuth_offsets_rad'])
    model_parameters.row_elevations_rad = np.array(lidar_calib_params['row_elevations_rad'])
    
    lidar_model = RowOffsetStructuredSpinningLidarModel(
        parameters=model_parameters,
        angles_to_columns_map_init=False,
        angles_to_columns_map_resolution_factor=3,
        device=device,
        dtype=torch.float32,
    )
    
    return lidar_model

def save_range_maps_to_tar(range_map, frame_indices, clip_id, tar_path, intensity_map=None):
    """
    Save sparse range and optional intensity maps to a WebDataset tar shard.

    Each frame is stored sparsely by writing npz entries for the indices and values
    of non-zero pixels to reduce storage and I/O.

    Args:
        range_map list of np.ndarray: Array of shape (T, H, W) with range values.
        frame_indices (list[int] | np.ndarray): Frame ids aligned with the first dim of `range_map`.
        clip_id (str): WebDataset key for the sample (used as `__key__`).
        tar_path (str): Output tar file path.
        intensity_map (List[np.ndarray] | None): Optional array of shape (T, H, W) with intensity values.

    """
    range_map_npz = {'__key__': clip_id}
    for idx, frame_idx in enumerate(frame_indices):
        c_range_map = range_map[idx]  # H, W
        
        # extract the indices of pixels with value > 0
        valid_pixels = np.where(c_range_map > 0)
        lidar_row, lidar_col = valid_pixels[0], valid_pixels[1]
        lidar_range = c_range_map[valid_pixels]
        
        # write lidar_row, lidar_col, valid_points, and lidar_range to addtional tar file 
        range_map_npz[f'{frame_idx}.lidar_row.npz'] = {'arr_0': lidar_row.astype(np.uint8)}  # 0-127
        range_map_npz[f'{frame_idx}.lidar_col.npz'] = {'arr_0': lidar_col.astype(np.uint16)}  # 0-3599
        range_map_npz[f'{frame_idx}.lidar_range.npz'] = {'arr_0': lidar_range.astype(np.float16)}
        
        if intensity_map is not None:
            lidar_intensity = intensity_map[idx][valid_pixels]
            range_map_npz[f'{frame_idx}.lidar_intensity.npz'] = {'arr_0': lidar_intensity.astype(np.float16)}
        
    write_to_tar(range_map_npz, tar_path)


def range_map_to_ray_directions(n_cols, sensor_elevation_angles):
    """
    Generate ray directions for each pixel in a range map.
    
    Args:
        n_cols: number of columns in the range map
        sensor_elevation_angles: list/array of elevation angles in degrees for each scan line
                                 (determines the number of rows)
    
    Returns:
        ray_directions: numpy array of shape (n_rows, n_cols, 3) containing unit vectors
                        for each pixel's ray direction
    """
    # Create arrays for azimuth and elevation angles
    azimuth_angles = np.linspace(np.pi, -np.pi, n_cols, endpoint=False)
    
    # Convert elevation angles from degrees to radians
    elevation_angles_rad = np.radians(sensor_elevation_angles)
    
    # Create meshgrid for all combinations of angles
    elevation_grid, azimuth_grid = np.meshgrid(elevation_angles_rad, azimuth_angles, indexing='ij')
    
    # Convert spherical coordinates to Cartesian coordinates (unit vectors)
    x = np.cos(elevation_grid) * np.cos(azimuth_grid)  # shape: (H, W)
    y = np.cos(elevation_grid) * np.sin(azimuth_grid)
    z = np.sin(elevation_grid)
    
    # Stack to create ray directions
    ray_directions = np.stack([x, y, z], axis=-1)  # shape: (H, W, 3)
    
    return ray_directions
    



def lidar_points_to_range_map_impl(points, n_rows, n_cols, sensor_elevation_angles, max_range=105, intensity=None):
    """
    Convert a LiDAR point cloud to a range map (and optional intensity map).

    Keeps only the nearest point per pixel using a flat index and scatter-min reduction.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) in the sensor frame.
        n_rows (int): Number of elevation rows in the output map.
        n_cols (int): Number of azimuth columns in the output map.
        sensor_elevation_angles (torch.Tensor): Shape (n_rows,), angles in degrees ordered by row.
        max_range (float): Maximum valid range threshold (meters). Defaults to 105.
        intensity (torch.Tensor | None): Optional per-point intensity values aligned with `points`.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - range_map of shape (n_rows, n_cols), dtype float32, on CUDA device
            - intensity_map of shape (n_rows, n_cols), dtype float32, on CUDA device
    """
    
    # Calculate spherical coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    range_values = torch.sqrt(x**2 + y**2 + z**2)
    valid_range = (range_values < max_range) & (range_values > 0)
    
    # Convert to right-hand coordinate system with y-axis at 0° and clockwise rotation
    azimuth = -torch.atan2(y, x) + torch.pi  # Shift azimuth to center 0° in the middle of the image
    col_idx = ((azimuth / (2*torch.pi)) * n_cols).type(torch.int32)
    col_idx = torch.clamp(col_idx, 0, n_cols-1)
    
    
    # Find closest elevation angle index
    elevation = torch.asin(z / range_values)
    elevation_degrees = elevation * 180 / torch.pi
    
    # prefilter the points
    epsilon = 0.1
    min_angle = torch.min(sensor_elevation_angles) - epsilon   
    max_angle = torch.max(sensor_elevation_angles) + epsilon
    valid_elevation = (elevation_degrees >= min_angle) & (elevation_degrees <= max_angle)
    valid_mask = valid_range & valid_elevation
    
    elevation_degrees = elevation_degrees[valid_mask]
    col_idx = col_idx[valid_mask]
    range_values = range_values[valid_mask]
    intensity_map = torch.zeros_like(range_values) if intensity is None else intensity[valid_mask]
   
    row_idx = torch.argmin(torch.abs(elevation_degrees[:, None] - sensor_elevation_angles), dim=1)
    valid_mask = (row_idx >= 0) & (row_idx < n_rows) & \
                (col_idx >= 0) & (col_idx < n_cols)
    
    
    # Apply minimum reduction using scatter_reduce
    flat_scatter_indices = row_idx[valid_mask] * n_cols + col_idx[valid_mask]
    src_range_values = range_values[valid_mask]
    src_intensity_values = intensity_map[valid_mask]

    # 3. Define output size
    output_size = n_rows * n_cols

    min_vals_flat, argmin_in_src = torch_scatter.scatter_min(
        src=src_range_values,
        index=flat_scatter_indices,
        dim=0,
        dim_size=output_size,
        fill_value=torch.inf
    )
    sel = min_vals_flat != torch.inf
    sel_argmin_in_src = argmin_in_src[sel]
    sel_src_intensity_values = src_intensity_values[sel_argmin_in_src]
        
    sel_src_range_values = src_range_values[sel_argmin_in_src]
    row_idx = row_idx[valid_mask][sel_argmin_in_src]
    col_idx = col_idx[valid_mask][sel_argmin_in_src]
    
    range_map = torch.full((n_rows, n_cols), 0, dtype=torch.float32, device=points.device)
    range_map[row_idx, col_idx] = sel_src_range_values
    intensity_map = torch.full((n_rows, n_cols), 0, dtype=torch.float32, device=points.device)
    intensity_map[row_idx, col_idx] = sel_src_intensity_values
    
    return range_map, intensity_map



def undo_motion_compensation_impl(
    xyz: np.ndarray, T_sensor_end_sensor_start: np.ndarray, timestamps_startend_us: list, timestamp_us: np.ndarray
) -> np.ndarray:
    """
    Undo motion compensation to bring rays into the time-dependent sensor frame.

    The input points are assumed to be motion-compensated to the beginning of the spin.

    Args:
        xyz (np.ndarray): Points in sensor space, shape (N, 3).
        T_sensor_end_sensor_start (np.ndarray): Relative pose from end-of-frame to start-of-frame
            in sensor space, shape (4, 4).
        timestamps_startend_us (list[int | float]): [start_timestamp_us, end_timestamp_us].
        timestamp_us (np.ndarray): Per-point timestamps (us), shape (N,).

    Returns:
        np.ndarray: Points after undoing motion compensation, shape (N, 3).
    """
    xyz = to_numpy_array(xyz)
    T_sensor_end_sensor_start = to_numpy_array(T_sensor_end_sensor_start)
    timestamp_us = to_numpy_array(timestamp_us)
    pose_interpolator = ncore_common.PoseInterpolator(
        np.stack([np.eye(4, dtype=np.float32),T_sensor_end_sensor_start]),
        timestamps_startend_us,
    )

    # Note: this interpolation will fail if the point's timestamps are outside of the frame's start/end times - issue dedicated error in that case
    assert (
        (timestamps_startend_us[0] <= timestamp_us).all() and (timestamp_us <= timestamps_startend_us[1]).all()
    ), (
        "undo_motion_compensation_impl: Lidar point timestamps out of frame timestamp bounds - "
        "this is an inconsistency in the dataset's internal data and needs to be fixed at dataset creation time"
    )
    T_sensor_end_sensor_pointtime = pose_interpolator.interpolate_to_timestamps(timestamp_us)

    xyz = ncore_transformations.transform_point_cloud(xyz[:, np.newaxis, :], T_sensor_end_sensor_pointtime).squeeze(1)

    return xyz


def undo_motion_compensation_to_world_points(sensor_model, points_list, pose_list, timestamps_list):
    """
    Undo motion compensation for a list of world-frame point clouds into sensor frame.

    Args:
        sensor_model: Configured NCORE LiDAR model used for shutter timing and angle computation.
        points_list (list[torch.Tensor]): List of tensors (N_i, 3) in world coordinates.
        pose_list (list[list[np.ndarray]]): For each frame i, [pose_start, pose_end] as 4x4 matrices
            (world_T_sensor).
        timestamps_list (list[list[int | float]]): For each frame i, [timestamp_start_us, timestamp_end_us].

    Returns:
        list[torch.Tensor]: List of tensors (N_i', 3) with points in sensor coordinates after undoing
            motion compensation. N_i' corresponds to valid points from the model query.
    """
    pts_undo_motion_compensation_list = []
    for idx, (pose, timestamps) in enumerate(zip(pose_list, timestamps_list)):
        pose_start, pose_end = pose
        timestamp_start, timestamp_end = timestamps
        
        # convert sensor points to world points
        c_world_points = points_list[idx].to(sensor_model.device).float()
        
        results = sensor_model.world_points_to_sensor_angles_shutter_pose(
            world_points=c_world_points,
            T_world_sensor_start=np.linalg.inv(pose_start),
            T_world_sensor_end=np.linalg.inv(pose_end),
            start_timestamp_us=timestamp_start,
            end_timestamp_us=timestamp_end,
            max_iterations=10,
            stop_mean_relative_time_error=1e-4,
            stop_delta_mean_relative_time_error=1e-6,
            return_T_world_sensors=False,
            return_valid_indices=True,
            return_timestamps=True,
            )
        
        valid_indices = results.valid_indices
        valid_points = c_world_points[valid_indices]
        # convert to camera space
        valid_points = transform_points_torch(valid_points, torch.from_numpy(np.linalg.inv(pose_start)).float().to(valid_points.device))
        
        timestamps_us = results.timestamps_us
        
        # now undo motion compensation 
        T_sensor_start_sensor_end = np.linalg.inv(pose_end) @ pose_start
        timestamps_startend_us = [timestamp_start, timestamp_end]
        pts_undo_motion_compensation = undo_motion_compensation_impl(valid_points, T_sensor_start_sensor_end, timestamps_startend_us, timestamps_us)
        pts_undo_motion_compensation = torch.from_numpy(pts_undo_motion_compensation).float()
        pts_undo_motion_compensation_list.append(pts_undo_motion_compensation)
        
    return pts_undo_motion_compensation_list

def project_list_of_points_and_intensity_to_range_map(
    points: List[torch.Tensor],
    device: torch.device,
    n_rows: int,
    n_cols: int,
    sensor_elevation_angles: torch.Tensor,
    intensity: List[torch.Tensor] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Project a list of LiDAR point clouds (and optional intensities) to range/intensity maps.

    Args:
        points (list[torch.Tensor]): List of tensors (N_t, 3) per frame in sensor frame.
        device (torch.device): Target device for computation.
        n_rows (int): Number of elevation rows in the output maps.
        n_cols (int): Number of azimuth columns in the output maps.
        sensor_elevation_angles (torch.Tensor): Shape (n_rows,), elevation angles in degrees.
        intensity (list[torch.Tensor] | None): Optional per-frame per-point intensity tensors aligned
            with `points`. If None, intensity maps are still returned (zeros at valid pixels).

    Returns:
        tuple[np.ndarray, np.ndarray | None]:
            - range_maps: Array (T, n_rows, n_cols) with range values (float32).
            - intensity_maps: Array (T, n_rows, n_cols) with intensity values (float32), or None
              if intensity inputs were not provided.
    """
    range_map_info = []
    intensity_map_info = []
    intensity  = intensity if intensity is not None else [None] * len(points)
    for idx, (frame_points, frame_intensity) in tqdm(enumerate(zip(points, intensity))):
        c_points = frame_points.to(device)
        c_intensity = frame_intensity.to(device) if frame_intensity is not None else None

        range_map_torch, intensity_map_torch = lidar_points_to_range_map_impl(
            c_points,
            n_rows,
            n_cols,
            sensor_elevation_angles,
            intensity=c_intensity
        )
        
        if idx % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
        range_map_info.append(range_map_torch.cpu().numpy())
        intensity_map_info.append(intensity_map_torch.cpu().numpy() if intensity_map_torch is not None else None)
    
    range_maps = np.stack(range_map_info, axis=0)
    intensity_maps = np.stack(intensity_map_info, axis=0) if intensity_map_info[0] is not None else None
    return range_maps, intensity_maps