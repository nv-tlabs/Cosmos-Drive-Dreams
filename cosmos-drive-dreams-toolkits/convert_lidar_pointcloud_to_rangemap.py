# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of this
# software and related documentation without an express license agreement from NVIDIA
# CORPORATION & AFFILIATES is strictly prohibited.

"""
python convert_lidar_pointcloud_to_rangemap.py
"""


import os
import json
import click
import torch
import numpy as np
import mediapy as media
from tqdm import tqdm

from utils.lidar_dataset import get_mads_dataloader
from utils.lidar_rangemap_utils import (
    undo_motion_compensation_to_world_points,
    init_ncore_sensor_model,
    init_ncore_default_sensor_model,
    project_list_of_points_and_intensity_to_range_map,
    save_range_maps_to_tar,
    range_map_to_ray_directions,
)
from utils.fill_in_road import estimate_road_surface_in_world
from utils.utils import apply_color_map_to_image, colorcode_depth_maps, transform_points_torch


def process_clip(batch, input_root, device, n_rows, n_cols, hdmap_output_root, rangemap_output_root, render_output_root, metadata_output_root, vis, fill_in_road, cmap):
    clip_id = batch['clip_id'][0]  # Get single clip ID from batch
    frame_indices = batch['frame_indices'][0]  # Get frame indices
    bbox_points_list = batch['bbox_points_list'][0]
    static_points = batch['static_points'][0]
    pose_list = batch['pose_list'][0]
    timestamps_list = batch['timestamps_list'][0]
    lidar_points_list = batch['lidar_points_list'][0]
    lidar_intensity_list = batch['intensity_list'][0]
    road_boundaries = batch['road_boundaries'][0]
    lane_points = batch['lane_points'][0]
    assert len(frame_indices) == len(bbox_points_list) == len(pose_list) == len(timestamps_list)
    
    ############################################################
    # init sensor model
    sensor_collection_path = os.path.join(
        input_root, 'lidar_sensor_config', f"{clip_id}.json"
    )
    if not os.path.exists(sensor_collection_path):
        sensor_model = init_ncore_default_sensor_model()
        print(f"Using default sensor model for {clip_id}")
    else:
        sensor_model = init_ncore_sensor_model(sensor_collection_path)
        print(f"Using more accurate sensor model for {clip_id}")
    sensor_elevation_angles = torch.rad2deg(sensor_model.row_elevations_rad)
    
    ############################################################
    # 1. Project lidar points (in sensor frame) to range map and save to tar
    lidar_range_maps, lidar_intensity_maps = project_list_of_points_and_intensity_to_range_map(
        lidar_points_list,
        device,
        n_rows,
        n_cols,
        sensor_elevation_angles,
        intensity=lidar_intensity_list
    )
    
    tar_path = os.path.join(rangemap_output_root, f"{clip_id}.tar")
    save_range_maps_to_tar(
        lidar_range_maps,
        frame_indices,
        clip_id,
        tar_path, 
        intensity_map=lidar_intensity_maps
    )
    
    ############################################################
    # 2. Save the metadata
    metadata_path = os.path.join(metadata_output_root, f"{clip_id}")
    metadata = {
        'pose_list': pose_list,
        'timestamps_list': timestamps_list,
        'frame_indices': frame_indices
    }
    np.savez(metadata_path, **metadata)

    
    ############################################################
    # 3. project hdmap points (in world frame) to the range map. if we directly run 3.2 then it will be super slow.
    if fill_in_road:
        ego_car_points = torch.from_numpy(np.array(pose_list)[:,0, :3, 3]).float()
        lane_points = torch.cat([ego_car_points, lane_points], dim=0).cuda()
        road_surface_points = estimate_road_surface_in_world(
            road_boundaries.cuda(), lane_points, 
            block_size=[100, 100], 
            voxel_sizes=[0.5, 0.5, 0.2],
            fine_voxel_sizes=[0.04, 0.04, 0.2]
        ).cpu()
        static_points = torch.cat([road_surface_points, static_points], dim=0)
        torch.cuda.empty_cache()
    
    hdmap_world_points = [torch.cat([bbox_points_list[i], static_points], dim=0) for i in range(len(bbox_points_list))]
    
    # 3.1 run the projection to reduce the number of points
    hdmap_world_points = [transform_points_torch(ele, torch.from_numpy(np.linalg.inv(pose_list[i][0])).float().to(ele.device)) for i, ele in enumerate(hdmap_world_points)]
    hdmap_range_maps, _ = project_list_of_points_and_intensity_to_range_map(
        hdmap_world_points,
        device,
        n_rows,
        n_cols,
        sensor_elevation_angles
    )
    hdmap_range_maps = torch.from_numpy(hdmap_range_maps).float().to(device)
    ray_directions = range_map_to_ray_directions(n_cols, sensor_elevation_angles.cpu().numpy()) # H, W, 3
    ray_directions = torch.from_numpy(ray_directions).to(device).float()
    hdmap_points = hdmap_range_maps[..., None] * ray_directions[None, ...] # T, H, W, 3
    hdmap_points_list = [hdmap_points[i][hdmap_range_maps[i] > 0] for i in range(hdmap_points.shape[0])]
    
    # 3.2 undo motion compensation to the hdmap points
    hdmap_world_points = [transform_points_torch(ele, torch.from_numpy(pose_list[i][0]).float().to(ele.device)) for i, ele in enumerate(hdmap_points_list)]
    hdmap_world_points = undo_motion_compensation_to_world_points(sensor_model, hdmap_world_points, pose_list, timestamps_list)
    
    # 3.3 project the hdmap points to the range map
    hdmap_range_maps, _ = project_list_of_points_and_intensity_to_range_map(
        hdmap_world_points,
        device,
        n_rows,
        n_cols,
        sensor_elevation_angles
    )
    tar_path = os.path.join(hdmap_output_root, f"{clip_id}.tar")
    save_range_maps_to_tar(
        hdmap_range_maps,
        frame_indices,
        clip_id,
        tar_path
    )
    
    ############################################################
    # 4. save the rendered videos of range map and hdmap
    if vis:
        hdmap_range_maps = torch.from_numpy(hdmap_range_maps)
        lidar_range_maps = torch.from_numpy(lidar_range_maps)
        lidar_intensity_maps = torch.from_numpy(lidar_intensity_maps)
        
        depth_maps = colorcode_depth_maps(torch.cat([lidar_range_maps, hdmap_range_maps], dim=1), cmap=cmap) # B x 3 x H x W
        intensity_maps = apply_color_map_to_image(lidar_intensity_maps, color_map="coolwarm") # B x 3 x H x W
        
        depth_maps = torch.cat([depth_maps, intensity_maps], dim=2)
        depth_maps = (depth_maps * 255 + 0.5).permute(0, 2, 3, 1).numpy().astype(np.uint8) # B x H x W x 3
        
        media.write_video(os.path.join(render_output_root, f"{clip_id}.mp4"), depth_maps, fps=10)

@click.command()
@click.option("--input_root", type=str, default="../assets/example", help="the root folder of the input data")
@click.option("--hdmap_output_root", type=str, default="dump/demo/hdmap", help="the root folder of the hdmap output data")
@click.option("--rangemap_output_root", type=str, default="dump/demo/lidar", help="the root folder of the lidar range map output data")
@click.option("--render_output_root", type=str, default="dump/demo/render", help="the root folder of the rendered output data")
@click.option("--metadata_output_root", type=str, default="dump/demo/metadata", help="the root folder of the metadata output data")
@click.option("--dataset", type=str, default="rds_hq", help=("the dataset name, 'rds_hq' or 'waymo', see the config in settings.json"))
@click.option("--split_filename", type=str, default="assets/lidar_split.txt", help="the filename of the split")
@click.option("--vis", type=bool, default=True, help="whether to visualize the range maps")
@click.option("--fill_in_road", type=bool, default=False, help="whether to fill in the road surface")
@click.option("--cmap", type=str, default="Spectral", help="the colormap to use")


def main(
    input_root: str,
    hdmap_output_root: str,
    rangemap_output_root: str,
    render_output_root: str,
    metadata_output_root: str,
    dataset: str,
    split_filename: str,
    vis: bool,
    fill_in_road: bool,
    cmap: str
):
    # Load settings
    with open(f'config/dataset_{dataset}.json', 'r') as file:
        settings = json.load(file)
    
    # Create output directories
    os.makedirs(hdmap_output_root, exist_ok=True)
    os.makedirs(rangemap_output_root, exist_ok=True)
    os.makedirs(render_output_root, exist_ok=True)
    os.makedirs(metadata_output_root, exist_ok=True)
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataloader
    dataloader, n_rows, n_cols = get_mads_dataloader(
        input_root=input_root,
        settings=settings,
        split_filename=split_filename,
        batch_size=1,
        num_workers=4
    )
    
    # Process each clip
    for batch in tqdm(dataloader, desc="Processing clips"):
        process_clip(batch, input_root, device, n_rows, n_cols, hdmap_output_root, rangemap_output_root, render_output_root, metadata_output_root, vis, fill_in_road, cmap)

        # Clean up GPU memory
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()