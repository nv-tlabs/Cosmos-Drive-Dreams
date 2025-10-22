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

import io
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import open3d as o3d
import os
from cosmos_predict1.utils.lidar_rangemap import range_map_to_ray_directions, load_pandar128_elevations, range_map_to_point_cloud

# Configure environment for better Kaleido performance
os.environ["KALEIDO_SCOPE_TIMEOUT"] = "60"  # Increase timeout to 60 seconds

# eye: camera position
# center: view direction
CAMERA_VIEWS = {
    "front_view_1": {
        "eye": {"x": -0.3, "y": 0, "z": 0.2},
        "center": {"x": 0.1, "y": 0, "z": 0},
    },
    "front_view_2": {
        "eye": {"x": -0.1, "y": 0, "z": 0.05},
        "center": {"x": 0.1, "y": 0, "z": 0},
    },
    "front_view_3": {
        "eye": {"x": -0.05, "y": 0, "z": 0.05},
        "center": {"x": 0.1, "y": 0, "z": 0},
    },
    "front_view_4": {
        "eye": {"x": -0.01, "y": 0, "z": 0.01},
        "center": {"x": 0.1, "y": 0, "z": 0},
    },
    "front_view_5": {
        "eye": {"x": -0.12, "y": 0, "z": 0.05},
        "center": {"x": 0.1, "y": 0, "z": 0},
    },
    "front_view_6": {
        "eye": {"x": -0.075, "y": 0, "z": 0.05},
        "center": {"x": 0.075, "y": 0, "z": 0.0},
    },
    "top_down_view_1": {
        "eye": {"x": 0, "y": -0.05, "z": 0.5},
        "center": {"x": 0, "y": -0.05, "z": 0},
    },
}


VIZ_KWARGS = {
    "point_size": 0.3,
    "camera_position": CAMERA_VIEWS["front_view_1"],
    "width": 1280,
    "height": 720,
    "bgcolor": [0.0, 0.0, 0.0],
    "range": 100,
}


def visualize_point_cloud(
    point_cloud,
    colors,
    point_size,
    camera_position,
    width,
    height,
    range=80,
    opacity=1.0,
    bgcolor=(0, 0, 0),
):
    # Create a scatter plot
    trace = go.Scatter3d(
        x=point_cloud[:, 0],
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode="markers",
        marker=dict(
            size=point_size,
            color=colors,
            opacity=opacity,
            line=dict(
                width=0,
            ),
        ),
    )
    color_str = f"rgba({bgcolor[0]},{bgcolor[1]},{bgcolor[2]})"

    fig = go.Figure(data=[trace])
    fig.update_layout(
        scene=dict(
            # xaxis=dict(range=[-80, 80], autorange=False),
            # yaxis=dict(range=[-80, 80], autorange=False),
            # zaxis=dict(range=[-80, 0], autorange=False),
            xaxis=dict(
                range=[-range, range],
                autorange=False,
                showbackground=False,
                showticklabels=False,
                zeroline=False,
                visible=False,
                showgrid=False,  # Ensure grid lines are turned off
            ),
            yaxis=dict(
                range=[-range, range],
                autorange=False,
                showbackground=False,
                showticklabels=False,
                zeroline=False,
                visible=False,
                showgrid=False,  # Ensure grid lines are turned off
            ),
            zaxis=dict(
                range=[-range, range],
                autorange=False,
                showbackground=False,
                showticklabels=False,
                zeroline=False,
                visible=False,
                showgrid=False,  # Ensure grid lines are turned off
            ),
            # aspectmode='data',
            aspectmode="cube",  # Set aspect mode to 'cube'
            camera=camera_position,  # Set camera position
        ),
        paper_bgcolor=color_str,  # Set background color of the plotting area to transparent
        plot_bgcolor=color_str,  # Set plot background to transparent
        margin=dict(l=0, r=0, b=0, t=0),  # Remove margins
    )

    # fig.write_image(output_file, width=width, height=height)
    fig_bytes = fig.to_image(width=width, height=height)

    # fig_bytes = fig.to_image(format="png", width=width, height=height)
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)[:, :, :3]


def vis_point_cloud_to_numpy(point_cloud, colors):
    img = visualize_point_cloud(point_cloud, colors, **VIZ_KWARGS)
    return img


def vis_point_cloud(point_cloud, colors, save_path):
    img = visualize_point_cloud(point_cloud, colors, **VIZ_KWARGS)
    img = Image.fromarray(img)
    img.save(save_path)

def o3d_statistical_filter(pts, nb_neighbors=20, std_ratio=1.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    inlier_cloud = pcd.select_by_index(ind)

    # get the numpy point cloud
    inlier_pts = np.asarray(inlier_cloud.points)
    return inlier_pts



def save_range_map_to_ply(
    range_map_input, save_path
):
    """
    Save the range map to a point cloud
    Input:
        range_map_input: numpy array of shape (H, W), unnormalized range values
    """

    # get the ray directions and ray directions
    input_points = range_map_to_point_cloud(range_map_input)

    n_pts = input_points.shape[0]
    input_colors = np.array([[1, 0.706, 0]] * n_pts, dtype=np.float32)


    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(input_points)
    pcd_ref.colors = o3d.utility.Vector3dVector(input_colors)
    
    o3d.io.write_point_cloud(save_path, pcd_ref)
    
    
def render_range_map_to_point_cloud(range_map_input, range_map_recon, filter_outlier=False):
    """
    Render the range map to a point cloud.
    Input:
        range_map_input: numpy array of shape (N, H, W), unnormalized range values
        range_map_recon: numpy array of shape (N, H, W), unnormalized range values
    Output:
        point_cloud: numpy array of shape (N, 3, H, W)
    """
    # unnormalize the range map
    valid_mask_input = range_map_input > 0
    valid_mask_recon = range_map_recon > 0

    # get the ray directions and ray directions
    elevation_angles = load_pandar128_elevations()
    assert len(elevation_angles) == range_map_input.shape[1]

    ray_directions = range_map_to_ray_directions(range_map_input.shape[2], elevation_angles)  # shape: (H, W, 3)
    ray_origins = np.zeros_like(ray_directions)  # shape: (H, W, 3)

    # get the point cloud
    bs = range_map_input.shape[0]

    vis_list = []
    for batch_idx in range(bs):
        c_range_map_input = range_map_input[batch_idx]  # shape: (H, W)
        c_range_map_recon = range_map_recon[batch_idx]  # shape: (H, W)
        c_ray_directions = ray_directions  # shape: (H, W, 3)
        c_ray_origins = ray_origins  # shape: (H, W, 3)

        c_valid_mask_input = valid_mask_input[batch_idx]
        c_valid_mask_recon = valid_mask_recon[batch_idx]

        input_points = (
            c_ray_origins[c_valid_mask_input]
            + c_ray_directions[c_valid_mask_input] * c_range_map_input[c_valid_mask_input][:, np.newaxis]
        )  # shape: (N, 3)
        recon_points = (
            c_ray_origins[c_valid_mask_recon]
            + c_ray_directions[c_valid_mask_recon] * c_range_map_recon[c_valid_mask_recon][:, np.newaxis]
        )  # shape: (N, 3)

        if filter_outlier:
            recon_points = o3d_statistical_filter(recon_points)

        n_pts = input_points.shape[0]
        input_colors = np.array([[1, 0.706, 0]] * n_pts, dtype=np.float32)
        n_pts = recon_points.shape[0]
        recon_colors = np.array([[0, 0.651, 0.929]] * n_pts, dtype=np.float32)

        input_img = vis_point_cloud_to_numpy(input_points, input_colors)  # shape: (H, W, 3)
        recon_img = vis_point_cloud_to_numpy(recon_points, recon_colors)  # shape: (H, W, 3)

        recon_images = np.concatenate([input_img, recon_img], axis=1)  # shape: (H, 2W, 3)
        vis_list.append(recon_images)

    vis_list = np.array(vis_list)  # shape: (bs, H, 2W, 3)

    return vis_list