# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all
# intellectual property and proprietary rights in and to this software,
# related documentation and any modifications thereto. Any use,
# reproduction, disclosure or distribution of this software and related
# documentation without an express license agreement from NVIDIA
# CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
from typing import List
from utils.minimap_utils import get_type_from_name
from utils.bbox_utils import CLASS_COLORS, build_cuboid_bounding_box, cuboid3d_to_polyline


def sample_points_along_line(start: np.ndarray, end: np.ndarray, density: float = 0.02) -> np.ndarray:
    """
    Sample points along a line segment with given density
    
    Args:
        start: (3,) array of start point
        end: (3,) array of end point
        density: distance between sampled points in meters
    
    Returns:
        points: (N, 3) array of sampled points
    """
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-2:
        return np.zeros((0, 3))
    num_points = max(2, int(np.ceil(length / density)))
    
    t = np.linspace(0, 1, num_points)
    points = start[None, :] + t[:, None] * direction[None, :]
    
    return points


def sample_points_in_cylinder(start: np.ndarray, end: np.ndarray, density: float = 0.02, radius: float = 0.1) -> np.ndarray:
    """
    Sample points in a cylinder along a line segment with given density (vectorized version)
    
    Args:
        start: (3,) array of start point
        end: (3,) array of end point
        radius: radius of the cylinder in meters
        density: approximate distance between sampled points in meters
    
    Returns:
        points: (N, 3) array of sampled points
    """
    # Calculate direction and length of the line
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-2:
        return np.zeros((0, 3))
    unit_direction = direction / length
    
    # Determine number of points along the axis and around the circle
    num_axial_points = max(2, int(np.ceil(length / density)))
    num_radial_points = max(4, int(np.ceil(2 * np.pi * radius / density)))
    
    # Generate axial points (shape: num_axial_points x 3)
    t = np.linspace(0, 1, num_axial_points)
    axial_points = start[None, :] + t[:, None] * direction[None, :]
    
    # Find perpendicular vectors
    if np.abs(unit_direction[0]) < 0.9:
        perpendicular = np.array([1.0, 0.0, 0.0])
    else:
        perpendicular = np.array([0.0, 1.0, 0.0])
    v1 = np.cross(unit_direction, perpendicular)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(unit_direction, v1)
    
    # Generate angles (shape: num_radial_points)
    angles = np.linspace(0, 2*np.pi, num_radial_points, endpoint=False)
    
    # Generate circle offsets (shape: num_radial_points x 3)
    circle_offsets = radius * (np.cos(angles)[:, None] * v1[None, :] + 
                             np.sin(angles)[:, None] * v2[None, :])
    
    # Add circle offsets to each axial point
    # Reshape axial_points to (num_axial_points, 1, 3)
    # Reshape circle_offsets to (1, num_radial_points, 3)
    # Result shape will be (num_axial_points, num_radial_points, 3)
    points = axial_points[:, None, :] + circle_offsets[None, :, :]
    
    # Reshape to (N, 3) where N = num_axial_points * num_radial_points
    points = points.reshape(-1, 3)
    
    return points



def extract_points_from_polyline(polylines: List[np.ndarray], density: float = 0.02, radius: float = 0.05) -> np.ndarray:
    """
    Extract points from polylines
    
    Args:
        polylines: list of polylines, each is (N, 3) array
        density: sampling density for points along lines
    
    Returns:
        points: (N, 3) array of sampled points
    """
    # 1. read all the polylines
    all_points = []
    for polyline in polylines:
        if len(polyline) < 2:
            continue
        if isinstance(polyline, list):
            polyline = np.array(polyline)

        # Sample dense points along the polyline
        for i in range(len(polyline) - 1):
            # points = sample_points_along_line(
            #     polyline[i],
            #     polyline[i + 1],
            #     density
            # )
            points = sample_points_in_cylinder(
                polyline[i],
                polyline[i + 1],
                density,
                radius
            )
            
            all_points.append(points)
            
    if not all_points:
        return np.zeros((0, 3))

    # 2. Combine all sampled points
    points = np.concatenate(all_points, axis=0)
    return points
    


def extract_points_from_hull(hulls: List[np.ndarray], density: float = 0.02) -> np.ndarray:
    """
    Extract points from hull boundaries and interior
    
    Args:
        hulls: list of hulls, each is (N, 3) array
        density: sampling density for points
        radius: radius for cylinder around boundary
    
    Returns:
        points: (M, 3) array of sampled points
    """
    from scipy.spatial import Delaunay
    
    all_points = []
    for hull in hulls:
        if len(hull) < 3:
            continue
        if isinstance(hull, list):
            hull = np.array(hull)
            
        # First sample points along boundaries using cylinders
        for i in range(len(hull)):
            boundary_points = sample_points_along_line(
                hull[i],
                hull[(i + 1) % len(hull)],
                density
            )
            all_points.append(boundary_points)
            
        # Project points onto their best-fitting plane
        # Find the normal of the plane
        centroid = np.mean(hull, axis=0)
        hull_centered = hull - centroid
        _, _, vh = np.linalg.svd(hull_centered)
        normal = vh[2]  # Third singular vector is normal to best-fitting plane
        
        # Create coordinate system in the plane
        if np.abs(normal[0]) < 0.9:
            v1 = np.array([1.0, 0.0, 0.0])
        else:
            v1 = np.array([0.0, 1.0, 0.0])
        v1 = v1 - np.dot(v1, normal) * normal
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        
        # Project hull vertices onto plane
        hull_2d = np.stack([
            np.dot(hull_centered, v1),
            np.dot(hull_centered, v2)
        ], axis=1)
        
        # Triangulate the hull
        tri = Delaunay(hull_2d)
        
        # Calculate bounding box of 2D projection
        min_coords = np.min(hull_2d, axis=0)
        max_coords = np.max(hull_2d, axis=0)
        
        # Create grid of points
        x = np.arange(min_coords[0], max_coords[0], density)
        y = np.arange(min_coords[1], max_coords[1], density)
        xx, yy = np.meshgrid(x, y)
        points_2d = np.stack([xx.ravel(), yy.ravel()], axis=1)
        
        # Find points inside the triangulation
        inside = tri.find_simplex(points_2d) >= 0
        points_2d = points_2d[inside]
        
        # Convert back to 3D
        points_3d = (centroid + 
                    points_2d[:, 0:1] * v1[None, :] + 
                    points_2d[:, 1:2] * v2[None, :])
        
        all_points.append(points_3d)
    
    if not all_points:
        return np.zeros((0, 3))
        
    # Combine all sampled points
    points = np.concatenate(all_points, axis=0)
    return points


def extract_points_from_hull_lines(hulls: List[np.ndarray], density: float = 0.02, radius: float = 0.05) -> np.ndarray:
    """
    Extract points from hull boundaries
    
    Args:
        hulls: list of hulls, each is (N, 3) array
        density: sampling density for points along the hull edges
    
    Returns:
        points: (M, 3) array of sampled points
    """
    all_points = []
    for hull in hulls:
        if len(hull) < 3:
            continue
        if isinstance(hull, list):
            hull = np.array(hull)
        
        # Sample points along hull edges
        for i in range(len(hull)):
            # Connect from current vertex to next vertex (with wrapping)
            # points = sample_points_along_line(
            #     hull[i],
            #     hull[(i + 1) % len(hull)],
            #     density
            # )
            points = sample_points_in_cylinder(
                hull[i],
                hull[(i + 1) % len(hull)],
                density,
                radius
            )
            
            all_points.append(points)
    
    if not all_points:
        return np.zeros((0, 3))
        
    # Combine all sampled points
    points = np.concatenate(all_points, axis=0)
    return points



def get_minimap_points(
    minimap_name,
    minimap_data_wo_meta_info,
    density: float = 0.02,
    radius: float = 0.05
):
    """
    Args:
        minimap_name: str, name of the minimap
        minimap_data_wo_meta_info: list of np.ndarray, results from simplify_minimap
        camera_poses: np.ndarray, shape (N, 4, 4), dtype=np.float32, camera poses of N frames
        camera_model: CameraModel, camera model
            
    Returns:
        minimaps_projection: np.ndarray, 
            shape (N, H, W, 3), dtype=np.uint8, projected minimap data across N frames
    """
    minimap_type = get_type_from_name(minimap_name)

    # 1. extract points
    if minimap_type == 'polygon':
        # extract points
        points = extract_points_from_hull(minimap_data_wo_meta_info, density)
        
    elif minimap_type == 'polyline':
        points = extract_points_from_polyline(minimap_data_wo_meta_info, density, radius)
        
    elif minimap_type == 'cuboid3d':
        points = extract_points_from_hull_lines(minimap_data_wo_meta_info, density, radius)
    else:
        raise ValueError(f"Invalid minimap type: {minimap_type}")
    
    return points


def get_bbox_points(all_object_info, frame_indices, density=0.02, radius=0.05):
    """
    Create a projection of bounding boxes on the minimap.
    Args:
        all_object_info: dict, containing all object info
        camera_poses: np.ndarray, shape (N, 4, 4), dtype=np.float32, camera to world transformation matrix
        valid_frame_ids: list[int], valid frame ids
        draw_heading: bool, whether to draw heading on the bounding boxes
        diff_color: bool, whether to use different colors for dynamic and static objects

    Returns:
        np.ndarray, shape (N, H, W, 3), dtype=np.uint8, projected bounding boxes on canvas
    """
    bbox_points_list = []
    for idx, frame_id in enumerate(frame_indices):
        current_object_info = all_object_info[f"{frame_id}.all_object_info.json"]

        polylines_cars = []
        polylines_trucks = []
        polylines_pedestrians = []
        polylines_cyclists = []
        polylines_others = []

        # sort tracking ids. avoid jittering when drawing bbox.
        tracking_ids = list(current_object_info.keys())
        tracking_ids.sort()

        # load the polylines
        for tracking_id in tracking_ids:
            object_info = current_object_info[tracking_id]

            if object_info['object_type'] not in CLASS_COLORS:
                if object_info['object_type'] == "Bus":
                    object_info['object_type'] = "Truck"
                elif object_info['object_type'] == 'Vehicle':
                    object_info['object_type'] = "Car"
                else:
                    object_info['object_type'] = "Others"

            object_to_world = np.array(object_info['object_to_world'])
            object_lwh = np.array(object_info['object_lwh'])
            cuboid_eight_vertices = build_cuboid_bounding_box(object_lwh[0], object_lwh[1], object_lwh[2], object_to_world)
            polyline = cuboid3d_to_polyline(cuboid_eight_vertices)

            # draw by the object type
            if object_info['object_type'] == "Car":
                polylines_cars.append(polyline)
            elif object_info['object_type'] == "Truck":
                polylines_trucks.append(polyline)
            elif object_info['object_type'] == "Pedestrian":
                polylines_pedestrians.append(polyline)
            elif object_info['object_type'] == "Cyclist":
                polylines_cyclists.append(polyline)
            else:
                polylines_others.append(polyline)
    
        # draw the polylines
        polylines = []
        polylines.extend(polylines_cars)
        polylines.extend(polylines_trucks)
        polylines.extend(polylines_pedestrians)
        polylines.extend(polylines_cyclists)
        polylines.extend(polylines_others)
        
        points = extract_points_from_polyline(polylines, density, radius)
        bbox_points_list.append(points)
        
    return bbox_points_list