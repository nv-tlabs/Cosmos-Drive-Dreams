# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import json

from pathlib import Path
from scipy.spatial.transform import Rotation as R
from termcolor import cprint
from utils.wds_utils import get_sample

MINIMAP_TO_RGB = json.load(open(Path(__file__).parent.parent / 'config' /'hdmap_color_config.json'))

MINIMAP_TO_TYPE = json.load(open(Path(__file__).parent.parent / 'config' /'hdmap_type_config.json'))

MINIMAP_TO_SEMANTIC_LABEL = {
    'lanelines': 5,
    'lanes': 5,
    'poles': 9,
    'road_boundaries': 5,
    'wait_lines': 10,
    'crosswalks': 5,
    'road_markings': 10,
}

def extract_vertices(minimap_data, vertices_list=None):
    if vertices_list is None:
        vertices_list = []

    if isinstance(minimap_data, dict):
        for key, value in minimap_data.items():
            if key == 'vertices':
                vertices_list.append(value)
            else:
                extract_vertices(value, vertices_list)

    elif isinstance(minimap_data, list):
        for item in minimap_data:
            extract_vertices(item, vertices_list)

    return vertices_list

def get_type_from_name(minimap_name):
    """
    Args:
        minimap_name: str, name of the minimap
    
    Returns:
        minimap_type: str, type of the minimap
    """
    if minimap_name in MINIMAP_TO_TYPE:
        return MINIMAP_TO_TYPE[minimap_name]
    else:
        raise ValueError(f"Invalid minimap name: {minimap_name}")


def cuboid3d_to_polyline(cuboid3d_eight_vertices):
    """
    Convert cuboid3d to polyline

    Args:
        cuboid3d_eight_vertices: np.ndarray, shape (8, 3), dtype=np.float32, 
            eight vertices of the cuboid3d
    
    Returns:
        polyline: np.ndarray, shape (N, 3), dtype=np.float32, 
            polyline vertices
    """
    if isinstance(cuboid3d_eight_vertices, list):
        cuboid3d_eight_vertices = np.array(cuboid3d_eight_vertices)

    connected_vertices_indices = [0,1,2,3,0,4,5,6,7,4,5,1,2,6,7,3]
    connected_polyline = np.array(cuboid3d_eight_vertices)[connected_vertices_indices]

    return connected_polyline


def simplify_minimap(minimap_wds_file):
    """
    Args:
        minimap_wds_file: path to the webdataset file containing minimap data.
        Note that cuboid3d are converted to polylines!!

    Returns:
        minimap_data_wo_meta_info: list of list of 3d points
            containing extracted minimap data, it represents a polyline or polygon
            [[[x, y, z], [x, y, z], ...]], 
            [[[x, y, z], [x, y, z], ...]], ...]
    
        -> minimap can be polygons, e.g., crosswalks, road_markings
        -> minimap can be polylines, e.g., lanelines, lanes, road_boundaries, wait_lines, poles
        -> minimap can be cuboid3d, e.g., traffic_signs, traffic_lights
    """

    minimap_raw_data = get_sample(minimap_wds_file)
    minimap_key_name = [key for key in minimap_raw_data.keys() if key.endswith('.json')][0]
    minimap_data = minimap_raw_data[minimap_key_name]
    minimap_data_wo_meta_info = extract_vertices(minimap_data)
    minimap_name = minimap_key_name.split('.')[0]

    # close the polygon
    if get_type_from_name(minimap_name) == 'polygon':
        for single_polygon in minimap_data_wo_meta_info:
            single_polygon.append(single_polygon[0])

    # 8 vertices, we can also make it polyline, just repeat some edges!
    if get_type_from_name(minimap_name) == 'cuboid3d':
        connected_vertices_indices = [0,1,2,3,0,4,5,6,7,4,5,1,2,6,7,3]
        for i, eight_vertices in enumerate(minimap_data_wo_meta_info):
            connected_polyline = cuboid3d_to_polyline(eight_vertices)
            minimap_data_wo_meta_info[i] = connected_polyline

    # for each polyline, if they are list, convert them to np.ndarray
    for i, polyline in enumerate(minimap_data_wo_meta_info):
        if isinstance(polyline, list):
            minimap_data_wo_meta_info[i] = np.array(polyline)

    return minimap_data_wo_meta_info, minimap_name


def create_minimap_projection(
    minimap_name,
    minimap_data_wo_meta_info,
    camera_poses,
    camera_model
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
    image_height, image_width = camera_model.height, camera_model.width
    # cprint(f"Processing minimap {minimap_name} with shape {image_height}x{image_width}", 'green')

    minimap_type = get_type_from_name(minimap_name)


    if minimap_type == 'polygon':
        projection_images = camera_model.draw_hull_depth(
            camera_poses,
            minimap_data_wo_meta_info,
            colors=np.array(MINIMAP_TO_RGB[minimap_name]),
        )
    elif minimap_type == 'polyline':
        if minimap_name == 'lanelines' or minimap_name == 'road_boundaries':
            segment_interval = 0.8
        else:
            segment_interval = 0

        projection_images = camera_model.draw_line_depth(
            camera_poses,
            minimap_data_wo_meta_info,
            colors=np.array(MINIMAP_TO_RGB[minimap_name]),
            segment_interval=segment_interval,
        )
    elif minimap_type == 'cuboid3d':
        projection_images = camera_model.draw_hull_depth(
            camera_poses,
            minimap_data_wo_meta_info,
            colors=np.array(MINIMAP_TO_RGB[minimap_name]),
        )
    else:
        raise ValueError(f"Invalid minimap type: {minimap_type}")
    
    return projection_images


def cuboid3d_update_vertices_remove_others(cuboid3d):
    """
    This is specific to the cuboid3d label of traffic_signs and traffic_lights. 

    Args:
        cuboid3d: dict
            {
                'center': {'x': 180.09763, 'y': 83.11682, 'z': -16.458818},  
                'orientation': {'x': -0.071179695, 'y': 0.0033680226, 'z': -0.36912337}, # yaw, pitch, roll
                'dimensions': {'x': 0.01, 'y': 0.64360946, 'z': 0.7865788}, # length, width, height
                'vertices': [{}, {}, {}, {}, {}, {}, {}, {}] # usually missing, 
            }

    Returns:
        updated cuboid3d: dict
            remove 'center', 'orientation', 'dimensions'
            add 'vertices' with 8 vertices
    """
    max_xyz = np.array([cuboid3d['dimensions']['x'] / 2.0, cuboid3d['dimensions']['y'] / 2.0, cuboid3d['dimensions']['z'] / 2.0])
    min_xyz = -max_xyz

    # just a kind reminder, the order of 8 vertices here is different from build_cuboid_bounding_box() in bbox_utils.py
    # but it does not matter.
    vertices_of_cuboid = np.zeros((8, 3), dtype=np.float32)
    vertices_of_cuboid[0] = np.array([min_xyz[0], min_xyz[1], min_xyz[2]])
    vertices_of_cuboid[1] = np.array([min_xyz[0], max_xyz[1], min_xyz[2]])
    vertices_of_cuboid[2] = np.array([max_xyz[0], max_xyz[1], min_xyz[2]])
    vertices_of_cuboid[3] = np.array([max_xyz[0], min_xyz[1], min_xyz[2]])
    vertices_of_cuboid[4] = np.array([min_xyz[0], min_xyz[1], max_xyz[2]])
    vertices_of_cuboid[5] = np.array([min_xyz[0], max_xyz[1], max_xyz[2]])
    vertices_of_cuboid[6] = np.array([max_xyz[0], max_xyz[1], max_xyz[2]])
    vertices_of_cuboid[7] = np.array([max_xyz[0], min_xyz[1], max_xyz[2]])

    yaw, pitch, roll = cuboid3d['orientation']['x'], cuboid3d['orientation']['y'], cuboid3d['orientation']['z']
    rotation_matrix = R.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()

    # Apply rotation. 
    vertices_of_cuboid = (rotation_matrix @ vertices_of_cuboid.T).T

    # Add center translation
    center = np.array([cuboid3d['center']['x'], cuboid3d['center']['y'], cuboid3d['center']['z']])
    vertices_of_cuboid = vertices_of_cuboid + center

    # following protobuf format
    cuboid3d['vertices'] = [{'x': vertex[0], 'y': vertex[1], 'z': vertex[2]} for vertex in vertices_of_cuboid.tolist()]
    cuboid3d.pop('center', None)
    cuboid3d.pop('orientation', None)
    cuboid3d.pop('dimensions', None)

    return cuboid3d 


def convert_cuboid3d_recursive(minimap_message):
    """
    Recursively traverse minimap_message and update any cuboid3d entries with vertices.
    
    Args:
        minimap_message: dict/list, potentially nested structure that may contain cuboid3d entries
        
    Returns:
        Updated minimap_message with cuboid3d entries converted to include vertices
    """
    if isinstance(minimap_message, dict):
        # If this is a cuboid3d dict, update it
        if 'cuboid3d' in minimap_message:
            minimap_message['cuboid3d'] = cuboid3d_update_vertices_remove_others(minimap_message['cuboid3d'])
        # Recursively process all dict values
        for key in minimap_message:
            minimap_message[key] = convert_cuboid3d_recursive(minimap_message[key])
            
    elif isinstance(minimap_message, list):
        # Recursively process all list items
        minimap_message = [convert_cuboid3d_recursive(item) for item in minimap_message]
        
    return minimap_message


def transform_decoded_label(decoded_label, transformation_matrix):
    """
    Apply transformation matrix to decoded label

    Args:
        decoded_label: dict, 
            decoded label from decode_static_label, can have several hierarchies, 
            but the last one is always numpy array with shape [N, 3]
        transformation_matrix: 4x4 numpy array, 
            transformation matrix we want to apply to the numpy array
    Returns:
        dict: transformed decoded_label with the same structure, but numpy array transformed
    """

    def transform_vertices(vertices, transformation_matrix):
        """
        Args:
            vertices: numpy array, shape [N, 3]
            transformation_matrix: 4x4 numpy array
        Returns:
            numpy array, shape [N, 3]
        """
        if vertices.shape == (0,):
            return vertices

        # add 1 to the vertices
        vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        transformed_vertices = np.dot(transformation_matrix, vertices.T).T
        return transformed_vertices[:, :3]
    
    # recursively apply transformation matrix to the vertices
    def transform(decoded_label, transformation_matrix):
        if isinstance(decoded_label, np.ndarray):
            return transform_vertices(decoded_label, transformation_matrix)
        elif isinstance(decoded_label, dict):
            transformed_label = {}
            for key, value in decoded_label.items():
                transformed_label[key] = transform(value, transformation_matrix)
            return transformed_label
        else:
            raise ValueError(f"Unknown type in decoded_label: {type(decoded_label)}")
        
    return transform(decoded_label, transformation_matrix)
