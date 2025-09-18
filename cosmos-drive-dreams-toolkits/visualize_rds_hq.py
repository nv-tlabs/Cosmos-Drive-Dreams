# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import click
import numpy as np
import os
import viser
import time
import viser.transforms as tf
import cv2
import json

from pycg import Isometry
from utils.wds_utils import write_to_tar, get_sample
from utils.minimap_utils import simplify_minimap
from utils.animation_utils import FreePoseAnimator, InterpType
from pathlib import Path

MINIMAP_TO_TYPE = json.load(open(Path(__file__).parent / 'config' /'hdmap_type_config.json'))
MINIMAP_TO_RGB = json.load(open(Path(__file__).parent / 'config' /'hdmap_color_config.json'))
DYNAMIC_OBJECT_RGB = [43, 96, 31]

dynamic_sampling_interval = 3
bev_map_h_pixel = 400
bev_map_w_pixel = 400
bev_map_h_meter = 60
bev_map_w_meter = 60

def grounding_pose(input_pose, reference_height=None, convention='flu'):
    """
    change input pose to be grounded.
    Args:
        input_pose: [4,4] matrix
        reference_height: float, optional
            if provided, the grounded pose use this height.
            it should be camera_height (obtained from the first frame) + ground_height (estimate from near lanes)
        convention: 'flu' or 'opencv'

    Returns:
        grounded_pose: [4,4] matrix, camera to world
    """
    if convention == 'flu':
        forward_dir = input_pose[:3, 0]
    elif convention == 'opencv':
        forward_dir = input_pose[:3, 2]
    else:
        raise ValueError(f"Invalid convention: {convention}")
    
    forward_dir_x, forward_dir_y = forward_dir[0], forward_dir[1]
    forward_dir_grounded = np.array([forward_dir_x, forward_dir_y, 0])
    forward_dir_grounded = forward_dir_grounded / np.linalg.norm(forward_dir_grounded)

    up_dir_grounded = np.array([0, 0, 1])

    left_dir_grounded = np.cross(up_dir_grounded, forward_dir_grounded)
    left_dir_grounded = left_dir_grounded / np.linalg.norm(left_dir_grounded)

    # update forward_dir_grounded again
    forward_dir_grounded = np.cross(left_dir_grounded, up_dir_grounded)
    forward_dir_grounded = forward_dir_grounded / np.linalg.norm(forward_dir_grounded)
    
    grounded_pose = np.eye(4)
    if convention == 'flu':
        grounded_pose[:3, 0] = forward_dir_grounded
        grounded_pose[:3, 1] = left_dir_grounded
        grounded_pose[:3, 2] = up_dir_grounded
    elif convention == 'opencv':
        grounded_pose[:3, 0] = -left_dir_grounded
        grounded_pose[:3, 1] = -up_dir_grounded
        grounded_pose[:3, 2] = forward_dir_grounded
    else:
        raise ValueError(f"Invalid convention: {convention}")
    
    grounded_pose[:3, 3] = input_pose[:3, 3]
    if reference_height is not None:
        grounded_pose[2, 3] = reference_height

    return grounded_pose

def get_client_camera_pose_matrix(client):
    client_camera_wxyz = client.camera.wxyz
    client_camera_position = client.camera.position
    client_camera_pose = tf.SE3(np.concatenate([client_camera_wxyz, client_camera_position]))

    return client_camera_pose.as_matrix()

def estimate_ground_height(front_camera_pose, lane_lines):
    """
    Estimate the ground height from the lane lines.

    Use x, y of front camera pose to find the nearest lane line point, 
    retrieve its z value as the ground height.

    Args:
        front_camera_pose: [4, 4]
        lane_lines: [N, 2, 3]

    Returns:
        ground_height: float
    """
    world_ego_xy = front_camera_pose[:2, 3]

    # find the nearest lane line point
    lane_line_points = lane_lines.reshape(-1, 3)
    l2_dist = np.linalg.norm(lane_line_points[:, :2] - world_ego_xy, axis=1)
    nearest_lane_line_point = lane_line_points[np.argmin(l2_dist)]

    ground_height = nearest_lane_line_point[2]
    return ground_height
    

def convert_dir_lwh_to_outline(front_dir, left_dir, up_dir, center_xyz, lwh):
    """
    Create the outline polyline of a bounding box from the direction vectors, lwh, and center.
    
    bbox format:
        ------> heading

       3 ---------------- 0
      /|                 /|
     / |                / |
    2 ---------------- 1  |
    |  |               |  |
    |  7 ------------- |- 4
    | /                | /
    6 ---------------- 5 


    """
    corners = np.array([front_dir * lwh[0] / 2 + left_dir * lwh[1] / 2 + up_dir * lwh[2] / 2,
                        front_dir * lwh[0] / 2 - left_dir * lwh[1] / 2 + up_dir * lwh[2] / 2,
                        -front_dir * lwh[0] / 2 - left_dir * lwh[1] / 2 + up_dir * lwh[2] / 2,
                        -front_dir * lwh[0] / 2 + left_dir * lwh[1] / 2 + up_dir * lwh[2] / 2,
                        front_dir * lwh[0] / 2 + left_dir * lwh[1] / 2 - up_dir * lwh[2] / 2,
                        front_dir * lwh[0] / 2 - left_dir * lwh[1] / 2 - up_dir * lwh[2] / 2,
                        -front_dir * lwh[0] / 2 - left_dir * lwh[1] / 2 - up_dir * lwh[2] / 2,
                        -front_dir * lwh[0] / 2 + left_dir * lwh[1] / 2 - up_dir * lwh[2] / 2])
    corners += center_xyz

    vertex_unicursal = [0,1,2,3,0,4,5,6,7,4,5,1] 

    return corners[vertex_unicursal]

def add_polyline(server, line_segments, name, color, line_width=3):
    """
    convert multiple polylines to a batch of 2-vertices line segments, and add them to the viser server
    """
    server.scene.add_line_segments(
        name=name,
        points=np.asarray(line_segments),
        colors=color,
        line_width=line_width,
    )


def world_coordinate_to_bev_image_coordinate(points_in_world, world_to_ego, bev_map_h_pixel, bev_map_w_pixel, bev_map_h_meter, bev_map_w_meter):
    """
    Convert world coordinate to bev image coordinate

    Args:
        points_in_world: [N, 3]
        world_to_ego: [4,4]
        bev_map_h_pixel: int
        bev_map_w_pixel: int
        bev_map_h_meter: float
        bev_map_w_meter: float
        
    Returns:
        x_coords: [N,]
        y_coords: [N,]
    """
    pixel_per_meter_h = bev_map_h_pixel / bev_map_h_meter
    pixel_per_meter_w = bev_map_w_pixel / bev_map_w_meter
    
    points_in_ego = (world_to_ego @ np.concatenate([points_in_world, np.ones((len(points_in_world), 1))], axis=1).T).T
    points_in_ego = points_in_ego[:,:3] # shape [N, 3]

    front_meter = points_in_ego[:, 0]
    left_meter = points_in_ego[:, 1]

    front_pixel = front_meter * pixel_per_meter_h # shape [N]
    left_pixel = left_meter * pixel_per_meter_w # shape [N]

    # image coordinate system:
    # -----> x 
    # |
    # v y 

    x_coords = (bev_map_w_pixel / 2 - left_pixel) # shape [N, 2], 2 is two points for one line segment
    y_coords = (bev_map_h_pixel / 2 - front_pixel) # shape [N, 2], 2 is two points for one line segment

    return x_coords, y_coords


def draw_bev_projection(gui_image_handler, client_camera_pose, label_to_line_segments, front_camera_pose_interpolator):
    """
    Draw the bev projection of the current frame, update the gui_image_handler.image

    --------------------
    |                  |
    |                  |
    |        ^ (front) |
    |        |         |
    |    <---o         |
    | (left)           |
    |                  |
    |                  |
    --------------------


    """
    label_to_visualize = ['lanelines', 'road_boundaries', 'current_dynamic_objects', 'ego_car']

    bev_map = np.ones((bev_map_h_pixel, bev_map_w_pixel, 3), dtype=np.uint8) * 255

    ego_pose_opencv = client_camera_pose # camera to world
    ego_pose_flu = np.concatenate([ego_pose_opencv[:,2:3], -ego_pose_opencv[:,0:1], -ego_pose_opencv[:,1:2], ego_pose_opencv[:,3:]], axis=1) # x,y,z, forward, left, up

    # we need a ego based on the ground!
    ego_pose_flu_grounded = grounding_pose(ego_pose_flu, convention='flu')

    # get the line segments in the ego frame
    for label_name in label_to_visualize:
        if label_name == 'current_dynamic_objects':
            color_uint8 = DYNAMIC_OBJECT_RGB
        elif label_name == 'ego_car':
            color_uint8 = (255, 0, 0)
        else:
            color_uint8 = MINIMAP_TO_RGB[label_name]
        
        ego_to_world = ego_pose_flu_grounded
        world_to_ego = np.linalg.inv(ego_to_world)

        line_segments_in_world = label_to_line_segments[label_name].reshape(-1, 3) # shape [N*2, 3]
        x_coords, y_coords = world_coordinate_to_bev_image_coordinate(line_segments_in_world, world_to_ego, bev_map_h_pixel, bev_map_w_pixel, bev_map_h_meter, bev_map_w_meter)
        x_coords = x_coords.reshape(-1, 2) # [N, 2]
        y_coords = y_coords.reshape(-1, 2) # [N, 2]


        # for each line segments in N, if all it's image coordinate is out of bound, skip it
        start_point_valid = (x_coords[:, 0] >= 0) & (x_coords[:, 0] < bev_map_w_pixel) & (y_coords[:, 0] >= 0) & (y_coords[:, 0] < bev_map_h_pixel)
        end_point_valid = (x_coords[:, 1] >= 0) & (x_coords[:, 1] < bev_map_w_pixel) & (y_coords[:, 1] >= 0) & (y_coords[:, 1] < bev_map_h_pixel)
        both_valid_line_segments = start_point_valid & end_point_valid

        both_valid_x_coords = x_coords[both_valid_line_segments].astype(np.int32)
        both_valid_y_coords = y_coords[both_valid_line_segments].astype(np.int32)

        for x, y in zip(both_valid_x_coords, both_valid_y_coords):
            cv2.line(bev_map, (x[0], y[0]), (x[1], y[1]), color_uint8, 2)

        # either valid line segment, start_point_valid or end_point_valid is valid. use XOR to get them
        either_valid_line_segments = np.logical_xor(start_point_valid, end_point_valid)
        either_valid_x_coords = x_coords[either_valid_line_segments].astype(np.int32)
        either_valid_y_coords = y_coords[either_valid_line_segments].astype(np.int32)

        for x, y in zip(either_valid_x_coords, either_valid_y_coords):
            cv2.line(bev_map, (x[0], y[0]), (x[1], y[1]), color_uint8, 2)


    cv2.circle(bev_map, (bev_map_h_pixel//2, bev_map_w_pixel//2), 3, (0, 0, 0), -1)

    # draw ego car trajectory
    t_min, t_max = front_camera_pose_interpolator.get_first_t(), front_camera_pose_interpolator.get_last_t()
    if t_min is not None and t_max is not None:
        front_camera_positions = []
        for t in range(t_min, t_max+1):
            front_camera_positions.append(front_camera_pose_interpolator.get_value(t).t)
        
        front_camera_positions = np.asarray(front_camera_positions) # shape [N, 3]
        x_coords, y_coords = world_coordinate_to_bev_image_coordinate(front_camera_positions, world_to_ego, bev_map_h_pixel, bev_map_w_pixel, bev_map_h_meter, bev_map_w_meter)

        valid_coord_idx = (x_coords >= 0) & (x_coords < bev_map_w_pixel) & (y_coords >= 0) & (y_coords < bev_map_h_pixel)
        
        x_coords = x_coords[valid_coord_idx]
        y_coords = y_coords[valid_coord_idx]
        
        x_coords = x_coords.astype(np.int32)
        y_coords = y_coords.astype(np.int32)
        
        for x, y in zip(x_coords, y_coords):
            cv2.circle(bev_map, (x, y), 1, (0, 0, 0), -1)

    gui_image_handler.image = bev_map


@click.command()
@click.option('--input_root', '-i', type=str, required=True,
               help='The root directory of the webdataset')
@click.option('--novel_pose_folder', '-np', type=str, default='novel_pose',
               help='The folder name of the novel pose data. If provided, we will render the novel ego trajectory')
@click.option('--dataset', '-d', type=str, default='rds_hq_mv',
               help='The dataset name, "rds_hq" or "rds_hq_mv" or "waymo"')
@click.option('--clip_id', '-c', type=str, help='clip id to visualize')
def main(input_root, novel_pose_folder, dataset, clip_id):
    server = viser.ViserServer()

    with open(f'config/dataset_{dataset}.json', 'r') as file:
        settings = json.load(file)

    label_to_line_segments = {}
    all_cameras = settings['CAMERAS']
    camera_poses = get_sample(os.path.join(input_root, f"pose/{clip_id}.tar"))
    frame_num = len([x for x in camera_poses.keys() if x.endswith(f'pose.{all_cameras[0]}.npy')])
    camera_height = camera_poses[f'000000.pose.{all_cameras[0]}.npy'][2, 3]

    all_camera_to_front_camera = [
        np.linalg.inv(camera_poses[f'000000.pose.{all_cameras[0]}.npy']) @ camera_poses[f'000000.pose.{camera}.npy']
        for camera in all_cameras
    ]

    ######### Setup GUI #########
    with server.gui.add_folder("Control"):
        gui_frame_slider = server.gui.add_slider(
            "Frame Selector",
            min=0,
            max=frame_num-1, # include frame_num-3.
            step=1,
            initial_value=0,
        )
        gui_frame_auto_play = server.gui.add_checkbox(
            "Auto Play",
            initial_value=False,
        )
        gui_record_button = server.gui.add_button(
            "Record Current Pose",
            hint="Record the camera pose for current frame",
        )
        gui_reset_button = server.gui.add_button(
            "Reset Recorded Poses",
            hint="Reset the recorded camera poses",
        )
        gui_force_grounded_button = server.gui.add_checkbox(
            "Force Pose to be Grounded",
            initial_value=True,
            hint="Force the recorded poses to be grounded" + \
                 "keep a fixed height as initial pose.",
        )
        gui_export_button = server.gui.add_button(
            "Export Poses",
            hint="Export the recorded camera poses for all frames",
        )
        gui_image_handler = server.gui.add_image(
            image = np.ones((bev_map_h_pixel, bev_map_w_pixel, 3), dtype=np.uint8) * 255,
            label="Bev Projection",
        )

    gui_record_button.__setattr__('frame_idx_to_front_camera_pose', {})
    gui_record_button.__setattr__(
        'front_camera_pose_interpolator', 
        FreePoseAnimator(interp_type=InterpType.BEZIER)
    )

    # >>> Setup notification for gui_record_button
    @gui_record_button.on_click
    def _(event: viser.GuiEvent):
        client = event.client
        assert client is not None

        # we record current camera pose.
        gui_record_button.frame_idx_to_front_camera_pose[gui_frame_slider.value] = \
            get_client_camera_pose_matrix(client)

        if gui_force_grounded_button.value:
            # get ground height
            ground_height = estimate_ground_height(
                front_camera_pose = gui_record_button.frame_idx_to_front_camera_pose[gui_frame_slider.value],
                lane_lines = label_to_line_segments['lanelines'],
            )

            gui_record_button.frame_idx_to_front_camera_pose[gui_frame_slider.value] = \
                grounding_pose(
                    gui_record_button.frame_idx_to_front_camera_pose[gui_frame_slider.value], 
                    reference_height=camera_height + ground_height,
                    convention='opencv'
                )

        gui_record_button.front_camera_pose_interpolator.set_keyframe(
            gui_frame_slider.value, 
            Isometry.from_matrix(gui_record_button.frame_idx_to_front_camera_pose[gui_frame_slider.value])
        )

        if hasattr(gui_record_button, 'notification_handle'):
            gui_record_button.notification_handle.remove()

        gui_record_button.__setattr__(
            'notification_handle', 
            client.add_notification(
                title=f"{len(gui_record_button.frame_idx_to_front_camera_pose)}" + \
                    f" camera poses are recorded: {sorted(list(gui_record_button.frame_idx_to_front_camera_pose.keys()))}",
                body=f"Just make sure the first (0) and last frame ({frame_num-1}) are recorded.",
                loading=False,
                with_close_button=True,
                auto_close=False,
                color='yellow',
            )
        )
    
    # >>> Setup gui_reset_button
    @gui_reset_button.on_click
    def _(event: viser.GuiEvent):
        client = event.client
        assert client is not None

        gui_record_button.frame_idx_to_front_camera_pose = {}
        gui_record_button.front_camera_pose_interpolator = FreePoseAnimator(interp_type=InterpType.BEZIER)
        
        client.add_notification(
            title="Recorded poses are reset",
            body="You can record poses again.",
            loading=False,
            with_close_button=True,
            auto_close=3000,
            color='blue',
        )

    # >>> Setup gui_export_button
    @gui_export_button.on_click
    def _(event: viser.GuiEvent):
        client = event.client
        assert client is not None

        # first check if the first frame and the last frame are recorded
        if (0 not in gui_record_button.frame_idx_to_front_camera_pose) or \
            (frame_num - 1 not in gui_record_button.frame_idx_to_front_camera_pose):
            client.add_notification(
                title="Error",
                body=f"Please record the first (0) and last ({frame_num-1}) frames.",
                loading=False,
                with_close_button=True,
                auto_close=3000,
                color='red',
            )
            return

        # get interpolated poses
        # ! in the final exporting, we make the orientation points to the driving direction
        interpolated_poses_sample = {"__key__": clip_id}
        for frame_idx in range(frame_num):
            interpolated_poses_sample[f'{frame_idx:06d}.pose.camera_front_wide_120fov.npy'] = \
                gui_record_button.front_camera_pose_interpolator.get_value(frame_idx).matrix
            
            # we also compute poses for other cameras
            for camera in all_cameras:
                if camera == 'camera_front_wide_120fov':
                    continue

                interpolated_poses_sample[f'{frame_idx:06d}.pose.{camera}.npy'] = \
                    interpolated_poses_sample[f'{frame_idx:06d}.pose.camera_front_wide_120fov.npy'] @ all_camera_to_front_camera[all_cameras.index(camera)]

        # save interpolated poses
        write_to_tar(interpolated_poses_sample, os.path.join(input_root, novel_pose_folder, f"{clip_id}.tar"))

        client.add_notification(
            title="Success",
            body=f"Interpolated poses are saved!",
            loading=False,
            with_close_button=True,
            auto_close=3000,
            color='green',
        )

    """
    Start to visualize the webdataset.
    """
    #### Add static attributes ####
    for label_name in settings['MINIMAP_TYPES']:
        if MINIMAP_TO_TYPE[label_name] in ['polyline', 'polygon']:
            minimap_wds_file = os.path.join(input_root, f"3d_{label_name}", f"{clip_id}.tar")

            # minimap_data_wo_meta_info is a list of vertices for polyline or polygon
            minimap_data_wo_meta_info, minimap_name = simplify_minimap(minimap_wds_file)

            # convert minimap_data_wo_meta_info to line segments
            line_segments = []
            for vertices in minimap_data_wo_meta_info:
                line_segments.extend(list(zip(vertices[:-1], vertices[1:])))

            # add minimap_data_wo_meta_info back to static_visualization_config
            label_to_line_segments[label_name] = np.asarray(line_segments)
            if len(line_segments) > 0:
                add_polyline(server, line_segments, label_name, MINIMAP_TO_RGB[label_name])
        else:
            print(f"Label {label_name} is not visualized here.")

        
    #### Add dynamic object attributes ####
    frame_idx_to_dynamic_object_handler = {}
    frame_idx_to_dynamic_object_line_segments = {}
    frame_idx_to_ego_car_line_segments = {}
    all_object_info_all_frames = get_sample(os.path.join(input_root, f"all_object_info/{clip_id}.tar"))

    for frame_idx in range(0, frame_num, dynamic_sampling_interval):
        all_object_this_frame = all_object_info_all_frames[f'{frame_idx:06d}.all_object_info.json']
        line_segments_this_frame = []

        tracking_ids = all_object_this_frame.keys()
        
        # 1) create line segments for dynamic objects
        for tracking_id in tracking_ids:
            # get the dynamic object shape and pose info
            object_info = all_object_this_frame[tracking_id]
            object_to_world = np.array(object_info['object_to_world'])
            object_lwh = np.array(object_info['object_lwh'])

            center_xyz = object_to_world[:3, 3]
            heading_dir = object_to_world[:3, 0]
            left_dir = object_to_world[:3, 1]
            up_dir = object_to_world[:3, 2]
            
            # convert the dynamic object shape and pose info to line segments
            outline = convert_dir_lwh_to_outline(heading_dir, left_dir, up_dir, center_xyz, object_lwh) # shape [12, 3]
            line_segments_this_frame.extend(list(zip(outline[:-1], outline[1:])))

        # if empty, create a fake line segments
        if len(line_segments_this_frame) == 0:
            line_segments_this_frame = np.ones((1, 2, 3)) * 1e6

        # add the line segments to the viser server
        frame_idx_to_dynamic_object_handler[frame_idx] = server.scene.add_line_segments(
            name=f'dynamic_object at frame {frame_idx}',
            points=np.asarray(line_segments_this_frame),
            colors=DYNAMIC_OBJECT_RGB,
            line_width=6,
            visible=False,
        )

        frame_idx_to_dynamic_object_line_segments[frame_idx] = np.asarray(line_segments_this_frame)


    # 2) create line segments for ego car, every frame
    for frame_idx in range(0, frame_num):
        ego_car_position = camera_poses[f'{frame_idx:06d}.pose.{all_cameras[0]}.npy'][:3,3]
        ego_car_forward_dir = camera_poses[f'{frame_idx:06d}.pose.{all_cameras[0]}.npy'][:3,2]
        frame_idx_to_ego_car_line_segments[frame_idx] = np.stack([
            ego_car_position - ego_car_forward_dir * 1,
            ego_car_position + ego_car_forward_dir * 1,
        ], axis=0).reshape(1, 2, 3)

    #### Set initial camera view based on ego car's first position ####
    def set_initial_camera_view():
        """Set initial camera view to look at the ego car's first position"""
        # Use ego car's first position (frame 0)
        if 0 in frame_idx_to_ego_car_line_segments:
            ego_car_first_position = camera_poses[f'000000.pose.{all_cameras[0]}.npy'][:3, 3]
            ego_car_first_forward_dir = camera_poses[f'000000.pose.{all_cameras[0]}.npy'][:3, 2]
            
            # Set camera position above and behind the ego car's first position
            camera_position = ego_car_first_position + np.array([0, 0, 15]) - ego_car_first_forward_dir * 10
            look_at_point = ego_car_first_position
            
            # Set up direction (z-axis up)
            up_direction = np.array([0, 0, 1])
            
            print(f"Setting initial camera view based on ego car first position:")
            print(f"  Ego car position: {ego_car_first_position}")
            print(f"  Camera position: {camera_position}")
            print(f"  Look at point: {look_at_point}")
            
            # Apply to all current clients
            for client_id, client in server.get_clients().items():
                client.camera.position = camera_position
                client.camera.look_at = look_at_point
                client.camera.up_direction = up_direction
                client.camera.fov = np.pi / 3  # 60 degrees
        else:
            print("No ego car position found, using default camera view")

    # Set initial view for existing clients
    set_initial_camera_view()
    
    # Set initial view for new clients that connect later
    @server.on_client_connect
    def on_new_client_connect(client: viser.ClientHandle):
        # Use ego car's first position for new clients
        if 0 in frame_idx_to_ego_car_line_segments:
            ego_car_first_position = camera_poses[f'000000.pose.{all_cameras[0]}.npy'][:3, 3]
            ego_car_first_forward_dir = camera_poses[f'000000.pose.{all_cameras[0]}.npy'][:3, 2]
            
            camera_position = ego_car_first_position + np.array([0, 0, 15]) - ego_car_first_forward_dir * 10
            look_at_point = ego_car_first_position
            up_direction = np.array([0, 0, 1])
            
            # Small delay to ensure client is ready
            time.sleep(0.1)
            client.camera.position = camera_position
            client.camera.look_at = look_at_point
            client.camera.up_direction = up_direction
            client.camera.fov = np.pi / 3

    """
    while-true loop for visualization
    """
    prev_frame_idx = 0
    while True:
        current_frame_idx = gui_frame_slider.value
        current_frame_idx_divisible_by_3 = current_frame_idx // 3 * 3
        frame_idx_to_dynamic_object_handler[current_frame_idx_divisible_by_3].visible = True
        label_to_line_segments['current_dynamic_objects'] = frame_idx_to_dynamic_object_line_segments[current_frame_idx_divisible_by_3]
        label_to_line_segments['ego_car'] = frame_idx_to_ego_car_line_segments[current_frame_idx]

        # create frustum for front camera at this frame
        camera_pose = camera_poses[f'{current_frame_idx:06d}.pose.{all_cameras[0]}.npy'] # [4,4] camera to world. opencv convention
        camera_wxyz = tf.SE3.from_matrix(camera_pose).wxyz_xyz[:4]
        camera_position = camera_pose[:3,3]
        server.scene.add_camera_frustum(
            name='front_camera',
            fov=60 / 180 * np.pi,
            aspect=16 / 9,
            color = np.array([1.0, 0, 0]),
            wxyz = camera_wxyz,
            position = camera_position,
            line_width = 5,
            scale = 0.8
        )

        # create bev projection for current frame
        for id, client in server.get_clients().items():
            draw_bev_projection(
                gui_image_handler, 
                get_client_camera_pose_matrix(client), 
                label_to_line_segments,
                gui_record_button.front_camera_pose_interpolator,
            )
        
        if current_frame_idx_divisible_by_3 != prev_frame_idx:
            frame_idx_to_dynamic_object_handler[prev_frame_idx].visible = False
            prev_frame_idx = current_frame_idx_divisible_by_3

        if gui_frame_auto_play.value:
            gui_frame_slider.value = (gui_frame_slider.value + dynamic_sampling_interval) % frame_num
            time.sleep(1/3)

        time.sleep(1/10)

if __name__ == '__main__':
    main()
