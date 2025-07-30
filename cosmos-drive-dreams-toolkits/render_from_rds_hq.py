# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import numpy as np
import click
import imageio as imageio_v1
import torch
import cv2
import json
import decord
import ray

from tqdm import tqdm
from pathlib import Path
from termcolor import cprint, colored
from pathlib import Path
from utils.wds_utils import get_sample
from utils.bbox_utils import create_bbox_projection, interpolate_bbox, fix_static_objects
from utils.minimap_utils import create_minimap_projection, simplify_minimap
from utils.pcd_utils import batch_move_points_within_bboxes_sparse, forward_warp_multiframes_sparse_depth_only
from utils.camera.pinhole import PinholeCamera
from utils.camera.ftheta import FThetaCamera
from utils.ray_utils import ray_remote, wait_for_futures

USE_RAY = True

def get_nearest_lidar_indices(high_indices, step=3):
    low_indices = []
    low_indices_times_step = []
    for hi in high_indices:
        # Round to the nearest multiple of 'step'
        low_idx = int(round(hi / step))
        low_idx_times_step = low_idx * step
        low_indices.append(low_idx)
        low_indices_times_step.append(low_idx_times_step)
    return low_indices_times_step, low_indices


def prepare_input(input_root, clip_id, settings, camera_type, post_training, resize_resolution, novel_pose_folder):
    """
    Prepare input for rendering.
    Args:
        input_root: the root folder of the input data
        clip_id: the id of the clip
        settings: the settings of the dataset
        camera_type: the type of camera model, 'pinhole' or 'ftheta'
        post_training: if True, use the post-training settings
        resize_resolution: the resolution of the output video. It helps us to set the correct camera model.
        novel_pose_folder: the folder name of the novel pose data. If provided, we will render the novel ego trajectory.

    Returns:
        camera_name_to_camera_poses: the pose of the camera. dict of numpy array of shape (T, 4, 4)
        render_frame_ids: the valid frame ids used for rendering. list of int
        all_object_info: the object info. list of dict
        camera_name_to_camera_model: the camera models. dict of camera model objects
    """
    INPUT_POSE_FPS = settings['INPUT_POSE_FPS']

    if post_training: 
        TARGET_RENDER_FPS = settings['POST_TRAINING']['TARGET_RENDER_FPS']
        TARGET_CHUNK_FRAME = settings['POST_TRAINING']['TARGET_CHUNK_FRAME']
        OVERLAP_FRAME = settings['POST_TRAINING']['OVERLAP_FRAME']
        MAX_CHUNK = settings['POST_TRAINING'].get('MAX_CHUNK', -1)
    else:
        TARGET_RENDER_FPS = settings['NOT_POST_TRAINING']['TARGET_RENDER_FPS']
        TARGET_CHUNK_FRAME = settings['NOT_POST_TRAINING']['TARGET_CHUNK_FRAME']
        OVERLAP_FRAME = settings['NOT_POST_TRAINING']['OVERLAP_FRAME']
        MAX_CHUNK = settings['NOT_POST_TRAINING'].get('MAX_CHUNK', -1)

    camera_name_to_camera_model = {}
    camera_name_to_camera_poses = {}

    resize_w, resize_h = resize_resolution
    # generate map projection for the instance buffer
    for camera_name in settings['CAMERAS']:
        print(f"Processing {clip_id} {camera_name}...")
        # load pose
        pose_folder = 'pose' if novel_pose_folder is None else novel_pose_folder
        pose_file = os.path.join(input_root, pose_folder, f"{clip_id}.tar")
        camera_key = f"pose.{camera_name}.npy"
        pose_data = get_sample(pose_file)
        pose_data_this_cam = {k: v for k, v in pose_data.items() if camera_key in k}
        pose_all_frames = np.stack([pose_data_this_cam[k] for k in sorted(pose_data_this_cam.keys())])

        frame_num = pose_all_frames.shape[0] # measured in INPUT_POSE_FPS, usually 30
        render_frame_ids = list(range(0, frame_num, INPUT_POSE_FPS // TARGET_RENDER_FPS))

        if MAX_CHUNK > 0:
            expected_frame_num = TARGET_CHUNK_FRAME + (TARGET_CHUNK_FRAME - OVERLAP_FRAME) * (MAX_CHUNK - 1)
            render_frame_ids = render_frame_ids[:expected_frame_num]
            print(f"Processing {clip_id} {camera_name} with {len(render_frame_ids)} frames only due to max_chunk={MAX_CHUNK}")

        # interpolate bbox from 10Hz to 30Hz
        all_object_info_file = os.path.join(input_root, 'all_object_info', f"{clip_id}.tar")
        all_object_info = get_sample(all_object_info_file)

        # fix static objects in jitter
        all_object_info = fix_static_objects(all_object_info)

        # interpolate bbox if box label FPS is lower than TARGET_RENDER_FPS
        all_object_info = interpolate_bbox(all_object_info, render_frame_ids)

        # read intrinsic and build camera model.
        if camera_type == "pinhole":
            intrinsic_file = os.path.join(input_root, f'{camera_type}_intrinsic', f"{clip_id}.tar")
            intrinsic_data = get_sample(intrinsic_file)
            intrinsic_this_cam = intrinsic_data[f"{camera_type}_intrinsic.{camera_name}.npy"]
            camera_model = PinholeCamera.from_numpy(intrinsic_this_cam, device='cpu')

            rescale_h = resize_h / camera_model.height
            rescale_w = resize_w / camera_model.width
            camera_model.rescale(rescale_h, rescale_w)

        elif camera_type == "ftheta":
            intrinsic_file = os.path.join(input_root, f'{camera_type}_intrinsic', f"{clip_id}.tar")

            # 3-rd party dataset does not have ftheta intrinsic, we can use the default one from RDS-HQ
            if os.path.exists(intrinsic_file):
                intrinsic_data = get_sample(intrinsic_file)
                intrinsic_this_cam = intrinsic_data[f"{camera_type}_intrinsic.{camera_name}.npy"]

            else:
                cprint(f"Ftheta intrinsic file does not exist: {intrinsic_file}", 'red')
                cprint(f"===> So we will use default ftheta intrinsic for rendering", 'yellow', attrs=['bold'])
                intrinsic_file = 'assets/default_ftheta_intrinsic.tar'
                camera_name_in_rds_hq = settings['CAMERAS_TO_RDS_HQ'][camera_name]    
                intrinsic_data = get_sample(intrinsic_file)
                intrinsic_this_cam = intrinsic_data[f"{camera_type}_intrinsic.{camera_name_in_rds_hq}.npy"]

            camera_model = FThetaCamera.from_numpy(intrinsic_this_cam, device='cpu')
            rescale_h = resize_h / camera_model.height
            rescale_w = resize_w / camera_model.width

            assert abs(rescale_h - rescale_w) < 0.02 # only handle the case that height is downsampled by the same ratio, required by the ftheta camera model
            camera_model.rescale(rescale_h)
        else:
            raise ValueError(f"Invalid camera type: {camera_type}")

        camera_name_to_camera_model[camera_name] = camera_model
        camera_name_to_camera_poses[camera_name] = pose_all_frames

    return camera_name_to_camera_poses, render_frame_ids, all_object_info, camera_name_to_camera_model

def prepare_output(
        full_video, 
        render_frame_ids, 
        render_name, 
        settings, 
        output_root, 
        clip_id, 
        camera_name, 
        resize_resolution, 
        cosmos_resolution, 
        post_training,
        camera_type
    ):
    """
    Cut full video into small clips for cosmos training / inference and save them.

    Args:
        full_video: the full video to be cut. numpy array of shape (T, H, W, 3)
        render_frame_ids: the frame ids to be rendered. list of int
        render_name: the name of the rendered video. str. 'hdmap', 'lidar', or 'rgb'
        settings: the settings of the dataset
        output_root: the root folder of the output data
        clip_id: the id of the clip
        camera_name: the name of the camera
        camera_type: the type of camera model, 'pinhole' or 'ftheta'
    """
    if post_training: 
        TARGET_RENDER_FPS = settings['POST_TRAINING']['TARGET_RENDER_FPS']
        TARGET_CHUNK_FRAME = settings['POST_TRAINING']['TARGET_CHUNK_FRAME']
        OVERLAP_FRAME = settings['POST_TRAINING']['OVERLAP_FRAME']
        to_cosmos_resolution = settings['POST_TRAINING']['TO_COSMOS_RESOLUTION']
    else:
        TARGET_RENDER_FPS = settings['NOT_POST_TRAINING']['TARGET_RENDER_FPS']
        TARGET_CHUNK_FRAME = settings['NOT_POST_TRAINING']['TARGET_CHUNK_FRAME']
        OVERLAP_FRAME = settings['NOT_POST_TRAINING']['OVERLAP_FRAME']
        to_cosmos_resolution = settings['NOT_POST_TRAINING']['TO_COSMOS_RESOLUTION']

    # sometimes the full_video does not match the target resolution, e.g. 720 * 1277, we need to resize it
    if full_video.shape[1] != resize_resolution[1] or full_video.shape[2] != resize_resolution[0]:
        print(f"Resizing {clip_id} {render_name} video from {full_video.shape[1]}x{full_video.shape[2]} to {resize_resolution[1]}x{resize_resolution[0]}...")
        full_video = np.stack([cv2.resize(frame, resize_resolution, interpolation=cv2.INTER_LANCZOS4) for frame in full_video], axis=0)

    if cosmos_resolution != resize_resolution:
        if to_cosmos_resolution == 'resize':
            target_w, target_h = cosmos_resolution
            full_video = np.stack([cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4) for frame in full_video], axis=0)
        elif to_cosmos_resolution == 'center-crop':
            crop_w, crop_h = cosmos_resolution
            resize_w, resize_h = resize_resolution
            crop_x = (resize_w - crop_w) // 2
            crop_y = (resize_h - crop_h) // 2
            full_video = full_video[:, crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

    output_root_p = Path(output_root)
    camera_folder_name = f"{camera_type}_{camera_name}"
    (output_root_p / render_name / camera_folder_name).mkdir(parents=True, exist_ok=True)
    for cur_idx, i in enumerate(range(0, len(render_frame_ids), TARGET_CHUNK_FRAME - OVERLAP_FRAME)):
        if i + TARGET_CHUNK_FRAME > len(render_frame_ids):
            continue

        full_video_cut = full_video[i:i+TARGET_CHUNK_FRAME]

        # save HD map condition video
        output_writer = imageio_v1.get_writer(
            output_root_p / render_name / camera_folder_name / f"{clip_id}_{cur_idx}.mp4",
            fps=TARGET_RENDER_FPS,
            codec="libx264",
            macro_block_size=None,  # This makes sure num_frames is correct (by default it is rounded to 16x).
            ffmpeg_params=[
                "-crf", "18",     # Lower CRF for higher quality (0-51, lower is better)
                "-preset", "slow",   # Slower preset for better compression/quality
                "-pix_fmt", "yuv420p", # Ensures wide compatibility
            ],
        )
        for local_frame_idx in range(len(full_video_cut)):
            output_writer.append_data(full_video_cut[local_frame_idx])
        output_writer.close()


@ray_remote(use_ray=USE_RAY, num_gpus=0, num_cpus=4)
def render_sample_hdmap(
    input_root: str,
    output_root: str,
    clip_id: str,
    settings: dict,
    camera_type: str,
    post_training: bool = False,
    novel_pose_folder: str = None,
    resize_resolution: tuple[int, int] = (1280, 720),
    cosmos_resolution: tuple[int, int] = (1280, 704)
):
    minimap_types = settings['MINIMAP_TYPES']

    camera_name_to_camera_poses, render_frame_ids, all_object_info, camera_name_to_camera_model = \
        prepare_input(input_root, clip_id, settings, camera_type, post_training, resize_resolution, novel_pose_folder)
    minimap_wds_files = [
        os.path.join(input_root, f"3d_{minimap_type}", f"{clip_id}.tar") for minimap_type in minimap_types
    ]

    for camera_name, camera_model in camera_name_to_camera_model.items():
        pose_all_frames = camera_name_to_camera_poses[camera_name]

        minimaps_projection_merged = np.zeros((len(render_frame_ids), camera_model.height, camera_model.width, 3), dtype=np.uint8)
        for minimap_wds_file in minimap_wds_files:
            minimap_data_wo_meta_info, minimap_name = simplify_minimap(minimap_wds_file)

            # all static labels
            minimap_projection = create_minimap_projection(
                minimap_name,
                minimap_data_wo_meta_info,
                pose_all_frames[render_frame_ids],
                camera_model
            )
            minimaps_projection_merged = np.maximum(minimaps_projection_merged, minimap_projection)

        # add bounding box projection to the minimap
        bounding_box_projection = create_bbox_projection(
            all_object_info,
            pose_all_frames,
            render_frame_ids,
            camera_model,
        )
        minimaps_projection_merged = np.maximum(minimaps_projection_merged, bounding_box_projection)

        prepare_output(
            minimaps_projection_merged, 
            render_frame_ids, 
            'hdmap', 
            settings,
            output_root, 
            clip_id, 
            camera_name, 
            resize_resolution, 
            cosmos_resolution, 
            post_training,
            camera_type
        )


@ray_remote(use_ray=USE_RAY, num_gpus=1, num_cpus=2)
def render_sample_lidar(
    input_root: str,
    output_root: str,
    clip_id: str,
    settings: dict,
    camera_type: str,
    post_training: bool = False,
    novel_pose_folder: str = None,
    resize_resolution: tuple[int, int] = (1280, 720),
    cosmos_resolution: tuple[int, int] = (1280, 704),
    accumulate_lidar_frames: int = 2,
):
    INPUT_POSE_FPS = settings['INPUT_POSE_FPS']
    INPUT_LIDAR_FPS = settings['INPUT_LIDAR_FPS']

    resize_w, resize_h = resize_resolution
    camera_name_to_camera_poses, render_frame_ids, all_object_info, camera_name_to_camera_model = \
        prepare_input(input_root, clip_id, settings, camera_type, post_training, resize_resolution, novel_pose_folder)

    frame_num = len(render_frame_ids)

    for camera_name, camera_model in camera_name_to_camera_model.items():
        pose_all_frames = camera_name_to_camera_poses[camera_name]
        # load lidar data
        lidar_tar = os.path.join(input_root, 'lidar_raw', f"{clip_id}.tar")
        lidar_data = get_sample(lidar_tar) # 'xxxxxx.xyz' & 'xxxxxx.lidar_to_world'

        # generate lidar depth
        lidar_depth_list = []
        interp_rate = INPUT_POSE_FPS // INPUT_LIDAR_FPS
        for frame_idx in render_frame_ids:
            accumulate_idx = [min(max(0, frame_idx + i * interp_rate), frame_num-1) for i in range(-accumulate_lidar_frames, accumulate_lidar_frames+1)]
            low_fps_frame_indices, _ = get_nearest_lidar_indices(accumulate_idx, step=interp_rate)

            lidar_frame_ids = [
                int(key.split('.')[0]) for key in lidar_data.keys() if key.endswith('.lidar_raw.npz') and key.split('.')[0].isdigit()
            ]
            valid_lidar_frame_ids = [
                x for x in lidar_frame_ids if x <= render_frame_ids[-1]
            ]
            max_lidar_idx = max(valid_lidar_frame_ids)
            low_fps_frame_indices = [min(max_lidar_idx, idx) for idx in low_fps_frame_indices]

            accumulated_points, original_bbox_dicts, moved_bbox_dicts = [], [], []

            for i in range(len(accumulate_idx)):
                lidar_points = lidar_data[f'{low_fps_frame_indices[i]:06d}.lidar_raw.npz']['xyz'].astype(np.float32).reshape(-1, 3)

                distances = np.linalg.norm(lidar_points, axis=1)
                threshold = 3.0  
                lidar_points = lidar_points[distances >= threshold]
                lidar_to_world = lidar_data[f'{low_fps_frame_indices[i]:06d}.lidar_raw.npz']['lidar_to_world']
                lidar_points_world = (lidar_to_world @ np.concatenate([lidar_points, np.ones([lidar_points.shape[0], 1])], axis=1).T).T[:, :3]
                accumulated_points.append(torch.tensor(lidar_points_world, device='cuda'))
                original_bbox_dicts.append(all_object_info[f'{low_fps_frame_indices[i]:06d}.all_object_info.json'])
                moved_bbox_dicts.append(all_object_info[f'{frame_idx:06d}.all_object_info.json'])

            accumulated_points_corrected, _ = batch_move_points_within_bboxes_sparse(
                accumulated_points,
                original_bbox_dicts,
                moved_bbox_dicts,
                torch.tensor(low_fps_frame_indices),
                dynamic_only=True,
                bbox_scale=1.2, # enlarge bbox to include all points
            )
            lengths_tensor = torch.tensor([t.shape[0] for t in accumulated_points_corrected], device='cuda')
            cat_points = torch.cat(accumulated_points_corrected, dim=0)
            w2cs = torch.inverse(torch.tensor(pose_all_frames[frame_idx])).unsqueeze(0).cuda()
            if camera_type == 'ftheta':
                cx_cy_w_h = camera_model.intrinsics[:4]
                bw_poly = camera_model._bw_poly.coef
                is_bw_poly = 1
                intrinsics = torch.tensor(
                    [*cx_cy_w_h, *bw_poly, is_bw_poly]
                ).unsqueeze(0).to(cat_points)
            else:
                intrinsics = torch.tensor(camera_model.intrinsics).unsqueeze(0).to(cat_points)
            depth, mask = forward_warp_multiframes_sparse_depth_only(
                w2cs,
                intrinsics,
                buffer_points=cat_points.cuda(),
                buffer_length=lengths_tensor.cuda(),
                target_h=resize_h,
                target_w=resize_w,
                center_only=False,
                is_ftheta=(camera_type == 'ftheta'),
                expanded_kernel=4
            )            
            depth = depth.squeeze(0).squeeze(0).cpu().numpy()
            lidar_depth_list.append(depth)
        lidar_depth = np.stack(lidar_depth_list, axis=0) # T, H, W
        # clamp to 75
        lidar_depth_vis = np.clip(lidar_depth, 0, 75) / 75.0 * 255
        lidar_depth_vis = lidar_depth_vis.astype(np.uint8)
        lidar_depth_vis = np.stack([lidar_depth_vis] * 3, axis=-1) # T, H, W, 3

        # prepare output
        prepare_output(
            lidar_depth_vis,
            render_frame_ids,
            'lidar',
            settings,
            output_root,
            clip_id,
            camera_name,
            resize_resolution,
            cosmos_resolution,
            post_training,
            camera_type
        )


@ray_remote(use_ray=USE_RAY, num_gpus=0)
def render_sample_rgb(
    input_root: str,
    output_root: str,
    clip_id: str,
    settings: dict,
    camera_type: str,
    post_training: bool = False,
    novel_pose_folder: str = None,
    resize_resolution: tuple[int, int] = (1280, 720),
    cosmos_resolution: tuple[int, int] = (1280, 704),
):
    """
    This function is used to render / sample the RGB video for post-training use.
    """
    resize_w, resize_h = resize_resolution
    camera_name_to_camera_poses, render_frame_ids, all_object_info, camera_name_to_camera_model = \
        prepare_input(input_root, clip_id, settings, camera_type, post_training, resize_resolution, novel_pose_folder)

    for camera_name, camera_model in camera_name_to_camera_model.items():
        frame_idx_in_gt_video = [i // (settings['INPUT_POSE_FPS'] // settings['GT_VIDEO_FPS']) for i in render_frame_ids]
        pose_all_frames = camera_name_to_camera_poses[camera_name]
        # load rgb
        rgb_file = os.path.join(input_root, f'{camera_type}_{camera_name}', f"{clip_id}.mp4")
        # load all frames
        vr = decord.VideoReader(rgb_file)
        all_frames = vr.get_batch(frame_idx_in_gt_video).asnumpy()
        
        prepare_output(
            all_frames,
            frame_idx_in_gt_video,
            'videos',
            settings,
            output_root,
            clip_id,
            camera_name,
            resize_resolution,
            cosmos_resolution,
            post_training,
            camera_type
        )
            
@click.command()
@click.option("--input_root", '-i', type=str, help="the root folder of RDS-HQ or RDS-HQ format dataset")
@click.option("--clip_id_json", "-cj", type=str, default=None, help="exact clip id or path to the clip id json file. If provided, we just render the clips in the json file.")
@click.option("--output_root", '-o', type=str, required=True, help="the root folder for the output data")
@click.option("--dataset", "-d", type=str, default="rds_hq", help="the dataset name, 'rds_hq', 'rds_hd_mv', 'waymo' or 'waymo_mv', see xxx.json in config folder")
@click.option("--camera_type", "-c", type=str, default="ftheta", help="the type of camera model, 'pinhole' or 'ftheta'")
@click.option("--skip", "-s", multiple=True, help="can be 'hdmap' or 'lidar'")
@click.option("--post_training", "-p", type=bool, default=False, help="if True, output the RGB video for post-training")
@click.option("--novel_pose_folder", "-np", type=str, default=None, help="the folder name of the novel pose data. If provided, we will render the novel ego trajectory")
def main(input_root, clip_id_json, output_root, dataset, camera_type, skip, post_training, novel_pose_folder):
    if skip is not None:
        assert all(s in ['hdmap', 'lidar'] for s in skip), "skip must be in ['hdmap', 'lidar']"

    # Load settings
    with open(f'config/dataset_{dataset}.json', 'r') as file:
        settings = json.load(file)

    # extract resize_resolution and cosmos_resolution
    cosmos_resolution = settings['COSMOS_RESOLUTION']

    if post_training: # for post-training only
        assert dataset in ['waymo', 'waymo_mv', 'waymo_mv_short'], "post_training is only supported for waymo dataset"
        assert camera_type == 'pinhole', "post_training is only supported for waymo pinhole camera currently"
        resize_resolution = settings['POST_TRAINING']['RESIZE_RESOLUTION']
        to_cosmos_resolution = settings['POST_TRAINING']['TO_COSMOS_RESOLUTION']
    else:
        resize_resolution = settings['NOT_POST_TRAINING']['RESIZE_RESOLUTION']
        to_cosmos_resolution = settings['NOT_POST_TRAINING']['TO_COSMOS_RESOLUTION']


    if to_cosmos_resolution == 'resize':
        if resize_resolution != cosmos_resolution:
            print(f"Original Resolution -> 1. {colored('Resize', 'yellow', attrs=['bold'])} to {colored(resize_resolution, 'yellow', attrs=['bold'])} for {camera_type} camera model.")
            print(f"                    -> 2. {colored('Resize', 'yellow', attrs=['bold'])} to {colored(cosmos_resolution, 'yellow', attrs=['bold'])} for cosmos model.")
        else:
            print(f"Original Resolution -> 1. {colored('Resize', 'yellow', attrs=['bold'])} to {colored(cosmos_resolution, 'yellow', attrs=['bold'])} for {camera_type} camera model and cosmos model.")
    else:
        print(f"Original Resolution -> 1. {colored('Resize', 'yellow', attrs=['bold'])} to {colored(resize_resolution, 'yellow', attrs=['bold'])} for {camera_type} camera model.")
        print(f"                    -> 2. {colored('Center Crop', 'magenta', attrs=['bold'])} to {colored(cosmos_resolution, 'magenta', attrs=['bold'])} for cosmos model.")



    # if novel_pose_folder is provided, we also change the output folder
    if novel_pose_folder is not None:
        output_root = f'{output_root}_{novel_pose_folder}'

    # get all clip ids
    if clip_id_json is None:
        input_root_p = Path(input_root)
        pose_folder = 'pose' if novel_pose_folder is None else novel_pose_folder
        clip_list = (input_root_p / pose_folder).rglob('*.tar')
        clip_list = [c.stem for c in clip_list]
    elif clip_id_json.endswith('.json'):
        clip_list = json.load(open(clip_id_json))
    else:
        clip_list = [clip_id_json]

    # shuffle the clip list
    np.random.seed(0)
    np.random.shuffle(clip_list)

    if USE_RAY:
        ray.init()
        futures = []
        if 'hdmap' not in skip:
            futures.extend([render_sample_hdmap.remote(input_root, output_root, clip_id, settings, camera_type, post_training, novel_pose_folder, resize_resolution, cosmos_resolution) for clip_id in clip_list])
        if 'lidar' not in skip:
            futures.extend([render_sample_lidar.remote(input_root, output_root, clip_id, settings, camera_type, post_training, novel_pose_folder, resize_resolution, cosmos_resolution) for clip_id in clip_list])
        if post_training:
            futures.extend([render_sample_rgb.remote(input_root, output_root, clip_id, settings, camera_type, post_training, novel_pose_folder, resize_resolution, cosmos_resolution) for clip_id in clip_list])
        wait_for_futures(futures)
    else:
        for clip_id in tqdm(clip_list):
            if 'hdmap' not in skip:
                render_sample_hdmap(input_root, output_root, clip_id, settings, camera_type, post_training, novel_pose_folder, resize_resolution, cosmos_resolution)
            if 'lidar' not in skip:
                render_sample_lidar(input_root, output_root, clip_id, settings, camera_type, post_training, novel_pose_folder, resize_resolution, cosmos_resolution)
            if post_training:
                render_sample_rgb(input_root, output_root, clip_id, settings, camera_type, post_training, novel_pose_folder, resize_resolution, cosmos_resolution)

if __name__ == "__main__":
    main()
