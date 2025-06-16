import decord
import click
import os
import torch
import numpy as np
import imageio

from pathlib import Path
from utils.wds_utils import get_sample, write_to_tar
from utils.camera.ftheta import FThetaCamera
from utils.camera.pinhole import PinholeCamera

decord.bridge.set_bridge('torch')

def get_rectification_image_grid(ftheta_camera_model, pinhole_camera_model):
    """
    Get the image grid for rectification.

    Args:
        ftheta_camera_model: FThetaCamera model
        pinhole_camera_model: PinholeCamera model

    Returns:
        grid_sample_uv: Image grid for rectification, [H, W, 2], range [-1, 1]
    """
    rays = pinhole_camera_model.get_rays() # [H, W, 3], torch cuda tensor
    pixel_in_ftheta_model = ftheta_camera_model.ray2pixel(rays.reshape(-1, 3))
    uv_in_ftheta_model = ftheta_camera_model.pixel2uv(pixel_in_ftheta_model).reshape(rays.shape[0], rays.shape[1], 2) # [H, W, 2], range [0, 1]
    grid_sample_uv = uv_in_ftheta_model * 2 - 1 # [H, W, 2], range [-1, 1]

    return grid_sample_uv


def save_video(video, fps, H, W, video_save_quality, video_save_path):
    """Save video frames to file.

    Args:
        grid (np.ndarray): Video frames array [T,H,W,C]
        fps (int): Frames per second
        H (int): Frame height
        W (int): Frame width
        video_save_quality (int): Video encoding quality (0-10)
        video_save_path (str): Output video file path
    """
    kwargs = {
        "fps": fps,
        "quality": video_save_quality,
        "macro_block_size": 1,
        "ffmpeg_params": ["-s", f"{W}x{H}"],
        "output_params": ["-f", "mp4"],
    }
    imageio.mimsave(video_save_path, video, "mp4", **kwargs)


def rectify_one_video(
        input_video_path, 
        output_root, 
        rds_hq_folder, 
        clip_id,
        camera_name='camera_front_wide_120fov',
        target_pinhole_resolution=(720, 1280),
        cosmos_resolution_after_resize=(704, 1280),
        cosmos_resolution_before_resize=(720, 1280),
    ):
    # deal with ftheta and pinhole camera model
    ftheta_intrinsic = get_sample(os.path.join(rds_hq_folder, 'ftheta_intrinsic', f'{clip_id}.tar'))
    pinhole_intrinsic = get_sample(os.path.join(rds_hq_folder, 'pinhole_intrinsic', f'{clip_id}.tar'))

    ftheta_camera_model = FThetaCamera.from_numpy(ftheta_intrinsic[f'ftheta_intrinsic.{camera_name}.npy'], device='cuda')
    pinhole_camera_model = PinholeCamera.from_numpy(pinhole_intrinsic[f'pinhole_intrinsic.{camera_name}.npy'], device='cuda')

    ftheta_camera_model.rescale(ratio=cosmos_resolution_after_resize[0] / ftheta_camera_model.height)
    pinhole_camera_model.rescale(ratio_h=target_pinhole_resolution[0] / pinhole_camera_model.height, ratio_w=target_pinhole_resolution[1] / pinhole_camera_model.width)

    # video reader
    video_reader = decord.VideoReader(input_video_path)
    n_frames = len(video_reader)
    fps = video_reader.get_avg_fps()
    video_tensor = torch.stack([video_reader[i] for i in range(n_frames)]) # [T, H, W, 3]
    video_tensor = (video_tensor / 255.0).to(ftheta_camera_model.device).permute(0, 3, 1, 2) # [T, 3, H, W]

    # we first resize the video back to the resolution before resize
    video_tensor = torch.nn.functional.interpolate(video_tensor, size=cosmos_resolution_before_resize, mode='bilinear', align_corners=False)

    # then we rectify the video to pinhole camera model
    grid_sample_uv = get_rectification_image_grid(ftheta_camera_model, pinhole_camera_model)
    rectified_video_tensor = torch.nn.functional.grid_sample(video_tensor, grid_sample_uv.expand(n_frames, -1, -1, -1)) # [T, 3, H, W]
    rectified_video_array = rectified_video_tensor.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
    rectified_video_array = rectified_video_array.astype(np.uint8)

    # save the video
    output_root_p = Path(output_root)
    output_root_p.mkdir(parents=True, exist_ok=True)
    video_filename = Path(input_video_path).stem
    output_video_p = output_root_p / f'{video_filename}.mp4'
    save_video(rectified_video_array, fps, target_pinhole_resolution[0], target_pinhole_resolution[1], 8, output_video_p.as_posix())

    # save the pinhole intrinsic
    rectified_pinhole_intrinsic = pinhole_camera_model.intrinsics
    rectified_pinhole_intrinsic_p = output_root_p / f'{video_filename}.tar'
    rectified_pinhole_intrinsic_sample = {
        '__key__': clip_id,
        f'pinhole_intrinsic.{camera_name}.npy': rectified_pinhole_intrinsic
    }
    write_to_tar(rectified_pinhole_intrinsic_sample, rectified_pinhole_intrinsic_p.as_posix())
    
@click.command()
@click.option('--input_video_path', '-i', type=str, required=True, help='The path to the input video')
@click.option('--output_root', '-o', type=str, required=True, help='The path to the output root')
@click.option('--rds_hq_folder', '-r', type=str, required=True, help='The path to the rds_hq folder')
@click.option('--clip_id', '-c', type=str, required=True, help='The clip id')
@click.option('--camera_name', '-n', type=str, default='camera_front_wide_120fov', help='The camera name')
def main(input_video_path, output_root, rds_hq_folder, clip_id, camera_name):
    rectify_one_video(input_video_path, output_root, rds_hq_folder, clip_id, camera_name)

if __name__ == '__main__':
    main()