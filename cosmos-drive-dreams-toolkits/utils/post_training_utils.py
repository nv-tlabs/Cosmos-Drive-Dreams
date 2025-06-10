import os
import decord
import numpy as np
import sys
import imageio as imageio_v1
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from ray_utils import ray_remote, wait_for_futures

USE_RAY = True

@ray_remote(use_ray=USE_RAY, num_cpus=4)
def overlay_condition_on_rgb(
    rgb_video_path: str,
    condition_video_path: str,
    output_video_path: str,
    overlay_weight: float = 0.8,
    output_fps: int = 120,
):
    """
    Overlay the condition video on the rgb video.
    """
    rgb_video = decord.VideoReader(rgb_video_path)
    condition_video = decord.VideoReader(condition_video_path)

    assert len(rgb_video) == len(condition_video), f"RGB video and condition video must have the same length, but got {len(rgb_video)} and {len(condition_video)}"

    overlay_frames = []
    for i in range(len(rgb_video)):
        rgb_frame = rgb_video[i].asnumpy()
        condition_frame = condition_video[i].asnumpy()

        # overlay
        overlay_frame = (rgb_frame + overlay_weight * condition_frame).clip(0, 255).astype(np.uint8)
        overlay_frames.append(overlay_frame)

    overlay_frames = np.array(overlay_frames)
    
    output_writer = imageio_v1.get_writer(
        output_video_path,
        fps=output_fps,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=[
            "-crf", "18",
            "-preset", "slow",
            "-pix_fmt", "yuv420p",
        ],
    )
    for frame in overlay_frames:
        output_writer.append_data(frame)
    output_writer.close()


def overlay_condition_on_rgb_folder(
    rendered_root: str,
    condition_name: str = 'hdmap',
    camera_type: str = 'pinhole',
    camera_name: str = 'front',
    output_fps: int = 120,
    overlay_weight: float = 0.8,
):
    """
    Overlay the condition video on the rgb video in the rendered folder.
    """
    rendered_root_p = Path(rendered_root)
    rgb_video_p = rendered_root_p / 'videos' / f'{camera_type}_{camera_name}'
    condition_video_p = rendered_root_p / condition_name / f'{camera_type}_{camera_name}'
    output_video_p = rendered_root_p / f'{condition_name}_overlay' / f'{camera_type}_{camera_name}'
    output_video_p.mkdir(parents=True, exist_ok=True)

    rgb_videos = list(rgb_video_p.glob('*.mp4'))
    condition_videos = list(condition_video_p.glob('*.mp4'))

    rgb_videos.sort()
    condition_videos.sort()

    assert len(rgb_videos) == len(condition_videos), f"RGB videos and condition videos must have the same length, but got {len(rgb_videos)} and {len(condition_videos)}"

    if USE_RAY:
        futures = []
        for rgb_video, condition_video in zip(rgb_videos, condition_videos):
            output_video_path = output_video_p / rgb_video.name
            futures.append(overlay_condition_on_rgb.remote(rgb_video.as_posix(), condition_video.as_posix(), output_video_path.as_posix(), overlay_weight, output_fps))
        wait_for_futures(futures)
    else:
        for rgb_video, condition_video in zip(rgb_videos, condition_videos):
            output_video_path = output_video_p / rgb_video.name
            overlay_condition_on_rgb(rgb_video.as_posix(), condition_video.as_posix(), output_video_path.as_posix(), overlay_weight, output_fps)
