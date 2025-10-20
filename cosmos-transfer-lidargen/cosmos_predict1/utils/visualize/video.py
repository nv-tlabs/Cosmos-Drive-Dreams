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

from typing import IO, Any, Union
import os, subprocess
import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image as PILImage
from torch import Tensor
import mediapy as media
from PIL import Image

from cosmos_predict1.utils import log
from cosmos_predict1.utils.easy_io import easy_io


try:
    import ffmpegcv
except Exception as e:  # ImportError cannot catch all problems
    log.info(e)
    ffmpegcv = None


def save_video(grid, video_name, fps=30):
    grid = (grid * 255).astype(np.uint8)
    grid = np.transpose(grid, (1, 2, 3, 0))
    with ffmpegcv.VideoWriter(video_name, "h264", fps) as writer:
        for frame in grid:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            writer.write(frame)


def save_img_or_video(sample_C_T_H_W_in01: Tensor, save_fp_wo_ext: Union[str, IO[Any]], fps: int = 24) -> None:
    """
    Save a tensor as an image or video file based on shape

        Args:
        sample_C_T_H_W_in01 (Tensor): Input tensor with shape (C, T, H, W) in [0, 1] range.
        save_fp_wo_ext (Union[str, IO[Any]]): File path without extension or file-like object.
        fps (int): Frames per second for video. Default is 24.
    """
    assert sample_C_T_H_W_in01.ndim == 4, "Only support 4D tensor"
    assert isinstance(save_fp_wo_ext, str) or hasattr(
        save_fp_wo_ext, "write"
    ), "save_fp_wo_ext must be a string or file-like object"

    if torch.is_floating_point(sample_C_T_H_W_in01):
        sample_C_T_H_W_in01 = sample_C_T_H_W_in01.clamp(0, 1)
    else:
        assert sample_C_T_H_W_in01.dtype == torch.uint8, "Only support uint8 tensor"
        sample_C_T_H_W_in01 = sample_C_T_H_W_in01.float().div(255)

    if sample_C_T_H_W_in01.shape[1] == 1:
        save_obj = PILImage.fromarray(
            rearrange((sample_C_T_H_W_in01.cpu().float().numpy() * 255), "c 1 h w -> h w c").astype(np.uint8),
            mode="RGB",
        )
        ext = ".jpg" if isinstance(save_fp_wo_ext, str) else ""
        easy_io.dump(
            save_obj,
            f"{save_fp_wo_ext}{ext}" if isinstance(save_fp_wo_ext, str) else save_fp_wo_ext,
            file_format="jpg",
            format="JPEG",
            quality=85,
        )
    else:
        save_obj = rearrange((sample_C_T_H_W_in01.cpu().float().numpy() * 255), "c t h w -> t h w c").astype(np.uint8)
        ext = ".mp4" if isinstance(save_fp_wo_ext, str) else ""
        easy_io.dump(
            save_obj,
            f"{save_fp_wo_ext}{ext}" if isinstance(save_fp_wo_ext, str) else save_fp_wo_ext,
            file_format="mp4",
            format="mp4",
            fps=fps,
        )



def save_images_to_video(image_list, output_video_path, fps=30, frame_size=None, max_resolution=None):
    """
    Create a video from a list of images using mediapy.

    Parameters:
    - image_list: List of image file paths.
    - output_video_path: Path to the output video file.
    - fps: Frames per second for the output video.
    - frame_size: Size of the video frames (width, height). If None, it will use the size of the first image.
    - max_resolution: Maximum resolution (width) for the frames. If the input images are larger,
                     they will be resized maintaining aspect ratio. If None, no resizing is performed.
    """
    if not image_list:
        raise ValueError("The image list is empty.")

    # Read all images
    frames = []
    first_img = None
    target_size = None
    
    for image_path in image_list:
        try:
            # Read image using PIL and convert to numpy array
            img = np.array(Image.open(image_path))
            
            # Store first image for reference
            if first_img is None:
                first_img = img
                # Calculate target size if max_resolution is specified
                if max_resolution is not None and img.shape[1] > max_resolution:
                    aspect_ratio = img.shape[1] / img.shape[0]
                    target_size = (max_resolution, int(max_resolution / aspect_ratio))
                    print(f"Resizing images from {img.shape[1]}x{img.shape[0]} to {target_size[0]}x{target_size[1]}")
            
            # Resize if max_resolution is specified and image is larger
            if target_size is not None:
                img = np.array(Image.fromarray(img).resize(target_size, Image.Resampling.LANCZOS))
            # Or resize if frame_size is specified
            elif frame_size is not None:
                img = np.array(Image.fromarray(img).resize(frame_size, Image.Resampling.LANCZOS))

            frames.append(img)
        except Exception as e:
            print(f"Warning: Could not read image {image_path}: {e}. Skipping.")
            continue

    if not frames:
        raise ValueError("No valid images were found.")

    # Write video using mediapy
    media.write_video(output_video_path, frames, fps=fps)
    print(f"Video saved to {output_video_path}")
    
    

def stack_videos_vertically_with_ffmpeg(video_paths, captions, output_path="output_stacked.mp4"):
    """
    Stack videos vertically with captions using OpenCV and FFMPEG.

    Args:
        video_paths (list): List of video file paths.
        captions (list): List of captions corresponding to each video.
        output_path (str): Output file path for the stacked video.
    """
    # Add captions to each video
    captioned_videos = []
    for i, (video_path, caption) in enumerate(zip(video_paths, captions)):
        captioned_video = f"captioned_{i}.mp4"
        command = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"drawtext=text='{caption}':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5",
            "-codec:a",
            "copy",
            captioned_video,
        ]
        subprocess.run(command, check=True)
        captioned_videos.append(captioned_video)

    # Prepare the filter complex for stacking videos vertically
    input_args = []
    filter_complex = []
    for i, video in enumerate(captioned_videos):
        input_args.extend(["-i", video])
        filter_complex.append(f"[{i}:v:0]")

    filter_complex = "".join(filter_complex) + f"vstack=inputs={len(captioned_videos)}[v]"

    # Construct the ffmpeg command
    command = [
        "ffmpeg",
        *input_args,
        "-y",
        "-loglevel",
        "error",
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        output_path,
    ]

    subprocess.run(command, check=True)

    # Clean up intermediate files
    for video in captioned_videos:
        os.remove(video)
