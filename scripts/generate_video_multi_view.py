# Take hdmap and generate synthetic video based on diverse captions
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

import argparse
import einops
import json
import os
import math
import numpy as np
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Workaround to suppress MP warning

import sys
import copy
from io import BytesIO

import torch

from cosmos_transfer1.checkpoints import (BASE_7B_CHECKPOINT_AV_SAMPLE_PATH, BASE_7B_CHECKPOINT_PATH,
                                          BASE_v2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH,
                                          BASE_t2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH)
from cosmos_transfer1.utils import log, misc
from cosmos_transfer1.utils.io import read_prompts_from_file, save_video
from cosmos_transfer1.diffusion.inference.transfer_multiview import validate_controlnet_specs
from cosmos_transfer1.diffusion.inference.preprocessors import Preprocessors
from cosmos_transfer1.diffusion.inference.world_generation_pipeline import DiffusionControl2WorldMultiviewGenerationPipeline


valid_hint_keys = {"hdmap", "lidar"}
def load_controlnet_specs(cfg):
    with open(cfg.controlnet_specs, "r") as f:
        controlnet_specs_in = json.load(f)

    controlnet_specs = {}
    args = {}

    for hint_key, config in controlnet_specs_in.items():
        if hint_key in valid_hint_keys:
            controlnet_specs[hint_key] = config
        else:
            if type(config) == dict:
                raise ValueError(f"Invalid hint_key: {hint_key}. Must be one of {valid_hint_keys}")
            else:
                args[hint_key] = config
                continue
    return controlnet_specs, args


def create_video_grid(video_tensors, padding=0, n_row=2):
    """
    Arrange a list of PyTorch video tensors into a grid. When number of videos is not divisible by n_row, fit videos in
     last row with black padding on the sides.

    Args:
        video_tensors (list): List of PyTorch tensors with shape [T, C, H, W] where:
            - T is the number of frames
            - C is the number of channels (typically 3 for RGB)
            - H is the height of each frame
            - W is the width of each frame
        padding (int, optional): Padding between videos in pixels. Defaults to 0.
        n_row (int, optional): number of rows

    Returns:
        torch.Tensor: A tensor of shape [T, C, grid_H, grid_W] representing the video grid
    """

    # Check if all videos have the same number of frames, channels, and dimensions
    num_frames = video_tensors[0].shape[0]
    num_channels = video_tensors[0].shape[1]
    height = video_tensors[0].shape[2]
    width = video_tensors[0].shape[3]

    for i, video in enumerate(video_tensors):
        if video.shape[0] != num_frames:
            raise ValueError(f"Video {i} has {video.shape[0]} frames, expected {num_frames}")
        if video.shape[1] != num_channels:
            raise ValueError(f"Video {i} has {video.shape[1]} channels, expected {num_channels}")

    # Calculate grid dimensions
    n_vids = len(video_tensors)
    grid_height = n_row * height + (n_row - 1) * padding
    n_col = math.ceil(n_vids / n_row)
    grid_width = n_col * width + (n_col - 1) * padding
    n_last_row = n_vids - (n_row - 1) * n_col

    # Create an empty grid filled with zeros (black)
    grid = torch.zeros(
        (num_frames, num_channels, grid_height, grid_width),
        dtype=video_tensors[0].dtype,
        device=video_tensors[0].device,
    )

    # Place videos on the top row (4 videos)
    for i in range(n_row):
        if i == (n_row - 1):
            nc = n_last_row
            # Calculate the starting position for the bottom row to center the remaining videos
            bottom_row_width = nc * width + (nc - 1) * padding
            left_padding = (grid_width - bottom_row_width) // 2
        else:
            nc = n_col
            left_padding = 0
        for j in range(nc):
            vid = i * n_col + j
            y_offset = i * (height + padding)
            x_offset = left_padding + j * (width + padding)
            grid[:, :, y_offset : y_offset + height, x_offset : x_offset + width] = video_tensors[vid]

    return grid

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control to world generation demo script", conflict_handler="resolve")

    # Add transfer specific arguments
    parser.add_argument("--caption_path", type=str, required=True, help="folder containing the json files of captions")
    parser.add_argument("--input_path", type=str, required=True, help="folder containing the hdmap/lidar condition videos")
    parser.add_argument("--input_view_path", type=str, required=True,
                        help="folder containing the single view videos")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all.",
        help="negative prompt which the sampled video condition on",
    )
    parser.add_argument("--sigma_max", type=float, default=80.0, help="sigma_max for partial denoising")
    parser.add_argument(
        "--view_condition_video",
        type=str,
        default="",
        help="We require that only a single condition view is specified and this video is treated as conditioning for that view. "
             "This video/videos should have the same duration as control videos",
    )
    parser.add_argument(
        "--initial_condition_video",
        type=str,
        default="",
        help="Can be either a path to a mp4 or a directory. If it is a mp4, we assume"
             "that it is a video temporally concatenated with the same number of views as the model. "
             "If it is a directory, we assume that the file names evaluate to integers that correspond to a view index,"
             " e.g. '000.mp4', '003.mp4', '004.mp4'."
             "This video/videos should have at least num_input_frames number of frames for each view. Frames will be taken from the back"
             "of the video(s) if the duration of the video in each view exceed num_input_frames",
    )
    parser.add_argument(
        "--num_input_frames",
        type=int,
        default=1,
        help="Number of conditional frames for long video generation, not used in t2w",
        choices=[1, 9],
    )
    parser.add_argument(
        "--controlnet_specs",
        type=str,
        help="Path to JSON file specifying multicontrolnet configurations",
        required=True,
    )
    parser.add_argument(
        "--is_av_sample", action="store_true", help="Whether the model is an driving post-training model"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Base directory containing model checkpoints"
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="Cosmos-Tokenize1-CV8x8x8-720p",
        help="Tokenizer weights directory relative to checkpoint_dir",
    )
    parser.add_argument(
        "--video_save_folder",
        type=str,
        default="outputs/",
        help="Output folder for generating a batch of videos",
    )
    parser.add_argument(
        "--batch_input_path",
        type=str,
        help="Path to a JSONL file of input prompts for generating a batch of videos",
    )
    parser.add_argument("--num_steps", type=int, default=35, help="Number of diffusion sampling steps")
    parser.add_argument("--guidance", type=float, default=5, help="Classifier-free guidance scale value")
    parser.add_argument("--fps", type=int, default=24, help="FPS of the output video")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs used to run inference in parallel.")
    parser.add_argument(
        "--offload_diffusion_transformer",
        action="store_true",
        help="Offload DiT after inference",
    )
    parser.add_argument(
        "--offload_text_encoder_model",
        action="store_true",
        help="Offload text encoder model after inference",
    )
    parser.add_argument(
        "--offload_guardrail_models",
        action="store_true",
        help="Offload guardrail models after inference",
    )
    parser.add_argument(
        "--upsample_prompt",
        action="store_true",
        help="Upsample prompt using Pixtral upsampler model",
    )
    parser.add_argument(
        "--offload_prompt_upsampler",
        action="store_true",
        help="Offload prompt upsampler model after inference",
    )
    parser.add_argument("--n_clip_max", type=int, default=-1, help="Maximum number of video extension loop")
    cmd_args = parser.parse_args()

    # Load and parse JSON input
    control_inputs, json_args = load_controlnet_specs(cmd_args)

    log.info(f"control_inputs: {json.dumps(control_inputs, indent=4)}")
    log.info(f"args in json: {json.dumps(json_args, indent=4)}")

    # if parameters not set on command line, use the ones from the controlnet_specs
    # if both not set use command line defaults
    for key in json_args:
        if f"--{key}" not in sys.argv:
            setattr(cmd_args, key, json_args[key])

    log.info(f"final args: {json.dumps(vars(cmd_args), indent=4)}")

    return cmd_args, control_inputs


def demo(cfg,
         pipeline,
         control_inputs,
         input_view_vid,
         video_save_name,
         prompt,
         ):
    """Run control-to-world generation demo.

    This function handles the main control-to-world generation pipeline, including:
    - Setting up the random seed for reproducibility
    - Initializing the generation pipeline with the provided configuration
    - Processing single or multiple prompts/images/videos from input
    - Generating videos from prompts and images/videos
    - Saving the generated videos and corresponding prompts to disk

    Args:
        cfg (argparse.Namespace): Configuration namespace containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (guidance, steps, dimensions)
            - Input/output settings (prompts/images/videos, save paths)
            - Performance options (model offloading settings)

    The function will save:
        - Generated MP4 video files
        - Text files containing the processed prompts

    If guardrails block the generation, a critical log message is displayed
    and the function continues to the next prompt if available.
    """
    preprocessors = Preprocessors()

    current_prompt = [prompt,
                      "The video is captured from a camera mounted on a car. The camera is facing to the left.",
                      "The video is captured from a camera mounted on a car. The camera is facing to the right.",
                      "The video is captured from a camera mounted on a car. The camera is facing backwards.",
                      "The video is captured from a camera mounted on a car. The camera is facing the rear left side.",
                      "The video is captured from a camera mounted on a car. The camera is facing the rear right side."
                      ]
    current_video_path = ""
    video_save_subfolder = str(os.path.join(cfg.video_save_folder, video_save_name))
    os.makedirs(video_save_subfolder, exist_ok=True)
    current_control_inputs = copy.deepcopy(control_inputs)

    # if control inputs are not provided, run respective preprocessor (for seg and depth)
    preprocessors(current_video_path, current_prompt, current_control_inputs, video_save_subfolder)

    # Generate video
    generated_output = pipeline.generate(
        prompts=current_prompt,
        view_condition_video=input_view_vid,
        initial_condition_video=cfg.initial_condition_video,
        control_inputs=current_control_inputs,
    )
    if generated_output is None:
        log.critical("Guardrail blocked generation.")
    video, prompt = generated_output

    if device_rank == 0:
        # Save video
        video = torch.from_numpy(video)
        video_segments = einops.rearrange(video, "(v t) h w c -> v t c h w", v=6)
        video_segments_vthwc = video_segments.permute(0, 1, 3, 4, 2).numpy().astype(np.uint8)
        for i in range(6):
            save_video(
                video=video_segments_vthwc[i],
                fps=args.fps,
                H=video_segments_vthwc[i].shape[1],
                W=video_segments_vthwc[i].shape[2],
                video_save_quality=8,
                video_save_path=os.path.join(video_save_subfolder, f"{i}.mp4"),
            )

        grid = create_video_grid(
            [
                video_segments[1],
                video_segments[0],
                video_segments[2],
                video_segments[4],
                video_segments[3],
                video_segments[5],
            ],
            n_row=2,
        )

        grid = einops.rearrange(grid, "t c h w -> t h w c")
        grid = grid.numpy()
        save_video(
            video=grid,
            fps=cfg.fps,
            H=grid.shape[1],
            W=grid.shape[2],
            video_save_quality=8,
            video_save_path=os.path.join(video_save_subfolder, f"grid.mp4"),
        )

        # Save prompt to text file alongside video
        with open(os.path.join(video_save_subfolder, "prompt.json"), "wb") as f:
            f.write(";".join(prompt).encode("utf-8"))

        log.info(f"Saved video to {video_save_subfolder}")


if __name__ == "__main__":
    args, control_inputs = parse_arguments()

    caption_path = os.path.abspath(args.caption_path)
    data_path = os.path.abspath(args.input_path)
    input_view_path = os.path.abspath(args.input_view_path)
    args.video_save_folder = os.path.abspath(args.video_save_folder)
    # load all the json files in the caption_path

    json_files = [f for f in os.listdir(caption_path) if f.endswith('.json')]
    # load the json file and get the prompt
    prompts = []
    hdmaps = []
    lidars = []
    input_view_vids = []
    video_save_names = []
    view_names = ["ftheta_camera_front_wide_120fov",
                  "ftheta_camera_cross_left_120fov",
                  "ftheta_camera_cross_right_120fov",
                  "ftheta_camera_rear_tele_30fov",
                  "ftheta_camera_rear_left_70fov",
                  "ftheta_camera_rear_right_70fov"
                  ]
    for json_file in tqdm(json_files):
        sample_name = json_file.split('.')[0]
        with open(os.path.join(caption_path, json_file), 'r') as f:
            data = json.load(f)
            
        for variation in data.keys():
            prompt = data[variation]
            hdmap = []
            for vname in view_names:
                hdmap.append(os.path.join(data_path, "hdmap", vname, f"{sample_name}_0.mp4"))

            if not all([os.path.exists(hdmap_v) for hdmap_v in hdmap]):
                print(f"hdmap files {hdmap} does not exist")
                continue

            input_view_vid = os.path.join(input_view_path, f"{sample_name}_{variation}.mp4")
            # make sure the file exists
            if not os.path.exists(input_view_vid):
                print(f"input view file {input_view_vid} does not exist")
                continue

            if "lidar" in control_inputs.keys():
                lidar = []
                for vname in view_names:
                    lidar.append(os.path.join(data_path, "lidar", vname, f"{sample_name}_0.mp4"))

                if not all([os.path.exists(lidar_v) for lidar_v in lidar]):
                    print(f"lidar file {lidar} does not exist")
                    continue
            else:
                lidar = None

            video_save_name = f"{sample_name}_{variation}"

            # check if output already exists
            video_save_path = os.path.join(args.video_save_folder, video_save_name, "grid.mp4")
            if os.path.exists(video_save_path):
                print(f"video {video_save_name} already exists")
                continue
            
            # append the prompt and hdmap to the list
            prompts.append(prompt)
            hdmaps.append(hdmap)
            lidars.append(lidar)
            input_view_vids.append(input_view_vid)
            video_save_names.append(video_save_name)

    if len(prompts) == 0:
        print("No prompts found")
        exit(0)

    current_dir = os.getcwd()
    os.chdir("cosmos-transfer1")
    torch.enable_grad(False)
    torch.serialization.add_safe_globals([BytesIO])
    cfg = args
    misc.set_random_seed(cfg.seed)

    device_rank = 0
    process_group = None
    if cfg.num_gpus > 1:
        from megatron.core import parallel_state

        from cosmos_transfer1.utils import distributed

        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=cfg.num_gpus)
        process_group = parallel_state.get_context_parallel_group()

        device_rank = distributed.get_rank(process_group)

    if cfg.initial_condition_video:
        cfg.is_lvg_model = True
        checkpoint = BASE_v2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH
    else:
        cfg.is_lvg_model = False
        cfg.num_input_frames = 0
        checkpoint = BASE_t2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH

    # Initialize transfer generation model pipeline
    control_inputs = validate_controlnet_specs(cfg, control_inputs)

    pipeline = DiffusionControl2WorldMultiviewGenerationPipeline(
        checkpoint_dir=cfg.checkpoint_dir,
        checkpoint_name=checkpoint,
        offload_network=cfg.offload_diffusion_transformer,
        offload_text_encoder_model=cfg.offload_text_encoder_model,
        offload_guardrail_models=cfg.offload_guardrail_models,
        guidance=cfg.guidance,
        num_steps=cfg.num_steps,
        fps=cfg.fps,
        seed=cfg.seed,
        num_input_frames=cfg.num_input_frames,
        control_inputs=control_inputs,
        sigma_max=80.0,
        num_video_frames=57,
        process_group=process_group,
        height=576,
        width=1024,
        is_lvg_model=cfg.is_lvg_model,
        n_clip_max=cfg.n_clip_max,
        regional_prompts = []
    )


    for prompt, hdmap, lidar, input_view_vid, video_save_name in zip(prompts, hdmaps, lidars, input_view_vids, video_save_names):
        current_control_inputs = control_inputs.copy()
        if "hdmap" in current_control_inputs.keys():
            current_control_inputs["hdmap"]["input_control"] = hdmap
        if "lidar" in current_control_inputs.keys():
            current_control_inputs["lidar"]["input_control"] = lidar
        # run the demo
        demo(
            args,
            pipeline,
            current_control_inputs,
            input_view_vid,
            video_save_name,
            prompt,
        )

    os.chdir(current_dir)
    # clean up properly
    if cfg.num_gpus > 1:
        parallel_state.destroy_model_parallel()
        import torch.distributed as dist

        dist.destroy_process_group()
