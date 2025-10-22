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
import os

import torch
from megatron.core import parallel_state

from cosmos_predict1.diffusion.inference.inference_utils import add_common_arguments, remove_argument
from cosmos_predict1.diffusion.inference.world_generation_pipeline import DiffusionViewExtendMultiviewGenerationPipeline
from cosmos_predict1.utils import distributed, log, misc
from cosmos_predict1.utils.io import read_prompts_from_file, save_video

torch.enable_grad(False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)
    remove_argument(parser, "width")
    remove_argument(parser, "height")
    remove_argument(parser, "num_video_frames")
    parser.add_argument("--height", type=int, default=576, help="Height of video to sample")
    parser.add_argument("--width", type=int, default=1024, help="Width of video to sample")
    parser.add_argument("--n_views", type=int, default=6, help="Number of camera view")
    parser.add_argument(
        "--num_video_frames",
        type=int,
        default=57,
        choices=[57],
        help="Number of video frames to sample, this is per-camera frame number.",
    )
    # Add video2world specific arguments
    parser.add_argument(
        "--diffusion_transformer_dir",
        type=str,
        default="Cosmos-Predict1-7B-Video2World-Sample-AV-Single2MultiView/t2w_model.pt",
        help="DiT model weights directory name relative to checkpoint_dir",
        choices=[
            "Cosmos-Predict1-7B-Video2World-Sample-AV-Single2MultiView/t2w_model.pt",
            "Cosmos-Predict1-7B-Video2World-Sample-AV-Single2MultiView/v2w_model.pt",
        ],
    )
    parser.add_argument(
        "--prompt_left",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing to the left. ",
        help="Text prompt for generating left camera view video",
    )
    parser.add_argument(
        "--prompt_right",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing to the right.",
        help="Text prompt for generating right camera view video",
    )

    parser.add_argument(
        "--prompt_back",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing backwards.",
        help="Text prompt for generating rear camera view video",
    )
    parser.add_argument(
        "--prompt_back_left",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing the rear left side.",
        help="Text prompt for generating left camera view video",
    )
    parser.add_argument(
        "--prompt_back_right",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing the rear right side.",
        help="Text prompt for generating right camera view video",
    )
    parser.add_argument(
        "--frame_repeat_negative_condition",
        type=float,
        default=10.0,
        help="frame_repeat number to be used as negative condition",
    )
    parser.add_argument(
        "--view_condition_video",
        type=str,
        help="Input video path for view extension. Can be either a path to a mp4 or a directory. If it is a mp4, we require"
        "that only a single condition view is specified and this video is treated as conditioning for that view. "
        "If it is a directory, we assume that the file names evaluate to integers that correspond to a view index,"
        " e.g. '000.mp4', '003.mp4', '004.mp4'."
        "This video/videos should have at least num_video_frames number of frames. Frames will be taken from the front"
        "of the video(s) if the duration of the video exceed num_video_frames",
    )
    parser.add_argument(
        "--initial_condition_video",
        type=str,
        help="Input video/image for time extension. Can be either a path to a mp4 or a directory. If it is a mp4, we assume"
        "that it is a video temporally concatenated with the same number of views as the model. "
        "If it is a directory, we assume that the file names evaluate to integers that correspond to a view index,"
        " e.g. '000.mp4', '003.mp4', '004.mp4'."
        "This video/videos should have at least num_input_frames number of frames for each view. Frames will be taken from the back"
        "of the video(s) if the duration of the video in each view exceed num_input_frames",
    )
    parser.add_argument(
        "--condition_location",
        type=str,
        help="Which view/views to use as input condition and whether to use initial frame conditioning. Options are:"
        "'fixed_cam_{x1}_{x2}_{x3}' where x1 x2 x3 are integers indicating the input conditioning views,"
        "'first_cam, which is equivalent to fixed_cam_0,"
        "and 'first_cam_and_first_n', where the first view (front view) and initial frames are used as condition ",
    )
    parser.add_argument(
        "--num_input_frames",
        type=int,
        default=1,
        help="Number of input frames for video2world prediction, not used in the t2w setting",
        choices=[0, 1, 9],
    )
    parser.add_argument(
        "--view_cond_start_frame",
        type=int,
        default=0,
        help="Number of frames to skip in the view_condition_video from the start, useful if you want to extend a segment "
        "of the input video rather than from the beginning.",
    )
    return parser.parse_args()


def demo(args):
    """Run multi-view video-to-world generation demo.

    This function handles the main video-to-world generation pipeline, including:
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
    misc.set_random_seed(args.seed)
    inference_type = "video2world"
    # validate_args(args, inference_type)

    if args.num_gpus > 1:
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)

    # Initialize video2world generation model pipeline
    pipeline = DiffusionViewExtendMultiviewGenerationPipeline(
        inference_type=inference_type,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.diffusion_transformer_dir,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_guardrail_models=args.offload_guardrail_models,
        disable_guardrail=args.disable_guardrail,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_video_frames=args.num_video_frames,
        frame_repeat_negative_condition=args.frame_repeat_negative_condition,
        seed=args.seed,
        num_input_frames=args.num_input_frames,
        n_views=args.n_views,
    )

    # Handle multiple prompts if prompt file is provided
    if args.batch_input_path:
        log.info(f"Reading batch inputs from path: {args.batch_input_path}")
        prompts = read_prompts_from_file(args.batch_input_path)
    else:
        # Single prompt case
        prompts = [
            {
                "prompt": args.prompt,
                "prompt_left": args.prompt_left,
                "prompt_right": args.prompt_right,
                "prompt_back": args.prompt_back,
                "prompt_back_left": args.prompt_back_left,
                "prompt_back_right": args.prompt_back_right,
                "view_condition_input": args.view_condition_video,
                "initial_condition_input": args.initial_condition_video,
            }
        ]

    os.makedirs(args.video_save_folder, exist_ok=True)
    for i, input_dict in enumerate(prompts):
        current_view_condition_video = input_dict.pop("view_condition_input", None)
        if current_view_condition_video is None:
            log.critical("View condition input is missing, skipping world generation.")
            continue
        current_initial_condition_video = input_dict.pop("initial_condition_input", None)
        if current_initial_condition_video is None and "first_n" in args.condition_location:
            log.critical(
                "Initial condition input is missing but first_n is specified in condition location, skipping generation"
            )
            continue
        current_prompt = input_dict

        # Skip check input frames to avoid loading potentially multiple videos multiple times

        # Generate video
        generated_output = pipeline.generate(
            prompt=current_prompt,
            condition_location=args.condition_location,
            view_condition_video_path=current_view_condition_video,
            initial_condition_video_path=current_initial_condition_video,
            view_cond_start_frame=args.view_cond_start_frame,
        )
        if generated_output is None:
            log.critical("Guardrail blocked video2world generation.")
            continue
        [video_grid, video, gt_grid, gt], prompt = generated_output

        if args.batch_input_path:
            video_save_path = os.path.join(args.video_save_folder, f"{i}.mp4")
            video_grid_save_path = os.path.join(args.video_save_folder, f"{i}_grid.mp4")
            prompt_save_path = os.path.join(args.video_save_folder, f"{i}.txt")
        else:
            video_save_path = os.path.join(args.video_save_folder, f"{args.video_save_name}.mp4")
            video_grid_save_path = os.path.join(args.video_save_folder, f"{args.video_save_name}_grid.mp4")
            prompt_save_path = os.path.join(args.video_save_folder, f"{args.video_save_name}.txt")

        # Save video
        save_video(
            video=video,
            fps=args.fps,
            H=args.height,
            W=args.width,
            video_save_quality=8,
            video_save_path=video_save_path,
        )
        save_video(
            video=gt,
            fps=args.fps,
            H=args.height,
            W=args.width,
            video_save_quality=8,
            video_save_path=os.path.join(args.video_save_folder, f"{args.video_save_name}_gt.mp4"),
        )
        save_video(
            video=video_grid,
            fps=args.fps,
            H=args.height * 2,
            W=args.width * 3,
            video_save_quality=8,
            video_save_path=video_grid_save_path,
        )
        # Save prompt to text file alongside video
        with open(prompt_save_path, "wb") as f:
            for key, value in prompt.items():
                f.write(value.encode("utf-8"))
                f.write("\n".encode("utf-8"))

        log.info(f"Saved video to {video_save_path}")
        log.info(f"Saved prompt to {prompt_save_path}")

    # clean up properly
    if args.num_gpus > 1:
        parallel_state.destroy_model_parallel()
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_arguments()
    demo(args)
