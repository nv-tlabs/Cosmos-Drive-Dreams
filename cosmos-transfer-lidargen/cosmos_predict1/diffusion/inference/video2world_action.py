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

from cosmos_predict1.diffusion.inference.inference_utils import add_common_arguments, remove_argument, validate_args
from cosmos_predict1.diffusion.inference.world_generation_pipeline import DiffusionVideo2WorldActionGenerationPipeline
from cosmos_predict1.utils import distributed, log, misc
from cosmos_predict1.utils.io import save_video

torch.enable_grad(False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Action-conditional video to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)
    # remove unnecessary arguments
    remove_argument(parser, "prompt")
    remove_argument(parser, "negative_prompt")
    remove_argument(parser, "disable_prompt_upsampler")
    remove_argument(parser, "offload_prompt_upsampler")

    # Add video2world specific arguments
    parser.add_argument(
        "--diffusion_transformer_dir",
        type=str,
        default="Cosmos-Predict1-7B-Video2World_action_post-trained",
        help="DiT model weights directory name relative to checkpoint_dir",
        choices=[
            "Cosmos-Predict1-7B-Video2World_action_post-trained",
            "Cosmos-Predict1-14B-Video2World_action_post-trained",
        ],
    )
    # For image2video or long video generation
    parser.add_argument(
        "--input_image_or_video_path",
        type=str,
        help="Input video/image path for generating a single video",
    )
    parser.add_argument(
        "--action_annotation_path",
        type=str,
        help="Input action annotation path for generating a single video",
    )
    # check if num_input_frames / reusing num_video_frames is better
    parser.add_argument("--num_input_frames", type=int, default=2, help="Number of frames to condition")
    parser.add_argument("--output_path", type=str, default="outputs", help="Output path")
    parser.add_argument(
        "--num_of_latent_condition",
        type=int,
        default=1,
        help="Number of latent condition to condition on",
    )
    parser.add_argument(
        "--num_of_loops",
        type=int,
        default=1,
        help="Number of loops to generate video",
    )

    return parser.parse_args()


def demo(args: argparse.Namespace) -> None:
    """Run video-to-world generation with action control demo.

    This function loads a action-conditioned video-to-world pipeline,  including
    - Setting up the random seed for reproducibility
    - Initializing the generation pipeline with the provided configuration
    - Processing single or multiple prompts/images/videos from input
    - Generating videos from videos and action
    - Saving the generated videos and corresponding prompts to disk

    Args:
        cfg (argparse.Namespace): Configuration namespace containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (guidance, steps, dimensions)
            - Input/output settings (prompts/images/videos, save paths)

    The function will save:
        - Generated MP4 video files
    """
    misc.set_random_seed(args.seed)
    inference_type = "video2world_action"
    validate_args(args, inference_type)

    if args.num_gpus > 1:
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)

    # Initialize action-conditioned video2world generation model pipeline
    pipeline = DiffusionVideo2WorldActionGenerationPipeline(
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
        seed=args.seed,
        num_input_frames=args.num_input_frames,
    )

    generated_output = pipeline.generate(
        action_path=args.action_annotation_path,
        image_or_video_path=args.input_image_or_video_path,
    )
    if generated_output is None:
        log.critical("Guardrail blocked video2world generation.")
        return

    video = generated_output

    video_save_path = os.path.join(args.video_save_folder, f"{args.video_save_name}.mp4")

    # Save video
    save_video(
        video=video,
        fps=args.fps,
        H=args.height,
        W=args.width,
        video_save_quality=5,
        video_save_path=video_save_path,
    )

    log.info(f"Saved video to {video_save_path}")

    # clean up properly
    if args.num_gpus > 1:
        parallel_state.destroy_model_parallel()
        import torch.distributed as dist

        dist.destroy_process_group()

    return


if __name__ == "__main__":
    args = parse_arguments()
    demo(args)
