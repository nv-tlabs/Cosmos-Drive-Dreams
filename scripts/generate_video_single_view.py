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
import json
import os
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Workaround to suppress MP warning

import sys
from io import BytesIO

import torch

from cosmos_transfer1.checkpoints import BASE_7B_CHECKPOINT_AV_SAMPLE_PATH, BASE_7B_CHECKPOINT_PATH
from cosmos_transfer1.utils import log, misc
from cosmos_transfer1.utils.io import read_prompts_from_file, save_video
from cosmos_transfer1.diffusion.inference.inference_utils import validate_controlnet_specs
from cosmos_transfer1.diffusion.inference.preprocessors import Preprocessors
from cosmos_transfer1.diffusion.inference.world_generation_pipeline import DiffusionControl2WorldGenerationPipeline


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

    
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control to world generation demo script", conflict_handler="resolve")

    # Add transfer specific arguments
    parser.add_argument("--caption_path", type=str, required=True, help="folder containing the json files of captions")
    parser.add_argument("--input_path", type=str, required=True, help="folder containing the hdmap/lidar condition videos")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all.",
        help="negative prompt which the sampled video condition on",
    )
    parser.add_argument(
        "--input_video_path",
        type=str,
        default="",
        help="Optional input RGB video path",
    )
    parser.add_argument(
        "--num_input_frames",
        type=int,
        default=1,
        help="Number of conditional frames for long video generation",
        choices=[1],
    )
    parser.add_argument("--sigma_max", type=float, default=80.0, help="sigma_max for partial denoising")
    parser.add_argument(
        "--blur_strength",
        type=str,
        default="medium",
        choices=["very_low", "low", "medium", "high", "very_high"],
        help="blur strength.",
    )
    parser.add_argument(
        "--canny_threshold",
        type=str,
        default="medium",
        choices=["very_low", "low", "medium", "high", "very_high"],
        help="blur strength of canny threshold applied to input. Lower means less blur or more detected edges, which means higher fidelity to input.",
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
    if cfg.batch_input_path:
        log.info(f"Reading batch inputs from path: {cfg.batch_input_path}")
        prompts = read_prompts_from_file(cfg.batch_input_path)
    else:
        # Single prompt case
        prompts = [{"prompt": prompt, "visual_input": cfg.input_video_path}]

    os.makedirs(cfg.video_save_folder, exist_ok=True)
    for i, input_dict in enumerate(prompts):
        current_prompt = input_dict.get("prompt", None)
        current_video_path = input_dict.get("visual_input", None)

        # if control inputs are not provided, run respective preprocessor (for seg and depth)
        preprocessors(current_video_path, current_prompt, control_inputs, cfg.video_save_folder)

        # Generate video
        generated_output = pipeline.generate(
            prompt=current_prompt,
            video_path=current_video_path,
            negative_prompt=cfg.negative_prompt,
            control_inputs=control_inputs,
            save_folder=cfg.video_save_folder,
        )
        if generated_output is None:
            log.critical("Guardrail blocked generation.")
            continue
        video, prompt = generated_output
        video = video[0]
        prompt = prompt[0]
        if cfg.batch_input_path:
            video_save_path = os.path.join(cfg.video_save_folder, f"{i}.mp4")
            prompt_save_path = os.path.join(cfg.video_save_folder, f"{i}.txt")
        else:
            video_save_path = os.path.join(cfg.video_save_folder, f"{video_save_name}.mp4")
            prompt_save_path = os.path.join(cfg.video_save_folder, f"{video_save_name}.txt")

        if device_rank == 0:
            # Save video
            os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
            save_video(
                video=video,
                fps=cfg.fps,
                H=video.shape[1],
                W=video.shape[2],
                video_save_quality=8,
                video_save_path=video_save_path,
            )

            # Save prompt to text file alongside video
            with open(prompt_save_path, "wb") as f:
                f.write(prompt.encode("utf-8"))

            log.info(f"Saved video to {video_save_path}")
            log.info(f"Saved prompt to {prompt_save_path}")



if __name__ == "__main__":
    args, control_inputs = parse_arguments()

    caption_path = os.path.abspath(args.caption_path)
    data_path = os.path.abspath(args.input_path)
    args.video_save_folder = os.path.abspath(args.video_save_folder)
    # load all the json files in the caption_path

    json_files = [f for f in os.listdir(caption_path) if f.endswith('.json')]
    # load the json file and get the prompt
    prompts = []
    hdmaps = []
    lidars = []
    video_save_names = []

    for json_file in tqdm(json_files):
        sample_name = json_file.split('.')[0]
        with open(os.path.join(caption_path, json_file), 'r') as f:
            data = json.load(f)
            
        for variation in data.keys():
            prompt = data[variation]
            hdmap = os.path.join(data_path, "hdmap", "ftheta_camera_front_wide_120fov", f"{sample_name}_0.mp4")
            # make sure the file exists
            if not os.path.exists(hdmap):
                print(f"hdmap file {hdmap} does not exist")
                continue

            if "lidar" in control_inputs.keys():
                lidar = os.path.join(data_path, "lidar", "ftheta_camera_front_wide_120fov", f"{sample_name}_0.mp4")
                if not os.path.exists(lidar):
                    print(f"lidar file {lidar} does not exist")
                    continue
            else:
                lidar = None

            video_save_name = f"{sample_name}_{variation}"

            # check if output already exists
            video_save_path = os.path.join(args.video_save_folder, f"{video_save_name}.mp4")
            if os.path.exists(video_save_path):
                print(f"video {video_save_name} already exists")
                continue
            
            # append the prompt and hdmap to the list
            prompts.append(prompt)
            hdmaps.append(hdmap)
            lidars.append(lidar)
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

    checkpoint = BASE_7B_CHECKPOINT_AV_SAMPLE_PATH if cfg.is_av_sample else BASE_7B_CHECKPOINT_PATH

    # Initialize transfer generation model pipeline
    control_inputs = validate_controlnet_specs(cfg, control_inputs)
    
    pipeline = DiffusionControl2WorldGenerationPipeline(
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
        sigma_max=cfg.sigma_max,
        blur_strength=cfg.blur_strength,
        canny_threshold=cfg.canny_threshold,
        upsample_prompt=cfg.upsample_prompt,
        offload_prompt_upsampler=cfg.offload_prompt_upsampler,
        process_group=process_group,
        regional_prompts=[]
    )

    for prompt, hdmap, lidar, video_save_name in zip(prompts, hdmaps, lidars, video_save_names):
        current_control_inputs = control_inputs.copy()
        if "hdmap" in current_control_inputs.keys():
            current_control_inputs["hdmap"]["input_control"] = hdmap
        if "lidar" in current_control_inputs.keys():
            current_control_inputs["lidar"]["input_control"] = lidar
        # run the demo
        demo(
            cfg,
            pipeline,
            current_control_inputs,
            video_save_name,
            prompt,
        )

    os.chdir(current_dir)
    # clean up properly
    if cfg.num_gpus > 1:
        parallel_state.destroy_model_parallel()
        import torch.distributed as dist

        dist.destroy_process_group()
