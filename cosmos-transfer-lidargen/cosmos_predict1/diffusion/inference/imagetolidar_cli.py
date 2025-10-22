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
import importlib
import torch

from cosmos_predict1.utils import distributed, log
from cosmos_predict1.diffusion.inference.imagetolidar_pipeline import RGB2LidarInference

from cosmos_predict1.utils.config_helper import get_config_module, override

def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description="A cli to calculate mean and std of the latent space")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path")
    parser.add_argument("--experiment", type=str, default="text2world_imagetolidar", help="Experiment name")
    parser.add_argument("--config", default="cosmos_predict1/diffusion/training/config/config_lidar.py", help="Path to the config file")
    parser.add_argument("--save_iter", type=int, default=1, help="Save mean and std every save_iter iterations")
    parser.add_argument("--save_dir", type=str, default="results", help="Save file path")
    parser.add_argument("--context_parallel_size", type=int, default=1, help="Context parallel size")
    parser.add_argument("--ddim_iters", type=int, default=35, help="Number of iterations at inference time")
    parser.add_argument("--guidance", type=float, default=1.5, help="Guidance weight at inference time")
    parser.add_argument("--sigma_min", type=float, default=0.02, help="Sigma min at inference time")
    parser.add_argument("--near_buffer", type=float, default=1.5, help="Near buffer")
    parser.add_argument("--far_buffer", type=float, default=10, help="Far buffer")
    parser.add_argument("--n_views", type=int, default=3, help="Number of cameras")
    parser.add_argument("--vis", type=int, default=1, help="if zero, do not save vis")
    parser.add_argument(
        "--filter_outlier", action="store_true", help="Filter outlier point cloud using statistical filter"
    )
    parser.add_argument(
        "opts",
        help="""
            Modify config options at the end of the command. For Yacs configs, use
            space-separated "PATH.KEY VALUE" pairs.
            For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    config = override(config, args.opts)
    assert args.context_parallel_size == 1, "Context parallel size must be 1 for imagetolidar"

    if args.dryrun:
        log.info("Config:\n" + config.pretty_print(use_color=True))
    else:
        distributed.init()
        with torch.no_grad():
            inferencer = RGB2LidarInference(args, config)
            inferencer.run_inference()


if __name__ == "__main__":
    main()
