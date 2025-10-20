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


from __future__ import annotations

import os

import attrs
import wandb
import wandb.util
from omegaconf import DictConfig
from cosmos_predict1.utils.lazy_config.lazy import LazyConfig
from cosmos_predict1.utils import distributed, log
from cosmos_predict1.utils.easy_io import easy_io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cosmos_predict1.utils.config import Config
    from cosmos_predict1.utils.model import Model


@distributed.rank0_only
def init_wandb(config: Config, model: Model) -> None:
    """Initialize Weights & Biases (wandb) logger.

    Args:
        config (Config): The config object for the Imaginaire codebase.
        model (Model): The PyTorch model.
    """
    if isinstance(config.job, DictConfig):
        from cosmos_predict1.utils.config import JobConfig

        config_job = JobConfig(**config.job)
    else:
        config_job = config.job
    
    # init the wandb id
    wandb_id = wandb.util.generate_id()
    log.info(f"Generating new wandb ID: {wandb_id}")
    
    # write to a txt file
    with open(os.path.join(config_job.path_local, "wandb_id.txt"), "w") as f:
        f.write(wandb_id)
    
    # refactor config so that wandb better understands it
    local_safe_yaml_fp = LazyConfig.save_yaml(config, os.path.join(config_job.path_local, "config.yaml"))
    if os.path.exists(local_safe_yaml_fp):
        config_resolved = easy_io.load(local_safe_yaml_fp)
    else:
        config_resolved = attrs.asdict(config)
    # Initialize the wandb library.
    wandb.init(
        force=True,
        id=wandb_id,
        project=config_job.project,
        group=config_job.group,
        name=config_job.name,
        config=config_resolved,
        dir=config_job.path_local,
        resume="allow",
        mode=config_job.wandb_mode,
    )