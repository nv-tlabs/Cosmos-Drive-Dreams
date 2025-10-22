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

from hydra.core.config_store import ConfigStore

from cosmos_predict1.utils import log
from cosmos_predict1.utils.lazy_config import LazyDict

Cosmos_LidarTokenizer_CI8x8: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /network": "continuous_image"},
            {"override /loss": "image"},
            {"override /data_train": "lidar_range_map_rRow4"},
            {"override /data_val": "lidar_range_map_rRow4"},
            {"override /optimizer": "fused_adam"},
            {"override /callbacks": ["float32", "wandbLidar"]},
            "_self_",
        ],
        job=dict(
            group="tokenizer",
            name="Cosmos-LidarTokenizer-CI8x8",
        ),
        checkpoint=dict(
            load_path="checkpoints/Cosmos-Tokenizer-CI8x8-Lidar/Cosmos-0.1-Tokenizer-CI8x8/autoencoder.pt",
            load_training_state=False,
            save_iter=2000, 
            strict_resume=False,
            jit=dict(strict=False, dtype="float32"),
        ),
        trainer=dict(
            max_iter=10000,
            validation_iter=500,
            max_val_iter=1,
            logging_iter=100,
        ),
        model=dict(
            config=dict(
                network=dict(
                    resolution=512,  # ignore
                ),
                disc_optimizer=dict(
                    lr=0.00016,
                ),
                ema=dict(
                    enabled=False,  # disable ema
                ),
                precision="float32"
            )
        ),
        optimizer=dict(
            lr=0.00004,
        ),
    )
)

cs = ConfigStore.instance()

experiment_name = "cosmos_lidar_tokenizer"
log.info(f"Registering experiment: {experiment_name}")
cs.store(
    group="experiment",
    package="_global_",
    name=experiment_name,
    node=Cosmos_LidarTokenizer_CI8x8,
)