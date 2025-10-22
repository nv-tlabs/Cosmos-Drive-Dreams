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

from cosmos_predict1.diffusion.training.networks.general_dit_lvg import VideoExtendGeneralDIT
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict

Cosmos_Predict1_Video2World_7B_Action_Post_trained: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Cosmos_Predict1_Video2World_7B",
        ],
        job=dict(
            name="Cosmos_Predict1_Video2World_7B_Action_Post_trained",
        ),
        model=dict(
            # Use 16x2x32x40 latent shape for training
            latent_shape=[
                16,  # Latent channel dim
                2,  # Latent temporal dim
                32,  # Latent height dim
                40,  # Latent width dim
            ],
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
                extra_per_block_abs_pos_emb=False,
                legacy_patch_emb=False,
            ),
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    first_random_n_num_condition_t_max=1,
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                )
            ),
            tokenizer=dict(
                video_vae=dict(pixel_chunk_duration=1),
            ),
        ),
    )
)


cs = ConfigStore.instance()
for _item in [
    Cosmos_Predict1_Video2World_7B_Action_Post_trained,
]:
    cs.store(group="experiment", package="_global_", name=_item["job"]["name"], node=_item)
