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

from cosmos_predict1.diffusion.networks.general_dit_view_extend_multiview import MultiviewExtensionGeneralDIT
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict

Cosmos_Predict1_Video2World_7B_ViewExtend_Multiview: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Cosmos_Predict1_Text2World_7B_Multiview",
            {"override /conditioner": "video_cond_frame_repeat"},
            "_self_",
        ],
        job=dict(
            group="Text2World",
            name="Cosmos_Predict1_Video2World_7B_ViewExtend_Multiview",
        ),
        model=dict(
            latent_shape=[
                16,
                16,
                72,
                128,
            ],
            net=L(MultiviewExtensionGeneralDIT)(
                extra_per_block_abs_pos_emb=True,
                extra_per_block_abs_pos_emb_type="sincos",
                n_views=6,  # this indicates how many views to generate, can be overwritten at inference time
                n_views_emb=7,  # this indicates how many views the model has trained for, should not be changed at inference time
                view_condition_dim=6,
                add_repeat_frame_embedding=True,
            ),
            conditioner=dict(video_cond_bool=dict()),
        ),
    )
)


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=Cosmos_Predict1_Video2World_7B_ViewExtend_Multiview["job"]["name"],
    node=Cosmos_Predict1_Video2World_7B_ViewExtend_Multiview,
)
