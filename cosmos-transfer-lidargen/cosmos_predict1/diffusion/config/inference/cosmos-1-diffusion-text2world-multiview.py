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

from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict

Cosmos_Predict1_Text2World_7B_Multiview: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Cosmos_Predict1_Text2World_7B",
            {"override /net": "faditv2_multiview_7b"},
            {"override /conditioner": "add_fps_image_size_padding_mask_frame_repeat"},
            "_self_",
        ],
        job=dict(
            group="Text2World",
            name="Cosmos_Predict1_Text2World_7B_Multiview",
        ),
        model=dict(
            latent_shape=[
                16,
                16,
                88,
                160,
            ],
            net=dict(
                extra_per_block_abs_pos_emb=True,
                extra_per_block_abs_pos_emb_type="sincos",
            ),
            tokenizer=dict(
                video_vae=dict(
                    pixel_chunk_duration=57,
                )
            ),
        ),
    )
)

Cosmos_Predict1_Text2World_7B_Multiview_post_trained: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Cosmos_Predict1_Text2World_7B",
            {"override /net": "faditv2_multiview_7b"},
            {"override /conditioner": "add_fps_image_size_padding_mask_frame_repeat"},
            "_self_",
        ],
        job=dict(
            group="Text2World",
            name="Cosmos_Predict1_Text2World_7B_Multiview_post_trained",
        ),
        model=dict(
            net=dict(
                n_views=5,
                view_condition_dim=3,
                add_repeat_frame_embedding=False,
                extra_per_block_abs_pos_emb=True,
                extra_per_block_abs_pos_emb_type="sincos",
            ),
            latent_shape=[
                16,
                16,
                88,
                160,
            ],
            tokenizer=dict(
                video_vae=dict(
                    pixel_chunk_duration=57,
                )
            ),
        ),
    )
)

cs = ConfigStore.instance()
for _item in [
    Cosmos_Predict1_Text2World_7B_Multiview,
    Cosmos_Predict1_Text2World_7B_Multiview_post_trained,
]:
    cs.store(group="experiment", package="_global_", name=_item["job"]["name"], node=_item)
