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

from cosmos_predict1.diffusion.config.base.conditioner import (
    BaseVideoConditionerConfig,
    VideoConditionerFpsSizePaddingConfig,
    VideoConditionerFpsSizePaddingFrameRepeatConfig,
    VideoExtendConditionerConfig,
    VideoExtendConditionerFrameRepeatConfig,
    ViewConditionedVideoExtendConditionerFrameRepeatConfig,
)
from cosmos_predict1.diffusion.config.base.net import FADITV2_14B_Config, FADITV2_Multiview_Config, FADITV2Config
from cosmos_predict1.diffusion.config.base.tokenizer import get_cosmos_diffusion_tokenizer_comp8x8x8, get_cosmos_diffusion_lidar_tokenizer


def register_net(cs):
    cs.store(
        group="net",
        package="model.net",
        name="faditv2_7b",
        node=FADITV2Config,
    )
    cs.store(
        group="net",
        package="model.net",
        name="faditv2_14b",
        node=FADITV2_14B_Config,
    )
    cs.store(
        group="net",
        package="model.net",
        name="faditv2_multiview_7b",
        node=FADITV2_Multiview_Config,
    )


def register_conditioner(cs):
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="basic",
        node=BaseVideoConditionerConfig,
    )
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="add_fps_image_size_padding_mask",
        node=VideoConditionerFpsSizePaddingConfig,
    )
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="video_cond",
        node=VideoExtendConditionerConfig,
    )
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="add_fps_image_size_padding_mask_frame_repeat",
        node=VideoConditionerFpsSizePaddingFrameRepeatConfig,
    )
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="video_cond_frame_repeat",
        node=VideoExtendConditionerFrameRepeatConfig,
    )
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="view_conditioned_video_frame_repeat_cond",
        node=ViewConditionedVideoExtendConditionerFrameRepeatConfig,
    )


def register_tokenizer(cs):
    cs.store(
        group="tokenizer",
        package="model.tokenizer",
        name="cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624",
        node=get_cosmos_diffusion_tokenizer_comp8x8x8(resolution="720", chunk_duration=121),
    )
    
    cs.store(
        group="tokenizer",
        package="model.tokenizer",
        name="cosmos_lidar_diffusion_tokenizer",
        node=get_cosmos_diffusion_lidar_tokenizer(resolution="720", chunk_duration=1),
    )


def register_configs():
    cs = ConfigStore.instance()

    register_net(cs)
    register_conditioner(cs)
    register_tokenizer(cs)
