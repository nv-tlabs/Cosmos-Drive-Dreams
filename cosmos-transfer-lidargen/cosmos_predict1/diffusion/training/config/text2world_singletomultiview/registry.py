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
    FPSConfig,
    FrameRepeatConfig,
    ImageSizeConfig,
    NumFramesConfig,
    PaddingMaskConfig,
    TextConfig,
    VideoCondBoolConfig,
)
from cosmos_predict1.diffusion.training.conditioner import ViewConditionedVideoExtendConditioner
from cosmos_predict1.diffusion.training.config.text2world_singletomultiview.experiment import register_experiments
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict

ViewConditionedVideoExtendConditionerConfig: LazyDict = L(ViewConditionedVideoExtendConditioner)(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
    video_cond_bool=VideoCondBoolConfig(),
    frame_repeat=FrameRepeatConfig(),
)


def register_configs():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="view_conditioned_video_frame_repeat_cond",
        node=ViewConditionedVideoExtendConditionerConfig,
    )
    register_experiments(cs)
