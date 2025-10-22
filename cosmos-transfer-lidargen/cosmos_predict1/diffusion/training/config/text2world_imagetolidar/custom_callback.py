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

import torch

from cosmos_predict1.utils.config import Config
from cosmos_predict1.utils.lazy_config import PLACEHOLDER
from cosmos_predict1.utils.lazy_config import LazyCall as L

from cosmos_predict1.utils.callback import LowPrecisionCallback as BaseCallback
from cosmos_predict1.diffusion.training.callbacks.iter_speed import IterSpeed
from cosmos_predict1.utils.model import Model
from cosmos_predict1.utils.trainer import Trainer
from cosmos_predict1.utils.callbacks.grad_clip import GradClip


class ImageToLidarLowPrecisionCallback(BaseCallback):
    """
    Config with non-primitive type makes it difficult to override the option.
    The callback gets precision from model.precision instead.
    """

    def __init__(self, config: Config, trainer: Trainer, update_iter: int):
        self.config = config
        self.trainer = trainer
        self.update_iter = update_iter
        self.skip_tensor_name = ["video", "lidar_hdmap","timestamp_list","pose_sensor_start_end"]
        for view_idx in range(6):
            self.skip_tensor_name.append(f"camera_c2lidar_transform_{view_idx}")
            self.skip_tensor_name.append(f"camera_intrinsics_{view_idx}")

    def on_train_start(self, model: Model, iteration: int = 0) -> None:
        assert model.precision in [
            torch.bfloat16,
            torch.float16,
            torch.half,
        ], "LowPrecisionCallback must use a low precision dtype."
        self.precision_type = model.precision

    def on_training_step_start(self, model, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data[k]):
                if k not in self.skip_tensor_name:
                    data[k] = v.to(dtype=self.precision_type)


ImageToLidar_CALLBACKS = dict(
    grad_clip=L(GradClip)(
        model_key="model",
        fsdp_enabled=True,
    ),
    low_prec=L(ImageToLidarLowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
    iter_speed=L(IterSpeed)(
        every_n=200,
        hit_thres=5,
    )
)
