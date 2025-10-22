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

import omegaconf

from cosmos_predict1.diffusion.training.module.pretrained_vae import VideoJITTokenizer, JointImageLidarTokenizer, JITVAE
from cosmos_predict1.utils.lazy_config import LazyCall as L

TOKENIZER_OPTIONS = {}


def tokenizer_register(key):
    def decorator(func):
        TOKENIZER_OPTIONS[key] = func
        return func

    return decorator


@tokenizer_register("cosmos_diffusion_tokenizer_comp8x8x8")
def get_cosmos_tokenizer_comp8x8x8(
    resolution: str,
    chunk_duration: int,
) -> omegaconf.dictconfig.DictConfig:
    assert resolution in ["512", "720"]

    pixel_chunk_duration = chunk_duration
    temporal_compression_factor = 8
    spatial_compression_factor = 8

    return L(VideoJITTokenizer)(
        name="cosmos_diffusion_tokenizer_comp8x8x8",
        enc_fp="checkpoints/Cosmos-Transfer-LidarGen/Cosmos-Tokenize1-CV8x8x8-720p/encoder.jit",
        dec_fp="checkpoints/Cosmos-Transfer-LidarGen/Cosmos-Tokenize1-CV8x8x8-720p/decoder.jit",
        mean_std_fp="checkpoints/Cosmos-Transfer-LidarGen/Cosmos-Tokenize1-CV8x8x8-720p/mean_std.pt",
        latent_ch=16,
        is_bf16=True,
        pixel_chunk_duration=pixel_chunk_duration,
        temporal_compression_factor=temporal_compression_factor,
        spatial_compression_factor=spatial_compression_factor,
        spatial_resolution=resolution,
    )



@tokenizer_register("cosmos_lidar_diffusion_tokenizer")
def get_cosmos_diffusion_lidar_tokenizer(resolution: str, chunk_duration: int) -> omegaconf.dictconfig.DictConfig:
    pixel_chunk_duration = chunk_duration
    temporal_compression_factor = 8
    spatial_compression_factor = 8
    
    return L(JointImageLidarTokenizer)(
        name="lidar_rgb_image_vae",
        squeeze_for_lidar=True,
        squeeze_for_image=False,
        latent_ch=16,
        image_vae=L(JITVAE)(
            name="CV8x8x8",
            mean_std_fp="checkpoints/Cosmos-Transfer-LidarGen/Cosmos-Tokenize1-CV8x8x8-720p-image/mean_std.pt",
            enc_fp="checkpoints/Cosmos-Transfer-LidarGen/Cosmos-Tokenize1-CV8x8x8-720p-image/encoder.jit",
            dec_fp="checkpoints/Cosmos-Transfer-LidarGen/Cosmos-Tokenize1-CV8x8x8-720p-image/decoder.jit",
            latent_ch=16,
            is_image=False,
            is_bf16=True,
        ),
        lidar_vae=L(JITVAE)(
            name="CI512_rRow4_mValue-1_fp32",
            mean_std_fp="checkpoints/Cosmos-Tokenizer-CI8x8-Lidar/mean_std.pt",
            enc_fp="checkpoints/Cosmos-Tokenizer-CI8x8-Lidar/encoder.jit",
            dec_fp="checkpoints/Cosmos-Tokenizer-CI8x8-Lidar/decoder.jit",
            is_image=True,
            is_bf16=False,
            latent_ch=16,
        ),
        video_vae=L(VideoJITTokenizer)(
            name="cosmos_predict1_tokenizer",
            enc_fp="checkpoints/Cosmos-Transfer-LidarGen/Cosmos-Tokenize1-CV8x8x8-720p/encoder.jit",
            dec_fp="checkpoints/Cosmos-Transfer-LidarGen/Cosmos-Tokenize1-CV8x8x8-720p/decoder.jit",
            mean_std_fp="checkpoints/Cosmos-Transfer-LidarGen/Cosmos-Tokenize1-CV8x8x8-720p/mean_std.pt",
            latent_ch=16,
            is_bf16=True,
            pixel_chunk_duration=pixel_chunk_duration,
            temporal_compression_factor=temporal_compression_factor,
            spatial_compression_factor=spatial_compression_factor,
            spatial_resolution=resolution,
        ),
    )