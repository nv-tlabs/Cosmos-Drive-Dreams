# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import torch
from tqdm import tqdm

from cosmos_predict1.tokenizer.inference.image_lib import ImageTokenizer
from cosmos_predict1.tokenizer.inference.utils import pad_image_batch, unpad_image_batch
from cosmos_predict1.utils.lidar_rangemap import normalize_range_map, unnormalize_range_map, RangeMapDownsampler, range_map_to_ray_directions, load_pandar128_elevations


class LidarTokenizer(ImageTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @torch.no_grad()
    def autoencode(self, image: np.ndarray) -> np.ndarray:
        """Reconstrcuts a numpy image after embedding into a latent first.

        Args:
            image: The input image BxHxWx3 layout, range [0..255].
        Returns:
            The reconstructed image, layout BxHxWx3, range [0..255].
        """
        input_array = image.transpose(0, 3, 1, 2)
        input_tensor = torch.from_numpy(input_array).to(self._dtype).to(self._device)
        
        if self._full_model is not None:
            output_tensor = self._full_model(input_tensor)
            output_tensor = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
        else:
            output_latent = self.encode(input_tensor)[0]
            output_tensor = self.decode(output_latent)

        padded_output_image = output_tensor.float().cpu().numpy()
        padded_output_image = padded_output_image.transpose(0, 2, 3, 1)

        return padded_output_image

    
    @torch.no_grad()
    def forward(self, image: np.ndarray) -> np.ndarray:
        """Reconstructs an image using a pre-trained tokenizer.

        Args:
            image: The input image BxHxWxC layout, range [0..255].
        Returns:
            The reconstructed image in range [0..255], layout BxHxWxC.
        """
        padded_image, crop_region = pad_image_batch(image)
        padded_output_image = self.autoencode(padded_image)
        return unpad_image_batch(padded_output_image, crop_region)

class LidarProcessor:
    def __init__(
        self,
        checkpoint_enc=None,
        checkpoint_dec=None,
        dtype="bfloat16",
        device="cuda",
        min_range=0.75,
        max_range=75,
        n_rows_repeat=4,
        n_cols_repeat=1,
        downsample_factor_row=1,
        downsample_factor_col=2,
        downsample_method="scatter_min",
    ):
        """Initialize the LidarProcessor.

        Args:
            checkpoint: Path to full autoencoder JIT model
            checkpoint_enc: Path to encoder JIT model
            checkpoint_dec: Path to decoder JIT model
            mode: Backend mode ('torch' or 'jit')
            short_size: Size to resample inputs
            dtype: Precision type
            device: Device for model execution
            max_range: Maximum range value
            n_rows_repeat: Number of times to repeat each row
            n_cols_repeat: Number of times to repeat each column
            inverse_depth: Whether to process depth as inverse depth
            log_space: Whether to normalize in log space
            dynamic_range: Whether to dynamically adjust the range
            stack_frames: Whether to stack frames
        """
        self.min_range = min_range
        self.max_range = max_range
        self.n_rows_repeat = n_rows_repeat
        self.n_cols_repeat = n_cols_repeat
        self.downsample_factor_row = downsample_factor_row
        self.downsample_factor_col = downsample_factor_col
        self.downsample_method = downsample_method
        self.max_value = 1
        self.min_value = -1

        if checkpoint_enc is None and checkpoint_dec is None:
            raise ValueError("checkpoint_enc and checkpoint_dec must be provided")
        
        self.range_map_downsampler = RangeMapDownsampler(
            row_factor=downsample_factor_row,
            col_factor=downsample_factor_col,
            method=downsample_method,
        )

        # Initialize the autoencoder
        self.autoencoder = LidarTokenizer(
            checkpoint_enc=checkpoint_enc,
            checkpoint_dec=checkpoint_dec,
            device=device,
            dtype=dtype,
        )
        

    def process_video(self, input_video):
        """Process a range image video.

        Args:
            input_video: Input video array of shape [N, H, W], the values are the effective range values
            max_frames: Maximum number of frames to process

        Returns:
            Tuple of (processed_video, original_video, difference)
        """
        valid_mask = (input_video > self.min_range + 0.1) & (input_video < self.max_range - 0.1)
        
        elevation_angles = load_pandar128_elevations()
        ray_directions = range_map_to_ray_directions(input_video.shape[-1], elevation_angles)  # shape: (H, W, 3)
        ray_directions = np.repeat(ray_directions[np.newaxis, ...], input_video.shape[0], axis=0)
        
        # downsample the range map if needed 
        input_video, [ray_directions, valid_mask] = self.range_map_downsampler.downsample(
            input_video,
            extra_maps=[ray_directions, valid_mask],
        )
        
        input_video = np.clip(input_video, self.min_range, self.max_range)
        n_frames = input_video.shape[0]
        normalised_video = normalize_range_map(input_video, self.max_range, self.min_range, self.min_value, False)       
        output_images_list = []
        normalised_video = np.repeat(normalised_video[..., np.newaxis], 3, axis=-1)  # N H W 3
        for frame_idx in tqdm(range(n_frames)):
            image = normalised_video[frame_idx]
            image = np.repeat(image, self.n_rows_repeat, axis=0)
            image = np.repeat(image, self.n_cols_repeat, axis=1)

            batch_image = np.expand_dims(image, axis=0)
            output_image = self.autoencoder(batch_image)[0]

            output_image = output_image[
                self.n_rows_repeat // 2 :: self.n_rows_repeat,
                self.n_cols_repeat // 2 :: self.n_cols_repeat,
                :,
            ]
            output_image = output_image.mean(axis=-1)  # H W
            output_image = np.clip(output_image, self.min_value, self.max_value)

            output_image, _ = unnormalize_range_map(
                output_image,
                self.max_range,
                self.min_range,
                self.min_value,
                False,
                near_buffer=0.1,
                far_buffer=0.1,
                valid_mask=valid_mask[frame_idx],
            )

            output_images_list.append(output_image)
        

        output_video = np.stack(output_images_list, axis=0)
        return input_video, output_video, valid_mask, ray_directions