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

from typing import Dict, Tuple, Union
import numpy as np
import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor


from cosmos_predict1.utils import log
from cosmos_predict1.diffusion.training.conditioner import (
    DataType,
    VideoExtendCondition,
    ViewConditionedVideoExtendCondition,
)
from cosmos_predict1.diffusion.training.models.model_image import diffusion_fsdp_class_decorator
from cosmos_predict1.diffusion.training.context_parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict1.diffusion.training.models.model import broadcast_condition, _broadcast
from cosmos_predict1.diffusion.training.models.view_extend_model_multiview import MultiviewViewExtendDiffusionModel


class ImageToLidarModel(MultiviewViewExtendDiffusionModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.diffusion_dtype = torch.bfloat16
        
        self.rgb_latent_shape = config.rgb_latent_shape
        self.lidar_latent_shape = config.lidar_latent_shape
        self.rgb_shape = config.rgb_shape
        self.lidar_shape = config.lidar_shape
        

    # to update
    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # to update
    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    
    def pad_and_concat_lidar_rgb(
        self, lidar_latent_state: torch.Tensor, rgb_latent_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Pad the latent state to have the same spatial size
        Input:
            lidar_latent_state: B, C, T, H, W
            rgb_latent_state: B, C, T, H, W
        Output:
            padded_latent_state: B, C, 2*T, H, W
        """
        B_lidar, C_lidar, T_lidar, H_lidar, W_lidar = lidar_latent_state.shape
        B_rgb, C_rgb, T_rgb, H_rgb, W_rgb = rgb_latent_state.shape


        assert B_lidar == B_rgb
        assert C_lidar == C_rgb

        # Create padding tensor with zeros of target shape (88x224)
        padded_latent = torch.zeros(
            (B_lidar, C_lidar, T_lidar + T_rgb, max(H_lidar, H_rgb), max(W_lidar, W_rgb)),
            device=lidar_latent_state.device,
            dtype=lidar_latent_state.dtype,
        )

        # Place the lidar latent state in the upper left corner
        padded_latent[:, :, :T_lidar, :H_lidar, :W_lidar] = lidar_latent_state

        # Place the rgb latent state in the upper right corner
        padded_latent[:, :, T_lidar:, :H_rgb, :W_rgb] = rgb_latent_state

        return padded_latent.contiguous()
    
    
    def split_latent_state(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split the latent state into lidar and rgb latent states
        latent_state: B, C, 2*T, H, W
        """
        T = latent_state.shape[2]
        lidar_latent_state = latent_state[:, :, :1, :self.lidar_latent_shape[0], :self.lidar_latent_shape[1]]
        rgb_latent_state = latent_state[:, :, 1:, :self.rgb_latent_shape[0], :self.rgb_latent_shape[1]]
        return lidar_latent_state.contiguous(), rgb_latent_state.contiguous()
    
    def split_raw_state(self, raw_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split the raw state into lidar and rgb raw states
        raw_state: B, C, 2*T, H, W
        """
        T = raw_state.shape[2]  
        lidar_raw_state = raw_state[:, :, :1, :self.lidar_shape[0], :self.lidar_shape[1]]
        rgb_raw_state = raw_state[:, :, 1:, :self.rgb_shape[0], :self.rgb_shape[1]]
        return lidar_raw_state.contiguous(), rgb_raw_state.contiguous()
    
    
    def add_condition_video_indicator_and_video_input_mask(
        self, latent_state: torch.Tensor, condition: VideoExtendCondition, num_condition_t: Union[int, None] = None
    ) -> VideoExtendCondition:
        """Add condition_video_indicator and condition_video_input_mask to the condition object for video conditioning.
        condition_video_indicator is a binary tensor indicating the condition region in the latent state. 1x1xTx1x1 tensor.
        condition_video_input_mask will be concat with the input for the network.
        Args:
            latent_state (torch.Tensor): latent state tensor in shape B,C,T,H,W
            condition (VideoExtendCondition): condition object
            num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_cam"
        Returns:
            VideoExtendCondition: updated condition object
        """
        T = latent_state.shape[2]
        latent_dtype = latent_state.dtype
        condition_video_indicator = torch.zeros(1, 1, T, 1, 1, device=latent_state.device).type(
            latent_dtype
        )  # 1 for condition region

        condition_video_indicator = rearrange(
            condition_video_indicator, "B C (V T) H W -> (B V) C T H W", V=self.n_views
        )
        if self.config.conditioner.video_cond_bool.condition_location == "first_cam":
            # num_condition_t_max = self.config.conditioner.video_cond_bool.first_random_n_num_condition_t_max
            # assert (
            #     num_condition_t_max < T
            # ), f"num_condition_t_max should be less than T, get {num_condition_t_max}, {T}"
            # condition_video_indicator[:, :, 1:num_condition_t_max] += 1.0
            condition_video_indicator[1:] += 1.0
        else:
            raise NotImplementedError(
                f"condition_location {self.config.conditioner.video_cond_bool.condition_location} not implemented; training={self.training}"
            )

        condition_video_indicator = rearrange(
            condition_video_indicator, "(B V) C T H W -> B C (V T) H W", V=self.n_views
        )
        
        condition.gt_latent = latent_state
        condition.condition_video_indicator = condition_video_indicator

        B, C, T, H, W = latent_state.shape
        # Create additional input_mask channel, this will be concatenated to the input of the network
        # See design doc section (Implementation detail A.1 and A.2) for visualization
        ones_padding = torch.ones((B, 1, T, H, W), dtype=latent_state.dtype, device=latent_state.device)
        zeros_padding = torch.zeros((B, 1, T, H, W), dtype=latent_state.dtype, device=latent_state.device)
        assert condition.video_cond_bool is not None, "video_cond_bool should be set"

        # The input mask indicate whether the input is conditional region or not
        if condition.video_cond_bool:  # Condition one given video frames
            condition.condition_video_input_mask = (
                condition_video_indicator * ones_padding + (1 - condition_video_indicator) * zeros_padding
            )
        else:  # Unconditional case, use for cfg
            condition.condition_video_input_mask = zeros_padding

        to_cp = self.net.is_context_parallel_enabled
        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=True, to_cp=to_cp)
        else:
            assert not to_cp, "parallel_state is not initialized, context parallel should be turned off."

        return condition
    
    
    def get_data_and_condition(
        self, data_batch: dict[str, Tensor], num_condition_t: Union[int, None] = None
    ) -> Tuple[Tensor, ViewConditionedVideoExtendCondition]:
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        input_key = self.input_data_key  # by default it is video key
        is_image_batch = self.is_image_batch(data_batch)
        is_video_batch = not is_image_batch

        # Broadcast data and condition across TP and CP groups.
        for key in sorted(data_batch):
            data_batch[key] = _broadcast(data_batch[key], to_tp=True, to_cp=is_video_batch)

        if is_image_batch:
            input_key = self.input_image_key

        # GT Latent state
        lidar_raw_state = data_batch[input_key]
        with torch.no_grad():
            lidar_latent_state = self.vae.encode_lidar(lidar_raw_state).contiguous()
            lidar_latent_state = (
                lidar_latent_state.to(self.diffusion_dtype)
                if lidar_latent_state.dtype != self.diffusion_dtype
                else lidar_latent_state
            )
            lidar_raw_state = (
                lidar_raw_state.to(self.diffusion_dtype)
                if lidar_raw_state.dtype != self.diffusion_dtype
                else lidar_raw_state
            )
        
        # RGB latent state
        rgb_images_raw_state = [data_batch[f"image_rgb_{view_idx}"] for view_idx in data_batch["view_indices"][0][:-1]]
        rgb_images_raw_state = torch.cat(rgb_images_raw_state, dim=0).contiguous() # B*V, C, 1, H, W
        rgb_images_latent_state = self.vae.encode_image(rgb_images_raw_state).contiguous()  # B*V, C, 1, H, W
        
        # reshape to B, C, T, H, W
        rgb_images_latent_state = rearrange(rgb_images_latent_state, "(B V) C T H W -> B C (V T) H W", V=self.n_views - 1)
        rgb_images_raw_state = rearrange(rgb_images_raw_state, "(B V) C T H W -> B C (V T) H W", V=self.n_views - 1)

        rgb_images_latent_state = (
            rgb_images_latent_state.to(self.diffusion_dtype)
            if rgb_images_latent_state.dtype != self.diffusion_dtype
            else rgb_images_latent_state
        )
        rgb_images_raw_state = (
            rgb_images_raw_state.to(self.diffusion_dtype) if rgb_images_raw_state.dtype != self.diffusion_dtype else rgb_images_raw_state
        )
        
        latent_state = self.pad_and_concat_lidar_rgb(lidar_latent_state, rgb_images_latent_state)
        raw_state = self.pad_and_concat_lidar_rgb(lidar_raw_state, rgb_images_raw_state)
        condition = self.conditioner(data_batch)
        
        # set the gt latent state as well as the condition_video_indicator
        condition = self.add_condition_video_indicator_and_video_input_mask(latent_state, condition, num_condition_t)
        
        # this part is for handling the temporal compression factor, doesn't affect our image model
        if condition.data_type == DataType.VIDEO and "view_indices" in data_batch:
            comp_factor = self.vae.temporal_compression_factor
            # n_frames = data_batch['num_frames']
            view_indices = rearrange(data_batch["view_indices"], "B (V T) -> B V T", V=self.n_views)
            view_indices_B_V_0 = view_indices[:, :, :1]
            view_indices_B_V_1T = view_indices[:, :, 1:-1:comp_factor]
            view_indices_B_V_T = torch.cat([view_indices_B_V_0, view_indices_B_V_1T], dim=-1)
            condition.view_indices_B_T = rearrange(view_indices_B_V_T, "B V T -> B (V T)", V=self.n_views)
            condition.data_n_cameras = self.n_views
            log.debug(f"condition.data_n_cameras {self.n_views}")
        return raw_state, latent_state, condition
    
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        sigma_min: float = 0.02,
        condition_latent: Union[torch.Tensor, None] = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        add_input_frames_guidance: bool = False,
        guidance_other: Union[float, None] = None,
    ) -> Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Different from the base model, this function support condition latent as input, it will create a differnt x0_fn if condition latent is given.
        If this feature is stablized, we could consider to move this function to the base model.

        Args:
            condition_latent (Optional[torch.Tensor]): latent tensor in shape B,C,T,H,W as condition to generate video.
            num_condition_t (Optional[int]): number of condition latent T, if None, will use the whole first half

            add_input_frames_guidance (bool): add guidance to the input frames, used for cfg on input frames
        """
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        if is_image_batch:
            log.debug("image batch, call base model generate_samples_from_batch")
            return super().generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                seed=seed,
                state_shape=state_shape,
                n_sample=n_sample,
                is_negative_prompt=is_negative_prompt,
                num_steps=num_steps,
            )
        if n_sample is None:
            input_key = self.input_image_key if is_image_batch else self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            if is_image_batch:
                state_shape = (self.state_shape[0], 1, *self.state_shape[2:])  # C,T,H,W
            else:
                log.debug(f"Default Video state shape is used. {self.state_shape}")
                state_shape = self.state_shape

        assert condition_latent is not None, "condition_latent should be provided"
        x0_fn = self.get_x0_fn_from_batch_with_condition_latent(
            data_batch,
            guidance,
            is_negative_prompt=is_negative_prompt,
            condition_latent=condition_latent,
            num_condition_t=num_condition_t,
            condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            add_input_frames_guidance=add_input_frames_guidance,
            guidance_other=guidance_other,
        )

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)
        x_sigma_max = (
            torch.randn(n_sample, *state_shape, **self.tensor_kwargs, generator=generator) * self.sde.sigma_max
        )
        
        batch_size = x_sigma_max.shape[0]
        if batch_size > 1:  # use the same sigma for all samples in the batch
            x_sigma_max[1:] = x_sigma_max[0]
            x_sigma_max = x_sigma_max.contiguous()

        if self.net.is_context_parallel_enabled:
            x_sigma_max = rearrange(x_sigma_max, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)
            x_sigma_max = rearrange(x_sigma_max, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

        samples = self.sampler(x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=self.sde.sigma_max, sigma_min=sigma_min)
        if self.net.is_context_parallel_enabled:
            samples = rearrange(samples, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)
            samples = rearrange(samples, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
        return samples
    
@diffusion_fsdp_class_decorator
class FSDPImageToLidarDiffusionModel(ImageToLidarModel):
    pass
