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

from typing import Optional, Union

import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict1.diffusion.conditioner import VideoExtendCondition, ViewConditionedVideoExtendCondition
from cosmos_predict1.diffusion.model.model_t2w import broadcast_condition
from cosmos_predict1.diffusion.model.model_v2w_multiview import DiffusionMultiviewV2WModel
from cosmos_predict1.diffusion.module.parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict1.utils import log


class DiffusionMultiviewViewExtendModel(DiffusionMultiviewV2WModel):
    def __init__(self, config):
        super().__init__(config)
        self.condition_location = None

    def generate_samples_from_batch(
        self,
        data_batch: dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        condition_latent: Optional[torch.Tensor] = None,
        num_condition_t: Optional[int] = None,
        condition_augment_sigma: float = None,
        add_input_frames_guidance: bool = False,
        condition_location: Optional[str] = None,
    ) -> Tensor:
        """Generates video samples conditioned on input frames.

        Args:
            data_batch: Input data dictionary
            guidance: Classifier-free guidance scale
            seed: Random seed for reproducibility
            state_shape: Shape of output tensor (defaults to model's state shape)
            n_sample: Number of samples to generate (defaults to batch size)
            is_negative_prompt: Whether to use negative prompting
            num_steps: Number of denoising steps
            condition_latent: Conditioning frames tensor (B,C,T,H,W)
            num_condition_t: Number of frames to condition on
            condition_augment_sigma: Noise level for condition augmentation
            add_input_frames_guidance: Whether to apply guidance to input frames

        Returns:
            Generated video samples tensor
        """
        assert condition_latent is not None, "condition_latent should be provided"

        if condition_location is None:
            condition_location = self.condition_location

        condition, uncondition = self._get_conditions(
            data_batch,
            is_negative_prompt,
            condition_latent,
            num_condition_t,
            add_input_frames_guidance,
            condition_location,
        )

        self.scheduler.set_timesteps(num_steps)
        if n_sample is None:
            n_sample = condition_latent.shape[0]
        xt = torch.randn(size=(n_sample,) + tuple(state_shape), **self.tensor_kwargs) * self.scheduler.init_noise_sigma

        to_cp = self.net.is_context_parallel_enabled
        if to_cp:
            xt = rearrange(xt, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            xt = split_inputs_cp(x=xt, seq_dim=2, cp_group=self.net.cp_group)
            xt = rearrange(xt, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
        x0 = xt.clone()

        for t in self.scheduler.timesteps:
            self.scheduler._init_step_index(t)
            sigma = self.scheduler.sigmas[self.scheduler.step_index].to(**self.tensor_kwargs)
            # Form new noise from latent
            new_xt, latent, indicator = self._augment_noise_with_latent(
                xt, sigma, condition, condition_augment_sigma=condition_augment_sigma, seed=seed
            )

            new_xt = new_xt.to(**self.tensor_kwargs)
            new_xt_scaled = self.scheduler.scale_model_input(new_xt, timestep=t)
            # Predict the noise residual
            t = t.to(**self.tensor_kwargs)
            net_output_cond = self.net(x=new_xt_scaled, timesteps=t, **condition.to_dict())
            net_output_uncond = self.net(x=new_xt_scaled, timesteps=t, **uncondition.to_dict())
            net_output = net_output_cond + guidance * (net_output_cond - net_output_uncond)
            # Replace indicated output with latent
            latent_unscaled = self._reverse_precondition_output(latent, xt=new_xt, sigma=sigma)
            new_output = indicator * latent_unscaled + (1 - indicator) * net_output
            # Step the sampler
            scheduler_output = self.scheduler.step(new_output, t, new_xt)
            xt = scheduler_output.prev_sample
            x0 = scheduler_output.pred_original_sample
        samples = x0

        if to_cp:
            samples = rearrange(samples, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)
            samples = rearrange(samples, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

        return samples

    def _get_conditions(
        self,
        data_batch: dict,
        is_negative_prompt: bool = False,
        condition_latent: Optional[torch.Tensor] = None,
        num_condition_t: Optional[int] = None,
        add_input_frames_guidance: bool = False,
        condition_location: str = "first_cam",
    ):
        """Get the conditions for the model.

        Args:
            data_batch: Input data dictionary
            is_negative_prompt: Whether to use negative prompting
            condition_latent: Conditioning frames tensor (B,C,T,H,W)
            num_condition_t: Number of frames to condition on
            add_input_frames_guidance: Whether to apply guidance to input frames

        Returns:
            condition: Input conditions
            uncondition: Conditions removed/reduced to minimum (unconditioned)
        """
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        if "view_indices" in data_batch:
            comp_factor = self.vae.temporal_compression_factor
            view_indices = rearrange(data_batch["view_indices"], "B (V T) -> B V T", V=self.n_views)
            view_indices_B_V_0 = view_indices[:, :, :1]
            view_indices_B_V_1T = view_indices[:, :, 1:-1:comp_factor]
            view_indices_B_V_T = torch.cat([view_indices_B_V_0, view_indices_B_V_1T], dim=-1)
            condition.view_indices_B_T = rearrange(view_indices_B_V_T, "B V T -> B (V T)", V=self.n_views)
            uncondition.view_indices_B_T = condition.view_indices_B_T

        condition.video_cond_bool = True
        condition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, condition, num_condition_t, condition_location
        )
        uncondition.video_cond_bool = False if add_input_frames_guidance else True
        uncondition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, uncondition, num_condition_t, condition_location
        )
        assert condition.gt_latent.allclose(uncondition.gt_latent)

        # For inference, check if parallel_state is initialized
        to_cp = self.net.is_context_parallel_enabled
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=False, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=False, to_cp=to_cp)

        return condition, uncondition

    def add_condition_video_indicator_and_video_input_mask(
        self,
        latent_state: torch.Tensor,
        condition: ViewConditionedVideoExtendCondition,
        num_condition_t: Union[int, None] = None,
        condition_location: str = "first_cam",
    ) -> ViewConditionedVideoExtendCondition:
        """Add condition_video_indicator and condition_video_input_mask to the condition object for video conditioning.
        condition_video_indicator is a binary tensor indicating the condition region in the latent state. 1x1xTx1x1 tensor.
        condition_video_input_mask will be concat with the input for the network.
        Args:
            latent_state (torch.Tensor): latent state tensor in shape B,C,T,H,W
            condition (VideoExtendCondition): condition object
            num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_n"
        Returns:
            VideoExtendCondition: updated condition object
        """
        T = latent_state.shape[2]
        latent_dtype = latent_state.dtype
        condition_video_indicator = torch.zeros(1, 1, T, 1, 1, device=latent_state.device).type(
            latent_dtype
        )  # 1 for condition region

        # Only in inference to decide the condition region
        assert num_condition_t is not None, "num_condition_t should be provided"
        assert num_condition_t <= T, f"num_condition_t should be less than T, get {num_condition_t}, {T}"
        log.info(
            f"condition_location {condition_location}, num_condition_t {num_condition_t}, condition.video_cond_bool {condition.video_cond_bool}"
        )
        # condition on first cam
        condition_video_indicator = rearrange(condition_video_indicator, "B C (V T) H W -> B V C T H W", V=self.n_views)

        if condition_location == "first_cam":
            # condition on first cam
            condition_video_indicator[:, 0, :, :, :, :] += 1.0

        elif condition_location.startswith("fixed_cam_and_first_n"):
            # condition on a list of cameras specified through the string
            cond_vids = [int(c) for c in condition_location.split("_")[5:]]

            for vidx in cond_vids:
                condition_video_indicator[:, vidx, :, :, :, :] += 1.0
            # also condition on first n_condition_t frames
            condition_video_indicator[:, :, :, :num_condition_t] += 1.0
            condition_video_indicator = condition_video_indicator.clamp(max=1.0)

        elif condition_location.startswith("fixed_cam"):
            # condition on a list of cameras specified through the string
            cond_vids = [int(c) for c in condition_location.split("_")[2:]]

            for vidx in cond_vids:
                condition_video_indicator[:, vidx, :, :, :, :] += 1.0
            condition_video_indicator = torch.clamp(condition_video_indicator, 0, 1)

        elif condition_location == "first_cam_and_first_n":
            # condition on first cam
            condition_video_indicator[:, 0, :, :, :, :] += 1.0
            condition_video_indicator[:, :, :, :num_condition_t] += 1.0
            condition_video_indicator = condition_video_indicator.clamp(max=1.0)
        else:
            raise NotImplementedError(f"condition_location {condition_location} not implemented")

        condition_video_indicator = rearrange(
            condition_video_indicator, "B V C T H W  -> B C (V T) H W", V=self.n_views
        )

        condition.gt_latent = latent_state
        condition.condition_video_indicator = condition_video_indicator

        B, C, T, H, W = latent_state.shape
        # Create additional input_mask channel, this will be concatenated to the input of the network
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

        return condition
