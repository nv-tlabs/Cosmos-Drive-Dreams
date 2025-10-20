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

from dataclasses import dataclass, fields
from statistics import NormalDist
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor
from torch.distributed import broadcast_object_list, get_process_group_ranks
from torch.distributed.utils import _verify_param_shape_across_processes

from cosmos_predict1.diffusion.conditioner import BaseVideoCondition, DataType, VideoExtendCondition
from cosmos_predict1.diffusion.config.base.conditioner import VideoCondBoolConfig
from cosmos_predict1.diffusion.functional.batch_ops import batch_mul
from cosmos_predict1.diffusion.model.model_v2w import DiffusionV2WModel
from cosmos_predict1.diffusion.module.parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict1.diffusion.modules.denoiser_scaling import EDMScaling
from cosmos_predict1.diffusion.modules.res_sampler import Sampler
from cosmos_predict1.diffusion.training.modules.edm_sde import EDMSDE
from cosmos_predict1.diffusion.types import DenoisePrediction
from cosmos_predict1.utils import distributed, log, misc

IS_PREPROCESSED_KEY = "is_preprocessed"


@dataclass
class CosmosCondition:
    crossattn_emb: torch.Tensor
    crossattn_mask: torch.Tensor
    padding_mask: Optional[torch.Tensor] = None
    scalar_feature: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Optional[torch.Tensor]]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class VideoDenoisePrediction:
    x0: torch.Tensor  # clean data prediction
    eps: Optional[torch.Tensor] = None  # noise prediction
    logvar: Optional[torch.Tensor] = None  # log variance of noise prediction, can be used a confidence / uncertainty
    net_in: Optional[torch.Tensor] = None  # input to the network
    net_x0_pred: Optional[torch.Tensor] = None  # prediction of x0 from the network
    xt: Optional[torch.Tensor] = None  # input to the network, before muliply with c_in
    x0_pred_replaced: Optional[torch.Tensor] = None  # x0 prediction with condition region replaced by gt_latent


def normalize_condition_latent(condition_latent):
    """Normalize the condition latent tensor to have zero mean and unit variance
    Args:
        condition_latent (torch.Tensor): latent tensor in shape B,C,T,H,W
    """
    condition_latent_2D = rearrange(condition_latent, "b c t h w -> b c t (h w)")
    mean = condition_latent_2D.mean(dim=-1)
    std = condition_latent_2D.std(dim=-1)
    # bct -> bct11
    mean = mean.unsqueeze(-1).unsqueeze(-1)
    std = std.unsqueeze(-1).unsqueeze(-1)
    condition_latent = (condition_latent - mean) / std
    return condition_latent


def robust_broadcast(tensor: torch.Tensor, src: int, pg, is_check_shape: bool = False) -> torch.Tensor:
    """
    Perform a robust broadcast operation that works regardless of tensor shapes on different ranks.

    Args:
        tensor (torch.Tensor): The tensor to broadcast (on src rank) or receive (on other ranks).
        src (int): The source rank for the broadcast. Defaults to 0.

    Returns:
        torch.Tensor: The broadcasted tensor on all ranks.
    """
    # First, broadcast the shape of the tensor
    if distributed.get_rank() == src:
        shape = torch.tensor(tensor.shape).cuda()
    else:
        shape = torch.empty(tensor.dim(), dtype=torch.long).cuda()
    if is_check_shape:
        _verify_param_shape_across_processes(pg, [shape])
    torch.distributed.broadcast(shape, src, group=pg)

    # Resize the tensor on non-src ranks if necessary
    if distributed.get_rank() != src:
        tensor = tensor.new_empty(shape.tolist()).type_as(tensor)

    # Now broadcast the tensor data
    torch.distributed.broadcast(tensor, src, group=pg)

    return tensor


def _broadcast(item: torch.Tensor | str | None, to_tp: bool = True, to_cp: bool = True) -> torch.Tensor | str | None:
    """
    Broadcast the item from the minimum rank in the specified group(s).
    Since global rank = tp_rank + cp_rank * tp_size + ...
    First broadcast in the tp_group and then in the cp_group will
    ensure that the item is broadcasted across ranks in cp_group and tp_group.

    Parameters:
    - item: The item to broadcast (can be a torch.Tensor, str, or None).
    - to_tp: Whether to broadcast to the tensor model parallel group.
    - to_cp: Whether to broadcast to the context parallel group.
    """
    if not parallel_state.is_initialized():
        return item
    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_group = parallel_state.get_context_parallel_group()

    to_tp = to_tp and parallel_state.get_tensor_model_parallel_world_size() > 1
    to_cp = to_cp and parallel_state.get_context_parallel_world_size() > 1

    if to_tp:
        min_tp_rank = min(get_process_group_ranks(tp_group))

    if to_cp:
        min_cp_rank = min(get_process_group_ranks(cp_group))

    if isinstance(item, torch.Tensor):  # assume the device is cuda
        # log.info(f"{item.shape}", rank0_only=False)
        if to_tp:
            # torch.distributed.broadcast(item, min_tp_rank, group=tp_group)
            item = robust_broadcast(item, min_tp_rank, tp_group)
        if to_cp:
            # torch.distributed.broadcast(item, min_cp_rank, group=cp_group)
            item = robust_broadcast(item, min_cp_rank, cp_group)
    elif item is not None:
        broadcastable_list = [item]
        if to_tp:
            # log.info(f"{broadcastable_list}", rank0_only=False)
            broadcast_object_list(broadcastable_list, min_tp_rank, group=tp_group)
        if to_cp:
            broadcast_object_list(broadcastable_list, min_cp_rank, group=cp_group)

        item = broadcastable_list[0]
    return item


def broadcast_condition(condition: BaseVideoCondition, to_tp: bool = True, to_cp: bool = True) -> BaseVideoCondition:
    condition_kwargs = {}
    for k, v in condition.to_dict().items():
        if isinstance(v, torch.Tensor):
            assert not v.requires_grad, f"{k} requires gradient. the current impl does not support it"
        condition_kwargs[k] = _broadcast(v, to_tp=to_tp, to_cp=to_cp)
    condition = type(condition)(**condition_kwargs)
    return condition


class DiffusionActionV2WModel(DiffusionV2WModel):
    #######################################
    # Not implemented yet
    def __init__(self, config):
        super().__init__(config)

        self.sde = EDMSDE(
            p_mean=0.0,
            p_std=1.0,
            sigma_max=80,
            sigma_min=0.0002,
        )
        self.sampler = Sampler()
        self.scaling = EDMScaling(self.sigma_data)

        self.is_extend_model = True

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Warning:
            A warning is issued if the data has not been previously normalized.

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.

        Side Effects:
            Modifies the 'input_data_key' tensor within the 'data_batch' dictionary in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        input_key = self.input_data_key if input_key is None else input_key
        # only handle video batch
        if input_key in data_batch:
            # Check if the data has already been normalized and avoid re-normalizing
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert torch.is_floating_point(data_batch[input_key]), "Video data is not in float format."
                assert torch.all(
                    (data_batch[input_key] >= -1.0001) & (data_batch[input_key] <= 1.0001)
                ), f"Video data is not in the range [-1, 1]. get data range [{data_batch[input_key].min()}, {data_batch[input_key].max()}]"
            else:
                assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
                data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[IS_PREPROCESSED_KEY] = True

    def get_data_and_condition(
        self, data_batch: dict[str, Tensor], num_condition_t: Union[int, None] = None
    ) -> Tuple[Tensor, VideoExtendCondition]:
        # raw_state, latent_state, condition = super().get_data_and_condition(data_batch)

        # ################## from DiffusionV2WModel ###################
        self._normalize_video_databatch_inplace(data_batch)
        input_key = self.input_data_key  # by default it is video key
        is_video_batch = True

        # Broadcast data and condition across TP and CP groups.
        # sort keys to make sure the order is same, IMPORTANT! otherwise, nccl will hang!
        local_keys = sorted(list(data_batch.keys()))
        # log.critical(f"all keys {local_keys}", rank0_only=False)
        for key in local_keys:
            data_batch[key] = _broadcast(data_batch[key], to_tp=True, to_cp=is_video_batch)

        # Latent state
        raw_state = data_batch[input_key]
        latent_state = self.encode(raw_state).contiguous()

        # Condition
        condition = self.conditioner(data_batch)
        condition.data_type = DataType.VIDEO

        # VAE has randomness. CP/TP group should have the same encoded output.

        latent_state = _broadcast(latent_state, to_tp=True, to_cp=is_video_batch)
        condition = broadcast_condition(condition, to_tp=True, to_cp=is_video_batch)

        # ################## from DiffusionV2WModel ###################

        if self.config.conditioner.video_cond_bool.sample_tokens_start_from_p_or_i:
            latent_state = self.sample_tokens_start_from_p_or_i(latent_state)
        condition = self.add_condition_video_indicator_and_video_input_mask(
            latent_state, condition, num_condition_t=num_condition_t
        )
        if self.config.conditioner.video_cond_bool.add_pose_condition:
            condition = self.add_condition_pose(data_batch, condition)
        log.debug(f"condition.data_type {condition.data_type}")
        return raw_state, latent_state, condition

    def draw_augment_sigma_and_epsilon(
        self, size: int, condition: VideoExtendCondition, p_mean: float, p_std: float, multiplier: float
    ) -> Tensor:
        is_video_batch = condition.data_type == DataType.VIDEO
        del condition
        batch_size = size[0]
        epsilon = torch.randn(size, **self.tensor_kwargs)

        gaussian_dist = NormalDist(mu=p_mean, sigma=p_std)
        cdf_vals = np.random.uniform(size=(batch_size))
        samples_interval_gaussian = [gaussian_dist.inv_cdf(cdf_val) for cdf_val in cdf_vals]

        log_sigma = torch.tensor(samples_interval_gaussian, device="cuda")
        sigma_B = torch.exp(log_sigma).to(**self.tensor_kwargs)

        sigma_B = _broadcast(sigma_B * multiplier, to_tp=True, to_cp=is_video_batch)
        epsilon = _broadcast(epsilon, to_tp=True, to_cp=is_video_batch)
        return sigma_B, epsilon

    def augment_conditional_latent_frames(
        self,
        condition: VideoExtendCondition,
        cfg_video_cond_bool: VideoCondBoolConfig,
        gt_latent: Tensor,
        condition_video_augment_sigma_in_inference: float = 0.001,
        sigma: Tensor = None,
        seed_inference: int = 1,
    ) -> Union[VideoExtendCondition, Tensor]:
        """This function is used to augment the condition input with noise
        Args:
            condition (VideoExtendCondition): condition object
                condition_video_indicator: binary tensor indicating the region is condition(value=1) or generation(value=0). Bx1xTx1x1 tensor.
                condition_video_input_mask: input mask for the network input, indicating the condition region. B,1,T,H,W tensor. will be concat with the input for the network.
            cfg_video_cond_bool (VideoCondBoolConfig): video condition bool config
            gt_latent (Tensor): ground truth latent tensor in shape B,C,T,H,W
            condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference
            sigma (Tensor): noise level for the generation region
        Returns:
            VideoExtendCondition: updated condition object
                condition_video_augment_sigma: sigma for the condition region, feed to the network
            augment_latent (Tensor): augmented latent tensor in shape B,C,T,H,W

        """

        if cfg_video_cond_bool.apply_corruption_to_condition_region == "noise_with_sigma":
            # Training only, sample sigma for the condition region
            augment_sigma, _ = self.draw_augment_sigma_and_epsilon(
                gt_latent.shape,
                condition,
                cfg_video_cond_bool.augment_sigma_sample_p_mean,
                cfg_video_cond_bool.augment_sigma_sample_p_std,
                cfg_video_cond_bool.augment_sigma_sample_multiplier,
            )
            noise = torch.randn(*gt_latent.shape, **self.tensor_kwargs)

        elif cfg_video_cond_bool.apply_corruption_to_condition_region == "noise_with_sigma_fixed":
            # Inference only, use fixed sigma for the condition region
            log.debug(
                f"condition_video_augment_sigma_in_inference={condition_video_augment_sigma_in_inference}, sigma={sigma.flatten()[0]}"
            )
            assert (
                condition_video_augment_sigma_in_inference is not None
            ), "condition_video_augment_sigma_in_inference should be provided"
            augment_sigma = condition_video_augment_sigma_in_inference

            if augment_sigma >= sigma.flatten()[0]:
                # This is a inference trick! If the sampling sigma is smaller than the augment sigma, we will start denoising the condition region together.
                # This is achieved by setting all region as `generation`, i.e. value=0
                log.debug("augment_sigma larger than sigma or other frame, remove condition")
                condition.condition_video_indicator = condition.condition_video_indicator * 0

            augment_sigma = torch.tensor([augment_sigma], **self.tensor_kwargs)

            # Inference, use fixed seed
            noise = misc.arch_invariant_rand(
                gt_latent.shape,
                torch.float32,
                self.tensor_kwargs["device"],
                seed_inference,
            )
        else:
            raise ValueError(f"does not support {cfg_video_cond_bool.apply_corruption_to_condition_region}")

        # Now apply the augment_sigma to the gt_latent

        augment_latent = gt_latent + noise * augment_sigma.view(-1, 1, 1, 1, 1)
        _, _, c_in_augment, c_noise_augment = self.scaling(sigma=augment_sigma)

        if cfg_video_cond_bool.condition_on_augment_sigma:  # model takes augment_sigma as input
            if condition.condition_video_indicator.sum() > 0:  # has condition frames
                condition.condition_video_augment_sigma = c_noise_augment
            else:  # no condition frames
                condition.condition_video_augment_sigma = torch.zeros_like(c_noise_augment)

        # Multiply the whole latent with c_in_augment
        augment_latent_cin = batch_mul(augment_latent, c_in_augment)

        # Since the whole latent will multiply with c_in later, we devide the value to cancel the effect
        _, _, c_in, _ = self.scaling(sigma=sigma)
        augment_latent_cin = batch_mul(augment_latent_cin, 1 / c_in)

        return condition, augment_latent_cin

    def _denoise(self, xt: torch.Tensor, sigma: torch.Tensor, condition: CosmosCondition) -> DenoisePrediction:
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (CosmosCondition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred) and optional confidence (logvar).
        """

        if getattr(self.config, "use_dummy_temporal_dim", False):
            # When using video DiT model for image, we need to use a dummy temporal dimension.
            xt = xt.unsqueeze(2)

        xt = xt.to(**self.tensor_kwargs)
        sigma = sigma.to(**self.tensor_kwargs)
        # get precondition for the network
        c_skip, c_out, c_in, c_noise = self.scaling(sigma=sigma)

        # forward pass through the network
        net_output = self.net(
            x=batch_mul(c_in, xt),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps=c_noise,  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        )

        logvar = self.model.logvar(c_noise)
        x0_pred = batch_mul(c_skip, xt) + batch_mul(c_out, net_output)

        # get noise prediction based on sde
        eps_pred = batch_mul(xt - x0_pred, 1.0 / sigma)

        if getattr(self.config, "use_dummy_temporal_dim", False):
            x0_pred = x0_pred.squeeze(2)
            eps_pred = eps_pred.squeeze(2)

        return DenoisePrediction(x0_pred, eps_pred, logvar)

    def denoise(
        self,
        noise_x: Tensor,
        sigma: Tensor,
        condition: VideoExtendCondition,
        condition_video_augment_sigma_in_inference: float = 0.001,
        seed_inference: int = 1,
    ) -> VideoDenoisePrediction:
        """
        Denoise the noisy input tensor.

        Args:
            noise_x (Tensor): Noisy input tensor.
            sigma (Tensor): Noise level.
            condition (VideoExtendCondition): Condition for denoising.
            condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference

        Returns:
            Tensor: Denoised output tensor.
        """

        assert (
            condition.gt_latent is not None
        ), f"find None gt_latent in condition, likely didn't call self.add_condition_video_indicator_and_video_input_mask when preparing the condition or this is a image batch but condition.data_type is wrong, get {noise_x.shape}"
        gt_latent = condition.gt_latent
        cfg_video_cond_bool: VideoCondBoolConfig = self.config.conditioner.video_cond_bool

        condition_latent = gt_latent

        if cfg_video_cond_bool.normalize_condition_latent:
            condition_latent = normalize_condition_latent(condition_latent)

        # Augment the latent with different sigma value, and add the augment_sigma to the condition object if needed
        condition, augment_latent = self.augment_conditional_latent_frames(
            condition,
            cfg_video_cond_bool,
            condition_latent,
            condition_video_augment_sigma_in_inference,
            sigma,
            seed_inference=seed_inference,
        )

        condition_video_indicator = condition.condition_video_indicator  # [B, 1, T, 1, 1]
        if parallel_state.get_context_parallel_world_size() > 1:
            cp_group = parallel_state.get_context_parallel_group()
            condition_video_indicator = split_inputs_cp(condition_video_indicator, seq_dim=2, cp_group=cp_group)
            augment_latent = split_inputs_cp(augment_latent, seq_dim=2, cp_group=cp_group)
            gt_latent = split_inputs_cp(gt_latent, seq_dim=2, cp_group=cp_group)

        if not condition.video_cond_bool:
            # Unconditional case, drop out the condition region
            augment_latent = self.drop_out_condition_region(augment_latent, noise_x, cfg_video_cond_bool)

        # Compose the model input with condition region (augment_latent) and generation region (noise_x)
        new_noise_xt = condition_video_indicator * augment_latent + (1 - condition_video_indicator) * noise_x
        # Call the abse model
        denoise_pred = self._denoise(new_noise_xt, sigma, condition)

        x0_pred_replaced = condition_video_indicator * gt_latent + (1 - condition_video_indicator) * denoise_pred.x0
        if cfg_video_cond_bool.compute_loss_for_condition_region:
            # We also denoise the conditional region
            x0_pred = denoise_pred.x0
        else:
            x0_pred = x0_pred_replaced

        return VideoDenoisePrediction(
            x0=x0_pred,
            eps=batch_mul(noise_x - x0_pred, 1.0 / sigma),
            logvar=denoise_pred.logvar,
            net_in=batch_mul(1.0 / torch.sqrt(self.sigma_data**2 + sigma**2), new_noise_xt),
            net_x0_pred=denoise_pred.x0,
            xt=new_noise_xt,
            x0_pred_replaced=x0_pred_replaced,
        )

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        condition_latent: Union[torch.Tensor, None] = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        add_input_frames_guidance: bool = False,
        return_noise: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Different from the base model, this function support condition latent as input, it will create a differnt x0_fn if condition latent is given.
        If this feature is stablized, we could consider to move this function to the base model.

        Args:
            condition_latent (Optional[torch.Tensor]): latent tensor in shape B,C,T,H,W as condition to generate video.
            num_condition_t (Optional[int]): number of condition latent T, if None, will use the whole first half

            add_input_frames_guidance (bool): add guidance to the input frames, used for cfg on input frames
            return_noise (bool): return the initial noise or not, used for ODE pairs generation
        """
        self._normalize_video_databatch_inplace(data_batch)
        if n_sample is None:
            input_key = self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
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
            seed_inference=seed,  # Use for noise of augment sigma
        )

        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape), torch.float32, self.tensor_kwargs["device"], seed
            )
            * self.sde.sigma_max
        )
        if self.net.is_context_parallel_enabled:
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)

        samples = self.sampler(x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=self.sde.sigma_max)
        if self.net.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)

        if return_noise:
            if self.net.is_context_parallel_enabled:
                x_sigma_max = cat_outputs_cp(x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)
            return samples, x_sigma_max / self.sde.sigma_max

        return samples

    def get_x0_fn_from_batch_with_condition_latent(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        condition_latent: torch.Tensor = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        add_input_frames_guidance: bool = False,
        seed_inference: int = 1,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.
        Different from the base model, this function support condition latent as input, it will add the condition information into the condition and uncondition object.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true
        - condition_latent (torch.Tensor): latent tensor in shape B,C,T,H,W as condition to generate video.
        - num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_n"
        - condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference
        - add_input_frames_guidance (bool): add guidance to the input frames, used for cfg on input frames

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        condition.video_cond_bool = True
        condition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, condition, num_condition_t
        )
        if self.config.conditioner.video_cond_bool.add_pose_condition:
            condition = self.add_condition_pose(data_batch, condition)

        uncondition.video_cond_bool = False if add_input_frames_guidance else True
        uncondition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, uncondition, num_condition_t
        )
        if self.config.conditioner.video_cond_bool.add_pose_condition:
            uncondition = self.add_condition_pose(data_batch, uncondition)

        to_cp = self.net.is_context_parallel_enabled
        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=True, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=True, to_cp=to_cp)
        else:
            assert not to_cp, "parallel_state is not initialized, context parallel should be turned off."

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(
                noise_x,
                sigma,
                condition,
                condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                seed_inference=seed_inference,
            ).x0_pred_replaced
            uncond_x0 = self.denoise(
                noise_x,
                sigma,
                uncondition,
                condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                seed_inference=seed_inference,
            ).x0_pred_replaced
            return cond_x0 + guidance * (cond_x0 - uncond_x0)

        return x0_fn

    def add_condition_video_indicator_and_video_input_mask(
        self, latent_state: torch.Tensor, condition: VideoExtendCondition, num_condition_t: Union[int, None] = None
    ) -> VideoExtendCondition:
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
        if self.config.conditioner.video_cond_bool.condition_location == "first_n":
            # Only in inference to decide the condition region
            assert num_condition_t is not None, "num_condition_t should be provided"
            assert num_condition_t <= T, f"num_condition_t should be less than T, get {num_condition_t}, {T}"
            log.info(
                f"condition_location first_n, num_condition_t {num_condition_t}, condition.video_cond_bool {condition.video_cond_bool}"
            )
            condition_video_indicator[:, :, :num_condition_t] += 1.0
        elif self.config.conditioner.video_cond_bool.condition_location == "first_random_n":
            # Only in training
            num_condition_t_max = self.config.conditioner.video_cond_bool.first_random_n_num_condition_t_max
            assert (
                num_condition_t_max <= T
            ), f"num_condition_t_max should be less than T, get {num_condition_t_max}, {T}"
            assert num_condition_t_max >= self.config.conditioner.video_cond_bool.first_random_n_num_condition_t_min
            num_condition_t = torch.randint(
                self.config.conditioner.video_cond_bool.first_random_n_num_condition_t_min,
                num_condition_t_max + 1,
                (1,),
            ).item()
            condition_video_indicator[:, :, :num_condition_t] += 1.0

        elif self.config.conditioner.video_cond_bool.condition_location == "random":
            # Only in training
            condition_rate = self.config.conditioner.video_cond_bool.random_conditon_rate
            flag = torch.ones(1, 1, T, 1, 1, device=latent_state.device).type(latent_dtype) * condition_rate
            condition_video_indicator = torch.bernoulli(flag).type(latent_dtype).to(latent_state.device)
        else:
            raise NotImplementedError(
                f"condition_location {self.config.conditioner.video_cond_bool.condition_location} not implemented; training={self.training}"
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

    def add_condition_pose(self, data_batch: Dict, condition: VideoExtendCondition) -> VideoExtendCondition:
        """Add pose condition to the condition object. For camera control model
        Args:
            data_batch (Dict): data batch, with key "plucker_embeddings", in shape B,T,C,H,W
            latent_state (torch.Tensor): latent state tensor in shape B,C,T,H,W
            condition (VideoExtendCondition): condition object
            num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_n"
        Returns:
            VideoExtendCondition: updated condition object
        """
        assert (
            "plucker_embeddings" in data_batch or "plucker_embeddings_downsample" in data_batch.keys()
        ), f"plucker_embeddings should be in data_batch. only find {data_batch.keys()}"
        plucker_embeddings = (
            data_batch["plucker_embeddings"]
            if "plucker_embeddings_downsample" not in data_batch.keys()
            else data_batch["plucker_embeddings_downsample"]
        )
        condition.condition_video_pose = rearrange(plucker_embeddings, "b t c h w -> b c t h w").contiguous()
        to_cp = self.net.is_context_parallel_enabled
        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=True, to_cp=to_cp)
        else:
            assert not to_cp, "parallel_state is not initialized, context parallel should be turned off."

        return condition

    def sample_tokens_start_from_p_or_i(self, latent_state: torch.Tensor) -> torch.Tensor:
        """Sample the PPP... from the IPPP... sequence, only for video sequence
        Args:
            latent_state (torch.Tensor): latent state tensor in shape B,C,T,H,W
        Returns:
            torch.Tensor: sampled PPP tensor in shape B,C,T,H,W
        """
        B, C, T, H, W = latent_state.shape
        latent_dtype = latent_state.dtype
        T_target = self.state_shape[1]
        latent_state_sample = torch.zeros((B, C, T_target, H, W), dtype=latent_dtype, device=latent_state.device)
        t_start = torch.randint(0, T - T_target + 1, (1,))
        # broadcast to other device
        latent_state_sample = latent_state[:, :, t_start : t_start + T_target].contiguous()
        if parallel_state.is_initialized():
            latent_state_sample = _broadcast(latent_state_sample, to_tp=True, to_cp=True)

        return latent_state_sample
