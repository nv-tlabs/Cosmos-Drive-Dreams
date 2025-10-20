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

"""Tokenizer callbacks extended from base callbacks."""

import io
import time

import imageio
import numpy as np
import torch.distributed as dist
import torch
import torchvision
import wandb
from PIL import Image
from torch._dynamo.eval_frame import OptimizedModule as torch_OptimizedModule

from cosmos_predict1.utils.config import Config
from cosmos_predict1.utils.model import Model
from cosmos_predict1.utils.trainer import Trainer
from cosmos_predict1.utils import callback, distributed, log, wandb_utils
from cosmos_predict1.tokenizer.training.callbacks import make_video_grid
from cosmos_predict1.tokenizer.training.datasets.lidar_datasets.configs import COMMONDATA_CONFIG
from cosmos_predict1.tokenizer.training.model import PREDICTION

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)

from cosmos_predict1.utils.lidar_rangemap import (
    undo_row_col_temporal_repeat,
    unnormalize_and_reduce_channels,
)
from cosmos_predict1.utils.visualize.point_cloud import render_range_map_to_point_cloud


def wandb_range_image(range_map_input, range_map_recon):
    # range_map_input: [N,  H, W]
    # range_map_recon: [N,  H, W]

    device = range_map_input.device
    vis = render_range_map_to_point_cloud(
        range_map_input.cpu().numpy(), range_map_recon.cpu().numpy(), filter_outlier=False
    )  # shape: (bs, H, W, 3)

    vis_images = torch.from_numpy(vis).permute(0, 3, 1, 2).to(device)  # shape: (bs, 3, H, W)

    image_grid = torchvision.utils.make_grid(vis_images.clamp_(0, 1), nrow=1, pad_value=0, normalize=False)
    image_grid = image_grid.permute(1, 2, 0).cpu().detach().numpy()
    pil_image = Image.fromarray((image_grid * _UINT8_MAX_F).astype("uint8"))
    return wandb.Image(pil_image)


def wandb_range_video(range_map_input, range_map_recon, fps=10):
    # range_map_input: [N, T, H, W]
    # range_map_recon: [N, T, H, W]

    device = range_map_input.device
    vis = render_range_map_to_point_cloud(
        range_map_input.squeeze(0).cpu().numpy(), range_map_recon.squeeze(0).cpu().numpy(), filter_outlier=False
    )  # shape: (t, H, W, 3)
    vis_videos = torch.from_numpy(vis).unsqueeze(0).permute(0, -1, 1, 2, 3).to(device)  # shape: (1,3, t, H, W)

    video_grid = make_video_grid(vis_videos.clamp_(0, 1), nrow=vis_videos.shape[0], padding=0)
    mem_file = io.BytesIO()
    imageio.mimsave(mem_file, video_grid, fps=fps, format="mp4")
    mem_file.seek(0)
    return wandb.Video(mem_file, fps=fps, format="mp4")


def wandb_image(images, iteration):
    vis_images = (images + 1) / 2  # [-1,1] to [0,1]
    image_grid = torchvision.utils.make_grid(vis_images.clamp_(0, 1), nrow=1, pad_value=0, normalize=False)
    image_grid = image_grid.permute(1, 2, 0).cpu().detach().numpy()
    pil_image = Image.fromarray((image_grid * _UINT8_MAX_F).astype("uint8"))
    return wandb.Image(pil_image)


def wandb_video(videos, iteration, fps=10):
    # videos: [B, C, T, H, W]
    vis_videos = (videos + 1.0) / 2.0
    video_grid = make_video_grid(vis_videos.clamp_(0, 1), nrow=vis_videos.shape[0], padding=0)
    mem_file = io.BytesIO()
    imageio.mimsave(mem_file, video_grid, fps=fps, format="mp4")
    mem_file.seek(0)
    return wandb.Video(mem_file, fps=fps, format="mp4")


class WandBLidarCallback(callback.Callback):
    """Extends the base WandBCallback for Tokenizers."""

    def __init__(self, config: Config, trainer: Trainer):
        super().__init__(config, trainer)

    def on_train_start(self, model: Model, iteration: int = 0) -> None:
        wandb_utils.init_wandb(self.config, model=model)
        if distributed.is_rank0():
            self.start_iteration_time = time.time()

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:  # Log the curent learning rate.
        if iteration % self.config.trainer.logging_iter == 0 and distributed.is_rank0():
            if model_ddp.module.network.training:
                wandb.log({"optim/lr": scheduler.get_last_lr()[0]}, step=iteration)
                wandb.log({"optim/grad_scale": grad_scaler.get_scale()}, step=iteration)
            if model_ddp.module.disc is not None and model_ddp.module.disc.training:
                wandb.log({"optim_disc/lr": scheduler.get_last_lr()[0]}, step=iteration)
                wandb.log({"optim_disc/grad_scale": grad_scaler.get_scale()}, step=iteration)

    def on_training_step_end(
        self,
        model: Model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:  # Log the timing results (over a number of iterations) and the training loss.
        if (iteration % self.config.trainer.logging_iter == 0) and distributed.is_rank0():
            timer_results = self.trainer.training_timer.compute_average_results()
            network = model.network if not isinstance(model.network, torch_OptimizedModule) else model.network._orig_mod
            avg_time = (time.time() - self.start_iteration_time) / self.config.trainer.logging_iter
            wandb.log({"timer/iter": avg_time}, step=iteration)
            wandb.log({f"timer/{key}": value for key, value in timer_results.items()}, step=iteration)
            wandb.log({"train/loss": loss}, step=iteration)
            wandb.log({"iteration": iteration}, step=iteration)
            for loss_key in output_batch.get("loss", dict()):
                loss_val = output_batch["loss"][loss_key]
                wandb.log({f"train/{loss_key}": loss_val}, step=iteration)

            self.trainer.training_timer.reset()

    def on_validation_start(
        self, model: Model, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        # Cache for collecting data/output batches.
        self._val_cache: dict[str, Any] = dict(
            data_batches=[],
            output_batches=[],
            loss=torch.tensor(0.0, device="cuda"),
            sample_size=torch.tensor(0, device="cuda"),
        )
        
    def on_validation_step_end(
        self,
        model: Model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:  # Collect the validation batch and aggregate the overall loss.
        # Collect the validation batch and aggregate the overall loss.
        _input_key = model.get_input_key(data_batch)
        batch_size = data_batch[_input_key].shape[0]
        self._val_cache["loss"] += loss * batch_size
        self._val_cache["sample_size"] += batch_size

        input_images = data_batch[_input_key].float()  # shape: [N, 3, H, w]
        output_images = output_batch[PREDICTION].float()

        result_image = torch.cat((input_images, output_images), dim=-1)
        
        if result_image.ndim == 5 and result_image.shape[2] > 1:
            wandb_media = wandb_video(result_image, iteration)
        elif result_image.ndim == 5 and result_image.shape[2] == 1:
            wandb_media = wandb_image(result_image[:, :, 0, ...], iteration)
        else:
            wandb_media = wandb_image(result_image, iteration)

        # get the range_map
        dataset_name = self.config.dataloader_val.dataset.dataset_name
        dataset_config = COMMONDATA_CONFIG[dataset_name]
        is_lidar = data_batch["data_type"][0] == "lidar"
        assert dataset_config["min_value"] == -1
        if is_lidar:
            # clamp the input and output images to the range
            input_images = torch.clamp(input_images, -1, 1)
            output_images = torch.clamp(output_images, -1, 1)

            # reduce row and col repeat
            input_images = undo_row_col_temporal_repeat(
                input_images,
                dataset_config["repeat_row"],
                dataset_config["repeat_col"],
                dataset_config["repeat_temporal"],
            )
            output_images = undo_row_col_temporal_repeat(
                output_images,
                dataset_config["repeat_row"],
                dataset_config["repeat_col"],
                dataset_config["repeat_temporal"],
            )

            # unnormalize the range map and also reduce the 3 channels to 1
            input_images, valid_mask_input = unnormalize_and_reduce_channels(
                input_images,
                dataset_config["max_range"],
                dataset_config["min_range"],
                dataset_config["min_value"],
                dataset_config["inverse_depth"],
                dataset_config["to_three_channels"],
                0.5,
                0.5,
                dataset_config["inv_depth_threshold"],
            )  # shape: (N, H, W) or (N, T, H, W)
            output_images, valid_mask_recon = unnormalize_and_reduce_channels(
                output_images,
                dataset_config["max_range"],
                dataset_config["min_range"],
                dataset_config["min_value"],
                dataset_config["inverse_depth"],
                dataset_config["to_three_channels"],
                0.5,
                0.5,
                dataset_config["inv_depth_threshold"],
            )  # shape: (N, H, W) or (N, T, H, W)

            valid_mask = valid_mask_input & valid_mask_recon

            if input_images.ndim == 3:  # image tokenizer
                wandb_range_map_media = wandb_range_image(input_images, output_images)
            elif input_images.ndim == 4:  # video tokenizer
                wandb_range_map_media = wandb_range_video(input_images, output_images)

            range_map_input = input_images[valid_mask]
            range_map_recon = output_images[valid_mask]

            mae = torch.abs(range_map_input - range_map_recon).mean().item()
            rmse = torch.sqrt(torch.mean((range_map_input - range_map_recon) ** 2)).item()
            relative_error = (torch.abs(range_map_input - range_map_recon) / (range_map_input + 1e-6)).mean().item()

        else:
            mae = None
            wandb_range_map_media = None

        if wandb.run is not None:
            if mae is not None:
                wandb.log({"val/depth_mae": mae}, step=iteration)
                wandb.log({"val/depth_rmse": rmse}, step=iteration)
                wandb.log({"val/depth_relative_error": relative_error}, step=iteration)

            wandb.log({"val/loss": loss}, step=iteration)
            for loss_key in output_batch.get("loss", dict()):
                loss_val = output_batch["loss"][loss_key]
                wandb.log({f"val/{loss_key}": loss_val}, step=iteration)
            for metric_key in output_batch.get("metric", dict()):
                metric_val = output_batch["metric"][metric_key]
                wandb.log({f"val/{metric_key.upper()}": metric_val}, step=iteration)

            if is_lidar:
                wandb.log({"val/lidar_reconstruction": [wandb_media]}, step=iteration)
            else:
                wandb.log({"val/rgb_reconstruction": [wandb_media]}, step=iteration)

            if wandb_range_map_media is not None:
                wandb.log({"val/range_map": [wandb_range_map_media]}, step=iteration)

    def on_before_dataloading(self, iteration: int = 0) -> None:
        if iteration % self.config.trainer.logging_iter == 0 and distributed.is_rank0():
            self.start_iteration_time = time.time()
            
    def on_validation_end(self, model: Model, iteration: int = 0) -> None:
        # Compute the average validation loss across all devices.
        dist.all_reduce(self._val_cache["loss"], op=dist.ReduceOp.SUM)
        dist.all_reduce(self._val_cache["sample_size"], op=dist.ReduceOp.SUM)
        loss = self._val_cache["loss"].item() / self._val_cache["sample_size"]
        # Log data/stats of validation set to W&B.
        if distributed.is_rank0():
            log.info(f"Validation loss (iteration {iteration}): {loss:4f}")
            wandb.log({"val/loss": loss}, step=iteration)

    def on_train_end(self, model: Model, iteration: int = 0) -> None:
        wandb.finish()
