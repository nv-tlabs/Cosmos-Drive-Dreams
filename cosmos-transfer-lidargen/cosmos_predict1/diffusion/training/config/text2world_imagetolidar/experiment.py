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
from megatron.core import parallel_state
from torch.utils.data import DistributedSampler, DataLoader

from cosmos_predict1.diffusion.training.models.imagetolidar_model import FSDPImageToLidarDiffusionModel
from cosmos_predict1.diffusion.training.networks.general_dit_view_extend_multiview import MultiviewExtensionGeneralDIT
from cosmos_predict1.utils.lazy_config import PLACEHOLDER
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict

from cosmos_predict1.diffusion.training.datasets.lidar_dataset.imagetolidar_dataloader import get_lidar_dataset, dict_collation_fn


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )
    
dset_name = "images_to_lidar"
img_to_lidar_dataset = get_lidar_dataset(
    dset_name,
)

dataloader = L(DataLoader)(
        dataset=img_to_lidar_dataset,
        sampler=L(get_sampler)(dataset=img_to_lidar_dataset),
        batch_size=1,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=dict_collation_fn
    )

cs = ConfigStore.instance()
n_cameras = 4
num_frames = 1


text2world_imagetolidar = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_lidar_diffusion_tokenizer"},
            {"override /conditioner": "image_to_lidar_cond"},
            {"override /callbacks": ["image_to_lidar"]},  
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_text2world",
            name="text2world_imagetolidar",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=600,
            broadcast_via_filesystem=False,
            load_path="",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
        ),
        trainer=dict(
            max_iter=2000,
            distributed_parallelism="fsdp",
            logging_iter=10,
            grad_accum_iter=1,
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        ),
        model=dict(
            n_views=n_cameras,
            # Use 16x16x32x40 latent shape for training
            latent_shape=[
                16,  # Latent channel dim
                n_cameras,  # Latent temporal dim
                88,  # 88,  # Latent height dim
                224,  # 160,  # Latent width dim
            ],
            rgb_latent_shape = [88, 160],
            lidar_latent_shape = [64, 224],
            rgb_shape = [704, 1280],
            lidar_shape = [512, 1792],
            loss_reduce="mean",
            ema=dict(
                enabled=True,
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=32,
                sharding_strategy="hybrid",
            ),
            net=L(MultiviewExtensionGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=1,
                n_views=n_cameras,
                n_views_emb=n_cameras,
                view_condition_dim=n_cameras + 1,
                add_repeat_frame_embedding=False,
                extra_per_block_abs_pos_emb=True,
                extra_per_block_abs_pos_emb_type="sincos",
            ),
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_cam",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=n_cameras - 1,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
        ),
        model_obj=L(FSDPImageToLidarDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps~(when resume from 310000)
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader,
        dataloader_val=dataloader,
    )
)


def register_experiments(cs):
    cs.store(
        group="experiment",
        package="_global_",
        name="text2world_imagetolidar",
        node=text2world_imagetolidar,
    )