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
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.tokenizer.training.datasets.lidar_datasets.configs import COMMONDATA_CONFIG
from cosmos_predict1.diffusion.training.datasets.lidar_dataset.imagetolidar_dataloader import LidarRangeMapSampler, MyDataLoader
import numpy as np

def get_lidar_range_map_dataloader(
    dataset_name: str,
    shuffle=True,
    num_workers=4,
    prefetch_factor=4,
    **kwargs,
) -> omegaconf.dictconfig.DictConfig:
    return L(MyDataLoader)(
        dataset=L(LidarRangeMapDataset)(
            dataset_name=dataset_name,
            **kwargs,
        ),
        batch_size=1,
        num_workers=num_workers,
        shuffle=shuffle,
        prefetch_factor=prefetch_factor,
    )


class LidarRangeMapDataset:
    def __init__(self, dataset_name, **kwargs):
        self.dataset_name = dataset_name
        self.dataset_config = OmegaConf.create(COMMONDATA_CONFIG[dataset_name])
        self.extra_config = OmegaConf.create(kwargs)

    def build_dataset(self):
        config_dict = OmegaConf.to_container(self.dataset_config, resolve=True)
        return LidarRangeMapSamplerTokenizer(**config_dict)


class LidarRangeMapSamplerTokenizer(LidarRangeMapSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_lidar(self, example, lidar_key = "lidar"):
        range_maps = self.process_lidar(example["custom"][lidar_key]) if lidar_key in example["custom"] else None
        return range_maps

    def __getitem__(self, idx):
        if self.load_rgb_only:
            raise NotImplementedError
        if self.load_lidar_or_rgb:
            raise NotImplementedError
        if self.load_lidar_and_rgb:
            raise NotImplementedError

        # sample the video index
        lidar_data_idx = (idx % self.n_data_per_node) + self.data_start_idx  # frame idx
        lidar_frame_indices = self.sample_frame_indices(load_lidar=True)

        # load the data
        try:
            example = self.dataset._read_rgb_and_lidar(lidar_data_idx, lidar_frame_indices, ["custom"])
        except Exception as e:
            print(f"Error loading data for index {idx}: {e}")
            return self.__getitem__(np.random.randint(0, self.__len__()))

        lidar_rangemap = self.get_lidar(example)

        if self.is_video_tokenizer:
            key = "video"
        else:
            key = "images"

        sample = {}
        sample[key] = lidar_rangemap
        sample["__key__"] = example["__key__"]
        clip_name = "%s-%s-%d" % (self.data_name, example["__key__"], idx)
        sample["clip_name"] = clip_name
        sample["data_type"] = "lidar"

        del example

        return sample   


if __name__ == "__main__":
    pass
