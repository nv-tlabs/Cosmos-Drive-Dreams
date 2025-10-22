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

from copy import deepcopy

#####################################
# configs for the diffusion model
COMMONDATA_CONFIG = dict()

COMMONDATA_CONFIG["images_to_lidar"] = {
    "data_name": "mads_range_map",
    "root_path": "datasets/lidar_dataset_release",  # this is used for RGB
    "filter_list_path": "assets/lidar/lidar_split.lst",  #  113180 samples
    "t5_embedding_path": "datasets/lidar_dataset_release",
    "custom_folders": ["datasets/lidar_dataset_release"],
    "custom_fields": ["lidar"],
    "metadata_folder": "datasets/lidar_dataset_release/metadata",
    "max_range": 100,
    "min_range": 5,
    "downsample_factor_row": 1,
    "downsample_factor_col": 2,
    "repeat_row": 4,
    "repeat_col": 1,
    "repeat_temporal": 1,
    "lidar_crop_size": [-1, 1792],  # no crop
    "lidar_size": [512, 1792],  # make sure the feature map is divisible by 8
    "rgb_crop_size": [704, 1280],
    "rgb_resize_size": [720, 1280],
    "downsample_method": "scatter_min",
    "min_value": -1,
    "num_views": 3,
    "view_indices": [0, 3, 4],
    "sample_n_frames": 1,
    "lidar_length": 200,
    "frame_sampling_method": "random",  # sequential, random
    "frame_sampling_every_n_frames": 1,
    "sample_lidar_prob": 1.0,  # always sample lidar
    "is_video_tokenizer": False,
    "RGB_FPS": 30,
    "LIDAR_FPS": 10,
    "load_lidar_and_rgb": True,  # specified as projects/edify_video/v4/config/gen3c/config_gen3c.py
    "load_rgb_only": False,  # specified as projects/edify_video/v4/config/gen3c/config_gen3c.py
    "load_lidar_only": False,
    "load_lidar_or_rgb": False,
    "load_t5_embeddings": False,
    "inverse_depth": False,
    "to_three_channels": "repeat",  # "repeat" or "concat"
    "inv_depth_threshold": 20,  # if the final depth is below this threshold, then use the inverse depth
}