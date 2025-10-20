## Post-training diffusion-based Text2World models (with multi-view data)

### Model Support Matrix

We support the following Cosmos Diffusion models for post-training. Review the available models and their compute requirements for post-tuning and inference to determine the best model for your use case.

| Model Name                               | Model Status | Compute Requirements for Post-Training |
|----------------------------------------------|------------------|------------------------------------------|
| Cosmos-Predict1-7B-Text2World-Sample-AV-Multiview           | **Supported**    | 8 NVIDIA GPUs*                           |

**\*** `H100-80GB` or `A100-80GB` GPUs are recommended.

### Environment setup

Please refer to the Post-training section of [INSTALL.md](/INSTALL.md#post-training) for instructions on environment setup.

### Download checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```
3. Accept the [Llama-Guard-3-8B terms](https://huggingface.co/meta-llama/Llama-Guard-3-8B)

4. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7):
   ```bash
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B --model_types Text2World-Sample-AV-Multiview --checkpoint_dir checkpoints
   ```

### Examples

Post-training a Cosmos Diffusion-based WFM enables you to train the model to generate videos that are more specific to your use case.

There are 3 steps to post-training: downloading a dataset, preprocessing the data, and post-training the model.

#### 1. Download a Dataset

The first step is to download a dataset with videos and captions.

Example 1. You can use [Waymo Open Dataset](https://waymo.com/open/) for post-training.

Please follow the [instruction](https://github.com/nv-tlabs/cosmos-av-sample-toolkits?tab=readme-ov-file#convert-public-datasets) in [cosmos-av-sample-toolkits](https://github.com/nv-tlabs/cosmos-av-sample-toolkits) to download and convert the Waymo Open Dataset.

```bash
mkdir -p datasets/waymo/

# Run this command in cosmos-av-sample-toolkits to process the Waymo videos
python convert_waymo_to_rds_hq.py -i <WAYMO_TFRECORDS_FOLDER> -o datasets/waymo/videos -n 32
```

#### 2. Preprocessing the Data

Run the following command to pre-compute T5-XXL embeddings for the video captions used for post-training:

```bash
# The script will use the provided prompt, save the T5-XXL embeddings in pickle format.
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/get_t5_embeddings_from_waymo.py --dataset_path datasets/waymo --prompt "A video of car driving on the road."
```

Dataset folder format:
```
datasets/waymo/
├── cache/
│   ├── prefix_t5_embeddings_pinhole_front.pickle
│   ├── prefix_t5_embeddings_pinhole_front_left.pickle
│   ├── prefix_t5_embeddings_pinhole_front_right.pickle
│   ├── prefix_t5_embeddings_pinhole_side_left.pickle
│   └── prefix_t5_embeddings_pinhole_side_right.pickle
├── metas/
│   ├── pinhole_front
│       ├── *.txt
│   ├── pinhole_front_left
│   ├── pinhole_front_right
│   ├── pinhole_side_left
│   └── pinhole_side_right
├── videos/
│   ├── pinhole_front
│       ├── *.mp4
│   ├── pinhole_front_left
│   ├── pinhole_front_right
│   ├── pinhole_side_left
│   └── pinhole_side_right
├── t5_xxl/
│   ├── pinhole_front
│       ├── *.pickle
│   ├── pinhole_front_left
│   ├── pinhole_front_right
│   ├── pinhole_side_left
│   └── pinhole_side_right
```

#### 3. Post-train the Model

Run the following command to execute an example post-training job with the above data.
```bash
export OUTPUT_ROOT=checkpoints # default value
torchrun --nproc_per_node=8 -m cosmos_predict1.diffusion.training.train \
    --config=cosmos_predict1/diffusion/training/config/config_multiview.py \
    -- experiment=text2world_multiview_7b_example_waymo
```

Optionally, multi-node training can be done with
```bash
# 4-node training example.
torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 \
    -m cosmos_predict1.diffusion.training.train \
    --config=cosmos_predict1/diffusion/training/config/config_multiview.py \
    -- experiment=text2world_multiview_7b_example_waymo
```

The model will be post-trained using the above cosmos_nemo_assets dataset.
See the config `text2world_multiview_7b_example_waymo` defined in `cosmos_predict1/diffusion/training/config/text2world/experiment.py` to understand how the dataloader is determined.
```python
num_frames = 57
view_keys = ["pinhole_front_left", "pinhole_front", "pinhole_front_right", "pinhole_side_left", "pinhole_side_right"]
example_multiview_dataset_waymo = L(Dataset)(
    dataset_dir="datasets/waymo",
    sequence_interval=1,
    num_frames=num_frames,
    view_keys=view_keys,
    video_size=(480, 848),
)

dataloader_train = L(DataLoader)(
    dataset=example_multiview_dataset_waymo,
    sampler=L(get_sampler)(dataset=example_multiview_dataset_waymo),
    batch_size=1,
    drop_last=True,
)
...

text2world_multiview_7b_example_waymo = LazyDict(
    dict(
        ...
        dataloader_train=dataloader_train,
        ...
    )
)
...

```

The checkpoints will be saved to `${OUTPUT_ROOT}/PROJECT/GROUP/NAME`.
In the above example, `PROJECT` is `posttraining`, `GROUP` is `diffusion_text2world`, `NAME` is `text2world_multiview_7b_example_waymo`.

See the job config to understand how they are determined.
```python
text2world_multiview_7b_example_waymo = LazyDict(
    dict(
        ...
        job=dict(
            project="posttraining",
            group="diffusion_text2world",
            name="text2world_multiview_7b_example_waymo",
        ),
        ...
    )
)
```

During the training, the checkpoints will be saved in the below structure.
```
checkpoints/posttraining/diffusion_text2world/text2world_multiview_7b_example_waymo/checkpoints/
├── iter_{NUMBER}_reg_model.pt
├── iter_{NUMBER}_ema_model.pt
```


### Inference with the Post-trained Model Checkpoint

The inference can be done with the same interface as described in [examples/inference_diffusion_text2world.md](/examples/inference_diffusion_text2world.md).

#### 1. Copying checkpoint to Designated Location

The post-trained checkpoint needs to be copied to `checkpoints/Cosmos-Predict1-7B-Text2World-Sample-AV-Multiview_post-trained/model.pt`

For example, if a post-trained checkpoint (ema) with 1000 iterations is to be used,
```bash
# copy checkpoint to the designated location
mkdir checkpoints/Cosmos-Predict1-7B-Text2World-Sample-AV-Multiview_post-trained/
cp checkpoints/posttraining/diffusion_text2world/text2world_multiview_7b_example_waymo/checkpoints/iter_000001000_ema_model.pt checkpoints/Cosmos-Predict1-7B-Text2World-Sample-AV-Multiview_post-trained/model.pt
```
#### 2. Running the Inference

We will set the prompt with an environment variable first.
```bash
PROMPT="The video is captured from a camera mounted on a car. The camera is facing forward. \
The video is taken from the perspective of a vehicle's dashboard camera, showing a straight road flanked by snow-covered trees and a clear sky. \
The road is mostly empty, with no visible traffic or pedestrians. \
The sun is setting, casting a warm glow on the horizon and creating long shadows on the snow. \
The trees are tall and leafless, with some coniferous trees interspersed among the bare deciduous trees. \
The snow on the ground appears undisturbed, suggesting a quiet and peaceful setting."

PROMPT_LEFT="The video is captured from a camera mounted on a car. The camera is facing to the left. \
The video captures a series of images from a moving vehicle, showcasing a winter scene with snow-covered ground and trees. \
The sky is a gradient of blue and orange hues, indicating either sunrise or sunset. \
The trees are tall and predominantly coniferous, with some deciduous trees as well. \
The snow appears undisturbed, suggesting a quiet, possibly early morning setting. \
There are no visible people or animals, and the road is clear of traffic. \
The video has a fisheye lens effect, which gives a wide-angle view of the surroundings."

PROMPT_RIGHT="The video is captured from a camera mounted on a car. The camera is facing to the right. \
The video captures a series of images taken from a moving vehicle, showcasing a winter scene with snow-covered ground and trees. \
The sky is a gradient of blue hues, indicating either dawn or dusk. \
The trees are predominantly coniferous, with some bare deciduous trees. \
The snow appears fresh and undisturbed, suggesting recent snowfall. \
There are no visible people or animals, and the environment is serene and untouched. \
The perspective changes as the vehicle moves, providing different angles of the same landscape."

PROMPT_BACK="The video is captured from a camera mounted on a car. The camera is facing backwards. \
The video captures a sequence of frames showing a road covered in snow, with tire tracks visible on the surface. \
The road is flanked by tall, leafless trees, and the sky is a gradient of pink and blue hues, indicating either sunrise or sunset. \
The lighting conditions suggest it is either early morning or late evening. \
There are no visible signs of people or animals, and the road appears to be in a rural or less populated area. \
The vehicles in the video are moving at a steady pace, and there are no visible traffic signs or markings that stand out."

PROMPT_BACK_LEFT="The video is captured from a camera mounted on a car. The camera is facing the rear left side."

PROMPT_BACK_RIGHT="The video is captured from a camera mounted on a car. The camera is facing the rear right side."
```
Here is the command for Waymo data inference with five views. See the config `Cosmos_Predict1_Text2World_7B_Multiview_post_trained` defined in `cosmos_predict1/diffusion/config/inference/cosmos-1-diffusion-text2world-multiview.py` and make sure it's consistent with training config.
```bash
# Run the video generation command with a single gpu
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/text2world_multiview.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Text2World-Sample-AV-Multiview_post-trained \
    --n_views 5 \
    --prompt "${PROMPT}" \
    --prompt_left "${PROMPT_LEFT}" \
    --prompt_right "${PROMPT_RIGHT}" \
    --prompt_back_left "${PROMPT_BACK_LEFT}" \
    --prompt_back_right "${PROMPT_BACK_RIGHT}" \
    --video_save_name diffusion-text2world-multiview-7b-post-train
```
