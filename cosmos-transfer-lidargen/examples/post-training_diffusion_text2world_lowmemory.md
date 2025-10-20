## Post-training diffusion-based Text2World models

### Model Support Matrix

We support the following Cosmos Diffusion models for post-training. Review the available models and their compute requirements for post-tuning and inference to determine the best model for your use case.

| Model Name                               | Model Status | Compute Requirements for Post-Training |
|----------------------------------------------|------------------|------------------------------------------|
| Cosmos-Predict1-7B-Text2World           | **Supported**    | 8 NVIDIA GPUs*                           |
| Cosmos-Predict1-14B-Text2World          | **Supported**    | 8 NVIDIA GPUs* x 4 nodes                 |

**\*** `H100-80GB` or `A100-80GB` GPUs are recommended.
Optionally, reducing the video resolution and the number of frames can facilitate training with less number (4) of GPUs or with GPUs with lower memory (40GB).

### Environment setup

Please refer to the Post-training section of [INSTALL.md](/INSTALL.md#post-training) for instructions on environment setup.

### Download checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```

3. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7):
   ```bash
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B 14B --model_types Text2World
   ```

### Examples

Post-training a Cosmos Diffusion-based WFM enables you to train the model to generate videos that are more specific to your use case.

There are 3 steps to post-training: downloading a dataset, preprocessing the data, and post-training the model.

#### 1. Download a Dataset

The first step is to download a dataset with videos and captions.

You must provide a folder containing a collection of videos in **MP4 format**, preferably 720p. These videos should focus on the subject throughout the entire video so that each video chunk contains the subject.

You can use [nvidia/Cosmos-NeMo-Assets](https://huggingface.co/datasets/nvidia/Cosmos-NeMo-Assets) for post-training.

```bash
mkdir -p datasets/cosmos_nemo_assets/

# This command will download the videos for physical AI
huggingface-cli download nvidia/Cosmos-NeMo-Assets --repo-type dataset --local-dir datasets/cosmos_nemo_assets/ --include "*.mp4*"

mv datasets/cosmos_nemo_assets/nemo_diffusion_example_data datasets/cosmos_nemo_assets/videos
```

#### 2. Preprocessing the Data

Run the following command to pre-compute T5-XXL embeddings for the video captions used for post-training:

```bash
# The script will use the provided prompt, save the T5-XXL embeddings in pickle format.
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/get_t5_embeddings_from_cosmos_nemo_assets.py --dataset_path datasets/cosmos_nemo_assets --prompt "A video of sks teal robot."
```

Dataset folder format:
```
datasets/cosmos_nemo_assets/
├── metas/
│   ├── *.txt
├── videos/
│   ├── *.mp4
├── t5_xxl/
│   ├── *.pickle
```

#### 3. Post-train the Model

##### Cosmos-Predict1-7B-Text2World

* Training with 4 x 80GB GPUs

To run with 4 GPUs with H100/A100 80GB, run experiment `text2world_7b_example_cosmos_nemo_assets_4gpu_80gb`.
It trains with `cosmos_nemo_assets` data at 384x384 resolution, video length of 121 frames.

```bash
torchrun --nproc_per_node=4 -m cosmos_predict1.diffusion.training.train \
    --config=cosmos_predict1/diffusion/training/config/config.py \
    -- experiment=text2world_7b_example_cosmos_nemo_assets_4gpu_80gb
```

See the config `text2world_7b_example_cosmos_nemo_assets_4gpu_80gb` defined in `cosmos_predict1/diffusion/training/config/text2world/experiment.py` for details.
```python
n_length_4gpu_80gb = 15
num_frames_4gpu_80gb = 8 * n_length_4gpu_80gb + 1  # 121
example_video_dataset_cosmos_nemo_assets_4gpu_80gb = L(Dataset)(
    num_frames=num_frames_4gpu_80gb,
    video_size=(384, 384),  # a low-res example.
    ...
)

dataloader_train_cosmos_nemo_assets_4gpu_80gb = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_4gpu_80gb,
    ...
)
dataloader_val_cosmos_nemo_assets_4gpu_80gb = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_4gpu_80gb,
    ...
)
...

text2world_7b_example_cosmos_nemo_assets_4gpu_80gb = LazyDict(
    dict(
        job=dict(
            project="posttraining",
            group="diffusion_text2world",
            name="text2world_7b_example_cosmos_nemo_assets_4gpu_80gb",
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                48,  # Latent height dim
                48,  # Latent width dim
            ],
            vae=dict(
                pixel_chunk_duration=num_frames_4gpu_80gb,
                spatial_resolution="384",
            ),
            ...
        ),
        dataloader_train=dataloader_train_cosmos_nemo_assets_4gpu_80gb,
        ...
    )
)
...
```

The checkpoints will be saved to `${OUTPUT_ROOT}/PROJECT/GROUP/NAME`.
In the above example, `PROJECT` is `posttraining`, `GROUP` is `diffusion_text2world`, `NAME` is `text2world_7b_example_cosmos_nemo_assets_4gpu_80gb`.

During the training, the checkpoints will be saved in the below structure.
```
checkpoints/posttraining/diffusion_text2world/text2world_7b_example_cosmos_nemo_assets_4gpu_80gb/checkpoints/
├── iter_{NUMBER}_reg_model.pt
├── iter_{NUMBER}_ema_model.pt
```

* Training with 8 x 40GB GPUs
To run with 8 GPUs with A100 40GB, run experiment `text2world_7b_example_cosmos_nemo_assets_8gpu_40gb`.
It trains with `cosmos_nemo_assets` data at 384x384 resolution, video length of 33 frames.

```bash
torchrun --nproc_per_node=8 -m cosmos_predict1.diffusion.training.train \
    --config=cosmos_predict1/diffusion/training/config/config.py \
    -- experiment=text2world_7b_example_cosmos_nemo_assets_8gpu_40gb
```
See the config `text2world_7b_example_cosmos_nemo_assets_8gpu_40gb` defined in `cosmos_predict1/diffusion/training/config/text2world/experiment.py` for details.

```python
n_length_8gpu_40gb = 4
num_frames_8gpu_40gb = 8 * n_length_8gpu_40gb + 1  # 33
example_video_dataset_cosmos_nemo_assets_8gpu_40gb = L(Dataset)(
    num_frames=num_frames_8gpu_40gb,
    video_size=(384, 384),  # a low-res example for lower VRAM utilization without considering aspect ratio.
    ...
)

dataloader_train_cosmos_nemo_assets_8gpu_40gb = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_8gpu_40gb,
    ...
)
dataloader_val_cosmos_nemo_assets_8gpu_40gb = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_8gpu_40gb,
    ...
)

...

text2world_7b_example_cosmos_nemo_assets_8gpu_40gb = LazyDict(
    dict(
        job=dict(
            project="posttraining",
            group="diffusion_text2world",
            name="text2world_7b_example_cosmos_nemo_assets_8gpu_40gb",
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                48,  # Latent height dim
                48,  # Latent width dim
            ],
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            vae=dict(
                pixel_chunk_duration=num_frames_8gpu_40gb,
                spatial_resolution="384",
            ),
            ...
        ),
        dataloader_train=dataloader_train_cosmos_nemo_assets_8gpu_40gb,
        ...
    )
)
...
```


* Training with 4 x 40GB GPUs
To run with 4 GPUs with A100 40GB, run experiment `text2world_7b_example_cosmos_nemo_assets_4gpu_40gb`.
It trains with `cosmos_nemo_assets` data at 384x384 resolution, video length of 17 frames.

```bash
torchrun --nproc_per_node=4 -m cosmos_predict1.diffusion.training.train \
    --config=cosmos_predict1/diffusion/training/config/config.py \
    -- experiment=text2world_7b_example_cosmos_nemo_assets_4gpu_40gb
```
See the config `text2world_7b_example_cosmos_nemo_assets_4gpu_40gb` defined in `cosmos_predict1/diffusion/training/config/text2world/experiment.py` for details.

```python
n_length_4gpu_40gb = 2
num_frames_4gpu_40gb = 8 * n_length_4gpu_40gb + 1  # 17
example_video_dataset_cosmos_nemo_assets_4gpu_40gb = L(Dataset)(
    num_frames=num_frames_4gpu_40gb,
    video_size=(384, 384),  # a low-res example for lower VRAM utilization without considering aspect ratio.
    ...
)

dataloader_train_cosmos_nemo_assets_4gpu_40gb = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_4gpu_40gb,
    ...
)
dataloader_val_cosmos_nemo_assets_4gpu_40gb = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_4gpu_40gb,
    ...
)

...

text2world_7b_example_cosmos_nemo_assets_4gpu_40gb = LazyDict(
    dict(
        job=dict(
            project="posttraining",
            group="diffusion_text2world",
            name="text2world_7b_example_cosmos_nemo_assets_4gpu_40gb",
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                48,  # Latent height dim
                48,  # Latent width dim
            ],
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            vae=dict(
                pixel_chunk_duration=num_frames_4gpu_40gb,
                spatial_resolution="384",
            ),
            ...
        ),
        dataloader_train=dataloader_train_cosmos_nemo_assets_4gpu_40gb,
        ...
    )
)
...
```


#### 4. Inference with the Post-trained Model Checkpoint

The inference can be done with the same interface as described in [examples/inference_diffusion_text2world.md](inference_diffusion_text2world.md).

##### Cosmos-Predict1-7B-Text2World

1. Copying checkpoint to the designated location

The post-trained checkpoint needs to be copied to `checkpoints/Cosmos-Predict1-7B-Text2World_post-trained-4gpu_80gb/model.pt`

For example, if a post-trained checkpoint (ema) with 2000 iterations is to be used,
```bash
# copy checkpoint to the designated location
mkdir checkpoints/Cosmos-Predict1-7B-Text2World_post-trained-4gpu_80gb/
cp checkpoints/posttraining/diffusion_text2world/text2world_7b_example_cosmos_nemo_assets_4gpu_80gb/checkpoints/iter_000002000_ema_model.pt checkpoints/Cosmos-Predict1-7B-Text2World_post-trained-4gpu_80gb/model.pt
```

2. Running the inference

We will set the prompt with an environment variable first.
```bash
PROMPT="A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. \
The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. \
A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, \
suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. \
The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of \
field that keeps the focus on the robot while subtly blurring the background for a cinematic effect."
```

```bash
# Run the video generation command with a single gpu
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/text2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Text2World_post-trained-4gpu_80gb \
    --num_video_frames 121 \
    --prompt "${PROMPT}" \
    --offload_prompt_upsampler \
    --video_save_name diffusion-text2world-7b-post-trained_4gpu_80gb
```

The output file is located at `outputs/diffusion-text2world-7b-post-trained_4gpu_80gb.mp4`.

* Similarly, 8 GPU 40GB post-trained model inference can be done with

```bash
# copy checkpoint to the designated location
mkdir checkpoints/Cosmos-Predict1-7B-Text2World_post-trained-8gpu_40gb/
cp checkpoints/posttraining/diffusion_text2world/text2world_7b_example_cosmos_nemo_assets_8gpu_40gb/checkpoints/iter_000002000_reg_model.pt checkpoints/Cosmos-Predict1-7B-Text2World_post-trained-8gpu_40gb/model.pt
```

```bash
# Run the video generation command with a single gpu
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/text2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Text2World_post-trained-8gpu_40gb \
    --num_video_frames 33 \
    --prompt "${PROMPT}" \
    --offload_text_encoder_model \
    --offload_prompt_upsampler \
    --offload_guardrail_models \
    --video_save_name diffusion-text2world-7b-post-trained_8gpu_40gb
```

The output file is located at `outputs/diffusion-text2world-7b-post-trained_8gpu_40gb.mp4`.

* Similarly, 4 GPU 40GB post-trained model inference can be done with

Note that we use `reg` model here instead of `ema` as ema is disabled during posttraining to reduce memory consumption.
```bash
# copy checkpoint to the designated location
mkdir checkpoints/Cosmos-Predict1-7B-Text2World_post-trained-4gpu_40gb/
cp checkpoints/posttraining/diffusion_text2world/text2world_7b_example_cosmos_nemo_assets_4gpu_40gb/checkpoints/iter_000002000_reg_model.pt checkpoints/Cosmos-Predict1-7B-Text2World_post-trained-4gpu_40gb/model.pt
```

```bash
# Run the video generation command with a single gpu
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/text2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Text2World_post-trained-4gpu_40gb \
    --num_video_frames 17 \
    --prompt "${PROMPT}" \
    --offload_text_encoder_model \
    --offload_prompt_upsampler \
    --offload_guardrail_models \
    --video_save_name diffusion-text2world-7b-post-trained_4gpu_40gb
```

The output file is located at `outputs/diffusion-text2world-7b-post-trained_4gpu_40gb.mp4`.
