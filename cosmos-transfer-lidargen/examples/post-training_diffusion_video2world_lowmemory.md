## Post-training diffusion-based Video2World models

### Model Support Matrix

We support the following Cosmos Diffusion models for post-training. Review the available models and their compute requirements for post-tuning and inference to determine the best model for your use case.

| Model Name                               | Model Status | Compute Requirements for Post-Training |
|----------------------------------------------|------------------|------------------------------------------|
| Cosmos-Predict1-7B-Video2World           | **Supported**    | 8 NVIDIA GPUs*                           |

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
3. Accept the [LlamaGuard-7b terms](https://huggingface.co/meta-llama/LlamaGuard-7b)

4. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7):
   ```bash
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B --model_types Video2World --checkpoint_dir checkpoints
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

To run with 4 GPUs with H100/A100 80GB, run experiment `video2world_7b_example_cosmos_nemo_assets_4gpu_80gb`.
It trains with `cosmos_nemo_assets` data at 384x384 resolution, video length of 121 frames.

```bash
torchrun --nproc_per_node=4 -m cosmos_predict1.diffusion.training.train \
    --config=cosmos_predict1/diffusion/training/config/config.py \
    -- experiment=video2world_7b_example_cosmos_nemo_assets_4gpu_80gb
```

See the config `video2world_7b_example_cosmos_nemo_assets_4gpu_80gb` defined in `cosmos_predict1/diffusion/training/config/video2world/experiment.py` for details.
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

video2world_7b_example_cosmos_nemo_assets_4gpu_80gb = LazyDict(
    dict(
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_example_cosmos_nemo_assets_4gpu_80gb",
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
In the above example, `PROJECT` is `posttraining`, `GROUP` is `diffusion_video2world`, `NAME` is `video2world_7b_example_cosmos_nemo_assets_4gpu_80gb`.

During the training, the checkpoints will be saved in the below structure.
```
checkpoints/posttraining/diffusion_video2world/video2world_7b_example_cosmos_nemo_assets_4gpu_80gb/checkpoints/
├── iter_{NUMBER}_reg_model.pt
├── iter_{NUMBER}_ema_model.pt
```


* Training with 8 x 40GB GPUs
To run with 8 GPUs with A100 40GB, run experiment `video2world_7b_example_cosmos_nemo_assets_8gpu_40gb`.
It trains with `cosmos_nemo_assets` data at 384x384 resolution, video length of 25 frames.

```bash
torchrun --nproc_per_node=8 -m cosmos_predict1.diffusion.training.train \
    --config=cosmos_predict1/diffusion/training/config/config.py \
    -- experiment=video2world_7b_example_cosmos_nemo_assets_8gpu_40gb
```
See the config `video2world_7b_example_cosmos_nemo_assets_8gpu_40gb` defined in `cosmos_predict1/diffusion/training/config/video2world/experiment.py` for details.

```python
n_length_8gpu_40gb = 3
num_frames_8gpu_40gb = 8 * n_length_8gpu_40gb + 1  # 25
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

video2world_7b_example_cosmos_nemo_assets_8gpu_40gb = LazyDict(
    dict(
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_example_cosmos_nemo_assets_8gpu_40gb",
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
To run with 4 GPUs with A100 40GB, run experiment `video2world_7b_example_cosmos_nemo_assets_4gpu_40gb`.
It trains with `cosmos_nemo_assets` data at 384x384 resolution, video length of 25 frames.

```bash
torchrun --nproc_per_node=4 -m cosmos_predict1.diffusion.training.train \
    --config=cosmos_predict1/diffusion/training/config/config.py \
    -- experiment=video2world_7b_example_cosmos_nemo_assets_4gpu_40gb
```
See the config `video2world_7b_example_cosmos_nemo_assets_4gpu_40gb` defined in `cosmos_predict1/diffusion/training/config/video2world/experiment.py` for details.

```python
n_length_4gpu_40gb = 2
num_frames_4gpu_40gb = 8 * n_length_4gpu_40gb + 1  # 17
example_video_dataset_cosmos_nemo_assets_4gpu_40gb = L(Dataset)(
    num_frames=num_frames_4gpu_40gb,
    video_size=(192, 192),  # a low-res example for lower VRAM utilization without considering aspect ratio.
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

video2world_7b_example_cosmos_nemo_assets_4gpu_40gb = LazyDict(
    dict(
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_example_cosmos_nemo_assets_4gpu_40gb",
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                24,  # Latent height dim
                24,  # Latent width dim
            ],
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            vae=dict(
                pixel_chunk_duration=num_frames_4gpu_40gb,
                spatial_resolution="192",
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

The inference can be done with the same interface as described in [examples/inference_diffusion_video2world.md](/examples/inference_diffusion_video2world.md).

##### Cosmos-Predict1-7B-Video2World

1. Copying checkpoint to Designated Location

The post-trained checkpoint needs to be copied to `checkpoints/Cosmos-Predict1-7B-Video2World_post-trained-4gpu_80gb/model.pt`

For example, if a posttrained checkpoint (ema) with 2000 iterations is to be used,
```bash
# copy checkpoint to the designated location
mkdir checkpoints/Cosmos-Predict1-7B-Video2World_post-trained-4gpu_80gb/
cp checkpoints/posttraining/diffusion_video2world/video2world_7b_example_cosmos_nemo_assets_4gpu_80gb/checkpoints/iter_000002000_ema_model.pt checkpoints/Cosmos-Predict1-7B-Video2World_post-trained-4gpu_80gb/model.pt
```

2. Running the Inference

This is the basic example for running inference on the post-trained 7B model with a single image.
```bash
# Run the video generation command with a single gpu
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World_post-trained-4gpu_80gb \
    --input_image_or_video_path assets/diffusion/video2world_input0.jpg \
    --num_input_frames 1 \
    --offload_prompt_upsampler \
    --video_save_name diffusion-video2world-7b-post-trained-4gpu_80gb
```

* Similarly, 8 GPU 40GB post-trained model inference can be done with
```bash
# copy checkpoint to the designated location
mkdir checkpoints/Cosmos-Predict1-7B-Video2World_post-trained-8gpu_40gb/
cp checkpoints/posttraining/diffusion_video2world/video2world_7b_example_cosmos_nemo_assets_8gpu_40gb/checkpoints/iter_000002000_reg_model.pt checkpoints/Cosmos-Predict1-7B-Video2World_post-trained-8gpu_40gb/model.pt
```

```bash
# Run the video generation command with a single gpu
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World_post-trained-8gpu_40gb \
    --input_image_or_video_path assets/diffusion/video2world_input0.jpg \
    --num_input_frames 1 \
    --num_video_frames 25 \
    --offload_text_encoder_model \
    --offload_prompt_upsampler \
    --offload_guardrail_models \
    --video_save_name diffusion-video2world-7b-post-trained-8gpu_40gb
```

* Similarly, 4 GPU 40GB post-trained model inference can be done with
```bash
# copy checkpoint to the designated location
mkdir checkpoints/Cosmos-Predict1-7B-Video2World_post-trained-4gpu_40gb/
cp checkpoints/posttraining/diffusion_video2world/video2world_7b_example_cosmos_nemo_assets_4gpu_40gb/checkpoints/iter_000002000_reg_model.pt checkpoints/Cosmos-Predict1-7B-Video2World_post-trained-4gpu_40gb/model.pt
```

```bash
# Run the video generation command with a single gpu
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World_post-trained-4gpu_40gb \
    --input_image_or_video_path assets/diffusion/video2world_input0.jpg \
    --num_input_frames 1 \
    --num_video_frames 25 \
    --offload_text_encoder_model \
    --offload_prompt_upsampler \
    --offload_guardrail_models \
    --video_save_name diffusion-video2world-7b-post-trained-4gpu_40gb
```
