## Post-training diffusion-based Video2World models

### Model Support Matrix

We support the following Cosmos Diffusion models for post-training. Review the available models and their compute requirements for post-tuning and inference to determine the best model for your use case.

| Model Name                               | Model Status | Compute Requirements for Post-Training |
|----------------------------------------------|------------------|------------------------------------------|
| Cosmos-Predict1-7B-Video2World           | **Supported**    | 8 NVIDIA GPUs*                           |

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

Run the following command to execute an example post-training job with `cosmos_nemo_assets` data.
```bash
export OUTPUT_ROOT=checkpoints # default value
torchrun --nproc_per_node=8 -m cosmos_predict1.diffusion.training.train --config=cosmos_predict1/diffusion/training/config/config.py -- experiment=video2world_7b_example_cosmos_nemo_assets
```

The model will be post-trained using the above cosmos_nemo_assets dataset.
See the config `video2world_7b_example_cosmos_nemo_assets` defined in `cosmos_predict1/diffusion/training/config/video2world/experiment.py` to understand how the dataloader is determined.
```python
num_frames = 121
example_video_dataset_cosmos_nemo_assets = L(Dataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(720, 1280),
    start_frame_interval=1,
)

dataloader_train_cosmos_nemo_assets = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets),
    batch_size=1,
    drop_last=True,
)
...

video2world_7b_example_cosmos_nemo_assets = LazyDict(
    dict(
        ...
        dataloader_train=dataloader_train_cosmos_nemo_assets,
        ...
    )
)
...
```

The checkpoints will be saved to `${OUTPUT_ROOT}/PROJECT/GROUP/NAME`.
In the above example, `PROJECT` is `posttraining`, `GROUP` is `diffusion_video2world`, `NAME` is `video2world_7b_example_cosmos_nemo_assets`.

See the job config to understand how they are determined.
```python
video2world_7b_example_cosmos_nemo_assets = LazyDict(
    dict(
        ...
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_example_cosmos_nemo_assets",
        ),
        ...
    )
)
```

During the training, the checkpoints will be saved in the below structure.
```
checkpoints/posttraining/diffusion_video2world/video2world_7b_example_cosmos_nemo_assets/checkpoints/
├── iter_{NUMBER}_reg_model.pt
├── iter_{NUMBER}_ema_model.pt
```


##### Cosmos-Predict1-7B-Video2World with LoRA

Run the following command to execute an example LoRA post-training job with `cosmos_nemo_assets` data.
```bash
export OUTPUT_ROOT=checkpoints # default value
torchrun --nproc_per_node=4 -m cosmos_predict1.diffusion.training.train \
    --config=cosmos_predict1/diffusion/training/config/config.py \
    -- experiment=video2world_7b_lora_example_cosmos_nemo_assets
```
See the config `video2world_7b_lora_example_cosmos_nemo_assets` defined in `cosmos_predict1/diffusion/training/config/video2world/experiment.py` and `cosmos_predict1/diffusion/training/utils/layer_control/peft_control_config_parser.py` to understand how LoRA is enabled.
```python
video2world_7b_example_cosmos_nemo_assets = LazyDict(
    dict(
        defaults=[
            ...
            {"override /ckpt_klass": "peft"},
            ...
        ],
        trainer=dict(
            ...
            distributed_parallelism="ddp",
            ...
        )
        model=dict(
            ...
            peft_control=get_fa_ca_qv_lora_config(first_nblocks=28, rank=8, scale=1),
            ...
        ),
    )
)
```

During the training, the checkpoints will be saved in the below structure.
```
checkpoints/posttraining/diffusion_video2world/video2world_7b_lora_example_cosmos_nemo_assets/checkpoints/
├── iter_{NUMBER}_model.pt
```

`iter_{NUMBER}_model.pt` contains all weights (base model weights and LoRA weights tensors). When `ema=True`, the checkpoint will contain both regular and ema weights.


#### 4. Inference with the Post-trained Model Checkpoint

The inference can be done with the same interface as described in [examples/inference_diffusion_video2world.md](/examples/inference_diffusion_video2world.md).

##### Cosmos-Predict1-7B-Video2World

1. Copying checkpoint to Designated Location

The post-trained checkpoint needs to be copied to `checkpoints/Cosmos-Predict1-7B-Video2World_post-trained/model.pt`

For example, if a posttrained checkpoint (ema) with 2000 iterations is to be used,
```bash
# copy checkpoint to the designated location
mkdir checkpoints/Cosmos-Predict1-7B-Video2World_post-trained/
cp checkpoints/posttraining/diffusion_video2world/video2world_7b_example_cosmos_nemo_assets/checkpoints/iter_000002000_ema_model.pt checkpoints/Cosmos-Predict1-7B-Video2World_post-trained/model.pt
```

2. Running the Inference

This is the basic example for running inference on the post-trained 7B model with a single image.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World_post-trained \
    --input_image_or_video_path assets/diffusion/video2world_input0.jpg \
    --num_input_frames 1 \
    --offload_prompt_upsampler \
    --video_save_name diffusion-video2world-7b-post-trained
```

##### Cosmos-Predict1-7B-Video2World with LoRA

1. Copying checkpoint to the designated location.

The LoRA post-trained checkpoint needs to be copied to `checkpoints/Cosmos-Predict1-7B-Video2World_post-trained-lora/model.pt`

For example, if a LoRA post-trained checkpoint with 5000 iterations is to be used,
```bash
# copy checkpoint to the designated location
mkdir checkpoints/Cosmos-Predict1-7B-Video2World_post-trained-lora/
cp checkpoints/posttraining/diffusion_video2world/video2world_7b_lora_example_cosmos_nemo_assets/checkpoints/iter_000005000_model.pt checkpoints/Cosmos-Predict1-7B-Video2World_post-trained-lora/model.pt
```

2. Running the inference

The following command is then used to run inference on LoRA post-trained 7B model with a single image input.
```bash
NUM_GPUS=4
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict1/diffusion/inference/video2world.py \
    --num_gpus ${NUM_GPUS} \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World_post-trained-lora \
    --input_image_or_video_path assets/diffusion/video2world_input3.jpg \
    --num_input_frames 1 \
    --offload_prompt_upsampler \
    --video_save_name diffusion-video2world-7b-post-trained-lora
```
The output file is located at `outputs/diffusion-video2world-7b-post-trained-lora.mp4`.
