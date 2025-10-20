## Inference with diffusion-based WorldInterpolator models

### Environment setup

Please refer to the Inference section of [INSTALL.md](/INSTALL.md#inference) for instructions on environment setup.

### Download checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```
3. Accept the [LlamaGuard-7b terms](https://huggingface.co/meta-llama/LlamaGuard-7b)

4. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7):
   ```bash
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B 14B --model_types Video2World --checkpoint_dir checkpoints
   ```

### GPU memory requirements

We report the maximum observed GPU memory usage during end-to-end inference. Additionally, we offer a series of model offloading strategies to help users manage GPU memory usage effectively.

For GPUs with limited memory, we recommend fully offloading all models. For higher-end GPUs, users can select the most suitable offloading strategy considering the numbers provided below.

| Offloading Strategy                                                              | Cosmos-Predict1-7B-Video2World | Cosmos-Predict1-14B-Video2World |
|----------------------------------------------------------------------------------|---------|---------|
| Offload prompt upsampler                                                         | 76.5 GB | > 80.0 GB |
| Offload prompt upsampler & guardrails                                            | 59.9 GB | 73.3 GB |
| Offload prompt upsampler & guardrails & T5 encoder                               | 41.3 GB | 54.8 GB |
| Offload prompt upsampler & guardrails & T5 encoder & tokenizer                   | 41.1 GB | 54.5 GB |
| Offload prompt upsampler & guardrails & T5 encoder & tokenizer & diffusion model | 27.3 GB | 39.0 GB |

The numbers may vary depending on system specs and are for reference only.

### Examples


The inference script is `cosmos_predict1/diffusion/inference/world_interpolator.py`.
It requires the input argument `--input_image_or_video_path` (video input); if the prompt upsampler is disabled, `--prompt` (text input) must also be provided.
To see the complete list of available arguments, run
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/world_interpolator.py --help
```

#### Example 1: single generation
This is the basic example for running inference on the 7B model with input video. No text prompts are provided here.
```bash

CUDA_VISIBLE_DEVICES=1 python3 -m cosmos_predict1.diffusion.inference.world_interpolator \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-WorldInterpolator \
    --input_image_or_video_path assets/diffusion/interpolation_example.mp4  \
    --num_input_frames 1 \
    --offload_prompt_upsampler \
    --video_save_name diffusion-world-interpolator-7b \
    --num_video_frames 10 \
    --num_frame_pairs 2

```
