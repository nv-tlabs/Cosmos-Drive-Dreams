## Inference with diffusion-based Video2World models

### Environment setup

Please refer to the Inference section of [INSTALL.md](/INSTALL.md#inference) for instructions on environment setup.

### Download checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```
3. Accept the [Llama-Guard-3-8B terms](https://huggingface.co/meta-llama/Llama-Guard-3-8B)

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

There are two models available for diffusion world generation from text and image/video input: `Cosmos-Predict1-7B-Video2World` and `Cosmos-Predict1-14B-Video2World`.

The inference script is `cosmos_predict1/diffusion/inference/video2world.py`.
It requires the input argument `--input_image_or_video_path` (image/video input); if the prompt upsampler is disabled, `--prompt` (text input) must also be provided.
To see the complete list of available arguments, run
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py --help
```

#### Example 1: single generation
This is the basic example for running inference on the 7B model with a single image. No text prompts are provided here.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World \
    --input_image_or_video_path assets/diffusion/video2world_input0.jpg \
    --num_input_frames 1 \
    --offload_prompt_upsampler \
    --video_save_name diffusion-video2world-7b
```

#### Example 2: single generation on the 14B model with model offloading
We run inference on the 14B model with offloading flags enabled. This is suitable for low-memory GPUs. Model offloading is also required for the 14B model to avoid OOM.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-14B-Video2World \
    --input_image_or_video_path assets/diffusion/video2world_input0.jpg \
    --num_input_frames 1 \
    --offload_tokenizer \
    --offload_diffusion_transformer \
    --offload_text_encoder_model \
    --offload_prompt_upsampler \
    --offload_guardrail_models \
    --video_save_name diffusion-video2world-14b
```

#### Example 3: single generation with multi-GPU inference
This example runs parallelized inference on a single prompt using 8 GPUs.
```bash
NUM_GPUS=8
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict1/diffusion/inference/video2world.py \
    --num_gpus ${NUM_GPUS} \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World \
    --input_image_or_video_path assets/diffusion/video2world_input0.jpg \
    --num_input_frames 1 \
    --offload_prompt_upsampler \
    --video_save_name diffusion-video2world-7b
```

#### Example 4: batch generation
This example runs inference on a batch of prompts, provided through the `--batch_input_path` argument (path to a JSONL file).
Each line in the JSONL file must contain a `visual_input` field:
```json
{"visual_input": "path/to/video1.mp4"}
{"visual_input": "path/to/video2.mp4"}
```
Inference command (with 9 input frames):
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World \
    --batch_input_path assets/diffusion/batch_inputs/video2world_ps.jsonl \
    --num_input_frames 9 \
    --offload_prompt_upsampler \
    --video_save_folder diffusion-video2world-7b-batch
```

#### Example 5: batch generation without prompt upsampler
This example runs inference on a batch of prompts, provided through the `--batch_input_path` argument (path to a JSONL file).
The prompt upsampler is disabled, and thus each line in the JSONL file will need to include both `prompt` and `visual_input` fields.
```json
{"prompt": "prompt1", "visual_input": "path/to/video1.mp4"}
{"prompt": "prompt2", "visual_input": "path/to/video2.mp4"}
```
Inference command (with 9 input frames):
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World \
    --batch_input_path assets/diffusion/batch_inputs/video2world_wo_ps.jsonl \
    --num_input_frames 9 \
    --disable_prompt_upsampler \
    --video_save_folder diffusion-video2world-7b-batch-wo-ps
```
