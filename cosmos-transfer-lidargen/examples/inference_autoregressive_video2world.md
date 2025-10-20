## Inference with autoregressive-based Video2World models

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
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_autoregressive_checkpoints.py --model_sizes 5B 13B --checkpoint_dir checkpoints
   ```

### GPU memory requirements

We report the maximum observed GPU memory usage during end-to-end inference. Additionally, we offer a series of model offloading strategies to help users manage GPU memory usage effectively.

For GPUs with limited memory, we recommend fully offloading all models. For higher-end GPUs, users can select the most suitable offloading strategy considering the numbers provided below.

| Offloading Strategy | Cosmos-Predict1-5B-Video2World | Cosmos-Predict1-13B-Video2World |
|-------------|---------|---------|
| No offloading | 66.2 GB | > 80 GB |
| Offload guardrails | 58.7 GB | 76.6 GB |
| Offload guardrails & T5 encoder | 41.3 GB | 58.0 GB |
| Offload guardrails & T5 encoder & diffusion decoder | 29.0 GB | 46.9 GB |
| Offload guardrails & T5 encoder & diffusion decoder & tokenizer | 28.8 GB | 46.7 GB |
| Offload guardrails & T5 encoder & diffusion decoder & tokenizer & AR model | 21.1 GB | 30.9 GB |

The numbers may vary depending on system specs and are for reference only.

### Examples
There are two model types available for autoregressive world generation: `Cosmos-Predict1-5B-Video2World` and `Cosmos-Predict1-13B-Video2World`.

The inference script is `cosmos_predict1/autoregressive/inference/video2world.py`.
It requires both `--input_image_or_video_path` (image/video input) and `--prompt` (text input).
To see the complete list of available arguments, run
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/autoregressive/inference/video2world.py --help
```

We will set the prompt with an environment variable first.
```bash
PROMPT="A video recorded from a moving vehicle's perspective, capturing roads, buildings, landscapes, and changing weather and lighting conditions."
```

#### Example 1: single generation
This is the basic example for running inference on the 4B model.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/autoregressive/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --ar_model_dir Cosmos-Predict1-5B-Video2World \
    --input_type text_and_video \
    --input_image_or_video_path assets/autoregressive/input.mp4 \
    --prompt "${PROMPT}" \
    --top_p 0.7 \
    --temperature 1.0 \
    --offload_diffusion_decoder \
    --offload_tokenizer \
    --video_save_name autoregressive-video2world-5b
```

#### Example 2: single generation with multi-GPU inference
This example runs parallelized inference using 8 GPUs.
```bash
NUM_GPUS=8
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict1/autoregressive/inference/video2world.py \
    --num_gpus ${NUM_GPUS} \
    --checkpoint_dir checkpoints \
    --ar_model_dir Cosmos-Predict1-5B-Video2World \
    --input_type text_and_video \
    --input_image_or_video_path assets/autoregressive/input.mp4 \
    --prompt "${PROMPT}" \
    --top_p 0.7 \
    --temperature 1.0 \
    --offload_diffusion_decoder \
    --offload_tokenizer \
    --video_save_name autoregressive-video2world-5b-8gpu
```

#### Example 3: batch generation
This example runs inference on a batch of prompts, provided through the `--batch_input_path` argument (path to a JSONL file).
Each line in the JSONL file must contain both `prompt` and `visual_input` fields:
```json
{"prompt": "prompt1", "visual_input": "path/to/video1.mp4"}
{"prompt": "prompt2", "visual_input": "path/to/video2.mp4"}
```
Inference command:
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/autoregressive/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --ar_model_dir Cosmos-Predict1-5B-Video2World \
    --batch_input_path assets/diffusion/batch_inputs/video2world.jsonl \
    --top_p 0.7 \
    --temperature 1.0 \
    --offload_diffusion_decoder \
    --offload_tokenizer \
    --video_save_folder autoregressive-video2world-5b-batch
```
