## Inference with autoregressive-based base models

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
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_autoregressive_checkpoints.py --model_sizes 4B 12B --checkpoint_dir checkpoints
   ```

### GPU memory requirements

We report the maximum observed GPU memory usage during end-to-end inference. Additionally, we offer a series of model offloading strategies to help users manage GPU memory usage effectively.

For GPUs with limited memory, we recommend fully offloading all models. For higher-end GPUs, users can select the most suitable offloading strategy considering the numbers provided below.

| Offloading Strategy | Cosmos-Predict1-4B | Cosmos-Predict1-12B |
|-------------|---------|---------|
| No offloading | 31.3 GB | 47.5 GB |
| Offload guardrails | 28.9 GB | 45.2 GB |
| Offload guardrails & diffusion decoder | 28.5 GB | 43.1 GB |
| Offload guardrails & diffusion decoder & tokenizer | 27.3 GB | 42.9 GB |
| Offload guardrails & diffusion decoder & tokenizer & AR model | 18.7 GB | 27.4 GB |

The numbers may vary depending on system specs and are for reference only.

### Examples
There are two model types available for autoregressive world generation: `Cosmos-Predict1-4B` and `Cosmos-Predict1-12B`.
It requires the input argument `--input_image_or_video_path` (image/video input).
The inference script is `cosmos_predict1/autoregressive/inference/base.py`.
To see the complete list of available arguments, run
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/autoregressive/inference/base.py --help
```

#### Example 1: single generation
This is the basic example for running inference on the 4B model.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/autoregressive/inference/base.py \
    --checkpoint_dir checkpoints \
    --ar_model_dir Cosmos-Predict1-4B \
    --input_type video \
    --input_image_or_video_path assets/autoregressive/input.mp4 \
    --top_p 0.8 \
    --temperature 1.0 \
    --offload_diffusion_decoder \
    --offload_tokenizer \
    --video_save_name autoregressive-4b
```

#### Example 2: single generation with multi-GPU inference
This example runs parallelized inference using 8 GPUs.
```bash
NUM_GPUS=8
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict1/autoregressive/inference/base.py \
    --num_gpus ${NUM_GPUS} \
    --checkpoint_dir checkpoints \
    --ar_model_dir Cosmos-Predict1-4B \
    --input_type video \
    --input_image_or_video_path assets/autoregressive/input.mp4 \
    --top_p 0.8 \
    --temperature 1.0 \
    --offload_diffusion_decoder \
    --offload_tokenizer \
    --video_save_name autoregressive-4b-8gpu
```

#### Example 3: batch generation
This example runs inference on a batch of prompts, provided through the `--batch_input_path` argument (path to a JSONL file).
The JSONL file should contain one visual input per line in the following format, where each line must contain a `visual_input` field:
```json
{"visual_input": "path/to/video1.mp4"}
{"visual_input": "path/to/video2.mp4"}
```
Inference command:
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/autoregressive/inference/base.py \
    --checkpoint_dir checkpoints \
    --ar_model_dir Cosmos-Predict1-4B \
    --batch_input_path assets/diffusion/batch_inputs/text2world.jsonl \
    --top_p 0.8 \
    --temperature 1.0 \
    --offload_diffusion_decoder \
    --offload_tokenizer \
    --video_save_folder autoregressive-4b-batch
```
