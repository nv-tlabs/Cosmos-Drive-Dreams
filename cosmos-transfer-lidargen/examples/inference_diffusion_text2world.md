## Inference with diffusion-based Text2World models

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
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B 14B --model_types Text2World --checkpoint_dir checkpoints
   ```

### GPU memory requirements

We report the maximum observed GPU memory usage during end-to-end inference. Additionally, we offer a series of model offloading strategies to help users manage GPU memory usage effectively.

For GPUs with limited memory, we recommend fully offloading all models. For higher-end GPUs, users can select the most suitable offloading strategy considering the numbers provided below.

| Offloading Strategy | Cosmos-Predict1-7B-Text2World | Cosmos-Predict1-14B-Text2World |
|-------------|---------|---------|
| Offload prompt upsampler | 74.0 GB | > 80.0 GB |
| Offload prompt upsampler & guardrails | 57.1 GB | 70.5 GB |
| Offload prompt upsampler & guardrails & T5 encoder | 38.5 GB | 51.9 GB |
| Offload prompt upsampler & guardrails & T5 encoder & tokenizer | 38.3 GB | 51.7 GB |
| Offload prompt upsampler & guardrails & T5 encoder & tokenizer & diffusion model | 24.4 GB | 39.0 GB |

The numbers may vary depending on system specs and are for reference only.

### Examples

There are two models available for diffusion world generation from text input: `Cosmos-Predict1-7B-Text2World` and `Cosmos-Predict1-14B-Text2World`.

The inference script is `cosmos_predict1/diffusion/inference/text2world.py`.
It requires the input argument `--prompt` (text input).
To see the complete list of available arguments, run
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/text2world.py --help
```

We will set the prompt with an environment variable first.
```bash
PROMPT="A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. \
The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. \
A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, \
suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. \
The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of \
field that keeps the focus on the robot while subtly blurring the background for a cinematic effect."
```

#### Example 1: single generation
This is the basic example for running inference on the 7B model with a single prompt.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/text2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Text2World \
    --prompt "${PROMPT}" \
    --offload_prompt_upsampler \
    --video_save_name diffusion-text2world-7b
```

#### Example 2: single generation on the 14B model with model offloading
We run inference on the 14B model with offloading flags enabled. This is suitable for low-memory GPUs. Model offloading is also required for the 14B model to avoid OOM.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/text2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-14B-Text2World \
    --prompt "${PROMPT}" \
    --offload_tokenizer \
    --offload_diffusion_transformer \
    --offload_text_encoder_model \
    --offload_prompt_upsampler \
    --offload_guardrail_models \
    --video_save_name diffusion-text2world-14b
```

#### Example 3: single generation with multi-GPU inference
This example runs parallelized inference on a single prompt using 8 GPUs.
```bash
NUM_GPUS=8
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict1/diffusion/inference/text2world.py \
    --num_gpus ${NUM_GPUS} \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Text2World \
    --prompt "${PROMPT}" \
    --offload_prompt_upsampler \
    --video_save_name diffusion-text2world-7b-8gpu
```

#### Example 4: batch generation
This example runs inference on a batch of prompts, provided through the `--batch_input_path` argument (path to a JSONL file).
The JSONL file should contain one prompt per line in the following format, where each line must contain a `prompt` field:
```json
{"prompt": "prompt1"}
{"prompt": "prompt2"}
```
Inference command:
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/text2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Text2World \
    --batch_input_path assets/diffusion/batch_inputs/text2world.jsonl \
    --offload_prompt_upsampler \
    --video_save_folder diffusion-text2world-7b-batch
```
