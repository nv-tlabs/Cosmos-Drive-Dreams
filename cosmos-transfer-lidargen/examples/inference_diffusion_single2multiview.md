## Inference with diffusion-based SingleToMultiView-Sample-AV models

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
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B --model_types SingleToMultiView-Sample-AV --checkpoint_dir checkpoints
   ```

### Examples
We provide `Cosmos-Predict1-7B-Video2World-Sample-AV-Single2MultiView/t2w_model.pt` for world generation from front view video and text,
and `Cosmos-Predict1-7B-Video2World-Sample-AV-Single2MultiView/v2w_model.pt` for world generation from front view video and multiview initial frame(s).

The inference script is `cosmos_predict1/diffusion/inference/video2world_view_extend_multiview.py`.
To see the complete list of available arguments, run
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world_view_extend_multiview.py --help
```

We will set the prompt with an environment variable first.
```bash
PROMPT="The video is captured from a camera mounted on a car. The camera is facing forward. \
The video captures a nighttime drive through a suburban area. The road is illuminated by streetlights and vehicle headlights, creating a well-lit path. \
Palm trees line both sides of the street, adding a tropical feel to the environment. \
Several cars are parked along the right side of the road, and a few are driving in the opposite direction. \
The sky is overcast, with no visible stars or moon, indicating it is likely late evening or night. \
The overall scene is quiet and peaceful, with no pedestrians or significant traffic."

PROMPT_LEFT="The video is captured from a camera mounted on a car. The camera is facing to the left. \
The video captures a nighttime drive through a quiet neighborhood. A white car is seen turning left onto a street lined with palm trees and other trees. \
The street is illuminated by streetlights, and there are a few parked cars visible. \
The sky is overcast, and the overall scene is dimly lit, indicating it is nighttime."

PROMPT_RIGHT="The video is captured from a camera mounted on a car. The camera is facing to the right. \
The The video captures a nighttime driving scene on a multi-lane road. The road is bordered by a high concrete barrier. \
Several cars are visible, including a white van, a black sedan, and a red car. \
The traffic appears to be moving slowly, possibly due to congestion. The sky is overcast, and the lighting suggests it is nighttime. \
Trees and some buildings can be seen in the background, indicating an urban or suburban setting."

PROMPT_BACK="The video is captured from a camera mounted on a car. The camera is facing backwards. \
The video depicts a nighttime drive through a residential area. The street is illuminated by streetlights and the headlights of a car in front. \
The road is lined with parked cars and houses, and there are trees and a fence along the side. \
A white truck is seen turning left onto the street, and a person is standing near the fence. \
The scene is quiet and there are no visible pedestrians or other vehicles. The weather appears to be clear."

PROMPT_BACK_LEFT="The video is captured from a camera mounted on a car. The camera is facing the rear left side."

PROMPT_BACK_RIGHT="The video is captured from a camera mounted on a car. The camera is facing the rear right side."
```

#### Example 1: single view extension with Text condition
This is the basic example for running inference on the 7B single to multiview model with a single view input video.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world_view_extend_multiview.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World-Sample-AV-Single2MultiView/t2w_model.pt \
    --view_condition_video assets/diffusion/sv2mv_input_view.mp4 \
    --num_input_frames 1 \
    --condition_location "first_cam" \
    --prompt "${PROMPT}" \
    --prompt_left "${PROMPT_LEFT}" \
    --prompt_right "${PROMPT_RIGHT}" \
    --prompt_back "${PROMPT_BACK}" \
    --prompt_back_left "${PROMPT_BACK_LEFT}" \
    --prompt_back_right "${PROMPT_BACK_RIGHT}" \
    --video_save_name diffusion-single2multiview-text2world
```
Similar to other examples, multiple gpus can be leveraged in generation too:
```bash
NUM_GPUS=8
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict1/diffusion/inference/video2world_view_extend_multiview.py \
    --num_gpus ${NUM_GPUS} \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World-Sample-AV-Single2MultiView/t2w_model.pt \
    --view_condition_video assets/diffusion/sv2mv_input_view_2.mp4 \
    --num_input_frames 1 \
    --condition_location "first_cam" \
    --prompt "${PROMPT}" \
    --prompt_left "${PROMPT_LEFT}" \
    --prompt_right "${PROMPT_RIGHT}" \
    --prompt_back "${PROMPT_BACK}" \
    --prompt_back_left "${PROMPT_BACK_LEFT}" \
    --prompt_back_right "${PROMPT_BACK_RIGHT}" \
    --video_save_name diffusion-single2multiview-text2world-8gpu
```
#### Example 2: single view extension with initial frames condition
This example runs the front view + initial frames extension into multiview video.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world_view_extend_multiview.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World-Sample-AV-Single2MultiView/v2w_model.pt \
    --view_condition_video assets/diffusion/sv2mv_input_view_2.mp4 \
    --initial_condition_video assets/diffusion/sv2mv_initial_frames.mp4 \
    --num_input_frames 9 \
    --condition_location "first_cam_and_first_n" \
    --prompt "${PROMPT}" \
    --prompt_left "${PROMPT_LEFT}" \
    --prompt_right "${PROMPT_RIGHT}" \
    --prompt_back "${PROMPT_BACK}" \
    --prompt_back_left "${PROMPT_BACK_LEFT}" \
    --prompt_back_right "${PROMPT_BACK_RIGHT}" \
    --video_save_name diffusion-single2multiview-video2world
```

#### Example 3: single view extension with looped generation
This example uses the generation results of example 1 as initial frames input to the video2world model for long video generation.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world_view_extend_multiview.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World-Sample-AV-Single2MultiView/v2w_model.pt \
    --view_condition_video assets/diffusion/sv2mv_input_view.mp4 \
    --initial_condition_video outputs/diffusion-single2multiview-text2world.mp4 \
    --num_input_frames 9 \
    --view_cond_start_frame 48 \
    --condition_location "first_cam_and_first_n" \
    --prompt "${PROMPT}" \
    --prompt_left "${PROMPT_LEFT}" \
    --prompt_right "${PROMPT_RIGHT}" \
    --prompt_back "${PROMPT_BACK}" \
    --prompt_back_left "${PROMPT_BACK_LEFT}" \
    --prompt_back_right "${PROMPT_BACK_RIGHT}" \
    --video_save_name diffusion-single2multiview-video2world-lvg
```
