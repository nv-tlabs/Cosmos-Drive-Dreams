## Post-training diffusion-based SingleToMultiView-Sample-AV/ Models

### Model Support Matrix

We support the following Cosmos SingleToMultiView-Sample-AV models for post-training. Review the available models and their compute requirements for post-tuning and inference to determine the best model for your use case.

| Model Name                                               | Model Status   | Compute Requirements for Post-Training   |
|----------------------------------------------------------|----------------|------------------------------------------|
| Cosmos-Predict1-7B-Video2World-Sample-AV-Single2MultiView/t2w_model | **Supported**  | 8 NVIDIA GPUs*                           |
| Cosmos-Predict1-7B-Video2World-Sample-AV-Single2MultiView/v2w_model | **Supported**  | 8 NVIDIA GPUs*                           |

**\*** `H100-80GB` or `A100-80GB` GPUs are recommended.

### How we trained Cosmos-Predict1-7B-Video2World-Sample-AV-Single2MultiView

The Cosmos-Predict1-7B-Video2World-Sample-AV-Single2MultiView model was post-trained with the following specifications:

#### Dataset Used:

* 300,000 multi-view video clips from internal RDS dataset
* Each clip was 5 seconds long
* Total of approximately 400 hours of driving data
* The number of views per clip matched the intended inference use case (6 views in this case)
* Dataset size was sufficient to provide novel data during the first 20,000 iterations
* Training data included diverse scenarios to ensure good generalization

#### Training Configuration:

* Batch size: 32
* Total iterations: 20,000-40,000
* Initial performance improvements observed after 10,000 iterations
* Training continued until model convergence

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
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B --model_types SingleToMultiView-Sample-AV --checkpoint_dir checkpoints
   ```

### Examples

Post-training a Cosmos Diffusion-based WFM enables you to train the model to generate videos that are more specific to your use case.

#### 1. Data Preparation
The first step is to download a dataset with videos and captions and then preprocess it to our required format.

Example 1. You can use [Waymo Open Dataset](https://waymo.com/open/) for post-training.
Please follow the [instruction](https://github.com/nv-tlabs/cosmos-av-sample-toolkits/blob/main/docs/processing_waymo_for_predict1.md) in [cosmos-av-sample-toolkits](https://github.com/nv-tlabs/cosmos-av-sample-toolkits) to download and convert the Waymo Open Dataset.

The resulting folder structure should look like this:
```
<DATA_ROOT>/waymo/
├── cache/
│   ├── prefix_t5_embeddings_pinhole_front.pkl
│   ├── prefix_t5_embeddings_pinhole_front_left.pkl
│   ├── ... (embeddings for each view)
├── videos/
│   ├── pinhole_front
│       ├── *.mp4
│   ├── pinhole_front_left
│   ├── pinhole_front_right
│   ├── pinhole_side_left
│   ├── pinhole_side_right
│   ...
└── t5_xxl/
    ├── pinhole_front
        └── *.pkl
```

If you've used the multiview caption embedding option above, set `load_mv_emb` to True in `cosmos_predict1/diffusion/training/config/text2world_singletomultiview/experiment.py`

* While here we use Waymo dataset, any RGB multi-view video dataset can be used as long as it's organized in the above structure.

#### 2. Post-train the Model

Run the following command to execute an example post-training job with the above data.
```bash
export OUTPUT_ROOT=checkpoints # default value
torchrun --nproc_per_node=8 -m cosmos_predict1.diffusion.training.train \
    --config=cosmos_predict1/diffusion/training/config/config_multiview.py \
    -- experiment=text2world_singletomultiview_7b_example_waymo
```

Optionally, multi-node training can be done with
```bash
# 4-node training example.
torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 \
    -m cosmos_predict1.diffusion.training.train \
    --config=cosmos_predict1/diffusion/training/config/config_multiview.py \
    -- experiment=text2world_singletomultiview_7b_example_waymo
```

The model will be post-trained using the above waymo dataset.
See the config `text2world_singletomultiview_7b_example_waymo` defined in `cosmos_predict1/diffusion/training/config/text2world_singletomultiview/experiment.py` to understand how the dataloader is determined.
```python
train_num_views = 3
num_frames = 57
view_keys = ["pinhole_front_left", "pinhole_front", "pinhole_front_right", "pinhole_side_left", "pinhole_side_right"]
example_multiview_dataset_waymo = L(Dataset)(
    dataset_dir="cosmos-av-sample-toolkits/waymo",
    sequence_interval=1,
    num_frames=num_frames,
    view_keys=view_keys,
    video_size=(576, 1024),
    sample_n_views=train_num_views,
    caption_view_idx_map={0:0,1:1,2:2,3:4,4:5},
)

```

The checkpoints will be saved to `${OUTPUT_ROOT}/PROJECT/GROUP/NAME`.
In the above example, `PROJECT` is `posttraining`, `GROUP` is `diffusion_text2world`, `NAME` is `text2world_singletomultiview_7b_example_waymo`.

See the job config to understand how they are determined.
```python
text2world_singletomultiview_7b_example_waymo = LazyDict(
    dict(
        ...
        job=dict(
            project="posttraining",
            group="diffusion_text2world",
            name="text2world_singletomultiview_7b_example_waymo",
        ),
        ...
    )
)
```

During the training, the checkpoints will be saved in the below structure.
```
checkpoints/posttraining/diffusion_text2world/text2world_singletomultiview_7b_example_waymo/checkpoints/
├── iter_{NUMBER}_reg_model.pt
├── iter_{NUMBER}_ema_model.pt
```
#### Inference with the Post-trained Model Checkpoint
Inference can be done with the same interface as described in [examples/inference_diffusion_single2multiview.md](https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/examples/inference_diffusion_single2multiview.md).

1. **Copying checkpoint to Designated Location**
The post-trained checkpoint needs to be copied to checkpoints/Cosmos-Predict1-7B-SingleToMultiView_post-trained/model.pt.
For example, if a post-trained checkpoint (ema) with 1000 iterations is to be used,
```commandline
# copy checkpoint to the designated location
mkdir checkpoints/Cosmos-Predict1-7B-SingleToMultiView_post-trained/
cp checkpoints/posttraining/diffusion_text2world/text2world_singletomultiview_7b_example_waymo/checkpoints/iter_000001000_ema_model.pt checkpoints/Cosmos-Predict1-7B-SingleToMultiView_post-trained/model.pt
```
2. **Running Inference** We will set the prompt with an environment variable first.
```commandline
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
See the config `Cosmos_Predict1_Video2World_7B_ViewExtend_Multiview` defined in `cosmos_predict1/diffusion/config/inference/cosmos-1-diffusion-video2world-view_extend-multiview.py` and make sure it's consistent with training config.
```commandline
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world_view_extend_multiview.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-SingleToMultiView_post-trained \
    --view_condition_video assets/diffusion/sv2mv_input_view.mp4 \
    --num_input_frames 1 \
    --condition_location "first_cam" \
    --prompt "${PROMPT}" \
    --prompt_left "${PROMPT_LEFT}" \
    --prompt_right "${PROMPT_RIGHT}" \
    --prompt_back "${PROMPT_BACK}" \
    --prompt_back_left "${PROMPT_BACK_LEFT}" \
    --prompt_back_right "${PROMPT_BACK_RIGHT}" \
    --video_save_name diffusion-single2multiview-text2world-posttrained
```
