## Post-training diffusion-based Video2World models (with action control)

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

For example, you can use bridge dataset for post-training.

```bash
# Download metadata with video urls and captions
mkdir -p datasets/bridge
cd datasets/bridge
wget https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/bridge_train_data.tar.gz

# Unpack and clean up structure.
tar -xf bridge_train_data.tar.gz
mv opensource_robotdata/bridge/* .
rm -rf opensource_robotdata/
```


#### 2. Preprocessing the Data

Action control does not require T5-XXL embeddings.
No preprocessing is necessary.

Dataset folder format:
```
datasets/bridge/
├── annotation/
│   ├── train/
│   │   ├── *.json
│   ├── val/
│   │   ├── *.json
├── videos/
│   ├── train/
│   │   ├── 0/rgb.mp4
│   ├── val/
│   │   ├── 0/rgb.mp4
```


#### 3. Post-train the Model

Run the following command to execute an example post-training job with the above data.
```bash
export OUTPUT_ROOT=checkpoints # default value
torchrun --nproc_per_node=8 -m cosmos_predict1.diffusion.training.train --config=cosmos_predict1/diffusion/training/config/config.py -- experiment=video2world_action_bridge_2frames
```

The model will be post-trained using the above bridge dataset.
See the config `video2world_action_bridge_2frames` defined in `cosmos_predict1/diffusion/training/config/video2world_action/experiment.py` to understand how the dataloader is determined.

```python
bridge_train_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=2,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="train",
    load_action=True,
    load_t5_embeddings=False,
)

dataloader_train = L(DataLoader)(
    dataset=bridge_train_dataset,
    sampler=L(get_sampler)(dataset=bridge_train_dataset),
    batch_size=8,
    drop_last=True,
)
...

video2world_action_bridge_2frames = LazyDict(
    dict(
        ...
        dataloader_train=dataloader_train,
        ...
    )
)
...
```

The checkpoints will be saved to `${OUTPUT_ROOT}/PROJECT/GROUP/NAME`.
In the above example, `PROJECT` is `posttraining`, `GROUP` is `diffusion_video2world_action`, `NAME` is `video2world_action_bridge_2frames`.

See the job config to understand how they are determined.
```python
video2world_action_bridge_2frames = LazyDict(
    dict(
        ...
        job=dict(
            project="posttraining",
            group="diffusion_video2world_action",
            name="video2world_action_bridge_2frames",
        ),
        ...
    )
)
```

During the training, the checkpoints will be saved in the below structure.
```
checkpoints/posttraining/diffusion_video2world/video2world_7b_example_hdvila/checkpoints/
├── iter_{NUMBER}_reg_model.pt
├── iter_{NUMBER}_ema_model.pt
```


#### 4. Inference with the Post-trained Model Checkpoint

The inference can be done with the same interface as described in [examples/inference_diffusion_video2world.md](examples/inference_diffusion_video2world.md).

1. Copying checkpoint to Designated Location

The post-trained checkpoint needs to be copied to `checkpoints/Cosmos-Predict1-7B-Video2World_action_post-trained/model.pt`

For example, if a posttrained checkpoint (ema) with 1000 iterations is to be used,
```bash
# copy checkpoint to the designated location
mkdir checkpoints/Cosmos-Predict1-7B-Video2World_action_post-trained/
cp checkpoints/posttraining/diffusion_video2world_action/video2world_action_bridge_2frames/checkpoints/iter_000002000_ema_model.pt checkpoints/Cosmos-Predict1-7B-Video2World_action_post-trained/model.pt
```
2. Running the Inference

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world_action.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World_action_post-trained \
    --seed 0 \
    --input_image_or_video_path datasets/bridge/videos/test/346/rgb.mp4 \ --action_annotation_path datasets/bridge/annotation/test/346.json \
    --height 256 \
    --width 320 \
    --fps 3
```
