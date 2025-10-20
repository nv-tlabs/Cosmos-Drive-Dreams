## Post-training autoregressive-based base models

### Model Support Matrix

We support the following Cosmos Autoregressive models for post-training. Review the available models and their compute requirements for post-training and inference to determine the best model for your use case.

| Model Name                               | Model Status | Compute Requirements for Post-Training |
|----------------------------------------------|------------------|------------------------------------------|
| Cosmos-Predict1-4B           | **Supported**    | 1 NVIDIA GPUs*                           |

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
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_autoregressive_checkpoints.py --model_sizes 4B --checkpoint_dir checkpoints
   ```
4. For tensor parallel training, checkpoints need to be sharded to the target tensor model parallel size TP. Shard checkpoints to TP=4 with:
   ```bash
    python scripts/shard_autoregressive_base_checkpoints.py --checkpoint_path checkpoints/Cosmos-Predict1-4B/model.pt --model_size 4b --tensor_parallel_size 4
   ```
   This command will shard and save 4 TP checkpoint shards as `checkpoints/Cosmos-Predict1-4B/model_model_mp_{rank}.pt`

### Examples

Post-training a Cosmos Autoregressive WFM enables you to train the model to generate videos that are more specific to your use case.

There are 2 steps to post-training: downloading a dataset and post-training the model.


#### 1. Download a Dataset
The first step is to download a dataset with videos. You must provide a folder containing a collection of videos in **MP4 format**.

You can use [nvidia/Cosmos-NeMo-Assets](https://huggingface.co/datasets/nvidia/Cosmos-NeMo-Assets) for post-training.

```bash
mkdir -p datasets/cosmos_nemo_assets/

# This command will download the videos for physical AI
huggingface-cli download nvidia/Cosmos-NeMo-Assets --repo-type dataset --local-dir datasets/cosmos_nemo_assets/ --include "*.mp4*"

mv datasets/cosmos_nemo_assets/nemo_diffusion_example_data datasets/cosmos_nemo_assets/videos
```

#### 2. Post-train the model on a single GPU (TP = 1)

Run the following command to execute an example post-training job with the above data which is scaled to a lower resolution by the dataloader to fit on a single GPU

```bash
export OUTPUT_ROOT=checkpoints # default value
torchrun --nproc_per_node=1 -m cosmos_predict1.autoregressive.train --config=cosmos_predict1/autoregressive/configs/config.py -- experiment=base_4b_example_tealrobotsmall_tp1
```

The model will be post-trained using the above cosmos_nemo_assets dataset.
See the `VideoDataset` defined in `cosmos_predict1/autoregressive/datasets/video_dataset.py` and `register_training_data`  in `cosmos_predict1/autoregressive/configs/registry.py` to understand how the dataloader works and is registered.

The checkpoints will be saved to `${OUTPUT_ROOT}/PROJECT/GROUP/NAME`. In the above example,
`PROJECT` is `posttraining`, `GROUP` is `autoregressive_base`, `NAME` is `base_4b_example_tealrobotsmall_tp1`.

See the job config to understand how they are determined.
```python
base_4b_example_tealrobotsmall_tp1= LazyDict(
    dict(
        ...
        job=dict(
            project="posttraining",
            group="autoregressive_base",
            name="base_4b_example_tealrobotsmall_tp1",
        ),
        ...
    )
)
```

During the training, the checkpoints will be saved in the below structure.
```
checkpoints/posttraining/autoregressive_base/base_4b_example_tealrobotsmall_tp1/checkpoints/
├── iter_{NUMBER}.pt
```

#### 3. Post-train the model on 4 GPUs (TP = 4)

The model can also be post-trained on multiple GPUs using tensor parallelism. Run the following command to execute an example post-training job with the above data with higher resolution.

```bash
export OUTPUT_ROOT=checkpoints # default value
torchrun --nproc_per_node=4 -m cosmos_predict1.autoregressive.train --config=cosmos_predict1/autoregressive/configs/config.py -- experiment=base_4b_example_tealrobot_tp4
```

The checkpoints will be saved to `${OUTPUT_ROOT}/PROJECT/GROUP/NAME`. In the above example,
`PROJECT` is `posttraining`, `GROUP` is `autoregressive_base`, `NAME` is `base_4b_example_tealrobot_tp4`.

See the job config to understand how they are determined.
```python
base_4b_example_tealrobotsmall_tp4= LazyDict(
    dict(
        ...
        job=dict(
            project="posttraining",
            group="autoregressive_base",
            name="base_4b_example_tealrobotsmall_tp4",
        ),
        ...
    )
)
```

During the training, the sharded checkpoints will be saved in the below structure.
```
checkpoints/posttraining/autoregressive_base/base_4b_example_tealrobot_tp4/checkpoints/
├── iter_{NUMBER}.pt
├── iter_{NUMBER}_model_mp_0.pt
├── iter_{NUMBER}_model_mp_1.pt
├── iter_{NUMBER}_model_mp_2.pt
├── iter_{NUMBER}_model_mp_3.pt
```



### Inference with the Post-trained Model Checkpoint

The inference can be done with the same interface as described in [examples/inference_autoregressive_base.md](/examples/inference_autoregressive_base.md).

#### 1. Copying checkpoint to Designated Location

The post-trained checkpoint needs to be copied to `checkpoints/Cosmos-Predict1-4B-Base_post-trained/model.pt`

For example, with TP=1 if a posttrained checkpoint with 1000 iterations is to be used,
```bash
# copy checkpoint to the designated location
mkdir checkpoints/Cosmos-Predict1-4B-Base_post-trained/
cp checkpoints/posttraining/autoregressive_base/base_4b_example_tealrobotsmall_tp1/checkpoints/iter_000001000.pt checkpoints/Cosmos-Predict1-4B-Base_post-trained/model.pt
```

With TP=4, the postrained checkpoints are sharded and should first be merged into a single checkpoint for inference

```bash
# merge tensor parallel model shards
mkdir checkpoints/Cosmos-Predict1-4B-Base_post-trained/
python scripts/merge_autoregressive_tp_checkpoints.py --checkpoint_path checkpoints/posttraining/autoregressive_base/base_4b_example_tealrobot_tp4/checkpoints/iter_000001000.pt --output_path checkpoints/Cosmos-Predict1-4B-Base_post-trained/model.pt --model_size 4b --tensor_parallel_size 4
```

#### 2. Running the Inference

This is the basic example for running inference on the post-trained 4B model with a single video.

```bash
NUM_GPUS=<NUM_GPUS>
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict1/autoregressive/inference/base.py \
    --num_gpus ${NUM_GPUS} \
    --checkpoint_dir checkpoints \
    --ar_model_dir Cosmos-Predict1-4B-Base_post-trained \
    --input_type video \
    --input_image_or_video_path datasets/cosmos_nemo_assets/videos/output_oige_render_view_sub.mp4  \
    --top_p 0.8 \
    --temperature 1.0 \
    --offload_diffusion_decoder \
    --offload_tokenizer \
    --video_save_name autoregressive-4b-post-train
```
