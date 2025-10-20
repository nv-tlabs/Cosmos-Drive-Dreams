## Post-training tokenizer models

### Model Support Matrix

Cosmos-Tokenize1 provides tokenizers for both images and videos. For each type, we offer tokenizers that operate in both continuous and discrete latent spaces. Refer to the table below for a list of supported post-training scripts for video tokenizers.

| Model Name                               | Model Status | Compute Requirements for Post-Training |
|----------------------------------------------|------------------|------------------------------------------|
| Cosmos-Tokenize1-CV8x8x8-720p           | **Supported**    | 8 NVIDIA GPUs*                           |
| Cosmos-Tokenize1-DV8x16x16-720p         | **Supported**    | 8 NVIDIA GPUs*                           |
| Cosmos-Tokenize1-CV4x8x8-360p           | **Supported**    | 8 NVIDIA GPUs*                           |
| Cosmos-Tokenize1-DV4x8x8-360p           | **Supported**    | 8 NVIDIA GPUs*                           |


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
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python3 -m scripts.download_tokenizer_checkpoints --tokenizer_types CV8x8x8-720p DV8x16x16-720p CV4x8x8-360p DV4x8x8-360p --checkpoint_dir checkpoints
   ```

The downloaded files should be in the following structure:
```
checkpoints/
├── Cosmos-Tokenize1-CV8x8x8-720p
│   ├── config.json
│   ├── encoder.jit
│   ├── decoder.jit
│   ├── autoencoder.jit
│   └── model.pt
├── Cosmos-Tokenize1-DV8x16x16-720p
│   ├── config.json
│   ├── encoder.jit
│   ├── decoder.jit
│   ├── autoencoder.jit
│   └── model.pt
├── Cosmos-Tokenize1-CV4x8x8-360p
│   ├── config.json
│   ├── encoder.jit
│   ├── decoder.jit
│   ├── autoencoder.jit
│   └── model.pt
└── Cosmos-Tokenize1-DV4x8x8-360p
│   ├── config.json
│   ├── encoder.jit
│   ├── decoder.jit
│   ├── autoencoder.jit
│   └── model.pt
```

### Examples

Post-training a Cosmos Tokenizer allows you to fine-tune the model for your specific use cases.

There are 3 steps to post-training: downloading a dataset, preprocessing the data, and post-training the tokenizer model.

#### 1. Download a Dataset

The first step is to download a dataset with videos.

You must provide a folder containing a collection of videos in **MP4 format**, preferably 720p. These videos should focus be diverse enough to capture different scenarios.

For example, you can use a subset of [HD-VILA-100M](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m) dataset for post-training.

```bash
# Download metadata with video urls
mkdir -p datasets/hdvila
cd datasets/hdvila
wget https://huggingface.co/datasets/TempoFunk/hdvila-100M/resolve/main/hdvila-100M.jsonl
```

Run the following command to download the sample videos used for post-training:

```bash
# Requirements for Youtube video downloads & video clipping
pip install pytubefix ffmpeg
```

```bash
# The script will downlaod the original HD-VILA-100M videos, save the corresponding clips and the metadata.
python3 -m scripts.download_tokenizer_example_data --dataset_path datasets/hdvila --N_videos 128 --do_download --do_clip
```

The downloaded files should be in the following structure:
```
datasets/hdvila/
├── metas/
│   ├── *.json
└── videos/
    └── *.mp4
```

Finally, register the glob pattern to the mp4 files at [dataset_provider.py](cosmos_predict1/tokenizer/training/datasets/dataset_provider.py), as show below.
```python
_VIDEO_PATTERN_DICT = {
    "hdvila_video": "datasets/hdvila/videos/*mp4",
}
```

**Note**: As will be shown below, different resolution variants of the `hdvila_video` can be obtained by simply passing `hdvila_video<resolution>`. For instance, in the following examples, we use `hdvila_video360` and `hdvila_video720` to refer to the same hdvila videos but resized to the resolution 360p and 720p, respectively, at the time of training.



#### 2. Post-train the Video Tokenizers

Run the following command to execute a post-training job with the above data for `Cosmos-Tokenize1-CV8x8x8-720p`.
```bash
export OUTPUT_ROOT=checkpoints # default value
torchrun --nproc_per_node=8 -m cosmos_predict1.tokenizer.training.train \
    --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
    experiment=Cosmos_Tokenize1_CV8x8x8_720p_HDVILA
```

The tokenizer will be post-trained using the above hdvila dataset.
See the config `Cosmos_Tokenize1_CV8x8x8_720p_HDVILA` defined in `cosmos_predict1/tokenizer/training/configs/experiments/cosmos_tokenize1.py` to understand how the dataloader is determined.
```python
Cosmos_Tokenize1_CV8x8x8_720p_HDVILA: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/video_basic",
            {"override /network": "continuous_factorized_video"},
            {"override /data_train": "hdvila_video720"}, # hdvila_video resized to 720p at the time of training
            {"override /data_val": "hdvila_video720"}, # hdvila_video resized to 720p at the time of training
            "_self_",
        ],
        dataloader_train=dict(
            dataset=dict(
                crop_height=256,
                num_video_frames=121,
            ),
            batch_size=1,
        ),
        dataloader_val=dict(
            dataset=dict(
                crop_height=256,
                num_video_frames=121,
            ),
            batch_size=1,
        ),
        model=dict(
            config=dict(
                network=dict(
                    channels_mult=[2, 4, 4],
                    patch_size=4,
                    legacy_mode=False,
                    temporal_compression=8,
                    spatial_compression=8,
                )
            )
        ),
        job=dict(
            project="posttraining",
            group="tokenizer",
            name="Cosmos-Tokenize1-CV8x8x8-720p-HDVILA",
        ),
        checkpoint=dict(
            load_path="checkpoints/Cosmos-Tokenize1-CV8x8x8-720p/model.pt",
            strict_resume=True,
            load_training_state=True,
            jit=dict(input_shape=[1, 3, 17, 512, 512]),
        ),
    )
)
```

The checkpoints will be saved to `${OUTPUT_ROOT}/PROJECT/GROUP/NAME`.
In the above example, `PROJECT` is `posttraining`, `GROUP` is `tokenizer`, `NAME` is `Cosmos-Tokenize1-CV8x8x8-720p-HDVILA`. See the job config above to understand how they are specified.

During the training, the checkpoints will be saved in the below structure.
```bash
checkpoints/posttraining/tokenizer/Cosmos-Tokenize1-CV8x8x8-720p-HDVILA/checkpoints/
├── iter_{NUMBER}.pt
├── iter_{NUMBER}_enc.jit
├── iter_{NUMBER}_dec.jit
├── iter_{NUMBER}_ema.jit
```

### Inference with the Post-trained Model Checkpoint

The inference can be done with the same interface as described in [examples/inference_tokenizer.md](/examples/inference_tokenizer.md).

```bash
# Autoencoding videos using post-trained `Cosmos-Tokenize1-CV8x8x8-720p-HDVILA`.
model_name="Cosmos-Tokenize1-CV8x8x8-720p-HDVILA"
python3 -m cosmos_predict1.tokenizer.inference.video_cli \
    --video_pattern 'cosmos_predict1/tokenizer/test_data/*.mp4' \
    --checkpoint_enc checkpoints/posttraining/tokenizer/${model_name}/checkpoints/iter_${NUMBER}_enc.jit \
    --checkpoint_dec checkpoints/posttraining/tokenizer/${model_name}/checkpoints/iter_${NUMBER}_dec.jit
```
