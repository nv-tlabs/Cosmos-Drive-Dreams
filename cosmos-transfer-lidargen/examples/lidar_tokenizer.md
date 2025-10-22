## Post-training and Inference with Lidar Tokenizer Models

### Environment setup

Please refer to the Inference section of [INSTALL.md](/INSTALL.md#inference) for instructions on environment setup. Note that in addition to the post-training requirements, you'll also need to install the following libraries:

```bash
# required for plotly
apt-get -y install libnss3 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2
plotly_get_chrome
# ncore dependency, this is for Nvidia internal only
export YOUR_TOKEN=xxxx
pip install ncore --extra-index-url https://__token__:${YOUR_TOKEN}@gitlab-master.nvidia.com/api/v4/projects/61004/packages/pypi/simple
# additional libs
pip install jaxtyping kaleido pyquaternion av lru-dict OpenEXR==3.2.3 plotly open3d
```

### Model Support Matrix

Currently, we support continuous latent space tokenizers for lidar data.

| Model Name                               | Model Status | Compute Requirements for Post-Training |
|----------------------------------------------|------------------|------------------------------------------|
| Cosmos-Tokenizer-CI8x8-Lidar      | **Supported**    | NVIDIA GPUs*                           |

**\*** `H100-80GB` or `A100-80GB` GPUs are recommended.

We conducted evaluation of the fine-tuned lidar tokenizer on 100 randomly sampled lidar clips to assess reconstruction quality. The evaluation metrics include Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and Relative Error, which provide insights into the model's ability to accurately reconstruct lidar data.

| Model Name                          | RMSE (m)  | MAE (m)   | Relative Error |
|-------------------------------------|-----------|-----------|----------------|
| Cosmos-Tokenize1-CI8x8-360p (Image) | 1.302     | 0.450     | 0.022          |
| Cosmos-Tokenizer-CI8x8-Lidar | 0.289     | 0.218     | 0.011          |

The results demonstrate significant improvements in reconstruction accuracy after fine-tuning on lidar range map data. 


### Download checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```
3. Download the checkpoints
    ```
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="nvidia/Cosmos-Tokenizer-CI8x8-Lidar",local_dir="checkpoints/Cosmos-Tokenizer-CI8x8-Lidar")
    ```

The downloaded files should be in the following structure:
```
checkpoints/
├── Cosmos-Tokenizer-CI8x8-Lidar
│   ├── encoder.jit
│   ├── decoder.jit
│   └── mean_std.pt
```

## Post-training Lidar Tokenizers

Post-training a Cosmos Lidar Tokenizer allows you to fine-tune the model for your specific lidar point cloud use cases.

### Dataset Preparation

You must provide lidar point cloud data in a compatible format, please check [cosmos-lidar-preprocessing-release
](https://gitlab-master.nvidia.com/shengyuh/cosmos-lidar-preprocessing-release) for lidar preprocessing.

For example, you can use lidar datasets with the following structure:
```
datasets/lidar_dataset_release/
├── metadata/
│   └── *.npz
└── lidar/
    └── *.tar
```

Each tar file should contain point cloud data that can be processed by the lidar tokenizer.

### Post-training Configuration

The lidar tokenizer uses a custom experiment configuration. See the config [cosmos_lidar_tokenizer](../cosmos_predict1/tokenizer/training/configs/experiments/cosmos_lidar_tokenizer.py) defined in the training configuration files to understand how the dataloader and model are configured. Note that we post-train the tokenizer using ```fp32``` precision. 

### Post-training Command

Run the following command to execute a post-training job for the lidar tokenizer:

```bash
export OUTPUT_ROOT=checkpoints # default value
N_GPUS=4
torchrun --nproc_per_node=$N_GPUS -m cosmos_predict1.tokenizer.training.train \
    --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
    experiment=cosmos_lidar_tokenizer
```

The checkpoints will be saved to `${OUTPUT_ROOT}/PROJECT/GROUP/NAME`.
During the training, the checkpoints will be saved in the following structure:
```bash
checkpoints/posttraining/tokenizer/Cosmos-Tokenize1-CI8x8-360p-LIDAR/checkpoints/
├── iter_{NUMBER}.pt
├── iter_{NUMBER}_enc.jit
├── iter_{NUMBER}_dec.jit
├── iter_{NUMBER}_ema.jit
```

## Inference with Lidar Tokenizer Models

### Autoencoding Lidar Point Clouds

The lidar tokenizer can encode and decode point cloud data, providing reconstructions of the input point clouds.

```bash
# Autoencoding lidar point clouds using a post-trained model
sample_path=datasets/lidar_dataset_release/lidar/14d2ed36-1afa-11ed-88ea-00044bf65d5c_1660389768178294_1660389788178294.tar
python -m cosmos_predict1.tokenizer.inference.lidar_cli \
    --sample_path=$sample_path \
    --enc_path=checkpoints/Cosmos-Tokenizer-CI8x8-Lidar/encoder.jit \
    --dec_path=checkpoints/Cosmos-Tokenizer-CI8x8-Lidar/decoder.jit \
    --output_folder=dump_results/reconstructions \
    --tokenizer_dtype=float32 \
    --vis_pcd=1
```

Command Line Arguments

- `--sample_path`: Path to the lidar sample tar file
- `--enc_path`: Path to the encoder checkpoint (.jit file)
- `--dec_path`: Path to the decoder checkpoint (.jit file)
- `--output_folder`: Output directory for reconstructions
- `--tokenizer_dtype`: Data type for tokenizer (e.g., float32, float16)
- `--vis_pcd`: Enable point cloud visualization (1 for enabled, 0 for disabled)

### Output

The inference script will generate reconstructed point clouds in the specified output directory. When `--vis_pcd=1` is enabled, it will also provide visualizations of the original and reconstructed point clouds.

#### Range Map Visualization
<img src="../assets/lidar/rangemap_view.jpg" alt="Rangemap" style="max-width: 800px;">

#### Point Cloud Visualization
<img src="../assets/lidar/pointcloud_view.jpg" alt="Pointcloud" style="max-width: 800px;">


## Model Architecture

The lidar tokenizer is based on the Cosmos-Tokenize1-CI8x8-360p architecture, which provides:

- **Continuous latent space**: The model operates in a continuous latent space for smooth point cloud representations
- **Spatial compression**: 8x8 spatial compression factor for efficient encoding
- **Point cloud compatibility**: Specialized for processing 3D point cloud data


## Troubleshooting

- **Memory issues**: If you encounter memory issues, try reducing the batch size or using fewer GPUs
- **Data format**: Ensure your lidar data is in the correct tar format
- **Checkpoint compatibility**: Make sure you're using compatible encoder and decoder checkpoints from the same training run
