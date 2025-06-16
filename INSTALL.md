# Setup for Cosmos-Drive-Dreams
We recommend using Conda to set up your environment for running Cosmos-Drive-Dreams. Installation and running of Cosmos-Drive-Dreams have been tested on Ubuntu 20.04.5 and Ubuntu 22.04.5.

## Environment Setup
First, clone the Cosmos-Drive-Dreams source code:
```bash
git clone git@github.com:nv-tlabs/Cosmos-Drive-Dreams.git
cd Cosmos-Drive-Dreams
git submodule update --init --recursive
```

Then, setup the conda environment:
```bash
# Create the cosmos-transfer1 conda environment.
conda env create --file environment.yaml
# Activate the cosmos-transfer1 conda environment.
conda activate cosmos-drive-dreams
# Install the dependencies.
pip install -r requirements.txt
# Install vllm
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
export VLLM_ATTENTION_BACKEND=FLASHINFER
pip install vllm==0.9.0

# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.12
# Install Transformer engine.
pip install transformer-engine[pytorch]==2.4.0
```

## Model Downloading
Cosmos-Transfer1 checkpoints used in Cosmos-Drive-Dreams need to be downloaded manually and a license need to be accepted prior to using them. Please follow these instructions to download the checkpoints locally prior to running our examples. 

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token. Set the access token to 'Read' permission (default is 'Fine-grained').

2. Log in to Hugging Face with the access token:

```bash
huggingface-cli login
```

3. Accept the terms of [Llama-Guard-3-8B terms](https://huggingface.co/meta-llama/Llama-Guard-3-8B), [Cosmos-Tokenize1-CV8x8x8-720p](https://huggingface.co/nvidia/Cosmos-Tokenize1-CV8x8x8-720p), [Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1)

4. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-transfer1-67c9d328196453be6e568d3e):

```bash
cd cosmos-transfer1
PYTHONPATH=$(pwd) python scripts/download_checkpoints.py --output_dir checkpoints/ --model 7b_av
cd ..
```

Note that this will require about 300GB of free storage. 

5. The downloaded files should be in the following structure:

```
checkpoints/
├── nvidia
│   │
│   ├── Cosmos-Guardrail1
│   │   ├── README.md
│   │   ├── blocklist/...
│   │   ├── face_blur_filter/...
│   │   └── video_content_safety_filter/...
│   │
│   ├── Cosmos-Transfer1-7B-Sample-AV/
│   │   ├── base_model.pt
│   │   ├── hdmap_control.pt
│   │   └── lidar_control.pt
│   │
│   ├── Cosmos-Transfer1-7B-Sample-AV-Single2MultiView/
│   │   ├── t2w_base_model.pt
│   │   ├── t2w_hdmap_control.pt
│   │   ├── t2w_lidar_control.pt
│   │   ├── v2w_base_model.pt
│   │   ├── v2w_hdmap_control.pt
│   │   └── v2w_lidar_control.pt
│   │
│   └── Cosmos-Tokenize1-CV8x8x8-720p
│       ├── decoder.jit
│       ├── encoder.jit
│       ├── autoencoder.jit
│       └── mean_std.pt
│
├── depth-anything/...
├── facebook/...
├── google-t5/...
├── IDEA-Research/...
└── meta-llama/...
```