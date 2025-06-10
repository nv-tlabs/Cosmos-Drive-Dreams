# Cosmos-AV-Sample Toolkits
This repo provides toolkits for:

* A rendering script that converts RDS-HQ datasets into input videos (LiDAR and HDMAP) compatible with [**Cosmos-Transfer1-7B-Sample-AV**](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_sample_av.md).

* A conversion script that converts open-source datasets (e.g., Waymo Open Dataset) into RDS-HQ format, so you can reuse the rendering script to render the input videos.

* An interactive visualization tool to visualize the RDS-HQ dataset and generate novel ego trajectories.

* 10 examples (collected by NVIDIA) of input prompts in raw format to help understand how to interface with the model. 

**[[Paper]](https://arxiv.org/abs/2503.14492)**
**[[Model Code]](https://github.com/nvidia-cosmos/cosmos-transfer1)**
**[[Website]](https://research.nvidia.com/labs/dir/cosmos-transfer1/)**

<div align="center">
  <img src="../assets/av_example.gif" alt=""  width="1100" />
</div>

## Quick Start

We provide pre-rendered examples [here](https://huggingface.co/datasets/nvidia/Cosmos-Transfer1-7B-Sample-AV-Data-Example/tree/main/examples) (rendered HDMAP / rendered LiDAR / text prompts). You can download and use these examples to test [**Cosmos-Transfer1-7B-Sample-AV**](https://github.com/nvidia-cosmos/cosmos-transfer1)!


## Download Examples
We provide 10 examples of input prompts with HD map and LiDAR, to help test the model.
1. Add your SSH public key to your [user settings](https://huggingface.co/settings/keys) on Hugging Face.
2. Download the examples from [Hugging Face](https://huggingface.co/datasets/nvidia/Cosmos-Transfer1-7B-Sample-AV-Data-Example) (about 8GB):
```bash
git lfs install
git clone git@hf.co:datasets/nvidia/Cosmos-Transfer1-7B-Sample-AV-Data-Example
```

## Visualize Dataset
You can use `visualize_rds_hq.py` to visualize the RDS-HQ dataset.
```bash
python visualize_rds_hq.py -i <RDS_HQ_FOLDER> -c <CLIP_ID>
```
This python script will launch a [viser](https://github.com/nerfstudio-project/viser) server to visualize the 3D HD map world with dynamic bounding boxes. You can use 
- `w a s d` to control camera's position
- `q e` to control camera's z-axis coordinate

![viser](../assets/viser.png)

> [!NOTE]
> You can run this script in a server with VS Code + remote-ssh plugin. VS Code will automatically forward the port of viser to the local host. 


## Rendering Cosmos Input Control Video
You can use `render_from_rds_hq.py` to render the HD map + bounding box / LiDAR condition videos from RDS-HQ dataset. GPU is required for rendering LiDAR.
```bash
python render_from_rds_hq.py -i <RDS_HQ_FOLDER> -o <OUTPUT_FOLDER> [--skip hdmap] [--skip lidar]
```
This will automatically launch multiple jobs based on [Ray](https://docs.ray.io/en/releases-2.4.0/index.html). If you want to use single process (e.g. for debugging), you can set `USE_RAY=False` in `render_from_rds_hq.py`. You can add `--skip hdmap` or `--skip lidar` to skip the rendering of HD map and LiDAR, respectively. 

**RDS-HQ Rendering Results**
<div align="center">
  <img src="../assets/rds_hq_render.png" alt="RDS-HQ Rendering Results" width="800" />
</div>

> [!NOTE]
> If you're interested, we offer [documentation](../assets/ftheta.pdf) that explains the NVIDIA f-theta camera in detail.


The output folder structure will be like this. Note that `videos` will only be generated when setting `--post_training true`.
```bash
<OUTPUT_FOLDER>
├── hdmap
│   └── {camera_type}_{camera_name}
│       ├── <CLIP_ID>_0.mp4
│       ├── <CLIP_ID>_1.mp4
│       └── ...
│
├── lidar
│   └── {camera_type}_{camera_name}
│       ├── <CLIP_ID>_0.mp4
│       ├── <CLIP_ID>_1.mp4
│       └── ...
│
└── videos
    └── {camera_type}_{camera_name}
        ├── <CLIP_ID>_0.mp4
        ├── <CLIP_ID>_1.mp4
        └── ...
```


## Generate Novel Ego Trajectory
You can also use `visualize_rds_hq.py` to generate novel trajectories.
```bash
python visualize_rds_hq.py -i <RDS_HQ_FOLDER> [-np <NOVEL_POSE_FOLDER_NAME>]
```
Here `<NOVEL_POSE_FOLDER_NAME>` is the folder name for novel pose data. By default, it will be `novel_pose`. 

Using the panel on the right, you record keyframe poses (make sure include the first frame and the last frame), and the script will interpolate all intermediate poses and save them as a `.tar` file in the `novel_pose` folder at `<RDS_HQ_FOLDER>`. Then you can pass `--novel_pose_folder <NOVEL_POSE_FOLDER_NAME>` to the rendering script `render_from_rds_hq.py` to use the novel ego trajectory.
```bash
python render_from_rds_hq.py -i <RDS_HQ_FOLDER> -o <OUTPUT_FOLDER> -np <NOVEL_POSE_FOLDER_NAME>
```

## Convert Public Datasets

We provide a conversion and rendering script for the Waymo Open Dataset as an example of how information from another AV source can interface with the model. 

[Convert Waymo Open Dataset](./docs/convert_public_dataset.md)

**Waymo Rendering Results (use ftheta intrinsics in RDS-HQ)**
![Waymo Rendering Results](../assets/waymo_render_ftheta.png)


**Waymo Rendering Results (use pinhole intrinsics in Waymo Open Dataset)**
![Waymo Rendering Results](../assets/waymo_render_pinhole.png)


**Our Model with Waymo Inputs**
<div align="center">
  <img src="../assets/waymo_example.gif" alt=""  width="1100" />
</div>




## Prompting During Inference
We provide a captioning modification example to help users reproduce our results. To modify the weather in a certain prompt, we use a LLM. Below is an example transformation request:
```bash
Given the prompt:
"The video is captured from a camera mounted on a car. The camera is facing forward. The video depicts a driving scene in an urban environment. The car hood is white. The camera is positioned inside a vehicle, providing a first-person perspective of the road ahead. The street is lined with modern buildings, including a tall skyscraper on the right and a historic-looking building on the left. The road is clear of traffic, with only a few distant vehicles visible in the distance. The weather appears to be clear and sunny, with a blue sky and some clouds. The time of day seems to be daytime, as indicated by the bright sunlight and shadows. The scene is quiet and devoid of pedestrians or other obstacles, suggesting a smooth driving experience."
Modify the environment to:
1. Morning with fog
2. Golden hour with sunlight
3. Heavy snowy day
4. Heavy rainy day
5. Heavy fog in the evening
...
```
You can use the modified text prompts as input to our model.

## Citation
```bibtex
@misc{nvidia2025cosmostransfer1,
  title     = {Cosmos Transfer1: World Generation with Adaptive Multimodal Control},
  author    = {NVIDIA}, 
  year      = {2025},
  url       = {https://arxiv.org/abs/2503.14492}
}
```
