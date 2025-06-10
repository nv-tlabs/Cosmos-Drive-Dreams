## PostTraining
Make sure you have Waymo Open Dataset converted to RDS-HQ format.

### Step 1: Render HD map + bounding box / LiDAR condition video from RDS-HQ format
For single view post-training, run the following command:
```bash
python render_from_rds_hq.py -d waymo -i <WAYMO_RDS-HQ_FOLDER> -o <POST_TRAINING_WAYMO_TRANSFER1> -c pinhole -p True 
```
Or, for multiview training, run:
```bash
python render_from_rds_hq.py -d waymo_mv_short -i <WAYMO_RDS-HQ_FOLDER> -o <POST_TRAINING_WAYMO_TRANSFER1> -c pinhole -p True 
```

### Step 2: Create T5 Text Embeddings
Lastly, we need to create T5 text embeddings. 
Make sure you have completed Cosmos-predict1 installation and use the cosmos-predict1 environment for this step:
```bash
conda activate cosmos-predict1
```
We offer two set of captions, a more complete set of single view captions in `assets/waymo_caption.csv`, and a set of 5k multiview captions in `assets/waymo_multiview_texts.json`.
To use the 5k multiview captions in `assets/waymo_multiview_texts.json`:
```bash
python create_t5_embed_mv.py --text_file ./assets/waymo_multiview_texts.json --data_root <POST_TRAINING_WAYMO_TRANSFER1> # json stores multi-view caption
```
it will generate `t5_xxl/pinhole_*/*.pkl` embeddings for all 5 views.

Alternatively, to use the single view captions in `assets/waymo_caption.csv`:
```bash
python create_t5_embed.py --caption_file ./assets/waymo_caption.csv --data_root <POST_TRAINING_WAYMO_TRANSFER1> # csv stores single-view caption
```
it will only generate `t5_xxl/pinhole_front/*.pkl` embeddings.

The resulting folder structure should look like this if you are doing multiview training, or with only the `pinhole_front/` sub-folders if doing only front view training:
```bash
<POST_TRAINING_WAYMO_TRANSFER1>
├── cache/
│   ├── prefix_pinhole_front.pkl
│   ├── prefix_pinhole_front_left.pkl
│   ├── prefix_pinhole_front_right.pkl
│   ├── prefix_pinhole_side_left.pkl
│   └── prefix_pinhole_side_right.pkl
│
├── videos/
│   ├── pinhole_front/
│   │   └── {clip_id}_0.mp4
│   │   └── {clip_id}_1.mp4
│   │   └── ....mp4
│   │
│   ├── pinhole_left/
│   │   └── {clip_id}_0.mp4
│   │   └── {clip_id}_1.mp4
│   │   └── ....mp4
│   *
│
├── hdmap/
│   ├── pinhole_front/
│   │   └── {clip_id}_0.mp4
│   │   └── {clip_id}_1.mp4
│   │   └── ....mp4
│   │
│   ├── pinhole_left/
│   │   └── {clip_id}_0.mp4
│   │   └── {clip_id}_1.mp4
│   │   └── ....mp4
│   *
│   
├── lidar/
│   ├── pinhole_front/
│   │   └── {clip_id}_0.mp4
│   │   └── {clip_id}_1.mp4
│   │   └── ....mp4
│   │
│   ├── pinhole_left/
│   │   └── {clip_id}_0.mp4
│   │   └── {clip_id}_1.mp4
│   │   └── ....mp4
│   *
│
└── t5_xxl/
    ├── pinhole_front
    │   └── {clip_id}.pkl
    *
```
You are now ready to train cosmos-transfer models on Waymo!