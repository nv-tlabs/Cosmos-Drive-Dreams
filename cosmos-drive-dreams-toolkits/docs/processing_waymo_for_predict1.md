## PostTraining

### Step 1: Create a folder for predict-1 post-training
```bash
mkdir <POST_TRAINING_WAYMO_PREDICT1>
mkdir <POST_TRAINING_WAYMO_PREDICT1>/videos

ln -s <WAYMO_RDS-HQ_FOLDER>/pinhole_front <POST_TRAINING_WAYMO_PREDICT1>/videos/
ln -s <WAYMO_RDS-HQ_FOLDER>/pinhole_front_left <POST_TRAINING_WAYMO_PREDICT1>/videos/
ln -s <WAYMO_RDS-HQ_FOLDER>/pinhole_front_right <POST_TRAINING_WAYMO_PREDICT1>/videos/
ln -s <WAYMO_RDS-HQ_FOLDER>/pinhole_side_left <POST_TRAINING_WAYMO_PREDICT1>/videos/
ln -s <WAYMO_RDS-HQ_FOLDER>/pinhole_side_right <POST_TRAINING_WAYMO_PREDICT1>/videos/
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
python create_t5_embed_mv.py --text_file ./assets/waymo_multiview_texts.json --data_root <POST_TRAINING_WAYMO_PREDICT1> # json stores multi-view caption
```
it will generate `t5_xxl/pinhole_*/*.pkl` embeddings for all 5 views.

Alternatively, to use the single view captions in `assets/waymo_caption.csv`:
```bash
python create_t5_embed.py --caption_file ./assets/waymo_caption.csv --data_root <POST_TRAINING_WAYMO_PREDICT1> # csv stores single-view caption
```
it will only generate `t5_xxl/pinhole_front/*.pkl` embeddings.


The resulting folder structure should look like this:
```
<POST_TRAINING_WAYMO_PREDICT1>/
├── cache/
│   ├── prefix_t5_embeddings_pinhole_front.pkl
│   ├── prefix_t5_embeddings_pinhole_front_left.pkl
│   ├── prefix_t5_embeddings_pinhole_front_right.pkl
│   ├── prefix_t5_embeddings_pinhole_side_left.pkl
│   └── prefix_t5_embeddings_pinhole_side_right.pkl
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
You are now ready to train cosmos-predict1 models on Waymo!
