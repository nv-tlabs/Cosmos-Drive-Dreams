import os
import click
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from transformers import T5EncoderModel, T5TokenizerFast
import numpy as np
from copy import deepcopy
import torch

# Initialize T5 model , please modify your t5 path here
t5_path = "/mnt/scratch/cache/imageinaire/google-t5/t5-11b"
tokenizer = T5TokenizerFast.from_pretrained(t5_path)
encoder = T5EncoderModel.from_pretrained(t5_path)
encoder.to("cuda")
encoder.eval()
encoder.requires_grad_(False)

# Camera view list
CAMERA_VIEWS = ["pinhole_front", "pinhole_front_left", "pinhole_front_right", "pinhole_side_left", "pinhole_side_right"]

# Prompt prefixes for each view
PREFIX_PROMPTS = {
    "pinhole_front": "The video is captured from a camera mounted on a car. The camera is facing forward.",
    "pinhole_front_left": "The video is captured from a camera mounted on a car. The camera is facing forward and slightly to the left.",
    "pinhole_front_right": "The video is captured from a camera mounted on a car. The camera is facing forward and slightly to the right.",
    "pinhole_side_left": "The video is captured from a camera mounted on a car. The camera is facing to the left.",
    "pinhole_side_right": "The video is captured from a camera mounted on a car. The camera is facing to the right.",
}

def _encode_text(text: str):
    """Encode a single text into T5 embeddings
    
    Args:
        text (str): Input text to be encoded
        
    Returns:
        numpy.ndarray: T5 embeddings of shape (seq_len, hidden_size)
        
    Note:
        - If text length exceeds 512 tokens, it will be truncated
        - A warning will be printed if truncation occurs
    """
    # Check text length
    tokens = tokenizer.tokenize(text)
    if len(tokens) > 512:
        print(f"Warning: Text length ({len(tokens)} tokens) exceeds 512 tokens and will be truncated")
    
    # Encode text with max_length=512
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = encoder(**inputs)
    return outputs.last_hidden_state.cpu().numpy()

def save_prefix_embeddings(cache_dir: Path):
    """Generate and save T5 embeddings for prefix prompts"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for view, prompt in PREFIX_PROMPTS.items():
        # Generate embedding
        embedding = _encode_text(prompt)
        
        # Save to file
        save_path = cache_dir / f"prefix_{view}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(embedding, f)
        print(f"Saved prefix embedding for {view} to {save_path}")

def _get_video_path(data_root: Path, view: str, key: str) -> Path:
    """Get video path by matching key with video filename, ignoring clip id suffix
    
    Args:
        data_root (Path): Root directory of the dataset
        view (str): Camera view name
        key (str): Key from CSV file
        
    Returns:
        Path: Path to matching video file, or None if not found
    """
    videos_dir = data_root / "videos" / view
    if not videos_dir.exists():
        return None
        
    # List all video files
    video_files = list(videos_dir.glob("*.mp4"))
    
    # Find matching video file (ignoring clip id suffix)
    for video_file in video_files:
        # Remove .mp4 extension and clip id suffix
        video_name = video_file.stem
        if video_name.endswith("_0"):
            base_name = "_".join(video_name.split("_")[:-1])
        else:
            base_name = video_name
        
        if base_name == key:
            return video_file
            
    return None

def _get_clip_id(video_path: Path) -> str:
    """Extract clip_id from video filename
    
    Args:
        video_path (Path): Path to the video file
        
    Returns:
        str: clip_id (e.g. "_0")
    """
    if video_path is None:
        return ""
    return "_" + video_path.stem.split("_")[-1]

@click.command()
@click.option("--text_file", type=str, default="assets/waymo_multiview_texts.json", help="Path to JSON file containing text descriptions")
@click.option("--data_root", type=str, default="waymo_mv", help="Path to data root directory")
def main(text_file: str, data_root: str):
    """Main function: Process text and generate T5 embeddings"""
    data_root = Path(data_root)
    
    # Generate and save prefix embeddings
    cache_dir = data_root / "cache"
    save_prefix_embeddings(cache_dir)
    
    # Read JSON file
    with open(text_file, 'r') as f:
        text_data = json.load(f)
    
    # Process text and generate embeddings for each view
    for view in CAMERA_VIEWS:
        # Create necessary directories
        metas_dir = data_root / "metas" / view
        t5_dir = data_root / "t5_xxl" / view
        metas_dir.mkdir(parents=True, exist_ok=True)
        t5_dir.mkdir(parents=True, exist_ok=True)
        
        # Process text and embeddings for each video
        for key, caption in tqdm(text_data.items(), desc=f"Processing {view}"):
            # Check if this key belongs to current view
            if not key.endswith(f"_{view}"):
                continue
                
            # Extract base key (without view suffix)
            base_key = key[:-len(f"_{view}")]
            
            # Get video path by matching key
            video_path = _get_video_path(data_root, view, base_key)
            if video_path is None:
                continue
                
            # Get clip id
            clip_id = _get_clip_id(video_path)
            
            # Save text file
            txt_path = metas_dir / f"{base_key}{clip_id}.txt"
            with open(txt_path, "w") as f:
                f.write(caption)
            
            # Generate and save T5 embeddings
            pkl_path = t5_dir / f"{base_key}{clip_id}.pkl"
            if not pkl_path.exists():
                # Generate T5 embeddings
                t5_embedding = [_encode_text(caption)]
                
                # Save embedding file
                with open(pkl_path, "wb") as f:
                    pickle.dump(t5_embedding, f)

if __name__ == "__main__":
    main() 