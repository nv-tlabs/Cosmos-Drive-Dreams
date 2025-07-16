"""
srun -A nvr_torontoai_videogen -p cpu -t 23:59:00 --cpus-per-task=64 --exclusive --pty bash
ray start --head --port 39668 --num-cpus 0 --num-gpus 0
python crop_traffic_lights.py -cj /lustre/fsw/portfolios/nvr/users/yiflu/Data/cosmos-mads-dataset-all-in-one/all_in_one_clip_ids_143k.json -o /lustre/fsw/portfolios/nvr/users/tianshic/cosmos-drive-dreams/traffic_lights_status_143k_clips -d rds_hq_mv -c ftheta -n 10 -m clip
"""
import os
# Set HF_HOME environment variable
os.environ['HF_HOME'] = '/lustre/fsw/portfolios/nvr/users/tianshic/huggingface/'

import numpy as np
import click
import imageio as imageio_v1
import torch
import cv2
import json
import decord
import ray
import time
import threading
from typing import Optional, Tuple, Union
from PIL import Image

from tqdm import tqdm
from pathlib import Path
from termcolor import cprint, colored
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel
import torch.nn as nn
from torchvision import transforms
from utils.wds_utils import get_sample, write_to_tar
from utils.bbox_utils import create_bbox_projection, interpolate_bbox, fix_static_objects, build_cuboid_bounding_box
from utils.minimap_utils import create_minimap_projection, simplify_minimap
from utils.pcd_utils import batch_move_points_within_bboxes_sparse, forward_warp_multiframes_sparse_depth_only
from utils.camera.pinhole import PinholeCamera
from utils.camera.ftheta import FThetaCamera
from utils.ray_utils import ray_remote, wait_for_futures

USE_RAY = True
DEBUG=False
if DEBUG:
    USE_RAY = False

class TrafficLightCLIPClassifier(nn.Module):
    """Traffic light classifier using CLIP backbone with MLP head."""
    
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", num_classes=5, hidden_dim=512, dropout=0.3):
        super().__init__()
        
        # Load CLIP model from transformers
        self.clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.bfloat16)
        self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Get CLIP feature dimension by running a forward pass with dummy input
        # This ensures we get the actual output dimension from get_image_features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16)
            clip_features = self.clip_model.get_image_features(pixel_values=dummy_input)
            clip_dim = clip_features.shape[-1]
        
        # MLP prediction head
        self.classifier = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x):
        # Extract CLIP features
        with torch.no_grad():
            clip_outputs = self.clip_model.get_image_features(pixel_values=x)
            clip_features = clip_outputs.float()
        
        # Pass through MLP head
        logits = self.classifier(clip_features)
        return logits

def load_clip_model(model_path, device='cuda'):
    """Load CLIP-based traffic light classifier."""
    try:
        print(f"Loading CLIP model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['config']
        class_names = checkpoint['class_names']
        
        # Create model
        model = TrafficLightCLIPClassifier(
            clip_model_name=config['clip_model'],
            num_classes=len(class_names),
            hidden_dim=config['hidden_dim'],
            dropout=config['dropout']
        ).to(device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create transform for preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        print(f"CLIP model loaded successfully!")
        print(f"Classes: {class_names}")
        
        return model, transform, class_names
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return None, None, None

def analyze_traffic_light_with_clip(image, model, transform, class_names, device='cuda'):
    """Analyze a traffic light image using CLIP-based classifier."""
    try:
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Preprocess image
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class_idx = int(torch.argmax(probabilities, dim=1).item())
            confidence = probabilities[0, predicted_class_idx].item()
        
        # Direct mapping from class index to traffic light state
        # Expected class order: ['Green', 'Occluded', 'Red', 'Unknown', 'Yellow']
        index_to_state = {
            0: 'green',     # Green
            1: 'unknown',   # Occluded -> unknown
            2: 'red',       # Red
            3: 'unknown',   # Unknown
            4: 'yellow'     # Yellow
        }
        
        # Return the corresponding state, default to 'unknown' if index not found
        return index_to_state.get(predicted_class_idx, 'unknown')
        
    except Exception as e:
        print(f"Error analyzing image with CLIP model: {e}")
        return 'error'

def load_qwen_model(device='cuda'):
    """Load Qwen2.5-VL model for traffic light analysis."""
    try:
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor
    except Exception as e:
        print(f"Error loading Qwen model: {e}")
        return None, None

def analyze_traffic_light_with_qwen(image, model, processor):
    """Analyze a traffic light image using Qwen to determine visibility and state."""
    try:
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Prepare the prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text", 
                        "text": "Is a traffic light visible in this image? Do not confuse the traffic light with other objects such as trafic signs, especially the stop sign. If yes, what state is the traffic light in? Please respond with one of: 'red', 'green', 'yellow', 'left-turn', or 'unknown' if no traffic light is clearly visible."
                    }
                ]
            }
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process inputs
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            #videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        
        # Parse the response
        output_text = output_text.lower().strip()
        
        # Determine the state
        if 'unknown' in output_text or 'no traffic light' in output_text:
            return 'unknown'
        elif 'red' in output_text:
            return 'red'
        elif 'green' in output_text:
            return 'green'
        elif 'yellow' in output_text:
            return 'yellow'
        elif 'left-turn' in output_text or 'left turn' in output_text:
            return 'left-turn'
        else:
            return 'unknown'
            
    except Exception as e:
        print(f"Error analyzing image with Qwen: {e}")
        return 'error'

def process_vision_info(messages):
    """Helper function to process vision info from messages."""
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if isinstance(message, dict) and "content" in message:
            for content in message["content"]:
                if content.get("type") == "image":
                    image_inputs.append(content["image"])
                elif content.get("type") == "video":
                    video_inputs.append(content["video"])
    
    return image_inputs, video_inputs

@ray_remote(use_ray=USE_RAY, num_gpus=1, num_cpus=4)
def crop_traffic_lights(input_root: str, output_root: str, clip_id: str, settings: dict, camera_type: str, model_type: str = "qwen", clip_model_path: Optional[str] = None, post_training: bool = False, novel_pose_folder: Optional[str] = None, n_skip_frames: int = 0):
    """
    Crop traffic lights from camera video frames using 3D cuboid annotations from MADS dataset.
    
    This function processes a single clip from the MADS dataset structure and extracts traffic light
    regions by projecting 3D cuboid annotations to 2D camera views and cropping the corresponding
    image regions.
    
    Data Pipeline Plan:
    ==================
    
    1. **Load Traffic Light Annotations**:
       - Read 3D traffic light cuboids from: {input_root}/3d_traffic_lights/{clip_id}.tar
       - Extract cuboid vertices from: data['traffic_lights.json']['labels'][i]['labelData']['shape3d']['cuboid3d']['vertices']
       - Each cuboid has 8 corner vertices in 3D world coordinates
    
    2. **Load Camera Data**:
       - Read camera video from: {input_root}/{camera_type}_{camera_name}/{clip_id}.mp4
       - Load camera poses from: {input_root}/pose/{clip_id}.tar (camera-to-world transforms)
       - Load camera intrinsics from: {input_root}/{camera_type}_intrinsic/{clip_id}.tar
       - Support both 'ftheta' and 'pinhole' camera models
    
    3. **Synchronization**:
       - Camera videos are at 30 FPS
       - 3D annotations are static (no temporal info in cuboids)
       - Match video frames with corresponding camera poses
    
    4. **3D to 2D Projection**:
       - For each frame and each traffic light cuboid:
         a. Transform 3D cuboid vertices from world to camera coordinates using camera pose
         b. Project camera coordinates to 2D image coordinates using camera intrinsics
         c. Compute 2D bounding box from projected cuboid vertices (min/max x,y)
         d. Apply padding/margins to ensure full traffic light is captured
    
    5. **Cropping and Saving**:
       - Extract image regions based on 2D bounding boxes
       - Filter out invalid crops (too small, out of bounds, etc.)
       - Save cropped images with metadata:
         * Filename format: {clip_id}_{camera_name}_{frame_idx}_{traffic_light_idx}.jpg
         * Metadata: original bbox, 3D cuboid info, camera info
    
    6. **Output Structure**:
       ```
       {output_root}/
       ├── cropped_images/
       │   ├── {camera_type}_{camera_name}/
       │   │   ├── {clip_id}_{frame_idx}_{traffic_light_idx}.jpg
       │   │   └── ...
       └── metadata/
           ├── {clip_id}_{camera_name}_metadata.json
           └── ...
       ```
    
    Args:
        input_root (str): Root directory of MADS dataset 
        output_root (str): Output directory for cropped traffic light images
        clip_id (str): Unique clip identifier (e.g., "812d1e08-a5b9-4cb3-9425-5cd80bf55303_19432580104_19452580104")
        settings (dict): Dataset configuration settings
        camera_type (str): Camera model type ('ftheta' or 'pinhole')
        model_type (str): Model type ('qwen' or 'clip')
        clip_model_path (str): Path to CLIP model checkpoint
        post_training (bool): Whether to use post-training settings
        novel_pose_folder (Optional[str]): Alternative pose folder name if using novel trajectories
        n_skip_frames (int): Number of frames to skip between processed frames (default: 0, process all frames)
    
    Returns:
        None: Saves cropped images and metadata to output_root
    
    Notes:
        - Handles multiple cameras per clip based on settings['CAMERAS']
        - Filters traffic lights that are not visible or too small in camera view
        - Supports both ftheta (fisheye) and pinhole camera models
        - Uses ray for parallel processing across clips
    """
    
    print(f"Processing {clip_id} for traffic light cropping...")
    
    # Load traffic light analysis model based on model_type
    print(f"Loading {model_type} model for traffic light analysis...")
    
    if model_type == "qwen":
        qwen_model, qwen_processor = load_qwen_model()
        if qwen_model is None or qwen_processor is None:
            print("Failed to load Qwen model, skipping analysis")
            return
        clip_model = None
        clip_transform = None
        clip_class_names = None
    elif model_type == "clip":
        if clip_model_path is None:
            print("Error: clip_model_path must be provided when using CLIP model")
            return
        if not os.path.exists(clip_model_path):
            print(f"Error: CLIP model checkpoint not found at {clip_model_path}")
            return
        clip_model, clip_transform, clip_class_names = load_clip_model(clip_model_path)
        if clip_model is None:
            print("Failed to load CLIP model, skipping analysis")
            return
        qwen_model = None
        qwen_processor = None
    else:
        print(f"Error: Unsupported model_type '{model_type}'. Use 'qwen' or 'clip'")
        return
    
    # Create output directories
    output_root_p = Path(output_root)
    (output_root_p / "cropped_images").mkdir(parents=True, exist_ok=True)
    (output_root_p / "metadata").mkdir(parents=True, exist_ok=True)
    
    # Store traffic light states for each camera
    clip_traffic_light_states = {}
    
    try:
        # 1. Load traffic light annotations
        traffic_lights_file = os.path.join(input_root, '3d_traffic_lights', f"{clip_id}.tar")
        if not os.path.exists(traffic_lights_file):
            print(f"Warning: Traffic lights file not found for {clip_id}")
            return
        
        traffic_lights_data = get_sample(traffic_lights_file)
        traffic_lights_json = traffic_lights_data['traffic_lights.json']
        traffic_light_labels = traffic_lights_json['labels']
        
        # Extract 3D cuboids
        traffic_light_cuboids = []
        for i, label in enumerate(traffic_light_labels):
            if 'labelData' in label and 'shape3d' in label['labelData'] and 'cuboid3d' in label['labelData']['shape3d']:
                vertices = label['labelData']['shape3d']['cuboid3d']['vertices']
                if len(vertices) == 8:  # Valid cuboid
                    # Extract feature_id from attributes
                    feature_id = None
                    if 'attributes' in label['labelData']['shape3d']:
                        for attribute in label['labelData']['shape3d']['attributes']:
                            if attribute.get('name') == 'feature_id':
                                feature_id = attribute.get('text')
                                break
                    
                    traffic_light_cuboids.append({
                        'index': i,
                        'vertices': np.array(vertices, dtype=np.float32),
                        'label_info': label,
                        'feature_id': feature_id
                    })
        
        if len(traffic_light_cuboids) == 0:
            print(f"No valid traffic light cuboids found for {clip_id}")
            return
        
        print(f"Found {len(traffic_light_cuboids)} traffic light cuboids")
        
        # Get total number of frames by checking the first camera
        first_camera = settings['CAMERAS'][0]
        pose_folder = 'pose' if novel_pose_folder is None else novel_pose_folder
        pose_file = os.path.join(input_root, pose_folder, f"{clip_id}.tar")
        if os.path.exists(pose_file):
            pose_data = get_sample(pose_file)
            camera_key = f"pose.{first_camera}.npy"
            pose_data_this_cam = {k: v for k, v in pose_data.items() if camera_key in k}
            if len(pose_data_this_cam) > 0:
                camera_poses = np.stack([pose_data_this_cam[k] for k in sorted(pose_data_this_cam.keys())])
                total_num_frames = camera_poses.shape[0]
            else:
                total_num_frames = 0
        else:
            total_num_frames = 0
        
        # Process each camera
        for camera_name in settings['CAMERAS']:
            print(f"Processing camera: {camera_name}")
            
            # 2. Load camera data
            # Load camera poses
            pose_folder = 'pose' if novel_pose_folder is None else novel_pose_folder
            pose_file = os.path.join(input_root, pose_folder, f"{clip_id}.tar")
            
            if not os.path.exists(pose_file):
                print(f"Warning: Pose file not found for {clip_id}")
                continue
                
            pose_data = get_sample(pose_file)
            camera_key = f"pose.{camera_name}.npy"
            pose_data_this_cam = {k: v for k, v in pose_data.items() if camera_key in k}
            
            if len(pose_data_this_cam) == 0:
                print(f"Warning: No pose data found for camera {camera_name}")
                continue
                
            camera_poses = np.stack([pose_data_this_cam[k] for k in sorted(pose_data_this_cam.keys())])
            num_frames = camera_poses.shape[0]
            
            # Load camera intrinsics
            intrinsic_file = os.path.join(input_root, f'{camera_type}_intrinsic', f"{clip_id}.tar")
            
            if not os.path.exists(intrinsic_file):
                if camera_type == "ftheta":
                    print(f"Warning: Ftheta intrinsic file not found, using default")
                    intrinsic_file = 'assets/default_ftheta_intrinsic.tar'
                    camera_name_in_rds_hq = settings.get('CAMERAS_TO_RDS_HQ', {}).get(camera_name, camera_name)
                    intrinsic_data = get_sample(intrinsic_file)
                    intrinsic_this_cam = intrinsic_data[f"{camera_type}_intrinsic.{camera_name_in_rds_hq}.npy"]
                else:
                    print(f"Warning: Intrinsic file not found for {clip_id}")
                    continue
            else:
                intrinsic_data = get_sample(intrinsic_file)
                intrinsic_this_cam = intrinsic_data[f"{camera_type}_intrinsic.{camera_name}.npy"]
            
            # Create camera model
            if camera_type == "pinhole":
                camera_model = PinholeCamera.from_numpy(intrinsic_this_cam, device='cpu')
                
            elif camera_type == "ftheta":
                camera_model = FThetaCamera.from_numpy(intrinsic_this_cam, device='cpu')
                
            else:
                raise ValueError(f"Invalid camera type: {camera_type}")
            
            # Load camera video
            video_file = os.path.join(input_root, f'{camera_type}_{camera_name}', f"{clip_id}.mp4")
            
            if not os.path.exists(video_file):
                print(f"Warning: Video file not found for {camera_name}")
                continue
            
            vr = decord.VideoReader(video_file)
            total_frames = len(vr)
            
            # 3. Process each frame
            camera_folder_name = f"{camera_type}_{camera_name}"
            
            crop_metadata = []
            crop_count = 0
            
            for frame_idx in range(0, min(num_frames, total_frames), n_skip_frames + 1):
                # Load frame
                try:
                    frame = vr[frame_idx].asnumpy()
                except Exception as e:
                    print(f"Error loading frame {frame_idx}: {e}")
                    continue
                
                camera_to_world = camera_poses[frame_idx]
                world_to_camera = np.linalg.inv(camera_to_world)
                
                # 4. Process each traffic light
                for tl_idx, traffic_light in enumerate(traffic_light_cuboids):
                    vertices_3d = traffic_light['vertices']  # 8x3 array
                    
                    # Transform 3D vertices to camera coordinates
                    vertices_cam = camera_model.transform_points_np(vertices_3d, world_to_camera)
                    
                    # Check if any vertices are in front of camera
                    if (vertices_cam[:, 2] <= 0).all():
                        continue  # All vertices behind camera
                    
                    # Project to 2D pixels
                    if camera_type == "pinhole":
                        pixels_2d = camera_model.ray2pixel_np(vertices_cam)
                    else:  # ftheta
                        pixels_2d = camera_model.ray2pixel_np(vertices_cam)
                    
                    # Filter out points behind camera for bbox computation
                    valid_mask = vertices_cam[:, 2] > 0
                    if not valid_mask.any():
                        continue
                    
                    valid_pixels = pixels_2d[valid_mask]
                    
                    # Compute 2D bounding box
                    min_x = np.min(valid_pixels[:, 0])
                    max_x = np.max(valid_pixels[:, 0])
                    min_y = np.min(valid_pixels[:, 1])
                    max_y = np.max(valid_pixels[:, 1])
                    
                    # Apply padding
                    padding = 0.2 * (max_y - min_y)  # pixels
                    min_x = max(0, int(min_x - padding))
                    max_x = min(frame.shape[1], int(max_x + padding))
                    min_y = max(0, int(min_y - padding))
                    max_y = min(frame.shape[0], int(max_y + padding))
                    
                    # Check if bounding box is valid
                    bbox_width = max_x - min_x
                    bbox_height = max_y - min_y
                    
                    if bbox_width < 32 or bbox_height < 32:  # Too small
                        continue
                    
                    if bbox_width > frame.shape[1] * 0.8 or bbox_height > frame.shape[0] * 0.8:  # Too large
                        continue
                    
                    # 5. Crop and save image
                    cropped_image = frame[min_y:max_y, min_x:max_x]
                    
                    if cropped_image.size == 0:
                        continue
                    
                    # Analyze traffic light state with selected model
                    if model_type == "qwen":
                        traffic_light_state = analyze_traffic_light_with_qwen(cropped_image, qwen_model, qwen_processor)
                    elif model_type == "clip":
                        traffic_light_state = analyze_traffic_light_with_clip(cropped_image, clip_model, clip_transform, clip_class_names)
                    else:
                        traffic_light_state = 'error'
                    
                    # Create directory structure for this clip and traffic light
                    tl_output_dir = output_root_p / "cropped_images" / clip_id / str(tl_idx)
                    tl_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save cropped image with new naming format
                    crop_filename = f"{frame_idx:06d}_{camera_folder_name}_{traffic_light_state}.png"
                    crop_path = tl_output_dir / crop_filename
                    if DEBUG:
                        cv2.imwrite(str(crop_path), cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
                    
                    # Store traffic light state analysis
                    traffic_light_analysis = {
                        'clip_id': clip_id,
                        'camera_name': camera_name,
                        'frame_idx': frame_idx,
                        'traffic_light_idx': tl_idx,
                        'filename': crop_filename,
                        'state': traffic_light_state,
                        'bbox_2d': {
                            'min_x': min_x,
                            'max_x': max_x,
                            'min_y': min_y,
                            'max_y': max_y
                        }
                    }
                    
                    # Store in clip_traffic_light_states using (tl_idx, frame_idx) as key
                    key = (tl_idx, frame_idx)
                    if key not in clip_traffic_light_states:
                        clip_traffic_light_states[key] = {}
                    clip_traffic_light_states[key][camera_name] = {
                        'state': traffic_light_state,
                        'analysis': traffic_light_analysis
                    }
                    
                    # Save metadata
                    metadata_entry = {
                        'clip_id': clip_id,
                        'camera_name': camera_name,
                        'camera_type': camera_type,
                        'frame_idx': frame_idx,
                        'traffic_light_idx': tl_idx,
                        'filename': crop_filename,
                        'traffic_light_state': traffic_light_state,
                        'bbox_2d': {
                            'min_x': min_x,
                            'max_x': max_x,
                            'min_y': min_y,
                            'max_y': max_y
                        },
                        'vertices_3d': vertices_3d.tolist(),
                        'vertices_cam': vertices_cam.tolist(),
                        'pixels_2d': pixels_2d.tolist(),
                        'camera_pose': camera_to_world.tolist(),
                        'original_label_info': traffic_light['label_info'],
                        'feature_id': traffic_light['feature_id']
                    }
                    
                    crop_metadata.append(metadata_entry)
                    crop_count += 1
            
            # Save metadata for this camera
            if crop_metadata:
                metadata_filename = f"{clip_id}_{camera_name}_metadata.json"
                metadata_path = output_root_p / "metadata" / metadata_filename
                
                # with open(metadata_path, 'w') as f:
                #     json.dump(crop_metadata, f, indent=2)
                
                print(f"Saved {crop_count} cropped traffic lights for camera {camera_name}")
            else:
                print(f"No valid traffic light crops found for camera {camera_name}")
        
        # Aggregate traffic light states across cameras using majority voting
        def aggregate_states_with_priority(states_list):
            """Aggregate traffic light states with priority voting."""
            if not states_list:
                return 'unknown'
            
            # Count occurrences
            state_counts = {}
            for state in states_list:
                state_counts[state] = state_counts.get(state, 0) + 1
            
            # Priority order: red, green, yellow, left-turn, then others
            priority_states = ['red', 'green', 'yellow', 'left-turn']
            
            # Find the most common state among priority states first
            for priority_state in priority_states:
                if priority_state in state_counts and state_counts[priority_state] > 0:
                    # Check if this priority state has majority or ties with other priority states
                    max_count = max(state_counts.values())
                    if state_counts[priority_state] == max_count:
                        return priority_state
            
            # If no priority states, return the most common state
            if state_counts:
                most_common_state = max(state_counts.items(), key=lambda x: x[1])[0]
                return str(most_common_state)
            
            return 'unknown'
        
        # Create final aggregated structure
        num_traffic_lights = len(traffic_light_cuboids)
        aggregated_traffic_light_states = {}
        
        for tl_idx in range(num_traffic_lights):
            # Create list of states for this traffic light across all frames
            traffic_light_frames = ['skipped'] * total_num_frames
            
            for frame_idx in range(total_num_frames):
                key = (tl_idx, frame_idx)
                if key in clip_traffic_light_states:
                    # Get states from all cameras for this traffic light and frame
                    camera_states = []
                    for camera_name, data in clip_traffic_light_states[key].items():
                        camera_states.append(data['state'])
                    
                    # Aggregate using majority voting with priority
                    aggregated_state = aggregate_states_with_priority(camera_states)
                    traffic_light_frames[frame_idx] = aggregated_state
                # If key not in clip_traffic_light_states, it remains 'skipped'
            
            # Second pass: Replace 'skipped' states with closest valid states
            def interpolate_skipped_states(states_list):
                """Replace 'skipped' states with closest valid states in time."""
                valid_states = ['red', 'green', 'yellow', 'left-turn', 'unknown']
                interpolated_states = states_list.copy()
                
                for i, state in enumerate(states_list):
                    if state == 'skipped':
                        # Find closest valid state
                        closest_state = 'unknown'  # default fallback
                        min_distance = float('inf')
                        
                        # Search both directions from current frame
                        for j, other_state in enumerate(states_list):
                            if other_state in valid_states:
                                distance = abs(i - j)
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_state = other_state
                        
                        interpolated_states[i] = closest_state
                
                return interpolated_states
            
            traffic_light_frames = interpolate_skipped_states(traffic_light_frames)
            
            # Store as dictionary with state and feature_id
            aggregated_traffic_light_states[tl_idx] = {
                'state': traffic_light_frames,
                'feature_id': traffic_light_cuboids[tl_idx]['feature_id']
            }
        
        print(f"Created aggregated states for {num_traffic_lights} traffic lights across {total_num_frames} frames")
        
        # Save traffic light states to tar file for the entire clip
        if clip_traffic_light_states or aggregated_traffic_light_states:
            traffic_light_states_tar_path = output_root_p / "traffic_light_states" / f"{clip_id}.tar"
            # Ensure the directory exists
            (output_root_p / "traffic_light_states").mkdir(parents=True, exist_ok=True)
            traffic_light_data = {"__key__": clip_id}
            
            # Save raw per-camera data
            raw_camera_data = {}
            for (tl_idx, frame_idx), camera_data in clip_traffic_light_states.items():
                key_str = f"tl_{tl_idx}_frame_{frame_idx}"
                if key_str not in raw_camera_data:
                    raw_camera_data[key_str] = {}
                for camera_name, data in camera_data.items():
                    raw_camera_data[key_str][camera_name] = data['analysis']
            
            traffic_light_data['raw_camera_states.json'] = json.dumps(raw_camera_data, indent=2)
            
            # Save aggregated states
            aggregated_data = {
                'num_traffic_lights': num_traffic_lights,
                'num_frames': total_num_frames,
                'traffic_light_states': aggregated_traffic_light_states,
                'description': 'Dictionary where each key corresponds to one traffic light index, each containing state list per frame and feature_id'
            }
            traffic_light_data['aggregated_states.json'] = json.dumps(aggregated_data, indent=2)
            
            # Save summary with just the states for quick access
            summary_data = {
                'aggregated_by_traffic_light': {str(tl_idx): data['state'] for tl_idx, data in aggregated_traffic_light_states.items()},
                'feature_ids': {str(tl_idx): data['feature_id'] for tl_idx, data in aggregated_traffic_light_states.items()},
                'frame_skip_setting': n_skip_frames
            }
            traffic_light_data['summary.json'] = json.dumps(summary_data, indent=2)
            
            write_to_tar(traffic_light_data, traffic_light_states_tar_path)
            print(f"Saved traffic light states to {traffic_light_states_tar_path}")
    
    except Exception as e:
        print(f"Error processing {clip_id}: {e}")
        import traceback
        traceback.print_exc()

def is_clip_completed(output_root: str, clip_id: str) -> bool:
    """Check if a clip is already completed by checking if output tar file exists."""
    output_path = Path(output_root) / "traffic_light_states" / f"{clip_id}.tar"
    return output_path.exists()

def filter_completed_clips(clip_list: list, output_root: str, skip_completed: bool = True) -> tuple:
    """Filter out already completed clips and return statistics."""
    if not skip_completed:
        return clip_list, 0, 0
    
    completed_clips = []
    remaining_clips = []
    
    print("Checking for already completed clips...")
    for clip_id in tqdm(clip_list, desc="Checking completion status"):
        if is_clip_completed(output_root, clip_id):
            completed_clips.append(clip_id)
        else:
            remaining_clips.append(clip_id)
    
    print(f"Found {len(completed_clips)} already completed clips")
    print(f"Remaining {len(remaining_clips)} clips to process")
    
    return remaining_clips, len(completed_clips), len(clip_list)

@click.command()
@click.option("--input_root", '-i', type=str, help="the root folder of RDS-HQ or RDS-HQ format dataset")
@click.option("--clip_id_json", "-cj", type=str, default=None, help="exact clip id or path to the clip id json file. If provided, we just render the clips in the json file.")
@click.option("--output_root", '-o', type=str, required=True, help="the root folder for the output data")
@click.option("--dataset", "-d", type=str, default="rds_hq", help="the dataset name, 'rds_hq', 'rds_hd_mv', 'waymo' or 'waymo_mv', see xxx.json in config folder")
@click.option("--camera_type", "-c", type=str, default="ftheta", help="the type of camera model, 'pinhole' or 'ftheta'")
@click.option("--model_type", "-m", type=str, default="qwen", help="model type for traffic light analysis: 'qwen' or 'clip'")
@click.option("--clip_model_path", "-cp", type=str, default="training_output/best_model.pth", help="path to CLIP model checkpoint (used when model_type='clip')")
@click.option("--n_skip_frames", "-n", type=int, default=0, help="number of frames to skip between processed frames (0 means process all frames)")
@click.option("--skip_completed", "-sc", is_flag=True, default=True, help="skip clips that already have output files (default: True)")
def main(input_root, clip_id_json, output_root, dataset, camera_type, model_type, clip_model_path, n_skip_frames, skip_completed):
    """
    Crop traffic lights from MADS dataset video frames.
    
    This script processes video clips from the MADS dataset and extracts traffic light regions
    by projecting 3D cuboid annotations to 2D camera views. Traffic light states are analyzed
    using either Qwen2.5-VL or a CLIP-based classifier.
    
    Model Options:
    - qwen: Uses Qwen2.5-VL model for zero-shot traffic light analysis
    - clip: Uses a trained CLIP-based classifier (requires model checkpoint)
    """
    
    # Validate model type
    if model_type not in ["qwen", "clip"]:
        print(f"Error: Invalid model_type '{model_type}'. Must be 'qwen' or 'clip'")
        return
    
    # Validate CLIP model path if using CLIP
    if model_type == "clip":
        if not os.path.exists(clip_model_path):
            print(f"Error: CLIP model checkpoint not found at {clip_model_path}")
            print("Please train a model first using train_light_model.py or specify the correct path")
            return
    
    # Load settings
    try:
        with open(f'config/dataset_{dataset}.json', 'r') as file:
            settings = json.load(file)
    except FileNotFoundError:
        print(f"Error: Config file 'config/dataset_{dataset}.json' not found")
        return
    
    # Get clip list
    if clip_id_json is None:
        # Use default MADS dataset path if not provided
        if input_root is None:
            input_root = "/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_holodeck/cosmos-mads-dataset-all-in-one"
        
        input_root_p = Path(input_root)
        pose_folder = 'pose'
        clip_list = list((input_root_p / pose_folder).rglob('*.tar'))
        clip_list = [c.stem for c in clip_list]
    elif clip_id_json.endswith('.json'):
        with open(clip_id_json, 'r') as f:
            clip_list = json.load(f)
    else:
        clip_list = [clip_id_json]
    
    # Use default MADS dataset path if not provided
    if input_root is None:
        input_root = "/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_holodeck/cosmos-mads-dataset-all-in-one"
    
    if not clip_list:
        print("No clips found to process")
        return

    # Filter out already completed clips
    original_count = len(clip_list)
    clip_list, completed_count, total_count = filter_completed_clips(clip_list, output_root, skip_completed)
    
    if len(clip_list) == 0:
        print("All clips are already completed!")
        return
    
    if skip_completed and completed_count > 0:
        print(f"\nFiltering summary:")
        print(f"  Total clips: {total_count}")
        print(f"  Already completed: {completed_count} ({completed_count/total_count*100:.1f}%)")
        print(f"  Remaining to process: {len(clip_list)} ({len(clip_list)/total_count*100:.1f}%)")
        print()

    print(f"Found {len(clip_list)} clips to process")
    print(f"Using camera type: {camera_type}")
    print(f"Using model type: {model_type}")
    if model_type == "clip":
        print(f"CLIP model path: {clip_model_path}")
    
    # Process clips
    if USE_RAY:
        ray.init()
        futures = [
            crop_traffic_lights.remote(
                input_root, output_root, clip_id, settings, camera_type, 
                model_type, clip_model_path, False, None, n_skip_frames
            ) 
            for clip_id in clip_list
        ]
        wait_for_futures(futures)
    else:
        for clip_id in tqdm(clip_list, desc="Processing clips"):
            crop_traffic_lights(
                input_root, output_root, clip_id, settings, camera_type,
                model_type, clip_model_path, False, None, n_skip_frames
            )
    
    print(f"Traffic light cropping completed. Results saved to: {output_root}")

if __name__ == "__main__":
    main()  