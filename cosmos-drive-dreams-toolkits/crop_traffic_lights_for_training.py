import os
# Set HF_HOME environment variable
os.environ['HF_HOME'] = '/lustre/fsw/portfolios/nvr/users/tianshic/huggingface/'

import numpy as np
import click
import cv2
import json
import decord
import ray
from typing import Optional
from pathlib import Path
from tqdm import tqdm
from utils.wds_utils import get_sample, write_to_tar
from utils.bbox_utils import create_bbox_projection, interpolate_bbox, fix_static_objects, build_cuboid_bounding_box
from utils.minimap_utils import create_minimap_projection, simplify_minimap
from utils.pcd_utils import batch_move_points_within_bboxes_sparse, forward_warp_multiframes_sparse_depth_only
from utils.camera.pinhole import PinholeCamera
from utils.camera.ftheta import FThetaCamera
from utils.ray_utils import ray_remote, wait_for_futures

USE_RAY = False

def extract_clip_id_from_image_name(image_name):
    """Extract clip ID from image name."""
    # Image name format: "clip_id.frame_time.projected.jpg"
    return image_name.split('.')[0]

@ray_remote(use_ray=USE_RAY, num_gpus=0, num_cpus=4)
def crop_traffic_lights(input_root: str, output_root: str, clip_id: str, settings: dict, camera_type: str, post_training: bool = False, novel_pose_folder: Optional[str] = None, n_skip_frames: int = 0, target_frame_indices: Optional[list] = None, frame_to_feature_status: Optional[dict] = None):
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
        post_training (bool): Whether to use post-training settings
        novel_pose_folder (Optional[str]): Alternative pose folder name if using novel trajectories
        n_skip_frames (int): Number of frames to skip between processed frames (default: 0, process all frames)
        target_frame_indices (Optional[list]): List of frame indices to process (default: None, process all frames)
        frame_to_feature_status (Optional[dict]): Dictionary mapping frame indices to feature IDs and their status
    
    Returns:
        None: Saves cropped images and metadata to output_root
    
    Notes:
        - Handles multiple cameras per clip based on settings['CAMERAS']
        - Filters traffic lights that are not visible or too small in camera view
        - Supports both ftheta (fisheye) and pinhole camera models
        - Uses ray for parallel processing across clips
    """
    
    print(f"Processing {clip_id} for traffic light cropping...")
    
    # Create output directories
    output_root_p = Path(output_root)
    (output_root_p / "cropped_images").mkdir(parents=True, exist_ok=True)
    (output_root_p / "metadata").mkdir(parents=True, exist_ok=True)
    
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
            (output_root_p / "cropped_images" / camera_folder_name).mkdir(parents=True, exist_ok=True)
            
            crop_metadata = []
            crop_count = 0
            
            # Determine which frames to process
            if target_frame_indices is not None:
                # Only process specified frame indices
                frames_to_process = [idx for idx in target_frame_indices if idx < min(num_frames, total_frames)]
                print(f"Processing {len(frames_to_process)} specified frames (out of {len(target_frame_indices)} requested)")
            else:
                # Process all frames with skip intervals (original behavior)
                frames_to_process = list(range(0, min(num_frames, total_frames), n_skip_frames + 1))
                print(f"Processing {len(frames_to_process)} frames with skip interval {n_skip_frames}")
            
            for frame_idx in frames_to_process:
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
                    feature_id = traffic_light['feature_id']
                    
                    # Check if this traffic light's feature_id exists in the label for this frame
                    if frame_to_feature_status is not None:
                        if frame_idx not in frame_to_feature_status:
                            continue  # No label data for this frame
                        
                        frame_feature_status = frame_to_feature_status[frame_idx]
                        if feature_id not in frame_feature_status:
                            continue  # This traffic light's feature_id not in label for this frame
                        
                        # Get the status for this traffic light
                        traffic_light_status = frame_feature_status[feature_id]
                    else:
                        # If no feature status info provided, use 'unknown'
                        traffic_light_status = 'unknown'
                    
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
                    padding = 0.4 * (max_y - min_y)  # pixels
                    min_x = max(0, int(min_x - padding))
                    max_x = min(frame.shape[1], int(max_x + padding))
                    min_y = max(0, int(min_y - padding))
                    max_y = min(frame.shape[0], int(max_y + padding))
                    
                    # Check if bounding box is valid
                    bbox_width = max_x - min_x
                    bbox_height = max_y - min_y
                    
                    if bbox_width < 52 or bbox_height < 52:  # Too small
                        continue
                    
                    if bbox_width > frame.shape[1] * 0.8 or bbox_height > frame.shape[0] * 0.8:  # Too large
                        continue
                    
                    # 5. Crop and save image
                    cropped_image = frame[min_y:max_y, min_x:max_x]
                    
                    if cropped_image.size == 0:
                        continue
                    
                    # Create status-based subfolder
                    status_folder = output_root_p / "cropped_images" / traffic_light_status
                    status_folder.mkdir(parents=True, exist_ok=True)
                    
                    # Save cropped image
                    crop_filename = f"{clip_id}_{camera_name}_{frame_idx:06d}_{tl_idx:03d}.jpg"
                    crop_path = status_folder / crop_filename
                    
                    cv2.imwrite(str(crop_path), cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
                    
                    # Save metadata
                    metadata_entry = {
                        'clip_id': clip_id,
                        'camera_name': camera_name,
                        'camera_type': camera_type,
                        'frame_idx': frame_idx,
                        'traffic_light_idx': tl_idx,
                        'filename': crop_filename,
                        'traffic_light_status': traffic_light_status,
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
                
                with open(metadata_path, 'w') as f:
                    json.dump(crop_metadata, f, indent=2)
                
                print(f"Saved {crop_count} cropped traffic lights for camera {camera_name}")
            else:
                print(f"No valid traffic light crops found for camera {camera_name}")
    
    except Exception as e:
        print(f"Error processing {clip_id}: {e}")
        import traceback
        traceback.print_exc()

def process_label_file(label_file_path: str, input_root: str, output_root: str, settings: dict, camera_type: str, n_skip_frames: int = 0):
    """Process a single label file to extract clip ID and crop traffic lights."""
    
    print(f"Processing label file: {label_file_path}")
    
    try:
        # Read the label file
        with open(label_file_path, 'r') as f:
            label_data = json.load(f)
        
        if not label_data:
            print(f"Warning: Label file {label_file_path} is empty")
            return False
        
        # Extract clip ID from the first element
        first_entry = label_data[0]
        image_name = first_entry['image_name']
        clip_id = extract_clip_id_from_image_name(image_name)
        
        # Extract frame indices and feature_id_to_status information
        frame_indices = set()
        frame_to_feature_status = {}  # frame_idx -> {feature_id: status}
        skipped_entries = 0
        
        for entry in label_data:
            if 'frame_idx' in entry:
                frame_idx = entry['frame_idx']
                frame_indices.add(frame_idx)
                
                # Extract feature_id_to_status if available
                if 'feature_id_to_status' in entry:
                    frame_to_feature_status[frame_idx] = entry['feature_id_to_status']
                else:
                    # Fallback to empty dict if no feature_id_to_status
                    frame_to_feature_status[frame_idx] = {}
            else:
                skipped_entries += 1
        
        # Convert to sorted list
        frame_indices = sorted(list(frame_indices))
        
        # Count total feature IDs across all frames
        all_feature_ids = set()
        for feature_status_dict in frame_to_feature_status.values():
            all_feature_ids.update(feature_status_dict.keys())
        
        print(f"Extracted clip ID: {clip_id}")
        print(f"Label data contains {len(label_data)} entries")
        print(f"Found {len(frame_indices)} unique frame indices to process")
        print(f"Found {len(all_feature_ids)} unique feature IDs across all frames")
        if skipped_entries > 0:
            print(f"Skipped {skipped_entries} entries without frame_idx")
        
        if not frame_indices:
            print("Warning: No valid frame indices found in label file")
            return False
        
        # Run crop_traffic_lights on the clip with specific frame indices and feature status info
        crop_traffic_lights(
            input_root=input_root,
            output_root=output_root,
            clip_id=clip_id,
            settings=settings,
            camera_type=camera_type,
            post_training=False,
            novel_pose_folder=None,
            n_skip_frames=n_skip_frames,
            target_frame_indices=frame_indices,
            frame_to_feature_status=frame_to_feature_status
        )
        
        print(f"Successfully processed clip {clip_id}")
        return True
        
    except Exception as e:
        print(f"Error processing label file {label_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

@click.command()
@click.option("--label_path", '-l', type=str, required=True, help="Path to folder containing JSON label files, or path to a single JSON file")
@click.option("--input_root", '-i', type=str, help="the root folder of RDS-HQ or RDS-HQ format dataset")
@click.option("--output_root", '-o', type=str, required=True, help="the root folder for the output data")
@click.option("--dataset", "-d", type=str, default="rds_hq", help="the dataset name, 'rds_hq', 'rds_hd_mv', 'waymo' or 'waymo_mv', see xxx.json in config folder")
@click.option("--camera_type", "-c", type=str, default="ftheta", help="the type of camera model, 'pinhole' or 'ftheta'")
@click.option("--n_skip_frames", "-n", type=int, default=0, help="number of frames to skip between processed frames (0 means process all frames)")
@click.option("--pattern", default="*.json", help="File pattern to match in the folder (default: *.json)")
def main(label_path, input_root, output_root, dataset, camera_type, n_skip_frames, pattern):
    """
    Crop traffic lights from MADS dataset video frames using label files.
    
    This script processes label files to extract clip IDs and then crops traffic light regions
    by projecting 3D cuboid annotations to 2D camera views.
    """
    
    # Load settings
    try:
        with open(f'config/dataset_{dataset}.json', 'r') as file:
            settings = json.load(file)
    except FileNotFoundError:
        print(f"Error: Config file 'config/dataset_{dataset}.json' not found")
        return
    
    # Use default MADS dataset path if not provided
    if input_root is None:
        input_root = "/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_holodeck/cosmos-mads-dataset-all-in-one"
    
    # Determine if we're processing a single file or multiple files
    label_path_p = Path(label_path)
    
    if not label_path_p.exists():
        print(f"Error: Path '{label_path}' not found")
        return
    
    if label_path_p.is_file():
        label_files = [label_path_p]
    elif label_path_p.is_dir():
        label_files = list(label_path_p.glob(pattern))
        if not label_files:
            print(f"Error: No files matching pattern '{pattern}' found in '{label_path}'")
            return
    else:
        print(f"Error: '{label_path}' is neither a file nor a directory")
        return
    
    print(f"Found {len(label_files)} label file(s) to process")
    print(f"Using camera type: {camera_type}")
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    
    # Process each label file
    successful_files = []
    failed_files = []
    
    if USE_RAY:
        ray.init()
        futures = []
        for label_file in label_files:
            future = process_label_file.remote(
                str(label_file), input_root, output_root, settings, camera_type, n_skip_frames
            )
            futures.append(future)
        
        results = wait_for_futures(futures)
        if results is not None:
            for i, result in enumerate(results):
                if result:
                    successful_files.append(label_files[i])
                else:
                    failed_files.append(label_files[i])
        else:
            print("Warning: Ray processing returned no results")
            failed_files = label_files.copy()
    else:
        for i, label_file in enumerate(tqdm(label_files, desc="Processing label files")):
            print(f"\n{'='*80}")
            print(f"Processing file {i+1}/{len(label_files)}: {label_file}")
            print(f"{'='*80}")
            
            success = process_label_file(
                str(label_file), input_root, output_root, settings, camera_type, n_skip_frames
            )
            
            if success:
                successful_files.append(label_file)
            else:
                failed_files.append(label_file)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total files found: {len(label_files)}")
    print(f"Successfully processed: {len(successful_files)}")
    print(f"Failed: {len(failed_files)}")
    
    if successful_files:
        print(f"\nSuccessful files:")
        for f in successful_files:
            print(f"  - {f}")
    
    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")
    
    print(f"\nTraffic light cropping completed. Results saved to: {output_root}")

if __name__ == "__main__":
    main()  