#!/usr/bin/env python3

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from crop_traffic_lights import crop_traffic_lights

def extract_clip_id_from_image_name(image_name):
    """Extract clip ID from image name."""
    # Image name format: "clip_id.frame_time.projected.jpg"
    return image_name.split('.')[0]

def normalize_state(state):
    """Normalize state to lowercase for comparison."""
    if state is None:
        return 'unknown'
    state_lower = state.lower().replace(' ', '-')
    # Treat "Occluded" as "unknown"
    if state_lower == 'occluded':
        return 'unknown'
    return state_lower

def calculate_accuracy(ground_truth_data, predicted_data, clip_id):
    """Calculate accuracy between ground truth and predicted traffic light states."""
    
    # Group ground truth data by frame_idx
    gt_by_frame = {}
    for entry in ground_truth_data:
        frame_idx = entry['frame_idx']
        if frame_idx not in gt_by_frame:
            gt_by_frame[frame_idx] = {}
        
        # Extract traffic light states using feature_id_to_status
        if 'feature_id_to_status' in entry:
            for feature_id, status in entry['feature_id_to_status'].items():
                gt_by_frame[frame_idx][feature_id] = normalize_state(status)
        else:
            # Fallback to old format for backward compatibility
            for key, value in entry['traffic_light_status'].items():
                if key.startswith('Traffic Light Status '):
                    tl_idx = int(key.split('Traffic Light Status ')[1])
                    gt_by_frame[frame_idx][tl_idx] = normalize_state(value)
    
    print(f"Ground truth data covers {len(gt_by_frame)} frames")
    
    # Load predicted data from crop_traffic_lights output
    if not predicted_data:
        print("No predicted data available")
        return 0.0, 0, 0, {}
    
    # Create mapping from feature_id to predicted data
    pred_by_feature_id = {}
    if isinstance(predicted_data, dict):
        for tl_key, tl_data in predicted_data.items():
            if 'feature_id' in tl_data and 'state' in tl_data:
                feature_id = tl_data['feature_id']
                pred_by_feature_id[feature_id] = tl_data['state']
        
        num_traffic_lights = len(predicted_data)
        num_frames = len(list(predicted_data.values())[0]['state']) if num_traffic_lights > 0 else 0
        print(f"Predicted data: {num_traffic_lights} traffic lights, {num_frames} frames")
        print(f"Feature IDs available: {list(pred_by_feature_id.keys())}")
    else:
        # Legacy list format - create dummy feature IDs
        for tl_idx, tl_states in enumerate(predicted_data):
            pred_by_feature_id[tl_idx] = tl_states
        
        num_traffic_lights = len(predicted_data)
        num_frames = len(predicted_data[0]) if num_traffic_lights > 0 else 0
        print(f"Predicted data (legacy format): {num_traffic_lights} traffic lights, {num_frames} frames")
        print(f"Legacy indices: {list(pred_by_feature_id.keys())}")
    
    # Initialize per-class metrics
    classes = ['red', 'green', 'yellow', 'left-turn', 'unknown']
    class_metrics = {}
    for cls in classes:
        class_metrics[cls] = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_negatives': 0
        }
    
    # Compare predictions with ground truth
    total_comparisons = 0
    correct_predictions = 0
    
    # Track which feature IDs are found in both GT and predictions
    gt_feature_ids = set()
    for frame_data in gt_by_frame.values():
        gt_feature_ids.update(frame_data.keys())
    
    pred_feature_ids = set(pred_by_feature_id.keys())
    common_feature_ids = gt_feature_ids.intersection(pred_feature_ids)
    missing_in_pred = gt_feature_ids - pred_feature_ids
    missing_in_gt = pred_feature_ids - gt_feature_ids
    
    print(f"Ground truth feature IDs: {len(gt_feature_ids)}")
    print(f"Predicted feature IDs: {len(pred_feature_ids)}")
    print(f"Common feature IDs: {len(common_feature_ids)}")
    if missing_in_pred:
        print(f"Missing in predictions: {missing_in_pred}")
    if missing_in_gt:
        print(f"Missing in ground truth: {missing_in_gt}")
    
    for frame_idx in gt_by_frame:
        if frame_idx >= num_frames:
            continue
            
        gt_frame = gt_by_frame[frame_idx]
        
        for feature_id in gt_frame:
            if feature_id not in pred_by_feature_id:
                continue
                
            gt_state = gt_frame[feature_id]
            pred_state = normalize_state(pred_by_feature_id[feature_id][frame_idx])
            
            total_comparisons += 1
            
            # Overall accuracy
            if gt_state == pred_state:
                correct_predictions += 1
            
            # Per-class metrics
            for cls in classes:
                if gt_state == cls and pred_state == cls:
                    class_metrics[cls]['true_positives'] += 1
                elif gt_state != cls and pred_state == cls:
                    class_metrics[cls]['false_positives'] += 1
                elif gt_state == cls and pred_state != cls:
                    class_metrics[cls]['false_negatives'] += 1
                else:  # gt_state != cls and pred_state != cls
                    class_metrics[cls]['true_negatives'] += 1
            
            # Debug: print mismatches for first few
            if total_comparisons <= 10 and gt_state != pred_state:
                print(f"  Mismatch at frame {frame_idx}, feature_id {feature_id}: GT='{gt_state}' vs Pred='{pred_state}'")
    
    # Calculate per-class precision, recall, and F1
    per_class_results = {}
    for cls in classes:
        metrics = class_metrics[cls]
        tp = metrics['true_positives']
        fp = metrics['false_positives']
        fn = metrics['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_results[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn,  # Number of actual instances of this class
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    # Calculate macro averages
    macro_precision = sum(per_class_results[cls]['precision'] for cls in classes) / len(classes)
    macro_recall = sum(per_class_results[cls]['recall'] for cls in classes) / len(classes)
    macro_f1 = sum(per_class_results[cls]['f1'] for cls in classes) / len(classes)
    
    # Calculate weighted averages (weighted by support)
    total_support = sum(per_class_results[cls]['support'] for cls in classes)
    if total_support > 0:
        weighted_precision = sum(per_class_results[cls]['precision'] * per_class_results[cls]['support'] for cls in classes) / total_support
        weighted_recall = sum(per_class_results[cls]['recall'] * per_class_results[cls]['support'] for cls in classes) / total_support
        weighted_f1 = sum(per_class_results[cls]['f1'] * per_class_results[cls]['support'] for cls in classes) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0
    
    overall_accuracy = correct_predictions / total_comparisons if total_comparisons > 0 else 0.0
    
    results = {
        'overall_accuracy': overall_accuracy,
        'correct_predictions': correct_predictions,
        'total_comparisons': total_comparisons,
        'per_class': per_class_results,
        'macro_avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'weighted_avg': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        }
    }
    
    return overall_accuracy, correct_predictions, total_comparisons, results

def aggregate_results(all_results):
    """Aggregate results from multiple files."""
    if not all_results:
        return None
    
    # Initialize aggregated metrics
    classes = ['red', 'green', 'yellow', 'left-turn', 'unknown']
    aggregated_class_metrics = {}
    for cls in classes:
        aggregated_class_metrics[cls] = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_negatives': 0
        }
    
    total_correct = 0
    total_comparisons = 0
    
    # Aggregate across all files
    for result in all_results:
        total_correct += result['correct_predictions']
        total_comparisons += result['total_comparisons']
        
        for cls in classes:
            metrics = result['per_class'][cls]
            aggregated_class_metrics[cls]['true_positives'] += metrics['tp']
            aggregated_class_metrics[cls]['false_positives'] += metrics['fp']
            aggregated_class_metrics[cls]['false_negatives'] += metrics['fn']
            aggregated_class_metrics[cls]['true_negatives'] += metrics.get('true_negatives', 0)
    
    # Calculate aggregated per-class metrics
    per_class_results = {}
    for cls in classes:
        metrics = aggregated_class_metrics[cls]
        tp = metrics['true_positives']
        fp = metrics['false_positives']
        fn = metrics['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_results[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    # Calculate macro and weighted averages
    macro_precision = sum(per_class_results[cls]['precision'] for cls in classes) / len(classes)
    macro_recall = sum(per_class_results[cls]['recall'] for cls in classes) / len(classes)
    macro_f1 = sum(per_class_results[cls]['f1'] for cls in classes) / len(classes)
    
    total_support = sum(per_class_results[cls]['support'] for cls in classes)
    if total_support > 0:
        weighted_precision = sum(per_class_results[cls]['precision'] * per_class_results[cls]['support'] for cls in classes) / total_support
        weighted_recall = sum(per_class_results[cls]['recall'] * per_class_results[cls]['support'] for cls in classes) / total_support
        weighted_f1 = sum(per_class_results[cls]['f1'] * per_class_results[cls]['support'] for cls in classes) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0
    
    overall_accuracy = total_correct / total_comparisons if total_comparisons > 0 else 0.0
    
    return {
        'overall_accuracy': overall_accuracy,
        'correct_predictions': total_correct,
        'total_comparisons': total_comparisons,
        'per_class': per_class_results,
        'macro_avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'weighted_avg': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        }
    }

def print_results(results, title="RESULTS"):
    """Print formatted results."""
    print(f"\n=== {title} ===")
    print(f"Correct predictions: {results['correct_predictions']}")
    print(f"Total comparisons: {results['total_comparisons']}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
    
    print(f"\n=== PER-CLASS METRICS ===")
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<8} {'TP':<5} {'FP':<5} {'FN':<5}")
    print("-" * 75)
    
    for cls, metrics in results['per_class'].items():
        print(f"{cls:<12} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
              f"{metrics['f1']:<10.4f} {metrics['support']:<8} {metrics['tp']:<5} "
              f"{metrics['fp']:<5} {metrics['fn']:<5}")
    
    print("-" * 75)
    print(f"{'Macro Avg':<12} {results['macro_avg']['precision']:<10.4f} "
          f"{results['macro_avg']['recall']:<10.4f} {results['macro_avg']['f1']:<10.4f}")
    print(f"{'Weighted Avg':<12} {results['weighted_avg']['precision']:<10.4f} "
          f"{results['weighted_avg']['recall']:<10.4f} {results['weighted_avg']['f1']:<10.4f}")
    
    # Show class distribution
    print(f"\n=== CLASS DISTRIBUTION ===")
    total_gt_instances = sum(metrics['support'] for metrics in results['per_class'].values())
    for cls, metrics in results['per_class'].items():
        percentage = (metrics['support'] / total_gt_instances * 100) if total_gt_instances > 0 else 0
        print(f"{cls:<12}: {metrics['support']:>6} instances ({percentage:>5.1f}%)")
    print(f"{'Total':<12}: {total_gt_instances:>6} instances")

def test_traffic_light_labeling(label_file_path, input_root=None, output_root=None):
    """Test traffic light labeling accuracy."""
    
    print(f"Reading label file: {label_file_path}")
    
    # Read the label file
    with open(label_file_path, 'r') as f:
        label_data = json.load(f)
    
    if not label_data:
        print("Error: Label file is empty")
        return None
    
    # Extract clip ID from the first element
    first_entry = label_data[0]
    image_name = first_entry['image_name']
    clip_id = extract_clip_id_from_image_name(image_name)
    
    print(f"Extracted clip ID: {clip_id}")
    print(f"Label data contains {len(label_data)} entries")
    
    # Set default paths if not provided
    if input_root is None:
        input_root = "/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_holodeck/cosmos-mads-dataset-all-in-one"
    
    if output_root is None:
        output_root = tempfile.mkdtemp(prefix="traffic_light_test_")
        print(f"Using temporary output directory: {output_root}")
    
    # Load dataset settings
    settings_file = 'config/dataset_rds_hq_front_mv.json'
    try:
        with open(settings_file, 'r') as f:
            settings = json.load(f)
    except FileNotFoundError:
        print(f"Error: Settings file '{settings_file}' not found")
        print("Using default settings")
        settings = {
            'CAMERAS': ['camera_front_wide_120fov', 'camera_cross_left_120fov', 'camera_cross_right_120fov']
        }
    
    print(f"Processing clip with settings: {len(settings.get('CAMERAS', []))} cameras")
    
    try:
        # Run crop_traffic_lights on the clip
        print("Running crop_traffic_lights...")
        crop_traffic_lights(
            input_root=input_root,
            output_root=output_root,
            clip_id=clip_id,
            settings=settings,
            camera_type='ftheta',
            post_training=False,
            novel_pose_folder=None,
            n_skip_frames=29
        )
        
        # Read the results
        results_file = Path(output_root) / "traffic_light_states" / f"{clip_id}.tar"
        
        if not results_file.exists():
            print(f"Error: Results file not found at {results_file}")
            return None
        
        # Extract aggregated results
        from utils.wds_utils import get_sample
        results_data = get_sample(str(results_file))
        
        if 'aggregated_states.json' not in results_data:
            print("Error: aggregated_states.json not found in results")
            return None
        
        aggregated_data = results_data['aggregated_states.json']
        predicted_states = aggregated_data['traffic_light_states']
        
        print(f"Loaded predicted states: {len(predicted_states)} traffic lights")
        
        # Calculate accuracy
        accuracy, correct, total, results = calculate_accuracy(label_data, predicted_states, clip_id)
        
        print_results(results, f"ACCURACY RESULTS - {clip_id}")
        
        # Clean up temporary directory if we created it
        if output_root.startswith('/tmp'):
            import shutil
            shutil.rmtree(output_root)
            print(f"\nCleaned up temporary directory: {output_root}")
        
        return results
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run the test."""
    parser = argparse.ArgumentParser(
        description="Test traffic light labeling accuracy on multiple label files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all JSON files in a folder
  python test_traffic_light_label.py traffic_lights_status_894_clips/tl_review/
  
  # Process with custom input and output paths
  python test_traffic_light_label.py traffic_lights_status_894_clips/tl_review/ \\
    --input-root /path/to/dataset --output-root /path/to/output
  
  # Process without aggregating results
  python test_traffic_light_label.py traffic_lights_status_894_clips/tl_review/ \\
    --no-aggregate
        """
    )
    
    parser.add_argument(
        'label_path',
        help='Path to folder containing JSON label files, or path to a single JSON file'
    )
    
    parser.add_argument(
        '--input-root',
        help='Root directory of the input dataset (default: /lustre/fsw/portfolios/nvr/projects/nvr_torontoai_holodeck/cosmos-mads-dataset-all-in-one)'
    )
    
    parser.add_argument(
        '--output-root',
        help='Root directory for output files (default: temporary directory)'
    )
    
    parser.add_argument(
        '--no-aggregate',
        action='store_true',
        help='Do not show aggregated results across all files'
    )
    
    parser.add_argument(
        '--pattern',
        default='*.json',
        help='File pattern to match in the folder (default: *.json)'
    )
    
    args = parser.parse_args()
    
    label_path = Path(args.label_path)
    
    if not label_path.exists():
        print(f"Error: Path '{label_path}' not found")
        return 1
    
    # Determine if we're processing a single file or multiple files
    if label_path.is_file():
        label_files = [label_path]
    elif label_path.is_dir():
        label_files = list(label_path.glob(args.pattern))
        if not label_files:
            print(f"Error: No files matching pattern '{args.pattern}' found in '{label_path}'")
            return 1
    else:
        print(f"Error: '{label_path}' is neither a file nor a directory")
        return 1
    
    print(f"Found {len(label_files)} label file(s) to process")
    
    # Process each file
    all_results = []
    successful_files = []
    
    for i, label_file in enumerate(label_files, 1):
        print(f"\n{'='*80}")
        print(f"Processing file {i}/{len(label_files)}: {label_file}")
        print(f"{'='*80}")
        
        try:
            result = test_traffic_light_labeling(
                str(label_file),
                args.input_root,
                args.output_root
            )
            
            if result is not None:
                all_results.append(result)
                successful_files.append(label_file)
                
                # Print current aggregated results after each file (if multiple files and not disabled)
                if not args.no_aggregate and len(all_results) > 1:
                    print(f"\n{'='*80}")
                    current_aggregated = aggregate_results(all_results)
                    if current_aggregated:
                        print_results(current_aggregated, f"CURRENT AGGREGATED RESULTS ({len(all_results)} files processed)")
                elif len(all_results) == 1:
                    # For the first file, just show a summary
                    print(f"\n{'='*80}")
                    print(f"FIRST FILE PROCESSED - BASELINE METRICS")
                    print(f"{'='*80}")
                    
            else:
                print(f"Failed to process {label_file}")
                
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Show summary and aggregated results
    print(f"\n{'='*80}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total files found: {len(label_files)}")
    print(f"Successfully processed: {len(successful_files)}")
    print(f"Failed: {len(label_files) - len(successful_files)}")
    
    if successful_files:
        print(f"\nSuccessful files:")
        for f in successful_files:
            print(f"  - {f}")
    
    # Show aggregated results
    if not args.no_aggregate and len(all_results) > 1:
        print(f"\n{'='*80}")
        aggregated = aggregate_results(all_results)
        if aggregated:
            print_results(aggregated, f"AGGREGATED RESULTS ({len(all_results)} files)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
