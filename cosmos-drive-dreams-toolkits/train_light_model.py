#!/usr/bin/env python3
import os
# Set HF_HOME environment variable
os.environ['HF_HOME'] = '/lustre/fsw/portfolios/nvr/users/tianshic/huggingface/'

import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
# from transformers import CLIPConfig, CLIPModel, CLIPProcessor
from transformers import AutoProcessor, AutoModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import click
from PIL import Image
import warnings
import random
warnings.filterwarnings("ignore")

class TrafficLightCLIPClassifier(nn.Module):
    """Traffic light classifier using CLIP backbone with MLP head."""
    
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", num_classes=5, hidden_dim=512, dropout=0.3):
        super().__init__()
        
        # Load CLIP model from transformers
        # self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        # self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
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
        
        print(f"CLIP feature dimension: {clip_dim}")
        
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

def simple_train_test_split(indices, labels, test_size=0.2, random_state=42):
    """Simple stratified train-test split implementation."""
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Group indices by class
    class_indices = {}
    for idx, label in zip(indices, labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    train_indices = []
    test_indices = []
    
    # Split each class proportionally
    for label, class_idx_list in class_indices.items():
        random.shuffle(class_idx_list)
        n_test = int(len(class_idx_list) * test_size)
        test_indices.extend(class_idx_list[:n_test])
        train_indices.extend(class_idx_list[n_test:])
    
    return train_indices, test_indices

def calculate_metrics(y_true, y_pred, class_names):
    """Calculate classification metrics manually."""
    num_classes = len(class_names)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Per-class metrics
    class_metrics = {}
    
    for i, class_name in enumerate(class_names):
        # True positives, false positives, false negatives
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        tn = np.sum((y_true != i) & (y_pred != i))
        
        # Precision, recall, f1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn
        
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        }
    
    # Macro averages
    macro_precision = np.mean([metrics['precision'] for metrics in class_metrics.values()])
    macro_recall = np.mean([metrics['recall'] for metrics in class_metrics.values()])
    macro_f1 = np.mean([metrics['f1-score'] for metrics in class_metrics.values()])
    
    # Weighted averages
    total_support = sum([metrics['support'] for metrics in class_metrics.values()])
    if total_support > 0:
        weighted_precision = sum([metrics['precision'] * metrics['support'] for metrics in class_metrics.values()]) / total_support
        weighted_recall = sum([metrics['recall'] * metrics['support'] for metrics in class_metrics.values()]) / total_support
        weighted_f1 = sum([metrics['f1-score'] * metrics['support'] for metrics in class_metrics.values()]) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0
    
    report = {
        'accuracy': str(accuracy),
        'macro avg': {
            'precision': str(macro_precision),
            'recall': str(macro_recall),
            'f1-score': str(macro_f1),
            'support': int(total_support)
        },
        'weighted avg': {
            'precision': str(weighted_precision),
            'recall': str(weighted_recall),
            'f1-score': str(weighted_f1),
            'support': int(total_support)
        }
    }
    
    # Add per-class metrics
    for class_name, metrics in class_metrics.items():
        report[class_name] = str(metrics)
    
    return report

def simple_confusion_matrix(y_true, y_pred, num_classes):
    """Simple confusion matrix implementation."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label][pred_label] += 1
    return cm

def get_overall_class_distribution(dataset_path):
    """Get overall class distribution from the dataset path for information."""
    class_counts = {}
    
    for class_name in os.listdir(dataset_path):
        class_dir = Path(dataset_path) / class_name
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg")))
            class_counts[class_name] = count
    
    print("Overall dataset class distribution:")
    total_samples = sum(class_counts.values())
    for class_name, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    return class_counts

def get_class_weights_from_subset(dataset, subset_indices):
    """Calculate class weights for balanced sampling from a subset of the dataset."""
    class_counts = {}
    
    # Count classes only in the subset
    for idx in subset_indices:
        _, class_idx = dataset.samples[idx]
        class_name = list(dataset.class_to_idx.keys())[list(dataset.class_to_idx.values()).index(class_idx)]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("Training set class distribution:")
    total_samples = sum(class_counts.values())
    for class_name, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    return class_counts

def create_weighted_sampler(subset_dataset, subset_indices, class_counts):
    """Create weighted random sampler for class balancing on a subset."""
    # Get the underlying dataset
    if hasattr(subset_dataset, 'dataset'):
        # This is a Subset object
        full_dataset = subset_dataset.dataset
    else:
        full_dataset = subset_dataset
    
    # Map class names to indices
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Calculate weights (inverse frequency)
    total_samples = sum(class_counts.values())
    class_weights = {}
    for class_name, count in class_counts.items():
        class_weights[class_name] = total_samples / (len(class_counts) * count)
    
    # Create sample weights only for the subset
    sample_weights = []
    for idx in subset_indices:
        _, class_idx = full_dataset.samples[idx]
        class_name = idx_to_class[class_idx]
        sample_weights.append(class_weights[class_name])
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def create_transforms():
    """Create data transforms."""
    # CLIP-style preprocessing with augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    return train_transform, val_transform

def split_dataset(dataset, val_split=0.2, random_state=42):
    """Split dataset into train and validation sets."""
    # Get all indices
    indices = list(range(len(dataset)))
    
    # Get labels for stratified split
    labels = [dataset.samples[i][1] for i in indices]
    
    # Stratified split to maintain class distribution
    train_indices, val_indices = simple_train_test_split(
        indices, 
        labels,
        test_size=val_split, 
        random_state=random_state
    )
    
    return train_indices, val_indices

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Update progress bar
        accuracy = 100. * correct / total
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{accuracy:.2f}%'})
    
    return total_loss / len(dataloader), correct / total

def validate_epoch(model, dataloader, criterion, device, class_names):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
            
            # Update progress bar
            accuracy = 100. * correct / total
            avg_loss = total_loss / len(dataloader.dataset) * dataloader.batch_size
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{accuracy:.2f}%'})
    
    # Calculate detailed metrics
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    
    # Classification report
    report = calculate_metrics(all_targets, all_preds, class_names)
    
    return avg_loss, accuracy, report, all_preds, all_targets

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = simple_confusion_matrix(y_true, y_pred, len(class_names))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_training_curves(train_losses, train_accs, val_losses, val_accs, save_path):
    """Plot and save training curves."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_model(model_path, data_path, output_dir, batch_size=32):
    """Test mode: Load best model and analyze misclassified samples."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model checkpoint
    print(f"Loading model from {model_path}...")
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
    
    print(f"Model loaded successfully!")
    print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
    print(f"Classes: {class_names}")
    
    # Create test dataset (same as validation)
    _, val_transform = create_transforms()
    test_dataset = datasets.ImageFolder(root=data_path, transform=val_transform)
    
    # Use same validation split as training
    train_indices, val_indices = split_dataset(test_dataset, config.get('val_split', 0.2))
    val_dataset = Subset(test_dataset, val_indices)
    
    test_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    # Create output directory for misclassified samples
    misclassified_dir = Path(output_dir) / "misclassified_analysis"
    misclassified_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference and collect misclassified samples
    print("Running inference on validation set...")
    misclassified_samples = []
    correct_count = 0
    total_count = 0
    
    # Also create per-class confusion directories
    for gt_class in class_names:
        for pred_class in class_names:
            if gt_class != pred_class:
                confusion_dir = misclassified_dir / f"GT_{gt_class}_PRED_{pred_class}"
                confusion_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing")):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            # Check each sample in the batch
            for i in range(data.size(0)):
                sample_idx = batch_idx * batch_size + i
                if sample_idx >= len(val_dataset):
                    break
                    
                actual_idx = val_indices[sample_idx]
                original_path = Path(test_dataset.samples[actual_idx][0])
                
                gt_label = target[i].item()
                pred_label = pred[i].item()
                gt_class = class_names[gt_label]
                pred_class = class_names[pred_label]
                
                total_count += 1
                
                if gt_label == pred_label:
                    correct_count += 1
                else:
                    # Misclassified sample
                    # Load original image (without normalization)
                    original_img = Image.open(original_path).convert('RGB')
                    
                    # Save to confusion-specific directory
                    confusion_dir = misclassified_dir / f"GT_{gt_class}_PRED_{pred_class}"
                    save_filename = f"{original_path.stem}_conf_{output[i].max().item():.3f}.jpg"
                    save_path = confusion_dir / save_filename
                    
                    # Save the image
                    original_img.save(save_path)
                    
                    # Store sample info
                    confidence = torch.softmax(output[i], dim=0).max().item()
                    misclassified_samples.append({
                        'original_path': str(original_path),
                        'saved_path': str(save_path),
                        'gt_class': gt_class,
                        'pred_class': pred_class,
                        'confidence': confidence,
                        'gt_label': gt_label,
                        'pred_label': pred_label
                    })
    
    # Calculate and print statistics
    accuracy = correct_count / total_count
    print(f"\nTest Results:")
    print(f"Total samples: {total_count}")
    print(f"Correct predictions: {correct_count}")
    print(f"Misclassified samples: {len(misclassified_samples)}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save misclassified samples report
    report_data = {
        'test_accuracy': accuracy,
        'total_samples': total_count,
        'correct_predictions': correct_count,
        'misclassified_count': len(misclassified_samples),
        'class_names': class_names,
        'misclassified_samples': misclassified_samples
    }
    
    with open(misclassified_dir / "misclassified_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    # Create a summary visualization
    create_misclassification_summary(misclassified_samples, class_names, misclassified_dir)
    
    print(f"\nMisclassified samples saved to: {misclassified_dir}")
    print("Folder structure:")
    print("  GT_{true_class}_PRED_{predicted_class}/")
    print("    - Contains images misclassified as predicted_class when true class was true_class")
    print("    - Filenames include confidence scores")

def create_misclassification_summary(misclassified_samples, class_names, output_dir):
    """Create a summary visualization of misclassifications."""
    if not misclassified_samples:
        print("No misclassified samples to visualize.")
        return
    
    # Create confusion matrix for misclassifications
    num_classes = len(class_names)
    confusion_counts = np.zeros((num_classes, num_classes), dtype=int)
    
    for sample in misclassified_samples:
        gt_idx = sample['gt_label']
        pred_idx = sample['pred_label']
        confusion_counts[gt_idx][pred_idx] += 1
    
    # Plot confusion matrix for misclassifications only
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_counts, annot=True, fmt='d', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of misclassifications'})
    plt.title('Misclassification Matrix\n(Shows count of GT class → Predicted class errors)')
    plt.xlabel('Predicted Class')
    plt.ylabel('Ground Truth Class')
    plt.tight_layout()
    plt.savefig(output_dir / "misclassification_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create confidence distribution plot
    confidences = [float(sample['confidence']) for sample in misclassified_samples]
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Confidence Distribution for Misclassified Samples')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Count')
    plt.axvline(float(np.mean(confidences)), color='red', linestyle='--',
                label=f'Mean: {float(np.mean(confidences)):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "misclassification_confidence_dist.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary statistics by class
    class_errors = {}
    for sample in misclassified_samples:
        gt_class = sample['gt_class']
        if gt_class not in class_errors:
            class_errors[gt_class] = []
        class_errors[gt_class].append(sample)
    
    print("\nMisclassification summary by class:")
    for class_name in class_names:
        if class_name in class_errors:
            errors = class_errors[class_name]
            avg_conf = np.mean([float(s['confidence']) for s in errors])
            print(f"  {class_name}: {len(errors)} errors, avg confidence: {avg_conf:.3f}")
        else:
            print(f"  {class_name}: 0 errors")

@click.command()
@click.option("--data_path", "-d", type=str, default="./output_train/cropped_images", help="Path to cropped_images directory")
@click.option("--output_dir", "-o", type=str, default="./training_output", help="Directory to save training outputs")
@click.option("--batch_size", "-b", type=int, default=32, help="Batch size for training")
@click.option("--epochs", "-e", type=int, default=50, help="Number of training epochs")
@click.option("--learning_rate", "-lr", type=float, default=1e-4, help="Learning rate")
@click.option("--val_split", "-vs", type=float, default=0.2, help="Validation split ratio")
@click.option("--clip_model", "-cm", type=str, default="openai/clip-vit-base-patch32", help="CLIP model variant")
@click.option("--hidden_dim", "-hd", type=int, default=512, help="Hidden dimension for MLP head")
@click.option("--dropout", "-dr", type=float, default=0.3, help="Dropout rate")
@click.option("--weight_decay", "-wd", type=float, default=1e-5, help="Weight decay")
@click.option("--save_interval", "-si", type=int, default=5, help="Save model every N epochs")
@click.option("--test_mode", "-t", is_flag=True, help="Test mode: analyze misclassified samples from trained model")
def main(data_path, output_dir, batch_size, epochs, learning_rate, val_split, 
         clip_model, hidden_dim, dropout, weight_decay, save_interval, test_mode):
    """Train traffic light classifier using CLIP backbone."""
    
    # Check if test mode
    if test_mode:
        model_path = Path(output_dir) / "best_model.pth"
        if not model_path.exists():
            print(f"Error: Model checkpoint not found at {model_path}")
            print("Please train a model first or specify the correct output directory.")
            return
        
        test_model(model_path, data_path, output_dir, batch_size)
        return
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training configuration
    config = {
        "data_path": data_path,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "val_split": val_split,
        "clip_model": clip_model,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "device": str(device)
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Load dataset
    print("Loading dataset...")
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise ValueError(f"Data path {data_path} does not exist")
    
    # Show overall dataset distribution
    overall_class_counts = get_overall_class_distribution(data_path)
    
    # Create transforms
    train_transform, val_transform = create_transforms()
    
    # Load full dataset to get class information
    full_dataset = datasets.ImageFolder(root=data_path, transform=train_transform)
    class_names = list(full_dataset.class_to_idx.keys())
    num_classes = len(class_names)
    
    print(f"Found {num_classes} classes: {class_names}")
    print(f"Total samples: {len(full_dataset)}")
    
    # Split dataset
    train_indices, val_indices = split_dataset(full_dataset, val_split)
    print(f"Train samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    
    # Get class weights for balanced sampling (based on training set only)
    class_counts = get_class_weights_from_subset(full_dataset, train_indices)
    
    # Create train and validation datasets
    train_dataset = Subset(full_dataset, train_indices)
    
    # Create validation dataset with validation transforms
    val_full_dataset = datasets.ImageFolder(root=data_path, transform=val_transform)
    val_dataset = Subset(val_full_dataset, val_indices)
    
    # Create weighted sampler for balanced training
    train_sampler = create_weighted_sampler(train_dataset, train_indices, class_counts)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print(f"Creating model with CLIP backbone: {clip_model}")
    model = TrafficLightCLIPClassifier(
        clip_model_name=clip_model,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler (fix verbose parameter)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("\nStarting training...")
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_report, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device, class_names
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'class_names': class_names,
                'config': config,
            }, output_dir / "best_model.pth")
            
            # Save detailed validation report
            with open(output_dir / "best_validation_report.json", "w") as f:
                json.dump(val_report, f, indent=2)
            
            # Save confusion matrix
            plot_confusion_matrix(
                val_targets, val_preds, class_names, 
                output_dir / "best_confusion_matrix.png"
            )
        
        # Save checkpoint periodically
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
                'class_names': class_names,
                'config': config,
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pth")
    
    # Save final model and results
    torch.save({
        'epoch': epochs-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'class_names': class_names,
        'config': config,
    }, output_dir / "final_model.pth")
    
    # Save training curves
    save_training_curves(
        train_losses, train_accs, val_losses, val_accs,
        output_dir / "training_curves.png"
    )
    
    # Final evaluation
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final validation accuracy: {val_acc:.4f}")
    
    # Save final metrics
    final_metrics = {
        "best_val_accuracy": float(best_val_acc),
        "final_val_accuracy": float(val_acc),
        "final_train_accuracy": float(train_acc),
        "total_epochs": epochs,
        "class_names": class_names,
        "class_distribution": class_counts
    }
    
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\nAll outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
