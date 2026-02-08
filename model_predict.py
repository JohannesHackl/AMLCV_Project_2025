import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import seaborn as sns

# Configuration
DATASET_PATH = 'merged_dataset'
MODEL_PATH = 'models/best_model.pth'
OUTPUT_DIR = 'visualization_results'
SPLIT = 'valid'  # or 'test' or 'train'
NUM_CLASSES = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class info
CLASS_NAMES = ['background', 'box', 'bag', 'barcode']
CLASS_COLORS = {
    0: [0, 0, 0],         # background - black
    1: [255, 77, 77],     # box - red
    2: [77, 255, 77],     # bag - green
    3: [77, 77, 255]      # barcode - blue
}

print(f"Using device: {DEVICE}")

class COCOSegmentationDataset(Dataset):
    """COCO format dataset for semantic segmentation"""
    
    def __init__(self, dataset_path, split='train'):
        self.dataset_path = Path(dataset_path)
        self.split = split
        
        # Load annotations
        ann_file = self.dataset_path / split / '_annotations.coco.json'
        
        print(f"Loading annotations from: {ann_file}")
        
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # Group annotations by image
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        print(f"{split} dataset: {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        
        # Load image
        img_path = self.dataset_path / self.split / img_info['file_name']
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        
        # Create segmentation mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        # Fill mask with annotations
        if img_id in self.img_to_anns:
            for ann in self.img_to_anns[img_id]:
                category_id = ann['category_id']
                
                if 'segmentation' in ann and ann['segmentation']:
                    for polygon in ann['segmentation']:
                        if len(polygon) >= 6:
                            pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(mask, [pts], category_id)
        
        # Convert to tensors
        image_array = np.array(image)
        image_tensor = image_array.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).long()
        
        return image_tensor, mask_tensor, image_array, mask, img_info['file_name']

def get_model(num_classes):
    """Load model architecture"""
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights)
    model.classifier = DeepLabHead(960, num_classes)  # type: ignore
    model.aux_classifier = FCNHead(40, num_classes)  # type: ignore
    return model

def mask_to_color(mask, colors):
    """Convert class mask to RGB colored image"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        color_mask[mask == class_id] = color
    return color_mask

def create_overlay(image, mask, alpha=0.5):
    """Create semi-transparent overlay of mask on image"""
    colored_mask = mask_to_color(mask, CLASS_COLORS)
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    return overlay

def visualize_single_prediction(image, gt_mask, pred_mask, filename, save_path):
    """Create detailed visualization for a single image"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Image, Ground Truth, Prediction
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    gt_colored = mask_to_color(gt_mask, CLASS_COLORS)
    axes[0, 1].imshow(gt_colored)
    axes[0, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    pred_colored = mask_to_color(pred_mask, CLASS_COLORS)
    axes[0, 2].imshow(pred_colored)
    axes[0, 2].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Overlays and Error Map
    gt_overlay = create_overlay(image, gt_mask, alpha=0.4)
    axes[1, 0].imshow(gt_overlay)
    axes[1, 0].set_title('Ground Truth Overlay', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    pred_overlay = create_overlay(image, pred_mask, alpha=0.4)
    axes[1, 1].imshow(pred_overlay)
    axes[1, 1].set_title('Prediction Overlay', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Error map: correct = green, incorrect = red
    error_map = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    correct = gt_mask == pred_mask
    error_map[correct] = [0, 255, 0]  # Green for correct
    error_map[~correct] = [255, 0, 0]  # Red for incorrect
    
    axes[1, 2].imshow(error_map)
    axes[1, 2].set_title('Error Map (Green=Correct, Red=Wrong)', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Add legend
    legend_elements = [mpatches.Patch(facecolor=tuple(np.array(CLASS_COLORS[i])/255), 
                                     label=CLASS_NAMES[i]) 
                      for i in range(NUM_CLASSES)]
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.98), ncol=NUM_CLASSES, fontsize=12)
    
    plt.suptitle(f'Segmentation Results: {filename}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path.name}")

def create_grid_visualization(images_data, save_path, grid_size=(4, 4)):
    """Create a grid of multiple predictions"""
    
    print(f"\n  Creating grid visualization with {len(images_data)} samples...")
    
    rows, cols = grid_size
    num_samples = min(len(images_data), rows * cols)
    
    fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 9, rows * 3))
    
    # Ensure axes is 2D array
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        row = idx // cols
        col = idx % cols
        
        image, gt_mask, pred_mask = images_data[idx]
        
        # Original image
        axes[row, col*3].imshow(image)
        axes[row, col*3].axis('off')
        if row == 0:
            axes[row, col*3].set_title('Image', fontsize=10, fontweight='bold')
        
        # Ground truth
        gt_colored = mask_to_color(gt_mask, CLASS_COLORS)
        axes[row, col*3 + 1].imshow(gt_colored)
        axes[row, col*3 + 1].axis('off')
        if row == 0:
            axes[row, col*3 + 1].set_title('Ground Truth', fontsize=10, fontweight='bold')
        
        # Prediction
        pred_colored = mask_to_color(pred_mask, CLASS_COLORS)
        axes[row, col*3 + 2].imshow(pred_colored)
        axes[row, col*3 + 2].axis('off')
        if row == 0:
            axes[row, col*3 + 2].set_title('Prediction', fontsize=10, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col*3].axis('off')
        axes[row, col*3 + 1].axis('off')
        axes[row, col*3 + 2].axis('off')
    
    # Add legend
    legend_elements = [mpatches.Patch(facecolor=tuple(np.array(CLASS_COLORS[i])/255), 
                                     label=CLASS_NAMES[i]) 
                      for i in range(NUM_CLASSES)]
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.99), ncol=NUM_CLASSES, fontsize=12)
    
    plt.suptitle('Segmentation Results Grid', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")

def create_class_specific_visualizations(images_data, save_dir):
    """Create visualizations for each class separately"""
    
    print("\n  Creating class-specific visualizations...")
    
    for class_id in range(1, NUM_CLASSES):  # Skip background
        class_name = CLASS_NAMES[class_id]
        
        print(f"    Processing class: {class_name}")
        
        fig, axes = plt.subplots(4, 6, figsize=(24, 16))
        axes = axes.flatten()
        
        count = 0
        for image, gt_mask, pred_mask, filename in images_data:
            if count >= 24:
                break
            
            # Check if this class exists in ground truth
            if class_id not in gt_mask:
                continue
            
            # Create binary masks for this class
            gt_binary = (gt_mask == class_id).astype(np.uint8) * 255
            pred_binary = (pred_mask == class_id).astype(np.uint8) * 255
            
            # Overlay on image
            overlay = image.copy()
            overlay[gt_binary > 0] = overlay[gt_binary > 0] * 0.5 + np.array(CLASS_COLORS[class_id]) * 0.5
            
            # Show prediction contours
            contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            overlay_with_pred = overlay.copy()
            cv2.drawContours(overlay_with_pred, contours, -1, (255, 255, 0), 2)
            
            axes[count].imshow(overlay_with_pred.astype(np.uint8))
            axes[count].set_title(f'{filename[:20]}...', fontsize=8)
            axes[count].axis('off')
            
            count += 1
        
        # Hide unused subplots
        for idx in range(count, 24):
            axes[idx].axis('off')
        
        plt.suptitle(f'Class: {class_name.upper()} (Ground Truth=filled, Prediction=yellow outline)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_path = save_dir / f'class_{class_name}_visualization.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"      ✓ Saved: {save_path.name} ({count} examples)")

def create_confusion_heatmap(confusion_matrix, save_path):
    """Create a heatmap of the confusion matrix"""
    
    print("\n  Creating confusion matrix heatmap...")
    
    # Normalize confusion matrix
    confusion_norm = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-6)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Pixel Counts)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('Ground Truth', fontsize=12)
    
    # Normalized
    sns.heatmap(confusion_norm, annot=True, fmt='.3f', cmap='RdYlGn', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('Ground Truth', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")

def visualize_predictions():
    """Main visualization function"""
    
    print("="*70)
    print("STARTING VISUALIZATION PIPELINE")
    print("="*70)
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_path.absolute()}")
    
    individual_dir = output_path / 'individual_predictions'
    individual_dir.mkdir(exist_ok=True)
    print(f"Individual predictions directory: {individual_dir.absolute()}")
    
    # Load dataset
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    dataset = COCOSegmentationDataset(DATASET_PATH, split=SPLIT)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    print(f"Model path: {MODEL_PATH}")
    
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model = get_model(NUM_CLASSES)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    print(f"✓ Validation loss: {checkpoint['val_loss']:.4f}")
    
    # Collect predictions
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS")
    print("="*70)
    
    all_data = []
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    
    with torch.no_grad():
        for idx, (image_tensor, mask_tensor, image_array, mask_array, filename) in enumerate(tqdm(dataloader, desc="Processing images")):
            image_tensor = image_tensor.to(DEVICE)
            
            # Predict
            output = model(image_tensor)
            pred = output['out'].argmax(dim=1).squeeze(0).cpu().numpy()
            
            # Get arrays
            image = image_array[0].numpy()
            gt_mask = mask_array[0].numpy()
            
            # Update confusion matrix
            for i in range(NUM_CLASSES):
                for j in range(NUM_CLASSES):
                    confusion_matrix[i, j] += np.sum((gt_mask == i) & (pred == j))
            
            # Store data
            all_data.append((image, gt_mask, pred, filename[0]))
    
    print(f"\n✓ Generated {len(all_data)} predictions")
    
    # Create individual visualizations
    print("\n" + "="*70)
    print("CREATING INDIVIDUAL VISUALIZATIONS (first 20)")
    print("="*70)
    
    for idx in range(min(20, len(all_data))):
        image, gt_mask, pred, filename = all_data[idx]
        save_path = individual_dir / f'prediction_{idx:03d}_{filename}'
        visualize_single_prediction(image, gt_mask, pred, filename, save_path)
    
    print(f"\n✓ Created {min(20, len(all_data))} individual visualizations")
    
    # Create grid visualization
    print("\n" + "="*70)
    print("CREATING GRID VISUALIZATION")
    print("="*70)
    grid_data = [(img, gt, pred) for img, gt, pred, _ in all_data[:16]]
    create_grid_visualization(grid_data, output_path / 'predictions_grid.png', grid_size=(4, 4))
    
    # Create class-specific visualizations
    print("\n" + "="*70)
    print("CREATING CLASS-SPECIFIC VISUALIZATIONS")
    print("="*70)
    create_class_specific_visualizations(all_data, output_path)
    
    # Create confusion matrix heatmap
    print("\n" + "="*70)
    print("CREATING CONFUSION MATRIX")
    print("="*70)
    create_confusion_heatmap(confusion_matrix, output_path / 'confusion_matrix.png')
    
    # Calculate and display metrics
    print("\n" + "="*70)
    print("FINAL METRICS")
    print("="*70)
    
    iou_per_class = []
    for i in range(NUM_CLASSES):
        intersection = confusion_matrix[i, i]
        union = confusion_matrix[i, :].sum() + confusion_matrix[:, i].sum() - intersection
        iou = intersection / (union + 1e-6)
        iou_per_class.append(iou)
        print(f"{CLASS_NAMES[i]:12s}: IoU = {iou:.4f}")
    
    mean_iou = np.mean(iou_per_class)
    print(f"\nMean IoU: {mean_iou:.4f}")
    
    # Calculate pixel accuracy
    pixel_accuracy = np.trace(confusion_matrix) / (np.sum(confusion_matrix) + 1e-6)
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    
    print("\n" + "="*70)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_path.absolute()}/")
    print(f"  - individual_predictions/ ({min(20, len(all_data))} detailed predictions)")
    print(f"  - predictions_grid.png (4x4 grid overview)")
    print(f"  - class_box_visualization.png")
    print(f"  - class_bag_visualization.png")
    print(f"  - class_barcode_visualization.png")
    print(f"  - confusion_matrix.png")
    print("\n" + "="*70)

if __name__ == '__main__':
    try:
        visualize_predictions()
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR OCCURRED:")
        print(f"{'='*70}")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}")