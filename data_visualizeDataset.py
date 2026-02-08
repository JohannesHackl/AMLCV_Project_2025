import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
import random

# Configuration
DATASET_PATH = 'merged_dataset_augError'
SPLITS = ['train', 'valid', 'test']

# Colors for each category (RGB)
CATEGORY_COLORS = {
    1: (255, 0, 0),      # box - red
    2: (0, 255, 0),      # bag - green
    3: (0, 0, 255)       # barcode - blue
}

CATEGORY_NAMES = {
    1: 'box',
    2: 'bag',
    3: 'barcode'
}

def draw_segmentation(image, annotation, color, alpha=0.5):
    """Draw segmentation polygons on image"""
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, 'RGBA')
    
    if 'segmentation' in annotation:
        for polygon in annotation['segmentation']:
            if len(polygon) >= 6:
                # Convert flat list to list of tuples
                points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                # Draw filled polygon
                draw.polygon(points, fill=color + (int(255 * alpha),), outline=color + (255,), width=2)
    
    return Image.blend(image, overlay, alpha)

def draw_bbox(draw, bbox, color, label_text):
    """Draw bounding box with label"""
    x, y, w, h = bbox
    # Draw rectangle
    draw.rectangle([x, y, x + w, y + h], outline=color + (255,), width=2)
    # Draw label background
    draw.rectangle([x, y - 15, x + len(label_text) * 8, y], fill=color + (200,))
    # Draw label text
    draw.text((x + 2, y - 13), label_text, fill=(255, 255, 255, 255))

def visualize_sample(image_path, annotations, save_path=None, show=True):
    """Visualize a single image with its annotations"""
    # Load image
    image = Image.open(image_path).convert('RGBA')
    
    # Draw segmentations
    for ann in annotations:
        cat_id = ann['category_id']
        color = CATEGORY_COLORS.get(cat_id, (128, 128, 128))
        image = draw_segmentation(image, ann, color, alpha=0.4)
    
    # Draw bboxes on top
    draw = ImageDraw.Draw(image)
    for ann in annotations:
        cat_id = ann['category_id']
        color = CATEGORY_COLORS.get(cat_id, (128, 128, 128))
        label = CATEGORY_NAMES.get(cat_id, 'unknown')
        draw_bbox(draw, ann['bbox'], color, label)
    
    # Convert back to RGB for display/save
    image = image.convert('RGB')
    
    if save_path:
        image.save(save_path)
    
    if show:
        plt.figure(figsize=(12, 12))
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return image

def visualize_random_samples(split='train', num_samples=5, save_dir=None):
    """Visualize random samples from a split"""
    dataset_path = Path(DATASET_PATH)
    ann_file = dataset_path / split / '_annotations.coco.json'
    img_dir = dataset_path / split
    
    # Load annotations
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Group annotations by image
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Select random images that have annotations
    images_with_anns = [img for img in coco_data['images'] if img['id'] in img_to_anns]
    
    if len(images_with_anns) < num_samples:
        print(f"Warning: Only {len(images_with_anns)} images with annotations available")
        num_samples = len(images_with_anns)
    
    selected_images = random.sample(images_with_anns, num_samples)
    
    # Create save directory if specified
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nVisualizing {num_samples} random samples from {split} split...")
    print("="*60)
    
    for idx, img_info in enumerate(selected_images):
        img_path = img_dir / img_info['file_name']
        annotations = img_to_anns[img_info['id']]
        
        print(f"\nImage {idx+1}/{num_samples}: {img_info['file_name']}")
        print(f"  Annotations: {len(annotations)}")
        # for ann in annotations:
        #     cat_name = CATEGORY_NAMES.get(ann['category_id'], 'unknown')
        #     print(f"    - {cat_name}")
        
        save_file = None
        if save_dir:
            save_file = Path(save_dir) / f"{split}_sample_{idx+1}_{img_info['file_name']}"
        
        visualize_sample(img_path, annotations, save_path=save_file, show=False)
    
    print(f"\n✓ Visualization complete!")
    if save_dir:
        print(f"  Saved to: {save_dir}")

def visualize_specific_image(split, image_filename, save_path=None):
    """Visualize a specific image by filename"""
    dataset_path = Path(DATASET_PATH)
    ann_file = dataset_path / split / '_annotations.coco.json'
    img_dir = dataset_path / split
    
    # Load annotations
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Find the image
    img_info = None
    for img in coco_data['images']:
        if img['file_name'] == image_filename:
            img_info = img
            break
    
    if not img_info:
        print(f"Error: Image {image_filename} not found in {split} split")
        return
    
    # Get annotations for this image
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_info['id']]
    
    img_path = img_dir / image_filename
    
    print(f"\nVisualizing: {image_filename}")
    print(f"Annotations: {len(annotations)}")
    for ann in annotations:
        cat_name = CATEGORY_NAMES.get(ann['category_id'], 'unknown')
        print(f"  - {cat_name}")
    
    visualize_sample(img_path, annotations, save_path=save_path, show=True)

def create_comparison_grid(split='train', num_samples=4, save_path='visualization_grid.png'):
    """Create a grid showing multiple samples side by side"""
    dataset_path = Path(DATASET_PATH)
    ann_file = dataset_path / split / '_annotations.coco.json'
    img_dir = dataset_path / split
    
    # Load annotations
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Group annotations by image
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Select random images
    images_with_anns = [img for img in coco_data['images'] if img['id'] in img_to_anns]
    selected_images = random.sample(images_with_anns, min(num_samples, len(images_with_anns)))
    
    # Create grid
    n_cols = 2
    n_rows = (len(selected_images) + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_info in enumerate(selected_images):
        row = idx // n_cols
        col = idx % n_cols
        
        img_path = img_dir / img_info['file_name']
        annotations = img_to_anns[img_info['id']]
        
        # Visualize
        vis_img = visualize_sample(img_path, annotations, save_path=None, show=False)
        
        # Plot
        axes[row, col].imshow(vis_img)
        axes[row, col].axis('off')
        
        # Title with annotation counts
        cat_counts = {}
        for ann in annotations:
            cat_name = CATEGORY_NAMES.get(ann['category_id'], 'unknown')
            cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
        
        title = f"{img_info['file_name']}\n"
        title += ", ".join([f"{name}: {count}" for name, count in cat_counts.items()])
        axes[row, col].set_title(title, fontsize=10)
    
    # Hide empty subplots
    for idx in range(len(selected_images), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Grid saved to: {save_path}")
    
    plt.show()

def check_dataset_stats(split):
    """Print dataset statistics"""
    dataset_path = Path(DATASET_PATH)
    ann_file = dataset_path / split / '_annotations.coco.json'
    
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Dataset Statistics - {split.upper()} Split")
    print(f"{'='*60}")
    
    print(f"\nTotal Images: {len(coco_data['images'])}")
    print(f"Total Annotations: {len(coco_data['annotations'])}")
    
    # Count by category
    cat_counts = {}
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        cat_name = CATEGORY_NAMES.get(cat_id, 'unknown')
        cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
    
    print(f"\nAnnotations by Category:")
    for cat_name, count in sorted(cat_counts.items()):
        print(f"  {cat_name}: {count}")
    
    # Images per category
    img_to_cats = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        if img_id not in img_to_cats:
            img_to_cats[img_id] = set()
        img_to_cats[img_id].add(cat_id)
    
    cat_img_counts = {}
    for cat_id in CATEGORY_NAMES.keys():
        count = sum(1 for cats in img_to_cats.values() if cat_id in cats)
        cat_img_counts[CATEGORY_NAMES[cat_id]] = count
    
    print(f"\nImages containing each category:")
    for cat_name, count in sorted(cat_img_counts.items()):
        print(f"  {cat_name}: {count}")

if __name__ == '__main__':
    # Example usage
    print("Dataset Visualization Tool")
    print("="*60)
    
    # Check stats for all splits
    for split in SPLITS:
        split_path = Path(DATASET_PATH) / split
        if split_path.exists():
            check_dataset_stats(split)
    
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    # Visualize random samples from train
    visualize_random_samples(split='train', num_samples=3, save_dir='visualizations/train')
    
    # Visualize random samples from valid
    visualize_random_samples(split='valid', num_samples=3, save_dir='visualizations/valid')
    
    # Create comparison grid
    create_comparison_grid(split='train', num_samples=6, save_path='visualizations/train_grid.png')
    
    print("\n" + "="*60)
    print("✓ All visualizations complete!")
    print("="*60)
    
    # Uncomment to visualize a specific image:
    # visualize_specific_image('train', 'dataset1_image_001.jpg', save_path='specific_vis.png')