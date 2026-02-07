import json
import os
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

# Configuration
DATASETS_CONFIG = {
    'dataset1': {
        'path': 'D:/My Files/My Downloads/AMLCV25_Project/1_box_barcode',
        'label_map': {
            'boxes': 'box',
            'labels': 'barcode'
        },
        'exclude_labels': ['Background Box']
    },
    'dataset2': {
        'path': 'D:/My Files/My Downloads/AMLCV25_Project/2_box',
        'label_map': {
            'Carton': 'box'
        },
        'exclude_labels': []
    },
    'dataset3': {
        'path': 'D:/My Files/My Downloads/AMLCV25_Project/3_barcode',
        'label_map': {
            'Labels': 'barcode'
        },
        'exclude_labels': []
    },
    'dataset4': {
        'path': 'D:/My Files/My Downloads/AMLCV25_Project/4_box_bag',
        'label_map': {
            'Box': 'box',
            'Flyer': 'bag'
        },
        'exclude_labels': []
    },
    'dataset5': {
        'path': 'D:/My Files/My Downloads/AMLCV25_Project/5_barcode',
        'label_map': {
            'DHL': 'barcode',
            'FedEx Express': 'barcode',
            'FedEx Ground': 'barcode',
            'UPS 2ND DAY': 'barcode',
            'UPS Ground': 'barcode',
            'UPS Standard': 'barcode'
        },
        'exclude_labels': []
    },
}


TARGET_SIZE = (512, 512)
OUTPUT_DIR = 'merged_dataset'

# Unified label names (background=0 is implicit, not in annotations)
UNIFIED_CATEGORIES = [
    {'id': 1, 'name': 'box'},
    {'id': 2, 'name': 'bag'},
    {'id': 3, 'name': 'barcode'}
]

def resize_with_padding(image, target_size):
    """Resize image maintaining aspect ratio with padding"""
    img_ratio = image.width / image.height
    target_ratio = target_size[0] / target_size[1]
    
    if img_ratio > target_ratio:
        # Width is limiting factor
        new_width = target_size[0]
        new_height = int(target_size[0] / img_ratio)
    else:
        # Height is limiting factor
        new_height = target_size[1]
        new_width = int(target_size[1] * img_ratio)
    
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create padded image
    padded = Image.new('RGB', target_size, (0, 0, 0))
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    padded.paste(resized, (paste_x, paste_y))
    
    return padded, (paste_x, paste_y, new_width, new_height)

def scale_segmentation(segmentation, scale_x, scale_y, offset_x, offset_y):
    """Scale COCO segmentation polygon coordinates"""
    scaled_seg = []
    for polygon in segmentation:
        scaled_poly = []
        for i in range(0, len(polygon), 2):
            x = polygon[i] * scale_x + offset_x
            y = polygon[i + 1] * scale_y + offset_y
            scaled_poly.extend([x, y])
        scaled_seg.append(scaled_poly)
    return scaled_seg

def scale_bbox(bbox, scale_x, scale_y, offset_x, offset_y):
    """Scale COCO bounding box [x, y, width, height]"""
    x, y, w, h = bbox
    return [
        x * scale_x + offset_x,
        y * scale_y + offset_y,
        w * scale_x,
        h * scale_y
    ]

def process_dataset(dataset_name, config, split, unified_cat_map, 
                   image_id_offset, ann_id_offset):
    """Process one dataset split"""
    dataset_path = Path(config['path'])
    ann_file = dataset_path / split / '_annotations.coco.json'
    img_dir = dataset_path / split
    
    if not ann_file.exists():
        print(f"Warning: {ann_file} not found, skipping")
        return None, 0, 0
    
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create category name to ID mapping for this dataset
    old_cat_map = {cat['name']: cat['id'] for cat in coco_data['categories']}
    
    # Build reverse mapping: old_id -> new_id
    old_id_to_new_id = {}
    for old_name, old_id in old_cat_map.items():
        # Check if this label should be excluded
        if old_name in config['exclude_labels']:
            continue
        
        # Map to unified name
        unified_name = config['label_map'].get(old_name, old_name)
        
        # Get unified ID
        if unified_name in unified_cat_map:
            old_id_to_new_id[old_id] = unified_cat_map[unified_name]
    
    processed_images = []
    processed_annotations = []
    
    for img_info in tqdm(coco_data['images'], desc=f"{dataset_name}/{split}"):
        old_img_id = img_info['id']
        new_img_id = old_img_id + image_id_offset
        
        # Load and resize image
        img_path = img_dir / img_info['file_name']
        if not img_path.exists():
            print(f"Warning: {img_path} not found, skipping")
            continue
        
        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size
        
        resized_img, (offset_x, offset_y, new_w, new_h) = resize_with_padding(
            image, TARGET_SIZE
        )
        
        # Calculate scaling factors
        scale_x = new_w / orig_width
        scale_y = new_h / orig_height
        
        # Save resized image
        output_img_dir = Path(OUTPUT_DIR) / split
        output_img_dir.mkdir(parents=True, exist_ok=True)
        
        new_filename = f"{dataset_name}_{img_info['file_name']}"
        output_img_path = output_img_dir / new_filename
        resized_img.save(output_img_path)
        
        # Add to processed images
        processed_images.append({
            'id': new_img_id,
            'file_name': new_filename,
            'width': TARGET_SIZE[0],
            'height': TARGET_SIZE[1]
        })
        
        # Process annotations for this image
        img_annotations = [ann for ann in coco_data['annotations'] 
                          if ann['image_id'] == old_img_id]
        
        for ann in img_annotations:
            old_cat_id = ann['category_id']
            
            # Skip if category is excluded or not mapped
            if old_cat_id not in old_id_to_new_id:
                continue
            
            new_ann = {
                'id': ann['id'] + ann_id_offset,
                'image_id': new_img_id,
                'category_id': old_id_to_new_id[old_cat_id],
                'iscrowd': ann.get('iscrowd', 0)
            }
            
            # Scale bbox
            if 'bbox' in ann:
                new_ann['bbox'] = scale_bbox(
                    ann['bbox'], scale_x, scale_y, offset_x, offset_y
                )
                new_ann['area'] = new_ann['bbox'][2] * new_ann['bbox'][3]
            
            # Scale segmentation
            if 'segmentation' in ann:
                new_ann['segmentation'] = scale_segmentation(
                    ann['segmentation'], scale_x, scale_y, offset_x, offset_y
                )
            
            processed_annotations.append(new_ann)
    
    # Calculate max IDs for next dataset
    max_img_id = max([img['id'] for img in processed_images]) if processed_images else image_id_offset
    max_ann_id = max([ann['id'] for ann in processed_annotations]) if processed_annotations else ann_id_offset
    
    return {
        'images': processed_images,
        'annotations': processed_annotations
    }, max_img_id, max_ann_id

def merge_datasets():
    """Main function to merge all datasets"""
    
    # Create unified category mapping
    unified_cat_map = {cat['name']: cat['id'] for cat in UNIFIED_CATEGORIES}
    
    for split in ['train', 'valid', 'test']:
        print(f"\n{'='*50}")
        print(f"Processing {split} split")
        print(f"{'='*50}")
        
        merged_data = {
            'images': [],
            'annotations': [],
            'categories': UNIFIED_CATEGORIES
        }
        
        image_id_offset = 0
        ann_id_offset = 0
        
        for dataset_name, config in DATASETS_CONFIG.items():
            result, max_img_id, max_ann_id = process_dataset(
                dataset_name, config, split, unified_cat_map,
                image_id_offset, ann_id_offset
            )
            
            if result:
                merged_data['images'].extend(result['images'])
                merged_data['annotations'].extend(result['annotations'])
                
                # Update offsets for next dataset
                image_id_offset = max_img_id + 1
                ann_id_offset = max_ann_id + 1
        
        # Save merged annotations
        output_ann_path = Path(OUTPUT_DIR) / split / '_annotations.coco.json'
        output_ann_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_ann_path, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        print(f"\n{split} split complete:")
        print(f"  Images: {len(merged_data['images'])}")
        print(f"  Annotations: {len(merged_data['annotations'])}")

if __name__ == '__main__':
    # Install required packages if needed:
    # pip install Pillow tqdm numpy
    
    merge_datasets()
    print(f"\n✓ Merged dataset saved to: {OUTPUT_DIR}")