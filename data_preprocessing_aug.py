import json
import os
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A
import cv2

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

# Augmentation settings
AUGMENT_DATASETS = ['dataset1', 'dataset3', 'dataset5']  # datasets with barcodes to augment
NUM_AUGMENTATIONS_PER_IMAGE = 3  # how many augmented versions to create

# Unified label names (background=0 is implicit, not in annotations)
UNIFIED_CATEGORIES = [
    {'id': 1, 'name': 'box'},
    {'id': 2, 'name': 'bag'},
    {'id': 3, 'name': 'barcode'}
]

LANCZOS = Image.Resampling.LANCZOS

# Handle different Pillow versions
# try:
#     LANCZOS = Image.Resampling.LANCZOS
# except AttributeError:
#     LANCZOS = Image.LANCZOS


# Define augmentation pipeline
transform = A.Compose(
    transforms=[
        A.OneOf([
            A.Rotate(limit=15, p=1.0, border_mode=cv2.BORDER_CONSTANT),
            A.Affine(shear=(-10, 10), p=1.0),
            A.Perspective(scale=(0.05, 0.1), p=1.0),
        ], p=0.7),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.6),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.4),
        A.GaussNoise(p=0.3),
    ],
    bbox_params=A.BboxParams(
        format='coco', 
        label_fields=['category_ids'], 
        min_visibility=0.3,
        clip=True
    ),
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=True)
)  # type: ignore

def clip_bbox(bbox):
    """Ensure bbox is within [0, 1] range for normalized coordinates"""
    x, y, w, h = bbox
    # Clip to valid range
    x = max(0.0, min(x, 1.0))
    y = max(0.0, min(y, 1.0))
    # Adjust width/height if they exceed boundaries
    w = min(w, 1.0 - x)
    h = min(h, 1.0 - y)
    return [x, y, w, h]

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
    
    resized = image.resize((new_width, new_height), LANCZOS)
    
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

def augment_image_and_annotations(image, annotations, img_id_base, ann_id_base):
    """Create augmented versions of an image with updated annotations"""
    augmented_data = []
    
    for aug_idx in range(NUM_AUGMENTATIONS_PER_IMAGE):
        # Prepare bboxes - convert to normalized coordinates
        img_width = image.width
        img_height = image.height
        
        bboxes = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            # Normalize to [0, 1]
            norm_bbox = [
                x / img_width,
                y / img_height,
                w / img_width,
                h / img_height
            ]
            bboxes.append(norm_bbox)
        
        category_ids = [ann['category_id'] for ann in annotations]
        
        # Convert segmentation polygons to keypoints for transformation
        all_keypoints = []
        keypoint_to_ann = []
        ann_polygon_counts = []
        
        for ann_idx, ann in enumerate(annotations):
            polygon_count = 0
            if 'segmentation' in ann and ann['segmentation']:
                for polygon in ann['segmentation']:
                    if len(polygon) >= 6:
                        points = []
                        for i in range(0, len(polygon), 2):
                            # Clamp coordinates to valid range [0, img_size - 0.01]
                            x = max(0.0, min(polygon[i], img_width - 0.01))
                            y = max(0.0, min(polygon[i+1], img_height - 0.01))
                            points.append((x, y))
                        all_keypoints.extend(points)
                        keypoint_to_ann.extend([ann_idx] * len(points))
                        polygon_count += 1
            ann_polygon_counts.append(polygon_count)
        
        try:
            # Apply augmentation
            transformed = transform(
                image=np.array(image),
                bboxes=bboxes,
                category_ids=category_ids,
                keypoints=all_keypoints if all_keypoints else [(0, 0)]
            )
        except Exception as e:
            print(f"Warning: Augmentation failed, skipping this version: {e}")
            continue
        
        if not transformed['bboxes']:
            # All bboxes were removed, skip this augmentation
            continue
        
        aug_image = Image.fromarray(transformed['image'])
        aug_bboxes_norm = transformed['bboxes']
        aug_keypoints = transformed.get('keypoints', [])
        
        # Convert bboxes back to pixel coordinates
        aug_bboxes = []
        for norm_bbox in aug_bboxes_norm:
            x, y, w, h = norm_bbox
            pixel_bbox = [
                x * TARGET_SIZE[0],
                y * TARGET_SIZE[1],
                w * TARGET_SIZE[0],
                h * TARGET_SIZE[1]
            ]
            aug_bboxes.append(pixel_bbox)
        
        # Reconstruct annotations
        aug_annotations = []
        keypoint_idx = 0
        
        for ann_idx, ann in enumerate(annotations):
            if ann_idx >= len(aug_bboxes):
                # Skip corresponding keypoints
                if ann_polygon_counts[ann_idx] > 0:
                    points_to_skip = sum(
                        len(ann['segmentation'][p]) // 2 
                        for p in range(ann_polygon_counts[ann_idx])
                    )
                    keypoint_idx += points_to_skip
                continue
                
            new_ann = {
                'id': ann_id_base + aug_idx * 10000 + ann_idx,
                'image_id': img_id_base + aug_idx + 1,
                'category_id': ann['category_id'],
                'bbox': aug_bboxes[ann_idx],
                'area': aug_bboxes[ann_idx][2] * aug_bboxes[ann_idx][3],
                'iscrowd': ann.get('iscrowd', 0)
            }
            
            # Reconstruct segmentation from transformed keypoints
            if 'segmentation' in ann and ann['segmentation'] and ann_polygon_counts[ann_idx] > 0:
                new_segmentation = []
                
                for poly_idx in range(ann_polygon_counts[ann_idx]):
                    original_polygon = ann['segmentation'][poly_idx]
                    num_points = len(original_polygon) // 2
                    new_polygon = []
                    
                    for _ in range(num_points):
                        if keypoint_idx < len(aug_keypoints):
                            kp = aug_keypoints[keypoint_idx]
                            # Clamp keypoints to image boundaries
                            x_clamped = max(0.0, min(float(kp[0]), TARGET_SIZE[0] - 0.01))
                            y_clamped = max(0.0, min(float(kp[1]), TARGET_SIZE[1] - 0.01))
                            new_polygon.extend([x_clamped, y_clamped])
                            keypoint_idx += 1
                        else:
                            keypoint_idx += 1
                            break
                    
                    if len(new_polygon) >= 6:
                        new_segmentation.append(new_polygon)
                
                if new_segmentation:
                    new_ann['segmentation'] = new_segmentation
            
            aug_annotations.append(new_ann)
        
        if aug_annotations:
            augmented_data.append({
                'image': aug_image,
                'annotations': aug_annotations
            })
    
    return augmented_data

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
    current_max_ann_id = ann_id_offset
    
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
        
        processed_anns_for_image = []
        
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
            
            processed_anns_for_image.append(new_ann)
            processed_annotations.append(new_ann)
        
        # Update max annotation ID
        if processed_anns_for_image:
            current_max_ann_id = max(current_max_ann_id, 
                                    max(ann['id'] for ann in processed_anns_for_image))
        
        # Check if this image contains barcodes AND should be augmented
        has_barcode = any(ann['category_id'] == 3 for ann in processed_anns_for_image)
        should_augment = (dataset_name in AUGMENT_DATASETS and 
                         split == 'train' and 
                         has_barcode)
        
        if should_augment:
            # Create augmented versions
            aug_data = augment_image_and_annotations(
                resized_img, 
                processed_anns_for_image,
                new_img_id,
                current_max_ann_id + 1
            )
            
            for aug_idx, aug in enumerate(aug_data):
                aug_filename = f"{dataset_name}_aug{aug_idx}_{img_info['file_name']}"
                aug_img_path = output_img_dir / aug_filename
                aug['image'].save(aug_img_path)
                
                aug_img_id = new_img_id + (aug_idx + 1) * 100000
                
                processed_images.append({
                    'id': aug_img_id,
                    'file_name': aug_filename,
                    'width': TARGET_SIZE[0],
                    'height': TARGET_SIZE[1]
                })
                
                # Update image_id in augmented annotations
                for ann in aug['annotations']:
                    ann['image_id'] = aug_img_id
                
                processed_annotations.extend(aug['annotations'])
                
                if aug['annotations']:
                    current_max_ann_id = max(current_max_ann_id,
                                            max(ann['id'] for ann in aug['annotations']))
    
    # Calculate max IDs for next dataset
    max_img_id = max([img['id'] for img in processed_images]) if processed_images else image_id_offset
    max_ann_id = current_max_ann_id
    
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
        print(f"  Categories: {merged_data['categories']}")

if __name__ == '__main__':
    # Install required packages if needed:
    # pip install Pillow tqdm numpy albumentations opencv-python
    
    print("Starting dataset merging and augmentation...")
    print(f"Target size: {TARGET_SIZE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Datasets to augment: {AUGMENT_DATASETS}")
    print(f"Augmentations per barcode image: {NUM_AUGMENTATIONS_PER_IMAGE}")
    
    merge_datasets()
    print(f"\n✓ Merged dataset saved to: {OUTPUT_DIR}")