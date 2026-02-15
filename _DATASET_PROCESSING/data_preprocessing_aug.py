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
AUGMENT_DATASETS = []   # e.g. ['dataset1', 'dataset3', 'dataset5']
NUM_AUGMENTATIONS_PER_IMAGE = 3

# Unified label names
UNIFIED_CATEGORIES = [
    {'id': 1, 'name': 'box'},
    {'id': 2, 'name': 'bag'},
    {'id': 3, 'name': 'barcode'}
]

LANCZOS = Image.Resampling.LANCZOS

# Augmentation pipeline - CRITICAL: remove_invisible=False to keep all keypoints
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
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)  # KEEP ALL KEYPOINTS
)

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
    """Create augmented versions with proper keypoint tracking"""
    augmented_data = []
    
    img_width = image.width
    img_height = image.height
    
    for aug_idx in range(NUM_AUGMENTATIONS_PER_IMAGE):
        # Prepare normalized bboxes
        bboxes = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            norm_bbox = [
                x / img_width,
                y / img_height,
                w / img_width,
                h / img_height
            ]
            bboxes.append(norm_bbox)
        
        category_ids = [ann['category_id'] for ann in annotations]
        
        # Build keypoint list with metadata
        all_keypoints = []
        keypoint_meta = []  # Store (ann_idx, poly_idx, point_idx_in_poly)
        
        for ann_idx, ann in enumerate(annotations):
            if 'segmentation' in ann and ann['segmentation']:
                for poly_idx, polygon in enumerate(ann['segmentation']):
                    if len(polygon) >= 6:
                        num_points = len(polygon) // 2
                        for pt_idx in range(num_points):
                            x = polygon[pt_idx * 2]
                            y = polygon[pt_idx * 2 + 1]
                            all_keypoints.append((x, y))
                            keypoint_meta.append((ann_idx, poly_idx, pt_idx))
        
        # If no segmentations, skip augmentation for this image
        if not all_keypoints:
            continue
        
        try:
            # Apply augmentation
            transformed = transform(
                image=np.array(image),
                bboxes=bboxes,
                category_ids=category_ids,
                keypoints=all_keypoints
            )
        except Exception as e:
            print(f"Warning: Augmentation failed: {e}")
            continue
        
        # Check if any bboxes survived
        if not transformed['bboxes']:
            continue
        
        aug_image = Image.fromarray(transformed['image'])
        aug_bboxes_norm = transformed['bboxes']
        aug_keypoints = transformed['keypoints']
        aug_category_ids = transformed['category_ids']
        
        # Verify keypoint count matches (since remove_invisible=False)
        if len(aug_keypoints) != len(keypoint_meta):
            print(f"Warning: Keypoint count mismatch {len(aug_keypoints)} != {len(keypoint_meta)}, skipping")
            continue
        
        # Convert bboxes to pixel coordinates
        aug_bboxes = []
        for norm_bbox in aug_bboxes_norm:
            x, y, w, h = norm_bbox
            pixel_bbox = [
                x * img_width,
                y * img_height,
                w * img_width,
                h * img_height
            ]
            aug_bboxes.append(pixel_bbox)
        
        # Reconstruct segmentations using metadata
        # Initialize structure: dict[ann_idx][poly_idx] = list of (x, y) tuples
        seg_builder = {}
        for ann_idx in range(len(annotations)):
            if 'segmentation' in annotations[ann_idx]:
                num_polys = len(annotations[ann_idx]['segmentation'])
                seg_builder[ann_idx] = [[] for _ in range(num_polys)]
        
        # Fill in transformed keypoints
        for kp_idx, (x, y) in enumerate(aug_keypoints):
            ann_idx, poly_idx, pt_idx = keypoint_meta[kp_idx]
            
            # Clamp to image bounds
            x = max(0.0, min(float(x), img_width - 0.01))
            y = max(0.0, min(float(y), img_height - 0.01))
            
            # Add to appropriate polygon (store as list for ordering)
            if ann_idx in seg_builder and poly_idx < len(seg_builder[ann_idx]):
                # Ensure we're adding points in order
                while len(seg_builder[ann_idx][poly_idx]) < pt_idx:
                    seg_builder[ann_idx][poly_idx].append(None)  # Placeholder
                
                if pt_idx < len(seg_builder[ann_idx][poly_idx]):
                    seg_builder[ann_idx][poly_idx][pt_idx] = (x, y)
                else:
                    seg_builder[ann_idx][poly_idx].append((x, y))
        
        # Convert to COCO format
        reconstructed_segs = {}
        for ann_idx, polys in seg_builder.items():
            valid_polys = []
            for poly_points in polys:
                # Filter out None placeholders and convert to flat list
                points = [p for p in poly_points if p is not None]
                if len(points) >= 3:  # At least 3 points for valid polygon
                    flat = []
                    for x, y in points:
                        flat.extend([x, y])
                    valid_polys.append(flat)
            
            if valid_polys:
                reconstructed_segs[ann_idx] = valid_polys
        
        # Build final annotations (only for bboxes that survived)
        aug_annotations = []
        for i, (bbox, cat_id) in enumerate(zip(aug_bboxes, aug_category_ids)):
            if bbox[2] < 1 or bbox[3] < 1:
                continue
            
            new_ann = {
                'id': ann_id_base + aug_idx * 10000 + i,
                'image_id': img_id_base,
                'category_id': cat_id,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0
            }
            
            # Add segmentation if available for this annotation
            if i in reconstructed_segs:
                new_ann['segmentation'] = reconstructed_segs[i]
            
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
    
    # Create category mapping
    old_cat_map = {cat['name']: cat['id'] for cat in coco_data['categories']}
    
    old_id_to_new_id = {}
    for old_name, old_id in old_cat_map.items():
        if old_name in config['exclude_labels']:
            continue
        
        unified_name = config['label_map'].get(old_name, old_name)
        
        if unified_name in unified_cat_map:
            old_id_to_new_id[old_id] = unified_cat_map[unified_name]
    
    processed_images = []
    processed_annotations = []
    current_max_ann_id = ann_id_offset
    
    for img_info in tqdm(coco_data['images'], desc=f"{dataset_name}/{split}"):
        old_img_id = img_info['id']
        new_img_id = old_img_id + image_id_offset
        
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
        
        if processed_anns_for_image:
            current_max_ann_id = max(current_max_ann_id, 
                                    max(ann['id'] for ann in processed_anns_for_image))
        
        # Augmentation for barcode images in training
        has_barcode = any(ann['category_id'] == 3 for ann in processed_anns_for_image)
        should_augment = (dataset_name in AUGMENT_DATASETS and 
                         split == 'train' and 
                         has_barcode)
        
        if should_augment:
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
                
                for ann in aug['annotations']:
                    ann['image_id'] = aug_img_id
                
                processed_annotations.extend(aug['annotations'])
                
                if aug['annotations']:
                    current_max_ann_id = max(current_max_ann_id,
                                            max(ann['id'] for ann in aug['annotations']))
    
    max_img_id = max([img['id'] for img in processed_images]) if processed_images else image_id_offset
    max_ann_id = current_max_ann_id
    
    return {
        'images': processed_images,
        'annotations': processed_annotations
    }, max_img_id, max_ann_id

def merge_datasets():
    """Main function to merge all datasets"""
    
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
                
                image_id_offset = max_img_id + 1
                ann_id_offset = max_ann_id + 1
        
        output_ann_path = Path(OUTPUT_DIR) / split / '_annotations.coco.json'
        output_ann_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_ann_path, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        print(f"\n{split} split complete:")
        print(f"  Images: {len(merged_data['images'])}")
        print(f"  Annotations: {len(merged_data['annotations'])}")

if __name__ == '__main__':
    print("Starting dataset merging and augmentation...")
    print(f"Target size: {TARGET_SIZE}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    merge_datasets()
    print(f"\n✓ Merged dataset saved to: {OUTPUT_DIR}")