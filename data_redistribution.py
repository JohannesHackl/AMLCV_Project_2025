import json
import shutil
from pathlib import Path
from collections import defaultdict
import random

DATASET_PATH = 'merged_dataset'
RANDOM_SEED = 42

def redistribute_splits():
    """Redistribute data to ensure all classes in all splits"""
    
    print("="*70)
    print("REDISTRIBUTING DATASET SPLITS")
    print("="*70)
    
    # Load all data
    all_data = {'train': None, 'valid': None, 'test': None}
    
    for split in ['train', 'valid', 'test']:
        ann_file = Path(DATASET_PATH) / split / '_annotations.coco.json'
        with open(ann_file, 'r') as f:
            all_data[split] = json.load(f)
    
    # Check what's missing
    categories = all_data['train']['categories']
    
    print("\nCurrent class distribution:")
    for split in ['train', 'valid', 'test']:
        class_counts = defaultdict(int)
        for ann in all_data[split]['annotations']:
            class_counts[ann['category_id']] += 1
        
        print(f"\n{split.upper()}:")
        for cat in categories:
            count = class_counts.get(cat['id'], 0)
            print(f"  {cat['name']}: {count}")
    
    # Strategy: If test is missing classes, take some from valid
    print("\n" + "="*70)
    print("SOLUTION: Taking samples from valid to create balanced test set")
    print("="*70)
    
    # Combine valid and test
    combined_images = all_data['valid']['images'] + all_data['test']['images']
    combined_annotations = all_data['valid']['annotations'] + all_data['test']['annotations']
    
    # Group images by which classes they contain
    image_classes = defaultdict(set)
    image_to_anns = defaultdict(list)
    
    for ann in combined_annotations:
        image_classes[ann['image_id']].add(ann['category_id'])
        image_to_anns[ann['image_id']].append(ann)
    
    # Separate images by class content
    images_with_bag = [img for img in combined_images if 2 in image_classes[img['id']]]
    images_without_bag = [img for img in combined_images if 2 not in image_classes[img['id']]]
    
    print(f"\nImages with 'bag': {len(images_with_bag)}")
    print(f"Images without 'bag': {len(images_without_bag)}")
    
    # Desired split ratio (approximately)
    total_images = len(combined_images)
    target_test_size = int(total_images * 0.15)  # 15% for test
    target_valid_size = total_images - target_test_size  # Rest for valid
    
    print(f"\nTarget test size: {target_test_size}")
    print(f"Target valid size: {target_valid_size}")
    
    # Ensure test has bags
    random.seed(RANDOM_SEED)
    random.shuffle(images_with_bag)
    random.shuffle(images_without_bag)
    
    # Allocate to test: make sure we have bags
    test_bag_count = max(10, int(len(images_with_bag) * 0.2))  # At least 10 or 20% of bag images
    test_images_bag = images_with_bag[:test_bag_count]
    
    # Fill rest of test from non-bag images
    remaining_test_slots = target_test_size - len(test_images_bag)
    test_images_other = images_without_bag[:remaining_test_slots]
    
    test_images = test_images_bag + test_images_other
    
    # Rest goes to valid
    valid_images_bag = images_with_bag[test_bag_count:]
    valid_images_other = images_without_bag[remaining_test_slots:]
    valid_images = valid_images_bag + valid_images_other
    
    # Create new splits
    test_img_ids = {img['id'] for img in test_images}
    valid_img_ids = {img['id'] for img in valid_images}
    
    new_test_anns = [ann for ann in combined_annotations if ann['image_id'] in test_img_ids]
    new_valid_anns = [ann for ann in combined_annotations if ann['image_id'] in valid_img_ids]
    
    new_test_data = {
        'images': test_images,
        'annotations': new_test_anns,
        'categories': categories
    }
    
    new_valid_data = {
        'images': valid_images,
        'annotations': new_valid_anns,
        'categories': categories
    }
    
    # Backup old data
    print("\n" + "="*70)
    print("Creating backups...")
    print("="*70)
    
    backup_dir = Path(DATASET_PATH + '_backup_before_redistribution')
    if not backup_dir.exists():
        shutil.copytree(DATASET_PATH, backup_dir)
        print(f"✓ Backup created at: {backup_dir}")
    else:
        print(f"✓ Backup already exists at: {backup_dir}")
    
    # Save new splits
    print("\n" + "="*70)
    print("Saving new splits...")
    print("="*70)
    
    # Save valid
    valid_ann_file = Path(DATASET_PATH) / 'valid' / '_annotations.coco.json'
    with open(valid_ann_file, 'w') as f:
        json.dump(new_valid_data, f, indent=2)
    
    # Save test
    test_ann_file = Path(DATASET_PATH) / 'test' / '_annotations.coco.json'
    with open(test_ann_file, 'w') as f:
        json.dump(new_test_data, f, indent=2)
    
    # Move image files
    print("\nMoving image files...")
    
    # Get current file locations
    valid_img_dir = Path(DATASET_PATH) / 'valid'
    test_img_dir = Path(DATASET_PATH) / 'test'
    
    # Track which images need to move
    current_test_files = {img['file_name'] for img in all_data['test']['images']}
    current_valid_files = {img['file_name'] for img in all_data['valid']['images']}
    
    new_test_files = {img['file_name'] for img in test_images}
    new_valid_files = {img['file_name'] for img in valid_images}
    
    # Files that need to move from valid to test
    valid_to_test = new_test_files & current_valid_files
    # Files that need to move from test to valid
    test_to_valid = new_valid_files & current_test_files
    
    print(f"  Moving {len(valid_to_test)} images from valid to test...")
    for filename in valid_to_test:
        src = valid_img_dir / filename
        dst = test_img_dir / filename
        if src.exists():
            shutil.move(str(src), str(dst))
    
    print(f"  Moving {len(test_to_valid)} images from test to valid...")
    for filename in test_to_valid:
        src = test_img_dir / filename
        dst = valid_img_dir / filename
        if src.exists():
            shutil.move(str(src), str(dst))
    
    # Verify new distribution
    print("\n" + "="*70)
    print("NEW CLASS DISTRIBUTION")
    print("="*70)
    
    for split_name, split_data in [('VALID', new_valid_data), ('TEST', new_test_data)]:
        class_counts = defaultdict(int)
        for ann in split_data['annotations']:
            class_counts[ann['category_id']] += 1
        
        print(f"\n{split_name}:")
        print(f"  Total images: {len(split_data['images'])}")
        print(f"  Total annotations: {len(split_data['annotations'])}")
        for cat in categories:
            count = class_counts.get(cat['id'], 0)
            status = "✓" if count > 0 else "✗"
            print(f"  {cat['name']}: {count} {status}")
    
    print("\n" + "="*70)
    print("✓ Redistribution complete!")
    print("="*70)
    print(f"Original data backed up to: {backup_dir}")
    print("You can now re-run your analysis script.")

if __name__ == '__main__':
    redistribute_splits()