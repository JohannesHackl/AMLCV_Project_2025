import json
from pathlib import Path
from collections import defaultdict

DATASET_PATH = 'merged_dataset'

def check_for_duplicates():
    """Check if there are duplicate annotations in the dataset"""
    
    for split in ['train', 'valid', 'test']:
        ann_file = Path(DATASET_PATH) / split / '_annotations.coco.json'
        
        if not ann_file.exists():
            print(f"Split {split} not found, skipping")
            continue
        
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"{split.upper()} Split Analysis")
        print(f"{'='*60}")
        
        print(f"\nTotal images: {len(coco_data['images'])}")
        print(f"Total annotations: {len(coco_data['annotations'])}")
        
        # Check for duplicate image IDs
        img_ids = [img['id'] for img in coco_data['images']]
        if len(img_ids) != len(set(img_ids)):
            print(f"⚠ WARNING: Duplicate image IDs found!")
            img_id_counts = defaultdict(int)
            for img_id in img_ids:
                img_id_counts[img_id] += 1
            duplicates = {k: v for k, v in img_id_counts.items() if v > 1}
            print(f"  Duplicate IDs: {duplicates}")
        
        # Check for duplicate annotation IDs
        ann_ids = [ann['id'] for ann in coco_data['annotations']]
        if len(ann_ids) != len(set(ann_ids)):
            print(f"⚠ WARNING: Duplicate annotation IDs found!")
        
        # Check annotations per image
        img_to_ann_count = defaultdict(int)
        for ann in coco_data['annotations']:
            img_to_ann_count[ann['image_id']] += 1
        
        ann_counts = list(img_to_ann_count.values())
        if ann_counts:
            print(f"\nAnnotations per image:")
            print(f"  Min: {min(ann_counts)}")
            print(f"  Max: {max(ann_counts)}")
            print(f"  Average: {sum(ann_counts) / len(ann_counts):.2f}")
            
            # Show images with unusually high annotation counts
            threshold = 20
            high_count_images = {img_id: count for img_id, count in img_to_ann_count.items() 
                                if count > threshold}
            
            if high_count_images:
                print(f"\n⚠ Images with >{threshold} annotations (possible duplicates):")
                for img_id, count in sorted(high_count_images.items(), key=lambda x: x[1], reverse=True)[:10]:
                    # Find image filename
                    img_info = next((img for img in coco_data['images'] if img['id'] == img_id), None)
                    if img_info:
                        print(f"  {img_info['file_name']}: {count} annotations")
                    else:
                        print(f"  Image ID {img_id}: {count} annotations")
        
        # Check for annotations outside image bounds
        out_of_bounds = 0
        for ann in coco_data['annotations']:
            bbox = ann['bbox']
            if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > 512 or bbox[1] + bbox[3] > 512:
                out_of_bounds += 1
        
        if out_of_bounds > 0:
            print(f"\n⚠ WARNING: {out_of_bounds} annotations have bboxes outside image bounds")

def remove_output_directory():
    """Remove the merged_dataset directory to start fresh"""
    import shutil
    
    output_path = Path(DATASET_PATH)
    
    if not output_path.exists():
        print(f"Directory {DATASET_PATH} does not exist. Nothing to delete.")
        return
    
    print(f"\n⚠ WARNING: This will DELETE the entire {DATASET_PATH} directory!")
    print("Make sure you have backups of your original source datasets.")
    response = input("Are you sure you want to continue? (type 'DELETE' to confirm): ")
    
    if response == 'DELETE':
        shutil.rmtree(output_path)
        print(f"✓ {DATASET_PATH} has been deleted.")
        print("You can now run the preprocessing script fresh.")
    else:
        print("Cancelled.")

if __name__ == '__main__':
    print("Dataset Cleanup Tool")
    print("="*60)
    
    print("\n1. Checking for issues...")
    check_for_duplicates()
    
    print("\n" + "="*60)
    print("\nOptions:")
    print("1. Delete merged_dataset directory and start fresh")
    print("2. Exit")
    
    choice = input("\nEnter choice (1 or 2): ")
    
    if choice == '1':
        remove_output_directory()
    else:
        print("Exited.")