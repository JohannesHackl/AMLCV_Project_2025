import json
from pathlib import Path
from collections import defaultdict

DATASET_PATH = 'merged_dataset'

def check_class_distribution():
    """Check if all classes exist in all splits"""
    
    splits = ['train', 'valid', 'test']
    
    print("="*70)
    print("CLASS PRESENCE CHECK")
    print("="*70)
    
    for split in splits:
        ann_file = Path(DATASET_PATH) / split / '_annotations.coco.json'
        
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        categories = {cat['id']: cat['name'] for cat in data['categories']}
        class_counts = defaultdict(int)
        
        for ann in data['annotations']:
            class_counts[ann['category_id']] += 1
        
        print(f"\n{split.upper()} split:")
        print(f"  Defined categories: {categories}")
        print(f"  Classes with instances:")
        for cid in sorted(categories.keys()):
            count = class_counts.get(cid, 0)
            status = "✓" if count > 0 else "✗ MISSING"
            print(f"    Class {cid} ({categories[cid]}): {count} instances {status}")
        
        missing_classes = [cid for cid in categories.keys() if class_counts.get(cid, 0) == 0]
        if missing_classes:
            print(f"  ⚠️  WARNING: Missing classes: {[categories[cid] for cid in missing_classes]}")

if __name__ == '__main__':
    check_class_distribution()