import json
from pathlib import Path

def check_overlap(dataset_path):
    dataset_path = Path(dataset_path)
    splits = ['train', 'valid', 'test']
    
    # Store sets of filenames for each split
    file_sets = {}
    
    for split in splits:
        ann_file = dataset_path / split / '_annotations.coco.json'
        if not ann_file.exists():
            print(f"Skipping {split} (file not found)")
            continue
            
        with open(ann_file, 'r') as f:
            data = json.load(f)
            # Create a set of filenames
            files = set(img['file_name'] for img in data['images'])
            file_sets[split] = files
            print(f"Loaded {len(files)} images from {split}")

    # Check intersections
    print("\n--- Checking Overlaps ---")
    overlaps_found = False
    
    split_names = list(file_sets.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            overlap = file_sets[s1].intersection(file_sets[s2])
            
            if overlap:
                overlaps_found = True
                print(f"⚠️  WARNING: Found {len(overlap)} duplicate images between '{s1}' and '{s2}'")
                print(f"   Sample duplicates: {list(overlap)[:3]}")
            else:
                print(f"✓ No overlap between '{s1}' and '{s2}'")

    if not overlaps_found:
        print("\n✅ Dataset split looks clean (no leakage detected).")

# Run the check
if __name__ == "__main__":
    # Update this path to match your folder structure
    DATASET_PATH = 'merged_dataset' 
    check_overlap(DATASET_PATH)