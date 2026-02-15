import json
import shutil
import random
import os
from pathlib import Path
from collections import defaultdict, Counter

# --- CONFIGURATION ---
DATASET_PATH = Path('merged_dataset')
BACKUP_PATH = Path('merged_dataset_preRedistribution')
RANDOM_SEED = 42

# Ratios: 70% Train, 20% Valid, 10% Test
SPLIT_RATIOS = {'train': 0.7, 'valid': 0.2, 'test': 0.1}

def load_and_merge_data():
    """
    Loads all splits, re-indexes IDs to ensure uniqueness, and creates a single global pool.
    """
    print("Loading, merging, and RE-INDEXING all current splits...")
    
    global_images = []
    global_annotations = []
    categories = None
    
    # Maps to track re-indexing
    # (split_name, original_id) -> new_global_id
    image_id_map = {} 
    
    # Counters for new IDs
    new_img_id_counter = 0
    new_ann_id_counter = 0
    
    # Map filename to current location
    file_location_map = {}
    
    # Check for duplicates by filename to prevent same image appearing twice
    seen_filenames = set()

    for split in ['train', 'valid', 'test']:
        split_dir = DATASET_PATH / split
        json_file = split_dir / '_annotations.coco.json'
        
        if not json_file.exists():
            print(f"Warning: {split} JSON not found, skipping.")
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        if categories is None:
            categories = data['categories']
        
        # 1. Process Images
        for img in data['images']:
            original_id = img['id']
            filename = img['file_name']
            
            # Safety check: Duplicate filenames (same image content)
            if filename in seen_filenames:
                print(f"  Warning: Skipping duplicate file {filename} found in multiple splits.")
                continue
            seen_filenames.add(filename)
            
            # Assign NEW unique ID
            new_id = new_img_id_counter
            image_id_map[(split, original_id)] = new_id
            
            # Update image object
            img['id'] = new_id
            global_images.append(img)
            
            # Track file location
            file_location_map[filename] = split_dir / filename
            
            new_img_id_counter += 1
            
        # 2. Process Annotations
        for ann in data['annotations']:
            original_img_id = ann['image_id']
            
            # Retrieve the NEW image ID for this annotation
            # If the image was skipped (duplicate), skip the annotation too
            if (split, original_img_id) not in image_id_map:
                continue
                
            new_img_id = image_id_map[(split, original_img_id)]
            
            # Update annotation
            ann['image_id'] = new_img_id
            ann['id'] = new_ann_id_counter # Re-index annotation ID too for safety
            
            global_annotations.append(ann)
            new_ann_id_counter += 1

    print(f"✓ Merged & Re-indexed: {len(global_images)} images, {len(global_annotations)} annotations.")
    return global_images, global_annotations, categories, file_location_map

def get_stratification_groups(images, annotations):
    """Groups images by their 'label signature'."""
    img_to_cats = defaultdict(set)
    for ann in annotations:
        img_to_cats[ann['image_id']].add(ann['category_id'])
    
    strat_groups = defaultdict(list)
    empty_images = []

    for img in images:
        cat_ids = img_to_cats.get(img['id'], set())
        if not cat_ids:
            empty_images.append(img)
        else:
            signature = tuple(sorted(list(cat_ids)))
            strat_groups[signature].append(img)
            
    return strat_groups, empty_images, img_to_cats

def split_list(items, ratios):
    """Splits a list into train/valid/test based on ratios."""
    random.shuffle(items)
    n = len(items)
    
    n_train = int(n * ratios['train'])
    n_valid = int(n * ratios['valid'])
    
    train_items = items[:n_train]
    valid_items = items[n_train : n_train + n_valid]
    test_items = items[n_train + n_valid:]
    
    return train_items, valid_items, test_items

def verify_class_coverage(splits, img_to_cats, all_cat_ids):
    """Checks if every split has at least one instance of every class."""
    global_counts = Counter()
    for cat_list in img_to_cats.values():
        global_counts.update(cat_list)
        
    for cat_id in all_cat_ids:
        if global_counts[cat_id] < 3:
            print(f"WARNING: Category ID {cat_id} has fewer than 3 instances total.")

    for split_name, images in splits.items():
        split_cats = set()
        for img in images:
            split_cats.update(img_to_cats.get(img['id'], set()))
        
        missing = set(all_cat_ids) - split_cats
        real_missing = [c for c in missing if global_counts[c] >= 3]
        
        if real_missing:
            return False, split_name, real_missing
            
    return True, None, None

def force_inject_classes(splits, img_to_cats, missing_cat, target_split):
    """Steals an image containing the missing class from 'train'."""
    print(f"  > Attempting to fix missing class {missing_cat} in {target_split}...")
    
    donor_img = None
    # Try to steal from train first
    for i, img in enumerate(splits['train']):
        if missing_cat in img_to_cats[img['id']]:
            donor_img = splits['train'].pop(i)
            break
    
    if donor_img:
        splits[target_split].append(donor_img)
        print(f"    ✓ Moved image {donor_img['id']} from train to {target_split}")
    else:
        print(f"    ✗ Could not find donor in train.")

def main():
    random.seed(RANDOM_SEED)
    print("="*60)
    print("  COCO DATASET REDISTRIBUTION (70/20/10) - RE-INDEXING")
    print("="*60)

    if not BACKUP_PATH.exists():
        print(f"Creating backup at {BACKUP_PATH}...")
        try:
            shutil.copytree(DATASET_PATH, BACKUP_PATH)
        except FileExistsError:
            pass
    else:
        print(f"Backup already exists at {BACKUP_PATH}")

    # Load & Merge & RE-INDEX
    images, annotations, categories, file_loc_map = load_and_merge_data()
    all_cat_ids = [c['id'] for c in categories]
    
    # Stratification
    strat_groups, empty_images, img_to_cats = get_stratification_groups(images, annotations)
    print(f"\nStratification groups: {len(strat_groups)}")
    
    # Split
    final_splits = {'train': [], 'valid': [], 'test': []}
    
    for signature, group_imgs in strat_groups.items():
        t, v, te = split_list(group_imgs, SPLIT_RATIOS)
        final_splits['train'].extend(t)
        final_splits['valid'].extend(v)
        final_splits['test'].extend(te)
        
    if empty_images:
        t, v, te = split_list(empty_images, SPLIT_RATIOS)
        final_splits['train'].extend(t)
        final_splits['valid'].extend(v)
        final_splits['test'].extend(te)

    # Verify Coverage
    for _ in range(5):
        ok, failed_split, missing_cats = verify_class_coverage(final_splits, img_to_cats, all_cat_ids)
        if ok:
            break
        print(f"\nCoverage check failed: {failed_split} is missing {missing_cats}")
        for cat in missing_cats:
            force_inject_classes(final_splits, img_to_cats, cat, failed_split)
    
    ok, failed_split, missing_cats = verify_class_coverage(final_splits, img_to_cats, all_cat_ids)
    if not ok:
        print(f"CRITICAL WARNING: {failed_split} is missing {missing_cats}. Proceeding anyway.")

    # Write & Move
    print("\nWriting new splits and moving files...")
    
    for split_name, split_images in final_splits.items():
        split_img_ids = {img['id'] for img in split_images}
        split_anns = [ann for ann in annotations if ann['image_id'] in split_img_ids]
        
        split_data = {
            'info': {'description': f"Redistributed {split_name} split"},
            'licenses': [],
            'categories': categories,
            'images': split_images,
            'annotations': split_anns
        }
        
        save_dir = DATASET_PATH / split_name
        save_dir.mkdir(exist_ok=True)
        with open(save_dir / '_annotations.coco.json', 'w') as f:
            json.dump(split_data, f, indent=2)
            
        move_count = 0
        for img in split_images:
            fname = img['file_name']
            src = file_loc_map[fname]
            dst = save_dir / fname
            
            # Check if source exists before moving
            if src.exists():
                # If src and dst are same (e.g. file was in train, stayed in train), do nothing
                if src.resolve() != dst.resolve():
                    shutil.move(str(src), str(dst))
                    move_count += 1
                
                # Update map: file is now at dst
                file_loc_map[fname] = dst
            else:
                 # It might have been moved in a previous iteration (e.g. valid -> train)
                 # We rely on file_loc_map being updated to catch this
                 pass
                
        print(f"  [{split_name.upper()}] Saved {len(split_images)} images.")

    print("\n" + "="*60)
    print("FINAL DISTRIBUTION REPORT")
    print("="*60)
    
    for split_name in ['train', 'valid', 'test']:
        imgs = final_splits[split_name]
        counts = defaultdict(int)
        for img in imgs:
            cats = img_to_cats.get(img['id'], [])
            for c in cats:
                counts[c] += 1
                
        print(f"\n{split_name.upper()} ({len(imgs)} images):")
        for cat in categories:
            print(f"  - {cat['name']:<15}: {counts[cat['id']]:4d}")

    print(f"\n✓ Success. Backup available at {BACKUP_PATH}")

if __name__ == '__main__':
    main()