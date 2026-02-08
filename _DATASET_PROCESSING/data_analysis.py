import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import cv2
from PIL import Image

# Configuration
DATASET_PATH = 'merged_dataset'  # Path to your merged dataset
OUTPUT_DIR = 'dataset_analysis/00'  # Where to save plots and tables

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_coco_data(split):
    """Load COCO annotations for a given split"""
    ann_file = Path(DATASET_PATH) / split / '_annotations.coco.json'
    if not ann_file.exists():
        print(f"Warning: {ann_file} not found")
        return None
    
    with open(ann_file, 'r') as f:
        return json.load(f)

def analyze_dataset():
    """Comprehensive dataset analysis"""
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Analyze each split
    splits = ['train', 'valid', 'test']
    all_stats = {}
    
    for split in splits:
        print(f"\n{'='*50}")
        print(f"Analyzing {split} split")
        print(f"{'='*50}")
        
        data = load_coco_data(split)
        if not data:
            continue
        
        stats = analyze_split(data, split)
        all_stats[split] = stats
    
    # Create combined visualizations
    if all_stats:
        create_combined_visualizations(all_stats)
        create_summary_tables(all_stats)
    
    print(f"\n✓ Analysis complete! Results saved to: {OUTPUT_DIR}")

def analyze_split(data, split):
    """Analyze a single dataset split"""
    
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    images = data['images']
    annotations = data['annotations']
    
    # Basic statistics
    num_images = len(images)
    num_annotations = len(annotations)
    num_classes = len(categories)
    
    print(f"\nBasic Statistics:")
    print(f"  Total images: {num_images}")
    print(f"  Total annotations: {num_annotations}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Average annotations per image: {num_annotations/num_images:.2f}")
    
    # Class distribution
    class_counts = defaultdict(int)
    for ann in annotations:
        class_counts[ann['category_id']] += 1
    
    # Annotations per image
    img_ann_counts = defaultdict(int)
    for ann in annotations:
        img_ann_counts[ann['image_id']] += 1
    
    ann_per_image = list(img_ann_counts.values())
    
    # Object sizes (bbox areas)
    bbox_areas = []
    class_bbox_areas = defaultdict(list)
    for ann in annotations:
        area = ann['bbox'][2] * ann['bbox'][3]  # width * height
        bbox_areas.append(area)
        class_bbox_areas[ann['category_id']].append(area)
    
    # Images per class (how many images contain each class)
    images_per_class = defaultdict(set)
    for ann in annotations:
        images_per_class[ann['category_id']].add(ann['image_id'])
    
    images_per_class_count = {k: len(v) for k, v in images_per_class.items()}
    
    stats = {
        'split': split,
        'num_images': num_images,
        'num_annotations': num_annotations,
        'num_classes': num_classes,
        'categories': categories,
        'class_counts': dict(class_counts),
        'ann_per_image': ann_per_image,
        'bbox_areas': bbox_areas,
        'class_bbox_areas': dict(class_bbox_areas),
        'images_per_class': images_per_class_count
    }
    
    # Create visualizations for this split
    create_split_visualizations(stats)
    
    return stats

def create_split_visualizations(stats):
    """Create visualizations for a single split"""
    
    split = stats['split']
    categories = stats['categories']
    
    # 1. Class Distribution - Instances per Class
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Bar plot - instances per class
    class_names = [categories[cid] for cid in sorted(stats['class_counts'].keys())]
    class_counts_sorted = [stats['class_counts'][cid] for cid in sorted(stats['class_counts'].keys())]
    
    ax = axes[0, 0]
    bars = ax.bar(class_names, class_counts_sorted, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_title(f'Instances per Class - {split.capitalize()}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Instances', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Images containing each class
    images_counts = [stats['images_per_class'][cid] for cid in sorted(stats['images_per_class'].keys())]
    
    ax = axes[0, 1]
    bars = ax.bar(class_names, images_counts, color=['#FFB6B9', '#6BCB77', '#4D96FF'])
    ax.set_title(f'Images per Class - {split.capitalize()}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Annotations per Image Distribution
    ax = axes[1, 0]
    ax.hist(stats['ann_per_image'], bins=30, edgecolor='black', color='#95E1D3', alpha=0.7)
    ax.set_title(f'Annotations per Image - {split.capitalize()}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Annotations', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.axvline(np.mean(stats['ann_per_image']), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(stats["ann_per_image"]):.2f}')
    ax.axvline(np.median(stats['ann_per_image']), color='blue', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(stats["ann_per_image"]):.1f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Object Size Distribution
    ax = axes[1, 1]
    ax.hist(stats['bbox_areas'], bins=50, edgecolor='black', color='#F38181', alpha=0.7)
    ax.set_title(f'Object Size Distribution - {split.capitalize()}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Bounding Box Area (pixels²)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.axvline(np.mean(stats['bbox_areas']), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(stats["bbox_areas"]):.0f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / f'{split}_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Box plot - Object sizes per class
    fig, ax = plt.subplots(figsize=(12, 6))
    
    box_data = [stats['class_bbox_areas'][cid] for cid in sorted(stats['class_bbox_areas'].keys())]
    bp = ax.boxplot(box_data, tick_labels=class_names, patch_artist=True,
                    medianprops=dict(color='red', linewidth=2),
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
    
    ax.set_title(f'Object Size Distribution by Class - {split.capitalize()}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Bounding Box Area (pixels²)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / f'{split}_size_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Visualizations saved for {split} split")

def create_combined_visualizations(all_stats):
    """Create visualizations comparing all splits"""
    
    # 1. Class distribution across splits
    fig, ax = plt.subplots(figsize=(14, 7))
    
    splits = list(all_stats.keys())
    categories = all_stats[splits[0]]['categories']
    class_ids = sorted(categories.keys())
    class_names = [categories[cid] for cid in class_ids]
    
    x = np.arange(len(class_names))
    width = 0.25
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, split in enumerate(splits):
        counts = [all_stats[split]['class_counts'].get(cid, 0) for cid in class_ids]
        offset = width * (i - 1)
        bars = ax.bar(x + offset, counts, width, label=split.capitalize(), 
                     color=colors[i], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Instances', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution Across Splits', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / 'combined_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Dataset size comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    split_names = [s.capitalize() for s in splits]
    num_images = [all_stats[s]['num_images'] for s in splits]
    num_annotations = [all_stats[s]['num_annotations'] for s in splits]
    
    x = np.arange(len(split_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, num_images, width, label='Images', 
                   color='#6BCB77', alpha=0.8)
    bars2 = ax.bar(x + width/2, num_annotations, width, label='Annotations', 
                   color='#4D96FF', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Split', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Size Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(split_names)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / 'dataset_size_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Class balance pie charts
    fig, axes = plt.subplots(1, len(splits), figsize=(6*len(splits), 6))
    if len(splits) == 1:
        axes = [axes]
    
    colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, split in enumerate(splits):
        counts = [all_stats[split]['class_counts'][cid] for cid in class_ids]
        
        axes[i].pie(counts, labels=class_names, autopct='%1.1f%%',
                   colors=colors_pie, startangle=90, textprops={'fontsize': 11})
        axes[i].set_title(f'{split.capitalize()} Split', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / 'class_balance_pies.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✓ Combined visualizations saved")

def create_summary_tables(all_stats):
    """Create summary tables in CSV and text format"""
    
    splits = list(all_stats.keys())
    categories = all_stats[splits[0]]['categories']
    
    # 1. Overall summary table
    summary_data = []
    for split in splits:
        stats = all_stats[split]
        summary_data.append({
            'Split': split.capitalize(),
            'Images': stats['num_images'],
            'Annotations': stats['num_annotations'],
            'Avg Ann/Image': f"{stats['num_annotations']/stats['num_images']:.2f}",
            'Min Ann/Image': min(stats['ann_per_image']),
            'Max Ann/Image': max(stats['ann_per_image']),
            'Median Ann/Image': f"{np.median(stats['ann_per_image']):.1f}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(Path(OUTPUT_DIR) / 'dataset_summary.csv', index=False)
    
    print(f"\n{'='*70}")
    print("DATASET SUMMARY")
    print('='*70)
    print(df_summary.to_string(index=False))
    
    # 2. Class distribution table
    class_data = []
    for cid in sorted(categories.keys()):
        class_name = categories[cid]
        row = {'Class': class_name, 'ID': cid}
        
        for split in splits:
            stats = all_stats[split]
            instances = stats['class_counts'].get(cid, 0)
            images = stats['images_per_class'].get(cid, 0)
            
            row[f'{split.capitalize()} Instances'] = instances
            row[f'{split.capitalize()} Images'] = images
            
            if instances > 0:
                row[f'{split.capitalize()} Inst/Img'] = f"{instances/images:.2f}"
            else:
                row[f'{split.capitalize()} Inst/Img'] = "0.00"
        
        class_data.append(row)
    
    df_classes = pd.DataFrame(class_data)
    df_classes.to_csv(Path(OUTPUT_DIR) / 'class_distribution.csv', index=False)
    
    print(f"\n{'='*70}")
    print("CLASS DISTRIBUTION")
    print('='*70)
    print(df_classes.to_string(index=False))
    
    # 3. Class balance analysis
    print(f"\n{'='*70}")
    print("CLASS BALANCE ANALYSIS")
    print('='*70)
    
    for split in splits:
        stats = all_stats[split]
        counts = [stats['class_counts'][cid] for cid in sorted(stats['class_counts'].keys())]
        
        if len(counts) > 1:
            imbalance_ratio = max(counts) / min(counts)
            print(f"\n{split.capitalize()} Split:")
            print(f"  Max instances: {max(counts)} ({categories[sorted(stats['class_counts'].keys())[np.argmax(counts)]]})")
            print(f"  Min instances: {min(counts)} ({categories[sorted(stats['class_counts'].keys())[np.argmin(counts)]]})")
            print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
            
            if imbalance_ratio < 1.5:
                print(f"  Status: Well balanced ✓")
            elif imbalance_ratio < 3:
                print(f"  Status: Moderately balanced")
            else:
                print(f"  Status: Imbalanced - consider class weighting")
    
    # 4. Object size statistics
    print(f"\n{'='*70}")
    print("OBJECT SIZE STATISTICS (in pixels²)")
    print('='*70)
    
    size_data = []
    for split in splits:
        stats = all_stats[split]
        for cid in sorted(categories.keys()):
            if cid in stats['class_bbox_areas']:
                areas = stats['class_bbox_areas'][cid]
                size_data.append({
                    'Split': split.capitalize(),
                    'Class': categories[cid],
                    'Min': f"{min(areas):.0f}",
                    'Max': f"{max(areas):.0f}",
                    'Mean': f"{np.mean(areas):.0f}",
                    'Median': f"{np.median(areas):.0f}",
                    'Std': f"{np.std(areas):.0f}"
                })
    
    df_sizes = pd.DataFrame(size_data)
    df_sizes.to_csv(Path(OUTPUT_DIR) / 'object_sizes.csv', index=False)
    print(df_sizes.to_string(index=False))
    
    print(f"\n  ✓ Summary tables saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    analyze_dataset()