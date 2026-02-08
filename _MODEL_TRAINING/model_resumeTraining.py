import torch
import torch.nn as nn
import torchvision
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
DATASET_PATH = 'merged_dataset'
MODEL_SAVE_PATH = 'models'
RESUME_FROM = 'models/best_model.pth'  # Which checkpoint to resume from
BATCH_SIZE = 8
NUM_ADDITIONAL_EPOCHS = 10  # How many MORE epochs to train
LEARNING_RATE = 0.0001  # Lower learning rate for fine-tuning
NUM_CLASSES = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

class COCOSegmentationDataset(Dataset):
    """COCO format dataset for semantic segmentation"""
    
    def __init__(self, dataset_path, split='train', transform=None):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform
        
        # Load annotations
        ann_file = self.dataset_path / split / '_annotations.coco.json'
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # Group annotations by image
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        print(f"{split} dataset: {len(self.images)} images, {len(self.annotations)} annotations")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        
        # Load image
        img_path = self.dataset_path / self.split / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Create segmentation mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        # Fill mask with annotations
        if img_id in self.img_to_anns:
            for ann in self.img_to_anns[img_id]:
                category_id = ann['category_id']
                
                if 'segmentation' in ann and ann['segmentation']:
                    # Draw polygon segmentation
                    for polygon in ann['segmentation']:
                        if len(polygon) >= 6:
                            pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(mask, [pts], category_id)
        
        # Convert to tensors
        image = np.array(image)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()
        
        return image, mask

def get_model(num_classes):
    """Load model architecture - must match original training exactly"""
    from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
    
    # Load with ImageNet weights to get the proper structure
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights)
    
    # Replace classifier heads
    model.classifier = DeepLabHead(960, num_classes)  # type: ignore
    model.aux_classifier = FCNHead(40, num_classes)  # type: ignore
    
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs['out'], masks)
        
        if 'aux' in outputs:
            aux_loss = criterion(outputs['aux'], masks)
            loss += 0.4 * aux_loss
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def validate(model, dataloader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Valid]')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs['out'], masks)
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def calculate_metrics(model, dataloader, device, num_classes):
    """Calculate IoU and other metrics"""
    model.eval()
    
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Calculating metrics'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = outputs['out'].argmax(dim=1)
            
            for pred, mask in zip(preds, masks):
                pred_np = pred.cpu().numpy().flatten()
                mask_np = mask.cpu().numpy().flatten()
                
                for i in range(num_classes):
                    for j in range(num_classes):
                        confusion_matrix[i, j] += np.sum((mask_np == i) & (pred_np == j))
    
    iou_per_class = []
    for i in range(num_classes):
        intersection = confusion_matrix[i, i]
        union = confusion_matrix[i, :].sum() + confusion_matrix[:, i].sum() - intersection
        iou = intersection / (union + 1e-6)
        iou_per_class.append(iou)
    
    mean_iou = np.mean(iou_per_class)
    
    return mean_iou, iou_per_class, confusion_matrix

def resume_training():
    """Resume training from checkpoint"""
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = COCOSegmentationDataset(DATASET_PATH, split='train')
    valid_dataset = COCOSegmentationDataset(DATASET_PATH, split='valid')
    
    num_workers = 0 if DEVICE.type == 'cpu' else 4
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))
    
    # Load checkpoint first
    print(f"\nLoading checkpoint from {RESUME_FROM}...")
    checkpoint = torch.load(RESUME_FROM, map_location=DEVICE)
    
    # Initialize model
    print("\nInitializing model...")
    model = get_model(NUM_CLASSES)
    
    # Load the saved weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("✓ Model weights loaded successfully")
    except Exception as e:
        print(f"Warning: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model = model.to(DEVICE)
    
    starting_epoch = checkpoint['epoch'] + 1
    
    # Handle history - CRITICAL for continuing loss curves
    if 'history' in checkpoint:
        previous_history = checkpoint['history']
        print(f"✓ Loaded training history: {len(previous_history['train_loss'])} epochs")
    else:
        previous_history = {'train_loss': [], 'val_loss': []}
        print("⚠ No history in checkpoint, starting fresh")
    
    print(f"Resuming from epoch {starting_epoch}")
    print(f"Previous best validation loss: {checkpoint.get('val_loss', 'N/A'):.4f}" if 'val_loss' in checkpoint else "Previous validation loss: N/A")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Load optimizer state
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE
            print(f"✓ Optimizer state loaded, learning rate set to {LEARNING_RATE}")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
    
    # Training loop
    print("\nResuming training...")
    print("="*70)
    
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    # Initialize history with previous data
    history = {
        'train_loss': previous_history.get('train_loss', []).copy(),
        'val_loss': previous_history.get('val_loss', []).copy()
    }
    
    total_epochs = starting_epoch + NUM_ADDITIONAL_EPOCHS
    
    for epoch in range(starting_epoch, total_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        
        # Validate
        val_loss = validate(model, valid_loader, criterion, DEVICE, epoch)
        
        # Append to history (this continues the curve!)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"\nEpoch {epoch+1}/{total_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Valid Loss: {val_loss:.4f}")
        
        # Save latest model EVERY epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': history,  # Save full history
        }, Path(MODEL_SAVE_PATH) / 'latest_model.pth')
        print(f"  ✓ Latest model saved (epoch {epoch+1})")
        
        # Save best model (only if improved)
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history,  # Save full history
            }, Path(MODEL_SAVE_PATH) / 'best_model.pth')
            print(f"  ✓ NEW BEST model saved! (improved by {improvement:.4f})")
        else:
            print(f"  No improvement (best: {best_val_loss:.4f}, current: {val_loss:.4f})")
        
        # Save checkpoint every 5 epochs (backup)
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history,  # Save full history
            }, Path(MODEL_SAVE_PATH) / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"  ✓ Checkpoint backup saved!")
        
        # Save loss curve plot EVERY epoch (so you can monitor progress)
        save_loss_curve(history, starting_epoch, Path(MODEL_SAVE_PATH))
        
        print("="*70)
    
    # Save final model
    torch.save({
        'epoch': total_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, Path(MODEL_SAVE_PATH) / 'final_model_continued.pth')
    
    # Save final loss curve
    save_loss_curve(history, starting_epoch, Path(MODEL_SAVE_PATH), final=True)
    
    # Calculate final metrics
    print("\nCalculating final metrics on validation set...")
    model.load_state_dict(torch.load(Path(MODEL_SAVE_PATH) / 'best_model.pth')['model_state_dict'])
    mean_iou, iou_per_class, conf_matrix = calculate_metrics(model, valid_loader, DEVICE, NUM_CLASSES)
    
    print("\n" + "="*70)
    print("FINAL METRICS (Validation Set)")
    print("="*70)
    print(f"Mean IoU: {mean_iou:.4f}")
    print("\nPer-class IoU:")
    class_names = ['background', 'box', 'bag', 'barcode']
    for i, (name, iou) in enumerate(zip(class_names, iou_per_class)):
        print(f"  {name}: {iou:.4f}")
    
    print("\n" + "="*70)
    print("✓ Continued training complete!")
    print("="*70)
    print(f"Models saved to: {MODEL_SAVE_PATH}/")
    print(f"  - best_model.pth (best validation loss: {best_val_loss:.4f})")
    print(f"  - latest_model.pth (most recent epoch: {total_epochs})")
    print(f"  - final_model_continued.pth (last epoch)")
    if total_epochs >= 5:
        print(f"  - checkpoint_epoch_*.pth (periodic backups)")
    print(f"  - loss_curve.png (updated training history)")

def save_loss_curve(history, resume_epoch=None, save_dir=Path('models'), final=False):
    """Save loss curve plot with resume indicator"""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # Plot losses
    ax.plot(epochs_range, history['train_loss'], label='Train Loss', 
            marker='o', linewidth=2, markersize=4, color='#3498db')
    ax.plot(epochs_range, history['val_loss'], label='Validation Loss', 
            marker='s', linewidth=2, markersize=4, color='#e74c3c')
    
    # Mark resume point if applicable
    if resume_epoch is not None and resume_epoch > 1:
        ax.axvline(x=resume_epoch, color='#2ecc71', linestyle='--', 
                   linewidth=2, label=f'Resumed at epoch {resume_epoch}', alpha=0.7)
        
        # Add shaded region for resumed training
        ax.axvspan(resume_epoch, len(history['train_loss']), 
                   alpha=0.1, color='#2ecc71', label='Resumed training')
    
    # Find and mark best validation loss
    if history['val_loss']:
        best_val_idx = np.argmin(history['val_loss']) + 1
        best_val_loss = min(history['val_loss'])
        ax.scatter([best_val_idx], [best_val_loss], color='gold', s=200, 
                   zorder=5, marker='*', edgecolors='black', linewidths=1.5,
                   label=f'Best Val Loss: {best_val_loss:.4f} (epoch {best_val_idx})')
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title('Training History', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics box
    if len(history['train_loss']) > 0:
        stats_text = f"Total Epochs: {len(history['train_loss'])}\n"
        stats_text += f"Final Train Loss: {history['train_loss'][-1]:.4f}\n"
        stats_text += f"Final Val Loss: {history['val_loss'][-1]:.4f}\n"
        stats_text += f"Best Val Loss: {min(history['val_loss']):.4f}"
        
        # Add text box with statistics
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    # Save with timestamp if not final
    if final:
        filename = 'loss_curve_final.png'
    else:
        filename = 'loss_curve.png'
    
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save data as CSV for external plotting
    df = pd.DataFrame({
        'epoch': list(epochs_range),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss']
    })
    df.to_csv(save_dir / 'training_history.csv', index=False)

if __name__ == '__main__':
    resume_training()