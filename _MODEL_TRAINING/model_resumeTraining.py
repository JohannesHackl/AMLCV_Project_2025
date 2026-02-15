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

# --- CONFIGURATION ---
DATASET_PATH = 'merged_dataset'
MODEL_SAVE_PATH = 'models'
RESUME_FROM = 'models/final_model_E30.pth'  # Checkpoint to resume
NUM_ADDITIONAL_EPOCHS = 20             # How many MORE epochs to train
LEARNING_RATE = 0.0001                 # Lower LR for fine-tuning
BATCH_SIZE = 16
NUM_CLASSES = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print(f"Using device: {DEVICE}")

# --- DATASET CLASS (Same as training) ---
class COCOSegmentationDataset(Dataset):
    def __init__(self, dataset_path, split='train', transform=None):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform
        
        ann_file = self.dataset_path / split / '_annotations.coco.json'
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        
        img_path = self.dataset_path / self.split / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        if img_id in self.img_to_anns:
            for ann in self.img_to_anns[img_id]:
                category_id = ann['category_id']
                if 'segmentation' in ann and ann['segmentation']:
                    for polygon in ann['segmentation']:
                        if len(polygon) >= 6:
                            pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(mask, [pts], category_id)
        
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()
        
        return image, mask

# --- MODEL DEFINITION (Must match fixed training script) ---
def get_model(num_classes):
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights)
    
    # Main Classifier (960 channels input)
    model.classifier = DeepLabHead(960, num_classes)
    
    # Aux Classifier (40 channels input - CRITICAL FIX)
    model.aux_classifier = FCNHead(40, num_classes)
    
    return model

# --- UTILS ---
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
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
            loss += 0.4 * criterion(outputs['aux'], masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device, epoch):
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
    return running_loss / len(dataloader)

def save_loss_curve(history, resume_epoch, save_path):
    """Plots the full history and marks where training resumed."""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='.')
    plt.plot(epochs, history['val_loss'], label='Valid Loss', marker='.')
    
    # Draw vertical line at resume point
    if resume_epoch < len(epochs):
        plt.axvline(x=resume_epoch, color='r', linestyle='--', label='Resumed Here')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve (Continuous)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path / 'training_history_continued.png')
    plt.close()

# --- MAIN RESUME FUNCTION ---
def resume_training():
    Path(MODEL_SAVE_PATH).mkdir(exist_ok=True)
    
    # 1. Load Data
    print("\nLoading datasets...")
    train_dataset = COCOSegmentationDataset(DATASET_PATH, split='train')
    valid_dataset = COCOSegmentationDataset(DATASET_PATH, split='valid')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Initialize Model
    print(f"\nInitializing model (Device: {DEVICE})...")
    model = get_model(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Load Checkpoint
    print(f"Loading checkpoint: {RESUME_FROM}")
    checkpoint = torch.load(RESUME_FROM, map_location=DEVICE)
    
    # Load Weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load Optimizer (Optional: usually better to reset if changing LR, but we load here)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Force the new Learning Rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE
    
    # Recover History and Epoch
    start_epoch = checkpoint.get('epoch', 0) + 1
    previous_history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
    
    # Ensure history is a list (in case it was saved differently)
    if not isinstance(previous_history['train_loss'], list):
        previous_history = {'train_loss': [], 'val_loss': []}
        
    print(f"✓ Resuming from Epoch {start_epoch}")
    print(f"✓ Previous History Length: {len(previous_history['train_loss'])} epochs")
    
    # 4. Training Loop
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    # Current session history starts as a copy of previous
    history = {
        'train_loss': previous_history['train_loss'].copy(),
        'val_loss': previous_history['val_loss'].copy()
    }
    
    total_epochs = start_epoch + NUM_ADDITIONAL_EPOCHS
    
    print("\nStarting Fine-Tuning...")
    print("="*70)
    
    for epoch in range(start_epoch, total_epochs):
        # Train & Valid
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        val_loss = validate(model, valid_loader, criterion, DEVICE, epoch)
        
        # Update History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{total_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save Checkpoint (Best)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'history': history  # Save combined history
            }, Path(MODEL_SAVE_PATH) / 'best_model.pth')
            print("  ✓ New Best Model Saved!")
            
        # Save Latest (Always)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'history': history
        }, Path(MODEL_SAVE_PATH) / 'latest_model.pth')
        
        # Update Plot immediately
        save_loss_curve(history, start_epoch, Path(MODEL_SAVE_PATH))

    print("\n" + "="*70)
    print("✓ Training Resumed & Completed.")
    print(f"Saved extended plot to {MODEL_SAVE_PATH}/training_history_continued.png")

if __name__ == '__main__':
    resume_training()
