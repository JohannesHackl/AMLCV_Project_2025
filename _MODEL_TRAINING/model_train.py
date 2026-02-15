import torch
import torch.nn as nn
import torchvision
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
DATASET_PATH = 'merged_dataset'
MODEL_SAVE_PATH = 'models'
BATCH_SIZE = 12
NUM_EPOCHS = 50  # Increased since we have early stopping
LEARNING_RATE = 0.001
NUM_CLASSES = 4  # background + 3 classes
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class COCOSegmentationDataset(Dataset):
    """COCO format dataset for semantic segmentation with Albumentations"""
    
    def __init__(self, dataset_path, split='train', transform=None):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform
        
        # Load annotations
        ann_file = self.dataset_path / split / '_annotations.coco.json'
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
            
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        
        # Group annotations by image
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        print(f"{split} dataset: {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        
        # Load image
        img_path = self.dataset_path / self.split / img_info['file_name']
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Create segmentation mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        # Fill mask with annotations
        if img_id in self.img_to_anns:
            for ann in self.img_to_anns[img_id]:
                category_id = ann['category_id']
                if 'segmentation' in ann and ann['segmentation']:
                    for polygon in ann['segmentation']:
                        if len(polygon) >= 6:
                            pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(mask, [pts], category_id)
        
        # Apply Albumentations (Augmentation + Normalization + ToTensor)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Fallback if no transform provided (shouldn't happen with correct setup)
            transform = A.Compose([A.Normalize(), ToTensorV2()])
            augmented = transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.long()

def get_transforms(split='train'):
    """Get Albumentations transforms for segmentation"""
    if split == 'train':
        return A.Compose([
            # Geometric Augmentations
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.Affine(shear=(-10, 10), p=0.3),
            
            # Color/Noise Augmentations
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.RandomGamma(gamma_limit=(80, 120)),
            ], p=0.5),
            A.GaussNoise(p=0.2),
            
            # Normalization & Conversion
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            # Validation: Only Normalize & Convert
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

def compute_class_weights(dataloader, num_classes, device):
    """Calculate class weights based on pixel frequency"""
    print("Computing class weights...")
    pixel_counts = torch.zeros(num_classes, device=device)
    
    # We only need to check a subset to get a good estimate
    num_batches = 0
    max_batches = 50 
    
    for _, masks in dataloader:
        masks = masks.to(device)
        for i in range(num_classes):
            pixel_counts[i] += (masks == i).sum()
        
        num_batches += 1
        if num_batches >= max_batches:
            break
            
    total_pixels = pixel_counts.sum()
    # Formula: total / (num_classes * frequency)
    weights = total_pixels / (num_classes * pixel_counts)
    
    # Normalize weights so they aren't too large
    weights = weights / weights.mean()
    
    print(f"Class Weights: {weights.cpu().numpy()}")
    return weights

def get_model(num_classes, pretrained=True):
    if pretrained:
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        model = deeplabv3_mobilenet_v3_large(weights=weights)
    else:
        model = deeplabv3_mobilenet_v3_large(weights=None)
    
    model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(960, num_classes)
    model.aux_classifier = torchvision.models.segmentation.fcn.FCNHead(40, num_classes)
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train() # Dropout ON
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
    model.eval() # Dropout OFF
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

def visualize_predictions(model, dataset, device, num_samples=4):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    colors = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]} # B, R, G, B
    
    # Denormalize for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(0, len(dataset))
            # Get raw image for visualization (bypass transform normalization for display)
            # Hack: access dataset internals to get clean image
            img_info = dataset.images[idx]
            raw_img = Image.open(dataset.dataset_path / dataset.split / img_info['file_name']).convert('RGB')
            raw_img = np.array(raw_img)
            
            # Get tensor for model
            image_tensor, mask_tensor = dataset[idx]
            
            output = model(image_tensor.unsqueeze(0).to(device))
            pred = output['out'].argmax(dim=1).squeeze(0).cpu().numpy()
            mask = mask_tensor.cpu().numpy()
            
            # Create colored masks
            mask_colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
            pred_colored = np.zeros((*pred.shape, 3), dtype=np.uint8)
            
            for class_id, color in colors.items():
                mask_colored[mask == class_id] = color
                pred_colored[pred == class_id] = color
            
            axes[i, 0].imshow(raw_img)
            axes[i, 0].set_title('Input Image')
            axes[i, 1].imshow(mask_colored)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 2].imshow(pred_colored)
            axes[i, 2].set_title('Prediction')
            
    plt.tight_layout()
    plt.savefig(Path(MODEL_SAVE_PATH) / 'predictions_visualization.png')
    plt.close()

def train():
    Path(MODEL_SAVE_PATH).mkdir(exist_ok=True)
    
    # 1. Setup Data with Augmentations
    print("\nLoading datasets...")
    train_dataset = COCOSegmentationDataset(DATASET_PATH, split='train', transform=get_transforms('train'))
    valid_dataset = COCOSegmentationDataset(DATASET_PATH, split='valid', transform=get_transforms('valid'))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Setup Model & Class Weights
    model = get_model(NUM_CLASSES, pretrained=True).to(DEVICE)
    
    # Compute weights from a subset of training data
    class_weights = compute_class_weights(train_loader, NUM_CLASSES, DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 3. Optimizer (AdamW) & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # 4. Training Loop with Early Stopping
    print("\nStarting training...")
    best_val_loss = float('inf')
    early_stop_patience = 7
    no_improve_count = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    for epoch in range(NUM_EPOCHS):
        # Train & Validate
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        val_loss = validate(model, valid_loader, criterion, DEVICE, epoch)
        
        # Step Scheduler
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        scheduler.step(val_loss)
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        print(f"  Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Early Stopping & Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, Path(MODEL_SAVE_PATH) / 'best_model.pth')
            print("  ✓ New best model saved!")
        else:
            no_improve_count += 1
            print(f"  ! No improvement for {no_improve_count}/{early_stop_patience} epochs")
            
        if no_improve_count >= early_stop_patience:
            print("\nEarly stopping triggered!")
            break
            
    # Final cleanup
    print("Generating final visualizations...")
    visualize_predictions(model, valid_dataset, DEVICE)
    
    # Plot history
    plt.figure()
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Valid')
    plt.legend()
    plt.savefig(Path(MODEL_SAVE_PATH) / 'training_history.png')
    plt.close()

if __name__ == '__main__':
    train()