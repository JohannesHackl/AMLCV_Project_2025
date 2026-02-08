import torch
import torch.nn as nn
import torchvision
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
DATASET_PATH = 'merged_dataset'
MODEL_SAVE_PATH = 'models'
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 4  # background + 3 classes (box, bag, barcode)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
# + Add scheduler
# + Add early stopping

# If GPU memory errors
BATCH_SIZE = 12  # or 8

# If training too slow
NUM_EPOCHS = 30  # minimum

# If overfitting (train loss << val loss)
# Add data augmentation or reduce epochs
"""

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
                        if len(polygon) >= 6:  # At least 3 points
                            pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(mask, [pts], category_id)
        
        # Convert to tensors
        image = np.array(image)
        
        # Simple normalization
        image = image.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()
        
        return image, mask

def get_model(num_classes, pretrained=True):
    """Load pre-trained DeepLabV3 with MobileNetV3 backbone"""
    
    # Load pre-trained model with updated API
    from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
    
    if pretrained:
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        model = deeplabv3_mobilenet_v3_large(weights=weights)
    else:
        model = deeplabv3_mobilenet_v3_large(weights=None)
    
    # Replace the entire classifier head
    # The classifier is a Sequential with: Conv2d -> BatchNorm2d -> ReLU -> Dropout -> Conv2d
    model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(960, num_classes)  # type: ignore
    
    # Replace auxiliary classifier
    model.aux_classifier = torchvision.models.segmentation.fcn.FCNHead(960, num_classes)  # type: ignore
    
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # DeepLabV3 returns a dict with 'out' and 'aux' keys
        loss = criterion(outputs['out'], masks)
        
        # Add auxiliary loss if available
        if 'aux' in outputs:
            aux_loss = criterion(outputs['aux'], masks)
            loss += 0.4 * aux_loss
        
        # Backward pass
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
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Calculating metrics'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = outputs['out'].argmax(dim=1)
            
            # Update confusion matrix
            for pred, mask in zip(preds, masks):
                pred_np = pred.cpu().numpy().flatten()
                mask_np = mask.cpu().numpy().flatten()
                
                for i in range(num_classes):
                    for j in range(num_classes):
                        confusion_matrix[i, j] += np.sum((mask_np == i) & (pred_np == j))
    
    # Calculate IoU for each class
    iou_per_class = []
    for i in range(num_classes):
        intersection = confusion_matrix[i, i]
        union = confusion_matrix[i, :].sum() + confusion_matrix[:, i].sum() - intersection
        iou = intersection / (union + 1e-6)
        iou_per_class.append(iou)
    
    mean_iou = np.mean(iou_per_class)
    
    return mean_iou, iou_per_class, confusion_matrix

def visualize_predictions(model, dataset, device, num_samples=4):
    """Visualize some predictions"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    # Class colors for visualization
    colors = {
        0: [0, 0, 0],       # background - black
        1: [255, 0, 0],     # box - red
        2: [0, 255, 0],     # bag - green
        3: [0, 0, 255]      # barcode - blue
    }
    
    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(0, len(dataset))
            image, mask = dataset[idx]
            
            # Predict
            image_input = image.unsqueeze(0).to(device)
            output = model(image_input)
            pred = output['out'].argmax(dim=1).squeeze(0).cpu().numpy()
            
            # Convert to displayable format
            image_display = image.permute(1, 2, 0).numpy()
            
            # Create colored masks
            mask_colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
            pred_colored = np.zeros((*pred.shape, 3), dtype=np.uint8)
            
            for class_id, color in colors.items():
                mask_colored[mask == class_id] = color
                pred_colored[pred == class_id] = color
            
            # Plot
            axes[i, 0].imshow(image_display)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask_colored)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_colored)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(Path(MODEL_SAVE_PATH) / 'predictions_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualizations saved to {MODEL_SAVE_PATH}/predictions_visualization.png")

def train():
    """Main training function"""
    
    # Create model directory
    Path(MODEL_SAVE_PATH).mkdir(exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = COCOSegmentationDataset(DATASET_PATH, split='train')
    valid_dataset = COCOSegmentationDataset(DATASET_PATH, split='valid')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=4, pin_memory=True)
    
    # Initialize model
    print("\nInitializing model...")
    model = get_model(NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\nStarting training...")
    print("="*70)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        
        # Validate
        val_loss = validate(model, valid_loader, criterion, DEVICE, epoch)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Valid Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, Path(MODEL_SAVE_PATH) / 'best_model.pth')
            print(f"  ✓ Best model saved!")
        
        print("="*70)
    
    # Save final model
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, Path(MODEL_SAVE_PATH) / 'final_model.pth')
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(MODEL_SAVE_PATH) / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
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
    
    # Visualize predictions
    print("\nGenerating prediction visualizations...")
    visualize_predictions(model, valid_dataset, DEVICE, num_samples=4)
    
    print("\n" + "="*70)
    print("✓ Training complete!")
    print("="*70)
    print(f"Models saved to: {MODEL_SAVE_PATH}/")
    print(f"  - best_model.pth (best validation loss)")
    print(f"  - final_model.pth (last epoch)")
    print(f"  - training_history.png")
    print(f"  - predictions_visualization.png")

if __name__ == '__main__':    
    train()