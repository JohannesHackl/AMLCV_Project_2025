# Semantic Segmentation of Boxes and Barcodes

This project implements a lightweight semantic segmentation pipeline for industrial warehouse automation. It is designed to detect and segment boxes, barcodes, and plastic bags from heterogeneous image sources, optimized for edge deployment.

## Project Overview

In logistics environments, accurate identification of packages (boxes) and markers (barcodes) is critical. This project utilizes a **MobileNetV3-DeepLabV3** architecture to achieve high-precision semantic segmentation while maintaining real-time performance on resource-constrained hardware like the Jetson Nano and Raspberry Pi 4.

### Key Results
- **Mean IoU:** 0.9433 (across all 4 classes)
- **Box IoU:** 0.9480
- **Barcode IoU:** 0.9393
- **Plastic Bag IoU:** 0.9282
- **Parameters:** 5.8M (approx. 129 MB)

## Repository Structure

- `_DATASET_PROCESSING/`: Scripts for data preprocessing, augmentation, and redistribution.
- `_MODEL_TRAINING/`: Implementation of the training loop, early stopping, and prediction scripts.
- `dataset_analysis/`: Statistical summaries of class distributions and object sizes.
- `dataset_visualizations/`: Overlay visualizations for training and validation sets.
- `merged_dataset/`: The unified dataset in COCO JSON format across train/valid/test splits.
- `models/`: Saved model checkpoints and historical IoU results.
- `main.tex`: LaTeX source for the project technical report.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- `requirements.txt` dependencies (run `pip install -r requirements.txt`)

### Training the Model
To start training from scratch:
```powershell
python _MODEL_TRAINING/model_train.py
```

### Inference
To run predictions on a specific image set:
```powershell
python _MODEL_TRAINING/model_predict.py
```

## Dataset Unification
The project combines five heterogeneous datasets into a unified 4-class schema:
1. **Background**
2. **Box/Carton**
3. **Plastic Bag/Flyer**
4. **Barcode**

Stratified sampling and class-weighted loss functions were employed to manage the severe class imbalance (91% boxes vs 11% barcodes).

## Authors
- **Amanda Lima Soares da Cunha** (amli@itu.dk)
- **Johannes Alexander Hackl** (jhac@itu.dk)

*IT University of Copenhagen - Advance Machine Learning for Computer Vision Project 2025*
