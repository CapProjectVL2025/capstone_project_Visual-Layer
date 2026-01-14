# datasets.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
import torch
import os
import json

def imagenet_dataloaders(data_root, img_size=224, batch_size=64, num_workers=4, pin_memory=True):
    """
    Builds ImageFolder dataloaders for train/val.
    Expects data_root/train and data_root/val
    """
    data_root = Path(data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    assert train_dir.exists() and val_dir.exists(), f"Train/Val directories missing under {data_root}"

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize(int(img_size*256/224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)

    return train_loader, val_loader, len(train_ds.classes)

# Add COCO dataset
