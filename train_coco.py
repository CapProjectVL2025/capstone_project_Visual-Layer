# -----------------------------
# COCO ViT-L/16 Single-Label Training
# -----------------------------

# -----------------------------
# Imports
# -----------------------------
from tqdm import tqdm
from PIL import Image
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

import timm
import wandb
from datasets import load_dataset
from torch import amp

# -----------------------------
# W&B + Dataset Config (KEPT)
# -----------------------------
ENTITY = "bhavyaranjan-university-of-california-santa-barbara"
PROJECT = "Capstone_Training_Runs"

DATASET_NAME = "detection-datasets/coco"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"

# -----------------------------
# Hyperparameters (KEPT BEHAVIOR)
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10
LR = 5e-5
WEIGHT_DECAY = 1e-2
MODEL_NAME = "vit_base_patch32_224" # vit_base_patch32_224, vit_large_patch16_224
CHECKPOINT_PATH = "vit_l16_streamed_singlelabel_ckpt.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# -----------------------------
# W&B Init (STREAMED VERSION)
# -----------------------------
wandb.init(
    entity=ENTITY,
    project=PROJECT,
    config={
        "architecture": MODEL_NAME,
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "single_label": True,
        "streaming": True,
    },
)

# -----------------------------
# Transforms (MATCH BASE PIPELINE)
# -----------------------------
train_tfms = T.Compose([
    T.RandomResizedCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

val_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# -----------------------------
# Model
# -----------------------------
model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=80,
    drop_rate=0.2,
    drop_path_rate=0.1,
).to(DEVICE)

wandb.watch(model, log="all", log_freq=100)

# -----------------------------
# Optimizer / Loss / Scheduler
# -----------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)

scaler = amp.GradScaler()

# -----------------------------
# COCO Utilities (Single-label, deterministic)
# -----------------------------

def get_single_label(objects):
    bboxes = objects["bbox"]
    labels = objects["category"]

    if len(bboxes) == 0:
        return None, None

    areas = [w * h for (_, _, w, h) in bboxes]
    idx = max(range(len(areas)), key=lambda i: areas[i])
    return bboxes[idx], labels[idx]


def crop_object(image, bbox):
    x, y, w, h = bbox
    return image.crop((x, y, x + w, y + h))

# -----------------------------
# Load STREAMED Datasets
# -----------------------------
print("Loading streamed datasets...")

train_stream = load_dataset(
    DATASET_NAME,
    split=TRAIN_SPLIT,
    streaming=True,
).shuffle(buffer_size=10_000, seed=42)

val_stream = load_dataset(
    DATASET_NAME,
    split=VAL_SPLIT,
    streaming=True,
)

# -----------------------------
# Training Loop (Dataloader-equivalent)
# -----------------------------
for epoch in range(EPOCHS):
    # -------------------------
    # TRAIN
    # -------------------------
    model.train()
    total_loss, total = 0.0, 0
    batch_imgs, batch_labels = [], []

    for sample in tqdm(train_stream, desc=f"Epoch {epoch+1}/{EPOCHS} - Train"):
        bbox, label = get_single_label(sample["objects"])
        if bbox is None:
            continue

        try:
            img = crop_object(sample["image"], bbox).convert("RGB")
            img = train_tfms(img)
        except Exception:
            continue

        batch_imgs.append(img)
        batch_labels.append(label)

        if len(batch_imgs) == BATCH_SIZE:
            images = torch.stack(batch_imgs).to(DEVICE)
            labels = torch.tensor(batch_labels, device=DEVICE)

            optimizer.zero_grad()
            with amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

            batch_imgs.clear()
            batch_labels.clear()

        if total >= 118_000:
            break

    train_loss = total_loss / max(total, 1)

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    val_loss, val_total = 0.0, 0
    batch_imgs, batch_labels = [], []

    with torch.no_grad():
        for sample in tqdm(val_stream, desc=f"Epoch {epoch+1}/{EPOCHS} - Val"):
            bbox, label = get_single_label(sample["objects"])
            if bbox is None:
                continue

            try:
                img = crop_object(sample["image"], bbox).convert("RGB")
                img = val_tfms(img)
            except Exception:
                continue

            batch_imgs.append(img)
            batch_labels.append(label)

            if len(batch_imgs) == BATCH_SIZE:
                images = torch.stack(batch_imgs).to(DEVICE)
                labels = torch.tensor(batch_labels, device=DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_total += images.size(0)

                batch_imgs.clear()
                batch_labels.clear()

            if val_total >= 5_000:
                break

    val_loss /= max(val_total, 1)

    scheduler.step()

    # -------------------------
    # Logging / Checkpointing
    # -------------------------
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "lr": scheduler.get_last_lr()[0],
    })

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    wandb.save(CHECKPOINT_PATH)

wandb.finish()
print("Training complete.")
