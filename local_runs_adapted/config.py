# config.py - central config for experiments (edit here)
import os
from pathlib import Path

# ---- dataset / storage ----
ROOT = Path.cwd()
DATA_ROOT = Path("/content/data/imagenet1k_imagefolder")   # ImageFolder with train/val subdirs
OUTPUT_DIR = ROOT / "runs"                                 # where checkpoints + plots are written
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- model selection ----
MODEL_NAME = "vit_base_patch32_224"   # "vit_base_patch32_224" (ViT-B/32)
PRETRAINED = False                    # True/False: use pretrained weights

# ---- training hyperparams ----
IMG_SIZE = 224
BATCH_SIZE = 64
ACCUM_STEPS = 2        # gradient accumulation (effective batch = BATCH_SIZE * ACCUM_STEPS)
EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

# ---- dataloader ----
NUM_WORKERS = 6
PIN_MEMORY = True

# ---- logging / checkpoint ----
CHECKPOINT_EVERY = 2   # save checkpoint every N epochs
CHECKPOINT_NAME = "checkpoint_epoch{epoch}.pth"
FINAL_CHECKPOINT = "model_final.pth"
SAVE_METRICS = True

# ---- wandb ----
USE_WANDB = True           
WANDB_PROJECT = "vision-train-pipeline"
WANDB_ENTITY = None       
WANDB_NOTEBOOK_NAME = None

# ---- misc ----
SEED = 42
DEVICE = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" or os.environ.get("FORCE_CUDA")) else ("cuda" if __import__("torch").cuda.is_available() else "cpu")
