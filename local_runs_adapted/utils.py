# utils.py
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import wandb
from datetime import datetime
import os

def save_checkpoint(model, out_dir, epoch, name_template="checkpoint_epoch{epoch}.pth"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = name_template.format(epoch=epoch)
    path = out_dir / fname
    torch.save(model.state_dict(), path)
    return path

def save_metrics_json(out_dir, metrics):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "metrics.json"
    with open(p, "w") as f:
        json.dump(metrics, f, indent=2)
    return p

def plot_losses(train_losses, val_losses, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7,4))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.legend(); plt.tight_layout()
    p = out_dir / f"loss_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(p, dpi=150)
    plt.close()
    return p

def init_wandb(config):
    if not config.USE_WANDB:
        return None
    # require environment variable WANDB_API_KEY to be set or have logged in
    wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY, config={
        "model": config.MODEL_NAME,
        "img_size": config.IMG_SIZE,
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "lr": config.LR,
        "wd": config.WEIGHT_DECAY,
        "accum_steps": config.ACCUM_STEPS,
    })
    return wandb
