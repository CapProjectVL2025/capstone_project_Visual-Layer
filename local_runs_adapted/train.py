# train.py
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm

import config as cfg
from datasets import imagenet_dataloaders
from models import build_model
from utils import save_checkpoint, save_metrics_json, plot_losses, init_wandb

def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train():
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    print("Device:", device)

    # dataloaders
    train_loader, val_loader, num_classes = imagenet_dataloaders(
        cfg.DATA_ROOT, img_size=cfg.IMG_SIZE, batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Classes: {num_classes}")

    # model
    model = build_model(cfg.MODEL_NAME, num_classes, pretrained=cfg.PRETRAINED, device=device)

    # loss/opt/scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    scaler = GradScaler()

    # W&B
    wandb = init_wandb(cfg) if cfg.USE_WANDB else None
    if wandb:
        wandb.watch(model, log="all", log_freq=200)

    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        # ---------- train ----------
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} - Train", leave=False)
        for step, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
                loss = loss / cfg.ACCUM_STEPS
            scaler.scale(loss).backward()
            if (step + 1) % cfg.ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = images.size(0)
            running_loss += loss.item() * cfg.ACCUM_STEPS * batch_size
            total += batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            pbar.set_postfix({"loss": f"{running_loss/total:.4f}", "acc": f"{correct/total:.4f}"})

        train_loss = running_loss / total
        train_acc = correct / total

        # ---------- val ----------
        model.eval()
        val_loss_accum = 0.0
        val_total = 0
        val_corr = 0
        with torch.no_grad():
            pbarv = tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} - Val", leave=False)
            for images, labels in pbarv:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                bs = images.size(0)
                val_loss_accum += loss.item() * bs
                val_total += bs
                val_corr += (logits.argmax(dim=1) == labels).sum().item()
                pbarv.set_postfix({"val_loss": f"{val_loss_accum/val_total:.4f}", "val_acc": f"{val_corr/val_total:.4f}"})

        val_loss = val_loss_accum / val_total
        val_acc = val_corr / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # scheduler and timing
        scheduler.step()
        epoch_time = time.time() - t0
        print(f"[{epoch}/{cfg.EPOCHS}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={epoch_time:.1f}s")

        # W&B log
        if wandb:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"]})

        # checkpoint + artifact
        if epoch % cfg.CHECKPOINT_EVERY == 0:
            path = save_checkpoint(model, cfg.OUTPUT_DIR, epoch, name_template=cfg.CHECKPOINT_NAME)
            print("Saved:", path)
            if wandb:
                wandb.save(str(path))
                art = wandb.Artifact(f"model_checkpoint_epoch{epoch}", type="model")
                art.add_file(str(path))
                wandb.log_artifact(art)

        # free memory if used
        torch.cuda.empty_cache()

    # final saves
    final_ckpt = Path(cfg.OUTPUT_DIR) / cfg.FINAL_CHECKPOINT
    torch.save(model.state_dict(), final_ckpt)
    print("Saved final model to", final_ckpt)
    if wandb:
        wandb.save(str(final_ckpt))
        fa = wandb.Artifact("final_model", type="model")
        fa.add_file(str(final_ckpt))
        wandb.log_artifact(fa)
        wandb.finish()

    # save metrics & plot
    metrics = {"train_loss": train_losses, "val_loss": val_losses, "val_acc": val_accs}
    if cfg.SAVE_METRICS:
        save_metrics_json(cfg.OUTPUT_DIR, metrics)
        p = plot_losses(train_losses, val_losses, cfg.OUTPUT_DIR)
        print("Saved plot to", p)

if __name__ == "__main__":
    train()
