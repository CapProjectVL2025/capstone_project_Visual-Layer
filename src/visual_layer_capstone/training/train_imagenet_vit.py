#!/usr/bin/env python3
"""Train ViT on streamed COCO with optional noisy label overrides."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

def get_single_label(objects):
    bboxes = objects.get("bbox", [])
    labels = objects.get("category", [])
    if len(bboxes) == 0:
        return None, None

    areas = [w * h for (_, _, w, h) in bboxes]
    idx = max(range(len(areas)), key=lambda i: areas[i])
    return bboxes[idx], labels[idx]


def crop_object(image, bbox):
    x, y, w, h = bbox
    return image.crop((x, y, x + w, y + h))


def build_transforms(img_size: int, transforms_mod):
    train_tfms = transforms_mod.Compose(
        [
            transforms_mod.RandomResizedCrop(img_size),
            transforms_mod.RandomHorizontalFlip(),
            transforms_mod.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms_mod.ToTensor(),
            transforms_mod.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    val_tfms = transforms_mod.Compose(
        [
            transforms_mod.Resize((img_size, img_size)),
            transforms_mod.ToTensor(),
            transforms_mod.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return train_tfms, val_tfms


def load_noisy_lookup(noisy_labels_csv: Optional[str]) -> dict[int, int]:
    import pandas as pd

    if not noisy_labels_csv:
        return {}

    noisy_df = pd.read_csv(noisy_labels_csv)
    if "image_id" not in noisy_df.columns or "label" not in noisy_df.columns:
        raise ValueError("Noisy labels CSV must include columns: image_id, label")

    return {
        int(row.image_id): int(row.label)
        for _, row in noisy_df.iterrows()
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train ViT with optional noisy labels")

    ap.add_argument("--dataset-name", type=str, default="detection-datasets/coco")
    ap.add_argument("--train-split", type=str, default="train")
    ap.add_argument("--val-split", type=str, default="val")
    ap.add_argument("--noisy-labels-csv", type=str, default="")

    ap.add_argument("--model-name", type=str, default="vit_base_patch32_224")
    ap.add_argument("--num-classes", type=int, default=80)
    ap.add_argument("--img-size", type=int, default=224)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--label-smoothing", type=float, default=0.1)

    ap.add_argument("--train-target", type=int, default=118_000)
    ap.add_argument("--val-target", type=int, default=5_000)
    ap.add_argument("--shuffle-buffer", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    ap.add_argument("--checkpoint-prefix", type=str, default="vit_b32_coco")

    ap.add_argument("--use-wandb", action="store_true")
    ap.add_argument("--wandb-entity", type=str, default="")
    ap.add_argument("--wandb-project", type=str, default="visual_layer_capstone")
    ap.add_argument("--wandb-run-name", type=str, default="")

    return ap.parse_args()


def main() -> int:
    args = parse_args()
    from datasets import load_dataset
    from torch import amp
    from tqdm import tqdm
    import timm
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms as T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    noisy_lookup = load_noisy_lookup(args.noisy_labels_csv)
    if noisy_lookup:
        print(f"[train] loaded noisy labels: {len(noisy_lookup)}")

    run = None
    if args.use_wandb:
        import wandb

        run = wandb.init(
            entity=args.wandb_entity or None,
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            config=vars(args),
        )

    train_tfms, val_tfms = build_transforms(args.img_size, T)

    model = timm.create_model(
        args.model_name,
        pretrained=False,
        num_classes=args.num_classes,
        drop_rate=0.2,
        drop_path_rate=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    amp_enabled = device.type == "cuda"
    scaler = amp.GradScaler(enabled=amp_enabled)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_stream = load_dataset(
            args.dataset_name,
            split=args.train_split,
            streaming=True,
        ).shuffle(buffer_size=args.shuffle_buffer, seed=args.seed + epoch)

        val_stream = load_dataset(
            args.dataset_name,
            split=args.val_split,
            streaming=True,
        )

        model.train()
        total_loss, total = 0.0, 0
        noisy_used, clean_used = 0, 0
        batch_imgs, batch_labels = [], []

        for sample in tqdm(train_stream, desc=f"Epoch {epoch + 1}/{args.epochs} - Train"):
            bbox, clean_label = get_single_label(sample.get("objects", {}))
            if bbox is None:
                continue

            image_id = int(sample["image_id"])
            if image_id in noisy_lookup:
                label = noisy_lookup[image_id]
                noisy_used += 1
            else:
                label = clean_label
                clean_used += 1

            try:
                img = crop_object(sample["image"], bbox).convert("RGB")
                img = train_tfms(img)
            except Exception:
                continue

            batch_imgs.append(img)
            batch_labels.append(label)

            if len(batch_imgs) == args.batch_size:
                images = torch.stack(batch_imgs).to(device)
                labels = torch.tensor(batch_labels, device=device)

                optimizer.zero_grad()
                with amp.autocast(device_type="cuda", enabled=amp_enabled):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item() * images.size(0)
                total += images.size(0)
                batch_imgs.clear()
                batch_labels.clear()

            if total >= args.train_target:
                break

        train_loss = total_loss / max(total, 1)

        model.eval()
        val_loss, val_total = 0.0, 0
        batch_imgs, batch_labels = [], []

        with torch.no_grad():
            for sample in tqdm(val_stream, desc=f"Epoch {epoch + 1}/{args.epochs} - Val"):
                bbox, label = get_single_label(sample.get("objects", {}))
                if bbox is None:
                    continue

                try:
                    img = crop_object(sample["image"], bbox).convert("RGB")
                    img = val_tfms(img)
                except Exception:
                    continue

                batch_imgs.append(img)
                batch_labels.append(label)

                if len(batch_imgs) == args.batch_size:
                    images = torch.stack(batch_imgs).to(device)
                    labels = torch.tensor(batch_labels, device=device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    val_total += images.size(0)
                    batch_imgs.clear()
                    batch_labels.clear()

                if val_total >= args.val_target:
                    break

        val_loss = val_loss / max(val_total, 1)
        scheduler.step()

        ckpt_path = ckpt_dir / f"{args.checkpoint_prefix}_epoch_{epoch + 1:02d}.pth"
        torch.save(model.state_dict(), ckpt_path)

        log_payload = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "noisy_labels_used": noisy_used,
            "clean_labels_used": clean_used,
        }

        if run is not None:
            run.log(log_payload)
            run.save(str(ckpt_path))

        print(
            f"[train] epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} noisy_used={noisy_used} clean_used={clean_used} "
            f"checkpoint={ckpt_path}"
        )

    if run is not None:
        run.finish()

    print("[train] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
