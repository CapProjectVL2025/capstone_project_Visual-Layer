#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

NOISY_LABELS="${NOISY_LABELS:-noise/cluster_labels_10.csv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"

$PYTHON_BIN -m visual_layer_capstone.training.train_imagenet_vit \
  --dataset-name detection-datasets/coco \
  --train-split train \
  --val-split val \
  --noisy-labels-csv "$NOISY_LABELS" \
  --model-name vit_base_patch32_224 \
  --num-classes 80 \
  --epochs 30 \
  --batch-size 64 \
  --lr 5e-5 \
  --weight-decay 1e-2 \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoint-prefix vit_b32_coco_noisy
