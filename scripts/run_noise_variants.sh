#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CLEAN_LABELS="${CLEAN_LABELS:-labels_clean.csv}"
EMBEDDINGS_NPY="${EMBEDDINGS_NPY:-embeddings.npy}"
NOISE_DIR="${NOISE_DIR:-noise}"
mkdir -p "$NOISE_DIR"

$PYTHON_BIN -m visual_layer_capstone.noise.random_noise \
  --labels "$CLEAN_LABELS" \
  --output-labels "$NOISE_DIR/random_labels_10.csv" \
  --log-file "$NOISE_DIR/random_log_10.csv" \
  --noise-level 0.10 \
  --random-seed 42

$PYTHON_BIN -m visual_layer_capstone.noise.border_noise \
  --embeddings "$EMBEDDINGS_NPY" \
  --labels "$CLEAN_LABELS" \
  --output-labels "$NOISE_DIR/border_labels_10.csv" \
  --log-file "$NOISE_DIR/border_log_10.csv" \
  --noise-level 0.10 \
  --metric cosine \
  --boundary-k 25 \
  --boundary-top-frac 0.25 \
  --nn-k 50 \
  --random-seed 42

$PYTHON_BIN -m visual_layer_capstone.noise.cluster_noise \
  --embeddings "$EMBEDDINGS_NPY" \
  --labels "$CLEAN_LABELS" \
  --output-labels "$NOISE_DIR/cluster_labels_10.csv" \
  --log-file "$NOISE_DIR/cluster_log_10.csv" \
  --noise-level 0.10 \
  --metric cosine \
  --cluster-size 50 \
  --nn-k 50 \
  --random-seed 42
