#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere, but ensure we operate relative to the repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NOISE_SCRIPT="$ROOT_DIR/alec/noise_injection.py"

$PYTHON_BIN "$NOISE_SCRIPT" \
  --embeddings alec/embeddings.npy \
  --labels alec/labels_clean.csv \
  --id-column vector_id \
  --label-column label \
  --output-labels alec/labels_random_10.csv \
  --log-file alec/log_random_10.csv \
  --mode random \
  --noise-level 0.10 \
  --cluster-size 1 \
  --metric cosine \
  --random-seed 42

$PYTHON_BIN "$NOISE_SCRIPT" \
  --embeddings alec/embeddings.npy \
  --labels alec/labels_clean.csv \
  --id-column vector_id \
  --label-column label \
  --output-labels alec/labels_cluster_k5_10.csv \
  --log-file alec/log_cluster_k5_10.csv \
  --mode cluster \
  --noise-level 0.10 \
  --cluster-size 5 \
  --metric cosine \
  --random-seed 42

$PYTHON_BIN "$NOISE_SCRIPT" \
  --embeddings alec/embeddings.npy \
  --labels alec/labels_clean.csv \
  --id-column vector_id \
  --label-column label \
  --output-labels alec/labels_border_10.csv \
  --log-file alec/log_border_10.csv \
  --mode border \
  --noise-level 0.10 \
  --boundary-k 10 \
  --boundary-top-frac 0.25 \
  --metric cosine \
  --random-seed 42

echo "All noise variants generated."
