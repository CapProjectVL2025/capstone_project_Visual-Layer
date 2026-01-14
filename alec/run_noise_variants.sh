#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere, but ensure we operate relative to the repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NOISE_SCRIPT="$ROOT_DIR/alec/noise_injection.py"

$PYTHON_BIN "$NOISE_SCRIPT" \
  --embeddings embeddings.npy \
  --labels labels_clean.csv \
  --id-column vector_id \
  --label-column label \
  --output-labels labels_nn_10.csv \
  --log-file log_nn_10.csv \
  --mode nearest_neighbor \
  --noise-level 0.10 \
  --cluster-size 1 \
  --metric cosine \
  --random-seed 42

$PYTHON_BIN "$NOISE_SCRIPT" \
  --embeddings embeddings.npy \
  --labels labels_clean.csv \
  --id-column vector_id \
  --label-column label \
  --output-labels labels_cluster_k5_10.csv \
  --log-file log_cluster_k5_10.csv \
  --mode cluster \
  --noise-level 0.10 \
  --cluster-size 5 \
  --metric cosine \
  --random-seed 42

$PYTHON_BIN "$NOISE_SCRIPT" \
  --embeddings embeddings.npy \
  --labels labels_clean.csv \
  --id-column vector_id \
  --label-column label \
  --output-labels labels_boundary_cluster_k5_10.csv \
  --log-file log_boundary_cluster_k5_10.csv \
  --mode boundary_cluster \
  --noise-level 0.10 \
  --cluster-size 5 \
  --boundary-k 10 \
  --metric cosine \
  --random-seed 42

echo "All noise variants generated."