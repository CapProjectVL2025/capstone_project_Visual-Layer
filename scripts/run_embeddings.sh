#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

METADATA_CSV="${METADATA_CSV:-metadata/embeddings.csv}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-embeddings}"
OUT_NPY="${OUT_NPY:-embeddings.npy}"
OUT_LABELS="${OUT_LABELS:-labels_clean.csv}"

$PYTHON_BIN -m visual_layer_capstone.embeddings.export_embeddings metadata-to-npy \
  --metadata-csv "$METADATA_CSV" \
  --embeddings-dir "$EMBEDDINGS_DIR" \
  --out-npy "$OUT_NPY" \
  --out-labels "$OUT_LABELS"
