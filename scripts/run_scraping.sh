#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET_ID="${VL_DATASET_ID:-}"
if [[ -z "$DATASET_ID" ]]; then
  echo "Set VL_DATASET_ID to run scraping/export." >&2
  exit 1
fi

$PYTHON_BIN -m visual_layer_capstone.scraping.visual_layer_scrape export-dataset \
  --dataset-id "$DATASET_ID" \
  --output-zip "${OUTPUT_ZIP:-./imagenet_export.zip}" \
  --format json \
  --poll-interval 30 \
  --max-wait 7200
