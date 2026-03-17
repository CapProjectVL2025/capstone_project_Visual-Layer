# Visual Layer Capstone: Reproducible Noise Pipeline

This repository contains the finalized, reproducible code for UCSB CMPSC 189A-B (Visual Layer team).

Users recreate data and outputs locally by providing their own paths.

## Project Goal
The project studies how **dataset label noise structure** affects vision model behavior.

Instead of only random label corruption, this codebase supports:
- random flips,
- **border-aware** flips near class boundaries in embedding space,
- **cluster-aware** flips that corrupt local neighborhoods coherently.

The final workflow also includes:
- ImageNet metadata cleaning with policy-based pruning,
- embedding generation/packing,
- noise visualization,
- streamed COCO ViT training with optional noisy-label overrides.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
├── src/
│   └── visual_layer_capstone/
│       ├── scraping/
│       ├── cleaning/
│       ├── embeddings/
│       ├── noise/
│       ├── visualization/
│       ├── training/
│       └── pipeline/
├── scripts/
└── docs/
```

`src/visual_layer_capstone` is organized as an implicit namespace package (no `__init__.py` files).

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## Stage-by-Stage Reproduction

### 1) (Optional) Export from Visual Layer
Requires environment credentials:
- `VL_API_KEY`
- `VL_API_SECRET`
- `VL_BASE_URL` (optional)
- `VL_DATASET_ID`

```bash
./scripts/run_scraping.sh
```

### 2) Clean and analyze ImageNet metadata
Core module:
`python -m visual_layer_capstone.cleaning.apply_clusters_to_imagenet`

Example:
```bash
python -m visual_layer_capstone.cleaning.apply_clusters_to_imagenet from-filtered-json \
  --partition-mode cluster \
  --cluster-input-dir PATH_TO_VL_EXPORTS \
  --policy configs/imagenet_cleaning_policy.yaml \
  --output-root PATH_TO_CLEANING_OUTPUT_ROOT \
  --mode metadata-only
```

Then analyze:
```bash
python -m visual_layer_capstone.cleaning.apply_clusters_to_imagenet analyze-cleaning-run \
  --run-dir PATH_TO_CLEANING_RUN_DIR
```

### 3) Pack embeddings + aligned labels
```bash
./scripts/run_embeddings.sh
```

`run_embeddings.sh` accepts path overrides via environment variables:
- `METADATA_CSV`
- `EMBEDDINGS_DIR`
- `OUT_NPY`
- `OUT_LABELS`

Default local outputs used by scripts/modules (all user-overridable):
- metadata CSV: `metadata/embeddings.csv`
- embedding vectors: `embeddings/`
- packed array: `embeddings.npy`
- clean labels: `labels_clean.csv`
- noisy labels/logs: `noise/`
- training checkpoints: `checkpoints/`
- plots: `plots/`

Or directly:
```bash
python -m visual_layer_capstone.embeddings.export_embeddings metadata-to-npy \
  --metadata-csv PATH_TO_EMBEDDING_METADATA_CSV \
  --embeddings-dir PATH_TO_EMBEDDING_PT_DIR \
  --out-npy PATH_TO_OUTPUT_EMBEDDINGS_NPY \
  --out-labels PATH_TO_OUTPUT_LABELS_CSV
```

### 4) Generate noise variants
```bash
./scripts/run_noise_variants.sh
```

`run_noise_variants.sh` accepts path overrides via environment variables:
- `CLEAN_LABELS`
- `EMBEDDINGS_NPY`
- `NOISE_DIR`

### 5) Visualize clean vs noisy labels
```bash
python -m visual_layer_capstone.visualization.noise_viz \
  --embeddings PATH_TO_EMBEDDINGS_NPY \
  --labels-clean PATH_TO_CLEAN_LABELS_CSV \
  --noisy-random PATH_TO_RANDOM_NOISY_LABELS_CSV \
  --noisy-border PATH_TO_BORDER_NOISY_LABELS_CSV \
  --noisy-cluster PATH_TO_CLUSTER_NOISY_LABELS_CSV \
  --log-random PATH_TO_RANDOM_LOG_CSV \
  --log-border PATH_TO_BORDER_LOG_CSV \
  --log-cluster PATH_TO_CLUSTER_LOG_CSV \
  --out-dir PATH_TO_PLOTS_DIR
```

### 6) Train ViT with noisy labels
```bash
./scripts/run_training.sh
```

Override paths:
```bash
NOISY_LABELS=PATH_TO_NOISY_LABELS_CSV ./scripts/run_training.sh
CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR ./scripts/run_training.sh
```

### 7) Run full configured pipeline
```bash
python -m visual_layer_capstone.pipeline.run_pipeline --config configs/pipeline.yaml
```

Enable stages by setting `enabled: true` in `configs/pipeline.yaml`.

## Key Configs
- `configs/imagenet_scrape.yaml`: Visual Layer export defaults.
- `configs/imagenet_cleaning_policy.yaml`: primary ImageNet policy used in cleaning runs.
- `configs/imagenet_cleaning_policy_conservative.yaml`: stricter cleaning policy variant for sensitivity checks.
- `configs/embeddings.yaml`: embedding extraction + pack settings.
- `configs/noise_random.yaml`: shared noise defaults and output paths.
- `configs/training_vit_b32.yaml`: baseline training hyperparameters.
- `configs/pipeline.yaml`: orchestrated stage list.

All config path fields are placeholders (`PATH_TO_...`) and should be replaced per environment.
