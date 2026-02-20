# CMPSC189 - Visual Layer

**Team Members:**  
- Alec Song  
- Rushil Gupta
- Kushagra Kanaujia
- Saeed Arellano
- Bhavya Ranjan

## Overview

This project explores how *structured label noise* affects model performance on large-scale vision datasets.  

## Project Summary
This repo focuses on an embedding-driven noise pipeline:

1. Build a clean label table aligned with embeddings.
2. Inject controlled noise into labels using one of several modes.
3. Visualize and inspect where the noisy labels land in t-SNE space.
4. Feed noisy labels into downstream training/evaluation pipelines.

## Data Model (Important)
In this workflow, each row is typically an **object-level embedding** (not necessarily one row per image).

- `embeddings.npy`: shape `[N, D]`
- clean labels CSV: `N` rows with at least:
  - `vector_id` (or your chosen id column)
  - `label`
- row `i` in labels must correspond to row `i` in embeddings.

`noise_injection.py` validates this row alignment for embedding-based modes.

## Noise Injection Script
Script: `alec/scripts/noise_injection.py`

### Supported modes
- `random`: randomly picks points and flips each to a random different class.
- `border`: prioritizes boundary-like points (small margin between nearest same-class and nearest different-class neighbors), then fills from global pool if needed.
- `cluster`: picks seed points and flips local neighbor clusters to a **single target label per seed**.

### Outputs
- `--output-labels`: noisy label CSV (same columns as input labels, label column replaced)
- `--log-file`: per-change log CSV with `index`, `original_label`, `new_label`, `reason`

For `border` mode:
- `reason=border` means changed from boundary pool
- `reason=border_fill_nn` means changed during fill phase
- runtime prints counts for both

## Environment
Install dependencies (example):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib torch datasets tqdm
```

## Quick Start
Run from `alec/` so relative paths stay simple:

```bash
cd /Users/alecsong/School/capstone/capstone_project_Visual-Layer/alec
mkdir -p labels logs plots
```

### 1) Optional: build `embeddings.npy` + clean labels from metadata

```bash
python3 scripts/export_embeddings_npy.py \
  --metadata-csv /Users/alecsong/School/capstone/metadata/embeddings.csv \
  --embeddings-dir /Users/alecsong/School/capstone/embeddings \
  --out-npy embeddings.npy \
  --out-labels labels/labels_clean.csv \
  --max-rows 0
```

`--max-rows 0` means no cap (use all available rows).

### 2) (Optional) create a faster 10k subset for debugging

```bash
python3 - <<'PY'
import numpy as np, pandas as pd
X = np.load('embeddings.npy')
df = pd.read_csv('labels/labels_clean.csv')
rng = np.random.RandomState(42)
idx = np.sort(rng.choice(len(df), size=10000, replace=False))
np.save('embeddings_10k.npy', X[idx])
df.iloc[idx].reset_index(drop=True).to_csv('labels/labels_clean_10k.csv', index=False)
print('wrote embeddings_10k.npy and labels/labels_clean_10k.csv')
PY
```

## Generate Noisy Labels

### Example: random / border / cluster at 10%

```bash
python3 scripts/noise_injection.py \
  --labels labels/labels_clean.csv \
  --output-labels labels/coco_labels_random_10.csv \
  --log-file logs/coco_log_random_10.csv \
  --mode random \
  --noise-level 0.10 \
  --random-seed 42

python3 scripts/noise_injection.py \
  --embeddings embeddings.npy \
  --labels labels/labels_clean.csv \
  --output-labels labels/coco_labels_border_10.csv \
  --log-file logs/coco_log_border_10.csv \
  --mode border \
  --noise-level 0.10 \
  --boundary-k 25 \
  --boundary-top-frac 0.25 \
  --nn-k 50 \
  --metric cosine \
  --random-seed 42

python3 scripts/noise_injection.py \
  --embeddings embeddings.npy \
  --labels labels/labels_clean.csv \
  --output-labels labels/coco_labels_cluster_10.csv \
  --log-file logs/coco_log_cluster_10.csv \
  --mode cluster \
  --noise-level 0.10 \
  --cluster-size 50 \
  --nn-k 50 \
  --metric cosine \
  --random-seed 42
```

### Example: generate 5%, 15%, 30% for `cluster` and `border`

```bash
for p in 05 15 30; do
  lv="0.${p}"

  python3 scripts/noise_injection.py \
    --embeddings embeddings.npy \
    --labels labels/labels_clean.csv \
    --output-labels "labels/coco_labels_cluster_${p}.csv" \
    --log-file "logs/coco_log_cluster_${p}.csv" \
    --mode cluster \
    --noise-level "$lv" \
    --cluster-size 50 \
    --nn-k 50 \
    --metric cosine \
    --random-seed 42

  python3 scripts/noise_injection.py \
    --embeddings embeddings.npy \
    --labels labels/labels_clean.csv \
    --output-labels "labels/coco_labels_border_${p}.csv" \
    --log-file "logs/coco_log_border_${p}.csv" \
    --mode border \
    --noise-level "$lv" \
    --boundary-k 25 \
    --boundary-top-frac 0.25 \
    --nn-k 50 \
    --metric cosine \
    --random-seed 42

done
```

## Visualizing Noise

### A) Cluster coherence diagnostic

```bash
python3 scripts/viz_issue_cluster_incoherence.py \
  --embeddings embeddings_10k.npy \
  --labels labels/labels_clean_10k.csv \
  --log logs/coco_log_cluster_10.csv \
  --out plots/issue_cluster_incoherence.png
```

What this shows:
- gray = background points
- colored = changed points
- color = `new_label`
- coherent clusters should tend to be one color per local cluster

### B) Pairwise clean vs noisy t-SNE view
Use `viz_tsne_mislabels.py` with clean labels plus one or more noisy CSVs:

```bash
python3 scripts/viz_tsne_mislabels.py \
  --embeddings embeddings_10k.npy \
  --labels-clean labels/labels_clean_10k.csv \
  --classes 2 39 \
  --noisy-random labels/coco_labels_random_10.csv \
  --noisy-border labels/coco_labels_border_10.csv \
  --noisy-cluster labels/coco_labels_cluster_10.csv \
  --out-dir plots
```

It writes one image per provided noise type, each with:
- left: clean labels
- right: corrupted labels
- `X` markers on changed points

## Practical Notes
- `random` mode does not require embeddings.
- `border` and `cluster` require embeddings and are much heavier at full dataset scale.
- For fast iteration, use the 10k subset.
- For full-size runs, prefer a high-memory VM.

## Troubleshooting
- Row mismatch error: your labels CSV and embeddings matrix are not aligned.
- Missing label/id columns: pass `--label-column` and `--id-column` explicitly.
- Slow runtime: reduce dataset size first (`embeddings_10k.npy`) and tune `nn-k`, `boundary-k`, `cluster-size`.

