# Design: Reproducible Visual Layer Pipeline

## Objective
This repository reproduces the Visual Layer capstone experiments on how structured dataset noise impacts downstream model performance.

Core study principles reflected in this codebase:
- compare clean vs noisy training data,
- inject noise using realistic patterns in embedding space (not just random flips),
- evaluate training behavior under controlled corruption levels.

Repository policy for reproducibility:
- no committed datasets, archives, model artifacts, or notebooks,
- all filesystem paths are provided through CLI flags, environment variables, or `configs/*.yaml`,
- local artifacts are created by users during execution.

## Pipeline Stages

### 1) Export / Scrape metadata
Module: `visual_layer_capstone.scraping.visual_layer_scrape`

Purpose:
- connect to Visual Layer API,
- request dataset export,
- poll async export status,
- download final export zip.

Reproducibility-oriented behavior:
- output destination is explicitly caller-provided (`--output-zip` / `OUTPUT_ZIP`),
- API calls use retry/backoff + token refresh to tolerate transient failures,
- supports both `VL_API_*` and legacy credential env names.

### 2) Clean ImageNet metadata and analyze policy outcomes
Module: `visual_layer_capstone.cleaning.apply_clusters_to_imagenet`

Purpose:
- merge Visual Layer JSON exports,
- dedupe by media id/file,
- apply policy rules (uniqueness, issue confidence, tags, cluster dedupe),
- emit keep/drop manifests + per-image decisions,
- run analysis and variant comparisons.

### 3) Build embeddings and aligned label tables
Module: `visual_layer_capstone.embeddings.export_embeddings`

Purpose:
- generate object-level embeddings from streamed COCO (`extract-coco-clip`),
- convert metadata-linked `.pt` vectors into packed `embeddings.npy` + aligned `labels_clean.csv` (`metadata-to-npy`).

### 4) Inject controlled noise
Modules:
- `visual_layer_capstone.noise.random_noise`
- `visual_layer_capstone.noise.border_noise`
- `visual_layer_capstone.noise.cluster_noise`

Noise modes:
- random: uniform label flips to different classes,
- border: prioritize points near class boundaries in embedding space,
- cluster: flip neighborhoods around seeded points to induce coherent local corruption.

All modes write:
- corrupted label CSV,
- per-change log CSV (`index`, `original_label`, `new_label`, `reason`).

### 5) Visualize noise behavior
Module: `visual_layer_capstone.visualization.noise_viz`

Purpose:
- t-SNE projection and side-by-side clean/noisy comparisons,
- overlay changed points by corruption mechanism/log reason,
- class-pair diagnostics for ambiguous classes.

### 6) Train ViT under clean/noisy supervision
Module: `visual_layer_capstone.training.train_imagenet_vit`

Purpose:
- streamed COCO training with largest-object crop policy,
- optional noisy label override from CSV,
- checkpoint writing to user-controlled path (`--checkpoint-dir`, default `checkpoints`),
- optional W&B logging for reproducible comparisons.

### 7) Orchestrate full runs
Module: `visual_layer_capstone.pipeline.run_pipeline`

Purpose:
- run ordered stage commands from `configs/pipeline.yaml`.

## Data Contracts

### Embeddings + Labels alignment
`embeddings.npy` row `i` must correspond to `labels_clean.csv` row `i`.

Required label columns for noise modules:
- `vector_id` (default id column),
- `label` (default label column).

### Noise logs
Each noise run log row includes:
- `index` (row index in labels/embeddings),
- `original_label`,
- `new_label`,
- `reason` (noise-specific provenance).

## Configuration Strategy
YAML files in `configs/` capture fixed experimental defaults:
- `imagenet_scrape.yaml`
- `imagenet_cleaning_policy.yaml`
- `imagenet_cleaning_policy_conservative.yaml`
- `embeddings.yaml`
- `noise_random.yaml`
- `training_vit_b32.yaml`
- `pipeline.yaml`

Configs intentionally use placeholder paths (`PATH_TO_...`) that must be set by the reproducing user.

## Package Layout Note
`src/visual_layer_capstone` uses an implicit namespace-package layout (no `__init__.py` files). Module execution uses:

```bash
python -m visual_layer_capstone.<submodule>...
```

with `PYTHONPATH=src` (as done by scripts in `scripts/`).

## Reproducibility Guarantees
- deterministic random seeds for noise injection,
- explicit run scripts for each stage in `scripts/`,
- no hidden repository-artifact path assumptions (all artifact paths are caller-selected),
- logs and manifests emitted for every transformation stage.
