# ImageNet-1K Dataset Cleaning with Visual Layer

This repository provides a reproducible dataset-cleaning pipeline for ImageNet-1K.
It applies a configurable quality policy to Visual Layer metadata exports and
produces a fully traceable set of artifacts suitable for noise-robustness research.

**Team:** Alec Song, Rushil Gupta, Kushagra Kanaujia, Saeed Arellano, Bhavya Ranjan

---

## What this repo reproduces

Given local Visual Layer metadata exports for ImageNet-1K, this pipeline:

1. Merges and deduplicates all partition exports into a single record set.
2. Applies a versioned cleaning policy (confidence thresholds, uniqueness filtering, tag rules).
3. Writes a complete set of guaranteed artifacts — cleaned records, drop decisions, filenames, summary.
4. Generates per-run analysis: drop reason counts, class impact, uniqueness distributions, issue matrices.
5. Supports side-by-side comparison of multiple policy variants.

Visual Layer is the **metadata source** — it provides per-image quality scores,
issue detections, and cluster assignments. The public workflow starts from
JSON exports already on disk; a live Visual Layer API connection is **not
required** to reproduce results.

---

## What is included vs excluded

| Included | Excluded |
|---|---|
| `clean_imagenet1k/imagenet1k_cleaning_chunked.py` — main pipeline script | Live VL API calls (optional, internal) |
| `clean_imagenet1k/cleaning_policy.yaml` — versioned policy | Raw image files (not redistributed) |
| `tests/` — smoke tests and fixture | Noise-injection scripts (research tooling only) |
| `requirements.txt` — locked dependencies | |

---

## Installation

Python 3.8 or later is required.

```bash
git clone https://github.com/CapProjectVL2025/capstone_project_Visual-Layer.git
cd capstone_project_Visual-Layer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Reproducing the paper workflow

### Step 1 — Run the cleaning pipeline

Place your Visual Layer JSON exports in a directory (e.g. `data/vl_exports/`).
Each file must contain a top-level `"media_items"` list.
See [clean_imagenet1k/README.md](clean_imagenet1k/README.md) for the exact input format.

```bash
python3 clean_imagenet1k/imagenet1k_cleaning_chunked.py from-filtered-json \
  --partition-mode cluster \
  --cluster-input-dir data/vl_exports \
  --policy clean_imagenet1k/cleaning_policy.yaml \
  --output-root data \
  --mode metadata-only
```

Outputs are written to `data/clean_cluster_groups/` for cluster-mode runs and
`data/clean_class_groups/` for class-mode runs.

### Step 2 — Analyze the cleaning run

```bash
python3 clean_imagenet1k/imagenet1k_cleaning_chunked.py analyze-cleaning-run \
  --run-dir data/clean_cluster_groups
```

Add `--skip-plots` to omit PNG generation.

### Step 3 — Compare policy variants (optional)

```bash
python3 clean_imagenet1k/imagenet1k_cleaning_chunked.py compare-cleaning-runs \
  --run-dirs data/run_conservative data/run_balanced data/run_aggressive \
  --output-file data/policy_comparison.csv
```

### Step 4 — Stress-test policy variants (optional)

```bash
python3 clean_imagenet1k/imagenet1k_cleaning_chunked.py stress-test-policy-variants \
  --input-dir data/vl_exports \
  --policy clean_imagenet1k/cleaning_policy.yaml \
  --output-root data/policy_sweep
```

---

## Output artifact contract

### Run outputs (`from-filtered-json`)

| File | Description |
|---|---|
| `raw_merged_metadata.json` | All records before any cleaning |
| `cleaned_imagenet1k.json` | Records kept by the policy |
| `metadata.json` | Compatibility alias for `cleaned_imagenet1k.json` |
| `dropped_metadata.json` | Records dropped by the policy |
| `prune_decisions.jsonl` | Per-image keep/drop decision with reasons |
| `keep_filenames.txt` | Filenames of kept images |
| `drop_filenames.txt` | Filenames of dropped images |
| `cleaning_summary.json` | Run metadata: timestamp, counts, artifact paths |
| `cleaning_policy.yaml` | Snapshot of the policy applied during this run |
| `README.md` | Short artifact guide for the run directory |

### Analysis outputs (`analyze-cleaning-run`)

| File | Description |
|---|---|
| `analysis/policy_analysis_summary.json` | Top drop reasons, flagged classes, policy snapshot |
| `analysis/drop_reason_counts.csv` | Image count per drop reason with bucket classification |
| `analysis/drop_reason_overlap.csv` | Pairwise Jaccard similarity between drop reasons |
| `analysis/drop_reason_combinations.csv` | Multi-reason drop patterns |
| `analysis/class_impact.csv` | Per-class drop fraction, issue hit rate, uniqueness stats |
| `analysis/uniqueness_by_decision.csv` | Uniqueness score distribution (kept vs dropped) |
| `analysis/issue_confidence_by_decision.csv` | Issue confidence distribution (kept vs dropped) |
| `analysis/issue_types_by_label.csv` | Issue type count matrix by class label |
| `analysis/issue_types_by_label_plot_groups.csv` | Grouped version for visualization |

### Optional outputs

| File | Description |
|---|---|
| `analysis/plots/*.png` | Generated when matplotlib is available and `--skip-plots` is not set |

### Sweep outputs (`stress-test-policy-variants`)

| File | Description |
|---|---|
| `policy_comparison.csv` | Side-by-side metrics across all variants |
| `policy_stress_test_summary.json` | Per-variant run summary |

---

## Smoke tests

Verify the installation and all public commands against a small fixture:

```bash
python3 -m unittest tests.test_policy_behavior -v
python3 -m pytest tests/ -v
```

Or run individually:

```bash
# CLI help
python3 clean_imagenet1k/imagenet1k_cleaning_chunked.py --help

# Fixture-based run
python3 clean_imagenet1k/imagenet1k_cleaning_chunked.py from-filtered-json \
  --partition-mode cluster \
  --cluster-input-dir tests/fixtures/sample_export \
  --policy clean_imagenet1k/cleaning_policy.yaml \
  --output-root /tmp/test_run \
  --mode metadata-only \
  --expected-total-media-items 10
```

---

## Limitations

- Raw ImageNet-1K images are not included; only metadata exports and analysis artifacts are provided.
- Visual Layer is used upstream to generate the local JSON exports consumed by this repo; the published workflow begins from those local exports.
- Plot generation requires `matplotlib`; all other commands run without it.
- The `--mode with-images` flag requires the original image files to be present locally.

---

## Citation

If you use this pipeline or the cleaned dataset in your research, please cite:

```
@misc{visuallayer-imagenet-cleaning-2025,
  title  = {Reproducible ImageNet-1K Dataset Cleaning with Visual Layer},
  author = {Song, Alec and Gupta, Rushil and Kanaujia, Kushagra and Arellano, Saeed and Ranjan, Bhavya},
  year   = {2026},
  url    = {https://github.com/CapProjectVL2025/capstone_project_Visual-Layer}
}
```
