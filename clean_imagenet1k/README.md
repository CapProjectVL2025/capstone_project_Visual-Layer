# clean_imagenet1k

This directory contains the main cleaning pipeline script and the versioned cleaning policy.

## Files

| File | Description |
|---|---|
| `imagenet1k_cleaning_chunked.py` | Main public pipeline script |
| `cleaning_policy.yaml` | Versioned cleaning policy applied to ImageNet-1K |
| `connect_vl_api_f.py` | Visual Layer API helper retained for future integrations; not required by the public workflow |

---

## Input format

The `from-filtered-json` command reads a directory of JSON files, each produced
by a Visual Layer export. Every file must be a JSON object with a top-level
`"media_items"` list:

```json
{
  "media_items": [
    {
      "media_id": "abc123",
      "file_name": "n01440764/n01440764_1.JPEG",
      "uniqueness_score": 0.82,
      "metadata_items": [
        {
          "type": "image_label",
          "properties": { "category_name": "tench" }
        },
        {
          "type": "issue",
          "properties": { "issue_type": "blurry", "confidence": 0.91 }
        },
        {
          "type": "user_tag",
          "tag_name": "poor_quality"
        }
      ]
    }
  ]
}
```

Files are discovered recursively using the `--glob` pattern (default: `**/*.json`).
Multiple files in the directory are merged and deduplicated by `media_id`.

---

## Commands

### `from-filtered-json` — core cleaning pipeline

Merges local exports, applies the cleaning policy, and writes all guaranteed
run artifacts.

```bash
python3 clean_imagenet1k/imagenet1k_cleaning_chunked.py from-filtered-json \
  --partition-mode cluster \
  --cluster-input-dir data/vl_exports \
  --policy clean_imagenet1k/cleaning_policy.yaml \
  --output-root data \
  --mode metadata-only
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--partition-mode` | required | `cluster`, `class`, or `both` |
| `--cluster-input-dir` | — | Directory of JSON export files |
| `--policy` | required | Path to `cleaning_policy.yaml` |
| `--output-root` | `data` | Parent directory for run output folder |
| `--mode` | `metadata-only` | `metadata-only` or `with-images` |
| `--skip-invalid` | off | Skip unreadable JSON files instead of failing |
| `--expected-total-media-items` | 1,331,167 | Expected dataset size for coverage check |

---

### `analyze-cleaning-run` — per-run analysis

Reads `prune_decisions.jsonl` and `cleaning_summary.json` from a completed run
and writes analysis CSVs and optional plots.

```bash
python3 clean_imagenet1k/imagenet1k_cleaning_chunked.py analyze-cleaning-run \
  --run-dir data/clean_cluster_groups
```

Add `--skip-plots` to skip PNG generation (useful in headless environments).

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--run-dir` | required | Path to the run output directory |
| `--output-dir` | `<run-dir>/analysis` | Where to write analysis outputs |
| `--top-classes` | 20 | Classes by drop fraction to include in class-based reports, summaries, and plots |
| `--skip-plots` | off | Omit PNG generation |
| `--min-class-size` | 50 | Minimum class size before over-pruning flags fire |

---

### `compare-cleaning-runs` — cross-run comparison

Reads `cleaning_summary.json` from two or more completed runs and writes a
side-by-side comparison CSV.

```bash
python3 clean_imagenet1k/imagenet1k_cleaning_chunked.py compare-cleaning-runs \
  --run-dirs data/run_conservative data/run_balanced data/run_aggressive \
  --output-file data/policy_comparison.csv
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--run-dirs` | required | One or more run directories |
| `--output-file` | `data/policy_comparison.csv` | CSV output path |

---

### `stress-test-policy-variants` — policy sweep

Applies conservative, balanced, and aggressive variants of a base policy to the
same input directory and writes per-variant run folders plus a sweep summary.

```bash
python3 clean_imagenet1k/imagenet1k_cleaning_chunked.py stress-test-policy-variants \
  --input-dir data/vl_exports \
  --policy clean_imagenet1k/cleaning_policy.yaml \
  --output-root data/policy_sweep
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--input-dir` | required | Shared input export directory |
| `--policy` | required | Base policy YAML |
| `--output-root` | required | Root for variant run folders |
| `--variants` | conservative balanced aggressive | Variants to generate |
| `--skip-invalid` | off | Skip unreadable JSON files |

---

## Major functions

These are the main code paths to understand when reading
`imagenet1k_cleaning_chunked.py`:

| Function | Responsibility |
|---|---|
| `ingest_filtered_exports` | Load JSON export files and flatten `media_items` into a dataframe |
| `normalize_records` | Extract labels, issues, tags, and uniqueness scores into stable tabular fields |
| `merge_and_dedupe` | Merge export rows and drop duplicates by `media_id` or `file_name` |
| `validate_merged_coverage` | Check merged row count against the expected dataset size |
| `apply_cleaning_policy` | Apply uniqueness, issue, and tag-based drop rules and record reasons |
| `save_outputs` | Write canonical run artifacts including decisions, manifests, summary, and policy snapshot |
| `_write_cleaning_run_analysis` | Build the analysis CSV/JSON outputs for a completed run |
| `compare_cleaning_runs` | Summarize multiple completed runs into a single comparison CSV |
| `stress_test_policy_variants` | Derive multiple policy variants, run them on shared input, and compare results |

---

## Output tree — single run

```
data/clean_cluster_groups/
├── raw_merged_metadata.json          All records before cleaning
├── cleaned_imagenet1k.json           Records kept by the policy
├── metadata.json                     Compatibility alias for cleaned output
├── dropped_metadata.json             Records dropped by the policy
├── prune_decisions.jsonl             Per-image keep/drop decision + reasons
├── keep_filenames.txt                Filenames of kept images
├── drop_filenames.txt                Filenames of dropped images
├── cleaning_summary.json             Run metadata: timestamp, counts, paths
├── cleaning_policy.yaml              Policy snapshot for this run
├── README.md                         Short artifact description
└── analysis/
    ├── policy_analysis_summary.json
    ├── drop_reason_counts.csv
    ├── drop_reason_overlap.csv
    ├── drop_reason_combinations.csv
    ├── class_impact.csv
    ├── uniqueness_by_decision.csv
    ├── issue_confidence_by_decision.csv
    ├── issue_types_by_label.csv
    ├── issue_types_by_label_plot_groups.csv
    └── plots/                        (optional — skipped with --skip-plots)
        ├── kept_vs_dropped_pie_chart.png
        ├── top_classes_by_drop_fraction.png
        ├── issue_types_top_dropped_classes.png
        └── issue_types_by_label_pages/
```

---

## Output tree — policy sweep

```
data/policy_sweep/
├── policy_comparison.csv
├── policy_stress_test_summary.json
├── clean_policy_sweep_<base-policy>_conservative_groups/
│   └── (same structure as single run)
├── clean_policy_sweep_<base-policy>_balanced_groups/
│   └── (same structure as single run)
└── clean_policy_sweep_<base-policy>_aggressive_groups/
    └── (same structure as single run)
```

---

## Notes

- Plots are optional outputs. All commands except plot generation run without `matplotlib`.
- Set `MPLCONFIGDIR=/tmp/matplotlib` if you encounter font-cache errors in locked environments.
- `connect_vl_api_f.py` is retained in the directory for future integrations, but the documented public workflow does not require it.
- The published workflow reproduces the same artifact types as historical policy-sweep runs such as `data/imagenet1k_vl_policy_sweep_no_cluster/`, although the run-directory names may differ from older internal experiments.
