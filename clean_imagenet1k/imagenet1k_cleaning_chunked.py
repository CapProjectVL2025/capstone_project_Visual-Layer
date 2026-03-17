#!/usr/bin/env python3
"""
ImageNet-1K Cleaning Pipeline
==============================

Applies a configurable cleaning policy to Visual Layer metadata exports
and produces a reproducible set of artifacts for dataset-quality research.

Public workflow (no live API required):
    1. Start from local Visual Layer JSON exports already on disk.
    2. Run ``from-filtered-json`` to merge, deduplicate, apply policy, and
       write all guaranteed run artifacts.
    3. Run ``analyze-cleaning-run`` to produce analysis CSVs and optional plots.
    4. Run ``compare-cleaning-runs`` to compare policy variants side-by-side.
    5. Run ``stress-test-policy-variants`` to sweep conservative / balanced /
       aggressive variants over the same input.

Supported public commands:
    from-filtered-json          Core cleaning pipeline from local exports.
    analyze-cleaning-run        Per-run analysis from prune_decisions.jsonl.
    compare-cleaning-runs       Cross-run policy comparison CSV.
    stress-test-policy-variants Policy sweep over a shared input directory.

Guaranteed run outputs (from-filtered-json):
    raw_merged_metadata.json    All records before cleaning.
    cleaned_imagenet1k.json     Records kept by the policy.
    metadata.json               Compatibility alias for cleaned output.
    dropped_metadata.json       Records dropped by the policy.
    prune_decisions.jsonl       Per-image keep/drop decision with reasons.
    keep_filenames.txt          Filenames of kept images.
    drop_filenames.txt          Filenames of dropped images.
    cleaning_summary.json       Run metadata and counts.
    cleaning_policy.yaml        Policy snapshot applied during this run.
    README.md                   Short artifact description for the run directory.

Guaranteed analysis outputs (analyze-cleaning-run):
    analysis/policy_analysis_summary.json
    analysis/drop_reason_counts.csv
    analysis/drop_reason_overlap.csv
    analysis/drop_reason_combinations.csv
    analysis/class_impact.csv
    analysis/uniqueness_by_decision.csv
    analysis/issue_confidence_by_decision.csv
    analysis/issue_types_by_label.csv
    analysis/issue_types_by_label_plot_groups.csv

Optional outputs:
    analysis/plots/*.png        Generated only when matplotlib is available
                                and --skip-plots is not set.
"""

import argparse
from copy import deepcopy
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from tqdm import tqdm

INTERNAL_COLUMNS = {"__source_export_file", "__source_partition"}
IMAGENET1K_EXPECTED_MEDIA_ITEMS = 1_331_167


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def normalize_issue_type(name: str) -> str:
    if not name:
        return ""
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "mislabels": "mislabel",
        "mislabeled": "mislabel",
        "label_outlier": "label_outlier",
        "label_outliers": "label_outlier",
        "outlier": "visual_outlier",
        "outliers": "visual_outlier",
        "visual_outliers": "visual_outlier",
    }
    return aliases.get(key, key)


def _normalize_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _normalize_issue_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    issues: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        issue_type = normalize_issue_type(item.get("issue_type", ""))
        conf_raw = item.get("confidence", 0.0)
        try:
            confidence = float(conf_raw)
        except Exception:
            confidence = 0.0
        if issue_type:
            issues.append({"issue_type": issue_type, "confidence": confidence})
    return issues


def _reason_bucket(reason: str) -> str:
    if not reason:
        return "other"
    if reason.startswith("low_uniqueness<") or reason == "duplicate_in_cluster":
        return "likely_redundant"
    if reason.startswith("issue_"):
        issue_name = normalize_issue_type(reason.replace("issue_", "", 1))
        if issue_name in {"mislabel", "corrupted"}:
            return "likely_harmful"
        if issue_name in {"label_outlier", "visual_outlier"}:
            return "review_atypical"
        if issue_name in {"blurry", "dark", "overexposed"}:
            return "quality_related"
        return "issue_other"
    if reason.startswith("user_tag:"):
        lowered = reason.lower()
        if any(tag in lowered for tag in ("wrong_class", "incorrect_label", "mislabeled")):
            return "likely_harmful"
        return "manual_review"
    return "other"


def _group_issue_type_for_plot(issue_type: str) -> str:
    normalized = normalize_issue_type(issue_type)
    plot_aliases = {
        "dark": "visual_outlier",
    }
    return plot_aliases.get(normalized, normalized)


def _series_stats(series: pd.Series) -> Dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": int(numeric.count()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "min": float(numeric.min()),
        "p10": float(numeric.quantile(0.10)),
        "p25": float(numeric.quantile(0.25)),
        "p75": float(numeric.quantile(0.75)),
        "p90": float(numeric.quantile(0.90)),
        "max": float(numeric.max()),
    }


def _load_cleaning_run(run_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[Dict[str, Any]]]:
    prune_path = run_dir / "prune_decisions.jsonl"
    summary_path = run_dir / "cleaning_summary.json"
    policy_path = run_dir / "cleaning_policy.yaml"

    if not prune_path.exists():
        raise FileNotFoundError(f"Missing prune decisions file: {prune_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing cleaning summary file: {summary_path}")

    prune_df = pd.read_json(prune_path, lines=True)
    with open(summary_path, "r") as f:
        summary = json.load(f)

    policy = None
    if policy_path.exists():
        with open(policy_path, "r") as f:
            policy = yaml.safe_load(f)

    return prune_df, summary, policy


def _prepare_prune_analysis_df(prune_df: pd.DataFrame) -> pd.DataFrame:
    df = prune_df.copy()

    if "keep" not in df.columns:
        raise ValueError("prune_decisions.jsonl is missing required column 'keep'")

    df["keep"] = df["keep"].fillna(False).astype(bool)
    if "label" not in df.columns:
        df["label"] = "unknown"
    df["label"] = df["label"].fillna("unknown").astype(str)
    df.loc[df["label"].str.strip() == "", "label"] = "unknown"

    if "uniqueness_score" not in df.columns:
        df["uniqueness_score"] = 0.0
    df["uniqueness_score"] = pd.to_numeric(df["uniqueness_score"], errors="coerce")

    if "drop_reasons" not in df.columns:
        df["drop_reasons"] = [[] for _ in range(len(df))]
    df["drop_reasons"] = df["drop_reasons"].apply(_normalize_string_list)
    df["reason_count"] = df["drop_reasons"].apply(len)
    df["reason_combo"] = df["drop_reasons"].apply(lambda x: " | ".join(sorted(set(x))) if x else "")

    if "issues" not in df.columns:
        df["issues"] = [[] for _ in range(len(df))]
    df["issues"] = df["issues"].apply(_normalize_issue_list)
    df["issue_types"] = df["issues"].apply(lambda x: sorted({normalize_issue_type(i.get("issue_type", "")) for i in x if i.get("issue_type")}))
    df["issue_count"] = df["issue_types"].apply(len)
    df["has_issues"] = df["issue_count"] > 0
    df["max_issue_confidence"] = df["issues"].apply(
        lambda x: max((float(i.get("confidence", 0.0)) for i in x), default=None)
    )

    df["decision"] = df["keep"].map({True: "kept", False: "dropped"})
    return df


def _build_drop_reason_counts(df: pd.DataFrame) -> pd.DataFrame:
    dropped = df[~df["keep"]].copy()
    total_images = len(df)
    total_dropped = len(dropped)
    if dropped.empty:
        return pd.DataFrame(
            columns=["reason", "reason_bucket", "count", "share_of_total", "share_of_dropped"]
        )

    exploded = dropped[["drop_reasons"]].explode("drop_reasons")
    exploded = exploded.dropna(subset=["drop_reasons"])
    exploded = exploded.rename(columns={"drop_reasons": "reason"})

    counts = exploded["reason"].value_counts().rename_axis("reason").reset_index(name="count")
    counts["reason_bucket"] = counts["reason"].apply(_reason_bucket)
    counts["share_of_total"] = counts["count"] / total_images if total_images else 0.0
    counts["share_of_dropped"] = counts["count"] / total_dropped if total_dropped else 0.0
    return counts.sort_values(["count", "reason"], ascending=[False, True]).reset_index(drop=True)


def _build_drop_reason_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    dropped = df[~df["keep"]].copy()
    pair_counts: Counter[Tuple[str, str]] = Counter()
    single_counts: Counter[str] = Counter()

    for reasons in dropped["drop_reasons"]:
        unique_reasons = sorted(set(reasons))
        for reason in unique_reasons:
            single_counts[reason] += 1
        for reason_a, reason_b in combinations(unique_reasons, 2):
            pair_counts[(reason_a, reason_b)] += 1

    rows: List[Dict[str, Any]] = []
    for (reason_a, reason_b), shared_count in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0])):
        denom = single_counts[reason_a] + single_counts[reason_b] - shared_count
        rows.append(
            {
                "reason_a": reason_a,
                "reason_b": reason_b,
                "shared_count": int(shared_count),
                "jaccard": float(shared_count / denom) if denom else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=["reason_a", "reason_b", "shared_count", "jaccard"])


def _build_drop_reason_combinations(df: pd.DataFrame) -> pd.DataFrame:
    dropped = df[~df["keep"]].copy()
    if dropped.empty:
        return pd.DataFrame(columns=["reason_combo", "count", "reason_count"])

    counts = dropped["reason_combo"].value_counts().rename_axis("reason_combo").reset_index(name="count")
    counts["reason_count"] = counts["reason_combo"].apply(lambda x: 0 if not x else len(str(x).split(" | ")))
    return counts.sort_values(["count", "reason_combo"], ascending=[False, True]).reset_index(drop=True)


def _build_class_impact(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = df.groupby("label", dropna=False)

    for label, group in grouped:
        total = int(len(group))
        kept = int(group["keep"].sum())
        dropped = total - kept
        dropped_rows = group[~group["keep"]]
        drop_reason_counter: Counter[str] = Counter()
        for reasons in dropped_rows["drop_reasons"]:
            drop_reason_counter.update(set(reasons))
        top_drop_reason = drop_reason_counter.most_common(1)[0][0] if drop_reason_counter else ""

        rows.append(
            {
                "label": label,
                "total_images": total,
                "kept_images": kept,
                "dropped_images": dropped,
                "drop_fraction": float(dropped / total) if total else 0.0,
                "issue_hit_rate": float(group["has_issues"].mean()) if total else 0.0,
                "multi_reason_drop_rate": float((dropped_rows["reason_count"] > 1).mean()) if dropped else 0.0,
                "mean_uniqueness_all": float(group["uniqueness_score"].dropna().mean()) if group["uniqueness_score"].notna().any() else None,
                "mean_uniqueness_kept": float(group.loc[group["keep"], "uniqueness_score"].dropna().mean()) if kept and group.loc[group["keep"], "uniqueness_score"].notna().any() else None,
                "mean_uniqueness_dropped": float(dropped_rows["uniqueness_score"].dropna().mean()) if dropped and dropped_rows["uniqueness_score"].notna().any() else None,
                "top_drop_reason": top_drop_reason,
            }
        )

    class_impact = pd.DataFrame(rows)
    return class_impact.sort_values(["drop_fraction", "dropped_images", "label"], ascending=[False, False, True]).reset_index(drop=True)


def _annotate_over_pruning(
    class_impact: pd.DataFrame,
    min_class_size: int,
    min_dropped_images: int,
) -> pd.DataFrame:
    annotated = class_impact.copy()
    if annotated.empty:
        annotated["drop_fraction_robust_z"] = pd.Series(dtype=float)
        annotated["over_pruned_severity"] = pd.Series(dtype=str)
        annotated["over_pruned_flag"] = pd.Series(dtype=bool)
        return annotated

    eligible_mask = (
        (annotated["total_images"] >= int(min_class_size))
        & (annotated["dropped_images"] >= int(min_dropped_images))
    )
    eligible = annotated[eligible_mask]

    annotated["drop_fraction_robust_z"] = 0.0
    annotated["over_pruned_severity"] = "normal"
    annotated["over_pruned_flag"] = False

    if eligible.empty:
        return annotated

    median = float(eligible["drop_fraction"].median())
    mad = float((eligible["drop_fraction"] - median).abs().median())
    if mad > 0:
        annotated["drop_fraction_robust_z"] = 0.6745 * (annotated["drop_fraction"] - median) / mad

    p90 = float(eligible["drop_fraction"].quantile(0.90))
    p95 = float(eligible["drop_fraction"].quantile(0.95))

    elevated_mask = eligible_mask & (
        (annotated["drop_fraction"] >= p90)
        | (annotated["drop_fraction_robust_z"] >= 2.0)
    )
    high_mask = eligible_mask & (
        (annotated["drop_fraction"] >= p95)
        | (annotated["drop_fraction_robust_z"] >= 3.0)
    )

    annotated.loc[elevated_mask, "over_pruned_severity"] = "elevated"
    annotated.loc[high_mask, "over_pruned_severity"] = "high"
    annotated["over_pruned_flag"] = annotated["over_pruned_severity"].isin(["elevated", "high"])

    return annotated.sort_values(
        ["over_pruned_flag", "drop_fraction", "dropped_images", "label"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def _build_distribution_summary(df: pd.DataFrame, value_col: str, output_col_name: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for decision, group in df.groupby("decision"):
        stats = _series_stats(group[value_col])
        stats[output_col_name] = decision
        rows.append(stats)
    ordered_cols = [output_col_name, "count", "mean", "median", "min", "p10", "p25", "p75", "p90", "max"]
    return pd.DataFrame(rows)[ordered_cols]


def _build_issue_type_by_label_matrix(
    df: pd.DataFrame,
    class_impact: pd.DataFrame,
) -> pd.DataFrame:
    label_order = class_impact["label"].tolist() if not class_impact.empty else sorted(df["label"].dropna().astype(str).unique())
    issue_rows: List[Dict[str, Any]] = []
    dropped = df[~df["keep"]].copy()

    for _, row in dropped.iterrows():
        for issue_type in row.get("issue_types", []):
            issue_rows.append({"label": row["label"], "issue_type": issue_type})

    if issue_rows:
        matrix = pd.DataFrame(issue_rows).groupby(["label", "issue_type"]).size().unstack(fill_value=0)
    else:
        matrix = pd.DataFrame(index=label_order)

    if label_order:
        matrix = matrix.reindex(label_order, fill_value=0)
    matrix.index.name = "label"
    return matrix


def _group_issue_matrix_for_plot(issue_matrix: pd.DataFrame) -> pd.DataFrame:
    if issue_matrix.empty:
        return issue_matrix.copy()
    grouped_columns = [_group_issue_type_for_plot(col) for col in issue_matrix.columns]
    grouped = issue_matrix.copy()
    grouped.columns = grouped_columns
    grouped = grouped.T.groupby(level=0).sum().T
    grouped.index.name = issue_matrix.index.name
    return grouped


def _build_policy_analysis_summary(
    df: pd.DataFrame,
    summary: Dict[str, Any],
    policy: Optional[Dict[str, Any]],
    reason_counts: pd.DataFrame,
    overlap_df: pd.DataFrame,
    combo_df: pd.DataFrame,
    class_impact: pd.DataFrame,
    run_dir: Path,
    output_dir: Path,
    top_classes_limit: int,
) -> Dict[str, Any]:
    total_images = int(len(df))
    kept_images = int(df["keep"].sum())
    dropped_images = total_images - kept_images
    drop_bucket_counts = (
        reason_counts.groupby("reason_bucket")["count"].sum().sort_values(ascending=False).to_dict()
        if not reason_counts.empty
        else {}
    )
    combo_counts = {
        "single_reason": int((df["reason_count"] == 1).sum()),
        "two_reasons": int((df["reason_count"] == 2).sum()),
        "three_or_more_reasons": int((df["reason_count"] >= 3).sum()),
    }

    top_classes = class_impact.head(top_classes_limit)[
        ["label", "drop_fraction", "dropped_images", "top_drop_reason", "over_pruned_flag", "over_pruned_severity"]
    ].to_dict(orient="records")
    top_reason_pairs = overlap_df.head(20).to_dict(orient="records") if not overlap_df.empty else []
    top_reason_combinations = combo_df.head(20).to_dict(orient="records") if not combo_df.empty else []
    flagged_classes = class_impact[class_impact["over_pruned_flag"]].head(top_classes_limit)[
        ["label", "drop_fraction", "dropped_images", "top_drop_reason", "over_pruned_severity"]
    ].to_dict(orient="records")

    summary_payload: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "analysis_output_dir": str(output_dir),
        "policy_name": summary.get("policy_name") or (policy or {}).get("policy_name", "unknown"),
        "total_images": total_images,
        "kept": kept_images,
        "dropped": dropped_images,
        "drop_rate": float(dropped_images / total_images) if total_images else 0.0,
        "consistency_checks": {
            "summary_total_matches_prune_rows": int(summary.get("total_images", -1)) == total_images,
            "summary_kept_matches_prune_rows": int(summary.get("kept", -1)) == kept_images,
            "summary_dropped_matches_prune_rows": int(summary.get("dropped", -1)) == dropped_images,
        },
        "drop_bucket_counts": drop_bucket_counts,
        "multi_reason_breakdown": combo_counts,
        "top_drop_reasons": reason_counts.head(20).to_dict(orient="records"),
        "top_reason_pairs": top_reason_pairs,
        "top_reason_combinations": top_reason_combinations,
        "top_over_pruned_classes": top_classes,
        "flagged_over_pruned_classes": flagged_classes,
        "artifacts": {
            "drop_reason_counts_csv": str(output_dir / "drop_reason_counts.csv"),
            "drop_reason_overlap_csv": str(output_dir / "drop_reason_overlap.csv"),
            "drop_reason_combinations_csv": str(output_dir / "drop_reason_combinations.csv"),
            "class_impact_csv": str(output_dir / "class_impact.csv"),
            "uniqueness_by_decision_csv": str(output_dir / "uniqueness_by_decision.csv"),
            "issue_confidence_by_decision_csv": str(output_dir / "issue_confidence_by_decision.csv"),
            "issue_types_by_label_csv": str(output_dir / "issue_types_by_label.csv"),
        },
    }

    if policy:
        summary_payload["policy_snapshot"] = {
            "policy_version": policy.get("policy_version"),
            "policy_name": policy.get("policy_name"),
            "uniqueness_threshold": policy.get("uniqueness_threshold"),
            "dedupe_by_cluster": policy.get("dedupe_by_cluster"),
            "drop_issues": policy.get("drop_issues", []),
            "drop_tags": policy.get("drop_tags", []),
        }

    return summary_payload


def _import_matplotlib():
    try:
        mpl_cache = Path(os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"))
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        print(f"WARNING: Plot generation skipped: {exc}")
        return None


def _generate_analysis_plots(
    total_images: int,
    kept_images: int,
    dropped_images: int,
    policy_name: str,
    class_impact: pd.DataFrame,
    issue_plot_matrix: pd.DataFrame,
    output_dir: Path,
    top_classes_limit: int,
) -> List[str]:
    plt = _import_matplotlib()
    if plt is None:
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []

    if total_images > 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        counts = [kept_images, dropped_images]
        labels = [f"Kept\n{kept_images:,} ({kept_images / total_images:.1%})", f"Dropped\n{dropped_images:,} ({dropped_images / total_images:.1%})"]
        colors = ["#3b8a64", "#c44e4e"]
        ax.pie(counts, labels=labels, colors=colors, startangle=90, counterclock=False)
        ax.set_title(f"ImageNet Images Kept vs Dropped\n{policy_name.replace('_', ' ').title()}")
        fig.tight_layout()
        path = plots_dir / "kept_vs_dropped_pie_chart.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        generated.append(str(path))

    if not class_impact.empty:
        top_classes = class_impact.head(top_classes_limit).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, max(6, top_classes_limit * 0.3)))
        ax.barh(top_classes["label"], top_classes["drop_fraction"], color="#8a5a44")
        ax.set_title(f"Top {len(top_classes)} Classes by Drop Fraction")
        ax.set_xlabel("Drop Fraction")
        ax.set_ylabel("Class Label")
        fig.tight_layout()
        path = plots_dir / "top_classes_by_drop_fraction.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        generated.append(str(path))

        top_class_names = list(class_impact.head(min(top_classes_limit, 10))["label"])
        available_labels = [label for label in top_class_names if label in issue_plot_matrix.index]
        if available_labels:
            stacked = issue_plot_matrix.reindex(available_labels, fill_value=0)
            if stacked.to_numpy().sum() > 0:
                fig, ax = plt.subplots(figsize=(12, 7))
                stacked.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
                ax.set_title("Issue Types Across Top Dropped Classes")
                ax.set_xlabel("Class Label")
                ax.set_ylabel("Dropped Images with Issue")
                ax.legend(title="Issue Type", bbox_to_anchor=(1.02, 1), loc="upper left")
                fig.tight_layout()
                path = plots_dir / "issue_types_top_dropped_classes.png"
                fig.savefig(path, dpi=180)
                plt.close(fig)
                generated.append(str(path))

    return generated


def _generate_issue_type_by_label_pages(
    issue_matrix: pd.DataFrame,
    output_dir: Path,
    page_size: int = 40,
) -> List[str]:
    plt = _import_matplotlib()
    if plt is None or issue_matrix.empty:
        return []

    plots_dir = output_dir / "plots" / "issue_types_by_label_pages"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []

    nonzero_matrix = issue_matrix.loc[issue_matrix.sum(axis=1) > 0]
    if nonzero_matrix.empty:
        return []

    for page_idx, start in enumerate(range(0, len(nonzero_matrix), page_size), start=1):
        subset = nonzero_matrix.iloc[start:start + page_size]
        fig_height = max(8, len(subset) * 0.28)
        fig, ax = plt.subplots(figsize=(14, fig_height))
        subset.iloc[::-1].plot(kind="barh", stacked=True, ax=ax, colormap="tab20")
        ax.set_title(f"Issue Types Across Dropped Labels (Page {page_idx})")
        ax.set_xlabel("Dropped Images with Issue")
        ax.set_ylabel("Class Label")
        ax.legend(title="Issue Type", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        path = plots_dir / f"issue_types_by_label_page_{page_idx:03d}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        generated.append(str(path))

    return generated


def _write_cleaning_run_analysis(
    run_dir: Path,
    output_dir: Path,
    top_classes: int,
    skip_plots: bool,
    min_class_size: int,
    min_dropped_images: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    prune_df, summary, policy = _load_cleaning_run(run_dir)
    df = _prepare_prune_analysis_df(prune_df)

    reason_counts = _build_drop_reason_counts(df)
    overlap_df = _build_drop_reason_overlaps(df)
    combo_df = _build_drop_reason_combinations(df)
    class_impact = _annotate_over_pruning(
        _build_class_impact(df),
        min_class_size=min_class_size,
        min_dropped_images=min_dropped_images,
    )
    issue_matrix = _build_issue_type_by_label_matrix(df, class_impact)
    issue_plot_matrix = _group_issue_matrix_for_plot(issue_matrix)
    uniqueness_stats = _build_distribution_summary(df, "uniqueness_score", "decision")
    issue_conf_stats = _build_distribution_summary(df, "max_issue_confidence", "decision")

    reason_counts.to_csv(output_dir / "drop_reason_counts.csv", index=False)
    overlap_df.to_csv(output_dir / "drop_reason_overlap.csv", index=False)
    combo_df.to_csv(output_dir / "drop_reason_combinations.csv", index=False)
    class_impact.to_csv(output_dir / "class_impact.csv", index=False)
    uniqueness_stats.to_csv(output_dir / "uniqueness_by_decision.csv", index=False)
    issue_conf_stats.to_csv(output_dir / "issue_confidence_by_decision.csv", index=False)
    issue_matrix.to_csv(output_dir / "issue_types_by_label.csv")
    issue_plot_matrix.to_csv(output_dir / "issue_types_by_label_plot_groups.csv")

    analysis_summary = _build_policy_analysis_summary(
        df=df,
        summary=summary,
        policy=policy,
        reason_counts=reason_counts,
        overlap_df=overlap_df,
        combo_df=combo_df,
        class_impact=class_impact,
        run_dir=run_dir,
        output_dir=output_dir,
        top_classes_limit=top_classes,
    )

    if skip_plots:
        plot_paths: List[str] = []
    else:
        plot_paths = _generate_analysis_plots(
            total_images=analysis_summary["total_images"],
            kept_images=analysis_summary["kept"],
            dropped_images=analysis_summary["dropped"],
            policy_name=analysis_summary["policy_name"],
            class_impact=class_impact,
            issue_plot_matrix=issue_plot_matrix,
            output_dir=output_dir,
            top_classes_limit=top_classes,
        )
        plot_paths.extend(
            _generate_issue_type_by_label_pages(
                issue_matrix=issue_plot_matrix,
                output_dir=output_dir,
            )
        )
    analysis_summary["artifacts"]["plots"] = plot_paths
    analysis_summary["over_pruning_thresholds"] = {
        "min_class_size": int(min_class_size),
        "min_dropped_images": int(min_dropped_images),
    }

    with open(output_dir / "policy_analysis_summary.json", "w") as f:
        json.dump(analysis_summary, f, indent=2)

    return analysis_summary


def analyze_cleaning_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"

    analysis_summary = _write_cleaning_run_analysis(
        run_dir=run_dir,
        output_dir=output_dir,
        top_classes=args.top_classes,
        skip_plots=args.skip_plots,
        min_class_size=args.min_class_size,
        min_dropped_images=args.min_dropped_images,
    )

    print_header("Cleaning Analysis Complete")
    print(f"Run dir: {run_dir}")
    print(f"Analysis dir: {output_dir}")
    print(f"Total images: {analysis_summary['total_images']:,}")
    print(f"Dropped images: {analysis_summary['dropped']:,} ({analysis_summary['drop_rate']:.2%})")
    return 0


def _variant_uniqueness_threshold(base_policy: Dict[str, Any], variant_name: str) -> Optional[float]:
    base_raw = base_policy.get("uniqueness_threshold")
    base_threshold = float(base_raw) if base_raw is not None else None
    if variant_name == "aggressive":
        return base_threshold
    target = 0.20 if variant_name == "conservative" else 0.25
    if base_threshold is None:
        return target
    return min(base_threshold, target)


def _variant_issue_floor(issue_type: str, variant_name: str) -> Optional[float]:
    if variant_name == "aggressive":
        return None
    if variant_name == "conservative":
        if issue_type == "corrupted":
            return 0.5
        return 0.9
    balanced_defaults = {
        "mislabel": 0.8,
        "label_outlier": 0.85,
        "visual_outlier": 0.8,
        "blurry": 0.85,
        "dark": 0.9,
        "overexposed": 0.9,
        "corrupted": 0.5,
    }
    return balanced_defaults.get(issue_type, 0.8)


def build_policy_variant(base_policy: Dict[str, Any], variant_name: str) -> Dict[str, Any]:
    normalized_variant = str(variant_name).strip().lower()
    if normalized_variant not in {"conservative", "balanced", "aggressive"}:
        raise ValueError(f"Unsupported policy variant: {variant_name}")

    variant = deepcopy(base_policy)
    base_name = str(base_policy.get("policy_name", "policy")).strip() or "policy"
    if normalized_variant == "aggressive":
        if "aggressive" not in base_name.lower():
            variant["policy_name"] = f"{base_name}_aggressive"
    else:
        variant["policy_name"] = f"{base_name}_{normalized_variant}"

    variant["analysis_variant"] = normalized_variant
    variant["uniqueness_threshold"] = _variant_uniqueness_threshold(base_policy, normalized_variant)

    updated_rules: List[Dict[str, Any]] = []
    for issue_rule in variant.get("drop_issues", []):
        if not isinstance(issue_rule, dict):
            continue
        updated = deepcopy(issue_rule)
        issue_type = normalize_issue_type(updated.get("issue_type", ""))
        floor = _variant_issue_floor(issue_type, normalized_variant)
        if floor is not None:
            try:
                current = float(updated.get("min_confidence", 0.0))
            except Exception:
                current = 0.0
            updated["min_confidence"] = max(current, floor)
        updated_rules.append(updated)
    variant["drop_issues"] = updated_rules
    return variant


def _sanitize_name(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "run"


def stress_test_policy_variants(args: argparse.Namespace) -> int:
    source_partition = args.source_partition or "policy_sweep"
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    base_policy = load_policy(args.policy)
    raw_df = ingest_filtered_exports(
        input_dir=input_dir,
        glob_pattern=args.glob,
        source_partition=source_partition,
        skip_invalid=args.skip_invalid,
    )
    norm_df = normalize_records(raw_df)
    merged_df = merge_and_dedupe(norm_df)
    coverage = validate_merged_coverage(
        merged_df=merged_df,
        expected_total_media_items=args.expected_total_media_items,
        require_full_coverage=args.require_full_coverage,
    )

    generated_policy_dir = output_root / "_generated_policies"
    generated_policy_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows: List[Dict[str, Any]] = []
    run_dirs: List[Path] = []

    for variant_name in args.variants:
        variant_policy = build_policy_variant(base_policy, variant_name)
        variant_slug = _sanitize_name(variant_policy.get("policy_name", variant_name))
        variant_policy_path = generated_policy_dir / f"{variant_slug}.yaml"
        with open(variant_policy_path, "w") as f:
            yaml.safe_dump(variant_policy, f, sort_keys=False)

        keep_df, drop_df, drop_reasons = apply_cleaning_policy(merged_df, variant_policy)
        run_dir = output_root / f"clean_{source_partition}_{variant_slug}_groups"
        save_outputs(
            merged_df=merged_df,
            keep_df=keep_df,
            drop_df=drop_df,
            drop_reasons=drop_reasons,
            policy=variant_policy,
            out_dir=run_dir,
            run_mode="metadata-only",
            policy_path=str(variant_policy_path),
            images_root=None,
        )
        analysis_summary = _write_cleaning_run_analysis(
            run_dir=run_dir,
            output_dir=run_dir / "analysis",
            top_classes=args.top_classes,
            skip_plots=args.skip_plots,
            min_class_size=args.min_class_size,
            min_dropped_images=args.min_dropped_images,
        )
        comparison_rows.append(
            {
                "policy_name": analysis_summary["policy_name"],
                "variant": variant_name,
                "run_dir": str(run_dir),
                "total_images": analysis_summary["total_images"],
                "kept": analysis_summary["kept"],
                "dropped": analysis_summary["dropped"],
                "drop_rate": analysis_summary["drop_rate"],
                "flagged_over_pruned_classes": len(analysis_summary.get("flagged_over_pruned_classes", [])),
                "top_drop_reason": analysis_summary.get("top_drop_reasons", [{}])[0].get("reason", "")
                if analysis_summary.get("top_drop_reasons")
                else "",
            }
        )
        run_dirs.append(run_dir)

    comparison_df = pd.DataFrame(comparison_rows).sort_values(["drop_rate", "policy_name"], ascending=[False, True])
    comparison_path = output_root / "policy_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    combined_summary = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "source_partition": source_partition,
        "base_policy": args.policy,
        "coverage_check": coverage,
        "variants": list(args.variants),
        "run_dirs": [str(p) for p in run_dirs],
        "comparison_csv": str(comparison_path),
    }
    combined_summary_path = output_root / "policy_stress_test_summary.json"
    with open(combined_summary_path, "w") as f:
        json.dump(combined_summary, f, indent=2)

    print_header("Policy Stress Test Complete")
    print(f"Input dir: {input_dir}")
    print(f"Output root: {output_root}")
    print(f"Saved comparison: {comparison_path}")
    return 0


def compare_cleaning_runs(args: argparse.Namespace) -> int:
    run_dirs = [Path(p) for p in args.run_dirs]
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        prune_df, summary, policy = _load_cleaning_run(run_dir)
        df = _prepare_prune_analysis_df(prune_df)
        dropped = df[~df["keep"]]
        reason_counter: Counter[str] = Counter()
        for reasons in dropped["drop_reasons"]:
            reason_counter.update(set(reasons))
        top_reason = reason_counter.most_common(1)[0][0] if reason_counter else ""
        rows.append(
            {
                "run_dir": str(run_dir),
                "policy_name": summary.get("policy_name") or (policy or {}).get("policy_name", "unknown"),
                "total_images": int(len(df)),
                "kept": int(df["keep"].sum()),
                "dropped": int((~df["keep"]).sum()),
                "drop_rate": float((~df["keep"]).mean()) if len(df) else 0.0,
                "unique_drop_reasons": int(len(reason_counter)),
                "top_drop_reason": top_reason,
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values(["drop_rate", "policy_name"], ascending=[False, True])
    comparison_df.to_csv(output_path, index=False)

    print_header("Cleaning Policy Comparison Complete")
    print(f"Saved comparison: {output_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean ImageNet-1K from filtered export JSON files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    filt = subparsers.add_parser("from-filtered-json", help="Run clean-only pipeline from filtered JSON exports")
    filt.add_argument("--partition-mode", choices=["cluster", "class", "both"], required=True)
    filt.add_argument("--cluster-input-dir", default=None, help="Folder of cluster-group filtered JSON exports")
    filt.add_argument("--class-input-dir", default=None, help="Folder of class-group filtered JSON exports")
    filt.add_argument("--policy", required=True, help="Path to cleaning_policy.yaml")
    filt.add_argument("--output-root", default="data", help="Root output folder")
    filt.add_argument("--mode", choices=["metadata-only", "with-images"], default="metadata-only")
    filt.add_argument("--images-root", default=None, help="Root folder containing original images (required for with-images)")
    filt.add_argument("--glob", default="**/*.json", help="Glob pattern for filtered export files")
    filt.add_argument("--skip-invalid", action="store_true", help="Skip invalid JSON files instead of failing")
    filt.add_argument(
        "--expected-total-media-items",
        type=int,
        default=IMAGENET1K_EXPECTED_MEDIA_ITEMS,
        help=(
            "Expected full dataset size used for coverage reporting "
            f"(default: {IMAGENET1K_EXPECTED_MEDIA_ITEMS:,})."
        ),
    )
    filt.add_argument(
        "--require-full-coverage",
        action="store_true",
        help="Fail run unless merged unique count matches --expected-total-media-items",
    )

    analyze_cmd = subparsers.add_parser(
        "analyze-cleaning-run",
        help="Analyze a completed cleaning run from prune_decisions.jsonl + cleaning_summary.json",
    )
    analyze_cmd.add_argument("--run-dir", required=True, help="Path to clean_*_groups run directory")
    analyze_cmd.add_argument(
        "--output-dir",
        default=None,
        help="Directory for analysis outputs (default: <run-dir>/analysis)",
    )
    analyze_cmd.add_argument(
        "--top-classes",
        type=int,
        default=20,
        help="Number of top classes by drop fraction to include in reports and plots (default: 20)",
    )
    analyze_cmd.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip optional PNG plot generation",
    )
    analyze_cmd.add_argument(
        "--min-class-size",
        type=int,
        default=50,
        help="Minimum class size before over-pruning flags are evaluated (default: 50)",
    )
    analyze_cmd.add_argument(
        "--min-dropped-images",
        type=int,
        default=10,
        help="Minimum dropped images before over-pruning flags are evaluated (default: 10)",
    )

    compare_cmd = subparsers.add_parser(
        "compare-cleaning-runs",
        help="Compare multiple completed cleaning runs and write policy_comparison.csv",
    )
    compare_cmd.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One or more clean_*_groups run directories to compare",
    )
    compare_cmd.add_argument(
        "--output-file",
        default="data/policy_comparison.csv",
        help="CSV output path for run comparison (default: data/policy_comparison.csv)",
    )

    stress_cmd = subparsers.add_parser(
        "stress-test-policy-variants",
        help="Apply conservative/balanced/aggressive policy variants to the same filtered JSON input",
    )
    stress_cmd.add_argument("--input-dir", required=True, help="Folder of filtered JSON exports to evaluate")
    stress_cmd.add_argument("--policy", required=True, help="Base cleaning policy YAML used to derive variants")
    stress_cmd.add_argument("--output-root", required=True, help="Root folder for policy-sweep outputs")
    stress_cmd.add_argument("--glob", default="**/*.json", help="Glob pattern for filtered export files")
    stress_cmd.add_argument("--skip-invalid", action="store_true", help="Skip invalid JSON files instead of failing")
    stress_cmd.add_argument(
        "--expected-total-media-items",
        type=int,
        default=IMAGENET1K_EXPECTED_MEDIA_ITEMS,
        help=(
            "Expected full dataset size used for coverage reporting "
            f"(default: {IMAGENET1K_EXPECTED_MEDIA_ITEMS:,})."
        ),
    )
    stress_cmd.add_argument(
        "--require-full-coverage",
        action="store_true",
        help="Fail run unless merged unique count matches --expected-total-media-items",
    )
    stress_cmd.add_argument(
        "--variants",
        nargs="+",
        default=["conservative", "balanced", "aggressive"],
        help="Policy variants to derive from the base policy (default: conservative balanced aggressive)",
    )
    stress_cmd.add_argument(
        "--source-partition",
        default="policy_sweep",
        help="Name used in generated run directory labels (default: policy_sweep)",
    )
    stress_cmd.add_argument(
        "--top-classes",
        type=int,
        default=20,
        help="Number of top classes by drop fraction to include in reports and plots (default: 20)",
    )
    stress_cmd.add_argument(
        "--min-class-size",
        type=int,
        default=50,
        help="Minimum class size before over-pruning flags are evaluated (default: 50)",
    )
    stress_cmd.add_argument(
        "--min-dropped-images",
        type=int,
        default=10,
        help="Minimum dropped images before over-pruning flags are evaluated (default: 10)",
    )
    stress_cmd.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip optional PNG plot generation for each variant",
    )

    args = parser.parse_args()

    if args.command == "from-filtered-json":
        if args.partition_mode in ("cluster", "both") and not args.cluster_input_dir:
            parser.error("--cluster-input-dir is required for partition-mode cluster/both")
        if args.partition_mode in ("class", "both") and not args.class_input_dir:
            parser.error("--class-input-dir is required for partition-mode class/both")
        if args.mode == "with-images" and not args.images_root:
            parser.error("--images-root is required when --mode with-images")

    return args


def load_policy(policy_path: str) -> Dict[str, Any]:
    p = Path(policy_path)
    if not p.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    with open(p, "r") as f:
        policy = yaml.safe_load(f)
    return policy


def ingest_filtered_exports(
    input_dir: Path,
    glob_pattern: str,
    source_partition: str,
    skip_invalid: bool = False,
) -> pd.DataFrame:
    files = sorted(input_dir.glob(glob_pattern))
    files = [f for f in files if f.is_file()]
    if not files:
        raise FileNotFoundError(f"No JSON files found in {input_dir} with glob '{glob_pattern}'")

    all_rows: List[Dict[str, Any]] = []
    for path in tqdm(files, desc=f"Loading {source_partition} exports"):
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            if not isinstance(payload, dict) or "media_items" not in payload:
                raise ValueError("Expected top-level object with 'media_items'")
            media_items = payload.get("media_items", [])
            if not isinstance(media_items, list):
                raise ValueError("'media_items' must be a list")

            for row in media_items:
                if not isinstance(row, dict):
                    continue
                record = dict(row)
                record["__source_export_file"] = str(path)
                record["__source_partition"] = source_partition
                all_rows.append(record)
        except Exception as e:
            if skip_invalid:
                print(f"Skipping invalid file {path}: {e}")
                continue
            raise

    if not all_rows:
        raise RuntimeError(f"No media rows could be loaded from {input_dir}")

    return pd.DataFrame(all_rows)


def _parse_metadata_items(items: Any) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    if not isinstance(items, list):
        return [], [], []

    issues: List[Dict[str, Any]] = []
    tags: List[str] = []
    labels: List[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).strip().lower()
        props = item.get("properties", {})
        if not isinstance(props, dict):
            props = {}

        if item_type == "issue":
            issue_type = normalize_issue_type(props.get("issue_type") or item.get("issue_type") or "")
            conf_raw = props.get("confidence", item.get("confidence", 0.0))
            try:
                confidence = float(conf_raw)
            except Exception:
                confidence = 0.0
            issues.append({"issue_type": issue_type, "confidence": confidence})
        elif item_type == "user_tag":
            tag_name = props.get("tag_name") or item.get("tag_name")
            if tag_name:
                tags.append(str(tag_name))
        elif item_type == "image_label":
            category = props.get("category_name") or item.get("category_name")
            if category:
                labels.append(str(category))

    return issues, tags, labels


def normalize_records(df: pd.DataFrame) -> pd.DataFrame:
    if "metadata_items" not in df.columns:
        raise ValueError("metadata_items column not found in filtered exports")

    parsed = df["metadata_items"].apply(_parse_metadata_items)
    df = df.copy()
    df["issues"] = parsed.apply(lambda x: x[0])
    df["tags"] = parsed.apply(lambda x: x[1])
    df["labels"] = parsed.apply(lambda x: x[2])
    df["label"] = df["labels"].apply(lambda x: x[0] if x else "unknown")

    if "uniqueness_score" in df.columns:
        df["uniqueness_score"] = pd.to_numeric(df["uniqueness_score"], errors="coerce").fillna(0.0)
    else:
        df["uniqueness_score"] = 0.0

    return df


def merge_and_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    merged = df.copy()
    before = len(merged)
    if "media_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["media_id"], keep="first")
    elif "file_name" in merged.columns:
        merged = merged.drop_duplicates(subset=["file_name"], keep="first")
    print(f"Merged rows after dedupe: {before:,} -> {len(merged):,}")
    return merged


def validate_merged_coverage(
    merged_df: pd.DataFrame,
    expected_total_media_items: Optional[int],
    require_full_coverage: bool,
)-> Dict[str, Any]:
    merged_count = int(len(merged_df))
    if expected_total_media_items is None:
        return {
            "expected_total_media_items": None,
            "merged_total_media_items": merged_count,
            "difference_signed": None,
            "difference_absolute": None,
            "missing_from_expected": None,
            "excess_over_expected": None,
        }

    expected = int(expected_total_media_items)
    diff_signed = merged_count - expected
    diff_abs = abs(diff_signed)
    missing = max(0, expected - merged_count)
    excess = max(0, merged_count - expected)

    coverage = {
        "expected_total_media_items": expected,
        "merged_total_media_items": merged_count,
        "difference_signed": diff_signed,
        "difference_absolute": diff_abs,
        "missing_from_expected": missing,
        "excess_over_expected": excess,
    }

    if diff_signed == 0:
        print(
            "Coverage check passed: "
            f"merged={merged_count:,}, expected={expected:,}, difference=0"
        )
        return coverage

    msg = (
        "Coverage check mismatch: "
        f"merged={merged_count:,}, expected={expected:,}, "
        f"missing={missing:,}, excess={excess:,}, absolute_difference={diff_abs:,}. "
        "Your partition exports may be incomplete or overlapping."
    )
    if require_full_coverage:
        raise RuntimeError(msg)
    print(f"WARNING: {msg}")
    return coverage


def apply_cleaning_policy(df: pd.DataFrame, policy: Dict[str, Any]):
    drop_reasons: Dict[int, List[str]] = defaultdict(list)
    drop_mask = pd.Series([False] * len(df), index=df.index)

    threshold = policy.get("uniqueness_threshold")
    if threshold is not None:
        low_mask = df["uniqueness_score"] < float(threshold)
        drop_mask |= low_mask
        for idx in df[low_mask].index:
            drop_reasons[idx].append(f"low_uniqueness<{threshold}")

    if policy.get("dedupe_by_cluster", False) and "cluster_id" in df.columns:
        valid_cluster = df["cluster_id"].notna() & (df["cluster_id"].astype(str).str.strip() != "")
        for _, group in df[valid_cluster].groupby("cluster_id"):
            if len(group) <= 1:
                continue
            keep_idx = group["uniqueness_score"].idxmax()
            for idx in group.index:
                if idx == keep_idx:
                    continue
                drop_mask.loc[idx] = True
                drop_reasons[idx].append("duplicate_in_cluster")

    for issue_rule in policy.get("drop_issues", []):
        policy_issue = normalize_issue_type(issue_rule.get("issue_type", ""))
        min_conf = float(issue_rule.get("min_confidence", 1.0))
        if not policy_issue:
            continue

        issue_mask = df["issues"].apply(
            lambda issues: any(
                normalize_issue_type(i.get("issue_type", "")) == policy_issue
                and float(i.get("confidence", 0.0)) >= min_conf
                for i in issues
            )
        )
        drop_mask |= issue_mask
        for idx in df[issue_mask].index:
            drop_reasons[idx].append(f"issue_{policy_issue}")

    drop_tags = set(policy.get("drop_tags", []))
    if drop_tags:
        tag_mask = df["tags"].apply(lambda tags: any(t in drop_tags for t in tags))
        drop_mask |= tag_mask
        for idx in df[tag_mask].index:
            matches = [t for t in df.loc[idx, "tags"] if t in drop_tags]
            drop_reasons[idx].append(f"user_tag: {', '.join(matches)}")

    keep_df = df[~drop_mask].copy()
    drop_df = df[drop_mask].copy()
    drop_df["drop_reasons"] = drop_df.index.map(lambda i: " | ".join(drop_reasons.get(i, [])))

    return keep_df, drop_df, drop_reasons


def _drop_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in INTERNAL_COLUMNS]
    return df[cols].copy()


def copy_images(keep_df: pd.DataFrame, images_root: Path, out_dir: Path) -> Tuple[int, int]:
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    missing_paths: List[str] = []

    for _, row in tqdm(keep_df.iterrows(), total=len(keep_df), desc="Copying cleaned images"):
        file_path = row.get("file_path")
        file_name = row.get("file_name")

        candidates: List[Path] = []
        if isinstance(file_path, str) and file_path.strip():
            fp = Path(file_path)
            candidates.append(fp if fp.is_absolute() else images_root / fp)
        if isinstance(file_name, str) and file_name.strip():
            candidates.append(images_root / file_name)

        src = next((p for p in candidates if p.exists()), None)
        if src is None:
            missing += 1
            missing_paths.append(str(file_path or file_name))
            continue

        rel = Path(file_path) if isinstance(file_path, str) and file_path.strip() else Path(file_name)
        dst = images_out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    if missing_paths:
        with open(out_dir / "missing_images.txt", "w") as f:
            for p in missing_paths:
                f.write(f"{p}\n")

    return copied, missing


def save_outputs(
    merged_df: pd.DataFrame,
    keep_df: pd.DataFrame,
    drop_df: pd.DataFrame,
    drop_reasons: Dict[int, List[str]],
    policy: Dict[str, Any],
    out_dir: Path,
    run_mode: str,
    policy_path: str,
    images_root: Optional[Path],
):
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_merged_path = out_dir / "raw_merged_metadata.json"
    _drop_internal_columns(merged_df).to_json(raw_merged_path, orient="records", lines=True)

    cleaned_path = out_dir / "cleaned_imagenet1k.json"
    clean_df = _drop_internal_columns(keep_df)
    clean_df.to_json(cleaned_path, orient="records", lines=True)

    # compatibility alias
    clean_df.to_json(out_dir / "metadata.json", orient="records", lines=True)

    dropped_path = out_dir / "dropped_metadata.json"
    _drop_internal_columns(drop_df).to_json(dropped_path, orient="records", lines=True)

    keep_manifest = out_dir / "keep_filenames.txt"
    drop_manifest = out_dir / "drop_filenames.txt"
    if "file_name" in keep_df.columns:
        keep_df["file_name"].to_csv(keep_manifest, index=False, header=False)
    else:
        keep_manifest.write_text("")
    if "file_name" in drop_df.columns:
        drop_df["file_name"].to_csv(drop_manifest, index=False, header=False)
    else:
        drop_manifest.write_text("")

    now = datetime.now().isoformat()
    keep_idx = set(keep_df.index.tolist())
    prune_rows = []
    for idx, row in merged_df.iterrows():
        prune_rows.append(
            {
                "media_id": row.get("media_id"),
                "file_name": row.get("file_name"),
                "keep": idx in keep_idx,
                "drop_reasons": drop_reasons.get(idx, []),
                "uniqueness_score": row.get("uniqueness_score"),
                "cluster_id": row.get("cluster_id"),
                "label": row.get("label"),
                "issues": row.get("issues", []),
                "tags": row.get("tags", []),
                "policy_name": policy.get("policy_name", "unknown"),
                "run_mode": run_mode,
                "source_partition": row.get("__source_partition"),
                "source_export_file": row.get("__source_export_file"),
                "timestamp": now,
            }
        )

    prune_path = out_dir / "prune_decisions.jsonl"
    pd.DataFrame(prune_rows).to_json(prune_path, orient="records", lines=True)

    copied = 0
    missing = 0
    if run_mode == "with-images":
        copied, missing = copy_images(keep_df, images_root=images_root, out_dir=out_dir)

    summary = {
        "timestamp": now,
        "policy_name": policy.get("policy_name", "unknown"),
        "run_mode": run_mode,
        "total_images": int(len(merged_df)),
        "kept": int(len(keep_df)),
        "dropped": int(len(drop_df)),
        "drop_rate": float(len(drop_df) / len(merged_df)) if len(merged_df) else 0.0,
        "drop_by_reason": dict(Counter([r for reasons in drop_reasons.values() for r in reasons])),
        "images_copied": int(copied),
        "images_missing": int(missing),
        "artifacts": {
            "raw_merged_metadata_json": str(raw_merged_path),
            "cleaned_imagenet1k_json": str(cleaned_path),
            "dropped_metadata_json": str(dropped_path),
            "prune_decisions_jsonl": str(prune_path),
        },
    }
    with open(out_dir / "cleaning_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    shutil.copy2(policy_path, out_dir / "cleaning_policy.yaml")

    readme = out_dir / "README.md"
    readme.write_text(
        "# Clean ImageNet-1K (Phase 1)\n\n"
        "This folder contains clean-only outputs (no noise generation).\n\n"
        "Key files:\n"
        "- raw_merged_metadata.json\n"
        "- cleaned_imagenet1k.json\n"
        "- dropped_metadata.json\n"
        "- prune_decisions.jsonl\n"
        "- cleaning_summary.json\n"
    )


def run_partition(
    partition_name: str,
    input_dir: Path,
    policy: Dict[str, Any],
    policy_path: str,
    output_root: Path,
    run_mode: str,
    images_root: Optional[Path],
    glob_pattern: str,
    skip_invalid: bool,
    expected_total_media_items: Optional[int],
    require_full_coverage: bool,
) -> Dict[str, Any]:
    print_header(f"Run Partition: {partition_name}")
    print(f"Input dir: {input_dir}")

    raw_df = ingest_filtered_exports(
        input_dir=input_dir,
        glob_pattern=glob_pattern,
        source_partition=partition_name,
        skip_invalid=skip_invalid,
    )
    norm_df = normalize_records(raw_df)
    merged_df = merge_and_dedupe(norm_df)
    # Enforce requested workflow order:
    # 1) Merge all partition exports into raw_merged_metadata.json
    # 2) Validate merged coverage
    # 3) Apply cleaning policy to produce cleaned_imagenet1k.json
    coverage = validate_merged_coverage(
        merged_df=merged_df,
        expected_total_media_items=expected_total_media_items,
        require_full_coverage=require_full_coverage,
    )
    keep_df, drop_df, drop_reasons = apply_cleaning_policy(merged_df, policy)

    out_dir = output_root / f"clean_{partition_name}_groups"
    save_outputs(
        merged_df=merged_df,
        keep_df=keep_df,
        drop_df=drop_df,
        drop_reasons=drop_reasons,
        policy=policy,
        out_dir=out_dir,
        run_mode=run_mode,
        policy_path=policy_path,
        images_root=images_root,
    )

    result = {
        "partition": partition_name,
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "total": int(len(merged_df)),
        "kept": int(len(keep_df)),
        "dropped": int(len(drop_df)),
        "coverage_check": coverage,
        "raw_merged_metadata_json": str(out_dir / "raw_merged_metadata.json"),
        "cleaned_imagenet1k_json": str(out_dir / "cleaned_imagenet1k.json"),
    }
    expected_total = coverage.get("expected_total_media_items")
    if expected_total is not None:
        print(
            "Coverage summary: "
            f"expected={expected_total:,}, "
            f"merged={int(coverage['merged_total_media_items']):,}, "
            f"missing={int(coverage['missing_from_expected']):,}, "
            f"excess={int(coverage['excess_over_expected']):,}, "
            f"difference={int(coverage['difference_signed']):+,}"
        )
    print(f"Completed {partition_name}: kept={result['kept']:,}, dropped={result['dropped']:,}")
    return result


def run_from_filtered_json(args: argparse.Namespace) -> int:
    policy = load_policy(args.policy)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    images_root = Path(args.images_root) if args.images_root else None

    run_results: List[Dict[str, Any]] = []

    if args.partition_mode in ("cluster", "both"):
        run_results.append(
            run_partition(
                partition_name="cluster",
                input_dir=Path(args.cluster_input_dir),
                policy=policy,
                policy_path=args.policy,
                output_root=output_root,
                run_mode=args.mode,
                images_root=images_root,
                glob_pattern=args.glob,
                skip_invalid=args.skip_invalid,
                expected_total_media_items=args.expected_total_media_items,
                require_full_coverage=args.require_full_coverage,
            )
        )

    if args.partition_mode in ("class", "both"):
        run_results.append(
            run_partition(
                partition_name="class",
                input_dir=Path(args.class_input_dir),
                policy=policy,
                policy_path=args.policy,
                output_root=output_root,
                run_mode=args.mode,
                images_root=images_root,
                glob_pattern=args.glob,
                skip_invalid=args.skip_invalid,
                expected_total_media_items=args.expected_total_media_items,
                require_full_coverage=args.require_full_coverage,
            )
        )

    if args.partition_mode == "both":
        combined = {
            "timestamp": datetime.now().isoformat(),
            "run_mode": args.mode,
            "policy": args.policy,
            "results": run_results,
        }
        combined_path = output_root / "clean_combined_runs_summary.json"
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"Saved combined summary: {combined_path}")

    print_header("Phase 1 Complete")
    for rr in run_results:
        print(f"[{rr['partition']}] raw: {rr['raw_merged_metadata_json']}")
        print(f"[{rr['partition']}] clean: {rr['cleaned_imagenet1k_json']}")
    return 0


def main() -> int:
    args = parse_args()

    try:
        if args.command == "from-filtered-json":
            return run_from_filtered_json(args)
        if args.command == "analyze-cleaning-run":
            return analyze_cleaning_run(args)
        if args.command == "compare-cleaning-runs":
            return compare_cleaning_runs(args)
        if args.command == "stress-test-policy-variants":
            return stress_test_policy_variants(args)
        raise ValueError(f"Unsupported command: {args.command}")
    except KeyboardInterrupt:
        print("Interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
