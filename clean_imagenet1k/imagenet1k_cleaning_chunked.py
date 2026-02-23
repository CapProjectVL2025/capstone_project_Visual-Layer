#!/usr/bin/env python3
"""
Phase 1 clean-only pipeline for ImageNet-1K using filtered Visual Layer exports.

This script does not generate label noise. It merges filtered exports,
applies cleaning policy, and writes clean + traceability artifacts.
"""

import argparse
import json
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from connect_vl_api_f import VisualLayerAPIClient

load_dotenv()

INTERNAL_COLUMNS = {"__source_export_file", "__source_partition"}
IMAGENET1K_EXPECTED_MEDIA_ITEMS = 1_331_167


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def cleanup_python_cache(script_dir: Path):
    removed = 0
    for cache_dir in script_dir.rglob("__pycache__"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors=True)
            removed += 1
    for pyc_file in script_dir.rglob("*.pyc"):
        if pyc_file.is_file():
            pyc_file.unlink(missing_ok=True)
            removed += 1
    if removed > 0:
        print(f"\nCleaned Python cache artifacts: {removed}")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean ImageNet-1K from filtered export JSON files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry = subparsers.add_parser("dry-run-auth", help="Validate Visual Layer auth + dataset access")
    dry.add_argument("--dataset-id", required=True, help="Visual Layer dataset ID")

    export_cmd = subparsers.add_parser("export-from-vl", help="Step 1: export filtered partitions from Visual Layer")
    export_cmd.add_argument("--dataset-id", required=True, help="Visual Layer dataset ID")
    export_cmd.add_argument(
        "--partitions-config",
        default=None,
        help="Path to JSON config describing partition filters",
    )
    export_cmd.add_argument(
        "--cluster-ids-file",
        default=None,
        help="Optional text file of cluster IDs (one per line) to auto-build partition filters",
    )
    export_cmd.add_argument(
        "--entity-type",
        default="IMAGES",
        help="Entity type for export request (default: IMAGES)",
    )
    export_cmd.add_argument(
        "--threshold",
        default="1",
        help="Threshold parameter for export request (default: 1)",
    )
    export_cmd.add_argument("--export-dir", required=True, help="Output directory for downloaded partition exports")
    export_cmd.add_argument("--include-images", action="store_true", help="Include images in export zip files")
    export_cmd.add_argument("--file-prefix", default="partition", help="Filename prefix for export archives")
    export_cmd.add_argument(
        "--sub-partition-size",
        type=int,
        default=10000,
        help="Fallback chunk size when export is rejected for too many entities (default: 10000)",
    )
    export_cmd.add_argument(
        "--offset-param",
        default="offset",
        help="Query param name used for pagination offset in fallback mode (default: offset)",
    )
    export_cmd.add_argument(
        "--limit-param",
        default="limit",
        help="Query param name used for pagination limit in fallback mode (default: limit)",
    )
    export_cmd.add_argument(
        "--max-sub-partitions",
        type=int,
        default=10000,
        help="Safety limit for fallback sub-partitions per partition (default: 10000)",
    )

    discover_cmd = subparsers.add_parser(
        "discover-cluster-ids",
        help="Discover cluster IDs from Visual Layer via API and write cluster_ids.txt",
    )
    discover_cmd.add_argument("--dataset-id", required=True, help="Visual Layer dataset ID")
    discover_cmd.add_argument(
        "--output-file",
        default="clean_imagenet1k/cluster_ids.txt",
        help="Output text file (one cluster UUID per line)",
    )
    discover_cmd.add_argument(
        "--temp-dir",
        default="data/_cluster_discovery_tmp",
        help="Temp directory for discovery export artifacts",
    )
    discover_cmd.add_argument(
        "--entity-type",
        default="CLUSTERS",
        help="Entity type used for discovery export (default: CLUSTERS)",
    )
    discover_cmd.add_argument(
        "--threshold",
        default="1",
        help="Threshold parameter for discovery export (default: 1)",
    )
    discover_cmd.add_argument(
        "--page-size",
        type=int,
        default=10000,
        help="Page size for fallback IMAGES discovery paging (default: 10000)",
    )
    discover_cmd.add_argument(
        "--offset-param",
        default="offset",
        help="Query param name used for discovery paging offset (default: offset)",
    )
    discover_cmd.add_argument(
        "--limit-param",
        default="limit",
        help="Query param name used for discovery paging limit (default: limit)",
    )
    discover_cmd.add_argument(
        "--max-pages",
        type=int,
        default=10000,
        help="Safety cap for discovery paging requests (default: 10000)",
    )

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

    args = parser.parse_args()

    if args.command == "from-filtered-json":
        if args.partition_mode in ("cluster", "both") and not args.cluster_input_dir:
            parser.error("--cluster-input-dir is required for partition-mode cluster/both")
        if args.partition_mode in ("class", "both") and not args.class_input_dir:
            parser.error("--class-input-dir is required for partition-mode class/both")
        if args.mode == "with-images" and not args.images_root:
            parser.error("--images-root is required when --mode with-images")
    elif args.command == "export-from-vl":
        if not args.partitions_config and not args.cluster_ids_file:
            parser.error("Provide either --partitions-config or --cluster-ids-file")
        if args.partitions_config and args.cluster_ids_file:
            parser.error("Use only one of --partitions-config or --cluster-ids-file")

    return args


def load_policy(policy_path: str) -> Dict[str, Any]:
    p = Path(policy_path)
    if not p.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    with open(p, "r") as f:
        policy = yaml.safe_load(f)
    return policy


def load_partitions_config(config_path: str) -> List[Dict[str, Any]]:
    """
    Load partition config for Visual Layer filtered exports.

    Expected JSON format:
    {
      "partitions": [
        {"name": "class_group_01", "extra_params": {"...": "..."}},
        {"name": "class_group_02", "extra_params": {"...": "..."}}
      ]
    }
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Partitions config not found: {config_path}")

    with open(p, "r") as f:
        payload = json.load(f)

    parts = payload.get("partitions", [])
    if not isinstance(parts, list) or not parts:
        raise ValueError("Partitions config must contain non-empty 'partitions' list")

    normalized: List[Dict[str, Any]] = []
    for i, part in enumerate(parts):
        if not isinstance(part, dict):
            raise ValueError(f"Partition index {i} must be an object")
        name = str(part.get("name", f"partition_{i+1:03d}")).strip()
        extra_params = part.get("extra_params", {})
        if not isinstance(extra_params, dict):
            raise ValueError(f"Partition '{name}' must include object 'extra_params'")
        normalized.append({"name": name, "extra_params": extra_params})
    return normalized


def load_cluster_ids(cluster_ids_file: str) -> List[str]:
    p = Path(cluster_ids_file)
    if not p.exists():
        raise FileNotFoundError(f"Cluster IDs file not found: {cluster_ids_file}")
    lines = [ln.strip() for ln in p.read_text().splitlines()]
    ids = [ln for ln in lines if ln and not ln.startswith("#")]
    if not ids:
        raise ValueError(f"No cluster IDs found in {cluster_ids_file}")
    return ids


def build_partitions_from_cluster_ids(
    cluster_ids: List[str],
    entity_type: str,
    threshold: str,
) -> List[Dict[str, Any]]:
    partitions: List[Dict[str, Any]] = []
    for idx, cluster_id in enumerate(cluster_ids, start=1):
        partitions.append(
            {
                "name": f"cluster_{idx:05d}",
                "extra_params": {
                    "entity_type": entity_type,
                    "threshold": str(threshold),
                    "cluster_id": cluster_id,
                },
            }
        )
    return partitions


def _collect_cluster_ids_from_obj(obj: Any, ids: Set[str]) -> None:
    if isinstance(obj, dict):
        cluster_id = obj.get("cluster_id")
        if isinstance(cluster_id, str) and cluster_id.strip():
            ids.add(cluster_id.strip())
        for v in obj.values():
            _collect_cluster_ids_from_obj(v, ids)
        return
    if isinstance(obj, list):
        for item in obj:
            _collect_cluster_ids_from_obj(item, ids)


def _is_entities_exceeded_error(err: Exception) -> bool:
    text = str(err).lower()
    return "exceeds threshold" in text or ("entities" in text and "rejected" in text)


def _extract_export(zip_path: Path, extract_dir: Path) -> int:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    json_files = sorted(p for p in extract_dir.glob("**/*.json") if p.is_file())
    media_items_count = 0
    for path in json_files:
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                media_items = payload.get("media_items")
                if isinstance(media_items, list):
                    media_items_count += len(media_items)
        except Exception:
            continue
    return media_items_count


def discover_cluster_ids_from_vl(
    dataset_id: str,
    output_file: str,
    temp_dir: str,
    entity_type: str = "CLUSTERS",
    threshold: str = "1",
    page_size: int = 10000,
    offset_param: str = "offset",
    limit_param: str = "limit",
    max_pages: int = 10000,
) -> int:
    print_header("Discover Cluster IDs From Visual Layer")
    out_path = Path(output_file)
    tmp_root = Path(temp_dir)
    tmp_root.mkdir(parents=True, exist_ok=True)

    client = VisualLayerAPIClient()
    if not client.test_connection(dataset_id):
        return 1

    def run_discovery_export(
        extra_params: Dict[str, Any],
        export_suffix: str,
    ) -> int:
        file_name = f"cluster_discovery_{export_suffix}.zip"
        task_id = client.export_dataset(
            dataset_id=dataset_id,
            format="json",
            include_images=False,
            file_name=file_name,
            extra_params=extra_params,
        )
        download_url = client.wait_for_export(dataset_id, task_id)

        run_dir = tmp_root / export_suffix
        run_dir.mkdir(parents=True, exist_ok=True)
        zip_path = run_dir / file_name
        client.download_export(download_url, str(zip_path))

        extract_dir = run_dir / "extracted"
        media_count = _extract_export(zip_path=zip_path, extract_dir=extract_dir)

        json_files = sorted(p for p in extract_dir.glob("**/*.json") if p.is_file())
        if not json_files:
            return 0
        for json_path in json_files:
            with open(json_path, "r") as f:
                payload = json.load(f)
            _collect_cluster_ids_from_obj(payload, cluster_ids)
        return media_count

    cluster_ids: Set[str] = set()
    base_params = {"entity_type": entity_type, "threshold": str(threshold)}
    try:
        run_discovery_export(
            extra_params=base_params,
            export_suffix=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    except Exception as e:
        # Some environments reject CLUSTERS with 500; fallback to paged IMAGES.
        if str(entity_type).upper() != "CLUSTERS":
            raise
        print(
            "Discovery with entity_type=CLUSTERS failed; "
            "falling back to paged entity_type=IMAGES discovery..."
        )
        offset = 0
        pages = 0
        while pages < max_pages:
            pages += 1
            current_limit = int(page_size)
            while True:
                suffix = f"images_page_{pages:05d}"
                params = {
                    "entity_type": "IMAGES",
                    "threshold": str(threshold),
                    offset_param: offset,
                    limit_param: current_limit,
                }
                print(
                    f"  Discovery page {pages}: {offset_param}={offset}, "
                    f"{limit_param}={current_limit}"
                )
                try:
                    media_count = run_discovery_export(extra_params=params, export_suffix=suffix)
                    break
                except Exception as page_err:
                    if _is_entities_exceeded_error(page_err) and current_limit > 1:
                        current_limit = max(1, current_limit // 2)
                        print(
                            f"  Page too large; retrying with {limit_param}={current_limit}"
                        )
                        continue
                    raise

            if media_count <= 0:
                print("  Reached empty page; discovery complete.")
                break

            offset += media_count
            if media_count < current_limit:
                print("  Final partial page reached; discovery complete.")
                break
        else:
            raise RuntimeError(
                f"Discovery exceeded max pages ({max_pages}). Increase --max-pages."
            )

    if not cluster_ids:
        raise RuntimeError(
            "No cluster_id values found in discovery export. "
            "Try --entity-type IMAGES and confirm the export contains cluster_id fields."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sorted(cluster_ids)) + "\n")
    print(f"Discovered cluster IDs: {len(cluster_ids):,}")
    print(f"Saved cluster IDs file: {out_path}")
    return 0


def export_partitions_from_vl(
    dataset_id: str,
    partitions_config: Optional[str],
    export_dir: str,
    include_images: bool = False,
    file_prefix: str = "partition",
    cluster_ids_file: Optional[str] = None,
    entity_type: str = "IMAGES",
    threshold: str = "1",
    sub_partition_size: int = 10000,
    offset_param: str = "offset",
    limit_param: str = "limit",
    max_sub_partitions: int = 10000,
) -> int:
    print_header("Step 1: Export Filtered Partitions From Visual Layer")
    if cluster_ids_file:
        cluster_ids = load_cluster_ids(cluster_ids_file)
        partitions = build_partitions_from_cluster_ids(
            cluster_ids=cluster_ids,
            entity_type=entity_type,
            threshold=threshold,
        )
    elif partitions_config:
        partitions = load_partitions_config(partitions_config)
    else:
        raise ValueError("Either partitions_config or cluster_ids_file must be provided")
    out_root = Path(export_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    client = VisualLayerAPIClient()
    if not client.test_connection(dataset_id):
        return 1

    results: List[Dict[str, Any]] = []

    def run_single_export(
        part_name: str,
        params: Dict[str, Any],
        seq_idx: int,
        suffix: Optional[str] = None,
    ) -> Dict[str, Any]:
        suffix_part = f"_{suffix}" if suffix else ""
        file_name = f"{file_prefix}_{seq_idx:03d}_{part_name}{suffix_part}.zip"
        task_id = client.export_dataset(
            dataset_id=dataset_id,
            format="json",
            include_images=include_images,
            file_name=file_name,
            extra_params=params,
        )
        download_url = client.wait_for_export(dataset_id, task_id)

        export_folder = part_name if not suffix else f"{part_name}/{suffix}"
        part_dir = out_root / export_folder
        part_dir.mkdir(parents=True, exist_ok=True)

        zip_path = part_dir / file_name
        client.download_export(download_url, str(zip_path))

        extract_dir = part_dir / "extracted"
        media_items_count = _extract_export(zip_path=zip_path, extract_dir=extract_dir)
        json_files = list(extract_dir.glob("**/*.json"))

        return {
            "name": part_name,
            "suffix": suffix,
            "zip_path": str(zip_path),
            "extract_dir": str(extract_dir),
            "json_files_found": len(json_files),
            "media_items_count": media_items_count,
            "extra_params": params,
        }

    for idx, part in enumerate(partitions, start=1):
        name = part["name"]
        params = part["extra_params"]

        print(f"\nExporting partition {idx}/{len(partitions)}: {name}")
        try:
            res = run_single_export(part_name=name, params=params, seq_idx=idx)
            results.append(res)
            print(f"  Extracted JSON files: {res['json_files_found']}")
            continue
        except Exception as e:
            if not _is_entities_exceeded_error(e):
                raise
            print(
                "  Export exceeded entity threshold; switching to automatic "
                f"sub-partitioning with {sub_partition_size:,} items/chunk..."
            )

        # Fallback: offset/limit paging for oversized partitions.
        offset = 0
        sub_idx = 0
        while sub_idx < max_sub_partitions:
            sub_idx += 1
            current_limit = int(sub_partition_size)
            while True:
                chunk_suffix = f"chunk_{sub_idx:05d}"
                chunk_params = {**params, offset_param: offset, limit_param: current_limit}
                print(
                    f"    Chunk {sub_idx}: {offset_param}={offset}, "
                    f"{limit_param}={current_limit}"
                )
                try:
                    chunk_res = run_single_export(
                        part_name=name,
                        params=chunk_params,
                        seq_idx=idx,
                        suffix=chunk_suffix,
                    )
                    break
                except Exception as chunk_err:
                    if _is_entities_exceeded_error(chunk_err) and current_limit > 1:
                        current_limit = max(1, current_limit // 2)
                        print(f"    Chunk still too large; retrying with {limit_param}={current_limit}")
                        continue
                    raise

            chunk_count = int(chunk_res.get("media_items_count", 0))
            chunk_res["chunk_index"] = sub_idx
            chunk_res["chunk_offset"] = offset
            chunk_res["chunk_limit"] = current_limit
            results.append(chunk_res)
            print(f"    Chunk exported rows: {chunk_count:,}")

            if chunk_count <= 0:
                print("    Reached empty chunk; partition export complete.")
                break

            offset += chunk_count
            if chunk_count < current_limit:
                print("    Final partial chunk reached; partition export complete.")
                break
        else:
            raise RuntimeError(
                f"Exceeded max sub-partitions ({max_sub_partitions}) for partition '{name}'. "
                "Increase --max-sub-partitions or review partition filters."
            )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset_id": dataset_id,
        "include_images": include_images,
        "partitions_config": partitions_config,
        "cluster_ids_file": cluster_ids_file,
        "entity_type": entity_type,
        "threshold": str(threshold),
        "sub_partition_size": int(sub_partition_size),
        "offset_param": offset_param,
        "limit_param": limit_param,
        "max_sub_partitions": int(max_sub_partitions),
        "export_dir": str(out_root),
        "results": results,
    }
    summary_path = out_root / "export_partitions_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved export summary: {summary_path}")
    print("Export partitions complete.")
    return 0


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


def run_dry_run_auth(dataset_id: str) -> int:
    print_header("Dry Run Auth")
    client = VisualLayerAPIClient()
    ok = client.test_connection(dataset_id)
    if ok:
        print("Dry-run auth passed")
        return 0
    print("Dry-run auth failed")
    return 1


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
    script_dir = Path(__file__).resolve().parent

    try:
        if args.command == "dry-run-auth":
            return run_dry_run_auth(args.dataset_id)
        if args.command == "export-from-vl":
            return export_partitions_from_vl(
                dataset_id=args.dataset_id,
                partitions_config=args.partitions_config,
                export_dir=args.export_dir,
                include_images=args.include_images,
                file_prefix=args.file_prefix,
                cluster_ids_file=args.cluster_ids_file,
                entity_type=args.entity_type,
                threshold=args.threshold,
                sub_partition_size=args.sub_partition_size,
                offset_param=args.offset_param,
                limit_param=args.limit_param,
                max_sub_partitions=args.max_sub_partitions,
            )
        if args.command == "discover-cluster-ids":
            return discover_cluster_ids_from_vl(
                dataset_id=args.dataset_id,
                output_file=args.output_file,
                temp_dir=args.temp_dir,
                entity_type=args.entity_type,
                threshold=args.threshold,
                page_size=args.page_size,
                offset_param=args.offset_param,
                limit_param=args.limit_param,
                max_pages=args.max_pages,
            )
        if args.command == "from-filtered-json":
            return run_from_filtered_json(args)
        raise ValueError(f"Unsupported command: {args.command}")
    except KeyboardInterrupt:
        print("Interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        cleanup_python_cache(script_dir)


if __name__ == "__main__":
    sys.exit(main())
