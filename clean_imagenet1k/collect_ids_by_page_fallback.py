#!/usr/bin/env python3
"""
Collect IDs by Visual Layer dataset page with fallback semantics:
- If page has cluster IDs -> write cluster IDs for that page.
- Else -> write image IDs for that page.

Outputs:
- Per-page ID files: ids_page_00001.txt, ids_page_00002.txt, ...
- Merged deduplicated ID file
- Summary JSON
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from dotenv import load_dotenv

load_dotenv()

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
CLUSTER_PATH_RE = re.compile(
    r"/cluster/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:[/?#]|$)",
    re.IGNORECASE,
)
CLUSTER_QUERY_RE = re.compile(
    r"(?:[?&](?:cluster_id|clusterId)=)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:[&#]|$)",
    re.IGNORECASE,
)
IMAGE_PATH_RE = re.compile(
    r"/data/image/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:[/?#]|$)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect page IDs with cluster->image fallback.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, required=True)
    parser.add_argument("--stop-after-empty-pages", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--headful", action="store_true")
    parser.add_argument("--base-url", default=os.environ.get("VL_BASE_URL", "https://app.visual-layer.com"))
    parser.add_argument("--storage-state", default=os.environ.get("VL_STORAGE_STATE", ".vl_storage_state.json"))
    parser.add_argument("--out-dir", default="clean_imagenet1k/by_page_ids")
    parser.add_argument("--merged-out", default="clean_imagenet1k/ids_merged.txt")
    parser.add_argument("--overwrite-existing-pages", action="store_true")
    return parser.parse_args()


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(lines)
    path.write_text("\n".join(materialized) + ("\n" if materialized else ""), encoding="utf-8")


def format_labeled_lines(cluster_ids: Set[str], image_ids: Set[str]) -> List[str]:
    lines: List[str] = []
    lines.extend(f"cluster:{value}" for value in sorted(cluster_ids))
    lines.extend(f"image:{value}" for value in sorted(image_ids))
    return lines


def load_labeled_ids(path: Path) -> Tuple[Set[str], Set[str]]:
    cluster_ids: Set[str] = set()
    image_ids: Set[str] = set()
    if not path.exists():
        return cluster_ids, image_ids
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip().lower()
        if not value:
            continue
        if value.startswith("cluster:"):
            candidate = value.split(":", 1)[1].strip()
            if candidate and UUID_RE.match(candidate):
                cluster_ids.add(candidate)
            continue
        if value.startswith("image:"):
            candidate = value.split(":", 1)[1].strip()
            if candidate and UUID_RE.match(candidate):
                image_ids.add(candidate)
            continue
        # Backward compatibility: old files had plain UUIDs.
        if UUID_RE.match(value):
            cluster_ids.add(value)
    return cluster_ids, image_ids


def extract_from_text(text: str) -> Tuple[Set[str], Set[str]]:
    clusters: Set[str] = set()
    images: Set[str] = set()
    if not text:
        return clusters, images
    raw = str(text)
    clusters.update(v.lower() for v in CLUSTER_PATH_RE.findall(raw))
    clusters.update(v.lower() for v in CLUSTER_QUERY_RE.findall(raw))
    images.update(v.lower() for v in IMAGE_PATH_RE.findall(raw))
    return clusters, images


def walk_payload_for_ids(obj: Any) -> Tuple[Set[str], Set[str]]:
    clusters: Set[str] = set()
    images: Set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("cluster_id", "clusterId", "cluster_uuid", "clusterUuid"):
                value = node.get(key)
                if isinstance(value, str) and UUID_RE.match(value):
                    clusters.add(value.lower())
            for key in ("media_id", "mediaId", "image_id", "imageId"):
                value = node.get(key)
                if isinstance(value, str) and UUID_RE.match(value):
                    images.add(value.lower())
            for key in ("url", "uri", "href", "link"):
                value = node.get(key)
                if isinstance(value, str):
                    c_ids, i_ids = extract_from_text(value)
                    clusters.update(c_ids)
                    images.update(i_ids)
            for child in node.values():
                walk(child)
            return
        if isinstance(node, list):
            for child in node:
                walk(child)

    walk(obj)
    return clusters, images


def main() -> int:
    args = parse_args()
    if args.start_page < 1:
        args.start_page = 1
    if args.end_page < args.start_page:
        print("--end-page must be >= --start-page", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    merged_out = Path(args.merged_out)
    storage_state = Path(args.storage_state)
    base_url = str(args.base_url).rstrip("/")
    dataset_id_lc = str(args.dataset_id).strip().lower()

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_out.parent.mkdir(parents=True, exist_ok=True)

    all_cluster_ids: Set[str] = set()
    all_image_ids: Set[str] = set()
    page_summaries: List[Dict[str, Any]] = []
    empty_streak = 0
    last_processed_page: Optional[int] = None

    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as exc:
        print(f"Playwright unavailable: {exc}", file=sys.stderr)
        return 3

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headful)
        context_kwargs: Dict[str, Any] = {}
        if storage_state.exists():
            context_kwargs["storage_state"] = str(storage_state)
        else:
            print(f"Warning: storage state not found: {storage_state}", file=sys.stderr)
        context = browser.new_context(**context_kwargs)
        page = context.new_page()

        network_clusters: Set[str] = set()
        network_images: Set[str] = set()

        def on_response(response: Any) -> None:
            try:
                url_l = str(response.url).lower()
                if "/api/" not in url_l and "graphql" not in url_l:
                    return
                headers = response.headers if isinstance(response.headers, dict) else {}
                content_type = str(headers.get("content-type", "")).lower()
                if "json" not in content_type and "graphql" not in url_l:
                    return
                payload = response.json()
            except Exception:
                return
            c_ids, i_ids = walk_payload_for_ids(payload)
            network_clusters.update(c_ids)
            network_images.update(i_ids)

        page.on("response", on_response)

        for page_num in range(args.start_page, args.end_page + 1):
            page_file = out_dir / f"ids_page_{page_num:05d}.txt"
            if page_file.exists() and not args.overwrite_existing_pages:
                existing_clusters, existing_images = load_labeled_ids(page_file)
                all_cluster_ids.update(existing_clusters)
                all_image_ids.update(existing_images)
                write_lines(merged_out, format_labeled_lines(all_cluster_ids, all_image_ids))
                page_summaries.append(
                    {
                        "page": page_num,
                        "status": "existing",
                        "cluster_ids_count": len(existing_clusters),
                        "image_ids_count": len(existing_images),
                        "ids_count": len(existing_clusters) + len(existing_images),
                        "file": str(page_file),
                        "timestamp": now_utc_iso(),
                    }
                )
                print(
                    (
                        f"[page {page_num}] reused existing ids "
                        f"(clusters={len(existing_clusters)} images={len(existing_images)})"
                    ),
                    file=sys.stderr,
                )
                last_processed_page = page_num
                continue

            network_clusters = set()
            network_images = set()
            url = f"{base_url}/dataset/{args.dataset_id}/data?page={page_num}"
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=args.timeout_ms)
            except Exception as exc:
                print(f"[page {page_num}] navigation failed: {exc}", file=sys.stderr)
                break
            try:
                page.wait_for_load_state("networkidle", timeout=min(args.timeout_ms, 8000))
            except Exception:
                pass

            page_clusters: Set[str] = set(network_clusters)
            page_images: Set[str] = set(network_images)

            c_ids, i_ids = extract_from_text(page.url)
            page_clusters.update(c_ids)
            page_images.update(i_ids)

            try:
                hrefs = page.eval_on_selector_all("a[href]", "els => els.map(el => el.href)")
            except Exception:
                hrefs = []
            if isinstance(hrefs, list):
                for href in hrefs:
                    if not isinstance(href, str):
                        continue
                    c_ids, i_ids = extract_from_text(href)
                    page_clusters.update(c_ids)
                    page_images.update(i_ids)

            page_clusters.discard(dataset_id_lc)
            page_images.discard(dataset_id_lc)

            # Requested behavior: if no clusters on page, fallback to image IDs.
            selected = page_clusters if page_clusters else page_images
            selected_kind = "cluster" if page_clusters else "image"

            if selected_kind == "cluster":
                selected_clusters = set(selected)
                selected_images: Set[str] = set()
            else:
                selected_clusters = set()
                selected_images = set(selected)

            write_lines(page_file, format_labeled_lines(selected_clusters, selected_images))
            all_cluster_ids.update(selected_clusters)
            all_image_ids.update(selected_images)
            write_lines(merged_out, format_labeled_lines(all_cluster_ids, all_image_ids))

            ids_count = len(selected)
            if ids_count == 0:
                empty_streak += 1
            else:
                empty_streak = 0

            page_summaries.append(
                {
                    "page": page_num,
                    "status": "collected",
                    "kind_used": selected_kind,
                    "cluster_ids_found": len(page_clusters),
                    "image_ids_found": len(page_images),
                    "ids_written": ids_count,
                    "file": str(page_file),
                    "url": page.url,
                    "timestamp": now_utc_iso(),
                }
            )
            print(
                (
                    f"[page {page_num}] kind={selected_kind} wrote={ids_count} "
                    f"merged_total={len(all_cluster_ids) + len(all_image_ids)}"
                ),
                file=sys.stderr,
            )
            last_processed_page = page_num

            if args.stop_after_empty_pages > 0 and empty_streak >= args.stop_after_empty_pages:
                print(
                    f"Stopping after {empty_streak} consecutive empty pages at page {page_num}",
                    file=sys.stderr,
                )
                break

        context.close()
        browser.close()

    summary = {
        "dataset_id": args.dataset_id,
        "start_page": args.start_page,
        "end_page_requested": args.end_page,
        "last_processed_page": last_processed_page,
        "merged_cluster_ids_total": len(all_cluster_ids),
        "merged_image_ids_total": len(all_image_ids),
        "merged_ids_total": len(all_cluster_ids) + len(all_image_ids),
        "merged_ids_file": str(merged_out),
        "pages": page_summaries,
        "timestamp": now_utc_iso(),
    }
    summary_path = out_dir / "ids_by_page_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote merged IDs: {merged_out}")
    print(f"Wrote summary:    {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
