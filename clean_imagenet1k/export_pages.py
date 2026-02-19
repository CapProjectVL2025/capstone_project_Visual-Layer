#!/usr/bin/env python3
"""
Reproducible Visual Layer cluster discovery + per-cluster metadata export.

Features:
- Deterministic `cluster_ids.txt` discovery from paged exploration metadata
- Strict retry behavior for 429/5xx/network and invalid-JSON 200 bodies
- Per-cluster raw + normalized metadata exports
- Oversized-cluster fallback with offset/limit sub-partitioning
- Checkpoint + failure manifests for recoverable reruns
- Built-in auto-resume loop for incomplete runs
"""

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import jwt
import requests
from dotenv import load_dotenv

load_dotenv()

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
UUID_IN_TEXT_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)
CLUSTER_PATH_UUID_RE = re.compile(
    r"/cluster/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:[/?#]|$)",
    re.IGNORECASE,
)
CLUSTER_QUERY_UUID_RE = re.compile(
    r"(?:[?&](?:cluster_id|clusterId)=)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:[&#]|$)",
    re.IGNORECASE,
)
TRANSIENT_HTTP = {429, 500, 502, 503, 504}
TERMINAL_EXPORT_FAILURES = {"FAILED", "REJECTED", "CANCELED", "CANCELLED"}
TOO_LARGE_HINTS = ("exceeds threshold", "entities", "rejected")


class RequestError(RuntimeError):
    def __init__(self, message: str, kind: str, status: int = 0, snippet: str = ""):
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.snippet = snippet


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_backoff_seconds(
    attempt: int,
    base_seconds: float,
    max_seconds: float,
    min_seconds: float = 0.0,
) -> float:
    wait = min(max_seconds, base_seconds * (2 ** max(0, attempt - 1)))
    jitter = random.uniform(0.0, min(1.0, wait / 4.0 if wait > 0 else 0.0))
    return max(min_seconds, wait + jitter)


def sleep_with_backoff(
    attempt: int,
    base_seconds: float,
    max_seconds: float,
    min_seconds: float = 0.0,
) -> None:
    time.sleep(compute_backoff_seconds(attempt, base_seconds, max_seconds, min_seconds=min_seconds))


def normalize_issue_type(name: str) -> str:
    raw = str(name or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "outlier": "visual_outlier",
        "outliers": "visual_outlier",
        "mislabels": "mislabel",
        "mislabeled": "mislabel",
        "label_outliers": "label_outlier",
    }
    return aliases.get(raw, raw)


def is_quality_issue(issue_type: str) -> bool:
    key = str(issue_type or "").lower()
    quality_tokens = (
        "blur",
        "dark",
        "exposed",
        "quality",
        "resolution",
        "noise",
        "artifact",
        "compression",
    )
    return any(token in key for token in quality_tokens)


class JWTAuthProvider:
    def __init__(self, api_key: str, api_secret: str, token_expiry_minutes: int = 60):
        self.api_key = api_key
        self.api_secret = api_secret
        self.token_expiry_minutes = token_expiry_minutes
        self.token: Optional[str] = None
        self.refresh_at: Optional[datetime] = None

    @classmethod
    def from_env(cls) -> "JWTAuthProvider":
        api_key = os.environ.get("VL_API_KEY") or os.environ.get("api_key")
        api_secret = os.environ.get("VL_API_SECRET") or os.environ.get("api_secret")
        if not api_key or not api_secret:
            raise RuntimeError(
                "Missing API credentials. Set VL_API_KEY/VL_API_SECRET "
                "or api_key/api_secret in your environment or .env file."
            )
        return cls(api_key=api_key, api_secret=api_secret)

    def _generate_token(self) -> None:
        now = datetime.now(timezone.utc)
        expiration = now + timedelta(minutes=self.token_expiry_minutes)
        payload = {
            "sub": self.api_key,
            "iat": int(now.timestamp()),
            "exp": int(expiration.timestamp()),
            "iss": "sdk",
        }
        self.token = jwt.encode(
            payload=payload,
            key=self.api_secret,
            algorithm="HS256",
            headers={"kid": self.api_key, "typ": "JWT"},
        )
        self.refresh_at = expiration - timedelta(minutes=5)

    def headers(self, force_refresh: bool = False) -> Dict[str, str]:
        now = datetime.now(timezone.utc)
        if force_refresh or not self.token or not self.refresh_at or now >= self.refresh_at:
            self._generate_token()
        return {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}


def vl_base_url() -> str:
    return os.environ.get("VL_BASE_URL", "https://app.visual-layer.com").rstrip("/")


def resolve_dataset_id(dataset_id_arg: Optional[str]) -> str:
    dataset_id = (
        dataset_id_arg
        or os.environ.get("VL_DATASET_ID")
        or os.environ.get("dataset_id")
        or os.environ.get("DATASET_ID")
    )
    if not dataset_id:
        raise RuntimeError(
            "Missing dataset ID. Provide --dataset-id or set VL_DATASET_ID/dataset_id in .env."
        )
    return str(dataset_id).strip()


def _looks_like_media_item(obj: Dict[str, Any]) -> bool:
    media_keys = ("media_id", "file_name", "file_path", "media_type", "metadata_items")
    return any(k in obj for k in media_keys)


def _cluster_context_present(obj: Dict[str, Any]) -> bool:
    type_val = str(obj.get("type", "")).lower()
    entity_type = str(obj.get("entity_type", "")).upper()
    kind_val = str(obj.get("kind", "")).lower()
    if "cluster" in type_val or entity_type == "CLUSTERS" or "cluster" in kind_val:
        return True
    return any("cluster" in str(k).lower() for k in obj.keys())


def walk_for_cluster_ids(obj: Any) -> Iterable[str]:
    """
    Recursively discover cluster UUIDs across possible payload shapes.
    Handles:
    - explicit keys like cluster_id / cluster_uuid
    - cluster objects where UUID is under id/uuid
    - cluster URLs that embed UUID values
    """
    if isinstance(obj, dict):
        # Common explicit cluster-ID fields.
        for field in ("cluster_id", "clusterId", "cluster_uuid", "clusterUuid"):
            value = obj.get(field)
            if isinstance(value, str) and UUID_RE.match(value):
                yield value

        has_cluster_context = _cluster_context_present(obj)
        is_media_like = _looks_like_media_item(obj)

        # Some exploration payloads expose cluster UUID under generic id/uuid keys.
        if has_cluster_context and not is_media_like:
            for field in ("id", "uuid"):
                value = obj.get(field)
                if isinstance(value, str) and UUID_RE.match(value):
                    yield value

        # Some payloads embed cluster UUID in cluster URLs.
        for field in ("url", "uri", "href", "link"):
            value = obj.get(field)
            if isinstance(value, str) and "cluster" in value.lower():
                cluster_uuid = extract_uuid_from_text(value)
                if cluster_uuid:
                    yield cluster_uuid

        for value in obj.values():
            yield from walk_for_cluster_ids(value)
        return

    if isinstance(obj, list):
        for item in obj:
            yield from walk_for_cluster_ids(item)


def extract_uuid_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    raw = str(text)

    # Prefer explicit cluster path segment, e.g. /data/cluster/<uuid>?...
    match = CLUSTER_PATH_UUID_RE.search(raw)
    if match:
        value = match.group(1).lower()
        return value if UUID_RE.match(value) else None

    # Then prefer explicit cluster_id query parameter.
    match = CLUSTER_QUERY_UUID_RE.search(raw)
    if match:
        value = match.group(1).lower()
        return value if UUID_RE.match(value) else None

    # Fallback: choose the last UUID found in text.
    # Visual Layer URLs often include dataset UUID first and cluster UUID later.
    all_matches = UUID_IN_TEXT_RE.findall(raw)
    if not all_matches:
        return None
    for candidate in reversed(all_matches):
        value = candidate.lower()
        if UUID_RE.match(value):
            return value
    return None


def extract_cluster_uuid_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    raw = str(text)
    match = CLUSTER_PATH_UUID_RE.search(raw)
    if match:
        value = match.group(1).lower()
        return value if UUID_RE.match(value) else None
    match = CLUSTER_QUERY_UUID_RE.search(raw)
    if match:
        value = match.group(1).lower()
        return value if UUID_RE.match(value) else None
    return None


def extract_uuid_excluding(text: str, blocked_uuid: Optional[str]) -> Optional[str]:
    if not text:
        return None
    blocked = (blocked_uuid or "").strip().lower()
    for candidate in reversed(UUID_IN_TEXT_RE.findall(str(text))):
        value = candidate.lower()
        if UUID_RE.match(value) and (not blocked or value != blocked):
            return value
    return None


def has_cluster_ids_file_values(path: Path) -> bool:
    if not path.exists():
        return False
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        if UUID_RE.match(value):
            return True
    return False


def get_query_int(url: str, key: str, default: int) -> int:
    try:
        query = dict(parse_qsl(urlparse(url).query, keep_blank_values=True))
        return int(query.get(key, default))
    except Exception:
        return default


def set_query_value(url: str, key: str, value: Any) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query[str(key)] = str(value)
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def discover_cluster_ids_via_ui(
    dataset_id: str,
    out_path: Path,
    flag_out: Optional[Path],
    checkpoint_path: Path,
    failures_path: Path,
    ui_list_url: Optional[str],
    ui_cluster_selector: str,
    ui_next_selector: str,
    ui_max_pages: int,
    ui_max_clusters_per_page: int,
    ui_timeout_ms: int,
    ui_headful: bool,
    ui_storage_state: Optional[str],
    ui_save_storage_state: Optional[str],
    ui_manual_login: bool,
    ui_force_click: bool,
) -> int:
    failures: List[Dict[str, Any]] = []
    cluster_ids: Set[str] = set()
    pages_scanned = 0
    list_url = ui_list_url or f"{vl_base_url()}/dataset/{dataset_id}/data"
    start_page_number = max(1, get_query_int(list_url, "page", 1))
    current_list_url = set_query_value(list_url, "page", start_page_number)
    dataset_id_lc = str(dataset_id).strip().lower()

    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as exc:
        failure = {
            "kind": "ui_dependency_missing",
            "status": 0,
            "snippet": "",
            "message": f"Playwright unavailable: {exc}",
            "timestamp": now_utc_iso(),
        }
        failures.append(failure)
        json_dump(
            failures_path,
            {
                "dataset_id": dataset_id,
                "completed": False,
                "phase": "discover-cluster-ids-ui",
                "failures": failures,
            },
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        json_dump(
            out_path.parent / "discover_summary.json",
            {
                "dataset_id": dataset_id,
                "cluster_ids_file": str(out_path),
                "cluster_count": 0,
                "pages_successful": 0,
                "entity_type_used": "UI",
                "completed": False,
                "failures": failures,
                "timestamp": now_utc_iso(),
            },
        )
        return 5

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=not ui_headful)
            context_kwargs: Dict[str, Any] = {}
            if ui_storage_state:
                storage_path = Path(ui_storage_state)
                if storage_path.exists():
                    context_kwargs["storage_state"] = str(storage_path)
                else:
                    failures.append(
                        {
                            "kind": "ui_storage_state_missing",
                            "status": 0,
                            "snippet": "",
                            "message": f"Storage state file not found: {storage_path}",
                            "timestamp": now_utc_iso(),
                        }
                    )
            context = browser.new_context(**context_kwargs)
            page = context.new_page()
            page.goto(current_list_url, wait_until="domcontentloaded", timeout=ui_timeout_ms)
            page.wait_for_timeout(1000)

            if ui_manual_login:
                print(
                    "Complete login in the opened browser window, then press Enter here to continue...",
                    file=sys.stderr,
                )
                input()
                # Respect the page the user left open only if it is the expected dataset list.
                expected_prefix = f"/dataset/{dataset_id}/data"
                if expected_prefix in page.url:
                    current_list_url = page.url
                else:
                    current_list_url = set_query_value(list_url, "page", start_page_number)
                    page.goto(current_list_url, wait_until="domcontentloaded", timeout=ui_timeout_ms)
                    page.wait_for_timeout(800)

            if ui_save_storage_state:
                context.storage_state(path=ui_save_storage_state)

            selector = ui_cluster_selector.strip() if ui_cluster_selector else "a[href*='/cluster/']"
            fallback_selectors = [
                "a[href*='/cluster/']",
                ":text-matches(\"^[0-9,]+\\s+Images$\", \"i\")",
                "text=/\\d+\\s+Images/i",
            ]
            next_selector = ui_next_selector.strip()
            page_index = 1

            while page_index <= ui_max_pages:
                pages_scanned += 1
                page.wait_for_timeout(400)
                dataset_page_number = start_page_number + (page_index - 1)

                deadline = time.time() + max(1, int(ui_timeout_ms)) / 1000.0
                active_selector = selector
                items = page.locator(active_selector)
                count = 0
                selector_attempts: List[Tuple[str, int]] = []
                while True:
                    selector_attempts = []
                    chosen_selector = None
                    chosen_items = None
                    chosen_count = 0

                    seen_candidates: Set[str] = set()
                    for candidate in [active_selector, *fallback_selectors]:
                        if candidate in seen_candidates:
                            continue
                        seen_candidates.add(candidate)
                        try:
                            candidate_items = page.locator(candidate)
                            candidate_count = candidate_items.count()
                        except Exception:
                            candidate_items = page.locator("xpath=//*")
                            candidate_count = 0
                        selector_attempts.append((candidate, candidate_count))
                        if candidate_count > 0 and chosen_selector is None:
                            chosen_selector = candidate
                            chosen_items = candidate_items
                            chosen_count = candidate_count

                    if chosen_selector is not None:
                        active_selector = chosen_selector
                        items = chosen_items  # type: ignore[assignment]
                        count = chosen_count
                        break

                    if time.time() >= deadline:
                        break
                    page.wait_for_timeout(500)

                if count == 0:
                    debug_html = out_path.parent / "ui_debug_last_page.html"
                    debug_png = out_path.parent / "ui_debug_last_page.png"
                    try:
                        debug_html.parent.mkdir(parents=True, exist_ok=True)
                        debug_html.write_text(page.content(), encoding="utf-8")
                    except Exception:
                        pass
                    try:
                        page.screenshot(path=str(debug_png), full_page=True)
                    except Exception:
                        pass
                    attempts_msg = ", ".join(f"'{sel}' -> {cnt}" for sel, cnt in selector_attempts)
                    failures.append(
                        {
                            "kind": "ui_no_cluster_items",
                            "status": 0,
                            "snippet": "",
                            "message": (
                                f"No cluster UI items found on page {page_index}. "
                                f"Selector attempts: {attempts_msg}. "
                                f"Debug files: {debug_html}, {debug_png}"
                            ),
                            "timestamp": now_utc_iso(),
                        }
                    )
                    break

                limit = min(count, ui_max_clusters_per_page)
                if count > ui_max_clusters_per_page:
                    failures.append(
                        {
                            "kind": "ui_cluster_page_truncated",
                            "status": 0,
                            "snippet": "",
                            "message": (
                                f"Page {dataset_page_number} has {count} cluster cards but "
                                f"ui_max_clusters_per_page={ui_max_clusters_per_page}. "
                                "Increase --ui-max-clusters-per-page to avoid partial discovery."
                            ),
                            "timestamp": now_utc_iso(),
                        }
                    )
                print(
                    f"[ui page {dataset_page_number}] visiting {limit}/{count} cluster cards",
                    file=sys.stderr,
                )
                item_idx = 0
                new_ids_on_page = 0
                visited_card_signatures: Set[str] = set()
                while item_idx < limit:
                    items = page.locator(active_selector)
                    current_count = items.count()
                    if item_idx >= current_count:
                        break
                    item = items.nth(item_idx)
                    click_target = item

                    href_value = item.get_attribute("href") or ""
                    if not href_value:
                        # Cluster cards are often wrappers with a nested <a>; inspect that first
                        # so we can extract cluster_id without navigating.
                        try:
                            nested_link = item.locator("a[href*='/cluster/']").first
                            if nested_link.count() > 0:
                                href_value = nested_link.get_attribute("href") or ""
                        except Exception:
                            href_value = href_value or ""
                    href_cluster_id = (
                        extract_cluster_uuid_from_text(href_value)
                        or extract_uuid_excluding(href_value, dataset_id_lc)
                    )
                    if href_cluster_id and not ui_force_click:
                        before = len(cluster_ids)
                        cluster_ids.add(href_cluster_id)
                        if len(cluster_ids) > before:
                            new_ids_on_page += 1
                        item_idx += 1
                        continue

                    # Skip non-cluster links to avoid permission errors on unrelated entities.
                    if href_value and "/cluster/" not in href_value:
                        item_idx += 1
                        continue

                    # If this is already a cluster link but UUID was not parseable, skip by default
                    # instead of clicking into potentially inaccessible entities.
                    if href_value and "/cluster/" in href_value and not ui_force_click and not href_cluster_id:
                        failures.append(
                            {
                                "kind": "ui_cluster_href_missing_uuid",
                                "status": 0,
                                "snippet": "",
                                "message": (
                                    f"Cluster link without parseable UUID on page {page_index}, "
                                    f"index {item_idx}: '{href_value}'"
                                ),
                                "timestamp": now_utc_iso(),
                            }
                        )
                        item_idx += 1
                        continue

                    # For text-based selectors, resolve a local cluster-card box and click
                    # coordinates so each text match maps to its own card.
                    click_box = None
                    if "text=" in active_selector or ":text" in active_selector:
                        try:
                            resolved = item.evaluate(
                                """
                                (el) => {
                                  function toBox(r) {
                                    return {x: r.x, y: r.y, width: r.width, height: r.height};
                                  }
                                  function labelCount(text) {
                                    const m = (text || "").match(/\\b\\d[\\d,]*\\s+Images\\b/gi);
                                    return m ? m.length : 0;
                                  }
                                  let best = null;
                                  let cur = el;
                                  while (cur && cur !== document.body) {
                                    const r = cur.getBoundingClientRect();
                                    const labels = labelCount(cur.innerText || "");
                                    const imgCount = cur.querySelectorAll("img").length;
                                    const isCardLike =
                                      r.width >= 140 &&
                                      r.width <= 700 &&
                                      r.height >= 120 &&
                                      r.height <= 700 &&
                                      labels >= 1 &&
                                      labels <= 2 &&
                                      imgCount >= 4 &&
                                      imgCount <= 40;
                                    if (isCardLike) {
                                      const area = r.width * r.height;
                                      if (!best || area < best.area) {
                                        best = {x: r.x, y: r.y, width: r.width, height: r.height, area};
                                      }
                                    }
                                    cur = cur.parentElement;
                                  }
                                  if (best) {
                                    return {x: best.x, y: best.y, width: best.width, height: best.height};
                                  }
                                  const r = el.getBoundingClientRect();
                                  return toBox(r);
                                }
                                """
                            )
                            if isinstance(resolved, dict):
                                click_box = {
                                    "x": float(resolved.get("x", 0.0)),
                                    "y": float(resolved.get("y", 0.0)),
                                    "width": float(resolved.get("width", 0.0)),
                                    "height": float(resolved.get("height", 0.0)),
                                }
                        except Exception:
                            click_box = None

                    card_signature = ""
                    try:
                        click_target.scroll_into_view_if_needed(timeout=ui_timeout_ms)
                    except Exception:
                        pass
                    if not click_box:
                        try:
                            raw_box = click_target.bounding_box()
                            if raw_box:
                                click_box = {
                                    "x": float(raw_box.get("x", 0.0)),
                                    "y": float(raw_box.get("y", 0.0)),
                                    "width": float(raw_box.get("width", 0.0)),
                                    "height": float(raw_box.get("height", 0.0)),
                                }
                        except Exception:
                            click_box = None
                    if click_box:
                        card_signature = (
                            f"{int(round(click_box['x']))}:"
                            f"{int(round(click_box['y']))}:"
                            f"{int(round(click_box['width']))}:"
                            f"{int(round(click_box['height']))}"
                        )
                    if card_signature:
                        if card_signature in visited_card_signatures:
                            item_idx += 1
                            continue
                        visited_card_signatures.add(card_signature)

                    # Guardrail: never click outside the main dataset grid region.
                    if click_box and (click_box["x"] < 40 or click_box["y"] < 140):
                        failures.append(
                            {
                                "kind": "ui_click_box_outside_grid",
                                "status": 0,
                                "snippet": "",
                                "message": (
                                    f"Skipping suspicious click target on page {page_index}, index {item_idx}: "
                                    f"{click_box}"
                                ),
                                "timestamp": now_utc_iso(),
                            }
                        )
                        item_idx += 1
                        continue

                    before_url = page.url
                    try:
                        if click_box and click_box["width"] > 0 and click_box["height"] > 0:
                            click_x = click_box["x"] + max(8.0, min(click_box["width"] / 2.0, click_box["width"] - 8.0))
                            click_y = click_box["y"] + max(8.0, min(click_box["height"] * 0.2, click_box["height"] - 8.0))
                            page.mouse.click(click_x, click_y)
                        else:
                            click_target.click(timeout=ui_timeout_ms)
                        try:
                            page.wait_for_url("**/cluster/**", timeout=min(5000, ui_timeout_ms))
                        except Exception:
                            page.wait_for_timeout(500)
                    except Exception as exc:
                        failures.append(
                            {
                                "kind": "ui_click_failed",
                                "status": 0,
                                "snippet": "",
                                "message": f"Click failed on page {page_index}, index {item_idx}: {exc}",
                                "timestamp": now_utc_iso(),
                            }
                        )
                        item_idx += 1
                        continue

                    if page.url == before_url and click_box:
                        # Retry by clicking the center of the visual card.
                        try:
                            page.mouse.click(
                                click_box["x"] + (click_box["width"] / 2.0),
                                click_box["y"] + min(click_box["height"] * 0.7, click_box["height"] - 4),
                            )
                            try:
                                page.wait_for_url("**/cluster/**", timeout=min(4000, ui_timeout_ms))
                            except Exception:
                                page.wait_for_timeout(400)
                        except Exception:
                            pass

                    cluster_id = (
                        extract_cluster_uuid_from_text(page.url)
                        or extract_uuid_excluding(page.url, dataset_id_lc)
                        or href_cluster_id
                    )
                    if not cluster_id and page.url != before_url:
                        try:
                            hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
                            if isinstance(hrefs, list):
                                for href in hrefs:
                                    cluster_id = (
                                        extract_cluster_uuid_from_text(str(href))
                                        or extract_uuid_excluding(str(href), dataset_id_lc)
                                    )
                                    if cluster_id:
                                        break
                        except Exception:
                            pass

                    if cluster_id:
                        before = len(cluster_ids)
                        cluster_ids.add(cluster_id)
                        if len(cluster_ids) > before:
                            new_ids_on_page += 1
                    else:
                        failures.append(
                            {
                                "kind": "ui_cluster_id_missing_after_click",
                                "status": 0,
                                "snippet": "",
                                "message": (
                                    f"Could not extract cluster_id after click on page {page_index}, "
                                    f"index {item_idx}, url '{page.url}'"
                                ),
                                "timestamp": now_utc_iso(),
                            }
                        )

                    if page.url != before_url:
                        try:
                            # Deterministic return: navigate directly to the list URL we were on
                            # before opening the cluster details.
                            page.goto(before_url, wait_until="domcontentloaded", timeout=ui_timeout_ms)
                            page.wait_for_timeout(350)
                        except Exception as exc:
                            failures.append(
                                {
                                    "kind": "ui_return_to_list_failed",
                                    "status": 0,
                                    "snippet": "",
                                    "message": f"Failed to return to list page after click: {exc}",
                                    "timestamp": now_utc_iso(),
                                }
                            )
                            break
                    else:
                        # Some UIs open side panels instead of full navigation.
                        try:
                            page.keyboard.press("Escape")
                            page.wait_for_timeout(250)
                        except Exception:
                            pass
                    item_idx += 1

                json_dump(
                    checkpoint_path,
                    {
                        "dataset_id": dataset_id,
                        "discovery_mode": "ui",
                        "pages_scanned": pages_scanned,
                        "current_ui_page": page_index,
                        "current_dataset_page": dataset_page_number,
                        "cluster_ids_discovered": len(cluster_ids),
                        "new_ids_on_page": new_ids_on_page,
                        "timestamp": now_utc_iso(),
                    },
                )
                print(
                    f"[ui page {dataset_page_number}] discovered {new_ids_on_page} new cluster_ids "
                    f"(total {len(cluster_ids)})",
                    file=sys.stderr,
                )

                if not next_selector:
                    if page_index >= ui_max_pages:
                        break
                    next_dataset_page = start_page_number + page_index
                    next_list_url = set_query_value(current_list_url, "page", next_dataset_page)
                    try:
                        page.goto(next_list_url, wait_until="domcontentloaded", timeout=ui_timeout_ms)
                        page.wait_for_timeout(900)
                        current_list_url = next_list_url
                        page_index += 1
                        continue
                    except Exception as exc:
                        failures.append(
                            {
                                "kind": "ui_next_page_failed",
                                "status": 0,
                                "snippet": "",
                                "message": f"Failed to navigate to next page URL '{next_list_url}': {exc}",
                                "timestamp": now_utc_iso(),
                            }
                        )
                        break

                next_button = page.locator(next_selector).first
                if next_button.count() == 0:
                    break
                disabled = next_button.get_attribute("disabled")
                aria_disabled = next_button.get_attribute("aria-disabled")
                class_attr = (next_button.get_attribute("class") or "").lower()
                if disabled is not None or aria_disabled == "true" or "disabled" in class_attr:
                    break

                try:
                    next_button.click(timeout=ui_timeout_ms)
                    page.wait_for_timeout(900)
                    current_list_url = page.url
                except Exception as exc:
                    failures.append(
                        {
                            "kind": "ui_next_page_failed",
                            "status": 0,
                            "snippet": "",
                            "message": f"Failed to navigate to next page: {exc}",
                            "timestamp": now_utc_iso(),
                        }
                    )
                    break
                page_index += 1

            context.close()
            browser.close()
    except Exception as exc:
        failures.append(
            {
                "kind": "ui_automation_fatal",
                "status": 0,
                "snippet": "",
                "message": str(exc),
                "timestamp": now_utc_iso(),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_ids = sorted(cluster_ids)
    out_path.write_text("\n".join(sorted_ids) + ("\n" if sorted_ids else ""), encoding="utf-8")

    if not failures and sorted_ids and flag_out:
        flag_out.parent.mkdir(parents=True, exist_ok=True)
        flag_out.write_text(
            "READY=1\n"
            f"CLUSTERS={len(sorted_ids)}\n"
            f"DATASET_ID={dataset_id}\n",
            encoding="utf-8",
        )

    json_dump(
        out_path.parent / "discover_summary.json",
        {
            "dataset_id": dataset_id,
            "cluster_ids_file": str(out_path),
            "cluster_count": len(sorted_ids),
            "pages_successful": pages_scanned,
            "entity_type_used": "UI",
            "completed": len(failures) == 0 and len(sorted_ids) > 0,
            "failures": failures,
            "timestamp": now_utc_iso(),
        },
    )

    if failures or not sorted_ids:
        if not failures and not sorted_ids:
            failures.append(
                {
                    "kind": "no_cluster_ids",
                    "status": 0,
                    "snippet": "",
                    "message": "No cluster IDs discovered via UI automation.",
                    "timestamp": now_utc_iso(),
                }
            )
        json_dump(
            failures_path,
            {
                "dataset_id": dataset_id,
                "completed": False,
                "phase": "discover-cluster-ids-ui",
                "failures": failures,
            },
        )
        print("UI discovery incomplete. See discover_summary/failures manifest.", file=sys.stderr)
        return 4

    print(f"Wrote {len(sorted_ids)} unique cluster_ids to: {out_path}")
    if flag_out:
        print(f"Wrote cleaning flag to: {flag_out}")
    return 0


def request_json_with_retry(
    session: requests.Session,
    auth: JWTAuthProvider,
    method: str,
    url: str,
    timeout_s: int,
    max_retries: int,
    retry_base_s: float,
    retry_max_s: float,
    unit_desc: str,
    params: Optional[Dict[str, Any]] = None,
    allow_json_retry: bool = True,
    cooldown_on_429_s: float = 0.0,
) -> Any:
    forced_auth_refresh = False
    for attempt in range(1, max_retries + 1):
        try:
            response = session.request(
                method=method,
                url=url,
                headers=auth.headers(),
                params=params,
                timeout=timeout_s,
            )
        except requests.exceptions.RequestException as exc:
            if attempt < max_retries:
                print(
                    f"{unit_desc} network error ({exc}); retry {attempt}/{max_retries}",
                    file=sys.stderr,
                )
                sleep_with_backoff(attempt, retry_base_s, retry_max_s)
                continue
            raise RequestError(str(exc), kind="network")

        status = response.status_code
        text = response.text

        if status in (401, 403) and not forced_auth_refresh:
            forced_auth_refresh = True
            auth.headers(force_refresh=True)
            if attempt < max_retries:
                print(f"{unit_desc} auth status {status}; refreshing JWT and retrying...", file=sys.stderr)
                sleep_with_backoff(attempt, retry_base_s, retry_max_s)
                continue

        if status in TRANSIENT_HTTP:
            if attempt < max_retries:
                print(f"{unit_desc} transient HTTP {status}; retry {attempt}/{max_retries}", file=sys.stderr)
                min_wait = cooldown_on_429_s if status == 429 else 0.0
                sleep_with_backoff(attempt, retry_base_s, retry_max_s, min_seconds=min_wait)
                continue
            snippet = text[:300].replace("\n", " ")
            raise RequestError(
                f"{unit_desc} transient HTTP exhausted: {status}",
                kind="http",
                status=status,
                snippet=snippet,
            )

        if status != 200:
            snippet = text[:300].replace("\n", " ")
            kind = "auth" if status in (401, 403) else "http"
            raise RequestError(
                f"{unit_desc} HTTP {status}",
                kind=kind,
                status=status,
                snippet=snippet,
            )

        try:
            return response.json()
        except Exception:
            if allow_json_retry and attempt < max_retries:
                print(f"{unit_desc} invalid JSON body; retry {attempt}/{max_retries}", file=sys.stderr)
                sleep_with_backoff(attempt, retry_base_s, retry_max_s)
                continue
            snippet = text[:300].replace("\n", " ")
            raise RequestError(
                f"{unit_desc} invalid JSON response",
                kind="json",
                status=status,
                snippet=snippet,
            )

    raise RequestError(f"{unit_desc} request failed unexpectedly", kind="unknown")


def download_file_with_retry(
    url: str,
    output_path: Path,
    timeout_s: int,
    max_retries: int,
    retry_base_s: float,
    retry_max_s: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=timeout_s)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return
        except requests.exceptions.RequestException as exc:
            if attempt < max_retries:
                print(f"[download] error ({exc}); retry {attempt}/{max_retries}", file=sys.stderr)
                sleep_with_backoff(attempt, retry_base_s, retry_max_s)
                continue
            raise RequestError(f"Download failed: {exc}", kind="download")


def build_explore_url(dataset_id: str, threshold: int, entity_type: str, page_number: int) -> str:
    base = vl_base_url()
    return (
        f"{base}/api/v1/explore/{dataset_id}/exploration_metadata"
        f"?threshold={threshold}&entity_type={entity_type}&page_number={page_number}"
    )


def build_export_start_url(dataset_id: str) -> str:
    return f"{vl_base_url()}/api/v1/dataset/{dataset_id}/export_context_async"


def build_export_status_url(dataset_id: str) -> str:
    return f"{vl_base_url()}/api/v1/dataset/{dataset_id}/export_status"


def is_entities_exceeded_error(text: str) -> bool:
    lower = str(text).lower()
    return "exceeds threshold" in lower or ("entities" in lower and "rejected" in lower)


def extract_export_zip(zip_path: Path, extract_dir: Path) -> List[Path]:
    if extract_dir.exists():
        shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    return sorted(path for path in extract_dir.glob("**/*.json") if path.is_file())


def parse_metadata_items(items: Any) -> Tuple[List[Dict[str, Any]], List[str], List[str], List[str], float]:
    if not isinstance(items, list):
        return [], [], [], [], 0.0
    issues: List[Dict[str, Any]] = []
    tags: List[str] = []
    labels: List[str] = []
    quality_issue_types: Set[str] = set()
    quality_max_conf = 0.0
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).strip().lower()
        props = item.get("properties", {})
        if not isinstance(props, dict):
            props = {}

        if item_type == "issue":
            issue_type = normalize_issue_type(props.get("issue_type") or item.get("issue_type") or "")
            confidence = parse_float(props.get("confidence", item.get("confidence", 0.0)), 0.0)
            issues.append({"issue_type": issue_type, "confidence": confidence})
            if is_quality_issue(issue_type):
                quality_issue_types.add(issue_type)
                if confidence > quality_max_conf:
                    quality_max_conf = confidence
        elif item_type == "user_tag":
            tag_name = props.get("tag_name") or item.get("tag_name")
            if tag_name:
                tags.append(str(tag_name))
        elif item_type == "image_label":
            category_name = props.get("category_name") or item.get("category_name")
            if category_name:
                labels.append(str(category_name))

    return issues, tags, labels, sorted(quality_issue_types), quality_max_conf


def normalize_record(
    row: Dict[str, Any],
    run_id: str,
    dataset_id: str,
    cluster_id: str,
    chunk_id: str,
    source_raw_path: str,
) -> Dict[str, Any]:
    issues, tags, labels, quality_issue_types, quality_max_conf = parse_metadata_items(row.get("metadata_items"))
    quality_fields = {k: v for k, v in row.items() if "quality" in str(k).lower() and k != "metadata_items"}
    return {
        "media_id": row.get("media_id"),
        "file_name": row.get("file_name"),
        "file_path": row.get("file_path"),
        "cluster_id": row.get("cluster_id", cluster_id),
        "uniqueness_score": parse_float(row.get("uniqueness_score"), 0.0),
        "width": parse_int(row.get("width"), 0),
        "height": parse_int(row.get("height"), 0),
        "file_size": row.get("file_size"),
        "metadata_items": row.get("metadata_items", []),
        "label": labels[0] if labels else None,
        "issues": issues,
        "tags": tags,
        "quality_issue_types": quality_issue_types,
        "quality_issue_max_confidence": quality_max_conf,
        "quality_score": row.get("quality_score"),
        "quality_fields": quality_fields if quality_fields else None,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "chunk_id": chunk_id,
        "source_raw_path": source_raw_path,
        "exported_at": now_utc_iso(),
    }


def append_normalized_records(
    raw_json_paths: Sequence[Path],
    jsonl_path: Path,
    run_id: str,
    dataset_id: str,
    cluster_id: str,
    chunk_id: str,
) -> int:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(jsonl_path, "a", encoding="utf-8") as out:
        for raw_path in raw_json_paths:
            with open(raw_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            media_items = payload.get("media_items", []) if isinstance(payload, dict) else []
            if not isinstance(media_items, list):
                continue
            for row in media_items:
                if not isinstance(row, dict):
                    continue
                out.write(
                    json.dumps(
                        normalize_record(
                            row=row,
                            run_id=run_id,
                            dataset_id=dataset_id,
                            cluster_id=cluster_id,
                            chunk_id=chunk_id,
                            source_raw_path=str(raw_path),
                        )
                    )
                    + "\n"
                )
                written += 1
    return written


def start_export_task(
    session: requests.Session,
    auth: JWTAuthProvider,
    dataset_id: str,
    file_name: str,
    extra_params: Dict[str, Any],
    timeout_s: int,
    max_retries: int,
    retry_base_s: float,
    retry_max_s: float,
    cooldown_on_429_s: float,
) -> str:
    payload = request_json_with_retry(
        session=session,
        auth=auth,
        method="GET",
        url=build_export_start_url(dataset_id),
        timeout_s=timeout_s,
        max_retries=max_retries,
        retry_base_s=retry_base_s,
        retry_max_s=retry_max_s,
        cooldown_on_429_s=cooldown_on_429_s,
        unit_desc=f"[export start {file_name}]",
        params={
            "file_name": file_name,
            "export_format": "json",
            "include_images": "false",
            **extra_params,
        },
        allow_json_retry=True,
    )
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"Export rejected: {payload.get('error')}")
    task_id = payload.get("id") if isinstance(payload, dict) else None
    if not task_id:
        raise RuntimeError(f"Missing export task ID in response: {payload}")
    return str(task_id)


def wait_for_export_task(
    session: requests.Session,
    auth: JWTAuthProvider,
    dataset_id: str,
    task_id: str,
    timeout_s: int,
    max_retries: int,
    retry_base_s: float,
    retry_max_s: float,
    cooldown_on_429_s: float,
    poll_interval_s: int,
    max_wait_s: int,
) -> str:
    start_time = time.time()
    while time.time() - start_time <= max_wait_s:
        payload = request_json_with_retry(
            session=session,
            auth=auth,
            method="GET",
            url=build_export_status_url(dataset_id),
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_base_s=retry_base_s,
            retry_max_s=retry_max_s,
            cooldown_on_429_s=cooldown_on_429_s,
            unit_desc=f"[export status {task_id}]",
            params={"export_task_id": task_id},
            allow_json_retry=True,
        )
        status = str(payload.get("status", "PENDING")).upper()
        if status == "COMPLETED":
            fields = ("download_uri", "download_url", "downloadUri", "url", "uri", "file_url", "downloadUrl")
            for field in fields:
                if payload.get(field):
                    return str(payload[field])
            result = payload.get("result")
            if isinstance(result, dict):
                for field in fields:
                    if result.get(field):
                        return str(result[field])
            raise RuntimeError(f"Export completed without download URL: {payload}")
        if status in TERMINAL_EXPORT_FAILURES:
            message = payload.get("result_message") or payload.get("error") or payload.get("message") or "unknown"
            raise RuntimeError(f"Export failed: {message}")
        time.sleep(poll_interval_s)
    raise RuntimeError(f"Export polling timed out for task {task_id}")


def run_single_export(
    session: requests.Session,
    auth: JWTAuthProvider,
    dataset_id: str,
    cluster_id: str,
    suffix: str,
    export_params: Dict[str, Any],
    raw_dir: Path,
    normalized_jsonl: Path,
    run_id: str,
    timeout_s: int,
    max_retries: int,
    retry_base_s: float,
    retry_max_s: float,
    cooldown_on_429_s: float,
    poll_interval_s: int,
    max_wait_s: int,
) -> Dict[str, Any]:
    file_name = f"cluster_{cluster_id}_{suffix}.zip"
    task_id = start_export_task(
        session=session,
        auth=auth,
        dataset_id=dataset_id,
        file_name=file_name,
        extra_params=export_params,
        timeout_s=timeout_s,
        max_retries=max_retries,
        retry_base_s=retry_base_s,
        retry_max_s=retry_max_s,
        cooldown_on_429_s=cooldown_on_429_s,
    )
    download_url = wait_for_export_task(
        session=session,
        auth=auth,
        dataset_id=dataset_id,
        task_id=task_id,
        timeout_s=timeout_s,
        max_retries=max_retries,
        retry_base_s=retry_base_s,
        retry_max_s=retry_max_s,
        cooldown_on_429_s=cooldown_on_429_s,
        poll_interval_s=poll_interval_s,
        max_wait_s=max_wait_s,
    )

    zip_path = raw_dir / "exports" / file_name
    download_file_with_retry(
        url=download_url,
        output_path=zip_path,
        timeout_s=max(120, timeout_s),
        max_retries=max_retries,
        retry_base_s=retry_base_s,
        retry_max_s=retry_max_s,
    )

    extracted_dir = raw_dir / f"extracted_{suffix}"
    extracted_json_paths = extract_export_zip(zip_path=zip_path, extract_dir=extracted_dir)
    if not extracted_json_paths:
        raise RuntimeError(f"No JSON files found in export archive {zip_path}")

    raw_json_paths: List[Path] = []
    media_items_count = 0
    for idx, extracted_json in enumerate(extracted_json_paths, start=1):
        raw_target = raw_dir / f"metadata_{suffix}_{idx:05d}.json"
        shutil.copy2(extracted_json, raw_target)
        raw_json_paths.append(raw_target)
        with open(raw_target, "r", encoding="utf-8") as f:
            payload = json.load(f)
        items = payload.get("media_items", []) if isinstance(payload, dict) else []
        if isinstance(items, list):
            media_items_count += len(items)

    if suffix == "full" and raw_json_paths:
        shutil.copy2(raw_json_paths[0], raw_dir / "metadata.json")

    normalized_rows = append_normalized_records(
        raw_json_paths=raw_json_paths,
        jsonl_path=normalized_jsonl,
        run_id=run_id,
        dataset_id=dataset_id,
        cluster_id=cluster_id,
        chunk_id=suffix,
    )

    return {
        "suffix": suffix,
        "task_id": task_id,
        "download_url": download_url,
        "zip_path": str(zip_path),
        "media_items_count": media_items_count,
        "normalized_rows_written": normalized_rows,
        "raw_json_files": [str(path) for path in raw_json_paths],
    }


def write_cluster_chunk_files(cluster_ids: Sequence[str], chunk_size: int, chunks_dir: Path) -> List[Dict[str, Any]]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_meta: List[Dict[str, Any]] = []
    for idx in range(0, len(cluster_ids), chunk_size):
        chunk_index = (idx // chunk_size) + 1
        chunk_path = chunks_dir / f"chunk_{chunk_index:05d}.txt"
        chunk_values = list(cluster_ids[idx : idx + chunk_size])
        chunk_path.write_text("\n".join(chunk_values) + "\n", encoding="utf-8")
        chunk_meta.append(
            {
                "index": chunk_index,
                "path": str(chunk_path),
                "count": len(chunk_values),
                "sha256": sha256_file(chunk_path),
            }
        )
    return chunk_meta


def load_cluster_ids(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Cluster IDs file not found: {path}")
    values: Set[str] = set()
    invalid: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        if not UUID_RE.match(value):
            invalid.append(value)
            continue
        values.add(value)
    if invalid:
        raise ValueError(
            "Invalid cluster_id values found in file: "
            + ", ".join(invalid[:5])
            + ("..." if len(invalid) > 5 else "")
        )
    if not values:
        raise ValueError(f"No cluster IDs found in {path}")
    return sorted(values)


def run_export_for_cluster(
    session: requests.Session,
    auth: JWTAuthProvider,
    dataset_id: str,
    cluster_id: str,
    run_id: str,
    cluster_dir: Path,
    entity_type: str,
    threshold: str,
    sub_partition_size: int,
    offset_param: str,
    limit_param: str,
    max_sub_partitions: int,
    timeout_s: int,
    max_retries: int,
    retry_base_s: float,
    retry_max_s: float,
    cooldown_on_429_s: float,
    poll_interval_s: int,
    max_wait_s: int,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    if cluster_dir.exists():
        shutil.rmtree(cluster_dir, ignore_errors=True)
    raw_dir = cluster_dir / "raw"
    normalized_dir = cluster_dir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized_jsonl = normalized_dir / "records.jsonl"
    if normalized_jsonl.exists():
        normalized_jsonl.unlink()

    base_params = {"entity_type": entity_type, "threshold": str(threshold), "cluster_id": cluster_id}
    chunks: List[Dict[str, Any]] = []
    total_media_items = 0
    total_normalized_rows = 0
    fallback_used = False

    try:
        full = run_single_export(
            session=session,
            auth=auth,
            dataset_id=dataset_id,
            cluster_id=cluster_id,
            suffix="full",
            export_params=base_params,
            raw_dir=raw_dir,
            normalized_jsonl=normalized_jsonl,
            run_id=run_id,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_base_s=retry_base_s,
            retry_max_s=retry_max_s,
            cooldown_on_429_s=cooldown_on_429_s,
            poll_interval_s=poll_interval_s,
            max_wait_s=max_wait_s,
        )
        chunks.append(full)
        total_media_items += int(full["media_items_count"])
        total_normalized_rows += int(full["normalized_rows_written"])
        if on_progress:
            on_progress(
                {
                    "status": "chunk_completed",
                    "chunk_suffix": "full",
                    "chunk_media_items": int(full["media_items_count"]),
                    "chunks_completed": 1,
                    "fallback_used": False,
                }
            )
    except Exception as exc:
        if not is_entities_exceeded_error(str(exc)):
            raise
        fallback_used = True
        # Restart the cluster folder so fallback output is clean and deterministic.
        if cluster_dir.exists():
            shutil.rmtree(cluster_dir, ignore_errors=True)
        raw_dir = cluster_dir / "raw"
        normalized_dir = cluster_dir / "normalized"
        raw_dir.mkdir(parents=True, exist_ok=True)
        normalized_dir.mkdir(parents=True, exist_ok=True)
        normalized_jsonl = normalized_dir / "records.jsonl"

        offset = 0
        sub_idx = 0
        while sub_idx < max_sub_partitions:
            sub_idx += 1
            current_limit = int(sub_partition_size)
            while True:
                chunk_suffix = f"chunk_{sub_idx:05d}"
                chunk_params = {**base_params, offset_param: offset, limit_param: current_limit}
                try:
                    chunk_res = run_single_export(
                        session=session,
                        auth=auth,
                        dataset_id=dataset_id,
                        cluster_id=cluster_id,
                        suffix=chunk_suffix,
                        export_params=chunk_params,
                        raw_dir=raw_dir,
                        normalized_jsonl=normalized_jsonl,
                        run_id=run_id,
                        timeout_s=timeout_s,
                        max_retries=max_retries,
                        retry_base_s=retry_base_s,
                        retry_max_s=retry_max_s,
                        cooldown_on_429_s=cooldown_on_429_s,
                        poll_interval_s=poll_interval_s,
                        max_wait_s=max_wait_s,
                    )
                    chunk_res["chunk_offset"] = offset
                    chunk_res["chunk_limit"] = current_limit
                    break
                except Exception as chunk_exc:
                    if is_entities_exceeded_error(str(chunk_exc)) and current_limit > 1:
                        current_limit = max(1, current_limit // 2)
                        continue
                    raise

            chunks.append(chunk_res)
            chunk_count = int(chunk_res["media_items_count"])
            total_media_items += chunk_count
            total_normalized_rows += int(chunk_res["normalized_rows_written"])
            if on_progress:
                on_progress(
                    {
                        "status": "chunk_completed",
                        "chunk_suffix": chunk_suffix,
                        "chunk_media_items": chunk_count,
                        "chunk_offset": offset,
                        "chunk_limit": current_limit,
                        "chunks_completed": len(chunks),
                        "fallback_used": True,
                    }
                )

            if chunk_count <= 0:
                break
            offset += chunk_count
            if chunk_count < current_limit:
                break
        else:
            raise RuntimeError(
                f"Exceeded max sub-partitions ({max_sub_partitions}) for cluster {cluster_id}"
            )

    stats = {
        "cluster_id": cluster_id,
        "rows_exported": total_media_items,
        "rows_normalized": total_normalized_rows,
        "chunks_exported": len(chunks),
        "too_large_fallback_used": fallback_used,
        "chunks": chunks,
        "timestamp": now_utc_iso(),
    }
    json_dump(cluster_dir / "stats.json", stats)
    return stats


def discover_cluster_ids(
    dataset_id: str,
    out_path: Path,
    flag_out: Optional[Path],
    entity_type: str,
    threshold: int,
    max_pages: int,
    stop_after_empty_pages: int,
    timeout_s: int,
    max_retries: int,
    retry_base_s: float,
    retry_max_s: float,
    cooldown_on_429_s: float,
    checkpoint_path: Path,
    failures_path: Path,
    allow_entity_fallback: bool = True,
) -> int:
    auth = JWTAuthProvider.from_env()
    session = requests.Session()
    cluster_ids: Set[str] = set()
    empty_streak = 0
    failures: List[Dict[str, Any]] = []
    pages_successful = 0

    for page in range(0, max_pages):
        url = build_explore_url(
            dataset_id=dataset_id,
            threshold=threshold,
            entity_type=entity_type,
            page_number=page,
        )
        try:
            payload = request_json_with_retry(
                session=session,
                auth=auth,
                method="GET",
                url=url,
                timeout_s=timeout_s,
                max_retries=max_retries,
                retry_base_s=retry_base_s,
                retry_max_s=retry_max_s,
                cooldown_on_429_s=cooldown_on_429_s,
                unit_desc=f"[discover page={page}]",
                allow_json_retry=True,
            )
        except RequestError as exc:
            failure = {
                "page": page,
                "kind": exc.kind,
                "status": exc.status,
                "snippet": exc.snippet,
                "message": str(exc),
                "timestamp": now_utc_iso(),
            }
            failures.append(failure)
            json_dump(
                failures_path,
                {
                    "dataset_id": dataset_id,
                    "completed": False,
                    "phase": "discover-cluster-ids",
                    "failures": failures,
                },
            )
            break

        page_ids = set(walk_for_cluster_ids(payload))
        cluster_ids |= page_ids
        pages_successful += 1
        print(
            f"[page {page}] found {len(page_ids)} cluster_ids, total unique {len(cluster_ids)}",
            file=sys.stderr,
        )

        json_dump(
            checkpoint_path,
            {
                "dataset_id": dataset_id,
                "last_completed_page": page,
                "pages_successful": pages_successful,
                "cluster_ids_discovered": len(cluster_ids),
                "empty_streak": empty_streak,
                "timestamp": now_utc_iso(),
            },
        )

        if len(page_ids) == 0:
            empty_streak += 1
            if empty_streak >= stop_after_empty_pages:
                print(
                    f"Stopping after {empty_streak} consecutive empty pages.",
                    file=sys.stderr,
                )
                break
        else:
            empty_streak = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_ids = sorted(cluster_ids)

    # If IMAGES page crawl yields no cluster IDs, retry discovery in CLUSTERS mode once.
    if (
        not failures
        and not sorted_ids
        and allow_entity_fallback
        and str(entity_type).upper() != "CLUSTERS"
    ):
        print(
            "No cluster IDs found using entity_type="
            f"{entity_type}. Retrying discovery with entity_type=CLUSTERS...",
            file=sys.stderr,
        )
        return discover_cluster_ids(
            dataset_id=dataset_id,
            out_path=out_path,
            flag_out=flag_out,
            entity_type="CLUSTERS",
            threshold=threshold,
            max_pages=max_pages,
            stop_after_empty_pages=stop_after_empty_pages,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_base_s=retry_base_s,
            retry_max_s=retry_max_s,
            cooldown_on_429_s=cooldown_on_429_s,
            checkpoint_path=checkpoint_path,
            failures_path=failures_path,
            allow_entity_fallback=False,
        )

    # Treat "zero discovered clusters" as incomplete to avoid silent empty exports.
    if not failures and not sorted_ids:
        failures.append(
            {
                "kind": "no_cluster_ids",
                "status": 200,
                "snippet": "",
                "message": "No cluster IDs discovered from exploration pages.",
                "timestamp": now_utc_iso(),
            }
        )
        json_dump(
            failures_path,
            {
                "dataset_id": dataset_id,
                "completed": False,
                "phase": "discover-cluster-ids",
                "failures": failures,
            },
        )

    out_path.write_text("\n".join(sorted_ids) + ("\n" if sorted_ids else ""), encoding="utf-8")

    if not failures and sorted_ids and flag_out:
        flag_out.parent.mkdir(parents=True, exist_ok=True)
        flag_out.write_text(
            "READY=1\n"
            f"CLUSTERS={len(sorted_ids)}\n"
            f"DATASET_ID={dataset_id}\n",
            encoding="utf-8",
        )

    json_dump(
        out_path.parent / "discover_summary.json",
        {
            "dataset_id": dataset_id,
            "cluster_ids_file": str(out_path),
            "cluster_count": len(sorted_ids),
            "pages_successful": pages_successful,
            "entity_type_used": str(entity_type).upper(),
            "completed": len(failures) == 0 and len(sorted_ids) > 0,
            "failures": failures,
            "timestamp": now_utc_iso(),
        },
    )
    if failures:
        print("Discovery incomplete. See failure manifest for details.", file=sys.stderr)
        return 4
    print(f"Wrote {len(sorted_ids)} unique cluster_ids to: {out_path}")
    if flag_out:
        print(f"Wrote cleaning flag to: {flag_out}")
    return 0


def discover_cluster_ids_with_strategy(
    dataset_id: str,
    out_path: Path,
    flag_out: Optional[Path],
    discover_method: str,
    entity_type: str,
    threshold: int,
    max_pages: int,
    stop_after_empty_pages: int,
    timeout_s: int,
    max_retries: int,
    retry_base_s: float,
    retry_max_s: float,
    cooldown_on_429_s: float,
    checkpoint_path: Path,
    failures_path: Path,
    ui_list_url: Optional[str],
    ui_cluster_selector: str,
    ui_next_selector: str,
    ui_max_pages: int,
    ui_max_clusters_per_page: int,
    ui_timeout_ms: int,
    ui_headful: bool,
    ui_storage_state: Optional[str],
    ui_save_storage_state: Optional[str],
    ui_manual_login: bool,
    ui_force_click: bool,
) -> int:
    mode = str(discover_method).lower()
    if mode not in {"api", "ui", "auto"}:
        raise ValueError(f"Unsupported discover method: {discover_method}")

    if mode == "ui":
        return discover_cluster_ids_via_ui(
            dataset_id=dataset_id,
            out_path=out_path,
            flag_out=flag_out,
            checkpoint_path=checkpoint_path,
            failures_path=failures_path,
            ui_list_url=ui_list_url,
            ui_cluster_selector=ui_cluster_selector,
            ui_next_selector=ui_next_selector,
            ui_max_pages=ui_max_pages,
            ui_max_clusters_per_page=ui_max_clusters_per_page,
            ui_timeout_ms=ui_timeout_ms,
            ui_headful=ui_headful,
            ui_storage_state=ui_storage_state,
            ui_save_storage_state=ui_save_storage_state,
            ui_manual_login=ui_manual_login,
            ui_force_click=ui_force_click,
        )

    rc = discover_cluster_ids(
        dataset_id=dataset_id,
        out_path=out_path,
        flag_out=flag_out,
        entity_type=entity_type,
        threshold=threshold,
        max_pages=max_pages,
        stop_after_empty_pages=stop_after_empty_pages,
        timeout_s=timeout_s,
        max_retries=max_retries,
        retry_base_s=retry_base_s,
        retry_max_s=retry_max_s,
        cooldown_on_429_s=cooldown_on_429_s,
        checkpoint_path=checkpoint_path,
        failures_path=failures_path,
    )
    if mode == "api":
        return rc

    # mode == auto:
    # Only fallback if API path did not produce cluster IDs.
    if rc != 0 and not has_cluster_ids_file_values(out_path):
        print(
            "API discovery did not return cluster IDs. Falling back to UI discovery...",
            file=sys.stderr,
        )
        return discover_cluster_ids_via_ui(
            dataset_id=dataset_id,
            out_path=out_path,
            flag_out=flag_out,
            checkpoint_path=checkpoint_path,
            failures_path=failures_path,
            ui_list_url=ui_list_url,
            ui_cluster_selector=ui_cluster_selector,
            ui_next_selector=ui_next_selector,
            ui_max_pages=ui_max_pages,
            ui_max_clusters_per_page=ui_max_clusters_per_page,
            ui_timeout_ms=ui_timeout_ms,
            ui_headful=ui_headful,
            ui_storage_state=ui_storage_state,
            ui_save_storage_state=ui_save_storage_state,
            ui_manual_login=ui_manual_login,
            ui_force_click=ui_force_click,
        )
    return rc


def ensure_unique_run_root(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{base.name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def write_run_manifest(run_root: Path, checkpoint: Dict[str, Any]) -> None:
    manifest = {
        "run_id": checkpoint.get("run_id"),
        "dataset_id": checkpoint.get("dataset_id"),
        "run_root": checkpoint.get("run_root"),
        "cluster_count": checkpoint.get("cluster_count"),
        "exported_clusters": checkpoint.get("exported_clusters"),
        "next_index": checkpoint.get("next_index"),
        "completed": checkpoint.get("completed"),
        "created_at": checkpoint.get("created_at"),
        "updated_at": checkpoint.get("updated_at"),
        "cluster_ids_file": checkpoint.get("cluster_ids_file"),
        "cluster_ids_sha256": checkpoint.get("cluster_ids_sha256"),
        "chunk_files": checkpoint.get("chunk_files", []),
        "failures_count": len(checkpoint.get("failures", [])),
        "assumptions": {
            "filter_dataset_doc_page_was_unavailable_in-runtime": True,
            "implementation_based_on_accessible_visual_layer_docs_and_existing_repo_api_patterns": True,
        },
    }
    json_dump(run_root / "manifests" / "run_manifest.json", manifest)


def save_checkpoint(checkpoint_path: Path, checkpoint: Dict[str, Any]) -> None:
    checkpoint["updated_at"] = now_utc_iso()
    json_dump(checkpoint_path, checkpoint)


def find_latest_checkpoint(output_root: Path) -> Optional[Path]:
    if not output_root.exists():
        return None
    candidates = sorted(
        (p / "manifests" / "checkpoint.json" for p in output_root.iterdir() if p.is_dir() and p.name.startswith("run_")),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def start_export_run(
    dataset_id: str,
    cluster_ids_file: Path,
    output_root: Path,
    chunk_size: int,
    checkpoint_file: Optional[Path],
    command_config: Dict[str, Any],
) -> Tuple[Path, Path, Dict[str, Any]]:
    cluster_ids = load_cluster_ids(cluster_ids_file)
    run_root = ensure_unique_run_root(output_root)
    manifests_dir = run_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (run_root / "metadata_by_cluster").mkdir(parents=True, exist_ok=True)

    copied_cluster_ids = run_root / "cluster_ids.txt"
    copied_cluster_ids.write_text("\n".join(cluster_ids) + "\n", encoding="utf-8")
    chunk_meta = write_cluster_chunk_files(
        cluster_ids=cluster_ids,
        chunk_size=chunk_size,
        chunks_dir=run_root / "cluster_chunks",
    )
    cluster_ids_sha = sha256_file(copied_cluster_ids)

    checkpoint = {
        "schema_version": 1,
        "run_id": run_root.name,
        "dataset_id": dataset_id,
        "run_root": str(run_root),
        "created_at": now_utc_iso(),
        "updated_at": now_utc_iso(),
        "completed": False,
        "cluster_ids": cluster_ids,
        "cluster_count": len(cluster_ids),
        "cluster_ids_file": str(copied_cluster_ids),
        "cluster_ids_sha256": cluster_ids_sha,
        "chunk_files": chunk_meta,
        "next_index": 0,
        "exported_clusters": 0,
        "cluster_state": {},
        "failures": [],
        "config": command_config,
    }

    cp_path = checkpoint_file if checkpoint_file else (manifests_dir / "checkpoint.json")
    save_checkpoint(cp_path, checkpoint)
    json_dump(
        run_root / "manifests" / "failures.json",
        {"run_id": run_root.name, "completed": False, "failures": []},
    )
    write_run_manifest(run_root, checkpoint)
    return run_root, cp_path, checkpoint


def load_resume_checkpoint(checkpoint_path: Path) -> Tuple[Path, Dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    run_root = Path(checkpoint.get("run_root", ""))
    if not run_root.exists():
        raise FileNotFoundError(
            f"Run root from checkpoint does not exist: {run_root}"
        )
    return run_root, checkpoint


def export_cluster_metadata_once(args: argparse.Namespace, checkpoint_path: Optional[Path], resume: bool) -> Tuple[int, Path]:
    auth = JWTAuthProvider.from_env()
    session = requests.Session()

    output_root = Path(args.output_root)
    if resume:
        resolved_checkpoint = checkpoint_path
        if resolved_checkpoint is None:
            resolved_checkpoint = find_latest_checkpoint(output_root)
            if resolved_checkpoint is None:
                raise FileNotFoundError(
                    f"No checkpoint found under output root: {output_root}"
                )
        run_root, checkpoint = load_resume_checkpoint(resolved_checkpoint)
        cp_path = resolved_checkpoint
    else:
        run_root, cp_path, checkpoint = start_export_run(
            dataset_id=args.dataset_id,
            cluster_ids_file=Path(args.cluster_ids_file),
            output_root=output_root,
            chunk_size=args.chunk_size,
            checkpoint_file=checkpoint_path,
            command_config={
                "entity_type": args.entity_type,
                "threshold": str(args.threshold),
                "sub_partition_size": args.sub_partition_size,
                "offset_param": args.offset_param,
                "limit_param": args.limit_param,
                "max_sub_partitions": args.max_sub_partitions,
                "timeout": args.timeout,
                "max_retries": args.max_retries,
                "retry_base_seconds": args.retry_base_seconds,
                "retry_max_seconds": args.retry_max_seconds,
                "cooldown_on_429_seconds": args.cooldown_on_429_seconds,
                "poll_interval_seconds": args.poll_interval_seconds,
                "max_wait_seconds": args.max_wait_seconds,
            },
        )

    cluster_ids: List[str] = checkpoint.get("cluster_ids", [])
    cluster_state: Dict[str, Any] = checkpoint.setdefault("cluster_state", {})
    failures: List[Dict[str, Any]] = checkpoint.setdefault("failures", [])
    start_index = int(checkpoint.get("next_index", 0))
    failures_path = run_root / "manifests" / "failures.json"

    for idx in range(start_index, len(cluster_ids)):
        cluster_id = cluster_ids[idx]
        cluster_state[cluster_id] = {
            "status": "running",
            "started_at": now_utc_iso(),
            "index": idx,
        }
        save_checkpoint(cp_path, checkpoint)
        write_run_manifest(run_root, checkpoint)

        cluster_dir = run_root / "metadata_by_cluster" / cluster_id

        def progress_callback(progress: Dict[str, Any]) -> None:
            state = cluster_state.setdefault(cluster_id, {})
            state["progress"] = progress
            state["status"] = "running"
            state["updated_at"] = now_utc_iso()
            save_checkpoint(cp_path, checkpoint)

        try:
            stats = run_export_for_cluster(
                session=session,
                auth=auth,
                dataset_id=args.dataset_id,
                cluster_id=cluster_id,
                run_id=str(checkpoint.get("run_id")),
                cluster_dir=cluster_dir,
                entity_type=args.entity_type,
                threshold=str(args.threshold),
                sub_partition_size=args.sub_partition_size,
                offset_param=args.offset_param,
                limit_param=args.limit_param,
                max_sub_partitions=args.max_sub_partitions,
                timeout_s=args.timeout,
                max_retries=args.max_retries,
                retry_base_s=args.retry_base_seconds,
                retry_max_s=args.retry_max_seconds,
                cooldown_on_429_s=args.cooldown_on_429_seconds,
                poll_interval_s=args.poll_interval_seconds,
                max_wait_s=args.max_wait_seconds,
                on_progress=progress_callback,
            )
            cluster_state[cluster_id] = {
                "status": "completed",
                "finished_at": now_utc_iso(),
                "index": idx,
                "stats": stats,
            }
            checkpoint["next_index"] = idx + 1
            checkpoint["exported_clusters"] = int(checkpoint.get("exported_clusters", 0)) + 1
            save_checkpoint(cp_path, checkpoint)
            write_run_manifest(run_root, checkpoint)
            print(
                f"[cluster {idx + 1}/{len(cluster_ids)}] {cluster_id}: exported "
                f"{stats['rows_exported']} items across {stats['chunks_exported']} chunk(s)"
            )
        except Exception as exc:
            failure = {
                "cluster_id": cluster_id,
                "index": idx,
                "message": str(exc),
                "timestamp": now_utc_iso(),
            }
            failures.append(failure)
            cluster_state[cluster_id] = {
                "status": "failed",
                "failed_at": now_utc_iso(),
                "index": idx,
                "error": str(exc),
            }
            checkpoint["completed"] = False
            checkpoint["next_index"] = idx
            save_checkpoint(cp_path, checkpoint)
            json_dump(
                failures_path,
                {
                    "run_id": checkpoint.get("run_id"),
                    "completed": False,
                    "failures": failures,
                },
            )
            write_run_manifest(run_root, checkpoint)
            print(
                f"[cluster {idx + 1}/{len(cluster_ids)}] {cluster_id}: failed ({exc})",
                file=sys.stderr,
            )
            return 7, cp_path

    checkpoint["completed"] = True
    checkpoint["finished_at"] = now_utc_iso()
    save_checkpoint(cp_path, checkpoint)
    json_dump(
        failures_path,
        {
            "run_id": checkpoint.get("run_id"),
            "completed": True,
            "failures": failures,
        },
    )
    write_run_manifest(run_root, checkpoint)
    print(f"Run complete. Output: {run_root}")
    return 0, cp_path


def export_cluster_metadata(args: argparse.Namespace) -> int:
    checkpoint_path = Path(args.checkpoint_file) if args.checkpoint_file else None
    resume = bool(args.resume)
    cycles = 0
    while True:
        rc, cp_path = export_cluster_metadata_once(args=args, checkpoint_path=checkpoint_path, resume=resume)
        if rc == 0:
            return 0
        if not args.auto_resume:
            return rc
        cycles += 1
        if cycles > args.max_auto_resume_cycles:
            print(
                "Auto-resume exhausted max cycles "
                f"({args.max_auto_resume_cycles}). Last checkpoint: {cp_path}",
                file=sys.stderr,
            )
            return rc
        print(
            f"Auto-resume cycle {cycles}/{args.max_auto_resume_cycles} "
            f"sleeping {args.auto_resume_wait_seconds}s before retry...",
            file=sys.stderr,
        )
        time.sleep(args.auto_resume_wait_seconds)
        checkpoint_path = cp_path
        resume = True


def run_all(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    discover_checkpoint = out_path.parent / "discover_checkpoint.json"
    discover_failures = out_path.parent / "discover_failures.json"

    rc = discover_cluster_ids_with_strategy(
        dataset_id=args.dataset_id,
        out_path=out_path,
        flag_out=Path(args.flag_out) if args.flag_out else None,
        discover_method=args.discover_method,
        entity_type=args.entity_type,
        threshold=args.threshold,
        max_pages=args.max_pages,
        stop_after_empty_pages=args.stop_after_empty_pages,
        timeout_s=args.timeout,
        max_retries=args.max_retries,
        retry_base_s=args.retry_base_seconds,
        retry_max_s=args.retry_max_seconds,
        cooldown_on_429_s=args.cooldown_on_429_seconds,
        checkpoint_path=discover_checkpoint,
        failures_path=discover_failures,
        ui_list_url=args.ui_list_url,
        ui_cluster_selector=args.ui_cluster_selector,
        ui_next_selector=args.ui_next_selector,
        ui_max_pages=args.ui_max_pages,
        ui_max_clusters_per_page=args.ui_max_clusters_per_page,
        ui_timeout_ms=args.ui_timeout_ms,
        ui_headful=args.ui_headful,
        ui_storage_state=args.ui_storage_state,
        ui_save_storage_state=args.ui_save_storage_state,
        ui_manual_login=args.ui_manual_login,
        ui_force_click=args.ui_force_click,
    )
    if rc != 0:
        return rc

    export_args = argparse.Namespace(
        dataset_id=args.dataset_id,
        cluster_ids_file=str(out_path),
        output_root=args.output_root,
        chunk_size=args.chunk_size,
        resume=False,
        checkpoint_file=None,
        entity_type=args.entity_type,
        threshold=str(args.threshold),
        sub_partition_size=args.sub_partition_size,
        offset_param=args.offset_param,
        limit_param=args.limit_param,
        max_sub_partitions=args.max_sub_partitions,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        retry_max_seconds=args.retry_max_seconds,
        cooldown_on_429_seconds=args.cooldown_on_429_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        max_wait_seconds=args.max_wait_seconds,
        auto_resume=args.auto_resume,
        max_auto_resume_cycles=args.max_auto_resume_cycles,
        auto_resume_wait_seconds=args.auto_resume_wait_seconds,
    )
    return export_cluster_metadata(export_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible Visual Layer cluster discovery and per-cluster metadata export."
    )
    subparsers = parser.add_subparsers(dest="command")

    discover = subparsers.add_parser(
        "discover-cluster-ids",
        help="Discover unique cluster IDs from exploration_metadata pages.",
    )
    discover.add_argument(
        "--dataset-id",
        required=False,
        help="Visual Layer dataset ID. Optional if VL_DATASET_ID/dataset_id is set in .env.",
    )
    discover.add_argument("--out", default="clean_imagenet1k/cluster_ids.txt")
    discover.add_argument("--flag-out", default="CLEANING_READY.flag")
    discover.add_argument("--entity-type", default="IMAGES")
    discover.add_argument("--threshold", type=int, default=1)
    discover.add_argument("--max-pages", type=int, default=5000)
    discover.add_argument("--stop-after-empty-pages", type=int, default=2)
    discover.add_argument("--timeout", type=int, default=30)
    discover.add_argument("--max-retries", type=int, default=10)
    discover.add_argument("--retry-base-seconds", type=float, default=2.0)
    discover.add_argument("--retry-max-seconds", type=float, default=120.0)
    discover.add_argument("--cooldown-on-429-seconds", type=float, default=0.0)
    discover.add_argument("--discover-method", choices=["api", "ui", "auto"], default="api")
    discover.add_argument(
        "--ui-list-url",
        default=None,
        help="Clusters list URL to crawl in UI mode. Defaults to <VL_BASE_URL>/dataset/<dataset_id>/data",
    )
    discover.add_argument(
        "--ui-cluster-selector",
        default="a[href*='/cluster/']",
        help="CSS selector for clickable cluster items in UI mode.",
    )
    discover.add_argument(
        "--ui-next-selector",
        default="",
        help="Optional CSS selector for next-page button in UI mode.",
    )
    discover.add_argument("--ui-max-pages", type=int, default=200)
    discover.add_argument("--ui-max-clusters-per-page", type=int, default=1000)
    discover.add_argument("--ui-timeout-ms", type=int, default=30000)
    discover.add_argument("--ui-headful", action="store_true")
    discover.add_argument("--ui-storage-state", default=None, help="Playwright storage-state JSON file.")
    discover.add_argument(
        "--ui-save-storage-state",
        default=None,
        help="Path to save Playwright storage state after manual login.",
    )
    discover.add_argument(
        "--ui-manual-login",
        action="store_true",
        help="Pause for manual login in browser before crawling clusters.",
    )
    discover.add_argument(
        "--ui-force-click",
        action="store_true",
        help=(
            "Force clicking each cluster UI item to resolve cluster IDs from navigation URL. "
            "Default behavior prefers parsing IDs from href without clicking."
        ),
    )

    export = subparsers.add_parser(
        "export-cluster-metadata",
        help="Export per-cluster metadata into raw JSON + normalized JSONL artifacts.",
    )
    export.add_argument(
        "--dataset-id",
        required=False,
        help="Visual Layer dataset ID. Optional if VL_DATASET_ID/dataset_id is set in .env.",
    )
    export.add_argument("--cluster-ids-file", required=True)
    export.add_argument("--output-root", default="data/cluster_exports_from_vl")
    export.add_argument("--chunk-size", type=int, default=100)
    export.add_argument("--resume", action="store_true")
    export.add_argument("--checkpoint-file", default=None)
    export.add_argument("--entity-type", default="IMAGES")
    export.add_argument("--threshold", default="1")
    export.add_argument("--sub-partition-size", type=int, default=10000)
    export.add_argument("--offset-param", default="offset")
    export.add_argument("--limit-param", default="limit")
    export.add_argument("--max-sub-partitions", type=int, default=10000)
    export.add_argument("--timeout", type=int, default=60)
    export.add_argument("--max-retries", type=int, default=10)
    export.add_argument("--retry-base-seconds", type=float, default=2.0)
    export.add_argument("--retry-max-seconds", type=float, default=120.0)
    export.add_argument("--cooldown-on-429-seconds", type=float, default=0.0)
    export.add_argument("--poll-interval-seconds", type=int, default=10)
    export.add_argument("--max-wait-seconds", type=int, default=3600)
    export.add_argument("--auto-resume", action="store_true")
    export.add_argument("--max-auto-resume-cycles", type=int, default=20)
    export.add_argument("--auto-resume-wait-seconds", type=int, default=60)

    run_all_cmd = subparsers.add_parser(
        "run-all",
        help="Run discovery first, then export per-cluster metadata.",
    )
    run_all_cmd.add_argument(
        "--dataset-id",
        required=False,
        help="Visual Layer dataset ID. Optional if VL_DATASET_ID/dataset_id is set in .env.",
    )
    run_all_cmd.add_argument("--out", default="clean_imagenet1k/cluster_ids.txt")
    run_all_cmd.add_argument("--flag-out", default="CLEANING_READY.flag")
    run_all_cmd.add_argument("--output-root", default="data/cluster_exports_from_vl")
    run_all_cmd.add_argument("--entity-type", default="IMAGES")
    run_all_cmd.add_argument("--threshold", type=int, default=1)
    run_all_cmd.add_argument("--max-pages", type=int, default=5000)
    run_all_cmd.add_argument("--stop-after-empty-pages", type=int, default=2)
    run_all_cmd.add_argument("--chunk-size", type=int, default=100)
    run_all_cmd.add_argument("--sub-partition-size", type=int, default=10000)
    run_all_cmd.add_argument("--offset-param", default="offset")
    run_all_cmd.add_argument("--limit-param", default="limit")
    run_all_cmd.add_argument("--max-sub-partitions", type=int, default=10000)
    run_all_cmd.add_argument("--timeout", type=int, default=60)
    run_all_cmd.add_argument("--max-retries", type=int, default=10)
    run_all_cmd.add_argument("--retry-base-seconds", type=float, default=2.0)
    run_all_cmd.add_argument("--retry-max-seconds", type=float, default=120.0)
    run_all_cmd.add_argument("--cooldown-on-429-seconds", type=float, default=0.0)
    run_all_cmd.add_argument("--discover-method", choices=["api", "ui", "auto"], default="api")
    run_all_cmd.add_argument(
        "--ui-list-url",
        default=None,
        help="Clusters list URL to crawl in UI mode. Defaults to <VL_BASE_URL>/dataset/<dataset_id>/data",
    )
    run_all_cmd.add_argument(
        "--ui-cluster-selector",
        default="a[href*='/cluster/']",
        help="CSS selector for clickable cluster items in UI mode.",
    )
    run_all_cmd.add_argument(
        "--ui-next-selector",
        default="",
        help="Optional CSS selector for next-page button in UI mode.",
    )
    run_all_cmd.add_argument("--ui-max-pages", type=int, default=200)
    run_all_cmd.add_argument("--ui-max-clusters-per-page", type=int, default=1000)
    run_all_cmd.add_argument("--ui-timeout-ms", type=int, default=30000)
    run_all_cmd.add_argument("--ui-headful", action="store_true")
    run_all_cmd.add_argument("--ui-storage-state", default=None, help="Playwright storage-state JSON file.")
    run_all_cmd.add_argument(
        "--ui-save-storage-state",
        default=None,
        help="Path to save Playwright storage state after manual login.",
    )
    run_all_cmd.add_argument(
        "--ui-manual-login",
        action="store_true",
        help="Pause for manual login in browser before crawling clusters.",
    )
    run_all_cmd.add_argument(
        "--ui-force-click",
        action="store_true",
        help=(
            "Force clicking each cluster UI item to resolve cluster IDs from navigation URL. "
            "Default behavior prefers parsing IDs from href without clicking."
        ),
    )
    run_all_cmd.add_argument("--poll-interval-seconds", type=int, default=10)
    run_all_cmd.add_argument("--max-wait-seconds", type=int, default=3600)
    run_all_cmd.add_argument("--auto-resume", action="store_true")
    run_all_cmd.add_argument("--max-auto-resume-cycles", type=int, default=20)
    run_all_cmd.add_argument("--auto-resume-wait-seconds", type=int, default=60)

    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    raw = list(argv if argv is not None else sys.argv[1:])

    commands = {"discover-cluster-ids", "export-cluster-metadata", "run-all"}
    if raw and raw[0] not in commands and raw[0] not in ("-h", "--help"):
        # Backward-compat mode: old invocation without subcommand behaves as discovery.
        raw = ["discover-cluster-ids", *raw]
    if not raw:
        parser.print_help()
        raise SystemExit(2)
    return parser.parse_args(raw)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.dataset_id = resolve_dataset_id(getattr(args, "dataset_id", None))
    if args.command == "discover-cluster-ids":
        out_path = Path(args.out)
        return discover_cluster_ids_with_strategy(
            dataset_id=args.dataset_id,
            out_path=out_path,
            flag_out=Path(args.flag_out) if args.flag_out else None,
            discover_method=args.discover_method,
            entity_type=args.entity_type,
            threshold=args.threshold,
            max_pages=args.max_pages,
            stop_after_empty_pages=args.stop_after_empty_pages,
            timeout_s=args.timeout,
            max_retries=args.max_retries,
            retry_base_s=args.retry_base_seconds,
            retry_max_s=args.retry_max_seconds,
            cooldown_on_429_s=args.cooldown_on_429_seconds,
            checkpoint_path=out_path.parent / "discover_checkpoint.json",
            failures_path=out_path.parent / "discover_failures.json",
            ui_list_url=args.ui_list_url,
            ui_cluster_selector=args.ui_cluster_selector,
            ui_next_selector=args.ui_next_selector,
            ui_max_pages=args.ui_max_pages,
            ui_max_clusters_per_page=args.ui_max_clusters_per_page,
            ui_timeout_ms=args.ui_timeout_ms,
            ui_headful=args.ui_headful,
            ui_storage_state=args.ui_storage_state,
            ui_save_storage_state=args.ui_save_storage_state,
            ui_manual_login=args.ui_manual_login,
            ui_force_click=args.ui_force_click,
        )
    if args.command == "export-cluster-metadata":
        return export_cluster_metadata(args)
    if args.command == "run-all":
        return run_all(args)
    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
