#!/usr/bin/env bash
# Crawl Visual Layer UI one dataset page at a time and persist cluster IDs per page.
# Then merge all per-page files into one deduplicated cluster_ids file.
set -u -o pipefail

usage() {
  cat <<'EOF'
Usage:
  discover_cluster_ids_by_page.sh \
    --dataset-id <uuid> \
    --start-page <n> \
    --end-page <n> \
    [--stop-after-empty-pages <n>] \
    [--out-dir <dir>] \
    [--merged-out <file>] \
    [--discover-root <dir>] \
    [--base-url <url>] \
    [--python <python-bin>] \
    [-- <extra export_pages.py args>]

Examples:
  ./clean_imagenet1k/discover_cluster_ids_by_page.sh \
    --dataset-id "$VL_DATASET_ID" \
    --start-page 1 \
    --end-page 200 \
    --out-dir /data/saeed/data/cluster_exports_full/by_page_ids \
    --merged-out /data/saeed/cluster_ids_full_vm.txt \
    --discover-root /data/saeed/data/cluster_exports_full/discover_by_page \
    -- --ui-storage-state /data/saeed/.vl_storage_state.json

Notes:
  - UI list pages are 1-indexed. If --start-page is 0, it is adjusted to 1.
  - By default, the script stops after 5 consecutive pages with 0 IDs.
  - Any extra args after '--' are passed directly to 'export_pages.py discover-cluster-ids'.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT_SCRIPT="${SCRIPT_DIR}/export_pages.py"

DATASET_ID="${VL_DATASET_ID:-}"
START_PAGE=1
END_PAGE=""
STOP_AFTER_EMPTY_PAGES=5
OUT_DIR="clean_imagenet1k/by_page_cluster_ids"
MERGED_OUT="clean_imagenet1k/cluster_ids_merged.txt"
DISCOVER_ROOT=""
BASE_URL="${VL_BASE_URL:-https://app.visual-layer.com}"
PYTHON_BIN="python3"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-id)
      DATASET_ID="${2:-}"
      shift 2
      ;;
    --start-page)
      START_PAGE="${2:-}"
      shift 2
      ;;
    --end-page)
      END_PAGE="${2:-}"
      shift 2
      ;;
    --stop-after-empty-pages)
      STOP_AFTER_EMPTY_PAGES="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --merged-out)
      MERGED_OUT="${2:-}"
      shift 2
      ;;
    --discover-root)
      DISCOVER_ROOT="${2:-}"
      shift 2
      ;;
    --base-url)
      BASE_URL="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${DATASET_ID}" ]]; then
  echo "Missing --dataset-id (or VL_DATASET_ID)." >&2
  exit 2
fi
if [[ -z "${END_PAGE}" ]]; then
  echo "Missing --end-page." >&2
  exit 2
fi
if [[ ! "${START_PAGE}" =~ ^[0-9]+$ ]] || [[ ! "${END_PAGE}" =~ ^[0-9]+$ ]]; then
  echo "--start-page and --end-page must be non-negative integers." >&2
  exit 2
fi
if [[ ! "${STOP_AFTER_EMPTY_PAGES}" =~ ^[0-9]+$ ]]; then
  echo "--stop-after-empty-pages must be a non-negative integer." >&2
  exit 2
fi
if (( START_PAGE < 1 )); then
  echo "UI pages are 1-indexed. Adjusting --start-page ${START_PAGE} -> 1." >&2
  START_PAGE=1
fi
if (( END_PAGE < START_PAGE )); then
  echo "--end-page (${END_PAGE}) must be >= --start-page (${START_PAGE})." >&2
  exit 2
fi
if [[ ! -f "${EXPORT_SCRIPT}" ]]; then
  echo "Could not find export script at ${EXPORT_SCRIPT}" >&2
  exit 2
fi

if [[ -z "${DISCOVER_ROOT}" ]]; then
  DISCOVER_ROOT="${OUT_DIR}/discover"
fi

mkdir -p "${OUT_DIR}" "${DISCOVER_ROOT}"
mkdir -p "$(dirname "${MERGED_OUT}")"

failed_pages=()
success_pages=0
pages_with_ids=0
empty_streak=0
empty_streak_pages=()
tail_stop_triggered=0
tail_stop_pages=()

for (( page=START_PAGE; page<=END_PAGE; page++ )); do
  page_tag="$(printf "%05d" "${page}")"
  page_out="${OUT_DIR}/cluster_ids_page_${page_tag}.txt"
  page_discover="${DISCOVER_ROOT}/page_${page_tag}"
  ui_list_url="${BASE_URL%/}/dataset/${DATASET_ID}/data?page=${page}"
  rc=0
  page_ids=0

  echo "[page ${page}] discover -> ${page_out}" >&2
  if "${PYTHON_BIN}" "${EXPORT_SCRIPT}" discover-cluster-ids \
      --dataset-id "${DATASET_ID}" \
      --discover-method ui \
      --out "${page_out}" \
      --discover-dir "${page_discover}" \
      --ui-list-url "${ui_list_url}" \
      --ui-max-pages 1 \
      "${EXTRA_ARGS[@]}"; then
    rc=0
  else
    rc=$?
    failed_pages+=("${page}")
  fi

  if [[ -f "${page_out}" ]]; then
    page_ids="$(wc -l < "${page_out}" | tr -d '[:space:]')"
  else
    page_ids=0
  fi

  if (( rc == 0 )); then
    success_pages=$((success_pages + 1))
    echo "[page ${page}] wrote ${page_ids} cluster_ids" >&2
  else
    echo "[page ${page}] failed (rc=${rc}), file_ids=${page_ids}" >&2
  fi

  if (( page_ids > 0 )); then
    pages_with_ids=$((pages_with_ids + 1))
    empty_streak=0
    empty_streak_pages=()
    continue
  fi

  empty_streak=$((empty_streak + 1))
  empty_streak_pages+=("${page}")
  if (( STOP_AFTER_EMPTY_PAGES > 0 && empty_streak >= STOP_AFTER_EMPTY_PAGES )); then
    tail_stop_triggered=1
    tail_stop_pages=("${empty_streak_pages[@]}")
    echo "Stopping after ${empty_streak} consecutive empty pages: ${tail_stop_pages[*]}" >&2
    break
  fi
done

tmp_merge="$(mktemp "${MERGED_OUT}.tmp.XXXXXX")"
shopt -s nullglob
page_files=( "${OUT_DIR}"/cluster_ids_page_*.txt )
shopt -u nullglob

if (( ${#page_files[@]} == 0 )); then
  rm -f "${tmp_merge}"
  echo "No per-page cluster ID files found under ${OUT_DIR}" >&2
  exit 1
fi

cat "${page_files[@]}" \
  | sed '/^[[:space:]]*$/d;/^[[:space:]]*#/d' \
  | tr '[:upper:]' '[:lower:]' \
  | sort -u > "${tmp_merge}"
mv "${tmp_merge}" "${MERGED_OUT}"
merged_count="$(wc -l < "${MERGED_OUT}" | tr -d '[:space:]')"

echo "Merged ${#page_files[@]} page files into ${MERGED_OUT} (${merged_count} unique IDs)." >&2
echo "Pages succeeded: ${success_pages}" >&2
echo "Pages with IDs: ${pages_with_ids}" >&2
if (( tail_stop_triggered == 1 )); then
  echo "Tail stop triggered on empty pages: ${tail_stop_pages[*]}" >&2
fi

hard_failed_pages=()
if (( ${#failed_pages[@]} > 0 )); then
  for fp in "${failed_pages[@]}"; do
    in_tail=0
    if (( tail_stop_triggered == 1 )); then
      for tp in "${tail_stop_pages[@]}"; do
        if [[ "${fp}" == "${tp}" ]]; then
          in_tail=1
          break
        fi
      done
    fi
    if (( in_tail == 0 )); then
      hard_failed_pages+=("${fp}")
    fi
  done
fi

if (( ${#hard_failed_pages[@]} > 0 )); then
  echo "Pages failed (${#hard_failed_pages[@]}): ${hard_failed_pages[*]}" >&2
  exit 3
fi

exit 0
