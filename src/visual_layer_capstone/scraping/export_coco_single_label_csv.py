import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def get_largest_bbox_label(objects) -> tuple[Optional[list], Optional[int], Optional[int]]:
    bboxes = objects["bbox"]
    labels = objects["category"]
    bbox_ids = objects.get("bbox_id", None)

    if len(bboxes) == 0:
        return None, None, None

    areas = [w * h for (_, _, w, h) in bboxes]
    idx = max(range(len(areas)), key=lambda i: areas[i])

    bbox = bboxes[idx]
    label = labels[idx]
    bbox_id = bbox_ids[idx] if bbox_ids is not None else None
    return bbox, label, bbox_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="detection-datasets/coco")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--streaming", action="store_true", help="Stream dataset instead of loading fully.")
    ap.add_argument("--max-rows", type=int, default=0, help="Optional cap for debugging (0 = no cap).")
    args = ap.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = out_path.resolve()

    print(f"[export] loading dataset={args.dataset} split={args.split} streaming={args.streaming}")
    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    rows = []
    count = 0
    for sample in tqdm(ds, desc="Exporting"):
        image_id = sample.get("image_id")
        file_name = sample.get("file_name", None)
        objects = sample.get("objects", {})

        _, label, bbox_id = get_largest_bbox_label(objects)
        if label is None:
            continue

        rows.append({
            "image_id": image_id,
            "object_id": bbox_id,
            "label": label,
            "file_name": file_name,
        })
        count += 1
        if args.max_rows > 0 and count >= args.max_rows:
            break

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_path), index=False)
    print(f"[export] wrote {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
