#!/usr/bin/env python3
"""Embedding export utilities.

Subcommands:
- extract-coco-clip: stream COCO, crop objects, export CLIP embeddings + metadata.
- metadata-to-npy: convert per-object `.pt` vectors from metadata CSV into a single `.npy`.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def crop_object(image: Image.Image, bbox) -> Image.Image:
    x, y, w, h = bbox
    return image.crop((x, y, x + w, y + h))


def extract_coco_clip(args: argparse.Namespace) -> int:
    from sentence_transformers import SentenceTransformer

    embed_dir = Path(args.embeddings_dir)
    meta_csv = Path(args.metadata_csv)
    meta_json = Path(args.metadata_json)

    embed_dir.mkdir(parents=True, exist_ok=True)
    meta_csv.parent.mkdir(parents=True, exist_ok=True)
    meta_json.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract-coco-clip] device={device}")
    print(f"[extract-coco-clip] loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name, device=device)

    ds = load_dataset(args.dataset_name, split=args.split, streaming=args.streaming)

    rows = []
    count = 0
    for sample in tqdm(ds, desc="Embedding COCO objects"):
        image: Image.Image = sample["image"]
        image_id = sample.get("image_id")
        objects = sample.get("objects", {})
        bboxes = objects.get("bbox", [])
        labels = objects.get("category", [])
        object_ids = objects.get("bbox_id", [None] * len(bboxes))

        for bbox, label, object_id in zip(bboxes, labels, object_ids):
            try:
                crop = crop_object(image, bbox).convert("RGB")
            except Exception:
                continue

            with torch.no_grad():
                emb = model.encode(crop, convert_to_tensor=True, normalize_embeddings=True)

            vector_id = str(uuid.uuid4())
            vector_path = embed_dir / f"{vector_id}.pt"
            torch.save(emb.cpu(), vector_path)

            rows.append(
                {
                    "vector_id": vector_id,
                    "vector_path": str(vector_path),
                    "image_id": image_id,
                    "object_id": object_id,
                    "label": label,
                }
            )
            count += 1

            if args.max_samples > 0 and count >= args.max_samples:
                break

        if args.max_samples > 0 and count >= args.max_samples:
            break

    df = pd.DataFrame(rows)
    df.to_csv(meta_csv, index=False)
    with meta_json.open("w") as f:
        json.dump(rows, f, indent=2)

    print(f"[extract-coco-clip] wrote vectors: {embed_dir}")
    print(f"[extract-coco-clip] wrote metadata csv: {meta_csv} rows={len(df)}")
    print(f"[extract-coco-clip] wrote metadata json: {meta_json}")
    return 0


def infer_columns(df: pd.DataFrame):
    path_candidates = ["vector_path", "path", "embedding_path", "pt_path", "file_path"]
    label_candidates = ["label", "category", "category_id", "class", "class_id"]
    id_candidates = ["vector_id", "id", "embedding_id", "uuid"]

    path_col = next((c for c in path_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)
    id_col = next((c for c in id_candidates if c in df.columns), None)
    return id_col, path_col, label_col


def find_existing_upwards(rel_path: str, start_dir: Path, max_up: int = 5) -> Path | None:
    rel = Path(rel_path)
    cur = start_dir
    for _ in range(max_up + 1):
        candidate = (cur / rel).resolve()
        if candidate.exists():
            return candidate
        cur = cur.parent
    return None


def load_pt_embedding(pt_path: Path) -> np.ndarray:
    t = torch.load(str(pt_path), map_location="cpu")
    if isinstance(t, torch.Tensor):
        emb = t
    elif isinstance(t, dict):
        if "embedding" in t:
            emb = t["embedding"]
        elif "emb" in t:
            emb = t["emb"]
        else:
            tensor_vals = [v for v in t.values() if isinstance(v, torch.Tensor)]
            if not tensor_vals:
                raise ValueError(f"No tensor found in dict at {pt_path}. Keys: {list(t.keys())}")
            emb = tensor_vals[0]
    else:
        raise ValueError(f"Unsupported .pt content type at {pt_path}: {type(t)}")

    return emb.detach().cpu().float().view(-1).numpy()


def resolve_vector_path(raw_path: str, metadata_parent: Path, embeddings_dir: Optional[Path]) -> Path:
    p = Path(str(raw_path)).expanduser()

    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()

    cand = (metadata_parent / p).resolve()
    if cand.exists():
        return cand

    if embeddings_dir is not None:
        cand = (embeddings_dir / p.name).resolve()
        if cand.exists():
            return cand
        cand = (embeddings_dir / p).resolve()
        if cand.exists():
            return cand

    cand = (metadata_parent / "embeddings" / p.name).resolve()
    if cand.exists():
        return cand

    raise FileNotFoundError(
        f"Could not resolve embedding path '{raw_path}'. Tried metadata_parent={metadata_parent}, "
        f"embeddings_dir={embeddings_dir}."
    )


def metadata_to_npy(args: argparse.Namespace) -> int:
    cwd = Path.cwd()

    meta_path = Path(args.metadata_csv)
    if not meta_path.exists():
        found = find_existing_upwards(args.metadata_csv, cwd, max_up=args.max_up)
        if found is None:
            raise FileNotFoundError(
                f"metadata CSV not found: {args.metadata_csv} (searched up to {args.max_up} parent dirs)"
            )
        meta_path = found

    metadata_parent = meta_path.parent.resolve()

    emb_dir = Path(args.embeddings_dir)
    if emb_dir.exists():
        emb_dir = emb_dir.resolve()
    else:
        found_dir = find_existing_upwards(args.embeddings_dir, cwd, max_up=args.max_up)
        emb_dir = found_dir.resolve() if found_dir is not None else None

    df = pd.read_csv(meta_path).reset_index(drop=True)
    id_col, path_col, label_col = infer_columns(df)

    if args.id_col:
        id_col = args.id_col
    if args.path_col:
        path_col = args.path_col
    if args.label_col:
        label_col = args.label_col

    if path_col is None or label_col is None:
        raise ValueError(
            f"Could not infer path/label columns. Found: {df.columns.tolist()}. "
            "Use --path-col and --label-col to override."
        )

    if args.max_rows > 0 and len(df) > args.max_rows:
        df = df.iloc[: args.max_rows].reset_index(drop=True)

    resolved_paths = [
        resolve_vector_path(raw, metadata_parent=metadata_parent, embeddings_dir=emb_dir)
        for raw in df[path_col].tolist()
    ]

    embs = []
    dim = None
    for i, p in enumerate(resolved_paths):
        e = load_pt_embedding(p)
        if dim is None:
            dim = e.shape[0]
        elif e.shape[0] != dim:
            raise ValueError(f"Embedding dim mismatch at row {i}: got {e.shape[0]}, expected {dim}")
        embs.append(e)

    X = np.vstack(embs).astype(np.float32)

    out_npy = Path(args.out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, X)

    out = pd.DataFrame()
    if id_col is not None and id_col in df.columns:
        out["vector_id"] = df[id_col].astype(str).values
    else:
        out["vector_id"] = [f"row_{i}" for i in range(len(df))]

    out["label"] = df[label_col].values
    out["vector_path"] = [str(p) for p in resolved_paths]
    for extra in ["image_id", "object_id"]:
        if extra in df.columns:
            out[extra] = df[extra].values

    out_labels = Path(args.out_labels)
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_labels, index=False)

    print(f"[metadata-to-npy] metadata_csv={meta_path}")
    print(f"[metadata-to-npy] embeddings_dir={emb_dir}")
    print(f"[metadata-to-npy] embeddings_npy={out_npy} shape={X.shape}")
    print(f"[metadata-to-npy] labels_csv={out_labels} rows={len(out)}")
    print(f"[metadata-to-npy] columns=id:{id_col} path:{path_col} label:{label_col}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding export utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    clip_cmd = sub.add_parser("extract-coco-clip", help="Stream COCO and export CLIP embeddings")
    clip_cmd.add_argument("--dataset-name", type=str, default="detection-datasets/coco")
    clip_cmd.add_argument("--split", type=str, default="train")
    clip_cmd.add_argument("--model-name", type=str, default="sentence-transformers/clip-ViT-B-32")
    clip_cmd.add_argument("--embeddings-dir", type=str, default="embeddings")
    clip_cmd.add_argument("--metadata-csv", type=str, default="metadata/embeddings.csv")
    clip_cmd.add_argument("--metadata-json", type=str, default="metadata/embeddings.json")
    clip_cmd.add_argument("--streaming", action="store_true")
    clip_cmd.add_argument("--max-samples", type=int, default=0)

    npy_cmd = sub.add_parser("metadata-to-npy", help="Pack .pt vectors from metadata into one .npy")
    npy_cmd.add_argument("--metadata-csv", type=str, default="metadata/embeddings.csv")
    npy_cmd.add_argument("--embeddings-dir", type=str, default="embeddings")
    npy_cmd.add_argument("--out-npy", type=str, default="embeddings.npy")
    npy_cmd.add_argument("--out-labels", type=str, default="labels_clean.csv")
    npy_cmd.add_argument("--max-rows", type=int, default=0)
    npy_cmd.add_argument("--id-col", type=str, default="")
    npy_cmd.add_argument("--path-col", type=str, default="")
    npy_cmd.add_argument("--label-col", type=str, default="")
    npy_cmd.add_argument("--max-up", type=int, default=5)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "extract-coco-clip":
        return extract_coco_clip(args)
    if args.command == "metadata-to-npy":
        return metadata_to_npy(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
