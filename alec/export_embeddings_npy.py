#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path


def infer_columns(df: pd.DataFrame):
    path_candidates = ["vector_path", "path", "embedding_path", "pt_path", "file_path"]
    label_candidates = ["label", "category", "category_id", "class", "class_id"]
    id_candidates = ["vector_id", "id", "embedding_id", "uuid"]

    path_col = next((c for c in path_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)
    id_col = next((c for c in id_candidates if c in df.columns), None)

    return id_col, path_col, label_col


def find_existing_upwards(rel_path: str, start_dir: Path, max_up: int = 5) -> Path | None:
    """
    Try rel_path in start_dir, then in start_dir/.., start_dir/../.., etc.
    Returns the first Path that exists, else None.
    """
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

    emb = emb.detach().cpu().float().view(-1).numpy()
    return emb


def resolve_vector_path(raw_path: str, metadata_parent: Path, embeddings_dir: Path | None) -> Path:
    """
    Resolve a vector_path like 'embeddings/<uuid>.pt' robustly.

    Tries, in order:
    1) raw_path as absolute or relative to cwd
    2) metadata_parent / raw_path
    3) embeddings_dir / basename(raw_path) (if embeddings_dir provided)
    4) embeddings_dir / raw_path (if embeddings_dir provided)
    5) metadata_parent / 'embeddings' / basename(raw_path)
    """
    p = Path(str(raw_path))

    # 1) Absolute or relative to current working directory
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()

    # 2) Relative to metadata's parent directory (important for your case)
    cand = (metadata_parent / p).resolve()
    if cand.exists():
        return cand

    # 3/4) Relative to provided embeddings_dir
    if embeddings_dir is not None:
        cand = (embeddings_dir / p.name).resolve()
        if cand.exists():
            return cand
        cand = (embeddings_dir / p).resolve()
        if cand.exists():
            return cand

    # 5) Common layout: metadata_parent/embeddings/<file>.pt
    cand = (metadata_parent / "embeddings" / p.name).resolve()
    if cand.exists():
        return cand

    raise FileNotFoundError(
        f"Could not resolve embedding path: '{raw_path}'. "
        f"Tried cwd, metadata_parent='{metadata_parent}', embeddings_dir='{embeddings_dir}'."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-csv", type=str, default="metadata/embeddings.csv")
    ap.add_argument("--embeddings-dir", type=str, default="embeddings")
    ap.add_argument("--out-npy", type=str, default="alec/embeddings.npy")
    ap.add_argument("--out-labels", type=str, default="alec/labels_clean.csv")
    ap.add_argument("--max-rows", type=int, default=10000, help="Cap rows for TSNE friendliness")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--id-col", type=str, default="", help="Optional override")
    ap.add_argument("--path-col", type=str, default="", help="Optional override")
    ap.add_argument("--label-col", type=str, default="", help="Optional override")
    ap.add_argument("--max-up", type=int, default=5, help="How many parent dirs to search upward")
    args = ap.parse_args()

    cwd = Path.cwd()

    # --- Locate metadata CSV robustly ---
    meta_path = Path(args.metadata_csv)
    if not meta_path.exists():
        found = find_existing_upwards(args.metadata_csv, cwd, max_up=args.max_up)
        if found is None:
            # Special-case: common repo layout where metadata is one level above project dir
            found = find_existing_upwards("metadata/embeddings.csv", cwd, max_up=args.max_up)
        if found is None:
            raise FileNotFoundError(f"metadata CSV not found: {args.metadata_csv} (searched up to {args.max_up} levels)")
        meta_path = found

    metadata_parent = meta_path.parent.resolve()

    # --- Locate embeddings dir robustly ---
    emb_dir = Path(args.embeddings_dir)
    if not emb_dir.exists():
        found_dir = find_existing_upwards(args.embeddings_dir, cwd, max_up=args.max_up)
        if found_dir is None:
            # Try sibling of metadata folder: <capstone_root>/embeddings
            sibling = (metadata_parent.parent / "embeddings").resolve()
            if sibling.exists():
                found_dir = sibling
        if found_dir is None:
            # Fall back to None; resolve_vector_path can still use metadata_parent/embeddings
            emb_dir = None
        else:
            emb_dir = found_dir
    else:
        emb_dir = emb_dir.resolve()

    df = pd.read_csv(str(meta_path))
    id_col, path_col, label_col = infer_columns(df)

    if args.id_col:
        id_col = args.id_col
    if args.path_col:
        path_col = args.path_col
    if args.label_col:
        label_col = args.label_col

    if path_col is None or label_col is None:
        raise ValueError(
            f"Could not infer required columns. Found: {df.columns.tolist()}. "
            f"Need a path column and a label column. Use --path-col / --label-col to override."
        )

    # Deterministic subset: take first max_rows
    df = df.reset_index(drop=True)
    if len(df) > args.max_rows:
        df = df.iloc[:args.max_rows].reset_index(drop=True)

    # Resolve paths and load embeddings
    resolved_paths: list[Path] = []
    for raw in df[path_col].tolist():
        resolved_paths.append(resolve_vector_path(raw, metadata_parent=metadata_parent, embeddings_dir=emb_dir))

    embs = []
    dim = None
    for i, p in enumerate(resolved_paths):
        e = load_pt_embedding(p)
        if dim is None:
            dim = e.shape[0]
        elif e.shape[0] != dim:
            raise ValueError(f"Embedding dim mismatch at row {i}: got {e.shape[0]}, expected {dim}")
        embs.append(e)

    X = np.vstack(embs).astype(np.float32)  # [N, D]
    out_npy = Path(args.out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_npy), X)

    out = pd.DataFrame()
    if id_col is not None and id_col in df.columns:
        out["vector_id"] = df[id_col].astype(str).values
    else:
        out["vector_id"] = [f"row_{i}" for i in range(len(df))]

    out["label"] = df[label_col].values
    out["vector_path"] = [str(p) for p in resolved_paths]
    # Optional columns for debugging
    for extra in ["image_id", "object_id"]:
        if extra in df.columns:
            out[extra] = df[extra].values

    out_labels = Path(args.out_labels)
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(str(out_labels), index=False)

    print("Export complete")
    print(f"  metadata_csv: {meta_path}")
    print(f"  embeddings_dir: {emb_dir if emb_dir is not None else '(inferred via metadata parent)'}")
    print(f"  embeddings.npy: {out_npy}  shape={X.shape}")
    print(f"  labels_clean.csv: {out_labels} rows={len(out)}")
    print(f"  columns used: id={id_col}, path={path_col}, label={label_col}")


if __name__ == "__main__":
    main()
