#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def find_existing_upwards(rel_path: str, start_dir: Path, max_up: int = 5) -> Path | None:
    rel = Path(rel_path)
    cur = start_dir
    for _ in range(max_up + 1):
        candidate = (cur / rel).resolve()
        if candidate.exists():
            return candidate
        cur = cur.parent
    return None


def resolve_input_path(raw_path: str, cwd: Path, max_up: int = 5, base_dir: Path | None = None) -> Path:
    p = Path(raw_path)
    if base_dir is not None and not p.is_absolute():
        cand = (base_dir / p).resolve()
        if cand.exists():
            return cand
    if p.exists():
        return p.resolve()
    found = find_existing_upwards(raw_path, cwd, max_up=max_up)
    if found is not None:
        return found
    raise FileNotFoundError(f"Could not find path: {raw_path} (searched up to {max_up} parent dirs)")


def load_inputs(embeddings_path: str, labels_path: str, label_col: str, max_up: int):
    cwd = Path.cwd()
    repo_root = Path(__file__).resolve().parents[1]
    emb_path = resolve_input_path(embeddings_path, cwd=cwd, max_up=max_up, base_dir=repo_root)
    labels_path = resolve_input_path(labels_path, cwd=cwd, max_up=max_up, base_dir=repo_root)

    X = np.load(str(emb_path))
    df = pd.read_csv(str(labels_path))

    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' in labels CSV.")
    if X.shape[0] != len(df):
        raise ValueError(f"Row mismatch: embeddings rows={X.shape[0]} vs labels rows={len(df)}")

    return X, df


def load_noisy_labels(
    noisy_path: str,
    label_col: str,
    df_clean: pd.DataFrame,
    keep: np.ndarray,
    max_up: int,
):
    path = resolve_input_path(
        noisy_path,
        Path.cwd(),
        max_up=max_up,
        base_dir=Path(__file__).resolve().parents[1],
    )
    df_noisy = pd.read_csv(str(path))
    if label_col not in df_noisy.columns:
        raise ValueError(f"Missing label column '{label_col}' in noisy CSV: {path}")
    if len(df_noisy) != len(df_clean):
        raise ValueError(
            f"Noisy labels rows={len(df_noisy)} does not match clean labels rows={len(df_clean)}: {path}"
        )
    y_noisy = df_noisy[label_col].values
    return y_noisy[keep].astype(str), path


def load_reasons_for_subset(log_path: str, keep_global_idx: np.ndarray, max_up: int) -> tuple[np.ndarray, Path]:
    path = resolve_input_path(
        log_path,
        Path.cwd(),
        max_up=max_up,
        base_dir=Path(__file__).resolve().parents[1],
    )
    df_log = pd.read_csv(str(path))
    if "index" not in df_log.columns or "reason" not in df_log.columns:
        raise ValueError(f"Log file missing required columns 'index'/'reason': {path}")

    reason_map = dict(zip(df_log["index"].astype(int).tolist(), df_log["reason"].astype(str).tolist()))
    reasons = np.array([reason_map.get(int(i), "") for i in keep_global_idx], dtype=object)
    return reasons, path


def centroid_distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return 1.0 - float(np.dot(a, b) / denom)
    return float(np.linalg.norm(a - b))


def pick_class_pair_by_distance(
    X: np.ndarray,
    labels: np.ndarray,
    min_count: int,
    candidate_top_k: int,
    metric: str,
    target_dist: float,
):
    labels = labels.astype(str)
    classes, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    classes = classes[order]
    counts = counts[order]
    if min_count > 0:
        mask = counts >= min_count
        classes = classes[mask]
    if candidate_top_k > 0 and len(classes) > candidate_top_k:
        classes = classes[:candidate_top_k]
    if len(classes) < 2:
        raise ValueError("Need at least two classes after filtering.")

    centroids = {}
    for cls in classes:
        idx = labels == cls
        centroids[cls] = X[idx].mean(axis=0)

    best_pair = None
    best_dist = None
    best_gap = None
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            ci = centroids[classes[i]]
            cj = centroids[classes[j]]
            d = centroid_distance(ci, cj, metric)
            gap = abs(d - target_dist)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_dist = d
                best_pair = (str(classes[i]), str(classes[j]))

    if best_pair is None:
        raise ValueError("Failed to select a class pair.")
    return best_pair, best_dist


def resolve_class_pair(requested: tuple[str, str], labels: np.ndarray) -> tuple[str, str]:
    labels_str = labels.astype(str)
    unique = pd.unique(labels_str)
    lower_map = {}
    for lab in unique:
        key = lab.lower()
        if key not in lower_map:
            lower_map[key] = lab

    resolved = []
    missing = []
    for req in requested:
        key = str(req).lower()
        if key in lower_map:
            resolved.append(lower_map[key])
        else:
            missing.append(req)

    if missing:
        sample = ", ".join(list(unique)[:10])
        raise ValueError(
            f"Requested class(es) not found: {', '.join(missing)}. "
            f"Available labels look like: {sample}"
        )

    return (resolved[0], resolved[1])


def sample_pair_indices(labels: np.ndarray, class_pair: tuple[str, str], max_points: int, seed: int) -> np.ndarray:
    labels = labels.astype(str)
    keep = np.where(np.isin(labels, list(class_pair)))[0]
    if keep.size == 0:
        raise ValueError("No points left after class pair filtering.")

    rng = np.random.RandomState(seed)
    if max_points > 0 and keep.size > max_points:
        keep = rng.choice(keep, size=max_points, replace=False)
        keep = np.sort(keep)
    return keep


def compute_tsne_coords(X, seed=42, pca_dim=50, perplexity=30, n_iter=1500):
    n = X.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for t-SNE.")

    pca_dim = min(pca_dim, X.shape[1], max(2, n - 1))
    Xp = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)

    max_perp = max(2, (n - 1) // 3)
    perplexity = min(perplexity, max_perp, n - 1)

    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    if "max_iter" in TSNE.__init__.__code__.co_varnames:
        kwargs["max_iter"] = n_iter
    else:
        kwargs["n_iter"] = n_iter

    return TSNE(**kwargs).fit_transform(Xp)


def build_class_colors(class_pair: tuple[str, str]):
    return {class_pair[0]: "#d62728", class_pair[1]: "#1f77b4", "Other": "#7f7f7f"}


def map_labels_to_pair(labels: np.ndarray, class_pair: tuple[str, str]) -> tuple[np.ndarray, bool]:
    labels = labels.astype(str)
    mapped = []
    has_other = False
    for lab in labels:
        if lab in class_pair:
            mapped.append(lab)
        else:
            mapped.append("Other")
            has_other = True
    return np.array(mapped, dtype=object), has_other


def build_overlay_specs(noise_type: str, changed_mask: np.ndarray, reasons: np.ndarray | None):
    overlays = []

    if reasons is None:
        overlays.append(dict(mask=changed_mask, marker="x", color="black", size=11, lw=1.1, label="Changed"))
        return overlays

    if noise_type == "random":
        overlays.append(dict(mask=changed_mask, marker="x", color="black", size=11, lw=1.1, label="Random flip"))
        return overlays

    if noise_type == "border":
        m_border = (reasons == "border") & changed_mask
        m_fill = (reasons == "border_fill_nn") & changed_mask
        m_other = changed_mask & ~(m_border | m_fill)
        overlays.append(dict(mask=m_border, marker="x", color="#ff7f00", size=11, lw=1.1, label="Boundary-seeded"))
        overlays.append(dict(mask=m_fill, marker="x", color="black", size=11, lw=1.1, label="Fill"))
        if m_other.any():
            overlays.append(dict(mask=m_other, marker="x", color="#9467bd", size=11, lw=1.1, label="Other changed"))
        return overlays

    # cluster
    seed_reason = np.array([r if str(r).startswith("cluster_seed_") else "" for r in reasons], dtype=object)
    changed_seed = seed_reason[changed_mask]
    changed_seed = changed_seed[changed_seed != ""]

    if changed_seed.size == 0:
        overlays.append(dict(mask=changed_mask, marker="x", color="black", size=11, lw=1.1, label="Changed"))
        return overlays

    seed_counts = pd.Series(changed_seed).value_counts()
    top = seed_counts.head(4).index.tolist()
    palette = ["#2ca02c", "#e377c2", "#17becf", "#bcbd22"]

    claimed = np.zeros_like(changed_mask, dtype=bool)
    for i, seed in enumerate(top):
        m = (seed_reason == seed) & changed_mask
        claimed |= m
        seed_id = seed.replace("cluster_seed_", "")
        overlays.append(
            dict(mask=m, marker="x", color=palette[i % len(palette)], size=11, lw=1.1,
                 label=f"Cluster {seed_id} ({int(m.sum())})")
        )

    remainder = changed_mask & ~claimed
    if remainder.any():
        overlays.append(dict(mask=remainder, marker="x", color="black", size=11, lw=1.1, label="Other changed"))

    return overlays


def plot_pair(
    Z,
    labels,
    class_order,
    class_colors,
    title,
    ax,
    point_size,
    point_alpha,
    overlays=None,
):
    labels = labels.astype(str)
    colors = [class_colors[v] for v in labels]
    ax.scatter(Z[:, 0], Z[:, 1], c=colors, s=point_size, alpha=point_alpha, linewidths=0)

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6,
               markerfacecolor=class_colors[c], markeredgecolor="none", label=c)
        for c in class_order
    ]

    if overlays:
        for ov in overlays:
            m = ov["mask"]
            if not m.any():
                continue
            ax.scatter(
                Z[m, 0], Z[m, 1], marker=ov["marker"], c=ov["color"],
                s=point_size + ov["size"], linewidths=ov["lw"]
            )
            handles.append(
                Line2D([0], [0], marker=ov["marker"], linestyle="None", markersize=7,
                       markeredgecolor=ov["color"], label=ov["label"])
            )

    ax.set_title(title)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, default="embeddings_10k.npy")
    ap.add_argument("--labels-clean", type=str, default="labels/labels_clean_10k.csv")
    ap.add_argument("--label-col", type=str, default="label")

    ap.add_argument("--out-dir", type=str, default="plots")
    ap.add_argument("--out-name", type=str, default="")

    ap.add_argument("--classes", nargs=2, default=["2", "39"])
    ap.add_argument("--target-centroid-dist", type=float, default=None)
    ap.add_argument("--class-metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    ap.add_argument("--min-class-count", type=int, default=50)
    ap.add_argument("--candidate-top-k", type=int, default=50)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-points", type=int, default=5000)

    ap.add_argument("--noisy-random", type=str, default="")
    ap.add_argument("--noisy-border", type=str, default="")
    ap.add_argument("--noisy-cluster", type=str, default="")
    ap.add_argument("--log-random", type=str, default="")
    ap.add_argument("--log-border", type=str, default="")
    ap.add_argument("--log-cluster", type=str, default="")

    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--perplexity", type=int, default=30)
    ap.add_argument("--n-iter", type=int, default=1500)
    ap.add_argument("--point-size", type=float, default=7.0)
    ap.add_argument("--point-alpha", type=float, default=0.75)
    ap.add_argument("--max-up", type=int, default=5)
    args = ap.parse_args()

    X, df_clean = load_inputs(args.embeddings, args.labels_clean, args.label_col, max_up=args.max_up)
    y_full = df_clean[args.label_col].values

    if args.classes:
        class_pair = resolve_class_pair((args.classes[0], args.classes[1]), y_full)
        pair_dist = None
    else:
        if args.target_centroid_dist is None:
            raise ValueError("--target-centroid-dist is required when --classes is not provided.")
        class_pair, pair_dist = pick_class_pair_by_distance(
            X, y_full, min_count=args.min_class_count,
            candidate_top_k=args.candidate_top_k, metric=args.class_metric,
            target_dist=args.target_centroid_dist,
        )

    keep = sample_pair_indices(y_full, class_pair, max_points=args.max_points, seed=args.seed)
    Xs = X[keep]
    y_clean = y_full[keep].astype(str)

    Z = compute_tsne_coords(
        Xs, seed=args.seed, pca_dim=args.pca_dim,
        perplexity=args.perplexity, n_iter=args.n_iter,
    )

    class_colors = build_class_colors(class_pair)
    y_clean_mapped, _ = map_labels_to_pair(y_clean, class_pair)

    noise_inputs = {
        "random": (args.noisy_random, args.log_random),
        "border": (args.noisy_border, args.log_border),
        "cluster": (args.noisy_cluster, args.log_cluster),
    }
    noise_inputs = {k: v for k, v in noise_inputs.items() if v[0]}
    if not noise_inputs:
        raise ValueError("Provide at least one of --noisy-random/--noisy-border/--noisy-cluster.")

    xmin, xmax = float(Z[:, 0].min()), float(Z[:, 0].max())
    ymin, ymax = float(Z[:, 1].min()), float(Z[:, 1].max())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for noise_type, (noisy_path, log_path) in noise_inputs.items():
        y_noisy, resolved_noisy_path = load_noisy_labels(
            noisy_path, args.label_col, df_clean, keep, max_up=args.max_up
        )
        changed_mask = y_noisy != y_clean
        changed_pct = 100.0 * (changed_mask.sum() / max(1, len(changed_mask)))

        reasons = None
        if log_path:
            reasons, resolved_log_path = load_reasons_for_subset(log_path, keep, max_up=args.max_up)
        else:
            resolved_log_path = None

        y_noisy_mapped, has_other = map_labels_to_pair(y_noisy, class_pair)
        class_order = list(class_pair)
        if has_other:
            class_order.append("Other")

        overlays = build_overlay_specs(noise_type, changed_mask, reasons)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        plot_pair(
            Z, y_clean_mapped, class_order[:2], class_colors,
            "t-SNE (Clean Labels)", axes[0],
            point_size=args.point_size, point_alpha=args.point_alpha,
            overlays=None,
        )
        plot_pair(
            Z, y_noisy_mapped, class_order, class_colors,
            "t-SNE (Corrupted Labels)", axes[1],
            point_size=args.point_size, point_alpha=args.point_alpha,
            overlays=overlays,
        )

        for ax in axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        title = f"Noise: {noise_type} | changed={changed_pct:.1f}% | {class_pair[0]} vs {class_pair[1]}"
        if pair_dist is not None:
            title += f" (centroid dist={pair_dist:.4f})"
        fig.suptitle(title)

        fig.tight_layout()
        if args.out_name:
            base = Path(args.out_name)
            if base.suffix:
                out_name = f"{base.stem}_{noise_type}{base.suffix}"
            else:
                out_name = f"{base.name}_{noise_type}.png"
        else:
            out_name = f"tsne_{noise_type}_{safe_name(class_pair[0])}_{safe_name(class_pair[1])}.png"
        out_path = out_dir / out_name
        fig.savefig(str(out_path), dpi=220)
        plt.close(fig)

        print(f"[viz] noise={noise_type} labels={resolved_noisy_path}")
        if resolved_log_path is not None:
            print(f"[viz] noise={noise_type} log={resolved_log_path}")
        print(f"[viz] changed={changed_mask.sum()} ({changed_pct:.1f}%)")
        print(f"[viz] wrote: {out_path}")

    print(f"[viz] selected classes: {class_pair[0]}, {class_pair[1]}")


if __name__ == "__main__":
    main()
