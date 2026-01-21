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
from sklearn.neighbors import NearestNeighbors


def find_existing_upwards(rel_path: str, start_dir: Path, max_up: int = 5) -> Path | None:
    rel = Path(rel_path)
    cur = start_dir
    for _ in range(max_up + 1):
        candidate = (cur / rel).resolve()
        if candidate.exists():
            return candidate
        cur = cur.parent
    return None


def resolve_input_path(raw_path: str, cwd: Path, max_up: int = 5) -> Path:
    p = Path(raw_path)
    if p.exists():
        return p.resolve()
    found = find_existing_upwards(raw_path, cwd, max_up=max_up)
    if found is not None:
        return found
    raise FileNotFoundError(f"Could not find path: {raw_path} (searched up to {max_up} parent dirs)")


def load_inputs(embeddings_path: str, labels_path: str, label_col: str, max_up: int):
    cwd = Path.cwd()
    emb_path = resolve_input_path(embeddings_path, cwd=cwd, max_up=max_up)
    labels_path = resolve_input_path(labels_path, cwd=cwd, max_up=max_up)

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
    path = resolve_input_path(noisy_path, Path.cwd(), max_up=max_up)
    df_noisy = pd.read_csv(str(path))
    if label_col not in df_noisy.columns:
        raise ValueError(f"Missing label column '{label_col}' in noisy CSV: {path}")
    if len(df_noisy) != len(df_clean):
        raise ValueError(
            f"Noisy labels rows={len(df_noisy)} does not match clean labels rows={len(df_clean)}: {path}"
        )
    y_noisy = df_noisy[label_col].values
    return y_noisy[keep].astype(str), path


def centroid_distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return 1.0 - float(np.dot(a, b) / denom)
    return float(np.linalg.norm(a - b))


def pick_similar_class_pair(
    X: np.ndarray,
    labels: np.ndarray,
    min_count: int,
    candidate_top_k: int,
    metric: str,
):
    labels = labels.astype(str)
    classes, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    classes = classes[order]
    counts = counts[order]
    if min_count > 0:
        mask = counts >= min_count
        classes = classes[mask]
        counts = counts[mask]
    if candidate_top_k > 0 and len(classes) > candidate_top_k:
        classes = classes[:candidate_top_k]
        counts = counts[:candidate_top_k]
    if len(classes) < 2:
        raise ValueError("Need at least two classes after filtering.")

    centroids = {}
    for cls in classes:
        idx = labels == cls
        centroids[cls] = X[idx].mean(axis=0)

    best_pair = None
    best_dist = None
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            ci = centroids[classes[i]]
            cj = centroids[classes[j]]
            d = centroid_distance(ci, cj, metric)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_pair = (str(classes[i]), str(classes[j]))

    if best_pair is None:
        raise ValueError("Failed to select a class pair.")
    return best_pair, best_dist


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
        counts = counts[mask]
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


def sample_pair_indices(labels: np.ndarray, class_pair: tuple[str, str], args) -> np.ndarray:
    labels = labels.astype(str)
    keep = np.where(np.isin(labels, list(class_pair)))[0]
    if keep.size == 0:
        raise ValueError("No points left after class pair filtering.")

    rng = np.random.RandomState(args.seed)

    if args.max_per_class > 0:
        selected = []
        for cls in class_pair:
            idx = keep[labels[keep] == cls]
            if idx.size > args.max_per_class:
                idx = rng.choice(idx, size=args.max_per_class, replace=False)
            selected.append(idx)
        keep = np.sort(np.concatenate(selected))

    if args.max_points and args.max_points > 0 and keep.size > args.max_points:
        if args.sample == "first":
            keep = keep[: args.max_points]
        else:
            keep = rng.choice(keep, size=args.max_points, replace=False)
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
    palette = ["#e41a1c", "#377eb8"]
    return {class_pair[0]: palette[0], class_pair[1]: palette[1], "Other": "#7f7f7f"}


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


def build_knn(X: np.ndarray, metric: str, k: int):
    k = max(2, min(k, X.shape[0]))
    nn = NearestNeighbors(metric=metric, n_neighbors=k)
    nn.fit(X)
    dists, neigh = nn.kneighbors(X, n_neighbors=k)
    return dists, neigh


def nearest_diff_neighbor(neigh_row: np.ndarray, labels: np.ndarray, i: int):
    yi = labels[i]
    for j in neigh_row:
        if j == i:
            continue
        if labels[j] != yi:
            return int(j)
    return None


def margin_hardness_scores(dists: np.ndarray, neigh: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = y.shape[0]
    scores = np.full(n, np.inf, dtype=np.float32)

    for i in range(n):
        yi = y[i]
        d_same = None
        d_diff = None
        for dist, j in zip(dists[i], neigh[i]):
            if j == i:
                continue
            if y[j] == yi and d_same is None:
                d_same = float(dist)
            if y[j] != yi and d_diff is None:
                d_diff = float(dist)
            if d_same is not None and d_diff is not None:
                break
        if d_same is not None and d_diff is not None:
            scores[i] = d_diff - d_same
    return scores


def inject_nn_exact(X, y, metric, noise_level, random_seed, nn_k):
    n = len(y)
    target = int(np.floor(noise_level * n))
    rng = np.random.RandomState(random_seed)

    _, neigh = build_knn(X, metric=metric, k=nn_k)

    indices = np.arange(n)
    rng.shuffle(indices)

    y_new = y.copy()
    changed = np.zeros(n, dtype=bool)
    changes = 0
    for i in indices:
        if changes >= target:
            break
        j = nearest_diff_neighbor(neigh[i], y_new, i)
        if j is None:
            continue
        new_label = y_new[j]
        if new_label != y_new[i]:
            y_new[i] = new_label
            changed[i] = True
            changes += 1

    return y_new, changed


def inject_boundary_nearest(X, y, metric, noise_level, random_seed, boundary_k, nn_k):
    n = len(y)
    target = int(np.floor(noise_level * n))
    rng = np.random.RandomState(random_seed)

    dists_b, neigh_b = build_knn(X, metric=metric, k=boundary_k)
    scores = margin_hardness_scores(dists_b, neigh_b, y)
    order = np.argsort(scores)
    order = order[np.isfinite(scores[order])]
    if order.size == 0:
        return inject_nn_exact(X, y, metric, noise_level, random_seed, nn_k)

    _, neigh = build_knn(X, metric=metric, k=nn_k)
    y_new = y.copy()
    changed = np.zeros(n, dtype=bool)
    changes = 0

    for i in order:
        if changes >= target:
            break
        j = nearest_diff_neighbor(neigh[i], y_new, i)
        if j is None:
            continue
        new_label = y_new[j]
        if new_label != y_new[i]:
            y_new[i] = new_label
            changed[i] = True
            changes += 1

    if changes < target:
        indices = np.arange(n)
        rng.shuffle(indices)
        for i in indices:
            if changes >= target:
                break
            if changed[i]:
                continue
            j = nearest_diff_neighbor(neigh[i], y_new, i)
            if j is None:
                continue
            new_label = y_new[j]
            if new_label != y_new[i]:
                y_new[i] = new_label
                changed[i] = True
                changes += 1

    return y_new, changed


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s)


def plot_pair(
    Z,
    labels,
    class_order,
    class_colors,
    title,
    ax,
    point_size,
    point_alpha,
    changed_mask=None,
):
    labels = labels.astype(str)
    colors = [class_colors[v] for v in labels]
    ax.scatter(Z[:, 0], Z[:, 1], c=colors, s=point_size, alpha=point_alpha, linewidths=0)
    if changed_mask is not None and changed_mask.any():
        ax.scatter(
            Z[changed_mask, 0], Z[changed_mask, 1],
            marker="x", c="black", s=point_size + 6, linewidths=1.0
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6,
               markerfacecolor=class_colors[c], markeredgecolor="none", label=c)
        for c in class_order
    ]
    if changed_mask is not None and changed_mask.any():
        handles.append(
            Line2D([0], [0], marker="x", linestyle="None", markersize=7,
                   markeredgecolor="black", label="Changed")
        )
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, default="alec/embeddings.npy")
    ap.add_argument("--labels-clean", type=str, default="alec/labels_clean.csv")
    ap.add_argument("--label-col", type=str, default="label")

    ap.add_argument("--out-dir", type=str, default="alec/plots")
    ap.add_argument("--out-name", type=str, default="")

    ap.add_argument("--classes", nargs=2, default=[],
                    help="Optional pair of classes to visualize (exact match).")
    ap.add_argument("--min-class-count", type=int, default=200)
    ap.add_argument("--candidate-top-k", type=int, default=50)
    ap.add_argument("--target-centroid-dist", type=float, default=None,
                    help="Pick a class pair whose centroid distance is closest to this value.")
    ap.add_argument("--class-metric", type=str, default="cosine",
                    choices=["cosine", "euclidean"])

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-points", type=int, default=10000)
    ap.add_argument("--max-per-class", type=int, default=0)
    ap.add_argument("--sample", type=str, choices=["first", "random"], default="random")

    ap.add_argument("--noisy-random", type=str, default="")
    ap.add_argument("--noisy-border", type=str, default="")
    ap.add_argument("--noisy-cluster", type=str, default="")

    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--perplexity", type=int, default=30)
    ap.add_argument("--n-iter", type=int, default=1500)
    ap.add_argument("--point-size", type=float, default=6.0)
    ap.add_argument("--point-alpha", type=float, default=0.7)
    ap.add_argument("--recompute-tsne", action="store_true",
                    help="Accepted for compatibility; t-SNE is always recomputed in this script.")
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
            target_dist=args.target_centroid_dist
        )

    keep = sample_pair_indices(y_full, class_pair, args)
    Xs = X[keep]
    y_clean = y_full[keep].astype(str)

    Z = compute_tsne_coords(
        Xs, seed=args.seed, pca_dim=args.pca_dim,
        perplexity=args.perplexity, n_iter=args.n_iter
    )

    class_colors = build_class_colors(class_pair)
    y_clean_mapped, _ = map_labels_to_pair(y_clean, class_pair)

    noise_inputs = {
        "random": args.noisy_random,
        "border": args.noisy_border,
        "cluster": args.noisy_cluster,
    }
    noise_inputs = {k: v for k, v in noise_inputs.items() if v}
    if not noise_inputs:
        raise ValueError("Provide at least one of --noisy-random/--noisy-border/--noisy-cluster.")

    xmin, xmax = float(Z[:, 0].min()), float(Z[:, 0].max())
    ymin, ymax = float(Z[:, 1].min()), float(Z[:, 1].max())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for noise_type, noisy_path in noise_inputs.items():
        y_noisy, resolved_path = load_noisy_labels(
            noisy_path, args.label_col, df_clean, keep, max_up=args.max_up
        )
        changed_mask = y_noisy != y_clean
        changed_pct = 100.0 * (changed_mask.sum() / max(1, len(changed_mask)))

        y_noisy_mapped, has_other = map_labels_to_pair(y_noisy, class_pair)
        class_order = list(class_pair)
        if has_other:
            class_order.append("Other")

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        plot_pair(
            Z, y_clean_mapped, class_order[:2], class_colors,
            "t-SNE (Clean Labels)", axes[0],
            point_size=args.point_size, point_alpha=args.point_alpha
        )
        plot_pair(
            Z, y_noisy_mapped, class_order, class_colors,
            f"t-SNE (Corrupted Labels)", axes[1],
            point_size=args.point_size, point_alpha=args.point_alpha,
            changed_mask=changed_mask
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
        fig.savefig(str(out_path), dpi=200)
        plt.close(fig)

        print(f"[viz] noise={noise_type} path={resolved_path}")
        print(f"[viz] changed={changed_mask.sum()} ({changed_pct:.1f}%)")
        print(f"[viz] wrote: {out_path}")

    print(f"[viz] selected classes: {class_pair[0]}, {class_pair[1]}")
    if pair_dist is not None:
        print(f"[viz] centroid distance ({args.class_metric}): {pair_dist:.4f}")


if __name__ == "__main__":
    main()
