#!/usr/bin/env python3
import os
import argparse
import inspect
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


def compute_boundary_points(X, y, k=10, metric="cosine"):
    n = X.shape[0]
    k = max(2, min(k, n))
    nn = NearestNeighbors(metric=metric, n_neighbors=k)
    nn.fit(X)
    _, neigh = nn.kneighbors(X, n_neighbors=k)

    boundary = np.zeros(n, dtype=bool)
    for i in range(n):
        neigh_i = neigh[i]
        neigh_i = neigh_i[neigh_i != i]
        if neigh_i.size == 0:
            continue
        labs = y[neigh_i]
        if len(set(labs.tolist())) > 1:
            boundary[i] = True
    return boundary


def compute_tsne_coords(X, seed=42, pca_dim=50, perplexity=30, n_iter=1500):
    pca_dim = min(pca_dim, X.shape[1])
    Xp = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)

    sig = inspect.signature(TSNE.__init__)
    params = sig.parameters

    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    if "max_iter" in params:
        kwargs["max_iter"] = n_iter
    else:
        kwargs["n_iter"] = n_iter

    Z = TSNE(**kwargs).fit_transform(Xp)
    return Z


def make_topk_label_mapping(y, top_k=10):
    """
    Returns:
      y_plot: labels with infrequent classes mapped to "Other"
      classes: ordered list of legend class names (top_k + optional "Other")
    """
    s = pd.Series(y)
    counts = s.value_counts()
    top = counts.index[:top_k].tolist()
    y_plot = np.array([v if v in top else "Other" for v in y], dtype=object)

    classes = [str(v) for v in top]
    if "Other" in y_plot:
        classes.append("Other")
    return y_plot, classes


def save_plot_with_legend(
    Z,
    y_clean,
    changed_mask,
    out_path,
    title,
    boundary_mask=None,
    top_k_legend=10,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Reduce legend complexity: top-K classes + Other
    y_plot, legend_classes = make_topk_label_mapping(y_clean, top_k=top_k_legend)

    # Assign color indices
    uniq = list(dict.fromkeys(legend_classes))  # preserve order
    mapping = {lab: i for i, lab in enumerate(uniq)}
    c = np.array([mapping[str(v)] for v in y_plot], dtype=int)

    plt.figure(figsize=(11, 8))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=c, s=8, alpha=0.85)

    # boundary overlay (hollow circles)
    if boundary_mask is not None:
        b = boundary_mask.astype(bool)
        plt.scatter(Z[b, 0], Z[b, 1], s=22, facecolors="none", linewidths=1.0)

    # changed labels overlay (X)
    m = changed_mask.astype(bool)
    plt.scatter(Z[m, 0], Z[m, 1], s=45, marker="x", linewidths=1.8)

    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")

    # Build legend handles
    handles = []
    # class handles: use proxy artists
    for lab in uniq:
        idx = mapping[lab]
        handles.append(Line2D([0], [0], marker='o', linestyle='None', markersize=6,
                              label=f"Class {lab}"))

    # add overlays to legend
    handles.append(Line2D([0], [0], marker='x', linestyle='None', markersize=7, label="Changed label"))
    if boundary_mask is not None:
        handles.append(Line2D([0], [0], marker='o', linestyle='None', markersize=7,
                              markerfacecolor='none', label="Boundary pool"))

    plt.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    if not os.path.exists(out_path):
        raise RuntimeError(f"Plot not written: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, required=True)
    ap.add_argument("--labels-clean", type=str, required=True)
    ap.add_argument("--label-col", type=str, default="label")

    ap.add_argument("--out-dir", type=str, default="plots")
    ap.add_argument("--variants", nargs="+", default=[])

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-points", type=int, default=10000)

    ap.add_argument("--recompute-tsne", action="store_true")
    ap.add_argument("--tsne-coords", type=str, default="tsne_coords.npy")
    ap.add_argument("--keep-idx", type=str, default="tsne_keep_idx.npy")

    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--perplexity", type=int, default=30)
    ap.add_argument("--n-iter", type=int, default=1500)

    ap.add_argument("--metric", type=str, default="cosine")
    ap.add_argument("--show-boundary", action="store_true")
    ap.add_argument("--boundary-k", type=int, default=10)

    ap.add_argument("--legend-top-k", type=int, default=10)
    args = ap.parse_args()

    print("[viz] cwd:", os.getcwd())
    print("[viz] out_dir:", args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    X = np.load(args.embeddings)
    df_clean = pd.read_csv(args.labels_clean)
    y_clean_full = df_clean[args.label_col].values

    if X.shape[0] != len(df_clean):
        raise ValueError(f"Row mismatch: embeddings rows={X.shape[0]} vs labels rows={len(df_clean)}")

    n = X.shape[0]
    rng = np.random.RandomState(args.seed)
    if n > args.max_points:
        keep = rng.choice(n, size=args.max_points, replace=False)
        keep.sort()
    else:
        keep = np.arange(n)

    Xs = X[keep]
    ys = y_clean_full[keep]

    need_recompute = args.recompute_tsne or (not os.path.exists(args.tsne_coords)) or (not os.path.exists(args.keep_idx))
    if need_recompute:
        print(f"[viz] computing t-SNE on {Xs.shape[0]} points (D={Xs.shape[1]}) ...")
        Z = compute_tsne_coords(Xs, seed=args.seed, pca_dim=args.pca_dim,
                                perplexity=args.perplexity, n_iter=args.n_iter)
        np.save(args.tsne_coords, Z)
        np.save(args.keep_idx, keep)
        print(f"[viz] saved coords: {args.tsne_coords}")
    else:
        Z = np.load(args.tsne_coords)
        keep_loaded = np.load(args.keep_idx)
        if keep_loaded.shape[0] != keep.shape[0] or not np.all(keep_loaded == keep):
            print("[viz] keep idx mismatch; forcing recompute.")
            Z = compute_tsne_coords(Xs, seed=args.seed, pca_dim=args.pca_dim,
                                    perplexity=args.perplexity, n_iter=args.n_iter)
            np.save(args.tsne_coords, Z)
            np.save(args.keep_idx, keep)

    boundary_mask = None
    if args.show_boundary:
        boundary_mask = compute_boundary_points(Xs, ys, k=args.boundary_k, metric=args.metric)
        print(f"[viz] boundary points: {int(boundary_mask.sum())} / {len(boundary_mask)}")

    # Clean plot
    clean_path = os.path.join(args.out_dir, "tsne_clean.png")
    save_plot_with_legend(
        Z, ys, changed_mask=np.zeros_like(ys, dtype=bool),
        boundary_mask=boundary_mask,
        out_path=clean_path,
        title="t-SNE (colored by clean label) | no corruption",
        top_k_legend=args.legend_top_k,
    )
    print("[viz] saved:", clean_path)

    # Auto variants if none provided
    if not args.variants:
        candidates = [
            "labels_nn_10.csv",
            "labels_cluster_k5_10.csv",
            "labels_boundary_cluster_k5_10.csv",
        ]
        args.variants = [c for c in candidates if os.path.exists(c)]
        print("[viz] auto variants:", args.variants)

    for var_path in args.variants:
        if not os.path.exists(var_path):
            print("[viz] skipping missing:", var_path)
            continue

        df_var = pd.read_csv(var_path)
        if len(df_var) != len(df_clean):
            raise ValueError(f"Variant {var_path} rows={len(df_var)} does not match labels_clean rows={len(df_clean)}")

        y_var_full = df_var[args.label_col].values
        y_var_s = y_var_full[keep]
        changed = (ys != y_var_s)
        n_changed = int(changed.sum())

        base = os.path.splitext(os.path.basename(var_path))[0]
        out_path = os.path.join(args.out_dir, f"tsne_{base}.png")
        save_plot_with_legend(
            Z, ys, changed_mask=changed,
            boundary_mask=boundary_mask,
            out_path=out_path,
            title=f"t-SNE (colored by clean label) | X = changed labels ({n_changed}) | {base}",
            top_k_legend=args.legend_top_k,
        )
        print("[viz] saved:", out_path)

    print("[viz] done.")


if __name__ == "__main__":
    main()