#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def compute_tsne(X, seed=0, pca_dim=50, perplexity=30, n_iter=1500):
    pca_dim = min(pca_dim, X.shape[1], max(2, X.shape[0] - 1))
    Xp = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    max_perp = max(2, (X.shape[0] - 1) // 3)
    perplexity = min(perplexity, max_perp, X.shape[0] - 1)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, default="embeddings_10k.npy")
    ap.add_argument("--labels", type=str, default="labels/labels_clean_10k.csv")
    ap.add_argument("--log", type=str, default="logs/coco_log_cluster_10k_10.csv")
    ap.add_argument("--out", type=str, default="plots/issue_cluster_incoherence_10k.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-bg", type=int, default=4000)
    ap.add_argument("--top-seeds", type=int, default=3)
    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--perplexity", type=int, default=30)
    ap.add_argument("--n-iter", type=int, default=1500)
    args = ap.parse_args()

    X = np.load(args.embeddings)
    df = pd.read_csv(args.labels)
    labels = df["label"].astype(str).values

    log = pd.read_csv(args.log)
    log = log[log["reason"].astype(str).str.startswith("cluster_seed_")].copy()
    if log.empty:
        raise ValueError("No cluster_seed_* entries found in log.")

    log["seed"] = log["reason"].str.replace("cluster_seed_", "", regex=False).astype(int)
    top_seeds = log["seed"].value_counts().head(args.top_seeds).index.tolist()
    log = log[log["seed"].isin(top_seeds)]
    changed_idx = log["index"].astype(int).values

    rng = np.random.RandomState(args.seed)
    bg_n = min(args.max_bg, len(labels))
    bg_idx = rng.choice(len(labels), size=bg_n, replace=False)

    idx = np.unique(np.concatenate([bg_idx, changed_idx]))
    Xsub = X[idx]

    Z = compute_tsne(
        Xsub, seed=args.seed, pca_dim=args.pca_dim,
        perplexity=args.perplexity, n_iter=args.n_iter
    )

    pos = {orig: i for i, orig in enumerate(idx)}
    changed_pos = [pos[i] for i in changed_idx if i in pos]
    changed_pos = np.array(changed_pos, dtype=int)

    log_idx = log.set_index("index")
    new_labels = [str(log_idx.loc[i, "new_label"]) for i in changed_idx if i in pos]

    plt.figure(figsize=(10, 8))
    plt.scatter(Z[:, 0], Z[:, 1], s=4, c="#d0d0d0", alpha=0.4, linewidths=0)

    uniq = sorted(set(new_labels))
    palette = plt.cm.tab10(np.linspace(0, 1, max(3, len(uniq))))
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(uniq)}
    colors = [color_map[lab] for lab in new_labels]

    plt.scatter(
        Z[changed_pos, 0], Z[changed_pos, 1],
        s=18, c=colors, edgecolors="black", linewidths=0.2, alpha=0.9
    )

    handles = []
    for lab in uniq[:10]:
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=color_map[lab], markersize=6, label=f"new={lab}")
        )
    plt.legend(handles=handles, title="New labels (sample)", loc="upper right", fontsize=8)

    plt.title("Cluster noise: changed points colored by new label")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
