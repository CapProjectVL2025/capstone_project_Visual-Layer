#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors




def resolve_repo_path(raw_path: str, repo_root: Path) -> str:
    if not raw_path:
        return raw_path
    p = Path(raw_path)
    if p.is_absolute():
        return str(p)
    return str((repo_root / p).resolve())


def load_inputs(embeddings_path: str, labels_path: str, id_col: str, label_col: str, require_embeddings: bool):
    repo_root = Path(__file__).resolve().parents[1]
    labels_path = resolve_repo_path(labels_path, repo_root)
    embeddings_path = resolve_repo_path(embeddings_path, repo_root)
    df = pd.read_csv(labels_path)

    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' in labels CSV.")

    X = None
    if require_embeddings:
        X = np.load(embeddings_path)
        if X.shape[0] != len(df):
            raise ValueError(
                f"Row mismatch: embeddings rows={X.shape[0]} vs labels rows={len(df)}"
            )

    if id_col not in df.columns:
        raise ValueError(f"Missing id column '{id_col}' in labels CSV.")

    ids = df[id_col].astype(str).values
    y = df[label_col].values
    return X, df, ids, y


def target_num_changes(n: int, noise_level: float) -> int:
    if noise_level < 0 or noise_level > 1:
        raise ValueError("noise_level must be in [0, 1]")
    return int(np.floor(noise_level * n))


def build_knn(X: np.ndarray, metric: str, k: int):
    k = max(2, min(k, X.shape[0]))
    nn = NearestNeighbors(metric=metric, n_neighbors=k)
    nn.fit(X)
    dists, neigh = nn.kneighbors(X, n_neighbors=k)
    return dists, neigh


def nearest_diff_neighbor(neigh_row: np.ndarray, labels: np.ndarray, i: int):
    """Return the nearest neighbor index with a different label, or None."""
    yi = labels[i]
    for j in neigh_row:
        if j == i:
            continue
        if labels[j] != yi:
            return int(j)
    return None


def margin_hardness_scores(dists: np.ndarray, neigh: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    For each point i:
      d_same = distance to nearest same-label neighbor
      d_diff = distance to nearest different-label neighbor
      score = d_diff - d_same  (smaller => closer to boundary / harder)
    If either neighbor type not found, score = +inf (not boundary-like).
    """
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


def pick_boundary_candidates(X: np.ndarray, y: np.ndarray, metric: str, boundary_k: int):
    """
    Strong boundary candidate selection using margin hardness.
    Returns indices sorted from most boundary-like (smallest margin) to least.
    """
    dists, neigh = build_knn(X, metric=metric, k=boundary_k)
    scores = margin_hardness_scores(dists, neigh, y)
    order = np.argsort(scores)  # smallest margin first
    # filter out inf scores
    order = order[np.isfinite(scores[order])]
    return order, dists, neigh, scores


def choose_flip_label_from_neighbor(y: np.ndarray, neighbor_idx: int):
    return y[neighbor_idx]


def apply_point_flip(y_new: np.ndarray, i: int, new_label, changed_mask: np.ndarray, log_rows: list, reason: str):
    old = y_new[i]
    if old == new_label:
        return False
    y_new[i] = new_label
    changed_mask[i] = True
    log_rows.append(
        dict(index=i, original_label=old, new_label=new_label, reason=reason)
    )
    return True


def inject_random_exact(X, y, noise_level, random_seed):
    """
    Exact budget random label flips:
      - select random points
      - assign each a random label (uniform) that is different from its current label
    """
    n = len(y)
    target = target_num_changes(n, noise_level)
    rng = np.random.RandomState(random_seed)

    labels = np.unique(y)
    if labels.size < 2:
        raise ValueError("Need at least 2 classes for random label flips.")

    indices = np.arange(n)
    rng.shuffle(indices)

    y_new = y.copy()
    changed = np.zeros(n, dtype=bool)
    log_rows = []

    changes = 0
    for i in indices:
        if changes >= target:
            break
        cur = y_new[i]
        pool = labels[labels != cur]
        if pool.size == 0:
            continue
        new_label = rng.choice(pool)
        if apply_point_flip(y_new, i, new_label, changed, log_rows, reason="random"):
            changes += 1

    return y_new, log_rows


def inject_nn_exact(X, y, metric, noise_level, random_seed, nn_k):
    """
    Exact budget NN: select points (random order) and flip each to its nearest DIFFERENT-label neighbor.
    """
    n = len(y)
    target = target_num_changes(n, noise_level)
    rng = np.random.RandomState(random_seed)

    # KNN once
    dists, neigh = build_knn(X, metric=metric, k=nn_k)

    indices = np.arange(n)
    rng.shuffle(indices)

    y_new = y.copy()
    changed = np.zeros(n, dtype=bool)
    log_rows = []

    changes = 0
    for i in indices:
        if changes >= target:
            break
        j = nearest_diff_neighbor(neigh[i], y_new, i)
        if j is None:
            continue
        new_label = choose_flip_label_from_neighbor(y_new, j)
        if apply_point_flip(y_new, i, new_label, changed, log_rows, reason="nn_diff"):
            changes += 1

    return y_new, log_rows


def inject_border_exact(X, y, metric, noise_level, random_seed, boundary_k, nn_k, boundary_top_frac):
    """
    Exact budget boundary nearest:
      - compute boundary candidates by margin hardness (d_diff - d_same), smaller is more boundary-like
      - restrict to top boundary_top_frac of candidates
      - flip those points to their nearest DIFFERENT-label neighbor
    """
    n = len(y)
    target = target_num_changes(n, noise_level)
    rng = np.random.RandomState(random_seed)

    order, _, _, _ = pick_boundary_candidates(X, y, metric=metric, boundary_k=boundary_k)
    if len(order) == 0:
        return inject_nn_exact(X, y, metric, noise_level, random_seed, nn_k)

    top_n = max(1, int(np.floor(boundary_top_frac * len(order))))
    boundary_pool = order[:top_n].copy()
    rng.shuffle(boundary_pool)

    _, neigh = build_knn(X, metric=metric, k=nn_k)
    y_new = y.copy()
    changed = np.zeros(n, dtype=bool)
    log_rows = []

    changes = 0
    for i in boundary_pool:
        if changes >= target:
            break
        j = nearest_diff_neighbor(neigh[i], y_new, i)
        if j is None:
            continue
        new_label = choose_flip_label_from_neighbor(y_new, j)
        if apply_point_flip(y_new, i, new_label, changed, log_rows, reason="border"):
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
            new_label = choose_flip_label_from_neighbor(y_new, j)
            if apply_point_flip(y_new, i, new_label, changed, log_rows, reason="border_fill_nn"):
                changes += 1

    return y_new, log_rows


def inject_cluster_exact(X, y, metric, noise_level, random_seed, cluster_size, nn_k):
    """
    Exact budget cluster:
      - pick random seed points (not yet changed)
      - for each seed, define a local cluster as its nearest neighbors (including itself)
      - flip as many points in that cluster as needed to hit the exact target
      - each cluster uses a single target label (nearest different-label neighbor of the seed)
    """
    n = len(y)
    target = target_num_changes(n, noise_level)
    rng = np.random.RandomState(random_seed)

    # KNN for cluster membership + diff-label target
    dists, neigh = build_knn(X, metric=metric, k=max(nn_k, cluster_size + 5))

    y_new = y.copy()
    changed = np.zeros(n, dtype=bool)
    log_rows = []

    indices = np.arange(n)
    rng.shuffle(indices)

    changes = 0
    for seed in indices:
        if changes >= target:
            break
        if changed[seed]:
            continue

        # Choose a single target label for this cluster (from seed's nearest diff-label neighbor)
        seed_nn = nearest_diff_neighbor(neigh[seed], y_new, seed)
        if seed_nn is None:
            continue
        cluster_target = choose_flip_label_from_neighbor(y_new, seed_nn)

        # cluster members: closest cluster_size points (include seed)
        members = []
        for j in neigh[seed]:
            if len(members) >= cluster_size:
                break
            members.append(int(j))

        # Attempt flips within the cluster until budget met
        for i in members:
            if changes >= target:
                break
            if changed[i]:
                continue
            if apply_point_flip(y_new, i, cluster_target, changed, log_rows, reason=f"cluster_seed_{seed}"):
                changes += 1

    return y_new, log_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, required=False, default="")
    ap.add_argument("--labels", type=str, required=True)
    ap.add_argument("--id-column", type=str, default="vector_id")
    ap.add_argument("--label-column", type=str, default="label")

    ap.add_argument("--output-labels", type=str, required=True)
    ap.add_argument("--log-file", type=str, required=True)

    ap.add_argument("--mode", type=str, required=True,
                    choices=["random", "border", "cluster"])
    ap.add_argument("--noise-level", type=float, required=True)

    ap.add_argument("--metric", type=str, default="cosine")
    ap.add_argument("--random-seed", type=int, default=42)

    ap.add_argument("--cluster-size", type=int, default=1)
    ap.add_argument("--nn-k", type=int, default=50, help="K for neighbor search (diff-label selection)")
    ap.add_argument("--boundary-k", type=int, default=25, help="K for boundary hardness scoring")
    ap.add_argument("--boundary-top-frac", type=float, default=0.25,
                    help="Use top fraction of boundary candidates as seeds (smaller => more boundary focused)")

    args = ap.parse_args()

    if "--random-seed" not in sys.argv:
        print('[noise_injection] warning: --random-seed not provided; using default ' + f"{args.random_seed}. Results are seed-dependent.")

    require_embeddings = args.mode in ("border", "cluster")
    if require_embeddings and not args.embeddings:
        raise ValueError("--embeddings is required for border/cluster modes.")
    X, df, ids, y = load_inputs(
        args.embeddings, args.labels, args.id_column, args.label_column, require_embeddings=require_embeddings
    )

    if args.cluster_size < 1:
        raise ValueError("cluster-size must be >= 1")
    if args.nn_k < 2:
        raise ValueError("nn-k must be >= 2")
    if args.boundary_k < 2:
        raise ValueError("boundary-k must be >= 2")
    if not (0 < args.boundary_top_frac <= 1.0):
        raise ValueError("boundary-top-frac must be in (0,1]")

    # Run selected injection
    if args.mode == "random":
        y_new, log_rows = inject_random_exact(
            X, y, noise_level=args.noise_level, random_seed=args.random_seed
        )
    elif args.mode == "border":
        y_new, log_rows = inject_border_exact(
            X, y, metric=args.metric, noise_level=args.noise_level,
            random_seed=args.random_seed, boundary_k=args.boundary_k,
            nn_k=args.nn_k, boundary_top_frac=args.boundary_top_frac
        )
        border_count = sum(1 for r in log_rows if r.get("reason") == "border")
        fill_count = sum(1 for r in log_rows if r.get("reason") == "border_fill_nn")
        print(f"[noise_injection] border pool flips: {border_count}")
        print(f"[noise_injection] fill flips:        {fill_count}")
        if fill_count > 0:
            print("[noise_injection] warning: fill flips were needed to reach target.")
    elif args.mode == "cluster":
        y_new, log_rows = inject_cluster_exact(
            X, y, metric=args.metric, noise_level=args.noise_level,
            random_seed=args.random_seed, cluster_size=args.cluster_size, nn_k=args.nn_k
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    # Write output labels CSV: preserve all original columns, only replace label column
    out_df = df.copy()
    out_df[args.label_column] = y_new
    out_df.to_csv(args.output_labels, index=False)

    # Write log
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(args.log_file, index=False)

    # Print summary
    changed = (y_new != y)
    n_changed = int(changed.sum())
    target = target_num_changes(len(y), args.noise_level)
    print(f"[noise_injection] mode={args.mode}")
    print(f"[noise_injection] target_changes={target} actual_changes={n_changed} (N={len(y)})")
    print(f"[noise_injection] wrote labels: {args.output_labels}")
    print(f"[noise_injection] wrote log:    {args.log_file}")


if __name__ == "__main__":
    main()
