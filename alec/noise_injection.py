#!/usr/bin/env python3
"""
noise_injection.py

Injects structured label noise into a dataset using image embeddings.

Supported noise modes:
- nearest_neighbor:
    For each selected example, flip its label to the label of its nearest neighbor in
    embedding space (seeds are chosen randomly across the dataset).

- cluster:
    Build clusters of size `cluster_size` by grouping each chosen seed with its
    nearest neighbors. For each cluster, flip *all* labels in the cluster to the
    label of the nearest example (outside the cluster) in embedding space.

- boundary_nearest:
    First identify "boundary points" whose neighbors contain multiple labels.
    Choose seeds only from these boundary points, then flip each seed's label
    to the label of its nearest neighbor. This targets hard / ambiguous points.

- boundary_cluster:
    Same boundary-point selection as above, but build clusters around boundary
    seeds and flip entire clusters at or near the class boundary.

Special case:
- cluster_size = 1 simulates flipping individual images (no multi-image clusters).

Inputs:
- Embeddings: .npy file of shape [N, D]
- Labels: CSV with at least [id_column, label_column]. Order must match embeddings.

Outputs:
- Noisy labels CSV (same schema as input CSV, but with label_column updated).
- Log CSV with columns [index, id, original_label, new_label, mode, cluster_id, ...].

Usage example:

python noise_injection.py \
    --embeddings embeddings.npy \
    --labels labels.csv \
    --id-column id \
    --label-column label \
    --output-labels labels_noisy.csv \
    --log-file noise_log.csv \
    --mode boundary_cluster \
    --noise-level 0.1 \
    --cluster-size 5 \
    --metric cosine \
    --boundary-k 10 \
    --random-seed 42
"""

import argparse
import json
import os
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# -------------------- I/O helpers --------------------


def load_embeddings(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    emb = np.load(path)
    if emb.ndim != 2:
        raise ValueError(f"Expected embeddings shape [N, D], got {emb.shape}")
    return emb


def load_labels(path: str, id_column: str, label_column: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Labels file not found: {path}")
    df = pd.read_csv(path)
    if id_column not in df.columns:
        raise ValueError(f"id_column '{id_column}' not in labels CSV columns: {df.columns.tolist()}")
    if label_column not in df.columns:
        raise ValueError(f"label_column '{label_column}' not in labels CSV columns: {df.columns.tolist()}")
    return df


def save_noisy_labels(
    df: pd.DataFrame,
    noisy_labels: np.ndarray,
    label_column: str,
    output_path: str,
) -> None:
    df_out = df.copy()
    df_out[label_column] = noisy_labels
    df_out.to_csv(output_path, index=False)


def save_logs(
    logs: List[Dict],
    df: pd.DataFrame,
    id_column: str,
    output_path: str,
) -> None:
    if not logs:
        pd.DataFrame(
            columns=["index", id_column, "original_label", "new_label", "mode", "cluster_id"]
        ).to_csv(output_path, index=False)
        return

    log_df = pd.DataFrame(logs)
    log_df[id_column] = log_df["index"].apply(lambda i: df.iloc[int(i)][id_column])
    cols = ["index", id_column] + [c for c in log_df.columns if c not in ("index", id_column)]
    log_df = log_df[cols]
    log_df.to_csv(output_path, index=False)


# -------------------- Nearest neighbor utilities --------------------


def build_nn_index(
    embeddings: np.ndarray,
    metric: str = "cosine",
) -> NearestNeighbors:
    """
    Build a NearestNeighbors index for the given embeddings.
    """
    nn = NearestNeighbors(metric=metric, algorithm="auto")
    nn.fit(embeddings)
    return nn


def select_indices_to_corrupt(
    n_samples: int,
    noise_level: float,
    random_state: np.random.RandomState,
    candidate_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Select a subset of indices to corrupt.

    noise_level is a fraction in (0, 1].

    If candidate_indices is None, indices are drawn from [0, n_samples).
    Otherwise, indices are drawn from candidate_indices.
    """
    if not (0.0 < noise_level <= 1.0):
        raise ValueError("noise_level must be in (0, 1].")

    if candidate_indices is None:
        pool = np.arange(n_samples)
    else:
        candidate_indices = np.asarray(candidate_indices, dtype=int)
        if candidate_indices.size == 0:
            raise ValueError("candidate_indices is empty; cannot select seeds.")
        pool = candidate_indices

    n_pool = pool.shape[0]
    n_corrupt = max(1, int(round(n_pool * noise_level)))
    if n_corrupt > n_pool:
        n_corrupt = n_pool

    pool = pool.copy()
    random_state.shuffle(pool)
    return pool[:n_corrupt]


# -------------------- Boundary detection --------------------


def find_boundary_points(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_neighbors: int,
    metric: str,
) -> np.ndarray:
    """
    Identify "boundary points" whose neighborhood contains multiple labels.

    For each point:
      - Find k nearest neighbors (including itself).
      - Look at the labels of neighbors (excluding self index for clarity).
      - If the neighbor labels contain more than one unique class, mark this
        point as a boundary point.

    Returns:
        boundary_indices: array of indices of boundary points.
    """
    n_samples = embeddings.shape[0]
    k_neighbors = max(2, min(k_neighbors, n_samples))  # at least 2, at most N

    nn = NearestNeighbors(metric=metric, n_neighbors=k_neighbors)
    nn.fit(embeddings)
    dists, neighbors = nn.kneighbors(embeddings, n_neighbors=k_neighbors)

    boundary_indices: List[int] = []

    for i in range(n_samples):
        neighbor_idxs = neighbors[i]
        # Exclude self from neighbors for label diversity check if present
        neighbor_idxs = [j for j in neighbor_idxs if j != i]
        if not neighbor_idxs:
            continue
        neighbor_labels = labels[neighbor_idxs]
        if len(set(neighbor_labels.tolist())) > 1:
            boundary_indices.append(i)

    return np.asarray(boundary_indices, dtype=int)


# -------------------- Noise application methods --------------------


def apply_nearest_neighbor_noise(
    embeddings: np.ndarray,
    labels: np.ndarray,
    noise_level: float,
    metric: str,
    random_state: np.random.RandomState,
    seeds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Flip labels of selected examples to the label of their nearest neighbor.

    If seeds is None, seeds are selected randomly across all samples.
    Otherwise, only the provided seed indices are corrupted.
    """
    n_samples = embeddings.shape[0]
    new_labels = labels.copy()
    logs: List[Dict] = []

    nn = build_nn_index(embeddings, metric=metric)
    distances, indices = nn.kneighbors(embeddings, n_neighbors=2)

    if seeds is None:
        seeds = select_indices_to_corrupt(n_samples, noise_level, random_state)
    else:
        seeds = np.asarray(seeds, dtype=int)

    for idx in seeds:
        neighbor_idx = indices[idx, 1]  # 0 is self, 1 is closest other
        orig_label = new_labels[idx]
        new_label = new_labels[neighbor_idx]
        if orig_label == new_label:
            continue
        new_labels[idx] = new_label
        logs.append(
            {
                "index": int(idx),
                "original_label": str(orig_label),
                "new_label": str(new_label),
                "mode": "nearest_neighbor",
                "cluster_id": int(idx),  # degenerate "cluster" of size 1
                "neighbor_index": int(neighbor_idx),
            }
        )

    return new_labels, logs


def build_clusters_by_knn(
    embeddings: np.ndarray,
    seeds: np.ndarray,
    cluster_size: int,
    metric: str,
) -> List[np.ndarray]:
    """
    Build disjoint clusters of up to `cluster_size` points around each seed
    using nearest neighbors in embedding space.

    Greedy: for each seed, grab its nearest neighbors that are not already
    assigned to some cluster until cluster_size is reached.
    """
    n_samples = embeddings.shape[0]
    if cluster_size < 1:
        raise ValueError("cluster_size must be >= 1")

    used = np.zeros(n_samples, dtype=bool)
    seeds = np.asarray(seeds, dtype=int)

    nn = build_nn_index(embeddings, metric=metric)
    # Precompute neighbors for all points, up to cluster_size
    n_neighbors = min(cluster_size, n_samples)
    dists, neighbors = nn.kneighbors(embeddings, n_neighbors=n_neighbors)

    clusters: List[np.ndarray] = []

    for seed in seeds:
        if used[seed]:
            continue

        cluster_indices = [seed]
        used[seed] = True

        # Add nearest neighbors not yet used
        for nb in neighbors[seed]:
            if len(cluster_indices) >= cluster_size:
                break
            if not used[nb]:
                cluster_indices.append(int(nb))
                used[nb] = True

        clusters.append(np.asarray(cluster_indices, dtype=int))

    return clusters


def apply_cluster_noise(
    embeddings: np.ndarray,
    labels: np.ndarray,
    noise_level: float,
    cluster_size: int,
    metric: str,
    random_state: np.random.RandomState,
    seeds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Flip labels of clusters of examples.

    Procedure:
    - Choose seeds according to noise_level (or use provided seeds).
    - Around each seed, form a cluster of up to `cluster_size` points using KNN.
    - For each cluster:
        - Compute cluster centroid.
        - Find nearest example *outside* the cluster.
        - Flip all labels in the cluster to that example's label.

    Special case:
    - If cluster_size == 1, this behaves like per-example centroid noise.
    """
    n_samples = embeddings.shape[0]
    if cluster_size < 1:
        raise ValueError("cluster_size must be >= 1")

    new_labels = labels.copy()
    logs: List[Dict] = []

    if seeds is None:
        seeds = select_indices_to_corrupt(n_samples, noise_level, random_state)
    else:
        seeds = np.asarray(seeds, dtype=int)

    # Build clusters around seeds
    clusters = build_clusters_by_knn(embeddings, seeds, cluster_size, metric)

    # NN index for centroid-to-point queries
    global_nn = build_nn_index(embeddings, metric=metric)
    n_neighbors = min(10, n_samples)

    for cluster_id, cluster_indices in enumerate(clusters):
        cluster_indices = np.asarray(cluster_indices, dtype=int)
        cluster_embs = embeddings[cluster_indices]
        centroid = cluster_embs.mean(axis=0, keepdims=True)

        dists, neigh_idxs = global_nn.kneighbors(centroid, n_neighbors=n_neighbors)
        neigh_idxs = neigh_idxs[0]

        cluster_set = set(cluster_indices.tolist())
        target_idx: Optional[int] = None
        for candidate in neigh_idxs:
            if candidate not in cluster_set:
                target_idx = int(candidate)
                break

        if target_idx is None:
            # Couldn't find a target outside the cluster; skip
            continue

        target_label = new_labels[target_idx]

        for idx in cluster_indices:
            orig_label = new_labels[idx]
            if orig_label == target_label:
                continue
            new_labels[idx] = target_label
            logs.append(
                {
                    "index": int(idx),
                    "original_label": str(orig_label),
                    "new_label": str(target_label),
                    "mode": "cluster",
                    "cluster_id": int(cluster_id),
                    "cluster_size": int(len(cluster_indices)),
                    "target_index": int(target_idx),
                }
            )

    return new_labels, logs


# -------------------- Boundary-based noise wrappers --------------------


def apply_boundary_nearest_noise(
    embeddings: np.ndarray,
    labels: np.ndarray,
    noise_level: float,
    metric: str,
    boundary_k: int,
    random_state: np.random.RandomState,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Apply nearest-neighbor noise, but only to points near the class boundary.

    Boundary points are those whose k-nearest neighbors contain multiple labels.
    """
    boundary_indices = find_boundary_points(
        embeddings=embeddings,
        labels=labels,
        k_neighbors=boundary_k,
        metric=metric,
    )
    if boundary_indices.size == 0:
        raise ValueError("No boundary points found; cannot apply boundary_nearest noise.")

    seeds = select_indices_to_corrupt(
        n_samples=embeddings.shape[0],
        noise_level=noise_level,
        random_state=random_state,
        candidate_indices=boundary_indices,
    )

    noisy_labels, logs = apply_nearest_neighbor_noise(
        embeddings=embeddings,
        labels=labels,
        noise_level=noise_level,  # noise_level unused when seeds provided, but kept for summary
        metric=metric,
        random_state=random_state,
        seeds=seeds,
    )

    # Overwrite mode in logs to make it explicit
    for log in logs:
        log["mode"] = "boundary_nearest"

    return noisy_labels, logs


def apply_boundary_cluster_noise(
    embeddings: np.ndarray,
    labels: np.ndarray,
    noise_level: float,
    cluster_size: int,
    metric: str,
    boundary_k: int,
    random_state: np.random.RandomState,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Apply cluster noise, but only around boundary points.

    Seeds come from boundary points, and clusters are grown around them in
    embedding space.
    """
    boundary_indices = find_boundary_points(
        embeddings=embeddings,
        labels=labels,
        k_neighbors=boundary_k,
        metric=metric,
    )
    if boundary_indices.size == 0:
        raise ValueError("No boundary points found; cannot apply boundary_cluster noise.")

    seeds = select_indices_to_corrupt(
        n_samples=embeddings.shape[0],
        noise_level=noise_level,
        random_state=random_state,
        candidate_indices=boundary_indices,
    )

    noisy_labels, logs = apply_cluster_noise(
        embeddings=embeddings,
        labels=labels,
        noise_level=noise_level,  # noise_level unused when seeds provided, but kept for summary
        cluster_size=cluster_size,
        metric=metric,
        random_state=random_state,
        seeds=seeds,
    )

    # Overwrite mode in logs to make it explicit
    for log in logs:
        log["mode"] = "boundary_cluster"

    return noisy_labels, logs


# -------------------- CLI --------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject structured label noise using embeddings.")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings .npy file")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels CSV file")
    parser.add_argument("--id-column", type=str, default="id", help="ID column in labels CSV")
    parser.add_argument("--label-column", type=str, default="label", help="Label column in labels CSV")
    parser.add_argument("--output-labels", type=str, required=True, help="Path to write noisy labels CSV")
    parser.add_argument("--log-file", type=str, required=True, help="Path to write noise log CSV")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["nearest_neighbor", "cluster", "boundary_nearest", "boundary_cluster"],
        required=True,
        help="Noise mode to use",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        required=True,
        help="Fraction of dataset (or boundary pool) to seed noise on (0 < noise_level <= 1)",
    )
    parser.add_argument(
        "--cluster-size",
        type=int,
        default=1,
        help="Cluster size k for cluster-based modes. k=1 simulates individual flips.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        help="Distance metric for NearestNeighbors (e.g., 'cosine', 'euclidean')",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--boundary-k",
        type=int,
        default=10,
        help="Number of neighbors to use when detecting boundary points (for boundary_* modes).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.random_seed)

    embeddings = load_embeddings(args.embeddings)
    df_labels = load_labels(args.labels, args.id_column, args.label_column)

    if embeddings.shape[0] != len(df_labels):
        raise ValueError(
            f"Embeddings count ({embeddings.shape[0]}) does not match labels rows ({len(df_labels)})"
        )

    labels = df_labels[args.label_column].values

    if args.mode == "nearest_neighbor":
        noisy_labels, logs = apply_nearest_neighbor_noise(
            embeddings=embeddings,
            labels=labels,
            noise_level=args.noise_level,
            metric=args.metric,
            random_state=rng,
        )
    elif args.mode == "cluster":
        noisy_labels, logs = apply_cluster_noise(
            embeddings=embeddings,
            labels=labels,
            noise_level=args.noise_level,
            cluster_size=args.cluster_size,
            metric=args.metric,
            random_state=rng,
        )
    elif args.mode == "boundary_nearest":
        noisy_labels, logs = apply_boundary_nearest_noise(
            embeddings=embeddings,
            labels=labels,
            noise_level=args.noise_level,
            metric=args.metric,
            boundary_k=args.boundary_k,
            random_state=rng,
        )
    elif args.mode == "boundary_cluster":
        noisy_labels, logs = apply_boundary_cluster_noise(
            embeddings=embeddings,
            labels=labels,
            noise_level=args.noise_level,
            cluster_size=args.cluster_size,
            metric=args.metric,
            boundary_k=args.boundary_k,
            random_state=rng,
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    save_noisy_labels(df_labels, noisy_labels, args.label_column, args.output_labels)
    save_logs(logs, df_labels, args.id_column, args.log_file)

    summary = {
        "mode": args.mode,
        "noise_level": args.noise_level,
        "cluster_size": args.cluster_size,
        "metric": args.metric,
        "random_seed": args.random_seed,
        "boundary_k": args.boundary_k,
        "n_samples": int(embeddings.shape[0]),
        "n_changes": int(len(logs)),
    }
    print("Noise injection summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
