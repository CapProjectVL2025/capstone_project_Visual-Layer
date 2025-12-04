#!/usr/bin/env python3
"""
noise_injection.py

Injects structured label noise into a dataset using image embeddings.

Supported noise modes:
- nearest_neighbor:
    For each selected example, flip its label to the label of its nearest neighbor in
    embedding space.

- cluster:
    Build clusters of size `cluster_size` by grouping each chosen seed with its
    nearest neighbors. For each cluster, flip *all* labels in the cluster to the
    label of the nearest example (outside the cluster) in embedding space.

Special case:
- cluster_size = 1 simulates flipping individual images (no multi-image clusters).

Inputs:
- Embeddings: .npy file of shape [N, D]
- Labels: CSV with at least [id_column, label_column]. Order must match embeddings.

Outputs:
- Noisy labels CSV (same schema as input CSV, but with label_column updated).
- Log CSV with columns [index, id, original_label, new_label, mode, cluster_id].

Usage example:

python noise_injection.py \
    --embeddings embeddings.npy \
    --labels labels.csv \
    --id-column id \
    --label-column label \
    --output-labels labels_noisy.csv \
    --log-file noise_log.csv \
    --mode nearest_neighbor \
    --noise-level 0.1 \
    --cluster-size 1 \
    --metric cosine \
    --random-seed 42
"""

import argparse
import json
import os

from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


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


def build_nn_index(
    embeddings: np.ndarray,
    metric: str = "cosine",
    n_neighbors: int = 2,
) -> NearestNeighbors:
    """
    Build a NearestNeighbors index.

    n_neighbors=2 means: nearest neighbor list will contain [self, nearest_other].
    """
    nn = NearestNeighbors(metric=metric, algorithm="auto")
    nn.fit(embeddings)
    return nn


def select_indices_to_corrupt(
    n_samples: int,
    noise_level: float,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """
    Select a random subset of indices to corrupt (as seeds).

    noise_level is a fraction in (0, 1].
    """
    if not (0.0 < noise_level <= 1.0):
        raise ValueError("noise_level must be in (0, 1].")
    n_corrupt = max(1, int(round(n_samples * noise_level)))
    all_indices = np.arange(n_samples)
    random_state.shuffle(all_indices)
    return all_indices[:n_corrupt]


def apply_nearest_neighbor_noise(
    embeddings: np.ndarray,
    labels: np.ndarray,
    noise_level: float,
    metric: str,
    random_state: np.random.RandomState,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Flip labels of randomly selected examples to the label of their nearest neighbor.

    Returns:
        new_labels: updated labels array
        logs: list of dicts describing each change
    """
    n_samples = embeddings.shape[0]
    new_labels = labels.copy()
    logs: List[Dict] = []

    nn = build_nn_index(embeddings, metric=metric, n_neighbors=2)

    # For each sample, its nearest neighbors (including itself)
    distances, indices = nn.kneighbors(embeddings, n_neighbors=2)

    seeds = select_indices_to_corrupt(n_samples, noise_level, random_state)

    for idx in seeds:
        neighbor_idx = indices[idx, 1]  # 0 is self, 1 is closest other
        orig_label = new_labels[idx]
        new_label = new_labels[neighbor_idx]
        if orig_label == new_label:
            # No actual change; skip for logging clarity
            continue
        new_labels[idx] = new_label
        logs.append(
            {
                "index": int(idx),
                "original_label": str(orig_label),
                "new_label": str(new_label),
                "mode": "nearest_neighbor",
                "cluster_id": int(idx),  # cluster of size 1
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

    Greedy: we walk seeds in order, grab its nearest neighbors that are not
    already assigned to some cluster, until the cluster has size `cluster_size`.

    Returns:
        clusters: list of arrays of indices
    """
    n_samples = embeddings.shape[0]
    used = np.zeros(n_samples, dtype=bool)

    # Each query needs up to cluster_size neighbors
    nn = build_nn_index(embeddings, metric=metric, n_neighbors=min(cluster_size, n_samples))

    distances, indices = nn.kneighbors(embeddings, n_neighbors=min(cluster_size, n_samples))

    clusters: List[np.ndarray] = []

    for seed in seeds:
        if used[seed]:
            continue

        cluster_indices = [seed]
        used[seed] = True

        # Walk through the neighbors of seed, add until cluster_size reached
        for nb in indices[seed]:
            if len(cluster_indices) >= cluster_size:
                break
            if not used[nb]:
                cluster_indices.append(int(nb))
                used[nb] = True

        clusters.append(np.array(cluster_indices, dtype=int))

    return clusters


def apply_cluster_noise(
    embeddings: np.ndarray,
    labels: np.ndarray,
    noise_level: float,
    cluster_size: int,
    metric: str,
    random_state: np.random.RandomState,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Flip labels of clusters of examples.

    Procedure:
    - Choose seeds according to noise_level.
    - Around each seed, form a cluster of up to `cluster_size` points using KNN.
    - For each cluster:
        - Compute cluster centroid.
        - Find nearest example *outside* the cluster.
        - Flip all labels in the cluster to that example's label.

    Special case:
    - If cluster_size == 1, this behaves like per-example centroid noise.

    Returns:
        new_labels: updated labels array
        logs: list of dicts describing each change
    """
    n_samples = embeddings.shape[0]
    if cluster_size < 1:
        raise ValueError("cluster_size must be >= 1")

    new_labels = labels.copy()
    logs: List[Dict] = []

    # Select seeds for clusters
    seeds = select_indices_to_corrupt(n_samples, noise_level, random_state)

    # Build clusters around seeds
    clusters = build_clusters_by_knn(embeddings, seeds, cluster_size, metric)

    # For finding nearest examples to cluster centroids
    global_nn = build_nn_index(embeddings, metric=metric, n_neighbors=min(10, n_samples))

    for cluster_id, cluster_indices in enumerate(clusters):
        cluster_indices = np.array(cluster_indices, dtype=int)
        cluster_embs = embeddings[cluster_indices]
        centroid = cluster_embs.mean(axis=0, keepdims=True)

        # Find nearest examples to centroid
        dists, neigh_idxs = global_nn.kneighbors(centroid, n_neighbors=min(10, n_samples))
        neigh_idxs = neigh_idxs[0]

        # Choose nearest example that is NOT in the cluster
        cluster_set = set(cluster_indices.tolist())
        target_idx: Optional[int] = None
        for candidate in neigh_idxs:
            if candidate not in cluster_set:
                target_idx = int(candidate)
                break

        if target_idx is None:
            # Fall back: skip this cluster if no suitable target
            continue

        target_label = new_labels[target_idx]

        # Flip all labels in the cluster
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
        # Still create an empty file for consistency
        pd.DataFrame(columns=["index", id_column, "original_label", "new_label", "mode", "cluster_id"]).to_csv(
            output_path, index=False
        )
        return

    # Enrich logs with IDs
    log_df = pd.DataFrame(logs)
    # Assume "index" column refers to row index in df
    log_df[id_column] = log_df["index"].apply(lambda i: df.iloc[int(i)][id_column])
    # Put id column near front
    cols = ["index", id_column] + [c for c in log_df.columns if c not in ("index", id_column)]
    log_df = log_df[cols]
    log_df.to_csv(output_path, index=False)


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
        choices=["nearest_neighbor", "cluster"],
        required=True,
        help="Noise mode to use",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        required=True,
        help="Fraction of dataset to seed noise on (0 < noise_level <= 1)",
    )
    parser.add_argument(
        "--cluster-size",
        type=int,
        default=1,
        help="Cluster size k. k=1 simulates individual flips.",
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
        "n_samples": int(embeddings.shape[0]),
        "n_changes": int(len(logs)),
    }
    print("Noise injection summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
