#!/usr/bin/env python3
"""Cluster-aware label-noise injection."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import _core


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inject cluster-based noise.")
    ap.add_argument("--embeddings", type=str, required=True)
    ap.add_argument("--labels", type=str, required=True)
    ap.add_argument("--id-column", type=str, default="vector_id")
    ap.add_argument("--label-column", type=str, default="label")
    ap.add_argument("--output-labels", type=str, required=True)
    ap.add_argument("--log-file", type=str, required=True)
    ap.add_argument("--noise-level", type=float, required=True)
    ap.add_argument("--metric", type=str, default="cosine")
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--cluster-size", type=int, default=50)
    ap.add_argument("--nn-k", type=int, default=50)
    ap.add_argument("--pair-classes", nargs=2, default=[])
    return ap.parse_args()


def run(args: argparse.Namespace) -> int:
    X, df, _, y = _core.load_inputs(
        embeddings_path=args.embeddings,
        labels_path=args.labels,
        id_col=args.id_column,
        label_col=args.label_column,
        require_embeddings=True,
    )

    if args.pair_classes:
        y_new, log_rows, pair_n, target = _core.inject_cluster_pair_exact(
            X=X,
            y=y,
            metric=args.metric,
            noise_level=args.noise_level,
            random_seed=args.random_seed,
            cluster_size=args.cluster_size,
            nn_k=args.nn_k,
            pair_classes=(args.pair_classes[0], args.pair_classes[1]),
        )
        print(f"[cluster_noise] pair_classes={args.pair_classes[0]},{args.pair_classes[1]}")
        print(f"[cluster_noise] pair_candidates={pair_n}")
    else:
        y_new, log_rows = _core.inject_cluster_exact(
            X=X,
            y=y,
            metric=args.metric,
            noise_level=args.noise_level,
            random_seed=args.random_seed,
            cluster_size=args.cluster_size,
            nn_k=args.nn_k,
        )
        target = _core.target_num_changes(len(y), args.noise_level)

    out_df = df.copy()
    out_df[args.label_column] = y_new

    output_labels = Path(args.output_labels)
    output_labels.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_labels, index=False)

    log_file = Path(args.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(log_file, index=False)

    changed = sum(1 for _ in log_rows)
    print(f"[cluster_noise] target_changes={target}")
    print(f"[cluster_noise] actual_changes={changed}")
    print(f"[cluster_noise] wrote labels: {output_labels}")
    print(f"[cluster_noise] wrote log: {log_file}")

    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
