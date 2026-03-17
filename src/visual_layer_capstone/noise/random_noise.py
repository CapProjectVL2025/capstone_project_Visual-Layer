#!/usr/bin/env python3
"""Random label-noise injection."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import _core


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inject random label flips.")
    ap.add_argument("--labels", type=str, required=True)
    ap.add_argument("--id-column", type=str, default="vector_id")
    ap.add_argument("--label-column", type=str, default="label")
    ap.add_argument("--output-labels", type=str, required=True)
    ap.add_argument("--log-file", type=str, required=True)
    ap.add_argument("--noise-level", type=float, required=True)
    ap.add_argument("--random-seed", type=int, default=42)
    return ap.parse_args()


def run(args: argparse.Namespace) -> int:
    _, df, _, y = _core.load_inputs(
        embeddings_path="",
        labels_path=args.labels,
        id_col=args.id_column,
        label_col=args.label_column,
        require_embeddings=False,
    )

    y_new, log_rows = _core.inject_random_exact(
        X=None,
        y=y,
        noise_level=args.noise_level,
        random_seed=args.random_seed,
    )

    out_df = df.copy()
    out_df[args.label_column] = y_new

    output_labels = Path(args.output_labels)
    output_labels.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_labels, index=False)

    log_file = Path(args.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(log_file, index=False)

    changed = sum(1 for r in log_rows)
    target = _core.target_num_changes(len(y), args.noise_level)

    print(f"[random_noise] target_changes={target}")
    print(f"[random_noise] actual_changes={changed}")
    print(f"[random_noise] wrote labels: {output_labels}")
    print(f"[random_noise] wrote log: {log_file}")

    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
