#!/usr/bin/env python3
"""Generic end-to-end pipeline runner.

The pipeline config should define ordered stages:

stages:
  - name: embeddings_pack
    enabled: true
    module: visual_layer_capstone.embeddings.export_embeddings
    args: ["metadata-to-npy", "--metadata-csv", "PATH_TO_METADATA_CSV"]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run reproducibility pipeline stages")
    ap.add_argument("--config", type=str, default="configs/pipeline.yaml")
    ap.add_argument("--stop-on-error", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    with config_path.open("r") as f:
        config = yaml.safe_load(f) or {}

    stages = config.get("stages", [])
    if not isinstance(stages, list):
        raise ValueError("Pipeline config field 'stages' must be a list")

    for stage in stages:
        name = stage.get("name", "unnamed_stage")
        enabled = bool(stage.get("enabled", True))
        module = stage.get("module")
        module_args = stage.get("args", [])

        if not enabled:
            print(f"[pipeline] skip: {name}")
            continue
        if not module:
            raise ValueError(f"Stage '{name}' missing required key 'module'")
        if not isinstance(module_args, list):
            raise ValueError(f"Stage '{name}' key 'args' must be a list")

        cmd = [sys.executable, "-m", module] + [str(a) for a in module_args]
        print(f"[pipeline] run: {name}")
        print(f"[pipeline] cmd: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            print(f"[pipeline] stage failed: {name} (code={result.returncode})")
            if args.stop_on_error:
                return result.returncode

    print("[pipeline] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
