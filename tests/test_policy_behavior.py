"""
Research-level behavior tests for the public ImageNet-1K cleaning workflow.

These tests verify the semantic behavior of the cleaning policy on the shipped
synthetic fixture rather than only checking file existence. They use the Python
standard library ``unittest`` runner so they can execute in environments where
``pytest`` is not installed.

Run with:
    python3 -m unittest tests.test_policy_behavior -v
"""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "clean_imagenet1k" / "imagenet1k_cleaning_chunked.py"
POLICY = REPO_ROOT / "clean_imagenet1k" / "cleaning_policy.yaml"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "sample_export"


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run the public cleaning CLI and capture output."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
    )


class TestPolicyBehavior(unittest.TestCase):
    """Validate the public cleaning logic against expected fixture outcomes."""

    maxDiff = None

    def test_exact_prune_decisions_for_fixture(self) -> None:
        """Each fixture record should resolve to the expected keep/drop outcome."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            result = run_cli(
                "from-filtered-json",
                "--partition-mode", "cluster",
                "--cluster-input-dir", str(FIXTURE_DIR),
                "--policy", str(POLICY),
                "--output-root", str(output_root),
                "--mode", "metadata-only",
                "--expected-total-media-items", "10",
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            run_dir = output_root / "clean_cluster_groups"
            rows = [
                json.loads(line)
                for line in (run_dir / "prune_decisions.jsonl").read_text().splitlines()
                if line.strip()
            ]
            observed = {
                row["media_id"]: {
                    "keep": row["keep"],
                    "drop_reasons": row["drop_reasons"],
                }
                for row in rows
            }

            expected = {
                "img_001": {"keep": False, "drop_reasons": ["issue_blurry"]},
                "img_002": {"keep": True, "drop_reasons": []},
                "img_003": {"keep": False, "drop_reasons": ["low_uniqueness<0.3"]},
                "img_004": {"keep": False, "drop_reasons": ["issue_mislabel"]},
                "img_005": {"keep": True, "drop_reasons": []},
                "img_006": {"keep": False, "drop_reasons": ["low_uniqueness<0.3", "issue_dark"]},
                "img_007": {"keep": False, "drop_reasons": ["issue_visual_outlier"]},
                "img_008": {"keep": False, "drop_reasons": ["user_tag: poor_quality"]},
                "img_009": {"keep": True, "drop_reasons": []},
                "img_010": {"keep": False, "drop_reasons": ["issue_label_outlier"]},
            }
            self.assertEqual(observed, expected)

            summary = json.loads((run_dir / "cleaning_summary.json").read_text())
            self.assertEqual(summary["total_images"], 10)
            self.assertEqual(summary["kept"], 3)
            self.assertEqual(summary["dropped"], 7)
            self.assertEqual(
                summary["drop_by_reason"],
                {
                    "low_uniqueness<0.3": 2,
                    "issue_dark": 1,
                    "issue_mislabel": 1,
                    "issue_label_outlier": 1,
                    "issue_visual_outlier": 1,
                    "issue_blurry": 1,
                    "user_tag: poor_quality": 1,
                },
            )

    def test_policy_sweep_changes_drop_rate_in_expected_direction(self) -> None:
        """Conservative should retain more images than balanced/aggressive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            result = run_cli(
                "stress-test-policy-variants",
                "--input-dir", str(FIXTURE_DIR),
                "--policy", str(POLICY),
                "--output-root", str(output_root),
                "--expected-total-media-items", "10",
                "--skip-plots",
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            with open(output_root / "policy_comparison.csv", newline="") as f:
                rows = list(csv.DictReader(f))

            by_variant = {row["variant"]: row for row in rows}
            self.assertEqual(set(by_variant.keys()), {"aggressive", "balanced", "conservative"})

            self.assertEqual(int(by_variant["aggressive"]["kept"]), 3)
            self.assertEqual(int(by_variant["balanced"]["kept"]), 3)
            self.assertEqual(int(by_variant["conservative"]["kept"]), 6)

            self.assertAlmostEqual(float(by_variant["aggressive"]["drop_rate"]), 0.7)
            self.assertAlmostEqual(float(by_variant["balanced"]["drop_rate"]), 0.7)
            self.assertAlmostEqual(float(by_variant["conservative"]["drop_rate"]), 0.4)

            self.assertGreater(
                int(by_variant["conservative"]["kept"]),
                int(by_variant["balanced"]["kept"]),
            )
            self.assertGreaterEqual(
                int(by_variant["balanced"]["kept"]),
                int(by_variant["aggressive"]["kept"]),
            )


if __name__ == "__main__":
    unittest.main()
