"""
Smoke tests for the public CLI of imagenet1k_cleaning_chunked.py.

These tests verify all four public commands work end-to-end on a small
fixture and produce the documented guaranteed output artifacts.

Run with:
    python3 -m pytest tests/ -v
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "clean_imagenet1k" / "imagenet1k_cleaning_chunked.py"
POLICY = REPO_ROOT / "clean_imagenet1k" / "cleaning_policy.yaml"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "sample_export"


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run the cleaning script with the given arguments."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
    )
    return result


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------

def test_help():
    """--help exits 0 and prints usage."""
    result = run_cli("--help")
    assert result.returncode == 0
    assert "from-filtered-json" in result.stdout
    assert "analyze-cleaning-run" in result.stdout
    assert "compare-cleaning-runs" in result.stdout
    assert "stress-test-policy-variants" in result.stdout
    assert "dry-run-auth" not in result.stdout
    assert "export-from-vl" not in result.stdout
    assert "discover-cluster-ids" not in result.stdout


# ---------------------------------------------------------------------------
# from-filtered-json
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cleaning_run_dir(tmp_path_factory):
    """Run from-filtered-json on the fixture and return the run directory."""
    output_root = tmp_path_factory.mktemp("from_filtered_json")
    result = run_cli(
        "from-filtered-json",
        "--partition-mode", "cluster",
        "--cluster-input-dir", str(FIXTURE_DIR),
        "--policy", str(POLICY),
        "--output-root", str(output_root),
        "--mode", "metadata-only",
        "--expected-total-media-items", "10",
    )
    assert result.returncode == 0, f"from-filtered-json failed:\n{result.stderr}"

    # The script writes to a subdirectory matching clean_*_cluster_groups
    run_dirs = list(output_root.glob("*_cluster_groups"))
    assert run_dirs, f"No run directory found under {output_root}"
    return run_dirs[0]


GUARANTEED_RUN_OUTPUTS = [
    "raw_merged_metadata.json",
    "cleaned_imagenet1k.json",
    "metadata.json",
    "dropped_metadata.json",
    "prune_decisions.jsonl",
    "keep_filenames.txt",
    "drop_filenames.txt",
    "cleaning_summary.json",
    "cleaning_policy.yaml",
    "README.md",
]


@pytest.mark.parametrize("filename", GUARANTEED_RUN_OUTPUTS)
def test_from_filtered_json_outputs_exist(cleaning_run_dir, filename):
    """All guaranteed run artifacts must be present after from-filtered-json."""
    assert (cleaning_run_dir / filename).exists(), (
        f"Missing guaranteed output: {filename}"
    )


def test_cleaning_summary_schema(cleaning_run_dir):
    """cleaning_summary.json must have required top-level keys."""
    summary = json.loads((cleaning_run_dir / "cleaning_summary.json").read_text())
    for key in ("timestamp", "policy_name", "total_images", "kept", "dropped"):
        assert key in summary, f"cleaning_summary.json missing key: {key}"


def test_prune_decisions_has_records(cleaning_run_dir):
    """prune_decisions.jsonl must have exactly one decision per input record."""
    lines = (cleaning_run_dir / "prune_decisions.jsonl").read_text().splitlines()
    lines = [l for l in lines if l.strip()]
    assert len(lines) == 10, f"Expected 10 prune decisions, got {len(lines)}"
    for line in lines:
        decision = json.loads(line)
        assert "keep" in decision, f"prune_decisions.jsonl record missing 'keep' field: {decision}"
        assert isinstance(decision["keep"], bool)


# ---------------------------------------------------------------------------
# analyze-cleaning-run
# ---------------------------------------------------------------------------

GUARANTEED_ANALYSIS_OUTPUTS = [
    "analysis/policy_analysis_summary.json",
    "analysis/drop_reason_counts.csv",
    "analysis/drop_reason_overlap.csv",
    "analysis/drop_reason_combinations.csv",
    "analysis/class_impact.csv",
    "analysis/uniqueness_by_decision.csv",
    "analysis/issue_confidence_by_decision.csv",
    "analysis/issue_types_by_label.csv",
    "analysis/issue_types_by_label_plot_groups.csv",
]


@pytest.fixture(scope="module")
def analysis_dir(cleaning_run_dir):
    """Run analyze-cleaning-run on the fixture run and return the analysis dir."""
    result = run_cli(
        "analyze-cleaning-run",
        "--run-dir", str(cleaning_run_dir),
        "--skip-plots",
    )
    assert result.returncode == 0, f"analyze-cleaning-run failed:\n{result.stderr}"
    return cleaning_run_dir / "analysis"


@pytest.mark.parametrize("relative_path", GUARANTEED_ANALYSIS_OUTPUTS)
def test_analyze_cleaning_run_outputs_exist(cleaning_run_dir, analysis_dir, relative_path):
    """All guaranteed analysis artifacts must be present."""
    assert (cleaning_run_dir / relative_path).exists(), (
        f"Missing analysis output: {relative_path}"
    )


def test_analyze_cleaning_run_honors_top_classes(cleaning_run_dir, tmp_path):
    """Class-based summary lists should respect --top-classes."""
    analysis_dir = tmp_path / "analysis_top2"
    result = run_cli(
        "analyze-cleaning-run",
        "--run-dir", str(cleaning_run_dir),
        "--output-dir", str(analysis_dir),
        "--top-classes", "2",
        "--skip-plots",
    )
    assert result.returncode == 0, f"analyze-cleaning-run --top-classes failed:\n{result.stderr}"

    summary = json.loads((analysis_dir / "policy_analysis_summary.json").read_text())
    assert len(summary["top_over_pruned_classes"]) <= 2
    assert len(summary["flagged_over_pruned_classes"]) <= 2


# ---------------------------------------------------------------------------
# compare-cleaning-runs
# ---------------------------------------------------------------------------

def test_compare_cleaning_runs(cleaning_run_dir, tmp_path):
    """compare-cleaning-runs with a single run should produce a comparison CSV."""
    output_csv = tmp_path / "policy_comparison.csv"
    result = run_cli(
        "compare-cleaning-runs",
        "--run-dirs", str(cleaning_run_dir),
        "--output-file", str(output_csv),
    )
    assert result.returncode == 0, f"compare-cleaning-runs failed:\n{result.stderr}"
    assert output_csv.exists(), "policy_comparison.csv not created"
    content = output_csv.read_text()
    assert len(content.strip().splitlines()) >= 2, "Comparison CSV must have header + at least one data row"


# ---------------------------------------------------------------------------
# stress-test-policy-variants
# ---------------------------------------------------------------------------

def test_stress_test_policy_variants(tmp_path):
    """stress-test-policy-variants should produce per-variant run folders."""
    output_root = tmp_path / "sweep"
    result = run_cli(
        "stress-test-policy-variants",
        "--input-dir", str(FIXTURE_DIR),
        "--policy", str(POLICY),
        "--output-root", str(output_root),
        "--expected-total-media-items", "10",
        "--top-classes", "2",
    )
    assert result.returncode == 0, f"stress-test-policy-variants failed:\n{result.stderr}"

    # At least one variant run directory must exist (named clean_*_groups)
    variant_dirs = list(output_root.glob("clean_*_groups"))
    assert variant_dirs, "No variant run directories found under sweep output root"

    # Each variant dir must have a cleaning_summary.json
    for vdir in variant_dirs:
        assert (vdir / "cleaning_summary.json").exists(), (
            f"Missing cleaning_summary.json in variant dir: {vdir.name}"
        )
        summary = json.loads((vdir / "analysis" / "policy_analysis_summary.json").read_text())
        assert len(summary["top_over_pruned_classes"]) <= 2
        assert len(summary["flagged_over_pruned_classes"]) <= 2

    # Sweep-level summary must exist
    assert (output_root / "policy_stress_test_summary.json").exists()
    assert (output_root / "policy_comparison.csv").exists()
