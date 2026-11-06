# OWNER: M4 - do not edit unless you are M4
"""Tests for the experiment runner and baseline plotting integration (Checkpoint G8)."""
from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from mm.config import Config
from mm.run_experiments import main, run_experiments


def test_run_experiments_quick_mode() -> None:
    """Verify that run_experiments in quick mode generates raw CSVs, fronts, and baseline plots."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = Config(
            k=2,
            dt=10.0,
            max_wait=120.0,
            gap0=100.0,
            gap_alpha=1.0,
            seeds=[42, 43],
            algorithms=["sort_skill", "fifo", "greedy_nearest"],
            use_f3=False,
        )

        summary_df, solutions_df = run_experiments(
            config=cfg,
            quick=True,
            results_dir=tmp_path,
            verbose=False,
        )

        # Check returned DataFrames
        assert not summary_df.empty, "Summary DataFrame should not be empty"
        assert not solutions_df.empty, "Solutions DataFrame should not be empty"

        # Check required columns in summary_df
        required_summary_cols = {
            "scenario",
            "seed",
            "algorithm",
            "runtime_sec",
            "n_solutions",
            "feasibility_rate",
            "mean_f1",
            "mean_f2",
            "non_dominated_count",
        }
        assert required_summary_cols.issubset(set(summary_df.columns))

        # Check raw CSV persistence
        raw_summary_csv = tmp_path / "raw" / "experiments_summary.csv"
        raw_solutions_csv = tmp_path / "raw" / "solutions_detail.csv"
        assert raw_summary_csv.exists(), "experiments_summary.csv must exist in raw/"
        assert raw_solutions_csv.exists(), "solutions_detail.csv must exist in raw/"

        # Check front CSV files
        fronts_dir = tmp_path / "fronts"
        assert fronts_dir.exists(), "fronts/ directory must exist"
        front_files = list(fronts_dir.glob("*.csv"))
        assert len(front_files) >= 3, f"Expected at least 3 front files, got {len(front_files)}"

        # Check baseline figure generation
        figures_dir = tmp_path / "figures"
        assert figures_dir.exists(), "figures/ directory must exist"
        pareto_fig = figures_dir / "baselines_pareto_light.png"
        boxplot_fig = figures_dir / "baselines_boxplot_light.png"
        assert pareto_fig.exists(), "baselines_pareto_light.png must be created"
        assert boxplot_fig.exists(), "baselines_boxplot_light.png must be created"


def test_run_experiments_skips_unregistered_algorithm() -> None:
    """Verify that an unavailable algorithm is skipped gracefully without raising errors."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = Config(
            seeds=[42],
            algorithms=["non_existent_matcher_xyz", "sort_skill"],
        )

        summary_df, solutions_df = run_experiments(
            config=cfg,
            scenarios=["light"],
            seeds=[42],
            algorithms=["non_existent_matcher_xyz", "sort_skill"],
            results_dir=tmp_path,
            verbose=False,
        )

        # Only the available algorithm should be present in results
        assert "sort_skill" in summary_df["algorithm"].values
        assert "non_existent_matcher_xyz" not in summary_df["algorithm"].values


def test_run_experiments_cli_quick() -> None:
    """Verify that the CLI entrypoint executes cleanly with --quick and custom results dir."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        exit_code = main(["--quick", "--algorithms", "fifo", "--results-dir", tmp_dir])
        assert exit_code == 0
        summary_csv = Path(tmp_dir) / "raw" / "experiments_summary.csv"
        assert summary_csv.exists()

