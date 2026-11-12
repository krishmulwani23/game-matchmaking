# OWNER: M2 - do not edit unless you are M2
"""Unit and integration tests for statistical analysis and conflict study (Checkpoint G13)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from mm.conflict_study import run_conflict_study, sweep_parameter
from mm.stats import (
    cliffs_delta,
    compare_algorithms,
    load_raw_results,
    pairwise_comparison_matrix,
    rank_biserial_correlation,
    wilcoxon_signed_rank_test,
)


# =========================================================================
# 1. Wilcoxon Signed-Rank Test & Effect Size Tests
# =========================================================================


def test_wilcoxon_identical_distributions() -> None:
    """When distributions are identical, test reports p=1.0 and zero effect size."""
    x = [10.0, 20.0, 30.0, 40.0, 50.0]
    y = [10.0, 20.0, 30.0, 40.0, 50.0]

    res = wilcoxon_signed_rank_test(x, y)
    assert res["p_value"] == 1.0
    assert res["rank_biserial"] == 0.0
    assert res["cliffs_delta"] == 0.0
    assert res["effect_size_r"] == 0.0
    assert res["n_nonzero"] == 0


def test_wilcoxon_strictly_dominant() -> None:
    """When x is strictly greater than y for all pairs, rank-biserial is 1.0."""
    x = [15.0, 25.0, 35.0, 45.0, 55.0]
    y = [10.0, 20.0, 30.0, 40.0, 50.0]

    res = wilcoxon_signed_rank_test(x, y)
    # W- is 0, so rank_biserial = (W+ - 0)/W+ = 1.0
    assert res["rank_biserial"] == 1.0
    assert res["cliffs_delta"] > 0.0
    assert res["p_value"] < 0.1
    assert res["n_nonzero"] == 5


def test_cliffs_delta_values() -> None:
    """Verify Cliff's delta on deterministic samples."""
    x1 = [10, 20, 30]
    x2 = [1, 2, 3]
    # Every element in x1 is strictly greater than every element in x2
    assert cliffs_delta(x1, x2) == 1.0
    assert cliffs_delta(x2, x1) == -1.0
    assert cliffs_delta(x1, x1) == 0.0


def test_wilcoxon_error_handling() -> None:
    """Wilcoxon test raises ValueError on unequal length inputs."""
    with pytest.raises(ValueError, match="identical length"):
        wilcoxon_signed_rank_test([1.0, 2.0], [1.0, 2.0, 3.0])

    # Empty inputs return default null record
    empty_res = wilcoxon_signed_rank_test([], [])
    assert empty_res["n_pairs"] == 0
    assert empty_res["p_value"] == 1.0


# =========================================================================
# 2. Reading Raw Results (RAM's G12 Artifacts)
# =========================================================================


def test_load_raw_results() -> None:
    """Verify reading results/raw/*.csv strictly read-only."""
    summary_df, detail_df = load_raw_results()
    assert not summary_df.empty, "results/raw/experiments_summary.csv must exist and have content"
    assert "algorithm" in summary_df.columns
    assert "scenario" in summary_df.columns
    assert "hypervolume" in summary_df.columns

    if not detail_df.empty:
        assert "algorithm" in detail_df.columns
        assert "f1_imbalance" in detail_df.columns


def test_compare_algorithms_from_raw() -> None:
    """Test pairwise comparison between two algorithms using raw results."""
    summary_df, _ = load_raw_results()
    assert not summary_df.empty

    comp = compare_algorithms(summary_df, "nsga2", "sort_skill", metric="hypervolume")
    assert isinstance(comp, dict)
    assert comp["algo_a"] == "nsga2"
    assert comp["algo_b"] == "sort_skill"
    assert "p_value" in comp
    assert "rank_biserial" in comp
    assert "cliffs_delta" in comp


def test_pairwise_comparison_matrix() -> None:
    """Verify generation of full pairwise comparison matrix."""
    summary_df, _ = load_raw_results()
    matrix = pairwise_comparison_matrix(summary_df, metric="hypervolume", algorithms=["nsga2", "sort_skill", "fifo"])

    assert isinstance(matrix, pd.DataFrame)
    assert len(matrix) == 3  # (3 choose 2 = 3 pairs)
    expected_cols = [
        "algo_a",
        "algo_b",
        "metric",
        "mean_a",
        "mean_b",
        "diff_mean",
        "statistic",
        "p_value",
        "rank_biserial",
        "cliffs_delta",
        "significant_05",
    ]
    for col in expected_cols:
        assert col in matrix.columns


# =========================================================================
# 3. Conflict Study Parameter Sweep Tests
# =========================================================================


def test_sweep_parameter() -> None:
    """Verify sweep_parameter executes over specified values and scenarios."""
    df_sweep = sweep_parameter(
        param_name="max_wait",
        param_values=[60.0, 120.0],
        scenarios=["light"],
        seeds=[42],
        baselines=["sort_skill", "fifo"],
    )

    assert isinstance(df_sweep, pd.DataFrame)
    # 2 values * 1 scenario * 1 seed * 2 baselines = 4 records
    assert len(df_sweep) == 4
    expected_cols = [
        "sweep_param",
        "param_value",
        "scenario",
        "seed",
        "algorithm",
        "f1_imbalance",
        "f2_wait",
        "match_rate",
        "is_feasible",
    ]
    for col in expected_cols:
        assert col in df_sweep.columns

    # All baseline solutions must be feasible
    assert df_sweep["is_feasible"].all()


def test_run_conflict_study_quick(tmp_path: Path) -> None:
    """Run full conflict study in quick mode with temporary directory output."""
    test_grid = {
        "max_wait": [60.0, 120.0],
        "k": [1, 2],
    }

    master_df, sweep_dfs = run_conflict_study(
        output_dir=tmp_path,
        scenarios=["light"],
        seeds=[42],
        baselines=["sort_skill", "greedy_nearest"],
        sweeps_dict=test_grid,
        verbose=False,
    )

    assert isinstance(master_df, pd.DataFrame)
    assert len(master_df) > 0
    assert (tmp_path / "conflict_study_table.csv").exists()
    assert (tmp_path / "sweep_max_wait.csv").exists()
    assert (tmp_path / "sweep_k.csv").exists()

