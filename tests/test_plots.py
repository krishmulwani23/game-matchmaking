# OWNER: M4 - do not edit unless you are M4
"""Unit tests for visualization utilities (mm.plots)."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mm.plots import (
    boxplot_wait_imbalance,
    boxplot_wait_vs_imbalance,
    pareto_plot,
    tradeoff_curve,
)


@pytest.fixture(autouse=True)
def close_plots() -> None:
    """Ensure matplotlib figures are cleaned up after each test."""
    yield
    plt.close("all")


def test_pareto_plot_synthetic(tmp_path: Path) -> None:
    """Test pareto_plot generation and file output on synthetic algorithm fronts."""
    fronts = {
        "sort_skill": np.array([[12.0, 120.0], [18.0, 95.0], [25.0, 75.0]]),
        "nsga2": np.array([
            [8.0, 110.0],
            [14.0, 70.0],
            [20.0, 50.0],
            [16.0, 85.0],  # dominated point to verify faint scatter
        ]),
    }
    save_file = tmp_path / "pareto_test.png"
    fig, ax = pareto_plot(fronts, save_path=save_file)

    assert fig is not None
    assert ax is not None
    assert save_file.exists()
    assert save_file.stat().st_size > 0


def test_tradeoff_curve_synthetic(tmp_path: Path) -> None:
    """Test tradeoff_curve plotting operating points on a synthetic front."""
    F = np.array([
        [1.0, 10.0],
        [2.0, 5.0],
        [3.0, 3.0],
        [5.0, 2.0],
        [10.0, 1.0],
    ])
    save_file = tmp_path / "tradeoff_test.png"
    fig, ax = tradeoff_curve(F, name="Synthetic GA", save_path=save_file)

    assert fig is not None
    assert ax is not None
    assert save_file.exists()
    assert save_file.stat().st_size > 0


def test_tradeoff_curve_empty_raises() -> None:
    """Empty front must raise ValueError."""
    with pytest.raises(ValueError, match="Cannot plot trade-off curve for empty points"):
        tradeoff_curve(np.empty((0, 2)))


def test_boxplot_wait_vs_imbalance_dataframe(tmp_path: Path) -> None:
    """Test boxplot_wait_vs_imbalance with pandas DataFrame input."""
    # Synthetic multi-seed benchmark results
    records = []
    rng = np.random.default_rng(42)
    for algo, mean_imb, mean_w in [("sort_skill", 15.0, 100.0), ("fifo", 35.0, 45.0), ("nsga2", 18.0, 50.0)]:
        for seed in range(5):
            records.append({
                "algorithm": algo,
                "seed": seed,
                "f1_imbalance": mean_imb + rng.normal(0, 2),
                "f2_wait": mean_w + rng.normal(0, 5),
            })
    df = pd.DataFrame(records)

    save_file = tmp_path / "boxplot_df.png"
    fig, (ax1, ax2) = boxplot_wait_vs_imbalance(df, save_path=save_file)

    assert fig is not None
    assert ax1 is not None and ax2 is not None
    assert save_file.exists()
    assert save_file.stat().st_size > 0


def test_boxplot_wait_vs_imbalance_dict(tmp_path: Path) -> None:
    """Test boxplot_wait_vs_imbalance with nested dictionary input."""
    data = {
        "baseline_sort": {
            "imbalance": [10.5, 11.2, 10.8, 12.0],
            "wait": [95.0, 102.0, 98.0, 105.0],
        },
        "baseline_fifo": {
            "imbalance": [25.0, 27.5, 24.2, 26.0],
            "wait": [45.0, 42.0, 46.0, 44.0],
        },
    }
    save_file = tmp_path / "boxplot_dict.png"
    fig, axes = boxplot_wait_vs_imbalance(data, save_path=save_file)

    assert fig is not None
    assert len(axes) == 2
    assert save_file.exists()
    assert save_file.stat().st_size > 0


def test_boxplot_alias() -> None:
    """Verify alias boxplot_wait_imbalance points to the same function."""
    assert boxplot_wait_imbalance is boxplot_wait_vs_imbalance
