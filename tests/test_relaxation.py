# OWNER: M4 - do not edit unless you are M4
"""Unit tests for adaptive skill gap relaxation and rule derivation (Checkpoint G12)."""
from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from mm.config import Config
from mm.relaxation import (
    RelaxationRule,
    derive_relaxation_rule,
    fit_relaxation_rule,
    generate_relaxation_comparison_table,
    plot_relaxation_rolling_validation,
    plot_relaxation_rule,
    validate_relaxation_rule,
)


def test_relaxation_rule_evaluation():
    """Verify scalar and vectorized evaluation of linear and piecewise rules."""
    rule_linear = RelaxationRule(gap0=100.0, alpha=1.25)
    assert rule_linear.evaluate(0.0) == 100.0
    assert rule_linear.evaluate(10.0) == 112.5
    assert rule_linear.evaluate(40.0) == 150.0
    assert "gap(w) = 100.0 + 1.25 * w" in rule_linear.formula_str

    arr = np.array([0.0, 20.0, 40.0])
    res = rule_linear.evaluate(arr)
    assert np.allclose(res, [100.0, 125.0, 150.0])

    # Piecewise grace period
    rule_piecewise = RelaxationRule(gap0=100.0, alpha=1.5, w_thresh=20.0)
    assert rule_piecewise.evaluate(10.0) == 100.0  # Within grace period
    assert rule_piecewise.evaluate(20.0) == 100.0
    assert rule_piecewise.evaluate(30.0) == 115.0  # (30 - 20) * 1.5 + 100
    assert "max(0, w - 20.0)" in rule_piecewise.formula_str


def test_fit_relaxation_rule():
    """Verify rule fitting from empirical wait times and skill spreads."""
    rng = np.random.default_rng(42)
    waits = rng.uniform(5.0, 100.0, size=50)
    # diffs = 100 + 1.2 * wait + noise
    diffs = 100.0 + 1.2 * waits + rng.normal(0, 5.0, size=50)

    rule = fit_relaxation_rule(waits, diffs, gap0=100.0, percentile=90.0)
    assert isinstance(rule, RelaxationRule)
    assert rule.gap0 == 100.0
    assert 0.8 <= rule.alpha <= 2.0


def test_generate_relaxation_comparison_table():
    """Verify that the comparison table generates required policies and columns."""
    rule = RelaxationRule(gap0=100.0, alpha=1.25)
    table = generate_relaxation_comparison_table(rule, gap0=100.0, max_wait=120.0)

    assert not table.empty
    expected_cols = {
        "policy",
        "alpha",
        "formula",
        "gap_at_w0",
        "gap_at_w60",
        "gap_at_w120",
        "fairness_rating",
        "queue_risk",
        "tradeoff_verdict",
    }
    assert expected_cols.issubset(set(table.columns))
    assert len(table) == 5  # 5 candidate policies evaluated
    assert "Fixed Strict Gap" in table["policy"].values
    assert "Recommended Adaptive" in table["policy"].values


def test_derive_relaxation_rule_end_to_end():
    """Verify derive_relaxation_rule with DataFrame input and disk output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        dummy_solutions = pd.DataFrame({
            "scenario": ["light"] * 10,
            "seed": [42] * 10,
            "algorithm": ["nsga2"] * 10,
            "f1_imbalance": [2.5, 3.0, 4.5, 6.0, 1.5, 8.0, 3.2, 5.1, 4.0, 2.0],
            "f2_wait": [150.0, 120.0, 90.0, 60.0, 200.0, 40.0, 110.0, 75.0, 95.0, 180.0],
        })

        rule, comp_df = derive_relaxation_rule(
            solutions_data=dummy_solutions,
            gap0=100.0,
            results_dir=tmp_path,
        )

        assert isinstance(rule, RelaxationRule)
        assert not comp_df.empty
        out_csv = tmp_path / "raw" / "relaxation_table.csv"
        if not out_csv.exists():
            out_csv = tmp_path / "relaxation_table.csv"
        assert out_csv.exists(), "relaxation_table.csv must be saved to results_dir"


def test_validate_relaxation_rule_quick():
    """Verify that validate_relaxation_rule runs rolling_sim and produces comparative metrics."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = Config(k=2, dt=10.0, max_wait=60.0, gap0=100.0, gap_alpha=1.0)
        rule = RelaxationRule(gap0=100.0, alpha=1.2)

        val_df = validate_relaxation_rule(
            rule=rule,
            scenarios=["light"],
            seeds=[42],
            fixed_alpha=0.0,
            n_rounds=3,  # Fast 3 rounds for unit test
            config=cfg,
            results_dir=tmp_path,
            verbose=False,
        )

        assert not val_df.empty
        assert "policy" in val_df.columns
        assert "mean_wait_sec" in val_df.columns
        assert "max_queue_len" in val_df.columns

        val_csv = tmp_path / "raw" / "relaxation_validation.csv"
        if not val_csv.exists():
            val_csv = tmp_path / "relaxation_validation.csv"
        assert val_csv.exists(), "relaxation_validation.csv must be written to disk"


def test_plot_relaxation_functions():
    """Verify that relaxation plotting utilities generate valid figures without errors."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        rule = RelaxationRule(gap0=100.0, alpha=1.25)

        # Plot 1: relaxation curve
        rule_fig_path = tmp_path / "test_rule_curve.png"
        fig1, ax1 = plot_relaxation_rule(
            rule=rule,
            empirical_waits=[10.0, 30.0, 50.0],
            empirical_diffs=[110.0, 140.0, 160.0],
            save_path=rule_fig_path,
        )
        assert rule_fig_path.exists()

        # Plot 2: rolling validation comparison
        val_fig_path = tmp_path / "test_rolling_val.png"
        dummy_val_df = pd.DataFrame([
            {"scenario": "light", "policy": "Fixed Gap (alpha=0)", "mean_wait_sec": 45.0, "max_queue_len": 24},
            {"scenario": "light", "policy": "Suggested Rule", "mean_wait_sec": 22.0, "max_queue_len": 12},
        ])
        fig2, axes2 = plot_relaxation_rolling_validation(
            validation_df=dummy_val_df,
            save_path=val_fig_path,
        )
        assert val_fig_path.exists()

