# OWNER: M3 - do not edit unless you are M3
"""Unit tests and acceptance benchmarks for weighted optimization algorithms (Checkpoint G6)."""
from __future__ import annotations

import numpy as np
import pytest

from mm import registry
from mm.interfaces import Matcher
from mm.objectives import evaluate
from mm.simulator import make_instance
from mm.types import Instance, Player, Solution
from mm.validate import is_feasible
from mm.weighted import (
    GAWeightedMatcher,
    SAWeightedMatcher,
    compute_baseline_normalization,
    ga_weighted,
    sa_weighted,
    sweep_weights,
)


def _make_light_instance(seed: int = 42) -> Instance:
    """Helper to generate a light test instance."""
    return make_instance(n_players=24, k=2, seed=seed, scenario="light")


def test_weighted_matchers_registration() -> None:
    """Verify sa_weighted and ga_weighted are registered and adhere to Matcher protocol."""
    names = registry.list_names()
    assert "sa_weighted" in names
    assert "ga_weighted" in names

    sa_matcher = registry.get("sa_weighted")
    assert isinstance(sa_matcher, Matcher)
    assert sa_matcher.name == "sa_weighted"

    ga_matcher = registry.get("ga_weighted")
    assert isinstance(ga_matcher, Matcher)
    assert ga_matcher.name == "ga_weighted"


def test_weighted_empty_instance() -> None:
    """Verify both algorithms handle an empty queue gracefully."""
    inst = Instance(
        players=[],
        k=2,
        dt=10.0,
        max_wait=120.0,
        gap0=100.0,
        gap_alpha=1.0,
        seed=42,
    )
    sa = registry.get("sa_weighted")
    ga = registry.get("ga_weighted")

    sol_sa = sa.solve(inst)
    assert len(sol_sa) == 1
    assert sol_sa[0].n_matches() == 0

    sol_ga = ga.solve(inst)
    assert len(sol_ga) == 1
    assert sol_ga[0].n_matches() == 0


def test_baseline_normalization_bounds() -> None:
    """Verify baseline normalization bounds computation produces valid positive ranges."""
    inst = _make_light_instance(seed=42)
    f1_min, f1_max, f2_min, f2_max = compute_baseline_normalization(inst)

    assert f1_min >= 0.0
    assert f1_max >= f1_min
    assert f2_min >= 0.0
    assert f2_max >= f2_min


def test_sa_weighted_w1_favors_minimum_imbalance() -> None:
    """Acceptance test G6: For SA, w=1 favors minimum imbalance over w=0."""
    inst = _make_light_instance(seed=42)
    sa = registry.get("sa_weighted")

    sol_w1 = sa.solve(inst, weight=1.0, seed=123)[0]
    sol_w0 = sa.solve(inst, weight=0.0, seed=123)[0]

    eval_w1 = evaluate(inst, sol_w1)
    eval_w0 = evaluate(inst, sol_w0)

    # w=1 must have lower or equal imbalance (f1) compared to w=0
    assert eval_w1[0] <= eval_w0[0], f"SA w=1 f1 ({eval_w1[0]}) should be <= w=0 f1 ({eval_w0[0]})"
    assert is_feasible(inst, sol_w1) is True
    assert is_feasible(inst, sol_w0) is True


def test_sa_weighted_w0_favors_minimum_wait() -> None:
    """Acceptance test G6: For SA, w=0 favors minimum waiting time over w=1."""
    inst = _make_light_instance(seed=42)
    sa = registry.get("sa_weighted")

    sol_w1 = sa.solve(inst, weight=1.0, seed=123)[0]
    sol_w0 = sa.solve(inst, weight=0.0, seed=123)[0]

    eval_w1 = evaluate(inst, sol_w1)
    eval_w0 = evaluate(inst, sol_w0)

    # w=0 must have lower or equal waiting time (f2) compared to w=1
    assert eval_w0[1] <= eval_w1[1], f"SA w=0 f2 ({eval_w0[1]}) should be <= w=1 f2 ({eval_w1[1]})"


def test_ga_weighted_w1_favors_minimum_imbalance() -> None:
    """Acceptance test G6: For GA, w=1 favors minimum imbalance over w=0."""
    inst = _make_light_instance(seed=42)
    ga = registry.get("ga_weighted")

    sol_w1 = ga.solve(inst, weight=1.0, seed=456)[0]
    sol_w0 = ga.solve(inst, weight=0.0, seed=456)[0]

    eval_w1 = evaluate(inst, sol_w1)
    eval_w0 = evaluate(inst, sol_w0)

    assert eval_w1[0] <= eval_w0[0], f"GA w=1 f1 ({eval_w1[0]}) should be <= w=0 f1 ({eval_w0[0]})"
    assert is_feasible(inst, sol_w1) is True
    assert is_feasible(inst, sol_w0) is True


def test_ga_weighted_w0_favors_minimum_wait() -> None:
    """Acceptance test G6: For GA, w=0 favors minimum waiting time over w=1."""
    inst = _make_light_instance(seed=42)
    ga = registry.get("ga_weighted")

    sol_w1 = ga.solve(inst, weight=1.0, seed=456)[0]
    sol_w0 = ga.solve(inst, weight=0.0, seed=456)[0]

    eval_w1 = evaluate(inst, sol_w1)
    eval_w0 = evaluate(inst, sol_w0)

    assert eval_w0[1] <= eval_w1[1], f"GA w=0 f2 ({eval_w0[1]}) should be <= w=1 f2 ({eval_w1[1]})"


def test_sa_weighted_deterministic_per_seed() -> None:
    """Acceptance test G6: Simulated Annealing is strictly deterministic for a fixed seed."""
    inst = _make_light_instance(seed=99)
    sa = registry.get("sa_weighted")

    sol_run1 = sa.solve(inst, weight=0.5, seed=777)[0]
    sol_run2 = sa.solve(inst, weight=0.5, seed=777)[0]

    np.testing.assert_array_equal(sol_run1.match, sol_run2.match)
    np.testing.assert_array_equal(sol_run1.team, sol_run2.team)
    np.testing.assert_allclose(
        evaluate(inst, sol_run1), evaluate(inst, sol_run2), atol=1e-5
    )


def test_ga_weighted_deterministic_per_seed() -> None:
    """Acceptance test G6: Genetic Algorithm is strictly deterministic for a fixed seed."""
    inst = _make_light_instance(seed=99)
    ga = registry.get("ga_weighted")

    sol_run1 = ga.solve(inst, weight=0.5, seed=888)[0]
    sol_run2 = ga.solve(inst, weight=0.5, seed=888)[0]

    np.testing.assert_array_equal(sol_run1.match, sol_run2.match)
    np.testing.assert_array_equal(sol_run1.team, sol_run2.team)
    np.testing.assert_allclose(
        evaluate(inst, sol_run1), evaluate(inst, sol_run2), atol=1e-5
    )


def test_sweep_weights_sa() -> None:
    """Verify sweep_weights returns the expected number of feasible solutions across the grid."""
    inst = _make_light_instance(seed=42)
    grid = np.linspace(0.0, 1.0, 11)

    solutions = sweep_weights(inst, weights=grid, algorithm="sa_weighted", seed=42)
    assert len(solutions) == 11
    for sol in solutions:
        assert is_feasible(inst, sol) is True


def test_sweep_weights_ga() -> None:
    """Verify sweep_weights works with GA and returns feasible solutions."""
    inst = _make_light_instance(seed=42)
    grid = np.linspace(0.0, 1.0, 5)

    solutions = sweep_weights(inst, weights=grid, algorithm="ga_weighted", seed=42)
    assert len(solutions) == 5
    for sol in solutions:
        assert is_feasible(inst, sol) is True


def test_matcher_solve_sweep_kwarg() -> None:
    """Verify passing weights to solve() returns multi-solution front."""
    inst = _make_light_instance(seed=42)
    sa = registry.get("sa_weighted")
    ga = registry.get("ga_weighted")

    grid = np.linspace(0.0, 1.0, 5)
    sols_sa = sa.solve(inst, weights=grid, seed=42)
    assert len(sols_sa) == 5

    sols_ga = ga.solve(inst, weights=grid, seed=42)
    assert len(sols_ga) == 5

