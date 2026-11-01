# OWNER: M3 - do not edit unless you are M3
"""Unit tests and acceptance benchmarks for NSGA-II multi-objective optimization (Checkpoint G5)."""
from __future__ import annotations

import numpy as np
import pytest

from mm import registry
from mm.interfaces import Matcher
from mm.nsga2 import NSGA2Matcher, nsga2
from mm.objectives import evaluate
from mm.pareto import is_non_dominated
from mm.repair import repair
from mm.simulator import make_instance
from mm.types import Instance, Player, Solution
from mm.validate import is_feasible


def _make_baseline_solutions(inst: Instance) -> dict[str, Solution]:
    """Helper to retrieve or build contract-compliant baseline solutions for comparison."""
    baselines: dict[str, Solution] = {}
    n = len(inst.players)
    match_size = 2 * inst.k

    # 1. sort_skill
    try:
        matcher = registry.get("sort_skill")
        baselines["sort_skill"] = matcher.solve(inst)[0]
    except (KeyError, NotImplementedError, Exception):
        # Contract fixture: sort by skill, snake draft into matches
        sorted_p = sorted(range(n), key=lambda i: inst.players[i].skill)
        m = np.full(n, -1, dtype=int)
        t = np.zeros(n, dtype=int)
        for match_idx in range(n // match_size):
            block = sorted_p[match_idx * match_size : (match_idx + 1) * match_size]
            for r, p in enumerate(block):
                m[p] = match_idx
                t[p] = 0 if r in (0, 3) else 1
        baselines["sort_skill"] = repair(inst, Solution(match=m, team=t))

    # 2. fifo
    try:
        matcher = registry.get("fifo")
        baselines["fifo"] = matcher.solve(inst)[0]
    except (KeyError, NotImplementedError, Exception):
        # Contract fixture: longest wait first, snake draft
        sorted_p = sorted(range(n), key=lambda i: inst.players[i].wait, reverse=True)
        m = np.full(n, -1, dtype=int)
        t = np.zeros(n, dtype=int)
        for match_idx in range(n // match_size):
            block = sorted_p[match_idx * match_size : (match_idx + 1) * match_size]
            for r, p in enumerate(block):
                m[p] = match_idx
                t[p] = 0 if r in (0, 3) else 1
        baselines["fifo"] = repair(inst, Solution(match=m, team=t))

    # 3. greedy_nearest
    try:
        matcher = registry.get("greedy_nearest")
        baselines["greedy_nearest"] = matcher.solve(inst)[0]
    except (KeyError, NotImplementedError, Exception):
        # Contract fixture: start from longest waiting, add nearest skill
        unmatched = set(range(n))
        m = np.full(n, -1, dtype=int)
        t = np.zeros(n, dtype=int)
        match_idx = 0
        while len(unmatched) >= match_size:
            oldest = max(unmatched, key=lambda i: inst.players[i].wait)
            unmatched.remove(oldest)
            group = [oldest]
            for _ in range(match_size - 1):
                nearest = min(
                    unmatched,
                    key=lambda i: abs(inst.players[i].skill - inst.players[oldest].skill),
                )
                unmatched.remove(nearest)
                group.append(nearest)
            group.sort(key=lambda i: inst.players[i].skill)
            for r, p in enumerate(group):
                m[p] = match_idx
                t[p] = 0 if r in (0, 3) else 1
            match_idx += 1
        baselines["greedy_nearest"] = repair(inst, Solution(match=m, team=t))

    return baselines


def test_nsga2_registration() -> None:
    """Verify NSGA2Matcher is registered under 'nsga2' and satisfies Matcher protocol."""
    assert "nsga2" in registry.list_names()
    matcher = registry.get("nsga2")
    assert isinstance(matcher, Matcher)
    assert matcher.name == "nsga2"


def test_nsga2_empty_instance() -> None:
    """Verify NSGA2Matcher handles an empty queue gracefully."""
    inst = Instance(
        players=[],
        k=2,
        dt=10.0,
        max_wait=120.0,
        gap0=100.0,
        gap_alpha=1.0,
        seed=42,
    )
    matcher = registry.get("nsga2")
    solutions = matcher.solve(inst)
    assert len(solutions) == 1
    assert solutions[0].n_matches() == 0


def test_nsga2_small_instance_front_non_empty_and_all_feasible() -> None:
    """Acceptance test G5: On a small instance, front is non-empty and all solutions are feasible."""
    inst = make_instance(n_players=24, k=2, seed=42, scenario="light")
    matcher = registry.get("nsga2")

    solutions = matcher.solve(inst, pop_size=30, n_gen=15, seed=42)

    assert len(solutions) > 0, "Pareto front must be non-empty"
    for idx, sol in enumerate(solutions):
        assert is_feasible(inst, sol) is True, f"Solution {idx} in front violates constraints"


def test_nsga2_deterministic_for_fixed_seed() -> None:
    """Acceptance test G5: A run must be strictly deterministic for a given seed."""
    inst = make_instance(n_players=24, k=2, seed=123, scenario="light")
    matcher = registry.get("nsga2")

    run1 = matcher.solve(inst, pop_size=20, n_gen=10, seed=999)
    run2 = matcher.solve(inst, pop_size=20, n_gen=10, seed=999)

    assert len(run1) == len(run2), "Identical seeds must produce identical front lengths"

    F1 = np.array([evaluate(inst, s) for s in run1])
    F2 = np.array([evaluate(inst, s) for s in run2])
    np.testing.assert_allclose(F1, F2, atol=1e-5, err_msg="Objectives differ between deterministic runs")

    for s1, s2 in zip(run1, run2):
        np.testing.assert_array_equal(s1.match, s2.match)
        np.testing.assert_array_equal(s1.team, s2.team)


def test_nsga2_beats_or_equals_each_baseline() -> None:
    """Acceptance test G5: NSGA-II front must beat or equal each baseline in at least one objective."""
    inst = make_instance(n_players=24, k=2, seed=42, scenario="light")
    baselines = _make_baseline_solutions(inst)

    matcher = registry.get("nsga2")
    front = matcher.solve(inst, pop_size=40, n_gen=20, seed=42)
    assert len(front) > 0

    F_front = np.array([evaluate(inst, s) for s in front])

    for b_name, b_sol in baselines.items():
        b_fit = evaluate(inst, b_sol)
        beats_f1 = np.any(F_front[:, 0] <= b_fit[0] + 1e-4)
        beats_f2 = np.any(F_front[:, 1] <= b_fit[1] + 1e-4)
        assert (
            beats_f1 or beats_f2
        ), f"NSGA-II failed to beat or equal baseline '{b_name}' in at least one objective (baseline={b_fit})"


def test_nsga2_mutual_non_dominance() -> None:
    """Verify that all solutions on the returned front are mutually non-dominated."""
    inst = make_instance(n_players=24, k=2, seed=42, scenario="light")
    matcher = registry.get("nsga2")

    front = matcher.solve(inst, pop_size=30, n_gen=15, seed=42)
    assert len(front) > 0

    F_front = np.array([evaluate(inst, s) for s in front])
    nd_mask = is_non_dominated(F_front)

    assert np.all(nd_mask), "Returned front contains mutually dominated solutions"


def test_nsga2_with_use_f3() -> None:
    """Verify NSGA-II optimizes 3 objectives when use_f3=True."""
    inst = make_instance(n_players=24, k=2, seed=42, scenario="light")
    matcher = registry.get("nsga2")

    front = matcher.solve(inst, pop_size=20, n_gen=10, seed=42, use_f3=True)
    assert len(front) > 0

    for sol in front:
        assert is_feasible(inst, sol) is True
        fit = evaluate(inst, sol, use_f3=True)
        assert len(fit) == 3

