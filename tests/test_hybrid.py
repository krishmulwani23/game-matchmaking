# OWNER: M3 - do not edit unless you are M3
"""Tests for Checkpoint G11: Hybrid Greedy + Local Search algorithm."""

import numpy as np
import pytest

from mm import registry
from mm.interfaces import Matcher
from mm.objectives import evaluate
from mm.simulator import make_instance
from mm.types import Instance, Player, Solution
from mm.validate import is_feasible, validate
from mm.hybrid import hybrid_greedy_ls, HybridGreedyLSMatcher, local_search_swap


def test_registration():
    """hybrid_greedy_ls must be registered and implement the Matcher protocol."""
    assert "hybrid_greedy_ls" in registry.list_names()
    matcher = registry.get("hybrid_greedy_ls")
    assert isinstance(matcher, Matcher)
    assert matcher.name == "hybrid_greedy_ls"


def test_empty_instance():
    """Hybrid matcher handles an empty queue gracefully."""
    inst = Instance(
        players=[],
        k=2,
        dt=10.0,
        max_wait=120.0,
        gap0=100.0,
        gap_alpha=1.0,
        seed=42,
    )
    matcher = registry.get("hybrid_greedy_ls")
    sols = matcher.solve(inst, seed=42)
    assert len(sols) == 1
    assert is_feasible(inst, sols[0])
    assert len(sols[0].match) == 0


def test_insufficient_players():
    """When players < 2k, returns a single valid solution with all players unmatched."""
    players = [
        Player(id=0, skill=1500.0, wait=10.0, arrival=0.0),
        Player(id=1, skill=1520.0, wait=20.0, arrival=0.0),
    ]
    inst = Instance(
        players=players,
        k=2,
        dt=10.0,
        max_wait=120.0,
        gap0=100.0,
        gap_alpha=1.0,
        seed=42,
    )
    matcher = registry.get("hybrid_greedy_ls")
    sols = matcher.solve(inst, seed=42)
    assert len(sols) == 1
    assert is_feasible(inst, sols[0])
    assert np.all(sols[0].match == -1)


def test_baseline_seeding_and_no_ls():
    """When enable_ls=False, seeded candidates are returned directly without local search."""
    inst = make_instance(24, k=2, seed=42)
    matcher = registry.get("hybrid_greedy_ls")

    sols_no_ls = matcher.solve(inst, seed=42, enable_ls=False)
    assert len(sols_no_ls) > 0
    for s in sols_no_ls:
        assert is_feasible(inst, s)


def test_all_returned_solutions_are_feasible():
    """All solutions in the returned Pareto front must be strictly feasible."""
    for n in [24, 60]:
        inst = make_instance(n, k=2, seed=123)
        matcher = registry.get("hybrid_greedy_ls")
        sols = matcher.solve(inst, seed=123, max_evals=50)
        assert len(sols) > 0
        for s in sols:
            violations = validate(inst, s)
            assert len(violations) == 0, f"Violations found: {violations}"
            assert is_feasible(inst, s)


def test_starvation_feasibility():
    """Starving players must be strictly matched in all returned solutions."""
    inst = make_instance(32, k=2, seed=99)
    # Force two players to be starving
    inst.players[0].wait = 115.0
    inst.players[1].wait = 118.0

    matcher = registry.get("hybrid_greedy_ls")
    sols = matcher.solve(inst, seed=99, max_evals=50)
    assert len(sols) > 0
    for s in sols:
        assert is_feasible(inst, s)
        assert s.match[0] >= 0, "Starving player 0 was not matched"
        assert s.match[1] >= 0, "Starving player 1 was not matched"


def test_local_search_improves_or_preserves_fitness():
    """Local search must never accept an inferior solution on the scalarized objective."""
    inst = make_instance(40, k=2, seed=42)
    # Seed with fifo baseline
    fifo = registry.get("fifo")
    initial_sol = fifo.solve(inst)[0]
    assert is_feasible(inst, initial_sol)

    weight = 0.8
    # Baseline bounds for normalization
    from mm.weighted import compute_baseline_normalization
    bounds = compute_baseline_normalization(inst)

    improved_sol = local_search_swap(
        inst=inst,
        initial_sol=initial_sol,
        weight=weight,
        bounds=bounds,
        max_evals=100,
        rng=np.random.default_rng(42),
    )

    f1_min, f1_max, f2_min, f2_max = bounds
    f_init = evaluate(inst, initial_sol)
    f1_norm_init = (f_init[0] - f1_min) / max(f1_max - f1_min, 1.0)
    f2_norm_init = (f_init[1] - f2_min) / max(f2_max - f2_min, 1.0)
    j_init = weight * f1_norm_init + (1.0 - weight) * f2_norm_init

    f_imp = evaluate(inst, improved_sol)
    f1_norm_imp = (f_imp[0] - f1_min) / max(f1_max - f1_min, 1.0)
    f2_norm_imp = (f_imp[1] - f2_min) / max(f2_max - f2_min, 1.0)
    j_imp = weight * f1_norm_imp + (1.0 - weight) * f2_norm_imp

    assert is_feasible(inst, improved_sol)
    assert j_imp <= j_init + 1e-9


def test_deterministic_for_fixed_seed():
    """Hybrid matcher must be strictly deterministic when provided the same seed."""
    inst = make_instance(36, k=2, seed=77)
    matcher = registry.get("hybrid_greedy_ls")

    sols_a = matcher.solve(inst, seed=42, max_evals=50)
    sols_b = matcher.solve(inst, seed=42, max_evals=50)

    assert len(sols_a) == len(sols_b)
    for s_a, s_b in zip(sols_a, sols_b):
        np.testing.assert_array_equal(s_a.match, s_b.match)
        np.testing.assert_array_equal(s_a.team, s_b.team)


def test_non_dominated_front():
    """All solutions returned on the front must be mutually non-dominated."""
    inst = make_instance(40, k=2, seed=42)
    matcher = registry.get("hybrid_greedy_ls")
    sols = matcher.solve(inst, seed=42, max_evals=60)

    F = np.array([evaluate(inst, s) for s in sols])
    n = len(sols)
    for i in range(n):
        for j in range(n):
            if i != j:
                # i should not strictly dominate j if both are in non-dominated set
                strictly_better_in_all = np.all(F[i] <= F[j]) and np.any(F[i] < F[j])
                assert not strictly_better_in_all, f"Solution {i} dominates solution {j}"
