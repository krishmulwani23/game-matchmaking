# OWNER: M2 - do not edit unless you are M2
"""Unit tests and acceptance benchmarks for baseline matchers (Checkpoint G3).

Tests the three registered baseline Matchers:
- "sort_skill"
- "fifo"
- "greedy_nearest"

Verifies:
1. Registry registration and protocol compliance.
2. Invariants: match size (2k) and team balance (k vs k).
3. Hard constraint feasibility on synthetic instances across light/medium/heavy
   scenarios and 5 seeds (42, 43, 44, 45, 46).
4. Starvation constraint satisfaction.
5. Determinism on identical instances.
6. Edge case robustness (empty queue, n < 2k).
"""
from __future__ import annotations

import numpy as np
import pytest

from mm.baselines import _snake_draft_teams, fifo, greedy_nearest, sort_skill
from mm.config import load_config
from mm.interfaces import Matcher
from mm.objectives import evaluate, f1_imbalance, f2_wait
import mm.registry as registry
from mm.simulator import make_instance
from mm.types import Instance, Player, Solution
from mm.validate import is_feasible, validate

SEEDS = [42, 43, 44, 45, 46]
SCENARIOS = ["light", "medium", "heavy"]
BASELINE_NAMES = ["sort_skill", "fifo", "greedy_nearest"]


def _create_handbuilt_instance(k: int = 2) -> Instance:
    """Create a deterministic 8-player hand-built instance with k=2 (4 players/match)."""
    players = [
        Player(id=0, skill=1500.0, wait=10.0, arrival=0.0, role=0, ping=30.0),
        Player(id=1, skill=1510.0, wait=25.0, arrival=1.0, role=1, ping=35.0),
        Player(id=2, skill=1495.0, wait=5.0, arrival=2.0, role=2, ping=28.0),
        Player(id=3, skill=1520.0, wait=15.0, arrival=3.0, role=1, ping=32.0),
        Player(id=4, skill=1600.0, wait=8.0, arrival=4.0, role=0, ping=40.0),
        Player(id=5, skill=1590.0, wait=40.0, arrival=5.0, role=1, ping=42.0),
        Player(id=6, skill=1610.0, wait=30.0, arrival=6.0, role=2, ping=38.0),
        Player(id=7, skill=1605.0, wait=2.0, arrival=7.0, role=1, ping=45.0),
    ]
    return Instance(
        players=players,
        k=k,
        dt=10.0,
        max_wait=120.0,
        gap0=100.0,
        gap_alpha=1.0,
        seed=42,
    )


# =========================================================================
# 1. Registry & Matcher Protocol Tests
# =========================================================================


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_baseline_registered(name: str) -> None:
    """All baselines must be discoverable in mm.registry."""
    names = registry.list_names()
    assert name in names, f"Baseline '{name}' not found in registry: {names}"

    matcher = registry.get(name)
    assert isinstance(matcher, Matcher), f"'{name}' does not satisfy Matcher protocol"
    assert matcher.name == name


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_matcher_protocol_output(name: str) -> None:
    """Matcher.solve must return a list containing a single feasible Solution."""
    inst = _create_handbuilt_instance()
    matcher = registry.get(name)
    solutions = matcher.solve(inst)

    assert isinstance(solutions, list)
    assert len(solutions) == 1
    sol = solutions[0]
    assert isinstance(sol, Solution)
    assert len(sol.match) == len(inst.players)
    assert len(sol.team) == len(inst.players)
    assert is_feasible(inst, sol)


# =========================================================================
# 2. Snake Draft Helper Tests
# =========================================================================


def test_snake_draft_distribution() -> None:
    """Snake draft must produce exactly equal team splits for any 2k size."""
    for k in [1, 2, 3, 4, 5]:
        match_size = 2 * k
        teams = _snake_draft_teams(match_size)
        assert len(teams) == match_size
        assert teams.count(0) == k
        assert teams.count(1) == k

    # Known pattern for k=2 (size 4): [0, 1, 1, 0]
    assert _snake_draft_teams(4) == [0, 1, 1, 0]
    # Known pattern for k=1 (size 2): [0, 1]
    assert _snake_draft_teams(2) == [0, 1]


# =========================================================================
# 3. Acceptance Benchmark: Feasibility on Synthetic Instances
# =========================================================================


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_synthetic_feasibility(scenario: str, seed: int, name: str) -> None:
    """Acceptance check: all baselines return feasible solutions on 5 seeds."""
    inst = make_instance(scenario=scenario, seed=seed)
    matcher = registry.get(name)
    sol = matcher.solve(inst)[0]

    violations = validate(inst, sol)
    assert violations == [], (
        f"Baseline '{name}' produced {len(violations)} violations on {scenario} (seed {seed}):\n"
        + "\n".join(violations[:5])
    )
    assert is_feasible(inst, sol) is True


# =========================================================================
# 4. Invariants & Balance Tests
# =========================================================================


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_match_and_team_invariants(name: str) -> None:
    """Each formed match must contain exactly 2k players, k per team."""
    inst = make_instance(scenario="medium", seed=42)
    matcher = registry.get(name)
    sol = matcher.solve(inst)[0]

    matched_ids = np.unique(sol.match[sol.match >= 0])
    assert len(matched_ids) > 0, f"Baseline '{name}' formed 0 matches on medium scenario"

    for m in matched_ids:
        in_match = [i for i, mid in enumerate(sol.match) if mid == m]
        assert len(in_match) == 2 * inst.k
        team0 = sum(1 for i in in_match if sol.team[i] == 0)
        team1 = sum(1 for i in in_match if sol.team[i] == 1)
        assert team0 == inst.k
        assert team1 == inst.k


# =========================================================================
# 5. Starvation Constraint Handling
# =========================================================================


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_starving_player_forced_matching(name: str) -> None:
    """Players whose wait + dt exceeds max_wait must be matched."""
    inst = _create_handbuilt_instance()
    # Force player 1 to exceed max_wait
    inst.players[1].wait = 115.0  # 115 + 10 = 125 > 120 (max_wait)

    matcher = registry.get(name)
    sol = matcher.solve(inst)[0]

    assert is_feasible(inst, sol)
    assert sol.match[1] >= 0, f"Starving player 1 remained unmatched in '{name}'"


# =========================================================================
# 6. Determinism & Functional Calling Tests
# =========================================================================


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_baseline_determinism(name: str) -> None:
    """Running a baseline twice on the same instance yields identical solutions."""
    inst = make_instance(scenario="light", seed=42)
    matcher = registry.get(name)

    sol1 = matcher.solve(inst)[0]
    sol2 = matcher.solve(inst)[0]

    np.testing.assert_array_equal(sol1.match, sol2.match)
    np.testing.assert_array_equal(sol1.team, sol2.team)


def test_functional_interfaces() -> None:
    """Module-level functions sort_skill, fifo, greedy_nearest return feasible solutions."""
    inst = _create_handbuilt_instance()

    s_sol = sort_skill(inst)
    f_sol = fifo(inst)
    g_sol = greedy_nearest(inst)

    for sol in [s_sol, f_sol, g_sol]:
        assert isinstance(sol, Solution)
        assert is_feasible(inst, sol)


# =========================================================================
# 7. Edge Cases: Empty Queue & Insufficient Players
# =========================================================================


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_empty_instance(name: str) -> None:
    """Handling of empty instance (0 players)."""
    inst = Instance(
        players=[],
        k=2,
        dt=10.0,
        max_wait=120.0,
        gap0=100.0,
        gap_alpha=1.0,
        seed=42,
    )
    matcher = registry.get(name)
    sol = matcher.solve(inst)[0]

    assert len(sol.match) == 0
    assert len(sol.team) == 0
    assert sol.n_matches() == 0
    assert is_feasible(inst, sol)


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_insufficient_players(name: str) -> None:
    """Fewer players than match size (2k) results in 0 matches and all unmatched."""
    # 3 players with k=2 (needs 4 players for a match)
    players = [
        Player(id=0, skill=1500.0, wait=10.0, arrival=0.0),
        Player(id=1, skill=1510.0, wait=15.0, arrival=1.0),
        Player(id=2, skill=1490.0, wait=20.0, arrival=2.0),
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
    matcher = registry.get(name)
    sol = matcher.solve(inst)[0]

    assert sol.n_matches() == 0
    assert np.all(sol.match == -1)
    assert is_feasible(inst, sol)

