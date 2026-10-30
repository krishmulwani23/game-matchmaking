# OWNER: M3 - do not edit unless you are M3
"""Unit tests and acceptance benchmarks for constraint repair (Checkpoint G4)."""
from __future__ import annotations

import numpy as np
import pytest

from mm.encoding import decode, random_genome, random_feasible
from mm.repair import repair
from mm.simulator import make_instance
from mm.types import Instance, Player, Solution
from mm.validate import is_feasible, validate


def _create_handbuilt_instance(k: int = 1) -> Instance:
    """Create a deterministic 4-player instance with k=1 (2 players per match)."""
    players = [
        Player(id=0, skill=1500.0, wait=10.0, arrival=0.0, role=0, ping=30.0),
        Player(id=1, skill=1520.0, wait=5.0, arrival=1.0, role=1, ping=35.0),
        Player(id=2, skill=1490.0, wait=2.0, arrival=2.0, role=2, ping=40.0),
        Player(id=3, skill=1510.0, wait=1.0, arrival=3.0, role=1, ping=32.0),
    ]
    return Instance(
        players=players,
        k=k,
        dt=10.0,
        max_wait=120.0,
        gap0=50.0,
        gap_alpha=1.0,
        seed=42,
    )


def test_repair_already_feasible_solution() -> None:
    """Verify that an already feasible solution remains feasible after repair."""
    inst = _create_handbuilt_instance()
    sol = Solution(
        match=np.array([0, 0, 1, 1], dtype=int),
        team=np.array([0, 1, 0, 1], dtype=int),
    )
    assert is_feasible(inst, sol) is True

    repaired = repair(inst, sol)
    assert is_feasible(inst, repaired) is True
    assert repaired.n_matches() == 2


def test_repair_all_unmatched_without_starvation() -> None:
    """All unmatched remains feasible if no player exceeds max_wait."""
    inst = _create_handbuilt_instance()
    sol = Solution.empty(len(inst.players))
    assert is_feasible(inst, sol) is True

    repaired = repair(inst, sol)
    assert is_feasible(inst, repaired) is True
    assert repaired.n_matches() == 0


def test_repair_forces_starving_player_matching() -> None:
    """Repair must force-match a player whose wait + dt exceeds max_wait."""
    inst = _create_handbuilt_instance()
    # Player 0 has wait = 115.0 -> 115 + 10 = 125 > 120 max_wait
    inst.players[0].wait = 115.0

    # Start from an all-unmatched solution (violates starvation)
    sol = Solution.empty(len(inst.players))
    assert is_feasible(inst, sol) is False

    repaired = repair(inst, sol)
    assert is_feasible(inst, repaired) is True
    # Player 0 must be in a match
    assert repaired.match[0] >= 0
    assert repaired.n_matches() >= 1


def test_repair_resolves_extreme_skill_gap_violation() -> None:
    """Repair must resolve matches containing extreme skill outliers."""
    inst = _create_handbuilt_instance()
    # Make player 1 an extreme outlier (skill 3000 vs 1500, allowed gap ~60)
    inst.players[1].skill = 3000.0

    # Put player 0 and player 1 in Match 0 (violates skill gap)
    sol = Solution(
        match=np.array([0, 0, -1, -1], dtype=int),
        team=np.array([0, 1, 0, 0], dtype=int),
    )
    assert is_feasible(inst, sol) is False

    repaired = repair(inst, sol)
    violations = validate(inst, repaired)
    assert violations == []
    assert is_feasible(inst, repaired) is True


def test_repair_snake_draft_balancing() -> None:
    """Verify snake draft balances team skill assignments in a match."""
    # 4 players, k=2 (one 4-player match)
    # Skills: 1400, 1500, 1600, 1700
    players = [
        Player(id=0, skill=1400.0, wait=10.0, arrival=0.0),
        Player(id=1, skill=1500.0, wait=10.0, arrival=0.0),
        Player(id=2, skill=1600.0, wait=10.0, arrival=0.0),
        Player(id=3, skill=1700.0, wait=10.0, arrival=0.0),
    ]
    inst = Instance(
        players=players,
        k=2,
        dt=10.0,
        max_wait=120.0,
        gap0=500.0,
        gap_alpha=1.0,
        seed=42,
    )
    # Put all 4 players into match 0 with arbitrary teams
    sol = Solution(
        match=np.array([0, 0, 0, 0], dtype=int),
        team=np.array([0, 0, 0, 0], dtype=int),  # Invalid: all on team 0
    )
    assert is_feasible(inst, sol) is False

    repaired = repair(inst, sol)
    assert is_feasible(inst, repaired) is True

    # Check teams have exactly 2 players each
    t0_skills = [inst.players[i].skill for i in range(4) if repaired.team[i] == 0]
    t1_skills = [inst.players[i].skill for i in range(4) if repaired.team[i] == 1]
    assert len(t0_skills) == 2
    assert len(t1_skills) == 2

    # Snake draft pairs (1400, 1700) -> sum 3100, and (1500, 1600) -> sum 3100
    assert sum(t0_skills) == pytest.approx(3100.0)
    assert sum(t1_skills) == pytest.approx(3100.0)


def test_random_feasible_helper() -> None:
    """Verify random_feasible always returns a strictly feasible solution."""
    inst = make_instance(n_players=24, k=2, seed=999, scenario="light")
    rng = np.random.default_rng(888)

    for _ in range(10):
        sol = random_feasible(inst, rng=rng)
        assert is_feasible(inst, sol) is True


@pytest.mark.parametrize("scenario,n_samples", [
    ("light", 40),
    ("medium", 40),
    ("heavy", 20),
])
def test_repair_acceptance_criterion_feasibility_rate(scenario: str, n_samples: int) -> None:
    """Acceptance Test G4: After repair, validate.is_feasible must hold for >= 99% of random genomes."""
    inst = make_instance(scenario=scenario, seed=123)
    rng = np.random.default_rng(42)

    feasible_count = 0
    for _ in range(n_samples):
        # Generate raw random genome
        genome = random_genome(inst, rng=rng)
        raw_sol = decode(inst, genome)

        # Apply repair
        repaired_sol = repair(inst, raw_sol)

        if is_feasible(inst, repaired_sol):
            feasible_count += 1

    rate = feasible_count / n_samples
    assert rate >= 0.99, f"Feasibility rate for scenario '{scenario}' was {rate:.1%}, expected >= 99%"

