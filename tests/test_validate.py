# OWNER: M1 - do not edit unless you are M1
"""Tests for mm.validate constraint validator."""
from __future__ import annotations

import numpy as np
import pytest

from mm.types import Instance, Player, Solution
from mm.validate import is_feasible, validate


def _create_handbuilt_instance() -> Instance:
    """Create a small 4-player hand-built instance with k=1 (2 players per match)."""
    players = [
        Player(id=0, skill=1500.0, wait=10.0, arrival=0.0, role=0, ping=30.0),
        Player(id=1, skill=1520.0, wait=5.0, arrival=1.0, role=1, ping=35.0),
        Player(id=2, skill=1490.0, wait=2.0, arrival=2.0, role=2, ping=40.0),
        Player(id=3, skill=1510.0, wait=1.0, arrival=3.0, role=1, ping=32.0),
    ]
    return Instance(
        players=players,
        k=1,
        dt=10.0,
        max_wait=120.0,
        gap0=50.0,
        gap_alpha=1.0,
        seed=42,
    )


def test_feasible_hand_built_solution() -> None:
    """Verify that a properly constructed match configuration is feasible."""
    inst = _create_handbuilt_instance()
    # Match 0: player 0 (team 0) and player 1 (team 1)
    # Match 1: player 2 (team 0) and player 3 (team 1)
    sol = Solution(
        match=np.array([0, 0, 1, 1], dtype=int),
        team=np.array([0, 1, 0, 1], dtype=int),
    )

    violations = validate(inst, sol)
    assert violations == []
    assert is_feasible(inst, sol) is True


def test_empty_solution_feasible_when_no_starvation() -> None:
    """All unmatched players is feasible if no player exceeds max_wait."""
    inst = _create_handbuilt_instance()
    sol = Solution.empty(len(inst.players))
    assert is_feasible(inst, sol) is True


def test_violating_match_size() -> None:
    """Constraint violation: Match does not have exactly 2k players."""
    inst = _create_handbuilt_instance()  # k=1, expects 2 per match
    # Put 3 players into match 0
    sol = Solution(
        match=np.array([0, 0, 0, -1], dtype=int),
        team=np.array([0, 1, 0, 0], dtype=int),
    )

    violations = validate(inst, sol)
    assert len(violations) > 0
    assert any("has 3 players; expected exactly 2" in v for v in violations)
    assert is_feasible(inst, sol) is False


def test_violating_team_balance() -> None:
    """Constraint violation: Match does not have k players per team."""
    inst = _create_handbuilt_instance()  # k=1, expects 1 on team 0 and 1 on team 1
    # Match 0 has 2 players, but both are placed on team 0
    sol = Solution(
        match=np.array([0, 0, -1, -1], dtype=int),
        team=np.array([0, 0, 0, 0], dtype=int),
    )

    violations = validate(inst, sol)
    assert len(violations) > 0
    assert any("on team 0 and 0 on team 1" in v for v in violations)
    assert is_feasible(inst, sol) is False


def test_violating_starvation_max_wait() -> None:
    """Constraint violation: Player with wait + dt > max_wait left unmatched."""
    inst = _create_handbuilt_instance()
    # Modify player 0 wait to 115.0 (115 + 10 = 125 > 120 max_wait)
    inst.players[0].wait = 115.0

    # Leave player 0 unmatched
    sol = Solution.empty(len(inst.players))

    violations = validate(inst, sol)
    assert len(violations) > 0
    assert any("must be matched, but remains unmatched" in v for v in violations)
    assert is_feasible(inst, sol) is False


def test_violating_skill_gap() -> None:
    """Constraint violation: Skill gap between players exceeds relaxed limit."""
    inst = _create_handbuilt_instance()
    # gap0 = 50.0, gap_alpha = 1.0, max(wait) = 10.0 -> max allowed gap = 60.0
    # Set player 1 skill to 1700 (skill difference |1500 - 1700| = 200.0 > 60.0)
    inst.players[1].skill = 1700.0

    sol = Solution(
        match=np.array([0, 0, -1, -1], dtype=int),
        team=np.array([0, 1, 0, 0], dtype=int),
    )

    violations = validate(inst, sol)
    assert len(violations) > 0
    assert any("skill gap between player 0" in v and "> allowed" in v for v in violations)
    assert is_feasible(inst, sol) is False


def test_violating_ping_gap() -> None:
    """Constraint violation: Ping difference in a match exceeds max_ping_gap."""
    inst = _create_handbuilt_instance()
    # Set player 0 ping = 20.0, player 1 ping = 150.0 (gap = 130.0 > 100.0)
    inst.players[0].ping = 20.0
    inst.players[1].ping = 150.0

    sol = Solution(
        match=np.array([0, 0, -1, -1], dtype=int),
        team=np.array([0, 1, 0, 0], dtype=int),
    )

    violations = validate(inst, sol, max_ping_gap=100.0)
    assert len(violations) > 0
    assert any("ping gap 130.0 ms exceeds configured limit 100.0 ms" in v for v in violations)
    assert is_feasible(inst, sol, max_ping_gap=100.0) is False


def test_violating_array_dimensions() -> None:
    """Constraint violation: Solution array shape does not match player count."""
    inst = _create_handbuilt_instance()
    sol = Solution(
        match=np.array([0, 0], dtype=int),  # 2 instead of 4
        team=np.array([0, 1], dtype=int),
    )

    violations = validate(inst, sol)
    assert len(violations) > 0
    assert is_feasible(inst, sol) is False
