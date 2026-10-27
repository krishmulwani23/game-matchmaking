# OWNER: M1 - do not edit unless you are M1
"""Tests for mm.types data structures and helpers."""
from __future__ import annotations

import numpy as np
import pytest

from mm.types import Instance, Player, Solution


def test_player_instantiation_defaults() -> None:
    """Verify Player fields and default optional role/ping."""
    p = Player(id=1, skill=1500.0, wait=12.5, arrival=3.0)
    assert p.id == 1
    assert p.skill == 1500.0
    assert p.wait == 12.5
    assert p.arrival == 3.0
    assert p.role is None
    assert p.ping is None


def test_player_instantiation_with_optionals() -> None:
    """Verify Player with explicit role and ping values."""
    p = Player(id=2, skill=1620.0, wait=0.0, arrival=5.0, role=1, ping=42.0)
    assert p.role == 1
    assert p.ping == 42.0


def test_instance_instantiation() -> None:
    """Verify Instance fields and container attributes."""
    players = [
        Player(id=0, skill=1400.0, wait=5.0, arrival=0.0),
        Player(id=1, skill=1600.0, wait=10.0, arrival=0.0),
        Player(id=2, skill=1550.0, wait=2.0, arrival=1.0),
        Player(id=3, skill=1450.0, wait=8.0, arrival=1.0),
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
    assert len(inst.players) == 4
    assert inst.k == 2
    assert inst.dt == 10.0
    assert inst.max_wait == 120.0
    assert inst.gap0 == 100.0
    assert inst.gap_alpha == 1.0
    assert inst.seed == 42


def test_solution_empty() -> None:
    """Verify Solution.empty creates an all-unmatched solution."""
    sol = Solution.empty(6)
    assert isinstance(sol.match, np.ndarray)
    assert isinstance(sol.team, np.ndarray)
    assert sol.match.shape == (6,)
    assert sol.team.shape == (6,)
    assert np.all(sol.match == -1)
    assert sol.n_matches() == 0


def test_solution_empty_zero() -> None:
    """Verify Solution.empty works for n=0 players."""
    sol = Solution.empty(0)
    assert sol.match.shape == (0,)
    assert sol.n_matches() == 0


def test_solution_n_matches() -> None:
    """Verify Solution.n_matches counts unique non-negative match IDs."""
    # All unmatched
    sol_none = Solution(
        match=np.array([-1, -1, -1, -1]),
        team=np.zeros(4, dtype=int),
    )
    assert sol_none.n_matches() == 0

    # Single match (match index 0)
    sol_one = Solution(
        match=np.array([0, 0, 0, 0, -1]),
        team=np.array([0, 0, 1, 1, 0]),
    )
    assert sol_one.n_matches() == 1

    # Two matches
    sol_two = Solution(
        match=np.array([0, 0, 1, 1, -1, -1]),
        team=np.array([0, 1, 0, 1, 0, 0]),
    )
    assert sol_two.n_matches() == 2

    # Non-contiguous match indices (e.g. 5 and 9)
    sol_non_contiguous = Solution(
        match=np.array([5, 5, 9, 9, -1]),
        team=np.array([0, 1, 0, 1, 0]),
    )
    assert sol_non_contiguous.n_matches() == 2


def test_config_load() -> None:
    """Verify loading configuration from default.yaml."""
    from mm.config import Config, load_config

    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.k == 2
    assert cfg.dt == 10.0
    assert cfg.max_wait == 120.0
    assert cfg.gap0 == 100.0
    assert cfg.gap_alpha == 1.0
    assert cfg.queue_sizes.light == 24
    assert cfg.queue_sizes.medium == 100
    assert cfg.queue_sizes.heavy == 400
    assert cfg.seeds == [42, 43, 44, 45, 46]
    assert "sort_skill" in cfg.algorithms
    assert cfg.use_f3 is False

