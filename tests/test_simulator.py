# OWNER: M1 - do not edit unless you are M1
"""Tests for mm.simulator synthetic data generation."""
from __future__ import annotations

import pytest

from mm.simulator import make_instance


def test_same_seed_gives_identical_instances() -> None:
    """Verify that identical seeds produce identical synthetic instances."""
    inst1 = make_instance(n_players=20, k=2, seed=42, scenario="medium")
    inst2 = make_instance(n_players=20, k=2, seed=42, scenario="medium")

    assert inst1.k == inst2.k
    assert inst1.dt == inst2.dt
    assert inst1.max_wait == inst2.max_wait
    assert inst1.gap0 == inst2.gap0
    assert inst1.gap_alpha == inst2.gap_alpha
    assert inst1.seed == inst2.seed
    assert len(inst1.players) == len(inst2.players)

    for p1, p2 in zip(inst1.players, inst2.players):
        assert p1.id == p2.id
        assert p1.skill == p2.skill
        assert p1.wait == p2.wait
        assert p1.arrival == p2.arrival
        assert p1.role == p2.role
        assert p1.ping == p2.ping


def test_different_seeds_give_different_instances() -> None:
    """Verify that distinct seeds produce statistically different player samples."""
    inst1 = make_instance(n_players=20, seed=42)
    inst2 = make_instance(n_players=20, seed=999)

    skills1 = [p.skill for p in inst1.players]
    skills2 = [p.skill for p in inst2.players]
    assert skills1 != skills2


def test_scenario_player_counts() -> None:
    """Verify preset scenario player counts (light ~24, medium ~100, heavy ~400)."""
    inst_light = make_instance(scenario="light", seed=1)
    inst_medium = make_instance(scenario="medium", seed=1)
    inst_heavy = make_instance(scenario="heavy", seed=1)

    assert len(inst_light.players) == 24
    assert len(inst_medium.players) == 100
    assert len(inst_heavy.players) == 400


def test_player_attribute_ranges() -> None:
    """Verify generated values respect natural domain boundaries."""
    inst = make_instance(n_players=50, seed=10)
    for p in inst.players:
        assert p.skill >= 100.0
        assert p.wait >= 0.0
        assert p.arrival >= 0.0
        assert p.role in (0, 1, 2)
        assert p.ping is not None and p.ping > 0.0


def test_zero_players_instance() -> None:
    """Verify edge case with 0 players."""
    inst = make_instance(n_players=0, seed=1)
    assert len(inst.players) == 0
