# OWNER: M1 - do not edit unless you are M1
"""Tests for mm.registry and Matcher protocol."""
from __future__ import annotations

import pytest

from mm.interfaces import Matcher
from mm.registry import _ensure_loaded, get, list_names, register
from mm.types import Instance, Player, Solution


def _make_dummy_instance() -> Instance:
    players = [
        Player(id=0, skill=1500.0, wait=0.0, arrival=0.0),
        Player(id=1, skill=1520.0, wait=0.0, arrival=0.0),
    ]
    return Instance(
        players=players,
        k=1,
        dt=10.0,
        max_wait=60.0,
        gap0=50.0,
        gap_alpha=1.0,
        seed=1,
    )


def test_registry_survives_missing_modules() -> None:
    """Registry must not crash when lazy modules are empty or missing."""
    names = list_names()
    assert isinstance(names, list)


def test_register_and_get_class() -> None:
    """Register a Matcher class and retrieve an instance with get()."""
    @register("test_dummy_class")
    class DummyClassMatcher:
        name = "test_dummy_class"

        def solve(self, inst: Instance, **kw) -> list[Solution]:
            return [Solution.empty(len(inst.players))]

    assert "test_dummy_class" in list_names()
    matcher = get("test_dummy_class")
    assert isinstance(matcher, Matcher)
    assert matcher.name == "test_dummy_class"

    inst = _make_dummy_instance()
    solutions = matcher.solve(inst)
    assert len(solutions) == 1
    assert solutions[0].n_matches() == 0


def test_register_instance_directly() -> None:
    """Register an already-instantiated matcher."""
    class CustomMatcher:
        name = "test_custom_instance"

        def solve(self, inst: Instance, **kw) -> list[Solution]:
            return [Solution.empty(len(inst.players))]

    instance_obj = CustomMatcher()
    register("test_custom_instance")(instance_obj)

    assert "test_custom_instance" in list_names()
    retrieved = get("test_custom_instance")
    assert retrieved is instance_obj

    inst = _make_dummy_instance()
    assert len(retrieved.solve(inst)) == 1


def test_get_unknown_matcher_raises_key_error() -> None:
    """Attempting to get an unregistered matcher must raise KeyError."""
    with pytest.raises(KeyError, match="not registered"):
        get("non_existent_matcher_xyz")
