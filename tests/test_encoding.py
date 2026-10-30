# OWNER: M3 - do not edit unless you are M3
"""Unit tests for chromosome encoding, decoding, and genetic operators (Checkpoint G4)."""
from __future__ import annotations

import numpy as np
import pytest

from mm.encoding import (
    Genome,
    crossover_ox,
    decode,
    encode,
    get_match_bounds,
    mutate_insert,
    mutate_inversion,
    mutate_swap,
    random_feasible,
    random_genome,
    random_permutation,
)
from mm.simulator import make_instance
from mm.types import Instance, Player, Solution
from mm.validate import is_feasible, validate


def _create_handbuilt_instance() -> Instance:
    """Create a deterministic 6-player instance with k=1 (2 players per match)."""
    players = [
        Player(id=0, skill=1500.0, wait=10.0, arrival=0.0),
        Player(id=1, skill=1520.0, wait=5.0, arrival=1.0),
        Player(id=2, skill=1490.0, wait=2.0, arrival=2.0),
        Player(id=3, skill=1510.0, wait=1.0, arrival=3.0),
        Player(id=4, skill=1505.0, wait=0.0, arrival=4.0),
        Player(id=5, skill=1495.0, wait=0.0, arrival=5.0),
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


def test_genome_instantiation_and_copy() -> None:
    """Verify Genome creation, properties, and deep copy."""
    perm = np.array([2, 0, 1, 3], dtype=int)
    g = Genome(perm=perm, n_matches=1)
    assert len(g) == 4
    assert g.n_matches == 1
    np.testing.assert_array_equal(g.perm, perm)

    g_copy = g.copy()
    assert g_copy is not g
    assert g_copy.perm is not g.perm
    np.testing.assert_array_equal(g_copy.perm, g.perm)
    assert g_copy.n_matches == g.n_matches


def test_get_match_bounds() -> None:
    """Verify calculation of minimum and maximum matches."""
    inst = _create_handbuilt_instance()  # 6 players, k=1 -> match_size=2
    # No starving players: min=0, max=3
    min_m, max_m = get_match_bounds(inst)
    assert min_m == 0
    assert max_m == 3

    # Add 3 starving players (wait + dt > max_wait) -> requires at least ceil(3 / 2) = 2 matches
    inst.players[0].wait = 115.0
    inst.players[1].wait = 112.0
    inst.players[2].wait = 118.0
    min_m, max_m = get_match_bounds(inst)
    assert min_m == 2
    assert max_m == 3


def test_decode_basic() -> None:
    """Verify decode maps consecutive blocks to matches and teams."""
    inst = _create_handbuilt_instance()  # k=1, 2 players per match
    # perm: [0, 1, 2, 3, 4, 5], n_matches=2
    # Match 0: p0 (team 0), p1 (team 1)
    # Match 1: p2 (team 0), p3 (team 1)
    # Unmatched: p4, p5 (match=-1, team=0)
    genome = Genome(perm=np.array([0, 1, 2, 3, 4, 5], dtype=int), n_matches=2)
    sol = decode(inst, genome)

    expected_match = np.array([0, 0, 1, 1, -1, -1], dtype=int)
    expected_team = np.array([0, 1, 0, 1, 0, 0], dtype=int)

    np.testing.assert_array_equal(sol.match, expected_match)
    np.testing.assert_array_equal(sol.team, expected_team)
    assert sol.n_matches() == 2


def test_decode_with_raw_array() -> None:
    """Verify decode accepts raw 1D numpy array."""
    inst = _create_handbuilt_instance()
    perm = np.array([5, 4, 3, 2, 1, 0], dtype=int)
    sol = decode(inst, perm, n_matches=1)

    assert sol.n_matches() == 1
    # Match 0 has players 5 and 4
    assert sol.match[5] == 0 and sol.team[5] == 0
    assert sol.match[4] == 0 and sol.team[4] == 1
    # Players 3, 2, 1, 0 remain unmatched
    assert sol.match[3] == -1
    assert sol.match[2] == -1


def test_decode_empty_instance() -> None:
    """Verify decode handles empty instance gracefully."""
    inst = Instance(
        players=[],
        k=2,
        dt=10.0,
        max_wait=120.0,
        gap0=100.0,
        gap_alpha=1.0,
        seed=42,
    )
    sol = decode(inst, np.array([]))
    assert sol.match.shape == (0,)
    assert sol.team.shape == (0,)
    assert sol.n_matches() == 0


def test_encode_decode_roundtrip() -> None:
    """Verify round-trip property: decode(inst, encode(sol)) == sol."""
    inst = _create_handbuilt_instance()
    # Construct a valid solution with 2 matches and 2 unmatched
    orig_match = np.array([1, 0, 0, 1, -1, -1], dtype=int)
    orig_team = np.array([0, 0, 1, 1, 0, 0], dtype=int)
    sol = Solution(match=orig_match, team=orig_team)

    genome = encode(sol, inst)
    assert genome.n_matches == 2
    assert len(genome.perm) == 6
    # All player indices 0..5 present exactly once
    assert sorted(genome.perm.tolist()) == [0, 1, 2, 3, 4, 5]

    decoded_sol = decode(inst, genome)
    # The formed matches should have identical player compositions and team assignments
    assert decoded_sol.n_matches() == sol.n_matches()

    # Verify that teammates and opposing teams match original
    for m_id in range(decoded_sol.n_matches()):
        dec_t0 = set(np.where((decoded_sol.match == m_id) & (decoded_sol.team == 0))[0])
        dec_t1 = set(np.where((decoded_sol.match == m_id) & (decoded_sol.team == 1))[0])

        # Match against one of the original matches
        found_match = False
        for orig_m in [0, 1]:
            orig_t0 = set(np.where((sol.match == orig_m) & (sol.team == 0))[0])
            orig_t1 = set(np.where((sol.match == orig_m) & (sol.team == 1))[0])
            if dec_t0 == orig_t0 and dec_t1 == orig_t1:
                found_match = True
                break
        assert found_match, f"Decoded match {m_id} does not match any original match"

    # Verify unmatched set is identical
    dec_unmatched = set(np.where(decoded_sol.match < 0)[0])
    orig_unmatched = set(np.where(sol.match < 0)[0])
    assert dec_unmatched == orig_unmatched


def test_crossover_ox_preserves_permutation() -> None:
    """Verify Order Crossover (OX) always generates valid permutations without duplicates."""
    rng = np.random.default_rng(123)
    n = 20

    for _ in range(50):
        p1 = rng.permutation(n)
        p2 = rng.permutation(n)

        c1, c2 = crossover_ox(p1, p2, rng=rng)

        # Offspring length must equal parent length
        assert len(c1) == n
        assert len(c2) == n

        # Offspring must contain every element 0..n-1 exactly once
        np.testing.assert_array_equal(np.sort(c1), np.arange(n))
        np.testing.assert_array_equal(np.sort(c2), np.arange(n))


def test_crossover_ox_with_genome_objects() -> None:
    """Verify crossover_ox operates correctly with Genome objects."""
    rng = np.random.default_rng(456)
    n = 10
    g1 = Genome(perm=rng.permutation(n), n_matches=2)
    g2 = Genome(perm=rng.permutation(n), n_matches=3)

    c1, c2 = crossover_ox(g1, g2, rng=rng)
    assert isinstance(c1, Genome)
    assert isinstance(c2, Genome)
    assert len(c1) == n
    assert len(c2) == n
    np.testing.assert_array_equal(np.sort(c1.perm), np.arange(n))
    np.testing.assert_array_equal(np.sort(c2.perm), np.arange(n))


def test_mutate_swap_preserves_permutation() -> None:
    """Verify swap mutation preserves permutation integrity."""
    rng = np.random.default_rng(789)
    n = 15
    perm = rng.permutation(n)

    for _ in range(30):
        mutated = mutate_swap(perm, rng=rng)
        assert len(mutated) == n
        np.testing.assert_array_equal(np.sort(mutated), np.arange(n))


def test_mutate_insert_preserves_permutation() -> None:
    """Verify insert mutation preserves permutation integrity."""
    rng = np.random.default_rng(101)
    n = 15
    perm = rng.permutation(n)

    for _ in range(30):
        mutated = mutate_insert(perm, rng=rng)
        assert len(mutated) == n
        np.testing.assert_array_equal(np.sort(mutated), np.arange(n))


def test_mutate_inversion_preserves_permutation() -> None:
    """Verify inversion mutation preserves permutation integrity."""
    rng = np.random.default_rng(202)
    n = 15
    perm = rng.permutation(n)

    for _ in range(30):
        mutated = mutate_inversion(perm, rng=rng)
        assert len(mutated) == n
        np.testing.assert_array_equal(np.sort(mutated), np.arange(n))


def test_random_genome_creation() -> None:
    """Verify random_genome respects instance bounds and player count."""
    inst = _create_handbuilt_instance()
    rng = np.random.default_rng(303)

    g = random_genome(inst, rng=rng)
    assert len(g.perm) == len(inst.players)
    np.testing.assert_array_equal(np.sort(g.perm), np.arange(len(inst.players)))
    assert 0 <= g.n_matches <= len(inst.players) // (2 * inst.k)

