# OWNER: M2 - do not edit unless you are M2
"""Unit tests and benchmarks for matchmaking objectives (Checkpoint G2)."""
from __future__ import annotations

import time
import numpy as np
import pytest

from mm.config import Config
from mm.objectives import evaluate, f1_imbalance, f2_wait, f3_mismatch
from mm.types import Instance, Player, Solution


def _make_handbuilt_instance(
    players_data: list[tuple[int, float, float]],
    k: int = 2,
    dt: float = 10.0,
    max_wait: float = 120.0,
    gap0: float = 100.0,
    gap_alpha: float = 1.0,
) -> Instance:
    """Helper to build a deterministic Instance from (id, skill, wait) tuples."""
    players = [
        Player(id=p_id, skill=skill, wait=wait, arrival=0.0)
        for p_id, skill, wait in players_data
    ]
    return Instance(
        players=players,
        k=k,
        dt=dt,
        max_wait=max_wait,
        gap0=gap0,
        gap_alpha=gap_alpha,
        seed=42,
    )


def test_all_unmatched_solution() -> None:
    """Verify objectives when all players remain in queue (empty solution).

    - f1_imbalance must be 0.0 (no matches formed).
    - f2_wait must equal sum_i (wait_i + dt).
    """
    players_data = [
        (0, 1500.0, 20.0),
        (1, 1550.0, 15.0),
        (2, 1480.0, 30.0),
        (3, 1520.0, 5.0),
    ]
    inst = _make_handbuilt_instance(players_data, k=1, dt=10.0)
    sol = Solution.empty(len(inst.players))

    f1 = f1_imbalance(inst, sol)
    f2 = f2_wait(inst, sol)

    expected_f2 = (20.0 + 10.0) + (15.0 + 10.0) + (30.0 + 10.0) + (5.0 + 10.0)  # 110.0
    assert f1 == 0.0
    assert f2 == pytest.approx(expected_f2)

    res = evaluate(inst, sol)
    assert isinstance(res, np.ndarray)
    assert res.shape == (2,)
    np.testing.assert_allclose(res, [0.0, expected_f2])


def test_perfectly_balanced_match() -> None:
    """Verify objectives when all players are matched with identical team mean skills.

    - f1_imbalance must be 0.0.
    - f2_wait must be 0.0 (no unmatched players).
    """
    players_data = [
        (0, 1500.0, 10.0),
        (1, 1500.0, 12.0),
        (2, 1500.0, 8.0),
        (3, 1500.0, 15.0),
    ]
    inst = _make_handbuilt_instance(players_data, k=1, dt=10.0)

    # 2 matches of k=1 (2 players each)
    # Match 0: player 0 (team 0) vs player 1 (team 1) -> 1500 vs 1500
    # Match 1: player 2 (team 0) vs player 3 (team 1) -> 1500 vs 1500
    sol = Solution(
        match=np.array([0, 0, 1, 1], dtype=int),
        team=np.array([0, 1, 0, 1], dtype=int),
    )

    f1 = f1_imbalance(inst, sol)
    f2 = f2_wait(inst, sol)

    assert f1 == pytest.approx(0.0)
    assert f2 == pytest.approx(0.0)
    np.testing.assert_allclose(evaluate(inst, sol), [0.0, 0.0])


def test_worked_example_from_docs() -> None:
    """Verify the exact worked example documented in docs/objectives.md.

    6 players, k=2, dt=10.0.
    Players 0..3 in match 0, players 4 and 5 unmatched.
    Team 0 skills: [1550.0, 1480.0] -> mean = 1515.0
    Team 1 skills: [1530.0, 1500.0] -> mean = 1515.0
    Imbalance: |1515.0 - 1515.0| = 0.0
    Unmatched waits: (5.0 + 10.0) + (2.0 + 10.0) = 27.0
    """
    players_data = [
        (101, 1550.0, 25.0),
        (102, 1530.0, 15.0),
        (103, 1480.0, 50.0),
        (104, 1500.0, 40.0),
        (105, 1620.0, 5.0),
        (106, 1390.0, 2.0),
    ]
    inst = _make_handbuilt_instance(players_data, k=2, dt=10.0)
    sol = Solution(
        match=np.array([0, 0, 0, 0, -1, -1], dtype=int),
        team=np.array([0, 1, 0, 1, 0, 0], dtype=int),
    )

    f1 = f1_imbalance(inst, sol)
    f2 = f2_wait(inst, sol)

    assert f1 == pytest.approx(0.0)
    assert f2 == pytest.approx(27.0)
    np.testing.assert_allclose(evaluate(inst, sol), [0.0, 27.0])


def test_hand_built_fixture_with_known_f1_f2() -> None:
    """Multi-match fixture with known non-zero f1 and f2 values.

    10 players, k=2, dt=10.0.
    Match 0 (k=2):
      Team 0: p0 (1600), p1 (1550) -> mean = 1575.0
      Team 1: p2 (1500), p3 (1450) -> mean = 1475.0
      Imbalance = |1575 - 1475| = 100.0

    Match 1 (k=2):
      Team 0: p4 (1400), p5 (1420) -> mean = 1410.0
      Team 1: p6 (1380), p7 (1370) -> mean = 1375.0
      Imbalance = |1410 - 1375| = 35.0

    Mean imbalance f1 = (100.0 + 35.0) / 2 = 67.5.

    Unmatched:
      p8: wait 12.0 -> 12 + 10 = 22.0
      p9: wait 8.0  -> 8 + 10  = 18.0
      f2 = 22.0 + 18.0 = 40.0.
    """
    players_data = [
        (0, 1600.0, 5.0),
        (1, 1550.0, 6.0),
        (2, 1500.0, 7.0),
        (3, 1450.0, 8.0),
        (4, 1400.0, 9.0),
        (5, 1420.0, 10.0),
        (6, 1380.0, 11.0),
        (7, 1370.0, 12.0),
        (8, 1490.0, 12.0),
        (9, 1510.0, 8.0),
    ]
    inst = _make_handbuilt_instance(players_data, k=2, dt=10.0)
    sol = Solution(
        match=np.array([0, 0, 0, 0, 1, 1, 1, 1, -1, -1], dtype=int),
        team=np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=int),
    )

    f1 = f1_imbalance(inst, sol)
    f2 = f2_wait(inst, sol)

    assert f1 == pytest.approx(67.5)
    assert f2 == pytest.approx(40.0)
    np.testing.assert_allclose(evaluate(inst, sol), [67.5, 40.0])


def test_non_contiguous_match_ids() -> None:
    """Verify that arbitrary non-contiguous match IDs (e.g., 3 and 8) work correctly."""
    players_data = [
        (0, 1600.0, 0.0),
        (1, 1500.0, 0.0),
        (2, 1400.0, 0.0),
        (3, 1380.0, 0.0),
    ]
    inst = _make_handbuilt_instance(players_data, k=1, dt=10.0)
    # Match ID 8 for players 0,1; match ID 3 for players 2,3
    sol = Solution(
        match=np.array([8, 8, 3, 3], dtype=int),
        team=np.array([0, 1, 0, 1], dtype=int),
    )

    # Match 8 diff = |1600 - 1500| = 100
    # Match 3 diff = |1400 - 1380| = 20
    # Mean imbalance = (100 + 20) / 2 = 60.0
    f1 = f1_imbalance(inst, sol)
    assert f1 == pytest.approx(60.0)
    assert f2_wait(inst, sol) == pytest.approx(0.0)


def test_f3_stub_and_use_f3() -> None:
    """Verify f3_mismatch returns 0.0 when role/ping are missing and evaluate handles use_f3."""
    players_data = [
        (0, 1500.0, 10.0),
        (1, 1500.0, 10.0),
    ]
    inst = _make_handbuilt_instance(players_data, k=1, dt=10.0)
    sol = Solution(
        match=np.array([0, 0], dtype=int),
        team=np.array([0, 1], dtype=int),
    )

    # Missing role/ping
    assert f3_mismatch(inst, sol) == 0.0

    # evaluate without use_f3
    res_2 = evaluate(inst, sol, use_f3=False)
    assert res_2.shape == (2,)

    # evaluate with use_f3
    res_3 = evaluate(inst, sol, use_f3=True)
    assert res_3.shape == (3,)
    assert res_3[2] == 0.0

    # evaluate with config
    cfg_f3 = Config(use_f3=True)
    res_cfg = evaluate(inst, sol, cfg=cfg_f3)
    assert res_cfg.shape == (3,)


def test_empty_instance() -> None:
    """Edge case: Instance with 0 players."""
    inst = Instance(
        players=[],
        k=2,
        dt=10.0,
        max_wait=120.0,
        gap0=100.0,
        gap_alpha=1.0,
        seed=42,
    )
    sol = Solution.empty(0)
    res = evaluate(inst, sol)
    np.testing.assert_allclose(res, [0.0, 0.0])


def test_vectorized_performance_benchmark() -> None:
    """Acceptance test G2: 1,000 evaluations in under 1 s on a medium instance (100 players)."""
    n_players = 100
    k = 2
    # Create 100 players
    rng = np.random.default_rng(12345)
    skills = rng.normal(1500.0, 300.0, size=n_players)
    waits = rng.exponential(30.0, size=n_players)

    players = [
        Player(id=i, skill=float(skills[i]), wait=float(waits[i]), arrival=0.0)
        for i in range(n_players)
    ]
    inst = Instance(
        players=players,
        k=k,
        dt=10.0,
        max_wait=120.0,
        gap0=100.0,
        gap_alpha=1.0,
        seed=12345,
    )

    # Form 20 matches (80 players matched, 20 players unmatched)
    match_arr = np.full(n_players, -1, dtype=int)
    team_arr = np.zeros(n_players, dtype=int)

    # 20 matches * 4 players = 80 players
    for m in range(20):
        start = m * 4
        match_arr[start : start + 4] = m
        team_arr[start : start + 2] = 0
        team_arr[start + 2 : start + 4] = 1

    sol = Solution(match=match_arr, team=team_arr)

    # Warmup
    _ = evaluate(inst, sol)

    n_evals = 1000
    start_time = time.perf_counter()
    for _ in range(n_evals):
        _ = evaluate(inst, sol)
    elapsed = time.perf_counter() - start_time

    # Must be strictly under 1.0 s
    assert elapsed < 1.0, f"1,000 evaluations took {elapsed:.3f} s, expected < 1.0 s"
    evals_per_sec = n_evals / elapsed
    print(f"\n[PERFORMANCE] 1,000 evaluations completed in {elapsed * 1000:.2f} ms ({evals_per_sec:.0f} evals/sec)")
