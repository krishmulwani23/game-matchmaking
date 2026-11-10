# OWNER: M2 - do not edit unless you are M2
"""Unit tests for matchmaking performance metrics and baseline benchmarks (Checkpoint G9)."""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd
import pytest

from mm.benchmarks import run_baseline_benchmarks
from mm.config import Config
from mm.metrics import (
    compute_metrics,
    feasibility_rate,
    match_rate,
    matched_wait_max,
    matched_wait_mean,
    matched_wait_median,
    matched_wait_p95,
    max_imbalance,
    mean_imbalance,
    queue_length_after,
    time_execution,
    timed,
)
from mm.objectives import evaluate, f3_mismatch
from mm.types import Instance, Player, Solution
from mm.validate import is_feasible


def _create_handbuilt_instance() -> Instance:
    """Create a 6-player hand-built instance with known attributes.

    k = 2 (each match has 4 players, 2 per team).
    Roles: 0 (Tank), 1 (DPS), 2 (Support).
    """
    players = [
        # Match 0, Team 0:
        Player(id=0, skill=1500.0, wait=10.0, arrival=0.0, role=0, ping=20.0),
        Player(id=1, skill=1600.0, wait=20.0, arrival=1.0, role=0, ping=25.0),
        # Match 0, Team 1:
        Player(id=2, skill=1520.0, wait=30.0, arrival=2.0, role=0, ping=30.0),
        Player(id=3, skill=1580.0, wait=40.0, arrival=3.0, role=1, ping=45.0),
        # Unmatched in queue:
        Player(id=4, skill=1700.0, wait=50.0, arrival=4.0, role=2, ping=50.0),
        Player(id=5, skill=1400.0, wait=60.0, arrival=5.0, role=1, ping=60.0),
    ]
    return Instance(
        players=players,
        k=2,
        dt=10.0,
        max_wait=120.0,
        gap0=150.0,
        gap_alpha=1.0,
        seed=42,
    )


# =========================================================================
# 1. Matched Wait Statistics Tests
# =========================================================================


def test_matched_wait_statistics() -> None:
    """Verify mean, median, p95, and max wait calculations for matched players."""
    inst = _create_handbuilt_instance()
    # Match players 0, 1, 2, 3 (waits: 10, 20, 30, 40)
    sol = Solution(
        match=np.array([0, 0, 0, 0, -1, -1], dtype=int),
        team=np.array([0, 0, 1, 1, 0, 0], dtype=int),
    )

    waits = [10.0, 20.0, 30.0, 40.0]
    expected_mean = np.mean(waits)  # 25.0
    expected_median = np.median(waits)  # 25.0
    expected_p95 = np.percentile(waits, 95)  # 38.5
    expected_max = 40.0

    assert matched_wait_mean(inst, sol) == pytest.approx(expected_mean)
    assert matched_wait_median(inst, sol) == pytest.approx(expected_median)
    assert matched_wait_p95(inst, sol) == pytest.approx(expected_p95)
    assert matched_wait_max(inst, sol) == pytest.approx(expected_max)


def test_matched_wait_no_matches() -> None:
    """When no matches are formed, matched wait statistics return 0.0."""
    inst = _create_handbuilt_instance()
    sol = Solution.empty(len(inst.players))

    assert matched_wait_mean(inst, sol) == 0.0
    assert matched_wait_median(inst, sol) == 0.0
    assert matched_wait_p95(inst, sol) == 0.0
    assert matched_wait_max(inst, sol) == 0.0


# =========================================================================
# 2. Queue Length & Match Rate Tests
# =========================================================================


def test_queue_length_and_match_rate() -> None:
    """Verify queue length remaining and match rate proportions."""
    inst = _create_handbuilt_instance()
    # 4 matched, 2 unmatched
    sol = Solution(
        match=np.array([0, 0, 0, 0, -1, -1], dtype=int),
        team=np.array([0, 0, 1, 1, 0, 0], dtype=int),
    )

    assert queue_length_after(inst, sol) == 2
    assert match_rate(inst, sol) == pytest.approx(4 / 6)


def test_queue_length_and_match_rate_extremes() -> None:
    """Verify queue length and match rate for all matched and all unmatched."""
    inst = _create_handbuilt_instance()

    # All unmatched
    sol_none = Solution.empty(6)
    assert queue_length_after(inst, sol_none) == 6
    assert match_rate(inst, sol_none) == 0.0

    # Empty queue (0 players)
    inst_empty = Instance(
        players=[],
        k=2,
        dt=10.0,
        max_wait=120.0,
        gap0=100.0,
        gap_alpha=1.0,
        seed=42,
    )
    sol_empty = Solution.empty(0)
    assert queue_length_after(inst_empty, sol_empty) == 0
    assert match_rate(inst_empty, sol_empty) == 0.0


# =========================================================================
# 3. Imbalance Statistics Tests
# =========================================================================


def test_imbalance_metrics() -> None:
    """Verify mean_imbalance and max_imbalance across matches."""
    # Build 8 players with 2 matches
    players = [
        # Match 0: Team 0 (1500, 1500) -> mean 1500; Team 1 (1520, 1520) -> mean 1520; diff = 20
        Player(id=0, skill=1500.0, wait=10.0, arrival=0.0),
        Player(id=1, skill=1500.0, wait=10.0, arrival=0.0),
        Player(id=2, skill=1520.0, wait=10.0, arrival=0.0),
        Player(id=3, skill=1520.0, wait=10.0, arrival=0.0),
        # Match 1: Team 0 (1600, 1600) -> mean 1600; Team 1 (1650, 1650) -> mean 1650; diff = 50
        Player(id=4, skill=1600.0, wait=10.0, arrival=0.0),
        Player(id=5, skill=1600.0, wait=10.0, arrival=0.0),
        Player(id=6, skill=1650.0, wait=10.0, arrival=0.0),
        Player(id=7, skill=1650.0, wait=10.0, arrival=0.0),
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
    sol = Solution(
        match=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int),
        team=np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=int),
    )

    # Imbalances: Match 0 = 20, Match 1 = 50
    assert mean_imbalance(inst, sol) == pytest.approx(35.0)
    assert max_imbalance(inst, sol) == pytest.approx(50.0)


def test_imbalance_no_matches() -> None:
    """When no matches are formed, imbalance metrics return 0.0."""
    inst = _create_handbuilt_instance()
    sol = Solution.empty(len(inst.players))

    assert mean_imbalance(inst, sol) == 0.0
    assert max_imbalance(inst, sol) == 0.0


# =========================================================================
# 4. f3_mismatch: Role Coverage & Ping Spread
# =========================================================================


def test_f3_mismatch_exact_calculation() -> None:
    """Verify exact hand-computed calculation for f3_mismatch."""
    inst = _create_handbuilt_instance()
    # Match 0:
    # Team 0: players 0, 1. Roles: {0, 0} (only role 0). Target roles: min(k=2, distinct=3) = 2.
    # Deficit 0 = 2 - 1 = 1.
    # Team 1: players 2, 3. Roles: {0, 1}. Target roles: 2.
    # Deficit 1 = 2 - 2 = 0.
    # Role deficit = 0.5 * (1 + 0) = 0.5.
    # Pings in Match 0: [20.0, 25.0, 30.0, 45.0].
    # Ping spread = 45.0 - 20.0 = 25.0.
    # Expected f3 = 25.0 + 0.5 = 25.5.
    sol = Solution(
        match=np.array([0, 0, 0, 0, -1, -1], dtype=int),
        team=np.array([0, 0, 1, 1, 0, 0], dtype=int),
    )

    f3 = f3_mismatch(inst, sol)
    assert f3 == pytest.approx(25.5)

    # evaluate with use_f3=True
    eval_res = evaluate(inst, sol, use_f3=True)
    assert len(eval_res) == 3
    assert eval_res[2] == pytest.approx(25.5)


def test_f3_missing_data_returns_zero() -> None:
    """f3_mismatch must return 0.0 if any player lacks role or ping."""
    inst = _create_handbuilt_instance()
    sol = Solution(
        match=np.array([0, 0, 0, 0, -1, -1], dtype=int),
        team=np.array([0, 0, 1, 1, 0, 0], dtype=int),
    )

    # Missing role
    inst.players[0].role = None
    assert f3_mismatch(inst, sol) == 0.0

    # Missing ping
    inst.players[0].role = 0
    inst.players[1].ping = None
    assert f3_mismatch(inst, sol) == 0.0


def test_f3_behind_cfg_flag() -> None:
    """evaluate incorporates f3 only when use_f3 or cfg.use_f3 is True."""
    inst = _create_handbuilt_instance()
    sol = Solution(
        match=np.array([0, 0, 0, 0, -1, -1], dtype=int),
        team=np.array([0, 0, 1, 1, 0, 0], dtype=int),
    )

    cfg_off = Config(use_f3=False)
    cfg_on = Config(use_f3=True)

    assert evaluate(inst, sol, cfg=cfg_off).shape == (2,)
    assert evaluate(inst, sol, cfg=cfg_on).shape == (3,)


# =========================================================================
# 5. Runtime Wrapper & Feasibility Rate Tests
# =========================================================================


def test_time_execution_and_decorator() -> None:
    """Verify runtime measurement utility and decorator."""
    def dummy_task(duration: float, val: int) -> int:
        time.sleep(duration)
        return val * 2

    res, elapsed = time_execution(dummy_task, 0.01, 5)
    assert res == 10
    assert elapsed >= 0.005

    @timed
    def timed_dummy(val: int) -> int:
        return val + 1

    res_dec, elapsed_dec = timed_dummy(41)
    assert res_dec == 42
    assert elapsed_dec >= 0.0


def test_feasibility_rate() -> None:
    """Verify feasibility_rate over candidate solutions."""
    inst = _create_handbuilt_instance()
    # Feasible solution:
    sol_feas = Solution.empty(len(inst.players))
    # Infeasible solution (invalid team value):
    sol_infeas = Solution(
        match=np.array([0, 0, 0, 0, -1, -1], dtype=int),
        team=np.array([0, 2, 1, 1, 0, 0], dtype=int),
    )

    assert feasibility_rate(inst, [sol_feas, sol_feas]) == 1.0
    assert feasibility_rate(inst, [sol_feas, sol_infeas]) == 0.5
    assert feasibility_rate(inst, [sol_infeas, sol_infeas]) == 0.0
    assert feasibility_rate(inst, []) == 1.0


def test_compute_metrics_dict() -> None:
    """Verify compute_metrics aggregates all fields properly."""
    inst = _create_handbuilt_instance()
    sol = Solution(
        match=np.array([0, 0, 0, 0, -1, -1], dtype=int),
        team=np.array([0, 0, 1, 1, 0, 0], dtype=int),
    )

    m = compute_metrics(inst, sol, runtime_s=0.0123)
    assert isinstance(m, dict)
    expected_keys = {
        "n_matches",
        "queue_length",
        "match_rate",
        "mean_imbalance",
        "max_imbalance",
        "matched_wait_mean",
        "matched_wait_median",
        "matched_wait_p95",
        "matched_wait_max",
        "is_feasible",
        "runtime_s",
    }
    assert expected_keys.issubset(m.keys())
    assert m["n_matches"] == 1
    assert m["queue_length"] == 2
    assert m["is_feasible"] is True


# =========================================================================
# 6. Baseline Benchmarks Integration Test
# =========================================================================


def test_baseline_benchmarks_run(tmp_path: Path) -> None:
    """Run baseline benchmarks in quick mode and verify CSV output."""
    out_csv = tmp_path / "test_baselines.csv"

    df = run_baseline_benchmarks(
        scenarios=["light"],
        seeds=[42],
        baselines=["sort_skill", "fifo", "greedy_nearest"],
        output_path=out_csv,
        verbose=False,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # 1 scenario * 1 seed * 3 baselines
    assert out_csv.exists()

    # Verify reload from disk
    reloaded = pd.read_csv(out_csv)
    assert len(reloaded) == 3
    expected_cols = [
        "scenario",
        "seed",
        "algorithm",
        "n_players",
        "n_matches",
        "queue_length",
        "match_rate",
        "f1_imbalance",
        "f2_wait",
        "f3_mismatch",
        "mean_imbalance",
        "max_imbalance",
        "matched_wait_mean",
        "matched_wait_median",
        "matched_wait_p95",
        "matched_wait_max",
        "is_feasible",
        "runtime_s",
    ]
    for col in expected_cols:
        assert col in reloaded.columns

