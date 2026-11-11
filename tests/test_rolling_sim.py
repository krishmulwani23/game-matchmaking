# OWNER: M1 - do not edit unless you are M1
"""Tests for mm.rolling_sim multi-round queue simulation and scenario presets."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from mm import registry
from mm.config import Config, _parse_yaml_fallback, load_config
from mm.rolling_sim import run_rolling_sim, summarize_rolling_logs
from mm.types import Instance, Solution


class _NoopDummyMatcher:
    """Dummy matcher that never forms any matches (leaves everyone in queue)."""

    name = "test_dummy_noop"

    def solve(self, inst: Instance, **kw) -> list[Solution]:
        return [Solution.empty(len(inst.players))]


class _MultiFrontDummyMatcher:
    """Dummy matcher returning multiple solutions to test solution selection."""

    name = "test_dummy_multi"

    def solve(self, inst: Instance, **kw) -> list[Solution]:
        n = len(inst.players)
        empty = Solution.empty(n)
        if n < 2 * inst.k:
            return [empty]

        # Solution A: 1 match using first 2k players (lower f2 wait, possibly higher f1)
        m_one = np.full(n, -1, dtype=int)
        t_one = np.zeros(n, dtype=int)
        for i in range(2 * inst.k):
            m_one[i] = 0
            t_one[i] = 0 if i < inst.k else 1
        sol_one = Solution(match=m_one, team=t_one)

        # Solution B: sort by skill to form 1 fair match (lower f1 imbalance)
        order = sorted(range(n), key=lambda idx: inst.players[idx].skill)
        m_fair = np.full(n, -1, dtype=int)
        t_fair = np.zeros(n, dtype=int)
        for rank, idx in enumerate(order[: 2 * inst.k]):
            m_fair[idx] = 0
            t_fair[idx] = rank % 2
        sol_fair = Solution(match=m_fair, team=t_fair)

        return [empty, sol_one, sol_fair]


def test_scenario_presets_in_default_config() -> None:
    """Verify light, medium, heavy, and bursty presets load from config/default.yaml."""
    cfg = load_config()
    for name in ("light", "medium", "heavy", "bursty"):
        assert name in cfg.scenarios
        preset = cfg.scenarios[name]
        assert preset.initial_players >= 0
        assert preset.arrival_rate > 0.0
        assert preset.n_rounds > 0

    assert cfg.scenarios["bursty"].burst_prob > 0.0
    assert cfg.scenarios["bursty"].burst_multiplier > 1.0


def test_fallback_yaml_parser_reads_scenarios() -> None:
    """Verify fallback YAML parser parses nested scenario presets identically."""
    default_yaml = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    parsed = _parse_yaml_fallback(default_yaml.read_text(encoding="utf-8"))
    assert "scenarios" in parsed
    for name in ("light", "medium", "heavy", "bursty"):
        assert name in parsed["scenarios"]
    assert parsed["scenarios"]["bursty"]["burst_prob"] == pytest.approx(0.25)
    assert parsed["scenarios"]["bursty"]["burst_multiplier"] == pytest.approx(5.0)


def test_dummy_matcher_queue_grows(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a dummy matcher that matches nothing, queue length and wait must grow."""
    monkeypatch.setitem(registry._REGISTRY, "test_dummy_noop", _NoopDummyMatcher())
    cfg = load_config()
    n_rounds = 15

    df = run_rolling_sim(
        matcher="test_dummy_noop",
        scenario="light",
        n_rounds=n_rounds,
        seed=42,
        config=cfg,
    )

    required_cols = {
        "round",
        "time",
        "arrivals",
        "queue_length",
        "mean_wait",
        "max_wait",
        "mean_imbalance",
        "matches_formed",
    }
    assert required_cols.issubset(df.columns)
    assert len(df) == n_rounds
    assert (df["matches_formed"] == 0).all()
    assert (df["mean_imbalance"] == 0.0).all()
    assert df["queue_length"].is_monotonic_increasing

    expected_final_q = cfg.scenarios["light"].initial_players + int(df["arrivals"].sum())
    assert int(df["queue_length"].iloc[-1]) == expected_final_q
    assert int(df["queue_length"].iloc[-1]) > cfg.scenarios["light"].initial_players
    assert df["max_wait"].iloc[-1] == pytest.approx(n_rounds * cfg.dt)
    assert df["mean_wait"].iloc[-1] > df["mean_wait"].iloc[0]


def test_fifo_baseline_keeps_queue_bounded_in_light_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If mm.baselines is available, fifo must keep queue length bounded in light scenario."""
    if "fifo" not in registry.list_names():
        pytest.skip("mm.baselines 'fifo' is not registered on this branch")

    monkeypatch.setitem(registry._REGISTRY, "test_dummy_noop", _NoopDummyMatcher())
    cfg = load_config()

    for seed in (42, 43, 44):
        df_fifo = run_rolling_sim(
            matcher="fifo",
            scenario="light",
            n_rounds=40,
            seed=seed,
            config=cfg,
        )
        df_noop = run_rolling_sim(
            matcher="test_dummy_noop",
            scenario="light",
            n_rounds=40,
            seed=seed,
            config=cfg,
        )

        total_entered = cfg.scenarios["light"].initial_players + int(df_fifo["arrivals"].sum())
        assert df_fifo["matches_formed"].sum() > 0
        # Bounded well below the unmatched cumulative queue size
        assert int(df_fifo["queue_length"].max()) < 60
        assert int(df_fifo["queue_length"].iloc[-1]) < total_entered * 0.35
        assert int(df_fifo["queue_length"].iloc[-1]) < int(df_noop["queue_length"].iloc[-1])


def test_rolling_sim_deterministic_same_seed() -> None:
    """Identical seeds produce identical per-round logs."""
    matcher = "fifo" if "fifo" in registry.list_names() else _NoopDummyMatcher()
    df1 = run_rolling_sim(matcher=matcher, scenario="light", n_rounds=10, seed=123)
    df2 = run_rolling_sim(matcher=matcher, scenario="light", n_rounds=10, seed=123)
    pd.testing.assert_frame_equal(df1, df2)


def test_bursty_scenario_triggers_traffic_spikes() -> None:
    """Bursty preset produces burst rounds with elevated arrivals."""
    df = run_rolling_sim(
        matcher=_NoopDummyMatcher(),
        scenario="bursty",
        n_rounds=40,
        seed=7,
    )
    assert df["is_burst"].any()
    burst_mean = df.loc[df["is_burst"], "arrivals"].mean()
    normal_mean = df.loc[~df["is_burst"], "arrivals"].mean()
    assert burst_mean > normal_mean


def test_all_scenario_presets_run_with_fifo() -> None:
    """All four presets run cleanly with fifo and produce valid summary stats."""
    if "fifo" not in registry.list_names():
        pytest.skip("mm.baselines 'fifo' is not registered on this branch")

    for scenario in ("light", "medium", "heavy", "bursty"):
        df = run_rolling_sim(matcher="fifo", scenario=scenario, n_rounds=10, seed=42)
        summary = summarize_rolling_logs(df)
        assert summary["n_rounds"] == 10
        assert summary["total_matches"] > 0
        assert summary["final_queue_length"] >= 0


def test_multi_solution_selection_strategies() -> None:
    """Multi-objective matcher outputs work with balanced, min_f1, min_f2, and first."""
    matcher = _MultiFrontDummyMatcher()
    for strategy in ("balanced", "min_f1", "min_f2", "first"):
        df = run_rolling_sim(
            matcher=matcher,
            scenario="light",
            n_rounds=5,
            seed=42,
            select_strategy=strategy,
        )
        assert len(df) == 5

    df_empty_summary = summarize_rolling_logs(pd.DataFrame())
    assert df_empty_summary["n_rounds"] == 0


def test_empty_queue_and_invalid_inputs() -> None:
    """Empty queue edge case works and invalid scenario/strategy raise ValueError."""
    df = run_rolling_sim(
        matcher=_NoopDummyMatcher(),
        scenario="light",
        n_rounds=5,
        seed=1,
        initial_players=0,
        arrival_rate=0.0,
    )
    assert (df["queue_length"] == 0).all()
    assert (df["mean_wait"] == 0.0).all()
    assert (df["max_wait"] == 0.0).all()

    with pytest.raises(ValueError, match="Unknown scenario"):
        run_rolling_sim(matcher=_NoopDummyMatcher(), scenario="unknown_preset")

    with pytest.raises(ValueError, match="Unknown select_strategy"):
        run_rolling_sim(
            matcher=_MultiFrontDummyMatcher(),
            scenario="light",
            n_rounds=1,
            select_strategy="invalid_strat",
        )
