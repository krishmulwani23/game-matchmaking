# OWNER: M1 - do not edit unless you are M1
"""SYNTHETIC Multi-Round Rolling Queue Simulation (Checkpoint G10).

DISCLAIMER: All player attributes, arrivals, skills, waits, roles, and network
pings generated in this module are purely SYNTHETIC and derived from stochastic
distributions. No real player traces or proprietary datasets are used.

Each simulation round executes:
1. New synthetic players arrive according to a Poisson arrival process
   (supporting steady and bursty traffic presets).
2. A Matcher from the central registry solves the current queue instance.
3. Matched players leave the queue; unmatched players remain and accumulate
   `wait += dt`.
4. Per-round queue and match quality metrics are logged.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from mm import registry
from mm.config import Config, ScenarioPreset, load_config
from mm.interfaces import Matcher
from mm.objectives import evaluate, f1_imbalance
from mm.simulator import ROLE_PROBABILITIES, ROLES
from mm.types import Instance, Player, Solution
from mm.validate import is_feasible


def _generate_players(
    count: int,
    start_id: int,
    arrival_time: float,
    rng: np.random.Generator,
    include_roles: bool = True,
    include_ping: bool = True,
) -> list[Player]:
    """Generate `count` SYNTHETIC players arriving at `arrival_time`."""
    if count <= 0:
        return []

    skills = np.maximum(100.0, rng.normal(loc=1500.0, scale=300.0, size=count))
    roles = (
        rng.choice(ROLES, size=count, p=ROLE_PROBABILITIES)
        if include_roles
        else [None] * count
    )
    pings = (
        rng.lognormal(mean=3.5, sigma=0.5, size=count)
        if include_ping
        else [None] * count
    )

    players: list[Player] = []
    for offset in range(count):
        players.append(
            Player(
                id=start_id + offset,
                skill=float(round(skills[offset], 1)),
                wait=0.0,
                arrival=float(round(arrival_time, 4)),
                role=int(roles[offset]) if roles[offset] is not None else None,
                ping=float(round(pings[offset], 1)) if pings[offset] is not None else None,
            )
        )
    return players


def _resolve_matcher(matcher: str | Matcher) -> tuple[str, Matcher]:
    """Resolve a matcher name via registry or validate a Matcher object."""
    if isinstance(matcher, str):
        resolved = registry.get(matcher)
        return matcher, resolved
    name = getattr(matcher, "name", type(matcher).__name__)
    return str(name), matcher


def _resolve_scenario(cfg: Config, scenario: str) -> ScenarioPreset:
    """Look up a scenario preset by name from Config."""
    key = scenario.lower()
    if key in cfg.scenarios:
        return cfg.scenarios[key]
    available = sorted(cfg.scenarios.keys())
    raise ValueError(f"Unknown scenario '{scenario}'. Available presets: {available}")


def _select_solution(
    inst: Instance,
    solutions: list[Solution],
    strategy: str = "balanced",
) -> Solution:
    """Select one Solution from a Matcher's output list."""
    if not solutions:
        return Solution.empty(len(inst.players))
    if len(solutions) == 1 or strategy == "first":
        return solutions[0]

    objs = np.array([evaluate(inst, sol)[:2] for sol in solutions], dtype=float)
    if strategy in ("min_f1", "very_fair"):
        return solutions[int(np.argmin(objs[:, 0]))]
    if strategy in ("min_f2", "fast"):
        return solutions[int(np.argmin(objs[:, 1]))]
    if strategy in ("balanced", "knee"):
        mins = objs.min(axis=0)
        spans = np.where((objs.max(axis=0) - mins) > 0, objs.max(axis=0) - mins, 1.0)
        normed = (objs - mins) / spans
        dists = np.linalg.norm(normed, axis=1)
        return solutions[int(np.argmin(dists))]

    raise ValueError(
        f"Unknown select_strategy '{strategy}'. "
        "Expected one of: 'balanced', 'knee', 'min_f1', 'very_fair', 'min_f2', 'fast', 'first'."
    )


def run_rolling_sim(
    matcher: str | Matcher = "fifo",
    scenario: str = "light",
    n_rounds: int | None = None,
    seed: int = 42,
    config: Config | None = None,
    *,
    initial_players: int | None = None,
    arrival_rate: float | None = None,
    k: int | None = None,
    dt: float | None = None,
    max_wait: float | None = None,
    gap0: float | None = None,
    gap_alpha: float | None = None,
    burst_prob: float | None = None,
    burst_multiplier: float | None = None,
    include_roles: bool = True,
    include_ping: bool = True,
    max_ping_gap: float | None = 100.0,
    select_strategy: str = "balanced",
    matcher_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run a SYNTHETIC multi-round rolling queue simulation.

    All generated players and arrivals are SYNTHETIC.

    Parameters
    ----------
    matcher : str | Matcher
        Registered matcher name (resolved via `mm.registry.get`) or a Matcher instance.
    scenario : str
        Scenario preset key from `config/default.yaml` ('light', 'medium', 'heavy', 'bursty').
    n_rounds : int | None
        Number of simulation rounds. Defaults to the scenario preset value.
    seed : int
        Random seed for deterministic arrivals and player generation.
    config : Config | None
        Configuration object. Defaults to `load_config()`.
    initial_players, arrival_rate, k, dt, max_wait, gap0, gap_alpha,
    burst_prob, burst_multiplier : optional overrides
        Override parameters from Config or the chosen ScenarioPreset.
    include_roles, include_ping : bool
        Whether generated SYNTHETIC players include role and ping attributes.
    max_ping_gap : float | None
        Ping spread limit passed to `validate.is_feasible`.
    select_strategy : str
        How to choose a single solution when a multi-objective matcher returns > 1 solution.
    matcher_kwargs : dict[str, Any] | None
        Extra keyword arguments forwarded to `matcher.solve(inst, **kw)`.

    Returns
    -------
    pd.DataFrame
        Per-round simulation logs containing `round`, `time`, `scenario`, `matcher`,
        `arrivals`, `is_burst`, `queue_before`, `queue_length`, `matches_formed`,
        `players_matched`, `mean_wait`, `max_wait`, `matched_mean_wait`,
        `matched_max_wait`, `mean_imbalance`, and `feasible`.
    """
    cfg = config if config is not None else load_config()
    preset = _resolve_scenario(cfg, scenario)
    matcher_name, matcher_obj = _resolve_matcher(matcher)

    sim_rounds = int(n_rounds if n_rounds is not None else preset.n_rounds)
    sim_init = int(initial_players if initial_players is not None else preset.initial_players)
    sim_rate = float(arrival_rate if arrival_rate is not None else preset.arrival_rate)
    sim_burst_prob = float(burst_prob if burst_prob is not None else preset.burst_prob)
    sim_burst_mult = float(
        burst_multiplier if burst_multiplier is not None else preset.burst_multiplier
    )

    sim_k = int(k if k is not None else cfg.k)
    sim_dt = float(dt if dt is not None else cfg.dt)
    sim_max_wait = float(max_wait if max_wait is not None else cfg.max_wait)
    sim_gap0 = float(gap0 if gap0 is not None else cfg.gap0)
    sim_gap_alpha = float(gap_alpha if gap_alpha is not None else cfg.gap_alpha)
    solve_kw = dict(matcher_kwargs) if matcher_kwargs else {}

    rng = np.random.default_rng(seed)
    queue: list[Player] = _generate_players(
        count=sim_init,
        start_id=0,
        arrival_time=0.0,
        rng=rng,
        include_roles=include_roles,
        include_ping=include_ping,
    )
    next_player_id = sim_init

    records: list[dict[str, Any]] = []
    for round_idx in range(sim_rounds):
        current_time = round_idx * sim_dt
        is_burst = bool(sim_burst_prob > 0.0 and rng.random() < sim_burst_prob)
        effective_rate = sim_rate * (sim_burst_mult if is_burst else 1.0)
        n_arrivals = int(rng.poisson(max(0.0, effective_rate * sim_dt)))

        new_players = _generate_players(
            count=n_arrivals,
            start_id=next_player_id,
            arrival_time=current_time,
            rng=rng,
            include_roles=include_roles,
            include_ping=include_ping,
        )
        next_player_id += n_arrivals
        queue.extend(new_players)

        queue_before = len(queue)
        inst = Instance(
            players=list(queue),
            k=sim_k,
            dt=sim_dt,
            max_wait=sim_max_wait,
            gap0=sim_gap0,
            gap_alpha=sim_gap_alpha,
            seed=seed + round_idx,
        )

        if queue_before > 0:
            solutions = matcher_obj.solve(inst, **solve_kw)
            sol = _select_solution(inst, solutions, strategy=select_strategy)
        else:
            sol = Solution.empty(0)

        matches_formed = sol.n_matches()
        mean_imbalance = float(f1_imbalance(inst, sol)) if matches_formed > 0 else 0.0
        feasible = bool(is_feasible(inst, sol, max_ping_gap=max_ping_gap))

        matched_waits: list[float] = []
        remaining_queue: list[Player] = []
        for idx, player in enumerate(queue):
            if sol.match[idx] >= 0:
                matched_waits.append(player.wait)
            else:
                remaining_queue.append(
                    Player(
                        id=player.id,
                        skill=player.skill,
                        wait=round(player.wait + sim_dt, 6),
                        arrival=player.arrival,
                        role=player.role,
                        ping=player.ping,
                    )
                )

        queue = remaining_queue
        queue_length = len(queue)

        if queue_length > 0:
            rem_waits = [p.wait for p in queue]
            mean_wait = float(np.mean(rem_waits))
            max_wait_val = float(np.max(rem_waits))
        else:
            mean_wait = 0.0
            max_wait_val = 0.0

        if matched_waits:
            matched_mean_wait = float(np.mean(matched_waits))
            matched_max_wait = float(np.max(matched_waits))
        else:
            matched_mean_wait = 0.0
            matched_max_wait = 0.0

        records.append(
            {
                "round": round_idx,
                "time": round((round_idx + 1) * sim_dt, 4),
                "scenario": scenario,
                "matcher": matcher_name,
                "arrivals": n_arrivals,
                "is_burst": is_burst,
                "queue_before": queue_before,
                "queue_length": queue_length,
                "matches_formed": matches_formed,
                "players_matched": len(matched_waits),
                "mean_wait": round(mean_wait, 4),
                "max_wait": round(max_wait_val, 4),
                "matched_mean_wait": round(matched_mean_wait, 4),
                "matched_max_wait": round(matched_max_wait, 4),
                "mean_imbalance": round(mean_imbalance, 4),
                "feasible": feasible,
            }
        )

    return pd.DataFrame(records)


def summarize_rolling_logs(logs: pd.DataFrame) -> dict[str, float | int]:
    """Compute summary statistics across a rolling simulation log DataFrame."""
    if logs.empty:
        return {
            "n_rounds": 0,
            "final_queue_length": 0,
            "max_queue_length": 0,
            "mean_queue_length": 0.0,
            "total_arrivals": 0,
            "total_matches": 0,
            "total_players_matched": 0,
            "mean_wait": 0.0,
            "peak_max_wait": 0.0,
            "mean_imbalance": 0.0,
            "feasibility_rate": 1.0,
        }

    active_rounds = logs[logs["matches_formed"] > 0]
    mean_imb = (
        float(active_rounds["mean_imbalance"].mean())
        if not active_rounds.empty
        else 0.0
    )

    return {
        "n_rounds": int(len(logs)),
        "final_queue_length": int(logs["queue_length"].iloc[-1]),
        "max_queue_length": int(logs["queue_length"].max()),
        "mean_queue_length": float(logs["queue_length"].mean()),
        "total_arrivals": int(logs["arrivals"].sum()),
        "total_matches": int(logs["matches_formed"].sum()),
        "total_players_matched": int(logs["players_matched"].sum()),
        "mean_wait": float(logs["mean_wait"].mean()),
        "peak_max_wait": float(logs["max_wait"].max()),
        "mean_imbalance": mean_imb,
        "feasibility_rate": float(logs["feasible"].mean()),
    }
