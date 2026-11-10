# OWNER: M2 - do not edit unless you are M2
"""Matchmaking Performance & Queue Evaluation Metrics (Checkpoint G9).

Provides standardized operational and evaluation metrics:
- Matched player wait statistics: mean, median, 95th percentile, and max wait.
- Queue backlog: queue length remaining after the round.
- Throughput: match rate (fraction of queued players matched).
- Imbalance statistics: mean and maximum team skill discrepancy across matches.
- Runtime performance: timing utility wrapper / decorator.
- Feasibility verification: feasibility rate across solution batches.
- Comprehensive metric aggregation via `compute_metrics`.
"""
from __future__ import annotations

import functools
import time
from typing import Any, Callable, Sequence, TypeVar
import numpy as np

from mm.objectives import f1_imbalance
from mm.types import Instance, Solution
from mm.validate import is_feasible

T = TypeVar("T")


def _get_matched_waits(inst: Instance, sol: Solution) -> np.ndarray:
    """Extract wait times of all players placed into matches."""
    if len(inst.players) == 0 or len(sol.match) == 0:
        return np.empty(0, dtype=float)

    match = np.asarray(sol.match)
    matched_mask = match >= 0
    if not np.any(matched_mask):
        return np.empty(0, dtype=float)

    waits = np.array([p.wait for p in inst.players], dtype=float)
    return waits[matched_mask]


def matched_wait_mean(inst: Instance, sol: Solution) -> float:
    """Compute the arithmetic mean wait time of matched players.

    Returns 0.0 if no players were matched.
    """
    mw = _get_matched_waits(inst, sol)
    return float(np.mean(mw)) if len(mw) > 0 else 0.0


def matched_wait_median(inst: Instance, sol: Solution) -> float:
    """Compute the median wait time of matched players.

    Returns 0.0 if no players were matched.
    """
    mw = _get_matched_waits(inst, sol)
    return float(np.median(mw)) if len(mw) > 0 else 0.0


def matched_wait_p95(inst: Instance, sol: Solution) -> float:
    """Compute the 95th percentile wait time of matched players.

    Returns 0.0 if no players were matched.
    """
    mw = _get_matched_waits(inst, sol)
    return float(np.percentile(mw, 95)) if len(mw) > 0 else 0.0


def matched_wait_max(inst: Instance, sol: Solution) -> float:
    """Compute the maximum wait time among matched players.

    Returns 0.0 if no players were matched.
    """
    mw = _get_matched_waits(inst, sol)
    return float(np.max(mw)) if len(mw) > 0 else 0.0


def queue_length_after(inst: Instance, sol: Solution) -> int:
    """Return the number of unmatched players remaining in the queue."""
    if len(sol.match) == 0:
        return 0
    match = np.asarray(sol.match)
    return int(np.sum(match < 0))


def match_rate(inst: Instance, sol: Solution) -> float:
    """Compute the proportion of queued players matched in this round.

    Returns a value in [0.0, 1.0]. Returns 0.0 if the queue is empty.
    """
    n = len(inst.players)
    if n == 0 or len(sol.match) == 0:
        return 0.0
    match = np.asarray(sol.match)
    return float(np.sum(match >= 0) / n)


def mean_imbalance(
    inst: Instance,
    sol: Solution,
    include_spread: bool = False,
) -> float:
    """Compute the mean skill imbalance across all formed matches.

    Equivalent to f1_imbalance. Returns 0.0 if no matches are formed.
    """
    return float(f1_imbalance(inst, sol, include_spread=include_spread))


def max_imbalance(inst: Instance, sol: Solution) -> float:
    """Compute the maximum team skill imbalance across formed matches.

    For each formed match m:
        Delta_m = |mean_skill(team 0) - mean_skill(team 1)|
    Returns max(Delta_m), or 0.0 if no matches are formed.
    """
    if sol.n_matches() == 0 or len(inst.players) == 0:
        return 0.0

    match = np.asarray(sol.match)
    team = np.asarray(sol.team)
    matched_mask = match >= 0
    if not np.any(matched_mask):
        return 0.0

    matched_matches = match[matched_mask]
    matched_teams = team[matched_mask]
    skills = np.array([p.skill for p in inst.players], dtype=float)[matched_mask]

    unique_matches, match_idx = np.unique(matched_matches, return_inverse=True)
    n_matches = len(unique_matches)
    if n_matches == 0:
        return 0.0

    bin_ids = match_idx * 2 + matched_teams
    total_bins = 2 * n_matches

    sums = np.bincount(bin_ids, weights=skills, minlength=total_bins)
    counts = np.bincount(bin_ids, minlength=total_bins)

    t0_sums = sums[0::2]
    t0_counts = counts[0::2]
    t1_sums = sums[1::2]
    t1_counts = counts[1::2]

    t0_means = np.divide(t0_sums, t0_counts, out=np.zeros_like(t0_sums), where=t0_counts > 0)
    t1_means = np.divide(t1_sums, t1_counts, out=np.zeros_like(t1_sums), where=t1_counts > 0)

    imbalances = np.abs(t0_means - t1_means)
    return float(np.max(imbalances)) if len(imbalances) > 0 else 0.0


def time_execution(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> tuple[T, float]:
    """Execute a callable and measure elapsed wall-clock time in seconds.

    Parameters
    ----------
    func : Callable
        Function or callable to execute.
    *args : Any
        Positional arguments passed to func.
    **kwargs : Any
        Keyword arguments passed to func.

    Returns
    -------
    tuple[T, float]
        (result, elapsed_seconds)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def timed(func: Callable[..., T]) -> Callable[..., tuple[T, float]]:
    """Decorator to wrap a function and return (result, elapsed_seconds)."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[T, float]:
        return time_execution(func, *args, **kwargs)

    return wrapper


def feasibility_rate(
    inst: Instance,
    solutions: Sequence[Solution],
    max_ping_gap: float | None = 100.0,
) -> float:
    """Compute the fraction of candidate solutions that are feasible.

    Parameters
    ----------
    inst : Instance
        The problem instance.
    solutions : Sequence[Solution]
        List or sequence of solutions to evaluate.
    max_ping_gap : float | None, optional
        Maximum allowed ping gap limit. Default is 100.0 ms.

    Returns
    -------
    float
        Proportion of feasible solutions in [0.0, 1.0]. Returns 1.0 if empty.
    """
    if len(solutions) == 0:
        return 1.0

    n_feasible = sum(
        1 for sol in solutions if is_feasible(inst, sol, max_ping_gap=max_ping_gap)
    )
    return float(n_feasible / len(solutions))


def compute_metrics(
    inst: Instance,
    sol: Solution,
    runtime_s: float | None = None,
    max_ping_gap: float | None = 100.0,
) -> dict[str, Any]:
    """Compute a comprehensive dictionary of all matchmaking metrics for a solution.

    Parameters
    ----------
    inst : Instance
        The problem instance.
    sol : Solution
        The proposed solution.
    runtime_s : float | None, optional
        Measured execution runtime in seconds, if available.
    max_ping_gap : float | None, optional
        Maximum allowed ping gap limit. Default is 100.0 ms.

    Returns
    -------
    dict[str, Any]
        Dictionary of computed performance and operational metrics.
    """
    metrics: dict[str, Any] = {
        "n_matches": sol.n_matches(),
        "queue_length": queue_length_after(inst, sol),
        "match_rate": round(match_rate(inst, sol), 4),
        "mean_imbalance": round(mean_imbalance(inst, sol), 4),
        "max_imbalance": round(max_imbalance(inst, sol), 4),
        "matched_wait_mean": round(matched_wait_mean(inst, sol), 2),
        "matched_wait_median": round(matched_wait_median(inst, sol), 2),
        "matched_wait_p95": round(matched_wait_p95(inst, sol), 2),
        "matched_wait_max": round(matched_wait_max(inst, sol), 2),
        "is_feasible": is_feasible(inst, sol, max_ping_gap=max_ping_gap),
    }
    if runtime_s is not None:
        metrics["runtime_s"] = round(runtime_s, 6)
    return metrics
