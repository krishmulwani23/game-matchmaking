# OWNER: M2 - do not edit unless you are M2
"""Matchmaking optimization objectives.

Defines the core evaluation metrics for matchmaking solutions:
- f1_imbalance: Skill difference between teams within matches (minimized).
- f2_wait: Total waiting time of players remaining unmatched in queue (minimized).
- f3_mismatch: Optional role and ping mismatch penalty (minimized).
- evaluate: Multi-objective vector evaluator returning [f1, f2] or [f1, f2, f3].

Optimized with vectorized NumPy routines for high-throughput calls during
evolutionary search (NSGA-II, simulated annealing, genetic algorithms).
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from mm.types import Instance, Solution

if TYPE_CHECKING:
    from mm.config import Config


def _get_instance_arrays(inst: Instance) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve or cache vectorized skill and wait arrays for an Instance.

    Caching on the instance avoids repeated Python list and attribute lookups
    across tens of thousands of evaluation calls in genetic algorithms.
    """
    skills = getattr(inst, "_cached_skills", None)
    waits = getattr(inst, "_cached_waits", None)
    n_players = len(inst.players)

    if skills is None or len(skills) != n_players:
        skills = np.array([p.skill for p in inst.players], dtype=float)
        waits = np.array([p.wait for p in inst.players], dtype=float)
        object.__setattr__(inst, "_cached_skills", skills) if hasattr(inst, "__frozen__") else setattr(inst, "_cached_skills", skills)
        object.__setattr__(inst, "_cached_waits", waits) if hasattr(inst, "__frozen__") else setattr(inst, "_cached_waits", waits)

    return skills, waits


def f1_imbalance(
    inst: Instance,
    sol: Solution,
    include_spread: bool = False,
) -> float:
    """Compute the mean skill imbalance across all formed matches.

    Definition:
        mean over formed matches m of |mean_skill(team 0) - mean_skill(team 1)|.
        If no matches are formed, returns 0.0.

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance.
    sol : Solution
        The proposed match and team assignment solution.
    include_spread : bool, optional
        If True, adds the mean within-team skill spread to the match imbalance.
        Default is False.

    Returns
    -------
    float
        The average skill imbalance across formed matches.
    """
    match = sol.match
    matched_mask = match >= 0

    if not np.any(matched_mask):
        return 0.0

    skills, _ = _get_instance_arrays(inst)

    matched_match_ids = match[matched_mask]
    matched_teams = sol.team[matched_mask]
    matched_skills = skills[matched_mask]

    # Map arbitrary match IDs to dense 0..M-1 indices
    unique_matches, match_indices = np.unique(matched_match_ids, return_inverse=True)
    n_matches = len(unique_matches)

    if n_matches == 0:
        return 0.0

    # Bins: 2 * match_idx + team_id (team_id in {0, 1})
    bin_ids = match_indices * 2 + matched_teams
    total_bins = 2 * n_matches

    sums = np.bincount(bin_ids, weights=matched_skills, minlength=total_bins)
    counts = np.bincount(bin_ids, minlength=total_bins)

    t0_sums = sums[0::2]
    t0_counts = counts[0::2]
    t1_sums = sums[1::2]
    t1_counts = counts[1::2]

    # Guard against empty teams in invalid/partial solutions
    t0_means = np.divide(t0_sums, t0_counts, out=np.zeros_like(t0_sums), where=t0_counts > 0)
    t1_means = np.divide(t1_sums, t1_counts, out=np.zeros_like(t1_sums), where=t1_counts > 0)

    imbalances = np.abs(t0_means - t1_means)

    if include_spread:
        # Optional within-team skill spread: mean difference between max and min per team
        # Vectorized via bincount or grouped aggregations
        spreads = np.zeros(n_matches, dtype=float)
        for idx in range(n_matches):
            mask_0 = (match_indices == idx) & (matched_teams == 0)
            mask_1 = (match_indices == idx) & (matched_teams == 1)
            sp0 = (np.ptp(matched_skills[mask_0])) if np.count_nonzero(mask_0) > 1 else 0.0
            sp1 = (np.ptp(matched_skills[mask_1])) if np.count_nonzero(mask_1) > 1 else 0.0
            spreads[idx] = 0.5 * (sp0 + sp1)
        imbalances = imbalances + spreads

    return float(np.mean(imbalances))


def f2_wait(inst: Instance, sol: Solution) -> float:
    """Compute the total waiting time of unmatched players remaining in queue.

    Definition:
        sum over unmatched i of (wait_i + dt).
        If all players are matched, returns 0.0.

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance containing player wait times and dt.
    sol : Solution
        The proposed match and team assignment solution.

    Returns
    -------
    float
        The total accumulated waiting time including the additional round length dt.
    """
    match = sol.match
    unmatched_mask = match < 0

    if not np.any(unmatched_mask):
        return 0.0

    _, waits = _get_instance_arrays(inst)
    unmatched_waits = waits[unmatched_mask]
    n_unmatched = len(unmatched_waits)

    return float(np.sum(unmatched_waits) + n_unmatched * inst.dt)


def f3_mismatch(inst: Instance, sol: Solution) -> float:
    """Compute role and ping mismatch penalty across formed matches.

    Combines role coverage deficit per team with network latency spread:
    - For each formed match m:
        - target_roles = min(inst.k, count_distinct_available_roles)
        - team_0_deficit = max(0, target_roles - unique_roles(team 0))
        - team_1_deficit = max(0, target_roles - unique_roles(team 1))
        - role_deficit = 0.5 * (team_0_deficit + team_1_deficit)
        - ping_spread = max(ping) - min(ping)
        - match_mismatch = ping_spread + role_deficit
    - Overall f3 is the arithmetic mean across all formed matches.

    Returns 0.0 when role or ping data is missing or not modelled,
    or when no matches are formed.

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance.
    sol : Solution
        The proposed match and team assignment solution.

    Returns
    -------
    float
        Mismatch objective value (0.0 if role/ping missing or no matches).
    """
    if sol.n_matches() == 0 or len(inst.players) == 0:
        return 0.0

    # If any player lacks role or ping, return 0.0
    has_roles = all(p.role is not None for p in inst.players)
    has_pings = all(p.ping is not None for p in inst.players)

    if not (has_roles and has_pings):
        return 0.0

    match = sol.match
    matched_mask = match >= 0
    if not np.any(matched_mask):
        return 0.0

    matched_match_ids = match[matched_mask]
    matched_teams = sol.team[matched_mask]

    pings = np.array([p.ping for p in inst.players], dtype=float)[matched_mask]
    roles = np.array([p.role for p in inst.players], dtype=int)[matched_mask]

    unique_matches = np.unique(matched_match_ids)
    all_distinct_roles = len(set(p.role for p in inst.players if p.role is not None))
    target_roles = min(inst.k, all_distinct_roles)

    mismatch_vals: list[float] = []
    for m in unique_matches:
        mask_m = matched_match_ids == m
        m_pings = pings[mask_m]
        ping_spread = float(np.ptp(m_pings)) if len(m_pings) > 1 else 0.0

        m_teams = matched_teams[mask_m]
        m_roles = roles[mask_m]

        t0_roles = m_roles[m_teams == 0]
        t1_roles = m_roles[m_teams == 1]

        u0 = len(np.unique(t0_roles)) if len(t0_roles) > 0 else 0
        u1 = len(np.unique(t1_roles)) if len(t1_roles) > 0 else 0

        deficit_0 = max(0, target_roles - u0)
        deficit_1 = max(0, target_roles - u1)
        role_deficit = 0.5 * (deficit_0 + deficit_1)

        mismatch_vals.append(ping_spread + role_deficit)

    return float(np.mean(mismatch_vals)) if mismatch_vals else 0.0


def evaluate(
    inst: Instance,
    sol: Solution,
    use_f3: bool | None = None,
    cfg: Config | None = None,
) -> np.ndarray:
    """Evaluate a matchmaking solution on the defined objectives.

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance.
    sol : Solution
        The proposed match assignments.
    use_f3 : bool | None, optional
        Whether to include f3_mismatch in the returned objectives.
        If None, inspects cfg.use_f3 if cfg is provided, else False.
    cfg : Config | None, optional
        Optional project configuration to resolve use_f3.

    Returns
    -------
    np.ndarray
        Array of objective values, shape (2,) [f1, f2] or shape (3,) [f1, f2, f3].
        All objectives are to be minimized.
    """
    if use_f3 is None:
        use_f3 = cfg.use_f3 if cfg is not None else False

    f1 = f1_imbalance(inst, sol)
    f2 = f2_wait(inst, sol)

    if use_f3:
        f3 = f3_mismatch(inst, sol)
        return np.array([f1, f2, f3], dtype=float)

    return np.array([f1, f2], dtype=float)
