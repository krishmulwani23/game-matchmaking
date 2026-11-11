# OWNER: M3 - do not edit unless you are M3
"""Hybrid Greedy + Swap-Based Local Search Optimization (Checkpoint G11).

Implements the hybrid_greedy_ls matcher for queue-based multiplayer matchmaking:
1. Seeds initial candidate solutions using the registered 'greedy_nearest' and 'fifo'
   baseline algorithms via registry.get.
2. Applies swap-based local search across teams, matches, and queue players, accepting
   moves only when the scalarized objective improves and hard constraints remain satisfied.
3. Provides an 'enable_ls' comparison flag so Member 4 can benchmark greedy baseline
   seeding against local search improvements.
4. Returns the non-dominated Pareto-optimal set of feasible solutions.
"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from mm.config import load_config
from mm.encoding import decode, encode, get_match_bounds
from mm.interfaces import Matcher
from mm.objectives import evaluate
from mm.pareto import is_non_dominated
from mm.registry import get as registry_get, register
from mm.repair import _is_group_feasible, repair
from mm.types import Instance, Solution
from mm.validate import is_feasible
from mm.weighted import compute_baseline_normalization, scalarize_objective


def _snake_draft_teams_for_group(inst: Instance, group: Sequence[int]) -> list[int]:
    """Assign binary team designations (0 or 1) using snake draft sorted by skill."""
    group_sorted = sorted(group, key=lambda idx: inst.players[idx].skill)
    teams: list[int] = [0] * len(group)
    for rank, p_idx in enumerate(group_sorted):
        round_idx = rank // 2
        t = 0 if (round_idx % 2 == 0 and rank % 2 == 0) or (round_idx % 2 != 0 and rank % 2 != 0) else 1
        pos = group.index(p_idx)
        teams[pos] = t
    return teams


def local_search_swap(
    inst: Instance,
    initial_sol: Solution,
    weight: float,
    bounds: tuple[float, float, float, float],
    max_evals: int = 60,
    max_ping_gap: float | None = 100.0,
    use_f3: bool = False,
    rng: np.random.Generator | None = None,
) -> Solution:
    """Perform swap-based local search to improve a candidate solution.

    Explores three swap neighborhoods:
    1. Within-match team swaps: Swaps two players between opposing teams in the same match.
       Leaves match membership and pairwise constraints untouched while tuning team balance.
    2. Inter-match player swaps: Swaps players between distinct matches to reduce skill dispersion.
    3. Queue-to-match swaps: Swaps a non-starving match player with an unmatched queue player.

    A move is accepted if and only if it strictly improves the scalarized objective
    J(sol, w) = w * f1_norm + (1 - w) * f2_norm and satisfies all hard constraints.

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance.
    initial_sol : Solution
        Initial feasible solution to improve.
    weight : float
        Scalarization tradeoff weight in [0, 1].
    bounds : tuple[float, float, float, float]
        Normalization bounds (f1_min, f1_max, f2_min, f2_max).
    max_evals : int, optional
        Maximum number of swap neighborhood evaluations. Default is 60.
    max_ping_gap : float | None, optional
        Maximum allowed ping gap. Default is 100.0 ms.
    use_f3 : bool, optional
        Whether f3 objective is active.
    rng : np.random.Generator | None, optional
        RNG for random move ordering.

    Returns
    -------
    Solution
        The improved feasible solution.
    """
    if rng is None:
        rng = np.random.default_rng(inst.seed)

    curr_sol = initial_sol
    fit = evaluate(inst, curr_sol, use_f3=use_f3)
    curr_energy = scalarize_objective(float(fit[0]), float(fit[1]), weight, bounds)

    n_players = len(inst.players)
    eval_count = 0
    improved = True

    while improved and eval_count < max_evals:
        improved = False
        active_matches = [int(m) for m in np.unique(curr_sol.match) if m >= 0]
        if not active_matches:
            break

        # Neighborhood 1: Within-match team swaps (fine-tunes f1 without affecting f2 or feasibility)
        match_order = rng.permutation(active_matches)
        for m in match_order:
            if eval_count >= max_evals:
                break

            t0_players = [i for i in range(n_players) if curr_sol.match[i] == m and curr_sol.team[i] == 0]
            t1_players = [i for i in range(n_players) if curr_sol.match[i] == m and curr_sol.team[i] == 1]

            for p0 in t0_players:
                for p1 in t1_players:
                    eval_count += 1
                    cand_team = curr_sol.team.copy()
                    cand_team[p0] = 1
                    cand_team[p1] = 0

                    cand_sol = Solution(match=curr_sol.match.copy(), team=cand_team)
                    cand_fit = evaluate(inst, cand_sol, use_f3=use_f3)
                    cand_energy = scalarize_objective(float(cand_fit[0]), float(cand_fit[1]), weight, bounds)

                    if cand_energy < curr_energy - 1e-6:
                        curr_sol = cand_sol
                        curr_energy = cand_energy
                        improved = True
                        break
                if improved:
                    break

        if improved:
            continue

        # Neighborhood 2: Inter-match player swaps (migrates players between matches)
        if len(active_matches) >= 2 and eval_count < max_evals:
            m_pairs = list(rng.choice(active_matches, size=(min(10, len(active_matches)), 2), replace=True))
            for m1, m2 in m_pairs:
                if m1 == m2 or eval_count >= max_evals:
                    continue

                m1_members = [i for i in range(n_players) if curr_sol.match[i] == m1]
                m2_members = [i for i in range(n_players) if curr_sol.match[i] == m2]

                for p1 in m1_members:
                    for p2 in m2_members:
                        eval_count += 1
                        cand_m1 = [p2 if x == p1 else x for x in m1_members]
                        cand_m2 = [p1 if x == p2 else x for x in m2_members]

                        # Check pairwise constraints for both candidate matches
                        if _is_group_feasible(inst, cand_m1, max_ping_gap) and _is_group_feasible(inst, cand_m2, max_ping_gap):
                            cand_match = curr_sol.match.copy()
                            cand_team = curr_sol.team.copy()

                            cand_match[p1] = m2
                            cand_match[p2] = m1

                            # Rebalance teams via snake draft
                            t1 = _snake_draft_teams_for_group(inst, cand_m1)
                            for idx, pl in enumerate(cand_m1):
                                cand_team[pl] = t1[idx]

                            t2 = _snake_draft_teams_for_group(inst, cand_m2)
                            for idx, pl in enumerate(cand_m2):
                                cand_team[pl] = t2[idx]

                            cand_sol = Solution(match=cand_match, team=cand_team)
                            if is_feasible(inst, cand_sol, max_ping_gap=max_ping_gap):
                                cand_fit = evaluate(inst, cand_sol, use_f3=use_f3)
                                cand_energy = scalarize_objective(float(cand_fit[0]), float(cand_fit[1]), weight, bounds)

                                if cand_energy < curr_energy - 1e-6:
                                    curr_sol = cand_sol
                                    curr_energy = cand_energy
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break

        if improved:
            continue

        # Neighborhood 3: Queue-to-match swaps (swaps non-starving match player with queue player)
        unmatched = [i for i in range(n_players) if curr_sol.match[i] < 0]
        if unmatched and eval_count < max_evals:
            for m in active_matches:
                if eval_count >= max_evals or improved:
                    break
                m_players = [i for i in range(n_players) if curr_sol.match[i] == m]
                for p in m_players:
                    # Do not displace a starving player
                    if (inst.players[p].wait + inst.dt) > inst.max_wait:
                        continue
                    for u in unmatched:
                        eval_count += 1
                        cand_group = [u if x == p else x for x in m_players]
                        if _is_group_feasible(inst, cand_group, max_ping_gap):
                            cand_match = curr_sol.match.copy()
                            cand_team = curr_sol.team.copy()
                            cand_match[p] = -1
                            cand_team[p] = 0
                            cand_match[u] = m

                            teams = _snake_draft_teams_for_group(inst, cand_group)
                            for idx, pl in enumerate(cand_group):
                                cand_team[pl] = teams[idx]

                            cand_sol = Solution(match=cand_match, team=cand_team)
                            if is_feasible(inst, cand_sol, max_ping_gap=max_ping_gap):
                                cand_fit = evaluate(inst, cand_sol, use_f3=use_f3)
                                cand_energy = scalarize_objective(float(cand_fit[0]), float(cand_fit[1]), weight, bounds)

                                if cand_energy < curr_energy - 1e-6:
                                    curr_sol = cand_sol
                                    curr_energy = cand_energy
                                    improved = True
                                    break
                    if improved:
                        break

    return curr_sol


def hybrid_greedy_ls(
    inst: Instance,
    enable_ls: bool = True,
    seed: int | None = None,
    weights: Sequence[float] | None = None,
    max_evals: int = 60,
    use_f3: bool | None = None,
    max_ping_gap: float | None = 100.0,
    **kw: Any,
) -> list[Solution]:
    """Solve matchmaking using Hybrid Greedy Seeding + Swap-Based Local Search.

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance.
    enable_ls : bool, optional
        Whether to apply swap-based local search to the seeded candidates.
        If False, returns the seeded greedy baselines directly without local search.
        Default is True.
    seed : int | None, optional
        RNG seed for deterministic execution. Default is inst.seed.
    weights : Sequence[float] | None, optional
        Tradeoff weights w in [0, 1] to explore during local search.
        If None, a standard grid np.linspace(0.0, 1.0, 5) is used.
    max_evals : int, optional
        Maximum swap evaluations per local search pass. Default is 60.
    use_f3 : bool | None, optional
        Whether to evaluate the 3rd objective (role/ping mismatch). Default is False.
    max_ping_gap : float | None, optional
        Maximum allowable ping gap in matches. Default is 100.0 ms.

    Returns
    -------
    list[Solution]
        The non-dominated Pareto-optimal set of feasible solutions.
    """
    n_players = len(inst.players)
    if n_players == 0:
        return [Solution.empty(0)]

    resolved_seed = inst.seed if seed is None else int(seed)

    try:
        cfg = load_config()
        cfg_use_f3 = cfg.use_f3
    except Exception:
        cfg_use_f3 = False
    resolved_use_f3 = cfg_use_f3 if use_f3 is None else bool(use_f3)

    # 1. Seed Initial Candidate Population from greedy_nearest and fifo only
    seed_baselines: list[Solution] = []

    # Baseline 1: greedy_nearest
    try:
        gn_matcher = registry_get("greedy_nearest")
        sol_gn = gn_matcher.solve(inst)[0]
        if is_feasible(inst, sol_gn, max_ping_gap=max_ping_gap):
            seed_baselines.append(sol_gn)
    except Exception:
        pass

    # Baseline 2: fifo
    try:
        fifo_matcher = registry_get("fifo")
        sol_fifo = fifo_matcher.solve(inst)[0]
        if is_feasible(inst, sol_fifo, max_ping_gap=max_ping_gap):
            seed_baselines.append(sol_fifo)
    except Exception:
        pass

    # Safety fallback
    if not seed_baselines:
        seed_baselines.append(repair(inst, Solution.empty(n_players), max_ping_gap=max_ping_gap))

    # Derive match-count candidate variations from the two baseline seeds across [min_m, max_m]
    min_m, max_m = get_match_bounds(inst)
    candidate_solutions: list[Solution] = list(seed_baselines)

    for base_sol in seed_baselines:
        g = encode(base_sol, inst)
        for m in range(min_m, max_m + 1):
            s_cand = repair(inst, decode(inst, g, n_matches=m), max_ping_gap=max_ping_gap)
            if is_feasible(inst, s_cand, max_ping_gap=max_ping_gap):
                candidate_solutions.append(s_cand)

    # If local search is disabled, return the non-dominated set of baseline seeds without local search
    if not enable_ls:
        fits = np.array([evaluate(inst, s, use_f3=resolved_use_f3) for s in candidate_solutions], dtype=float)
        nd_mask = is_non_dominated(fits)
        return [candidate_solutions[i] for i in range(len(candidate_solutions)) if nd_mask[i]]

    # 2. Swap-Based Local Search
    bounds = compute_baseline_normalization(inst, use_f3=resolved_use_f3)
    target_weights = np.linspace(0.0, 1.0, 5) if weights is None else np.asarray(weights, dtype=float)

    collected_solutions: list[Solution] = list(candidate_solutions)

    # Apply local search to candidate solutions across the target weights
    for cand_idx, seed_sol in enumerate(candidate_solutions):
        for w_idx, w in enumerate(target_weights):
            step_seed = resolved_seed + cand_idx * 100 + w_idx
            step_rng = np.random.default_rng(step_seed)
            improved = local_search_swap(
                inst=inst,
                initial_sol=seed_sol,
                weight=float(w),
                bounds=bounds,
                max_evals=max_evals,
                max_ping_gap=max_ping_gap,
                use_f3=resolved_use_f3,
                rng=step_rng,
            )
            if is_feasible(inst, improved, max_ping_gap=max_ping_gap):
                collected_solutions.append(improved)

    # 3. Filter to Non-Dominated Pareto Set & Deduplicate
    all_fitness = np.array(
        [evaluate(inst, s, use_f3=resolved_use_f3) for s in collected_solutions],
        dtype=float,
    )
    nd_mask = is_non_dominated(all_fitness)
    pareto_candidates = [collected_solutions[i] for i in range(len(collected_solutions)) if nd_mask[i]]
    pareto_fitness = all_fitness[nd_mask]

    # Deduplicate solutions with identical objective coordinates
    unique_solutions: list[Solution] = []
    seen_fits: list[np.ndarray] = []

    for idx, sol in enumerate(pareto_candidates):
        fit_val = pareto_fitness[idx]
        is_dup = False
        for seen in seen_fits:
            if np.allclose(fit_val, seen, atol=1e-4):
                is_dup = True
                break
        if not is_dup:
            seen_fits.append(fit_val)
            unique_solutions.append(sol)

    # Sort final front by f1 ascending then f2
    if unique_solutions:
        sort_keys = [(seen_fits[i][0], seen_fits[i][1]) for i in range(len(unique_solutions))]
        sorted_order = sorted(range(len(unique_solutions)), key=lambda k: sort_keys[k])
        unique_solutions = [unique_solutions[i] for i in sorted_order]

    return unique_solutions


@register("hybrid_greedy_ls")
class HybridGreedyLSMatcher:
    """Registered Matcher for Hybrid Greedy + Swap-Based Local Search."""

    name: str = "hybrid_greedy_ls"

    def __init__(self, **default_kw: Any) -> None:
        self.default_kw = default_kw

    def solve(self, inst: Instance, **kw: Any) -> list[Solution]:
        """Solve matchmaking instance and return the non-dominated Pareto front.

        Parameters
        ----------
        inst : Instance
            The matchmaking problem instance.
        **kw : Any
            Optional parameter overrides (enable_ls, weights, max_evals, seed, use_f3, etc.).

        Returns
        -------
        list[Solution]
            List of feasible solutions forming the non-dominated front.
        """
        combined = {**self.default_kw, **kw}
        return hybrid_greedy_ls(inst, **combined)
