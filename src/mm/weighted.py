# OWNER: M3 - do not edit unless you are M3
"""Weighted scalarization optimization for queue-based matchmaking (Checkpoint G6).

Implements single-objective metaheuristics minimizing the scalarized tradeoff:
    J(sol, w) = w * f1_norm + (1 - w) * f2_norm

Features:
- sa_weighted: Simulated Annealing with geometric cooling.
- ga_weighted: Single-objective Genetic Algorithm with tournament selection & elitism.
- Normalization constants computed from baseline matchers evaluated on the same instance.
- sweep_weights: Systematic sweep across weight combinations (e.g., np.linspace(0, 1, 11)).
- Self-registration under 'sa_weighted' and 'ga_weighted'.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence
import numpy as np

from mm.config import load_config
from mm.encoding import (
    Genome,
    crossover_ox,
    decode,
    get_match_bounds,
    mutate_insert,
    mutate_inversion,
    mutate_swap,
    random_genome,
)
from mm.interfaces import Matcher
from mm.objectives import evaluate
from mm.registry import register
from mm.repair import repair
from mm.types import Instance, Solution


def compute_baseline_normalization(
    inst: Instance,
    use_f3: bool = False,
) -> tuple[float, float, float, float]:
    """Compute normalization constants (f1_min, f1_max, f2_min, f2_max) from baselines.

    Evaluates baseline solutions (sort_skill, fifo, greedy_nearest, empty, and full matching)
    on the instance to establish problem-scale normalization constants for f1 and f2.

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance.
    use_f3 : bool, optional
        Whether f3 objective is active.

    Returns
    -------
    tuple[float, float, float, float]
        (f1_min, f1_max, f2_min, f2_max)
    """
    cached = getattr(inst, "_cached_norm_bounds", None)
    if cached is not None:
        return cached

    n = len(inst.players)
    if n == 0:
        return (0.0, 1.0, 0.0, 1.0)

    match_size = 2 * inst.k
    f1_vals: list[float] = []
    f2_vals: list[float] = []

    # 1. Empty solution (all unmatched, high f2, 0.0 f1)
    sol_empty = repair(inst, Solution.empty(n))
    e_val = evaluate(inst, sol_empty, use_f3=use_f3)
    f1_vals.append(float(e_val[0]))
    f2_vals.append(float(e_val[1]))

    # 2. sort_skill fixture: sort by skill, snake draft
    if n >= match_size:
        sp = sorted(range(n), key=lambda i: inst.players[i].skill)
        m_arr = np.full(n, -1, dtype=int)
        t_arr = np.zeros(n, dtype=int)
        for m_idx in range(n // match_size):
            block = sp[m_idx * match_size : (m_idx + 1) * match_size]
            for r, p in enumerate(block):
                m_arr[p] = m_idx
                t_arr[p] = 0 if r in (0, 3) else 1
        sol_sort = repair(inst, Solution(match=m_arr, team=t_arr))
        s_val = evaluate(inst, sol_sort, use_f3=use_f3)
        f1_vals.append(float(s_val[0]))
        f2_vals.append(float(s_val[1]))

    # 3. fifo fixture: sort by wait descending, snake draft
    if n >= match_size:
        fp = sorted(range(n), key=lambda i: inst.players[i].wait, reverse=True)
        m_arr = np.full(n, -1, dtype=int)
        t_arr = np.zeros(n, dtype=int)
        for m_idx in range(n // match_size):
            block = fp[m_idx * match_size : (m_idx + 1) * match_size]
            for r, p in enumerate(block):
                m_arr[p] = m_idx
                t_arr[p] = 0 if r in (0, 3) else 1
        sol_fifo = repair(inst, Solution(match=m_arr, team=t_arr))
        f_val = evaluate(inst, sol_fifo, use_f3=use_f3)
        f1_vals.append(float(f_val[0]))
        f2_vals.append(float(f_val[1]))

    # 4. greedy_nearest fixture: oldest player paired with nearest skill
    if n >= match_size:
        unmatched = set(range(n))
        m_arr = np.full(n, -1, dtype=int)
        t_arr = np.zeros(n, dtype=int)
        m_idx = 0
        while len(unmatched) >= match_size:
            oldest = max(unmatched, key=lambda i: inst.players[i].wait)
            unmatched.remove(oldest)
            group = [oldest]
            for _ in range(match_size - 1):
                nearest = min(
                    unmatched,
                    key=lambda i: abs(inst.players[i].skill - inst.players[oldest].skill),
                )
                unmatched.remove(nearest)
                group.append(nearest)
            group.sort(key=lambda i: inst.players[i].skill)
            for r, p in enumerate(group):
                m_arr[p] = m_idx
                t_arr[p] = 0 if r in (0, 3) else 1
            m_idx += 1
        sol_gn = repair(inst, Solution(match=m_arr, team=t_arr))
        gn_val = evaluate(inst, sol_gn, use_f3=use_f3)
        f1_vals.append(float(gn_val[0]))
        f2_vals.append(float(gn_val[1]))

    # Compute bounds with safety margin
    f1_min = float(min(f1_vals))
    f1_max = float(max(f1_vals))
    f2_min = float(min(f2_vals))
    f2_max = float(max(f2_vals))

    bounds = (f1_min, f1_max, f2_min, f2_max)
    object.__setattr__(inst, "_cached_norm_bounds", bounds) if hasattr(inst, "__frozen__") else setattr(inst, "_cached_norm_bounds", bounds)
    return bounds


def scalarize_objective(
    f1: float,
    f2: float,
    weight: float,
    bounds: tuple[float, float, float, float],
) -> float:
    """Compute the normalized scalar objective: w * f1_norm + (1 - w) * f2_norm."""
    f1_min, f1_max, f2_min, f2_max = bounds
    df1 = max(f1_max - f1_min, 1.0)
    df2 = max(f2_max - f2_min, 1.0)

    f1_norm = (f1 - f1_min) / df1
    f2_norm = (f2 - f2_min) / df2

    w = float(np.clip(weight, 0.0, 1.0))
    return float(w * f1_norm + (1.0 - w) * f2_norm)


def sa_weighted(
    inst: Instance,
    weight: float = 0.5,
    seed: int | None = None,
    n_steps: int = 50,
    iter_per_step: int = 15,
    t_start: float = 1.0,
    cooling_rate: float = 0.92,
    use_f3: bool | None = None,
    max_ping_gap: float | None = 100.0,
    **kw: Any,
) -> Solution:
    """Solve matchmaking using Simulated Annealing with scalarized weighted objective.

    Minimizes w * f1_norm + (1 - w) * f2_norm using geometric cooling and Metropolis acceptance.

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance.
    weight : float, optional
        Tradeoff weight w in [0, 1] for skill imbalance vs waiting time. Default is 0.5.
    seed : int | None, optional
        RNG seed for deterministic execution. If None, uses inst.seed.
    n_steps : int, optional
        Number of temperature reduction steps. Default is 50.
    iter_per_step : int, optional
        Number of Markov iterations per temperature level. Default is 15.
    t_start : float, optional
        Initial temperature. Default is 1.0.
    cooling_rate : float, optional
        Geometric cooling decay factor alpha. Default is 0.92.
    use_f3 : bool | None, optional
        Whether f3 objective is active.
    max_ping_gap : float | None, optional
        Maximum allowed ping gap within matches. Default is 100.0.

    Returns
    -------
    Solution
        The best feasible solution discovered during the annealing run.
    """
    n_players = len(inst.players)
    if n_players == 0:
        return Solution.empty(0)

    resolved_seed = inst.seed if seed is None else int(seed)
    rng = np.random.default_rng(resolved_seed)
    resolved_w = float(np.clip(weight, 0.0, 1.0))
    bounds = compute_baseline_normalization(inst, use_f3=bool(use_f3))
    min_m, max_m = get_match_bounds(inst)

    def _eval(sol: Solution) -> tuple[float, float, float]:
        fit = evaluate(inst, sol, use_f3=bool(use_f3))
        f1, f2 = float(fit[0]), float(fit[1])
        energy = scalarize_objective(f1, f2, resolved_w, bounds)
        return energy, f1, f2

    # Initial state initialization: bias match count based on weight
    if resolved_w > 0.8:
        init_m = min_m
    elif resolved_w < 0.2:
        init_m = max_m
    else:
        init_m = int(rng.integers(min_m, max_m + 1)) if max_m >= min_m else 0

    curr_g = random_genome(inst, rng=rng, n_matches=init_m)
    curr_sol = repair(inst, decode(inst, curr_g), max_ping_gap=max_ping_gap)
    curr_energy, _, _ = _eval(curr_sol)

    best_sol = curr_sol
    best_energy = curr_energy

    T = t_start
    for _ in range(n_steps):
        for _ in range(iter_per_step):
            # Neighbor generation via mutation
            cand_perm = curr_g.perm.copy()
            cand_m = curr_g.n_matches

            op = rng.choice(["swap", "insert", "inversion"])
            if op == "swap":
                cand_perm = mutate_swap(cand_perm, rng=rng)
            elif op == "insert":
                cand_perm = mutate_insert(cand_perm, rng=rng)
            else:
                cand_perm = mutate_inversion(cand_perm, rng=rng)

            # Match count exploration
            if rng.random() < 0.35 and max_m > min_m:
                delta = int(rng.choice([-1, 1]))
                cand_m = int(np.clip(cand_m + delta, min_m, max_m))

            cand_g = Genome(perm=cand_perm, n_matches=cand_m)
            cand_sol = repair(inst, decode(inst, cand_g), max_ping_gap=max_ping_gap)
            cand_energy, _, _ = _eval(cand_sol)

            # Metropolis acceptance criterion
            dE = cand_energy - curr_energy
            if dE <= 0.0 or rng.random() < np.exp(-dE / max(T, 1e-7)):
                curr_g = cand_g
                curr_sol = cand_sol
                curr_energy = cand_energy

                if curr_energy < best_energy:
                    best_energy = curr_energy
                    best_sol = curr_sol

        T *= cooling_rate

    return best_sol


def ga_weighted(
    inst: Instance,
    weight: float = 0.5,
    seed: int | None = None,
    pop_size: int = 30,
    n_gen: int = 25,
    crossover_prob: float = 0.9,
    mutation_prob: float = 0.25,
    use_f3: bool | None = None,
    max_ping_gap: float | None = 100.0,
    **kw: Any,
) -> Solution:
    """Solve matchmaking using a Single-Objective GA with scalarized weighted objective.

    Minimizes w * f1_norm + (1 - w) * f2_norm using tournament selection and elitism.

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance.
    weight : float, optional
        Tradeoff weight w in [0, 1]. Default is 0.5.
    seed : int | None, optional
        RNG seed for deterministic execution. If None, uses inst.seed.
    pop_size : int, optional
        Population size. Default is 30.
    n_gen : int, optional
        Generations. Default is 25.
    crossover_prob : float, optional
        Order Crossover (OX1) probability. Default is 0.9.
    mutation_prob : float, optional
        Mutation probability. Default is 0.25.
    use_f3 : bool | None, optional
        Whether f3 objective is active.
    max_ping_gap : float | None, optional
        Maximum allowed ping gap. Default is 100.0.

    Returns
    -------
    Solution
        The best feasible solution discovered by the genetic algorithm.
    """
    n_players = len(inst.players)
    if n_players == 0:
        return Solution.empty(0)

    resolved_seed = inst.seed if seed is None else int(seed)
    rng = np.random.default_rng(resolved_seed)
    resolved_w = float(np.clip(weight, 0.0, 1.0))
    bounds = compute_baseline_normalization(inst, use_f3=bool(use_f3))
    min_m, max_m = get_match_bounds(inst)

    def _eval(sol: Solution) -> float:
        fit = evaluate(inst, sol, use_f3=bool(use_f3))
        f1, f2 = float(fit[0]), float(fit[1])
        return scalarize_objective(f1, f2, resolved_w, bounds)

    # 1. Population Initialization
    population_genomes: list[Genome] = []

    # Seed extreme match count individuals
    init_m = min_m if resolved_w > 0.5 else max_m
    population_genomes.append(random_genome(inst, rng=rng, n_matches=init_m))

    while len(population_genomes) < pop_size:
        m_cnt = int(rng.integers(min_m, max_m + 1)) if max_m >= min_m else 0
        population_genomes.append(random_genome(inst, rng=rng, n_matches=m_cnt))

    population_sols = [
        repair(inst, decode(inst, g), max_ping_gap=max_ping_gap)
        for g in population_genomes
    ]
    fitnesses = [_eval(s) for s in population_sols]

    # 2. Generational Evolution
    for _ in range(n_gen):
        # 2a. Elitism: retain top 2 best individuals
        elite_order = np.argsort(fitnesses)
        new_genomes = [population_genomes[elite_order[0]], population_genomes[elite_order[1]]]
        new_sols = [population_sols[elite_order[0]], population_sols[elite_order[1]]]
        new_fitnesses = [fitnesses[elite_order[0]], fitnesses[elite_order[1]]]

        # 2b. Selection and Reproduction
        while len(new_genomes) < pop_size:
            # Tournament selection
            i1, i2 = rng.choice(pop_size, size=2, replace=False)
            p1_idx = i1 if fitnesses[i1] <= fitnesses[i2] else i2

            i3, i4 = rng.choice(pop_size, size=2, replace=False)
            p2_idx = i3 if fitnesses[i3] <= fitnesses[i4] else i4

            p1 = population_genomes[p1_idx]
            p2 = population_genomes[p2_idx]

            # Crossover
            if rng.random() < crossover_prob:
                c1_perm, _ = crossover_ox(p1.perm, p2.perm, rng=rng)
                c1_m = p1.n_matches if rng.random() < 0.5 else p2.n_matches
            else:
                c1_perm = p1.perm.copy()
                c1_m = p1.n_matches

            # Mutation
            if rng.random() < mutation_prob:
                op = rng.choice(["swap", "insert", "inversion"])
                if op == "swap":
                    c1_perm = mutate_swap(c1_perm, rng=rng)
                elif op == "insert":
                    c1_perm = mutate_insert(c1_perm, rng=rng)
                else:
                    c1_perm = mutate_inversion(c1_perm, rng=rng)

            # Match count exploration
            if rng.random() < 0.35 and max_m > min_m:
                delta = int(rng.choice([-1, 1]))
                c1_m = int(np.clip(c1_m + delta, min_m, max_m))

            child_g = Genome(perm=c1_perm, n_matches=c1_m)
            child_sol = repair(inst, decode(inst, child_g), max_ping_gap=max_ping_gap)
            child_fit = _eval(child_sol)

            new_genomes.append(child_g)
            new_sols.append(child_sol)
            new_fitnesses.append(child_fit)

        population_genomes = new_genomes
        population_sols = new_sols
        fitnesses = new_fitnesses

    best_idx = int(np.argmin(fitnesses))
    return population_sols[best_idx]


def sweep_weights(
    inst: Instance,
    weights: Sequence[float] | np.ndarray = np.linspace(0.0, 1.0, 11),
    algorithm: str = "sa_weighted",
    seed: int | None = None,
    **kw: Any,
) -> list[Solution]:
    """Run an optimization algorithm once per weight across a weight grid.

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance.
    weights : Sequence[float] | np.ndarray, optional
        Array of weights w in [0, 1]. Default is np.linspace(0, 1, 11).
    algorithm : str, optional
        The algorithm to sweep: "sa_weighted" or "ga_weighted". Default is "sa_weighted".
    seed : int | None, optional
        Base RNG seed.
    **kw : Any
        Additional keyword arguments forwarded to the solver.

    Returns
    -------
    list[Solution]
        Collected list of solutions across the weight sweep.
    """
    resolved_seed = inst.seed if seed is None else int(seed)
    solutions: list[Solution] = []

    algo_func: Callable[..., Solution]
    if algorithm.lower() in ("sa_weighted", "sa"):
        algo_func = sa_weighted
    elif algorithm.lower() in ("ga_weighted", "ga"):
        algo_func = ga_weighted
    else:
        raise ValueError(f"Unknown weighted algorithm '{algorithm}'. Choose 'sa_weighted' or 'ga_weighted'.")

    for i, w in enumerate(weights):
        step_seed = resolved_seed + i * 1000
        sol = algo_func(inst, weight=float(w), seed=step_seed, **kw)
        solutions.append(sol)

    return solutions


@register("sa_weighted")
class SAWeightedMatcher:
    """Simulated Annealing Matcher for weighted multi-objective matchmaking."""

    name: str = "sa_weighted"

    def __init__(self, **default_kw: Any) -> None:
        self.default_kw = default_kw

    def solve(self, inst: Instance, **kw: Any) -> list[Solution]:
        """Solve instance.

        If 'weights' is specified, executes a weight sweep returning all solutions.
        Otherwise, runs single-weight optimization (default w=0.5) returning a list of length 1.
        """
        combined = {**self.default_kw, **kw}
        weights = combined.pop("weights", None)
        if weights is not None or combined.pop("sweep", False):
            w_grid = np.linspace(0.0, 1.0, 11) if weights is None else weights
            return sweep_weights(inst, weights=w_grid, algorithm="sa_weighted", **combined)

        w = combined.pop("weight", combined.pop("w", 0.5))
        sol = sa_weighted(inst, weight=w, **combined)
        return [sol]

    def sweep_weights(
        self,
        inst: Instance,
        weights: Sequence[float] | np.ndarray = np.linspace(0.0, 1.0, 11),
        **kw: Any,
    ) -> list[Solution]:
        """Run systematic weight sweep and return collected solutions."""
        combined = {**self.default_kw, **kw}
        return sweep_weights(inst, weights=weights, algorithm="sa_weighted", **combined)


@register("ga_weighted")
class GAWeightedMatcher:
    """Single-objective Genetic Algorithm Matcher for weighted matchmaking."""

    name: str = "ga_weighted"

    def __init__(self, **default_kw: Any) -> None:
        self.default_kw = default_kw

    def solve(self, inst: Instance, **kw: Any) -> list[Solution]:
        """Solve instance.

        If 'weights' is specified, executes a weight sweep returning all solutions.
        Otherwise, runs single-weight optimization (default w=0.5) returning a list of length 1.
        """
        combined = {**self.default_kw, **kw}
        weights = combined.pop("weights", None)
        if weights is not None or combined.pop("sweep", False):
            w_grid = np.linspace(0.0, 1.0, 11) if weights is None else weights
            return sweep_weights(inst, weights=w_grid, algorithm="ga_weighted", **combined)

        w = combined.pop("weight", combined.pop("w", 0.5))
        sol = ga_weighted(inst, weight=w, **combined)
        return [sol]

    def sweep_weights(
        self,
        inst: Instance,
        weights: Sequence[float] | np.ndarray = np.linspace(0.0, 1.0, 11),
        **kw: Any,
    ) -> list[Solution]:
        """Run systematic weight sweep and return collected solutions."""
        combined = {**self.default_kw, **kw}
        return sweep_weights(inst, weights=weights, algorithm="ga_weighted", **combined)
