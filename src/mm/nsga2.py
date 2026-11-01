# OWNER: M3 - do not edit unless you are M3
"""NSGA-II multi-objective genetic algorithm for queue-based matchmaking.

Implements the Non-dominated Sorting Genetic Algorithm II (Deb et al., 2002)
tailored for the discrete matchmaking grouping problem:
- Multi-objective minimization of f1 (skill imbalance) and f2 (waiting time).
- Chromosome representation via permutation and match count (mm.encoding).
- Constraint handling via systematic repair (mm.repair).
- Elitist (mu + lambda) selection with fast non-dominated sorting and crowding distance.
- Self-registration in the Matcher registry under the name 'nsga2'.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
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


@dataclass
class Individual:
    """An individual in the NSGA-II population."""

    genome: Genome
    sol: Solution
    fitness: np.ndarray  # shape (m,)
    rank: int = 0
    crowding_dist: float = 0.0


def fast_non_dominated_sort(fitness: np.ndarray) -> list[list[int]]:
    """Compute Pareto fronts using fast non-dominated sorting (Deb et al., 2002).

    Parameters
    ----------
    fitness : np.ndarray
        Array of shape (n, m) with m minimization objectives.

    Returns
    -------
    list[list[int]]
        List of fronts, where front[0] is the non-dominated Pareto front.
    """
    n = len(fitness)
    if n == 0:
        return []

    # pairwise comparison: p dominates q if p <= q (all) and p < q (any)
    le = fitness[:, None, :] <= fitness[None, :, :]
    lt = fitness[:, None, :] < fitness[None, :, :]
    dominates = np.all(le, axis=2) & np.any(lt, axis=2)

    # domination counts and dominated sets
    domination_counts = np.sum(dominates.T, axis=1)  # number of individuals dominating i
    dominated_sets: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        dominated_sets[i] = list(np.where(dominates[i])[0])

    fronts: list[list[int]] = []
    current_front = list(np.where(domination_counts == 0)[0])

    while current_front:
        fronts.append(current_front)
        next_front: list[int] = []
        for p in current_front:
            for q in dominated_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        current_front = next_front

    return fronts


def compute_crowding_distances(
    fitness: np.ndarray,
    front: Sequence[int],
) -> np.ndarray:
    """Calculate crowding distance for individuals in a single Pareto front.

    Parameters
    ----------
    fitness : np.ndarray
        Array of shape (n, m) of objective values.
    front : Sequence[int]
        Indices of individuals in the front.

    Returns
    -------
    np.ndarray
        Array of shape (len(front),) containing crowding distances.
    """
    k = len(front)
    if k == 0:
        return np.array([], dtype=float)
    if k <= 2:
        return np.full(k, np.inf, dtype=float)

    front_idx = np.asarray(front, dtype=int)
    m = fitness.shape[1]
    distances = np.zeros(k, dtype=float)

    for obj in range(m):
        obj_vals = fitness[front_idx, obj]
        sorted_order = np.argsort(obj_vals)

        # Boundary points receive infinite distance
        distances[sorted_order[0]] = np.inf
        distances[sorted_order[-1]] = np.inf

        val_min = obj_vals[sorted_order[0]]
        val_max = obj_vals[sorted_order[-1]]
        spread = val_max - val_min

        if spread > 1e-9:
            for i in range(1, k - 1):
                prev_val = obj_vals[sorted_order[i - 1]]
                next_val = obj_vals[sorted_order[i + 1]]
                distances[sorted_order[i]] += (next_val - prev_val) / spread

    return distances


def crowded_tournament(
    ind1: Individual,
    ind2: Individual,
    rng: np.random.Generator,
) -> Individual:
    """Perform binary tournament selection based on rank and crowding distance."""
    if ind1.rank < ind2.rank:
        return ind1
    if ind2.rank < ind1.rank:
        return ind2
    if ind1.crowding_dist > ind2.crowding_dist:
        return ind1
    if ind2.crowding_dist > ind1.crowding_dist:
        return ind2
    return ind1 if rng.random() < 0.5 else ind2


def _evaluate_individual(
    inst: Instance,
    genome: Genome,
    use_f3: bool = False,
    max_ping_gap: float | None = 100.0,
) -> Individual:
    """Decode, repair, and evaluate a genome to form a complete Individual."""
    raw_sol = decode(inst, genome)
    feasible_sol = repair(inst, raw_sol, max_ping_gap=max_ping_gap)
    fit = evaluate(inst, feasible_sol, use_f3=use_f3)
    return Individual(genome=genome, sol=feasible_sol, fitness=fit)


def nsga2(
    inst: Instance,
    pop_size: int | None = None,
    n_gen: int | None = None,
    seed: int | None = None,
    crossover_prob: float = 0.9,
    mutation_prob: float = 0.25,
    use_f3: bool | None = None,
    max_ping_gap: float | None = 100.0,
    **kw: Any,
) -> list[Solution]:
    """Run NSGA-II optimization to find the non-dominated Pareto front.

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance.
    pop_size : int | None, optional
        Population size. If None, loaded from configuration (default: 40).
    n_gen : int | None, optional
        Number of generations. If None, loaded from configuration (default: 30).
    seed : int | None, optional
        RNG seed for deterministic execution. If None, uses inst.seed.
    crossover_prob : float, optional
        Probability of applying Order Crossover (OX1). Default is 0.9.
    mutation_prob : float, optional
        Probability of applying mutation operators. Default is 0.25.
    use_f3 : bool | None, optional
        Whether to optimize 3 objectives (f1, f2, f3). Default is False.
    max_ping_gap : float | None, optional
        Maximum allowable ping gap in matches. Default is 100.0.

    Returns
    -------
    list[Solution]
        The non-dominated, feasible solutions on the Pareto front.
    """
    n_players = len(inst.players)
    if n_players == 0:
        return [Solution.empty(0)]

    # Resolve configuration defaults
    try:
        cfg = load_config()
        cfg_use_f3 = cfg.use_f3
    except Exception:
        cfg_use_f3 = False

    resolved_use_f3 = cfg_use_f3 if use_f3 is None else bool(use_f3)
    resolved_seed = inst.seed if seed is None else int(seed)
    resolved_pop = 40 if pop_size is None else int(pop_size)
    resolved_gen = 30 if n_gen is None else int(n_gen)

    # Ensure even population size for pairwise reproduction
    if resolved_pop % 2 != 0:
        resolved_pop += 1

    rng = np.random.default_rng(resolved_seed)
    min_m, max_m = get_match_bounds(inst)

    # -------------------------------------------------------------
    # 1. Initialize Population
    # -------------------------------------------------------------
    population: list[Individual] = []

    # Diversity injection: seed extreme match count individuals
    if max_m >= min_m:
        # Minimum matches individual (emphasizes fair matches / low f1)
        g_min = random_genome(inst, rng=rng, n_matches=min_m)
        population.append(_evaluate_individual(inst, g_min, resolved_use_f3, max_ping_gap))

        # Maximum matches individual (emphasizes low queue wait / low f2)
        if max_m > min_m:
            g_max = random_genome(inst, rng=rng, n_matches=max_m)
            population.append(_evaluate_individual(inst, g_max, resolved_use_f3, max_ping_gap))

    # Fill remaining population uniformly across allowable match bounds
    while len(population) < resolved_pop:
        m_count = int(rng.integers(min_m, max_m + 1)) if max_m >= min_m else 0
        g = random_genome(inst, rng=rng, n_matches=m_count)
        population.append(_evaluate_individual(inst, g, resolved_use_f3, max_ping_gap))

    # Initial non-dominated ranking and crowding distance
    fitness_matrix = np.array([ind.fitness for ind in population], dtype=float)
    fronts = fast_non_dominated_sort(fitness_matrix)
    for rank_idx, front in enumerate(fronts):
        dists = compute_crowding_distances(fitness_matrix, front)
        for i, idx in enumerate(front):
            population[idx].rank = rank_idx
            population[idx].crowding_dist = dists[i]

    # -------------------------------------------------------------
    # 2. Main Generational Loop
    # -------------------------------------------------------------
    for _ in range(resolved_gen):
        # 2a. Selection and Reproduction (produce offspring Q_t of size N)
        offspring_pool: list[Individual] = []

        while len(offspring_pool) < resolved_pop:
            # Binary tournament selection
            cand1, cand2 = rng.choice(population, size=2, replace=False)
            parent1 = crowded_tournament(cand1, cand2, rng)

            cand3, cand4 = rng.choice(population, size=2, replace=False)
            parent2 = crowded_tournament(cand3, cand4, rng)

            p1_perm = parent1.genome.perm
            p2_perm = parent2.genome.perm
            m1 = parent1.genome.n_matches
            m2 = parent2.genome.n_matches

            # Crossover
            if rng.random() < crossover_prob:
                c1_perm, c2_perm = crossover_ox(p1_perm, p2_perm, rng=rng)
                c1_m = m1 if rng.random() < 0.5 else m2
                c2_m = m2 if rng.random() < 0.5 else m1
            else:
                c1_perm, c2_perm = p1_perm.copy(), p2_perm.copy()
                c1_m, c2_m = m1, m2

            # Mutation
            for perm, m_val in [(c1_perm, c1_m), (c2_perm, c2_m)]:
                if len(offspring_pool) >= resolved_pop:
                    break

                child_perm = perm.copy()
                child_m = m_val

                if rng.random() < mutation_prob:
                    # Choose a mutation operator: swap, insert, or inversion
                    op = rng.choice(["swap", "insert", "inversion"])
                    if op == "swap":
                        child_perm = mutate_swap(child_perm, rng=rng)
                    elif op == "insert":
                        child_perm = mutate_insert(child_perm, rng=rng)
                    else:
                        child_perm = mutate_inversion(child_perm, rng=rng)

                # Match count perturbation
                if rng.random() < 0.3 and max_m > min_m:
                    delta = int(rng.choice([-1, 1]))
                    child_m = int(np.clip(child_m + delta, min_m, max_m))

                child_genome = Genome(perm=child_perm, n_matches=child_m)
                child_ind = _evaluate_individual(
                    inst, child_genome, resolved_use_f3, max_ping_gap
                )
                offspring_pool.append(child_ind)

        # 2b. Environmental Selection (mu + lambda elitism on 2N individuals)
        combined_pool = population + offspring_pool
        combined_fitness = np.array([ind.fitness for ind in combined_pool], dtype=float)
        combined_fronts = fast_non_dominated_sort(combined_fitness)

        next_generation: list[Individual] = []

        for rank_idx, front in enumerate(combined_fronts):
            dists = compute_crowding_distances(combined_fitness, front)
            for i, idx in enumerate(front):
                combined_pool[idx].rank = rank_idx
                combined_pool[idx].crowding_dist = dists[i]

            if len(next_generation) + len(front) <= resolved_pop:
                # Entire front fits
                for idx in front:
                    next_generation.append(combined_pool[idx])
            else:
                # Partial front: sort by crowding distance descending
                needed = resolved_pop - len(next_generation)
                front_order = np.argsort([-combined_pool[idx].crowding_dist for idx in front])
                for k in range(needed):
                    next_generation.append(combined_pool[front[front_order[k]]])
                break

        population = next_generation

    # -------------------------------------------------------------
    # 3. Final Front Extraction & Deduplication
    # -------------------------------------------------------------
    final_fitness = np.array([ind.fitness for ind in population], dtype=float)
    final_fronts = fast_non_dominated_sort(final_fitness)
    pareto_indices = final_fronts[0] if final_fronts else []

    # Filter to unique objective vectors / solutions
    pareto_solutions: list[Solution] = []
    seen_fitness: list[np.ndarray] = []

    for idx in pareto_indices:
        ind = population[idx]
        fit = ind.fitness
        # Deduplication check
        is_duplicate = False
        for seen in seen_fitness:
            if np.allclose(fit, seen, atol=1e-4):
                is_duplicate = True
                break
        if not is_duplicate:
            seen_fitness.append(fit)
            pareto_solutions.append(ind.sol)

    # Sort front solutions by f1 ascending (imbalance) then f2 (wait)
    if pareto_solutions:
        sort_keys = [
            (seen_fitness[i][0], seen_fitness[i][1]) for i in range(len(pareto_solutions))
        ]
        sorted_order = sorted(range(len(pareto_solutions)), key=lambda k: sort_keys[k])
        pareto_solutions = [pareto_solutions[i] for i in sorted_order]

    return pareto_solutions


@register("nsga2")
class NSGA2Matcher:
    """Multi-Objective NSGA-II Matcher.

    Implements the Matcher protocol for queue-based matchmaking optimization.
    """

    name: str = "nsga2"

    def __init__(self, **default_kw: Any) -> None:
        self.default_kw = default_kw

    def solve(self, inst: Instance, **kw: Any) -> list[Solution]:
        """Solve a matchmaking instance and return the non-dominated Pareto front.

        Parameters
        ----------
        inst : Instance
            The matchmaking problem instance.
        **kw : Any
            Optional algorithm overrides (pop_size, n_gen, seed, use_f3, etc.).

        Returns
        -------
        list[Solution]
            List of feasible solutions forming the non-dominated front.
        """
        combined_kw = {**self.default_kw, **kw}
        return nsga2(inst, **combined_kw)
