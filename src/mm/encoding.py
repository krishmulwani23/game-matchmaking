# OWNER: M3 - do not edit unless you are M3
"""Chromosome encoding, decoding, and genetic operators for matchmaking optimization.

Defines the genetic representation for queue-based matchmaking:
- Genome: Chromosome holding a player permutation and the number of matches formed.
- decode: Maps a Genome to a discrete matchmaking Solution (match and team arrays).
- encode: Reconstructs a Genome from an existing Solution (ensures round-trip property).
- crossover_ox: Order Crossover (OX1) preserving valid permutations.
- mutate_swap, mutate_insert, mutate_inversion: Permutation-preserving mutation operators.
- random_genome, random_feasible: Stochastic sampling and initialization helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Sequence
import numpy as np

from mm.types import Instance, Solution

if TYPE_CHECKING:
    from mm.repair import repair


@dataclass
class Genome:
    """Genetic representation of a matchmaking grouping.

    Parameters
    ----------
    perm : np.ndarray
        1D integer array representing a permutation of player indices 0..N-1.
    n_matches : int
        Number of matches to form from the first 2k * n_matches players in `perm`.
    """

    perm: np.ndarray
    n_matches: int

    def __post_init__(self) -> None:
        self.perm = np.asarray(self.perm, dtype=int)
        self.n_matches = int(self.n_matches)

    def copy(self) -> Genome:
        """Return a deep copy of this Genome."""
        return Genome(perm=self.perm.copy(), n_matches=self.n_matches)

    def __len__(self) -> int:
        return len(self.perm)


def get_match_bounds(inst: Instance) -> tuple[int, int]:
    """Compute the minimum and maximum allowable match counts for an instance.

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance.

    Returns
    -------
    tuple[int, int]
        (min_matches, max_matches).
        min_matches is derived from the number of starving players (wait + dt > max_wait)
        who must be matched to satisfy hard constraints.
        max_matches is the floor division of total players by match size 2k.
    """
    n_players = len(inst.players)
    match_size = 2 * inst.k
    if n_players < match_size or match_size <= 0:
        return (0, 0)

    max_matches = n_players // match_size

    # Count starving players who must be matched
    starving_count = sum(
        1 for p in inst.players if (p.wait + inst.dt) > inst.max_wait
    )
    min_matches = int(math.ceil(starving_count / match_size))
    min_matches = min(min_matches, max_matches)

    return (min_matches, max_matches)


def decode(
    inst: Instance,
    genome: Genome | np.ndarray | Sequence[int],
    n_matches: int | None = None,
) -> Solution:
    """Decode a chromosome genome into a discrete matchmaking Solution.

    The first 2k * n_matches players in the permutation are assigned to matches:
    - In match m:
      - The first k players are assigned to team 0.
      - The next k players are assigned to team 1.
    - All remaining players in the permutation stay unmatched in queue (match = -1, team = 0).

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance.
    genome : Genome | np.ndarray | Sequence[int]
        The chromosome to decode, either a Genome object or a 1D permutation array.
    n_matches : int | None, optional
        Number of matches to form. If None and `genome` is a Genome, uses `genome.n_matches`.
        If None and `genome` is an array, defaults to the maximum possible matches.

    Returns
    -------
    Solution
        A discrete Solution object with match and team arrays.
    """
    n_players = len(inst.players)
    if n_players == 0:
        return Solution.empty(0)

    if isinstance(genome, Genome):
        perm = genome.perm
        target_matches = genome.n_matches if n_matches is None else n_matches
    else:
        perm = np.asarray(genome, dtype=int)
        target_matches = n_players // (2 * inst.k) if n_matches is None else n_matches

    match_size = 2 * inst.k
    max_possible = n_players // match_size if match_size > 0 else 0
    m_count = max(0, min(int(target_matches), max_possible))

    match = np.full(n_players, -1, dtype=int)
    team = np.zeros(n_players, dtype=int)

    for m in range(m_count):
        start = m * match_size
        team0_indices = perm[start : start + inst.k]
        team1_indices = perm[start + inst.k : start + match_size]

        match[team0_indices] = m
        team[team0_indices] = 0

        match[team1_indices] = m
        team[team1_indices] = 1

    return Solution(match=match, team=team)


def encode(sol: Solution, inst: Instance | None = None) -> Genome:
    """Encode an existing Solution into a Genome chromosome.

    Preserves the round-trip invariant: decode(inst, encode(sol)) reproduces the
    exact match and team groupings of sol.

    Parameters
    ----------
    sol : Solution
        The matchmaking solution to encode.
    inst : Instance | None, optional
        Optional instance reference (used for player length check if provided).

    Returns
    -------
    Genome
        A Genome containing the packed permutation and number of formed matches.
    """
    n_players = len(sol.match)
    if n_players == 0:
        return Genome(perm=np.array([], dtype=int), n_matches=0)

    matched_ids = sorted([int(m) for m in np.unique(sol.match) if m >= 0])
    perm_list: list[int] = []

    # Pack matched players match by match, team 0 followed by team 1
    for m in matched_ids:
        t0_players = [i for i in range(n_players) if sol.match[i] == m and sol.team[i] == 0]
        t1_players = [i for i in range(n_players) if sol.match[i] == m and sol.team[i] == 1]
        perm_list.extend(t0_players)
        perm_list.extend(t1_players)

    # Append all remaining unmatched players
    unmatched_players = [i for i in range(n_players) if sol.match[i] < 0]
    perm_list.extend(unmatched_players)

    return Genome(perm=np.array(perm_list, dtype=int), n_matches=len(matched_ids))


def crossover_ox(
    parent1: Genome | np.ndarray,
    parent2: Genome | np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[Genome, Genome] | tuple[np.ndarray, np.ndarray]:
    """Perform Order Crossover (OX1) between two permutation chromosomes.

    Order Crossover preserves the relative order and adjacency of elements while
    guaranteeing that offspring are valid permutations with zero duplicates or missing items.

    Parameters
    ----------
    parent1 : Genome | np.ndarray
        First parent chromosome.
    parent2 : Genome | np.ndarray
        Second parent chromosome.
    rng : np.random.Generator | None, optional
        Random number generator.

    Returns
    -------
    tuple[Genome, Genome] or tuple[np.ndarray, np.ndarray]
        Two offspring preserving valid permutations.
    """
    if rng is None:
        rng = np.random.default_rng()

    is_genome_type = isinstance(parent1, Genome)
    p1 = parent1.perm if isinstance(parent1, Genome) else np.asarray(parent1, dtype=int)
    p2 = parent2.perm if isinstance(parent2, Genome) else np.asarray(parent2, dtype=int)

    n = len(p1)
    if n <= 1:
        if is_genome_type:
            g1 = parent1.copy() if isinstance(parent1, Genome) else Genome(p1.copy(), 0)
            g2 = parent2.copy() if isinstance(parent2, Genome) else Genome(p2.copy(), 0)
            return g1, g2
        return p1.copy(), p2.copy()

    # Pick two cut points
    pts = sorted(rng.choice(n + 1, size=2, replace=False))
    c1, c2 = pts[0], pts[1]

    def _ox_child(donor: np.ndarray, recipient: np.ndarray) -> np.ndarray:
        child = np.full(n, -1, dtype=int)
        child[c1:c2] = donor[c1:c2]
        in_child = set(donor[c1:c2])

        # Fill remaining positions cyclically starting from c2
        fill_idx = c2 % n
        for item in np.concatenate([recipient[c2:], recipient[:c2]]):
            if item not in in_child:
                child[fill_idx] = item
                fill_idx = (fill_idx + 1) % n
        return child

    c1_perm = _ox_child(p1, p2)
    c2_perm = _ox_child(p2, p1)

    if is_genome_type:
        m1 = parent1.n_matches if isinstance(parent1, Genome) else 0
        m2 = parent2.n_matches if isinstance(parent2, Genome) else 0
        # Offspring inherit match counts from parents (or can swap)
        child_g1 = Genome(perm=c1_perm, n_matches=m1)
        child_g2 = Genome(perm=c2_perm, n_matches=m2)
        return child_g1, child_g2

    return c1_perm, c2_perm


def mutate_swap(
    genome: Genome | np.ndarray,
    rng: np.random.Generator | None = None,
) -> Genome | np.ndarray:
    """Mutate a chromosome by swapping two randomly selected player positions.

    Parameters
    ----------
    genome : Genome | np.ndarray
        Chromosome to mutate.
    rng : np.random.Generator | None, optional
        Random number generator.

    Returns
    -------
    Genome | np.ndarray
        Mutated chromosome preserving valid permutation.
    """
    if rng is None:
        rng = np.random.default_rng()

    is_genome_type = isinstance(genome, Genome)
    perm = genome.perm.copy() if is_genome_type else np.asarray(genome, dtype=int).copy()

    n = len(perm)
    if n >= 2:
        i, j = rng.choice(n, size=2, replace=False)
        perm[i], perm[j] = perm[j], perm[i]

    if is_genome_type:
        return Genome(perm=perm, n_matches=genome.n_matches)
    return perm


def mutate_insert(
    genome: Genome | np.ndarray,
    rng: np.random.Generator | None = None,
) -> Genome | np.ndarray:
    """Mutate a chromosome by removing a player from position i and inserting at position j.

    Parameters
    ----------
    genome : Genome | np.ndarray
        Chromosome to mutate.
    rng : np.random.Generator | None, optional
        Random number generator.

    Returns
    -------
    Genome | np.ndarray
        Mutated chromosome preserving valid permutation.
    """
    if rng is None:
        rng = np.random.default_rng()

    is_genome_type = isinstance(genome, Genome)
    perm = genome.perm.copy() if is_genome_type else np.asarray(genome, dtype=int).copy()

    n = len(perm)
    if n >= 2:
        i, j = rng.choice(n, size=2, replace=False)
        val = perm[i]
        perm = np.delete(perm, i)
        perm = np.insert(perm, j, val)

    if is_genome_type:
        return Genome(perm=perm, n_matches=genome.n_matches)
    return perm


def mutate_inversion(
    genome: Genome | np.ndarray,
    rng: np.random.Generator | None = None,
) -> Genome | np.ndarray:
    """Mutate a chromosome by reversing a randomly chosen contiguous subsegment.

    Parameters
    ----------
    genome : Genome | np.ndarray
        Chromosome to mutate.
    rng : np.random.Generator | None, optional
        Random number generator.

    Returns
    -------
    Genome | np.ndarray
        Mutated chromosome preserving valid permutation.
    """
    if rng is None:
        rng = np.random.default_rng()

    is_genome_type = isinstance(genome, Genome)
    perm = genome.perm.copy() if is_genome_type else np.asarray(genome, dtype=int).copy()

    n = len(perm)
    if n >= 2:
        pts = sorted(rng.choice(n + 1, size=2, replace=False))
        c1, c2 = pts[0], pts[1]
        perm[c1:c2] = perm[c1:c2][::-1]

    if is_genome_type:
        return Genome(perm=perm, n_matches=genome.n_matches)
    return perm


def random_permutation(
    n: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a random permutation of integers 0..n-1."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.permutation(n)


def random_genome(
    inst: Instance,
    rng: np.random.Generator | None = None,
    n_matches: int | None = None,
) -> Genome:
    """Generate a random Genome chromosome for an instance.

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance.
    rng : np.random.Generator | None, optional
        Random number generator.
    n_matches : int | None, optional
        Explicit number of matches to form. If None, samples uniformly between
        min_matches and max_matches.

    Returns
    -------
    Genome
        A randomly initialized Genome.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_players = len(inst.players)
    perm = rng.permutation(n_players)

    if n_matches is not None:
        m = n_matches
    else:
        min_m, max_m = get_match_bounds(inst)
        m = int(rng.integers(min_m, max_m + 1)) if max_m >= min_m else 0

    return Genome(perm=perm, n_matches=m)


def random_feasible(
    inst: Instance,
    rng: np.random.Generator | None = None,
) -> Solution:
    """Generate a guaranteed-feasible Solution via random initialization and repair.

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance.
    rng : np.random.Generator | None, optional
        Random number generator.

    Returns
    -------
    Solution
        A validated, feasible Solution satisfying all constraints.
    """
    # Lazy import to avoid circular dependency
    from mm.repair import repair

    genome = random_genome(inst, rng=rng)
    sol = decode(inst, genome)
    return repair(inst, sol)
