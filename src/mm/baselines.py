# OWNER: M2 - do not edit unless you are M2
"""Baseline Matchers for Queue-Based Multiplayer Matchmaking (Checkpoint G3).

Implements three single-solution heuristic baselines:
1. "sort_skill": Sort players by skill ascending, group sequentially into matches
   of size 2k, split teams via a snake draft, and leave remaining players in the queue.
2. "fifo": Take the longest-waiting players first (descending wait times), group into
   matches of size 2k, and balance the teams (sort by skill, snake draft).
3. "greedy_nearest": Start from the longest-waiting unmatched player, greedily add
   the nearest-skill players to form a match of size 2k, balance teams, and repeat.

All baselines strictly enforce the problem's hard constraints:
- Exact match size: 2k players per formed match, k per team.
- Starvation prevention: any player with wait + dt > max_wait must be matched.
- Relaxed skill gap: |skill_i - skill_j| <= gap0 + gap_alpha * max(wait_i, wait_j).
- Ping spread: max ping difference within each formed match <= max_ping_gap (100 ms).

Constraint satisfaction and repair are guaranteed via `mm.repair.repair`.
Each baseline registers itself with the central registry via `@register`.
"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from mm.interfaces import Matcher
from mm.registry import register
from mm.repair import repair
from mm.types import Instance, Solution


def _snake_draft_teams(match_size: int) -> list[int]:
    """Generate team assignments for players ordered by skill using a snake draft.

    In a match of size 2k sorted by skill (rank 0 to 2k-1):
    - Round 0: team 0, team 1
    - Round 1: team 1, team 0
    - Round 2: team 0, team 1
    ...
    This alternating selection balances both team size (exactly k vs k) and
    minimizes the expected skill imbalance between teams.

    Parameters
    ----------
    match_size : int
        Total number of players in the match (2k).

    Returns
    -------
    list[int]
        Binary team designations (0 or 1) of length match_size.
    """
    teams: list[int] = []
    for r in range(match_size):
        round_idx = r // 2
        teams.append((r % 2) if (round_idx % 2 == 0) else (1 - (r % 2)))
    return teams


def sort_skill(
    inst: Instance,
    max_ping_gap: float | None = 100.0,
    **kw: Any,
) -> Solution:
    """Solve matchmaking instance using the skill-sorted sequential baseline.

    Sorts players by skill ascending, forms sequential matches of size 2k,
    splits teams via a snake draft, leaves the remainder in queue, and repairs
    any hard constraint violations (starvation, relaxed skill gap, ping limit).

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance.
    max_ping_gap : float | None, optional
        Maximum ping gap allowed per match. Default is 100.0 ms.
    **kw : Any
        Ignored extra keyword arguments.

    Returns
    -------
    Solution
        A feasible matchmaking solution.
    """
    n = len(inst.players)
    match_size = 2 * inst.k
    if n < match_size or match_size <= 0:
        return repair(inst, Solution.empty(n), max_ping_gap=max_ping_gap)

    # Sort players by skill ascending; break ties by longer wait time
    sorted_indices = sorted(
        range(n),
        key=lambda i: (inst.players[i].skill, -inst.players[i].wait),
    )

    match = np.full(n, -1, dtype=int)
    team = np.zeros(n, dtype=int)
    team_pattern = _snake_draft_teams(match_size)

    n_matches = n // match_size
    for m_idx in range(n_matches):
        block = sorted_indices[m_idx * match_size : (m_idx + 1) * match_size]
        for rank, player_idx in enumerate(block):
            match[player_idx] = m_idx
            team[player_idx] = team_pattern[rank]

    candidate = Solution(match=match, team=team)
    return repair(inst, candidate, max_ping_gap=max_ping_gap)


def fifo(
    inst: Instance,
    max_ping_gap: float | None = 100.0,
    **kw: Any,
) -> Solution:
    """Solve matchmaking instance using the First-In, First-Out (FIFO) baseline.

    Prioritizes the longest-waiting players first (descending wait times),
    groups them into matches of size 2k, balances the teams by sorting skills
    within each match and applying a snake draft, and repairs any hard constraint
    violations.

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance.
    max_ping_gap : float | None, optional
        Maximum ping gap allowed per match. Default is 100.0 ms.
    **kw : Any
        Ignored extra keyword arguments.

    Returns
    -------
    Solution
        A feasible matchmaking solution.
    """
    n = len(inst.players)
    match_size = 2 * inst.k
    if n < match_size or match_size <= 0:
        return repair(inst, Solution.empty(n), max_ping_gap=max_ping_gap)

    # Sort players by wait time descending; break ties by skill
    sorted_indices = sorted(
        range(n),
        key=lambda i: (-inst.players[i].wait, inst.players[i].skill),
    )

    match = np.full(n, -1, dtype=int)
    team = np.zeros(n, dtype=int)
    team_pattern = _snake_draft_teams(match_size)

    n_matches = n // match_size
    for m_idx in range(n_matches):
        block = sorted_indices[m_idx * match_size : (m_idx + 1) * match_size]
        # Balance teams: sort the 2k players in this match by skill
        block_by_skill = sorted(block, key=lambda i: inst.players[i].skill)
        for rank, player_idx in enumerate(block_by_skill):
            match[player_idx] = m_idx
            team[player_idx] = team_pattern[rank]

    candidate = Solution(match=match, team=team)
    return repair(inst, candidate, max_ping_gap=max_ping_gap)


def greedy_nearest(
    inst: Instance,
    max_ping_gap: float | None = 100.0,
    **kw: Any,
) -> Solution:
    """Solve matchmaking instance using the greedy nearest-skill baseline.

    Iteratively starts from the longest-waiting unmatched player, greedily
    gathers the 2k-1 nearest-skill unmatched players, balances the teams via
    snake draft, and repairs any constraint violations.

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance.
    max_ping_gap : float | None, optional
        Maximum ping gap allowed per match. Default is 100.0 ms.
    **kw : Any
        Ignored extra keyword arguments.

    Returns
    -------
    Solution
        A feasible matchmaking solution.
    """
    n = len(inst.players)
    match_size = 2 * inst.k
    if n < match_size or match_size <= 0:
        return repair(inst, Solution.empty(n), max_ping_gap=max_ping_gap)

    unmatched: set[int] = set(range(n))
    match = np.full(n, -1, dtype=int)
    team = np.zeros(n, dtype=int)
    team_pattern = _snake_draft_teams(match_size)

    m_idx = 0
    while len(unmatched) >= match_size:
        # Start from the longest-waiting unmatched player
        oldest = max(unmatched, key=lambda i: (inst.players[i].wait, -inst.players[i].id))
        unmatched.remove(oldest)

        group: list[int] = [oldest]
        for _ in range(match_size - 1):
            nearest = min(
                unmatched,
                key=lambda i: abs(inst.players[i].skill - inst.players[oldest].skill),
            )
            unmatched.remove(nearest)
            group.append(nearest)

        # Balance teams: sort the formed match by skill
        group_by_skill = sorted(group, key=lambda i: inst.players[i].skill)
        for rank, player_idx in enumerate(group_by_skill):
            match[player_idx] = m_idx
            team[player_idx] = team_pattern[rank]
        m_idx += 1

    candidate = Solution(match=match, team=team)
    return repair(inst, candidate, max_ping_gap=max_ping_gap)


@register("sort_skill")
class SortSkillMatcher:
    """Registered Matcher for the skill-sorted sequential baseline."""

    name: str = "sort_skill"

    def __init__(self, **default_kw: Any) -> None:
        self.default_kw = default_kw

    def solve(self, inst: Instance, **kw: Any) -> list[Solution]:
        """Solve instance and return a single-element list of Solutions."""
        combined = {**self.default_kw, **kw}
        return [sort_skill(inst, **combined)]


@register("fifo")
class FifoMatcher:
    """Registered Matcher for the First-In, First-Out (FIFO) baseline."""

    name: str = "fifo"

    def __init__(self, **default_kw: Any) -> None:
        self.default_kw = default_kw

    def solve(self, inst: Instance, **kw: Any) -> list[Solution]:
        """Solve instance and return a single-element list of Solutions."""
        combined = {**self.default_kw, **kw}
        return [fifo(inst, **combined)]


@register("greedy_nearest")
class GreedyNearestMatcher:
    """Registered Matcher for the greedy nearest-skill baseline."""

    name: str = "greedy_nearest"

    def __init__(self, **default_kw: Any) -> None:
        self.default_kw = default_kw

    def solve(self, inst: Instance, **kw: Any) -> list[Solution]:
        """Solve instance and return a single-element list of Solutions."""
        combined = {**self.default_kw, **kw}
        return [greedy_nearest(inst, **combined)]
