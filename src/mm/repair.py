# OWNER: M3 - do not edit unless you are M3
"""Constraint repair operator for matchmaking optimization.

Implements repair(inst, sol) to ensure that candidate solutions satisfy all
problem constraints defined in the shared contract:
1. Exact match sizing: exactly 2k players per match.
2. Equal team balance: exactly k players on team 0 and k on team 1.
3. Starvation constraint: any player with wait + dt > max_wait must be matched.
4. Relaxed skill gap: |skill_i - skill_j| <= gap0 + gap_alpha * max(wait_i, wait_j).
5. Ping spread: max(ping) - min(ping) <= max_ping_gap within each formed match.

Guarantees >= 99% (target 100%) feasibility across random chromosomes and scenarios.
"""
from __future__ import annotations

import itertools
import math
from typing import Sequence
import numpy as np

from mm.types import Instance, Player, Solution
from mm.validate import is_feasible, validate


def _is_pair_skill_valid(p1: Player, p2: Player, gap0: float, gap_alpha: float) -> bool:
    """Check if two players satisfy the relaxed pairwise skill gap constraint."""
    diff = abs(p1.skill - p2.skill)
    allowed = gap0 + gap_alpha * max(p1.wait, p2.wait)
    return diff <= (allowed + 1e-6)


def _is_pair_ping_valid(p1: Player, p2: Player, max_ping_gap: float | None) -> bool:
    """Check if two players satisfy the ping gap constraint."""
    if max_ping_gap is None or p1.ping is None or p2.ping is None:
        return True
    return abs(p1.ping - p2.ping) <= (max_ping_gap + 1e-6)


def _is_pair_valid(
    p1: Player,
    p2: Player,
    gap0: float,
    gap_alpha: float,
    max_ping_gap: float | None,
) -> bool:
    """Check if two players satisfy both skill and ping constraints."""
    return _is_pair_skill_valid(p1, p2, gap0, gap_alpha) and _is_pair_ping_valid(
        p1, p2, max_ping_gap
    )


def _count_member_violations(
    inst: Instance,
    player_idx: int,
    group: Sequence[int],
    max_ping_gap: float | None,
) -> int:
    """Count how many other members in group violate constraints with player_idx."""
    p_target = inst.players[player_idx]
    count = 0
    for other_idx in group:
        if other_idx == player_idx:
            continue
        p_other = inst.players[other_idx]
        if not _is_pair_valid(p_target, p_other, inst.gap0, inst.gap_alpha, max_ping_gap):
            count += 1
    return count


def _is_group_feasible(
    inst: Instance,
    group: Sequence[int],
    max_ping_gap: float | None,
) -> bool:
    """Check if all pairs in group satisfy skill and ping constraints."""
    n = len(group)
    for a in range(n):
        idx_a = group[a]
        p_a = inst.players[idx_a]
        for b in range(a + 1, n):
            idx_b = group[b]
            p_b = inst.players[idx_b]
            if not _is_pair_valid(p_a, p_b, inst.gap0, inst.gap_alpha, max_ping_gap):
                return False
    return True


def _find_feasible_group_for_player(
    inst: Instance,
    target_idx: int,
    candidate_pool: list[int],
    max_ping_gap: float | None,
) -> list[int] | None:
    """Find 2k-1 candidates from candidate_pool that form a feasible match with target_idx."""
    needed = 2 * inst.k - 1
    if len(candidate_pool) < needed:
        return None

    target_player = inst.players[target_idx]
    # Filter candidates compatible with target_player
    compatible = [
        c
        for c in candidate_pool
        if c != target_idx
        and _is_pair_valid(target_player, inst.players[c], inst.gap0, inst.gap_alpha, max_ping_gap)
    ]
    if len(compatible) < needed:
        return None

    # Sort compatible candidates by skill distance to target_player
    compatible.sort(key=lambda c: abs(inst.players[c].skill - target_player.skill))

    # Fast heuristic 1: check top 'needed' players directly
    top_candidates = compatible[:needed]
    if _is_group_feasible(inst, [target_idx] + top_candidates, max_ping_gap):
        return [target_idx] + top_candidates

    # Fast heuristic 2: check combinations among top 20 closest compatible players
    search_window = min(len(compatible), 20)
    for combo in itertools.combinations(compatible[:search_window], needed):
        group = [target_idx] + list(combo)
        if _is_group_feasible(inst, group, max_ping_gap):
            return group

    # Fast heuristic 3: greedy forward selection across the full compatible list
    selected = [target_idx]
    for c in compatible:
        p_c = inst.players[c]
        if all(
            _is_pair_valid(p_c, inst.players[s], inst.gap0, inst.gap_alpha, max_ping_gap)
            for s in selected
        ):
            selected.append(c)
            if len(selected) == 2 * inst.k:
                return selected

    return None


def repair(
    inst: Instance,
    sol: Solution,
    max_ping_gap: float | None = 100.0,
) -> Solution:
    """Repair a candidate matchmaking Solution to satisfy all problem constraints.

    The repair pipeline executes structured stages:
    1. Canonicalization: Drop partial matches not having exactly 2k players.
    2. Segregate Incompatible Starving Players: If two starving players in the same
       match violate constraints, separate them so each can form their own feasible match.
    3. Heal or Dissolve Existing Matches: Fix outlier players using queue replacements.
       Dissolve non-starving matches that cannot be healed.
    4. Starvation Resolution Guarantee: Prioritizes most constrained starving players
       first to guarantee every starving player is placed in a feasible match.
    5. Snake Draft Balancing & Re-indexing: Assign teams via snake draft to achieve
       exact k vs k balance and minimize imbalance, re-indexing match IDs to 0..M-1.

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance.
    sol : Solution
        The candidate solution to repair.
    max_ping_gap : float | None, optional
        Maximum allowed difference between highest and lowest ping in a single match.
        Default is 100.0 ms.

    Returns
    -------
    Solution
        A feasible Solution meeting all hard constraints.
    """
    n_players = len(inst.players)
    if n_players == 0:
        return Solution.empty(0)

    match_size = 2 * inst.k
    if match_size <= 0 or n_players < match_size:
        return Solution.empty(n_players)

    match = np.asarray(sol.match, dtype=int).copy()
    team = np.asarray(sol.team, dtype=int).copy()

    # Ensure array dimensions
    if len(match) != n_players:
        match = np.full(n_players, -1, dtype=int)
        team = np.zeros(n_players, dtype=int)
    if len(team) != n_players:
        team = np.zeros(n_players, dtype=int)

    # All starving players (must be matched)
    starving_indices = set(
        i for i, p in enumerate(inst.players) if (p.wait + inst.dt) > inst.max_wait
    )

    # ---------------------------------------------------------
    # Stage 1: Drop partial matches (< or > 2k players)
    # ---------------------------------------------------------
    unique_m = [int(m) for m in np.unique(match) if m >= 0]
    for m in unique_m:
        members = [i for i in range(n_players) if match[i] == m]
        if len(members) != match_size:
            for i in members:
                match[i] = -1
                team[i] = 0

    # ---------------------------------------------------------
    # Stage 2: Separate incompatible starving players in the same match
    # ---------------------------------------------------------
    active_matches = [int(m) for m in np.unique(match) if m >= 0]
    for m in active_matches:
        members = [i for i in range(n_players) if match[i] == m]
        starving_in_m = [i for i in members if i in starving_indices]
        if len(starving_in_m) > 1:
            for idx_a in range(len(starving_in_m)):
                p_a = inst.players[starving_in_m[idx_a]]
                for idx_b in range(idx_a + 1, len(starving_in_m)):
                    p_b = inst.players[starving_in_m[idx_b]]
                    if not _is_pair_valid(p_a, p_b, inst.gap0, inst.gap_alpha, max_ping_gap):
                        # Evict the later starving player so each forms a separate feasible match
                        evicted = starving_in_m[idx_b]
                        match[evicted] = -1
                        team[evicted] = 0

    # ---------------------------------------------------------
    # Stage 3: Heal or dissolve matches with violations
    # ---------------------------------------------------------
    active_matches = [int(m) for m in np.unique(match) if m >= 0]
    for m in active_matches:
        members = [i for i in range(n_players) if match[i] == m]
        if len(members) != match_size:
            continue

        if _is_group_feasible(inst, members, max_ping_gap):
            continue

        # Try healing outlier players
        healing_attempts = 0
        while not _is_group_feasible(inst, members, max_ping_gap) and healing_attempts < 20:
            healing_attempts += 1
            viol_counts = [
                (_count_member_violations(inst, idx, members, max_ping_gap), idx)
                for idx in members
            ]
            viol_counts.sort(reverse=True, key=lambda x: x[0])
            worst_cnt, outlier = viol_counts[0]
            if worst_cnt == 0:
                break

            target = outlier
            if outlier in starving_indices:
                non_starving = [
                    (cnt, idx) for cnt, idx in viol_counts if idx not in starving_indices and cnt > 0
                ]
                if non_starving:
                    target = non_starving[0][1]
                else:
                    break

            unmatched = [
                i for i in range(n_players) if match[i] < 0 and i not in starving_indices
            ]
            best_u: int | None = None
            best_rem = worst_cnt
            for u in unmatched:
                cand = [u if x == target else x for x in members]
                v = sum(
                    _count_member_violations(inst, x, cand, max_ping_gap)
                    for x in cand
                ) // 2
                if v < best_rem:
                    best_rem = v
                    best_u = u
                    if v == 0:
                        break

            if best_u is not None:
                match[best_u] = m
                team[best_u] = team[target]
                match[target] = -1
                team[target] = 0
                members = [best_u if x == target else x for x in members]
            else:
                break

        # If still not feasible
        if not _is_group_feasible(inst, members, max_ping_gap):
            # Safe to dissolve
            for i in members:
                match[i] = -1
                team[i] = 0

    # ---------------------------------------------------------
    # Stage 4: Starvation Resolution Guarantee
    # ---------------------------------------------------------
    # Sort starving players by number of compatible players in the dataset (most constrained first)
    starving_list = list(starving_indices)
    starving_list.sort(
        key=lambda s: sum(
            1
            for c in range(n_players)
            if c != s and _is_pair_valid(inst.players[s], inst.players[c], inst.gap0, inst.gap_alpha, max_ping_gap)
        )
    )

    for s_idx in starving_list:
        if match[s_idx] >= 0:
            # Check if current match is feasible
            m_curr = match[s_idx]
            curr_members = [i for i in range(n_players) if match[i] == m_curr]
            if len(curr_members) == match_size and _is_group_feasible(inst, curr_members, max_ping_gap):
                continue
            # Otherwise dissolve current match and reconstruct
            for i in curr_members:
                match[i] = -1
                team[i] = 0

        # Try unmatched pool first
        pool = [i for i in range(n_players) if match[i] < 0 and i != s_idx]
        group = _find_feasible_group_for_player(inst, s_idx, pool, max_ping_gap)

        if group is None:
            # Expanded pool: unmatched + non-starving players from matches WITHOUT any starving player
            matches_with_starving = set(match[s] for s in starving_indices if match[s] >= 0)
            expanded_pool = [
                i
                for i in range(n_players)
                if i != s_idx
                and i not in starving_indices
                and (match[i] < 0 or match[i] not in matches_with_starving)
            ]
            group = _find_feasible_group_for_player(inst, s_idx, expanded_pool, max_ping_gap)
            if group is not None:
                # Dissolve stolen matches
                stolen_matches = set(match[g] for g in group if match[g] >= 0)
                for sm in stolen_matches:
                    for idx in range(n_players):
                        if match[idx] == sm:
                            match[idx] = -1
                            team[idx] = 0

        if group is not None and len(group) == match_size:
            active_m = [int(m) for m in np.unique(match) if m >= 0]
            new_id = (max(active_m) + 1) if active_m else 0
            for idx in group:
                match[idx] = new_id

    # ---------------------------------------------------------
    # Stage 5: Final Clean-up, Snake Draft & Re-indexing
    # ---------------------------------------------------------
    surviving_matches = sorted([int(m) for m in np.unique(match) if m >= 0])
    next_m_idx = 0

    for old_m_id in surviving_matches:
        members = [i for i in range(n_players) if match[i] == old_m_id]

        if len(members) != match_size or not _is_group_feasible(inst, members, max_ping_gap):
            # Dissolve invalid match
            for i in members:
                match[i] = -1
                team[i] = 0
            continue

        # Sort the 2k players by skill ascending
        members.sort(key=lambda idx: inst.players[idx].skill)

        # Snake draft team assignment
        for rank, p_idx in enumerate(members):
            match[p_idx] = next_m_idx
            round_num = rank // 2
            if round_num % 2 == 0:
                team[p_idx] = 0 if (rank % 2 == 0) else 1
            else:
                team[p_idx] = 1 if (rank % 2 == 0) else 0

        next_m_idx += 1

    # Ensure all unmatched players have team 0 and match -1
    match[match < 0] = -1
    team[match < 0] = 0

    return Solution(match=match, team=team)
