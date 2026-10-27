# OWNER: M1 - do not edit unless you are M1
"""Constraint validation for matchmaking instances and solutions.

Checks feasibility of formed matches against hard constraints:
1. Match size (exactly 2k players per match, k per team).
2. Starvation prevention (if wait + dt > max_wait, player must be matched).
3. Relaxed pairwise skill gap (|skill_i - skill_j| <= gap0 + gap_alpha * max(wait_i, wait_j)).
4. Ping gap limit within each match (if ping is modelled).
"""
from __future__ import annotations

import numpy as np

from mm.types import Instance, Solution


def validate(
    inst: Instance,
    sol: Solution,
    max_ping_gap: float | None = 100.0,
) -> list[str]:
    """Validate a Solution against all constraints of the Instance.

    Parameters
    ----------
    inst : Instance
        The matchmaking problem instance containing players and parameters.
    sol : Solution
        The proposed solution containing match assignments and team designations.
    max_ping_gap : float | None
        Maximum allowed difference between highest and lowest ping in a single match.
        If None, ping gap constraint is skipped. Default is 100.0 ms.

    Returns
    -------
    list[str]
        List of human-readable violation descriptions. If empty, the solution is feasible.
    """
    violations: list[str] = []
    n_players = len(inst.players)

    # 1. Structural array checks
    if len(sol.match) != n_players:
        violations.append(
            f"Solution match array length {len(sol.match)} does not match player count {n_players}"
        )
        return violations

    if len(sol.team) != n_players:
        violations.append(
            f"Solution team array length {len(sol.team)} does not match player count {n_players}"
        )
        return violations

    # 2. Team validity for matched players
    for i in range(n_players):
        if sol.match[i] >= 0 and sol.team[i] not in (0, 1):
            violations.append(
                f"Player {inst.players[i].id} in match {sol.match[i]} has invalid team {sol.team[i]} (must be 0 or 1)"
            )

    # 3. Starvation constraint: if wait + dt > max_wait, player must be matched
    for i, p in enumerate(inst.players):
        if (p.wait + inst.dt) > inst.max_wait:
            if sol.match[i] < 0:
                violations.append(
                    f"Player {p.id} has wait + dt = {p.wait + inst.dt:.1f} > max_wait {inst.max_wait:.1f} "
                    f"and must be matched, but remains unmatched in the queue"
                )

    # 4. Per-match constraints
    matched_ids = np.unique(sol.match[sol.match >= 0])

    for m in matched_ids:
        # Find player indices in this match
        player_indices = [i for i, match_id in enumerate(sol.match) if match_id == m]
        m_count = len(player_indices)
        expected_size = 2 * inst.k

        # 4a. Match size: must be exactly 2k players
        if m_count != expected_size:
            violations.append(
                f"Match {m} has {m_count} players; expected exactly {expected_size} (2k with k={inst.k})"
            )

        # 4b. Team split: exactly k players in team 0 and k in team 1
        team0_count = sum(1 for i in player_indices if sol.team[i] == 0)
        team1_count = sum(1 for i in player_indices if sol.team[i] == 1)
        if team0_count != inst.k or team1_count != inst.k:
            violations.append(
                f"Match {m} has {team0_count} on team 0 and {team1_count} on team 1; "
                f"expected {inst.k} on each team"
            )

        # 4c. Pairwise skill gap: |skill_i - skill_j| <= gap0 + gap_alpha * max(wait_i, wait_j)
        for idx_a in range(len(player_indices)):
            for idx_b in range(idx_a + 1, len(player_indices)):
                i = player_indices[idx_a]
                j = player_indices[idx_b]
                p_i = inst.players[i]
                p_j = inst.players[j]

                actual_gap = abs(p_i.skill - p_j.skill)
                allowed_gap = inst.gap0 + inst.gap_alpha * max(p_i.wait, p_j.wait)

                # Tolerance for float comparisons
                if actual_gap > allowed_gap + 1e-6:
                    violations.append(
                        f"Match {m}: skill gap between player {p_i.id} ({p_i.skill:.1f}) and "
                        f"player {p_j.id} ({p_j.skill:.1f}) is {actual_gap:.1f} > allowed {allowed_gap:.1f}"
                    )

        # 4d. Ping gap: max(ping) - min(ping) <= max_ping_gap (if ping is modelled)
        if max_ping_gap is not None and len(player_indices) > 0:
            pings = [inst.players[i].ping for i in player_indices if inst.players[i].ping is not None]
            # Ping is considered modelled if all players in the match have ping values
            if len(pings) == len(player_indices):
                ping_gap = max(pings) - min(pings)
                if ping_gap > max_ping_gap + 1e-6:
                    violations.append(
                        f"Match {m}: ping gap {ping_gap:.1f} ms exceeds configured limit {max_ping_gap:.1f} ms"
                    )

    return violations


def is_feasible(
    inst: Instance,
    sol: Solution,
    max_ping_gap: float | None = 100.0,
) -> bool:
    """Check whether a Solution satisfies all constraints of the Instance.

    Returns True if no constraint violations are found, False otherwise.
    """
    return len(validate(inst, sol, max_ping_gap=max_ping_gap)) == 0
