# OWNER: M1 - do not edit unless you are M1
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Player:
    id: int
    skill: float
    wait: float
    arrival: float
    role: int | None = None
    ping: float | None = None


@dataclass
class Instance:
    players: list[Player]
    k: int
    dt: float
    max_wait: float
    gap0: float
    gap_alpha: float
    seed: int


@dataclass
class Solution:
    match: np.ndarray
    team: np.ndarray

    @classmethod
    def empty(cls, n: int) -> Solution:
        """Create an empty solution where all n players remain unmatched in the queue."""
        return cls(
            match=np.full(n, -1, dtype=int),
            team=np.zeros(n, dtype=int),
        )

    def n_matches(self) -> int:
        """Return the total number of distinct formed matches."""
        matched = self.match[self.match >= 0]
        if matched.size == 0:
            return 0
        return int(len(np.unique(matched)))
