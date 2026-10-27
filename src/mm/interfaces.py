# OWNER: M1 - do not edit unless you are M1
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from mm.types import Instance, Solution


@runtime_checkable
class Matcher(Protocol):
    name: str

    def solve(self, inst: Instance, **kw: Any) -> list[Solution]:
        """Solve a matchmaking instance and return a list of solutions."""
        ...
