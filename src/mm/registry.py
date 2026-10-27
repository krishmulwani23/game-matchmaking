# OWNER: M1 - do not edit unless you are M1
from __future__ import annotations

import importlib
from typing import Any, Callable

from mm.interfaces import Matcher

_REGISTRY: dict[str, Any] = {}
_LOADED: bool = False

_LAZY_MODULES: tuple[str, ...] = (
    "mm.baselines",
    "mm.nsga2",
    "mm.weighted",
    "mm.hybrid",
)


def _ensure_loaded() -> None:
    """Lazily import algorithm modules if not already loaded."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for mod_name in _LAZY_MODULES:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            # Module missing or missing optional dependencies: safe to ignore
            pass


def register(name_or_obj: str | Any = None) -> Any:
    """Register a Matcher class or instance in the registry.

    Can be used as:
        @register("sort_skill")
        class SortSkillMatcher: ...

        @register
        class FifoMatcher:
            name = "fifo"
            ...
    """
    def decorator(obj: Any) -> Any:
        registered_name = name if isinstance(name, str) else getattr(obj, "name", getattr(obj, "__name__", str(obj)))
        _REGISTRY[registered_name] = obj
        return obj

    if callable(name_or_obj) and not isinstance(name_or_obj, str):
        name = getattr(name_or_obj, "name", getattr(name_or_obj, "__name__", str(name_or_obj)))
        return decorator(name_or_obj)
    else:
        name = name_or_obj
        return decorator


def get(name: str) -> Matcher:
    """Return an instantiated Matcher by registered name."""
    _ensure_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"Matcher '{name}' is not registered. Available: {list_names()}")
    obj = _REGISTRY[name]
    if isinstance(obj, type):
        return obj()
    return obj


def list_names() -> list[str]:
    """Return a sorted list of registered matcher names."""
    _ensure_loaded()
    return sorted(_REGISTRY.keys())
