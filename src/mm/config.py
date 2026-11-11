# OWNER: M1 - do not edit unless you are M1
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


@dataclass
class QueueSizes:
    light: int = 24
    medium: int = 100
    heavy: int = 400


@dataclass
class ScenarioPreset:
    """Configuration preset for rolling queue simulation scenarios."""

    initial_players: int = 24
    arrival_rate: float = 0.6
    n_rounds: int = 30
    burst_prob: float = 0.0
    burst_multiplier: float = 1.0


def _default_scenarios() -> dict[str, ScenarioPreset]:
    return {
        "light": ScenarioPreset(
            initial_players=24,
            arrival_rate=0.6,
            n_rounds=30,
            burst_prob=0.0,
            burst_multiplier=1.0,
        ),
        "medium": ScenarioPreset(
            initial_players=100,
            arrival_rate=2.0,
            n_rounds=30,
            burst_prob=0.0,
            burst_multiplier=1.0,
        ),
        "heavy": ScenarioPreset(
            initial_players=400,
            arrival_rate=6.0,
            n_rounds=30,
            burst_prob=0.0,
            burst_multiplier=1.0,
        ),
        "bursty": ScenarioPreset(
            initial_players=24,
            arrival_rate=0.5,
            n_rounds=30,
            burst_prob=0.25,
            burst_multiplier=5.0,
        ),
    }


@dataclass
class Config:
    k: int = 2
    dt: float = 10.0
    max_wait: float = 120.0
    gap0: float = 100.0
    gap_alpha: float = 1.0
    queue_sizes: QueueSizes = field(default_factory=QueueSizes)
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    algorithms: list[str] = field(
        default_factory=lambda: [
            "sort_skill",
            "fifo",
            "greedy_nearest",
            "nsga2",
            "sa_weighted",
            "ga_weighted",
            "hybrid_greedy_ls",
        ]
    )
    use_f3: bool = False
    scenarios: dict[str, ScenarioPreset] = field(default_factory=_default_scenarios)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        return load_config(path)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def _coerce_scalar(val: str) -> Any:
    """Coerce a simple YAML scalar string into bool, int, float, or str."""
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    try:
        if "." not in val and "e" not in val.lower():
            return int(val)
        return float(val)
    except ValueError:
        return val.strip("\"'")


def _parse_yaml_fallback(text: str) -> dict[str, Any]:
    """Basic fallback parser for default.yaml when PyYAML is not yet installed."""
    data: dict[str, Any] = {}
    current_key: str | None = None
    current_subkey: str | None = None
    in_list = False
    in_dict = False

    for raw_line in text.splitlines():
        line = raw_line.split("#")[0].rstrip()
        if not line:
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()

        if indent == 0 and ":" in stripped:
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip()
            current_subkey = None
            if not val:
                current_key = key
                if key in ("queue_sizes", "scenarios"):
                    data[key] = {}
                    in_dict = True
                    in_list = False
                elif key in ("seeds", "algorithms"):
                    data[key] = []
                    in_list = True
                    in_dict = False
                else:
                    data[key] = {}
                    in_dict = True
                    in_list = False
            else:
                current_key = None
                in_list = False
                in_dict = False
                data[key] = _coerce_scalar(val)
        elif indent > 0 and current_key:
            if in_list and stripped.startswith("-"):
                item = stripped.lstrip("-").strip()
                data[current_key].append(_coerce_scalar(item))
            elif in_dict and ":" in stripped:
                sub_k, sub_v = stripped.split(":", 1)
                sub_k = sub_k.strip()
                sub_v = sub_v.strip()
                if indent <= 2 and not sub_v:
                    current_subkey = sub_k
                    data[current_key][current_subkey] = {}
                elif indent > 2 and current_subkey is not None and isinstance(
                    data[current_key].get(current_subkey), dict
                ):
                    data[current_key][current_subkey][sub_k] = _coerce_scalar(sub_v)
                else:
                    current_subkey = None
                    data[current_key][sub_k] = _coerce_scalar(sub_v)

    return data


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from a YAML file into a Config dataclass."""
    config_path = Path(path) if path is not None else _default_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    if yaml is not None:
        data = yaml.safe_load(content) or {}
    else:
        data = _parse_yaml_fallback(content)

    q_data = data.get("queue_sizes", {})
    queue_sizes = QueueSizes(
        light=int(q_data.get("light", 24)),
        medium=int(q_data.get("medium", 100)),
        heavy=int(q_data.get("heavy", 400)),
    )

    defaults = _default_scenarios()
    raw_scenarios = data.get("scenarios", {})
    scenarios: dict[str, ScenarioPreset] = {}
    if isinstance(raw_scenarios, dict) and raw_scenarios:
        for sc_name, sc_vals in raw_scenarios.items():
            if not isinstance(sc_vals, dict):
                continue
            fallback = defaults.get(sc_name, ScenarioPreset())
            scenarios[str(sc_name)] = ScenarioPreset(
                initial_players=int(sc_vals.get("initial_players", fallback.initial_players)),
                arrival_rate=float(sc_vals.get("arrival_rate", fallback.arrival_rate)),
                n_rounds=int(sc_vals.get("n_rounds", fallback.n_rounds)),
                burst_prob=float(sc_vals.get("burst_prob", fallback.burst_prob)),
                burst_multiplier=float(
                    sc_vals.get("burst_multiplier", fallback.burst_multiplier)
                ),
            )
    else:
        scenarios = defaults

    return Config(
        k=int(data.get("k", 2)),
        dt=float(data.get("dt", 10.0)),
        max_wait=float(data.get("max_wait", 120.0)),
        gap0=float(data.get("gap0", 100.0)),
        gap_alpha=float(data.get("gap_alpha", 1.0)),
        queue_sizes=queue_sizes,
        seeds=[int(s) for s in data.get("seeds", [42, 43, 44, 45, 46])],
        algorithms=list(data.get("algorithms", [])),
        use_f3=bool(data.get("use_f3", False)),
        scenarios=scenarios,
    )
