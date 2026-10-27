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

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        return load_config(path)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def _parse_yaml_fallback(text: str) -> dict[str, Any]:
    """Basic fallback parser for default.yaml when PyYAML is not yet installed."""
    data: dict[str, Any] = {}
    current_key: str | None = None
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
            if not val:
                current_key = key
                if key == "queue_sizes":
                    data[key] = {}
                    in_dict = True
                    in_list = False
                elif key in ("seeds", "algorithms"):
                    data[key] = []
                    in_list = True
                    in_dict = False
                else:
                    data[key] = {}
            else:
                current_key = None
                in_list = False
                in_dict = False
                if val.lower() in ("true", "false"):
                    data[key] = val.lower() == "true"
                elif val.replace(".", "", 1).isdigit():
                    data[key] = float(val) if "." in val else int(val)
                else:
                    data[key] = val.strip("\"'")
        elif indent > 0 and current_key:
            if in_dict and ":" in stripped:
                sub_k, sub_v = stripped.split(":", 1)
                sub_k = sub_k.strip()
                sub_v = sub_v.strip()
                data[current_key][sub_k] = int(sub_v) if sub_v.isdigit() else sub_v
            elif in_list and stripped.startswith("-"):
                item = stripped.lstrip("-").strip().strip("\"'")
                data[current_key].append(int(item) if item.isdigit() else item)

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
    )
