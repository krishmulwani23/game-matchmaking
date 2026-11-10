# OWNER: M2 - do not edit unless you are M2
"""Baseline Benchmarks Runner (Checkpoint G9).

Executes all registered baseline matchers across scenarios and seeds,
evaluates all performance metrics, and outputs a tidy DataFrame to
`results/baselines/baselines.csv`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Sequence
import pandas as pd

from mm.config import load_config
from mm.metrics import (
    compute_metrics,
    matched_wait_max,
    matched_wait_mean,
    matched_wait_median,
    matched_wait_p95,
    max_imbalance,
    mean_imbalance,
    queue_length_after,
    match_rate,
    time_execution,
)
from mm.objectives import f1_imbalance, f2_wait, f3_mismatch
import mm.registry as registry
from mm.simulator import make_instance
from mm.types import Instance, Solution
from mm.validate import is_feasible

DEFAULT_BASELINES = ["sort_skill", "fifo", "greedy_nearest"]
DEFAULT_SCENARIOS = ["light", "medium", "heavy"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def _resolve_repo_root() -> Path:
    """Resolve the repository root directory."""
    return Path(__file__).resolve().parents[2]


def run_baseline_benchmarks(
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    baselines: Sequence[str] = DEFAULT_BASELINES,
    output_path: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run registered baselines across scenario and seed pairs and record metrics.

    Parameters
    ----------
    scenarios : Sequence[str], optional
        List of scenarios to evaluate ('light', 'medium', 'heavy').
    seeds : Sequence[int], optional
        List of integer RNG seeds.
    baselines : Sequence[str], optional
        List of registered baseline matcher names.
    output_path : str | Path | None, optional
        Destination CSV file path. If None, writes to `results/baselines/baselines.csv`.
    verbose : bool, optional
        Whether to print benchmark progress to stdout.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame containing all benchmark records.
    """
    repo_root = _resolve_repo_root()
    if output_path is None:
        csv_file = repo_root / "results" / "baselines" / "baselines.csv"
    else:
        csv_file = Path(output_path)

    csv_file.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for scenario in scenarios:
        for seed in seeds:
            inst = make_instance(scenario=scenario, seed=seed)

            for b_name in baselines:
                matcher = registry.get(b_name)

                # Solve and measure execution time
                solutions, runtime_s = time_execution(matcher.solve, inst)
                sol = solutions[0] if len(solutions) > 0 else Solution.empty(len(inst.players))

                f1 = round(float(f1_imbalance(inst, sol)), 4)
                f2 = round(float(f2_wait(inst, sol)), 2)
                f3 = round(float(f3_mismatch(inst, sol)), 4)
                feasible = is_feasible(inst, sol)

                record = {
                    "scenario": scenario,
                    "seed": seed,
                    "algorithm": b_name,
                    "n_players": len(inst.players),
                    "n_matches": sol.n_matches(),
                    "queue_length": queue_length_after(inst, sol),
                    "match_rate": round(match_rate(inst, sol), 4),
                    "f1_imbalance": f1,
                    "f2_wait": f2,
                    "f3_mismatch": f3,
                    "mean_imbalance": round(mean_imbalance(inst, sol), 4),
                    "max_imbalance": round(max_imbalance(inst, sol), 4),
                    "matched_wait_mean": round(matched_wait_mean(inst, sol), 2),
                    "matched_wait_median": round(matched_wait_median(inst, sol), 2),
                    "matched_wait_p95": round(matched_wait_p95(inst, sol), 2),
                    "matched_wait_max": round(matched_wait_max(inst, sol), 2),
                    "is_feasible": feasible,
                    "runtime_s": round(runtime_s, 6),
                }
                records.append(record)
                if verbose:
                    print(
                        f"[{scenario.upper():6s} | seed={seed:2d}] {b_name:15s} "
                        f"-> f1={f1:7.3f}, f2={f2:7.1f}, matches={sol.n_matches():2d}, "
                        f"feasible={feasible}"
                    )

    df = pd.DataFrame(records)
    df.to_csv(csv_file, index=False)
    if verbose:
        print(f"\nSaved {len(df)} benchmark records to {csv_file}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline benchmarks and export CSV.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
        help="List of scenarios (e.g. light medium heavy).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="List of seeds (e.g. 42 43 44 45 46).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with light scenario and 2 seeds (42, 43).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output CSV file path.",
    )

    args = parser.parse_args()

    scenarios = ["light"] if args.quick else args.scenarios
    seeds = [42, 43] if args.quick else args.seeds

    run_baseline_benchmarks(
        scenarios=scenarios,
        seeds=seeds,
        output_path=args.output,
        verbose=True,
    )


if __name__ == "__main__":
    main()
