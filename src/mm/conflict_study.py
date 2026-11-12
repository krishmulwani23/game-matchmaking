# OWNER: M2 - do not edit unless you are M2
"""Conflict Study & Parameter Sensitivity Analysis Module (Checkpoint G13).

Demonstrates the fundamental conflict between match fairness (f1) and queue
latency (f2), proving why heuristic sorting alone is incapable of trading
fairness against waiting and why multi-objective optimization is strictly required.

Sweeps:
- max_wait: Deadline before starvation forces matching.
- gap_alpha: Dynamic relaxation rate of skill gap tolerance over time.
- k: Match team size (2k total players per match).
- queue_size: Queued player density and liquidity.

Outputs:
- Summary and per-parameter sweep tables to `results/conflict/`.
- Visualization plot data and summary figure to `results/conflict/conflict_tradeoffs.png`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mm.metrics import match_rate, matched_wait_mean, max_imbalance, queue_length_after
from mm.objectives import f1_imbalance, f2_wait
import mm.registry as registry
from mm.simulator import make_instance
from mm.types import Instance, Solution
from mm.validate import is_feasible

DEFAULT_BASELINES = ["sort_skill", "fifo", "greedy_nearest"]
DEFAULT_SCENARIOS = ["light", "medium", "heavy"]
DEFAULT_SEEDS = [42, 43, 44]

# Sweep parameter grids
SWEEPS: dict[str, list[Any]] = {
    "max_wait": [60.0, 90.0, 120.0, 180.0, 240.0],
    "gap_alpha": [0.0, 0.5, 1.0, 1.5, 2.0],
    "k": [1, 2, 3, 4],
    "queue_size": [24, 60, 100, 200, 400],
}


def _resolve_repo_root() -> Path:
    """Resolve the repository root directory."""
    return Path(__file__).resolve().parents[2]


def sweep_parameter(
    param_name: str,
    param_values: Sequence[Any],
    scenarios: Sequence[str] = ("light", "medium"),
    seeds: Sequence[int] = (42, 43),
    baselines: Sequence[str] = DEFAULT_BASELINES,
) -> pd.DataFrame:
    """Execute parametric sweep for a single specified parameter.

    Parameters
    ----------
    param_name : str
        Parameter to sweep ('max_wait', 'gap_alpha', 'k', 'queue_size').
    param_values : Sequence[Any]
        Values to evaluate.
    scenarios : Sequence[str], optional
        Scenarios to sweep across.
    seeds : Sequence[int], optional
        Seeds to average across.
    baselines : Sequence[str], optional
        Baseline matchers to evaluate.

    Returns
    -------
    pd.DataFrame
        DataFrame of sweep observations.
    """
    records: list[dict[str, Any]] = []

    for val in param_values:
        for sc in scenarios:
            for seed in seeds:
                kw: dict[str, Any] = {}
                n_players = None
                match_k = None

                if param_name == "max_wait":
                    kw["max_wait"] = float(val)
                elif param_name == "gap_alpha":
                    kw["gap_alpha"] = float(val)
                elif param_name == "k":
                    match_k = int(val)
                elif param_name == "queue_size":
                    n_players = int(val)

                inst = make_instance(
                    n_players=n_players,
                    k=match_k,
                    seed=seed,
                    scenario=sc,
                    **kw,
                )

                for b_name in baselines:
                    matcher = registry.get(b_name)
                    solutions = matcher.solve(inst)
                    sol = solutions[0] if len(solutions) > 0 else Solution.empty(len(inst.players))

                    f1 = float(f1_imbalance(inst, sol))
                    f2 = float(f2_wait(inst, sol))
                    m_rate = float(match_rate(inst, sol))
                    m_wait = float(matched_wait_mean(inst, sol))
                    feas = is_feasible(inst, sol)

                    records.append(
                        {
                            "sweep_param": param_name,
                            "param_value": val,
                            "scenario": sc,
                            "seed": seed,
                            "algorithm": b_name,
                            "n_players": len(inst.players),
                            "k": inst.k,
                            "max_wait": inst.max_wait,
                            "gap_alpha": inst.gap_alpha,
                            "n_matches": sol.n_matches(),
                            "match_rate": round(m_rate, 4),
                            "f1_imbalance": round(f1, 4),
                            "f2_wait": round(f2, 2),
                            "matched_wait_mean": round(m_wait, 2),
                            "max_imbalance": round(float(max_imbalance(inst, sol)), 4),
                            "is_feasible": feas,
                        }
                    )

    return pd.DataFrame(records)


def run_conflict_study(
    output_dir: str | Path | None = None,
    scenarios: Sequence[str] = ("light", "medium", "heavy"),
    seeds: Sequence[int] = (42, 43, 44),
    baselines: Sequence[str] = DEFAULT_BASELINES,
    sweeps_dict: dict[str, list[Any]] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run full conflict study sweeping max_wait, gap_alpha, k, and queue_size.

    Parameters
    ----------
    output_dir : str | Path | None, optional
        Target directory for CSV and figure outputs. Default is results/conflict/.
    scenarios : Sequence[str], optional
        Scenarios to evaluate.
    seeds : Sequence[int], optional
        Integer seeds to test.
    baselines : Sequence[str], optional
        List of baseline algorithm names.
    sweeps_dict : dict[str, list[Any]] | None, optional
        Custom grid dictionary.
    verbose : bool, optional
        Whether to print status to stdout.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, pd.DataFrame]]
        (master_table, dict_of_individual_sweep_dfs)
    """
    repo_root = _resolve_repo_root()
    base_out = Path(output_dir) if output_dir is not None else (repo_root / "results" / "conflict")
    base_out.mkdir(parents=True, exist_ok=True)

    grid = sweeps_dict if sweeps_dict is not None else SWEEPS

    all_dfs: list[pd.DataFrame] = []
    sweep_dfs: dict[str, pd.DataFrame] = {}

    for param, vals in grid.items():
        if verbose:
            print(f"Sweeping parameter: {param} across values {vals}...")
        df_p = sweep_parameter(
            param_name=param,
            param_values=vals,
            scenarios=scenarios,
            seeds=seeds,
            baselines=baselines,
        )
        sweep_dfs[param] = df_p
        all_dfs.append(df_p)

        # Save individual parameter sweep CSV
        csv_p = base_out / f"sweep_{param}.csv"
        df_p.to_csv(csv_p, index=False)

    master_df = pd.concat(all_dfs, ignore_index=True)
    master_csv = base_out / "conflict_study_table.csv"
    master_df.to_csv(master_csv, index=False)

    if verbose:
        print(f"Saved master conflict table ({len(master_df)} rows) to {master_csv}")

    # Generate summary plots
    try:
        plot_conflict_sweeps(sweep_dfs, output_path=base_out / "conflict_tradeoffs.png")
        if verbose:
            print(f"Generated conflict trade-off visualization at {base_out / 'conflict_tradeoffs.png'}")
    except Exception as e:
        if verbose:
            print(f"Warning: plot generation skipped: {e}")

    return master_df, sweep_dfs


def plot_conflict_sweeps(
    sweep_dfs: dict[str, pd.DataFrame],
    output_path: str | Path,
) -> None:
    """Generate multi-panel figure showing how baseline heuristics fail to trade off objectives."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {"sort_skill": "#1f77b4", "fifo": "#ff7f0e", "greedy_nearest": "#2ca02c"}

    # 1. Panel A: f1_imbalance vs max_wait (Tightening wait deadlines forces skill sacrifices)
    ax1 = axes[0, 0]
    if "max_wait" in sweep_dfs:
        df_mw = sweep_dfs["max_wait"]
        for algo, group in df_mw.groupby("algorithm"):
            agg = group.groupby("param_value")["f1_imbalance"].mean()
            ax1.plot(agg.index, agg.values, marker="o", label=algo, color=colors.get(algo, "gray"), linewidth=2)
        ax1.set_title("A: Skill Imbalance vs. Max Wait Deadline", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Max Wait W_max (seconds)")
        ax1.set_ylabel("Mean Skill Imbalance f1 (Elo)")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

    # 2. Panel B: f2_wait vs gap_alpha (Relaxation alpha impact on backlog)
    ax2 = axes[0, 1]
    if "gap_alpha" in sweep_dfs:
        df_ga = sweep_dfs["gap_alpha"]
        for algo, group in df_ga.groupby("algorithm"):
            agg = group.groupby("param_value")["f2_wait"].mean()
            ax2.plot(agg.index, agg.values, marker="s", label=algo, color=colors.get(algo, "gray"), linewidth=2)
        ax2.set_title("B: Queue Wait vs. Skill Gap Relaxation Rate alpha", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Skill Gap Relaxation Rate alpha")
        ax2.set_ylabel("Total Queue Wait f2 (seconds)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

    # 3. Panel C: match_rate vs k (Team capacity scaling)
    ax3 = axes[1, 0]
    if "k" in sweep_dfs:
        df_k = sweep_dfs["k"]
        for algo, group in df_k.groupby("algorithm"):
            agg = group.groupby("param_value")["match_rate"].mean()
            ax3.plot(agg.index, agg.values, marker="^", label=algo, color=colors.get(algo, "gray"), linewidth=2)
        ax3.set_title("C: Match Rate vs. Team Size k (Match Size 2k)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Players Per Team k")
        ax3.set_ylabel("Proportion Matched (Match Rate)")
        ax3.grid(True, linestyle="--", alpha=0.6)
        ax3.legend()

    # 4. Panel D: Direct Trade-off f1 vs f2 (Rigid single points vs Trade-off necessity)
    ax4 = axes[1, 1]
    if "queue_size" in sweep_dfs:
        df_qs = sweep_dfs["queue_size"]
        for algo, group in df_qs.groupby("algorithm"):
            ax4.scatter(
                group["f1_imbalance"],
                group["f2_wait"],
                label=algo,
                color=colors.get(algo, "gray"),
                s=70,
                alpha=0.8,
            )
        ax4.set_title("D: Objective Conflict: Imbalance vs. Latency", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Skill Imbalance f1 (minimized)")
        ax4.set_ylabel("Total Queue Wait f2 (minimized)")
        ax4.grid(True, linestyle="--", alpha=0.6)
        ax4.legend()

    fig.suptitle(
        "Empirical Conflict Study: Why Optimization is Required\n"
        "(Sorting & FIFO are trapped at fixed extremes and cannot navigate the Pareto front)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute conflict study and sensitivity sweeps.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick execution with reduced grids for testing.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target output directory for results/conflict/.",
    )
    args = parser.parse_args()

    scenarios = ["light", "medium"] if args.quick else ("light", "medium", "heavy")
    seeds = [42] if args.quick else (42, 43, 44)

    grid = None
    if args.quick:
        grid = {
            "max_wait": [60.0, 120.0],
            "gap_alpha": [0.5, 1.0],
            "k": [1, 2],
            "queue_size": [24, 100],
        }

    run_conflict_study(
        output_dir=args.output_dir,
        scenarios=scenarios,
        seeds=seeds,
        sweeps_dict=grid,
        verbose=True,
    )


if __name__ == "__main__":
    main()
