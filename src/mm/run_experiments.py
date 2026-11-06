# OWNER: M4 - do not edit unless you are M4
"""Experiment runner across scenarios, seeds, and matchmaking algorithms (Checkpoint G8).

Runs systematic evaluation batches:
- Builds instances via `simulator.make_instance` for specified scenarios and seeds.
- Obtains solutions from registered Matchers via `registry.get(name).solve(inst)`.
- Evaluates objective vectors (f1_imbalance, f2_wait, f3_mismatch) via `objectives.evaluate`.
- Checks hard constraint satisfaction and records feasibility rate via `validate.is_feasible`.
- Measures wall-clock runtime per algorithm execution.
- Skips unmerged/unregistered algorithms gracefully without failing.
- Outputs tidy tabular results to `results/raw/`, extracted Pareto fronts to `results/fronts/`,
  and initial baseline comparative visualizations to `results/figures/`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from mm.config import Config, load_config
from mm.objectives import evaluate
from mm.pareto import hypervolume, non_dominated, representative_points
from mm.plots import boxplot_wait_vs_imbalance, pareto_plot, tradeoff_curve
from mm.registry import get, list_names
from mm.simulator import make_instance
from mm.types import Instance, Solution
from mm.validate import is_feasible


def _resolve_repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def _parse_list_arg(val: Sequence[str] | str | None) -> list[str]:
    """Parse comma-separated or space-separated list arguments."""
    if val is None:
        return []
    if isinstance(val, str):
        val = [val]
    items: list[str] = []
    for chunk in val:
        for part in str(chunk).split(","):
            cleaned = part.strip()
            if cleaned:
                items.append(cleaned)
    return items


def run_experiments(
    config: Config | None = None,
    scenarios: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    algorithms: Sequence[str] | None = None,
    quick: bool = False,
    results_dir: str | Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the experimental grid over scenarios, seeds, and algorithms.

    Parameters
    ----------
    config : Config | None, optional
        Project configuration. If None, loaded from default.yaml.
    scenarios : Sequence[str] | None, optional
        List of scenarios to run ('light', 'medium', 'heavy').
    seeds : Sequence[int] | None, optional
        List of integer seeds.
    algorithms : Sequence[str] | None, optional
        List of algorithm names to run.
    quick : bool, optional
        Quick smoke-test mode: runs light scenario with 2 seeds.
    results_dir : str | Path | None, optional
        Base directory where raw, fronts, and figures directories reside.
    verbose : bool, optional
        Whether to print progress messages to stdout.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (summary_df, solutions_df) containing tidy experimental results.
    """
    if config is None:
        config = load_config()

    repo_root = _resolve_repo_root()
    base_res = Path(results_dir) if results_dir is not None else (repo_root / "results")
    raw_dir = base_res / "raw"
    fronts_dir = base_res / "fronts"
    figures_dir = base_res / "figures"

    raw_dir.mkdir(parents=True, exist_ok=True)
    fronts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Resolve scenarios
    if scenarios:
        selected_scenarios = list(scenarios)
    elif quick:
        selected_scenarios = ["light"]
    else:
        selected_scenarios = ["light", "medium", "heavy"]

    # Resolve seeds
    if seeds:
        selected_seeds = [int(s) for s in seeds]
    elif quick:
        selected_seeds = config.seeds[:2] if len(config.seeds) >= 2 else [42, 43]
    else:
        selected_seeds = list(config.seeds)

    # Resolve algorithms
    if algorithms:
        target_algorithms = list(algorithms)
    else:
        target_algorithms = list(config.algorithms)

    available_matchers = set(list_names())
    active_algorithms: list[str] = []
    skipped_algorithms: list[str] = []

    for algo in target_algorithms:
        if algo in available_matchers:
            active_algorithms.append(algo)
        else:
            skipped_algorithms.append(algo)

    if verbose:
        print(f"=== Matchmaking Experiment Runner (Checkpoint G8) ===")
        print(f"Scenarios : {selected_scenarios}")
        print(f"Seeds     : {selected_seeds}")
        print(f"Target    : {target_algorithms}")
        print(f"Active    : {active_algorithms}")
        if skipped_algorithms:
            for s in skipped_algorithms:
                print(f"[SKIP] Algorithm '{s}' is not registered/available. Skipping.")
        print("-" * 55)

    summary_rows: list[dict[str, Any]] = []
    solution_rows: list[dict[str, Any]] = []

    # Map to store all objective points for baseline comparison figures
    # Structure: scenario -> algo_name -> list of (f1, f2)
    scenario_algo_points: dict[str, dict[str, list[list[float]]]] = {
        sc: {} for sc in selected_scenarios
    }

    for sc_idx, scenario in enumerate(selected_scenarios, start=1):
        for seed_idx, seed in enumerate(selected_seeds, start=1):
            if verbose:
                print(f"[{sc_idx}/{len(selected_scenarios)}] Scenario: {scenario.upper():<7} | Seed: {seed}")

            # Instantiate problem instance deterministically
            inst = make_instance(
                k=config.k,
                seed=seed,
                scenario=scenario,
                dt=config.dt,
                max_wait=config.max_wait,
                gap0=config.gap0,
                gap_alpha=config.gap_alpha,
            )

            for algo_name in active_algorithms:
                matcher = get(algo_name)

                # Solve and measure wall-clock duration
                t_start = time.perf_counter()
                try:
                    solutions: list[Solution] = matcher.solve(inst)
                except Exception as exc:
                    if verbose:
                        print(f"  [ERROR] {algo_name} failed on scenario={scenario}, seed={seed}: {exc}")
                    continue
                runtime_sec = time.perf_counter() - t_start

                if not isinstance(solutions, list):
                    solutions = [solutions]

                n_solutions = len(solutions)
                if n_solutions == 0:
                    summary_rows.append({
                        "scenario": scenario,
                        "seed": seed,
                        "algorithm": algo_name,
                        "runtime_sec": round(runtime_sec, 5),
                        "n_solutions": 0,
                        "feasibility_rate": 0.0,
                        "mean_f1": np.nan,
                        "min_f1": np.nan,
                        "max_f1": np.nan,
                        "mean_f2": np.nan,
                        "min_f2": np.nan,
                        "max_f2": np.nan,
                        "non_dominated_count": 0,
                        "hypervolume": 0.0,
                    })
                    continue

                obj_list: list[list[float]] = []
                feas_list: list[bool] = []
                match_counts: list[int] = []

                for sol_idx, sol in enumerate(solutions):
                    feasible = is_feasible(inst, sol)
                    objs = evaluate(inst, sol, use_f3=config.use_f3, cfg=config)
                    feas_list.append(feasible)
                    obj_list.append(objs.tolist())
                    n_matches = sol.n_matches()
                    match_counts.append(n_matches)

                    sol_record: dict[str, Any] = {
                        "scenario": scenario,
                        "seed": seed,
                        "algorithm": algo_name,
                        "solution_id": sol_idx,
                        "is_feasible": feasible,
                        "n_matches": n_matches,
                        "f1_imbalance": round(float(objs[0]), 4),
                        "f2_wait": round(float(objs[1]), 4),
                        "runtime_sec": round(runtime_sec, 5),
                    }
                    if config.use_f3 and len(objs) >= 3:
                        sol_record["f3_mismatch"] = round(float(objs[2]), 4)

                    solution_rows.append(sol_record)

                obj_arr = np.array(obj_list, dtype=float)
                feas_rate = float(np.mean(feas_list))

                # Extract non-dominated points
                nd_pts = non_dominated(obj_arr)
                nd_count = len(nd_pts)

                # Compute hypervolume if 2D objectives
                hv_val = 0.0
                if obj_arr.shape[1] == 2 and nd_count > 0:
                    # Adaptive reference point (1.2 * max of observed in front or default safe upper bound)
                    ref_point = np.array([
                        max(float(np.max(nd_pts[:, 0]) * 1.2), 300.0),
                        max(float(np.max(nd_pts[:, 1]) * 1.2), 1000.0),
                    ])
                    try:
                        hv_val = float(hypervolume(nd_pts, ref_point=ref_point))
                    except Exception:
                        hv_val = 0.0

                # Save individual front CSV to results/fronts/
                front_path = fronts_dir / f"{scenario}_{seed}_{algo_name}_front.csv"
                front_cols = ["f1_imbalance", "f2_wait"]
                if config.use_f3 and nd_pts.shape[1] >= 3:
                    front_cols.append("f3_mismatch")
                front_df = pd.DataFrame(nd_pts[:, :len(front_cols)], columns=front_cols)
                front_df.to_csv(front_path, index=False)

                # Store for aggregate scenario figures
                if algo_name not in scenario_algo_points[scenario]:
                    scenario_algo_points[scenario][algo_name] = []
                scenario_algo_points[scenario][algo_name].extend(nd_pts[:, :2].tolist())

                summary_rows.append({
                    "scenario": scenario,
                    "seed": seed,
                    "algorithm": algo_name,
                    "runtime_sec": round(runtime_sec, 5),
                    "n_solutions": n_solutions,
                    "feasibility_rate": round(feas_rate, 4),
                    "mean_f1": round(float(np.mean(obj_arr[:, 0])), 4),
                    "min_f1": round(float(np.min(obj_arr[:, 0])), 4),
                    "max_f1": round(float(np.max(obj_arr[:, 0])), 4),
                    "mean_f2": round(float(np.mean(obj_arr[:, 1])), 4),
                    "min_f2": round(float(np.min(obj_arr[:, 1])), 4),
                    "max_f2": round(float(np.max(obj_arr[:, 1])), 4),
                    "non_dominated_count": nd_count,
                    "hypervolume": round(hv_val, 2),
                })

                if verbose:
                    print(
                        f"    - {algo_name:<16}: {n_solutions:>3} sol | "
                        f"feas: {feas_rate*100:>5.1f}% | "
                        f"f1: {np.mean(obj_arr[:, 0]):>6.2f} | "
                        f"f2: {np.mean(obj_arr[:, 1]):>8.2f} | "
                        f"{runtime_sec:>6.3f}s"
                    )

    summary_df = pd.DataFrame(summary_rows)
    solutions_df = pd.DataFrame(solution_rows)

    # Save tidy CSVs to results/raw/
    summary_path = raw_dir / "experiments_summary.csv"
    solutions_path = raw_dir / "solutions_detail.csv"
    summary_df.to_csv(summary_path, index=False)
    solutions_df.to_csv(solutions_path, index=False)

    if verbose:
        print("-" * 55)
        print(f"Saved summary CSV   -> {summary_path}")
        print(f"Saved solutions CSV -> {solutions_path}")
        print(f"Saved Pareto fronts -> {fronts_dir}")

    # Generate baseline figures in results/figures/
    _generate_baseline_figures(
        summary_df=summary_df,
        solutions_df=solutions_df,
        scenario_points=scenario_algo_points,
        figures_dir=figures_dir,
        verbose=verbose,
    )

    return summary_df, solutions_df


def _generate_baseline_figures(
    summary_df: pd.DataFrame,
    solutions_df: pd.DataFrame,
    scenario_points: dict[str, dict[str, list[list[float]]]],
    figures_dir: Path,
    verbose: bool = True,
) -> None:
    """Generate baseline comparative figures and save to results/figures/."""
    baseline_names = {"sort_skill", "fifo", "greedy_nearest"}

    for scenario, algo_dict in scenario_points.items():
        baseline_fronts: dict[str, np.ndarray] = {}
        for algo, pts in algo_dict.items():
            if algo in baseline_names and len(pts) > 0:
                baseline_fronts[algo] = non_dominated(np.array(pts, dtype=float))

        if not baseline_fronts:
            continue

        # 1. Multi-baseline Pareto front comparison
        pareto_fig_path = figures_dir / f"baselines_pareto_{scenario}.png"
        try:
            fig, ax = pareto_plot(
                fronts=baseline_fronts,
                title=f"Baseline Algorithms Comparison - {scenario.capitalize()} Scenario",
                save_path=pareto_fig_path,
            )
            if verbose:
                print(f"Generated figure    -> {pareto_fig_path}")
        except Exception as exc:
            if verbose:
                print(f"  [WARN] Failed to plot pareto front for {scenario}: {exc}")

        # 2. Distribution boxplot of wait vs imbalance across seeds
        if not solutions_df.empty:
            sc_mask = (solutions_df["scenario"] == scenario) & (
                solutions_df["algorithm"].isin(baseline_names)
            )
            sc_baseline_df = solutions_df[sc_mask]
            if not sc_baseline_df.empty:
                boxplot_fig_path = figures_dir / f"baselines_boxplot_{scenario}.png"
                try:
                    fig, axes = boxplot_wait_vs_imbalance(
                        data=sc_baseline_df,
                        title=f"Baselines Wait vs Imbalance - {scenario.capitalize()} Scenario",
                        imbalance_col="f1_imbalance",
                        wait_col="f2_wait",
                        algorithm_col="algorithm",
                        save_path=boxplot_fig_path,
                    )
                    if verbose:
                        print(f"Generated figure    -> {boxplot_fig_path}")
                except Exception as exc:
                    if verbose:
                        print(f"  [WARN] Failed to plot boxplot for {scenario}: {exc}")

        # 3. Overall trade-off curve across all baseline solutions
        all_baseline_pts: list[list[float]] = []
        for pts in baseline_fronts.values():
            all_baseline_pts.extend(pts.tolist())
        if len(all_baseline_pts) >= 3:
            tradeoff_fig_path = figures_dir / f"baselines_tradeoff_{scenario}.png"
            try:
                fig, ax = tradeoff_curve(
                    F=np.array(all_baseline_pts, dtype=float),
                    name="Combined Baselines",
                    title=f"Baseline Operating Trade-off Curve - {scenario.capitalize()} Scenario",
                    save_path=tradeoff_fig_path,
                )
                if verbose:
                    print(f"Generated figure    -> {tradeoff_fig_path}")
            except Exception as exc:
                if verbose:
                    print(f"  [WARN] Failed to plot tradeoff curve for {scenario}: {exc}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for running matchmaking experiments."""
    parser = argparse.ArgumentParser(
        description="Run matchmaking optimization experiments (Checkpoint G8).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: config/default.yaml).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick smoke test using light scenario and first 2 seeds.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="List of scenarios to run ('light', 'medium', 'heavy').",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="List of integer RNG seeds.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=None,
        help="List of algorithm names to evaluate.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Target results directory for raw, fronts, and figures.",
    )

    args = parser.parse_args(argv)

    config_path = Path(args.config) if args.config else None
    cfg = load_config(config_path)

    scenarios = _parse_list_arg(args.scenarios) if args.scenarios else None
    seeds = [int(s) for s in args.seeds] if args.seeds else None
    algorithms = _parse_list_arg(args.algorithms) if args.algorithms else None

    run_experiments(
        config=cfg,
        scenarios=scenarios,
        seeds=seeds,
        algorithms=algorithms,
        quick=args.quick,
        results_dir=args.results_dir,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
