# OWNER: M2 - do not edit unless you are M2
"""Statistical Analysis & Hypothesis Testing Module (Checkpoint G13).

Provides non-parametric statistical hypothesis testing and effect size
computations between matchmaking algorithms across seeds:
- Wilcoxon signed-rank test for paired algorithmic comparisons.
- Effect size metrics: Matched-pairs rank-biserial correlation and Cliff's delta.
- Automated parsing and pairing of raw experimental results from `results/raw/`.
- Full pairwise comparison matrices for multi-objective benchmarks.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd
from scipy import stats


def _resolve_repo_root() -> Path:
    """Resolve the repository root directory."""
    return Path(__file__).resolve().parents[2]


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Cliff's delta non-parametric effect size between two distributions.

    Cliff's delta measures the degree of overlap between two distributions:
        delta = (sum_{i,j} sign(x_i - y_j)) / (len(x) * len(y))

    Interpretation:
        |delta| < 0.147: Negligible
        0.147 <= |delta| < 0.33: Small
        0.33 <= |delta| < 0.474: Medium
        |delta| >= 0.474: Large

    Parameters
    ----------
    x : Sequence[float]
        First sample distribution.
    y : Sequence[float]
        Second sample distribution.

    Returns
    -------
    float
        Cliff's delta in [-1.0, 1.0].
    """
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    nx = len(arr_x)
    ny = len(arr_y)
    if nx == 0 or ny == 0:
        return 0.0

    # Vectorized outer comparison
    diffs = np.subtract.outer(arr_x, arr_y)
    greater = np.sum(diffs > 0)
    less = np.sum(diffs < 0)
    return float((greater - less) / (nx * ny))


def rank_biserial_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute matched-pairs rank-biserial correlation effect size.

    For paired samples (x_i, y_i) with non-zero differences d_i = x_i - y_i:
        r = (W+ - W-) / (W+ + W-)
    where W+ and W- are sums of positive and negative signed ranks.

    Returns
    -------
    float
        Rank-biserial correlation in [-1.0, 1.0].
    """
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    if len(arr_x) != len(arr_y) or len(arr_x) == 0:
        return 0.0

    diffs = arr_x - arr_y
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(nonzero))
    pos_ranks = np.sum(ranks[nonzero > 0])
    neg_ranks = np.sum(ranks[nonzero < 0])
    total_ranks = pos_ranks + neg_ranks
    if total_ranks == 0:
        return 0.0
    return float((pos_ranks - neg_ranks) / total_ranks)


def wilcoxon_signed_rank_test(
    x: Sequence[float],
    y: Sequence[float],
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Execute Wilcoxon signed-rank test on paired distributions.

    Parameters
    ----------
    x : Sequence[float]
        First distribution of observations across seeds.
    y : Sequence[float]
        Second distribution of observations across seeds (paired with x).
    alternative : str, optional
        Alternative hypothesis ('two-sided', 'greater', 'less').

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'statistic': Wilcoxon W statistic
        - 'p_value': Two-sided (or directional) asymptotic/exact p-value
        - 'rank_biserial': Rank-biserial correlation effect size
        - 'cliffs_delta': Cliff's delta effect size
        - 'effect_size_r': Standardized effect size r = Z / sqrt(N)
        - 'n_pairs': Total number of pairs
        - 'n_nonzero': Number of pairs with non-zero difference
    """
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    if len(arr_x) != len(arr_y):
        raise ValueError(f"Paired samples must have identical length: {len(arr_x)} vs {len(arr_y)}")

    n_pairs = len(arr_x)
    if n_pairs == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "rank_biserial": 0.0,
            "cliffs_delta": 0.0,
            "effect_size_r": 0.0,
            "n_pairs": 0,
            "n_nonzero": 0,
        }

    diffs = arr_x - arr_y
    nonzero = diffs[diffs != 0]
    n_nonzero = len(nonzero)

    rb = rank_biserial_correlation(arr_x, arr_y)
    cd = cliffs_delta(arr_x, arr_y)

    # If all differences are zero, distributions are identical
    if n_nonzero == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "rank_biserial": 0.0,
            "cliffs_delta": 0.0,
            "effect_size_r": 0.0,
            "n_pairs": n_pairs,
            "n_nonzero": 0,
        }

    # SciPy Wilcoxon signed rank test
    try:
        res = stats.wilcoxon(arr_x, arr_y, alternative=alternative)
        stat = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception:
        # Fallback for very small sample or specific zero patterns
        stat = 0.0
        p_val = 1.0

    # Normal approximation for z-score effect size r = Z / sqrt(N)
    # Expected mean of W: E[W] = n(n+1)/4, Var[W] = n(n+1)(2n+1)/24
    e_w = n_nonzero * (n_nonzero + 1) / 4.0
    var_w = n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24.0
    z_score = (stat - e_w) / math.sqrt(var_w) if var_w > 0 else 0.0
    effect_r = abs(z_score) / math.sqrt(n_nonzero) if n_nonzero > 0 else 0.0

    return {
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "rank_biserial": round(rb, 4),
        "cliffs_delta": round(cd, 4),
        "effect_size_r": round(effect_r, 4),
        "n_pairs": n_pairs,
        "n_nonzero": n_nonzero,
    }


def load_raw_results(
    raw_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read experimental results produced by RAM from results/raw/.

    Strictly READ-ONLY: never modifies or overwrites files in results/raw/.

    Parameters
    ----------
    raw_dir : str | Path | None, optional
        Path to raw results directory. If None, resolves to results/raw/.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (experiments_summary_df, solutions_detail_df)
    """
    repo_root = _resolve_repo_root()
    base_raw = Path(raw_dir) if raw_dir is not None else (repo_root / "results" / "raw")

    summary_file = base_raw / "experiments_summary.csv"
    detail_file = base_raw / "solutions_detail.csv"

    summary_df = pd.read_csv(summary_file) if summary_file.exists() else pd.DataFrame()
    detail_df = pd.read_csv(detail_file) if detail_file.exists() else pd.DataFrame()

    return summary_df, detail_df


def compare_algorithms(
    df: pd.DataFrame,
    algo_a: str,
    algo_b: str,
    metric: str = "hypervolume",
    scenario: str | None = None,
) -> dict[str, Any]:
    """Perform paired comparison between two algorithms across seeds.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns ['scenario', 'seed', 'algorithm', metric].
    algo_a : str
        Name of first algorithm.
    algo_b : str
        Name of second algorithm.
    metric : str, optional
        Metric column to compare. Default is 'hypervolume'.
    scenario : str | None, optional
        Scenario filter. If None, pairs across all scenarios and seeds.

    Returns
    -------
    dict[str, Any]
        Dictionary with comparison statistics, means, p-values, and effect sizes.
    """
    sub = df[df["algorithm"].isin([algo_a, algo_b])].copy()
    if scenario is not None:
        sub = sub[sub["scenario"] == scenario]

    if metric not in sub.columns:
        raise KeyError(f"Metric '{metric}' not found in DataFrame columns: {list(sub.columns)}")

    # Pivot to pair by (scenario, seed)
    piv = sub.pivot_table(
        index=["scenario", "seed"],
        columns="algorithm",
        values=metric,
        aggfunc="mean",
    ).dropna(subset=[algo_a, algo_b])

    vals_a = piv[algo_a].to_numpy(dtype=float)
    vals_b = piv[algo_b].to_numpy(dtype=float)

    test_res = wilcoxon_signed_rank_test(vals_a, vals_b)

    mean_a = float(np.mean(vals_a)) if len(vals_a) > 0 else 0.0
    mean_b = float(np.mean(vals_b)) if len(vals_b) > 0 else 0.0

    return {
        "algo_a": algo_a,
        "algo_b": algo_b,
        "scenario": scenario if scenario is not None else "all",
        "metric": metric,
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "diff_mean": round(mean_a - mean_b, 4),
        **test_res,
        "significant_05": test_res["p_value"] < 0.05,
    }


def pairwise_comparison_matrix(
    df: pd.DataFrame,
    metric: str = "hypervolume",
    scenario: str | None = None,
    algorithms: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate all pairwise Wilcoxon tests between algorithms for a given metric.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or summary DataFrame containing benchmark observations.
    metric : str, optional
        Metric to evaluate (e.g. 'hypervolume', 'mean_f1', 'mean_f2').
    scenario : str | None, optional
        Scenario filter or None for all.
    algorithms : Sequence[str] | None, optional
        List of algorithms to compare. If None, auto-detected from df.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame of pairwise comparison rows.
    """
    if algorithms is None:
        algorithms = sorted(df["algorithm"].unique().tolist())

    records: list[dict[str, Any]] = []
    n = len(algorithms)
    for i in range(n):
        for j in range(i + 1, n):
            algo_a = algorithms[i]
            algo_b = algorithms[j]
            comp = compare_algorithms(df, algo_a, algo_b, metric=metric, scenario=scenario)
            records.append(comp)

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run statistical Wilcoxon tests on raw results.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Path to results/raw directory.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="hypervolume",
        help="Metric to test (hypervolume, mean_f1, mean_f2).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path for stats comparisons.",
    )
    args = parser.parse_args()

    summary_df, _ = load_raw_results(args.raw_dir)
    if summary_df.empty:
        print("No raw experiments summary found. Run experiments first.")
        return

    print(f"Loaded {len(summary_df)} rows from experiments_summary.csv.")
    matrix = pairwise_comparison_matrix(summary_df, metric=args.metric)

    repo_root = _resolve_repo_root()
    out_file = (
        Path(args.output)
        if args.output is not None
        else (repo_root / "results" / "conflict" / "stats_comparisons.csv")
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out_file, index=False)
    print(f"Saved {len(matrix)} pairwise comparisons to {out_file}:\n")
    print(matrix[["algo_a", "algo_b", "metric", "diff_mean", "p_value", "rank_biserial", "significant_05"]].to_string())


if __name__ == "__main__":
    main()
