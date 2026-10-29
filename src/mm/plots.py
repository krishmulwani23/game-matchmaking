# OWNER: M4 - do not edit unless you are M4
"""Visualization utilities for matchmaking Pareto fronts, trade-offs, and distributions.

Provides plotting routines for:
- pareto_plot: multi-algorithm Pareto front comparison in objective space.
- tradeoff_curve: single-algorithm front highlighting the three operating points
  ('very_fair', 'balanced' knee point, 'fast').
- boxplot_wait_vs_imbalance: distributions of wait time and skill imbalance across seeds.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mm.pareto import is_non_dominated, non_dominated, representative_points

# Set non-interactive backend if no display is available
if matplotlib.get_backend().lower() not in ("agg", "svg", "pdf", "ps"):
    try:
        matplotlib.use("Agg")
    except Exception:
        pass


def pareto_plot(
    fronts: dict[str, np.ndarray | list[list[float]]],
    ax: plt.Axes | None = None,
    title: str = "Pareto Front Comparison",
    xlabel: str = "Mean Skill Imbalance (f1)",
    ylabel: str = "Total Wait Time (f2)",
    show_dominated: bool = True,
    step_front: bool = True,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multi-algorithm Pareto fronts in 2D objective space (minimization).

    Parameters
    ----------
    fronts : dict[str, np.ndarray | list[list[float]]]
        Dictionary mapping algorithm name to an (n, 2) array of objective points.
    ax : plt.Axes, optional
        Existing matplotlib axes. If None, a new figure and axis are created.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label (objective 1).
    ylabel : str, optional
        Y-axis label (objective 2).
    show_dominated : bool, optional
        Whether to show dominated points as faint markers. Default True.
    step_front : bool, optional
        Whether to connect non-dominated points with a Pareto front line. Default True.
    save_path : str | Path, optional
        Path to save the generated figure.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and Axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    else:
        fig = ax.get_figure()

    palette = sns.color_palette("tab10", n_colors=max(len(fronts), 1))

    for idx, (name, points) in enumerate(fronts.items()):
        color = palette[idx % len(palette)]
        arr = np.asarray(points, dtype=float)
        if len(arr) == 0:
            continue

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        nd_mask = is_non_dominated(arr)
        nd_points = arr[nd_mask]
        dom_points = arr[~nd_mask]

        # Plot dominated points if requested
        if show_dominated and len(dom_points) > 0:
            ax.scatter(
                dom_points[:, 0],
                dom_points[:, 1],
                color=color,
                alpha=0.25,
                s=25,
                marker="o",
            )

        # Plot non-dominated points
        if len(nd_points) > 0:
            # Sort non-dominated points along objective 0
            sort_idx = np.argsort(nd_points[:, 0])
            sorted_nd = nd_points[sort_idx]

            ax.scatter(
                sorted_nd[:, 0],
                sorted_nd[:, 1],
                color=color,
                alpha=0.9,
                s=55,
                marker="o",
                label=name,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )

            if step_front and len(sorted_nd) > 1:
                ax.plot(
                    sorted_nd[:, 0],
                    sorted_nd[:, 1],
                    color=color,
                    linestyle="--",
                    alpha=0.75,
                    linewidth=1.5,
                    zorder=3,
                )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)

    fig.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig, ax


def tradeoff_curve(
    F: np.ndarray,
    name: str = "Front",
    ax: plt.Axes | None = None,
    title: str = "Trade-off Curve & Operating Points",
    xlabel: str = "Mean Skill Imbalance (f1)",
    ylabel: str = "Total Wait Time (f2)",
    show_secant: bool = True,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a single trade-off curve and highlight the 3 labelled operating points.

    Operating points:
    - 'very_fair' (min f1): highlighted with square marker.
    - 'fast' (min f2): highlighted with diamond marker.
    - 'balanced' (knee point): highlighted with a star marker.

    Parameters
    ----------
    F : np.ndarray
        Array of shape (n, 2) containing objective points.
    name : str, optional
        Label for the algorithm or front.
    ax : plt.Axes, optional
        Existing axes.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    show_secant : bool, optional
        Whether to show the secant line connecting the extreme points.
    save_path : str | Path, optional
        Path to save the generated figure.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and Axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    else:
        fig = ax.get_figure()

    arr = np.asarray(F, dtype=float)
    if len(arr) == 0:
        raise ValueError("Cannot plot trade-off curve for empty points.")

    nd_pts = non_dominated(arr)
    sort_idx = np.argsort(nd_pts[:, 0])
    sorted_nd = nd_pts[sort_idx]

    # Plot continuous front curve
    ax.plot(
        sorted_nd[:, 0],
        sorted_nd[:, 1],
        color="#2b5c8f",
        linestyle="-",
        linewidth=2,
        label=f"{name} (Pareto front)",
        zorder=2,
    )
    ax.scatter(
        sorted_nd[:, 0],
        sorted_nd[:, 1],
        color="#2b5c8f",
        s=40,
        alpha=0.6,
        edgecolors="none",
        zorder=3,
    )

    # Labelled operating points
    op_pts = representative_points(arr)
    vf = op_pts["very_fair"]
    fast = op_pts["fast"]
    balanced = op_pts["balanced"]

    # Draw secant line between extremes if requested
    if show_secant and len(sorted_nd) > 2:
        ax.plot(
            [vf[0], fast[0]],
            [vf[1], fast[1]],
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label="Extremes secant line",
            zorder=1,
        )

    # Plot operating points
    ax.scatter(
        vf[0],
        vf[1],
        color="#2ca02c",
        s=120,
        marker="s",
        edgecolors="black",
        linewidths=1.2,
        label=f"very_fair ({vf[0]:.1f}, {vf[1]:.1f})",
        zorder=5,
    )
    ax.scatter(
        fast[0],
        fast[1],
        color="#1f77b4",
        s=120,
        marker="D",
        edgecolors="black",
        linewidths=1.2,
        label=f"fast ({fast[0]:.1f}, {fast[1]:.1f})",
        zorder=5,
    )
    ax.scatter(
        balanced[0],
        balanced[1],
        color="#d62728",
        s=180,
        marker="*",
        edgecolors="black",
        linewidths=1.2,
        label=f"balanced (knee) ({balanced[0]:.1f}, {balanced[1]:.1f})",
        zorder=6,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9.5)

    fig.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig, ax


def boxplot_wait_vs_imbalance(
    data: pd.DataFrame | dict[str, Any],
    ax: tuple[plt.Axes, plt.Axes] | None = None,
    title: str = "Wait Time and Skill Imbalance Distributions Across Seeds",
    imbalance_col: str = "f1_imbalance",
    wait_col: str = "f2_wait",
    algorithm_col: str = "algorithm",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Generate side-by-side boxplots of wait time and skill imbalance across seeds.

    Parameters
    ----------
    data : pd.DataFrame | dict[str, Any]
        DataFrame with columns for algorithm, imbalance, and wait; or dict
        with structure {algorithm: {'imbalance': [...], 'wait': [...]}}
        or {algorithm: (n_seeds, 2) array}.
    ax : tuple[plt.Axes, plt.Axes], optional
        Pair of Axes for imbalance and wait plots. If None, created automatically.
    title : str, optional
        Overall figure title.
    imbalance_col : str, optional
        Name of imbalance column in DataFrame.
    wait_col : str, optional
        Name of wait column in DataFrame.
    algorithm_col : str, optional
        Name of algorithm column in DataFrame.
    save_path : str | Path, optional
        Path to save figure.

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        Figure and pair of Axes.
    """
    df = _standardize_experiment_df(
        data,
        imbalance_col=imbalance_col,
        wait_col=wait_col,
        algorithm_col=algorithm_col,
    )

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    else:
        ax1, ax2 = ax
        fig = ax1.get_figure()

    # Left plot: Skill Imbalance
    sns.boxplot(
        data=df,
        x=algorithm_col,
        y=imbalance_col,
        hue=algorithm_col,
        palette="Set2",
        ax=ax1,
        width=0.45,
        legend=False,
    )
    sns.stripplot(
        data=df,
        x=algorithm_col,
        y=imbalance_col,
        color="black",
        alpha=0.5,
        jitter=0.15,
        size=5,
        ax=ax1,
    )
    ax1.set_title("Skill Imbalance (f1, lower = fairer)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Algorithm", fontsize=10)
    ax1.set_ylabel("Mean Team Imbalance", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax1.tick_params(axis="x", rotation=15)

    # Right plot: Wait Time
    sns.boxplot(
        data=df,
        x=algorithm_col,
        y=wait_col,
        hue=algorithm_col,
        palette="Set2",
        ax=ax2,
        width=0.45,
        legend=False,
    )
    sns.stripplot(
        data=df,
        x=algorithm_col,
        y=wait_col,
        color="black",
        alpha=0.5,
        jitter=0.15,
        size=5,
        ax=ax2,
    )
    ax2.set_title("Total Wait Time (f2, lower = faster)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Algorithm", fontsize=10)
    ax2.set_ylabel("Queue Wait Time (s)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax2.tick_params(axis="x", rotation=15)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig, (ax1, ax2)


# Alias for flexible naming
boxplot_wait_imbalance = boxplot_wait_vs_imbalance


def _standardize_experiment_df(
    data: pd.DataFrame | dict[str, Any],
    imbalance_col: str,
    wait_col: str,
    algorithm_col: str,
) -> pd.DataFrame:
    """Normalize input experiment data into a tidy DataFrame with expected columns."""
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        # Check column synonyms if exact name is absent
        synonyms = {
            imbalance_col: ["imbalance", "f1", "f1_imbalance", "skill_imbalance"],
            wait_col: ["wait", "f2", "f2_wait", "total_wait"],
            algorithm_col: ["algorithm", "algo", "method", "model"],
        }
        for target, syn_list in synonyms.items():
            if target not in df.columns:
                for syn in syn_list:
                    if syn in df.columns:
                        df[target] = df[syn]
                        break
        return df

    # If dict provided, convert to DataFrame
    rows = []
    if isinstance(data, dict):
        for algo, val in data.items():
            if isinstance(val, dict):
                # {algo: {'imbalance': [...], 'wait': [...]}}
                imb_vals = val.get("imbalance", val.get(imbalance_col, []))
                wait_vals = val.get("wait", val.get(wait_col, []))
                for imb, w in zip(imb_vals, wait_vals):
                    rows.append({algorithm_col: algo, imbalance_col: imb, wait_col: w})
            elif isinstance(val, (np.ndarray, list)):
                # {algo: array of shape (seeds, 2)}
                val_arr = np.asarray(val, dtype=float)
                if val_arr.ndim == 2 and val_arr.shape[1] >= 2:
                    for r in val_arr:
                        rows.append({algorithm_col: algo, imbalance_col: r[0], wait_col: r[1]})

    return pd.DataFrame(rows)
