# OWNER: M4 - do not edit unless you are M4
"""Pareto front analysis and multi-objective performance metrics.

Provides core routines for Pareto dominance filtering, exact 2D and general 3D
hypervolume calculation, spacing and spread distribution metrics, and labelled
operating points identification (very fair, fast, balanced knee point).
"""
from __future__ import annotations

import numpy as np


def is_non_dominated(F: np.ndarray) -> np.ndarray:
    """Return a boolean mask where True indicates a non-dominated point (minimization).

    A point p dominates point q if p[i] <= q[i] for all objectives i, and
    p[i] < q[i] for at least one objective.

    Parameters
    ----------
    F : np.ndarray
        Array of shape (n, m) representing n solutions evaluated on m objectives.

    Returns
    -------
    np.ndarray
        Boolean array of shape (n,) where True indicates non-dominated points.
    """
    F_arr = np.asarray(F, dtype=float)
    if F_arr.ndim == 1:
        F_arr = F_arr.reshape(-1, 1)

    n_points = len(F_arr)
    if n_points == 0:
        return np.zeros(0, dtype=bool)
    if n_points == 1:
        return np.ones(1, dtype=bool)

    # For standard front sizes, pairwise broadcast comparison is vectorized and fast
    if n_points <= 3000:
        # diff[j, i, :] = F[j, :] - F[i, :]
        # point j dominates point i if F[j] <= F[i] (all) and F[j] < F[i] (any)
        le = F_arr[:, None, :] <= F_arr[None, :, :]
        lt = F_arr[:, None, :] < F_arr[None, :, :]
        dominates = np.all(le, axis=2) & np.any(lt, axis=2)
        # Point i is dominated if any j dominates i
        is_dominated = np.any(dominates, axis=0)
        return ~is_dominated

    # Fallback chunked comparison for very large n to conserve memory
    non_dom = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not non_dom[i]:
            continue
        p = F_arr[i]
        # Points dominated by p
        dom_by_p = np.all(p <= F_arr, axis=1) & np.any(p < F_arr, axis=1)
        non_dom[dom_by_p] = False
        # If any active point dominates p, p is dominated
        doms_p = np.all(F_arr[non_dom] <= p, axis=1) & np.any(F_arr[non_dom] < p, axis=1)
        if np.any(doms_p):
            non_dom[i] = False

    return non_dom


def non_dominated(F: np.ndarray, return_mask: bool = False) -> np.ndarray:
    """Filter an (n, m) array of minimization objectives to the non-dominated set.

    Parameters
    ----------
    F : np.ndarray
        Array of shape (n, m) representing n objective vectors.
    return_mask : bool, optional
        If True, return a boolean mask of shape (n,). If False (default),
        return the filtered (k, m) array of non-dominated points.

    Returns
    -------
    np.ndarray
        Filtered (k, m) array or (n,) boolean mask.
    """
    F_arr = np.asarray(F, dtype=float)
    mask = is_non_dominated(F_arr)
    if return_mask:
        return mask
    if F_arr.ndim == 1:
        return F_arr[mask]
    return F_arr[mask, :]


def hypervolume(F: np.ndarray, ref: np.ndarray | list[float]) -> float:
    """Compute the hypervolume indicator for a set of points against a reference point.

    Assumes all objectives are to be minimized. Points that do not weakly dominate
    the reference point (i.e. points with any coordinate strictly greater than ref)
    do not contribute to the hypervolume.

    Uses an exact analytical rectangular strip algorithm for 2 objectives, and
    pymoo's generic hypervolume calculator (or recursive dimension slicing fallback)
    for 3 or more objectives.

    Parameters
    ----------
    F : np.ndarray
        Array of shape (n, m) containing objective vectors.
    ref : np.ndarray | list[float]
        Reference point of shape (m,). Must be strictly worse than the ideal point.

    Returns
    -------
    float
        Hypervolume dominated by the non-dominated set bounded by ref.
    """
    F_arr = np.asarray(F, dtype=float)
    if F_arr.ndim == 1:
        F_arr = F_arr.reshape(-1, 1)

    ref_arr = np.asarray(ref, dtype=float)
    m = len(ref_arr)

    if len(F_arr) == 0:
        return 0.0

    if F_arr.shape[1] != m:
        raise ValueError(f"Dimension mismatch: F has {F_arr.shape[1]} objectives, ref has {m}")

    # Keep only points that are weakly better than the reference point in all objectives
    valid = np.all(F_arr <= ref_arr, axis=1)
    F_valid = F_arr[valid]

    if len(F_valid) == 0:
        return 0.0

    # Filter to non-dominated front within the bounding reference point
    front = non_dominated(F_valid)
    if len(front) == 0:
        return 0.0

    # 1D case
    if m == 1:
        return float(ref_arr[0] - np.min(front[:, 0]))

    # 2D case: exact analytical algorithm by sorting and rectangular strip decomposition
    if m == 2:
        # Sort ascending by objective 0; tie-break ascending by objective 1
        order = np.lexsort((front[:, 1], front[:, 0]))
        sorted_front = front[order]

        # In case of duplicate objective 0 values, keep the one with lowest objective 1
        unique_x, unique_idx = np.unique(sorted_front[:, 0], return_index=True)
        pts = sorted_front[unique_idx]

        # Ensure strict Pareto property in 2D (monotonically decreasing in objective 1)
        filtered_pts = []
        min_y = float("inf")
        for x, y in pts:
            if y < min_y:
                filtered_pts.append((x, y))
                min_y = y

        pts = np.array(filtered_pts, dtype=float)
        k = len(pts)
        rx, ry = ref_arr[0], ref_arr[1]

        # Slicing into vertical rectangular strips
        # Area = sum_{i=0}^{k-2} (x_{i+1} - x_i) * (ry - y_i) + (rx - x_{k-1}) * (ry - y_{k-1})
        hv = 0.0
        for i in range(k - 1):
            width = pts[i + 1, 0] - pts[i, 0]
            height = ry - pts[i, 1]
            hv += width * height
        hv += (rx - pts[k - 1, 0]) * (ry - pts[k - 1, 1])

        return float(hv)

    # 3D and higher: generic fallback using pymoo
    try:
        from pymoo.indicators.hv import HV

        hv_indicator = HV(ref_point=ref_arr)
        return float(hv_indicator(front))
    except ImportError:
        # Fallback inclusion-exclusion / recursive slicer
        return _hypervolume_recursive(front, ref_arr)


def _hypervolume_recursive(front: np.ndarray, ref: np.ndarray) -> float:
    """Recursive dimension reduction for 3D+ hypervolume if pymoo is unavailable."""
    m = len(ref)
    if m == 2:
        return hypervolume(front, ref)

    # Sort points descending by the last objective
    order = np.argsort(-front[:, -1])
    sorted_pts = front[order]

    hv = 0.0
    current_ref_last = ref[-1]

    # Process slices along the last dimension
    for i in range(len(sorted_pts)):
        p = sorted_pts[i]
        h = current_ref_last - p[-1]
        if h > 0:
            sub_front = sorted_pts[i:, :-1]
            sub_ref = ref[:-1]
            sub_hv = hypervolume(sub_front, sub_ref)
            hv += h * sub_hv
            current_ref_last = p[-1]

    return float(hv)


def spacing(F: np.ndarray) -> float:
    """Compute Schott's spacing metric (S) to quantify front distribution uniformity.

    Measures the standard deviation of distances between consecutive / nearest points
    in the objective space. A spacing of 0 indicates an ideal equidistant distribution.

    Parameters
    ----------
    F : np.ndarray
        Array of shape (n, m) containing objective vectors.

    Returns
    -------
    float
        Schott's spacing metric S. Returns 0.0 if n <= 1.
    """
    F_arr = np.asarray(F, dtype=float)
    if F_arr.ndim == 1:
        F_arr = F_arr.reshape(-1, 1)

    front = non_dominated(F_arr)
    n = len(front)
    if n <= 1:
        return 0.0

    # Nearest neighbor Euclidean distance for each point
    diff = front[:, None, :] - front[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask diagonal (distance to self = 0)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)

    d_mean = np.mean(min_dists)
    s = np.sqrt(np.sum((min_dists - d_mean) ** 2) / (n - 1))
    return float(s)


def spread(F: np.ndarray, extreme_points: np.ndarray | None = None) -> float:
    """Compute Deb's spread metric (Delta) measuring both extent and uniformity.

    Parameters
    ----------
    F : np.ndarray
        Array of shape (n, 2) containing 2D objective vectors.
    extreme_points : np.ndarray, optional
        True extreme points of shape (2, 2) to evaluate boundary coverage.
        If None, extreme points of F are used (df = dl = 0).

    Returns
    -------
    float
        Spread metric Delta (lower is better, 0 is ideal uniform spread).
    """
    F_arr = np.asarray(F, dtype=float)
    if F_arr.ndim == 1:
        F_arr = F_arr.reshape(-1, 1)

    front = non_dominated(F_arr)
    n = len(front)
    if n < 2:
        return 0.0

    # Sort front by first objective
    order = np.argsort(front[:, 0])
    sorted_front = front[order]

    # Consecutive Euclidean distances
    consecutive_diffs = np.diff(sorted_front, axis=0)
    dists = np.linalg.norm(consecutive_diffs, axis=1)
    d_mean = np.mean(dists)

    df = 0.0
    dl = 0.0
    if extreme_points is not None:
        ext = np.asarray(extreme_points, dtype=float)
        if len(ext) >= 2:
            df = float(np.linalg.norm(sorted_front[0] - ext[0]))
            dl = float(np.linalg.norm(sorted_front[-1] - ext[1]))

    numerator = df + dl + float(np.sum(np.abs(dists - d_mean)))
    denominator = df + dl + float((n - 1) * d_mean)

    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def representative_points(
    F: np.ndarray,
    return_indices: bool = False,
) -> dict[str, np.ndarray | int]:
    """Identify three labelled operating points on the Pareto front.

    The three operating points correspond to:
    - 'very_fair': minimum f1 (skill imbalance), tie-breaking by min f2 (wait time).
    - 'fast': minimum f2 (wait time), tie-breaking by min f1 (skill imbalance).
    - 'balanced': knee point maximizing perpendicular distance to the secant line
      joining the normalized extremes.

    Parameters
    ----------
    F : np.ndarray
        Array of shape (n, m) with m >= 2, where column 0 is f1 and column 1 is f2.
    return_indices : bool, optional
        If True, returns integer indices of the points in original array F.
        If False (default), returns the coordinate vectors as np.ndarray.

    Returns
    -------
    dict[str, np.ndarray | int]
        Dictionary with keys 'very_fair', 'fast', and 'balanced'.
    """
    F_arr = np.asarray(F, dtype=float)
    n = len(F_arr)
    if n == 0:
        raise ValueError("Cannot select representative points from an empty front.")

    if F_arr.ndim < 2 or F_arr.shape[1] < 2:
        raise ValueError(f"F must have at least 2 objectives, got shape {F_arr.shape}")

    # Identify non-dominated indices
    nd_mask = is_non_dominated(F_arr)
    nd_indices = np.where(nd_mask)[0]

    if len(nd_indices) == 0:
        nd_indices = np.arange(n)

    nd_points = F_arr[nd_indices]

    # 'very_fair': min f1, tie-break by min f2
    vf_order = np.lexsort((nd_points[:, 1], nd_points[:, 0]))
    vf_idx = nd_indices[vf_order[0]]

    # 'fast': min f2, tie-break by min f1
    fast_order = np.lexsort((nd_points[:, 0], nd_points[:, 1]))
    fast_idx = nd_indices[fast_order[0]]

    # 'balanced': knee point
    if len(nd_indices) <= 2:
        # With 1 or 2 points, knee point is chosen as the point closest to ideal
        ideal = np.min(nd_points[:, :2], axis=0)
        norm_dist = np.linalg.norm(nd_points[:, :2] - ideal, axis=1)
        balanced_idx = nd_indices[np.argmin(norm_dist)]
    else:
        # Min-max normalization for objectives 0 and 1
        p_sub = nd_points[:, :2]
        f_min = np.min(p_sub, axis=0)
        f_max = np.max(p_sub, axis=0)
        f_range = np.where(f_max > f_min, f_max - f_min, 1.0)
        p_norm = (p_sub - f_min) / f_range

        # Normalized coordinates of extremes
        vf_norm = (F_arr[vf_idx, :2] - f_min) / f_range
        fast_norm = (F_arr[fast_idx, :2] - f_min) / f_range

        # Vector from vf_norm to fast_norm
        v = fast_norm - vf_norm
        v_norm = np.linalg.norm(v)

        if v_norm < 1e-9:
            balanced_idx = vf_idx
        else:
            # Perpendicular distance from p_norm to secant line
            # Distance in 2D = |v_x * (p_y - vf_y) - v_y * (p_x - vf_x)| / ||v||
            u = p_norm - vf_norm
            cross = np.abs(v[0] * u[:, 1] - v[1] * u[:, 0])
            distances = cross / v_norm
            balanced_sub_idx = np.argmax(distances)
            balanced_idx = nd_indices[balanced_sub_idx]

    if return_indices:
        return {
            "very_fair": int(vf_idx),
            "fast": int(fast_idx),
            "balanced": int(balanced_idx),
        }

    return {
        "very_fair": F_arr[vf_idx].copy(),
        "fast": F_arr[fast_idx].copy(),
        "balanced": F_arr[balanced_idx].copy(),
    }
