# OWNER: M4 - do not edit unless you are M4
"""Unit tests for Pareto analysis routines (mm.pareto)."""
from __future__ import annotations

import numpy as np
import pytest

from mm.pareto import (
    hypervolume,
    is_non_dominated,
    non_dominated,
    representative_points,
    spacing,
    spread,
)


def test_is_non_dominated_basic() -> None:
    """Test standard 2D Pareto dominance filtering."""
    # Points:
    # A: [1, 5] (non-dominated)
    # B: [2, 3] (non-dominated)
    # C: [5, 1] (non-dominated)
    # D: [3, 4] (dominated by B: 2 <= 3 and 3 <= 4)
    # E: [6, 6] (dominated by all)
    F = np.array([
        [1.0, 5.0],
        [2.0, 3.0],
        [5.0, 1.0],
        [3.0, 4.0],
        [6.0, 6.0],
    ])
    mask = is_non_dominated(F)
    assert np.array_equal(mask, [True, True, True, False, False])

    nd = non_dominated(F)
    assert nd.shape == (3, 2)
    assert np.allclose(nd, [[1.0, 5.0], [2.0, 3.0], [5.0, 1.0]])


def test_is_non_dominated_identical_points() -> None:
    """Identical points do not dominate each other and should remain non-dominated."""
    F = np.array([
        [2.0, 4.0],
        [2.0, 4.0],
        [5.0, 5.0],
    ])
    mask = is_non_dominated(F)
    assert np.array_equal(mask, [True, True, False])


def test_non_dominated_empty_and_single() -> None:
    """Edge cases: empty array and single-point front."""
    empty_F = np.empty((0, 2))
    assert len(non_dominated(empty_F)) == 0
    assert len(is_non_dominated(empty_F)) == 0

    single_F = np.array([[3.0, 4.0]])
    assert np.array_equal(is_non_dominated(single_F), [True])
    assert np.array_equal(non_dominated(single_F), [[3.0, 4.0]])


def test_non_dominated_return_mask() -> None:
    """Test return_mask parameter on non_dominated."""
    F = np.array([[1.0, 2.0], [2.0, 3.0]])
    mask = non_dominated(F, return_mask=True)
    assert np.array_equal(mask, [True, False])


def test_hypervolume_2d_known_case() -> None:
    """Test hypervolume against known analytical calculation.

    Analytical verification:
    Points: A = (2, 8), B = (5, 4). Reference point: ref = (10, 10).
    Vertical rectangular strips:
    - Strip 1: x in [2, 5], width = 3, y bounded by y_A = 8, height = (10 - 8) = 2 -> Area = 6.
    - Strip 2: x in [5, 10], width = 5, y bounded by y_B = 4, height = (10 - 4) = 6 -> Area = 30.
    Total Hypervolume = 6 + 30 = 36.0.
    """
    F = np.array([[2.0, 8.0], [5.0, 4.0]])
    ref = np.array([10.0, 10.0])

    hv = hypervolume(F, ref)
    assert pytest.approx(36.0) == hv

    # Compare with pymoo HV indicator
    from pymoo.indicators.hv import HV
    pymoo_hv = HV(ref_point=ref)(F)
    assert pytest.approx(pymoo_hv) == hv


def test_hypervolume_with_dominated_points_and_outside_ref() -> None:
    """Dominated points and points worse than ref must not distort hypervolume."""
    # Front is same A=(2, 8), B=(5, 4). Point C=(6, 9) is dominated. Point D=(11, 2) is outside ref.
    F = np.array([
        [2.0, 8.0],
        [5.0, 4.0],
        [6.0, 9.0],
        [11.0, 2.0],
    ])
    ref = np.array([10.0, 10.0])
    hv = hypervolume(F, ref)
    assert pytest.approx(36.0) == hv


def test_hypervolume_empty_and_outside() -> None:
    """Empty front or all points outside reference should return 0.0."""
    ref = np.array([10.0, 10.0])
    assert hypervolume(np.empty((0, 2)), ref) == 0.0

    F_outside = np.array([[12.0, 12.0], [15.0, 5.0]])
    assert hypervolume(F_outside, ref) == 0.0


def test_hypervolume_1d() -> None:
    """1D hypervolume should equal ref - min(F)."""
    F = np.array([[3.0], [5.0], [7.0]])
    ref = [10.0]
    assert pytest.approx(7.0) == hypervolume(F, ref)


def test_hypervolume_3d() -> None:
    """Test 3D hypervolume on known cube volume."""
    # Single point [2, 2, 2] bounded by [10, 10, 10] forms a cube of side 8 -> 512.0
    F = np.array([[2.0, 2.0, 2.0]])
    ref = np.array([10.0, 10.0, 10.0])
    hv = hypervolume(F, ref)
    assert pytest.approx(512.0) == hv


def test_spacing_equidistant() -> None:
    """Spacing of perfectly equidistant points should be 0.0."""
    # 5 points spaced exactly 2 units apart along x: (0, 0), (2, 0), (4, 0), (6, 0), (8, 0)
    F = np.array([[0.0, 8.0], [2.0, 6.0], [4.0, 4.0], [6.0, 2.0], [8.0, 0.0]])
    s = spacing(F)
    assert pytest.approx(0.0, abs=1e-6) == s


def test_spacing_small_and_non_uniform() -> None:
    """Spacing of <=1 points is 0; non-uniform spacing is > 0."""
    assert spacing(np.array([[1.0, 2.0]])) == 0.0
    assert spacing(np.empty((0, 2))) == 0.0

    # Non-uniform points: distances are 1, 10
    F = np.array([[0.0, 10.0], [1.0, 9.0], [11.0, 0.0]])
    assert spacing(F) > 0.0


def test_spread_metric() -> None:
    """Test Deb's spread metric Delta."""
    assert spread(np.empty((0, 2))) == 0.0
    assert spread(np.array([[1.0, 2.0]])) == 0.0

    # Uniform front
    F_uniform = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])
    delta = spread(F_uniform)
    assert pytest.approx(0.0, abs=1e-6) == delta


def test_representative_points_identification() -> None:
    """Test labelled operating points on a convex Pareto curve."""
    # Points along a trade-off curve where knee point is clearly at (3, 3)
    # (1, 10) -> very_fair (min f1 = 1)
    # (10, 1) -> fast (min f2 = 1)
    # (3, 3)  -> balanced knee point bending towards ideal (0,0)
    F = np.array([
        [1.0, 10.0],
        [2.0, 5.0],
        [3.0, 3.0],
        [5.0, 2.0],
        [10.0, 1.0],
    ])
    reps = representative_points(F)
    assert np.array_equal(reps["very_fair"], [1.0, 10.0])
    assert np.array_equal(reps["fast"], [10.0, 1.0])
    assert np.array_equal(reps["balanced"], [3.0, 3.0])

    # Test return_indices
    indices = representative_points(F, return_indices=True)
    assert indices["very_fair"] == 0
    assert indices["fast"] == 4
    assert indices["balanced"] == 2


def test_representative_points_edge_cases() -> None:
    """Edge cases: 1 point, 2 points, and empty array."""
    with pytest.raises(ValueError, match="Cannot select representative points"):
        representative_points(np.empty((0, 2)))

    # Single point
    single = np.array([[5.0, 5.0]])
    reps1 = representative_points(single)
    assert np.array_equal(reps1["very_fair"], [5.0, 5.0])
    assert np.array_equal(reps1["fast"], [5.0, 5.0])
    assert np.array_equal(reps1["balanced"], [5.0, 5.0])

    # Two points
    two_pts = np.array([[1.0, 8.0], [7.0, 2.0]])
    reps2 = representative_points(two_pts)
    assert np.array_equal(reps2["very_fair"], [1.0, 8.0])
    assert np.array_equal(reps2["fast"], [7.0, 2.0])
    assert "balanced" in reps2
