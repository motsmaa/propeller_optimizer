"""Sanity checks for the tabulated XFOIL polar (parsing + interpolation only —
model accuracy is judged by tests/test_validation_uiuc.py, not here)."""
from pathlib import Path

import numpy as np
import pytest

from propopt.solvers.polars import naca4412_polar

POLAR_DIR = Path(__file__).resolve().parent.parent / "data" / "polars"

pytestmark = pytest.mark.skipif(
    not POLAR_DIR.exists() or not list(POLAR_DIR.glob("naca4412_re*.txt")),
    reason="tabulated polar data missing (see data/polars/SOURCE.md)",
)


def test_parses_all_re_levels_and_plausible_midtable_values():
    polar = naca4412_polar(POLAR_DIR)
    n_files = len(list(POLAR_DIR.glob("naca4412_re*.txt")))
    assert len(polar.re_levels) == n_files  # every table parsed, none dropped
    assert np.all(np.diff(polar.re_levels) > 0)
    # broad plausibility band for a cambered section at alpha=5 deg, Re=1e5
    cl, cd = polar(np.radians([5.0]), np.array([1e5]))
    assert 0.5 < cl[0] < 1.3
    assert 0.005 < cd[0] < 0.06


def test_reynolds_is_clamped_not_extrapolated():
    polar = naca4412_polar(POLAR_DIR)
    a = np.radians([4.0])
    re_min, re_max = polar.re_levels[0], polar.re_levels[-1]
    lo = polar(a, np.array([re_min / 5.0]))   # below table min
    at_min = polar(a, np.array([re_min]))
    hi = polar(a, np.array([re_max * 10.0]))  # above table max
    at_max = polar(a, np.array([re_max]))
    assert lo[0] == pytest.approx(at_min[0]) and lo[1] == pytest.approx(at_min[1])
    assert hi[0] == pytest.approx(at_max[0]) and hi[1] == pytest.approx(at_max[1])


def test_alpha_extrapolation_is_continuous_and_finite():
    polar = naca4412_polar(POLAR_DIR)
    re = np.full(2, 5e4)
    # step across the Re=5e4 table's upper edge (alpha_max = 14 deg)
    cl_a, cd_a = polar(np.radians([13.9, 14.1]), re)
    assert abs(cl_a[1] - cl_a[0]) < 0.1 and abs(cd_a[1] - cd_a[0]) < 0.05
    # deep post-stall: finite, drag-dominated flat-plate behaviour
    cl_b, cd_b = polar(np.radians([60.0]), re[:1])
    assert np.isfinite(cl_b[0]) and np.isfinite(cd_b[0])
    assert cd_b[0] > 1.0
