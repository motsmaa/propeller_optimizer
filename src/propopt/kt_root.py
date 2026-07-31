"""Measured root stress-concentration factors Kt(r/t) — MEASURED DATA.

This module is a DATA TABLE plus an interpolator. It contains no model, no fit
and no tuning knob. It exists so `solvers/beam.py` can amplify its root stress
by numbers that were measured, instead of by the 2.5 placeholder it carried
from Phase 2b.

PROVENANCE (how these numbers were produced — Phase-4a slice 3b,
`hf/kt2d.py:wall_on_plate_kt`, reproducible with gmsh + CalculiX):

  * MODEL — 2-D plane strain, CPE6 (6-node quadratic triangles), unit
    out-of-plane depth. A wall of thickness t standing on a plate, blended by a
    circular fillet of radius r; i.e. the stepped-flat-bar geometry, with
    D/d = 8. Loads applied as CONSISTENT nodal vectors (3-point Gauss over the
    quadratic edge, weights L/6, 2L/3, L/6) — tributary-length weighting is
    wrong for quadratic elements and fails a uniform-tension patch test by
    tens of percent.
  * BENDING — a pure moment (linear traction, zero net force, zero shear); the
    load case that matches `beam.py`'s `sigma_bend`.
  * AXIAL — uniform tension; the load case that matches `beam.py`'s
    centrifugal `N/A`.
  * Kt = max in-plane principal stress on the fillet arc / sigma_nom, where
    sigma_nom is read from the wall itself well above the blend. On EVERY point
    below, sigma_nom reproduced the elementary value (6M/t^2 or F/t) to 0.00%.
  * VALIDATION — the same extraction, mesh strategy and load path, run on the
    Kirsch problem (circular hole in a wide plate, exact Kt = 3), converges
    MONOTONICALLY FROM BELOW and lands at -0.19% at the resolution used here
    (elem_across = 16) and +0.03% at elem_across = 32. The plate ratio is
    SATURATED: D/d = 4/6/8/12 give Kt within +/-0.05% of one another.
  * EXCLUDED — r/t = 0.01 is NOT in the table. It did not converge (Kt
    oscillated 4.08-4.78 across the refinement ladder while the element count
    barely grew). Reporting a number there would be reporting mesh noise.

RANGE IS A VALIDITY LIMIT, NOT A SAMPLING LIMIT — hence `kt_factors` RAISES
rather than clamping outside it:
  * below 0.020 is exactly where the model failed to converge; extrapolating a
    steepening curve into the region the FEA could not resolve would be
    inventing data;
  * above 0.500 the two face fillets collide (r > t/2 is not a filleted wall
    any more) and `wall_on_plate_kt` refuses to build the geometry at all.
Clamping would silently return a number that was never measured. For the
shipped design space neither end is reachable — see `tests/test_kt_root.py`.

INTERPOLATION is piecewise linear in log(Kt - 1) vs log(r/t). That form is
chosen because it (a) passes through every measured point exactly, (b) reduces
between points to a LOCAL power law, which is the functional form the data
actually follows, and (c) is monotone, so no interpolation artifact can invent
a peak between samples. The global power-law fits quoted in the README
(bending 1 + 0.1796 (r/t)^-0.7051, axial 1 + 0.3683 (r/t)^-0.6512) carry
5.9% / 6.8% maximum error and are DELIBERATELY NOT used in the stress path.

VALIDITY (inherited from the 2-D model, carried here so the caller cannot lose
it): plane strain holds at MID-CHORD, where the blade root is locally a long
prismatic wall. Near the leading and trailing edges the section is short in the
chordwise direction and the true state approaches plane STRESS; those regions
are not covered. A 3-D confirmation on the real filleted blade is outstanding.
"""
from __future__ import annotations

import numpy as np

from .contracts import SolverError

# r / t, where r is the fillet radius and t the wall (root) thickness.
R_OVER_T = np.array([0.020, 0.030, 0.040, 0.056, 0.075,
                     0.100, 0.150, 0.200, 0.300, 0.400, 0.500])

# Pure bending (matches beam.py sigma_bend).
KT_BENDING = np.array([3.6202, 3.0740, 2.7439, 2.4135, 2.1692,
                       1.9562, 1.7186, 1.5769, 1.4170, 1.3302, 1.2742])

# Uniform axial tension (matches beam.py centrifugal N/A).
KT_AXIAL = np.array([5.3453, 4.5089, 3.9906, 3.4698, 3.0867,
                     2.7493, 2.3384, 2.1016, 1.8122, 1.6427, 1.5290])

R_OVER_T_MIN = float(R_OVER_T[0])
R_OVER_T_MAX = float(R_OVER_T[-1])

_LOG_R = np.log(R_OVER_T)
_LOG_KT = {"bending": np.log(KT_BENDING - 1.0), "axial": np.log(KT_AXIAL - 1.0)}


def kt_interp(r_over_t: float, load: str) -> float:
    """Interpolate one measured curve. `load` is "bending" or "axial"."""
    try:
        log_kt = _LOG_KT[load]
    except KeyError:
        raise SolverError(f"load must be bending|axial, got {load!r}") from None
    x = float(r_over_t)
    if not np.isfinite(x):
        raise SolverError(f"non-finite root r/t: {x!r}")
    if not (R_OVER_T_MIN <= x <= R_OVER_T_MAX):
        raise SolverError(
            f"root r/t = {x:.4f} is outside the measured range "
            f"[{R_OVER_T_MIN}, {R_OVER_T_MAX}] — below it the 2-D model does "
            f"not converge, above it the fillets collide (r > t/2). See "
            f"propopt.kt_root; the table is not extrapolated.")
    return 1.0 + float(np.exp(np.interp(np.log(x), _LOG_R, log_kt)))


def kt_factors(r_over_t: float) -> tuple[float, float]:
    """(Kt_axial, Kt_bending) at a root fillet ratio r/t. Order matches the
    order the terms are composed in `beam.py`: axial first, then bending."""
    return kt_interp(r_over_t, "axial"), kt_interp(r_over_t, "bending")
