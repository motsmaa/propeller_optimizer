"""Phase-5 slice 5a: free-form 3-D spine, rotation-minimizing frames, feasibility.

PURE NUMPY. No gmsh, no CAD kernel — this is the core the optimizer would run,
so it stays inside the runtime dependency set. The OCC sweep that turns these
frames into a solid lives in `hf/loop_sweep.py`, under the `hf` extra.

WHY NOT FRENET. The Frenet normal is the curvature direction, which is
undefined at an inflection point and REVERSES through one. A free-form
optimizer generates inflections constantly (any S-shaped spine has one), so a
Frenet-swept section would flip 180 degrees mid-blade and the loft would fold
through itself. The frame here is ROTATION-MINIMIZING: it is parallel-
transported along the curve, so it has no curvature dependence and nothing to
flip. Implemented by the DOUBLE REFLECTION method (Wang, Juttler, Zheng & Liu,
ACM TOG 2008): reflect the previous frame in the plane bisecting the segment,
then again in the plane bisecting the two tangents. Two reflections compose to
a rotation, so orthonormality is preserved exactly rather than drifting the way
repeated Gram-Schmidt does, and the method is O(h^4) accurate against the true
transport.

FRAME (verified against geometry.build_blade_solid and occ_geometry.station_rings):
x tangential, y AXIAL = rotation axis, z radial = span. Radius about the axis
is hypot(x, z).

DEGENERACY IS A SolverError, NEVER A nan. A cusp or a zero-length segment mid-
spine would divide by zero inside the transport recurrence and silently poison
every downstream frame with nans; both are checked BEFORE the recurrence runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import LoopDesign, SolverError
from .geometry import DEFAULT_N_CHORD, DEFAULT_SECTION, SectionFn, section_polygon

# Seed conditioning thresholds, MEASURED not chosen by inspection — see
# rmf_seed_thresholds.py. Criterion: p95 seed-normal direction wobble <= 1.0 deg
# under a 1e-3 * radius_m perturbation of the first two control points.
#   m = |y_hat x t0| = hypot(tx, tz)   (sine of the angle from the ROTATION AXIS)
#   d = |P1 - P0| / radius_m
# That criterion was UNSATISFIABLE under the earlier radial seed (its best cell
# wobbled 1.52 deg) and had to be re-derived at 1e-4. The axis seed does not
# depend on P0, which removes one route for the perturbation, and the original
# criterion is met unmodified.
SEED_M_MIN = 0.50        # measured 0.584 deg here; 0.2 fails at 1.475 deg
SEED_D_MIN = 0.20        # measured 0.873 deg here; 0.1 fails at 1.787 deg

# A spine segment shorter than this fraction of radius_m is treated as a cusp /
# zero tangent: the transport recurrence divides by |segment|^2, so this is
# checked before the recurrence, not after it produces nans.
CUSP_SEG_MIN = 1e-9

SPINE_SAMPLES = 400      # dense samples for arc-length and feasibility tests


@dataclass(frozen=True)
class SweptFrames:
    """Arc-length-parameterised spine with a rotation-minimizing frame."""
    s: np.ndarray            # (M,) arc-length fraction, 0 -> 1
    points_m: np.ndarray     # (M, 3)
    tangent: np.ndarray      # (M, 3) unit
    normal: np.ndarray       # (M, 3) unit, transported (the twist datum)
    binormal: np.ndarray     # (M, 3) unit = tangent x normal
    length_m: float


# ------------------------------------------------------------------ spline ---


def _knots(n_ctrl: int, degree: int = 3) -> np.ndarray:
    """Clamped uniform knot vector: degree+1 zeros, uniform interior, degree+1 ones."""
    n_int = n_ctrl - degree - 1
    return np.concatenate([np.zeros(degree + 1),
                           (np.arange(1, n_int + 1) / (n_int + 1.0)
                            if n_int > 0 else np.empty(0)),
                           np.ones(degree + 1)])


def bspline_eval(ctrl: np.ndarray, u: np.ndarray, degree: int = 3) -> np.ndarray:
    """Clamped cubic B-spline at parameters u in [0, 1], by de Boor's algorithm.

    de Boor rather than an explicit basis sum because it is the numerically
    stable form: it is a sequence of convex combinations, so no evaluated point
    can escape the local control hull even at the clamped ends.
    """
    ctrl = np.asarray(ctrl, dtype=float)
    n = len(ctrl)
    if n < degree + 1:
        raise SolverError(f"a degree-{degree} spline needs >= {degree + 1} "
                          f"control points, got {n}")
    kv = _knots(n, degree)
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0 - 1e-12)
    out = np.empty((len(u), ctrl.shape[1]))
    for j, uu in enumerate(u):
        k = int(np.searchsorted(kv, uu, side="right") - 1)
        k = min(max(k, degree), n - 1)
        d = [ctrl[k - degree + i].copy() for i in range(degree + 1)]
        for r in range(1, degree + 1):
            for i in range(degree, r - 1, -1):
                lo = kv[k - degree + i]
                hi = kv[k + 1 + i - r]
                a = 0.0 if hi == lo else (uu - lo) / (hi - lo)
                d[i] = (1.0 - a) * d[i - 1] + a * d[i]
        out[j] = d[degree]
    return out


# -------------------------------------------------------------------- seed ---


def seed_normal(p0: np.ndarray, p1: np.ndarray, radius_m: float,
                *, check: bool = True) -> np.ndarray:
    """The contract's RMF seed at s = 0: normalise(y_hat x t0).

    Chosen by measurement (`seed_equivalence.py`): of the three candidates
    sharing the degeneracy |y_hat x t0|, only this one reproduces the radial
    path's twist datum, to 1.1e-13 m on a straight radial spine. The other two
    sit at exactly +90 and +180 degrees.

    The contract raises ValueError on exact degeneracy; that guard remains for
    hand-built designs. This raises SolverError on the wider, MEASURED
    ill-conditioned band, so a genome landing there is penalized rather than
    aborting the run.
    """
    d = np.asarray(p1, float) - np.asarray(p0, float)
    dn = float(np.linalg.norm(d))
    d_frac = dn / radius_m
    if check and d_frac < SEED_D_MIN:
        raise SolverError(
            f"RMF seed ill-conditioned: |P1-P0| = {d_frac:.4g} * radius_m is "
            f"below the measured floor {SEED_D_MIN:g}, so the start tangent "
            "direction - and hence the twist datum - is not a stable function "
            "of the design (see rmf_seed_thresholds.py)")
    if dn <= 0.0:
        raise SolverError("RMF seed: the first two control points coincide")
    t0 = d / dn
    # hypot(tx, tz), not sqrt(1 - ty**2): equal analytically, but the latter
    # loses its significant digits exactly where this check bites.
    m = float(np.hypot(t0[0], t0[2]))
    if check and m < SEED_M_MIN:
        raise SolverError(
            f"RMF seed ill-conditioned: the start tangent is within "
            f"{np.degrees(np.arcsin(min(m, 1.0))):.2f} deg of the ROTATION "
            f"AXIS (sin = {m:.4g} < {SEED_M_MIN:g}), so the projected seed "
            "normal - the datum every twist_rad is measured from - is not a "
            "stable function of the design")
    if m <= 0.0:
        raise SolverError("RMF seed: start tangent parallel to the rotation axis")
    return np.array([t0[2] / m, 0.0, -t0[0] / m])


# ---------------------------------------------------------------- transport ---


def rmf_frames(points: np.ndarray, n0: np.ndarray,
               radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Parallel-transport `n0` along a polyline by DOUBLE REFLECTION.

    Returns (tangent, normal), both (M, 3) unit. Raises SolverError on a cusp
    or zero-length segment BEFORE any division, so nans can never enter the
    recurrence and propagate silently to every later frame.
    """
    p = np.asarray(points, float)
    seg = np.diff(p, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    bad = np.flatnonzero(seg_len < CUSP_SEG_MIN * radius_m)
    if bad.size:
        raise SolverError(
            f"cusp / zero tangent in the spine at {bad.size} of {len(seg_len)} "
            f"segments (first at index {int(bad[0])}, length "
            f"{seg_len[bad[0]]:.3e} m): the transport recurrence divides by the "
            "segment length, so this is refused before it becomes a nan")

    t = np.empty_like(p)
    t[:-1] = seg / seg_len[:, None]
    t[-1] = t[-2]
    # interior tangents: average of adjacent segment directions, renormalised.
    # A zero here means the curve doubles back on itself within one sample.
    t[1:-1] = 0.5 * (t[0:-2] + t[1:-1])
    tn = np.linalg.norm(t, axis=1)
    rev = np.flatnonzero(tn < 1e-12)
    if rev.size:
        raise SolverError(
            f"spine reverses direction within one sample at index "
            f"{int(rev[0])} (cusp): the tangent is undefined there")
    t /= tn[:, None]

    nrm = np.empty_like(p)
    # The seed is defined against the curve's EXACT start tangent (the control
    # polygon's P1-P0), but the transport runs on the DISCRETE tangent of the
    # sampled polyline, and the two differ by a discretisation angle. Seeding
    # without re-projecting leaves the first section sheared - measured 1.8e-3
    # non-orthogonality at sample 0 against ~1e-16 everywhere else. Re-project
    # onto the tangent the recurrence actually uses; this is the same datum,
    # discretised consistently, and converges as the sampling rises.
    n_seed = n0 - float(n0 @ t[0]) * t[0]
    ln0 = float(np.linalg.norm(n_seed))
    if ln0 <= 0.0:
        raise SolverError(
            "RMF seed collapsed onto the discretised start tangent; the "
            "twist datum is undefined at this sampling")
    nrm[0] = n_seed / ln0
    for i in range(len(p) - 1):
        v1 = p[i + 1] - p[i]
        c1 = float(v1 @ v1)
        n_l = nrm[i] - (2.0 / c1) * float(v1 @ nrm[i]) * v1
        t_l = t[i] - (2.0 / c1) * float(v1 @ t[i]) * v1
        v2 = t[i + 1] - t_l
        c2 = float(v2 @ v2)
        # c2 == 0 means the reflected tangent already equals the next tangent
        # (a straight run): the first reflection alone is the transport.
        nrm[i + 1] = n_l if c2 <= 0.0 else n_l - (2.0 / c2) * float(v2 @ n_l) * v2
        nrm[i + 1] -= float(nrm[i + 1] @ t[i + 1]) * t[i + 1]
        ln = float(np.linalg.norm(nrm[i + 1]))
        if ln <= 0.0 or not np.isfinite(ln):
            raise SolverError(f"frame transport collapsed at sample {i + 1}")
        nrm[i + 1] /= ln
    return t, nrm


def spine_frames(design: LoopDesign, n_samples: int = SPINE_SAMPLES,
                 *, check_seed: bool = True) -> SweptFrames:
    """Sample the spine uniformly in ARC LENGTH and transport the frame along it.

    Arc-length parameterisation matters twice over: `station_s` is defined as an
    arc-length fraction in the contract, and the double-reflection error scales
    with segment length, so uniform segments spread that error evenly.
    """
    ctrl = np.asarray(design.spine_ctrl_m, float)
    u = np.linspace(0.0, 1.0, max(n_samples * 4, 64))
    dense = bspline_eval(ctrl, u)
    d = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = float(cum[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise SolverError(f"spine has non-physical arc length {total!r}")
    s = np.linspace(0.0, 1.0, n_samples)
    pts = np.column_stack([np.interp(s * total, cum, dense[:, k])
                           for k in range(3)])
    n0 = seed_normal(ctrl[0], ctrl[1], design.radius_m, check=check_seed)
    t, nrm = rmf_frames(pts, n0, design.radius_m)
    return SweptFrames(s, pts, t, nrm, np.cross(t, nrm), total)


# ------------------------------------------------------------- feasibility ---


def section_radius(chord_m: np.ndarray, section: SectionFn = DEFAULT_SECTION,
                   n_chord: int = DEFAULT_N_CHORD,
                   te_thickness_m: float = 2.0e-4) -> np.ndarray:
    """Circumradius of the placed section about the pitch axis, per station (m).

    Read off the SAME polygon the sweep lofts, so the clearance tests cannot
    drift from the geometry they are supposed to bound.
    """
    c = np.asarray(chord_m, float)
    out = np.empty(len(c))
    for i, ci in enumerate(c):
        poly = section_polygon(section, n_chord, te_thickness_m / ci)
        xy = np.column_stack([(poly[:, 0] - 0.25) * ci, poly[:, 1] * ci])
        out[i] = float(np.linalg.norm(xy, axis=1).max())
    return out


def min_curvature_radius(fr: SweptFrames) -> float:
    """Smallest osculating-circle radius along the spine (m), from the
    circumradius of consecutive sample triples: R = abc / 4A."""
    p = fr.points_m
    a = np.linalg.norm(p[1:-1] - p[:-2], axis=1)
    b = np.linalg.norm(p[2:] - p[1:-1], axis=1)
    c = np.linalg.norm(p[2:] - p[:-2], axis=1)
    area = 0.5 * np.linalg.norm(np.cross(p[1:-1] - p[:-2], p[2:] - p[:-2]), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(area > 0.0, a * b * c / (4.0 * area), np.inf)
    return float(np.nanmin(r))


def curvature_margin(fr: SweptFrames, sec_r: np.ndarray) -> float:
    """CHEAP self-intersection proxy, g <= 0 feasible.

    A tube of local radius rho swept along a curve folds through itself on the
    concave side as soon as the curvature radius drops below rho. This is
    necessary, not sufficient (it cannot see two distant parts of the spine
    approaching), which is why `self_intersection_margin` runs as well.
    """
    return float(sec_r.max() / min_curvature_radius(fr) - 1.0)


def self_intersection_margin(fr: SweptFrames, sec_r: np.ndarray,
                             exclude_radii: float = 1.5) -> float:
    """ACTUAL proximity test, g <= 0 feasible.

    Minimum distance between spine samples against the sum of the two local
    section radii, EXCLUDING pairs that are near each other ALONG THE ARC. That
    exclusion is not optional and it is not a fixed sample count: a tube of
    radius rho legitimately occupies a neighbourhood of size ~rho around its own
    spine, so any two spine points closer than ~rho in arc length are the same
    piece of tube and always "overlap" in the distance metric. Excluding a fixed
    number of samples instead makes the test report 100% self-intersection for
    every design, because the exclusion width then depends on the sampling
    density rather than on the geometry.

    `exclude_radii` is the exclusion half-width in units of the summed local
    section radii.
    """
    p = fr.points_m
    r = np.interp(fr.s, np.linspace(0.0, 1.0, len(sec_r)), sec_r)
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    lim = r[:, None] + r[None, :]
    arc = np.abs(fr.s[:, None] - fr.s[None, :]) * fr.length_m
    mask = arc > exclude_radii * lim
    if not mask.any():
        # the whole spine is shorter than one section: every part of it is the
        # same piece of tube, which is a degenerate blade, not a clean one
        return 1.0
    return float(np.max((lim[mask] - d[mask]) / lim[mask]))


def blade_clearance_margin(fr: SweptFrames, sec_r: np.ndarray,
                           n_blades: int) -> float:
    """Blade-to-blade clearance at 360/n_blades spacing, g <= 0 feasible."""
    if n_blades < 2:
        return -1.0
    th = 2.0 * np.pi / n_blades
    c, s = np.cos(th), np.sin(th)
    p = fr.points_m
    q = np.column_stack([p[:, 0] * c + p[:, 2] * s, p[:, 1],
                         -p[:, 0] * s + p[:, 2] * c])
    r = np.interp(fr.s, np.linspace(0.0, 1.0, len(sec_r)), sec_r)
    d = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=2)
    lim = r[:, None] + r[None, :]
    return float(np.max((lim - d) / lim))


def hub_junction_margin(fr: SweptFrames, sec_r: np.ndarray,
                        design: LoopDesign) -> tuple[float, float]:
    """Closure continuity at the hub, g <= 0 feasible, as (tangent, section).

    TANGENT: the spine must cross the hub surface TRANSVERSALLY at both feet.
    A spine that leaves along the hub tangent grazes the surface and the root
    junction has no well-defined intersection. Measured as |t . radial|,
    required >= sin(15 deg).

    SECTION: for a hub-closing loop the two feet both sit on the hub cylinder,
    so they collide if the spine returns too near its own start. Measured as
    the distance between the two end points against the sum of their section
    radii.
    """
    p, t = fr.points_m, fr.tangent
    worst = 0.0
    for idx in (0, -1):
        r_xy = np.hypot(p[idx, 0], p[idx, 2])
        if r_xy <= 0.0:
            return 1.0, 1.0
        radial = np.array([p[idx, 0] / r_xy, 0.0, p[idx, 2] / r_xy])
        worst = max(worst, float(np.sin(np.radians(15.0))
                                 - abs(float(t[idx] @ radial))))
    gap = float(np.linalg.norm(p[0] - p[-1]))
    lim = float(sec_r[0] + sec_r[-1])
    sect = (lim - gap) / lim if design.topology == "loop_hub" else -1.0
    return worst, float(sect)
