"""Phase-5 slice 5a: spine, rotation-minimizing frame, feasibility proxies.

The load-bearing test here is `test_rmf_does_not_flip_where_frenet_does`. The
whole reason for a rotation-minimizing frame is that Frenet reverses through an
inflection point; asserting RMF is smooth is worth little unless the same curve
is shown to break Frenet, so both are computed on one curve and compared.
"""
from __future__ import annotations

import numpy as np
import pytest

from propopt.contracts import LoopDesign, SolverError
from propopt.loop_geometry import (SEED_D_MIN, SEED_M_MIN, bspline_eval,
                                   rmf_frames, seed_normal, spine_frames)


def _design(**over) -> LoopDesign:
    base = dict(
        radius_m=0.127, hub_radius_m=0.02, n_blades=2, rpm=5000.0,
        topology="loop_hub",
        spine_ctrl_m=((0.02, 0.0, 0.0), (0.06, 0.03, 0.04),
                      (0.06, -0.03, -0.04), (0.0, 0.0, -0.02)),
        station_s=(0.0, 0.5, 1.0), chord_m=(0.012, 0.020, 0.012),
        twist_rad=(0.1, 0.2, 0.1))
    return LoopDesign(**{**base, **over})


# ------------------------------------------------------------------ spline ---


def test_clamped_spline_interpolates_its_end_control_points():
    ctrl = np.array([[0.01, 0.0, 0.0], [0.06, 0.03, 0.04],
                     [0.06, -0.03, -0.04], [0.0, 0.0, -0.01]])
    p = bspline_eval(ctrl, np.array([0.0, 1.0 - 1e-12]))
    assert np.allclose(p[0], ctrl[0], atol=1e-12)
    assert np.allclose(p[-1], ctrl[-1], atol=1e-6)


def test_spline_stays_inside_the_control_hull():
    """de Boor is a chain of convex combinations, so no sample may exceed the
    control points' bounding box — the property the contract's envelope check
    relies on."""
    rng = np.random.default_rng(3)
    ctrl = rng.uniform(-0.1, 0.1, size=(6, 3))
    p = bspline_eval(ctrl, np.linspace(0, 1, 200))
    assert np.all(p >= ctrl.min(axis=0) - 1e-12)
    assert np.all(p <= ctrl.max(axis=0) + 1e-12)


# -------------------------------------------------------------------- frame ---


def _frenet_normals(p):
    """Frenet normal = normalised curvature vector, by central differences."""
    d1 = np.gradient(p, axis=0)
    d2 = np.gradient(d1, axis=0)
    t = d1 / np.linalg.norm(d1, axis=1)[:, None]
    n = d2 - np.einsum("ij,ij->i", d2, t)[:, None] * t
    ln = np.linalg.norm(n, axis=1)
    return np.where(ln[:, None] > 1e-14, n / np.maximum(ln, 1e-30)[:, None], np.nan)


def _max_consecutive_turn_deg(v):
    d = np.einsum("ij,ij->i", v[:-1], v[1:])
    ok = np.isfinite(d)
    return float(np.degrees(np.arccos(np.clip(d[ok], -1, 1))).max())


def test_rmf_does_not_flip_where_frenet_does():
    """An S-shaped spine has an inflection; Frenet's normal reverses through it
    and RMF's does not. This is the entire justification for the frame choice."""
    s = np.linspace(-1.0, 1.0, 400)
    p = np.column_stack([0.05 * s, 0.02 * s ** 3, 0.03 * np.ones_like(s)])
    fren = _frenet_normals(p)
    n0 = np.array([0.0, 1.0, 0.0])
    n0 -= n0 @ (p[1] - p[0]) / np.linalg.norm(p[1] - p[0]) ** 2 * (p[1] - p[0])
    _, rmf = rmf_frames(p, n0 / np.linalg.norm(n0), 0.127)
    frenet_turn = _max_consecutive_turn_deg(fren)
    rmf_turn = _max_consecutive_turn_deg(rmf)
    assert frenet_turn > 90.0, f"Frenet did not flip ({frenet_turn:.1f} deg)"
    assert rmf_turn < 1.0, f"RMF turned {rmf_turn:.3f} deg between samples"


def test_rmf_stays_orthonormal_along_a_writhing_spine():
    rng = np.random.default_rng(11)
    ctrl = np.column_stack([rng.uniform(-0.1, 0.1, 7), rng.uniform(-0.06, 0.06, 7),
                            rng.uniform(-0.1, 0.1, 7)])
    p = bspline_eval(ctrl, np.linspace(0, 1, 500))
    d = p[1] - p[0]
    n0 = np.cross(d, [0, 0, 1.0])
    t, n = rmf_frames(p, n0 / np.linalg.norm(n0), 0.127)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-12)
    assert np.max(np.abs(np.einsum("ij,ij->i", t, n))) < 1e-12


def test_rmf_is_sampling_convergent():
    """Doubling the sample count must not change the transported frame at the
    end of the curve — a frame that flipped or drifted would not converge."""
    ctrl = np.array([[0.01, 0.0, 0.0], [0.06, 0.03, 0.04],
                     [0.06, -0.03, -0.04], [0.0, 0.0, -0.01]])
    ends = []
    for m in (500, 1000, 2000):
        p = bspline_eval(ctrl, np.linspace(0, 1, m))
        d = p[1] - p[0]
        n0 = np.cross(d, [0, 1.0, 0])
        _, n = rmf_frames(p, n0 / np.linalg.norm(n0), 0.127)
        ends.append(n[-1])
    a = np.degrees(np.arccos(np.clip(ends[0] @ ends[1], -1, 1)))
    b = np.degrees(np.arccos(np.clip(ends[1] @ ends[2], -1, 1)))
    assert b < a and b < 0.05, f"end-frame not converging: {a:.4f} -> {b:.4f} deg"


def test_cusp_is_a_solver_error_not_a_nan():
    """A zero-length segment divides by zero inside the transport recurrence.
    It must be refused BEFORE that, or every later frame silently becomes nan."""
    p = np.array([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.02, 0.0, 0.0],
                  [0.03, 0.0, 0.01], [0.04, 0.0, 0.02]])
    with pytest.raises(SolverError, match="cusp / zero tangent"):
        rmf_frames(p, np.array([0.0, 1.0, 0.0]), 0.127)


def test_spine_that_doubles_back_is_a_solver_error():
    p = np.array([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.01, 0.0, 0.0],
                  [0.02, 0.0, 0.0]])
    with pytest.raises(SolverError, match="cusp|reverses"):
        rmf_frames(p, np.array([0.0, 1.0, 0.0]), 0.127)


# --------------------------------------------------------------------- seed ---


def test_seed_is_perpendicular_to_the_tangent_and_lies_in_the_rotor_plane():
    """n0 = y_hat x t0 is perpendicular to BOTH the tangent and the rotation
    axis, so it has no axial component at all."""
    p0 = np.array([0.02, 0.0, 0.0])
    p1 = np.array([0.06, 0.03, 0.05])
    n = seed_normal(p0, p1, 0.127)
    t = (p1 - p0) / np.linalg.norm(p1 - p0)
    assert abs(float(n @ t)) < 1e-12
    assert abs(float(n[1])) < 1e-15, "seed must lie in the rotor plane"
    assert abs(float(np.linalg.norm(n)) - 1.0) < 1e-15


def test_seed_reproduces_the_radial_convention_on_a_radial_tangent():
    """THE equivalence pin. On a straight radial spine (t0 = +z) the seed must
    be +x, which is what makes a swept section coincide with PropellerDesign's
    at the same twist. +y or -x would rotate the twist datum by 90 or 180 deg —
    both were measured and rejected (seed_equivalence.py)."""
    n = seed_normal(np.array([0.0, 0.0, 0.02]), np.array([0.0, 0.0, 0.12]), 0.127)
    assert np.allclose(n, [1.0, 0.0, 0.0], atol=1e-15)


def test_a_straight_radial_blade_is_accepted():
    """The previous radial seed REFUSED this — the canonical propeller — as
    malformed, which is why the seed was moved onto the rotation axis."""
    LoopDesign(radius_m=0.127, hub_radius_m=0.02, n_blades=2, rpm=4590.0,
               topology="open",
               spine_ctrl_m=tuple((0.0, 0.0, z) for z in
                                  np.linspace(0.02, 0.127, 4)),
               station_s=(0.0, 0.5, 1.0), chord_m=(0.02, 0.02, 0.02),
               twist_rad=(0.3, 0.3, 0.3))


def test_ill_conditioned_seed_is_a_solver_error_not_a_value_error():
    """The contract's ValueErrors sit strictly INSIDE this band, so a genome
    that lands near degeneracy is penalized rather than aborting the run."""
    p0 = np.array([0.02, 0.0, 0.0])
    # tangent within the measured floor of the ROTATION AXIS
    m = 0.5 * SEED_M_MIN
    p1 = p0 + 0.3 * 0.127 * np.array([m, np.sqrt(1 - m * m), 0.0])
    with pytest.raises(SolverError, match="within .* of the ROTATION"):
        seed_normal(p0, p1, 0.127)
    # tangent too short
    p2 = p0 + 0.5 * SEED_D_MIN * 0.127 * np.array([1.0, 0.0, 0.0])
    with pytest.raises(SolverError, match=r"\|P1-P0\|"):
        seed_normal(p0, p2, 0.127)


def test_seed_band_is_wider_than_the_contract_guard():
    """If the SolverError band did not strictly contain the ValueError point,
    a genome could still reach the contract guard and abort a run."""
    assert SEED_M_MIN > 0.0 and SEED_D_MIN > 0.0


# ------------------------------------------------------------- frames end-to-end


def test_spine_frames_are_arc_length_uniform():
    fr = spine_frames(_design(), n_samples=300)
    d = np.linalg.norm(np.diff(fr.points_m, axis=0), axis=1)
    assert d.std() / d.mean() < 0.02
    assert fr.length_m > 0.0
    assert np.allclose(np.linalg.norm(fr.tangent, axis=1), 1.0, atol=1e-12)
    assert np.allclose(np.linalg.norm(fr.normal, axis=1), 1.0, atol=1e-12)
    assert np.max(np.abs(np.einsum("ij,ij->i", fr.tangent, fr.normal))) < 1e-10
