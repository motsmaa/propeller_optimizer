"""TWIST-CONVENTION EQUIVALENCE: which RMF seed reproduces the radial path?

A straight radial spine hub->tip at fixed `twist_rad` is the one configuration
where the swept representation and the radial representation must describe the
SAME blade. Whichever seed makes them coincide is the seed that makes
`twist_rad` mean the same thing in both, and the others differ from it by a
constant rotation about the tangent.

The offset is MEASURED, not derived, and a non-zero result is reported rather
than absorbed into the placement as a correction constant: a constant offset
hidden in the sweep would silently redefine twist for every loop design.

Three candidates, all with the identical degeneracy condition |y_hat x t0|:
  (a) n0 = normalise(y_hat - (y_hat.t0) t0)      the axis projected off t0
  (b) n0 = normalise(t0 x y_hat)
  (c) n0 = normalise(y_hat x t0)                 = -(b)
"""
from __future__ import annotations

import numpy as np

from propopt.contracts import LoopDesign, PropellerDesign
from propopt.geometry import (DEFAULT_N_CHORD, DEFAULT_SECTION,
                              PITCH_AXIS_CHORD_FRAC, section_polygon)
from propopt.hf.hover_case import (HoverCaseConfig, assert_leading_edge_first,
                                   assert_positive_incidence)
from propopt.hf.occ_geometry import station_rings
from propopt.loop_geometry import bspline_eval, rmf_frames

R, HUB, NB, RPM = 0.127, 0.02, 2, 4590.0
N_ST, N_CHORD, TE = 8, DEFAULT_N_CHORD, 2.0e-4
CHORD, TWIST = 0.020, 0.30            # fixed along the span, so any offset is
                                      # the SEED's, not a spanwise variation


def seed(variant: str, t0: np.ndarray) -> np.ndarray:
    y = np.array([0.0, 1.0, 0.0])
    if variant == "a":
        n = y - float(y @ t0) * t0
    elif variant == "b":
        n = np.cross(t0, y)
    elif variant == "c":
        n = np.cross(y, t0)
    else:
        raise ValueError(variant)
    return n / np.linalg.norm(n)


# ---- radial reference: the validated path, at the same chord/twist ----------
r_i = np.linspace(HUB, R, N_ST)
design_r = PropellerDesign(R, HUB, NB, RPM,
                           tuple([CHORD] * N_ST), tuple([TWIST] * N_ST),
                           tuple(float(v) for v in r_i))
ref = station_rings(design_r, DEFAULT_SECTION, N_CHORD, TE)

cfg = HoverCaseConfig(rpm=RPM)
assert_leading_edge_first(design_r, cfg)
assert_positive_incidence(design_r, cfg)
print("radial reference passes BOTH guards: leading-edge-first and "
      "positive-signed-incidence")
print("  (so any seed that reproduces it inherits both)")

# ---- swept: a straight radial spine, same stations -------------------------
ctrl = np.column_stack([np.zeros(4), np.zeros(4), np.linspace(HUB, R, 4)])
design_l = LoopDesign(
    radius_m=R, hub_radius_m=HUB, n_blades=NB, rpm=RPM, topology="open",
    spine_ctrl_m=tuple(tuple(float(v) for v in p) for p in ctrl),
    station_s=tuple(float(v) for v in np.linspace(0.0, 1.0, N_ST)),
    chord_m=tuple([CHORD] * N_ST), twist_rad=tuple([TWIST] * N_ST))

s = np.linspace(0.0, 1.0, N_ST)
dense = bspline_eval(ctrl, np.linspace(0, 1, 400))
d = np.linalg.norm(np.diff(dense, axis=0), axis=1)
cum = np.concatenate([[0.0], np.cumsum(d)])
pts_dense = np.column_stack([np.interp(s * cum[-1], cum, dense[:, k])
                             for k in range(3)])

poly = section_polygon(DEFAULT_SECTION, N_CHORD, TE / CHORD)
xi = (poly[:, 0] - PITCH_AXIS_CHORD_FRAC) * CHORD
eta = poly[:, 1] * CHORD
k = N_CHORD

print()
print("  variant   seed n0 at t0=+z        offset about the tangent   max |dx|")
print("  " + "-" * 68)
results = {}
for v in ("a", "b", "c"):
    t_all, n_all = rmf_frames(pts_dense, seed(v, np.array([0.0, 0.0, 1.0])), R)
    B = np.cross(t_all, n_all)
    c_t, s_t = np.cos(-TWIST), np.sin(-TWIST)
    a_c = xi * c_t - eta * s_t
    b_c = xi * s_t + eta * c_t
    got = (pts_dense[:, None, :] + a_c[None, :, None] * n_all[:, None, :]
           + b_c[None, :, None] * B[:, None, :])
    # signed angle about the tangent (+z) between reference and swept chord
    def chord_vec(ring):
        return 0.5 * (ring[:, k - 1, :] + ring[:, k, :]) - ring[:, 0, :]
    u, w = chord_vec(ref)[:, :2], chord_vec(got)[:, :2]
    ang = np.degrees(np.arctan2(u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0],
                                u[:, 0] * w[:, 0] + u[:, 1] * w[:, 1]))
    off = float(np.mean(ang))
    dev = float(np.max(np.abs(ang - off)))
    dx = float(np.max(np.abs(got - ref)))
    results[v] = (off, dx)
    print("  (%s)       %-22s %+9.4f deg (spread %.2e)  %.3e m"
          % (v, np.round(seed(v, np.array([0.0, 0.0, 1.0])), 6), off, dev, dx))

print()
zero = [v for v, (off, dx) in results.items() if abs(off) < 1e-9]
print("ZERO-OFFSET variants: %s" % (zero or "NONE"))
for v, (off, dx) in results.items():
    print("  (%s) offset %+8.3f deg, max vertex deviation %.3e m" % (v, off, dx))
if not zero:
    print("\nSTOP: no variant reproduces the radial convention exactly. Report "
          "rather than absorbing the offset as a constant.")
else:
    v = zero[0]
    print("\nCHOSEN: variant (%s). Max vertex deviation from the radial path "
          "%.3e m" % (v, results[v][1]))
