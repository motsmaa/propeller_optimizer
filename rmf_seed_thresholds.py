"""Measure the RMF seed conditioning, to CHOOSE the decode thresholds.

CONVENTION UNDER TEST (chosen by measurement in seed_equivalence.py, which
found it the only one of three candidates that reproduces the radial path's
twist datum exactly - the others sat at +90 and +180 degrees):

    n0 = normalise(y_hat x t0) = (tz, 0, -tx) / hypot(tx, tz)

The seed fixes the datum every section's `twist_rad` is measured from, so if
its DIRECTION moves under a small change in the control points, the whole
blade's twist is that uncertain. Two quantities can make it move:

  A  m = |y_hat x t0| = hypot(tx, tz) = sin(angle between t0 and the ROTATION
     AXIS). As the start tangent approaches the axis, the projection of the
     axis into the rotor plane shrinks and its DIRECTION becomes arbitrary.
  B  d = |P1 - P0| / radius_m. The start tangent is along P1 - P0, so as the
     first two control points converge the tangent direction becomes arbitrary.

Note this seed does NOT depend on P0, only on the start tangent direction. The
previous radial seed did, through the local radial direction, which gave the
perturbation a second route in.

Method: build a seed at a target (m, d), perturb BOTH control points by random
vectors of magnitude eps*radius_m, recompute the seed normal, and record the
angle between the perturbed and unperturbed normals. 4000 trials per cell;
report the 95th percentile.

CRITERION, PRE-DECLARED (the ORIGINAL one, restored): p95 seed-normal wobble
<= 1.0 degree at eps = 1e-3 * radius_m. Under the previous radial seed this
criterion was UNSATISFIABLE - even a well-conditioned seed wobbled 1.52 deg -
and had to be re-derived at eps = 1e-4. It is re-declared unmodified here to
test whether removing the P0 dependence recovers it.
"""
from __future__ import annotations

import numpy as np

R = 0.127
RNG = np.random.default_rng(20260909)
N_TRIAL = 4000
EPS = (1e-6, 1e-4, 1e-3)
CRIT_DEG = 1.0
CRIT_EPS = 1e-3


def seed_normal(p0, p1):
    """The chosen seed: y_hat x t0, normalised."""
    d = p1 - p0
    t = d / np.linalg.norm(d)
    n = np.array([t[2], 0.0, -t[0]])
    return n / np.linalg.norm(n)


def build(m, d_frac, rng):
    """P0 anywhere sane; P1 placed so the tangent makes angle asin(m) with the
    ROTATION AXIS, with the in-plane azimuth random so no special orientation
    is baked in."""
    th = rng.uniform(0, 2 * np.pi)
    p0 = np.array([0.02 * np.cos(th), rng.uniform(-0.02, 0.02),
                   0.02 * np.sin(th)])
    phi = rng.uniform(0, 2 * np.pi)
    # tangent at angle asin(m) from +y: axial component sqrt(1-m^2), in-plane m
    t0 = np.array([m * np.cos(phi), np.sqrt(max(1.0 - m * m, 0.0)),
                   m * np.sin(phi)])
    return p0, p0 + d_frac * R * t0


def wobble(m, d_frac, eps, rng):
    out = np.empty(N_TRIAL)
    for i in range(N_TRIAL):
        p0, p1 = build(m, d_frac, rng)
        n_ref = seed_normal(p0, p1)
        dp = rng.normal(size=(2, 3))
        dp *= eps * R / np.linalg.norm(dp, axis=1)[:, None]
        n_per = seed_normal(p0 + dp[0], p1 + dp[1])
        out[i] = np.degrees(np.arccos(np.clip(np.dot(n_ref, n_per), -1, 1)))
    return float(np.percentile(out, 95))


print("CONDITION A - m = hypot(tx, tz) = sin(angle from the ROTATION AXIS),")
print("              at a typical d = 0.30 R")
print("  m        " + "".join("  p95 wobble @ eps=%-7g" % e for e in EPS))
mA = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 5e-3, 2e-3, 1e-3, 1e-4]
rowsA = []
for m in mA:
    w = [wobble(m, 0.30, e, RNG) for e in EPS]
    rowsA.append((m, w))
    print("  %-8.4g " % m + "".join("  %20.3f" % x for x in w))

print()
print("CONDITION B - d = |P1-P0| / radius_m, at a safe m = 0.5")
print("  d/R      " + "".join("  p95 wobble @ eps=%-7g" % e for e in EPS))
dB = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 5e-3, 2e-3, 1e-3]
rowsB = []
for d in dB:
    w = [wobble(0.5, d, e, RNG) for e in EPS]
    rowsB.append((d, w))
    print("  %-8.4g " % d + "".join("  %20.3f" % x for x in w))

j = EPS.index(CRIT_EPS)
okA = [m for m, w in rowsA if w[j] <= CRIT_DEG]
okB = [d for d, w in rowsB if w[j] <= CRIT_DEG]
print()
print("PRE-DECLARED criterion: p95 wobble <= %.1f deg at eps = %g * radius_m"
      % (CRIT_DEG, CRIT_EPS))
if not okA or not okB:
    print("  RESULT: STILL UNSATISFIABLE on condition %s"
          % ("A" if not okA else "B"))
    print("  best cell A = %.3f deg at m = %.4g" % (rowsA[0][1][j], mA[0]))
else:
    print("  RESULT: SATISFIED UNMODIFIED (it was not, under the radial seed)")
    print("  condition A: satisfied down to m   = %.4g (%.3f deg); next cell "
          "%.4g fails (%.3f deg)"
          % (min(okA), dict(rowsA)[min(okA)][j],
             mA[mA.index(min(okA)) + 1] if mA.index(min(okA)) + 1 < len(mA) else float("nan"),
             dict(rowsA)[mA[mA.index(min(okA)) + 1]][j] if mA.index(min(okA)) + 1 < len(mA) else float("nan")))
    print("  condition B: satisfied down to d/R = %.4g (%.3f deg); next cell "
          "%.4g fails (%.3f deg)"
          % (min(okB), dict(rowsB)[min(okB)][j],
             dB[dB.index(min(okB)) + 1] if dB.index(min(okB)) + 1 < len(dB) else float("nan"),
             dict(rowsB)[dB[dB.index(min(okB)) + 1]][j] if dB.index(min(okB)) + 1 < len(dB) else float("nan")))
    print()
    print("CHOSEN THRESHOLDS: SEED_M_MIN = %.4g, SEED_D_MIN = %.4g"
          % (min(okA), min(okB)))
    print("Both are MEASURED-passing cells, not extrapolations.")
