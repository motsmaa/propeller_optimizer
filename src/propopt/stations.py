"""Radial-station placement schemes and quadrature weights.

`PropellerDesign.station_r_m` carries the actual station radii (the shared
ground truth read by aero, geometry and structure); the *schemes* that choose
them live here and are applied by problem_spec / validation when a design is
constructed. Keeping schemes out of contracts.py preserves the frozen seam.

Quadrature convention: each station owns the radial cell between its
neighbours' midpoints, clipped to [hub, tip], so the weights always sum to
exactly the full span (tip - hub). For `uniform_midpoint` stations this
reproduces the uniform dr = span/n cell width that BEMT and NullStruct used
before stations were shared (to within one ulp — the boundaries are computed
via midpoints rather than i*dr).
"""
from __future__ import annotations

import numpy as np


def uniform_midpoint(hub_radius_m: float, radius_m: float, n: int) -> tuple[float, ...]:
    """Midpoints of n uniform cells spanning [hub, tip] — the historical
    (Phase 0-2a) default. Computed with the exact expression BEMT used
    internally (hub + (i + 1/2) * dr) so the refactor is bit-preserving."""
    dr = (radius_m - hub_radius_m) / n
    return tuple(float(v) for v in hub_radius_m + (np.arange(n) + 0.5) * dr)


def cosine(hub_radius_m: float, radius_m: float, n: int) -> tuple[float, ...]:
    """Half-cosine spacing clustered toward the TIP (where loads and loss
    gradients are strongest), sampled at cell midpoints in the stretched
    coordinate so no station sits exactly on hub or tip."""
    u = (np.arange(n) + 0.5) / n
    r = hub_radius_m + (radius_m - hub_radius_m) * np.sin(0.5 * np.pi * u)
    return tuple(float(v) for v in r)


def station_weights(station_r_m, hub_radius_m: float, radius_m: float) -> np.ndarray:
    """Radial cell width owned by each station (m): cell boundaries are the
    midpoints between neighbouring stations, clipped to [hub, tip]. Sums to
    the full span. Positive whenever the design contract holds (strictly
    increasing stations within [hub, tip])."""
    r = np.asarray(station_r_m, dtype=float)
    bounds = np.empty(len(r) + 1)
    bounds[0] = hub_radius_m
    bounds[-1] = radius_m
    bounds[1:-1] = 0.5 * (r[:-1] + r[1:])
    w = np.diff(bounds)
    if np.any(w <= 0.0):
        raise ValueError("degenerate station spacing: non-positive cell width")
    return w
