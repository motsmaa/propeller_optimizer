"""Parsers for UIUC Propeller Database files + design construction.

Handles the Volume-1 plain-text formats (see data/uiuc/SOURCE.md):
  geometry:  columns  r/R  c/R  beta      (beta in degrees, measured from the
                                           rotation plane — same convention as
                                           PropellerDesign.twist_rad)
  static:    columns  RPM  CT  CP         (CT = T/(rho n^2 D^4),
                                           CP = P/(rho n^3 D^5), n in rev/s,
                                           D = diameter)

This module is for validation only: it builds a PropellerDesign from measured
geometry so the BEMT solver can be compared against measured static CT/CP.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..contracts import EvalResult, PropellerDesign
from ..stations import uniform_midpoint


def _parse_table(path: Path, n_cols: int) -> np.ndarray:
    """Whitespace-separated numeric rows; header/comment lines are skipped."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < n_cols:
            continue
        try:
            rows.append([float(p) for p in parts[:n_cols]])
        except ValueError:
            continue  # header line such as "r/R c/R beta"
    if not rows:
        raise ValueError(f"no numeric rows parsed from {path}")
    return np.asarray(rows, dtype=float)


@dataclass(frozen=True)
class UiucGeometry:
    r_over_r: np.ndarray   # radial stations, fraction of tip radius, increasing
    c_over_r: np.ndarray   # chord / tip radius
    beta_deg: np.ndarray   # blade angle from rotation plane, degrees


def load_geometry(path: Path) -> UiucGeometry:
    tbl = _parse_table(path, 3)
    r, c, b = tbl[:, 0], tbl[:, 1], tbl[:, 2]
    if not (np.all(np.diff(r) > 0) and 0.0 < r[0] and r[-1] <= 1.0 + 1e-9):
        raise ValueError(f"geometry r/R must be increasing within (0, 1]: {path}")
    if np.any(c <= 0) or np.any(b <= 0) or np.any(b >= 90):
        raise ValueError(f"geometry c/R or beta out of range: {path}")
    return UiucGeometry(r, c, b)


def load_static(path: Path) -> np.ndarray:
    """Return rows of (rpm, CT_meas, CP_meas)."""
    tbl = _parse_table(path, 3)
    if np.any(tbl <= 0):
        raise ValueError(f"static data must be positive (RPM, CT, CP): {path}")
    return tbl


def design_from_geometry(
    geom: UiucGeometry,
    diameter_m: float,
    n_blades: int,
    rpm: float,
    n_stations: int = 20,
) -> PropellerDesign:
    """Build a PropellerDesign by resampling UIUC geometry onto solver stations.

    The hub radius is taken as the innermost measured station (the data says
    nothing inboard of it). Chord and twist are linearly interpolated
    (numpy.interp) onto uniform-midpoint stations (stations.uniform_midpoint,
    carried on the design as station_r_m), which all lie strictly inside the
    measured [r/R_min, 1] range — no extrapolation occurs.
    """
    radius = diameter_m / 2.0
    hub = float(geom.r_over_r[0]) * radius
    stations = uniform_midpoint(hub, radius, n_stations)
    r_norm = np.asarray(stations) / radius  # as r/R
    chord = np.interp(r_norm, geom.r_over_r, geom.c_over_r) * radius
    twist = np.radians(np.interp(r_norm, geom.r_over_r, geom.beta_deg))
    return PropellerDesign(radius, hub, n_blades, rpm,
                           tuple(float(c) for c in chord),
                           tuple(float(t) for t in twist),
                           stations)


def thrust_power_coefficients(ev: EvalResult, rpm: float, diameter_m: float,
                              rho: float) -> tuple[float, float]:
    """UIUC/propeller convention: CT = T/(rho n^2 D^4), CP = P/(rho n^3 D^5),
    n in rev/s. Note rho cancels against the rho used inside BEMT, so the
    coefficient comparison is density-independent."""
    n = rpm / 60.0
    ct = ev.thrust_N / (rho * n ** 2 * diameter_m ** 4)
    cp = ev.power_W / (rho * n ** 3 * diameter_m ** 5)
    return ct, cp
