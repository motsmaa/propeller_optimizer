"""Tabulated, Reynolds-dependent airfoil polars for the BEMT polar seam.

`TabulatedPolar` implements the same `polar(alpha_rad, Re) -> (Cl, Cd)`
signature as `bemt.analytic_polar`, backed by XFOIL polar-dump tables at a
set of Reynolds numbers (see data/polars/SOURCE.md for provenance). The BEMT
solver is untouched: pass `BEMTAeroSolver(polar=TabulatedPolar.from_xfoil_files(...))`.

Interpolation / extrapolation policy:
  - alpha: linear interpolation within each Re table (numpy.interp on a
    precomputed uniform grid);
  - alpha BEYOND a table's measured range: linear blend over `BLEND_DEG`
    degrees from the table-edge value into a crude flat-plate post-stall
    model (Cl = 2 sin a cos a, Cd = 1.8 sin^2 a + 0.01). Clearly a fallback,
    not data — fine for lightly-loaded stations that stray past stall,
    wrong for deep-stall accuracy.
  - Re: linear interpolation in log(Re) between bracketing tables; Re outside
    the table bounds is CLAMPED to the nearest table (no extrapolation).
    With the shipped NACA 4412 set the lower bound is Re = 5e4, which small
    props undershoot inboard at low RPM — those stations silently get the
    5e4 polar (flagged in data/polars/SOURCE.md).
"""
from __future__ import annotations

import re as _re
from pathlib import Path

import numpy as np

BLEND_DEG = 4.0            # width of table-edge -> flat-plate blend
_GRID_DEG = np.arange(-90.0, 90.0 + 0.25, 0.25)  # common alpha grid, degrees


def _flat_plate(alpha_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Textbook flat-plate post-stall: Cl = 2 sin a cos a, Cd ~ 1.8 sin^2 a."""
    s, c = np.sin(alpha_rad), np.cos(alpha_rad)
    return 2.0 * s * c, 1.8 * s ** 2 + 0.01


def load_xfoil_polar(path: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Parse an XFOIL polar dump; return (Re, alpha_deg, Cl, Cd), alpha sorted."""
    reynolds = None
    rows = []
    for line in Path(path).read_text().splitlines():
        m = _re.search(r"Re\s*=\s*([0-9.]+)\s*e\s*([0-9]+)", line)
        if m:
            reynolds = float(m.group(1)) * 10.0 ** int(m.group(2))
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue  # header / separator line
        rows.append(vals[:3])  # alpha, CL, CD
    if reynolds is None or not rows:
        raise ValueError(f"not a recognizable XFOIL polar dump: {path}")
    tbl = np.asarray(rows, dtype=float)
    alpha, order = np.unique(tbl[:, 0], return_index=True)  # sort + dedupe
    cl, cd = tbl[order, 1], tbl[order, 2]
    if np.any(cd <= 0) or not np.all(np.isfinite(tbl)):
        raise ValueError(f"non-physical Cl/Cd values in {path}")
    return reynolds, alpha, cl, cd


class TabulatedPolar:
    """Callable polar interpolating XFOIL tables over (alpha, Re). See module
    docstring for the interpolation/extrapolation policy."""

    def __init__(self, tables: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]):
        if len(tables) < 2:
            raise ValueError("need tables at >= 2 Reynolds numbers to interpolate")
        tables = sorted(tables, key=lambda t: t[0])
        self.re_levels = np.array([t[0] for t in tables], dtype=float)
        if np.any(np.diff(self.re_levels) <= 0):
            raise ValueError("duplicate Reynolds levels")
        # Precompute Cl/Cd on a common alpha grid per Re level, with the
        # flat-plate blend applied outside each table's measured range.
        grid_rad = np.radians(_GRID_DEG)
        cl_fp, cd_fp = _flat_plate(grid_rad)
        self._cl_grid, self._cd_grid = [], []
        for _, a_deg, cl, cd in tables:
            cl_i = np.interp(_GRID_DEG, a_deg, cl)   # constant beyond edges
            cd_i = np.interp(_GRID_DEG, a_deg, cd)
            past = np.maximum(a_deg[0] - _GRID_DEG, _GRID_DEG - a_deg[-1])
            w_fp = np.clip(past / BLEND_DEG, 0.0, 1.0)  # 0 inside table -> 1 flat-plate
            self._cl_grid.append((1.0 - w_fp) * cl_i + w_fp * cl_fp)
            self._cd_grid.append((1.0 - w_fp) * cd_i + w_fp * cd_fp)
        self._cl_grid = np.asarray(self._cl_grid)
        self._cd_grid = np.asarray(self._cd_grid)
        self._log_re = np.log(self.re_levels)

    @classmethod
    def from_xfoil_files(cls, paths: list[Path]) -> "TabulatedPolar":
        return cls([load_xfoil_polar(p) for p in paths])

    def __call__(self, alpha: np.ndarray, reynolds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        alpha_deg = np.degrees(np.asarray(alpha, dtype=float))
        alpha_deg = np.clip(alpha_deg, _GRID_DEG[0], _GRID_DEG[-1])
        re_c = np.clip(np.asarray(reynolds, dtype=float), self.re_levels[0], self.re_levels[-1])
        re_c = np.broadcast_to(re_c, alpha_deg.shape)

        # per-level alpha interpolation, then log-Re blend between brackets
        cl_lv = np.stack([np.interp(alpha_deg, _GRID_DEG, g) for g in self._cl_grid])
        cd_lv = np.stack([np.interp(alpha_deg, _GRID_DEG, g) for g in self._cd_grid])
        hi = np.clip(np.searchsorted(self.re_levels, re_c), 1, len(self.re_levels) - 1)
        lo = hi - 1
        w = (np.log(re_c) - self._log_re[lo]) / (self._log_re[hi] - self._log_re[lo])
        w = np.clip(w, 0.0, 1.0)
        idx = np.arange(alpha_deg.size).reshape(alpha_deg.shape)
        cl = (1.0 - w) * cl_lv[lo, idx] + w * cl_lv[hi, idx]
        cd = (1.0 - w) * cd_lv[lo, idx] + w * cd_lv[hi, idx]
        return cl, cd


def naca4412_polar(data_dir: Path) -> TabulatedPolar:
    """Load the shipped NACA 4412 XFOIL set from data/polars/ (see SOURCE.md)."""
    paths = sorted(Path(data_dir).glob("naca4412_re*.txt"))
    if not paths:
        raise FileNotFoundError(f"no naca4412_re*.txt tables in {data_dir}")
    return TabulatedPolar.from_xfoil_files(paths)
