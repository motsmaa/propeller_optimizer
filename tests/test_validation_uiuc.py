"""Validation anchor: BEMT vs measured UIUC static data (APC Slow Flyer 10x4.7).

This test MEASURES model error against a real propeller. It must never be made
to pass by tuning polar constants, loss factors, or thresholds to a code
change — that would overfit one geometry and destroy the anchor's value.
The gating thresholds below were set at Phase-1 closeout (README "Phase 1
(closed)") to the *diagnosed data-fidelity limit* of the proxy-section polars;
widening them requires a new diagnosis, not a red CI. Data files + provenance:
data/uiuc/ and data/polars/ (SOURCE.md in each).

Parametrized over three polars through the BEMT polar seam (solver unchanged):
  analytic      - Phase-1 Re-independent polar                     (GATING)
  tab-re5e4     - NACA 4412 XFOIL tables, Re >= 5e4 only           (GATING)
  tab-extended  - + locally generated XFOIL tables down to Re=1e4  (NON-GATING,
                  xfail: edge-of-validity documentation of why the 4412 proxy
                  fails at low Re — see XFAIL_EXTENDED_REASON)
"""
from pathlib import Path

import pytest

from propopt.solvers.bemt import BEMTAeroSolver
from propopt.solvers.polars import TabulatedPolar
from propopt.units import RHO_AIR
from propopt.validation.uiuc import (
    design_from_geometry,
    load_geometry,
    load_static,
    thrust_power_coefficients,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GEOM_FILE = DATA_DIR / "uiuc" / "apcsf_10x4.7_geom.txt"
STATIC_FILE = DATA_DIR / "uiuc" / "apcsf_10x4.7_static_kt0835.txt"
POLAR_DIR = DATA_DIR / "polars"

DIAMETER_M = 0.254  # APC Slow Flyer 10x4.7: 10 in diameter
N_BLADES = 2
N_STATIONS = 20     # resampling is converged: 20 vs 80 stations moves CT/CP ~0.1%

# Phase-1 closeout thresholds, scoped explicitly to: validation against UIUC
# apcsf_10x4.7 using a NON-MATCHED NACA 4412 proxy section (12% thick vs the
# real ~5% thin cambered APC blade) with known-optimistic Ncrit=9 XFOIL
# polars. They encode that diagnosed input-data limit — NOT solver accuracy
# (see the assert message and README "Phase 1 (closed)" for the diagnosis).
# Replaced the 15%/20% placeholders once the residual was traced to the data.
CT_MAPE_LIMIT = 20.0  # percent
CP_MAPE_LIMIT = 25.0  # percent

XFAIL_EXTENDED_REASON = (
    "Non-gating documentation variant: XFOIL at Re=1e4 on a 12%-thick section "
    "is edge-of-validity; the ~-39% low-RPM CT error is directionally "
    "meaningful (it proved the low-RPM error flatness was Reynolds-clamping, "
    "not solver error) but not quantitative. Retained to document why the "
    "4412 proxy fails at low Re, not as a correctness gate. strict=False "
    "deliberately: this number is expected to move when a matched thin "
    "section lands, and neither a failure nor an incidental pass should flip "
    "CI for a variant that is documentation, not a tracked fix."
)


def _make_solver(polar_id: str) -> BEMTAeroSolver:
    if polar_id == "analytic":
        return BEMTAeroSolver(rho=RHO_AIR)  # rho cancels in the coefficients
    files = sorted(POLAR_DIR.glob("naca4412_re*.txt")) if POLAR_DIR.exists() else []
    if polar_id == "tab-re5e4":
        files = [p for p in files if int(p.stem.split("re")[1]) >= 50_000]
    if len(files) < 2:
        pytest.skip(
            "Tabulated polar data missing (see data/polars/SOURCE.md): "
            "naca4412_re<RE>.txt tables for RE 50000-500000 from "
            "http://airfoiltools.com/polar/text?polar=xf-naca4412-il-<RE>, "
            "plus XFOIL-generated RE 10000/20000/30000 for tab-extended"
        )
    # relax/max_iter are numerical settings on the public constructor, not
    # physics: the jagged low-Re XFOIL tables make the default relaxation
    # (0.35) oscillate. The converged fixed point is relaxation-independent
    # (relax 0.2 vs 0.05 agree to ~1e-6 N on this dataset).
    return BEMTAeroSolver(polar=TabulatedPolar.from_xfoil_files(files),
                          rho=RHO_AIR, relax=0.2, max_iter=1000)


@pytest.mark.parametrize("polar_id", [
    "analytic",
    "tab-re5e4",
    pytest.param("tab-extended",
                 marks=pytest.mark.xfail(reason=XFAIL_EXTENDED_REASON,
                                         strict=False)),
])
def test_bemt_vs_uiuc_apcsf_10x47_static(polar_id):
    if not (GEOM_FILE.exists() and STATIC_FILE.exists()):
        pytest.skip(
            "UIUC validation data missing. Download into data/uiuc/:\n"
            "  https://m-selig.ae.illinois.edu/props/volume-1/data/apcsf_10x4.7_geom.txt\n"
            "  https://m-selig.ae.illinois.edu/props/volume-1/data/apcsf_10x4.7_static_kt0835.txt"
        )
    geom = load_geometry(GEOM_FILE)
    static = load_static(STATIC_FILE)
    assert len(static) >= 5, "suspiciously few static test points parsed"

    solver = _make_solver(polar_id)
    lines = [f"{'RPM':>6} {'CT_meas':>8} {'CT_bemt':>8} {'%err':>7} "
             f"{'CP_meas':>8} {'CP_bemt':>8} {'%err':>7}"]
    ct_abs_errs, cp_abs_errs = [], []
    for rpm, ct_meas, cp_meas in static:
        design = design_from_geometry(geom, DIAMETER_M, N_BLADES, rpm,
                                      n_stations=N_STATIONS)
        ev = solver.evaluate(design)
        ct_bemt, cp_bemt = thrust_power_coefficients(ev, rpm, DIAMETER_M, RHO_AIR)
        ct_err = 100.0 * (ct_bemt - ct_meas) / ct_meas
        cp_err = 100.0 * (cp_bemt - cp_meas) / cp_meas
        ct_abs_errs.append(abs(ct_err))
        cp_abs_errs.append(abs(cp_err))
        lines.append(f"{rpm:6.0f} {ct_meas:8.4f} {ct_bemt:8.4f} {ct_err:+7.1f} "
                     f"{cp_meas:8.4f} {cp_bemt:8.4f} {cp_err:+7.1f}")

    ct_mape = sum(ct_abs_errs) / len(ct_abs_errs)
    cp_mape = sum(cp_abs_errs) / len(cp_abs_errs)
    lines.append(f"CT MAPE = {ct_mape:.1f}% (limit {CT_MAPE_LIMIT}%)   "
                 f"CP MAPE = {cp_mape:.1f}% (limit {CP_MAPE_LIMIT}%)")
    table = "\n".join(lines)
    print(f"\nBEMT ({polar_id} polar) vs UIUC APC Slow Flyer 10x4.7 (static):\n"
          + table)

    assert ct_mape <= CT_MAPE_LIMIT and cp_mape <= CP_MAPE_LIMIT, (
        f"BEMT ({polar_id} polar) vs UIUC apcsf_10x4.7 exceeded the Phase-1 "
        f"validation band (CT MAPE <= {CT_MAPE_LIMIT}%, CP MAPE <= "
        f"{CP_MAPE_LIMIT}%).\n"
        "Context for whoever sees this fail: these bounds are scoped to a "
        "NON-MATCHED proxy section (12%-thick NACA 4412, XFOIL Ncrit=9, "
        "known-optimistic profile drag at low Re) and encode that diagnosed "
        "input-data-fidelity limit, not solver error. The BEMT solver itself "
        "was validated as structurally correct at Phase-1 closeout: "
        "relaxation-independent fixed point (~1e-6 N), CP/CT ratio matching "
        "measurement to ~3%, and a high-RPM CT offset stable across two "
        "independent polar sets. A failure here therefore most likely means "
        "the input polars/geometry/resampling changed or a solver regression "
        "was introduced - diagnose first; do NOT widen the band or tune "
        "constants to silence it.\n" + table
    )
