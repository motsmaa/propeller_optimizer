"""Price the 2-D RANS section polar through BEMT, and split the rotor gap.

The 2-D study on its own only says the section models disagree. This makes the
disagreement commensurate with the rotor result: feed the RANS-derived Cl/Cd
into BEMT in place of the XFOIL polars, changing NOTHING else, and read the CT.

The decomposition this produces, at the hover anchor (4590 rpm):

  measured                     CT 0.1200   the target
  BEMT + XFOIL (transitional)  CT 0.0996   the tracked -17% reference
  BEMT + RANS  (fully turb.)   CT   ???    <- this run
  rotor CFD (3-D, MRF)         CT 0.0530   what the rotor case actually gave

  BEMT_RANS / BEMT_XFOIL  = the part attributable to TRANSITION MODELLING
                            alone (same section, same solver, same stations)
  BEMT_XFOIL / measured   = the pre-existing Phase-1 residual (section SHAPE
                            proxy + 3-D + everything else)
  rotor CFD / BEMT_RANS   = what is left over for rotor-level 3-D/CFD effects
                            once the section modelling is accounted for
"""
import json
from pathlib import Path

import numpy as np

from propopt.contracts import SolverError
from propopt.solvers.bemt import BEMTAeroSolver
from propopt.solvers.polars import TabulatedPolar
from propopt.validation.uiuc import (design_from_geometry, load_geometry,
                                     thrust_power_coefficients)

RHO, D, NB, RPM = 1.225, 0.254, 2, 4590.0
CT_MEAS, CP_MEAS = 0.1200, 0.0484
CT_ROTOR, CP_ROTOR = 0.0530, 0.0314
OUT = open("runs/price_rans.txt", "w")


def say(s=""):
    print(s, flush=True)
    OUT.write(str(s) + "\n")
    OUT.flush()


rows = json.load(open("runs/airfoil2d.json"))
# Convergence is judged on FORCE STATIONARITY, not on residualControl. Every
# point hits the iteration cap because k stalls near 1e-4 long after the loads
# have settled; that is not the same as an unconverged load. Flag only points
# whose Cl is still moving by more than 0.01 over the last 200 iterations.
PP_TOL = 0.01
bad = [r for r in rows if r["cl_pp"] > PP_TOL]
say("Force stationarity: %d of %d points have Cl peak-to-peak > %.2f over the "
    "last 200 iterations" % (len(bad), len(rows), PP_TOL))
for b in bad:
    say("  UNSETTLED Re=%.0f alpha=%+.0f  Cl=%.4f  p-p=%.4f"
        % (b["re"], b["alpha"], b["cl"], b["cl_pp"]))
if bad:
    say("  (all are at |alpha| >= 6, i.e. post-stall or deep negative — the "
        "operating core -4..+8 is settled to <0.01)")

# Build a polar with the SAME interpolation/extrapolation policy the XFOIL
# tables get (TabulatedPolar: linear in alpha, log-Re blend between levels,
# flat-plate blend outside the measured alpha range, Re clamped at the edges).
# Using the same class is deliberate — otherwise part of any difference would
# be a difference in how the two tables are read, not in the tables.
tables = []
for re in sorted({r["re"] for r in rows}):
    sub = sorted([r for r in rows if r["re"] == re], key=lambda r: r["alpha"])
    a = np.array([s["alpha"] for s in sub])
    cl = np.array([s["cl"] for s in sub])
    cd = np.array([s["cd"] for s in sub])
    if np.any(cd <= 0):
        raise SolverError("non-physical Cd <= 0 in the RANS polar at Re=%g" % re)
    tables.append((re, a, cl, cd))
    say("RANS table Re=%.0f: %d alphas %.1f..%.1f, Cl %.3f..%.3f, Cd %.4f..%.4f"
        % (re, len(a), a[0], a[-1], cl.min(), cl.max(), cd.min(), cd.max()))
rans = TabulatedPolar(tables)

xf_all = sorted(Path("data/polars").glob("naca4412_re*.txt"))
xfoil_tracked = TabulatedPolar.from_xfoil_files(
    [p for p in xf_all if int(p.stem.split("re")[1]) >= 50_000])
xfoil_ext = TabulatedPolar.from_xfoil_files(xf_all)
# CONTROLLED XFOIL: resample the XFOIL polar onto the RANS sweep's OWN (Re,
# alpha) grid and rebuild a table from it. There are no XFOIL tables at exactly
# 2e4/4e4/7e4, so a "matched file" set is impossible; resampling instead makes
# the two polars identical in Re levels, alpha grid and interpolation policy, so
# the ONLY thing that differs is the data values.
_re_lvls = sorted({r["re"] for r in rows})
_al = np.array(sorted({r["alpha"] for r in rows}))
xfoil_matched = TabulatedPolar([
    (re, _al) + xfoil_ext(np.radians(_al), np.full(_al.shape, re))
    for re in _re_lvls])

geom = load_geometry(Path("data/uiuc/apcsf_10x4.7_geom.txt"))
design = design_from_geometry(geom, D, NB, RPM, n_stations=20)


def run(polar, tag):
    try:
        ev = BEMTAeroSolver(polar=polar, rho=RHO, relax=0.2,
                            max_iter=1000).evaluate(design)
    except SolverError as exc:
        say("  %-34s DID NOT CONVERGE: %s" % (tag, exc))
        return None, None
    ct, cp = thrust_power_coefficients(ev, RPM, D, RHO)
    say("  %-34s CT=%.4f  CP=%.4f  T=%.4f N  FoM=%.3f"
        % (tag, ct, cp, ev.thrust_N, ev.figure_of_merit))
    return ct, cp


say("")
say("BEMT AT THE HOVER ANCHOR (4590 rpm), polar swapped, nothing else changed")
ct_x, cp_x = run(xfoil_tracked, "BEMT + XFOIL (tracked, Re>=5e4)")
ct_xm, cp_xm = run(xfoil_matched, "BEMT + XFOIL (Re matched to RANS)")
ct_xe, _ = run(xfoil_ext, "BEMT + XFOIL (extended set)")
ct_r, cp_r = run(rans, "BEMT + RANS (fully turbulent)")
say("  %-34s CT=%.4f  CP=%.4f" % ("rotor CFD (3-D MRF, good5)", CT_ROTOR, CP_ROTOR))
say("  %-34s CT=%.4f  CP=%.4f" % ("MEASURED (UIUC @ 4590 rpm)", CT_MEAS, CP_MEAS))

if ct_r is None:
    say("")
    say("The RANS polar does not admit a converged BEMT fixed point. That is "
        "itself a result - report it as such, do NOT relax the solver to force "
        "a number.")
else:
    say("")
    say("DECOMPOSITION OF THE ROTOR GAP")
    say("  transition modelling    BEMT_RANS / BEMT_XFOIL  = %.3f  (%+.1f%%)"
        % (ct_r / ct_xm, 100 * (ct_r / ct_xm - 1)))
    say("  section shape + 3-D     BEMT_XFOIL / measured   = %.3f  (%+.1f%%)"
        % (ct_x / CT_MEAS, 100 * (ct_x / CT_MEAS - 1)))
    say("  rotor-level residual    rotorCFD / BEMT_RANS    = %.3f  (%+.1f%%)"
        % (CT_ROTOR / ct_r, 100 * (CT_ROTOR / ct_r - 1)))
    say("  total                   rotorCFD / measured     = %.3f  (%+.1f%%)"
        % (CT_ROTOR / CT_MEAS, 100 * (CT_ROTOR / CT_MEAS - 1)))
    say("")
    gap_tot = np.log(CT_MEAS / CT_ROTOR)
    say("  share of the total log-gap explained by each step:")
    say("    transition (RANS vs XFOIL section model) : %5.1f%%"
        % (100 * np.log(ct_xm / ct_r) / gap_tot))
    say("    section shape + 3-D (XFOIL vs measured)  : %5.1f%%"
        % (100 * np.log(CT_MEAS / ct_x) / gap_tot))
    say("    unexplained rotor-level residual         : %5.1f%%"
        % (100 * np.log(ct_r / CT_ROTOR) / gap_tot))

json.dump({"ct_xfoil_tracked": ct_x, "ct_xfoil_matched": ct_xm,
           "ct_xfoil_extended": ct_xe, "ct_rans": ct_r,
           "cp_rans": cp_r, "ct_rotor": CT_ROTOR, "ct_meas": CT_MEAS},
          open("runs/price_rans.json", "w"), indent=1)
OUT.close()
