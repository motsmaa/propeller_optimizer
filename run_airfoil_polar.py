"""2-D section discriminator: fully-turbulent RANS vs XFOIL-with-transition.

Mesh ONCE, then sweep (Re, alpha) by rotating the freestream. Cl/Cd come from
the integrated surface force rotated into wind axes, per unit span.

Usage:  python run_airfoil_polar.py [--validate] [--reuse-mesh]
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from propopt.contracts import SolverError
from propopt.hf.airfoil_case import (AirfoilCaseConfig, build_airfoil_mesh_case,
                                     fix_boundary_types, write_airfoil_fields,
                                     write_run_control)
from propopt.hf.openfoam import (_wsl_sh, make_case, parse_checkmesh,
                                 parse_residuals, run_foam)
from propopt.hf.wsl import _quote, to_wsl_path
from propopt.solvers.polars import TabulatedPolar

VALIDATE = "--validate" in sys.argv
REUSE = "--reuse-mesh" in sys.argv
N_ITER = 3000
CFG = AirfoilCaseConfig()
# Sweep range DERIVED in alpha_range.py from the converged rotor fields (union
# of the thrust-weighted cores at both the CFD's own inflow and the
# momentum-consistent inflow at measured thrust): alpha -4.1..+7.7, Re 1.7e4..8.7e4.
# Extended past the derived +7.7 upper edge to +12: BEMT's OWN converged hover
# solution sits at alpha up to 9.4 deg, and the repriced polar must cover the
# range the BEMT fixed point visits, not only the range the CFD sits at.
ALPHAS = [-8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
RES = [2.0e4, 4.0e4, 7.0e4]
if VALIDATE:
    # Anchor point: Re 5e5, where natural transition on a 12% section sits close
    # to the leading edge, so fully-turbulent RANS and transitional XFOIL are
    # expected to agree in the linear range. See the caveat in the report: this
    # is a CODE-TO-CODE check, not an experimental anchor.
    #
    # BIGGER CHORD on purpose. Re 5e5 on the 0.0287 m rotor chord needs
    # U = 254 m/s (M ~ 0.75). The incompressible-vs-incompressible comparison
    # would still be formally consistent, but validating at a physically absurd
    # operating point is not worth it: a 0.20 m chord reaches Re 5e5 at 36.5 m/s.
    # surface_cell_m is rescaled with it to keep y+ in the same band.
    ALPHAS, RES = [-4, -2, 0, 2, 4, 6, 8], [5.0e5]
    CFG = AirfoilCaseConfig(chord_m=0.20, surface_cell_m=2.6e-4)

TAG = "airfoil2d_validate" if VALIDATE else "airfoil2d"
SRC = Path("runs/" + TAG)
OUT = open("runs/%s.txt" % TAG, "w")


def say(s=""):
    print(s, flush=True)
    OUT.write(str(s) + "\n")
    OUT.flush()


case = make_case(TAG, SRC)
if not REUSE:
    if SRC.exists():
        shutil.rmtree(SRC)
    geo = build_airfoil_mesh_case(case, CFG)
    write_run_control(SRC, CFG, N_ITER)
    case.sync_in()
    rm = run_foam("cartesian2DMesh", case, (), timeout=7200, measure_rss=False)
    say("cartesian2DMesh %.1fs  chord=%.4f m  farfield=%.3f m  poly pts=%d"
        % (rm.wall_s, geo["chord_m"], geo["far_field_m"], geo["poly_points"]))
    fix_boundary_types(case)
    rc = run_foam("checkMesh", case, (), timeout=1800, measure_rss=False)
    q = parse_checkmesh(rc.log_text)
    say("CELLS=%s mesh_ok=%s maxSkew=%s maxNonOrtho=%s"
        % (q.get("cells"), q.get("mesh_ok"), q.get("max_skewness"),
           q.get("max_non_orthogonality")))
    if "not 1D or 2D" in rc.log_text:
        raise SolverError("mesh is not recognised as 2-D after retyping "
                          "patches; the empty-direction check failed")
else:
    write_run_control(SRC, CFG, N_ITER)
    _wsl_sh("cp " + _quote(to_wsl_path(SRC / "system" / "controlDict")) + " "
            + _quote(case.work) + "/system/controlDict")

POLAR = TabulatedPolar.from_xfoil_files(
    sorted(Path("data/polars").glob("naca4412_re*.txt")))
rows = []
say("")
say("   Re      alpha    Cl_RANS   Cd_RANS   Cl_XFOIL  Cd_XFOIL   dCl     "
    "y+avg/max   Cl p-p  iters")

for re in RES:
    u = re * CFG.nu / CFG.chord_m
    for al in ALPHAS:
        write_airfoil_fields(SRC, CFG, u, float(al))
        _wsl_sh("cd " + _quote(case.work) + " && rm -rf [1-9]* processor* "
                "postProcessing && mkdir -p 0")
        for f in ("U", "p", "k", "omega", "epsilon", "nuTilda", "nut"):
            _wsl_sh("cp " + _quote(to_wsl_path(SRC / "0" / f)) + " "
                    + _quote(case.work) + "/0/" + f)
        r = run_foam("simpleFoam", case, (), timeout=7200, measure_rss=False,
                     log_name="log.a%+05.1f_re%.0f" % (al, re))
        res = parse_residuals(r.log_text)
        nit = len(res.get("Ux", []))

        # force: total (pressure + viscous) on the airfoil patch, per unit span
        txt = _wsl_sh("cat " + _quote(case.work)
                      + "/postProcessing/forces/0/force.dat").stdout
        dat = [l for l in txt.splitlines() if l.strip()
               and not l.startswith("#")]
        if not dat:
            raise SolverError("no force.dat rows for alpha=%s Re=%s" % (al, re))
        arr = np.array([[float(x) for x in
                         l.replace("(", " ").replace(")", " ").split()]
                        for l in dat])
        dz = CFG.dz_frac * CFG.chord_m
        qc = 0.5 * CFG.rho * u ** 2 * CFG.chord_m * dz
        a = np.radians(al)
        clh = (-arr[:, 1] * np.sin(a) + arr[:, 2] * np.cos(a)) / qc
        cl, cd = float(clh[-1]), float((arr[-1, 1] * np.cos(a)
                                        + arr[-1, 2] * np.sin(a)) / qc)
        # Force stationarity over the last 200 iterations is the real
        # convergence measure, exactly as in the rotor driver — residualControl
        # on k stalls long after the loads have settled.
        # ABSOLUTE peak-to-peak in Cl units, not relative: near zero lift a
        # relative measure divides by ~0 and reports a meaningless percentage.
        w = clh[-200:] if clh.size >= 200 else clh
        ppcl = float(w.max() - w.min())

        # The y+ function object prints "patch airfoil y+ : min = ..., max =
        # ..., average = ...". Matching on "yPlus" MISSES it — that exact grep
        # bug already cost a false "extraction gap" report on the rotor runs.
        yp = [l for l in r.log_text.splitlines() if "y+ :" in l]
        if not yp:
            raise SolverError("no y+ output for alpha=%s Re=%s; refusing to "
                              "report an unverified wall treatment" % (al, re))
        ypmax = float(yp[-1].split("max = ")[1].split(",")[0])
        ypav = float(yp[-1].split("average = ")[1].split()[0])

        clx, cdx = POLAR(np.radians(np.array([float(al)])), np.array([re]))
        rows.append(dict(re=re, alpha=float(al), cl=cl, cd=cd,
                         cl_xfoil=float(clx[0]), cd_xfoil=float(cdx[0]),
                         yplus_max=ypmax, yplus_avg=ypav, iters=nit,
                         cl_pp=ppcl, converged=nit < N_ITER))
        say("%8.0f %7.1f %10.4f %9.4f %10.4f %9.4f %+8.4f %5.1f/%-5.1f %8.5f"
            " %6d%s" % (re, al, cl, cd, clx[0], cdx[0], cl - clx[0], ypav,
                        ypmax, ppcl, nit, "" if nit < N_ITER else " *"))

json.dump(rows, open("runs/%s.json" % TAG, "w"), indent=1)
say("")
say("* = hit the iteration cap without meeting residualControl")
OUT.close()
