"""Mesh sensitivity for the 2-D section sweep.

WHY. The Re=5e5 validation ran on a 0.20 m chord with surface cells at 0.13% of
chord (86,741 cells). The production sweep runs on the 0.0287 m rotor chord with
surface cells at 1.57% of chord (5,968 cells) — deliberately, because that is
the ROTOR's own surface resolution (0.6-1.2% of chord) and the point is to test
the rotor's modelling. But it means the validation does not validate the sweep
mesh, and a Cl conclusion drawn on 6k cells needs its own convergence evidence.

CONFOUND, STATED UP FRONT. With no boundary layers the wall cell is isotropic,
so refining the surface cell changes the chordwise resolution AND the wall-normal
spacing (hence y+) together. A Cl shift therefore cannot be attributed to one of
them alone. This is the same class of confound as the rotor's
rotor_region_level test, where refining the wake also refined the wall. It is
reported, not hidden: what this check CAN establish is whether the sweep's Cl is
stable under refinement at all.
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

N_ITER = 3000
# --nut=nutUSpaldingWallFunction re-runs the refinement with a wall treatment
# that is VALID from the viscous sublayer through the log layer. Needed because
# refining the isotropic surface cell drives y+ below 10, where the default
# nutkWallFunction (which assumes the first cell sits in the LOG layer) is
# outside its validity and returns nonsense rather than a refined answer.
NUT = "nutkWallFunction"
for _a in sys.argv[1:]:
    if _a.startswith("--nut="):
        NUT = _a.split("=", 1)[1]
CELLS = [4.5e-4, 2.25e-4, 1.1e-4]      # rotor-like -> 4x finer
if "--fine-only" in sys.argv:
    CELLS = [1.1e-4]
ALPHAS = [-2.0, 0.0, 4.0]
RES = [2.0e4, 7.0e4]
SUF = "" if NUT == "nutkWallFunction" else "_spalding"
OUT = open("runs/airfoil2d_meshcheck%s.txt" % SUF, "w")


def say(s=""):
    print(s, flush=True)
    OUT.write(str(s) + "\n")
    OUT.flush()


rows = []
say("MESH SENSITIVITY of the 2-D sweep (chord 0.0287 m = the rotor mid-span)")
say("wall treatment: " + NUT)
say("surface cell is ISOTROPIC, so chordwise resolution and y+ move together.")
for cell in CELLS:
    cfg = AirfoilCaseConfig(surface_cell_m=cell, nut_wall_function=NUT)
    tag = "airfoil2d_m%d%s" % (round(cell * 1e6), SUF)
    src = Path("runs/" + tag)
    if src.exists():
        shutil.rmtree(src)
    case = make_case(tag, src)
    build_airfoil_mesh_case(case, cfg)
    write_run_control(src, cfg, N_ITER)
    case.sync_in()
    run_foam("cartesian2DMesh", case, (), timeout=7200, measure_rss=False)
    fix_boundary_types(case)
    rc = run_foam("checkMesh", case, (), timeout=1800, measure_rss=False)
    q = parse_checkmesh(rc.log_text)
    say("")
    say("surface_cell=%.2e m (%.2f%% chord)  CELLS=%s  maxSkew=%s"
        % (cell, 100 * cell / cfg.chord_m, q.get("cells"),
           q.get("max_skewness")))
    say("     Re    alpha      Cl        Cd     y+avg/max")
    for re in RES:
        u = re * cfg.nu / cfg.chord_m
        for al in ALPHAS:
            write_airfoil_fields(src, cfg, u, al)
            _wsl_sh("cd " + _quote(case.work) + " && rm -rf [1-9]* "
                    "postProcessing && mkdir -p 0")
            for f in ("U", "p", "k", "omega", "epsilon", "nuTilda", "nut"):
                _wsl_sh("cp " + _quote(to_wsl_path(src / "0" / f)) + " "
                        + _quote(case.work) + "/0/" + f)
            r = run_foam("simpleFoam", case, (), timeout=7200,
                         measure_rss=False,
                         log_name="log.a%+05.1f_re%.0f" % (al, re))
            txt = _wsl_sh("cat " + _quote(case.work)
                          + "/postProcessing/forces/0/force.dat").stdout
            dat = [l for l in txt.splitlines() if l.strip()
                   and not l.startswith("#")]
            if not dat:
                raise SolverError("no forces at cell=%g Re=%g a=%g"
                                  % (cell, re, al))
            v = [float(x) for x in
                 dat[-1].replace("(", " ").replace(")", " ").split()]
            dz = cfg.dz_frac * cfg.chord_m
            qc = 0.5 * cfg.rho * u ** 2 * cfg.chord_m * dz
            a = np.radians(al)
            cl = (-v[1] * np.sin(a) + v[2] * np.cos(a)) / qc
            cd = (v[1] * np.cos(a) + v[2] * np.sin(a)) / qc
            yp = [l for l in r.log_text.splitlines() if "y+ :" in l]
            if not yp:
                raise SolverError("no y+ at cell=%g Re=%g a=%g" % (cell, re, al))
            ypm = float(yp[-1].split("max = ")[1].split(",")[0])
            ypa = float(yp[-1].split("average = ")[1].split()[0])
            rows.append(dict(cell=cell, re=re, alpha=al, cl=cl, cd=cd,
                             yplus_avg=ypa, yplus_max=ypm,
                             cells=q.get("cells")))
            say("%8.0f %7.1f %9.4f %9.4f  %5.1f/%-5.1f" % (re, al, cl, cd,
                                                           ypa, ypm))

say("")
say("DRIFT vs the finest mesh (%.2e m):" % CELLS[-1])
say("  cell        Re    alpha    Cl       dCl vs finest")
fine = {(r["re"], r["alpha"]): r["cl"] for r in rows if r["cell"] == CELLS[-1]}
for r in rows:
    say("  %.2e %7.0f %6.1f %9.4f %+12.4f"
        % (r["cell"], r["re"], r["alpha"], r["cl"],
           r["cl"] - fine[(r["re"], r["alpha"])]))
json.dump(rows, open("runs/airfoil2d_meshcheck%s.json" % SUF, "w"), indent=1)
OUT.close()
