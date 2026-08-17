"""RUNG 1: advancing-flow validation against measured UIUC data.

WHY THIS RUN EXISTS. Steady frozen-rotor MRF failed in hover, and two mechanisms
are indistinguishable from hover data alone: (a) a setup/pipeline error somewhere
in geometry -> mesh -> section -> SST -> force extraction, or (b) the frozen-wake
model itself being invalid at V = 0, where the helix does not convect and sits as
a stack of standing vortex rings. At V > 0 the wake convects downstream and MRF
is inside its validity envelope, so this run tests (a) against MEASUREMENT with
everything else held identical to good5.

  within ~10% of measured -> the pipeline is validated end to end and hover
      steadiness is isolated as the sole CFD-specific problem.
  not within ~10%        -> there is a second problem, found before any transient
      spend.

The rpm is matched to a measured file EXACTLY rather than interpolated between
sweeps, because CT is weakly rpm-dependent through Reynolds number (see
data/uiuc/SOURCE.md). J is likewise taken from a measured row, so the comparison
involves no interpolation of the reference data at all.

Same extraction contract as run_hover_variant.py: reconstruct BEFORE sampling and
raise on any zero/missing result.
"""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from propopt.contracts import SolverError
from propopt.hf.hover_case import HoverCaseConfig, build_hover_case
from propopt.hf.openfoam import (_wsl_sh, make_case, parse_checkmesh,
                                 parse_residuals, parse_time_steps, run_foam)
from propopt.hf.wsl import _quote, to_wsl_path
from propopt.validation.uiuc import (design_from_geometry, load_advancing,
                                     load_geometry)

NAME = sys.argv[1]
J_WANT = float(sys.argv[2])
RPM = float(sys.argv[3]) if len(sys.argv) > 3 else 5018.0
FAR = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
NITER = int(sys.argv[5]) if len(sys.argv) > 5 else 2500
NP = int(sys.argv[6]) if len(sys.argv) > 6 else 8
RHO, D, R, NB = 1.225, 0.254, 0.127, 2
MEAS = {5018.0: "data/uiuc/apcsf_10x4.7_kt0837_5018.txt",
        4014.0: "data/uiuc/apcsf_10x4.7_kt0836_4014.txt"}[RPM]
SRC = Path("runs/adv_" + NAME)
OUT = open("runs/adv_" + NAME + ".txt", "w")


def say(s=""):
    print(s, flush=True)
    OUT.write(str(s) + "\n")
    OUT.flush()


# --- operating point, pinned to a MEASURED row (no reference interpolation) ---
tbl = load_advancing(Path(MEAS))
i_m = int(np.argmin(np.abs(tbl[:, 0] - J_WANT)))
J, CT_M, CP_M, ETA_M = (float(v) for v in tbl[i_m])
if abs(J - J_WANT) > 1e-9:
    say("NOTE requested J=%.4f snapped to measured row J=%.4f" % (J_WANT, J))
n_rev = RPM / 60.0
V = J * n_rev * D
say("RUNG 1 %s: rpm=%.0f J=%.4f -> V=%.4f m/s   (%s row %d)"
    % (NAME, RPM, J, V, Path(MEAS).name, i_m + 1))
say("  measured CT=%.4f CP=%.4f eta=%.3f" % (CT_M, CP_M, ETA_M))
# Wake convection per revolution: this is the number that makes MRF defensible
# here and not in hover. Blade-passage spacing is half of it for 2 blades.
say("  tip speed %.1f m/s, V/OmegaR=%.3f, wake convects %.2fR per rev "
    "(blade passage %.2fR)"
    % (2 * np.pi * n_rev * R, V / (2 * np.pi * n_rev * R),
       V / n_rev / R, V / n_rev / R / NB))
say("  blockage: disc area / box cross-section = %.2f%%"
    % (100.0 * np.pi * R ** 2 / (2 * FAR * R) ** 2))

geom = load_geometry(Path("data/uiuc/apcsf_10x4.7_geom.txt"))
design = design_from_geometry(geom, D, NB, RPM, n_stations=20)
# Everything except rpm and axial_inflow_m_s is good5's configuration verbatim.
cfg = HoverCaseConfig(rpm=RPM, periodic=False, background_cells_per_span=28,
                      surface_level=(4, 5), rotor_region_level=2,
                      far_field_radii=FAR, n_iterations=NITER, n_proc=NP,
                      n_layers=0, mrf_zone_radii=1.3,
                      mrf_zone_half_height_radii=0.4,
                      far_field_bc="totalPressure",   # ignored when advancing
                      axial_inflow_m_s=V)

if SRC.exists():
    shutil.rmtree(SRC)
case = make_case("adv_" + NAME, SRC)
build_hover_case(case, design, cfg)
case.sync_in()

stages = []
for app, args, npx in (("blockMesh", (), None), ("surfaceFeatureExtract", (), None),
                       ("decomposePar", ("-force",), None),
                       ("snappyHexMesh", ("-overwrite",), NP),
                       ("reconstructParMesh", ("-constant", "-mergeTol", "1e-6"), None),
                       ("checkMesh", ("-constant",), None)):
    r = run_foam(app, case, args, n_proc=npx, timeout=21600)
    stages.append(r)
    say("  %-22s%8.1fs  RSS %8.1f MB" % (app, r.wall_s, r.peak_rss_mb or float("nan")))
q = parse_checkmesh(stages[-1].log_text)
say("CELLS=%s mesh_ok=%s maxSkew=%s maxAR=%s maxNonOrtho=%s"
    % (q.get("cells"), q.get("mesh_ok"), q.get("max_skewness"),
       q.get("max_aspect_ratio"), q.get("max_non_orthogonality")))
mesh_wall = sum(s.wall_s for s in stages)

rts = run_foam("topoSet", case, (), timeout=3600)
zl = [l.strip() for l in rts.log_text.splitlines() if "now size" in l]
say("  MRF zone: " + (zl[-1] if zl else "NO CELLS SELECTED"))
if not zl or zl[-1].endswith(" 0"):
    raise SolverError(NAME + ": MRF cellZone is empty")
run_foam("decomposePar", case, ("-force",), timeout=3600)
rp = run_foam("potentialFoam", case, ("-writephi",), n_proc=NP, timeout=7200)
say("  potentialFoam         %8.1fs  RSS %8.1f MB" % (rp.wall_s, rp.peak_rss_mb))
r = run_foam("simpleFoam", case, (), n_proc=NP, timeout=86400)
say("  simpleFoam            %8.1fs  RSS %8.1f MB" % (r.wall_s, r.peak_rss_mb))
case.sync_out()

res = parse_residuals(r.log_text)
steps = parse_time_steps(r.log_text)
say("iters %d/%d converged=%s" % (len(steps), NITER, len(steps) < NITER))
for f, v in res.items():
    say("  %-7s %.3e -> %.3e (min %.3e)" % (f, v[0], v[-1], v.min()))


def load(sub, stem):
    rows = []
    for f in sorted((SRC / "postProcessing" / sub).rglob(stem + ".dat")):
        for ln in f.read_text().splitlines():
            s = ln.strip()
            if s and not s.startswith("#"):
                rows.append([float(x) for x in
                             s.replace("(", " ").replace(")", " ").split()])
    if not rows:
        raise SolverError(NAME + ": no rows in " + sub + "/" + stem + ".dat")
    w = min(len(x) for x in rows)
    return np.array([x[:w] for x in rows])


F, M = load("forces", "force"), load("forces", "moment")
N = 200
th, tq = F[:, 2], M[:, 2]
T, Q = float(th[-N:].mean()), float(tq[-N:].mean())
ppT = float((th[-N:].max() - th[-N:].min()) / abs(th[-N:].mean()))
ppQ = float((tq[-N:].max() - tq[-N:].min()) / abs(tq[-N:].mean()))
omega = RPM * 2 * np.pi / 60
P = abs(Q) * omega
ct = abs(T) / (RHO * n_rev ** 2 * D ** 4)
cp = P / (RHO * n_rev ** 3 * D ** 5)
eta = J * ct / cp
say("FORCES T=%+.5f N (p-p %.2f%%)  Q=%+.6f Nm (p-p %.2f%%)"
    % (T, ppT * 100, Q, ppQ * 100))
say("")
say("             CT        CP       eta")
say("  CFD     %8.4f  %8.4f  %8.3f" % (ct, cp, eta))
say("  meas    %8.4f  %8.4f  %8.3f" % (CT_M, CP_M, ETA_M))
say("  err     %+7.1f%%  %+7.1f%%  %+7.1f%%"
    % (100 * (ct / CT_M - 1), 100 * (cp / CP_M - 1), 100 * (eta / ETA_M - 1)))
verdict = "WITHIN 10%" if abs(ct / CT_M - 1) <= 0.10 else "OUTSIDE 10%"
say("  VERDICT on CT: " + verdict)

# --- wake: does the slipstream convect, unlike hover? ------------------------
rr = run_foam("reconstructPar", case, ("-latestTime",), timeout=14400,
              log_name="log.reconstructLatest")
latest_t = _wsl_sh("cd " + _quote(case.work)
                   + " && ls -d [0-9]* 2>/dev/null | sort -g | tail -1").stdout.strip()
if not latest_t or latest_t == "0":
    raise SolverError(NAME + ": no reconstructed time directory (got "
                      + repr(latest_t) + "); refusing to sample")
say("  reconstructed time: " + latest_t)

L = FAR * R
diag = "\n".join([
    "functions", "{",
    "    fluxMag { type surfaceFieldValue; libs (fieldFunctionObjects);",
    "      regionType patch; name farfield; operation sumMag; fields (phi);",
    "      writeFields false; log true; writeControl writeTime; }",
    "    fluxSum { type surfaceFieldValue; libs (fieldFunctionObjects);",
    "      regionType patch; name farfield; operation sum; fields (phi);",
    "      writeFields false; log true; writeControl writeTime; }",
    "    lines { type sets; libs (sampling); setFormat raw;",
    "      interpolationScheme cellPoint; fields (U p); writeControl writeTime;",
    "      sets {",
    "        wakeAxis { type uniform; axis y; start (0.0635 %.4f 0);"
    " end (0.0635 %.4f 0); nPoints 401; }" % (-L, L),
    "        radial { type uniform; axis x; start (0.0005 0 0);"
    " end (%.4f 0 0); nPoints 300; }" % L,
    "      } }", "}", ""])
cd = SRC / "system" / "controlDict"
cd.write_text(re.sub(r"functions\s*\{.*", diag, cd.read_text(), flags=re.S))
_wsl_sh("cp " + _quote(to_wsl_path(cd)) + " " + _quote(case.work)
        + "/system/controlDict")
rd = run_foam("simpleFoam", case, ("-postProcess", "-latestTime"),
              timeout=14400, log_name="log.diag")
case.sync_out()
fs = [l for l in rd.log_text.splitlines() if "farfield) of phi" in l]
if len(fs) < 2:
    raise SolverError(NAME + ": far-field flux objects produced no output")
say("FLUX " + " | ".join(x.strip() for x in fs[-2:]))

lat = _wsl_sh("cd " + _quote(case.work) + " && ls -d postProcessing/lines/* "
              "2>/dev/null | tail -1").stdout.strip()
if not lat:
    raise SolverError(NAME + ": no postProcessing/lines output")


def xy(nm):
    txt = _wsl_sh("cat " + _quote(case.work) + "/" + lat + "/" + nm).stdout
    a = np.array([[float(x) for x in l.split()] for l in txt.splitlines()
                  if l.strip() and not l.startswith("#")])
    if a.size == 0:
        raise SolverError(NAME + ": sample set " + nm + " is empty")
    return a


# Momentum theory WITH freestream: v_i = -V/2 + sqrt((V/2)^2 + T/(2 rho A)).
# The far-wake axial excess over the freestream is then 2 v_i.
vi_id = -0.5 * V + np.sqrt((0.5 * V) ** 2 + abs(T) / (2 * RHO * np.pi * R ** 2))
w = xy("wakeAxis_p_U.xy")
say("")
say("WAKE at r=0.5R.  u_exc = -Uy - V = slipstream excess over the freestream.")
say("  momentum theory: v_i=%.3f at the disc, 2*v_i=%.3f far downstream "
    "(freestream V=%.3f)" % (vi_id, 2 * vi_id, V))
say("  y/R     Uy      u_exc")
for yy in (FAR, 3, 2, 1, 0.5, 0, -0.5, -1, -2, -3, -FAR):
    uy = w[np.argmin(np.abs(w[:, 0] - yy * R)), 3]
    say("  %+5g %8.3f %8.3f" % (yy, uy, -uy - V))
down = w[w[:, 0] < -R]
rev = float(down[:, 3].max())
say("  downstream (y<-1R): max Uy = %+.3f  (POSITIVE = flow returning to the "
    "rotor, the hover failure mode)" % rev)
say("  slipstream %s" % ("CONVECTS (no reversal anywhere downstream)" if rev < 0
                         else "REVERSES - same failure as hover"))

rad = xy("radial_p_U.xy")
m = rad[:, 0] <= R
vd = float(2.0 / R ** 2 * np.trapezoid(rad[m, 3] * rad[m, 0], rad[m, 0]))
say("  disc-avg Uy = %+.3f m/s -> excess %.3f vs momentum v_i %.3f (ratio %.3f)"
    % (vd, -vd - V, vi_id, (-vd - V) / vi_id))

json.dump({"name": NAME, "rpm": RPM, "J": J, "V": V, "far": FAR,
           "cells": q.get("cells"), "T": T, "Q": Q, "ct": ct, "cp": cp,
           "eta": eta, "ct_meas": CT_M, "cp_meas": CP_M, "eta_meas": ETA_M,
           "ct_err": ct / CT_M - 1, "cp_err": cp / CP_M - 1,
           "eta_err": eta / ETA_M - 1, "verdict_ct": verdict,
           "ppT": ppT, "ppQ": ppQ, "iters": len(steps),
           "n_iter_requested": NITER, "vi_ideal": float(vi_id),
           "disc_avg_Uy": vd, "max_Uy_downstream": rev,
           "mesh_wall": mesh_wall, "solve_wall": r.wall_s,
           "solve_rss": r.peak_rss_mb},
          open("runs/summary_adv_" + NAME + ".json", "w"), indent=1)
OUT.close()
