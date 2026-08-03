"""One diagnostic variant: mesh -> potentialFoam -> simpleFoam -> forces + wake.

EXTRACTION CONTRACT: every derived quantity is asserted non-degenerate. The
solve runs in PARALLEL, so fields live in processor*/<time> and there is NO
reconstructed time directory; a serial `-postProcess -latestTime` then finds
nothing, samples nothing, and writes ZEROS that look like data. This module
reconstructs first and RAISES on any zero or missing result.
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
from propopt.validation.uiuc import design_from_geometry, load_geometry

NAME, FAR, LAY = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
NP = int(sys.argv[4]) if len(sys.argv) > 4 else 8
MRF_R = float(sys.argv[5]) if len(sys.argv) > 5 else 1.3
MRF_H = float(sys.argv[6]) if len(sys.argv) > 6 else 0.4
MRF_OFF = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
RPM, RHO, D, R = 4590.0, 1.225, 0.254, 0.127
SRC = Path("runs/uiuc_" + NAME)
OUT = open("runs/uiuc_" + NAME + ".txt", "w")


def say(s=""):
    print(s, flush=True)
    OUT.write(str(s) + "\n")
    OUT.flush()


geom = load_geometry(Path("data/uiuc/apcsf_10x4.7_geom.txt"))
design = design_from_geometry(geom, D, 2, RPM, n_stations=20)
base = dict(rpm=RPM, periodic=False, background_cells_per_span=28,
            surface_level=(4, 5), rotor_region_level=2, far_field_radii=FAR,
            n_iterations=2500, n_proc=NP,
            mrf_zone_radii=MRF_R, mrf_zone_half_height_radii=MRF_H)
if LAY:
    base.update(n_layers=12, layer_expansion=1.2, first_layer_m=2.6e-5,
                nut_wall_function="nutUSpaldingWallFunction")
else:
    base.update(n_layers=0)
cfg = HoverCaseConfig(**base)
say("VARIANT " + NAME + ": far=" + str(FAR) + "R layers=" + str(cfg.n_layers)
    + " mrf_r=" + str(MRF_R) + "R mrf_h=" + str(MRF_H) + "R off="
    + str(MRF_OFF) + "R np=" + str(NP))

if SRC.exists():
    shutil.rmtree(SRC)
case = make_case("uiuc_" + NAME, SRC)
build_hover_case(case, design, cfg)

if MRF_OFF:
    ts = SRC / "system" / "topoSetDict"
    t = ts.read_text()
    lo = -MRF_H * R + MRF_OFF * R
    hi = MRF_H * R + MRF_OFF * R
    t = re.sub(r"point1\s+\(0 [-0-9.eE+]+ 0\);", "point1  (0 %.6f 0);" % lo, t)
    t = re.sub(r"point2\s+\(0 [-0-9.eE+]+ 0\);", "point2  (0 %.6f 0);" % hi, t)
    ts.write_text(t)
    say("  MRF zone axial span %+.4f .. %+.4f m  (%+.2fR .. %+.2fR)"
        % (lo, hi, lo / R, hi / R))
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
mesh_rss = max((s.peak_rss_mb or 0) for s in stages)
say("MESH wall %.1fs peak RSS %.1f MB" % (mesh_wall, mesh_rss))

rts = run_foam("topoSet", case, ("-constant",), timeout=3600)
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
say("iters %d/2500 converged=%s" % (len(steps), len(steps) < 2500))
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


F = load("forces", "force")
M = load("forces", "moment")
N = 200
th = F[:, 2]
tq = M[:, 2]
T = float(th[-N:].mean())
Q = float(tq[-N:].mean())
ppT = float((th[-N:].max() - th[-N:].min()) / abs(th[-N:].mean()))
ppQ = float((tq[-N:].max() - tq[-N:].min()) / abs(tq[-N:].mean()))
omega = RPM * 2 * np.pi / 60
P = abs(Q) * omega
nrev = RPM / 60
ct = abs(T) / (RHO * nrev ** 2 * D ** 4)
cp = P / (RHO * nrev ** 3 * D ** 5)
say("FORCES T=%+.5f N (p-p %.2f%%)  Q=%+.6f Nm (p-p %.2f%%)"
    % (T, ppT * 100, Q, ppQ * 100))
say("CT=%.4f  CP=%.4f  P=%.2f W" % (ct, cp, P))

rr = run_foam("reconstructPar", case, ("-latestTime",), timeout=14400,
              log_name="log.reconstructLatest")
say("  reconstructPar        %8.1fs  RSS %8.1f MB" % (rr.wall_s, rr.peak_rss_mb))
latest_t = _wsl_sh("cd " + _quote(case.work)
                   + " && ls -d [0-9]* 2>/dev/null | sort -g | tail -1").stdout.strip()
if not latest_t or latest_t == "0":
    raise SolverError(NAME + ": no reconstructed time directory (got "
                      + repr(latest_t) + "); refusing to sample - this is the "
                      "silent-zeros failure mode")
say("  reconstructed time: " + latest_t)

L = FAR * R
diag = "\n".join([
    "functions", "{",
    "    fluxSum { type surfaceFieldValue; libs (fieldFunctionObjects);",
    "      regionType patch; name farfield; operation sum; fields (phi);",
    "      writeFields false; log true; writeControl writeTime; }",
    "    fluxMag { type surfaceFieldValue; libs (fieldFunctionObjects);",
    "      regionType patch; name farfield; operation sumMag; fields (phi);",
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
flux_mag = float(fs[-1].split("=")[-1])
if flux_mag == 0.0:
    raise SolverError(NAME + ": sumMag(phi) is exactly 0.0 - sampling failed")

lat = _wsl_sh("cd " + _quote(case.work)
              + " && ls -d postProcessing/lines/* 2>/dev/null | tail -1").stdout.strip()
if not lat:
    raise SolverError(NAME + ": no postProcessing/lines output")


def xy(nm):
    txt = _wsl_sh("cat " + _quote(case.work) + "/" + lat + "/" + nm).stdout
    a = np.array([[float(x) for x in l.split()] for l in txt.splitlines()
                  if l.strip() and not l.startswith("#")])
    if a.size == 0:
        raise SolverError(NAME + ": sample set " + nm + " is empty")
    return a


rad = xy("radial_p_U.xy")
m = rad[:, 0] <= R
vi = float(2.0 / R ** 2 * np.trapezoid(rad[m, 3] * rad[m, 0], rad[m, 0]))
vi_ideal = float(np.sqrt(abs(T) / (2 * RHO * np.pi * R ** 2)))
if vi == 0.0:
    raise SolverError(NAME + ": disc-averaged Uy is exactly 0.0 - sampling failed")
ratio = abs(vi) / vi_ideal
say("INDUCED disc-avg Uy=%+.3f m/s  ideal=%.3f  ratio=%.3f" % (vi, vi_ideal, ratio))
w = xy("wakeAxis_p_U.xy")
prof = []
for yy in (-FAR, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, FAR):
    prof.append("%+gR:%+.2f" % (yy, w[np.argmin(np.abs(w[:, 0] - yy * R)), 3]))
say("WAKE Uy(y/R) " + " ".join(prof))
down = w[w[:, 0] < -R]
say("RECIRC mean Uy for y<-1R = %+.3f m/s (negative = exiting, positive = returning)"
    % down[:, 3].mean())

json.dump({"name": NAME, "far": FAR, "layers": cfg.n_layers, "mrf_r": MRF_R,
           "mrf_h": MRF_H, "mrf_off": MRF_OFF, "cells": q.get("cells"),
           "T": T, "Q": Q, "ct": ct, "cp": cp, "ppT": ppT, "ppQ": ppQ,
           "vi": vi, "vi_ideal": vi_ideal, "ratio": ratio,
           "flux_sumMag": flux_mag, "recirc_mean_Uy": float(down[:, 3].mean()),
           "iters": len(steps), "mesh_wall": mesh_wall, "mesh_rss": mesh_rss,
           "solve_wall": r.wall_s, "solve_rss": r.peak_rss_mb},
          open("runs/summary_" + NAME + ".json", "w"), indent=1)
OUT.close()
