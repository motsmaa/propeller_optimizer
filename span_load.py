"""TEST 3a: WHERE on the blade is the thrust missing?

The forces function object's binData was silently ignored by this build (it
wrote totals only), so the spanwise loading is computed directly instead: dump
the blade patch as VTK, rebuild face centres / areas / normals from the polygon
connectivity, and integrate p*n_y*A into spanwise bins.

PRESSURE ONLY. From force.dat the split is pressure_y = +1.5787 N against
viscous_y = -0.0110 N, so viscous is 0.7% of thrust and its omission cannot
change the SHAPE of the spanwise deficit.

The shape discriminates:
  outer-span collapse -> surface under-resolution (chord shrinks outboard, so
                         cells/chord falls and the tip is only ~2 cells thick)
  inner-span collapse -> root / hub treatment
  uniform deficit     -> something global (frozen rotor, turbulence, geometry)
"""
import json, re
import numpy as np
from pathlib import Path
from propopt.contracts import SolverError
from propopt.hf.openfoam import _wsl_sh, make_case, run_foam
from propopt.hf.wsl import _quote, to_wsl_path
from propopt.solvers.bemt import BEMTAeroSolver
from propopt.solvers.polars import TabulatedPolar
from propopt.validation.uiuc import design_from_geometry, load_geometry

RHO, D, R, NB = 1.225, 0.254, 0.127, 2
CASES = {"good5": ("uiuc_good5", 4590.0), "J046": ("adv_J046", 5018.0)}

FN = """
functions
{
    bladeSurf
    {
        type surfaces; libs (sampling);
        surfaceFormat vtk; fields (p);
        writeControl writeTime;
        surfaces { blade { type patch; patches ("blade_.*"); interpolate false; } }
    }
}
"""


DT = {"Float32": "<f4", "Float64": "<f8", "Int32": "<i4", "Int64": "<i8",
      "UInt8": "<u1", "UInt32": "<u4", "UInt64": "<u8"}


def read_vtp(path):
    """Binary-base64 VTK XML PolyData -> (points, offsets, connectivity, cell p).

    Each DataArray is base64(uint64 byte-count || payload) when uncompressed,
    which is why the first 8 decoded bytes are stripped.
    """
    import base64
    txt = path.read_text(errors="replace")
    arr = {}
    for m in re.finditer(r"<DataArray([^>]*)>(.*?)</DataArray>", txt, re.S):
        at, body = m.group(1), m.group(2).strip()
        nm = re.search(r"Name='([^']*)'", at).group(1)
        ty = re.search(r"type='([^']*)'", at).group(1)
        nc = int((re.search(r"NumberOfComponents='([^']*)'", at)
                  or re.match(r"(1)", "1")).group(1))
        raw = base64.b64decode(body)
        n = int(np.frombuffer(raw[:8], dtype="<u8")[0])
        v = np.frombuffer(raw[8:8 + n], dtype=DT[ty])
        arr[nm] = v.reshape(-1, nc) if nc > 1 else v
    for k in ("Points", "connectivity", "offsets", "p"):
        if k not in arr:
            raise SolverError("VTP missing array " + k)
    return arr["Points"].astype(float), arr["offsets"], arr["connectivity"],         arr["p"].astype(float)


out = {}
for tag, (cdir, rpm) in CASES.items():
    src = Path("runs/" + cdir)
    case = make_case(cdir, src)
    cd = src / "system" / "controlDict"
    cd.write_text(re.sub(r"functions\s*\{.*", FN, cd.read_text(), flags=re.S))
    _wsl_sh("cp " + _quote(to_wsl_path(cd)) + " " + _quote(case.work)
            + "/system/controlDict")
    run_foam("simpleFoam", case, ("-postProcess", "-latestTime"), timeout=14400,
             log_name="log.spansurf", measure_rss=False)
    case.sync_out()
    vs = sorted((src / "postProcessing" / "bladeSurf").rglob("*.vtp"))
    if not vs:
        raise SolverError(tag + ": no blade VTP written")
    pts, offs, conn, p = read_vtp(vs[-1])
    starts = np.concatenate(([0], offs[:-1]))
    nf = len(offs)
    ctr = np.zeros((nf, 3))
    Sf = np.zeros((nf, 3))
    for q in range(nf):
        v = pts[conn[starts[q]:offs[q]]]
        c = v.mean(axis=0)
        a = np.zeros(3)
        for t in range(len(v)):
            a += 0.5 * np.cross(v[t] - c, v[(t + 1) % len(v)] - c)
        ctr[q], Sf[q] = c, a
    # thrust per face: rho * p * (-n_y) ... sign fixed by matching the total
    fy = RHO * p * Sf[:, 1]
    tot = fy.sum()
    print("%s: %d faces, sum(rho*p*Sf_y) = %+.4f N  (force.dat pressure_y "
          "reference below)" % (tag, nf, tot), flush=True)
    sgn = 1.0
    ref = [l for l in (src / "postProcessing" / "forces" / "0"
                       / "force.dat").read_text().splitlines()
           if l.strip() and not l.startswith("#")][-1]
    pref = float(ref.replace("(", " ").replace(")", " ").split()[5])
    if abs(tot + pref) < abs(tot - pref):
        sgn = -1.0
    fy *= sgn
    print("   force.dat pressure_y = %+.4f N ; sign %+.0f -> integrated %+.4f N"
          % (pref, sgn, fy.sum()), flush=True)
    if abs(fy.sum() - pref) / abs(pref) > 0.05:
        raise SolverError(tag + ": surface re-integration disagrees with "
                          "force.dat by >5%% (%.4f vs %.4f)" % (fy.sum(), pref))

    r_abs = np.abs(ctr[:, 2])
    edges = np.linspace(0.0, R, 21)
    rc = 0.5 * (edges[:-1] + edges[1:])
    dTdr = np.array([fy[(r_abs >= edges[i]) & (r_abs < edges[i + 1])].sum()
                     / (edges[i + 1] - edges[i]) for i in range(20)])

    design = design_from_geometry(load_geometry(
        Path("data/uiuc/apcsf_10x4.7_geom.txt")), D, NB, rpm, n_stations=20)
    pol = TabulatedPolar.from_xfoil_files(
        [q for q in sorted(Path("data/polars").glob("naca4412_re*.txt"))
         if int(q.stem.split("re")[1]) >= 50_000])
    tpl, _ = BEMTAeroSolver(polar=pol, rho=RHO, relax=0.2,
                            max_iter=1000).spanwise_loads(design)
    rb = np.asarray(design.station_r_m)
    bem = np.interp(rc, rb, np.asarray(tpl) * NB)
    ch = np.interp(rc / R, [0.15, 0.55, 1.0], [0.109, 0.226, 0.042]) * R
    print("   r/R   chord[mm] cells/chord  dT/dr CFD   dT/dr BEMT   ratio",
          flush=True)
    for i in range(3, 20):
        print("   %.3f %8.2f %10.1f %11.2f %12.2f %8.3f"
              % (rc[i] / R, 1e3 * ch[i], ch[i] / 3.44e-4, dTdr[i], bem[i],
                 dTdr[i] / bem[i] if bem[i] else float("nan")), flush=True)
    out[tag] = dict(r=rc.tolist(), dTdr_cfd=dTdr.tolist(),
                    dTdr_bemt=bem.tolist(), total=float(fy.sum()))
json.dump(out, open("runs/span_load.json", "w"), indent=1)
