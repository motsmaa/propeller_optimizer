"""TEST 3b: does the OUTBOARD loading recover with surface refinement?

Test 3a found the CFD/BEMT loading ratio collapsing monotonically outboard
(1.05 at r/R 0.22 -> 0.18 at 0.97), tracking cells/chord. Two explanations fit
equally: surface under-resolution outboard, or excessive tip-vortex downwash.

The discriminator is free - four completed runs differ ONLY in surface
refinement level, halving the surface cell each step:
    W23 (2,3) -> W34 (3,4) -> good5 (4,5) -> SURF56 (5,6)
Total CT is flat across them (0.0514/0.0550/0.0530/0.0485), which is why this
was recorded as eliminated. But a flat TOTAL is compatible with an outboard
recovery cancelled by an inboard or wake change, and the spanwise split was
never examined.

  outboard loading rises monotonically with refinement -> resolution is real
  outboard loading flat                                -> resolution is NOT the
                                                          cause; tip downwash is
"""
import base64, json, re
import numpy as np
from pathlib import Path
from propopt.contracts import SolverError
from propopt.hf.openfoam import _wsl_sh, make_case, run_foam
from propopt.hf.wsl import _quote, to_wsl_path

RHO, R = 1.225, 0.127
import sys
_ALL = {"W23": ("uiuc_W23", 1.376), "W34": ("uiuc_W34", 0.688),
        "good5": ("uiuc_good5", 0.344), "SURF56": ("uiuc_SURF56", 0.172),
        "A_mrf15": ("uiuc_A_mrf15", 0.344), "B_mrf40": ("uiuc_B_mrf40", 0.344),
        "C_far15": ("uiuc_C_far15", 0.344), "D_far20": ("uiuc_D_far20", 0.344),
        "F_inout": ("uiuc_F_inout", 0.344), "LONG": ("uiuc_LONG", 0.344),
        "J035": ("adv_J035", 0.344), "J046": ("adv_J046", 0.344)}
_sel = [a for a in sys.argv[1:] if not a.startswith("-")]
LADDER = [(t, _ALL[t][0], _ALL[t][1]) for t in (_sel or list(_ALL))]
OUTJ = "runs/span_" + ("_".join(_sel) if _sel else "all") + ".json"
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
    txt = path.read_text(errors="replace")
    arr = {}
    for m in re.finditer(r"<DataArray([^>]*)>(.*?)</DataArray>", txt, re.S):
        at, body = m.group(1), m.group(2).strip()
        nm = re.search(r"Name='([^']*)'", at).group(1)
        ty = re.search(r"type='([^']*)'", at).group(1)
        ncm = re.search(r"NumberOfComponents='([^']*)'", at)
        nc = int(ncm.group(1)) if ncm else 1
        raw = base64.b64decode(body)
        n = int(np.frombuffer(raw[:8], dtype="<u8")[0])
        v = np.frombuffer(raw[8:8 + n], dtype=DT[ty])
        arr[nm] = v.reshape(-1, nc) if nc > 1 else v
    return arr["Points"].astype(float), arr["offsets"], arr["connectivity"], \
        arr["p"].astype(float)


EDGES = np.linspace(0.0, R, 21)
RC = 0.5 * (EDGES[:-1] + EDGES[1:])
res = {}
for tag, cdir, cell_mm in LADDER:
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
        raise SolverError(tag + ": no blade VTP")
    pts, offs, conn, p = read_vtp(vs[-1])
    starts = np.concatenate(([0], offs[:-1]))
    nf = len(offs)
    ctr = np.zeros((nf, 3)); Sf = np.zeros((nf, 3))
    for q in range(nf):
        v = pts[conn[starts[q]:offs[q]]]
        c = v.mean(axis=0); a = np.zeros(3)
        for t in range(len(v)):
            a += 0.5 * np.cross(v[t] - c, v[(t + 1) % len(v)] - c)
        ctr[q], Sf[q] = c, a
    fy = RHO * p * Sf[:, 1]
    ref = [l for l in (src / "postProcessing" / "forces" / "0"
                       / "force.dat").read_text().splitlines()
           if l.strip() and not l.startswith("#")][-1]
    pref = float(ref.replace("(", " ").replace(")", " ").split()[5])
    if abs(fy.sum() - pref) / abs(pref) > 0.05:
        raise SolverError("%s: re-integration %.4f vs force.dat %.4f"
                          % (tag, fy.sum(), pref))
    ra = np.abs(ctr[:, 2])
    dTdr = np.array([fy[(ra >= EDGES[i]) & (ra < EDGES[i + 1])].sum()
                     / (EDGES[i + 1] - EDGES[i]) for i in range(20)])
    res[tag] = dict(cell_mm=cell_mm, faces=nf, total=float(fy.sum()),
                    dTdr=dTdr.tolist())
    print("%-7s cell=%.3f mm  faces=%6d  T_pressure=%+.4f N"
          % (tag, cell_mm, nf, fy.sum()), flush=True)

print("")
print("dT/dr [N/m] by surface cell size, and the INBOARD/OUTBOARD split")
print("  r/R    " + "".join("%10s" % t for t, _, _ in LADDER))
for i in range(2, 20):
    print("  %.3f  " % (RC[i] / R)
          + "".join("%10.2f" % res[t]["dTdr"][i] for t, _, _ in LADDER), flush=True)
print("")
inb = slice(2, 10)      # r/R 0.125-0.525
outb = slice(12, 20)    # r/R 0.625-0.975
print("  band            " + "".join("%10s" % t for t, _, _ in LADDER))
for nm, sl in (("inboard  <0.55R", inb), ("outboard >0.60R", outb)):
    v = [np.trapezoid(np.array(res[t]["dTdr"])[sl], RC[sl]) for t, _, _ in LADDER]
    print("  %-15s" % nm + "".join("%10.4f" % x for x in v), flush=True)
    print("   ratio to W23   " + "".join("%10.3f" % (x / v[0]) for x in v), flush=True)
json.dump(res, open(OUTJ, "w"), indent=1)
