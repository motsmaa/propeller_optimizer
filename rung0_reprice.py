"""RUNG 0: reprice CT from the CFD's OWN inflow. Post-processing only, no solve.

The question: is the CFD's low thrust (CT 0.0530) the blade responding CORRECTLY
to the wrong inflow it is given, or is the blade loading itself broken?

Method: extract the through-disc axial velocity from a converged CFD field,
form phi(r) = atan(v_axial / Omega r), alpha(r) = beta(r) - phi(r), and price
alpha through the SAME XFOIL polars BEMT uses, with the SAME blade-element
integral (bemt.py lines 158/186 -- copied here, not re-derived). If the repriced
CT lands near the CFD's directly-integrated 0.0530 rather than near BEMT's
0.0964, the recirculation is established as the cause.

FRAME (from hf/hover_case.py): x tangential, y AXIAL (rotation axis), z span.
Blades lie along +-z. Thrust acts along +y; the induced flow is -y, so the
through-disc speed is v_ax = -Uy. Rotation is about +y with omega NEGATIVE, so
the blade-travel unit vector at (x,y,z) is t = (-z, 0, x)/r and the fluid's
co-rotating swirl is u_sw = U.t = (-z*Ux + x*Uz)/r.

BLADE CONTAMINATION. The disc plane y=0 is the BLADE plane: a radial ray at a
blade azimuth runs inside the solid. Points inside the blade are simply absent
from the sample (no cell to interpolate), but points just OFF the surface carry
the blade's own bound-vortex up/downwash, which is exactly the self-induced part
BEMT accounts for through the lift itself and must NOT be counted as inflow. So
azimuths within BLADE_EXCL_DEG of a blade are dropped, and the y=+-0.10R planes
are reported alongside y=0 as a sensitivity check.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

from propopt.contracts import SolverError
from propopt.hf.openfoam import _wsl_sh, make_case, run_foam
from propopt.hf.wsl import _quote, to_wsl_path
from propopt.solvers.bemt import MU_AIR, BEMTAeroSolver
from propopt.solvers.polars import TabulatedPolar
from propopt.stations import station_weights
from propopt.validation.uiuc import (design_from_geometry, load_geometry,
                                     thrust_power_coefficients)

NAME = sys.argv[1] if len(sys.argv) > 1 else "good5"
RESAMPLE = "--reuse" not in sys.argv
RPM, RHO, D, R, NB = 4590.0, 1.225, 0.254, 0.127, 2
CT_CFD = {"good5": 0.05296398800138271, "LONG": 0.05111229016162383}[NAME]
CT_MEAS, CP_MEAS = 0.1200, 0.0484

N_AZ, AZ_OFF = 36, 5.0        # azimuthal rays; offset so none runs along a blade
N_RAD, R0F, R1F = 80, 0.10, 1.05
PLANES = (0.25, 0.10, 0.0, -0.10, -0.25)   # y/R: +upstream ... -wake
BLADE_AZ = (90.0, 270.0)       # blades lie along +-z
BLADE_EXCL_DEG = 20.0
OUT = open("runs/rung0_" + NAME + ".txt", "w")


def say(s=""):
    print(s, flush=True)
    OUT.write(str(s) + "\n")
    OUT.flush()


geom = load_geometry(Path("data/uiuc/apcsf_10x4.7_geom.txt"))
design = design_from_geometry(geom, D, NB, RPM, n_stations=20)
case = make_case("uiuc_" + NAME, Path("runs/uiuc_" + NAME))   # does NOT sync_in

# ---------------------------------------------------------------- sample -----
lat = _wsl_sh("cd " + _quote(case.work)
              + " && ls -d [0-9]* 2>/dev/null | sort -g | tail -1").stdout.strip()
if not lat or lat == "0":
    raise SolverError(NAME + ": no reconstructed time directory (got " + repr(lat)
                      + "); rung 0 needs the reconstructed field")
say("RUNG 0 on " + NAME + ": reconstructed time " + lat)

az = np.radians(AZ_OFF + 360.0 * np.arange(N_AZ) / N_AZ)
if RESAMPLE:
    sets = []
    for iy, yf in enumerate(PLANES):
        for ia, a in enumerate(az):
            r0, r1 = R0F * R, R1F * R
            s = (r0 * np.cos(a), yf * R, r0 * np.sin(a))
            e = (r1 * np.cos(a), yf * R, r1 * np.sin(a))
            sets.append("        s%d_%02d { type uniform; axis xyz; start (%.6f "
                        "%.6f %.6f); end (%.6f %.6f %.6f); nPoints %d; }"
                        % (iy, ia, s[0], s[1], s[2], e[0], e[1], e[2], N_RAD))
    fn = "\n".join(["functions", "{", "    discs { type sets; libs (sampling);",
                    "      setFormat raw; interpolationScheme cellPoint;",
                    "      fields (U p); writeControl writeTime;", "      sets", "      {"]
                   + sets + ["      } }", "}", ""])
    cdt = (Path("runs/uiuc_" + NAME) / "system" / "controlDict").read_text()
    tmp = Path("runs") / ("rung0_controlDict_" + NAME)
    tmp.write_text(re.sub(r"functions\s*\{.*", fn, cdt, flags=re.S))
    _wsl_sh("cp " + _quote(to_wsl_path(tmp)) + " " + _quote(case.work)
            + "/system/controlDict")
    rd = run_foam("simpleFoam", case, ("-postProcess", "-latestTime"),
                  timeout=14400, log_name="log.rung0")
    say("  postProcess %.1fs" % rd.wall_s)

sdir = _wsl_sh("cd " + _quote(case.work) + " && ls -d postProcessing/discs/* "
               "2>/dev/null | tail -1").stdout.strip()
if not sdir:
    raise SolverError(NAME + ": no postProcessing/discs output")
raw = _wsl_sh("cd " + _quote(case.work) + "/" + sdir
              + " && for f in s*_p_U.xy; do awk -v n=\"$f\" "
                "'!/^#/ && NF>0 {print n, $0}' \"$f\"; done")
if raw.returncode != 0 or not raw.stdout.strip():
    raise SolverError(NAME + ": sample files unreadable: " + raw.stderr[:300])

rows = {}
for ln in raw.stdout.splitlines():
    p = ln.split()
    if len(p) != 8:                     # name + x y z p Ux Uy Uz
        raise SolverError(NAME + ": expected 7 data columns (x y z p Ux Uy Uz), "
                          "got %d in %r" % (len(p) - 1, ln[:120]))
    m = re.match(r"s(\d+)_(\d+)_p_U\.xy", p[0])
    rows.setdefault(int(m.group(1)), []).append([float(v) for v in p[1:]])

# ------------------------------------------------- azimuthal averaging -------
RG = np.linspace(R0F * R, R1F * R, N_RAD)


def profile(iy, exclude_blades):
    """Azimuthally averaged (v_ax, u_swirl) on the RG radial grid for one plane.
    Returns (mean, median, n_used_per_radius)."""
    a = np.asarray(rows[iy])
    x, z, ux, uy, uz = a[:, 0], a[:, 2], a[:, 4], a[:, 5], a[:, 6]
    r = np.hypot(x, z)
    ok = r > 1e-9
    psi = np.degrees(np.arctan2(z[ok], x[ok])) % 360.0
    if exclude_blades:
        d = np.minimum.reduce([np.abs((psi - b + 180.0) % 360.0 - 180.0)
                               for b in BLADE_AZ])
        ok[np.flatnonzero(ok)[d < BLADE_EXCL_DEG]] = False
    r, x, z = r[ok], x[ok], z[ok]
    v_ax = -uy[ok]                                   # +ve = through-disc (-y)
    u_sw = (-z * ux[ok] + x * uz[ok]) / r            # +ve = co-rotating
    mean = np.full((2, RG.size), np.nan)
    med = np.full((2, RG.size), np.nan)
    n = np.zeros(RG.size, dtype=int)
    half = 0.5 * (RG[1] - RG[0])
    for i, rc in enumerate(RG):
        m = np.abs(r - rc) <= half
        n[i] = int(m.sum())
        if n[i]:
            mean[:, i] = [v_ax[m].mean(), u_sw[m].mean()]
            med[:, i] = [np.median(v_ax[m]), np.median(u_sw[m])]
    return mean, med, n


iy0 = PLANES.index(0.0)
mean0, med0, n0 = profile(iy0, True)
mean0_all, _, _ = profile(iy0, False)
if np.all(np.isnan(mean0[0])):
    raise SolverError(NAME + ": disc-plane profile is entirely empty")
say("  disc plane: %d..%d valid samples per radial bin (blades +-%.0f deg excluded)"
    % (n0.min(), n0.max(), BLADE_EXCL_DEG))

say("")
say("INFLOW v_ax(r) [m/s, +ve = through the disc], azimuthal mean")
say("  r/R    " + "".join("%9s" % ("y=%+.2fR" % p) for p in PLANES)
    + "   med(y=0)  all-az(y=0)")
prof = {p: profile(PLANES.index(p), True)[0] for p in PLANES}
for rf in (0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
    i = int(np.argmin(np.abs(RG - rf * R)))
    say("  %.2f   " % rf + "".join("%9.3f" % prof[p][0, i] for p in PLANES)
        + "%11.3f%12.3f" % (med0[0, i], mean0_all[0, i]))

# ------------------------------------------------------------- reprice -------
r = np.asarray(design.station_r_m, dtype=float)
c = np.asarray(design.chord_m, dtype=float)
beta = np.asarray(design.twist_rad, dtype=float)
w_r = station_weights(r, design.hub_radius_m, R)
omega = RPM * 2.0 * np.pi / 60.0
ut = omega * r
# The polar set must be the one the TRACKED BEMT reference uses, or the
# comparison prices two different sections. tests/test_validation_uiuc.py
# `tab-re5e4` (Re >= 5e4 only, relax=0.2) is what produces the headline
# CT 0.0964 / -20%; the extended set including Re 1e4-3e4 is XFAIL there
# (edge-of-validity XFOIL on a 12% section), so it is reported as a
# sensitivity, never as the primary.
_pf = sorted(Path("data/polars").glob("naca4412_re*.txt"))
POLARS = {
    "re>=5e4": TabulatedPolar.from_xfoil_files(
        [p for p in _pf if int(p.stem.split("re")[1]) >= 50_000]),
    "extended": TabulatedPolar.from_xfoil_files(_pf),
}
polar = POLARS["re>=5e4"]
good = ~np.isnan(mean0[0])
v_ax = np.interp(r, RG[good], mean0[0][good])
u_sw = np.interp(r, RG[good], mean0[1][good])


def price(v, utan, tag, pol=None):
    """BEMT's blade-element integral, verbatim (bemt.py 158/186), on an IMPOSED
    inflow. The Prandtl loss factor does not appear: it enters BEMT only through
    the momentum balance that SOLVES for v, and here v is measured, not solved."""
    phi = np.arctan2(v, utan)
    w = np.hypot(v, utan)
    alpha = beta - phi
    cl, cd = (pol or polar)(alpha, RHO * w * c / MU_AIR)
    sp, cp_ = np.sin(phi), np.cos(phi)
    q = 0.5 * RHO * w ** 2 * NB * c
    T = float(np.sum(q * (cl * cp_ - cd * sp) * w_r))
    Q = float(np.sum(q * (cl * sp + cd * cp_) * r * w_r))
    n = RPM / 60.0
    ct = T / (RHO * n ** 2 * D ** 4)
    cp = (Q * omega) / (RHO * n ** 3 * D ** 5)
    return dict(tag=tag, T=T, Q=Q, ct=ct, cp=cp, alpha_deg=np.degrees(alpha),
                phi_deg=np.degrees(phi), cl=cl, cd=cd,
                re=RHO * w * c / MU_AIR, dT=q * (cl * cp_ - cd * sp))


A = price(v_ax, ut, "CFD inflow, no swirl correction")
B = price(v_ax, ut - u_sw, "CFD inflow + swirl")
bemt = BEMTAeroSolver(polar=polar, relax=0.2, max_iter=1000)
ev = bemt.evaluate(design)
ct_b, cp_b = thrust_power_coefficients(ev, RPM, D, RHO)
_, _, _, _, _, sin_b, cos_b, _ = bemt._converged_state(design)
phi_b = np.arctan2(sin_b, cos_b)
v_bemt = ut * np.tan(phi_b)

say("")
say("ANGLES at the 20 BEMT stations (deg)")
say("  r/R    beta   phi_BEMT  phi_CFD  a_BEMT  a_CFD  a_CFD+swirl   v_BEMT  v_CFD")
for i in range(len(r)):
    say("  %.3f %7.2f %8.2f %8.2f %7.2f %6.2f %10.2f %10.2f %6.2f"
        % (r[i] / R, np.degrees(beta[i]), np.degrees(phi_b[i]), A["phi_deg"][i],
           np.degrees(beta[i] - phi_b[i]), A["alpha_deg"][i], B["alpha_deg"][i],
           v_bemt[i], v_ax[i]))

say("")
say("REPRICED THRUST")
say("  %-34s CT=%.4f  CP=%.4f  T=%.4f N" % ("BEMT (self-consistent inflow)",
                                            ct_b, cp_b, ev.thrust_N))
for X in (A, B):
    say("  %-34s CT=%.4f  CP=%.4f  T=%.4f N" % (X["tag"], X["ct"], X["cp"], X["T"]))
say("  %-34s CT=%.4f" % ("CFD, directly integrated", CT_CFD))
say("  %-34s CT=%.4f  CP=%.4f" % ("MEASURED (UIUC)", CT_MEAS, CP_MEAS))
say("")
say("CLOSURE  repriced/CFD = %.3f (no swirl)  %.3f (swirl)   [1.00 = closes]"
    % (A["ct"] / CT_CFD, B["ct"] / CT_CFD))
say("         repriced/BEMT = %.3f (no swirl)  %.3f (swirl)"
    % (A["ct"] / ct_b, B["ct"] / ct_b))

say("")
say("SENSITIVITY (does the closure depend on a choice I made?)")
for pid, pol in POLARS.items():
    sb = BEMTAeroSolver(polar=pol, relax=0.2, max_iter=1000)
    cb, _ = thrust_power_coefficients(sb.evaluate(design), RPM, D, RHO)
    sa = price(v_ax, ut, "", pol)
    say("  polar %-10s  BEMT CT=%.4f   repriced CT=%.4f   ratio to CFD %.3f"
        % (pid, cb, sa["ct"], sa["ct"] / CT_CFD))
for pl in PLANES:
    g = ~np.isnan(prof[pl][0])
    vp = np.interp(r, RG[g], prof[pl][0][g])
    sa = price(vp, ut, "")
    say("  inflow plane y=%+.2fR  repriced CT=%.4f   ratio to CFD %.3f"
        % (pl, sa["ct"], sa["ct"] / CT_CFD))
sa = price(np.interp(r, RG[good], mean0_all[0][good]), ut, "")
say("  all azimuths (blade-adjacent NOT excluded), y=0: CT=%.4f  ratio %.3f"
    % (sa["ct"], sa["ct"] / CT_CFD))

# ------------------------------------------------------------- inversion -----
# alpha is small, so CT is hypersensitive to phi and the plane-to-plane spread
# above is huge. The sharper question is INVERSE: by what uniform factor must
# the measured disc-plane inflow be scaled to reproduce a given CT? k near 1
# means that CT is consistent with the CFD's own measured inflow; k far from 1
# means it is not reachable from this flow field at all.
def invert(target):
    k = np.linspace(0.05, 1.60, 3101)
    ct = np.array([price(kk * v_ax, ut, "")["ct"] for kk in k])
    if not (ct.min() <= target <= ct.max()):
        return float("nan")
    i = int(np.argmin(np.abs(ct - target)))
    return float(k[i])


say("")
say("INVERSION  scale factor k on the measured disc-plane inflow needed to hit:")
for lbl, tgt in (("CFD's own CT 0.0530", CT_CFD), ("BEMT CT %.4f" % ct_b, ct_b),
                 ("measured CT 0.1200", CT_MEAS)):
    k = invert(tgt)
    say("  %-22s k = %.3f   -> v_ax(mid-span) %.2f m/s"
        % (lbl, k, k * float(np.interp(0.5 * R, RG[good], mean0[0][good]))))

say("")
say("SPANWISE THRUST dT/dr [N/m per blade-pair] and where the load is lost")
say("  r/R     a_BEMT  a_CFD    dT/dr_BEMT  dT/dr_CFD")
_, _, q_b, cl_b, cd_b, sb_, cb_, _ = bemt._converged_state(design)
dT_b = q_b * (cl_b * cb_ - cd_b * sb_)
for i in range(0, len(r), 2):
    say("  %.3f  %7.2f %7.2f %12.2f %11.2f"
        % (r[i] / R, np.degrees(beta[i] - phi_b[i]), A["alpha_deg"][i],
           dT_b[i], A["dT"][i]))
neg = np.flatnonzero(A["alpha_deg"] < 0)
if neg.size:
    say("  alpha < 0 outboard of r/R = %.3f  (%d of %d stations, min %.2f deg)"
        % (r[neg[0]] / R, neg.size, r.size, A["alpha_deg"].min()))
    say("  thrust from those stations: %+.4f N of %+.4f N total"
        % (float(np.sum((A["dT"] * w_r)[neg])), A["T"]))

json.dump({"name": NAME, "time": lat, "ct_cfd": CT_CFD, "ct_bemt": ct_b,
           "cp_bemt": cp_b, "ct_meas": CT_MEAS,
           "ct_reprice_noswirl": A["ct"], "cp_reprice_noswirl": A["cp"],
           "ct_reprice_swirl": B["ct"], "cp_reprice_swirl": B["cp"],
           "closure_noswirl": A["ct"] / CT_CFD, "closure_swirl": B["ct"] / CT_CFD,
           "r_over_R": (r / R).tolist(),
           "beta_deg": np.degrees(beta).tolist(),
           "v_ax_cfd": v_ax.tolist(), "u_swirl_cfd": u_sw.tolist(),
           "v_bemt": v_bemt.tolist(),
           "alpha_cfd_deg": A["alpha_deg"].tolist(),
           "alpha_cfd_swirl_deg": B["alpha_deg"].tolist(),
           "alpha_bemt_deg": np.degrees(beta - phi_b).tolist(),
           "cl_cfd": A["cl"].tolist(), "re_cfd": A["re"].tolist(),
           "planes_y_over_R": list(PLANES),
           "v_ax_planes": {str(p): np.where(np.isnan(prof[p][0]), None,
                                            prof[p][0]).tolist() for p in PLANES},
           "rg_over_R": (RG / R).tolist()},
          open("runs/rung0_" + NAME + ".json", "w"), indent=1)
OUT.close()
