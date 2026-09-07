"""TEST 4: does BLADE-ELEMENT THEORY with the 4412 proxy also predict the low
forward-flight thrust? If so, rung 1's "second problem" is the SECTION PROXY,
not a rotor-level CFD defect.

propopt's BEMT is hover-only (V=0 hardwired). This is the standard forward-
flight extension, same blade-element integral and same polar seam, with the
axial momentum balance carrying the freestream:

    dT_BE  = 1/2 rho W^2 B c (Cl cos phi - Cd sin phi)
    dT_mom = 4 pi r rho (V + v) v F dr        (F = Prandtl tip/root loss)
    W = hypot(V+v, Omega r),  phi = atan2(V+v, Omega r),  alpha = beta - phi

The question it settles: at J=0.35-0.46 the blade sits near zero incidence, so
the section's ZERO-LIFT ANGLE dominates the answer. If the 4412 proxy predicts
CT close to the CFD's rather than to measurement, then the CFD is reproducing
blade-element theory correctly and the gap is the proxy, not the solver.
"""
import numpy as np
from pathlib import Path
from propopt.solvers.bemt import MU_AIR, _prandtl_factor
from propopt.solvers.polars import TabulatedPolar
from propopt.stations import station_weights
from propopt.validation.uiuc import design_from_geometry, load_geometry

RHO, D, R, NB = 1.225, 0.254, 0.127, 2
PTS = [("J035", 5018.0, 0.349, 0.0709, 0.0413, 0.0416),
       ("J046", 5018.0, 0.460, 0.0486, 0.0351, 0.0260)]
xf = sorted(Path("data/polars").glob("naca4412_re*.txt"))
POLARS = {
    "XFOIL Re>=5e4": TabulatedPolar.from_xfoil_files(
        [p for p in xf if int(p.stem.split("re")[1]) >= 50_000]),
    "XFOIL extended": TabulatedPolar.from_xfoil_files(xf),
}


def fwd_bemt(design, polar, V, rpm, relax=0.2, nit=2000):
    r = np.asarray(design.station_r_m, float)
    c = np.asarray(design.chord_m, float)
    beta = np.asarray(design.twist_rad, float)
    w_r = station_weights(r, design.hub_radius_m, R)
    om = rpm * 2 * np.pi / 60.0
    ut = om * r
    v = np.full_like(r, 0.5)
    for _ in range(nit):
        va = V + v
        phi = np.arctan2(va, ut)
        W = np.hypot(va, ut)
        cl, cd = polar(beta - phi, RHO * W * c / MU_AIR)
        sp, cp = np.sin(phi), np.cos(phi)
        ss = np.maximum(sp, 1e-6)
        F = np.maximum(_prandtl_factor(0.5 * NB * (R - r) / (r * ss))
                       * _prandtl_factor(0.5 * NB * (r - design.hub_radius_m)
                                         / (r * ss)), 1e-3)
        dT = 0.5 * RHO * W ** 2 * NB * c * (cl * cp - cd * sp)
        # 4 pi r rho (V+v) v F = dT/dr  ->  v^2 + V v - dT/(4 pi r rho F) = 0
        k = np.maximum(dT, 0.0) / (4 * np.pi * r * RHO * F)
        vn = 0.5 * (-V + np.sqrt(V ** 2 + 4 * k))
        if np.max(np.abs(vn - v)) < 1e-9 * max(om * R, 1.0):
            v = vn
            break
        v = v + relax * (vn - v)
    va = V + v
    phi = np.arctan2(va, ut)
    W = np.hypot(va, ut)
    cl, cd = polar(beta - phi, RHO * W * c / MU_AIR)
    sp, cp = np.sin(phi), np.cos(phi)
    q = 0.5 * RHO * W ** 2 * NB * c
    T = float(np.sum(q * (cl * cp - cd * sp) * w_r))
    Q = float(np.sum(q * (cl * sp + cd * cp) * r * w_r))
    return T, Q * om, np.degrees(beta - phi), v, cl


geom = load_geometry(Path("data/uiuc/apcsf_10x4.7_geom.txt"))
print("FORWARD-FLIGHT BLADE-ELEMENT THEORY with the NACA 4412 proxy")
for tag, rpm, J, ct_m, cp_m, ct_cfd in PTS:
    n = rpm / 60.0
    V = J * n * D
    d = design_from_geometry(geom, D, NB, rpm, n_stations=20)
    print("")
    print("%s  J=%.3f  V=%.3f m/s   measured CT=%.4f   CFD CT=%.4f"
          % (tag, J, V, ct_m, ct_cfd))
    for pn, pol in POLARS.items():
        T, P, al, v, cl = fwd_bemt(d, pol, V, rpm)
        ct = T / (RHO * n ** 2 * D ** 4)
        cp = P / (RHO * n ** 3 * D ** 5)
        print("  %-16s CT=%+.4f  CP=%.4f  |  vs measured %+.1f%%  vs CFD %+.1f%%"
              % (pn, ct, cp, 100 * (ct / ct_m - 1), 100 * (ct / ct_cfd - 1)))
        rr = np.asarray(d.station_r_m) / R
        idx = [np.argmin(abs(rr - x)) for x in (0.3, 0.5, 0.7, 0.9)]
        print("     alpha at r/R 0.3/0.5/0.7/0.9 = "
              + " ".join("%+.2f" % al[i] for i in idx)
              + " deg   Cl = " + " ".join("%+.3f" % cl[i] for i in idx))
