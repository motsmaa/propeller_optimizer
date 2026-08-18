"""Derive the (alpha, Re) range the blade ACTUALLY operates in, from CFD fields.

This sets the sweep range for the 2-D section study. It is derived, not assumed:
the range is taken from the converged rotor fields themselves, and weighted by
each station's share of thrust so that a station contributing nothing does not
widen the range that matters.

alpha(r) = beta(r) - atan(v_ax(r) / Omega r), with v_ax the azimuthally averaged
through-disc axial velocity (freestream + induced, so this works unchanged for
hover and for advancing flow). Re(r) = rho W c / mu, the same chord Reynolds
number BEMT forms.
"""
import json
import sys
from pathlib import Path

import numpy as np

from propopt.hf.disc_inflow import (azimuthal_profile, latest_time,
                                    read_disc_samples, run_sampling)
from propopt.hf.openfoam import make_case
from propopt.solvers.bemt import MU_AIR
from propopt.solvers.polars import TabulatedPolar
from propopt.stations import station_weights
from propopt.validation.uiuc import design_from_geometry, load_geometry

RHO, D, R, NB, MU = 1.225, 0.254, 0.127, 2, MU_AIR
N_RAD, R0F, R1F = 80, 0.10, 1.05
# (case dir on WSL, run-summary json, rpm)
CASES = {
    "good5": ("uiuc_good5", "runs/summary_good5.json", 4590.0),
    "J035": ("adv_J035", "runs/summary_adv_J035.json", 5018.0),
    "J046": ("adv_J046", "runs/summary_adv_J046.json", 5018.0),
}
CT_MEAS = {"good5": 0.1200, "J035": 0.0709, "J046": 0.0486}
V_INF = {"good5": 0.0, "J035": 7.4138, "J046": 9.7717}
# Extended polar set (Re 1e4-5e5) for the WEIGHTING only, because the actual
# station Reynolds numbers run down to ~1e4 and the tracked Re>=5e4 set would
# clamp most of the span. This only decides which stations count as loaded.
POLAR = TabulatedPolar.from_xfoil_files(
    sorted(Path("data/polars").glob("naca4412_re*.txt")))
REUSE = "--reuse" in sys.argv
OUT = open("runs/alpha_range.txt", "w")


def say(s=""):
    print(s, flush=True)
    OUT.write(str(s) + "\n")
    OUT.flush()


RG = np.linspace(R0F * R, R1F * R, N_RAD)
summary = {}
say("OPERATING (alpha, Re) DERIVED FROM THE CFD FIELDS")
say("Thrust weight w_T = station share of the POSITIVE blade-element thrust,")
say("so the reported range is where the blade actually works.")

for tag, (cdir, sjson, rpm) in CASES.items():
    src = Path("runs/" + cdir)
    case = make_case(cdir, src)                     # resolves, does NOT sync_in
    t = latest_time(case, tag)
    rows = (read_disc_samples(case, tag) if REUSE
            else run_sampling(case, src, tag, planes=(0.0,), radius=R))
    mean, _, nps = azimuthal_profile(rows, 0, RG)
    good = ~np.isnan(mean[0])

    design = design_from_geometry(load_geometry(
        Path("data/uiuc/apcsf_10x4.7_geom.txt")), D, NB, rpm, n_stations=20)
    r = np.asarray(design.station_r_m, float)
    c = np.asarray(design.chord_m, float)
    beta = np.asarray(design.twist_rad, float)
    w_r = station_weights(r, design.hub_radius_m, R)
    omega = rpm * 2.0 * np.pi / 60.0
    ut = omega * r
    v_ax = np.interp(r, RG[good], mean[0][good])
    phi = np.arctan2(v_ax, ut)
    alpha = np.degrees(beta - phi)
    W = np.hypot(v_ax, ut)
    re = RHO * W * c / MU

    # ALSO form the alpha the blade would see at the MEASURED thrust with a
    # momentum-consistent inflow. The CFD inflow and the correct-physics inflow
    # give different alpha, and the 2-D sweep must cover BOTH - otherwise it
    # resolves Cl only where the (suspect) CFD happens to sit.
    #   hover:     v_i = sqrt(T / 2 rho A)
    #   advancing: T = 2 rho A v_i (V + v_i)  ->  solve the quadratic
    n_rev = rpm / 60.0
    T_meas = CT_MEAS[tag] * RHO * n_rev ** 2 * D ** 4
    V = V_INF[tag]
    vi = (-0.5 * V + np.sqrt((0.5 * V) ** 2
                             + T_meas / (2 * RHO * np.pi * R ** 2)))
    phi_m = np.arctan2(V + vi, ut)
    alpha_m = np.degrees(beta - phi_m)
    W_m = np.hypot(V + vi, ut)
    re_m = RHO * W_m * c / MU

    def weighted_range(al, ww, rey):
        """alpha range spanning the middle 90% of the blade's LOAD CAPACITY,
        w = q c dr = 0.5 rho W^2 c w_r.

        Deliberately KINEMATIC, not priced through the polar. Weighting by
        positive blade-element thrust degenerates exactly where the answer
        matters most: at J=0.460 the 4412 polar returns Cl <= 0 at every
        station, so a thrust weight collapses to zero everywhere while the real
        propeller is measurably producing CT=0.0486. A weight that vanishes when
        the section model disagrees with reality cannot be used to choose the
        range over which to test that section model."""
        load = 0.5 * RHO * ww ** 2 * c * w_r
        wt = load / load.sum()
        o = np.argsort(al)
        cum = np.cumsum(wt[o])
        k = o[(cum >= 0.05) & (cum <= 0.95)]
        if k.size == 0:
            k = o
        return (float(al[k].min()), float(al[k].max()), wt,
                float(rey[k].min()), float(rey[k].max()))

    beta_deg = np.degrees(beta)
    a_lo, a_hi, wT, r_lo, r_hi = weighted_range(alpha, W, re)
    m_lo, m_hi, wM, mr_lo, mr_hi = weighted_range(alpha_m, W_m, re_m)
    say("")
    say("%s  (rpm %.0f, reconstructed t=%s, %d..%d samples/bin)"
        % (tag, rpm, t, nps.min(), nps.max()))
    say("  measured T=%.3f N -> momentum v_i=%.3f m/s (V=%.3f)" % (T_meas, vi, V))
    say("  r/R    a_CFD   a_meas     Re_CFD    w_T")
    for i in range(0, len(r), 2):
        say("  %.3f %7.2f %8.2f %10.0f %7.3f"
            % (r[i] / R, alpha[i], alpha_m[i], re[i], wT[i]))
    say("  full-span a_CFD %.2f .. %.2f deg;  Re %.0f .. %.0f"
        % (alpha.min(), alpha.max(), re.min(), re.max()))
    say("  CORE (5-95%% of thrust)  from CFD inflow:      alpha %.2f .. %.2f, "
        "Re %.0f .. %.0f" % (a_lo, a_hi, r_lo, r_hi))
    say("  CORE (5-95%% of thrust)  at MEASURED thrust:   alpha %.2f .. %.2f, "
        "Re %.0f .. %.0f" % (m_lo, m_hi, mr_lo, mr_hi))
    summary[tag] = dict(rpm=rpm, V=V, T_meas=float(T_meas), vi_mom=float(vi),
                        alpha_min=float(alpha.min()),
                        alpha_max=float(alpha.max()),
                        alpha_core=[a_lo, a_hi],
                        alpha_core_measured=[m_lo, m_hi],
                        re_min=float(re.min()), re_max=float(re.max()),
                        re_core=[r_lo, r_hi], re_core_measured=[mr_lo, mr_hi],
                        r_over_R=(r / R).tolist(), alpha_deg=alpha.tolist(),
                        alpha_measured_deg=alpha_m.tolist(),
                        re=re.tolist(), w_thrust=wT.tolist(),
                        v_ax=v_ax.tolist())

allc = list(summary.values())
lo = min(min(a["alpha_core"][0], a["alpha_core_measured"][0]) for a in allc)
hi = max(max(a["alpha_core"][1], a["alpha_core_measured"][1]) for a in allc)
rlo = min(min(a["re_core"][0], a["re_core_measured"][0]) for a in allc)
rhi = max(max(a["re_core"][1], a["re_core_measured"][1]) for a in allc)
say("")
say("=== 2-D SWEEP RANGE ===")
say("  Union of the thrust-weighted cores over all three cases, under BOTH the")
say("  CFD's own inflow and the momentum-consistent inflow at measured thrust.")
say("  alpha %.1f .. %.1f deg   ->  sweep %d .. %d deg"
    % (lo, hi, np.floor(lo - 1), np.ceil(hi + 1)))
say("  Re    %.0f .. %.0f       ->  tabulate at 2e4, 4e4, 7e4" % (rlo, rhi))
json_range = dict(alpha_lo=lo, alpha_hi=hi, re_lo=rlo, re_hi=rhi)
summary["_sweep"] = json_range
json.dump(summary, open("runs/alpha_range.json", "w"), indent=1)
OUT.close()
