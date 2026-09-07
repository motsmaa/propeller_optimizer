"""TEST 1: recover thrust from the FAR-FIELD momentum balance, not the blade.

    T = -[ rho * sum_f phi_f U_y,f  +  (integral p n dA)_y ]

phi = (U.n)A outward-positive on boundary faces; the pressure term comes from a
`forces` object on the FARFIELD patch, which integrates rho*p*Sf with Sf
outward. Viscous traction on the far field is neglected (near-quiescent flow).

This touches none of the blade-force code path. Reported-vs-momentum thrust
therefore separates "the flow carries only that thrust" from "the blade
integration under-reports / momentum leaks at the MRF interface".
"""
import json, re
from pathlib import Path
from propopt.contracts import SolverError
from propopt.hf.openfoam import _wsl_sh, make_case, run_foam
from propopt.hf.wsl import _quote, to_wsl_path

RHO, D = 1.225, 0.254
CASES = {"good5": ("uiuc_good5", 1.5804, 0.1200, 4590.0),
         "J035":  ("adv_J035",   1.4848, 0.0709, 5018.0),
         "J046":  ("adv_J046",   0.9255, 0.0486, 5018.0)}

FN = """
functions
{
    momFlux
    {
        type surfaceFieldValue; libs (fieldFunctionObjects);
        regionType patch; name farfield; operation weightedSum;
        weightField phi; fields (U);
        writeFields false; log true; writeControl writeTime;
    }
    massFlux
    {
        type surfaceFieldValue; libs (fieldFunctionObjects);
        regionType patch; name farfield; operation sum; fields (phi);
        writeFields false; log true; writeControl writeTime;
    }
    farForce
    {
        type forces; libs (forces); patches (farfield);
        rho rhoInf; rhoInf %.4f; CofR (0 0 0);
        writeControl writeTime; log true;
    }
}
""" % RHO

out = {}
for tag, (cdir, T_rep, ct_meas, rpm) in CASES.items():
    src = Path("runs/" + cdir)
    case = make_case(cdir, src)
    cd = src / "system" / "controlDict"
    cd.write_text(re.sub(r"functions\s*\{.*", FN, cd.read_text(), flags=re.S))
    _wsl_sh("cp " + _quote(to_wsl_path(cd)) + " " + _quote(case.work)
            + "/system/controlDict")
    run_foam("simpleFoam", case, ("-postProcess", "-latestTime"), timeout=14400,
             log_name="log.momentum", measure_rss=False)
    case.sync_out()

    def vec(marker):
        ln = [l for l in (src / "log.momentum").read_text().splitlines()
              if marker in l]
        if not ln:
            raise SolverError(tag + ": no output for " + marker)
        m = re.search(r"=\s*\(([^)]*)\)", ln[-1])
        if not m:
            raise SolverError(tag + ": non-vector: " + ln[-1][:140])
        return [float(x) for x in m.group(1).split()]

    ln = [l for l in (src / "log.momentum").read_text().splitlines()
          if "sum(farfield) of phi" in l]
    mdot = float(ln[-1].split("=")[-1])
    mom = vec("weightedSum(farfield) of U")
    # pressure force on the far-field patch, straight from force.dat
    ff = sorted((src / "postProcessing" / "farForce").glob("*/force.dat"))
    if not ff:
        raise SolverError(tag + ": no farForce/force.dat")
    fr = [l for l in ff[-1].read_text().splitlines()
          if l.strip() and not l.startswith("#")]
    v = [float(x) for x in fr[-1].replace("(", " ").replace(")", " ").split()]
    p_y, visc_y = v[5], v[8]           # pressure_y, viscous_y
    T_mom = -(RHO * mom[1] + p_y)
    n = rpm / 60.0
    q = RHO * n ** 2 * D ** 4
    out[tag] = dict(T_reported=T_rep, T_momentum=T_mom, ratio=T_mom / T_rep,
                    ct_reported=abs(T_rep) / q, ct_momentum=abs(T_mom) / q,
                    ct_meas=ct_meas, mass_flux=mdot,
                    momflux_y=RHO * mom[1], pforce_y=p_y, visc_y=visc_y)
    print("%-6s T_blade=%+.4f N  T_momentum=%+.4f N  ratio=%.3f  net mdot=%.2e"
          % (tag, T_rep, T_mom, T_mom / T_rep, mdot), flush=True)
    print("       CT_blade=%.4f  CT_momentum=%.4f  CT_measured=%.4f"
          % (abs(T_rep) / q, abs(T_mom) / q, ct_meas), flush=True)
    print("       [rho*momflux_y=%+.5f  pforce_y=%+.5f  visc_y=%+.5f N]"
          % (RHO * mom[1], p_y, visc_y), flush=True)
json.dump(out, open("runs/momentum_check.json", "w"), indent=1)
