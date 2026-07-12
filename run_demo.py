"""Phase 2b demonstrator: NSGA-II with ALL FOUR objectives physical.

Thrust/FoM from BEMT, mass from the lofted-solid volume, safety factor from
the rotating-cantilever beam FE (PLA, Kt root guard). Resonance is a
feasibility CONSTRAINT (margin >= 0.15 vs blade-passing harmonics). NOTE the
A2 coupling: rpm is a free design variable, so the optimizer may satisfy
resonance by shifting rpm rather than stiffening blades — the printed rpm
range makes that lever visible."""
from propopt.contracts import SolverError
from propopt.optimize import optimize
from propopt.problem_spec import radial_spec
from propopt.solvers.beam import BeamStructSolver
from propopt.solvers.bemt import BEMTAeroSolver


def main() -> None:
    spec = radial_spec()
    beam = BeamStructSolver()
    res = optimize(spec, BEMTAeroSolver(), beam,
                   pop_size=60, n_gen=40, seed=0)
    print(f"Pareto points: {len(res)}   failed evals: {res.n_failed}")
    for i, name in enumerate(spec.obj_names):
        col = res.F[:, i]  # physical values
        print(f"  {name:16s} min={col.min():10.4g}  max={col.max():10.4g}")

    margins, f1s, rpms = [], [], []
    for x in res.X:
        design = spec.decode(x)
        try:
            a = beam.analyze(design)
        except SolverError:
            continue
        margins.append(a.resonance_margin)
        f1s.append(a.f1_hz)
        rpms.append(design.rpm)
    near = sum(1 for m in margins if m < 0.15)
    print(f"  f1 range: {min(f1s):.0f}-{max(f1s):.0f} Hz;  "
          f"resonance margin min={min(margins):.3f} max={max(margins):.3f};  "
          f"{near}/{len(margins)} designs inside the 0.15 keep-out band")
    print(f"  rpm range on front: {min(rpms):.0f}-{max(rpms):.0f} "
          "(free variable - the A2 resonance lever)")


if __name__ == "__main__":
    main()
