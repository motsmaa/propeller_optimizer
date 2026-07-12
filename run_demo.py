"""Phase 2a demonstrator: NSGA-II + BEMT aero + real geometric mass.

Thrust/FoM (BEMT) and mass (lofted-solid volume x density) are real physics;
only safety factor / stress remain Phase-0 placeholders (Phase 2b)."""
from propopt.optimize import optimize
from propopt.problem_spec import radial_spec
from propopt.solvers.bemt import BEMTAeroSolver
from propopt.solvers.geom_mass import GeometryMassStructSolver


def main() -> None:
    spec = radial_spec()
    res = optimize(spec, BEMTAeroSolver(), GeometryMassStructSolver(),
                   pop_size=60, n_gen=40, seed=0)
    print(f"Pareto points: {len(res)}   failed evals: {res.n_failed}")
    for i, name in enumerate(spec.obj_names):
        col = res.F[:, i]  # physical values
        print(f"  {name:16s} min={col.min():10.4g}  max={col.max():10.4g}")


if __name__ == "__main__":
    main()
