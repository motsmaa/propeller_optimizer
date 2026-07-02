"""Phase 0 demonstrator: NSGA-II + null evaluators -> Pareto front summary."""
from propopt.optimize import optimize
from propopt.problem_spec import radial_spec
from propopt.solvers.null import NullAeroSolver, NullStructSolver


def main() -> None:
    spec = radial_spec()
    res = optimize(spec, NullAeroSolver(), NullStructSolver(),
                   pop_size=60, n_gen=40, seed=0)
    print(f"Pareto points: {len(res)}   failed evals: {res.n_failed}")
    for i, name in enumerate(spec.obj_names):
        col = res.F[:, i]  # physical values
        print(f"  {name:16s} min={col.min():10.4g}  max={col.max():10.4g}")


if __name__ == "__main__":
    main()
