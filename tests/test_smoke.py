import numpy as np

from propopt.optimize import optimize
from propopt.problem_spec import radial_spec
from propopt.solvers.null import NullAeroSolver, NullStructSolver


def test_end_to_end_emits_pareto_front():
    spec = radial_spec()
    res = optimize(spec, NullAeroSolver(), NullStructSolver(),
                   pop_size=40, n_gen=20, seed=0)
    assert len(res) > 0                      # DoD: a front exists
    assert res.F.shape[1] == spec.n_obj      # correct objective count
    assert res.X.shape[1] == spec.n_var
    assert np.isfinite(res.F).all()
    assert res.n_failed == 0                 # null solvers never fail


def test_front_is_non_degenerate():
    spec = radial_spec()
    res = optimize(spec, NullAeroSolver(), NullStructSolver(),
                   pop_size=60, n_gen=30, seed=0)
    spreads = res.F.max(axis=0) - res.F.min(axis=0)   # res.F is physical
    assert (spreads > 0).sum() >= 2          # genuine trade-offs on >= 2 objectives
