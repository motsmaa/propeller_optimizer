import numpy as np

from propopt.optimize import optimize
from propopt.problem_spec import radial_spec
from propopt.solvers.null import NullAeroSolver, NullStructSolver


def _run(seed: int):
    spec = radial_spec()
    return optimize(spec, NullAeroSolver(), NullStructSolver(),
                    pop_size=40, n_gen=20, seed=seed)


def test_same_seed_is_reproducible():
    a, b = _run(0), _run(0)
    assert a.F.shape == b.F.shape
    assert np.array_equal(a.F, b.F)


def test_different_seed_differs():
    a, c = _run(0), _run(1)
    assert not (a.F.shape == c.F.shape and np.array_equal(a.F, c.F))
