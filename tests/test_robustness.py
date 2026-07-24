"""A1: a solver that fails on some designs must penalize those candidates rather
than crash the run, and the surviving Pareto front must exclude the failures.
This is the guarantee the null solvers cannot exercise (they never fail)."""
from propopt.contracts import AeroSolver, EvalResult, SolverError
from propopt.optimize import optimize
from propopt.problem_spec import radial_spec
from propopt.solvers.null import NullAeroSolver, NullStructSolver


class FlakyAero(AeroSolver):
    """Wraps a base solver but diverges above an rpm threshold (SolverError path)."""
    def __init__(self, base: AeroSolver, rpm_limit: float = 9000.0):
        self._base = base
        self._rpm_limit = rpm_limit

    def evaluate(self, design):
        if design.rpm > self._rpm_limit:
            raise SolverError("did not converge at high rpm")
        return self._base.evaluate(design)


class NanAero(AeroSolver):
    """Returns NaN -> EvalResult sanity check raises (SanityError path)."""
    def evaluate(self, design):
        return EvalResult(float("nan"), 1.0, 1.0, 0.5)


def test_flaky_solver_does_not_crash_and_excludes_failures():
    spec = radial_spec(motor=None)  # free-rpm: FlakyAero gates on the rpm gene
    res = optimize(spec, FlakyAero(NullAeroSolver()), NullStructSolver(),
                   pop_size=60, n_gen=30, seed=0)
    assert res.n_failed > 0                       # some candidates failed
    assert len(res) > 0                           # run still produced a real front
    assert (res.X[:, -1] <= 9000.0 + 1e-6).all()  # survivors are below the rpm limit


def test_all_failing_solver_completes_without_raising():
    spec = radial_spec(motor=None)  # free-rpm: FlakyAero gates on the rpm gene
    res = optimize(spec, NanAero(), NullStructSolver(),
                   pop_size=20, n_gen=10, seed=0)
    assert res.n_failed > 0                        # every eval was caught + penalized
