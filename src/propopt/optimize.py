"""Optimizer layer. Wires NSGA-II to whatever AeroSolver/StructSolver are injected,
through the ProblemSpec only. Knows nothing about radial stations or concrete solvers.

A failed evaluation (recoverable solver error, out-of-range result, arithmetic
blow-up) is converted into a penalized candidate so a single bad design cannot
kill a long run. Programming errors (TypeError, AttributeError, ...) are NOT
caught here — they should surface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from .contracts import AeroSolver, SolverError, StructSolver
from .problem_spec import ProblemSpec
from .units import SanityError

_RECOVERABLE = (SanityError, SolverError, ArithmeticError)


class PropellerProblem(ElementwiseProblem):
    def __init__(self, spec: ProblemSpec, aero: AeroSolver, struct: StructSolver,
                 penalty: float = 1e6):
        super().__init__(
            n_var=spec.n_var, n_obj=spec.n_obj,
            n_ieq_constr=spec.n_constr, xl=spec.xl, xu=spec.xu,
        )
        self._spec = spec
        self._aero = aero
        self._struct = struct
        self._penalty = float(penalty)
        self.n_failed = 0

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            design = self._spec.decode(x)
            ev = self._aero.evaluate(design)
            st = self._struct.evaluate(design)
            out["F"] = self._spec.to_min_F(ev, st)
            out["G"] = self._spec.eval_constraints(ev, st)
        except _RECOVERABLE:
            self.n_failed += 1
            out["F"] = np.full(self._spec.n_obj, self._penalty)
            out["G"] = np.full(self._spec.n_constr, self._penalty)


@dataclass
class ParetoResult:
    X: np.ndarray                 # decision vectors of the non-dominated set
    F: np.ndarray                 # PHYSICAL objective values (not MIN form)
    obj_names: tuple[str, ...]
    n_failed: int = 0             # evaluations penalized due to solver failure

    def __len__(self) -> int:
        return 0 if self.F is None else int(self.F.shape[0])


def optimize(
    spec: ProblemSpec,
    aero: AeroSolver,
    struct: StructSolver,
    pop_size: int = 60,
    n_gen: int = 40,
    seed: int = 0,
    penalty: float = 1e6,
) -> ParetoResult:
    problem = PropellerProblem(spec, aero, struct, penalty=penalty)
    algorithm = NSGA2(pop_size=pop_size)
    res = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
    F_min = res.F if res.F is not None else np.empty((0, spec.n_obj))
    X = res.X if res.X is not None else np.empty((0, spec.n_var))
    F_phys = spec.to_physical(np.atleast_2d(F_min))
    return ParetoResult(np.atleast_2d(X), F_phys, spec.obj_names, problem.n_failed)
