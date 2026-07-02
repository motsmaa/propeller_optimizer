"""propopt — generative propeller design optimizer (Phase 0 scaffolding)."""
from .contracts import (
    AeroSolver,
    EvalResult,
    PropellerDesign,
    SolverError,
    StructResult,
    StructSolver,
)
from .optimize import ParetoResult, optimize
from .problem_spec import Constraint, Objective, ProblemSpec, Sense, radial_spec

__all__ = [
    "AeroSolver", "StructSolver", "SolverError",
    "PropellerDesign", "EvalResult", "StructResult",
    "ProblemSpec", "Objective", "Constraint", "Sense", "radial_spec",
    "optimize", "ParetoResult",
]
