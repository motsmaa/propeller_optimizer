"""Phase 3: a run is fully specified by a RunConfig + seed.

RunConfig is a frozen dataclass holding every tunable of a run — motor,
material, design space, optimizer size, constraint parameters, and the fixed
hypervolume-normalization bounds. `configs/default.py` constructs one with the
justification for each value next to the value (the reason this is Python, not
JSON). `build_run` turns a RunConfig into the (spec, aero, struct) triple the
optimizer consumes; `run` executes it. Given the same RunConfig and seed the
result is bit-reproducible (NSGA-II is seeded, solvers are deterministic).

contracts.py and optimize.py are untouched: build_run wires the existing
radial_spec + solvers, and `run` calls the existing optimize().
"""
from __future__ import annotations

from dataclasses import dataclass

from .motor import DEFAULT_MOTOR, MotorSpec
from .optimize import ParetoResult, optimize
from .problem_spec import ProblemSpec, radial_spec
from .solvers.beam import PLA, BeamStructSolver, Material
from .solvers.bemt import BEMTAeroSolver

# Objective order is fixed by problem_spec.radial_spec: (thrust, fom, mass, sf).
OBJECTIVES = ("thrust_N", "figure_of_merit", "mass_kg", "safety_factor")


@dataclass(frozen=True)
class RunConfig:
    """Everything needed to reproduce a run. See configs/default.py for values
    with justification."""
    # --- design space ---
    n_stations: int
    radius_m: float
    hub_radius_m: float
    n_blades: int
    chord_bounds: tuple[float, float]
    twist_bounds: tuple[float, float]
    throttle_bounds: tuple[float, float]
    # --- physics ---
    motor: MotorSpec
    material: Material
    kt_root: float
    # --- constraints ---
    min_thrust_N: float
    resonance_margin_min: float
    # --- optimizer ---
    pop_size: int
    n_gen: int
    seed: int
    # --- hypervolume normalization (fixed bounds so HV is comparable across
    # seeds/generations; NOT physical targets — see analysis.hypervolume) ---
    # per objective, in OBJECTIVES order: the ideal (best) and nadir (worst)
    # physical value used to map objectives into a common [0,1] loss.
    hv_ideal: tuple[float, float, float, float]
    hv_nadir: tuple[float, float, float, float]
    # a Pareto design counts as "binding" on a constraint when |g| is within
    # this fraction of that constraint's own threshold scale (see analysis).
    binding_tol_frac: float = 0.02
    # run mode (seed-handling only; all physics/optimizer settings identical):
    #   "exploration" -> single seed `seed` (iteration/demos, exploration-grade)
    #   "decision"    -> aggregate the cross-seed non-dominated union of `seeds`
    mode: str = "exploration"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    def __post_init__(self) -> None:
        if not (0.0 <= self.hub_radius_m < self.radius_m):
            raise ValueError("require 0 <= hub_radius_m < radius_m")
        if self.n_stations < 2:
            raise ValueError("need >= 2 stations")
        for lo, hi in (self.chord_bounds, self.twist_bounds, self.throttle_bounds):
            if not lo < hi:
                raise ValueError("every bound must have lo < hi")
        if self.pop_size < 2 or self.n_gen < 1:
            raise ValueError("pop_size >= 2 and n_gen >= 1 required")
        # nadir must be strictly worse than ideal on every objective so the
        # normalized loss is well-defined and feasible points map inside [0,1)
        for name, a, b in zip(OBJECTIVES, self.hv_ideal, self.hv_nadir):
            if a == b:
                raise ValueError(f"hv_ideal == hv_nadir for {name}")
        if self.mode not in ("exploration", "decision"):
            raise ValueError(f"mode must be exploration|decision, got {self.mode!r}")
        if len(self.seeds) < 1:
            raise ValueError("seeds must be non-empty")


def build_run(cfg: RunConfig) -> tuple[ProblemSpec, BEMTAeroSolver, BeamStructSolver]:
    """Construct the (spec, aero, struct) triple from a config. ONE BEMT
    instance is shared across the equilibrium root-find, aero, and the beam so
    its converged-state memo is reused (see bemt.py)."""
    aero = BEMTAeroSolver()
    struct = BeamStructSolver(material=cfg.material, aero=aero, kt_root=cfg.kt_root)
    spec = radial_spec(
        n_stations=cfg.n_stations, radius_m=cfg.radius_m,
        hub_radius_m=cfg.hub_radius_m, n_blades=cfg.n_blades,
        chord_bounds=cfg.chord_bounds, twist_bounds=cfg.twist_bounds,
        min_thrust_N=cfg.min_thrust_N,
        resonance_margin_min=cfg.resonance_margin_min,
        motor=cfg.motor, aero_for_rpm=aero, throttle_bounds=cfg.throttle_bounds,
    )
    return spec, aero, struct


def run(cfg: RunConfig, seed: int | None = None) -> ParetoResult:
    """Execute the optimization for a config (seed overrides cfg.seed)."""
    spec, aero, struct = build_run(cfg)
    return optimize(spec, aero, struct, pop_size=cfg.pop_size,
                    n_gen=cfg.n_gen, seed=cfg.seed if seed is None else seed)
