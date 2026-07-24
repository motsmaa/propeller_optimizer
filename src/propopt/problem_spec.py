"""Config-driven problem specification.

This is the ONLY place that knows the genome is radial. The optimizer reads
n_var / bounds / n_obj / n_constr from a ProblemSpec and never hardcodes a
station count, so a later free-form-loop genome is a swap of this module alone.

Objectives bundle (name, sense, fn) together so the optimization direction and
the human-facing name can never drift out of sync — adding the Phase-6 noise
objective is one tuple entry, with no parallel sign array to maintain.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from .contracts import AeroSolver, EvalResult, PropellerDesign, StructResult
from .motor import DEFAULT_MOTOR, MotorSpec, solve_torque_equilibrium
from .stations import uniform_midpoint

ObjFn = Callable[[EvalResult, StructResult], float]
SpacingFn = Callable[[float, float, int], tuple[float, ...]]  # (hub, tip, n) -> radii


class Sense(Enum):
    MIN = 1   # multiplier mapping physical value -> pymoo MIN form
    MAX = -1  # (the value squares to 1, so it is its own inverse)


@dataclass(frozen=True)
class Objective:
    name: str
    sense: Sense
    fn: ObjFn


@dataclass(frozen=True)
class Constraint:
    name: str
    fn: ObjFn  # g(.) <= 0 is feasible (pymoo convention)


@dataclass(frozen=True)
class ProblemSpec:
    n_var: int
    xl: np.ndarray
    xu: np.ndarray
    decode: Callable[[np.ndarray], PropellerDesign]
    objectives: tuple[Objective, ...]
    constraints: tuple[Constraint, ...]

    @property
    def n_obj(self) -> int:
        return len(self.objectives)

    @property
    def n_constr(self) -> int:
        return len(self.constraints)

    @property
    def obj_names(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.objectives)

    @property
    def senses(self) -> np.ndarray:
        return np.array([o.sense.value for o in self.objectives], dtype=float)

    def to_min_F(self, ev: EvalResult, st: StructResult) -> np.ndarray:
        return np.array([o.sense.value * o.fn(ev, st) for o in self.objectives], dtype=float)

    def eval_constraints(self, ev: EvalResult, st: StructResult) -> np.ndarray:
        return np.array([c.fn(ev, st) for c in self.constraints], dtype=float)

    def to_physical(self, f_min: np.ndarray) -> np.ndarray:
        """Recover physical objective values from MIN-form: F_min * senses."""
        return np.asarray(f_min, dtype=float) * self.senses


def radial_spec(
    n_stations: int = 6,
    radius_m: float = 0.127,
    hub_radius_m: float = 0.01,
    n_blades: int = 2,
    chord_bounds: tuple[float, float] = (0.005, 0.030),
    twist_bounds: tuple[float, float] = (0.0, 0.6),
    rpm_bounds: tuple[float, float] = (2000.0, 12000.0),
    min_thrust_N: float = 2.0,
    spacing: SpacingFn = uniform_midpoint,
    resonance_margin_min: float = 0.15,
    motor: MotorSpec | None = DEFAULT_MOTOR,
    aero_for_rpm: AeroSolver | None = None,
    throttle_bounds: tuple[float, float] = (0.2, 1.0),
) -> ProblemSpec:
    """Radial-station spec.

    Phase 2c — MOTOR MODE (default, `motor` set): the last genome slot is
    THROTTLE in [throttle_min, 1]; rpm is solved per design as the
    motor-propeller torque equilibrium (see motor.py) inside `decode`, using
    `aero_for_rpm` (a BEMTAeroSolver by default) for Q_prop. Genome:
    [chord_0..n-1, twist_0..n-1, throttle].

    FREE-RPM MODE (`motor=None`): the pre-2c behaviour — the last slot is rpm
    in `rpm_bounds`, sampled directly. Kept for optimizer-plumbing/geometry
    tests that predate the motor model. Genome: [..., rpm].

    Station radii are fixed per spec by `spacing` (stations.py) and carried on
    every decoded design via PropellerDesign.station_r_m.
    Objectives: max thrust, max FoM, min mass, max safety factor.
    Constraints: safety_factor >= 1, thrust >= min_thrust_N,
    resonance_margin >= resonance_margin_min, and (motor mode only)
    motor current at equilibrium <= I_max.

    `throttle_min` = 0.2: below ~20% throttle the first-order motor model is
    least trustworthy (ESC PWM, cogging, back-EMF sensing all degrade) and the
    delivered thrust is negligible; it also keeps the equilibrium comfortably
    above the BEMT low-rpm floor.

    `resonance_margin_min` = 0.15 is a DELIBERATELY CONSERVATIVE first-pass
    value: the first-pass beam f1 estimate is itself uncertain (isotropic
    PLA, rigid hub, no fillet), so the keep-out band is wide on purpose —
    revisit when Phase-4 FEA calibrates the modal model.

    A2 RESOLUTION (Path 2): with `motor` set, rpm is no longer a free lever the
    optimizer can slide to satisfy the resonance constraint — it must earn
    resonance margin structurally (stiffen the blade) or by trading throttle
    (which also moves thrust). Throttle is the new operating relief valve;
    run_demo reports its distribution as the analogue of the pre-2c
    rpm-distribution diagnostic.
    """
    station_r = spacing(hub_radius_m, radius_m, n_stations)
    n_var = 2 * n_stations + 1
    last_bounds = throttle_bounds if motor is not None else rpm_bounds
    xl = np.concatenate([
        np.full(n_stations, chord_bounds[0]),
        np.full(n_stations, twist_bounds[0]),
        [last_bounds[0]],
    ])
    xu = np.concatenate([
        np.full(n_stations, chord_bounds[1]),
        np.full(n_stations, twist_bounds[1]),
        [last_bounds[1]],
    ])

    aero_rpm = aero_for_rpm
    if motor is not None and aero_rpm is None:
        from .solvers.bemt import BEMTAeroSolver  # local: avoid import cycle
        aero_rpm = BEMTAeroSolver()

    def _build(rpm: float, chord, twist) -> PropellerDesign:
        return PropellerDesign(radius_m, hub_radius_m, n_blades, rpm, chord, twist,
                               station_r)

    def decode(x: np.ndarray) -> PropellerDesign:
        chord = tuple(float(v) for v in x[:n_stations])
        twist = tuple(float(v) for v in x[n_stations:2 * n_stations])
        if motor is None:
            return _build(float(x[-1]), chord, twist)
        throttle = float(x[-1])
        # Q_prop(N) via the shared aero; SolverError from a non-bracketing
        # equilibrium (case c) or BEMT non-convergence propagates to decode and
        # is penalized by the optimizer. Over-current (case b) is NOT rejected
        # here — the equilibrium exists; the current constraint flags it.
        def prop_torque(rpm: float) -> float:
            return aero_rpm.evaluate(_build(rpm, chord, twist)).torque_Nm
        rpm = solve_torque_equilibrium(motor, throttle, prop_torque)
        return _build(rpm, chord, twist)

    objectives = (
        Objective("thrust_N", Sense.MAX, lambda ev, st: ev.thrust_N),
        Objective("figure_of_merit", Sense.MAX, lambda ev, st: ev.figure_of_merit),
        Objective("mass_kg", Sense.MIN, lambda ev, st: st.mass_kg),
        Objective("safety_factor", Sense.MAX, lambda ev, st: st.safety_factor),
    )
    constraints = (
        Constraint("safety_factor>=1", lambda ev, st: 1.0 - st.safety_factor),
        Constraint("thrust>=min", lambda ev, st: min_thrust_N - ev.thrust_N),
        Constraint("resonance_margin>=min",
                   lambda ev, st: resonance_margin_min - st.resonance_margin),
    )
    if motor is not None:
        # At equilibrium Q_prop == Q_motor, so I follows from ev.torque_Nm
        # alone — no rpm/design needed, so it fits the bundled (ev, st) seam.
        constraints += (
            Constraint("motor_current<=Imax",
                       lambda ev, st: motor.current_from_torque(ev.torque_Nm)
                       / motor.i_max_a - 1.0),
        )
    return ProblemSpec(n_var, xl, xu, decode, objectives, constraints)
