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

from .contracts import EvalResult, PropellerDesign, StructResult
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
) -> ProblemSpec:
    """Radial-station spec. Genome: [chord_0..chord_{n-1}, twist_0..twist_{n-1}, rpm].

    Station radii are fixed per spec by the `spacing` scheme (see stations.py)
    and carried on every decoded design via PropellerDesign.station_r_m.
    Objectives: max thrust, max FoM, min mass, max safety factor.
    Constraints: safety_factor >= 1, thrust >= min_thrust_N.
    """
    station_r = spacing(hub_radius_m, radius_m, n_stations)
    n_var = 2 * n_stations + 1
    xl = np.concatenate([
        np.full(n_stations, chord_bounds[0]),
        np.full(n_stations, twist_bounds[0]),
        [rpm_bounds[0]],
    ])
    xu = np.concatenate([
        np.full(n_stations, chord_bounds[1]),
        np.full(n_stations, twist_bounds[1]),
        [rpm_bounds[1]],
    ])

    def decode(x: np.ndarray) -> PropellerDesign:
        chord = tuple(float(v) for v in x[:n_stations])
        twist = tuple(float(v) for v in x[n_stations:2 * n_stations])
        rpm = float(x[-1])
        return PropellerDesign(radius_m, hub_radius_m, n_blades, rpm, chord, twist,
                               station_r)

    objectives = (
        Objective("thrust_N", Sense.MAX, lambda ev, st: ev.thrust_N),
        Objective("figure_of_merit", Sense.MAX, lambda ev, st: ev.figure_of_merit),
        Objective("mass_kg", Sense.MIN, lambda ev, st: st.mass_kg),
        Objective("safety_factor", Sense.MAX, lambda ev, st: st.safety_factor),
    )
    constraints = (
        Constraint("safety_factor>=1", lambda ev, st: 1.0 - st.safety_factor),
        Constraint("thrust>=min", lambda ev, st: min_thrust_N - ev.thrust_N),
    )
    return ProblemSpec(n_var, xl, xu, decode, objectives, constraints)
