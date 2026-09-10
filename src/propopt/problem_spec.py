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

from .contracts import (BLADE_TOPOLOGIES, AeroSolver, Design, EvalResult,
                        LoopDesign, PropellerDesign, SolverError, StructResult)
from .motor import DEFAULT_MOTOR, MotorSpec, solve_torque_equilibrium
from .loop_geometry import (SweptFrames, blade_clearance_margin,
                            curvature_margin, hub_junction_margin,
                            section_radius, seed_normal,
                            self_intersection_margin, spine_frames)
from .stations import uniform_midpoint

ObjFn = Callable[[EvalResult, StructResult], float]
GeomFn = Callable[[Design], float]   # g(design) <= 0 is feasible
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
class GeomConstraint:
    """A constraint on the DESIGN alone, needing no solver result.

    Separate from `Constraint` because its signature is different: geometric
    feasibility (self-intersection, blade clearance, hub closure) is knowable
    from the decoded design, before and independently of any aero or structural
    evaluation. Kept OUT of `constraints` so `ProblemSpec.n_constr` - which
    `optimize.py` passes to pymoo as `n_ieq_constr` - is unchanged, and the
    radial path keeps exactly the constraint count it had."""
    name: str
    fn: GeomFn


@dataclass(frozen=True)
class ProblemSpec:
    n_var: int
    xl: np.ndarray
    xu: np.ndarray
    decode: Callable[[np.ndarray], PropellerDesign]
    objectives: tuple[Objective, ...]
    constraints: tuple[Constraint, ...]
    # ADDITIVE and defaulted: `radial_spec` constructs ProblemSpec positionally
    # and does not pass this, so its signature, n_constr and behaviour are
    # untouched. Nothing evaluates these inside the optimizer loop yet - wiring
    # them into pymoo's G would change optimize.py, which is a separate
    # propose-and-stop.
    geom_constraints: tuple[GeomConstraint, ...] = ()

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

    def eval_geom(self, design: Design) -> np.ndarray:
        """Geometric feasibility of a decoded design; g <= 0 feasible."""
        return np.array([g.fn(design) for g in self.geom_constraints], dtype=float)

    @property
    def geom_names(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.geom_constraints)


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


def loop_spec(
    n_ctrl: int = 5,
    n_stations: int = 8,
    radius_m: float = 0.127,
    hub_radius_m: float = 0.01,
    n_blades: int = 2,
    topology: str = "loop_hub",
    chord_bounds: tuple[float, float] = (0.006, 0.026),
    twist_bounds: tuple[float, float] = (-0.35, 0.60),
    rpm_bounds: tuple[float, float] = (2000.0, 12000.0),
    axial_bound_radii: float = 0.60,
    te_thickness_m: float = 2.0e-4,
) -> ProblemSpec:
    """Free-form 3-D loop spec (Phase 5, slice 5a) — GEOMETRY ONLY.

    Sibling of `radial_spec` behind the same `ProblemSpec` interface. It
    carries NO objectives and NO constraints: there is no loop-aware aero or
    structural solver yet, and inventing placeholder objectives would let a
    later reader mistake them for physics. Geometric feasibility is exposed
    through `geom_constraints`, which nothing in the optimizer loop reads.

    GENOME, in cylindrical coordinates about the rotation axis:
        [r_1..r_K, theta_0..theta_{K-1}, y_0..y_{K-1}, chord_0..chord_{N-1},
         twist_0..twist_{N-1}, rpm]
    where K = n_ctrl. Control points are (r, theta, y) -> (r cos t, y, r sin t)
    in the codebase frame (x tangential, y axial, z radial).

    WHY CYLINDRICAL, AND WHY r IS BOX-BOUNDED. `LoopDesign` validates that every
    control point lies inside the radius envelope and, for `loop_hub`, that both
    endpoints sit on the hub cylinder — as ValueError, because a malformed
    design is a programming error. A genome must never be able to reach those:
    a ValueError propagating out of `decode` would abort the run instead of
    being penalized. Encoding r directly and box-bounding it to
    [hub_radius_m, radius_m] makes the envelope check unreachable by
    construction, and for `loop_hub` r on the FIRST and LAST control points is
    PINNED to hub_radius_m rather than encoded, which makes the endpoint check
    unreachable too. Hence r_1..r_K carries only the free radii — K-2 of them
    for `loop_hub`, K otherwise.

    `n_ctrl = 5` (justified, not guessed): a clamped cubic needs >= 4. At 4 the
    curve is a single Bezier segment whose endpoint tangents are forced along
    P0->P1 and P2->P3, so with both endpoints pinned to the hub the only
    remaining freedom is the two interior points, and the family cannot produce
    an out-and-back arc with independent foot directions. 5 adds one interior
    knot span, which is the minimum that lets the outbound and return legs bend
    independently. Starting minimal is deliberate: more control points widen the
    reachable family but also raise the self-intersection rate, and the gate is
    meant to measure the base topology, not a tuned one.

    `twist_bounds` opens to -0.35 rad, unlike the radial spec's 0.0 floor: a
    loop's return leg travels the other way round the rotation, so its sections
    need the opposite sign to make thrust in the same direction.

    AXIAL EXTENT is bounded here, in the genome box (`axial_bound_radii`), not
    in the contract — `Design` deliberately leaves it unbounded because it is a
    design-quality question, not a well-formedness one.
    """
    if topology not in BLADE_TOPOLOGIES:
        raise ValueError(f"topology must be one of {BLADE_TOPOLOGIES}, "
                         f"got {topology!r}")
    if n_ctrl < 4:
        raise ValueError(f"a cubic spine needs >= 4 control points, got {n_ctrl}")
    if n_stations < 2:
        raise ValueError(f"a swept design needs >= 2 stations, got {n_stations}")

    pinned = topology == "loop_hub"
    n_free_r = n_ctrl - 2 if pinned else n_ctrl
    y_max = axial_bound_radii * radius_m
    n_var = n_free_r + 2 * n_ctrl + 2 * n_stations + 1

    xl = np.concatenate([
        np.full(n_free_r, hub_radius_m),
        np.full(n_ctrl, -np.pi), np.full(n_ctrl, -y_max),
        np.full(n_stations, chord_bounds[0]),
        np.full(n_stations, twist_bounds[0]), [rpm_bounds[0]]])
    xu = np.concatenate([
        np.full(n_free_r, radius_m),
        np.full(n_ctrl, np.pi), np.full(n_ctrl, y_max),
        np.full(n_stations, chord_bounds[1]),
        np.full(n_stations, twist_bounds[1]), [rpm_bounds[1]]])

    # Stations are placed at uniform arc-length fractions. The endpoints are
    # INCLUDED (s = 0 and 1): for a hub-closing loop those are the two feet,
    # which are exactly the sections that must exist for the caps to close.
    station_s = tuple(float(v) for v in np.linspace(0.0, 1.0, n_stations))

    def decode(x: np.ndarray) -> LoopDesign:
        x = np.asarray(x, dtype=float)
        i = 0
        r_free = x[i:i + n_free_r]; i += n_free_r
        th = x[i:i + n_ctrl]; i += n_ctrl
        y = x[i:i + n_ctrl]; i += n_ctrl
        chord = x[i:i + n_stations]; i += n_stations
        twist = x[i:i + n_stations]; i += n_stations
        rpm = float(x[i])

        r = (np.concatenate([[hub_radius_m], r_free, [hub_radius_m]])
             if pinned else r_free)
        ctrl = tuple((float(ri * np.cos(ti)), float(yi), float(ri * np.sin(ti)))
                     for ri, ti, yi in zip(r, th, y))
        if topology != "loop_hub":
            raise SolverError(
                f"topology {topology!r} is not implemented in slice 5a. "
                "'loop_neighbour' (a blade closing on the next blade's root) "
                "needs an inter-blade junction the sweep generator does not "
                "build, and 'open' (a free tip) needs a tip-closure rule that "
                "has not been chosen. Only 'loop_hub' is implemented.")

        design = LoopDesign(
            radius_m=radius_m, hub_radius_m=hub_radius_m, n_blades=n_blades,
            rpm=rpm, topology=topology, spine_ctrl_m=ctrl,
            station_s=station_s,
            chord_m=tuple(float(v) for v in chord),
            twist_rad=tuple(float(v) for v in twist))
        # Seed conditioning is checked HERE, as a SolverError on the measured
        # ill-conditioned band, so a genome landing there is penalized. The
        # contract's exact-degeneracy ValueErrors sit strictly inside this band
        # and are therefore unreachable from a genome.
        seed_normal(np.asarray(ctrl[0]), np.asarray(ctrl[1]), radius_m)
        return design

    def _fr(d: Design) -> SweptFrames:
        return spine_frames(d)

    def _sec(d: Design) -> np.ndarray:
        return section_radius(np.asarray(d.chord_m), te_thickness_m=te_thickness_m)

    geom = (
        GeomConstraint("curvature_vs_section",
                       lambda d: curvature_margin(_fr(d), _sec(d))),
        GeomConstraint("self_intersection",
                       lambda d: self_intersection_margin(_fr(d), _sec(d))),
        GeomConstraint("blade_clearance",
                       lambda d: blade_clearance_margin(_fr(d), _sec(d), d.n_blades)),
        GeomConstraint("hub_closure_tangent",
                       lambda d: hub_junction_margin(_fr(d), _sec(d), d)[0]),
        GeomConstraint("hub_closure_section",
                       lambda d: hub_junction_margin(_fr(d), _sec(d), d)[1]),
    )
    return ProblemSpec(n_var, xl, xu, decode, (), (), geom)
