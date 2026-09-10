"""Frozen interface contracts — the architectural spine.

Nothing in the optimizer layer references concrete solver implementations; all
coupling goes through these dataclasses and ABCs. Keep them topology-agnostic:
the radial-station representation lives in `problem_spec`, not here.

Two blade topologies are represented, as SIBLINGS under `Design`:
`PropellerDesign` (radial stations, strictly increasing radius) and
`LoopDesign` (a section swept along a free-form 3-D spine, stations placed by
arc-length fraction because a loop revisits radii). They are deliberately not
one generalised class - see `Design`. The solver ABCs stay typed on
`PropellerDesign` until a loop-aware solver exists, so a radial solver cannot
silently accept a loop it has no way to evaluate.

Result dataclasses validate strictly (finite + physical range) so that solver
bugs surface loudly in tests and direct use. During optimization, the optimizer
catches these validation errors and penalizes the candidate (see optimize.py),
so strict contracts and a crash-free run coexist.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .units import require_finite, require_fraction, require_nonneg, require_positive


@dataclass(frozen=True)
class Design:
    """What every decoded design carries regardless of blade topology (SI).

    Shared base of the RADIAL (`PropellerDesign`) and LOOP (`LoopDesign`)
    representations, which are SIBLINGS: neither is a special case of the other.
    Dataclass equality and isinstance are class-exact, so a solver or cache
    written for one can never silently accept the other. Only topology-agnostic
    fields live here; `radius_m` is an ENVELOPE - no part of the blade lies
    outside it - which is what the radial tip radius already meant and what a
    loop needs it to mean.

    AXIAL EXTENT IS DELIBERATELY UNBOUNDED HERE. `radius_m` bounds the swept
    cylinder's RADIUS only; nothing in this contract limits how far a design
    reaches along the rotation axis. How far a blade may bulge axially is a
    design-quality question, not a well-formedness one, so it belongs in
    `geom_constraints` where it is PENALIZED, not in a validator that would
    reject it.

    ABSTRACT: instantiating `Design` directly raises TypeError. It carries no
    blade representation at all, so a bare `Design` is never a decodable
    candidate and constructing one is a programming error, not a bad design.
    """

    radius_m: float
    hub_radius_m: float
    n_blades: int
    rpm: float

    def __post_init__(self) -> None:
        if type(self) is Design:
            raise TypeError(
                "Design is abstract: instantiate PropellerDesign (radial) or "
                "LoopDesign (free-form spine)")
        require_positive("radius_m", self.radius_m)
        require_positive("rpm", self.rpm)
        if self.n_blades < 1:
            raise ValueError(f"n_blades must be >= 1, got {self.n_blades}")


@dataclass(frozen=True)
class PropellerDesign(Design):
    """A decoded RADIAL candidate design (SI units). Output of seam 1
    (genome -> geometry).

    Field ORDER and validation SEQUENCE are unchanged by the introduction of
    the base class: dataclass inheritance places base fields first, which is
    already the order this class had, and `super().__post_init__()` runs
    exactly the three checks that were previously the first three lines. The
    positional signature, `__eq__`, `__hash__` and `__repr__` are therefore
    bit-for-bit what they were - pinned in tests/test_contracts_loop.py.
    """

    chord_m: tuple[float, ...]    # per radial station, root -> tip
    twist_rad: tuple[float, ...]  # per radial station, root -> tip
    station_r_m: tuple[float, ...]  # radial position of each station (m),
                                    # strictly increasing, within
                                    # [hub_radius_m, radius_m]. The single
                                    # shared source of station placement for
                                    # aero, geometry and structure. Spacing
                                    # SCHEMES (uniform-midpoint, cosine, ...)
                                    # live in problem_spec/stations, not here.

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.chord_m) != len(self.twist_rad):
            raise ValueError("chord_m and twist_rad must have equal length")
        if not self.chord_m:
            raise ValueError("design must have at least one radial station")
        if len(self.station_r_m) != len(self.chord_m):
            raise ValueError("station_r_m and chord_m must have equal length")
        r = self.station_r_m
        for v in r:
            require_finite("station_r_m", v)
        if any(b <= a for a, b in zip(r, r[1:])):
            raise ValueError("station_r_m must be strictly increasing")
        if r[0] < self.hub_radius_m or r[-1] > self.radius_m:
            raise ValueError("station_r_m must lie within [hub_radius_m, radius_m]")


# Blade closure topologies. All three are WELL-FORMED names at the contract
# level; which of them the geometry layer can actually build is that layer's
# business and is reported there with SolverError. This preserves the failure
# layering recorded in the Phase-2a notes: ValueError = malformed design,
# SolverError = a valid design the pipeline cannot realise (a feasibility
# outcome the optimizer penalizes).
BLADE_TOPOLOGIES = ("loop_hub", "loop_neighbour", "open")

# Below this, the ROTATION AXIS and the spine tangent are treated as parallel
# and the rotation-minimizing frame has no well-defined seed. The quantity is
# |y_hat x t0| for a unit tangent, i.e. the sine of the angle between them, so
# 1e-6 is ~1e-6 rad.
_RMF_SEED_TOL = 1e-6


@dataclass(frozen=True)
class LoopDesign(Design):
    """A decoded FREE-FORM design: the section swept along a 3-D spine (SI).

    FRAME (verified against geometry.build_blade_solid and
    occ_geometry.station_rings, not assumed): x tangential, y AXIAL = the
    rotation axis, z radial = span. A point's radius about the axis is
    therefore hypot(x, z).

    The spine is a clamped cubic B-spline through `spine_ctrl_m`. Stations are
    placed by ARC-LENGTH FRACTION `station_s` in [0, 1] - the quantity that is
    legitimately strictly increasing along a loop. RADIUS IS DERIVED by the
    geometry layer from the spine and is deliberately NOT a field: a loop
    revisits radii, so no monotone invariant on radius can hold, and storing a
    derived value invites it to disagree with the curve.

    ROTATION-MINIMIZING FRAME, SEEDED HERE rather than in the geometry layer,
    so that `twist_rad` determines the geometry from this contract ALONE. At
    s = 0 the seed normal is the ROTATION AXIS crossed into the unit tangent:

        y_hat = (0, 1, 0)                            the rotation axis
        t0    = normalise(P1 - P0)      clamped-spline start tangent direction
        n0    = normalise(y_hat x t0) = (tz, 0, -tx) / hypot(tx, tz)

    and the frame is parallel-transported along the spine from there (never
    Frenet - Frenet flips at inflection points; see the sweep generator).
    `twist_rad` at a station is the rotation of the section about the local
    unit tangent, measured FROM that transported normal, positive by the
    right-hand rule about the tangent.

    WHY THIS SEED AND NOT ANOTHER - MEASURED, NOT ARGUED. Three candidates
    share the identical degeneracy condition |y_hat x t0| and differ only in
    direction: y_hat - (y_hat.t0)t0, t0 x y_hat, and y_hat x t0. On a straight
    radial spine at fixed twist - the one configuration where the swept and
    radial representations must describe the SAME blade - only `y_hat x t0`
    reproduces `PropellerDesign`'s section orientation, to 1.1e-13 m. The other
    two sit at exactly +90 and +180 degrees. Picking either would have silently
    redefined twist for every loop design relative to the radial path
    (`seed_equivalence.py`).

    WHY THE AXIS AND NOT THE RADIAL DIRECTION. An earlier version projected the
    LOCAL RADIAL direction off the tangent. That is degenerate when the start
    tangent is parallel to the radial direction - which is exactly the tangent
    a blade leaving a hub cylinder has, and exactly what `loop_hub` produces.
    It was not a theoretical concern: that seed REFUSED a straight radial
    blade, the canonical propeller, as malformed. Seeding off the fixed
    rotation axis inverts the degeneracy so that it bites only on a spine
    running purely AXIALLY at its foot.

    THE DEGENERATE SET IS EXCLUDED BY DECODE-SIDE REJECTION, not by anything in
    this contract. Nothing here constrains the foot TANGENT: the `loop_hub`
    branch below validates endpoint RADIUS only, and junction transversality is
    a soft penalty in another layer. `loop_spec.decode` raises SolverError when
    hypot(tx, tz) falls below a measured conditioning floor, which is strictly
    wider than the ValueError below, so a genome is penalized rather than
    aborting a run and the ValueError remains unreachable from one.

    A DEGENERATE SEED is a ValueError: a start tangent parallel to the ROTATION
    AXIS, or coincident first two control points, leave the frame - and hence
    the twist datum - undefined. That is malformedness, not infeasibility.

    ENVELOPE AND ENDPOINT CHECKS ARE UNREACHABLE FROM THE GENOME BY
    CONSTRUCTION. `loop_spec.decode` parameterises control points as
    (r, theta, y) with r box-bounded to [hub_radius_m, radius_m], and for
    `loop_hub` pins r = hub_radius_m on the first and last control points
    rather than encoding them. The checks below are retained as a guard
    against hand-built or programmatically-built designs; a genome can never
    trip them, because a ValueError here would abort a run instead of being
    penalized.
    """

    topology: str
    spine_ctrl_m: tuple[tuple[float, float, float], ...]
    station_s: tuple[float, ...]
    chord_m: tuple[float, ...]
    twist_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        require_nonneg("hub_radius_m", self.hub_radius_m)
        if self.hub_radius_m >= self.radius_m:
            raise ValueError(
                f"hub_radius_m must be < radius_m, got {self.hub_radius_m} "
                f">= {self.radius_m}")
        if self.topology not in BLADE_TOPOLOGIES:
            raise ValueError(f"topology must be one of {BLADE_TOPOLOGIES}, "
                             f"got {self.topology!r}")
        if len(self.spine_ctrl_m) < 4:
            raise ValueError("a cubic spine needs >= 4 control points, got "
                             f"{len(self.spine_ctrl_m)}")
        for p in self.spine_ctrl_m:
            if len(p) != 3:
                raise ValueError("spine control points must be 3-D")
            for v in p:
                require_finite("spine_ctrl_m", v)
            # A B-spline lies inside the convex hull of its control polygon and
            # a cylinder is convex, so bounding the CONTROL POINTS bounds the
            # whole curve - no curve evaluation is needed in the contract.
            if math.hypot(p[0], p[2]) > self.radius_m * (1.0 + 1e-9):
                raise ValueError("spine control point lies outside radius_m")
        if self.topology == "loop_hub":
            # A zero-radius hub cylinder IS the rotation axis, so both feet
            # would be planted on it and the blade would have no azimuth.
            # Checked only in this branch: other topologies do not pin their
            # endpoints to the hub and a zero hub is meaningless, not invalid,
            # for them.
            if self.hub_radius_m <= 0.0:
                raise ValueError(
                    "loop_hub needs hub_radius_m > 0: both spine endpoints lie "
                    "on the hub cylinder, and a zero-radius cylinder is the "
                    "rotation axis")
            tol = 1e-9 * self.radius_m
            for p in (self.spine_ctrl_m[0], self.spine_ctrl_m[-1]):
                if abs(math.hypot(p[0], p[2]) - self.hub_radius_m) > tol:
                    raise ValueError("loop_hub: both spine endpoints must lie "
                                     "on the hub cylinder")

        p0, p1 = self.spine_ctrl_m[0], self.spine_ctrl_m[1]
        d = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        dn = math.sqrt(sum(v * v for v in d))
        if dn <= 0.0:
            raise ValueError("degenerate RMF seed: the first two spine control "
                             "points coincide, so the start tangent is undefined")
        t0 = tuple(v / dn for v in d)
        # |y_hat x t0| for a unit t0, computed as hypot(tx, tz) rather than
        # sqrt(1 - ty**2). The two are equal analytically, but 1 - ty**2 loses
        # its significant digits exactly where this check bites (ty -> +-1),
        # which is the region the check exists to detect.
        if math.hypot(t0[0], t0[2]) < _RMF_SEED_TOL:
            raise ValueError(
                "degenerate RMF seed: the start tangent is parallel to the "
                "ROTATION AXIS, so the section-twist datum is undefined")

        n = len(self.station_s)
        if n < 2:
            raise ValueError(f"a swept design needs >= 2 stations, got {n}")
        if len(self.chord_m) != n or len(self.twist_rad) != n:
            raise ValueError(
                "station_s, chord_m and twist_rad must have equal length")
        for v in self.station_s:
            require_finite("station_s", v)
        if any(b <= a for a, b in zip(self.station_s, self.station_s[1:])):
            raise ValueError("station_s must be strictly increasing")
        if self.station_s[0] < 0.0 or self.station_s[-1] > 1.0:
            raise ValueError("station_s must lie within [0, 1]")
        for v in self.chord_m:
            require_positive("chord_m", v)
        for v in self.twist_rad:
            require_finite("twist_rad", v)


@dataclass(frozen=True)
class EvalResult:
    """Aerodynamic evaluation output (SI units). Output of the AeroSolver."""

    thrust_N: float
    torque_Nm: float
    power_W: float
    figure_of_merit: float  # dimensionless, (0, 1]

    def __post_init__(self) -> None:
        require_finite("thrust_N", self.thrust_N)
        require_finite("torque_Nm", self.torque_Nm)
        require_finite("power_W", self.power_W)
        require_fraction("figure_of_merit", self.figure_of_merit)


@dataclass(frozen=True)
class StructResult:
    """Structural evaluation output (SI units). Output of the StructSolver."""

    mass_kg: float
    safety_factor: float  # dimensionless
    max_stress_Pa: float
    first_freq_hz: float      # first (flapwise) natural frequency at design rpm
    resonance_margin: float   # min_n |f1 - n*BPF| / (n*BPF) over the first few
                              # blade-passing harmonics; 0 = on resonance

    def __post_init__(self) -> None:
        require_positive("mass_kg", self.mass_kg)
        require_positive("safety_factor", self.safety_factor)
        require_finite("max_stress_Pa", self.max_stress_Pa)
        require_positive("first_freq_hz", self.first_freq_hz)
        require_nonneg("resonance_margin", self.resonance_margin)


class SolverError(Exception):
    """Raised by a solver when an evaluation fails recoverably — e.g. BEMT/CFD
    non-convergence or a result the solver itself judges unusable. The optimizer
    catches this and penalizes the candidate instead of crashing the run."""


class AeroSolver(ABC):
    @abstractmethod
    def evaluate(self, design: PropellerDesign) -> EvalResult: ...


class StructSolver(ABC):
    @abstractmethod
    def evaluate(self, design: PropellerDesign) -> StructResult: ...
