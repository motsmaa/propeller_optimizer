"""Frozen interface contracts — the architectural spine.

Nothing in the optimizer layer references concrete solver implementations; all
coupling goes through these dataclasses and ABCs. Keep them topology-agnostic:
the radial-station representation lives in `problem_spec`, not here.

Result dataclasses validate strictly (finite + physical range) so that solver
bugs surface loudly in tests and direct use. During optimization, the optimizer
catches these validation errors and penalizes the candidate (see optimize.py),
so strict contracts and a crash-free run coexist.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .units import require_finite, require_fraction, require_nonneg, require_positive


@dataclass(frozen=True)
class PropellerDesign:
    """A decoded candidate design (SI units). Output of seam 1 (genome -> geometry)."""

    radius_m: float
    hub_radius_m: float
    n_blades: int
    rpm: float
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
        require_positive("radius_m", self.radius_m)
        require_positive("rpm", self.rpm)
        if self.n_blades < 1:
            raise ValueError(f"n_blades must be >= 1, got {self.n_blades}")
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
