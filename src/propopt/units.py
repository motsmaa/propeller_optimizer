"""SI units and sanity checks.

All quantities in this package are SI: metres, radians, seconds, kilograms,
newtons, pascals, watts, hertz. RPM is the one explicit exception and is always
named `rpm`. These helpers are called at solver boundaries so a NaN/inf or an
out-of-range value fails loudly instead of silently poisoning the Pareto front.
"""
from __future__ import annotations
import math

RHO_AIR = 1.225  # kg/m^3, sea-level ISA


class SanityError(ValueError):
    """Raised when a physical quantity violates a basic plausibility check."""


def require_finite(name: str, v: float) -> float:
    if not math.isfinite(v):
        raise SanityError(f"{name} must be finite, got {v!r}")
    return v


def require_positive(name: str, v: float) -> float:
    require_finite(name, v)
    if v <= 0:
        raise SanityError(f"{name} must be > 0, got {v!r}")
    return v


def require_nonneg(name: str, v: float) -> float:
    require_finite(name, v)
    if v < 0:
        raise SanityError(f"{name} must be >= 0, got {v!r}")
    return v


def require_fraction(name: str, v: float) -> float:
    require_finite(name, v)
    if not 0.0 <= v <= 1.0:
        raise SanityError(f"{name} must be in [0, 1], got {v!r}")
    return v
