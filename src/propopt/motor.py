"""Phase 2c: brushless-motor model + motor-propeller torque equilibrium.

Resolves the A2 open decision along Path 2: rpm is no longer a free genome
variable. It becomes the speed N at which the motor's torque equals the
propeller's aerodynamic torque, for a commanded THROTTLE. throttle is the new
operating variable in the genome.

First-order (DC-equivalent) brushless model — the standard textbook motor
constants, no thermal / magnetic-saturation / ESC-timing effects (see the
`MotorSpec` docstring for what is omitted):

    Kt        = 60 / (2*pi*KV)                 [N.m/A]  torque constant
    V         = throttle * V_max               [V]      applied voltage
    I(N)      = (V - N/KV) / Rm                [A]      current at speed N (rpm)
    Q_motor(N)= Kt * (I(N) - I0)               [N.m]    shaft torque
    N0        = KV * (V - I0*Rm)               [rpm]    no-load speed (Q=0)
    Q_stall   = Kt * (V/Rm - I0)               [N.m]    torque at N=0

Q_motor is LINEAR and DECREASING in N (rising back-EMF cuts current), from
Q_stall at N=0 to 0 at N0. Propeller torque Q_prop(N) rises with N. Their
single crossing on [floor, N0] is the operating point; the root-find exploits
that monotonic sign structure (g = Q_prop - Q_motor is negative at the floor,
positive at N0, increasing) with a hand-rolled Illinois bracket — numpy only,
no scipy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .contracts import SolverError

TWO_PI = 2.0 * math.pi
# 60/(2*pi); Kt[N.m/A] = KT_CONST / KV[rpm/V]. (The 9.5493 the anchor checks.)
KT_CONST = 60.0 / TWO_PI


@dataclass(frozen=True)
class MotorSpec:
    """First-order brushless-DC motor (SI, except KV in rpm/V and speeds in rpm
    by convention — consistent with PropellerDesign.rpm).

    OMITTED (documented first-pass limits, not modelled): winding thermal rise
    and its effect on Rm; magnetic saturation at high current (Kt fade); ESC
    commutation/timing losses and PWM behaviour below a few percent throttle;
    iron/eddy losses beyond the lumped I0. These make the model optimistic at
    the current/thermal limits — the current constraint uses I_max as a hard
    proxy for all of them.
    """
    name: str
    kv_rpm_per_v: float      # motor velocity constant [rpm/V]
    rm_ohm: float            # winding resistance [Ohm]
    i0_a: float              # no-load current [A]
    v_max_v: float           # supply voltage at full throttle [V]
    i_max_a: float           # current limit (continuous) [A]

    def __post_init__(self) -> None:
        for f in ("kv_rpm_per_v", "rm_ohm", "v_max_v", "i_max_a"):
            v = getattr(self, f)
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"MotorSpec.{f} must be positive and finite, got {v!r}")
        if not (math.isfinite(self.i0_a) and self.i0_a >= 0.0):
            raise ValueError(f"MotorSpec.i0_a must be >= 0, got {self.i0_a!r}")
        if self.i0_a * self.rm_ohm >= self.v_max_v:
            raise ValueError("i0*Rm >= V_max: motor cannot spin even at full throttle")

    @property
    def kt_nm_per_a(self) -> float:
        return KT_CONST / self.kv_rpm_per_v

    def voltage(self, throttle: float) -> float:
        return throttle * self.v_max_v

    def current(self, rpm: float, throttle: float) -> float:
        return (self.voltage(throttle) - rpm / self.kv_rpm_per_v) / self.rm_ohm

    def torque(self, rpm: float, throttle: float) -> float:
        return self.kt_nm_per_a * (self.current(rpm, throttle) - self.i0_a)

    def no_load_speed(self, throttle: float) -> float:
        return self.kv_rpm_per_v * (self.voltage(throttle) - self.i0_a * self.rm_ohm)

    def stall_torque(self, throttle: float) -> float:
        return self.kt_nm_per_a * (self.voltage(throttle) / self.rm_ohm - self.i0_a)

    def current_from_torque(self, torque_nm: float) -> float:
        """Invert Q_motor -> I. At the torque equilibrium Q_prop == Q_motor, so
        the operating current follows from the propeller torque alone — this is
        how the current-limit constraint reads I without needing rpm."""
        return torque_nm / self.kt_nm_per_a + self.i0_a


# Representative small outrunner in the 2216 class (~950 KV), a common match
# for a 10-inch prop on a 3S LiPo. Values are typical of that class's
# datasheets (SunnySky/T-Motor 2216-grade): KV 900-1000, Rm ~0.1 Ohm, no-load
# ~0.6 A, ~15-20 A continuous, 11.1 V (3S). NOT a specific SKU.
#
# NOTE: this default motor materially sets the achievable rpm band (no-load
# ~10.5k rpm here) and therefore the entire Pareto front — swapping it is the
# intended way to study a different power system, not a detail.
DEFAULT_MOTOR = MotorSpec("2216-class 950KV (3S)", 950.0, 0.10, 0.6, 11.1, 18.0)

# Below the bracket floor the equilibrium is treated as non-existent (case c):
# well under any spinning prop's operating point for DEFAULT_MOTOR (min no-load
# ~2050 rpm at throttle 0.2), and high enough that BEMT is numerically stable.
RPM_FLOOR = 100.0


def solve_torque_equilibrium(
    motor: MotorSpec,
    throttle: float,
    prop_torque: Callable[[float], float],
    *,
    rpm_floor: float = RPM_FLOOR,
    rtol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Speed N (rpm) where prop_torque(N) == motor.torque(N, throttle).

    `prop_torque(N) -> Q_prop [N.m]` is injected (BEMT in production, an
    analytic law in tests), so the solver is validated independently of BEMT.
    Raises SolverError when no equilibrium exists in [rpm_floor, no-load]
    (case c): motor too weak to spin the blade, or throttle too low to turn.
    """
    if not (0.0 < throttle <= 1.0):
        raise SolverError(f"throttle must be in (0, 1], got {throttle}")
    n_hi = motor.no_load_speed(throttle)
    if not (n_hi > rpm_floor):
        raise SolverError(f"no-load speed {n_hi:.1f} rpm <= floor {rpm_floor}: "
                          "throttle too low / motor too weak to spin")

    def g(n: float) -> float:
        return prop_torque(n) - motor.torque(n, throttle)

    a, b = rpm_floor, n_hi
    ga, gb = g(a), g(b)
    if not (math.isfinite(ga) and math.isfinite(gb)):
        raise SolverError("non-finite torque residual bracketing equilibrium")
    # Endpoint sign checks use a tolerance on a torque scale (stall torque): at
    # no-load Q_motor is 0 up to rounding, so g(n_hi) can be a tiny +/- value
    # that must read as "root here", not as "no equilibrium".
    tol = 1e-9 * motor.stall_torque(throttle)
    if ga > tol:
        raise SolverError("no torque equilibrium: propeller overloads the motor "
                          "near stall (cannot spin up)")
    if gb < -tol:
        raise SolverError("no torque equilibrium below no-load speed")
    if gb <= tol:
        return b
    if abs(ga) <= tol:
        return a

    # Brent's method (inverse-quadratic interpolation with bisection fallback):
    # keeps the sign bracket, converges superlinearly. On this wide bracket
    # (~100..10^4 rpm) plain regula-falsi/Illinois creeps at bisection rate
    # (~20 BEMT solves); Brent needs ~8. fa<0<fb hold from the checks above.
    tol = rtol * n_hi
    fa, fb = ga, gb
    c, fc, d, e = a, fa, b - a, b - a
    for _ in range(max_iter):
        if fb * fc > 0.0:                      # ensure c is the contrapoint
            c, fc, d, e = a, fa, b - a, b - a
        if abs(fc) < abs(fb):                  # b is the best estimate
            a, fa = b, fb
            b, fb = c, fc
            c, fc = a, fa
        tol1 = 2.0 * 2.22e-16 * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0.0:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p, q = 2.0 * xm * s, 1.0 - s
            else:
                q, r = fa / fc, fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e, d = d, p / q            # accept interpolation
            else:
                d = e = xm                 # fall back to bisection
        else:
            d = e = xm
        a, fa = b, fb
        b += d if abs(d) > tol1 else (tol1 if xm > 0.0 else -tol1)
        fb = g(b)
        if not math.isfinite(fb):
            raise SolverError("non-finite torque residual during root-find")
    raise SolverError(f"torque equilibrium did not converge in {max_iter} iters")
