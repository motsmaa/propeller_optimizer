"""Blade-element momentum theory (BEMT) aero solver — Phase 1.

Operating point: HOVER / STATIC ONLY (freestream V = 0). This is consistent
with figure of merit being a hover metric: FoM = ideal induced power / actual
power = T^(3/2) / (sqrt(2*rho*A) * P). Forward-flight evaluation would need an
explicit operating-condition argument on `evaluate` (open decision, see README).

Method: at each radial station, iterate the axial induced velocity `v` to
consistency between blade-element thrust and axial momentum theory with a
combined Prandtl tip + root loss factor:

    dT_BE  = 1/2 rho W^2 B c (Cl cos(phi) - Cd sin(phi)) dr
    dT_mom = 4 pi r rho v^2 F dr            (static momentum, swirl neglected)

Torque/power come from integrating the blade-element tangential force. The
fixed-point update is under-relaxed; failure to converge, or a degenerate /
non-physical input or intermediate state, raises SolverError (never a bare
exception), which the optimizer treats as a penalized candidate.

Airfoil polars enter only through a swappable `polar(alpha, Re) -> (Cl, Cd)`
callable (constructor argument), so tabulated/measured polars can be plugged in
without touching the solver. The default `analytic_polar` is a crude cambered
thin-airfoil model.

A2 / OPEN DECISION (flagged, not solved here): the genome treats `rpm` as a
free design variable with no motor coupling. Whether rpm stays free or becomes
the solved motor-prop torque-equilibrium point is deliberately NOT decided in
this phase; do not add a motor model here without revisiting that seam.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..contracts import AeroSolver, EvalResult, PropellerDesign, SolverError
from ..stations import station_weights
from ..units import RHO_AIR

MU_AIR = 1.81e-5        # Pa*s, dynamic viscosity of air at ~15 C
SPEED_OF_SOUND = 340.0  # m/s, sea-level ISA
MAX_TIP_MACH = 0.85     # beyond this the incompressible model is invalid

# `polar(alpha_rad, reynolds) -> (Cl, Cd)`; must accept/return numpy arrays.
PolarFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

# --- analytic default polar -------------------------------------------------
CL_ALPHA = 5.7     # 1/rad, ~0.9 * 2*pi (finite-thickness lift slope)
CL0 = 0.3          # camber offset, roughly a mildly cambered thin section
CL_CAP = 1.2       # symmetric stall cap on |Cl|
CD0 = 0.02         # profile drag floor
K_CD = 0.04        # quadratic drag polar coefficient, Cd = CD0 + K_CD*Cl^2
K_STALL = 1.0      # post-stall drag rise, ~flat-plate Cd at alpha = 90 deg


def analytic_polar(alpha: np.ndarray, reynolds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simple analytic polar: linear Cl capped at stall, quadratic Cd.

    WARNING: Reynolds number is accepted but IGNORED. At the chord Reynolds
    numbers of small props (3e4-3e5) real sections lose lift slope and gain
    a lot of profile drag; this polar is optimistic there. Plugging in
    tabulated low-Re polars plus a Re-correction layer is deliberately
    deferred — swap this function via the BEMTAeroSolver(polar=...) seam.
    """
    alpha = np.asarray(alpha, dtype=float)
    cl = np.clip(CL0 + CL_ALPHA * alpha, -CL_CAP, CL_CAP)
    # extra drag once past the stall angles implied by the Cl caps
    a_pos = (CL_CAP - CL0) / CL_ALPHA
    a_neg = (-CL_CAP - CL0) / CL_ALPHA
    excess = np.maximum(alpha - a_pos, 0.0) + np.maximum(a_neg - alpha, 0.0)
    cd = CD0 + K_CD * cl ** 2 + K_STALL * excess ** 2
    return cl, cd


def _prandtl_factor(f: np.ndarray) -> np.ndarray:
    return (2.0 / np.pi) * np.arccos(np.exp(-np.minimum(f, 50.0)))


class BEMTAeroSolver(AeroSolver):
    """Static (hover) BEMT with Prandtl tip/root losses. See module docstring."""

    def __init__(
        self,
        polar: PolarFn = analytic_polar,
        rho: float = RHO_AIR,
        max_iter: int = 500,
        rtol: float = 1e-7,
        relax: float = 0.35,
    ) -> None:
        self._polar = polar
        self._rho = rho
        self._max_iter = max_iter
        self._rtol = rtol
        self._relax = relax

    def evaluate(self, design: PropellerDesign) -> EvalResult:
        rho = self._rho
        radius, hub = design.radius_m, design.hub_radius_m
        n_blades = design.n_blades
        chord = np.asarray(design.chord_m, dtype=float)
        theta = np.asarray(design.twist_rad, dtype=float)
        omega = design.rpm * 2.0 * np.pi / 60.0

        # -- degeneracy / model-validity checks (SolverError, not a crash) --
        if not hub >= 0.0:
            raise SolverError(f"hub_radius_m must be >= 0, got {hub}")
        if hub >= radius:
            raise SolverError(f"hub radius {hub} m >= tip radius {radius} m")
        if np.any(chord <= 0.0) or not np.all(np.isfinite(chord)):
            raise SolverError("chord must be positive and finite at every station")
        if np.any(np.abs(theta) >= np.pi / 2) or not np.all(np.isfinite(theta)):
            raise SolverError("twist must be finite and within (-pi/2, pi/2)")
        tip_mach = omega * radius / SPEED_OF_SOUND
        if tip_mach > MAX_TIP_MACH:
            raise SolverError(f"tip Mach {tip_mach:.2f} exceeds {MAX_TIP_MACH} "
                              "(incompressible BEMT invalid)")

        # shared station placement (contract field) + quadrature cell widths:
        # each station owns the cell between neighbouring midpoints clipped to
        # [hub, R], so aero integrates over exactly [hub, R]
        r = np.asarray(design.station_r_m, dtype=float)
        w_r = station_weights(r, hub, radius)
        ut = omega * r                      # blade-section tangential speed
        v_scale = omega * radius            # velocity scale for tolerance

        # -- fixed-point iteration on axial induced velocity v(r) --
        v = 0.05 * ut  # small positive seed so phi, F start well-defined
        for _ in range(self._max_iter):
            phi = np.arctan2(v, ut)
            w = np.hypot(v, ut)
            cl, cd = self._polar(theta - phi, rho * w * chord / MU_AIR)
            sin_phi, cos_phi = np.sin(phi), np.cos(phi)

            sin_safe = np.maximum(sin_phi, 1e-6)  # guard only the loss factor
            f_loss = (_prandtl_factor(0.5 * n_blades * (radius - r) / (r * sin_safe))
                      * _prandtl_factor(0.5 * n_blades * (r - hub) / (r * sin_safe)))
            f_loss = np.maximum(f_loss, 1e-3)

            dtdr_be = 0.5 * rho * w ** 2 * n_blades * chord * (cl * cos_phi - cd * sin_phi)
            # static momentum has no real solution for negatively loaded
            # stations; treat their induced velocity as zero
            v_mom = np.sqrt(np.maximum(dtdr_be, 0.0) / (4.0 * np.pi * r * rho * f_loss))
            v_new = v + self._relax * (v_mom - v)
            if not np.all(np.isfinite(v_new)):
                raise SolverError("non-finite induced velocity during BEMT iteration")
            if np.max(np.abs(v_new - v)) < self._rtol * v_scale:
                v = v_new
                break
            v = v_new
        else:
            raise SolverError(f"BEMT did not converge in {self._max_iter} iterations")

        # -- converged state: integrate blade-element forces --
        phi = np.arctan2(v, ut)
        w = np.hypot(v, ut)
        cl, cd = self._polar(theta - phi, rho * w * chord / MU_AIR)
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)
        q_local = 0.5 * rho * w ** 2 * n_blades * chord
        thrust = float(np.sum(q_local * (cl * cos_phi - cd * sin_phi) * w_r))
        torque = float(np.sum(q_local * (cl * sin_phi + cd * cos_phi) * r * w_r))
        power = torque * omega
        if not (np.isfinite(thrust) and np.isfinite(torque) and power > 0.0):
            raise SolverError("non-physical integrated loads (T/Q/P)")

        disk_area = np.pi * radius ** 2
        if thrust > 0.0:
            fom = thrust ** 1.5 / (np.sqrt(2.0 * rho * disk_area) * power)
        else:
            fom = 0.0
        if fom >= 1.0:
            raise SolverError(f"figure of merit {fom:.3f} >= 1 beats the "
                              "actuator-disk ideal — solver inconsistency")
        return EvalResult(thrust, torque, power, fom)
