"""PLACEHOLDER solvers for Phase 0.

These are NOT physical. They are smooth, deterministic functions of the genome
chosen so the optimizer sees genuine multi-objective tension (and so the
end-to-end plumbing can be tested before any real physics exists).
  NullAeroSolver   -> replaced by BEMT in Phase 1.
  NullStructSolver -> replaced by reduced FEA in Phase 2.
"""
from __future__ import annotations

import numpy as np

from ..contracts import (
    AeroSolver,
    EvalResult,
    PropellerDesign,
    StructResult,
    StructSolver,
)
from ..stations import station_weights

RPM_REF = 12000.0      # normalisation reference
CHORD_REF = 0.030      # normalisation reference (m)
MAT_DENSITY = 1240.0   # kg/m^3, ~PLA placeholder

# MEANINGLESS SENTINELS for the modal fields the null solver cannot compute.
# Chosen absurdly large (a GHz blade mode, a resonance margin of a million)
# so that (1) no report or plot could mistake them for physical values, and
# (2) the resonance constraint never binds on the null path — the null
# solvers exist to test optimizer plumbing, and a binding fake constraint
# would make every null run infeasible and useless. The trade-off is that
# accidentally wiring the null solver into a resonance study will not trip
# the constraint; the absurd magnitude (1e9 Hz) is the tell — inspect f1 in
# any output before trusting margins.
SENTINEL_F1_HZ = 1.0e9
SENTINEL_MARGIN = 1.0e6


class NullAeroSolver(AeroSolver):
    def evaluate(self, design: PropellerDesign) -> EvalResult:
        chord = np.asarray(design.chord_m)
        twist = np.asarray(design.twist_rad)
        c = float(chord.mean()) / CHORD_REF          # ~0..1
        n = design.rpm / RPM_REF                      # ~0..1
        twist_var = float(twist.var())

        thrust = 8.0 * design.n_blades * c * n ** 2                    # ~0..16 N
        fom = 1.0 / (1.0 + 6.0 * twist_var + 1.5 * c + 0.4 * n)        # (0, 1)
        power = 5.0 * thrust * (0.5 + 0.5 * n) + 1e-3                  # > 0 W
        omega = design.rpm * 2.0 * np.pi / 60.0
        torque = power / omega
        return EvalResult(thrust, torque, power, fom)


class NullStructSolver(StructSolver):
    def evaluate(self, design: PropellerDesign) -> StructResult:
        chord = np.asarray(design.chord_m)
        w_r = station_weights(design.station_r_m, design.hub_radius_m, design.radius_m)
        thickness = 0.12 * chord
        mass = float(MAT_DENSITY * np.sum(chord * thickness * w_r) * design.n_blades)

        c = float(chord.mean()) / CHORD_REF
        n = design.rpm / RPM_REF
        sf = 6.0 * c / (0.05 + n ** 2)                # more material up, more rpm down
        stress = 2.0e8 * n ** 2 / (c + 1e-3)
        return StructResult(mass, max(sf, 1e-3), stress,
                            SENTINEL_F1_HZ, SENTINEL_MARGIN)
