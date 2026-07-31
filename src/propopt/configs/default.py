"""Default run: a 10-inch 2-blade PLA prop on a 2216-class 3S motor.

Every value carries its justification here so the rationale lives next to the
number. `pop_size`/`n_gen` are set from the convergence study (see
analysis.py / tests/test_convergence.py and the README Phase-3 section).
"""
from __future__ import annotations

import dataclasses

from ..config import RunConfig
from ..motor import DEFAULT_MOTOR
from ..solvers.beam import PLA


def exploration_config() -> RunConfig:
    """EXPLORATION-grade: a single seed, pop 60 / gen 40. For iteration and
    demos. Per the convergence study the single-seed final-HV spread is ~25%,
    so the exact edges of an exploration front carry seed noise — do NOT quote
    its numbers as final (use decision mode for that)."""
    return RunConfig(
        mode="exploration",    # single seed `seed`; exploration-grade.
        # --- design space ---
        n_stations=6,          # enough radial DOF for a useful twist/chord
                               # distribution; BEMT/beam are converged well
                               # below this station count (geometry O(1/n)).
        radius_m=0.127,        # 10-inch diameter (5-inch tip radius), matches
                               # the UIUC apcsf_10x4.7 validation prop.
        hub_radius_m=0.01,     # ~8% root cutout, typical small-prop hub.
        n_blades=2,            # 2-blade, matches the validation prop + motor.
        chord_bounds=(0.007, 0.030),   # 7-30 mm for a 10-inch prop. The lower
                                       # bound is set by the finite TE below,
                                       # two ways: the geometry guard rejects
                                       # chord < te/(0.5*0.12) = 6.67 mm, and
                                       # more fundamentally a 0.4 mm TE on a
                                       # 5 mm chord is 8% of chord — a wedge,
                                       # not a propeller section. 7 mm keeps
                                       # margin above the guard (TE <= 5.7% of
                                       # chord everywhere).
        twist_bounds=(0.0, 0.6),       # 0-34 deg geometric; BEMT rejects
                                       # |twist|>=pi/2, so this stays valid.
        throttle_bounds=(0.2, 1.0),    # >=20%: below that the first-order motor
                                       # model degrades (ESC PWM/cogging) and
                                       # thrust is negligible (see problem_spec).
        # --- physics ---
        motor=DEFAULT_MOTOR,   # 2216-class 950KV/3S; sets the rpm band and
                               # hence the whole front (see motor.py).
        material=PLA,          # printed PLA, isotropic first-pass (see beam.py).
        root_fillet_radius_m=2.0e-4,
                               # 0.20 mm intended root fillet. Sets the root
                               # stress concentration: the beam looks up
                               # MEASURED Kt_axial and Kt_bending at this
                               # design's own r/t (see propopt.kt_root, from
                               # Phase-4a slice 3b 2-D plane-strain FEA). Two
                               # independent reasons for this radius: half a
                               # 0.40 mm extrusion width is the smallest blend
                               # an FDM tool path can lay, and it is exactly the
                               # slice-3a constructibility bound te/2 for the
                               # 0.40 mm trailing edge below, so it is provably
                               # buildable on the real solid. At the demo
                               # front's root chords this lands at r/t ~ 0.056
                               # -> Kt_bending 2.41, Kt_axial 3.47. It replaces
                               # the old kt_root=2.5 placeholder, which was ~4%
                               # conservative on bending but left the
                               # centrifugal term entirely unamplified.
        te_thickness_m=4.0e-4, # 0.40 mm blunt trailing edge. TWO independent
                               # reasons for this exact value: (1) one 0.4 mm
                               # nozzle/extrusion width is the FDM
                               # manufacturing floor — below one extrusion
                               # width a slicer cannot lay a closed edge at
                               # all; (2) the mesh-quality study (README
                               # Phase-2d) puts it where quality resolves —
                               # gamma_min 0.000 -> 0.074 and gamma_p01
                               # <=0.08 -> 0.496 vs the Phase-4a sharp-TE
                               # baseline, and it unblocks gmsh remesh mode,
                               # which failed outright on the knife edge —
                               # without paying the extra mass of 0.5 mm.
        # --- constraints ---
        min_thrust_N=2.0,      # a design must lift something; excludes the
                               # trivial zero-pitch optimum.
        resonance_margin_min=0.15,     # conservative keep-out; first-pass beam
                                       # f1 is uncertain (see problem_spec).
        # --- optimizer (set from the convergence study, README Phase-3) ---
        pop_size=60,           # STUDY (5 seeds, ~20 ms/candidate, ~4 min): HV
        n_gen=40,              # rises monotonically, reaching 99% of its gen-40
                               # value by ~gen 37 (still creeping ~1-3%/10gen).
                               # 60x40 is the near-converged setting that fits a
                               # <=5 min 5-seed budget; final-HV seed spread is
                               # ~25%, so a single-seed front is exploration-
                               # grade, not decision-grade (aggregate seeds or
                               # raise pop for the latter).
        seed=0,                # exploration seed; runs reproduce from it.
        seeds=(0, 1, 2, 3, 4), # decision-mode aggregation seeds (>=5 for
                               # decision-grade; unused in exploration mode).
        # --- hypervolume normalization (fixed bounds, NOT physical targets;
        # chosen to bracket every feasible front across seeds so HV is
        # comparable — nadir sits just outside the feasible boundary so no
        # feasible design maps to loss==1). Order: thrust, fom, mass, sf. ---
        hv_ideal=(15.0, 1.0, 0.0, 10.0),      # best: high thrust, ideal FoM,
                                              # massless, very safe.
        hv_nadir=(1.5, 0.0, 0.035, 0.8),      # worst: below min_thrust(2),
                                              # zero FoM, heavier than any
                                              # design, below SF floor(1).
    )


def decision_config() -> RunConfig:
    """DECISION-grade: identical physics/optimizer to exploration, differing
    ONLY in seed-handling — aggregates the cross-seed non-dominated union of
    `seeds` (>=5) into one front. Per the convergence study (README Phase-3),
    5-seed aggregation is the decision-grade setting; single-seed fronts carry
    ~25% HV spread and are exploration-only."""
    return dataclasses.replace(exploration_config(), mode="decision")


# Backward-compatible alias: the default run is exploration-grade.
def default_config() -> RunConfig:
    return exploration_config()
