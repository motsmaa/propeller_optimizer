"""Root stress-concentration: the measured Kt(r/t) tables and how the beam
applies them.

The physics claim under test is NOT "Kt is 2.41". It is that BOTH root stress
components are amplified, each by the factor measured for its own load case,
looked up at the design's OWN root r/t — and that nothing in that path is
pinned to a particular design or trailing edge.
"""
import dataclasses

import numpy as np
import pytest

from propopt.configs.default import default_config
from propopt.contracts import PropellerDesign, SolverError
from propopt.geometry import DEFAULT_SECTION, section_thickness
from propopt.kt_root import (KT_AXIAL, KT_BENDING, R_OVER_T, R_OVER_T_MAX,
                             R_OVER_T_MIN, kt_factors, kt_interp)
from propopt.problem_spec import radial_spec
from propopt.solvers.beam import BeamStructSolver
from propopt.stations import uniform_midpoint

TE = 4.0e-4


def _design(rpm=6000.0, chord=None, n=6):
    chord = np.linspace(0.030, 0.012, n) if chord is None else chord
    return PropellerDesign(0.127, 0.0127, 2, rpm, tuple(chord),
                           tuple(np.linspace(0.45, 0.05, n)),
                           uniform_midpoint(0.0127, 0.127, n))


# ----- the tables ------------------------------------------------------------

def test_interpolation_reproduces_every_measured_point():
    """The interpolant must pass through the DATA, not near it — a fit that
    smooths the measurements would silently replace them (the power-law fits
    carry 5.9%/6.8% error and are deliberately not in this path)."""
    for x, kb, ka in zip(R_OVER_T, KT_BENDING, KT_AXIAL):
        assert kt_interp(x, "bending") == pytest.approx(kb, abs=1e-12)
        assert kt_interp(x, "axial") == pytest.approx(ka, abs=1e-12)


def test_curves_are_monotone_and_axial_exceeds_bending():
    """Two physical orderings the data must satisfy: a bigger fillet always
    concentrates less, and at a given radius uniform tension concentrates more
    than bending (the bending nominal is already a surface extreme)."""
    x = np.geomspace(R_OVER_T_MIN, R_OVER_T_MAX, 200)
    for load in ("bending", "axial"):
        kt = np.array([kt_interp(xi, load) for xi in x])
        assert np.all(np.diff(kt) < 0.0)
        assert np.all(kt > 1.0)
    assert np.all([kt_interp(xi, "axial") > kt_interp(xi, "bending") for xi in x])


def test_out_of_range_raises_and_says_why():
    """RAISE, not clamp. Below the table the 2-D model did not converge
    (r/t = 0.01 oscillated 4.08-4.78); above it the two face fillets collide.
    Both ends are validity limits, so a clamped value would be a number that
    was never measured."""
    for bad in (0.5 * R_OVER_T_MIN, 2.0 * R_OVER_T_MAX):
        with pytest.raises(SolverError, match="outside the measured range"):
            kt_factors(bad)
    with pytest.raises(SolverError):
        kt_factors(float("nan"))
    with pytest.raises(SolverError, match="bending|axial"):
        kt_interp(0.1, "torsion")


def test_shipped_design_space_lands_strictly_inside_the_table():
    """The guard raising in a real run would be a design-space error, not a
    modelling one — so check the whole space maps inside, with margin. Root
    thickness scales with root chord, so the extremes of chord_bounds bracket
    the reachable r/t."""
    cfg = default_config()
    r = cfg.root_fillet_radius_m
    t_norm = section_thickness(DEFAULT_SECTION, 25, 0.0)[:, 1].max()
    ratios = [r / (t_norm * c) for c in cfg.chord_bounds]
    assert R_OVER_T_MIN < min(ratios) and max(ratios) < R_OVER_T_MAX
    for x in ratios:
        kt_factors(x)                      # must not raise anywhere in between


# ----- how the beam applies them ---------------------------------------------

def test_both_components_are_amplified_by_their_own_factor():
    """THE regression that the axial term is no longer bare.

    The fillet radius changes Kt but changes NOTHING else — same geometry, same
    loads — so the root stress at three radii gives three equations
    sigma_i = Kt_axial(i)*X + Kt_bending(i)*Y in two unknowns (X = N/A,
    Y = sigma_bend). Solving the first two and PREDICTING the third is an
    independent check of the composition: it can only succeed if both
    coefficients are applied as claimed and the root is the governing node.
    Under the old model (bare N/A) the recovered X would come out ~0.
    """
    d = _design(rpm=9000.0)
    radii = (1.2e-4, 2.0e-4, 3.0e-4)
    sig, kts = [], []
    for r in radii:
        a = BeamStructSolver(root_fillet_radius_m=r, te_thickness_m=TE).analyze(d)
        sig.append(a.max_stress_Pa)
        kts.append((a.kt_axial, a.kt_bending))
    m = np.array(kts[:2])
    x_axial, y_bend = np.linalg.solve(m, np.array(sig[:2]))
    assert x_axial > 0.0 and y_bend > 0.0
    predicted = kts[2][0] * x_axial + kts[2][1] * y_bend
    assert predicted == pytest.approx(sig[2], rel=1e-9)
    # and the axial share is not a rounding artifact — it is a real fraction of
    # the root stress at this rpm, which is exactly why leaving it unamplified
    # mattered
    assert 0.1 < kts[1][0] * x_axial / sig[1] < 0.9


def test_amplification_is_root_only():
    """Kt is a junction effect. Away from the root the blade is a free
    prismatic section, so a change of fillet radius must not move the stress
    anywhere else — check via a design whose peak is NOT at the root."""
    solver_a = BeamStructSolver(root_fillet_radius_m=1.2e-4, te_thickness_m=TE)
    solver_b = BeamStructSolver(root_fillet_radius_m=4.0e-4, te_thickness_m=TE)
    # a fat root over a thin mid-span moves the governing section outboard
    chord = np.array([0.030, 0.030, 0.008, 0.008, 0.008, 0.008])
    d = _design(rpm=4000.0, chord=chord)
    a, b = solver_a.analyze(d), solver_b.analyze(d)
    # the two radii are far apart in Kt (bending 2.95 vs 1.90, axial 4.31 vs
    # 2.53), so BIT-IDENTICAL peak stress can only mean the root is not the
    # governing node and nothing outboard was touched
    assert b.kt_bending < 0.7 * a.kt_bending and b.kt_axial < 0.7 * a.kt_axial
    assert a.max_stress_Pa == pytest.approx(b.max_stress_Pa, rel=1e-12)


def test_bigger_fillet_is_safer():
    d = _design()
    sf = [BeamStructSolver(root_fillet_radius_m=r, te_thickness_m=TE)
          .analyze(d).safety_factor for r in (1.2e-4, 2.0e-4, 3.0e-4, 4.0e-4)]
    assert all(b > a for a, b in zip(sf, sf[1:]))


def test_r_over_t_is_computed_from_the_design_not_frozen():
    """r/t must move with the design's own root thickness. Halving the root
    chord halves t and therefore doubles r/t — nothing is pinned to the demo
    blade."""
    s = BeamStructSolver(te_thickness_m=TE)
    a = s.analyze(_design(chord=np.linspace(0.030, 0.012, 6)))
    b = s.analyze(_design(chord=np.linspace(0.015, 0.012, 6)))
    assert b.root_r_over_t == pytest.approx(2.0 * a.root_r_over_t, rel=1e-9)
    # a thinner root at a FIXED fillet radius is relatively better blended
    assert b.kt_bending < a.kt_bending and b.kt_axial < a.kt_axial
    assert a.root_thickness_m == pytest.approx(0.119977 * 0.030, rel=1e-3)


def test_te_thickness_is_live_in_the_root_read_and_what_it_moves():
    """te_thickness_m is wired into the root geometry, not ignored — but it is
    honest about WHAT it moves.

    It sets the constructibility bound EXACTLY (r_max = min(t_local)/2 = te/2).
    It does NOT move the maximum root thickness, and therefore does not move
    r/t or Kt, anywhere in the admissible range. That is a consequence of the
    wedge's design, not an oversight: the smoothstep is C1-flat at the
    max-thickness station, so it can only overtake the section's own maximum
    once te/c ~ 0.07 — and `geometry` already refuses te/c >= 0.06 as "a wedge,
    not an aerofoil" (TE_MAX_THICKNESS_FRACTION). The crossover is therefore
    unreachable by construction. Recorded as a measured property, and a
    desirable one: choosing a printable trailing edge does not silently change
    the root stress concentration."""
    d = _design()
    for te in (2.0e-4, 4.0e-4, 6.0e-4):
        a = BeamStructSolver(te_thickness_m=te).analyze(d)
        assert a.root_r_max_geom_m == pytest.approx(0.5 * te, rel=1e-9)
        assert a.root_thickness_m == pytest.approx(0.119977 * 0.030, rel=1e-3)
    # the crossover exists but lies outside what geometry will build
    t_of = lambda te_norm: section_thickness(DEFAULT_SECTION, 25, te_norm)[:, 1].max()
    assert t_of(0.06) == pytest.approx(t_of(0.0), rel=1e-12)   # inside the guard
    assert t_of(0.10) > 1.05 * t_of(0.0)                       # beyond it


def test_bad_fillet_radius_is_a_programming_error():
    for bad in (0.0, -1.0e-4, float("nan")):
        with pytest.raises(ValueError, match="root_fillet_radius_m"):
            BeamStructSolver(root_fillet_radius_m=bad)
    cfg = default_config()
    with pytest.raises(ValueError, match="root_fillet_radius_m"):
        dataclasses.replace(cfg, root_fillet_radius_m=0.0)


def test_front_designs_all_stay_inside_the_table():
    """Random genomes across the shipped bounds must never trip the range
    guard — if they did, the optimizer would be penalizing designs for a
    modelling limitation rather than a physical one."""
    solver = BeamStructSolver(te_thickness_m=TE)
    rng = np.random.default_rng(0)
    spec = radial_spec(n_stations=6, motor=None, chord_bounds=(0.007, 0.030))
    checked, ratios = 0, []
    for _ in range(60):
        d = spec.decode(spec.xl + rng.random(spec.n_var) * (spec.xu - spec.xl))
        try:
            a = solver.analyze(d)
        except SolverError as exc:
            assert "measured range" not in str(exc)   # never the Kt guard
            continue
        checked += 1
        ratios.append(a.root_r_over_t)
    assert checked > 40
    assert R_OVER_T_MIN < min(ratios) and max(ratios) < R_OVER_T_MAX
