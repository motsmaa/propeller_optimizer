"""Phase-2b anchor: the beam FE against closed-form beam theory, plus
physical-direction sanity checks on the full solver. The FE anchors prove the
assembly and eigensolve are correct; they are exact-form checks, not
tuned-tolerance comparisons."""
import numpy as np
import pytest

from propopt.contracts import PropellerDesign, SolverError, StructResult
from propopt.problem_spec import radial_spec
from propopt.solvers.beam import (
    BeamStructSolver,
    Material,
    centrifugal_tension,
    distributed_load_vector,
    first_frequency_hz,
    flapwise_system,
)
from propopt.solvers.bemt import BEMTAeroSolver
from propopt.stations import uniform_midpoint

N_ELEM = 20


def _uniform_beam(length=1.0, ei=100.0, rho_a=1.0):
    r = np.linspace(0.0, length, N_ELEM + 1)
    return r, np.full(N_ELEM, ei), np.full(N_ELEM, rho_a)


def make_design(rpm: float = 5000.0, scale: float = 1.0, n: int = 6) -> PropellerDesign:
    r = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    chord = tuple(scale * (0.025 - 0.013 * ri) for ri in r)
    twist = tuple(float(np.arctan(0.35 / (np.pi * ri))) for ri in r)
    return PropellerDesign(0.127, 0.0127, 2, rpm, chord, twist,
                           uniform_midpoint(0.0127, 0.127, n))


# ----- analytical anchors (FE vs closed form) --------------------------------

def test_tip_deflection_uniform_cantilever_uniform_load():
    """w_tip = q L^4 / (8 EI). Hermite elements + consistent load vector are
    nodally exact for a uniform Euler-Bernoulli beam -> near machine precision."""
    length, ei, q = 1.0, 100.0, 3.0
    r, ei_e, rho_a_e = _uniform_beam(length, ei)
    k, _ = flapwise_system(r, ei_e, rho_a_e, np.zeros(N_ELEM))
    w = np.linalg.solve(k, distributed_load_vector(r, np.full(N_ELEM, q)))
    assert w[-2] == pytest.approx(q * length ** 4 / (8.0 * ei), rel=1e-9)


def test_centrifugal_root_tension_uniform_bar():
    """N_root = 1/2 rho A omega^2 (R^2 - r0^2); midpoint integration is exact
    for the linear integrand -> machine precision."""
    rho_a, omega = 2.5, 300.0
    r = np.linspace(0.0, 1.0, N_ELEM + 1)
    n = centrifugal_tension(r, np.full(N_ELEM, rho_a), omega)
    assert n[0] == pytest.approx(0.5 * rho_a * omega ** 2 * 1.0 ** 2, rel=1e-12)
    assert n[-1] == 0.0


def test_first_frequency_uniform_cantilever():
    """omega_1 = (1.8751)^2 sqrt(EI / (rho A L^4)) = 3.5160 sqrt(...). FE with
    consistent mass converges from above at O(h^4); 20 elements are well
    inside rel 1e-4."""
    length, ei, rho_a = 1.0, 100.0, 1.0
    r, ei_e, rho_a_e = _uniform_beam(length, ei, rho_a)
    k, m = flapwise_system(r, ei_e, rho_a_e, np.zeros(N_ELEM))
    f_exact = 1.8751040687 ** 2 * np.sqrt(ei / (rho_a * length ** 4)) / (2 * np.pi)
    assert first_frequency_hz(k, m) == pytest.approx(f_exact, rel=1e-4)


# ----- physical-direction sanity on the full solver --------------------------

def test_stress_stiffening_f1_increases_with_rpm():
    solver = BeamStructSolver()
    f1 = [solver.analyze(make_design(rpm=rpm)).f1_hz
          for rpm in (2000.0, 4000.0, 8000.0, 12000.0)]
    assert all(b > a for a, b in zip(f1, f1[1:]))


def test_safety_factor_directions():
    solver = BeamStructSolver()
    sf_rpm = [solver.analyze(make_design(rpm=rpm)).safety_factor
              for rpm in (3000.0, 6000.0, 9000.0)]
    assert all(b < a for a, b in zip(sf_rpm, sf_rpm[1:]))  # faster -> more stress
    sf_chord = [solver.analyze(make_design(rpm=6000.0, scale=s)).safety_factor
                for s in (0.7, 1.0, 1.3)]
    assert all(b > a for a, b in zip(sf_chord, sf_chord[1:]))  # more material -> safer


def test_conforms_to_contract():
    st = BeamStructSolver().evaluate(make_design())
    assert isinstance(st, StructResult)
    assert st.mass_kg > 0.0 and st.safety_factor > 0.0 and st.max_stress_Pa > 0.0


def test_degenerate_inputs_raise_solver_error():
    # aero non-convergence propagates as SolverError
    flaky = BeamStructSolver(aero=BEMTAeroSolver(max_iter=2))
    with pytest.raises(SolverError):
        flaky.evaluate(make_design())
    # contract-valid but physically-unsolvable: negative chord
    bad = PropellerDesign(0.127, 0.0127, 2, 5000.0, (0.02, -0.01, 0.02),
                          (0.1, 0.1, 0.1), uniform_midpoint(0.0127, 0.127, 3))
    with pytest.raises(SolverError):
        BeamStructSolver().evaluate(bad)
    # bad material is a programming error, rejected at construction
    with pytest.raises(ValueError):
        Material("junk", 1240.0, -1.0, 50e6)


def test_resonance_constraint_is_active():
    """The resonance feasibility constraint must mark a near-resonance design
    infeasible (g > 0) and a well-separated one feasible (g < 0). At 3000 rpm
    the reference design's f1 (~93 Hz) sits ~7% from the first blade-passing
    harmonic (100 Hz) — inside the 0.15 keep-out band."""
    spec = radial_spec()
    idx = [c.name for c in spec.constraints].index("resonance_margin>=min")
    aero, beam = BEMTAeroSolver(), BeamStructSolver()

    near = make_design(rpm=3000.0)
    g = spec.eval_constraints(aero.evaluate(near), beam.evaluate(near))
    assert g[idx] > 0.0  # infeasible: too close to resonance

    safe = make_design(rpm=5000.0)  # margin ~0.27
    g = spec.eval_constraints(aero.evaluate(safe), beam.evaluate(safe))
    assert g[idx] < 0.0


def test_resonance_constraint_reshapes_front():
    """With the constraint disabled the optimizer parks designs near
    blade-passing resonance; enabling it must clear the front out of the
    keep-out band. (Small run: direction check, not a benchmark.)"""
    from propopt.optimize import optimize

    beam = BeamStructSolver()
    aero = BEMTAeroSolver()

    def front_margins(margin_min):
        spec = radial_spec(resonance_margin_min=margin_min, motor=None)
        res = optimize(spec, aero, beam, pop_size=32, n_gen=10, seed=0)
        margins = []
        for x in res.X:
            try:
                margins.append(beam.analyze(spec.decode(x)).resonance_margin)
            except SolverError:
                continue
        assert margins, "front produced no analyzable designs"
        return margins

    unconstrained = front_margins(0.0)
    constrained = front_margins(0.15)
    assert min(unconstrained) < 0.15  # the hazard is real in this problem
    assert all(m >= 0.15 for m in constrained)


def test_mass_consistency_with_geometry():
    """Beam element mass uses the span-exact average of the linearly
    interpolated chord^2, so it must equal the lofted-solid mass EXACTLY for
    untwisted blades.

    Twisted blades differ by design (verdict: genuine geometry effect, not an
    integration inconsistency): the triangulated loft — the actual printable
    artifact whose volume is the mass objective — encloses slightly more
    volume than the smooth twisted extrusion (non-planar quads chorded by
    triangles), by O(dtheta^2) per element / O(1/n) total; verified to vanish
    under station refinement and to be exactly zero untwisted. The beam
    tracks the smooth limit (integral of A dr, twist-invariant). Measured
    worst case over random genomes ~3.7%; asserted at 5%."""
    solver = BeamStructSolver()
    d0 = PropellerDesign(0.127, 0.0127, 2, 5000.0, (0.02,) * 6, (0.0,) * 6,
                         uniform_midpoint(0.0127, 0.127, 6))
    a0 = solver.analyze(d0)
    assert a0.beam_mass_kg == pytest.approx(a0.mass_kg, rel=1e-12)

    rng = np.random.default_rng(0)
    checked = 0
    for n_st in (4, 6, 8, 10):
        spec = radial_spec(n_stations=n_st, motor=None)  # mass is rpm-independent
        for _ in range(25):
            d = spec.decode(spec.xl + rng.random(spec.n_var) * (spec.xu - spec.xl))
            try:
                a = solver.analyze(d)
            except SolverError:
                continue  # BEMT stall non-convergence; penalized in the loop
            checked += 1
            assert a.beam_mass_kg == pytest.approx(a.mass_kg, rel=0.05)
    assert checked > 80  # the sweep must actually exercise the check


# --- exact span-averaged section properties with a blunt TE ---------------- #

def test_te_zero_section_path_is_pinned():
    """te=0 must keep running the pre-TE span-exact closed form. Pinned at full
    precision: the zero branch is untouched code, so any drift means the blunt-TE
    work leaked into the sharp-TE path.

    RE-PINNED once, when the root guard moved from a single kt_root=2.5 on the
    bending term to the MEASURED pair (Kt_axial, Kt_bending) on both terms:
    3.6097254650763246 -> 3.2212810180636900, a 10.8% reduction. That is a
    deliberate physics change, not drift — this design's root (2.869 mm thick)
    sits at r/t = 0.0697, where Kt_bending = 2.2262 (vs the 2.5 placeholder, so
    the bending term EASED) and Kt_axial = 3.1768 (vs 1.0, i.e. the centrifugal
    term was previously unamplified). f1 is UNCHANGED to full
    precision, which is the check that only the stress composition moved:
    Kt never enters the mass, stiffness or eigenproblem."""
    d = make_design()
    a = BeamStructSolver(aero=BEMTAeroSolver(), te_thickness_m=0.0).analyze(d)
    assert a.safety_factor == pytest.approx(3.2212810180636900, rel=1e-12)
    assert a.f1_hz == pytest.approx(122.43263498808244, rel=1e-12)
    assert a.root_r_over_t == pytest.approx(0.06970, rel=1e-3)


def test_blunt_te_area_is_exact_not_trapezoidal():
    """A(c) = A_base*c^2 + k_area*te*c is EXACT (the normalized area is exactly
    linear in te/c), so the element average must reproduce the analytic integral
    of A over a linearly tapered element — not the trapezoidal mean, which
    over-estimates c^2 by up to ~11.5% on a strong taper."""
    te = 4.0e-4
    s = BeamStructSolver(te_thickness_m=te)
    ci, cj = 0.030, 0.012
    c_nodes = np.array([ci, cj])
    area_e, _ = s._element_props(c_nodes)
    exact = s._a_base * (ci**2 + ci*cj + cj**2) / 3.0 + s._k_area * te * 0.5 * (ci + cj)
    assert area_e[0] == pytest.approx(exact, rel=1e-14)
    trapezoid = 0.5 * (s._area_abs(ci) + s._area_abs(cj))
    assert trapezoid > area_e[0]                      # the old bias, upward
    assert trapezoid / area_e[0] - 1 > 0.05           # ~11.5% on this taper


def test_gauss3_is_exact_for_the_sharp_te_quartic():
    """The EI quadrature must not degrade the sharp-TE case: I ~ c^4 with c
    linear in r is a quartic, and 3-point Gauss is exact to degree 5."""
    ci, cj = 0.030, 0.012
    xi = 0.5 * (np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)]) + 1.0)
    w = np.array([5/9, 8/9, 5/9])
    cg = ci + (cj - ci) * xi
    exact_c4 = (ci**4 + ci**3*cj + ci**2*cj**2 + ci*cj**3 + cj**4) / 5.0
    assert 0.5 * np.sum(w * cg**4) == pytest.approx(exact_c4, rel=1e-14)


def test_blunt_te_ei_quadrature_error_is_negligible():
    """For the rational (blunt-TE) part there is no closed form; check the
    3-point rule against a 20-point reference on the same element."""
    s = BeamStructSolver(te_thickness_m=4.0e-4)
    ci, cj = 0.030, 0.012
    xi = 0.5 * (np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)]) + 1.0)
    w = np.array([5/9, 8/9, 5/9])
    got = 0.5 * np.sum(w * s._i_chord_abs(ci + (cj - ci) * xi))
    xr, wr = np.polynomial.legendre.leggauss(20)
    ref = 0.5 * np.sum(wr * s._i_chord_abs(ci + (cj - ci) * 0.5 * (xr + 1.0)))
    assert abs(got - ref) / ref < 1e-8               # measured ~3e-11


def test_blunt_te_mass_consistency_restored():
    """Worst-case beam-vs-loft mass with the TE active must return to the
    sharp-TE level. The residual that remains is the twist/faceting effect
    (the loft is a faceted solid), NOT an averaging error."""
    te = 4.0e-4
    solver = BeamStructSolver(aero=BEMTAeroSolver(), te_thickness_m=te)
    rng = np.random.default_rng(0)
    worst, checked = 0.0, 0
    for n_st in (4, 6, 8, 10):
        spec = radial_spec(n_stations=n_st, motor=None, chord_bounds=(0.007, 0.030))
        for _ in range(25):
            d = spec.decode(spec.xl + rng.random(spec.n_var) * (spec.xu - spec.xl))
            try:
                a = solver.analyze(d)
            except SolverError:
                continue
            checked += 1
            worst = max(worst, abs(a.beam_mass_kg - a.mass_kg) / a.mass_kg)
    assert checked > 80
    assert worst < 0.04, f"worst {worst:.3%} (sharp-TE baseline ~3.4%)"
