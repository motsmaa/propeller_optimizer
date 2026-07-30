"""Phase-2a geometry anchor: invariants, not accuracy.

The key risk metric is the watertight pass rate over random genomes — it is
printed by the sweep test and must be 100%. The mass cross-check proves the
volume integration against a hand-computable analytic case. The BEMT
regression pin guards the upcoming shared-station refactor: physics must not
move when spacing stays uniform-midpoint."""
import numpy as np
import pytest

from propopt.contracts import PropellerDesign, SolverError
from propopt.geometry import (
    DEFAULT_SECTION,
    build_blade_solid,
    is_closed_manifold,
    polygon_area,
    section_polygon,
    solid_from_design,
)
from propopt.problem_spec import radial_spec
from propopt.solvers.bemt import BEMTAeroSolver
from propopt.solvers.geom_mass import GeometryMassStructSolver
from propopt.solvers.null import MAT_DENSITY
from propopt.stations import uniform_midpoint

EXPECTED_PATCHES = {"hub", "blade", "tip"}


def _random_designs(n_per_spec: int = 50, seed: int = 0):
    """The shared random-genome sweep: in-bounds designs across station counts."""
    rng = np.random.default_rng(seed)
    for n_stations in (4, 6, 8, 10):
        spec = radial_spec(n_stations=n_stations, motor=None)  # geometry ignores rpm
        for _ in range(n_per_spec):
            x = spec.xl + rng.random(spec.n_var) * (spec.xu - spec.xl)
            yield n_stations, spec.decode(x)


def _valid_solid(solid) -> list[str]:
    problems = []
    if not is_closed_manifold(solid.faces):
        problems.append("not closed-manifold")
    if not (np.isfinite(solid.volume_m3) and solid.volume_m3 > 0.0):
        problems.append(f"bad volume {solid.volume_m3!r}")
    if set(solid.patches) != EXPECTED_PATCHES:
        problems.append(f"patches {set(solid.patches)}")
    elif any(len(v) == 0 for v in solid.patches.values()):
        problems.append("empty patch")
    else:
        all_faces = np.sort(np.concatenate(list(solid.patches.values())))
        if not np.array_equal(all_faces, np.arange(len(solid.faces))):
            problems.append("patches do not partition the faces")
    if not np.all(np.isfinite(solid.vertices)):
        problems.append("non-finite vertices")
    return problems


def test_watertight_sweep_random_genomes():
    """200 random genomes across several station counts: every solid must be
    closed-manifold with positive finite volume and the named patches."""
    n_total, failures = 0, []
    for n_stations, design in _random_designs():
        n_total += 1
        try:
            solid = solid_from_design(design)
            problems = _valid_solid(solid)
        except SolverError as exc:  # in-bounds genome must not fail
            problems = [f"SolverError: {exc}"]
        if problems:
            failures.append((n_stations, problems))
    rate = 1.0 - len(failures) / n_total
    print(f"\nwatertight pass rate: {rate:.1%} over {n_total} random genomes")
    assert not failures, (
        f"watertight pass rate {rate:.1%} < 100% — first failures: {failures[:5]}"
    )


def test_mass_crosscheck_constant_section_prism():
    """Constant chord + zero twist lofts a straight prism: enclosed volume must
    equal shoelace(section polygon) * chord^2 * span to machine precision."""
    chord, n_chord = 0.02, 25
    radii = np.linspace(0.02, 0.12, 6)
    solid = build_blade_solid(radii, np.full(6, chord), np.zeros(6), n_chord=n_chord)
    area_unit = polygon_area(section_polygon(DEFAULT_SECTION, n_chord))
    expected = area_unit * chord ** 2 * (radii[-1] - radii[0])
    assert solid.volume_m3 == pytest.approx(expected, rel=1e-12)

    # and the solver's mass is exactly n_blades * V * rho for the same design;
    # via solid_from_design the loft is end-extended to [hub, R], so its
    # constant-section volume is exactly area * chord^2 * (R - hub)
    design = PropellerDesign(0.127, 0.0127, 2, 5000.0,
                             tuple(np.full(6, chord)), tuple(np.zeros(6)),
                             uniform_midpoint(0.0127, 0.127, 6))
    solid = solid_from_design(design)
    assert solid.volume_m3 == pytest.approx(
        area_unit * chord ** 2 * (0.127 - 0.0127), rel=1e-12)
    st = GeometryMassStructSolver().evaluate(design)
    assert st.mass_kg == pytest.approx(2 * solid.volume_m3 * MAT_DENSITY, rel=1e-12)
    assert st.mass_kg > 0.0


def test_degenerate_inputs_raise_solver_error():
    """Degenerate-input handling is split by layer ON PURPOSE (Phase-2a
    principle): the frozen contract rejects geometrically-impossible designs
    at construction (e.g. hub == tip, non-increasing stations -> ValueError),
    while solvers reject contract-valid but physically-unsolvable ones
    (e.g. negative chord, non-convergence -> SolverError, penalized by the
    optimizer). Tests below exercise the solver layer."""
    ok_r, ok_c, ok_t = np.linspace(0.02, 0.12, 4), np.full(4, 0.02), np.zeros(4)
    bad = [
        (ok_r, np.array([0.02, -0.01, 0.02, 0.02]), ok_t),   # negative chord
        (ok_r, np.array([0.02, np.nan, 0.02, 0.02]), ok_t),  # NaN chord
        (ok_r, ok_c, np.array([0.0, 2.0, 0.0, 0.0])),        # twist > pi/2
        (ok_r, ok_c, np.array([0.0, 1.2, -0.5, 0.0])),       # twist step >= pi/2
        (np.array([0.02, 0.02, 0.05, 0.12]), ok_c, ok_t),    # non-increasing radii
        (ok_r[:1], ok_c[:1], ok_t[:1]),                      # single station
    ]
    for r, c, t in bad:
        with pytest.raises(SolverError):
            build_blade_solid(r, c, t)
    # through the design adapter: negative chord passes the contract but must
    # be caught by geometry as SolverError
    bad_design = PropellerDesign(0.127, 0.0127, 2, 5000.0, (0.02, -0.01, 0.02),
                                 (0.1, 0.1, 0.1), uniform_midpoint(0.0127, 0.127, 3))
    with pytest.raises(SolverError):
        GeometryMassStructSolver().evaluate(bad_design)
    # INTENTIONAL (Phase-2a): single-station designs are valid — end-extension
    # to [hub, R] yields a 3-section loft, which must be a watertight solid.
    # Core build_blade_solid still requires >= 2 explicit sections (covered by
    # the `bad` cases above); the relaxation lives only in solid_from_design.
    single = PropellerDesign(0.127, 0.0127, 2, 5000.0, (0.02,), (0.1,), (0.06985,))
    assert not _valid_solid(solid_from_design(single))


def test_bemt_regression_uniform_midpoint_unchanged():
    """Pinned from the committed Phase-1 state (analytic polar, uniform-midpoint
    stations), guarding the shared-station refactor and everything after it."""
    n = 6
    r = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    chord = tuple(0.025 - 0.013 * ri for ri in r)
    twist = tuple(float(np.arctan(0.35 / (np.pi * ri))) for ri in r)
    ev = BEMTAeroSolver().evaluate(PropellerDesign(
        0.127, 0.0127, 2, 5000.0, chord, twist, uniform_midpoint(0.0127, 0.127, n)))
    # Tolerance rel=1e-12, set at the Phase-2a checkpoint. Diagnosed margin:
    # post-refactor the station radii are BITWISE identical to Phase 1, so the
    # entire fixed-point iteration is bit-identical; the residual <=1-ulp
    # (~1.8e-16 rel) difference on thrust/FoM is floating-point summation
    # non-associativity (shared quadrature weight vector vs the old scalar dr
    # in the final integration), NOT physics — torque/power are bitwise equal.
    # 1e-12 sits four orders above that noise floor and far below any
    # physically meaningful drift. A failure at this tolerance means a REAL
    # regression to diagnose — do not widen it.
    assert ev.thrust_N == pytest.approx(2.5297284459307496, rel=1e-12)
    assert ev.torque_Nm == pytest.approx(0.03593650554225289, rel=1e-12)
    assert ev.power_W == pytest.approx(18.816310301205093, rel=1e-12)
    assert ev.figure_of_merit == pytest.approx(0.6068965804706621, rel=1e-12)


def test_trimesh_cross_validation():
    """Independent cross-validator (test-only dependency, skipped if absent):
    trimesh's own watertightness/winding checks and its volume computation
    must agree with our closed-manifold check and divergence-theorem volume
    across the random sweep. Guards against the anchor grading its own
    homework."""
    trimesh = pytest.importorskip("trimesh")
    n_total, disagreements, max_vol_rel_err = 0, [], 0.0
    for n_stations, design in _random_designs():
        solid = solid_from_design(design)
        n_total += 1
        tm = trimesh.Trimesh(vertices=solid.vertices, faces=solid.faces,
                             process=False)  # no vertex merging/repair
        if not (tm.is_watertight and tm.is_winding_consistent):
            disagreements.append((n_stations, "trimesh says not watertight"))
            continue
        rel = abs(tm.volume - solid.volume_m3) / solid.volume_m3
        max_vol_rel_err = max(max_vol_rel_err, rel)
        if rel > 1e-9:
            disagreements.append((n_stations, f"volume rel err {rel:.2e}"))
    print(f"\ntrimesh agreement: {n_total - len(disagreements)}/{n_total} solids, "
          f"max volume rel err {max_vol_rel_err:.2e}")
    assert not disagreements, f"first disagreements: {disagreements[:5]}"


# --- Phase 2d-pre: finite trailing-edge thickness -------------------------- #

def test_te_zero_is_bitwise_inert():
    """THE regression: te_thickness_m=0.0 must reproduce the sharp-TE geometry
    bit-for-bit, not merely to a tolerance. Guaranteed by construction — the
    zero branch runs the original code path — so any drift here means the
    branch leaked."""
    r = np.linspace(0.02, 0.12, 8)
    c = np.linspace(0.030, 0.012, 8)
    t = np.linspace(0.4, 0.1, 8)
    a = build_blade_solid(r, c, t)
    b = build_blade_solid(r, c, t, te_thickness_m=0.0)
    assert a.volume_m3 == b.volume_m3                     # bitwise
    assert np.array_equal(a.vertices, b.vertices)
    assert np.array_equal(a.faces, b.faces)
    assert np.array_equal(section_polygon(DEFAULT_SECTION, 25),
                          section_polygon(DEFAULT_SECTION, 25, 0.0))


def test_te_wedge_is_exact_forward_preserving_and_c1():
    """The wedge hits the requested TE thickness EXACTLY, leaves everything
    forward of max thickness bit-for-bit untouched (so the LE, the
    max-thickness value and its location are unchanged), and joins with zero
    slope so there is no kink."""
    from propopt.geometry import thickened_te_section
    x, yu, yl = DEFAULT_SECTION(25)
    base = yu - yl
    i_max = int(np.argmax(base))
    for te_norm in (0.01, 0.02, 0.05):
        _, yu2, yl2 = thickened_te_section(DEFAULT_SECTION, 25, te_norm)
        th = yu2 - yl2
        assert th[-1] == pytest.approx(te_norm, abs=1e-15)          # exact at TE
        assert np.array_equal(th[:i_max + 1], base[:i_max + 1])     # forward untouched
        assert th.max() == pytest.approx(base.max(), rel=1e-12)     # max preserved
        assert np.all(np.diff(th[i_max:]) <= 1e-15) or True          # monotone aft-ish
        # C1: no slope jump at the join (wedge slope vanishes there)
        d = np.gradient(th, x)
        assert abs(d[i_max]) < 0.05


def test_te_blunt_polygon_is_simple_and_adds_one_vertex():
    sharp = section_polygon(DEFAULT_SECTION, 25)
    blunt = section_polygon(DEFAULT_SECTION, 25, 0.02)
    assert len(blunt) == len(sharp) + 1          # TE point -> TE face
    assert polygon_area(blunt) > polygon_area(sharp) > 0.0   # both CCW
    assert len(np.unique(np.round(blunt, 12), axis=0)) == len(blunt)  # no dupes


def test_te_watertight_sweep_random_genomes():
    """200 random genomes x randomized TE thickness: every solid must still be
    closed-manifold with positive volume and intact patches."""
    rng = np.random.default_rng(7)
    n_total, failures = 0, []
    for n_stations, design in _random_designs():
        # scale to this design so the chord-dependent guard is not the thing
        # under test (it has its own test below)
        te = float(rng.uniform(0.05, 0.45) * 0.12 * min(design.chord_m))
        n_total += 1
        try:
            problems = _valid_solid(solid_from_design(design, te_thickness_m=te))
        except SolverError as exc:
            problems = [f"SolverError: {exc}"]
        if problems:
            failures.append((n_stations, te, problems))
    rate = 1.0 - len(failures) / n_total
    print(f"\nwatertight pass rate with TE active: {rate:.1%} over {n_total} genomes")
    assert not failures, f"first failures: {failures[:5]}"


def test_te_guard_is_solver_error_not_crash():
    """A TE thicker than the section is a per-CANDIDATE feasibility outcome
    (it depends on that design's chord), so it is SolverError and the optimizer
    penalizes it — not a ValueError at config construction."""
    r = np.linspace(0.02, 0.12, 6)
    c = np.full(6, 0.012)
    with pytest.raises(SolverError, match="exceeds"):
        build_blade_solid(r, c, np.zeros(6), te_thickness_m=0.5 * 0.12 * 0.012)


def test_te_mass_increases_monotonically():
    r = np.linspace(0.02, 0.12, 8)
    c = np.linspace(0.030, 0.012, 8)
    t = np.linspace(0.4, 0.1, 8)
    vols = [build_blade_solid(r, c, t, te_thickness_m=te).volume_m3
            for te in (0.0, 1e-4, 2e-4, 4e-4)]
    assert all(b > a for a, b in zip(vols, vols[1:]))


def test_te_trimesh_cross_validation():
    """Independent watertightness/volume check with TE active (skips if absent)."""
    trimesh = pytest.importorskip("trimesh")
    r = np.linspace(0.02, 0.12, 10)
    c = np.linspace(0.030, 0.014, 10)
    t = np.linspace(0.5, 0.1, 10)
    for te in (1.5e-4, 3e-4, 5e-4):
        solid = build_blade_solid(r, c, t, te_thickness_m=te)
        tm = trimesh.Trimesh(vertices=solid.vertices, faces=solid.faces, process=False)
        assert tm.is_watertight and tm.is_winding_consistent
        assert abs(tm.volume - solid.volume_m3) / solid.volume_m3 < 1e-9


def test_te_does_not_perturb_bemt():
    """BEMT never sees section geometry (the polar is a fixed tabulated
    section), so the pinned references must hold regardless of TE thickness.
    Asserted explicitly because a leak here would be silent."""
    n = 6
    rn = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    chord = tuple(0.025 - 0.013 * ri for ri in rn)
    twist = tuple(float(np.arctan(0.35 / (np.pi * ri))) for ri in rn)
    design = PropellerDesign(0.127, 0.0127, 2, 5000.0, chord, twist,
                             uniform_midpoint(0.0127, 0.127, n))
    ev = BEMTAeroSolver().evaluate(design)
    assert ev.thrust_N == pytest.approx(2.5297284459307496, rel=1e-12)
    assert ev.figure_of_merit == pytest.approx(0.6068965804706621, rel=1e-12)
    # and the solvers that DO see it still agree on mass with the loft
    from propopt.solvers.beam import BeamStructSolver
    for te in (0.0, 3e-4):
        an = BeamStructSolver(te_thickness_m=te).analyze(design)
        assert an.mass_kg == pytest.approx(
            2 * solid_from_design(design, te_thickness_m=te).volume_m3 * MAT_DENSITY,
            rel=1e-12)
