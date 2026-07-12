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
        spec = radial_spec(n_stations=n_stations)
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
