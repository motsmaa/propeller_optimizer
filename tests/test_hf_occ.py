"""Phase-4a slice 2: the blade as an OpenCASCADE B-rep solid (HF path only).

These lock down the properties the fidelity gate established: the OCC
construction reads the SAME section and station definitions as the triangle
loft (verified bitwise, so the two cannot silently diverge), and it agrees with
the loft EXACTLY wherever the two representations are mathematically obliged to
agree — which turns out to be everywhere except a spanwise-twisted patch.
"""
import numpy as np
import pytest

from propopt.contracts import PropellerDesign, SolverError
from propopt.geometry import solid_from_design
from propopt.problem_spec import radial_spec
from propopt.stations import uniform_midpoint

TE = 4.0e-4
CHORD_BOUNDS = (0.007, 0.030)


def _has_gmsh() -> bool:
    try:
        import gmsh  # noqa: F401
        return True
    except ImportError:
        return False


needs_gmsh = pytest.mark.skipif(not _has_gmsh(),
                                reason="gmsh not installed (pip install -e '.[hf]')")


def _design(chord, twist, n=6):
    return PropellerDesign(0.127, 0.01, 2, 5000.0, tuple(chord), tuple(twist),
                           uniform_midpoint(0.01, 0.127, n))


def test_station_rings_match_the_triangle_loft_bitwise():
    """The OCC path restates the polygon -> 3-D placement (the loft does it
    vectorized in the optimizer's hot path and should not be restructured for
    the HF path). This asserts the restatement rather than trusting it — if the
    two ever diverge, THIS is what catches it. No gmsh needed."""
    from propopt.hf.occ_geometry import station_rings
    spec = radial_spec(motor=None, chord_bounds=CHORD_BOUNDS)
    for x in (spec.xl, spec.xu, 0.5 * (spec.xl + spec.xu)):
        d = spec.decode(x)
        rings = station_rings(d, te_thickness_m=TE)
        loft = solid_from_design(d, te_thickness_m=TE)
        assert np.array_equal(rings.reshape(-1, 3), loft.vertices)


@needs_gmsh
def test_occ_matches_loft_exactly_without_twist_variation():
    """Where the two representations are obliged to agree they agree to machine
    precision: a ruled patch between two stations of EQUAL twist is planar in
    the spanwise direction, so chording it with triangles removes nothing.
    Exactness here is what proves sections, stations, TE, placement and caps
    are shared rather than merely similar."""
    from propopt.hf.occ_geometry import occ_blade
    for twist in (0.0, 0.3, 0.6):                       # constant twist
        d = _design(np.linspace(0.030, 0.012, 6), np.full(6, twist))
        loft = solid_from_design(d, te_thickness_m=TE).volume_m3
        occ = occ_blade(d, te_thickness_m=TE).volume_m3
        assert occ == pytest.approx(loft, rel=1e-9)


@needs_gmsh
def test_occ_gap_is_twist_variation_only_and_vanishes_with_refinement():
    """The whole OCC-vs-loft difference is spanwise twist VARIATION, and it is
    the loft's first-order error rather than an OCC defect: refining stations
    shrinks it, and OCC converges to the smooth swept volume faster than the
    loft does."""
    from propopt.hf.occ_geometry import occ_blade
    gaps = []
    for n in (4, 8, 16):
        d = _design(np.linspace(0.030, 0.012, n), np.linspace(0.6, 0.0, n), n=n)
        loft = solid_from_design(d, te_thickness_m=TE).volume_m3
        occ = occ_blade(d, te_thickness_m=TE).volume_m3
        gaps.append(abs(occ - loft) / loft)
    assert all(b < a for a, b in zip(gaps, gaps[1:]))    # monotone shrinking
    assert gaps[-1] < 0.5 * gaps[0]                      # and substantially so


@needs_gmsh
def test_occ_construction_is_robust_across_the_design_space():
    """OCC lofting failure rate is the key risk metric for using a B-rep kernel
    in the HF path — a kernel that refuses awkward blades cannot carry the
    fillet work. Measured 0/20 on the fidelity sweep; this is a smaller guard."""
    from propopt.hf.occ_geometry import occ_blade
    spec = radial_spec(motor=None, chord_bounds=CHORD_BOUNDS)
    rng = np.random.default_rng(0)
    failures = 0
    for _ in range(8):
        d = spec.decode(spec.xl + rng.random(spec.n_var) * (spec.xu - spec.xl))
        try:
            blade = occ_blade(d, te_thickness_m=TE)
            assert blade.volume_m3 > 0.0
            # blunt TE keeps upper/lower trailing vertices distinct
            assert blade.blunt_te and blade.n_profile_points % 2 == 1
        except SolverError:
            failures += 1
    assert failures == 0, f"{failures}/8 OCC lofts failed"


def test_degenerate_design_raises_solver_error():
    """OCC failure is a feasibility outcome for the HF path, never a crash."""
    from propopt.hf.occ_geometry import station_rings
    d = _design(np.full(6, 0.02), np.zeros(6))
    with pytest.raises(SolverError):
        station_rings(d, te_thickness_m=-1.0)


def test_sharp_te_is_refused_explicitly():
    """A sharp TE gives a CUSPED profile wire — the degeneracy a B-rep kernel
    handles worst, and the same one that broke gmsh's STL remesh mode in the
    Phase-4a slice. Untested and therefore refused up front, rather than left
    to fail deeper inside OCC with an opaque message."""
    from propopt.hf.occ_geometry import station_rings
    d = _design(np.full(6, 0.02), np.full(6, 0.3))
    with pytest.raises(SolverError, match="finite trailing edge"):
        station_rings(d, te_thickness_m=0.0)


def test_root_cross_section_is_identical_to_the_loft():
    """THE property that makes a later Kt extraction safe: the root section is
    a design station (chord/twist held from the hub extension), not an
    interpolated one, and the section polygon is shared — so the region where
    root stress is read is the SAME geometry in both representations."""
    from propopt.hf.occ_geometry import station_rings
    from propopt.geometry import (DEFAULT_SECTION, extended_stations,
                                  polygon_area, section_polygon)
    d = _design(np.linspace(0.030, 0.012, 6), np.linspace(0.6, 0.0, 6))
    rings = station_rings(d, te_thickness_m=TE)
    loft = solid_from_design(d, te_thickness_m=TE)
    p = rings.shape[1]
    assert np.array_equal(rings[0], loft.vertices[:p])          # bitwise

    _, c_ext, _ = extended_stations(d)
    analytic = polygon_area(
        section_polygon(DEFAULT_SECTION, 25, TE / c_ext[0])) * c_ext[0] ** 2
    tri = loft.vertices[loft.faces[loft.patches["hub"]]]
    cap = float(np.sum(0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)))
    assert cap == pytest.approx(analytic, rel=1e-12)


@needs_gmsh
def test_volume_gap_does_not_accrue_at_the_root():
    """Where the OCC-vs-loft difference lives matters more than its size: the
    root and tip elements sit in the end-extension, where twist is HELD
    constant, so their ruled patches are untwisted and the two representations
    agree exactly there. The gap is mid-span only, so it cannot contaminate a
    root-stress ratio."""
    import contextlib, io, gmsh
    from propopt.geometry import build_blade_solid, extended_stations
    from propopt.hf.occ_geometry import _build_occ_solid, station_rings

    d = _design(np.linspace(0.030, 0.012, 6), np.linspace(0.6, 0.0, 6))
    r, c, t = extended_stations(d)
    rings = station_rings(d, te_thickness_m=TE)

    def occ_vol(sub):
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            with contextlib.redirect_stdout(io.StringIO()):
                return float(gmsh.model.occ.getMass(3, _build_occ_solid(gmsh, sub)))
        finally:
            gmsh.finalize()

    diffs = []
    for e in range(len(r) - 1):
        vl = build_blade_solid(r[e:e+2], c[e:e+2], t[e:e+2],
                               te_thickness_m=TE).volume_m3
        diffs.append(abs(occ_vol(rings[e:e+2]) - vl))
    # end-extension elements (constant twist) agree exactly; mid-span does not
    assert diffs[0] == pytest.approx(0.0, abs=1e-15), "root element must be exact"
    assert diffs[-1] == pytest.approx(0.0, abs=1e-15), "tip element must be exact"
    assert sum(diffs[1:-1]) > 0.0                    # the gap is real, mid-span
