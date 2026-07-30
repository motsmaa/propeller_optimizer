"""Phase-4a high-fidelity path: boundary, mesher, deck/runner/parser.

Layered by what each test needs, so a machine with neither gmsh nor CalculiX
still runs the pure-Python parts and skips the rest with a clear reason:
  * path translation      — no dependencies
  * meshing               — needs gmsh
  * deck/run/parse        — needs gmsh AND ccx
The expensive closed-form anchors live in the `slow` marker (see
tests/test_hf_anchors.py) so `pytest -q` stays fast.
"""
import numpy as np
import pytest

from propopt.contracts import SolverError
from propopt.hf.wsl import Runner, WSLError, to_wsl_path, wsl_available


def _has_gmsh() -> bool:
    try:
        import gmsh  # noqa: F401
        return True
    except ImportError:
        return False


def _has_ccx() -> bool:
    from propopt.hf.wsl import resolve_runner
    try:
        resolve_runner("ccx", env_var="PROPOPT_CCX")
        return True
    except WSLError:
        return False


needs_gmsh = pytest.mark.skipif(not _has_gmsh(),
                                reason="gmsh not installed (pip install -e '.[hf]')")
needs_ccx = pytest.mark.skipif(not _has_ccx(),
                               reason="CalculiX ccx not available (see src/propopt/hf/SOURCE.md)")


# ----- boundary (no external dependencies) ---------------------------------

def test_windows_to_wsl_path_translation():
    assert to_wsl_path(r"C:\a\b\c.inp") == "/mnt/c/a/b/c.inp"
    assert to_wsl_path(r"D:/x/y") == "/mnt/d/x/y"
    assert to_wsl_path("/usr/bin/ccx") == "/usr/bin/ccx"   # POSIX passthrough


def test_wsl_command_quotes_paths_with_spaces():
    argv, win_cwd = Runner("ccx", use_wsl=True).command(
        ["-i", "job"], cwd=r"C:\Program Files\x")
    assert win_cwd is None                     # cwd is set INSIDE the distro
    joined = " ".join(argv)
    assert "'/mnt/c/Program Files/x'" in joined
    argv2, win_cwd2 = Runner("ccx.exe", use_wsl=False).command(["-i", "job"], cwd="C:/tmp")
    assert argv2[0] == "ccx.exe" and win_cwd2 == "C:/tmp"   # native: no hop


# ----- mesher (gmsh only) ---------------------------------------------------

@needs_gmsh
def test_box_mesh_volume_and_order():
    from propopt.hf.mesh import mesh_box
    m = mesh_box(0.1, 0.02, 0.02, 6e-3, order=2)
    assert m.order == 2 and m.ccx_element_type == "C3D10"
    assert m.tets.shape[1] == 10
    assert m.volume_m3 == pytest.approx(0.1 * 0.02 * 0.02, rel=1e-9)
    assert m.quality_min > 0.0                 # no inverted/degenerate tets
    assert m.n_tets > 0 and m.n_nodes > 0


@needs_gmsh
def test_preserve_mode_keeps_loft_volume_exactly(tmp_path):
    """The STL facets bound the volume mesh unchanged, so the meshed volume
    equals the loft's exact divergence-theorem volume. This is the property that
    makes the FEA analyse the same solid the optimizer scored.

    The 1e-8 bound is the ASCII-STL text precision, not a mesher error: the
    exporter writes vertices at %.6e (7 significant digits), and recomputing the
    volume of the as-written surface reproduces the meshed volume exactly. The
    mesher itself is bit-exact on the boundary."""
    from propopt.export import _write_ascii_stl
    from propopt.geometry import build_blade_solid
    from propopt.hf.mesh import mesh_stl

    r = np.linspace(0.02, 0.12, 12)
    solid = build_blade_solid(r, np.full(12, 0.02), np.zeros(12), n_chord=25)
    stl = tmp_path / "bar.stl"
    _write_ascii_stl(stl, solid.vertices, solid.faces, "bar")
    m = mesh_stl(stl, 2e-3, order=1, reference_volume_m3=solid.volume_m3)
    assert m.volume_rel_error == pytest.approx(0.0, abs=1e-8)  # measured 5.3e-9


@needs_gmsh
def test_volume_drift_gate_raises(tmp_path):
    """A drift budget of 0 must trip on any non-exact mesh: the guard exists so
    a silently-different solid cannot reach the solver."""
    from propopt.export import _write_ascii_stl
    from propopt.geometry import build_blade_solid
    from propopt.hf.mesh import mesh_stl

    r = np.linspace(0.02, 0.12, 8)
    solid = build_blade_solid(r, np.full(8, 0.02), np.zeros(8), n_chord=21)
    stl = tmp_path / "bar.stl"
    _write_ascii_stl(stl, solid.vertices, solid.faces, "bar")
    with pytest.raises(SolverError, match="drift"):
        mesh_stl(stl, 2e-3, order=1, reference_volume_m3=solid.volume_m3 * 1.5,
                 max_volume_drift=0.0)


@needs_gmsh
def test_missing_stl_is_solver_error():
    from propopt.hf.mesh import mesh_stl
    with pytest.raises(SolverError):
        mesh_stl("does_not_exist.stl", 1e-3)


@needs_gmsh
def test_boundary_and_nodal_areas_are_consistent():
    from propopt.hf.mesh import boundary_triangles, mesh_box, nodal_areas
    m = mesh_box(0.1, 0.02, 0.02, 8e-3, order=1)
    tris = boundary_triangles(m.tets)
    areas = nodal_areas(m.nodes, tris)
    exact = 2 * (0.1 * 0.02 + 0.1 * 0.02 + 0.02 * 0.02)   # box surface area
    assert areas.sum() == pytest.approx(exact, rel=1e-9)


# ----- deck + runner + parser (gmsh AND ccx) -------------------------------

@needs_gmsh
@needs_ccx
def test_uniaxial_bar_matches_closed_form(tmp_path):
    """End-to-end chain on a case with an exact answer: a clamped bar under a
    known end load. This is the UNIT-SYSTEM guard — a mm/MPa mix-up shows up as
    a factor of 10^6, which no tolerance below could absorb.

    The primary assertion is mid-span axial stress (sigma = F/A), which is
    exact to ~0.03% here. Tip displacement is asserted only loosely: this bar is
    stubby (L/a = 5), so nodal end loading plus the clamp's suppression of
    Poisson contraction add ~2.6% of Saint-Venant end effect that sigma*L/E does
    not model. That was verified to be a real 3-D effect rather than a load-
    mapping artifact — equal-per-node and area-weighted loading agree to 5
    digits. The slender-bar bending/centrifugal anchors in test_hf_anchors.py are
    the tight convergence checks."""
    from propopt.hf.calculix import run_ccx, write_inp
    from propopt.hf.mesh import boundary_triangles, mesh_box, nodal_areas
    from propopt.solvers.beam import Material

    L, a = 0.1, 0.02
    mat = Material("t", 7800.0, 2.0e11, 2.5e8)
    m = mesh_box(a, a, L, 5e-3, order=2)
    z = m.nodes[:, 2]
    fixed = np.flatnonzero(z <= z.min() + 1e-12)
    top = (nodal_areas(m.nodes, boundary_triangles(m.tets)) > 0) & (z >= z.max() - 1e-12)
    total_N = 1000.0
    areas = nodal_areas(m.nodes, boundary_triangles(m.tets))
    f = np.zeros((len(m.nodes), 3))
    f[top, 2] = total_N * areas[top] / areas[top].sum()      # consistent traction
    res = run_ccx(write_inp(tmp_path / "bar.inp", m, mat, fixed_nodes=fixed,
                            nodal_forces_N=f, poisson=0.3))

    sigma_exact = total_N / (a * a)
    u_exact = sigma_exact * L / mat.elastic_modulus_Pa
    # primary check: mid-span axial stress, away from clamp and load patches
    cz = m.nodes[m.corners, 2].mean(axis=1)
    mid = np.abs(cz - L / 2) < 0.15 * L
    assert float(np.mean(res.stress_Pa[mid, 2])) == pytest.approx(sigma_exact, rel=0.01)
    # secondary: tip displacement, loose (end effects, see docstring)
    tip = np.flatnonzero(z >= z.max() - 1e-12)
    u = float(np.mean(res.displacement_m[np.searchsorted(res.node_ids, tip + 1), 2]))
    assert u == pytest.approx(u_exact, rel=0.05)              # measured +2.6%


@needs_gmsh
@needs_ccx
def test_bad_deck_raises_solver_error(tmp_path):
    """A deck ccx rejects must surface as SolverError, not a crash — meshing and
    solving failures are feasibility outcomes for a validator."""
    from propopt.hf.calculix import run_ccx
    bad = tmp_path / "bad.inp"
    bad.write_text("*NODE\n1, 0.0, 0.0, 0.0\n*STEP\n*STATIC\n*END STEP\n")
    with pytest.raises(SolverError):
        run_ccx(bad)


def test_clamp_set_cannot_be_empty(tmp_path):
    """No BC = a floating model = a meaningless run; refuse it up front."""
    from propopt.hf.calculix import write_inp
    from propopt.hf.mesh import TetMesh
    from propopt.solvers.beam import PLA
    m = TetMesh(nodes=np.zeros((4, 3)), tets=np.array([[0, 1, 2, 3]]),
                volume_m3=1.0, n_boundary_tris=4, quality_min=1.0,
                quality_mean=1.0, quality_p01=1.0, remeshed=False, order=1)
    with pytest.raises(SolverError, match="clamp"):
        write_inp(tmp_path / "x.inp", m, PLA, fixed_nodes=np.empty(0, dtype=int))
