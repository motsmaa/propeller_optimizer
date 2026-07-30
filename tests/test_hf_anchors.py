"""Phase-4a closed-form anchors (slow: each runs CalculiX on a refined mesh).

These are the tests that make a beam-vs-FEA comparison meaningful at all: they
prove the deck, the boundary conditions, the SI unit system, and the load
application against cases with exact answers, on a PRISMATIC specimen rather
than the blade. Deselected from `pytest -q`; run with `pytest -m slow`.

Tolerances are set to the *converged* discrepancy measured in the Phase-4a
report, and each has a stated physical cause — they are not slack to hide
error:
  * cantilever tip deflection lands ~0.5% BELOW Euler-Bernoulli because a fully
    clamped 3-D face also suppresses Poisson contraction and warping, which
    beam theory does not model (the FE model is genuinely stiffer).
  * centrifugal sigma_zz scatters ~1-3% about the analytic parabola because it
    is read as a mean of element-averaged integration-point stresses in a slab.
"""
import numpy as np
import pytest

from propopt.hf.wsl import WSLError


def _has(mod: str) -> bool:
    try:
        __import__(mod)
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


pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not _has("gmsh"), reason="gmsh not installed"),
              pytest.mark.skipif(not _has_ccx(), reason="CalculiX ccx not available")]

L, A_SIDE = 1.0, 0.04            # 1 m bar, 40 mm square: L/h = 25
NU = 0.3


def _material():
    from propopt.solvers.beam import Material
    return Material("anchor-steel", 7800.0, 2.1e11, 2.5e8)


def test_cantilever_tip_deflection_matches_beam_theory(tmp_path):
    """w_tip = q L^4 / (8 E I) for a uniform cantilever under uniform load."""
    from propopt.hf.calculix import run_ccx, write_inp
    from propopt.hf.mesh import boundary_triangles, mesh_box, nodal_areas

    mat = _material()
    q = 50.0
    inertia = A_SIDE ** 4 / 12.0
    w_exact = q * L ** 4 / (8.0 * mat.elastic_modulus_Pa * inertia)

    m = mesh_box(L, A_SIDE, A_SIDE, 0.013, order=2)      # span along X
    x = m.nodes[:, 0]
    fixed = np.flatnonzero(x <= x.min() + 1e-9)
    areas = nodal_areas(m.nodes, boundary_triangles(m.tets))
    side = (areas > 0) & (x > x.min() + 1e-9) & (x < x.max() - 1e-9)
    f = np.zeros((len(m.nodes), 3))
    f[side, 2] = q * L * areas[side] / areas[side].sum()   # uniform line load

    res = run_ccx(write_inp(tmp_path / "cant.inp", m, mat, fixed_nodes=fixed,
                            nodal_forces_N=f, poisson=NU))
    tip = np.flatnonzero(x >= x.max() - 1e-9)
    w = float(np.mean(res.displacement_m[np.searchsorted(res.node_ids, tip + 1), 2]))
    assert w == pytest.approx(w_exact, rel=0.02)          # measured -0.55%
    assert w < w_exact                                    # clamp over-constrains


def test_rotating_bar_axial_stress_matches_closed_form(tmp_path):
    """sigma_zz(z) = 0.5 rho omega^2 (R^2 - z^2) for a bar rotating about an
    axis through its root. Sampled away from the clamp, so the comparison is
    against the analytic profile rather than a single end value."""
    from propopt.hf.calculix import elements_in_slab, root_face_nodes, run_ccx, write_inp
    from propopt.hf.mesh import mesh_box

    mat = _material()
    omega = 6000.0 * 2.0 * np.pi / 60.0
    m = mesh_box(A_SIDE, A_SIDE, L, 0.013, order=2)       # span along Z
    res = run_ccx(write_inp(
        tmp_path / "cent.inp", m, mat, fixed_nodes=root_face_nodes(m, span_axis=2),
        omega_rad_s=omega, rotation_axis=(0.0, 1.0, 0.0),
        rotation_point=(0.0, 0.0, 0.0), poisson=NU))

    cz = m.nodes[m.corners, 2].mean(axis=1)
    for frac in (0.2, 0.5):
        z0 = frac * L
        sel = elements_in_slab(m, z0 - 0.03 * L, z0 + 0.03 * L)
        got = float(np.mean(res.stress_Pa[sel, 2]))
        exact = 0.5 * mat.density_kg_m3 * omega ** 2 * (L ** 2 - z0 ** 2)
        assert got == pytest.approx(exact, rel=0.04)      # measured within 1%
