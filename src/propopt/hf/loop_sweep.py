"""Phase-5 slice 5a: sweep the section along a 3-D spine with gmsh's OCC kernel.

Lives under `hf/` so gmsh stays in the optional extra; the frames it consumes
come from `loop_geometry`, which is pure numpy and inside the runtime deps.

METHOD: place the section wire at each of `n_loft_sections` arc-length stations
using the ROTATION-MINIMIZING frame, then `addThruSections(makeSolid=True,
makeRuled=True)`. Deliberately NOT `occ.addPipe`: OCC's pipe sweeps with a
Frenet or a fixed-trihedron law, and Frenet is exactly the frame that flips at
an inflection point. Placing the sections explicitly keeps the frame ours.

SECTION PLACEMENT reduces EXACTLY to the radial path's. `geometry.build_blade_solid`
writes, with the section's chordwise xi and thickness-wise eta:

    x_tangential = xi cos(-twist) - eta sin(-twist)
    y_axial      = xi sin(-twist) + eta cos(-twist)
    z_radial     = r                                     (the span direction)

Here the span direction is the spine TANGENT T, and the section plane is
spanned by the transported normal N and the binormal B = T x N. Mapping
xi -> N and eta -> B makes the two identical whenever (N, B, T) = (x, y, z),
so the loop generator is a strict generalisation rather than a second
convention. The rotation sense (-twist) is inherited, not re-derived; it was
settled empirically from blade surface pressure in Phase 4b.

SECTION SAMPLING IS DENSIFIED relative to the genome. Phase 4a established that
the as-scored station count (6 x 25) is too coarse and too anisotropic to mesh —
remeshing failed outright on it — and that densifying the sampling of the SAME
parametric design is what makes it meshable. Chord and twist are therefore
interpolated from the genome's stations onto `n_loft_sections` arc-length
positions. The design is unchanged; only how finely it is sampled changes.

FINITE TE IS MANDATORY, inherited from `occ_geometry.station_rings`: a sharp TE
gives a cusped profile wire, which is the case a B-rep kernel tolerates least.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..contracts import LoopDesign, SolverError
from ..geometry import (DEFAULT_N_CHORD, DEFAULT_SECTION, PITCH_AXIS_CHORD_FRAC,
                        SectionFn, section_polygon)
from ..loop_geometry import spine_frames
from .mesh import TetMesh, _extract, tet_volumes


@dataclass(frozen=True)
class SweptBlade:
    """Result of an OCC sweep (volume from the kernel, not from a mesh)."""
    volume_m3: float
    n_sections: int
    n_profile_points: int
    length_m: float
    min_section_gap_m: float
    surface_nodes: np.ndarray | None = None    # (N,3), shared vertices
    surface_tris: np.ndarray | None = None     # (T,3) into surface_nodes


def sweep_rings(design: LoopDesign,
                section: SectionFn = DEFAULT_SECTION,
                n_chord: int = DEFAULT_N_CHORD,
                te_thickness_m: float = 2.0e-4,
                n_loft_sections: int = 40) -> tuple[np.ndarray, float, float]:
    """(n_sections, P, 3) section vertices in METRES, plus spine length and the
    smallest gap between consecutive section planes.

    The gap is returned because `addThruSections` fails opaquely when
    consecutive wires touch or cross; reporting it turns that into a diagnosable
    number instead of an OCC message.
    """
    te = float(te_thickness_m)
    if not np.isfinite(te) or te <= 0.0:
        raise SolverError(
            "the swept path requires a finite trailing edge "
            f"(te_thickness_m > 0), got {te!r}; a sharp TE gives a cusped "
            "profile wire, which is unsupported in the HF path")
    if n_loft_sections < 2:
        raise SolverError("need >= 2 loft sections")

    fr = spine_frames(design)
    s_loft = np.linspace(0.0, 1.0, n_loft_sections)
    st_s = np.asarray(design.station_s, float)
    chord = np.interp(s_loft, st_s, np.asarray(design.chord_m, float))
    twist = np.interp(s_loft, st_s, np.asarray(design.twist_rad, float))
    if np.any(chord <= 0.0):
        raise SolverError("chord must be positive at every loft section")

    pts = np.column_stack([np.interp(s_loft, fr.s, fr.points_m[:, k])
                           for k in range(3)])
    T = np.column_stack([np.interp(s_loft, fr.s, fr.tangent[:, k]) for k in range(3)])
    N = np.column_stack([np.interp(s_loft, fr.s, fr.normal[:, k]) for k in range(3)])
    # Re-orthonormalise after interpolation: interpolating unit vectors does not
    # preserve unit length or orthogonality, and a non-orthonormal frame would
    # shear every section it places.
    T /= np.linalg.norm(T, axis=1)[:, None]
    N -= np.einsum("ij,ij->i", N, T)[:, None] * T
    nn = np.linalg.norm(N, axis=1)
    if np.any(nn <= 0.0):
        raise SolverError("interpolated frame normal collapsed onto the tangent")
    N /= nn[:, None]
    B = np.cross(T, N)

    rings = np.empty((n_loft_sections, 0, 3))
    out = []
    for i in range(n_loft_sections):
        poly = section_polygon(section, n_chord, te / chord[i])
        xi = (poly[:, 0] - PITCH_AXIS_CHORD_FRAC) * chord[i]
        eta = poly[:, 1] * chord[i]
        c_t, s_t = np.cos(-twist[i]), np.sin(-twist[i])
        a = xi * c_t - eta * s_t          # -> N   (the radial path's x)
        b = xi * s_t + eta * c_t          # -> B   (the radial path's y)
        out.append(pts[i] + a[:, None] * N[i] + b[:, None] * B[i])
    rings = np.stack(out)

    gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return rings, fr.length_m, float(gaps.min())


def _build(gmsh, rings: np.ndarray) -> int:
    n_st, p, _ = rings.shape
    try:
        wires = []
        for ring in rings:
            ids = [gmsh.model.occ.addPoint(*map(float, v)) for v in ring]
            lines = [gmsh.model.occ.addLine(ids[i], ids[(i + 1) % p])
                     for i in range(p)]
            wires.append(gmsh.model.occ.addCurveLoop(lines))
        res = gmsh.model.occ.addThruSections(wires, makeSolid=True, makeRuled=True)
        gmsh.model.occ.synchronize()
    except Exception as exc:
        raise SolverError(f"OCC sweep failed: {exc}") from exc
    vols = [tag for dim, tag in res if dim == 3]
    if len(vols) != 1:
        raise SolverError(f"OCC sweep produced {len(vols)} volumes (expected 1)")
    return vols[0]


def sweep_blade(design: LoopDesign,
                section: SectionFn = DEFAULT_SECTION,
                n_chord: int = DEFAULT_N_CHORD,
                te_thickness_m: float = 2.0e-4,
                n_loft_sections: int = 40,
                surface: bool = False,
                surface_size_m: float | None = None) -> SweptBlade:
    """Build the swept solid; with `surface=True` also return its surface mesh.

    The surface mesh is returned as SHARED-VERTEX (nodes, triangles) arrays,
    not as an STL, so an INDEPENDENT library (trimesh, test-only) can grade
    watertightness on exactly the convention the repo already uses for the
    triangle loft (`trimesh.Trimesh(..., process=False)`). Writing STL would
    defeat the check outright: STL is a triangle soup with no shared vertices,
    so `is_watertight` is False for every closed solid until the vertices are
    merged, and merging is a repair step the cross-validator must not perform.
    """
    try:
        import gmsh
    except ImportError as exc:
        raise SolverError("gmsh is not installed (pip install -e '.[hf]')") from exc

    rings, length, gap = sweep_rings(design, section, n_chord, te_thickness_m,
                                     n_loft_sections)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
        with contextlib.redirect_stdout(io.StringIO()):
            tag = _build(gmsh, rings)
            volume = float(gmsh.model.occ.getMass(3, tag))
            nodes = tris = None
            if surface:
                sz = surface_size_m or float(np.median(
                    np.linalg.norm(np.diff(rings[:, 0], axis=0), axis=1)))
                gmsh.option.setNumber("Mesh.MeshSizeMax", sz)
                gmsh.option.setNumber("Mesh.MeshSizeMin", sz / 4.0)
                gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
                try:
                    gmsh.model.mesh.generate(2)
                except Exception as exc:
                    raise SolverError(f"surface meshing failed: {exc}") from exc
                nodes, tris = _surface_mesh(gmsh)
        if not np.isfinite(volume) or volume <= 0.0:
            raise SolverError(f"swept solid has non-physical volume {volume!r}")
    finally:
        gmsh.finalize()
    return SweptBlade(volume, rings.shape[0], rings.shape[1], length, gap,
                      nodes, tris)


def _surface_mesh(gmsh):
    """Shared-vertex (nodes, triangles) of the current 2-D mesh."""
    ntags, coords, _ = gmsh.model.mesh.getNodes()
    etypes, _, enodes = gmsh.model.mesh.getElements(2)
    conn = np.empty((0, 3), dtype=np.int64)
    for et, nd in zip(etypes, enodes):
        if et == 2:                                   # 3-node triangle
            conn = np.asarray(nd, dtype=np.int64).reshape(-1, 3)
            break
    if conn.size == 0:
        raise SolverError("sweep produced no surface triangles")
    tags = np.asarray(ntags, dtype=np.int64)
    pts = np.asarray(coords, dtype=float).reshape(-1, 3)
    order = np.argsort(tags)
    tags, pts = tags[order], pts[order]
    faces = np.searchsorted(tags, conn)
    if np.any(tags[faces] != conn):
        raise SolverError("surface connectivity references unknown node tags")
    return pts, faces


def mesh_swept_blade(design: LoopDesign,
                     target_size_m: float,
                     *,
                     section: SectionFn = DEFAULT_SECTION,
                     n_chord: int = DEFAULT_N_CHORD,
                     te_thickness_m: float = 2.0e-4,
                     n_loft_sections: int = 40,
                     order: int = 1,
                     reference_volume_m3: float | None = None,
                     max_volume_drift: float = 0.02,
                     optimize: bool = True) -> TetMesh:
    """Tet-mesh the swept solid natively. WATERTIGHT IS NOT MESHABLE — Phase 4a
    got gamma_min = 0.000 from a watertight surface with a knife edge — so the
    gate reports quality, not just success."""
    try:
        import gmsh
    except ImportError as exc:
        raise SolverError("gmsh is not installed (pip install -e '.[hf]')") from exc

    rings, _, _ = sweep_rings(design, section, n_chord, te_thickness_m,
                              n_loft_sections)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(target_size_m))
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(target_size_m))
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.Optimize", 1 if optimize else 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1 if optimize else 0)
        with contextlib.redirect_stdout(io.StringIO()):
            _build(gmsh, rings)
            try:
                gmsh.model.mesh.generate(3)
                if order == 2:
                    gmsh.model.mesh.setOrder(2)
            except Exception as exc:
                raise SolverError(f"swept 3-D meshing failed: {exc}") from exc
            nodes, tets, n_tris, qual = _extract(gmsh, order, "swept-blade")
    finally:
        gmsh.finalize()

    vols = tet_volumes(nodes, tets)
    if np.any(vols <= 0.0):
        raise SolverError(
            f"{int(np.sum(vols <= 0.0))} non-positive-volume tets in the sweep")
    mesh = TetMesh(nodes=nodes, tets=tets, volume_m3=float(np.sum(vols)),
                   n_boundary_tris=n_tris, quality_min=float(qual.min()),
                   quality_mean=float(qual.mean()),
                   quality_p01=float(np.percentile(qual, 1.0)), quality=qual,
                   remeshed=False, order=int(order),
                   reference_volume_m3=reference_volume_m3)
    drift = mesh.volume_rel_error
    if drift is not None and abs(drift) > max_volume_drift:
        raise SolverError(
            f"swept mesh volume drifted {drift:+.2%} from the kernel volume "
            f"(limit {max_volume_drift:.1%})")
    return mesh
