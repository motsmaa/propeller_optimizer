"""ASCII STL surface -> volume tet mesh via gmsh (headless Python API).

UNITS: metres throughout. The STL written by `export._write_ascii_stl` is in
metres (the loft's vertices are SI), and nothing here rescales, so the node
coordinates handed to CalculiX are metres. See calculix.py for why that matters.

Two meshing modes, because the triangle loft forces a real trade-off:

  PRESERVE (remesh=False) — the STL facets become the boundary of the volume
    mesh unchanged. The meshed volume then equals the loft's exact
    divergence-theorem volume to round-off, so the FEA analyses precisely the
    solid the optimizer scored. Cost: the loft's facets are long thin
    station-to-station triangles (chordwise spacing ~1 mm vs spanwise ~15 mm),
    so boundary-conforming tets inherit that anisotropy.

  REMESH (remesh=True) — gmsh reclassifies the facets into a reparametrized
    geometry and generates a fresh isotropic surface mesh at `target_size_m`.
    Tet quality improves; the boundary is no longer bit-identical to the loft,
    so the enclosed volume DRIFTS. `TetMesh.volume_rel_error` reports the drift
    against the loft volume and `mesh_stl` raises SolverError past
    `max_volume_drift` — drift means the FEA is analysing a different blade
    than the optimizer scored, which silently invalidates any comparison.

Meshing failure is a FEASIBILITY outcome (SolverError), never a crash.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..contracts import SolverError

# Tet quality measure: gmsh "gamma" = inscribed/circumscribed radius ratio,
# normalized so 1.0 is equilateral and 0.0 is degenerate/flat. A common rule of
# thumb for trustworthy stress is min gamma > ~0.1 and mean > ~0.6.
QUALITY_MEASURE = "gamma"
GAMMA_FLOOR = 0.1

# gmsh's 10-node tet numbers its last two mid-edge nodes on edges (3,4) then
# (2,4); CalculiX/Abaqus C3D10 wants (2,4) then (3,4). Everything else agrees.
# Getting this wrong does NOT fail silently — ccx aborts with "nonpositive
# jacobian determinant" — but it costs a debugging cycle, so it is applied at
# extraction and the mesh is stored already in CalculiX order.
_GMSH_TO_CCX_TET10 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8])


@dataclass(frozen=True)
class TetMesh:
    """Tet mesh in METRES. `tets` is (M,4) for order 1 (C3D4) or (M,10) for
    order 2 (C3D10); the first four columns are always the corner nodes.

    Order 2 matters: linear tets lock in bending and badly UNDER-predict
    deflection, so the cantilever anchor needs C3D10 to converge at a sane
    element count."""
    nodes: np.ndarray            # (N, 3) float, metres
    tets: np.ndarray             # (M, 4|10) int, 0-based into `nodes`
    volume_m3: float             # summed tet volume (corner nodes)
    n_boundary_tris: int
    quality_min: float
    quality_mean: float
    quality_p01: float           # 1st percentile — the tail that ruins stress
    remeshed: bool
    order: int = 1
    reference_volume_m3: float | None = None   # loft volume, if supplied
    quality: np.ndarray | None = None          # per-element gamma, same order as tets

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_tets(self) -> int:
        return int(self.tets.shape[0])

    @property
    def corners(self) -> np.ndarray:
        return self.tets[:, :4]

    @property
    def ccx_element_type(self) -> str:
        return "C3D10" if self.order == 2 else "C3D4"

    @property
    def volume_rel_error(self) -> float | None:
        """Signed relative volume drift vs the loft's exact volume."""
        if self.reference_volume_m3 is None or self.reference_volume_m3 == 0.0:
            return None
        return (self.volume_m3 - self.reference_volume_m3) / self.reference_volume_m3


def tet_volumes(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Signed volume of each tet: det[b-a, c-a, d-a] / 6 (corner nodes only)."""
    a, b, c, d = (nodes[tets[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0


def boundary_triangles(tets: np.ndarray) -> np.ndarray:
    """Corner-node triangles on the mesh boundary: the faces owned by exactly
    one tet. Used to map surface tractions to nodes (see nodal_areas)."""
    corners = np.asarray(tets)[:, :4]
    faces = np.vstack([corners[:, [0, 2, 1]], corners[:, [0, 1, 3]],
                       corners[:, [1, 2, 3]], corners[:, [0, 3, 2]]])
    key = np.sort(faces, axis=1)
    _, idx, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    return faces[idx[counts == 1]]


def nodal_areas(nodes: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Tributary area per node (m^2): each triangle's area split equally among
    its three vertices. Sums to the total surface area, so distributing a force
    proportional to this reproduces a uniform traction in the limit."""
    p0, p1, p2 = (nodes[tris[:, i]] for i in range(3))
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    out = np.zeros(len(nodes))
    for col in range(3):
        np.add.at(out, tris[:, col], area / 3.0)
    return out


def mesh_stl(
    stl_path: str | Path,
    target_size_m: float,
    *,
    remesh: bool = False,
    order: int = 2,
    min_size_m: float | None = None,
    reference_volume_m3: float | None = None,
    max_volume_drift: float = 0.01,
    classify_angle_deg: float = 40.0,
    optimize: bool = True,
) -> TetMesh:
    """Tet-mesh the interior of a closed ASCII STL surface.

    `target_size_m` is the characteristic element length in METRES. Raises
    SolverError on any gmsh failure, an empty/degenerate mesh, inverted tets,
    or (remesh mode) volume drift beyond `max_volume_drift`.
    """
    try:
        import gmsh
    except ImportError as exc:  # optional `hf` extra
        raise SolverError("gmsh is not installed (pip install -e '.[hf]')") from exc

    stl_path = Path(stl_path)
    if not stl_path.is_file():
        raise SolverError(f"STL not found: {stl_path}")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(target_size_m))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin",
                              float(min_size_m if min_size_m is not None
                                    else target_size_m / 5.0))
        gmsh.option.setNumber("Mesh.Optimize", 1 if optimize else 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1 if optimize else 0)

        # gmsh chatters on stdout even at Verbosity 1; keep the library quiet so
        # a solver run does not pollute the caller's output.
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                gmsh.merge(str(stl_path))
            except Exception as exc:
                raise SolverError(f"gmsh could not read {stl_path.name}: {exc}") from exc

            if remesh:
                # Reclassify facets into sheets separated by >classify_angle_deg
                # and build a parametrization gmsh can mesh freshly. This is
                # what perturbs the boundary (hence the volume check below).
                gmsh.model.mesh.classifySurfaces(
                    np.deg2rad(classify_angle_deg), True, False,
                    np.deg2rad(classify_angle_deg))
                gmsh.model.mesh.createGeometry()

            surfaces = gmsh.model.getEntities(2)
            if not surfaces:
                raise SolverError(f"{stl_path.name}: no surface found after import")
            try:
                loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
                gmsh.model.geo.addVolume([loop])
                gmsh.model.geo.synchronize()
            except Exception as exc:
                raise SolverError(
                    f"{stl_path.name}: could not form a closed volume from the "
                    f"surface (is the STL watertight?): {exc}") from exc

            try:
                gmsh.model.mesh.generate(3)
            except Exception as exc:
                raise SolverError(f"{stl_path.name}: 3-D meshing failed: {exc}") from exc

            if order == 2:
                gmsh.model.mesh.setOrder(2)
            nodes, tets, n_tris, qual = _extract(gmsh, order, stl_path.name)
    finally:
        gmsh.finalize()

    vols = tet_volumes(nodes, tets)
    if np.any(vols <= 0.0):                          # inverted / flat elements
        n_bad = int(np.sum(vols <= 0.0))
        raise SolverError(f"{n_bad} non-positive-volume tets in the mesh")
    volume = float(np.sum(vols))
    if not np.isfinite(volume) or volume <= 0.0:
        raise SolverError(f"non-physical mesh volume {volume!r}")

    mesh = TetMesh(
        nodes=nodes, tets=tets, volume_m3=volume, n_boundary_tris=n_tris,
        quality_min=float(qual.min()), quality_mean=float(qual.mean()),
        quality_p01=float(np.percentile(qual, 1.0)), quality=qual,
        remeshed=bool(remesh), order=int(order),
        reference_volume_m3=reference_volume_m3,
    )
    drift = mesh.volume_rel_error
    if drift is not None and abs(drift) > max_volume_drift:
        raise SolverError(
            f"meshed volume drifted {drift:+.2%} from the loft volume "
            f"(limit {max_volume_drift:.1%}): the FEA would analyse a different "
            "solid than the optimizer scored")
    return mesh


def _extract(gmsh, order: int, label: str):
    """Pull nodes/tets/quality out of the current gmsh model, compacting the
    (1-based, possibly sparse) gmsh node tags to 0-based indices."""
    gtype = 11 if order == 2 else 4      # 11 = 10-node tet, 4 = 4-node tet
    ncol = 10 if order == 2 else 4
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    etypes, etags, enodes = gmsh.model.mesh.getElements(3)
    _, tri_tags, _ = gmsh.model.mesh.getElements(2)
    n_tris = int(sum(len(t) for t in tri_tags))

    tet_tags = np.empty(0, dtype=np.int64)
    conn = np.empty((0, ncol), dtype=np.int64)
    for etype, tags, nds in zip(etypes, etags, enodes):
        if etype == gtype:
            tet_tags = np.asarray(tags, dtype=np.int64)
            conn = np.asarray(nds, dtype=np.int64).reshape(-1, ncol)
            break
    if conn.size == 0:
        raise SolverError(f"{label}: mesher produced no order-{order} tetrahedra")

    qual = np.asarray(gmsh.model.mesh.getElementQualities(tet_tags, QUALITY_MEASURE),
                      dtype=float)
    tags = np.asarray(node_tags, dtype=np.int64)
    nodes = np.asarray(coords, dtype=float).reshape(-1, 3)
    perm = np.argsort(tags)
    tags, nodes = tags[perm], nodes[perm]
    tets = np.searchsorted(tags, conn)
    if np.any(tags[tets] != conn):
        raise SolverError(f"{label}: connectivity references unknown node tags")
    if order == 2:
        tets = tets[:, _GMSH_TO_CCX_TET10]
    return nodes, tets, n_tris, qual


def mesh_box(lx: float, ly: float, lz: float, target_size_m: float,
             *, order: int = 2, optimize: bool = True) -> TetMesh:
    """Tet-mesh a box [0,lx]x[0,ly]x[0,lz] (METRES) — the prismatic specimen for
    the closed-form anchors. Uses gmsh's OCC kernel, so the geometry is exact
    and any anchor error is the FE discretization, not a faceted boundary."""
    try:
        import gmsh
    except ImportError as exc:
        raise SolverError("gmsh is not installed (pip install -e '.[hf]')") from exc

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(target_size_m))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(target_size_m) / 3.0)
        gmsh.option.setNumber("Mesh.Optimize", 1 if optimize else 0)
        with contextlib.redirect_stdout(io.StringIO()):
            gmsh.model.occ.addBox(0.0, 0.0, 0.0, lx, ly, lz)
            gmsh.model.occ.synchronize()
            try:
                gmsh.model.mesh.generate(3)
                if order == 2:
                    gmsh.model.mesh.setOrder(2)
            except Exception as exc:
                raise SolverError(f"box meshing failed: {exc}") from exc
            nodes, tets, n_tris, qual = _extract(gmsh, order, "box")
    finally:
        gmsh.finalize()

    vols = tet_volumes(nodes, tets)
    if np.any(vols <= 0.0):
        raise SolverError("box mesh contains non-positive-volume tets")
    return TetMesh(nodes=nodes, tets=tets, volume_m3=float(np.sum(vols)),
                   n_boundary_tris=n_tris, quality_min=float(qual.min()),
                   quality_mean=float(qual.mean()),
                   quality_p01=float(np.percentile(qual, 1.0)), quality=qual,
                   remeshed=False, order=int(order),
                   reference_volume_m3=lx * ly * lz)
