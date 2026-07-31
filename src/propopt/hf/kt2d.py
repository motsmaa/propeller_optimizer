"""Phase-4a slice 3b: 2-D plane-strain stress-concentration models.

PURPOSE: produce the Kt(r/t) DESIGN CURVES for the blade-root fillet, so
`beam.py`'s root guard is selected by intended manufacturing radius rather than
left at the old 2.5 placeholder. This module is the MEASURING INSTRUMENT; the
resulting tables live in `propopt.kt_root`, which is what `beam.py` reads. Run
this only to re-derive or extend those tables (it needs gmsh + CalculiX);
nothing in the optimizer loop imports it.

KT DEFINITION — COMPOSES WITH THE BEAM. `beam.py` now computes

    sigma = Kt_axial * N/A + Kt_bending * sigma_bend

so each component is amplified by the factor measured for ITS OWN load case:

    Kt = sigma_peak(fillet surface) / sigma_nom(wall section)

with both taken from the SAME 2-D model so any modelling bias cancels, and
sigma_nom checked against the elementary 6M/t^2 (bending) or F/t (axial) that
`beam.py` uses. `load="bending"` applies a pure moment; `load="axial"` applies
uniform tension. Both are production curves — the axial one was previously
reported as a diagnostic only, because the beam did not amplify N/A at all.

UNITS: SI. Plane strain, unit out-of-plane depth, so forces are per metre.

PLANE-STRAIN VALIDITY: the section is cut perpendicular to the chord at
MID-CHORD, where the blade is locally a long prismatic wall and the
out-of-plane (chordwise) strain is genuinely suppressed. Near the LE and TE
the section is short in the chordwise direction and the true state approaches
plane STRESS; Kt there is not covered by this model. 2-D throughout — a 3-D
confirmation on the real filleted blade is a separate step.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..contracts import SolverError
from ..solvers.beam import Material
from .calculix import parse_dat, run_ccx

# 8-node serendipity quad = CalculiX CPE8 (plane strain). Quadratic elements
# are required: linear quads badly under-read a fillet peak.
GMSH_QUAD8 = 16
CCX_PLANE_STRAIN = "CPE8"
# CPE6 (6-node triangle) is used instead of CPE8 against curved boundaries:
# measured aspect ratio at the hole is 1.05-1.19 for triangles vs 1.14-2.32
# for recombined quads, and the quad AR DEGRADES with refinement (global max
# 2.67 -> 4.41) which is what corrupted the extrapolated nodal peak.
GMSH_TRI6 = 9
CCX_PLANE_STRAIN_TRI = "CPE6"


@dataclass(frozen=True)
class Kt2DResult:
    kt: float
    sigma_peak_Pa: float
    sigma_nom_Pa: float
    sigma_nom_analytic_Pa: float
    n_elements: int
    n_nodes: int
    elem_across_fillet: float

    @property
    def nom_error(self) -> float:
        return self.sigma_nom_Pa / self.sigma_nom_analytic_Pa - 1.0


def _mesh_planar(build, refine_curves, fine, coarse, zone, order2=True):
    """Mesh a planar OCC face with 8-node quads, refined near `refine_curves`.

    `build(gmsh) -> (surface_tags, refine_curve_tags)`. Curve tags are taken
    from the caller's own construction — never from diffing tags across a
    boolean."""
    try:
        import gmsh
    except ImportError as exc:
        raise SolverError("gmsh is not installed (pip install -e '.[hf]')") from exc
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
        with contextlib.redirect_stdout(io.StringIO()):
            build(gmsh)
            curves = refine_curves(gmsh)
            if curves:
                f = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(f, "CurvesList", curves)
                gmsh.model.mesh.field.setNumber(f, "Sampling", 400)
                th = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(th, "InField", f)
                gmsh.model.mesh.field.setNumber(th, "SizeMin", fine)
                gmsh.model.mesh.field.setNumber(th, "SizeMax", coarse)
                # the refined ZONE is a fixed geometric distance, NOT a multiple
                # of the element size: otherwise shrinking `fine` also shrinks the
                # refined region and the element count never grows, so a
                # "convergence ladder" would measure mesh noise instead of
                # convergence (observed: 259 -> 252 elements over a 4x refinement).
                gmsh.model.mesh.field.setNumber(th, "DistMin", fine)
                gmsh.model.mesh.field.setNumber(th, "DistMax", zone)
                gmsh.model.mesh.field.setAsBackgroundMesh(th)
            for o in ("Mesh.MeshSizeExtendFromBoundary",
                      "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"):
                gmsh.option.setNumber(o, 0)
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
            gmsh.option.setNumber("Mesh.MeshSizeMax", coarse)
            gmsh.option.setNumber("Mesh.MeshSizeMin", fine)
            try:
                gmsh.model.mesh.generate(2)
                if order2:
                    gmsh.model.mesh.setOrder(2)
            except Exception as exc:
                raise SolverError(f"2-D meshing failed: {exc}") from exc
            tags, coord, _ = gmsh.model.mesh.getNodes()
            etypes, etags, enodes = gmsh.model.mesh.getElements(2)
            conn = None
            for et, _, nd in zip(etypes, etags, enodes):
                if et == GMSH_QUAD8:
                    conn = np.asarray(nd, dtype=np.int64).reshape(-1, 8)
            if conn is None:
                raise SolverError("mesher produced no 8-node quads (CPE8)")
        nodes = np.asarray(coord, dtype=float).reshape(-1, 3)
        tg = np.asarray(tags, dtype=np.int64)
        perm = np.argsort(tg)
        tg, nodes = tg[perm], nodes[perm]
        elems = np.searchsorted(tg, conn)
        if np.any(tg[elems] != conn):
            raise SolverError("2-D connectivity references unknown nodes")
    finally:
        gmsh.finalize()
    return nodes[:, :2], elems


def write_plane_strain_inp(path, nodes2d, elems, material: Material,
                           fixed, nodal_forces, depth_m: float = 1.0,
                           elem_type: str = CCX_PLANE_STRAIN) -> Path:
    """CalculiX plane-strain deck. `fixed` maps 0-based node -> tuple of dofs
    (1=x, 2=y). `nodal_forces` is (N,2) in N per `depth_m` of depth."""
    path = Path(path)
    lines = ["** propopt slice-3b plane-strain Kt model",
             "** UNITS: SI (m, N, Pa). Plane strain, unit depth.",
             "*NODE, NSET=NALL"]
    for i, (x, y) in enumerate(nodes2d, start=1):
        lines.append(f"{i}, {x:.10e}, {y:.10e}, 0.0")
    lines.append(f"*ELEMENT, TYPE={elem_type}, ELSET=EALL")
    for e, c in enumerate(elems + 1, start=1):
        lines.append(f"{e}, " + ", ".join(str(int(v)) for v in c))
    lines += ["*MATERIAL, NAME=MAT", "*ELASTIC",
              f"{material.elastic_modulus_Pa:.6e}, {material.poisson:.6f}",
              "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT", f"{depth_m:.6e}",
              "*STEP", "*STATIC", "*BOUNDARY"]
    for n, dofs in fixed.items():
        for dof in dofs:
            lines.append(f"{int(n) + 1}, {dof}, {dof}, 0.0")
    loaded = np.flatnonzero(np.linalg.norm(nodal_forces, axis=1) > 0.0)
    if loaded.size:
        lines.append("*CLOAD")
        for n in loaded:
            for dof in (0, 1):
                f = float(nodal_forces[n, dof])
                if f != 0.0:
                    lines.append(f"{int(n) + 1}, {dof + 1}, {f:.10e}")
    # NODE FILE writes EXTRAPOLATED nodal stresses to .frd. That is what a
    # surface peak needs: an element-centroid value sits ~h/2 inside a steep
    # gradient and under-reads (measured ~13% low on the Kirsch anchor).
    lines += ["*NODE PRINT, NSET=NALL", "U",
              "*NODE FILE", "U,S",
              "*EL PRINT, ELSET=EALL", "S", "*END STEP", ""]
    path.write_text("\n".join(lines))
    return path


def edge_consistent_load(nodes2d, elems, on_edge, traction):
    """CONSISTENT nodal load vector for a traction on a quadratic (CPE8) edge.

    Tributary-length weighting is WRONG for quadratic elements: for a uniform
    traction the exact weights are (L/6, 2L/3, L/6) for (corner, midside,
    corner), not (~L/4, L/2, L/4). Using the wrong weights makes even a
    uniform-tension patch test fail by tens of percent at the loaded edge,
    which then contaminates both sigma_nom and the peak.

    `traction(x, y) -> (tx, ty)` in Pa. Integrated with 3-point Gauss, exact
    for a quadratic shape function times a linear traction (the pure-bending
    case), hence exact for both load cases used here.
    """
    # 8-node quad edges: (corner, midside, corner)
    EDGES = (((0, 3, 1), (1, 4, 2), (2, 5, 0)) if elems.shape[1] == 6
             else ((0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)))
    gx = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
    gw = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    f = np.zeros((len(nodes2d), 2))
    n_edges = 0
    for el in elems:
        for a, m, b in EDGES:
            ids = (el[a], el[m], el[b])
            if not all(on_edge[i] for i in ids):
                continue
            n_edges += 1
            pa, pm, pb = (nodes2d[i] for i in ids)
            for xi, w in zip(gx, gw):
                N = np.array([xi * (xi - 1.0) / 2.0, 1.0 - xi * xi,
                              xi * (xi + 1.0) / 2.0])
                dN = np.array([xi - 0.5, -2.0 * xi, xi + 0.5])
                jac = np.linalg.norm(dN[0] * pa + dN[1] * pm + dN[2] * pb)
                pt = N[0] * pa + N[1] * pm + N[2] * pb
                tx, ty = traction(pt[0], pt[1])
                for k, i in enumerate(ids):
                    f[i, 0] += w * N[k] * tx * jac
                    f[i, 1] += w * N[k] * ty * jac
    if n_edges == 0:
        raise SolverError("no element edges found on the loaded boundary")
    return f


def parse_frd_stresses(frd_path) -> dict:
    """Node id -> (sxx,syy,szz,sxy,syz,szx) from a CalculiX .frd STRESS block.
    Fixed width: ' -1' + 10-char node + 6 x 12-char values.

    NODE IDENTITY IS VERIFIED, not assumed: CalculiX expands plane-strain
    elements into a 3-D layer internally, but it writes results back on the
    ORIGINAL 2-D numbering — confirmed by parsing the .frd 2C coordinate block
    and matching it against the INP node-for-node (agreement limited only by
    the .frd's 6-significant-digit text format)."""
    out, inblock = {}, False
    for line in Path(frd_path).read_text().splitlines():
        if "STRESS" in line and line.lstrip().startswith("-4"):
            inblock = True
            continue
        if inblock:
            if line.startswith(" -3"):
                break
            if line.startswith(" -1"):
                try:
                    out[int(line[3:13])] = tuple(
                        float(line[13 + 12*k: 25 + 12*k]) for k in range(6))
                except ValueError:
                    continue
    if not out:
        raise SolverError(f"{Path(frd_path).name}: no nodal STRESS block")
    return out


def frd_node_coords(frd_path) -> dict:
    """Node id -> (x,y,z) from the .frd 2C block, for identity verification."""
    out, inblock = {}, False
    for line in Path(frd_path).read_text().splitlines():
        if line.strip().startswith("2C"):
            inblock = True
            continue
        if inblock:
            if line.startswith(" -3"):
                break
            if line.startswith(" -1"):
                try:
                    out[int(line[3:13])] = tuple(
                        float(line[13 + 12*k: 25 + 12*k]) for k in range(3))
                except ValueError:
                    continue
    return out


def nodal_stress_array(inp_path, n_nodes: int) -> np.ndarray:
    p = Path(inp_path)
    raw = parse_frd_stresses(p.parent / (p.with_suffix("").name + ".frd"))
    arr = np.zeros((n_nodes, 6))
    for nid, s in raw.items():
        if 1 <= nid <= n_nodes:
            arr[nid - 1] = s
    return arr


def verify_node_identity(inp_path, nodes2d, rtol=1e-5) -> float:
    """Max |coordinate| discrepancy between the INP and the .frd 2C block.
    Raises if the two numberings do not correspond.

    Tolerance is RELATIVE to the model size because .frd is a 6-significant-
    digit text format: on a 50 mm model that alone gives ~5e-8 m of apparent
    disagreement, which is formatting, not renumbering."""
    p = Path(inp_path)
    coords = frd_node_coords(p.parent / (p.with_suffix("").name + ".frd"))
    if len(coords) != len(nodes2d):
        raise SolverError(
            f"node-count mismatch: INP {len(nodes2d)}, .frd {len(coords)} — "
            "CalculiX renumbered, so results cannot be mapped back")
    worst = 0.0
    for nid, (x, y, _z) in coords.items():
        worst = max(worst, abs(x - nodes2d[nid-1, 0]), abs(y - nodes2d[nid-1, 1]))
    scale = float(np.abs(nodes2d).max())
    if worst > rtol * scale:
        raise SolverError(
            f"node identity broken: worst coord diff {worst:.3e} m "
            f"(> {rtol:.0e} x model size {scale:.3e} m)")
    return worst


def _mesh_quarter_hole(a, W, fine, coarse, zone):
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        with contextlib.redirect_stdout(io.StringIO()):
            rect = gmsh.model.occ.addRectangle(0, 0, 0, W, W)
            disk = gmsh.model.occ.addDisk(0, 0, 0, a, a)
            gmsh.model.occ.cut([(2, rect)], [(2, disk)])
            gmsh.model.occ.synchronize()
            arcs = []
            for dim, tag in gmsh.model.getEntities(1):
                lo, hi = gmsh.model.getParametrizationBounds(dim, tag)
                ts = np.linspace(float(lo[0]), float(hi[0]), 9)
                p = np.array(gmsh.model.getValue(dim, tag, list(ts))).reshape(-1, 3)
                if np.all(np.abs(np.hypot(p[:, 0], p[:, 1]) - a) < 1e-6 * a):
                    arcs.append(tag)
            f = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(f, "CurvesList", arcs)
            gmsh.model.mesh.field.setNumber(f, "Sampling", 600)
            th = gmsh.model.mesh.field.add("Threshold")
            # DistMin > 0 holds `fine` over a band of FIXED width, so halving
            # the element size actually fills that band with more elements and
            # the total count grows; with DistMin = 0 the size ramps up
            # immediately and the count stays flat (249 -> 275 over a 6x
            # refinement), which makes the ladder meaningless.
            for k, v in (("InField", f), ("SizeMin", fine), ("SizeMax", coarse),
                         ("DistMin", zone), ("DistMax", 6.0 * zone)):
                gmsh.model.mesh.field.setNumber(th, k, v)
            gmsh.model.mesh.field.setAsBackgroundMesh(th)
            for o in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints",
                      "Mesh.MeshSizeFromCurvature"):
                gmsh.option.setNumber(o, 0)
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
            gmsh.option.setNumber("Mesh.MeshSizeMax", coarse)
            # MeshSizeMin MUST be 0 here: setting it to `fine` suppresses the
            # background field and the feature stops refining (measured: arc
            # nodes stuck at 9->10 over a 4x refinement, vs 5->17 once freed).
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.setOrder(2)
            tg, cd, _ = gmsh.model.mesh.getNodes()
            et, _, en = gmsh.model.mesh.getElements(2)
            conn = None
            for e, n in zip(et, en):
                if e == GMSH_TRI6:
                    conn = np.asarray(n, dtype=np.int64).reshape(-1, 6)
            if conn is None:
                raise SolverError("no CPE6 triangles produced")
        nodes = np.asarray(cd).reshape(-1, 3)
        tg = np.asarray(tg, dtype=np.int64)
        perm = np.argsort(tg); tg, nodes = tg[perm], nodes[perm]
        elems = np.searchsorted(tg, conn)
        # drop orphan nodes (referenced by no element): CalculiX omits them
        # from .frd output, which would break the id->row mapping
        used = np.zeros(len(nodes), bool); used[elems.ravel()] = True
        remap = -np.ones(len(nodes), np.int64); remap[used] = np.arange(used.sum())
        nodes, elems = nodes[used], remap[elems]
    finally:
        gmsh.finalize()
    return nodes[:, :2], elems


def kirsch_kt(hole_r_m=1.0e-3, width_ratio=50.0, elem_across=8.0,
              material=None, work_dir="runs/kt2d", tag="kirsch"):
    """Quarter plate, central hole, uniaxial tension, plane strain.

    TARGET: Kt = 3.000 (Kirsch, infinite plate) on the GROSS section. The plate
    is sized so the finite-width correction is negligible rather than quoting a
    correction coefficient from memory: with width_ratio = W/a = 50 the hole
    diameter to plate width is d/W = 2a/W = 0.04, for which the finite-width
    effect on Kt_gross is well under 1%.

    Extraction is GEOMETRY-ANCHORED: peak is read on nodes lying on the hole
    boundary (|r-a| < 1e-6 a), and reported two ways — the raw nodal max and
    the maximum of a quadratic fit of sigma_xx against theta near the analytic
    peak location theta = 90 deg. sigma_nom is the applied traction, asserted
    against a far-field reading. Kt < 1 raises."""
    mat = material or Material("val", 7800.0, 2.1e11, 2.5e8, poisson=0.3)
    a, W, sig = float(hole_r_m), float(hole_r_m) * float(width_ratio), 1.0e6
    fine = a * (np.pi / 2) / float(elem_across)
    nodes, elems = _mesh_quarter_hole(a, W, fine, 0.5 * a, zone=1.0 * a)
    tol = 1e-9 * a
    fixed = {}
    for n in np.flatnonzero(np.abs(nodes[:, 0]) < tol):
        fixed[n] = (1,)
    for n in np.flatnonzero(np.abs(nodes[:, 1]) < tol):
        fixed[n] = fixed.get(n, ()) + (2,)
    f = edge_consistent_load(nodes, elems, np.abs(nodes[:, 0] - W) < tol,
                             lambda x, y: (sig, 0.0))
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    inp = write_plane_strain_inp(Path(work_dir) / f"{tag}.inp", nodes, elems,
                                 mat, fixed, f,
                                 elem_type=CCX_PLANE_STRAIN_TRI)
    run_ccx(inp)
    verify_node_identity(inp, nodes)
    sn = nodal_stress_array(inp, len(nodes))

    rad = np.hypot(nodes[:, 0], nodes[:, 1])
    on_hole = np.abs(rad - a) < 1e-6 * a
    if on_hole.sum() < 5:
        raise SolverError(f"only {on_hole.sum()} nodes on the hole boundary")
    theta = np.arctan2(nodes[on_hole, 1], nodes[on_hole, 0])
    sxx = sn[on_hole, 0]
    nodal_peak = float(sxx.max())
    near = np.abs(theta - np.pi / 2) < np.deg2rad(35.0)
    if near.sum() >= 3:
        c = np.polyfit(theta[near], sxx[near], 2)
        fitted_peak = float(np.polyval(c, -c[1] / (2 * c[0]))) if c[0] < 0 else nodal_peak
    else:
        fitted_peak = nodal_peak
    far = (np.abs(nodes[:, 0] - 0.9 * W) < 0.05 * W) & (nodes[:, 1] > 0.5 * W)
    nom_far = float(np.median(sn[far, 0])) if np.any(far) else sig
    kt = fitted_peak / sig
    if kt < 1.0:
        raise SolverError(f"Kt = {kt:.4f} < 1 — extraction failure, not a result")
    return {"kt": kt, "kt_nodal": nodal_peak / sig, "elements": len(elems),
            "nodes": len(nodes), "arc_nodes": int(on_hole.sum()),
            "nom_far_err": nom_far / sig - 1.0, "d_over_W": 2 * a / W}


def _mesh_wall(t, r, W, T, H, fine, coarse, zone):
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        with contextlib.redirect_stdout(io.StringIO()):
            o = gmsh.model.occ; h = 0.5 * t
            P = [o.addPoint(-0.5*W, -T, 0), o.addPoint(0.5*W, -T, 0),
                 o.addPoint(0.5*W, 0, 0), o.addPoint(h + r, 0, 0),
                 o.addPoint(h, r, 0), o.addPoint(h, H, 0),
                 o.addPoint(-h, H, 0), o.addPoint(-h, r, 0),
                 o.addPoint(-h - r, 0, 0), o.addPoint(-0.5*W, 0, 0)]
            cR = o.addPoint(h + r, r, 0); cL = o.addPoint(-h - r, r, 0)
            cur = [o.addLine(P[0], P[1]), o.addLine(P[1], P[2]), o.addLine(P[2], P[3]),
                   o.addCircleArc(P[3], cR, P[4]), o.addLine(P[4], P[5]),
                   o.addLine(P[5], P[6]), o.addLine(P[6], P[7]),
                   o.addCircleArc(P[7], cL, P[8]), o.addLine(P[8], P[9]),
                   o.addLine(P[9], P[0])]
            o.addPlaneSurface([o.addCurveLoop(cur)]); o.synchronize()
            f = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(f, "CurvesList", [cur[3], cur[7]])
            gmsh.model.mesh.field.setNumber(f, "Sampling", 800)
            th = gmsh.model.mesh.field.add("Threshold")
            for k, v in (("InField", f), ("SizeMin", fine), ("SizeMax", coarse),
                         ("DistMin", zone), ("DistMax", 6.0 * zone)):
                gmsh.model.mesh.field.setNumber(th, k, v)
            gmsh.model.mesh.field.setAsBackgroundMesh(th)
            for op in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints",
                       "Mesh.MeshSizeFromCurvature"):
                gmsh.option.setNumber(op, 0)
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
            gmsh.option.setNumber("Mesh.MeshSizeMax", coarse)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
            gmsh.model.mesh.generate(2); gmsh.model.mesh.setOrder(2)
            tg, cd, _ = gmsh.model.mesh.getNodes()
            et, _, en = gmsh.model.mesh.getElements(2)
            conn = None
            for e, n in zip(et, en):
                if e == GMSH_TRI6:
                    conn = np.asarray(n, dtype=np.int64).reshape(-1, 6)
            if conn is None:
                raise SolverError("no CPE6 triangles produced")
        nodes = np.asarray(cd).reshape(-1, 3); tg = np.asarray(tg, dtype=np.int64)
        perm = np.argsort(tg); tg, nodes = tg[perm], nodes[perm]
        elems = np.searchsorted(tg, conn)
        used = np.zeros(len(nodes), bool); used[elems.ravel()] = True
        remap = -np.ones(len(nodes), np.int64); remap[used] = np.arange(used.sum())
        nodes, elems = nodes[used], remap[elems]
    finally:
        gmsh.finalize()
    return nodes[:, :2], elems


def wall_on_plate_kt(t_m, r_m, *, load="bending", width_ratio=8.0,
                     plate_depth_ratio=4.0, wall_height_ratio=6.0,
                     elem_across=16.0, material=None,
                     work_dir="runs/kt2d", tag="wall"):
    """Wall of thickness t on a plate, fillet radius r, plane strain.

    This IS the stepped-flat-bar geometry (width_ratio = D/d). `load="bending"`
    applies a PURE MOMENT (linear traction, zero net force and zero shear) —
    the case matching `beam.py`'s `sigma_bend`. `load="axial"` applies uniform
    tension, matching `beam.py`'s centrifugal `N/A`. Both curves are shipped in
    `propopt.kt_root`."""
    mat = material or Material("pla", 1240.0, 3.5e9, 50.0e6, poisson=0.35)
    t, r = float(t_m), float(r_m)
    if not 0.0 < r <= 0.5 * t:
        raise SolverError(f"fillet r={r} invalid for wall thickness {t}")
    W, T, H = width_ratio * t, plate_depth_ratio * t, wall_height_ratio * t
    fine = r * (np.pi / 2) / float(elem_across)
    nodes, elems = _mesh_wall(t, r, W, T, H, fine, t / 6.0, zone=r)
    tol = 1e-9 * t
    fixed = {n: (1, 2) for n in np.flatnonzero(np.abs(nodes[:, 1] + T) < tol)}
    I = t ** 3 / 12.0
    sig_nom_analytic = 1.0e6
    if load == "bending":
        M = sig_nom_analytic * I / (0.5 * t)
        trac = lambda x, y: (0.0, M * x / I)
    else:
        trac = lambda x, y: (0.0, sig_nom_analytic)
    f = edge_consistent_load(nodes, elems, np.abs(nodes[:, 1] - H) < tol, trac)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    inp = write_plane_strain_inp(Path(work_dir) / f"{tag}.inp", nodes, elems,
                                 mat, fixed, f, elem_type=CCX_PLANE_STRAIN_TRI)
    run_ccx(inp); verify_node_identity(inp, nodes)
    sn = nodal_stress_array(inp, len(nodes))
    sxx, syy, sxy = sn[:, 0], sn[:, 1], sn[:, 3]
    prin = 0.5*(sxx+syy) + np.sqrt((0.5*(sxx-syy))**2 + sxy**2)
    on_face = (np.abs(nodes[:, 0] - 0.5*t) < 1e-7*t) & \
              (np.abs(nodes[:, 1] - 0.6*H) < 0.2*H)
    if not np.any(on_face):
        raise SolverError("no nodes in the nominal band")
    sig_nom = float(np.median(syy[on_face]))
    dR = np.hypot(nodes[:, 0] - (0.5*t + r), nodes[:, 1] - r)
    on_arc = (np.abs(dR - r) < 1e-6*r) & (nodes[:, 0] > 0)
    if on_arc.sum() < 4:
        raise SolverError(f"only {on_arc.sum()} nodes on the fillet arc")
    peak = float(prin[on_arc].max())
    kt = peak / sig_nom
    if kt < 1.0:
        raise SolverError(f"Kt = {kt:.4f} < 1 - extraction failure, not a result")
    return {"kt": kt, "sigma_nom": sig_nom, "nom_err": sig_nom/sig_nom_analytic - 1.0,
            "elements": len(elems), "arc_nodes": int(on_arc.sum()), "r_over_t": r/t}
