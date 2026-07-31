"""Phase-4a slice 3a: blade + hub boss + root fillet, in OCC. HF path only.

WHY: the beam model's root guard was a documented placeholder (kt_root = 2.5).
Calibrating it needs a root whose stress concentration is GEOMETRIC — produced
by a finite-radius fillet — rather than the artifact of a clamped flat face
that the slice-1/2 models produce. That requires a hub body and a real B-rep
fillet. (Slice 3b then measured the curves in 2-D; they ship in
`propopt.kt_root` and the beam reads them.)

NOTHING HERE IS HARDCODED TO THE CURRENT te_thickness_m OR TO ANY DESIGN.
Every length is derived from the design's own geometry:
  * hub radius        <- the root section's own reach from the rotation axis
  * hub length        <- the root chord (Saint-Venant)
  * penetration depth <- the root section's own max thickness
  * fillet bound      <- the section's own thickness distribution
  * mesh sizes        <- the fillet radius itself
The only bare numbers are dimensionless shape factors, each a named parameter
carrying its justification.

FRAME (from `geometry`): x tangential (rotor plane), y AXIAL = the rotation
axis, z radial = span. So the hub boss is a cylinder about the **y** axis and
the blade grows along +z. Distance from the rotation axis is hypot(x, z).

HUB RADIUS IS NOT `hub_radius_m`. `PropellerDesign.hub_radius_m` is the ROOT
CUTOUT radius — where the aerodynamic blade starts — not the radius of a boss
that could physically hold it. The root section is a chordwise plank ~1 chord
long lying tangent to that radius, so a cylinder of radius `hub_radius_m`
contains none of it (measured: 0% of root points, minimum containing radius
2.1x larger). Fusing them would give a tangent line-contact, not a junction,
and a fillet needs a CLOSED junction loop. The hub radius is therefore derived
to contain the root section with clearance, and `hub_radius_m` is used only as
the radius at which the blade's aerodynamic definition begins.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np

from ..contracts import PropellerDesign, SolverError
from ..geometry import (DEFAULT_N_CHORD, DEFAULT_SECTION, SectionFn,
                        extended_stations, section_thickness)
from .mesh import TetMesh, _extract, tet_volumes
from .occ_geometry import _build_occ_solid, station_rings

# --- dimensionless shape factors (the only bare numbers; each justified) ---

# Hub radius = HUB_CLEARANCE_FACTOR x (root section's max reach from the axis).
# > 1 so the root cap lies STRICTLY inside the boss: the fuse must be a
# transversal volume crossing, never a near-tangency, which is the fragile case
# for a boolean.
HUB_CLEARANCE_FACTOR = 1.15

# Hub length = HUB_LENGTH_FACTOR x root chord, centred on the blade. The clamp
# in 3b sits on the end faces, i.e. HUB_LENGTH_FACTOR/2 chords from the
# junction. Saint-Venant: a self-equilibrated end disturbance decays over ~1
# characteristic cross-sectional dimension, and the junction's characteristic
# dimension is the root chord, so 1 chord of standoff leaves a few-percent
# perturbation at the fillet.
HUB_LENGTH_FACTOR = 2.0

# Blade root is extended inward by PENETRATION_FACTOR x root max thickness
# before the fuse. One local thickness is the natural scale on which two
# surfaces stop being "nearly tangent"; it guarantees the extended cap is well
# inside the boss so the boolean sees a clean crossing.
PENETRATION_FACTOR = 1.0

# Elements across the fillet arc. 4-8 is the standard band for converged
# fillet-peak stress; 5 gives ~18 deg per element, enough to resolve the
# gradient without exploding the count.
N_ELEM_ACROSS_DEFAULT = 5


# Plate hub: lateral extent and thickness as multiples of the root chord. The
# extent must clear the blade footprint plus several fillet radii so the fillet
# never runs off the edge; the thickness sets the clamp standoff (Saint-Venant,
# same argument as the cylinder length).
PLATE_EXTENT_FACTOR = 2.0
PLATE_THICKNESS_FACTOR = 1.0


@dataclass(frozen=True)
class HubSpec:
    """Derived hub geometry (SI). Every field computed from the design.

    `kind` selects the hub body:
      "plate"    — a flat boss face perpendicular to the span, with the blade
                   emerging at z = hub_radius_m. THE ONLY KIND OCC WILL FILLET
                   against a lofted blade (see module docstring / slice-3a
                   factorial). Default.
      "cylinder" — a boss about the rotation axis. Geometrically the more
                   faithful hub, but OCC cannot blend a lofted blade against
                   it; retained so the finding stays reproducible.
    """
    kind: str
    penetration_m: float     # how far the blade root is pushed inside the hub
    root_chord_m: float
    root_thickness_m: float  # max thickness of the ROOT section (computed)
    root_reach_m: float      # root section's max distance from the rotation axis
    # cylinder form
    radius_m: float = 0.0    # boss radius about the rotation (y) axis
    length_m: float = 0.0    # axial extent, centred on y = 0
    # plate form
    top_z_m: float = 0.0     # the boss face the blade emerges from
    thickness_m: float = 0.0
    extent_m: float = 0.0
    centre_xy: tuple = (0.0, 0.0)

    @property
    def clamp_standoff_chords(self) -> float:
        """Distance from junction to the clamp face, in root chords."""
        d = 0.5 * self.length_m if self.kind == "cylinder" else self.thickness_m
        return d / self.root_chord_m


def root_thickness_distribution(design: PropellerDesign,
                                section: SectionFn = DEFAULT_SECTION,
                                n_chord: int = DEFAULT_N_CHORD,
                                te_thickness_m: float = 0.0) -> np.ndarray:
    """Local blade thickness (m) at each chordwise sample of the ROOT section.

    Read off the same polygon the loft and the OCC solid use, so it tracks
    te_thickness_m, chord and the section function automatically. The
    normalized read is `geometry.section_thickness` — shared with the beam's
    root r/t so the two cannot drift apart."""
    _, c, _ = extended_stations(design)
    c_root = float(c[0])
    return section_thickness(section, n_chord, te_thickness_m / c_root) * c_root


def analytic_fillet_bound(design: PropellerDesign,
                          section: SectionFn = DEFAULT_SECTION,
                          n_chord: int = DEFAULT_N_CHORD,
                          te_thickness_m: float = 0.0) -> dict:
    """r_max_geom = min(local thickness)/2 along the junction.

    Two fillets (one per blade face) collide when 2r exceeds the local blade
    thickness, so half the MINIMUM thickness bounds a constant-radius fillet.

    The exact leading-edge vertex is excluded: there the upper and lower
    surfaces MEET, so the thickness is identically zero and the two-sided
    collision argument does not apply — the surface is convex there and the
    limiter is the nose radius, reported separately. The minimum is otherwise
    found from the data, NOT assumed to be the trailing edge."""
    dist = root_thickness_distribution(design, section, n_chord, te_thickness_m)
    pos, thick = dist[:, 0], dist[:, 1]
    interior = thick > 0.0
    if not np.any(interior):
        raise SolverError("root section has no positive thickness anywhere")
    i_min = int(np.argmin(np.where(interior, thick, np.inf)))
    t_min = float(thick[i_min])
    c_root = float(pos[-1])
    # NACA-style nose radius ~ 1.1 (t/c)^2 c, the local scale at the LE where
    # the collision bound does not apply.
    t_max = float(thick.max())
    nose_radius = 1.1 * (t_max / c_root) ** 2 * c_root if c_root > 0 else 0.0
    return {
        "r_max_geom_m": 0.5 * t_min,
        "t_min_m": t_min,
        "t_min_at_chord_frac": float(pos[i_min] / c_root) if c_root else 0.0,
        "t_min_is_trailing_edge": bool(i_min == len(thick) - 1),
        "t_max_m": t_max,
        "le_nose_radius_m": nose_radius,
        "chord_m": c_root,
    }


def derive_hub(design: PropellerDesign,
               section: SectionFn = DEFAULT_SECTION,
               n_chord: int = DEFAULT_N_CHORD,
               te_thickness_m: float = 0.0,
               clearance_factor: float = HUB_CLEARANCE_FACTOR,
               length_factor: float = HUB_LENGTH_FACTOR,
               penetration_factor: float = PENETRATION_FACTOR,
               kind: str = "plate",
               extent_factor: float = PLATE_EXTENT_FACTOR,
               thickness_factor: float = PLATE_THICKNESS_FACTOR) -> HubSpec:
    """Size the hub from the design's own root geometry (see module doc)."""
    rings = station_rings(design, section, n_chord, te_thickness_m)
    root = rings[0]
    reach = float(np.hypot(root[:, 0], root[:, 2]).max())
    _, c, _ = extended_stations(design)
    c_root = float(c[0])
    t_root = float(root_thickness_distribution(
        design, section, n_chord, te_thickness_m)[:, 1].max())
    depth = penetration_factor * t_root
    z_root = float(root[0, 2])
    if depth >= z_root:
        # Pushing the cap past the rotation axis would be nonsense; cap it and
        # keep the derivation honest rather than silently producing a solid
        # that straddles the axis.
        depth = 0.9 * z_root
    if kind == "cylinder":
        return HubSpec("cylinder", depth, c_root, t_root, reach,
                       radius_m=clearance_factor * reach,
                       length_m=length_factor * c_root)
    if kind != "plate":
        raise SolverError(f"hub kind must be 'plate' or 'cylinder', got {kind!r}")
    # Plate: the boss face sits at the design's own root-cutout radius, so the
    # EXPOSED blade spans exactly [hub_radius_m, radius_m] — the same interval
    # the loft and the aero quadrature use — and the blade is pushed `depth`
    # below that face so the boolean is a transversal crossing.
    return HubSpec("plate", depth, c_root, t_root, reach,
                   top_z_m=z_root,
                   thickness_m=thickness_factor * c_root,
                   extent_m=extent_factor * c_root,
                   centre_xy=(float(root[:, 0].mean()), float(root[:, 1].mean())))


def _extended_rings(design, section, n_chord, te_thickness_m, hub: HubSpec):
    """Station rings with the root pushed `hub.penetration_m` further inboard,
    holding chord and twist (the same end-holding the loft uses)."""
    rings = station_rings(design, section, n_chord, te_thickness_m)
    inner = rings[0].copy()
    inner[:, 2] = rings[0, 0, 2] - hub.penetration_m
    if hub.kind == "cylinder":
        if float(np.hypot(inner[:, 0], inner[:, 2]).max()) >= hub.radius_m:
            raise SolverError(
                "extended blade root does not fit inside the hub boss; increase "
                "clearance_factor or reduce penetration_factor")
    elif hub.penetration_m >= hub.thickness_m:
        raise SolverError("penetration exceeds plate thickness")
    return np.concatenate([inner[None, :, :], rings], axis=0)


def _junction_curves(gmsh, hub: HubSpec, rel_tol: float = 1e-4) -> list[int]:
    """Curve tags lying ON the hub surface — the blade/hub junction loop.

    A curve qualifies when every sampled point sits on that surface: the
    cylinder of radius hub.radius_m about the y axis, or the plate's top plane
    z = hub.top_z_m. The hub's own boundary edges are excluded (the cylinder's
    end circles; the plate's outer rim)."""
    scale = hub.radius_m if hub.kind == "cylinder" else hub.root_chord_m
    tol = rel_tol * scale
    out = []
    for dim, tag in gmsh.model.getEntities(1):
        try:
            lo, hi = gmsh.model.getParametrizationBounds(dim, tag)
        except Exception:
            continue
        ts = np.linspace(float(lo[0]), float(hi[0]), 7)
        pts = np.array(gmsh.model.getValue(dim, tag, list(ts))).reshape(-1, 3)
        if hub.kind == "cylinder":
            d = np.hypot(pts[:, 0], pts[:, 2])
            on_hub = np.all(np.abs(d - hub.radius_m) < tol)
            on_rim = np.all(np.abs(np.abs(pts[:, 1]) - 0.5 * hub.length_m) < tol)
        else:
            on_hub = np.all(np.abs(pts[:, 2] - hub.top_z_m) < tol)
            half = 0.5 * hub.extent_m
            on_rim = (np.all(np.abs(np.abs(pts[:, 0] - hub.centre_xy[0]) - half) < tol)
                      or np.all(np.abs(np.abs(pts[:, 1] - hub.centre_xy[1]) - half) < tol))
        if on_hub and not on_rim:
            out.append(tag)
    return out


def build_filleted(gmsh, design: PropellerDesign, hub: HubSpec,
                   fillet_radius_m: float,
                   section: SectionFn = DEFAULT_SECTION,
                   n_chord: int = DEFAULT_N_CHORD,
                   te_thickness_m: float = 0.0) -> dict:
    """Blade + boss, fused, with the junction filleted. Caller owns the gmsh
    session. Returns diagnostics; raises SolverError on any OCC failure."""
    rings = _extended_rings(design, section, n_chord, te_thickness_m, hub)
    blade = _build_occ_solid(gmsh, rings)
    try:
        if hub.kind == "cylinder":
            boss = gmsh.model.occ.addCylinder(
                0.0, -0.5 * hub.length_m, 0.0, 0.0, hub.length_m, 0.0,
                hub.radius_m)
        else:
            half = 0.5 * hub.extent_m
            boss = gmsh.model.occ.addBox(
                hub.centre_xy[0] - half, hub.centre_xy[1] - half,
                hub.top_z_m - hub.thickness_m,
                hub.extent_m, hub.extent_m, hub.thickness_m)
        fused, _ = gmsh.model.occ.fuse([(3, blade)], [(3, boss)])
        gmsh.model.occ.synchronize()
    except Exception as exc:
        raise SolverError(f"blade/hub fuse failed: {exc}") from exc
    vols = [t for d, t in fused if d == 3]
    if len(vols) != 1:
        raise SolverError(f"fuse produced {len(vols)} volumes (expected 1)")
    v_fused = float(gmsh.model.occ.getMass(3, vols[0]))

    curves = _junction_curves(gmsh, hub)
    if not curves:
        raise SolverError("no blade/hub junction curves found on the boss surface")
    if fillet_radius_m <= 0.0:
        # no fillet applied: getMass IS trustworthy here (verified against a
        # converged tet mesh to ~3e-6 relative)
        return {"occ_volume_m3_unreliable": v_fused, "fused_volume_m3": v_fused,
                "n_junction_curves": len(curves), "volume_tag": vols[0],
                "new_surfaces": [], "radius_m": 0.0}

    before = {t for _, t in gmsh.model.getEntities(2)}
    try:
        out = gmsh.model.occ.fillet(vols, curves, [float(fillet_radius_m)],
                                    removeVolume=True)
        gmsh.model.occ.synchronize()
    except Exception as exc:
        raise SolverError(f"fillet r={fillet_radius_m:.6e} m failed: {exc}") from exc
    fvols = [t for d, t in out if d == 3]
    if len(fvols) != 1:
        raise SolverError(
            f"fillet r={fillet_radius_m:.6e} m produced {len(fvols)} volumes")
    v_fillet = float(gmsh.model.occ.getMass(3, fvols[0]))
    if not np.isfinite(v_fillet) or v_fillet <= 0.0:
        raise SolverError(f"filleted solid has non-physical volume {v_fillet!r}")
    return {
        # DO NOT USE THIS FOR VOLUME. `occ.getMass` is exact on the FUSED solid
        # (matches a converged tet mesh to ~3e-6 relative) but returns a wrong
        # value once the junction has been filleted. Proof it is an artifact
        # and not real material: the implied dV depends on the PLATE SIZE
        # (+31.26 mm^3 on a 2.0x1.0-chord plate vs +12.67 mm^3 on a 1.0x0.4
        # plate, same blade, same junction, same radius) — a local fillet
        # cannot do that. The wrong value also survives a BREP write and
        # re-read in a fresh session, so it is baked into the shape rather
        # than stale session state. A converged tet mesh of the SAME shape
        # gives +0.93 mm^3, i.e. 1.8x the ideal rolling-ball value, which is
        # physically sensible for a non-90-degree corner.
        # Use `mesh_filleted(...).volume_m3` when a volume is needed.
        "occ_volume_m3_unreliable": v_fillet,
        "fused_volume_m3": v_fused,
        "n_junction_curves": len(curves), "volume_tag": fvols[0],
        "new_surfaces": [t for _, t in gmsh.model.getEntities(2)
                         if t not in before],
        "radius_m": float(fillet_radius_m)}


def _session(fn):
    """Run `fn(gmsh)` inside a quiet gmsh session."""
    try:
        import gmsh
    except ImportError as exc:
        raise SolverError("gmsh is not installed (pip install -e '.[hf]')") from exc
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(gmsh)
    finally:
        gmsh.finalize()


def try_fillet(design: PropellerDesign, hub: HubSpec, radius_m: float,
               section: SectionFn = DEFAULT_SECTION,
               n_chord: int = DEFAULT_N_CHORD,
               te_thickness_m: float = 0.0) -> dict:
    """Attempt one radius; returns the diagnostics dict or raises SolverError."""
    return _session(lambda g: build_filleted(g, design, hub, radius_m, section,
                                             n_chord, te_thickness_m))


def max_fillet_radius(design: PropellerDesign, hub: HubSpec,
                      upper_m: float,
                      section: SectionFn = DEFAULT_SECTION,
                      n_chord: int = DEFAULT_N_CHORD,
                      te_thickness_m: float = 0.0,
                      rel_tol: float = 0.02,
                      max_iter: int = 24) -> dict:
    """Bisect for the largest radius OCC will actually build.

    `upper_m` is the analytic bound; the search brackets [0, upper] and
    converges to `rel_tol` of it. Returns the largest success and the smallest
    failure, with the failure message."""
    lo, hi = 0.0, float(upper_m)
    best, fail_msg, first_fail = None, None, None
    try:
        best = try_fillet(design, hub, hi, section, n_chord, te_thickness_m)
        return {"r_ok_m": hi, "r_fail_m": None, "message": None,
                "bracketed_at_analytic_bound": True, "result": best}
    except SolverError as exc:
        fail_msg, first_fail = str(exc), hi
    for _ in range(max_iter):
        if hi - lo <= rel_tol * upper_m:
            break
        mid = 0.5 * (lo + hi)
        try:
            best = try_fillet(design, hub, mid, section, n_chord, te_thickness_m)
            lo = mid
        except SolverError as exc:
            hi, fail_msg, first_fail = mid, str(exc), mid
    return {"r_ok_m": lo, "r_fail_m": first_fail, "message": fail_msg,
            "bracketed_at_analytic_bound": False, "result": best}


def mesh_filleted(design: PropellerDesign, hub: HubSpec, radius_m: float,
                  *, coarse_size_m: float,
                  n_elem_across: int = N_ELEM_ACROSS_DEFAULT,
                  section: SectionFn = DEFAULT_SECTION,
                  n_chord: int = DEFAULT_N_CHORD,
                  te_thickness_m: float = 0.0,
                  order: int = 2, optimize: bool = True) -> TetMesh:
    """Mesh the filleted solid with local refinement ON the fillet faces.

    Fillet-face size is `radius_m / n_elem_across` — derived from the radius
    itself, so it tracks automatically — growing to `coarse_size_m` away from
    the junction via a Distance+Threshold field."""
    fine = float(radius_m) / int(n_elem_across)

    def _run(gmsh):
        info = build_filleted(gmsh, design, hub, radius_m, section, n_chord,
                              te_thickness_m)
        faces = info["new_surfaces"]
        if faces:
            f_dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(f_dist, "SurfacesList", faces)
            gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)
            f_thr = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
            gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", fine)
            gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", coarse_size_m)
            # grow from the fillet over a few radii: close enough to keep the
            # gradient resolved, far enough to avoid a cliff in element size
            gmsh.model.mesh.field.setNumber(f_thr, "DistMin", radius_m)
            gmsh.model.mesh.field.setNumber(f_thr, "DistMax", 10.0 * radius_m)
            gmsh.model.mesh.field.setAsBackgroundMesh(f_thr)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", coarse_size_m)
        gmsh.option.setNumber("Mesh.MeshSizeMin", fine)
        gmsh.option.setNumber("Mesh.Optimize", 1 if optimize else 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1 if optimize else 0)
        try:
            gmsh.model.mesh.generate(3)
            if order == 2:
                gmsh.model.mesh.setOrder(2)
        except Exception as exc:
            raise SolverError(f"filleted-solid meshing failed: {exc}") from exc
        return _extract(gmsh, order, "filleted"), info

    (nodes, tets, n_tris, qual), info = _session(_run)
    vols = tet_volumes(nodes, tets)
    if np.any(vols <= 0.0):
        raise SolverError(
            f"{int(np.sum(vols <= 0.0))} non-positive-volume tets in the mesh")
    return TetMesh(nodes=nodes, tets=tets, volume_m3=float(np.sum(vols)),
                   n_boundary_tris=n_tris, quality_min=float(qual.min()),
                   quality_mean=float(qual.mean()),
                   quality_p01=float(np.percentile(qual, 1.0)), quality=qual,
                   remeshed=False, order=int(order),
                   reference_volume_m3=info["fused_volume_m3"])
