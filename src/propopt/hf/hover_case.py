"""Phase-4b slice 1: a static-hover MRF RANS case for one propeller design.

VALIDATOR ONLY. Not an AeroSolver, not imported by optimize.py.

FRAME (inherited from `geometry`, not reinvented): x tangential (rotor plane),
y AXIAL = the rotation axis, z radial = span. So the rotation axis is **y**,
thrust acts along y, and the blade grows along +z.

SECTOR. A B-bladed rotor in hover is periodic under rotation by 360/B degrees,
so only one passage is meshed. `sector_deg` and `periodic` are CONFIG, not
assumptions baked into the mesh writer: Phase 5 will want a full 360-degree
domain (unequal blades, ground effect, inflow at incidence) and that must be a
config change, not a rewrite.

For B = 2 the sector is 180 degrees, which is the half-space z > 0. Its two
bounding half-planes ({z=0, x>0} and {z=0, x<0}) are COPLANAR — a 180-degree
rotational cyclic is the degenerate case where the periodic pair lies in one
plane. It is still a valid pair: rotating {z=0, x<0} by 180 degrees about y
lands on {z=0, x>0} with reversed normal, which is exactly the cyclic
condition. The background mesh is therefore two blocks split at x = 0, because
blockMesh cannot split one face of one block into two patches.

STATIC HOVER FAR FIELD — the classic difficulty. With V_inf = 0 there is no
freestream to prescribe. The rotor ENTRAINS fluid radially and expels a
slipstream axially, so the outer boundary must let flow cross it in BOTH
directions, chosen locally by the solution:

  * a fixedValue inlet would IMPOSE a flow that does not exist and fix the
    induced velocity by fiat — it would decide the answer;
  * a slip or symmetry wall would CONFINE the induced flow, turning the domain
    into a shrouded test rig (a wind-tunnel-wall / blockage error that inflates
    thrust);
  * so the outer boundary is a constant-TOTAL-pressure opening:
    `pressureInletOutletVelocity` on U with `totalPressure` (p0 = 0) on p.
    Where fluid enters, static pressure is reduced by the entering dynamic
    head (the correct Bernoulli relation for entrainment from a quiescent
    reservoir); where it leaves, it becomes zeroGradient. This is the standard
    open boundary for isolated hovering rotors.

Its correctness depends on the boundary sitting in near-quiescent flow, i.e.
on DOMAIN SIZE, which is a convergence parameter exactly like mesh density.
This slice does NOT establish domain independence — that belongs with the
mesh-independence study.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..contracts import PropellerDesign, SolverError
from ..geometry import DEFAULT_N_CHORD, DEFAULT_SECTION, SectionFn, extended_stations
from .occ_geometry import _build_occ_solid, station_rings
from .openfoam import FoamCase, write_dict

# Sea-level ISA air, the same properties the BEMT solver uses, so a CT/CP
# comparison is not confounded by a different atmosphere.
RHO_AIR = 1.225          # kg/m^3
_FF_U = {"totalPressure": "pressureInletOutletVelocity; value uniform (0 0 0)",
         "inletOutlet": "inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0)"}
_FF_P = {"totalPressure": "totalPressure; p0 uniform 0; value uniform 0",
         "inletOutlet": "fixedValue; value uniform 0"}
NU_AIR = 1.4607e-5       # m^2/s (kinematic)


@dataclass(frozen=True)
class HoverCaseConfig:
    """Everything that defines the CFD case. No value below is hardcoded in the
    writer; all of it is echoed into the case for provenance."""
    rpm: float
    # --- domain (multiples of tip radius) ---
    far_field_radii: float = 5.0      # box half-extent / R
    # --- sector ---
    periodic: bool = True             # False -> full 360-degree domain
    sector_deg: float | None = None   # None -> 360 / n_blades
    # --- background mesh ---
    background_cells_per_span: int = 24   # cells per tip radius (whole block/2)
    core_radii: float = 1.3          # uniform fine core half-extent / R
    core_cell_frac: float = 0.55     # fraction of background cells in the core
    # --- snappyHexMesh refinement ---
    surface_level: tuple[int, int] = (5, 6)
    rotor_region_level: int = 3       # refinement box around the rotor disk
    mrf_zone_radii: float = 1.3       # MRF cellZone radius / R
    mrf_zone_half_height_radii: float = 0.4   # MRF cellZone half-height / R
    n_layers: int = 3
    layer_expansion: float = 1.25
    first_layer_rel: float = 0.35     # final layer thickness / local cell size
    # ABSOLUTE first-layer thickness (m). When set, layer sizing switches to
    # absolute mode, which is the only way to aim at a y+ target: relative
    # sizing ties the layer to the local octree cell, so the y+ it produces
    # moves with the refinement level instead of staying where you put it.
    first_layer_m: float | None = None
    # Wall treatment for nut. nutkWallFunction assumes the first cell sits in
    # the LOG layer. nutUSpaldingWallFunction is a continuous blend valid from
    # the viscous sublayer through the log layer, which is what a y+ ~ 1 mesh
    # needs (and what a mesh straddling the buffer layer needs too).
    nut_wall_function: str = "nutkWallFunction"
    # --- physics ---
    rho: float = RHO_AIR
    nu: float = NU_AIR
    turbulence_model: str = "kOmegaSST"
    # Far-field opening type. "totalPressure": constant-total-pressure opening
    # (p0 = 0) with pressureInletOutletVelocity - the standard hover opening.
    # "inletOutlet": fixed STATIC pressure with a switching velocity BC. The
    # two differ in how much entrainment they permit, which is exactly the
    # suspect when a wake fails to exit.
    far_field_bc: str = "totalPressure"
    # --- geometry ---
    te_thickness_m: float = 2.0e-4
    n_chord: int = DEFAULT_N_CHORD
    stl_size_m: float = 4.0e-4        # STL facet size (finer than any CFD cell)
    # --- run control ---
    n_iterations: int = 3000
    n_proc: int = 8
    write_interval: int = 500

    def sector(self, n_blades: int) -> float:
        if not self.periodic:
            return 360.0
        return float(self.sector_deg if self.sector_deg is not None
                     else 360.0 / n_blades)

    def omega_rad_s(self) -> float:
        """Unsigned shaft speed."""
        return float(self.rpm) * 2.0 * np.pi / 60.0

    def omega_signed(self) -> float:
        """SIGNED rotation rate about +y for MRF.

        NEGATIVE, and that is not arbitrary. `geometry.station_rings` places the
        chord with the leading edge at xi = -0.25c and the trailing edge at
        +0.75c, i.e. the LE sits at NEGATIVE x. A blade must advance
        leading-edge-first, so its velocity must point in -x, and
        (omega_y * y_hat) x r has x-component omega_y * z, which is negative for
        a blade at +z only when omega_y < 0.

        A positive omega here runs the propeller BACKWARDS: it still produces
        thrust toward the suction side, so the thrust SIGN looks correct, but
        lift collapses while drag does not (measured: CT -44%, CP -13% against
        UIUC). Nothing else in propopt carries a rotation sense - BEMT uses
        |Omega r| and the structural path uses centrifugal magnitude - so this
        is the only place the sense exists and the only place it can be wrong.
        `leading_edge_first_cos` is the cross-check; `build_hover_case` calls it.
        """
        return -self.omega_rad_s()


def write_blade_stl(design: PropellerDesign, path: str | Path, cfg: HoverCaseConfig,
                    section: SectionFn = DEFAULT_SECTION, name: str = "blade") -> dict:
    """Triangulated blade SURFACE for snappyHexMesh, in METRES.

    Built from the SAME OCC solid the structural HF path uses
    (`occ_geometry.station_rings` + `_build_occ_solid`), so the CFD blade is the
    same design as the BEMT blade by construction. ASCII STL with a named solid
    so snappy can address it as a region."""
    try:
        import gmsh
    except ImportError as exc:
        raise SolverError("gmsh is not installed (pip install -e '.[hf]')") from exc

    rings = station_rings(design, section, cfg.n_chord, cfg.te_thickness_m)
    # One blade per passage in a periodic sector; ALL of them in a full domain.
    n_copies = 1 if cfg.periodic else int(design.n_blades)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(cfg.stl_size_m))
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(cfg.stl_size_m) / 4.0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.option.setNumber("Mesh.Binary", 0)
        # one solid per PHYSICAL GROUP -> the STL carries named regions, so
        # snappy makes a separate patch for the root cap and its force can be
        # integrated on its own (it has no BEMT analogue).
        gmsh.option.setNumber("Mesh.StlOneSolidPerSurface", 2)
        with contextlib.redirect_stdout(io.StringIO()):
            volume = 0.0
            vol_tags: list[int] = []
            for k in range(n_copies):
                th = 2.0 * np.pi * k / max(n_copies, 1)
                r = rings.copy()
                # rotate about the y axis (the rotation axis in this frame)
                r[:, :, 0] = rings[:, :, 0] * np.cos(th) + rings[:, :, 2] * np.sin(th)
                r[:, :, 2] = -rings[:, :, 0] * np.sin(th) + rings[:, :, 2] * np.cos(th)
                tag = _build_occ_solid(gmsh, r)
                volume += float(gmsh.model.occ.getMass(3, tag))
                vol_tags.append(tag)
            # Classify each solid's boundary faces by distance from the
            # rotation axis: the ROOT cap is the nearest planar face, the TIP
            # cap the farthest. Found from the geometry, not assumed by index.
            root_s, skin_s = [], []
            for vt in vol_tags:
                faces = [s for d, s in gmsh.model.getBoundary([(3, vt)],
                                                              oriented=False) if d == 2]
                rad = []
                for s in faces:
                    cx, _cy, cz = gmsh.model.occ.getCenterOfMass(2, s)
                    rad.append(float(np.hypot(cx, cz)))
                i_root = int(np.argmin(rad))
                root_s.append(faces[i_root])
                skin_s += [s for j, s in enumerate(faces) if j != i_root]
            gmsh.model.addPhysicalGroup(2, skin_s, name="skin")
            gmsh.model.addPhysicalGroup(2, root_s, name="root")
            try:
                gmsh.model.mesh.generate(2)
            except Exception as exc:
                raise SolverError(f"blade surface meshing failed: {exc}") from exc
            gmsh.write(str(path))
            _, _, tri = gmsh.model.mesh.getElements(2)
            n_tri = sum(len(t) for t in tri) // 3
    finally:
        gmsh.finalize()
    regions = sorted(set(ln.split()[1] for ln in path.read_text().splitlines()
                         if ln.startswith("solid") and len(ln.split()) > 1))
    return {"stl": path, "n_triangles": n_tri, "occ_volume_m3": volume,
            "regions": regions}


def _grading(L: float, core_half: float, n: int, core_cell_frac: float,
             expansion: float = 8.0) -> str:
    """blockMesh multi-grading: a uniform fine CORE containing the rotor, with
    the cells expanding to the far field outside it. Three segments per
    direction, each (lengthFraction cellFraction expansionRatio)."""
    if not 0.0 < core_half < L:
        raise SolverError(f"core half-extent {core_half} must be inside {L}")
    if not 0.0 < core_cell_frac < 1.0:
        raise SolverError("core_cell_frac must be in (0, 1)")
    f_core = core_half / L                      # of the HALF length
    f_out = (1.0 - f_core) / 2.0                # each outer segment, of FULL length
    c_out = (1.0 - core_cell_frac) / 2.0
    return (f"( ({f_out:.6f} {c_out:.6f} {1.0/expansion:.6f}) "
            f"({f_core:.6f} {core_cell_frac:.6f} 1) "
            f"({f_out:.6f} {c_out:.6f} {expansion:.6f}) )")


def _blockmesh(L: float, n: int, periodic: bool,
               core_half: float = 0.0, core_cell_frac: float = 0.55) -> str:
    """Two-block background mesh for the z>0 half-space (180-degree sector).

    Two blocks, split at x = 0, because the cyclic pair is the two halves of the
    z = 0 plane and blockMesh cannot split one block face into two patches."""
    if not periodic:
        # Full 360-degree box: no periodic pair at all, so no axis degeneracy.
        vs = [(x, y, z) for z in (-L, L) for y in (-L, L) for x in (-L, L)]
        verts = "\n".join(f"    ({a:.6f} {b:.6f} {c:.6f})" for a, b, c in vs)
        m = 2 * n
        # GRADED, not uniform. snappyHexMesh runs its surface-refinement loop
        # ONCE, on the raw background mesh, and abandons it if no cell edge
        # intersects the surface ("Stopping refining since too few cells
        # selected"). A blade 3.4 mm thick with a 25 mm chord does not cut the
        # corner edges of a 40 mm cell, so a uniform background silently
        # produces a mesh with NO surface refinement at all — measured: levels
        # (5,6) and (6,7) gave identical 177 k-cell meshes. The core must
        # therefore already resolve the blade before snappy starts. Grading
        # buys that without paying for a uniform fine mesh out to the far field.
        grade = _grading(L, core_half, n, core_cell_frac)
        return f"""
scale   1;

vertices
(
{verts}
);

blocks ( hex (0 1 3 2 4 5 7 6) ({m} {m} {m}) simpleGrading ({grade} {grade} {grade}) );

edges ();

boundary
(
    farfield
    {{
        type    patch;
        faces
        (
            (0 1 3 2)   // z = -L
            (4 6 7 5)   // z = +L
            (0 4 5 1)   // y = -L
            (2 3 7 6)   // y = +L
            (0 2 6 4)   // x = -L
            (1 5 7 3)   // x = +L
        );
    }}
);

mergePatchPairs ();
"""
    v = []
    for iz in (0.0, L):
        for iy in (-L, L):
            for ix in (-L, 0.0, L):
                v.append((ix, iy, iz))
    # index(ix,iy,iz) = ix + 3*iy + 6*iz
    verts = "\n".join(f"    ({a:.6f} {b:.6f} {c:.6f})" for a, b, c in v)
    nx, ny, nz = n, 2 * n, n            # y spans 2L, x spans 2L (n per block)
    return f"""
scale   1;

vertices
(
{verts}
);

blocks
(
    hex (0 1 4 3 6 7 10 9)  ({nx} {ny} {nz}) simpleGrading (1 1 1)
    hex (1 2 5 4 7 8 11 10) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges ();

boundary
(
    cyclic_neg
    {{
        type            cyclic;
        neighbourPatch  cyclic_pos;
        transform       rotational;
        rotationAxis    (0 1 0);
        rotationCentre  (0 0 0);
        // A 180-degree sector is the degenerate periodic case: both halves of
        // the pair lie in the SAME plane (z = 0) and meet ON the rotation axis,
        // where cell size collapses. snappyHexMesh perturbs points there, and
        // the default 1e-4 area-match tolerance rejects the result. Relaxed to
        // 1e-2; the ACTUAL worst mismatch is reported in the mesh log so the
        // number is measured, not assumed benign.
        matchTolerance  0.01;
        faces ( (0 3 4 1) );
    }}
    cyclic_pos
    {{
        type            cyclic;
        neighbourPatch  cyclic_neg;
        transform       rotational;
        rotationAxis    (0 1 0);
        rotationCentre  (0 0 0);
        // A 180-degree sector is the degenerate periodic case: both halves of
        // the pair lie in the SAME plane (z = 0) and meet ON the rotation axis,
        // where cell size collapses. snappyHexMesh perturbs points there, and
        // the default 1e-4 area-match tolerance rejects the result. Relaxed to
        // 1e-2; the ACTUAL worst mismatch is reported in the mesh log so the
        // number is measured, not assumed benign.
        matchTolerance  0.01;
        faces ( (1 4 5 2) );
    }}
    farfield
    {{
        type            patch;
        faces
        (
            (0 1 7 6) (1 2 8 7)          // y = -L
            (3 9 10 4) (4 10 11 5)       // y = +L
            (0 6 9 3)                    // x = -L
            (2 5 11 8)                   // x = +L
            (6 7 10 9) (7 8 11 10)       // z = +L (outer radial)
        );
    }}
);

mergePatchPairs ();
"""



def leading_edge_first_cos(design: PropellerDesign, cfg: HoverCaseConfig,
                           section: SectionFn = DEFAULT_SECTION) -> np.ndarray:
    """cos of the angle between the chord (LE->TE) and the direction of travel,
    at EVERY station. Must be NEGATIVE everywhere: if the leading edge leads,
    the vector TE->LE points along the motion, so LE->TE opposes it.

    Checked at every station, not just mid-span, because a twist-SIGN error
    would leave a correct mid-span reading while reversing the root or tip."""
    rings = station_rings(design, section, cfg.n_chord, cfg.te_thickness_m)
    k = cfg.n_chord
    om = np.array([0.0, cfg.omega_signed(), 0.0])
    out = []
    for st in rings:
        le = st[0]
        te = 0.5 * (st[k - 1] + st[k])
        chord = (te - le)[:2]
        travel = np.cross(om, le)[:2]
        nc, nt = np.linalg.norm(chord), np.linalg.norm(travel)
        if nc == 0.0 or nt == 0.0:
            raise SolverError("degenerate chord or zero rotation rate")
        out.append(float(np.dot(chord / nc, travel / nt)))
    return np.asarray(out)


def assert_leading_edge_first(design: PropellerDesign, cfg: HoverCaseConfig,
                              section: SectionFn = DEFAULT_SECTION) -> np.ndarray:
    """Raise unless every station advances leading-edge-first."""
    cos = leading_edge_first_cos(design, cfg, section)
    bad = np.flatnonzero(cos >= 0.0)
    if bad.size:
        raise SolverError(
            f"ROTATION IS BACKWARDS at {bad.size} of {cos.size} stations "
            f"(cos = {np.round(cos[bad], 4).tolist()}): the trailing edge leads. "
            f"MRF omega about +y is {cfg.omega_signed():+.4f} rad/s; the blade "
            "must advance leading-edge-first. See HoverCaseConfig.omega_signed.")
    return cos



def signed_incidence_deg(design: PropellerDesign, cfg: HoverCaseConfig,
                         section: SectionFn = DEFAULT_SECTION) -> np.ndarray:
    """Signed angle of attack (deg) at every station, for the CONFIGURED omega.

    LE-ordering alone is NOT enough: a blade can lead with its nose and still
    sit at negative incidence, producing no thrust. This measures the angle
    between the relative wind and the chord, signed by the SUCTION side.

    alpha = +asin(wind_hat . n_suction). Lift acts toward the suction side only
    when the relative wind ARRIVES on the pressure side, i.e. when the wind has
    a positive component along the suction normal. The sign was inverted here
    once; it was settled EMPIRICALLY, by measuring which blade face actually
    carried the higher pressure in a solved field (see README), not by algebra.

    The suction normal is taken between UPPER and LOWER surface points at the
    SAME chordwise index (polygon layout: lower j <-> upper p-j). Sampling
    mismatched indices yields a vector nearly PARALLEL to the chord and a
    sign-flipped answer - that error cost a run and a wrong diagnosis, so the
    perpendicularity is asserted here."""
    rings = station_rings(design, section, cfg.n_chord, cfg.te_thickness_m)
    k = cfg.n_chord
    pn = rings.shape[1]
    j = k // 2
    om = np.array([0.0, cfg.omega_signed(), 0.0])
    out = []
    for st in rings:
        le = st[0]
        te = 0.5 * (st[k - 1] + st[k])
        chord = (te - le)[:2]
        chord = chord / np.linalg.norm(chord)
        n = (st[pn - j][:2] - st[j][:2])
        n = n / np.linalg.norm(n)
        if abs(float(np.dot(chord, n))) > 1e-6:
            raise SolverError(
                "suction normal is not perpendicular to the chord "
                f"(dot = {float(np.dot(chord, n)):.4f}) - mismatched polygon "
                "indices; the incidence sign would be meaningless")
        travel = np.cross(om, le)[:2]
        travel = travel / np.linalg.norm(travel)
        out.append(float(np.degrees(np.arcsin(np.clip(np.dot(-travel, n), -1.0, 1.0)))))
    return np.asarray(out)


def assert_positive_incidence(design: PropellerDesign, cfg: HoverCaseConfig,
                              section: SectionFn = DEFAULT_SECTION) -> np.ndarray:
    """Raise unless every station sees POSITIVE angle of attack."""
    a = signed_incidence_deg(design, cfg, section)
    bad = np.flatnonzero(a <= 0.0)
    if bad.size:
        raise SolverError(
            f"NEGATIVE INCIDENCE at {bad.size} of {a.size} stations "
            f"(alpha = {np.round(a[bad], 2).tolist()} deg): the blade leads with "
            "its nose but is pitched the wrong way, so it makes no thrust. "
            f"MRF omega about +y is {cfg.omega_signed():+.4f} rad/s.")
    return a


def build_hover_case(case: FoamCase, design: PropellerDesign, cfg: HoverCaseConfig,
                     section: SectionFn = DEFAULT_SECTION) -> dict:
    """Write every dictionary for the case. Returns a provenance dict."""
    src = case.src
    R = float(design.radius_m)
    L = cfg.far_field_radii * R
    omega = cfg.omega_signed()
    sector = cfg.sector(design.n_blades)
    if cfg.periodic and abs(sector - 180.0) > 1e-9:
        raise SolverError(
            f"the periodic writer implements the 180-degree sector only "
            f"(got {sector} deg for {design.n_blades} blades). The sector is "
            "config-driven; a new blockMesh writer is needed for other angles.")

    # HARD GUARD: a backwards rotor still makes thrust in the right direction,
    # so nothing downstream would flag it. Check before building anything.
    assert_leading_edge_first(design, cfg, section)
    assert_positive_incidence(design, cfg, section)

    geo = write_blade_stl(design, src / "constant" / "triSurface" / "blade.stl",
                          cfg, section)

    # --- mesh ---------------------------------------------------------------
    core_half = cfg.core_radii * R
    write_dict(src, "system", "blockMeshDict",
               _blockmesh(L, cfg.background_cells_per_span, cfg.periodic,
                          core_half, cfg.core_cell_frac))

    # background cell size IN THE CORE, which is what governs whether snappy
    # detects the blade at all
    m = 2 * cfg.background_cells_per_span
    bg = 2.0 * core_half / max(int(m * cfg.core_cell_frac), 1)
    finest = bg / (2 ** cfg.surface_level[1])
    r_box = 1.25 * R
    write_dict(src, "system", "snappyHexMeshDict", f"""
castellatedMesh true;
snap            true;
addLayers       {'true' if cfg.n_layers > 0 else 'false'};

geometry
{{
    blade.stl {{ type triSurfaceMesh; name blade; }}
    rotorBox
    {{
        type searchableBox;
        min ({-r_box:.6f} {-0.5*r_box:.6f} {0.0 if cfg.periodic else -r_box:.6f});
        max ({ r_box:.6f} { 0.5*r_box:.6f} {r_box:.6f});
    }}
}}

castellatedMeshControls
{{
    maxLocalCells       2000000;
    maxGlobalCells      12000000;
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 3;

    features ();

    refinementSurfaces
    {{
        blade
        {{
            level ({cfg.surface_level[0]} {cfg.surface_level[1]});
            regions
            {{
                skin {{ level ({cfg.surface_level[0]} {cfg.surface_level[1]});
                        patchInfo {{ type wall; }} }}
                root {{ level ({cfg.surface_level[0]} {cfg.surface_level[1]});
                        patchInfo {{ type wall; }} }}
            }}
        }}
    }}

    resolveFeatureAngle 30;

    refinementRegions
    {{
        rotorBox {{ mode inside; levels ((1e15 {cfg.rotor_region_level})); }}
    }}

    // a point that is unambiguously INSIDE the fluid and outside the blade:
    // well outboard of the tip, off-axis, above the rotor plane.
    locationInMesh ({0.37*L:.6f} {0.41*L:.6f} {0.53*L:.6f});
    allowFreeStandingZoneFaces false;
}}

snapControls
{{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      50;
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap     true;
    explicitFeatureSnap     false;
    multiRegionFeatureSnap  false;
}}

addLayersControls
{{
    relativeSizes       {'false' if cfg.first_layer_m else 'true'};
    layers {{ "blade_.*" {{ nSurfaceLayers {cfg.n_layers}; }} }}
    expansionRatio      {cfg.layer_expansion};
    {f'firstLayerThickness {cfg.first_layer_m:.8e};' if cfg.first_layer_m
       else f'finalLayerThickness {cfg.first_layer_rel};'}
    minThickness        {(cfg.first_layer_m * 0.1) if cfg.first_layer_m else 0.05:.8e};
    nGrow               0;
    featureAngle        130;
    slipFeatureAngle    30;
    nRelaxIter          5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals      3;
    nSmoothThickness    10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle  90;
    nBufferCellsNoExtrude 0;
    nLayerIter          50;
    nRelaxedIter        20;
}}

meshQualityControls
{{
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minVol              1e-13;
    minTetQuality       1e-15;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.02;
    minVolRatio         0.01;
    minTriangleTwist    -1;
    nSmoothScale        4;
    errorReduction      0.75;
}}

mergeTolerance 1e-6;
""")

    write_dict(src, "system", "meshQualityDict",
               '#includeEtc "caseDicts/meshQualityDict"')

    write_dict(src, "system", "decomposeParDict", f"""
numberOfSubdomains {cfg.n_proc};
method          scotch;
""")

    write_dict(src, "system", "surfaceFeatureExtractDict", """
blade.stl
{
    extractionMethod    extractFromSurface;
    includedAngle       150;
    writeObj            yes;
}
""")

    # --- physics ------------------------------------------------------------
    write_dict(src, "constant", "transportProperties", f"""
// KINEMATIC viscosity (m^2/s). simpleFoam is incompressible: p is p/rho.
transportModel  Newtonian;
nu              {cfg.nu:.6e};
""")
    write_dict(src, "constant", "turbulenceProperties", f"""
simulationType  RAS;
RAS
{{
    RASModel        {cfg.turbulence_model};
    turbulence      on;
    printCoeffs     on;
}}
""")
    # Whole-domain MRF: for an isolated rotor in hover the flow is steady in the
    # rotating frame everywhere, so one frame covers the domain (equivalent to a
    # single rotating frame). The far field is declared non-rotating so the
    # quiescent outer boundary stays in the absolute frame; the blade is not
    # listed, so it rotates with the frame, which is what makes it a moving wall.
    # MRF over a ROTOR CELLZONE, not the whole mesh. v2512 rejects
    # `cellZone none` outright ("cannot find MRF cellZone none"), and a bounded
    # zone is the better model anyway: the rotating frame covers the blade, the
    # far field stays in the absolute frame where the quiescent open boundary is
    # posed. The zone is sized from the design (1.3 R radius, +/-0.4 R axially),
    # so it contains the blade with clearance at any scale.
    zone_r = cfg.mrf_zone_radii * R
    zone_h = cfg.mrf_zone_half_height_radii * R
    write_dict(src, "system", "topoSetDict", f"""
actions
(
    {{
        name    rotor;  type cellSet;     action new;
        source  cylinderToCell;
        point1  (0 {-zone_h:.6f} 0);
        point2  (0 { zone_h:.6f} 0);
        radius  {zone_r:.6f};
    }}
    {{
        name    rotor;  type cellZoneSet; action new;
        source  setToCellZone;  set rotor;
    }}
);
""")
    write_dict(src, "constant", "MRFProperties", f"""
MRF1
{{
    cellZone        rotor;
    active          yes;
    nonRotatingPatches ( );
    origin          (0 0 0);
    axis            (0 1 0);            // propopt frame: y is the rotation axis
    omega           {omega:.9f};        // rad/s = 2*pi*{cfg.rpm}/60
}}
""")

    # --- fields -------------------------------------------------------------
    # Quiescent far field. k is small but non-zero (a true zero would make the
    # SST equations degenerate); omega from a length scale of one tip radius.
    # Free-stream turbulence pair, DERIVED not guessed. Hover has no U_inf, so
    # the reference velocity is the nominal induced velocity from momentum
    # theory at the measured thrust, v_i = sqrt(T / 2 rho A) ~ 5 m/s.
    #   I = 1%, U_ref = 5 m/s  ->  k = 1.5 (I U_ref)^2 = 3.75e-3
    # The TEXTBOOK epsilon (Cmu^0.75 k^1.5 / L, L = 0.1R) gives nu_t/nu = 29:
    # a "quiescent" far field carrying 29x laminar eddy viscosity, which is
    # what diverged the first k-epsilon attempt. epsilon is instead set so the
    # far-field eddy viscosity is laminar-order, nu_t/nu = 1, which is what
    # still air actually means:  epsilon = Cmu k^2 / nu.
    k_inf = 3.75e-3
    omega_inf = 1.0
    # Cmu^0.75 k^1.5 / L with Cmu=0.09, L = one tip radius
    eps_inf = 0.09 * k_inf ** 2 / cfg.nu          # -> nu_t/nu = 1
    write_dict(src, "0", "U", """
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{
    "blade_.*" { type noSlip; }
    farfield   { type FFU; }
    cyclic_neg { type cyclic; }
    cyclic_pos { type cyclic; }
}
""".replace("FFU", _FF_U[cfg.far_field_bc]), cls="volVectorField")

    write_dict(src, "0", "p", """
// KINEMATIC pressure (m^2/s^2), NOT pascals: simpleFoam is incompressible.
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    "blade_.*" { type zeroGradient; }
    farfield   { type FFP; }
    cyclic_neg { type cyclic; }
    cyclic_pos { type cyclic; }
}
""".replace("FFP", _FF_P[cfg.far_field_bc]), cls="volScalarField")

    write_dict(src, "0", "k", f"""
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {k_inf};
boundaryField
{{
    "blade_.*" {{ type kqRWallFunction; value uniform {k_inf}; }}
    farfield   {{ type inletOutlet; inletValue uniform {k_inf}; value uniform {k_inf}; }}
    cyclic_neg {{ type cyclic; }}
    cyclic_pos {{ type cyclic; }}
}}
""", cls="volScalarField")

    write_dict(src, "0", "omega", f"""
dimensions      [0 0 -1 0 0 0 0];
internalField   uniform {omega_inf};
boundaryField
{{
    "blade_.*" {{ type omegaWallFunction; value uniform {omega_inf}; }}
    farfield   {{ type inletOutlet; inletValue uniform {omega_inf}; value uniform {omega_inf}; }}
    cyclic_neg {{ type cyclic; }}
    cyclic_pos {{ type cyclic; }}
}}
""", cls="volScalarField")

    # epsilon is written unconditionally: kEpsilon reads (k, epsilon) and
    # kOmegaSST reads (k, omega), so an unused field is inert. Writing both
    # makes turbulence_model a pure config switch with no case-structure change.
    write_dict(src, "0", "epsilon", f"""
dimensions      [0 2 -3 0 0 0 0];
internalField   uniform {eps_inf};
boundaryField
{{
    "blade_.*" {{ type epsilonWallFunction; value uniform {eps_inf}; }}
    farfield   {{ type inletOutlet; inletValue uniform {eps_inf}; value uniform {eps_inf}; }}
}}
""", cls="volScalarField")

    write_dict(src, "0", "nut", """
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    "blade_.*" { type NUTWF; value uniform 0; }
    farfield   { type calculated; value uniform 0; }
    cyclic_neg { type cyclic; }
    cyclic_pos { type cyclic; }
}
""".replace("NUTWF", cfg.nut_wall_function), cls="volScalarField")

    # --- run control --------------------------------------------------------
    # forces: rho must be given explicitly or an incompressible run returns
    # force/rho. CofR is the rotor hub; the torque about the y axis is what the
    # motor must supply, and the y force is thrust.
    write_dict(src, "system", "controlDict", f"""
application     simpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {cfg.n_iterations};
deltaT          1;
writeControl    timeStep;
writeInterval   {cfg.write_interval};
purgeWrite      2;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    forces
    {{
        type            forces;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        log             false;
        patches         ("blade_.*");
        rho             rhoInf;
        rhoInf          {cfg.rho};      // SI kg/m^3 - without this the
                                        // incompressible run reports force/rho
        CofR            (0 0 0);
    }}
    forcesRoot
    {{
        // the exposed bluff root cap has NO BEMT analogue; integrated on its
        // own so its share of thrust and torque is visible, not buried
        type            forces;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        log             false;
        patches         (blade_root);
        rho             rhoInf;
        rhoInf          {cfg.rho};
        CofR            (0 0 0);
    }}
    yPlus
    {{
        type            yPlus;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
        writeFields     true;
        log             true;
    }}
}}
""")

    write_dict(src, "system", "fvSchemes", """
ddtSchemes      { default steadyState; }
gradSchemes     { default cellLimited Gauss linear 1; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss limitedLinear 1;
    div(phi,omega)  bounded Gauss limitedLinear 1;
    // epsilon is listed unconditionally so turbulence_model stays a pure
    // config switch: kEpsilon needs div(phi,epsilon), kOmegaSST needs
    // div(phi,omega), and an unused entry is inert.
    div(phi,epsilon) bounded Gauss limitedLinear 1;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited corrected 0.33; }
interpolationSchemes { default linear; }
snGradSchemes   { default limited corrected 0.33; }
wallDist        { method meshWave; }
""")

    write_dict(src, "system", "fvSolution", """
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-08;
        relTol          0.01;
        smoother        GaussSeidel;
    }
    Phi
    {
        // potentialFoam initialisation: a divergence-free starting field.
        // Starting simpleFoam from uniform zero puts the whole domain in an
        // impulsive transient that a large, graded far field cannot absorb.
        solver          GAMG;
        tolerance       1e-07;
        relTol          0.01;
        smoother        GaussSeidel;
    }
    "(U|k|omega|epsilon)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-08;
        relTol          0.01;
    }
}
potentialFlow
{
    nNonOrthogonalCorrectors 10;
}
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent      yes;
    residualControl { p 1e-5; U 1e-5; "(k|omega|epsilon)" 1e-5; }
}
relaxationFactors
{
    equations { U 0.7; p 0.7; "(k|omega|epsilon)" 0.5; ".*" 0.7; }
}
""")

    return {
        "rpm": cfg.rpm, "omega_rad_s": omega, "sector_deg": sector,
        "n_blades": design.n_blades, "radius_m": R,
        "domain_half_extent_m": L, "domain_radii": cfg.far_field_radii,
        "background_cell_m": bg, "finest_cell_m": finest,
        "mrf_zone_radius_m": zone_r, "mrf_zone_half_height_m": zone_h,
        "stl_triangles": geo["n_triangles"], "occ_volume_m3": geo["occ_volume_m3"],
        "te_thickness_m": cfg.te_thickness_m,
        "rho": cfg.rho, "nu": cfg.nu, "turbulence": cfg.turbulence_model,
    }
