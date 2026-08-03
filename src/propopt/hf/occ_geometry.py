"""Phase-4a slice 2: the blade as an OpenCASCADE B-rep solid (HF path only).

WHY: a fillet is a B-rep operation. It cannot be applied to a triangle mesh, so
the root-fillet work needs a real CAD solid. This module builds one with gmsh's
BUNDLED OCC kernel (`gmsh.model.occ`) — no new dependency, no STEP round-trip.

WHAT IT IS NOT: this does not replace the optimizer's triangle loft. That loft
stays the geometry the optimizer scores and the mass of record; this is additive
and lives entirely in `hf/`. Every design evaluated in the loop still goes
through `geometry.build_blade_solid`.

SHARED DEFINITIONS. The aerofoil is NOT reimplemented here: section shape comes
from `geometry.section_polygon` (including the blunt-TE treatment) and station
placement from `geometry.extended_stations`. Two section definitions is exactly
the divergence the Phase-2a shared-station work removed. The one thing this
module restates is the polygon -> 3-D placement (rotate by twist about the
quarter-chord, stack along the radial axis), because `build_blade_solid` does it
in vectorized form inside the optimizer's hot path and should not be
restructured for the HF path's benefit. `tests/test_hf_occ.py` asserts the
station rings produced here are BITWISE equal to the triangle loft's vertices,
so the restatement is verified rather than trusted.

BLUNT TRAILING EDGE. With `te_thickness_m > 0` the section polygon carries TWO
distinct trailing vertices (TE-lower then TE-upper), so the profile wire simply
gains one extra short segment: the TE face edge. Nothing special is done for it
— it is an ordinary polyline segment, and lofting it between stations generates
the flat TE band as a normal ruled face. With a SHARP TE the two coincide and
the wire has a cusp there; OCC tolerates it less well (see the failure-rate
report), which is one more reason the 0.4 mm default matters.

PROFILE CURVE CHOICE. The wire is a POLYLINE through the same polygon vertices
the triangle loft uses, and the loft is RULED (`makeRuled=True`). That is the
faithful choice for a fidelity gate: the section is then identical to the
triangle loft's, so the only remaining difference is spanwise — OCC spans each
station pair with a bilinear ruled patch where the triangle loft chords that
same patch with two triangles. A spline profile would have been smoother but
would bulge outside the polygon, adding a second, confounding volume difference
on top of the twist effect this slice is trying to measure.

UNITS: SI (metres) throughout, same as the loft and the CalculiX deck.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..contracts import PropellerDesign, SolverError
from ..geometry import (
    DEFAULT_N_CHORD,
    DEFAULT_SECTION,
    PITCH_AXIS_CHORD_FRAC,
    SectionFn,
    extended_stations,
    section_polygon,
)
from .mesh import QUALITY_MEASURE, TetMesh, _extract, tet_volumes


@dataclass(frozen=True)
class OccBlade:
    """Result of an OCC construction (volume from the kernel, not from a mesh)."""
    volume_m3: float
    n_stations: int
    n_profile_points: int
    blunt_te: bool


def station_rings(design: PropellerDesign,
                  section: SectionFn = DEFAULT_SECTION,
                  n_chord: int = DEFAULT_N_CHORD,
                  te_thickness_m: float = 0.0) -> np.ndarray:
    """(n_stations, P, 3) section vertices in METRES — the same points the
    triangle loft uses, in the same order.

    Mirrors `geometry.build_blade_solid`'s placement: chordwise coordinate
    measured from the quarter-chord pitch axis, rotated by the local twist in
    the (tangential, axial) plane, stacked along the radial axis. Verified
    bitwise against the loft in tests."""
    r, c, t = extended_stations(design)
    te = float(te_thickness_m)
    if not np.isfinite(te) or te < 0.0:
        raise SolverError(f"te_thickness_m must be finite and >= 0, got {te!r}")
    if te <= 0.0:
        # A sharp TE makes the profile wire CUSPED: upper and lower surfaces
        # meet at a single point with zero included angle. That is the case a
        # B-rep kernel tolerates least well — it is the same degeneracy that
        # made gmsh's STL remesh mode fail outright in the Phase-4a slice — and
        # it is untested here. Refused explicitly rather than left to fail
        # somewhere deeper in OCC with an opaque message. The HF path requires
        # the finite TE that is the adopted default (0.4 mm).
        raise SolverError(
            "OCC construction requires a finite trailing edge "
            "(te_thickness_m > 0); a sharp TE gives a cusped profile wire, "
            "which is unsupported in the HF path")
    if np.any(c <= 0.0):
        raise SolverError("chord must be positive at every station")

    if te > 0.0:
        polys = np.stack([section_polygon(section, n_chord, te / ci) for ci in c])
        xi = (polys[:, :, 0] - PITCH_AXIS_CHORD_FRAC) * c[:, None]
        eta = polys[:, :, 1] * c[:, None]
    else:
        poly = section_polygon(section, n_chord)
        xi = (poly[:, 0] - PITCH_AXIS_CHORD_FRAC)[None, :] * c[:, None]
        eta = poly[:, 1][None, :] * c[:, None]

        # NOTE: rotation sense. The section is rotated by MINUS the twist.
    # The chord is laid out with the leading edge at xi = -0.25c (i.e. at
    # NEGATIVE x), so a +twist rotation puts the trailing edge ABOVE the
    # leading edge relative to the oncoming wind, giving negative
    # incidence for the only rotation sense that leads with the nose.
    # Settled empirically from blade surface pressure in a solved field,
    # not by algebra -- see hf/hover_case.signed_incidence_deg.
    cos_t, sin_t = np.cos(-t)[:, None], np.sin(-t)[:, None]
    rings = np.empty((len(r), xi.shape[1], 3))
    rings[:, :, 0] = xi * cos_t - eta * sin_t      # tangential (rotor plane)
    rings[:, :, 1] = xi * sin_t + eta * cos_t      # axial
    rings[:, :, 2] = r[:, None]                    # radial (span)
    return rings


def _build_occ_solid(gmsh, rings: np.ndarray) -> int:
    """Create the lofted solid in the CURRENT gmsh model; return its volume tag.

    Caller owns gmsh.initialize()/finalize(). Raises SolverError on any OCC
    failure — a B-rep kernel refusing a loft is a feasibility outcome for the
    HF path, not a crash."""
    n_st, p, _ = rings.shape
    if n_st < 2:
        raise SolverError("need >= 2 stations to loft an OCC solid")
    try:
        wires = []
        for ring in rings:
            pts = [gmsh.model.occ.addPoint(*map(float, v)) for v in ring]
            lines = [gmsh.model.occ.addLine(pts[i], pts[(i + 1) % p])
                     for i in range(p)]
            wires.append(gmsh.model.occ.addCurveLoop(lines))
        # makeSolid=True caps root and tip with planar faces (the section wires
        # are planar), so the result is a single closed solid, not a shell.
        # makeRuled=True spans each station pair with a ruled surface, the
        # direct B-rep analogue of the triangle loft's chorded quads.
        out = gmsh.model.occ.addThruSections(wires, makeSolid=True, makeRuled=True)
        gmsh.model.occ.synchronize()
    except Exception as exc:
        raise SolverError(f"OCC loft failed: {exc}") from exc

    vols = [tag for dim, tag in out if dim == 3]
    if len(vols) != 1:
        raise SolverError(
            f"OCC loft produced {len(vols)} volumes (expected exactly 1)")
    return vols[0]


def occ_blade(design: PropellerDesign,
              section: SectionFn = DEFAULT_SECTION,
              n_chord: int = DEFAULT_N_CHORD,
              te_thickness_m: float = 0.0,
              step_path: str | Path | None = None) -> OccBlade:
    """Build the OCC solid and return its kernel-computed volume.

    `step_path` optionally writes a STEP file for inspection; it is NOT part of
    the pipeline (nothing reads it back)."""
    try:
        import gmsh
    except ImportError as exc:
        raise SolverError("gmsh is not installed (pip install -e '.[hf]')") from exc

    rings = station_rings(design, section, n_chord, te_thickness_m)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
        with contextlib.redirect_stdout(io.StringIO()):
            tag = _build_occ_solid(gmsh, rings)
            volume = float(gmsh.model.occ.getMass(3, tag))
            if step_path is not None:
                gmsh.write(str(step_path))
        if not np.isfinite(volume) or volume <= 0.0:
            raise SolverError(f"OCC solid has non-physical volume {volume!r}")
    finally:
        gmsh.finalize()
    return OccBlade(volume, rings.shape[0], rings.shape[1], te_thickness_m > 0.0)


def mesh_occ_blade(design: PropellerDesign,
                   target_size_m: float,
                   *,
                   section: SectionFn = DEFAULT_SECTION,
                   n_chord: int = DEFAULT_N_CHORD,
                   te_thickness_m: float = 0.0,
                   order: int = 2,
                   min_size_m: float | None = None,
                   curvature_elements: float = 0.0,
                   reference_volume_m3: float | None = None,
                   max_volume_drift: float = 0.01,
                   optimize: bool = True) -> TetMesh:
    """Tet-mesh the OCC solid NATIVELY — no STL intermediate.

    gmsh meshes the B-rep directly, so the boundary is the analytic surface
    rather than a facet approximation, and `curvature_elements` (> 0) enables
    curvature-adaptive sizing: the leading edge gets refined because it is
    curved, not because someone guessed a size for it.
    """
    try:
        import gmsh
    except ImportError as exc:
        raise SolverError("gmsh is not installed (pip install -e '.[hf]')") from exc

    rings = station_rings(design, section, n_chord, te_thickness_m)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(target_size_m))
        gmsh.option.setNumber("Mesh.MeshSizeMin",
                              float(min_size_m if min_size_m is not None
                                    else target_size_m))
        if curvature_elements > 0.0:
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",
                                  float(curvature_elements))
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.Optimize", 1 if optimize else 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1 if optimize else 0)
        with contextlib.redirect_stdout(io.StringIO()):
            _build_occ_solid(gmsh, rings)
            try:
                gmsh.model.mesh.generate(3)
                if order == 2:
                    gmsh.model.mesh.setOrder(2)
            except Exception as exc:
                raise SolverError(f"OCC 3-D meshing failed: {exc}") from exc
            nodes, tets, n_tris, qual = _extract(gmsh, order, "occ-blade")
    finally:
        gmsh.finalize()

    vols = tet_volumes(nodes, tets)
    if np.any(vols <= 0.0):
        raise SolverError(
            f"{int(np.sum(vols <= 0.0))} non-positive-volume tets in the OCC mesh")
    volume = float(np.sum(vols))
    mesh = TetMesh(nodes=nodes, tets=tets, volume_m3=volume, n_boundary_tris=n_tris,
                   quality_min=float(qual.min()), quality_mean=float(qual.mean()),
                   quality_p01=float(np.percentile(qual, 1.0)), quality=qual,
                   remeshed=False, order=int(order),
                   reference_volume_m3=reference_volume_m3)
    drift = mesh.volume_rel_error
    if drift is not None and abs(drift) > max_volume_drift:
        raise SolverError(
            f"OCC mesh volume drifted {drift:+.2%} from the reference "
            f"(limit {max_volume_drift:.1%})")
    return mesh
