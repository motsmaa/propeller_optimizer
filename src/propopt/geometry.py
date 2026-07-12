"""Parametric blade geometry: design -> watertight triangulated solid (Phase 2a).

Engine choice: PURE NUMPY STRUCTURED LOFT — no CAD kernel in the inner loop.
Rationale (assessed against FreeCAD-headless and OCC-based kernels such as
CadQuery/build123d): the optimizer builds thousands of solids per run, and a
B-rep kernel there is the wrong tool — OCC lofts fail opaquely and
nondeterministically on unusual sections, cost 0.1-1 s each, and drag in a
heavyweight non-pip-friendly dependency. A structured triangle mesh is
watertight BY CONSTRUCTION (closed-manifold connectivity is a topological
property of the generator, independent of coordinates), builds in ~1 ms,
gives an exact enclosed volume via the divergence theorem, and is exactly the
representation a reduced structural solver and STL export consume. A CAD
kernel (build123d/STEP) belongs later, once per *final* Pareto design for
manufacturing export — not per genome. Every build is verified closed-manifold
and positive-volume; failures raise SolverError, never a silent bad solid.

Solid produced: ONE blade (root cap + lofted skin + tip cap), with named
patches 'hub' (root cap: the hub-mating face), 'blade' (lofted skin), 'tip'
(tip cap). Total design mass = n_blades * blade volume * density; the hub
body itself is NOT modelled (its dimensions are not in the genome — flagged
assumption).

Assumptions (stated per Phase-2 report):
- Section shape enters ONLY through the `SectionFn` seam. Default is a NACA
  4412-like parametric 4-digit section (4% camber @ 40%, 12% thick, closed
  trailing edge, vertical camber/thickness composition) to stay consistent
  with the Phase-1 aero polar proxy; swap both together when a matched thin
  section lands.
- Sections are stacked with the pitch axis at 25% chord, rotated by twist
  about the radial axis; no rake/skew/cone.
- Station radii come from the shared contract field `design.station_r_m`;
  `solid_from_design` extends the loft with end-held sections to hub and R,
  so the solid spans exactly the same interval [hub, R] the aero quadrature
  integrates (see stations.station_weights).
- No exact global self-intersection test. Simplicity is guaranteed per
  section by construction (upper > lower for thickness > 0), and loft
  self-intersection is excluded for |twist| < pi/2 per station, adjacent
  twist steps < pi/2, and strictly increasing radii — all enforced with
  SolverError. Residual risk is limited to pathological chord/gap aspect
  ratios far outside the genome bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .contracts import PropellerDesign, SolverError

PITCH_AXIS_CHORD_FRAC = 0.25  # sections rotate about the quarter-chord point
DEFAULT_N_CHORD = 25          # chordwise samples per surface (section has 2n-2 verts)

# section(n_chord) -> (x, y_upper, y_lower), unit chord: x[0]=0 (LE), x[-1]=1
# (TE), y_upper[0]==y_lower[0], y_upper[-1]==y_lower[-1] (closed TE), and
# y_upper > y_lower strictly between. This is the section-shape seam.
SectionFn = Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]]


def make_naca4_section(camber: float = 0.04, camber_pos: float = 0.4,
                       thickness: float = 0.12) -> SectionFn:
    """Parametric NACA 4-digit section (default ~4412) with a CLOSED trailing
    edge (-0.1036 quartic coefficient) so the loft can be watertight. Camber
    and thickness are composed vertically (not normal to the camber line) —
    a standard thin-section approximation that guarantees a simple polygon."""
    if thickness <= 0.0:
        raise ValueError("section thickness must be > 0 for a solid")

    def section(n_chord: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_chord)))  # cosine spacing
        yt = 5.0 * thickness * (0.2969 * np.sqrt(x) - 0.1260 * x
                                - 0.3516 * x ** 2 + 0.2843 * x ** 3
                                - 0.1036 * x ** 4)  # half-thickness, closed TE
        yc = np.zeros_like(x)
        if camber > 0.0:
            p = camber_pos
            fwd = x < p
            yc[fwd] = camber / p ** 2 * (2 * p * x[fwd] - x[fwd] ** 2)
            yc[~fwd] = camber / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[~fwd] - x[~fwd] ** 2)
        return x, yc + yt, yc - yt

    return section


DEFAULT_SECTION: SectionFn = make_naca4_section()


def section_polygon(section: SectionFn, n_chord: int) -> np.ndarray:
    """Closed simple polygon (P, 2), CCW: LE -> lower surface -> TE -> upper
    surface -> back to LE. LE and TE each appear exactly once (P = 2*n_chord - 2)."""
    x, yu, yl = section(n_chord)
    k = len(x)
    px = np.concatenate([[x[0]], x[1:k - 1], [x[-1]], x[k - 2:0:-1]])
    py = np.concatenate([[yl[0]], yl[1:k - 1], [yl[-1]], yu[k - 2:0:-1]])
    return np.stack([px, py], axis=1)


def polygon_area(poly: np.ndarray) -> float:
    """Shoelace area (positive for CCW)."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def is_closed_manifold(faces: np.ndarray) -> bool:
    """True iff every directed edge occurs exactly once and its reverse occurs
    too: the mesh is closed, 2-manifold and consistently oriented."""
    edges: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for e in ((a, b), (b, c), (c, a)):
            e = (int(e[0]), int(e[1]))
            edges[e] = edges.get(e, 0) + 1
    return all(n == 1 and edges.get((b, a), 0) == 1 for (a, b), n in edges.items())


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Enclosed volume via the divergence theorem (positive iff faces are
    outward-oriented). Exact for the polyhedron the mesh bounds."""
    v0, v1, v2 = (vertices[faces[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->", v0, np.cross(v1, v2)) / 6.0)


@dataclass(frozen=True)
class BladeSolid:
    """Watertight triangulated solid of ONE blade (SI: metres, m^3)."""
    vertices: np.ndarray              # (V, 3) float
    faces: np.ndarray                 # (F, 3) int, outward-oriented
    patches: dict[str, np.ndarray]    # 'hub' | 'blade' | 'tip' -> face indices
    volume_m3: float


def _cap_triangles(k: int, ring: np.ndarray, top: bool) -> list[tuple[int, int, int]]:
    """Triangulate one section polygon by zipping the (x-monotone) upper and
    lower vertex rows. `ring` maps polygon position -> global vertex id.
    top=True winds outward +z (tip cap), top=False outward -z (root cap)."""
    p = 2 * k - 2
    low = [0] + list(range(1, k - 1)) + [k - 1]           # LE, lower interior, TE
    up = [0] + [p - j for j in range(1, k - 1)] + [k - 1]  # same columns, upper row
    tris = []
    for j in range(k - 1):
        a, b, c, d = low[j], low[j + 1], up[j + 1], up[j]
        for tri in ((a, b, c), (a, c, d)):
            if len({tri[0], tri[1], tri[2]}) == 3:  # skip LE/TE degenerates
                t = tri if top else (tri[0], tri[2], tri[1])
                tris.append((int(ring[t[0]]), int(ring[t[1]]), int(ring[t[2]])))
    return tris


def build_blade_solid(
    radii_m: np.ndarray,
    chord_m: np.ndarray,
    twist_rad: np.ndarray,
    section: SectionFn = DEFAULT_SECTION,
    n_chord: int = DEFAULT_N_CHORD,
) -> BladeSolid:
    """Loft `section` through the stations -> verified watertight solid.

    Raises SolverError on any degenerate input or if the built mesh fails the
    closed-manifold / positive-volume verification (never returns silently
    broken geometry)."""
    r = np.asarray(radii_m, dtype=float)
    c = np.asarray(chord_m, dtype=float)
    t = np.asarray(twist_rad, dtype=float)
    if not (len(r) == len(c) == len(t)):
        raise SolverError("stations: radii/chord/twist lengths differ")
    if len(r) < 2:
        raise SolverError("need >= 2 stations to loft a solid")
    if not np.all(np.isfinite(r)) or np.any(np.diff(r) <= 0.0):
        raise SolverError("station radii must be finite and strictly increasing")
    if not np.all(np.isfinite(c)) or np.any(c <= 0.0):
        raise SolverError("chord must be positive and finite at every station")
    if not np.all(np.isfinite(t)) or np.any(np.abs(t) >= np.pi / 2):
        raise SolverError("twist must be finite and within (-pi/2, pi/2)")
    if np.any(np.abs(np.diff(t)) >= np.pi / 2):
        raise SolverError("adjacent-station twist step >= pi/2: loft would self-intersect")

    poly = section_polygon(section, n_chord)          # (P, 2) unit chord, CCW
    n_st, p = len(r), len(poly)
    xi = (poly[:, 0] - PITCH_AXIS_CHORD_FRAC)[None, :] * c[:, None]
    eta = poly[:, 1][None, :] * c[:, None]
    cos_t, sin_t = np.cos(t)[:, None], np.sin(t)[:, None]
    verts = np.empty((n_st, p, 3))
    verts[:, :, 0] = xi * cos_t - eta * sin_t          # tangential (rotor plane)
    verts[:, :, 1] = xi * sin_t + eta * cos_t          # axial
    verts[:, :, 2] = r[:, None]                        # radial (span)
    vertices = verts.reshape(-1, 3)

    faces: list[tuple[int, int, int]] = []
    ring = lambda i: np.arange(i * p, (i + 1) * p)
    for i in range(n_st - 1):                          # lofted skin
        a, b = ring(i), ring(i + 1)
        for j in range(p):
            q = (j + 1) % p
            faces.append((int(a[j]), int(a[q]), int(b[q])))
            faces.append((int(a[j]), int(b[q]), int(b[j])))
    n_skin = len(faces)
    faces += _cap_triangles(n_chord, ring(0), top=False)        # root cap (-z out)
    n_root = len(faces) - n_skin
    faces += _cap_triangles(n_chord, ring(n_st - 1), top=True)  # tip cap (+z out)
    faces_arr = np.asarray(faces, dtype=int)

    if not is_closed_manifold(faces_arr):
        raise SolverError("generated blade mesh is not a closed manifold")
    volume = mesh_volume(vertices, faces_arr)
    if not np.isfinite(volume) or volume <= 0.0:
        raise SolverError(f"generated blade solid has non-physical volume {volume!r}")

    patches = {
        "blade": np.arange(0, n_skin),
        "hub": np.arange(n_skin, n_skin + n_root),
        "tip": np.arange(n_skin + n_root, len(faces_arr)),
    }
    return BladeSolid(vertices, faces_arr, patches, volume)


def solid_from_design(design: PropellerDesign,
                      section: SectionFn = DEFAULT_SECTION,
                      n_chord: int = DEFAULT_N_CHORD) -> BladeSolid:
    """Build one blade solid from the design's shared stations.

    Lofts through `design.station_r_m`, EXTENDED with end-held sections
    (chord/twist frozen at the end-station values) to hub_radius_m and
    radius_m. The solid therefore covers exactly the interval [hub, R] that
    the aero quadrature integrates (stations own midpoint-bounded cells
    clipped to [hub, R]) — domain consistency between aero and geometry.
    End-holding matches the quadrature's treatment of the outer half-cells
    (station properties extended over their whole cell)."""
    r = np.asarray(design.station_r_m, dtype=float)
    c = np.asarray(design.chord_m, dtype=float)
    t = np.asarray(design.twist_rad, dtype=float)
    if r[0] > design.hub_radius_m:
        r = np.concatenate([[design.hub_radius_m], r])
        c, t = np.concatenate([[c[0]], c]), np.concatenate([[t[0]], t])
    if r[-1] < design.radius_m:
        r = np.concatenate([r, [design.radius_m]])
        c, t = np.concatenate([c, [c[-1]]]), np.concatenate([t, [t[-1]]])
    return build_blade_solid(r, c, t, section, n_chord)
