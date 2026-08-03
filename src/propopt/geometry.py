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
# A trailing edge may not exceed this fraction of the thinnest station's max
# thickness; beyond it the "wedge" stops being a trailing edge and the section
# stops being an aerofoil. Enforced per candidate (chord-dependent).
TE_MAX_THICKNESS_FRACTION = 0.5

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


def thickened_te_section(section: SectionFn, n_chord: int, te_thickness_norm: float):
    """`section` with a smooth wedge added so the TE reaches `te_thickness_norm`
    TOTAL thickness (chord-normalized). Returns (x, y_upper, y_lower).

    Functional form — a cubic smoothstep in the aft region only:

        u(x)     = clip((x - x_tmax) / (1 - x_tmax), 0, 1)
        S(u)     = 3u^2 - 2u^3                       (Hermite smoothstep)
        y_upper += (te/2) * S(u),  y_lower -= (te/2) * S(u)

    Chosen because it satisfies every requirement exactly and with no free
    constants to tune:
      * S(0) = S'(0) = 0, so the wedge and its slope vanish AT the
        max-thickness station -> the forward section (including the leading
        edge and the max-thickness point and value) is bit-for-bit untouched,
        and the join is C1 with no kink;
      * S(1) = 1, so the TE thickness is EXACTLY the requested value, not
        asymptotically close to it;
      * S'(1) = 0, so the surface meets the blunt TE face tangentially rather
        than as a corner, which is what keeps the aft facets well-shaped.

    `x_tmax` is taken as the SAMPLED argmax of the base thickness (the seam
    only exposes samples, not an analytic section), so "unchanged forward of
    max thickness" holds exactly on the sampled points the loft actually uses.
    """
    x, yu, yl = section(n_chord)
    if te_thickness_norm <= 0.0:
        return x, yu, yl
    x_tmax = float(x[int(np.argmax(yu - yl))])
    denom = 1.0 - x_tmax
    if denom <= 0.0:                       # max thickness at the TE: no aft region
        raise SolverError("section max thickness is at the trailing edge; "
                          "cannot add a TE wedge")
    u = np.clip((x - x_tmax) / denom, 0.0, 1.0)
    half = 0.5 * te_thickness_norm * (3.0 * u ** 2 - 2.0 * u ** 3)
    return x, yu + half, yl - half


def section_polygon(section: SectionFn, n_chord: int,
                    te_thickness_norm: float = 0.0) -> np.ndarray:
    """Closed simple polygon (P, 2), CCW: LE -> lower surface -> TE -> upper
    surface -> back to LE.

    SHARP TE (te_thickness_norm == 0, the default): LE and TE each appear
    exactly once, P = 2*n_chord - 2. This path is byte-for-byte the original
    Phase-2a code so a zero TE thickness is inert by construction.

    BLUNT TE (te_thickness_norm > 0): the TE becomes a short flat FACE, so the
    upper and lower trailing points are distinct and P = 2*n_chord - 1. The
    polygon stays simple and closed; only the vertex count and the aft
    connectivity change (see `_cap_triangles`).
    """
    if te_thickness_norm <= 0.0:
        x, yu, yl = section(n_chord)
        k = len(x)
        px = np.concatenate([[x[0]], x[1:k - 1], [x[-1]], x[k - 2:0:-1]])
        py = np.concatenate([[yl[0]], yl[1:k - 1], [yl[-1]], yu[k - 2:0:-1]])
        return np.stack([px, py], axis=1)

    x, yu, yl = thickened_te_section(section, n_chord, te_thickness_norm)
    k = len(x)
    # ... LE, lower surface, TE-lower, TE-upper, upper surface, back to LE
    px = np.concatenate([[x[0]], x[1:k - 1], [x[-1]], [x[-1]], x[k - 2:0:-1]])
    py = np.concatenate([[yl[0]], yl[1:k - 1], [yl[-1]], [yu[-1]], yu[k - 2:0:-1]])
    return np.stack([px, py], axis=1)


def section_thickness(section: SectionFn, n_chord: int,
                      te_thickness_norm: float = 0.0) -> np.ndarray:
    """Local thickness of the section, read off the SAME polygon the loft uses.

    Returns (n_chord, 2): chord-normalized chordwise position and local
    thickness (upper minus lower at that station), both in units of chord. The
    leading-edge entry is identically zero (the surfaces meet there); the
    trailing-edge entry is exactly `te_thickness_norm`.

    This is the one definition of "how thick is the blade here". The root
    fillet's geometric bound (`hf.hub_fillet`) and the beam's root r/t
    (`solvers.beam`) both read it, so neither can drift from the lofted solid
    or from each other when the section, the chordwise sampling, or the
    trailing-edge thickness changes.
    """
    poly = section_polygon(section, n_chord, te_thickness_norm)
    k = n_chord
    p = len(poly)                                    # 2k-2 sharp, 2k-1 blunt
    blunt = p == 2 * k - 1
    x, y = poly[:, 0], poly[:, 1]
    # polygon layout: LE, lower surface, TE-lower, [TE-upper,] upper reversed.
    # A sharp TE shares one vertex between the two surfaces; a blunt TE has two.
    lower = np.concatenate([[0], np.arange(1, k - 1), [k - 1]])
    upper = np.concatenate([[0], p - np.arange(1, k - 1), [k if blunt else k - 1]])
    return np.column_stack([x[lower], y[upper] - y[lower]])


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


def _cap_triangles(k: int, ring: np.ndarray, top: bool,
                   blunt_te: bool = False) -> list[tuple[int, int, int]]:
    """Triangulate one section polygon by zipping the (x-monotone) upper and
    lower vertex rows. `ring` maps polygon position -> global vertex id.
    top=True winds outward +z (tip cap), top=False outward -z (root cap).

    Sharp TE: both rows END at the same vertex (k-1), so the final column is a
    triangle and one degenerate is dropped. Blunt TE: the rows end at DISTINCT
    vertices (k-1 lower, k upper) and the final column is a genuine quad — the
    only connectivity change, and the reason a blunt cap has one more triangle
    per section than a sharp one.
    """
    if blunt_te:
        p = 2 * k - 1
        low = [0] + list(range(1, k - 1)) + [k - 1]              # ..., TE lower
        up = [0] + [p - j for j in range(1, k - 1)] + [k]        # ..., TE upper
    else:
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
    te_thickness_m: float = 0.0,
) -> BladeSolid:
    """Loft `section` through the stations -> verified watertight solid.

    `te_thickness_m` is an ABSOLUTE trailing-edge thickness in metres, not a
    percentage of chord: the floor it represents is a manufacturing limit
    (printed PLA cannot hold an edge thinner than roughly a nozzle width), and
    a percentage would leave the small-chord tip stations knife-edged — exactly
    where the mesh sliver problem is worst. Because it is absolute, the
    CHORD-NORMALIZED trailing thickness differs per station, so each station
    gets its own section polygon; with te_thickness_m == 0 the original single
    shared polygon is used unchanged.

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

    te = float(te_thickness_m)
    if not np.isfinite(te) or te < 0.0:
        raise SolverError(f"te_thickness_m must be finite and >= 0, got {te!r}")
    blunt = te > 0.0
    if blunt:
        # A trailing edge thicker than the section itself is nonsense. Checked
        # here (not at config construction) because it depends on the CHORD of
        # this candidate: a config-level value can be valid for one design and
        # impossible for a thinner one, so it is a per-candidate feasibility
        # outcome -> SolverError, which the optimizer penalizes.
        x0, yu0, yl0 = section(n_chord)
        t_max_norm = float(np.max(yu0 - yl0))
        limit = TE_MAX_THICKNESS_FRACTION * t_max_norm * float(c.min())
        if te >= limit:
            raise SolverError(
                f"te_thickness_m {te:.4e} m exceeds {TE_MAX_THICKNESS_FRACTION:.0%} "
                f"of the thinnest station's max thickness ({limit:.4e} m)")

    n_st = len(r)
    if blunt:
        # absolute TE thickness -> per-station normalized thickness te/chord
        polys = np.stack([section_polygon(section, n_chord, te / ci) for ci in c])
        p = polys.shape[1]
        xi = (polys[:, :, 0] - PITCH_AXIS_CHORD_FRAC) * c[:, None]
        eta = polys[:, :, 1] * c[:, None]
    else:
        poly = section_polygon(section, n_chord)      # (P, 2) unit chord, CCW
        p = len(poly)
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
    faces += _cap_triangles(n_chord, ring(0), top=False, blunt_te=blunt)
    n_root = len(faces) - n_skin
    faces += _cap_triangles(n_chord, ring(n_st - 1), top=True, blunt_te=blunt)
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
                      n_chord: int = DEFAULT_N_CHORD,
                      te_thickness_m: float = 0.0) -> BladeSolid:
    """Build one blade solid from the design's shared stations.

    Lofts through `design.station_r_m`, EXTENDED with end-held sections
    (chord/twist frozen at the end-station values) to hub_radius_m and
    radius_m. The solid therefore covers exactly the interval [hub, R] that
    the aero quadrature integrates (stations own midpoint-bounded cells
    clipped to [hub, R]) — domain consistency between aero and geometry.
    End-holding matches the quadrature's treatment of the outer half-cells
    (station properties extended over their whole cell)."""
    r, c, t = extended_stations(design)
    return build_blade_solid(r, c, t, section, n_chord, te_thickness_m)


def extended_stations(design: PropellerDesign):
    """(radii, chord, twist) after end-extension to [hub_radius_m, radius_m].

    The ONE definition of where a blade's sections sit — the triangle loft and
    the Phase-4 OCC construction both read it, so an HF solid cannot silently
    drift from the geometry the optimizer scored."""
    r = np.asarray(design.station_r_m, dtype=float)
    c = np.asarray(design.chord_m, dtype=float)
    t = np.asarray(design.twist_rad, dtype=float)
    if r[0] > design.hub_radius_m:
        r = np.concatenate([[design.hub_radius_m], r])
        c, t = np.concatenate([[c[0]], c]), np.concatenate([[t[0]], t])
    if r[-1] < design.radius_m:
        r = np.concatenate([r, [design.radius_m]])
        c, t = np.concatenate([c, [c[-1]]]), np.concatenate([t, [t[-1]]])
    return r, c, t
