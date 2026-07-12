"""Phase 2b: reduced structural solver — rotating-cantilever 1-D beam FE.

The blade is a cantilever clamped at the hub radius, discretized on the
shared stations (design.station_r_m) as Euler-Bernoulli beam elements with
Hermite-cubic shape functions (2 nodes x [deflection, slope]). This is a
SURROGATE: solid/meshed FEA is reserved for Phase-4 validation, not the loop.

Physics included:
- Centrifugal tension N(r) from the integrated mass outboard of each radius —
  it stresses the blade axially AND stiffens flapwise bending (geometric /
  stress-stiffening matrix in the modal problem). Spin softening (-Omega^2 M)
  applies to IN-PLANE (lead-lag) motion, not to flapwise deflection (parallel
  to the rotation axis), and the modal solve here is flapwise-only — so it is
  correctly absent, not omitted (documented modelling choice).
- Static bending from the BEMT per-blade spanwise loads (flapwise = thrust
  direction, edgewise = tangential), moments by exact cantilever statics on
  the station-cell quadrature; plus centrifugal axial stress.
- Static strength: max extreme-fiber stress (axial + conservatively
  superposed corner bending, moments rotated by local twist into the section
  frame; products of inertia neglected — first-pass) -> von Mises of a
  uniaxial state is |sigma|; SF = yield / sigma_peak.
- Modal: first flapwise natural frequency from the generalized eigenproblem
  (K_elastic + K_geometric) phi = omega^2 M phi via Cholesky + numpy eigh,
  plus a resonance margin against blade-passing harmonics n*B*rpm/60.

SCOPE LIMITS of the resonance model (documented, not solved here): only
blade-passing harmonics n=1..n_harmonics are checked against f1. There is NO
1P/shaft-order excitation term (relevant to rotor imbalance, not modelled),
NO forward-flight forcing (this solver is static/hover, consistent with
BEMT), and NO damping or forced-response amplitude estimate — the margin is a
frequency keep-out band, not a vibration-amplitude prediction. A proper
Campbell diagram (multiple mode shapes vs. rpm, all excitation orders) is
Phase-4 material, once solid FEA is in the loop.

Root stress concentration (REQUIRED GUARD): the 1-D model is blind to the
3-D root/hub junction, so the root-node bending stress is multiplied by a
documented factor KT_ROOT (default 2.5 — Peterson-type charts put unfilleted
or small-fillet rectangular-to-hub junctions at Kt ~ 1.5-3; we take the
conservative middle-high end). This is a CONSERVATIVE PLACEHOLDER to be
calibrated against Phase-4 solid FEA — the optimizer must not be allowed to
exploit a root the surrogate cannot see.

Element properties are span-averages of the linearly-interpolated chord that
the geometry loft uses (area ~ c^2 averaged exactly, EI ~ c^4 averaged
exactly), so the beam's integrated mass matches the lofted solid volume
exactly for untwisted blades (twist perturbs the discrete loft volume only
at second order). Section constants come from the same section polygon as
the geometry module — one section seam, three consumers.

Material is an explicit parameter (dataclass), not a hidden constant. The
default PLA uses typical filament-datasheet values; printed PLA is
ANISOTROPIC (layer adhesion can halve transverse strength) and the isotropic
von Mises treatment here is a documented first-pass simplification — Kt and
the yield value must be read with that caveat.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import PropellerDesign, SolverError, StructResult, StructSolver
from ..geometry import DEFAULT_N_CHORD, DEFAULT_SECTION, SectionFn, section_polygon, solid_from_design
from ..stations import station_weights
from .bemt import BEMTAeroSolver
from .null import MAT_DENSITY


@dataclass(frozen=True)
class Material:
    """Isotropic material (SI). Isotropy is a first-pass approximation for
    printed parts — see module docstring."""
    name: str
    density_kg_m3: float
    elastic_modulus_Pa: float
    yield_strength_Pa: float

    def __post_init__(self) -> None:
        for f in ("density_kg_m3", "elastic_modulus_Pa", "yield_strength_Pa"):
            v = getattr(self, f)
            if not (np.isfinite(v) and v > 0.0):
                raise ValueError(f"Material.{f} must be positive and finite, got {v!r}")


# Typical PLA filament datasheet values (e.g. Prusament/Ultimaker PLA TDS:
# E ~ 2.9-3.6 GPa, tensile strength ~ 45-60 MPa). Density matches the
# geometry/mass constant MAT_DENSITY so mass stays consistent across solvers.
PLA = Material("PLA (printed; isotropic first-pass)", MAT_DENSITY, 3.5e9, 50.0e6)

KT_ROOT_DEFAULT = 2.5  # root stress-concentration guard, see module docstring


@dataclass(frozen=True)
class UnitSection:
    """Section constants for the unit-chord polygon; scale by chord c as
    area*c^2, i_**c^4, d_**c."""
    area: float
    i_chord: float   # about the centroidal chordwise axis (flapwise bending)
    i_thick: float   # about the centroidal thickness axis (edgewise bending)
    d_thick: float   # max fiber distance, thickness direction
    d_chord: float   # max fiber distance, chordwise direction


def unit_section_properties(section: SectionFn = DEFAULT_SECTION,
                            n_chord: int = DEFAULT_N_CHORD) -> UnitSection:
    """Exact polygon (Green's theorem) area / centroid / second moments of the
    same section polygon the geometry loft uses."""
    poly = section_polygon(section, n_chord)
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = 0.5 * float(np.sum(cross))
    cx = float(np.sum((x + x1) * cross)) / (6.0 * area)
    cy = float(np.sum((y + y1) * cross)) / (6.0 * area)
    ixx = float(np.sum((y * y + y * y1 + y1 * y1) * cross)) / 12.0 - area * cy ** 2
    iyy = float(np.sum((x * x + x * x1 + x1 * x1) * cross)) / 12.0 - area * cx ** 2
    return UnitSection(area, ixx, iyy,
                       float(np.max(np.abs(y - cy))), float(np.max(np.abs(x - cx))))


# ----- beam FE core (flapwise, clamped at node 0) ---------------------------

def _element_matrices(length: float, ei: float, rho_a: float, tension: float):
    l_ = length
    k_e = ei / l_ ** 3 * np.array([
        [12.0, 6 * l_, -12.0, 6 * l_],
        [6 * l_, 4 * l_ ** 2, -6 * l_, 2 * l_ ** 2],
        [-12.0, -6 * l_, 12.0, -6 * l_],
        [6 * l_, 2 * l_ ** 2, -6 * l_, 4 * l_ ** 2]])
    k_g = tension / (30.0 * l_) * np.array([
        [36.0, 3 * l_, -36.0, 3 * l_],
        [3 * l_, 4 * l_ ** 2, -3 * l_, -l_ ** 2],
        [-36.0, -3 * l_, 36.0, -3 * l_],
        [3 * l_, -l_ ** 2, -3 * l_, 4 * l_ ** 2]])
    m_e = rho_a * l_ / 420.0 * np.array([
        [156.0, 22 * l_, 54.0, -13 * l_],
        [22 * l_, 4 * l_ ** 2, 13 * l_, -3 * l_ ** 2],
        [54.0, 13 * l_, 156.0, -22 * l_],
        [-13 * l_, -3 * l_ ** 2, -22 * l_, 4 * l_ ** 2]])
    return k_e, k_g, m_e


def flapwise_system(r_nodes: np.ndarray, ei_elem: np.ndarray,
                    rho_a_elem: np.ndarray, tension_elem: np.ndarray):
    """Assemble the clamped-root flapwise system. Returns (K, M) with the
    geometric (stress-stiffening) matrix already added to K and the two
    root DOFs (deflection, slope at node 0) eliminated."""
    n_nodes = len(r_nodes)
    ndof = 2 * n_nodes
    k = np.zeros((ndof, ndof))
    m = np.zeros((ndof, ndof))
    for e in range(n_nodes - 1):
        length = float(r_nodes[e + 1] - r_nodes[e])
        k_e, k_g, m_e = _element_matrices(length, float(ei_elem[e]),
                                          float(rho_a_elem[e]), float(tension_elem[e]))
        sl = slice(2 * e, 2 * e + 4)
        k[sl, sl] += k_e + k_g
        m[sl, sl] += m_e
    return k[2:, 2:], m[2:, 2:]


def distributed_load_vector(r_nodes: np.ndarray, p_elem: np.ndarray) -> np.ndarray:
    """Consistent nodal load vector for element-wise uniform transverse load
    p_elem (N/m), root DOFs eliminated. With Hermite elements this makes the
    static FE solution nodally EXACT for a uniform beam under uniform load."""
    ndof = 2 * len(r_nodes)
    f = np.zeros(ndof)
    for e in range(len(r_nodes) - 1):
        l_ = float(r_nodes[e + 1] - r_nodes[e])
        f[2 * e: 2 * e + 4] += p_elem[e] * l_ * np.array([0.5, l_ / 12.0, 0.5, -l_ / 12.0])
    return f[2:]


def centrifugal_tension(r_nodes: np.ndarray, rho_a_elem: np.ndarray,
                        omega: float) -> np.ndarray:
    """Axial tension at each node from the mass outboard of it:
    N(r) = integral_r^R rho*A(xi)*omega^2*xi dxi, element-wise exact for the
    element-average rho*A (midpoint rule is exact for the linear integrand)."""
    mid = 0.5 * (r_nodes[:-1] + r_nodes[1:])
    seg = rho_a_elem * omega ** 2 * mid * np.diff(r_nodes)
    n = np.zeros(len(r_nodes))
    n[:-1] = np.cumsum(seg[::-1])[::-1]
    return n


def first_frequency_hz(k: np.ndarray, m: np.ndarray) -> float:
    """Smallest eigenfrequency of K phi = omega^2 M phi (Cholesky + eigh)."""
    try:
        chol = np.linalg.cholesky(m)
        a = np.linalg.solve(chol, np.linalg.solve(chol, k).T).T
        lam = np.linalg.eigvalsh(0.5 * (a + a.T))
    except np.linalg.LinAlgError as exc:
        raise SolverError(f"beam eigenproblem failed: {exc}") from exc
    lam1 = float(lam[0])
    if not np.isfinite(lam1) or lam1 <= 0.0:
        raise SolverError(f"non-physical first eigenvalue {lam1!r}")
    return float(np.sqrt(lam1) / (2.0 * np.pi))


# ----- the solver ------------------------------------------------------------

@dataclass(frozen=True)
class BeamAnalysis:
    """Full Phase-2b structural output (StructResult carries only the frozen
    contract triple; the rest is reported through this object)."""
    mass_kg: float             # from the lofted solid volume (all blades)
    beam_mass_kg: float        # beam quadrature mass (all blades), consistency check
    safety_factor: float       # yield / peak stress (Kt included at root)
    max_stress_Pa: float
    f1_hz: float               # first flapwise natural frequency at design rpm
    excitation_hz: tuple[float, ...]  # blade-passing harmonics n*B*rpm/60
    resonance_margin: float    # min_n |f1 - f_n| / f_n
    tip_deflection_m: float


class BeamStructSolver(StructSolver):
    """See module docstring. `aero` must provide spanwise_loads(design)
    (BEMTAeroSolver does); it is re-run internally at the design rpm, so the
    aero cost per evaluation roughly doubles — accepted for Phase 2b."""

    def __init__(self,
                 material: Material = PLA,
                 aero: BEMTAeroSolver | None = None,
                 kt_root: float = KT_ROOT_DEFAULT,
                 section: SectionFn = DEFAULT_SECTION,
                 n_chord: int = DEFAULT_N_CHORD,
                 n_harmonics: int = 3) -> None:
        if kt_root < 1.0:
            raise ValueError(f"kt_root must be >= 1, got {kt_root}")
        self._mat = material
        self._aero = aero if aero is not None else BEMTAeroSolver()
        self._kt = float(kt_root)
        self._section = section
        self._n_chord = n_chord
        self._n_harm = int(n_harmonics)
        self._unit = unit_section_properties(section, n_chord)

    # -- helpers --------------------------------------------------------------

    def _nodes(self, design: PropellerDesign):
        """Node radii [hub, stations..., R] (deduped ends) with end-held chord
        and twist — mirrors the geometry loft's end-extension exactly."""
        r = np.asarray(design.station_r_m, dtype=float)
        c = np.asarray(design.chord_m, dtype=float)
        t = np.asarray(design.twist_rad, dtype=float)
        if r[0] > design.hub_radius_m:
            r = np.concatenate([[design.hub_radius_m], r])
            c, t = np.concatenate([[c[0]], c]), np.concatenate([[t[0]], t])
        if r[-1] < design.radius_m:
            r = np.concatenate([r, [design.radius_m]])
            c, t = np.concatenate([c, [c[-1]]]), np.concatenate([t, [t[-1]]])
        if len(r) < 2:
            raise SolverError("beam needs a nonzero span (hub == station == tip?)")
        return r, c, t

    def _element_props(self, c_nodes: np.ndarray):
        """Span-exact element averages of the linearly-interpolated chord:
        mean(c^2) = (ci^2 + ci*cj + cj^2)/3 (matches the loft volume exactly),
        mean(c^4) = (ci^4 + ci^3 cj + ci^2 cj^2 + ci cj^3 + cj^4)/5."""
        ci, cj = c_nodes[:-1], c_nodes[1:]
        c2 = (ci ** 2 + ci * cj + cj ** 2) / 3.0
        c4 = (ci ** 4 + ci ** 3 * cj + ci ** 2 * cj ** 2 + ci * cj ** 3 + cj ** 4) / 5.0
        area_e = self._unit.area * c2
        ei_e = self._mat.elastic_modulus_Pa * self._unit.i_chord * c4
        return area_e, ei_e

    # -- main -----------------------------------------------------------------

    def analyze(self, design: PropellerDesign) -> BeamAnalysis:
        mat, unit = self._mat, self._unit
        omega = design.rpm * 2.0 * np.pi / 60.0
        p_flap, p_edge = self._aero.spanwise_loads(design)   # per blade, N/m
        w_cell = station_weights(design.station_r_m,
                                 design.hub_radius_m, design.radius_m)
        r_st = np.asarray(design.station_r_m, dtype=float)

        r_n, c_n, t_n = self._nodes(design)
        area_e, ei_e = self._element_props(c_n)
        rho_a_e = mat.density_kg_m3 * area_e
        lengths = np.diff(r_n)
        beam_mass = design.n_blades * float(np.sum(rho_a_e * lengths))
        solid = solid_from_design(design, self._section, self._n_chord)
        mass = design.n_blades * solid.volume_m3 * mat.density_kg_m3

        # centrifugal tension at nodes; element tension = end average
        n_node = centrifugal_tension(r_n, rho_a_e, omega)
        n_elem = 0.5 * (n_node[:-1] + n_node[1:])

        # -- static stress by exact cantilever statics on the station cells --
        # evaluation points: every node except the free tip
        r_eval, c_eval, t_eval = r_n[:-1], c_n[:-1], t_n[:-1]
        arm = np.maximum(r_st[None, :] - r_eval[:, None], 0.0)
        m_flap = arm @ (p_flap * w_cell)          # N*m, flapwise bending
        m_edge = arm @ (p_edge * w_cell)          # N*m, edgewise bending
        # rotate the moment vector by local twist into the section frame
        cos_t, sin_t = np.cos(t_eval), np.sin(t_eval)
        m_1 = m_flap * cos_t + m_edge * sin_t     # bends in thickness direction
        m_2 = -m_flap * sin_t + m_edge * cos_t    # bends in chordwise direction
        i_c = unit.i_chord * c_eval ** 4
        i_t = unit.i_thick * c_eval ** 4
        area_n = unit.area * c_eval ** 2
        sigma_bend = (np.abs(m_1) * unit.d_thick * c_eval / i_c
                      + np.abs(m_2) * unit.d_chord * c_eval / i_t)
        kt = np.ones_like(sigma_bend)
        kt[0] = self._kt                           # root/hub junction guard
        sigma = n_node[:-1] / area_n + kt * sigma_bend
        if not np.all(np.isfinite(sigma)):
            raise SolverError("non-finite beam stress")
        sigma_peak = float(np.max(sigma))
        if sigma_peak <= 0.0:
            raise SolverError(f"non-physical peak stress {sigma_peak!r}")
        safety_factor = mat.yield_strength_Pa / sigma_peak

        # -- static tip deflection (flapwise FE; reported, not an objective) --
        p_node = np.interp(r_n, r_st, p_flap)      # end-held via interp clamping
        p_elem = 0.5 * (p_node[:-1] + p_node[1:])
        k_red, m_red = flapwise_system(r_n, ei_e, rho_a_e, n_elem)
        f_red = distributed_load_vector(r_n, p_elem)
        try:
            defl = np.linalg.solve(k_red, f_red)
        except np.linalg.LinAlgError as exc:
            raise SolverError(f"singular beam stiffness: {exc}") from exc
        tip_deflection = float(defl[-2])
        if not np.isfinite(tip_deflection):
            raise SolverError("non-finite beam deflection")

        # -- modal: first flapwise frequency with centrifugal stiffening --
        f1 = first_frequency_hz(k_red, m_red)
        harmonics = tuple(n * design.n_blades * design.rpm / 60.0
                          for n in range(1, self._n_harm + 1))
        margin = min(abs(f1 - f) / f for f in harmonics)

        return BeamAnalysis(mass, beam_mass, safety_factor, sigma_peak,
                            f1, harmonics, margin, tip_deflection)

    def evaluate(self, design: PropellerDesign) -> StructResult:
        a = self.analyze(design)
        return StructResult(a.mass_kg, a.safety_factor, a.max_stress_Pa,
                            a.f1_hz, a.resonance_margin)
