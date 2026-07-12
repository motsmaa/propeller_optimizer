"""Phase 2a structural stand-in: REAL mass from the lofted solid, PLACEHOLDER
strength. Mass = n_blades * blade volume * density (SI), where the volume
comes from the verified-watertight triangulated loft in propopt.geometry.
Safety factor and stress are still the Phase-0 null formulas — Phase 2b
(reduced structural model) replaces them; do not read structural meaning into
them. Geometry failures surface as SolverError and are penalized by the
optimizer, never silently swallowed."""
from __future__ import annotations

from ..contracts import PropellerDesign, StructResult, StructSolver
from ..geometry import DEFAULT_N_CHORD, DEFAULT_SECTION, SectionFn, solid_from_design
from .null import MAT_DENSITY, NullStructSolver


class GeometryMassStructSolver(StructSolver):
    def __init__(self, density_kg_m3: float = MAT_DENSITY,
                 section: SectionFn = DEFAULT_SECTION,
                 n_chord: int = DEFAULT_N_CHORD) -> None:
        self._density = float(density_kg_m3)
        self._section = section
        self._n_chord = n_chord
        self._null = NullStructSolver()  # placeholder SF/stress source

    def evaluate(self, design: PropellerDesign) -> StructResult:
        solid = solid_from_design(design, self._section, self._n_chord)
        mass = design.n_blades * solid.volume_m3 * self._density
        placeholder = self._null.evaluate(design)
        # modal fields pass through the null solver's documented sentinels —
        # this solver has no modal model either
        return StructResult(mass, placeholder.safety_factor, placeholder.max_stress_Pa,
                            placeholder.first_freq_hz, placeholder.resonance_margin)
