"""Phase 3 export: front -> JSON, selected designs -> STL.

JSON (not CSV) because each Pareto record is heterogeneous and nested — a
variable-length decision vector plus scalar objectives plus a constraint
name->value map plus an activity map. CSV would force a flat, brittle schema
with per-constraint columns; JSON keeps the record self-describing and carries
the config echo + convergence + activity summary in one file that re-reads
without a schema. numpy-only (json is stdlib).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .analysis import DesignRecord
from .config import RunConfig
from .geometry import solid_from_design


def _config_echo(cfg: RunConfig) -> dict:
    """A reproducibility record: config + seed reproduce the run exactly."""
    return {
        "n_stations": cfg.n_stations, "radius_m": cfg.radius_m,
        "hub_radius_m": cfg.hub_radius_m, "n_blades": cfg.n_blades,
        "chord_bounds": list(cfg.chord_bounds), "twist_bounds": list(cfg.twist_bounds),
        "throttle_bounds": list(cfg.throttle_bounds),
        "motor": {"name": cfg.motor.name, "kv": cfg.motor.kv_rpm_per_v,
                  "rm_ohm": cfg.motor.rm_ohm, "i0_a": cfg.motor.i0_a,
                  "v_max_v": cfg.motor.v_max_v, "i_max_a": cfg.motor.i_max_a}
        if cfg.motor is not None else None,
        "material": {"name": cfg.material.name,
                     "E_Pa": cfg.material.elastic_modulus_Pa,
                     "yield_Pa": cfg.material.yield_strength_Pa,
                     "rho": cfg.material.density_kg_m3},
        "kt_root": cfg.kt_root, "min_thrust_N": cfg.min_thrust_N,
        "resonance_margin_min": cfg.resonance_margin_min,
        "pop_size": cfg.pop_size, "n_gen": cfg.n_gen, "seed": cfg.seed,
        "binding_tol_frac": cfg.binding_tol_frac,
        # grade provenance: which mode/seeds produced this output
        "mode": cfg.mode,
        "seeds": list(cfg.seeds) if cfg.mode == "decision" else [cfg.seed],
    }


def export_front_json(path: str | Path, cfg: RunConfig,
                      records: list[DesignRecord], summary: dict,
                      convergence: dict | None = None) -> Path:
    """Write the full front + activity summary + config echo to JSON."""
    path = Path(path)
    payload = {
        "config": _config_echo(cfg),
        "constraint_activity": summary,
        "convergence": convergence,   # optional seed-variance study
        "designs": [asdict(r) for r in records],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def select_representative(records: list[DesignRecord]) -> dict[str, int]:
    """Pick designs worth exporting as solids: the single-objective champion of
    each objective (the four corners of the trade-off). Explicit and
    reproducible; these are the designs a reader most wants to inspect."""
    if not records:
        return {}
    idx = {
        "max_thrust": int(np.argmax([r.objectives["thrust_N"] for r in records])),
        "max_fom": int(np.argmax([r.objectives["figure_of_merit"] for r in records])),
        "min_mass": int(np.argmin([r.objectives["mass_kg"] for r in records])),
        "max_safety": int(np.argmax([r.objectives["safety_factor"] for r in records])),
    }
    return idx


def _write_ascii_stl(path: Path, vertices: np.ndarray, faces: np.ndarray, name: str) -> None:
    tris = vertices[faces]                        # (F, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)
    lines = [f"solid {name}"]
    for (a, b, c), (nx, ny, nz) in zip(tris, n):
        lines.append(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}")
        lines.append("    outer loop")
        for v in (a, b, c):
            lines.append(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines))


def export_selected_stl(out_dir: str | Path, cfg: RunConfig,
                        records: list[DesignRecord]) -> list[Path]:
    """Export the representative designs' watertight blade solids as ASCII STL."""
    from .config import build_run
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec, _, _ = build_run(cfg)
    written = []
    for label, i in select_representative(records).items():
        design = spec.decode(np.asarray(records[i].x, dtype=float))
        solid = solid_from_design(design)
        p = out_dir / f"blade_{label}.stl"
        _write_ascii_stl(p, solid.vertices, solid.faces, label)
        written.append(p)
    return written
