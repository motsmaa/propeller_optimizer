"""CalculiX static structural validator: write INP -> run ccx -> parse .dat.

=============================================================================
UNITS: STRICT SI EVERYWHERE IN THE DECK — metres, kilograms, seconds, newtons,
pascals, kg/m^3, rad/s. CalculiX is unit-agnostic (it just multiplies the
numbers it is given), so mixing mm/MPa is the classic silent failure mode: it
runs happily and returns stresses wrong by 10^6. Node coordinates come from the
loft/STL in metres and are NEVER rescaled here; E and rho come straight off the
shared `Material`.
=============================================================================

This is a VALIDATOR, not an inner-loop solver. It implements StructSolver so it
can be swapped in for a like-for-like comparison against the reduced beam
model, but it must never be wired into optimize.py — a run costs seconds to
minutes, versus ~20 ms for a whole reduced evaluation.

Boundary condition mirrors the beam's rigid-hub clamp: every node on the
root-end face (minimum spanwise coordinate) is fixed in all 3 DOF. That makes
the comparison like-for-like, and it is also the reason the resolved stress
concentration here is PARTIAL — see the Phase-4a report: with no fillet and no
hub body, a fully clamped flat face is itself a stress-concentration artifact.

Loads (both optional, superposable in one static step):
  * centrifugal — CalculiX's own `*DLOAD ... CENTRIF` body load, given omega^2
    and the rotation axis, so the body force is exact and not something this
    module discretizes.
  * surface tractions — nodal `*CLOAD` distributed over boundary nodes by
    tributary area (see mesh.nodal_areas). Used for the per-station aero band
    loads; that mapping is an approximation, and the report says so.

modal (first_freq_hz / resonance_margin) is OUT OF SCOPE for this slice; the
StructResult fields are required by the frozen contract, so they carry the same
kind of unmistakable sentinels null.py uses, documented below.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..contracts import PropellerDesign, SolverError, StructResult, StructSolver
from ..geometry import DEFAULT_N_CHORD, DEFAULT_SECTION, SectionFn, solid_from_design
from ..solvers.beam import PLA, Material
from .mesh import TetMesh, boundary_triangles, nodal_areas
from .wsl import Runner, WSLError, resolve_runner

# Modal is out of scope this slice. These are the same "absurd on inspection"
# sentinel discipline as solvers/null.py: a GHz first mode and a millionfold
# resonance margin cannot be mistaken for physics and cannot silently satisfy
# a resonance constraint. THEY ARE NOT REAL VALUES.
SENTINEL_F1_HZ = 1.0e9
SENTINEL_MARGIN = 1.0e6

_STEP_RE = re.compile(r"^\s*S T E P", re.M)


@dataclass
class CcxResult:
    """Parsed ccx output (SI)."""
    node_ids: np.ndarray             # 1-based ccx node ids, as printed
    displacement_m: np.ndarray       # (N, 3)
    elem_ids: np.ndarray             # 1-based element ids
    stress_Pa: np.ndarray            # (M, 6) sxx syy szz sxy sxz syz, elem-averaged
    stdout: str = field(repr=False, default="")

    @property
    def von_mises_Pa(self) -> np.ndarray:
        sxx, syy, szz, sxy, sxz, syz = self.stress_Pa.T
        return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                       + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2))

    @property
    def max_displacement_m(self) -> float:
        return float(np.linalg.norm(self.displacement_m, axis=1).max())


# --------------------------------------------------------------------------- #
# deck writing
# --------------------------------------------------------------------------- #

def write_inp(
    path: str | Path,
    mesh: TetMesh,
    material: Material,
    *,
    fixed_nodes: np.ndarray,
    omega_rad_s: float = 0.0,
    rotation_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    rotation_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
    nodal_forces_N: np.ndarray | None = None,
    poisson: float | None = None,
) -> Path:
    """Write a single-step linear-static INP. All quantities SI (see module doc).

    `fixed_nodes` / `nodal_forces_N` are indexed 0-based into `mesh.nodes`;
    ccx ids are those + 1. `nodal_forces_N` is (N, 3) in newtons. Poisson's
    ratio comes from the shared `Material` (one definition across fidelities);
    `poisson` overrides it only for a deliberate sensitivity study.
    """
    path = Path(path)
    nu = material.poisson if poisson is None else float(poisson)
    if fixed_nodes.size == 0:
        raise SolverError("no nodes selected for the clamp: model would float")
    lines: list[str] = [
        "** propopt Phase-4a CalculiX deck",
        "** UNITS: SI — m, kg, s, N, Pa, kg/m^3, rad/s. Do NOT mix mm/MPa.",
        f"** material: {material.name}",
        "*NODE, NSET=NALL",
    ]
    for i, (x, y, z) in enumerate(mesh.nodes, start=1):
        lines.append(f"{i}, {x:.10e}, {y:.10e}, {z:.10e}")

    lines.append(f"*ELEMENT, TYPE={mesh.ccx_element_type}, ELSET=EALL")
    for e, conn in enumerate(mesh.tets + 1, start=1):
        # ccx tolerates continuation lines; keep <=8 fields per line for C3D10
        head = ", ".join(str(int(v)) for v in conn[:8])
        lines.append(f"{e}, {head}")
        if len(conn) > 8:
            lines.append("      " + ", ".join(str(int(v)) for v in conn[8:]))

    lines += _nset("FIXED", fixed_nodes + 1)
    lines += [
        f"*MATERIAL, NAME=MAT",
        "*ELASTIC",
        f"{material.elastic_modulus_Pa:.6e}, {nu:.6f}",
        "*DENSITY",
        f"{material.density_kg_m3:.6e}",
        "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT",
        "*STEP",
        "*STATIC",
        "*BOUNDARY",
        "FIXED, 1, 3, 0.0",
    ]

    if omega_rad_s != 0.0:
        # CENTRIF magnitude is omega^2 [rad^2/s^2]; then a point on the axis and
        # the axis direction cosines. Body force is applied by ccx itself.
        ax, ay, az = rotation_axis
        px, py, pz = rotation_point
        lines += [
            "*DLOAD",
            f"EALL, CENTRIF, {omega_rad_s ** 2:.10e}, {px:.6e}, {py:.6e}, {pz:.6e}, "
            f"{ax:.6e}, {ay:.6e}, {az:.6e}",
        ]

    if nodal_forces_N is not None:
        mag = np.linalg.norm(nodal_forces_N, axis=1)
        loaded = np.flatnonzero(mag > 0.0)
        if loaded.size:
            lines.append("*CLOAD")
            for n in loaded:
                for dof in range(3):
                    f = float(nodal_forces_N[n, dof])
                    if f != 0.0:
                        lines.append(f"{int(n) + 1}, {dof + 1}, {f:.10e}")

    lines += [
        "*NODE PRINT, NSET=NALL",
        "U",
        "*EL PRINT, ELSET=EALL",
        "S",
        "*END STEP",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def _nset(name: str, ids: np.ndarray) -> list[str]:
    out = [f"*NSET, NSET={name}"]
    ids = np.asarray(ids, dtype=int).ravel()
    for i in range(0, len(ids), 8):
        out.append(", ".join(str(int(v)) for v in ids[i:i + 8]))
    return out


# --------------------------------------------------------------------------- #
# running + parsing
# --------------------------------------------------------------------------- #

def run_ccx(inp_path: str | Path, runner: Runner | None = None,
            timeout: float = 1800.0) -> CcxResult:
    """Run ccx on `inp_path` and parse its .dat. Raises SolverError on a missing
    binary, non-zero exit, a missing/empty .dat, or a reported non-convergence
    — all of which are feasibility outcomes for a validator, never crashes."""
    inp_path = Path(inp_path)
    job = inp_path.with_suffix("")
    try:
        runner = runner or resolve_runner("ccx", env_var="PROPOPT_CCX")
        cp = runner.run(["-i", job.name], cwd=inp_path.parent, timeout=timeout)
    except WSLError as exc:
        raise SolverError(f"could not run ccx: {exc}") from exc

    out = (cp.stdout or "") + (cp.stderr or "")
    if cp.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-12:])
        raise SolverError(f"ccx exited {cp.returncode}:\n{tail}")
    low = out.lower()
    for bad in ("did not converge", "no convergence", "*error"):
        if bad in low:
            tail = "\n".join(out.strip().splitlines()[-12:])
            raise SolverError(f"ccx reported non-convergence ({bad!r}):\n{tail}")

    # NB: append, never with_suffix — a job name may legitimately contain dots
    # (e.g. "cant_0.02"), and with_suffix would eat the ".02".
    dat = job.parent / (job.name + ".dat")
    if not dat.is_file() or dat.stat().st_size == 0:
        raise SolverError(f"ccx produced no results ({dat.name} missing/empty)")
    res = parse_dat(dat)
    res.stdout = out
    return res


def parse_dat(dat_path: str | Path) -> CcxResult:
    """Parse displacements and (integration-point) stresses from a ccx .dat.

    Integration-point stresses are averaged per element — the honest
    element-level value. Stress at a single node is NOT recovered by averaging
    integration points across neighbouring elements here; peak stress is
    reported as a max over elements, which is slightly non-conservative near a
    singularity but avoids inventing a smoothing scheme.
    """
    text = Path(dat_path).read_text()
    if _STEP_RE.search(text) is None:
        raise SolverError(f"{Path(dat_path).name}: no step results found")

    disp: dict[int, tuple[float, float, float]] = {}
    stress: dict[int, list[tuple[float, ...]]] = {}
    mode = None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("displacements"):
            mode = "u"
            continue
        if low.startswith("stresses"):
            mode = "s"
            continue
        if low.startswith(("s t e p", "increment", "forces", "strains", "energy")):
            mode = None
            continue
        parts = s.split()
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            mode = None
            continue
        if mode == "u" and len(vals) >= 4:
            disp[int(vals[0])] = (vals[1], vals[2], vals[3])
        elif mode == "s" and len(vals) >= 8:
            stress.setdefault(int(vals[0]), []).append(tuple(vals[2:8]))

    if not disp:
        raise SolverError(f"{Path(dat_path).name}: no displacements parsed")
    nids = np.array(sorted(disp), dtype=int)
    u = np.array([disp[i] for i in nids], dtype=float)
    if stress:
        eids = np.array(sorted(stress), dtype=int)
        s = np.array([np.mean(stress[e], axis=0) for e in eids], dtype=float)
    else:
        eids, s = np.empty(0, dtype=int), np.empty((0, 6))
    return CcxResult(nids, u, eids, s)


# --------------------------------------------------------------------------- #
# load mapping helpers
# --------------------------------------------------------------------------- #

def band_traction_forces(mesh: TetMesh, station_r_m: np.ndarray,
                         band_force_N: np.ndarray, direction: np.ndarray,
                         span_axis: int = 2) -> np.ndarray:
    """Map per-station band forces onto boundary nodes by tributary area.

    Each station owns a spanwise band (bounded by neighbour midpoints); every
    boundary node in that band receives a share of the band's force weighted by
    its tributary area, so the result is a piecewise-uniform surface traction.

    THIS IS AN APPROXIMATION. The reduced model applies each station's load as a
    point resultant on the beam axis; here it is smeared over that band's wetted
    surface. The two are not identical, and the difference is neither model's
    error — it is the mapping's. Stated in the report.
    """
    tris = boundary_triangles(mesh.tets)
    areas = nodal_areas(mesh.nodes, tris)
    span = mesh.nodes[:, span_axis]
    r = np.asarray(station_r_m, dtype=float)
    band_force_N = np.asarray(band_force_N, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if len(r) != len(band_force_N):
        raise SolverError("station radii and band forces have different lengths")

    edges = np.empty(len(r) + 1)
    edges[0], edges[-1] = span.min(), span.max()
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    on_surface = areas > 0.0

    forces = np.zeros((len(mesh.nodes), 3))
    for i, f_total in enumerate(band_force_N):
        lo, hi = edges[i], edges[i + 1]
        sel = on_surface & (span >= lo) & (span <= hi if i == len(r) - 1 else span < hi)
        w = areas[sel]
        if w.sum() <= 0.0:
            continue                              # empty band: nothing wetted
        share = (w / w.sum())[:, None] * f_total
        forces[sel] += share * direction[None, :]
    return forces


def root_face_nodes(mesh: TetMesh, span_axis: int = 2,
                    tol_m: float = 1e-9) -> np.ndarray:
    """0-based indices of nodes on the minimum-span (root) face — the clamp set,
    mirroring the beam's rigid hub."""
    span = mesh.nodes[:, span_axis]
    return np.flatnonzero(span <= span.min() + tol_m)


def elements_in_slab(mesh: TetMesh, lo_m: float, hi_m: float,
                     span_axis: int = 2) -> np.ndarray:
    """0-based element indices whose corner centroid lies in a spanwise slab.
    Used to read stress in a root band while excluding the clamped-face layer
    (where a fully-fixed flat face is itself a numerical concentration)."""
    cz = mesh.nodes[mesh.corners, span_axis].mean(axis=1)
    return np.flatnonzero((cz >= lo_m) & (cz <= hi_m))


# --------------------------------------------------------------------------- #
# the StructSolver
# --------------------------------------------------------------------------- #

class CalculiXStructSolver(StructSolver):
    """High-fidelity static validator behind the same ABC as the beam.

    NOT for the optimizer loop (see module docstring). `mass_kg` is the LOFT
    volume x density — deliberately the same quantity the optimizer scored, not
    the tet-mesh volume, so the mass objective stays consistent across
    fidelities; the mesh-vs-loft volume difference is exposed separately as a
    diagnostic on the returned analysis.
    """

    def __init__(self, material: Material = PLA, *,
                 target_size_m: float = 5e-4, order: int = 2,
                 aero=None, section: SectionFn = DEFAULT_SECTION,
                 n_chord: int = DEFAULT_N_CHORD, te_thickness_m: float = 0.0,
                 work_dir: str | Path | None = None,
                 ccx_binary: str | None = None, keep_files: bool = True) -> None:
        self._mat = material
        self._target = float(target_size_m)
        self._order = int(order)
        self._aero = aero
        self._section = section
        self._n_chord = n_chord
        self._te = float(te_thickness_m)   # Poisson comes from `material`
        self._work = Path(work_dir) if work_dir else Path("runs/hf_ccx")
        self._runner = resolve_runner("ccx", binary=ccx_binary, env_var="PROPOPT_CCX")
        self._keep = keep_files

    def evaluate(self, design: PropellerDesign) -> StructResult:
        an = self.analyze(design)
        return StructResult(an["mass_kg"], an["safety_factor"], an["max_stress_Pa"],
                            SENTINEL_F1_HZ, SENTINEL_MARGIN)

    def analyze(self, design: PropellerDesign, *, with_aero: bool = True,
                stl_path: str | Path | None = None,
                mesh: TetMesh | None = None, tag: str = "blade") -> dict:
        """Run one static case and return the full diagnostic dict (superset of
        StructResult, which the frozen contract cannot carry)."""
        from .mesh import mesh_stl
        from ..export import _write_ascii_stl

        work = self._work / tag
        work.mkdir(parents=True, exist_ok=True)
        solid = solid_from_design(design, self._section, self._n_chord, self._te)

        if mesh is None:
            if stl_path is None:
                stl_path = work / f"{tag}.stl"
                _write_ascii_stl(stl_path, solid.vertices, solid.faces, tag)
            mesh = mesh_stl(stl_path, self._target, order=self._order,
                            reference_volume_m3=solid.volume_m3,
                            max_volume_drift=1.0)

        omega = design.rpm * 2.0 * np.pi / 60.0
        forces = None
        if with_aero:
            if self._aero is None:
                raise SolverError("with_aero=True requires an aero solver")
            p_flap, p_edge = self._aero.spanwise_loads(design)
            from ..stations import station_weights
            w = station_weights(design.station_r_m, design.hub_radius_m,
                                design.radius_m)
            r = np.asarray(design.station_r_m, dtype=float)
            # geometry frame: x tangential, y axial (thrust), z radial (span)
            forces = (band_traction_forces(mesh, r, p_flap * w, np.array([0.0, 1.0, 0.0]))
                      + band_traction_forces(mesh, r, p_edge * w, np.array([1.0, 0.0, 0.0])))

        inp = write_inp(work / f"{tag}.inp", mesh, self._mat,
                        fixed_nodes=root_face_nodes(mesh), omega_rad_s=omega,
                        nodal_forces_N=forces)
        res = run_ccx(inp, self._runner)

        vm = res.von_mises_Pa
        if vm.size == 0:
            raise SolverError("ccx returned no element stresses")
        peak = float(vm.max())
        if not np.isfinite(peak) or peak <= 0.0:
            raise SolverError(f"non-physical peak stress {peak!r}")
        mass = design.n_blades * solid.volume_m3 * self._mat.density_kg_m3
        if not self._keep:
            shutil.rmtree(work, ignore_errors=True)
        return {
            "mass_kg": mass,
            "safety_factor": self._mat.yield_strength_Pa / peak,
            "max_stress_Pa": peak,
            "loft_volume_m3": solid.volume_m3,
            "mesh_volume_m3": mesh.volume_m3,
            "mesh_volume_rel_error": mesh.volume_rel_error,
            "n_tets": mesh.n_tets, "n_nodes": mesh.n_nodes,
            "quality_min": mesh.quality_min, "quality_mean": mesh.quality_mean,
            "result": res, "mesh": mesh,
        }
