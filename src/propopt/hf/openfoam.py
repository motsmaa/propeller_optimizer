"""Phase-4b: OpenFOAM case construction, execution and extraction.

A VALIDATOR, NOT AN INNER-LOOP SOLVER. Nothing here is imported by
`optimize.py` and nothing here implements `AeroSolver`. A CFD case costs
minutes; the optimizer evaluates thousands of candidates. This exists to
measure how far BEMT is from a resolved solution, not to replace it.

UNITS: strict SI (m, s, kg, N, Pa, kg/m^3) — stated again in a header comment
inside every case file this module writes, because OpenFOAM is unit-agnostic.

ONE TRAP WORTH STATING TWICE: `simpleFoam` is INCOMPRESSIBLE, so its pressure
field is KINEMATIC (p/rho, m^2/s^2) and its `nu` is kinematic viscosity. Forces
from the `forces` function object are only in newtons because the dictionary
carries `rho rhoInf; rhoInf <value>;` — the same density the BEMT solver uses.
Omitting that silently returns force/rho.

BOUNDARY BETWEEN FIDELITIES: this module builds its geometry from the SAME
`PropellerDesign` and the SAME OCC path (`hf/occ_geometry.station_rings`) that
the structural HF path uses, so the CFD blade and the BEMT blade are the same
design by construction rather than by agreement.

WHERE A CASE RUNS — and why it is not where it is written. OpenFOAM's
`fileName` type REJECTS SPACES: `fileName::stripInvalid()` deletes them and
then treats the result as fatal. This repository lives under
`...\Propeller optimizer\...`, so no OpenFOAM case can execute in the project
tree at all — measured, not anticipated. Cases are therefore written on the
Windows side (the record, version-controllable, where logs come back to) and
COPIED to a space-free directory on WSL's own ext4 filesystem to run.

That is the right place independently of the space problem: `/mnt/c` is a **9p**
mount, and OpenFOAM is a many-small-file workload (one file per field per
processor per write). Running on 9p would make the cost measurement a
measurement of the filesystem. Only small text (the case dictionaries in, the
logs and `postProcessing` out) crosses the boundary; mesh and field data stay
on ext4.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..contracts import SolverError
from .wsl import (Runner, WSLError, _quote, find_openfoam_bashrc,
                  openfoam_runner, to_wsl_path)

FOAM_BANNER = r"""/*--------------------------------*- C++ -*----------------------------------*\
| propopt Phase-4b generated case file. DO NOT EDIT BY HAND.                   |
| UNITS: STRICT SI - metres, seconds, kilograms, newtons, Pa, kg/m^3.          |
| NOTE: simpleFoam is INCOMPRESSIBLE. p is KINEMATIC pressure (p/rho, m^2/s^2) |
|       and nu is KINEMATIC viscosity (m^2/s). Forces are newtons only because |
|       the forces function object is given rhoInf explicitly.                 |
\*---------------------------------------------------------------------------*/
"""


def foam_dict(cls: str, obj: str, body: str, location: str | None = None) -> str:
    """A complete OpenFOAM dictionary file as text."""
    loc = f'    location    "{location}";\n' if location else ""
    return (f"{FOAM_BANNER}FoamFile\n{{\n    version     2.0;\n"
            f"    format      ascii;\n    class       {cls};\n{loc}"
            f"    object      {obj};\n}}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            f"{body}\n\n// ******************************************* //\n")


def write_dict(case_dir: Path, location: str, obj: str, body: str,
               cls: str = "dictionary") -> Path:
    d = Path(case_dir) / location
    d.mkdir(parents=True, exist_ok=True)
    p = d / obj
    p.write_text(foam_dict(cls, obj, body, location), encoding="utf-8", newline="\n")
    return p


# --- running ----------------------------------------------------------------

# Sample the whole process SESSION, not just the launched process: an mpirun of
# N ranks holds N x RSS simultaneously, and it is the SUM that has to fit in the
# WSL memory ceiling. GNU `time -v` reports the max over single processes and
# would under-report a parallel run by roughly the rank count.
#
# Matching is on SESSION id, not process group: `setsid` starts a new session
# and MPI launchers put ranks in their own process groups within it, so a
# pgid match silently misses every rank (measured: it reported 14 MB for an
# 8-rank snappyHexMesh). Session id is inherited by the whole tree.
# The running maximum is written to a FILE on every new high-water mark, not
# only echoed at exit: a long mesh or solve that is killed by a timeout would
# otherwise report no memory at all — which is exactly when the number matters
# most (measured: a 50-minute snappyHexMesh died with no RSS recorded).
RSS_FILE = ".propopt_peak_rss_kb"
_RSS_SAMPLER = r"""
_rssfile="$1"; shift
: > "$_rssfile"
setsid "$@" &
_pid=$!
_max=0
while kill -0 $_pid 2>/dev/null; do
  _cur=$(ps -eo rss=,sid= 2>/dev/null | awk -v s="$_pid" '$2==s {t+=$1} END {print t+0}')
  if [ -n "$_cur" ] && [ "$_cur" -gt "$_max" ]; then _max=$_cur; echo "$_max" > "$_rssfile"; fi
  sleep 0.1
done
wait $_pid; _rc=$?
echo "$_max" > "$_rssfile"
echo "PROPOPT_PEAK_RSS_KB=$_max" 1>&2
exit $_rc
"""


FOAM_ROOT_ENV = "PROPOPT_FOAM_ROOT"
_DEFAULT_FOAM_ROOT = "$HOME/.propopt/foam"


def _wsl_sh(snippet: str, cwd: str | os.PathLike = "/") -> subprocess.CompletedProcess:
    """Run a plain shell snippet inside WSL (no OpenFOAM environment needed).
    Goes through the same Runner so the boundary stays one place."""
    argv, _ = Runner("bash", use_wsl=True).shell(snippet, cwd)
    return subprocess.run(argv, capture_output=True, text=True, timeout=1800)


@dataclass(frozen=True)
class FoamCase:
    """A case with two locations: where it is WRITTEN and where it RUNS.

    `src`  — Windows-side directory. The case dictionaries are authored here and
             logs / postProcessing are copied back here. This is the record.
    `work` — POSIX directory on WSL's ext4. Where OpenFOAM actually executes;
             must contain no spaces (see module docstring).
    """
    name: str
    src: Path
    work: str

    def sync_in(self) -> None:
        """Copy the authored case into the run directory, replacing it."""
        posix_src = to_wsl_path(self.src)
        cp = _wsl_sh(f"rm -rf {_quote(self.work)} && mkdir -p {_quote(self.work)} && "
                     f"cp -a {_quote(posix_src)}/. {_quote(self.work)}/")
        if cp.returncode != 0:
            raise SolverError(f"could not stage case {self.name}: {cp.stderr[:400]}")

    def sync_out(self, patterns=("log.*", "postProcessing", "*.obj")) -> None:
        """Copy results back to the Windows side. Mesh/field data stay on ext4."""
        posix_src = to_wsl_path(self.src)
        globs = " ".join(_quote(p) for p in patterns)
        cp = _wsl_sh(f"cd {_quote(self.work)} && for g in {globs}; do "
                     f"cp -a $g {_quote(posix_src)}/ 2>/dev/null || true; done")
        if cp.returncode != 0:
            raise SolverError(f"could not retrieve case {self.name}: {cp.stderr[:400]}")

    def disk_usage_mb(self) -> float:
        cp = _wsl_sh(f"du -sm {_quote(self.work)} 2>/dev/null | cut -f1")
        try:
            return float(cp.stdout.strip().splitlines()[0])
        except (ValueError, IndexError):
            return float("nan")


def make_case(name: str, src_dir: str | os.PathLike, root: str | None = None) -> FoamCase:
    """Create a case rooted at a space-free POSIX directory inside WSL."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise SolverError(f"case name {name!r} must be space-free and simple "
                          "(OpenFOAM rejects spaces in paths)")
    base = root or os.environ.get(FOAM_ROOT_ENV) or _DEFAULT_FOAM_ROOT
    cp = _wsl_sh(f'mkdir -p "{base}/{name}" && cd "{base}/{name}" && pwd')
    work = cp.stdout.strip().splitlines()[-1] if cp.returncode == 0 else ""
    if not work or " " in work:
        raise SolverError(
            f"could not create a space-free run directory under {base!r} "
            f"(got {work!r}): {cp.stderr[:300]}")
    src = Path(src_dir)
    src.mkdir(parents=True, exist_ok=True)
    return FoamCase(name, src, work)


@dataclass(frozen=True)
class FoamRun:
    """One application invocation: what it cost and where its log is."""
    application: str
    argv_tail: str
    returncode: int
    wall_s: float
    peak_rss_mb: float | None
    log_path: Path

    @property
    def log_text(self) -> str:
        return self.log_path.read_text(encoding="utf-8", errors="replace")


def run_foam(application: str, case: FoamCase, args=(), *,
             log_name: str | None = None, timeout: float = 7200.0,
             n_proc: int | None = None, measure_rss: bool = True,
             bashrc: str | None = None) -> FoamRun:
    """Run one OpenFOAM application in `case.work`, logging to `case.src`.

    Non-zero exit -> SolverError carrying the log tail. A mesher that refuses
    the geometry or a solver that diverges is a FEASIBILITY OUTCOME of the HF
    path, not a crash, so it is reported through the same exception the rest of
    the solver seam uses.
    """
    case_dir = case.work
    args = [str(a) for a in args]
    if n_proc and n_proc > 1:
        cmd = ["mpirun", "-np", str(int(n_proc)), application, *args, "-parallel"]
    else:
        cmd = [application, *args]

    quoted = " ".join(_quote(c) for c in cmd)
    snippet = (f"bash -c {_quote(_RSS_SAMPLER)} _ {_quote(RSS_FILE)} {quoted}"
               if measure_rss else quoted)
    try:
        script = bashrc or find_openfoam_bashrc()
        argv, _ = openfoam_runner(application, bashrc=script).shell(snippet, case_dir)
    except WSLError as exc:
        raise SolverError(f"OpenFOAM environment unavailable: {exc}") from exc

    def _persisted_rss() -> float | None:
        got = _wsl_sh(f"cat {_quote(case_dir + '/' + RSS_FILE)} 2>/dev/null")
        try:
            return int(got.stdout.strip()) / 1024.0
        except (ValueError, AttributeError):
            return None

    t0 = time.perf_counter()
    try:
        cp = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        peak = _persisted_rss()
        raise SolverError(
            f"{application} exceeded the {timeout:.0f} s timeout in {case_dir}"
            + (f" (peak RSS reached {peak:.0f} MB)" if peak else "")) from exc
    except OSError as exc:
        raise SolverError(f"could not launch {application}: {exc}") from exc
    wall = time.perf_counter() - t0

    log_path = case.src / (log_name or f"log.{application}")
    log_path.write_text(cp.stdout + ("\n--- stderr ---\n" + cp.stderr if cp.stderr else ""),
                        encoding="utf-8", newline="\n")

    peak = None
    m = re.search(r"PROPOPT_PEAK_RSS_KB=(\d+)", cp.stderr or "")
    if m:
        peak = int(m.group(1)) / 1024.0
    if not peak:
        peak = _persisted_rss()        # e.g. the marker was lost in a long log

    if cp.returncode != 0:
        tail = "\n".join((cp.stdout or "").splitlines()[-25:])
        err = (cp.stderr or "").strip()[:600]
        raise SolverError(
            f"{application} failed (exit {cp.returncode}) in {case_dir}\n"
            f"--- log tail ---\n{tail}\n--- stderr ---\n{err}")
    return FoamRun(application, " ".join(args), cp.returncode, wall, peak, log_path)


# --- parsing ----------------------------------------------------------------

def parse_checkmesh(text: str) -> dict:
    """checkMesh -> {cells, points, faces, failed_checks, max_skewness,
    max_non_orthogonality, mesh_ok}. Parses the report; does not judge it."""
    out: dict = {"failed_checks": [], "mesh_ok": None}
    for key, pat in (("cells", r"^\s*cells:\s+(\d+)"),
                     ("points", r"^\s*points:\s+(\d+)"),
                     ("faces", r"^\s*faces:\s+(\d+)")):
        m = re.search(pat, text, re.M)
        if m:
            out[key] = int(m.group(1))
    m = re.search(r"Max skewness = ([0-9.eE+-]+)", text)
    if m:
        out["max_skewness"] = float(m.group(1))
    m = re.search(r"Mesh non-orthogonality Max: ([0-9.eE+-]+)\s+average: ([0-9.eE+-]+)", text)
    if m:
        out["max_non_orthogonality"] = float(m.group(1))
        out["avg_non_orthogonality"] = float(m.group(2))
    m = re.search(r"Max aspect ratio = ([0-9.eE+-]+)", text)
    if m:
        out["max_aspect_ratio"] = float(m.group(1))
    out["failed_checks"] = re.findall(r"\*\*\*(.+)", text)
    if "Mesh OK." in text:
        out["mesh_ok"] = True
    elif "Failed" in text and "mesh checks" in text:
        out["mesh_ok"] = False
    return out


_RES_RE = re.compile(
    r"Solving for (\w+), Initial residual = ([0-9.eE+-]+), "
    r"Final residual = ([0-9.eE+-]+)")


def parse_residuals(text: str) -> dict[str, np.ndarray]:
    """field -> initial residual per occurrence, in order. simpleFoam prints one
    per outer iteration per field."""
    out: dict[str, list[float]] = {}
    for fld, init, _fin in _RES_RE.findall(text):
        out.setdefault(fld, []).append(float(init))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def parse_time_steps(text: str) -> np.ndarray:
    return np.asarray([float(t) for t in re.findall(r"^Time = ([0-9.eE+-]+)", text, re.M)])


def parse_forces_dat(path: str | os.PathLike) -> np.ndarray:
    """postProcessing forces.dat -> (n, k) numeric rows, comments stripped.

    The `forces` function object's column layout has changed between releases,
    so the caller must select columns by reading the file's own header rather
    than by a remembered index — see `forces_header`."""
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append([float(v) for v in s.replace("(", " ").replace(")", " ").split()])
    if not rows:
        raise SolverError(f"no numeric rows in {path}")
    width = min(len(r) for r in rows)
    return np.asarray([r[:width] for r in rows], dtype=float)


def forces_header(path: str | os.PathLike) -> list[str]:
    """The last comment line of a forces.dat — the column legend as written by
    THIS build, so column indices are read, never assumed."""
    head = [ln for ln in Path(path).read_text(errors="replace").splitlines()
            if ln.strip().startswith("#")]
    return head
