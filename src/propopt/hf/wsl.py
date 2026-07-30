"""Windows<->WSL boundary: path translation + subprocess invocation.

ONE place owns the boundary, because Phase 4b (OpenFOAM) crosses it too. The
contract is deliberately TEXT-ONLY and one-directional per file:

    Python (Windows)  writes  the input deck  ->
    solver (WSL/Linux) reads it, writes results ->
    Python (Windows)  parses the results

Nothing is streamed across the boundary and no shared library is loaded across
it; the only coupling is files on a path both sides can see (a Windows drive,
which WSL mounts under /mnt/<drive>) plus one `wsl -e` call. That keeps the
failure modes to two: a path that does not translate, and a non-zero exit.

Solver location resolution order (never a hardcoded distro or versioned path):
  1. explicit `binary=` argument
  2. $PROPOPT_CCX (or the caller's env var) — may be a Linux or Windows path
  3. the bare name on the Linux PATH inside WSL
  4. the bare name on the Windows PATH (native build, no WSL hop)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

WSL_EXE = "wsl"
_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


class WSLError(RuntimeError):
    """The Windows<->WSL boundary itself failed (not a solver-physics failure).
    Callers translate this into SolverError at the solver seam."""


def to_wsl_path(path: str | os.PathLike) -> str:
    """Windows path -> WSL path. C:\\a\\b -> /mnt/c/a/b. Already-POSIX paths
    pass through unchanged so a $PROPOPT_CCX of '/usr/bin/ccx' still works.

    Uses the absolute path: relative Windows paths have no meaningful meaning
    on the other side of the boundary (WSL's cwd is not Windows' cwd)."""
    raw = str(path)
    if not _DRIVE_RE.match(raw) and "\\" not in raw:
        return raw                                  # already POSIX-ish
    resolved = str(Path(raw).resolve())
    m = _DRIVE_RE.match(resolved)
    if not m:
        raise WSLError(f"cannot translate to a WSL path: {raw!r}")
    drive, rest = m.group(1).lower(), m.group(2).replace("\\", "/")
    return str(PurePosixPath(f"/mnt/{drive}") / rest)


def wsl_available() -> bool:
    """True if a WSL distro can actually run a command (not merely installed)."""
    if shutil.which(WSL_EXE) is None:
        return False
    try:
        return subprocess.run([WSL_EXE, "-e", "true"], capture_output=True,
                              timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@dataclass(frozen=True)
class Runner:
    """How to invoke one solver binary, on whichever side of the boundary it
    lives. `use_wsl=False` means a native Windows build (no translation)."""
    binary: str
    use_wsl: bool

    def command(self, args: list[str], cwd: str | os.PathLike) -> tuple[list[str], str | None]:
        """Full argv plus the Windows cwd to run it from. Under WSL the working
        directory is set INSIDE the distro (cd to the translated dir) so the
        solver writes its results next to its input, as ccx expects."""
        if not self.use_wsl:
            return [self.binary, *args], str(cwd)
        inner = " ".join(_quote(a) for a in [self.binary, *args])
        return [WSL_EXE, "-e", "bash", "-lc",
                f"cd {_quote(to_wsl_path(cwd))} && {inner}"], None

    def run(self, args: list[str], cwd: str | os.PathLike,
            timeout: float = 1800.0) -> subprocess.CompletedProcess:
        argv, win_cwd = self.command(args, cwd)
        try:
            return subprocess.run(argv, cwd=win_cwd, capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise WSLError(f"{self.binary} exceeded {timeout:.0f}s timeout") from exc
        except OSError as exc:
            raise WSLError(f"could not launch {self.binary}: {exc}") from exc


def _quote(s: str) -> str:
    """Single-quote for bash -lc; a POSIX path never needs more than this."""
    return "'" + s.replace("'", "'\\''") + "'"


def resolve_runner(name: str, binary: str | None = None,
                   env_var: str | None = None) -> Runner:
    """Locate a solver binary per the documented order. Raises WSLError with
    actionable text when nothing is found (this is a setup problem, and the
    caller should surface it as such rather than as a physics failure)."""
    candidates: list[str] = []
    if binary:
        candidates.append(binary)
    if env_var and os.environ.get(env_var):
        candidates.append(os.environ[env_var])

    for cand in candidates:                          # explicit wins, either side
        if Path(cand).exists():                      # a Windows-visible file
            is_win = Path(cand).suffix.lower() == ".exe"
            return Runner(cand if is_win else to_wsl_path(cand), use_wsl=not is_win)
        if wsl_available() and _wsl_has(cand):
            return Runner(cand, use_wsl=True)
        if shutil.which(cand):
            return Runner(cand, use_wsl=False)

    if wsl_available() and _wsl_has(name):           # bare name inside WSL
        return Runner(name, use_wsl=True)
    if shutil.which(name):                           # native Windows build
        return Runner(name, use_wsl=False)
    raise WSLError(
        f"{name!r} not found. Install it inside WSL "
        f"(e.g. `sudo apt install -y calculix-ccx` for ccx), or set "
        f"${env_var or 'the binary path'} to its location. See "
        "src/propopt/hf/SOURCE.md."
    )


def _wsl_has(cand: str) -> bool:
    try:
        cp = subprocess.run([WSL_EXE, "-e", "bash", "-lc",
                             f"command -v {_quote(cand)}"],
                            capture_output=True, text=True, timeout=60)
        return cp.returncode == 0 and bool(cp.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False
