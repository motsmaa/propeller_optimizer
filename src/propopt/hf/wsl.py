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

OPENFOAM IS DIFFERENT: it is an ENVIRONMENT, not a binary on the PATH. Its
applications are only reachable after `etc/bashrc` has been sourced, which also
sets $FOAM_* used by the applications themselves. So a Runner may carry an
`env_script` that is sourced before the command in the SAME shell. That is why
the sourcing lives here and not in a second helper: the boundary stays one
place, and `Runner.command()` remains the only builder of a WSL argv.
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

# Distinct exit code for "the solver environment did not initialise", so that
# a setup failure is never mistaken for a solver-physics failure.
ENV_INIT_FAILED = 97


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
    lives. `use_wsl=False` means a native Windows build (no translation).

    `env_script` is a POSIX path sourced before the command in the same shell
    (OpenFOAM's `etc/bashrc`). None for solvers that are plain binaries."""
    binary: str
    use_wsl: bool
    env_script: str | None = None

    def command(self, args: list[str], cwd: str | os.PathLike) -> tuple[list[str], str | None]:
        """Full argv plus the Windows cwd to run it from. Under WSL the working
        directory is set INSIDE the distro (cd to the translated dir) so the
        solver writes its results next to its input, as ccx expects."""
        if not self.use_wsl:
            if self.env_script:
                raise WSLError("env_script is meaningless for a native Windows binary")
            return [self.binary, *args], str(cwd)
        return self.shell(" ".join(_quote(a) for a in [self.binary, *args]), cwd)

    def shell(self, snippet: str, cwd: str | os.PathLike) -> tuple[list[str], str | None]:
        """argv for an ARBITRARY shell snippet, run in `cwd` with the
        environment sourced. Exists so callers that need a pipeline, an
        `mpirun` wrapper or a resource sampler do not rebuild the boundary
        themselves — this stays the only place a WSL argv is constructed."""
        if not self.use_wsl:
            raise WSLError("shell() is WSL-only; a native binary has no shell hop")
        return [WSL_EXE, "-e", "bash", "-lc",
                f"{self._env_prefix()}cd {_quote(to_wsl_path(cwd))} && {snippet}"], None

    def _env_prefix(self) -> str:
        """Shell prefix that sources `env_script` and PROVES it worked.

        NOT `set -e`. OpenFOAM's `etc/bashrc` is not `set -e`-safe: measured on
        v2512, `. etc/bashrc` returns 0 on its own, but under `set -e` the shell
        aborts partway THROUGH it (its config scripts use non-zero-returning
        probes as ordinary statements). `set -e` would therefore turn every
        successful run into a silent failure.

        The guard is instead a positive POSTCONDITION: $WM_PROJECT_DIR is set
        only once the environment is actually built, so testing it catches a
        half-initialised environment — which is the thing we care about — rather
        than testing that no command anywhere in a third-party script returned
        non-zero. Source output goes to stderr so stdout stays clean for the
        application's own output."""
        if not self.env_script:
            return ""
        q = _quote(self.env_script)
        return (f". {q} 1>&2 || true; "
                f'[ -n "${{WM_PROJECT_DIR:-}}" ] || {{ echo "propopt: sourcing '
                f'{self.env_script} did not initialise the OpenFOAM environment '
                f'(WM_PROJECT_DIR unset)" 1>&2; exit {ENV_INIT_FAILED}; }}; ')

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


# --- OpenFOAM: an environment, not a binary ---------------------------------

# Searched in order; the HIGHEST version found wins (`sort -V`). No version is
# hardcoded — v2512 today, v2606 tomorrow, both are matched by the glob.
OPENFOAM_BASHRC_GLOBS = (
    "/usr/lib/openfoam/openfoam*/etc/bashrc",   # openfoam.com (ESI) .deb
    "/opt/openfoam*/etc/bashrc",                # openfoam.org (Foundation) .deb
    "/usr/lib/openfoam*/etc/bashrc",            # some source builds
)
OPENFOAM_ENV_VAR = "PROPOPT_OPENFOAM_BASHRC"


def find_openfoam_bashrc(env_var: str = OPENFOAM_ENV_VAR) -> str:
    """POSIX path of the OpenFOAM `etc/bashrc` inside WSL.

    Order: $PROPOPT_OPENFOAM_BASHRC (Windows or POSIX path), then the globs
    above, highest version last. Raises WSLError with an actionable message —
    a missing OpenFOAM is a SETUP problem, and the caller must surface it as
    that rather than as a physics failure."""
    override = os.environ.get(env_var)
    if override:
        cand = to_wsl_path(override)
        if not _wsl_test_f(cand):
            raise WSLError(f"${env_var} = {override!r} but that file does not "
                           "exist inside WSL")
        return cand
    if not wsl_available():
        raise WSLError("WSL is not available; OpenFOAM runs inside the distro")
    listing = " ".join(OPENFOAM_BASHRC_GLOBS)
    try:
        cp = subprocess.run([WSL_EXE, "-e", "bash", "-lc",
                             f"ls -1 {listing} 2>/dev/null | sort -V"],
                            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise WSLError(f"could not search for OpenFOAM inside WSL: {exc}") from exc
    hits = [ln.strip() for ln in cp.stdout.splitlines() if ln.strip()]
    if not hits:
        raise WSLError(
            "OpenFOAM not found inside WSL (looked for "
            f"{', '.join(OPENFOAM_BASHRC_GLOBS)}). Install the ESI package, "
            "e.g. `sudo apt-get install -y openfoam2512-default` after adding "
            "the dl.openfoam.com repository, or set "
            f"${env_var} to an existing etc/bashrc. "
            "See src/propopt/hf/SOURCE.md.")
    return hits[-1]


def openfoam_runner(application: str, bashrc: str | None = None) -> Runner:
    """A Runner for one OpenFOAM application, with the environment sourced."""
    return Runner(application, use_wsl=True,
                  env_script=bashrc or find_openfoam_bashrc())


def openfoam_available() -> bool:
    """True if an OpenFOAM environment can be located (for test skipping)."""
    try:
        find_openfoam_bashrc()
        return True
    except WSLError:
        return False


def openfoam_version(bashrc: str | None = None) -> dict:
    """Identify the installed OpenFOAM: {'bashrc', 'version', 'fork', 'api'}.

    `fork` distinguishes the two upstreams, because they differ in dictionary
    syntax in places and a case written for one is not guaranteed to run on the
    other — so the provenance record must say which produced a number."""
    script = bashrc or find_openfoam_bashrc()
    probe = Runner("bash", use_wsl=True, env_script=script)
    argv, _ = probe.command(
        ["-c", 'echo "VERSION=${WM_PROJECT_VERSION:-}"; '
               'echo "API=${FOAM_API:-${WM_PROJECT_API:-}}"; '
               'echo "DIR=${WM_PROJECT_DIR:-}"; '
               'command -v simpleFoam || true'],
        Path.cwd())
    cp = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    if cp.returncode != 0:
        raise WSLError(f"sourcing {script} failed (exit {cp.returncode}): "
                       f"{cp.stderr.strip()[:400]}")
    out = dict(ln.split("=", 1) for ln in cp.stdout.splitlines() if "=" in ln)
    version = out.get("VERSION", "").strip()
    return {
        "bashrc": script,
        "version": version,
        "api": out.get("API", "").strip(),
        # ESI versions are numeric (2512); Foundation versions are small
        # integers or 'dev' (12, dev). The install path disambiguates too.
        "fork": ("openfoam.com (ESI/OpenCFD)" if "/usr/lib/openfoam/" in script
                 else "openfoam.org (Foundation)"),
        "simpleFoam": cp.stdout.strip().splitlines()[-1]
                      if "simpleFoam" in cp.stdout else "",
    }


def _wsl_test_f(posix_path: str) -> bool:
    try:
        return subprocess.run([WSL_EXE, "-e", "bash", "-lc",
                               f"test -f {_quote(posix_path)}"],
                              capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _wsl_has(cand: str) -> bool:
    try:
        cp = subprocess.run([WSL_EXE, "-e", "bash", "-lc",
                             f"command -v {_quote(cand)}"],
                            capture_output=True, text=True, timeout=60)
        return cp.returncode == 0 and bool(cp.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False
