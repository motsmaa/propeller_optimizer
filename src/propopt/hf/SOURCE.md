# Phase-4 high-fidelity toolchain — provenance

Same discipline as `data/uiuc/SOURCE.md`: record exactly what produced the
numbers, because behaviour is version-dependent.

Recorded 2026-07-29.

## CalculiX (structural solver)

| item | value |
|---|---|
| binary | `ccx` |
| version | **2.21** (`ccx -v` -> "This is Version 2.21") |
| Debian package | `calculix-ccx` 2.21-1build1, amd64 |
| install route | **WSL2 + Debian/Ubuntu package** — `apt-get install -y calculix-ccx` inside the WSL distro |
| distro | Ubuntu 26.04 LTS (`resolute`), WSL version 2 |
| path inside WSL | `/usr/bin/ccx` |

`ccx` is a **SYSTEM BINARY, not a pip package** — it cannot go in
`pyproject.toml`. It is reached across the Windows<->WSL boundary by
`propopt.hf.wsl`, which resolves the binary in this order: explicit
`binary=` argument, `$PROPOPT_CCX`, the Linux PATH inside WSL, then the
Windows PATH (for a native build). No distro name or version-numbered path is
hardcoded anywhere.

**Why the version is recorded:** CalculiX INP keyword support varies by
release, and this bites specifically when *modal-with-preload* arrives
(`*FREQUENCY, PERTURBATION` following a `*STATIC` step). 2.21 supports
perturbation frequency analysis; older 2.1x releases differ in how the
preloaded stiffness is carried between steps. Any change of this version
invalidates the anchors below and they must be re-run.

## gmsh (mesher)

| item | value |
|---|---|
| version | **4.15.2** |
| install route | pip wheel from PyPI (the gmsh project's own upstream SDK wheel) |
| declared in | `pyproject.toml` `[project.optional-dependencies]` -> `hf` |

gmsh IS pip-installable, so it is an optional extra (`pip install -e '.[hf]'`)
and never a core runtime dependency. Core deps remain
`["numpy", "pymoo>=0.6"]`.

## OpenFOAM (CFD solver — Phase 4b)

Recorded 2026-08-01.

| item | value |
|---|---|
| fork | **openfoam.com (ESI/OpenCFD)** |
| version | **v2512** (`$WM_PROJECT_VERSION`), API **2512** (`$FOAM_API`) |
| Debian packages | `openfoam2512`, `-common`, `-default`, `-dev`, `-source`, `-tools`, `-tutorials`, all **2512.0-2**, amd64 |
| install route | explicit keyring + pinned apt package from `https://dl.openfoam.com/repos/deb resolute main` |
| distro | Ubuntu 26.04 LTS (`resolute`), WSL version 2 |
| environment | `/usr/lib/openfoam/openfoam2512/etc/bashrc` |
| applications | `/usr/lib/openfoam/openfoam2512/platforms/linux64GccDPInt32Opt/bin/` |

Install (the signing key is dearmored into its own keyring and the repo is
scoped to it with `signed-by`; nothing is piped into a root shell):

```bash
curl -fsSL https://dl.openfoam.com/pubkey.gpg | sudo gpg --dearmor -o /usr/share/keyrings/openfoam.gpg
echo "deb [signed-by=/usr/share/keyrings/openfoam.gpg] https://dl.openfoam.com/repos/deb resolute main" \
  | sudo tee /etc/apt/sources.list.d/openfoam.list
sudo apt-get update && sudo apt-get install -y openfoam2512-default
```

**Why ESI and not the Foundation build.** The openfoam.org repository publishes
**no numbered stable release for `resolute`** — only `openfoam-dev`, dated
rolling snapshots (`20260623`, `20260710` at the time of writing). A validator
whose output is meant to be citable cannot sit on a base that changes under it
between `apt update`s. ESI publishes versioned, side-by-side packages, so the
exact version is pinned and recorded — the same discipline applied to ccx and
gmsh above. The two forks also differ in dictionary syntax in places, so a case
written for one is not guaranteed to run unchanged on the other; **which fork
produced a number is part of the number.**

**OpenFOAM is a SYSTEM ENVIRONMENT, not a binary and not a pip package.** Its
applications are only on `$PATH` after `etc/bashrc` is sourced, which also sets
the `$FOAM_*` and `$WM_*` variables the applications themselves read. It is
reached through `propopt.hf.wsl`, the single Windows<->WSL translation point:
`Runner.env_script` sources the environment in the same shell as the command.
Discovery is by glob over both forks' install locations, highest version by
`sort -V`, overridable with `$PROPOPT_OPENFOAM_BASHRC`. **No version or distro
path is hardcoded.**

**`set -e` must NOT be used to guard the sourcing** (measured on v2512):
`. etc/bashrc` returns 0 on its own, but under `set -e` the shell aborts
partway *through* it, because its config scripts use non-zero-returning probes
as ordinary statements — so `set -e` turns every successful run into a silent
failure. The guard is instead a positive postcondition, `$WM_PROJECT_DIR`
non-empty, which is what actually distinguishes a built environment from a
half-built one. Failing that test exits with the distinct code
`wsl.ENV_INIT_FAILED` (97) so a setup problem is never reported as a
solver-physics failure.

## Units

Every INP deck written by `hf/calculix.py` is **strict SI**: metres, kilograms,
seconds, newtons, pascals, kg/m^3. CalculiX is unit-agnostic — it multiplies
whatever numbers it is given — so mixing mm/MPa is the classic silent failure.
The STL exported by `export._write_ascii_stl` is already in metres (the loft's
vertices are SI) and nothing in `hf/` rescales it.

The same applies to OpenFOAM, and with one extra trap: **`simpleFoam` is an
INCOMPRESSIBLE solver, so its pressure field `p` is kinematic — p/rho, in
m^2/s^2, not pascals.** Forces are therefore only in newtons if the `forces`
function object is given `rhoInf` explicitly with `rho rhoInf`. Every case
written by `hf/openfoam.py` states its unit system in a header comment in the
case files themselves, sets `rhoInf` to the same air density the BEMT solver
uses, and reports thrust/torque in SI. Lengths are metres, matching the STL
handed to `snappyHexMesh` without rescaling.
