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

## Units

Every INP deck written by `hf/calculix.py` is **strict SI**: metres, kilograms,
seconds, newtons, pascals, kg/m^3. CalculiX is unit-agnostic — it multiplies
whatever numbers it is given — so mixing mm/MPa is the classic silent failure.
The STL exported by `export._write_ascii_stl` is already in metres (the loft's
vertices are SI) and nothing in `hf/` rescales it.
