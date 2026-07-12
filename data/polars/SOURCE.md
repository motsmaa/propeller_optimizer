# Airfoil polar tables — provenance

Section: **NACA 4412** — a *proxy* for the (unpublished) APC Slow Flyer blade
section: comparable camber (4%) but thicker (12%) than the real thin blade.
Chosen because the preferred thinner sections (e.g. NACA 4406, Selig low-Re
props sections) are not available in the source database.

Tool: **XFOIL 6.96, Ncrit = 9, Mach 0, free transition** — tables are the raw
XFOIL polar dumps published by airfoiltools.com (retrieved 2026-07-11). XFOIL
was not available locally in this environment; these are pre-computed XFOIL
results from a public source, saved unmodified. NOT hand-edited, NOT fitted to
any propeller data.

| file | Re | alpha range (deg) | URL |
|---|---|---|---|
| `naca4412_re50000.txt`  | 5.0e4 | -9.25 .. +14.0  | http://airfoiltools.com/polar/text?polar=xf-naca4412-il-50000 |
| `naca4412_re100000.txt` | 1.0e5 | -9.5 .. +17.0   | http://airfoiltools.com/polar/text?polar=xf-naca4412-il-100000 |
| `naca4412_re200000.txt` | 2.0e5 | -8.5 .. +17.25  | http://airfoiltools.com/polar/text?polar=xf-naca4412-il-200000 |
| `naca4412_re500000.txt` | 5.0e5 | -12.25 .. +17.25| http://airfoiltools.com/polar/text?polar=xf-naca4412-il-500000 |

## Low-Re extension (generated locally, 2026-07-12)

Generated with **XFOIL 6.99** (precompiled Windows build `XFOIL6.99.zip` from
https://web.mit.edu/drela/Public/web/xfoil/, no local compilation; default
paneling). Settings matched to the airfoiltools tables: `NACA 4412`, viscous,
**Ncrit = 9 (default), Mach 0, free transition, ITER 200**. Sweep alpha
0 -> +18 deg step 0.25, then re-initialized (`INIT`) and 0 -> -10 deg, both
accumulated into one polar file (rows sorted by the loader). Unconverged
points are NOT stored by XFOIL and were not forced; raw dumps unmodified.

| file | Re | converged | dropped (VISCAL non-convergence) |
|---|---|---|---|
| `naca4412_re10000.txt` | 1.0e4 | 110/113 | 3 |
| `naca4412_re20000.txt` | 2.0e4 | 113/113 | 0 |
| `naca4412_re30000.txt` | 3.0e4 | 110/113 | 3 |

**Validity warning:** Re = 1e4 on a 12%-thick section is at the edge of
XFOIL's validity (massive laminar separation; at Re = 3e4 the boundary-layer
march emitted ~1.5e3 bubble-region warnings even on converged points). Treat
sub-5e4 values as direction-only, not truth.

## Validation thresholds tied to this dataset (Phase-1 closeout)

The UIUC anchor (tests/test_validation_uiuc.py) gates at **CT MAPE <= 20%,
CP MAPE <= 25%** for the analytic and Re>=5e4 tabulated variants. Those bounds
are scoped to THIS dataset: a non-matched 12%-thick NACA 4412 proxy for the
real ~5%-thick APC blade section, with known-optimistic Ncrit=9 XFOIL drag at
low Re. They encode that diagnosed input-data limit, not solver accuracy (the
solver was validated structurally at closeout — see README "Phase 1
(closed)"). If the section/dataset here is ever replaced with a matched thin
section, the thresholds should be re-derived, tighter. The Re->1e4 extended
variant is non-gating (xfail): sub-5e4 XFOIL on this thick section is
edge-of-validity, direction-only.

Known coverage limits (see propopt.solvers.polars for how these are handled):
- No data below Re = 1e4; the very lowest APC SF 10x4.7 root stations at low
  RPM (~5.5e3) still clamp to the 1e4 table.
- Alpha outside a table's range falls back to a flat-plate post-stall model
  via a short blend — crude, clearly marked in code.
- XFOIL with Ncrit=9 is known to be optimistic (laminar-bubble effects
  underpredicted) at low Re versus wind-tunnel measurements.
