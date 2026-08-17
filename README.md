# propopt — Phase 0 (scaffolding + frozen contracts)

Thin end-to-end slice: NSGA-II runs against placeholder evaluators through the
frozen interface contracts, before any real physics exists.

## Layout
```
src/propopt/
  contracts.py      PropellerDesign, EvalResult, StructResult, ABCs, SolverError (SI)
  stations.py       station spacing schemes (uniform_midpoint, cosine) + quadrature weights
  motor.py          Phase 2c: BLDC motor model + torque-equilibrium root-find (Brent)
  hf/               Phase 4a: high-fidelity validators (NOT the inner loop)
    wsl.py          Windows<->WSL boundary: path translation + subprocess (4b reuses)
    mesh.py         STL -> tet mesh via gmsh; volume-drift + quality gates
    occ_geometry.py slice 2: blade as an OCC B-rep solid + native tet meshing
    hub_fillet.py   slice 3a: hub boss + root fillet; derived bounds, validity
    calculix.py     INP writer, ccx runner, .dat parser, CalculiXStructSolver
    SOURCE.md       toolchain provenance (ccx 2.21, gmsh 4.15.2, install route)
  config.py         Phase 3: RunConfig (frozen) + build_run/run; single source of truth
  configs/default.py Phase 3: the default RunConfig with justification per value
  analysis.py       Phase 3: hypervolume convergence + constraint-activity breakdown
  export.py         Phase 3: front -> JSON, selected designs -> STL
  plotting.py       Phase 3: scatter-matrix + parallel-coords (optional matplotlib)
  units.py          SI conventions + sanity-check assertions
  problem_spec.py   config-driven spec; objectives bundle (name, sense, fn)
  optimize.py       pymoo NSGA-II wiring + failed-eval penalty policy
  solvers/null.py   placeholder aero/struct (kept as tests/reference)
  solvers/bemt.py   Phase 1: static/hover BEMT aero, swappable polar seam
  solvers/polars.py TabulatedPolar: Re-interpolating XFOIL table polar
  solvers/geom_mass.py  Phase 2a: real mass from the lofted solid, placeholder SF
  solvers/beam.py   Phase 2b: rotating-cantilever beam FE (static + modal), real SF
  geometry.py       Phase 2a: design -> verified-watertight triangulated blade solid
  validation/       UIUC propeller-DB parsers (validation only)
data/               raw measured/XFOIL datasets, unmodified + SOURCE.md provenance
tests/              contracts, smoke (DoD), reproducibility, robustness, BEMT,
                    polar tables, UIUC validation anchor
run_demo.py         prints a Pareto-front summary
```

## Run
```
pip install -e .            # or: pip install numpy "pymoo>=0.6"
PYTHONPATH=src python run_demo.py
PYTHONPATH=src pytest -q
```

## Design notes
- **Objectives are atomic** `(name, sense, fn)` triples; there is no parallel sign
  array to keep in sync. Adding an objective is one tuple entry.
- **`ParetoResult.F` is physical** (already un-negated); MIN-form lives only inside
  the optimizer.
- **Failed evaluations don't crash runs.** A `SolverError`, an out-of-range result
  (caught by the sanity layer), or an arithmetic blow-up penalizes that candidate;
  genuine programming errors still propagate. `ParetoResult.n_failed` reports the
  count so a high failure rate is visible.

## Definition of done (met)
NSGA-II emits a non-empty, non-degenerate Pareto front of correct shape on the
null evaluators; runs are reproducible by seed; a partially-failing solver
completes with failures excluded from the front; contracts are frozen and sanity-checked.

## Phase 1 (closed): BEMT aero
`solvers/bemt.py` implements static/hover blade-element momentum theory behind
the frozen `AeroSolver` ABC (Prandtl tip+root losses, `SolverError` on
non-convergence or degenerate input). Airfoil data enters only through a
swappable `polar(alpha, Re) -> (Cl, Cd)` callable; the default analytic polar
ignores Reynolds number; `solvers/polars.py` adds `TabulatedPolar`, a
Re-interpolating table polar (currently NACA 4412 XFOIL data in
`data/polars/`, a proxy section — see its SOURCE.md). Validation anchor:
`tests/test_validation_uiuc.py` compares BEMT against measured UIUC static
data for the APC Slow Flyer 10x4.7 (raw files + provenance in `data/uiuc/`,
parser in `validation/uiuc.py`), parametrized over three polars.

### Phase-1 closeout
**The BEMT solver is validated as structurally correct; no residual anchor
error was traced into the solver.** Evidence: the fixed point is
relaxation-independent (relax 0.2 vs 0.05 agree to ~1e-6 N); the predicted
power-per-thrust ratio CP/CT matches measurement to ~3%; the high-RPM CT
offset is stable across two independent polar sets; and station resampling is
discretization-converged (20 vs 80 stations moves CT/CP ~0.1%).

Residual anchor error is attributed to three input causes:
1. **Non-matched proxy section** — 12%-thick NACA 4412 vs the real ~5%-thick
   thin cambered APC blade section. Dominates at low Re: with genuine
   sub-5e4 XFOIL tables, low-RPM CT error reaches -39% (thick sections
   collapse at low Re in XFOIL; the real thin section does not).
2. **Known-optimistic XFOIL Ncrit=9 profile drag at low Re** (laminar-bubble
   losses underpredicted), visible as the unreproduced CP-vs-RPM drift.
3. **An unresolved split inside the stable ~-17% high-RPM CT offset** between
   a section lift deficit (est. dCl0 ~ 0.15 +/- 0.05) and a possible 1-2 deg
   blade-datum/twist-convention offset. These two are entangled and were NOT
   separated: both are RPM-independent in the clamping-free regime, so this
   dataset cannot distinguish them.

Validation gates (justification embedded in the test's assert message and
data/polars/SOURCE.md): the `analytic` and `tab-re5e4` anchor variants gate at
**CT MAPE <= 20%, CP MAPE <= 25%**, scoped explicitly to validation against
UIUC apcsf_10x4.7 using the non-matched 4412/Ncrit=9 proxy — these bounds
encode the diagnosed data limitation above, not solver accuracy (current:
18.2%/20.2% and 16.6%/19.5%, both passing). The `tab-extended` (Re down to
1e4) variant is **non-gating** (`xfail(strict=False)`): XFOIL at Re=1e4 on a
12%-thick section is edge-of-validity, so its -39% low-RPM figure is
directional documentation of the proxy's low-Re failure, not a correctness
gate.

**Deferred refinement path (backlog, not scheduled):** matched thin-section
polars (measured UIUC LSAT data or XFOIL on an actual thin geometry) plus a
blade-datum/twist-convention audit to resolve the section-vs-datum fork in
(3). Pull this only if a later phase shows BEMT lift accuracy is the binding
constraint on a decision.

**A2 (RESOLVED in Phase 2c, Path 2):** `rpm` was a free design variable with no
motor coupling. It is now solved from a motor-propeller torque equilibrium;
throttle is the operating variable in the genome. See "Phase 2c" below.

## Phase 2a (in progress): parametric geometry
`geometry.py` lofts a parametric section (seam: `SectionFn`; default NACA
4412-like, matching the Phase-1 aero polar proxy) through the radial stations
into a triangulated blade solid with named patches (`hub` root cap, `blade`
skin, `tip` cap). Engine: pure-numpy structured loft — watertight BY
CONSTRUCTION (closed-manifold connectivity is topological), ~1 ms/genome, no
CAD kernel in the optimizer loop (a B-rep kernel is reserved for once-per-
final-design STEP export later; OCC-class lofts are too slow/opaque for
thousands of genomes). Every build is verified closed-manifold + positive-
volume; violations raise `SolverError`. `solvers/geom_mass.py` wires REAL
mass (n_blades x volume x PLA density) into the optimizer; thrust/FoM/mass
are now real physics, safety factor/stress remain Phase-0 placeholders until
Phase 2b. Anchor (`tests/test_geometry.py`): 100% watertight over 200 random
in-bounds genomes; prism volume matches the shoelace-analytic value to 1e-12;
degenerate genomes raise `SolverError`; BEMT numbers are pinned against the
shared-station refactor.

**Shared stations (landed):** `PropellerDesign.station_r_m` carries the
station radii (strictly increasing, within [hub, R]) as the single source of
station placement; spacing schemes (`uniform_midpoint`, `cosine`) and the
quadrature-weights helper live in `stations.py`. BEMT, geometry and
NullStruct all read the shared field — the duplicated uniform-midpoint
assumptions are gone. Domain consistency: aero integrates midpoint-bounded
cells clipped to [hub, R], and `solid_from_design` end-extends the loft to
the same [hub, R], so aero and geometry describe the same interval. The
refactor is regression-gated at rel 1e-12 (justification inline at the
assertion in `tests/test_geometry.py`): with uniform-midpoint spacing the
pinned Phase-1 BEMT values reproduce to <= 1 ulp (torque/power bitwise;
thrust/FoM 1 ulp from quadrature-weight summation reassociation — diagnosed,
not a physics change). Dev/test-only dependency: `trimesh` independently
cross-validates watertightness and volume in the test suite (skips if
absent); it is deliberately NOT a runtime dependency — the package runtime
stays numpy-only (+pymoo).

Phase-2a assumptions and deliberate semantics:
- **Quadrature**: each station owns the radial cell bounded by neighbour
  midpoints, clipped to [hub, R]; weights always sum to the full span.
- **Loft end-extension**: chord/twist are held constant from the end stations
  out to hub and R — no tip taper. Tip-rounding is a possible later
  refinement, not current behaviour.
- **Cosine spacing** samples cell midpoints in the stretched coordinate, so
  no station sits exactly on hub or tip.
- **Single-station designs are valid** (intentional): end-extension yields a
  3-section loft; core `build_blade_solid` still requires >= 2 explicit
  sections.
- **Failure layering (principle)**: the frozen contract rejects
  geometrically-impossible designs at construction (`ValueError`, e.g.
  hub == tip / non-increasing stations); solvers reject contract-valid but
  physically-unsolvable ones (`SolverError`, penalized by the optimizer).

### Placement twist-sign defect (found 2026-08, fixed)

Sections were rotated by **+twist** while the chord is laid out with the
**leading edge at xi = -0.25c**, i.e. at NEGATIVE x. Those two conventions are
incompatible: the only rotation sense that leads with the nose then puts the
blade at NEGATIVE incidence. Every lofted blade was effectively pitched
nose-down, and no rotation direction made it a working propeller. The fix
rotates by `-twist`; `station_rings` and `build_blade_solid` remain bitwise
identical to each other.

**Why nothing caught it for three phases.** No consumer of the geometry could
see it:

| consumer | why it was blind |
|---|---|
| BEMT | reads `twist_rad` directly and uses \|Omega r\|; never touches the 3-D placement |
| beam / structural | centrifugal magnitude and section bending; both placement-sign-invariant |
| loft volume / mass | a proper rotation preserves volume; orientation-insensitive |
| mesh quality, watertightness | topological, not orientational |

Only a **3-D placement consumer** could observe it, and the project had none
until Phase 4b. It was found empirically: sampling blade surface pressure in a
solved CFD field showed the face the code labelled "suction" carrying the
HIGHER pressure across r/R 0.2-0.8 (by 99-267 m^2/s^2). Two sign-algebra
attempts got it wrong in opposite directions before the measurement settled it.

**Guards added** (`tests/test_hf_rotation.py`, enforced in `build_hover_case`):
`assert_leading_edge_first` (chord vs direction of travel) and
`assert_positive_incidence` (alpha = +asin(wind . n_suction), measured against
the suction normal sampled at MATCHED chordwise indices). Each catches a
different failure; neither alone would have caught this one.

**Blast radius.** Loft volume and mass shift by **0.376%** (UIUC blade
4.913726e-6 -> 4.895264e-6 m^3) from triangle-chording of the reversed twist.
That is the same O(dtheta^2) faceting effect already documented above and sits
well inside its 5% band, so **prior mass, structural and Pareto results stand**.
Pinned BEMT references, UIUC anchors and the beam SF/f1 pins are unchanged to
full precision.

**Affected artifacts — every STL exported before this fix is pitched the wrong
way and is not a valid propeller**, independent of CFD:
`runs/decision/stl/blade_*.stl`, `runs/exploration/stl/*`, and any STL handed
out from a Pareto front via `export.export_selected_stl`. Re-export them. The
numbers those runs reported remain valid; only the printable orientation was
wrong.

### Phase 4b baseline (good5): first physically-coherent CFD result

5R box, 1,184,958 cells, no layers, MRF (rotor cellZone r=1.3R, +/-0.4R),
kOmegaSST wall functions, potentialFoam init, 2500 iterations (1.96 h, 8 cores,
2731 MB). Driver: `run_hover_variant.py` (assertion-hardened: reconstructs
before sampling and RAISES on any zero/missing result, after an earlier version
silently reported zeros for a parallel run).

| | CT | CP |
|---|---|---|
| CFD good5 | 0.0530 | 0.0314 |
| measured UIUC @ 4590 rpm | 0.1200 | 0.0484 |
| BEMT | 0.0996 (-17.0%) | 0.0384 (-20.7%) |

The BEMT row previously read 0.0964 / 0.0381; that was **stale by 3.3% on CT**.
The tracked value is whatever `tests/test_validation_uiuc.py::test_bemt_vs_uiuc_apcsf_10x47_static[tab-re5e4]`
prints at 4590 rpm, which is 0.0996 / 0.0384. Corrected during rung 0, when the
reprice needed the reference it was being compared against.

The three-way gap decomposes cleanly and multiplicatively:

| step | ratio | attributable to |
|---|---|---|
| CFD / BEMT | x1.88 | the wake recirculation (see rung 0 below) |
| BEMT / measured | x1.20 | the Phase-1 residual (proxy section, transition, 3-D) |
| product = CFD / measured | x2.26 | |

alpha = +22.30 / +13.93 / +5.51 deg at root/mid/tip, equal to local twist to
0.0000 deg. Forces stationary (thrust p-p 1.02%, torque 0.49% over the last 200
iterations); residuals U ~9.5e-4, p 1.7e-6; mass conserved (net far-field flux
8.2e-6 m^3/s).

**What is right:** the inflow is textbook - smooth near-uniform acceleration
from -2.2 to -5.5 m/s over +2R to +0.5R, and a properly loaded annulus peaking
-10.5 m/s at the tip radius.

**What is not:** CT is -56% against measurement, and THE WAKE DOES NOT
PROPAGATE. Volume flux (r<=2.5R) peaks at -1.482 m^3/s at -0.5R, collapses to
-0.965 at -1R and REVERSES to +0.281 at -2R, where every radius is flowing back
toward the rotor. No contraction, no flux conservation - a recirculating torus,
not a slipstream.

This matters because it can no longer be blamed on a badly-loaded rotor (the
earlier nose-down explanation, now falsified): the rotor IS loaded and the
inflow IS coherent, and the wake still turns around within 1R. The
induced-velocity ratio is therefore NOT reported - momentum theory's v_i
presumes a developed contracting slipstream, and quoting it here would repeat
the phantom-thrust error made earlier in this phase.

Open: whether the confinement is domain size, far-field BC, MRF zone extent, or
whether a steady rotating-frame model can sustain a hover slipstream at all.

## Phase 4b (CLOSED, negative result): steady MRF cannot sustain a hover slipstream

**Outcome: steady frozen-rotor MRF does not produce a coherent hover slipstream
on any mesh resolvable within a 9 GB / 8-core budget.** Every non-transient
explanation was tested and eliminated. Transient sliding-mesh/AMI is required.
This is a negative result, not a failure: the eliminations are the deliverable.

### Independent finding: blade placement twist sign (real bug, fixed)

Sections were rotated by +twist against a leading-edge-at--x convention, so
every lofted blade was pitched nose-down and no rotation sense made it a working
propeller. Fixed (rotate by -twist); guards added
(`tests/test_hf_rotation.py`). **Exported STLs before that fix are invalid
propellers.** Loft volume/mass shifted 0.376%, inside the documented faceting
band, so prior mass/structural/Pareto results stand. See the Phase-2a section.

### Eliminated as causes of the wake reversal

| variable | range tested | effect |
|---|---|---|
| domain size | 5R, 10R, 15R, 20R | none (CT within noise) |
| far-field BC | totalPressure, inletOutlet | none |
| MRF zone extent | +/-0.4R, -2.5R, -4R | artifact only, non-monotonic |
| wall treatment | y+ 11.1 - 83.4 (mean) | none |
| blade surface resolution | 0.344 -> 0.172 mm | none (CT -8%, below noise) |
| iteration count | 2500 -> 10000 | none (residuals plateau ~1e-3) |
| diagnostics | validated vs shipped tutorial | correct |
| MRF implementation | Taylor-Couette analytic | 0.24% max error |

### The established finding, stated precisely

**The wake reversal is a numerical-diffusion artifact in the sense that COARSE
wake meshes SUPPRESS it — it is not a real feature that fine meshes resolve.**
Flux at -2R (r <= 1.5R) tracks wake cell size and nothing else:

| wake cell | 2.75 mm | 5.5 mm | 11.0 mm |
|---|---|---|---|
| flux at -2R | +0.078 .. +0.107 (reverses) | +0.053 | -0.426 .. -0.540 (exits) |

Four runs at 2.75 mm (good5, W23, ISO, LONG, SURF56) all reverse, spanning
y+ 11-28. Two runs at 11.0 mm both exit, spanning y+ 38-83. y+ ranges OVERLAP
across the two behaviours, so wall treatment cannot be the controlling variable
and wake resolution can. The uncomfortable direction: our FINEST meshes show the
recirculation, so refining further would make it worse, not better.

### The CT deficit - EXPLAINED by rung 0 (supersedes the "unexplained" reading)

CT sits at **-56% to -60%** against measured (0.1200) across every adequately
converged variant, unmoved by everything in the table above.

This section previously read "unexplained, and distinct from the BEMT residual",
and treated the CT deficit and the wake recirculation as two independent
problems. **Rung 0 falsified that: they are one problem.** The CFD-specific
x1.88 is the recirculation; only the x1.20 BEMT/measured step remains open, and
that one is the pre-existing Phase-1 residual on the same geometry.

### Rung 0: reprice CT from the CFD's OWN inflow (post-processing, no solve)

`rung0_reprice.py`. Extract the through-disc axial velocity from the converged
field, form phi(r) = atan(v_ax / Omega r) and alpha(r) = beta(r) - phi(r), then
price alpha through the SAME XFOIL polars and the SAME blade-element integral
BEMT uses (`bemt.py` lines 158/186, copied not re-derived). Cost ~18 s per case.

**The direct reprice does NOT close as originally framed**: at the disc plane it
gives CT 0.0251 against the CFD's directly-integrated 0.0530, a factor 0.47.
alpha is small, so CT is hypersensitive to the sampling plane - 0.0641 at
y=+0.25R through 0.0251 at y=0 to 0.0002 at y=-0.25R. The bracket contains
0.0530, but a method with that spread confirms nothing on its own.

**The inversion is the test that works.** By what uniform factor k must the
measured disc-plane inflow be scaled to reproduce a given CT?

| target | k | implied v_ax at mid-span |
|---|---|---|
| **CFD's own CT 0.0530** | **0.86** | 7.2 m/s |
| BEMT CT 0.0996 | 0.61 | 5.1 m/s |
| measured CT 0.1200 | 0.45 | 3.8 m/s |

The CFD's thrust is consistent with its own measured inflow to within 14%.
BEMT's CT needs a 39% inflow reduction and measurement 55% - neither value
occurs anywhere in the field, at any plane or radius. k = 0.86 rather than 1.00
is expected, not residual: sampling at y=0 is sampling IN the blade plane, so it
includes part of the blade's own bound-vortex downwash, which blade-element
theory accounts for through the lift itself rather than through the inflow.

**Summary statistic:** BEMT's local inflow is ~1.15x its own momentum-theory
ideal (5.2-5.9 vs 4.89 m/s at its 2.97 N) - normal for a real rotor. The CFD's
is **2.35x** its own. That gap is the entire anomaly.

**Recirculation, measured directly** rather than inferred from a single ray
(36 azimuths x 5 axial planes, azimuths within 20 deg of a blade excluded):

| v_ax [m/s] | y=+0.25R | y=+0.10R | y=0 | y=-0.10R | y=-0.25R |
|---|---|---|---|---|---|
| r/R 0.15 | 5.61 | 4.52 | 1.76 | 0.20 | **-1.28** |
| r/R 0.50 | 6.94 | 7.72 | 8.41 | 9.16 | 9.72 |
| r/R 1.00 | 7.01 | 8.66 | 10.09 | 11.11 | 11.83 |

Two signatures. The air arrives **already at 70% of its disc-plane speed a
quarter-radius upstream** - a real hover slipstream accelerates from ~0 far
upstream to v_i at the disc; this one is pre-accelerated because it is being
recycled. And the inner disc carries **negative** downstream axial velocity
(-1.3 to -1.6 m/s at r/R 0.15-0.20): flow returning up through the rotor. That
is the torus core, measured.

**The alpha(r) collapse.** alpha falls from +8.95 deg at the root monotonically
through **zero at r/R = 0.639**, reaching -3.9 deg at the tip; 9 of 20 stations
sit at negative incidence, against BEMT's +2.3 to +9.4 deg everywhere positive.

| r/R | alpha BEMT | alpha CFD | dT/dr BEMT | dT/dr CFD |
|---|---|---|---|---|
| 0.51 | 6.48 | 1.47 | 31.0 | 15.4 |
| 0.68 | 3.78 | -0.52 | 44.5 | 15.9 |
| 0.85 | 2.58 | -2.02 | 40.6 | **-0.6** |
| 0.94 | 2.33 | -3.17 | 23.3 | **-13.1** |

Precision point: NACA 4412 has alpha_0L ~ -4 deg, so negative alpha here is LOW
lift, not REVERSED lift - those stations still net +0.15 N of the 0.75 N total.
dT/dr turns negative only outboard of r/R ~ 0.82, where Cl ~ 0 and the
-Cd sin(phi) drag projection takes over. The blade's entire outer working
section, normally where most thrust is made, is simply unloaded.

**Robustness.** Convergence-independent: LONG (10,000 iters) reproduces good5
(2,500) at k = 0.867 vs 0.858 and the identical zero crossing r/R = 0.639, ~1%
apart, well inside the 3.6% force noise floor. Polar-set-independent (Re>=5e4
0.0251, extended 0.0224). Not blade contamination (all azimuths 0.0164 vs
excluded 0.0251, same verdict; 28 valid samples per radial bin). Co-rotating
swirl is present with the physically correct sign and makes it slightly worse
(0.0193); BEMT neglects swirl entirely, so that is a minor extra term, not a
driver.

**Conclusion: the blade is responding correctly to the flow it is given.** No
separate blade-loading defect is needed to explain CT 0.0530. The CT deficit and
the wake recirculation are ONE problem, not two.

### k-epsilon: UNTESTED BY DEFAULT, not eliminated

Two bounded attempts both diverged (Fy -> -2.0e+65 at iteration 6; 1.0e+93 at
iteration 8 with a derived free-stream pair at nu_t/nu = 1.00 and relaxation
reduced to 0.5 on the turbulence equations). **This does NOT show the turbulence
model is irrelevant** - only that k-epsilon could not be made to run within the
agreed budget. Unexplored: bounded/limited schemes on epsilon, staged startup,
alternative near-wall treatment. Do not read this row as a completed test.
Discretisation schemes and the frozen-rotor assumption itself are likewise
untested.

### Transient sizing - a COST ESTIMATE, not a commitment

Anchors: 1.19M cells -> 2.7 GB, 2.8 s/iter steady on 8 cores; shipped transient
propeller tutorial 526k cells, 3073 steps, 1.8 h on 4 ranks. Hover needs ~360
steps/rev (1 deg/step at 4590 rpm) and 8-10 revs to develop -> ~3000-3600 steps,
each 3-5x a steady iteration.

| level | cells | RAM/run | wall-clock (8 cores) | fits 9 GB? |
|---|---|---|---|---|
| coarse | 1.2M | ~3.5 GB | ~30-40 h | yes |
| medium | 2.5M | ~7 GB | ~65-85 h | marginal |
| fine | 5M | ~14 GB | ~130-170 h | **no** |

~9 runs (setup + 3 mesh levels + 2 timestep + retries) = 400-600 h serial
locally: does not fit. Needs >=32 GB/node and 32-64 cores, ~2500-4000
core-hours. **Not provisioned, not committed.**

### STANDING RULE: an unexercised config field is an unvalidated config field

k-epsilon failed on THREE missing dictionary entries (`epsilon` field writer,
`div(phi,epsilon)`, `fvSolution` solver entry) that surfaced only on first use,
after two diverged 2-hour runs. A case-writer smoke test - build + mesh + 2
iterations on a tiny mesh - finds this class in seconds. **Before any large
compute commitment, exercise every config field once.** `runs/audit.py` is that
harness.

Pre-flight audit, 19 configurations, ~20 minutes. First run **16 passed / 3 failed**; all three fixed, now **19/19**. The harness distinguishes a genuine pass from an EXPECTED REFUSAL, so a guard that correctly refuses a degenerate configuration counts as a pass and a guard that silently accepts one counts as a failure:

| field | first run | after fix |
|---|---|---|
| `turbulence_model=SpalartAllmaras` | FAIL - missing `0/nuTilda` (same class as the epsilon gaps) | **OK** - field + `div(phi,nuTilda)` + solver entry added |
| `n_layers=3` (relative sizing) | FAIL - topoSet point-count mismatch on a layered mesh | **OK** - `topoSet` no longer pinned to `-constant`; layer addition moves points, so a stale point count was read |
| `periodic=True` (180 sector) | FAIL - cyclic face-area mismatch 66.7% | **OK (refused)** - now raises `SolverError` up front rather than producing a bad mesh; the geometry is NOT fixed, the degeneracy is documented and the caller is pointed at `periodic=False` |
| kOmega, kEpsilon, realizableKE, inletOutlet, nutUSpalding, core_radii, core_cell_frac, stl_size_m, n_chord, te_thickness_m, write_interval, rho, nu, mrf_zone_radii, mrf_zone_half_height_radii | pass |

Note `kEpsilon` now PASSES the smoke test - the dictionary gaps are fixed; it is
the SOLVE that diverges. Those are different failures and should not be conflated.

## Phase 2b (in progress): reduced structural solver
`solvers/beam.py` replaces the null safety factor with a rotating-cantilever
1-D beam FE on the shared stations (Euler-Bernoulli / Hermite elements,
numpy-only): centrifugal tension (stressing + stress-stiffening), static
bending from the BEMT per-blade spanwise loads (`BEMTAeroSolver.
spanwise_loads`, the aero->structure seam), extreme-fiber stress with a
documented root stress-concentration guard (Kt=2.5, conservative placeholder
until Phase-4 FEA calibrates it — **SUPERSEDED**, see "Phase 4a slice 3b/3c"
below: measured factors now amplify BOTH stress components), SF = yield/peak.
Modal: first flapwise
frequency with centrifugal stiffening + resonance margin vs blade-passing
harmonics (currently REPORTED only; contract extension pending approval).
Material is an explicit parameter — default PLA (E=3.5 GPa, yield=50 MPa,
rho=1240 kg/m^3, typical filament-datasheet values); printed-PLA anisotropy
is a documented isotropic first-pass simplification. Anchors
(`tests/test_beam.py`): FE matches closed-form tip deflection (qL^4/8EI,
nodally exact), centrifugal root tension (exact), and the uniform-cantilever
first frequency (3.516 sqrt(EI/m L^4), rel 1e-4); f1 rises with rpm; SF falls
with rpm and rises with chord; beam mass equals the lofted-solid mass exactly
untwisted (<=5% twisted, measured ~3.7% worst). All four objectives are now
physical; with real PLA strength the demo's max feasible thrust drops from
~46 N to ~13 N and the optimizer rides the SF=1 boundary.

**Resonance constraint (landed):** `StructResult` carries `first_freq_hz`
and `resonance_margin`; `radial_spec` enforces
`resonance_margin >= resonance_margin_min` (default **0.15** — deliberately
conservative because the first-pass beam f1 is itself uncertain: isotropic
PLA, rigid hub, no fillet; revisit when Phase-4 FEA calibrates the modal
model). Without it the optimizer parked 14/60 front designs inside the
keep-out band (min margin 0.004); with it, 0/60 (min 0.170) at essentially
unchanged thrust. **A2 coupling (known, unresolved):** rpm is a free design
variable, so the optimizer satisfies resonance mainly by shifting rpm — the
front's rpm range compressed from 3358-8403 to 3803-7798 under the
constraint. Do not read structural meaning into rpm placement until A2 is
decided. Null/geom-mass solvers carry documented ABSURD sentinels for the
modal fields (f1 = 1e9 Hz, margin = 1e6): unmistakably fake on inspection,
non-binding so plumbing tests keep working.

**Mass-discrepancy verdict (investigated):** the <=3.7% beam-vs-geometry
mass difference on twisted blades is a genuine geometry effect, not an
integration inconsistency. The mass objective is computed on the FACETED
TRIANGULATED LOFT (the actual printable artifact), which is therefore a
slight CONSERVATIVE OVER-ESTIMATE of the smooth-blade mass — the triangles
chording the non-planar quads between rotated twisted sections enclose
marginally more volume than the smooth extrusion. The beam model instead
tracks the smooth ∫A dr limit, which is exactly twist-invariant. These are
two legitimately different shapes: the effect is exactly zero untwisted,
scales as O(dtheta^2) per element, and vanishes under station refinement.
KNOWN AND EXPECTED, not a bug — documented here and in
`tests/test_beam.py::test_mass_consistency_with_geometry`.

### A2 — RESOLVED in Phase 2c (Path 2): rpm is now solved, not free
The resonance constraint is satisfiable two ways: stiffen the blade, or move
the excitation frequency by changing rpm. While rpm was a free design
variable (A2 open), NSGA-II took the cheap path — enabling the constraint
compressed the front's rpm distribution (8-10k bucket emptied, median
6940 -> 6552 rpm) at ~zero objective cost rather than forcing stiffer blades.
Phase 2c resolves this: rpm LEAVES the genome and is solved from the
motor-propeller torque equilibrium (see "Phase 2c" below); **throttle**
(the operating variable the pilot actually commands) replaces it. The
optimizer can no longer slide rpm independently of a physical motor/blade
pairing — full before/after evidence is in the Phase 2c section.

**Resonance model scope limits (documented, not solved):** the margin checks
only blade-passing harmonics n=1..3 against the first flapwise mode. There is
no 1P/shaft-order excitation term, no forward-flight forcing (consistent with
BEMT's static/hover scope), and no damping or forced-response amplitude
estimate — it is a frequency keep-out band, not a vibration prediction. A
full Campbell diagram (multiple modes, all excitation orders, vs. rpm) is
Phase-4 material once solid FEA is in the loop.

## Phase 2c (checkpoint): motor-propeller torque equilibrium (A2, Path 2)
`motor.py` adds a first-order brushless-DC model (`MotorSpec`: KV, Rm, I0,
V_max, I_max) and solves the operating point. rpm LEAVES the genome; the last
slot becomes **throttle in [0.2, 1]** and `radial_spec.decode` solves the
speed N where motor torque equals propeller torque
(`Q_motor(N)=Kt((V-N/KV)/Rm - I0)`, linear decreasing; `Q_prop(N)` from BEMT,
rising) via a hand-rolled **Brent** root-find on [floor, no-load speed]. rpm
stays a `PropellerDesign` field (now computed), so contracts.py and
optimize.py are untouched: the equilibrium runs inside the `decode` closure,
which binds `MotorSpec` + a shared BEMT solver. `motor=None` restores the
pre-2c free-rpm genome (used by the plumbing/geometry/resonance tests).

Feasibility (bred away, never crashes): **(b) over-current** — an equilibrium
exists but the current it draws exceeds I_max — is a bundled `Constraint`
(g <= 0); at equilibrium `Q_prop == Q_motor`, so the current is recovered
from `ev.torque_Nm` alone (`I = torque/Kt + I0`), fitting the `(ev, st)` seam
with no signature change. **(c) no equilibrium** (motor too weak / throttle
too low to bracket a crossing) raises `SolverError` from `decode` -> penalized
by the optimizer, same discipline as the BEMT stall path.

Default motor: **2216-class 950 KV on 3S** (Rm 0.10 Ohm, I0 0.6 A, 11.1 V,
18 A limit) — representative of that datasheet class, not a specific SKU. It
materially sets the achievable rpm band (no-load ~10.5k rpm) and hence the
whole front; swapping it is the intended way to study a different power
system. First-order model OMITS thermal/Rm-drift, magnetic saturation, ESC
timing/PWM below a few % throttle — the current constraint uses I_max as the
lumped proxy.

Cost: the root-find calls BEMT (Brent ~6 solves) plus `evaluate` and beam's
`spanwise_loads`. A 1-slot equality-checked memo on
`BEMTAeroSolver._converged_state`, with one BEMT instance shared across
decode/aero/beam, reuses the last solve: **~7 BEMT/candidate with the memo,
8 without, vs 2 pre-2c** (~3.5x). Anchors (`tests/test_motor.py`): Kt =
9.5493/KV; zero-prop-torque equilibrium == no-load speed; stall torque; the
equilibrium against an analytic Q=kN^2 prop matches the hand-solved quadratic
root (BEMT-independent); efficiency Q.omega/VI < 1; monotonic throttle->N and
blade->N; over-current and no-equilibrium flagged, not crashed.

**A2 made visible (demo, seed 0, 60/60):** with the motor coupling, **every**
front design de-rates throttle (0/60 at full throttle; throttle 0.36-0.85,
median 0.65) — throttle is the relief valve rpm used to be. Equilibrium rpm
spans 3.3-7.6k (motor-bounded, not the free 2-12k), the current constraint
BINDS at the high-thrust edge (max 17.9 A vs 18 A limit), and peak feasible
thrust drops **13.3 N -> 9.4 N** vs the free-rpm front: the motor, not an
arbitrary rpm choice, now limits thrust. Objective values for over-current
equilibria assume the ESC could supply the current (it can't) but are excluded
from the feasible front, so they don't affect results.

**What caps throttle — ANSWERED in Phase 3 (see below):** the constraint-
activity breakdown shows most of the front (53/60) is constraint-INTERIOR —
throttle is set by the objective trade-off, not a hard cap. Only the extremes
bind: the min-thrust corner sits on thrust>=min, and the high-throttle corner
approaches the SF and current limits jointly (throttle correlates +0.86 with
both g[SF] and g[current]).

## Phase 3 (checkpoint): config-driven runs, convergence, post-processing
A run is fully specified by a `RunConfig` (frozen dataclass, `config.py`) +
seed; `configs/default.py` constructs the default with each value's
justification inline (the reason config is Python, not JSON). `build_run(cfg)`
wires the existing radial_spec + solvers (sharing one BEMT so its memo helps);
`run(cfg, seed)` calls the existing `optimize` — contracts.py and optimize.py
untouched. Runs are bit-reproducible from config+seed. Entry point:
`python run_demo.py [--mode exploration|decision] [--seed N] [--out DIR] [--convergence]`.

**Two run modes (differ ONLY in seed-handling; identical physics, bounds,
optimizer, constraints):**
- **exploration** — a single seed, pop 60 / gen 40. For iteration and demos.
  Exploration-grade: the convergence study measured ~25% single-seed final-HV
  spread, so exact front edges carry seed noise.
- **decision** — aggregates a configured seed list (default 5) by merging their
  feasible archives into ONE cross-seed non-dominated front (union of all
  designs, then keep the cross-seed Pareto set — an actual aggregated front
  that export/plot/STL consume, NOT HV-averaging). Aggregation can only add
  coverage, so the aggregated HV is >= every constituent seed's HV.
  `configs/default.py` exposes `exploration_config()` and `decision_config()`;
  `decision_config` is `replace(exploration, mode="decision")` — the split is
  literally the mode field. `analysis.aggregate_front` reuses `run()` (the same
  optimize path the convergence study drives) over the seeds, then the pure
  `aggregate_results` merge. Every export's JSON `config` echo carries `mode`
  and `seeds`, so no output is ambiguous about its grade.

**Convergence (`analysis.py`; slow test `tests/test_convergence.py`, run with
`pytest -m slow`).** Measured cost ~20 ms/candidate, so a 5-seed study at
pop 60 x gen 40 = 12k evals ~ 4 min fits a <=5 min budget. Hypervolume is
computed on the CUMULATIVE feasible non-dominated archive (monotonic by
construction — the right convergence metric; the current-population front is
not) against a FIXED reference: objectives are mapped to a common [0,1] loss
using per-objective ideal/nadir bounds held constant in the config (thrust
15/1.5 N, FoM 1/0, mass 0/0.035 kg, SF 10/0.8), ref point = 1. These bounds
bracket every feasible front so HV is comparable across seeds/gens — they are
normalization constants, not physical targets, and the nadir sits just outside
the feasible boundary so no feasible design maps to loss 1. Result (5 seeds):
HV rises monotonically within every seed, reaching 99% of its gen-40 value by
~gen 37 but still creeping ~1-3%/10gen; final-HV mean ~0.075, seed rel-spread
~25% (re-measured after the 7 mm chord bound; was ~18% at the 5 mm bound).
**Verdict: near-converged but seed-dependent** — a single-seed front is
exploration-grade; for decision-grade, aggregate >=5 seeds or raise pop.
**Recommendation: pop 60 / gen 40** (baked into `default.py`) as the
budget-fitting near-converged setting.

> **Re-measured under the corrected root Kt** (see "Phase 4a slice 3b/3c"):
> seed rel-spread fell 25% -> 12.2% and final-HV mean 0.075 -> 0.059. pop 60 /
> gen 40 is retained; the HV bounds above still bracket the new front.

**Constraint activity (the open 2c question, answered).** Each Pareto design's
g-vector is classified binding / slack / violated with a per-constraint
tolerance = 2% of that constraint's own threshold scale (SF->1, thrust->
min_thrust, resonance->margin_min, current->1; a single absolute tol would be
meaningless across differently-scaled g's). Across the seed-0 front: 53/60
designs are fully interior (all constraints slack); only 7/60 bind any
constraint. **What caps throttle:** the low-throttle end (0.36) is capped by
thrust>=min (can't de-rate below 2 N); the high-throttle end (0.85) is jointly
bounded by SF and motor-current (both correlate +0.86 with throttle; the
max-thrust design sits at SF 1.37, I 17/18 A) but is not tightly binding at
2% tol; the bulk of the front is set by the four-way objective trade-off, not
a hard cap. So the answer is "a mix, and mostly not a hard constraint."

**Export + plotting.** The front exports to JSON (`export.py`) — chosen over
CSV because each record is nested/heterogeneous (variable-length decision
vector + objectives + constraint name->g map + activity map); JSON keeps it
self-describing with the config echo, activity summary, and convergence in one
re-readable file. The four objective-corner designs (max thrust / max FoM /
min mass / max safety) export as watertight ASCII STL. Plotting (`plotting.py`)
provides a pairwise objective scatter matrix and a parallel-coordinates plot,
with constraint-binding designs distinguished, saved via the non-interactive
Agg backend. **matplotlib is an OPTIONAL extra** (`pip install -e '.[plot]'`),
imported lazily; absent, plotting raises a clear message and the run still
produces its JSON/STL. Core runtime deps stay `["numpy", "pymoo>=0.6"]`.

**Grade of earlier-phase numbers (retroactive honesty).** Every front shown in
earlier phases — the Phase-2c demo, the A2 before/after (13.3->9.4 N, throttle
0.36-0.85, current 17.9/18 A), and the 2c/3 constraint-activity breakdown —
was produced by a SINGLE seed, i.e. exploration-grade. The qualitative
conclusions hold (rpm is motor-bounded; throttle is the relief valve; the
front is largely constraint-interior; peak thrust drops under the motor
coupling), but the exact quantitative edges carry the ~25% single-seed HV
noise. Decision-grade numbers require the aggregated multi-seed front (decision
mode). Read prior single-seed figures as representative, not final.

## Phase 4a (first slice): CalculiX structural HF path

Purpose: prove the mesher -> deck-writer -> runner -> parser chain end to end on
ONE design, and find out whether meshing the triangle loft will fight us. No
modal, no robustness sweep, no Kt calibration in this slice. `hf/` is a
VALIDATOR path — `CalculiXStructSolver` implements `StructSolver` so it can be
compared like-for-like against the beam, but it is never wired into
`optimize.py` (seconds-to-minutes per case vs ~20 ms for a whole reduced
evaluation).

Toolchain (see `hf/SOURCE.md`): **CalculiX 2.21** via WSL2 + Debian
`calculix-ccx`, **gmsh 4.15.2** via pip under the `hf` extra. `ccx` is a system
binary and cannot be a pip dependency; core deps remain
`["numpy","pymoo>=0.6"]`. The Windows<->WSL boundary is text-only and lives in
one module (`hf/wsl.py`) because OpenFOAM reuses it in 4b: Python writes the
INP, ccx reads it and writes `.dat`/`.frd`, Python parses. Binary location is
configurable (arg / `$PROPOPT_CCX` / WSL PATH / Windows PATH) — no hardcoded
distro or versioned path. **All decks are strict SI** (m, kg, s, N, Pa); the
STL is already metres and nothing rescales it.

### Closed-form anchors (prismatic bar, not the blade) — `tests/test_hf_anchors.py`
| anchor | closed form | converged FE | note |
|---|---|---|---|
| cantilever tip deflection | `qL^4/8EI` | **-0.46%** at 16k tets | converging (-0.79/-0.55/-0.52/-0.46%); lands *below* beam theory because a fully clamped 3-D face also suppresses Poisson contraction and warping |
| rotating-bar axial stress | `0.5*rho*omega^2*(R^2-z^2)` | **within ~1%** at 0.2R and 0.5R | sampled against the analytic profile away from the clamp, so no clamp artifact |

Wall-clock (order-2 tets, one static case): 1k tets ~3 s, 5k ~11 s, 9k ~19 s,
16k ~37-53 s, 101k ~4.3 min. **ccx dominates; meshing is 0.1-14 s.** This is the
number that sizes every later HF cost estimate.

### The blade (decision-mode max-thrust champion, front index 93 of 226)
thrust 10.439 N, rpm 8537, SF_beam 1.293, mass 16.40 g, chord 17.0-29.9 mm.

**4.1 centrifugal only** — an exact comparison (both models compute it
independently, no load mapping involved). Compared as section-integrated axial
force `N = (1/t) * sum(sigma_zz,e * V_e)` over a slab off the clamp, which is
volume-weighted and therefore directly comparable to the beam's `N`:

| grid | tets | gamma mean | dV vs loft | N_fea | vs beam |
|---|---|---|---|---|---|
| coarse | 5,767 | 0.351 | -0.41% | 400.4 N | +2.50% |
| medium | 10,265 | 0.353 | -0.37% | 378.8 N | -3.02% |
| fine | 101,063 | 0.724 | -0.36% | 393.5 N | **+0.74%** |

The centrifugal path (CalculiX `CENTRIF` body load, omega^2, axis, density)
agrees with the beam's analytic `N/A` to **0.74%** on the fine grid. Coarse/
medium scatter +-3% tracks mesh quality, not physics.

**4.2 combined (centrifugal + aero band tractions).** The band mapping
conserves total force exactly: 5.2194 N/blade x 2 = 10.439 N vs the front's
10.439 N. Three-way root stress, fine grid:

| source | root stress | ratio to beam-raw |
|---|---|---|
| beam raw (no Kt) | 15.47 MPa | 1.00x |
| beam x Kt=2.5 | 38.66 MPa | 2.50x |
| CalculiX peak von Mises | 17.89 MPa | **1.16x** |
| CalculiX root-band von Mises | 14.82 MPa | 0.96x |

**THIS COMPARISON IS PARTIAL AND IS NOT A CALIBRATED Kt.** The current geometry
has no root fillet and no hub body, so the FEA resolves concentration from the
section shape and the clamped face only — not the fillet effect that usually
dominates real root concentration. The 1.16x therefore does NOT refute Kt=2.5;
it measures a different, smaller set of mechanisms. Two further error sources
belong to neither model: the band-traction mapping (the beam applies each
station's load as a point resultant on the beam axis; the FEA smears it over
that band's wetted surface), and mesh quality at the trailing edge. Note also
that peak von Mises does not converge monotonically (16.8 / 16.3 / 17.9 MPa) —
it is a local extremum on a sliver-prone mesh — whereas the root-band value is
stable to ~1.5% (14.61 / 14.56 / 14.82). **Use the root-band metric, not the raw
peak, for cross-fidelity comparison.**

### Verdict: is meshing the triangle loft viable?
**Viable but not as-exported — and the blocker is geometric, not a resolution
problem.** Three findings:

1. **The trailing edge is a knife edge.** The default section closes the TE
   exactly (`yt(1)=0`, measured thickness -3.3e-17), so the solid pinches to
   zero thickness along the whole TE. Sliver tets there are unavoidable:
   `gamma_min = 0.000` at EVERY grid and target size tried, and the 1st
   percentile stays <= 0.08 even when the mean reaches 0.72. Refinement does
   not fix it; it only shrinks the slivers. It also drives runaway refinement —
   with gmsh's default min-size of target/5, the "fine" blade grid exploded to
   a 33 MB deck; clamping `min_size_m = target_size_m` fixed that.
2. **The as-exported surface is too coarse and too anisotropic to mesh.** The
   loft writes 6 stations x 25 chordwise points (764 facets), i.e. ~1 mm
   chordwise vs ~15 mm spanwise triangles. Preserve mode cannot refine past
   that boundary (1,250 tets, `gamma_mean = 0.125` — unusable for stress), and
   remesh mode fails outright: `classifySurfaces` cannot form closed loops
   across the knife edges ("1D mesh seems not to be forming a closed loop").
   **Densifying the surface sampling of the same parametric design is what
   makes it work** (30-50 stations x 31-49 chordwise), reaching
   `gamma_mean = 0.72`.
3. **Volume fidelity is good, and preserve mode is bit-exact.** Densified
   sampling agrees with the as-scored loft volume to **-0.36%** (within the 1%
   flag), converging as sampling rises (-0.22% at 20x25 -> -0.018% at 80x81).
   In preserve mode the meshed volume matches the loft's divergence-theorem
   volume to **5.3e-9 relative**, and that residual is entirely the ASCII-STL
   `%.6e` text precision (recomputing the as-written surface reproduces the
   mesh volume exactly) — the mesher does not move the boundary at all.

So meshing is **not** the Phase-4 bottleneck by cost (0.1-14 s vs 12-256 s for
ccx), but it IS a correctness risk until the knife edge is addressed. Two
follow-ons, both deliberately NOT done in this slice: give the HF path a
documented densified-sampling option (surface sampling is currently a
`build_blade_solid` argument the HF path has to drive by hand), and decide the
trailing-edge treatment (a finite TE thickness would remove the sliver floor
entirely). The Phase-2d root-thickness blend is a separate approved change and
was deliberately kept out so that a meshing failure here had exactly one
suspect.

Modal is out of scope this slice: `CalculiXStructSolver` returns the same kind
of unmistakable sentinels `null.py` uses for `first_freq_hz` / `resonance_margin`
(1e9 Hz / 1e6) — **they are not real values**.

## Finite trailing-edge thickness (adopted, 0.4 mm default)

The section previously closed its trailing edge exactly (measured thickness
-3.3e-17), producing a zero-thickness knife edge along the whole span. Phase 4a
proved that is a *geometric* dead end for meshing, not a resolution problem:
`gamma_min = 0.000` at every grid and target size tried, 1st percentile <= 0.08
even at `gamma_mean` 0.72, runaway refinement, and gmsh remesh mode failing
outright ("1D mesh seems not to be forming a closed loop").

`geometry.py` now takes an ABSOLUTE `te_thickness_m` (metres, not %-chord) and
adds a cubic-smoothstep wedge aft of max thickness:
`u = clip((x-x_tmax)/(1-x_tmax),0,1)`, `S(u) = 3u^2-2u^3`, `y_up += (te/2)S`,
`y_lo -= (te/2)S`. `S(0)=S'(0)=0` leaves everything forward of max thickness
bit-for-bit untouched and joins C1 with no kink; `S(1)=1` hits the requested
thickness exactly; `S'(1)=0` meets the blunt face tangentially. Absolute rather
than %-chord because the floor is a manufacturing limit, and a percentage would
leave the small-chord tip stations knife-edged — exactly where meshing is worst.

Topology: the TE point becomes a short flat FACE, so the section polygon gains
one vertex (P = 2k-2 -> 2k-1) and each cap's final column becomes a real quad
instead of a degenerate triangle. `te_thickness_m = 0.0` runs the original code
path and is **bitwise** inert (volume, vertices, faces all identical).

**Adopted default: `te_thickness_m = 4.0e-4` (0.40 mm)** — one nozzle/extrusion
width is the FDM manufacturing floor, and it is where mesh quality resolves
(`gamma_min` 0.000 -> 0.074, `gamma_p01` <= 0.08 -> 0.496, `gamma_mean` 0.72 ->
0.813) without the mass cost of 0.5 mm. It also **unblocks gmsh remesh mode**,
which now succeeds on the as-exported 764-facet STL (~96k tets, -0.17% volume
drift) — so the HF path no longer needs densified re-sampling. Preserve mode
remains poor regardless of TE; that is a surface-sampling problem, a separate
issue from the knife edge.

`chord_bounds` lower bound raised **5 mm -> 7 mm** to match: the geometry guard
rejects chord < te/(0.5*0.12) = 6.67 mm, and a 0.4 mm TE on a 5 mm chord is 8%
of chord — a wedge, not a propeller section. At 7 mm the TE is <= 5.7% of chord
everywhere. Failed evaluations per run: 17 (pre-TE) -> 40 (TE at 5 mm bounds)
-> **10** (TE at 7 mm bounds).

### Known limitation: blunt-TE base drag is NOT in the aero model
The geometry now has a finite trailing edge, but the BEMT polars are XFOIL runs
on a **sharp-TE** NACA 4412 and cannot represent **base drag** — the pressure
deficit behind a blunt TE. BEMT is therefore **optimistic on drag by an
unquantified amount that scales with te/chord**, which at the adopted default
runs 1.6% of chord at the 25 mm root to 5.7% at the 7 mm tip station. Part of
the top-end thrust and safety-factor improvement seen when TE was enabled is
attributable to this omission rather than to real aerodynamic gain, and FoM is
correspondingly optimistic. Quantifying it is a **Phase-4b CFD** task: RANS
resolves base drag natively, so the CFD cross-check measures this directly
rather than requiring a correlation. Until then, treat blunt-TE designs' drag,
power and FoM as optimistic and do not read the TE-enabled thrust/SF gains as
purely physical.

### Fixed: beam-vs-loft mass consistency with TE active (was 11.46%, restored to ~3.3%)
The blunt TE initially broke the beam's span-exact `c^2`/`c^4` averaging and fell
back to a trapezoidal mean of per-node absolute properties, which biased worst-case
`|beam - loft|/loft` from 3.40% (sharp TE) to 11.46% (0.4 mm TE) over 100 random
twisted genomes. Fixed with an exact decomposition, not a tuned correction:

- **Area is exactly linear in TE thickness.** `A(c) = A_base*c^2 + k_area*te*c`
  holds identically (measured dA/d(te_norm) constant to 10 digits), so both
  terms have exact span-averages over a linearly tapered element:
  `mean(c^2) = (ci^2+ci*cj+cj^2)/3`, `mean(c) = (ci+cj)/2`. `k_area` is taken
  from the POLYGON (not the analytic `0.5*(1-x_tmax)`), so the beam stays
  consistent with the loft, which is the faceted polygon and the mass of record.
- **The second moment I has NO closed form** — and this is a real
  distinction, not just more work. `I_centroidal = I0 - Q^2/A`, and with a
  blunt TE, `I0`, `Q`, `A` are polynomials in chord of DIFFERENT degrees, so
  their combination is a RATIONAL function of chord, not a polynomial; no
  finite span-average formula exists. The centroid-shift term `A*ybar^2` is
  ~51% of `I0` for this section, so it cannot be dropped to force a
  polynomial either. Used instead: **3-point Gauss-Legendre** quadrature on
  pointwise-exact values per element. This is exact to degree 5, and the
  sharp-TE case is exactly quartic in chord, so the sharp-TE closed form is
  reproduced EXACTLY (measured rel err 0.00e+00), not merely approximated;
  the rational blunt-TE correction is resolved to ~2.9e-11 against a
  20-point reference.
- **Residual, kept separate from the averaging fix (as it should be):**
  +0.128% between the polygon's numerical `k_area` and the analytic
  `0.5*(1-x_tmax)` — this is n_chord=25 faceting the beam deliberately
  inherits to match the loft, not an averaging error, and it does not
  change with this fix.

Result: worst-case beam-vs-loft mass with TE active is **3.29%** (n=100),
at or below the pre-TE 3.40% baseline. Station-refinement is also tighter at
every count (e.g. 2.51% vs 3.40% at n=4 stations).

**Impact on results is small, and corrects the direction, not just the
magnitude.** Across the exploration front (60/60 designs, seed 0) the
trapezoidal error was NOT optimistic on safety factor — it was mildly
CONSERVATIVE: SF rose (never fell) by up to +2.80% (median +0.16%) under the
exact fix, because the old over-estimated area over-estimated centrifugal
tension and therefore stress. `f1` shifted -1.70% to +0.87% (median -0.48%);
the negative shifts move toward resonance, which is the one genuinely
non-conservative direction here, but no design changed feasibility on either
SF (>=1) or the resonance-margin constraint (>=0.15) at the current bounds —
0 infeasible under either formulation, 0 newly infeasible.

**Exported fronts on disk predate this fix.** `runs/decision/front.json` and
`runs/my_run/front.json` (gitignored, not part of this commit) were produced
before the exact-averaging correction landed; a fresh decision-mode run would
differ slightly — mainly a small relaxation of SF near the constraint
boundary, per the shifts above. Re-run `run_demo.py` for figures that reflect
the corrected beam.

## Phase 4a slice 2: OCC B-rep geometry for the HF path

A fillet is a B-rep operation — it cannot be applied to a triangle mesh — so the
root-fillet work needs a real CAD solid. `hf/occ_geometry.py` builds the blade
with gmsh's **bundled** OpenCASCADE kernel (`gmsh.model.occ`): no new dependency,
no STEP round-trip, and gmsh meshes the B-rep natively so there is no STL
intermediate.

**Two representations of the same design, and which is of record.** The
optimizer keeps using the triangle loft, which remains the **mass of record**;
the HF path uses the OCC solid. They are not different blades — they are the
same station data at different discretization orders, converging to the same
smooth swept limit:

| stations | triangle loft vs smooth | OCC vs smooth |
|---|---|---|
| 4  | +0.943% | -0.470% |
| 8  | +0.631% | -0.104% |
| 16 | +0.347% | -0.0246% |
| 32 | +0.181% | -0.0060% |
| 64 | +0.092% | -0.0015% |

The loft converges **O(1/n)**, OCC **O(1/n^2)**. So the OCC-vs-loft gap is
dominated by the LOFT's first-order error, not by an OCC defect.

**Measured fidelity band** over 20 designs spanning the genome bounds
(te = 0.4 mm): median **-0.0000%**, worst **1.52%**, range -1.52% .. +0.95%,
**0/20 lofting failures**, ~0.12-0.19 s per construction. The difference is
driven by spanwise twist VARIATION and nothing else — with constant twist (any
value) or taper-only, OCC and the loft agree to **0.000000%**, which is what
proves sections, stations, TE treatment, placement and caps are genuinely
shared rather than merely similar. (Note the sign: the loft is LARGER than
smooth, consistent with the Phase-2b measurement that the triangulated loft
encloses more volume than the smooth extrusion.)

**Why the band does not propagate into the optimizer.** The HF path exists to
extract a DIMENSIONLESS ratio — a stress-concentration factor Kt — not absolute
stresses. A ratio taken within one representation is insensitive to a global
volume offset between representations. More specifically, the root is exactly
identical between the two: station 0 is a design station (the hub end-extension
holds chord and twist constant), so the root element's ruled patch is untwisted
and OCC and the loft agree there **exactly**. Measured on a high-twist design,
the root element carries **0.0%** of the total volume difference while holding
15.3% of the volume; the whole gap is mid-span (elements 1-5, -1.12% to -1.20%
each). Root cross-section area agrees analytic == loft cap == OCC kernel face to
**4.4e-16** relative. Kt is therefore read from geometry that is common to both.

**Sharp trailing edges are unsupported in the HF path.** `te_thickness_m = 0`
gives a CUSPED profile wire (upper and lower surfaces meeting at zero included
angle) — the degeneracy a B-rep kernel handles worst, and the same one that made
gmsh's STL remesh mode fail outright in the first 4a slice. Rather than leave it
to fail deep inside OCC with an opaque message, `station_rings` raises
`SolverError` up front. The adopted 0.4 mm default avoids it.

### Fillet geometry validation (slice 3a)

**`occ.getMass` is unreliable on a filleted solid; the mesh is authoritative.**
It is exact on the FUSED solid (agrees with a converged tet mesh to ~3e-6
relative) but wrong once the junction is filleted. Two independent proofs that
this is an artifact rather than real material: the implied dV depends on the
HUB PLATE SIZE (+31.26 mm^3 on a 2.0x1.0-chord plate vs +12.67 mm^3 on a
1.0x0.4 plate — same blade, same junction, same radius, and a *local* fillet
cannot do that), and the wrong value survives a BREP write and re-read in a
fresh session, so it is baked into the shape rather than stale session state.
`build_filleted` therefore returns it as `occ_volume_m3_unreliable`; anything
needing a volume must mesh. A guard test asserts the plate-size dependence, so
if a future gmsh/OCC fixes this the test fails loudly.

**The blend itself is geometrically correct.** Measured from the surface mesh
by cutting planes perpendicular to the junction and fitting a circle through
the blend quadrant — a procedure first validated on a box-on-plate control,
where it recovers the requested radius exactly:

| station | r = 0.100 mm | r = 0.200 mm |
|---|---|---|
| control (box on plate) | 0.10000 (0.00%) | 0.20000 (0.00%) |
| blade mid-chord | 0.10000 (-0.00%) | 0.20004 (+0.02%) |
| blade near-TE | 0.09999 (-0.01%) | 0.19999 (-0.00%) |
| blade LE region | 0.08794 (-12.1%) | 0.17298 (-13.5%) |

**No trailing-edge collapse**: even at r = 0.200 mm = te_thickness_m/2 exactly
— the theoretical limit where the upper and lower blends meet — the near-TE
station still fits the requested radius to 0.00%. The leading-edge fit reads
12-14% low, but that is a MEASUREMENT artifact of fitting a circle across a
sharply curved junction (nose radius 0.473 mm vs r = 0.1-0.2 mm), not real
distortion: the LE band's fillet VOLUME is 0.98x ideal, whereas a genuinely
13%-under-radius blend would carry ~0.76x.

**Fillet volume is uniform and follows the rolling-ball law.** Classifying tets
that lie above the junction plane and outside the root section (the blade
section is exactly constant over that band, since chord and twist are held from
the hub out to the first design station):

| band | volume | share | vs ideal (1-pi/4)r^2 L |
|---|---|---|---|
| LE (u < 0.15) | 0.0822 mm^3 | 16.5% | 0.98x |
| mid (0.15-0.75) | 0.2888 mm^3 | 58.1% | 0.93x |
| TE (u >= 0.75) | 0.1264 mm^3 | 25.4% | 0.95x |
| **total** | **0.4974 mm^3** | | **0.94x** |

There is no dV excess anywhere — every band sits slightly BELOW the ideal
90-degree-corner law, as expected where the blade meets the plate at an obtuse
angle. (An earlier coarse-mesh estimate of ~1.8x was itself mesh error: the
fillet is 0.2 mm and that mesh was 1 mm, so the feature was smaller than one
element.)

**The spline-profile junction path is rejected.** Replacing the faceted
polyline profile with a closed BSpline gives only 2 junction curves instead of
49, but their total length is 122.7 mm = exactly 2x the section perimeter — the
junction loop is represented TWICE. Its `getMass` dV goes negative (a concave
fillet cannot remove material) and its mesh dV is +166 mm^3. The polyline
profile is the sound route.

**Method note carried forward:** never identify post-boolean entities by
diffing tag sets — OCC re-tags modified entities, which silently corrupted an
earlier audit (it reported a 1410 mm "blend radius" and a 46x-too-large face
area). Query post-fillet entities fresh instead.

## Phase 4a slice 3b/3c: measured root Kt, applied to BOTH stress components

`kt_root = 2.5` is gone. The beam now composes the root stress as

```
sigma_root = Kt_axial * (N/A) + Kt_bending * sigma_bend
```

with both factors **measured**, looked up at each design's own root ratio
`r/t`. The tables ship in `src/propopt/kt_root.py`; the instrument that
produced them is `src/propopt/hf/kt2d.py` (needs gmsh + CalculiX, never
imported by the optimizer loop).

### The measured curves

2-D plane strain, CPE6 quadratic triangles, unit depth. Wall of thickness `t`
on a plate, circular fillet radius `r` (i.e. the stepped-flat-bar geometry).
`bending` = pure moment (zero net force, zero shear); `axial` = uniform
tension. Kt = max in-plane principal stress on the fillet arc / sigma_nom read
from the wall itself. `sigma_nom` reproduced the elementary `6M/t^2` and `F/t`
to **0.00% on every row**.

| r/t | Kt bending | Kt axial |
|---|---|---|
| 0.020 | 3.6202 | 5.3453 |
| 0.030 | 3.0740 | 4.5089 |
| 0.040 | 2.7439 | 3.9906 |
| **0.056** | **2.4135** | **3.4698** |
| 0.075 | 2.1692 | 3.0867 |
| 0.100 | 1.9562 | 2.7493 |
| 0.150 | 1.7186 | 2.3384 |
| 0.200 | 1.5769 | 2.1016 |
| 0.300 | 1.4170 | 1.8122 |
| 0.400 | 1.3302 | 1.6427 |
| 0.500 | 1.2742 | 1.5290 |

### Validation basis

* **Kirsch anchor** — the same extraction, mesh strategy and consistent-load
  path run on a circular hole in a wide plate (exact Kt = 3.000) converges
  **monotonically from below**: -3.02% / -1.27% / -1.03% / -0.39% / **-0.19%**
  / -0.03% / +0.03% at `elem_across` = 4 / 6 / 8 / 12 / **16** / 24 / 32, with
  the element count growing throughout (23.4k -> 27.5k). The curves above are
  measured at `elem_across = 16`.
* **Plate ratio saturated** — D/d = 4 / 6 / 8 / 12 give Kt within +/-0.05% of
  one another. D/d = 8 is used.
* **Consistent nodal loads** — 3-point Gauss over the quadratic edge (weights
  L/6, 2L/3, L/6). Tributary-length weighting is wrong for quadratic elements
  and fails a uniform-tension patch test by tens of percent; that bug, not mesh
  resolution, was what produced the earlier Kt = 0.71 reading.
* **r/t = 0.01 is EXCLUDED** — it did not converge (Kt oscillated 4.08-4.78
  across the ladder while the element count barely grew). The table starts at
  0.020.

### How it is applied

The fillet radius is a **manufacturing intent, not a genome variable**:
`RunConfig.root_fillet_radius_m`, default **0.20 mm**. The optimizer cannot
trade it — it must not be able to buy safety factor by wishing for a bigger
blend. Two independent reasons for 0.20 mm: half a 0.40 mm extrusion width is
the smallest blend an FDM tool path can lay, and it is exactly the slice-3a
constructibility bound `min(local thickness)/2 = te/2` for the shipped 0.40 mm
trailing edge, so the fillet is provably buildable on the real solid.

`t` is the design's **computed** root thickness (max of the root section's own
thickness distribution, read from `geometry.section_thickness` — the same seam
the loft and the hub-fillet bound use). Nothing is pinned to one design: across
the 7-30 mm chord bounds the root lands at r/t = 0.056-0.238, always strictly
inside the table.

**What `te_thickness_m` moves, honestly:** it sets the constructibility bound
exactly (`r_max = te/2`, reported per design as
`BeamAnalysis.root_r_max_geom_m`). It does **not** move `t`, and therefore does
not move Kt, anywhere in the admissible range — the TE wedge is a smoothstep
that is C1-flat at the max-thickness station, so it only overtakes the section
maximum at te/c ~ 0.07, and `geometry` already refuses te/c >= 0.06 as "a
wedge, not an aerofoil". The crossover is unreachable by construction. That is
a desirable property: choosing a printable trailing edge does not silently
change the root stress concentration.

**Interpolation** is piecewise linear in `log(Kt - 1)` vs `log(r/t)`: it passes
through every measured point exactly, reduces between points to a local power
law (the form the data follows), and is monotone. The global power-law fits
(`bending 1 + 0.1796 (r/t)^-0.7051`, `axial 1 + 0.3683 (r/t)^-0.6512`) carry
5.9% / 6.8% maximum error and are **deliberately not in the stress path**.

**Out of range RAISES, it does not clamp.** Below 0.020 is exactly where the
model failed to converge; above 0.500 the two face fillets collide (`r > t/2`
is not a filleted wall). Both ends are validity limits, not sampling limits, so
a clamped value would be a number that was never measured.

### What was wrong before

`kt_root = 2.5` multiplied **only** `sigma_bend`; the centrifugal `N/A` was
left bare. Measured at the shipped root ratio r/t = 0.056:

| term | old factor | measured | verdict |
|---|---|---|---|
| bending | 2.5 | **2.41** | placeholder was 4% conservative — a good guess |
| axial | **1.0** | **3.47** | the real gap: centrifugal tension unamplified |

The bending placeholder was fine. The unamplified axial term was not: at the
root of a fast-spinning blade the centrifugal stress is a large share of the
total, and it was carrying no junction concentration at all.

### Impact (decision mode, 5 seeds, pop 60 x gen 40)

The before-front is regenerated from the current config with the old model
patched in, so the **only** difference between the two fronts is the Kt
composition. (`runs/decision/front.json` predates the 7 mm chord floor and
cannot serve as a baseline.)

**Same 229 designs, two stress models** — root r/t 0.0556-0.0923, Kt_axial
2.84-3.48 (was 1.0), Kt_bending 2.01-2.42 (was 2.5):

| | min | median | mean | max |
|---|---|---|---|---|
| SF before | 1.004 | 2.743 | 3.360 | 8.247 |
| SF after | 0.939 | 2.317 | 2.719 | 6.161 |
| ratio | 0.693 | 0.808 | 0.844 | 1.128 |

Median SF falls **19.2%**. Only **2 of 229** designs (0.9%) drop below SF = 1 —
the correction is a broad ~20% haircut, not a cliff. The `ratio > 1` tail is
real: thin-rooted designs sit at higher r/t where the measured bending factor
(2.01) is *below* the old 2.5 placeholder by more than the new axial term costs
them.

**Re-optimized front vs old front** (229 -> 200 designs):

| objective | OLD min | OLD max | NEW min | NEW max | median shift |
|---|---|---|---|---|---|
| thrust_N | 2.0008 | 10.2321 | 2.0011 | 10.2973 | -4.5% |
| figure_of_merit | 0.3675 | 0.6546 | 0.3494 | 0.6502 | +2.4% |
| mass_kg | 0.0038 | 0.0204 | 0.0040 | 0.0195 | -8.5% |
| safety_factor | 1.0043 | 8.2475 | 1.0000 | 6.3034 | -22.5% |

**What the optimizer trades to recover feasibility: rpm, not material.** Median
rpm drops 7.7% (6345 -> 5856) and throttle 5.6%, while median mass *falls* 8.5%
and chord barely moves (-3.3%). Centrifugal stress scales with omega^2, so
under an amplified axial term the cheapest way to buy back safety factor is to
spin slower — and the lost thrust is partly recovered with 7.4% more twist,
which is why figure of merit *rises* 2.4%. The thrust ceiling is untouched
(10.23 -> 10.30 N): the max-thrust corner was never SF-limited.

**HV bounds still bracket the new front** — no rescaling needed:

| objective | ideal | nadir | new best | new worst | headroom to ideal | max loss |
|---|---|---|---|---|---|---|
| thrust_N | 15.0 | 1.5 | 10.2973 | 2.0011 | 34.8% | 0.963 |
| figure_of_merit | 1.0 | 0.0 | 0.6502 | 0.3494 | 35.0% | 0.651 |
| mass_kg | 0.0 | 0.035 | 0.0040 | 0.0195 | 11.4% | 0.558 |
| safety_factor | 10.0 | 0.8 | 6.3034 | 1.0000 | 40.2% | 0.978 |

Every normalized loss lies strictly inside [0,1); **zero** points are pinned at
the nadir. The SF nadir of 0.8 still sits just below the constraint floor as
designed (worst feasible SF = 1.0000, loss 0.978 — tight but not clipped). The
SF ideal of 10 is now 40.2% away from the best achieved design, versus 19.0%
before, so the SF axis carries more unused headroom than the other three; the
bounds remain valid (they are normalization constants, not targets) but the SF
dimension is now the least-exercised. HV falls 0.0919 -> 0.0654, which is the
correct direction: the same search over a genuinely smaller feasible region.

### Convergence re-check under the corrected physics

`n_var` is unchanged but the feasible region shrank, so pop 60 / gen 40 was
re-measured rather than assumed (5 seeds):

| seed | HV gen 1 | gen 20 | gen 30 | gen 40 | monotonic | creep, last 10 gen |
|---|---|---|---|---|---|---|
| 0 | 0.01570 | 0.05385 | 0.05935 | 0.06234 | yes | 5.0% |
| 1 | 0.02010 | 0.05047 | 0.05717 | 0.05957 | yes | 4.2% |
| 2 | 0.01710 | 0.05400 | 0.05972 | 0.06325 | yes | 5.9% |
| 3 | 0.01274 | 0.04664 | 0.05190 | 0.05597 | yes | 7.8% |
| 4 | 0.01859 | 0.05081 | 0.05358 | 0.05630 | yes | 5.1% |

**Final-HV seed spread: 12.2%, down from 25.1%.** The seeds now agree with each
other roughly twice as well — amplifying the axial term sharpened the feasible
region, so there is less for a seed to get lucky about. (`SEED_SPREAD_LIMIT` in
`tests/test_convergence.py` tightened 0.35 -> 0.25 accordingly.)

Each *individual* run is, however, slightly further from its own asymptote than
before. A gen-80 probe (seeds 0-1) puts gen 40 at **88.5%** of the gen-80
hypervolume, with creep 24.3% / 11.7% / 4.6% / 5.2% / 2.5% / 2.9% / 1.9% per
10 generations at gen 20 / 30 / 40 / 50 / 60 / 70 / 80. **pop 60 x gen 40 is
retained** as the exploration-grade setting: it is still the near-converged
point that fits the <=5 min 5-seed budget, and its cross-seed agreement
improved. Runs wanting the last ~10% of hypervolume should raise `n_gen` toward
70-80; that is a config decision and was not made here, since changing it would
invalidate the fronts reported above.

### Outstanding

* **3-D confirmation.** These are 2-D plane-strain numbers. Plane strain holds
  at MID-CHORD, where the blade root is locally a long prismatic wall; near the
  leading and trailing edges the section is short chordwise and the true state
  approaches plane STRESS, which is not covered. Confirming the curve on the
  real filleted 3-D blade is the next structural step.
* **Variable-radius fillet.** A constant radius is bounded by `te/2` at the
  trailing edge, which is where it runs out first. A variable-radius blend
  would allow a much larger (and much less concentrating) fillet over the
  mid-chord region where the stress actually peaks.
