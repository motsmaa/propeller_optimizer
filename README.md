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

## Phase 2b (in progress): reduced structural solver
`solvers/beam.py` replaces the null safety factor with a rotating-cantilever
1-D beam FE on the shared stations (Euler-Bernoulli / Hermite elements,
numpy-only): centrifugal tension (stressing + stress-stiffening), static
bending from the BEMT per-blade spanwise loads (`BEMTAeroSolver.
spanwise_loads`, the aero->structure seam), extreme-fiber stress with a
documented root stress-concentration guard (Kt=2.5, conservative placeholder
until Phase-4 FEA calibrates it), SF = yield/peak. Modal: first flapwise
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
