# propopt — Phase 0 (scaffolding + frozen contracts)

Thin end-to-end slice: NSGA-II runs against placeholder evaluators through the
frozen interface contracts, before any real physics exists.

## Layout
```
src/propopt/
  contracts.py      PropellerDesign, EvalResult, StructResult, ABCs, SolverError (SI)
  units.py          SI conventions + sanity-check assertions
  problem_spec.py   config-driven spec; objectives bundle (name, sense, fn)
  optimize.py       pymoo NSGA-II wiring + failed-eval penalty policy
  solvers/null.py   placeholder aero/struct (kept as tests/reference)
  solvers/bemt.py   Phase 1: static/hover BEMT aero, swappable polar seam
  solvers/polars.py TabulatedPolar: Re-interpolating XFOIL table polar
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

**Open decision (A2, do not implement without discussion):** `rpm` is still a
free design variable with no motor coupling. The alternative — solving rpm from
a motor-prop torque equilibrium — changes the genome and possibly the
`evaluate` signature (explicit operating-condition argument). Deferred.
