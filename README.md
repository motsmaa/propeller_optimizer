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
  solvers/null.py   placeholder aero/struct (replaced in Phases 1-2)
tests/              contracts, smoke (DoD), reproducibility, robustness
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

## Next (Phase 1)
Replace `NullAeroSolver` with a custom BEMT solver behind the same `AeroSolver`
ABC, raising `SolverError` on non-convergence; validate against published
propeller data. Open decisions flagged for Phase 1: motor model + operating
condition (does rpm stay a design variable or become a solved equilibrium), and
whether `evaluate` gains an explicit operating-condition argument.
