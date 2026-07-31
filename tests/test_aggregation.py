"""Phase 3: decision-mode cross-seed front aggregation. Fast (small runs)."""
import dataclasses

import numpy as np

from propopt.analysis import _non_dominated_mask, aggregate_results, hypervolume
from propopt.config import build_run, run
from propopt.configs.default import decision_config


def test_decision_aggregation_dominates_each_seed():
    cfg = dataclasses.replace(decision_config(), pop_size=20, n_gen=5, seeds=(0, 1))
    spec, _, _ = build_run(cfg)

    per_seed = [run(cfg, seed=s) for s in cfg.seeds]        # run each seed once
    hv_each = [hypervolume(r.F, cfg) for r in per_seed]
    agg = aggregate_results(per_seed, spec)                 # pure merge, no re-run

    # (a) the aggregate is non-dominated across the union it was built from
    assert len(agg) > 0
    assert _non_dominated_mask(agg.F * spec.senses).all()
    # every aggregate design came from some seed's front
    union_f = np.vstack([np.atleast_2d(r.F) for r in per_seed])
    for row in agg.F:
        assert np.any(np.all(np.isclose(union_f, row), axis=1))

    # (b) aggregation cannot lose designs: HV is no worse than any single seed
    hv_agg = hypervolume(agg.F, cfg)
    assert hv_agg >= max(hv_each) - 1e-9


def test_mode_selects_seed_handling_only():
    """Exploration and decision configs are identical except mode/seed-handling
    — same physics, bounds, optimizer, constraints."""
    ex = decision_config()
    from propopt.configs.default import exploration_config
    xp = exploration_config()
    shared = ("n_stations", "radius_m", "hub_radius_m", "n_blades", "chord_bounds",
              "twist_bounds", "throttle_bounds", "motor", "material",
              "root_fillet_radius_m", "te_thickness_m",
              "min_thrust_N", "resonance_margin_min", "pop_size", "n_gen",
              "hv_ideal", "hv_nadir", "binding_tol_frac")
    for f in shared:
        assert getattr(ex, f) == getattr(xp, f), f
    assert (xp.mode, ex.mode) == ("exploration", "decision")
