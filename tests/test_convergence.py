"""Phase 3 convergence characterization (SLOW — deselected by default; run
with `pytest -m slow`). Asserts convergence BEHAVIOR, not exact HV values:
cumulative-archive HV is monotonic within each seed, and the final-HV spread
across seeds stays within a stated bound. ~2-3 min at the default config."""
import numpy as np
import pytest

from propopt.analysis import seed_variance
from propopt.configs.default import default_config

pytestmark = pytest.mark.slow

# Measured 5-seed final-HV rel-spread ~18% at pop60/gen40; 0.35 gives headroom
# for run-to-run variance while still catching a broken/divergent run.
SEED_SPREAD_LIMIT = 0.35


def test_hv_monotonic_within_seed_and_seed_spread_bounded():
    cfg = default_config()
    sv = seed_variance(cfg, (0, 1, 2))

    # cumulative feasible-archive HV cannot decrease within a run
    assert all(sv["monotonic"].values()), sv["monotonic"]

    # every seed made real progress (final HV well above the gen-1 value)
    for s, hist in sv["histories"].items():
        h = np.asarray(hist)
        assert h[-1] > 2.0 * h[0], f"seed {s} barely improved: {h[0]:.4f}->{h[-1]:.4f}"

    # fronts are near-converged: final-HV spread across seeds is bounded
    assert sv["final_hv_rel_spread"] < SEED_SPREAD_LIMIT, sv["final_hv_rel_spread"]
