"""Phase 3: constraint-activity classification + hypervolume properties.
Fast — no optimization runs here."""
import numpy as np
import pytest

from propopt.analysis import (
    _non_dominated_mask,
    classify,
    constraint_scales,
    hypervolume,
    normalized_losses,
)
from propopt.configs.default import default_config
from propopt.problem_spec import radial_spec


def test_constraint_scales_are_per_constraint():
    cfg = default_config()
    spec = radial_spec(motor=cfg.motor)
    names = [c.name for c in spec.constraints]
    scales = constraint_scales(spec, cfg)
    smap = dict(zip(names, scales))
    assert smap["safety_factor>=1"] == 1.0
    assert smap["thrust>=min"] == cfg.min_thrust_N          # 2.0, not 1.0
    assert smap["resonance_margin>=min"] == cfg.resonance_margin_min  # 0.15
    assert smap["motor_current<=Imax"] == 1.0


def test_classify_uses_the_right_scale_per_constraint():
    # scales: SF=1, thrust=2, resonance=0.15, current=1; tol_frac=0.02
    scales = np.array([1.0, 2.0, 0.15, 1.0])
    # g just inside each 2% band -> binding; comfortably negative -> slack;
    # positive beyond band -> violated
    g = np.array([-0.01,      # |0.01| < 0.02*1   -> binding
                  -0.5,       # -0.5  < -0.02*2    -> slack
                  0.0027,     # 0.0027 < 0.02*0.15=0.003 -> binding (near +0)
                  0.5])       # 0.5   > 0.02*1     -> violated
    assert classify(g, scales, 0.02) == ["binding", "slack", "binding", "violated"]


def test_hypervolume_bounds_and_monotonicity():
    cfg = default_config()
    ideal = np.array(cfg.hv_ideal)[None, :]
    nadir = np.array(cfg.hv_nadir)[None, :]
    assert hypervolume(ideal, cfg) == pytest.approx(1.0, abs=1e-9)  # ideal fills unit cube
    assert hypervolume(nadir, cfg) == pytest.approx(0.0, abs=1e-9)  # nadir -> zero
    # adding a non-dominated point never decreases HV
    mid = 0.5 * (ideal + nadir)
    hv1 = hypervolume(mid, cfg)
    better = np.vstack([mid, 0.25 * ideal + 0.75 * nadir + 0.0])  # a spread point
    assert hypervolume(better, cfg) >= hv1 - 1e-12
    assert 0.0 < hv1 < 1.0


def test_normalized_losses_direction():
    cfg = default_config()
    # a strong design: high thrust, high FoM, low mass, high SF -> small loss
    strong = np.array([[12.0, 0.6, 0.004, 6.0]])
    weak = np.array([[2.1, 0.32, 0.018, 1.05]])
    assert normalized_losses(strong, cfg).sum() < normalized_losses(weak, cfg).sum()


def test_non_dominated_mask():
    # minimization: (0,1) and (1,0) non-dominated; (2,2) dominated
    p = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [0.5, 0.5]])
    mask = _non_dominated_mask(p)
    assert list(mask) == [True, True, False, True]  # (0.5,0.5) not dominated by (0,1)/(1,0)
    dom = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert list(_non_dominated_mask(dom)) == [True, False]
