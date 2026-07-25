"""Phase 3: RunConfig validation + reproducibility from config + seed."""
import dataclasses

import numpy as np
import pytest

from propopt.config import build_run, run
from propopt.configs.default import default_config


def _small():
    """A tiny but real motor-mode config for fast end-to-end checks."""
    return dataclasses.replace(default_config(), pop_size=20, n_gen=4)


def test_config_is_frozen_and_validates():
    cfg = default_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 5  # type: ignore[misc]
    with pytest.raises(ValueError):
        dataclasses.replace(cfg, hub_radius_m=cfg.radius_m)      # hub >= tip
    with pytest.raises(ValueError):
        dataclasses.replace(cfg, chord_bounds=(0.03, 0.005))     # lo >= hi
    with pytest.raises(ValueError):
        dataclasses.replace(cfg, hv_ideal=cfg.hv_nadir)          # ideal == nadir


def test_build_run_wires_motor_mode():
    spec, aero, struct = build_run(default_config())
    names = [c.name for c in spec.constraints]
    assert "motor_current<=Imax" in names   # motor mode active
    assert spec.n_obj == 4

    # the beam and the equilibrium share one BEMT instance (memo reuse)
    assert struct._aero is aero


def test_run_is_reproducible_from_config_and_seed():
    cfg = _small()
    a = run(cfg, seed=0)
    b = run(cfg, seed=0)
    assert a.F.shape == b.F.shape
    assert np.array_equal(a.F, b.F)          # bit-identical fronts
    assert np.array_equal(a.X, b.X)
    c = run(cfg, seed=1)
    assert not (a.F.shape == c.F.shape and np.array_equal(a.F, c.F))
