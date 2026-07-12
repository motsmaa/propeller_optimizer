import dataclasses

import pytest

from propopt.contracts import (
    AeroSolver,
    EvalResult,
    PropellerDesign,
    StructResult,
    StructSolver,
)
from propopt.solvers.null import NullAeroSolver, NullStructSolver
from propopt.stations import uniform_midpoint
from propopt.units import SanityError


def _design() -> PropellerDesign:
    return PropellerDesign(0.127, 0.01, 2, 8000.0, (0.02, 0.018, 0.015), (0.3, 0.2, 0.1),
                           uniform_midpoint(0.01, 0.127, 3))


def test_design_is_frozen():
    d = _design()
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.rpm = 9000.0  # type: ignore[misc]


def test_design_validates_lengths():
    with pytest.raises(ValueError):
        PropellerDesign(0.127, 0.01, 2, 8000.0, (0.02, 0.018), (0.3,),
                        uniform_midpoint(0.01, 0.127, 2))


def test_design_validates_stations():
    chord, twist = (0.02, 0.018, 0.015), (0.3, 0.2, 0.1)
    with pytest.raises(ValueError):  # length mismatch
        PropellerDesign(0.127, 0.01, 2, 8000.0, chord, twist, (0.03, 0.08))
    with pytest.raises(ValueError):  # not strictly increasing
        PropellerDesign(0.127, 0.01, 2, 8000.0, chord, twist, (0.03, 0.03, 0.08))
    with pytest.raises(ValueError):  # outside [hub, radius]
        PropellerDesign(0.127, 0.01, 2, 8000.0, chord, twist, (0.005, 0.05, 0.10))
    with pytest.raises(ValueError):
        PropellerDesign(0.127, 0.01, 2, 8000.0, chord, twist, (0.03, 0.05, 0.13))
    with pytest.raises(SanityError):  # non-finite
        PropellerDesign(0.127, 0.01, 2, 8000.0, chord, twist, (0.03, float("nan"), 0.10))


def test_abcs_cannot_instantiate():
    with pytest.raises(TypeError):
        AeroSolver()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        StructSolver()  # type: ignore[abstract]


def test_null_solvers_conform_and_pass_sanity():
    d = _design()
    ev = NullAeroSolver().evaluate(d)
    st = NullStructSolver().evaluate(d)
    assert isinstance(ev, EvalResult)
    assert isinstance(st, StructResult)
    assert ev.thrust_N >= 0.0
    assert 0.0 < ev.figure_of_merit <= 1.0
    assert st.mass_kg > 0.0 and st.safety_factor > 0.0


def test_sanity_rejects_bad_values():
    with pytest.raises(SanityError):
        EvalResult(float("nan"), 1.0, 1.0, 0.5)
    with pytest.raises(SanityError):
        EvalResult(1.0, 1.0, 1.0, 1.5)  # FoM out of [0, 1]
    with pytest.raises(SanityError):
        StructResult(-1.0, 2.0, 1e6, 100.0, 0.5)    # negative mass
    with pytest.raises(SanityError):
        StructResult(0.01, 2.0, 1e6, 0.0, 0.5)      # first frequency must be > 0
    with pytest.raises(SanityError):
        StructResult(0.01, 2.0, 1e6, 100.0, -0.1)   # margin must be >= 0
    ok = StructResult(0.01, 2.0, 1e6, 100.0, 0.0)   # on-resonance is representable
    assert ok.resonance_margin == 0.0
