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
from propopt.units import SanityError


def _design() -> PropellerDesign:
    return PropellerDesign(0.127, 0.01, 2, 8000.0, (0.02, 0.018, 0.015), (0.3, 0.2, 0.1))


def test_design_is_frozen():
    d = _design()
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.rpm = 9000.0  # type: ignore[misc]


def test_design_validates_lengths():
    with pytest.raises(ValueError):
        PropellerDesign(0.127, 0.01, 2, 8000.0, (0.02, 0.018), (0.3,))


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
        StructResult(-1.0, 2.0, 1e6)    # negative mass
