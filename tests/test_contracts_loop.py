"""Contract pins for the Phase-5 topology split (radial vs loop).

The radial path must be BIT-FOR-BIT unchanged by the introduction of the
`Design` base class. Dataclass inheritance places base fields first, which was
already `PropellerDesign`'s field order, so the positional signature, repr,
hash and equality should all be untouched — but "should" is what these tests
replace. The repr is pinned against a LITERAL rather than a recomputed value:
a test that builds its expectation from the same object it is checking would
pass even if the field order changed.
"""
from __future__ import annotations

import dataclasses

import pytest

from propopt.contracts import (BLADE_TOPOLOGIES, Design, LoopDesign,
                               PropellerDesign)

RADIAL = dict(radius_m=0.127, hub_radius_m=0.01, n_blades=2, rpm=5000.0,
              chord_m=(0.02, 0.018), twist_rad=(0.30, 0.20),
              station_r_m=(0.04, 0.09))


def _radial(**over) -> PropellerDesign:
    return PropellerDesign(**{**RADIAL, **over})


def _loop(**over) -> LoopDesign:
    base = dict(
        radius_m=0.127, hub_radius_m=0.01, n_blades=2, rpm=5000.0,
        topology="loop_hub",
        # starts on the hub cylinder at theta=0 heading outward-and-axial, so
        # the start tangent is NOT parallel to the radial direction
        spine_ctrl_m=((0.01, 0.0, 0.0), (0.05, 0.02, 0.01),
                      (0.05, -0.02, 0.06), (0.0, 0.0, 0.01)),
        station_s=(0.0, 0.5, 1.0),
        chord_m=(0.012, 0.020, 0.012),
        twist_rad=(0.1, 0.2, 0.1),
    )
    return LoopDesign(**{**base, **over})


# --------------------------------------------------------------- radial pins


def test_propeller_design_field_order_is_pinned():
    """Positional construction is used by problem_spec._build and
    validation.uiuc; a reordering would silently transpose arguments."""
    assert [f.name for f in dataclasses.fields(PropellerDesign)] == [
        "radius_m", "hub_radius_m", "n_blades", "rpm",
        "chord_m", "twist_rad", "station_r_m",
    ]


def test_propeller_design_positional_construction_still_works():
    d = PropellerDesign(0.127, 0.01, 2, 5000.0, (0.02, 0.018), (0.30, 0.20),
                        (0.04, 0.09))
    assert d == _radial()


def test_propeller_design_repr_pinned_against_a_literal():
    assert repr(_radial()) == (
        "PropellerDesign(radius_m=0.127, hub_radius_m=0.01, n_blades=2, "
        "rpm=5000.0, chord_m=(0.02, 0.018), twist_rad=(0.3, 0.2), "
        "station_r_m=(0.04, 0.09))"
    )


def test_propeller_design_hash_is_stable_and_equality_holds():
    a, b = _radial(), _radial()
    assert a == b and hash(a) == hash(b)
    assert hash(a) == hash(_radial())          # stable across constructions
    assert a != _radial(rpm=5001.0)


def test_radial_validation_sequence_unchanged():
    with pytest.raises(ValueError, match="radius_m"):
        _radial(radius_m=-1.0)
    with pytest.raises(ValueError, match="n_blades"):
        _radial(n_blades=0)
    with pytest.raises(ValueError, match="strictly increasing"):
        _radial(station_r_m=(0.09, 0.04))
    with pytest.raises(ValueError, match="within"):
        _radial(station_r_m=(0.04, 0.2))


# ------------------------------------------------------- base class is abstract


def test_bare_design_cannot_be_instantiated():
    with pytest.raises(TypeError, match="abstract"):
        Design(0.127, 0.01, 2, 5000.0)


def test_subclasses_are_class_exact_and_never_equal():
    """A cache or solver keyed on one design type must never accept the other,
    even when every shared field agrees."""
    r, l = _radial(), _loop()
    assert isinstance(r, Design) and isinstance(l, Design)
    assert not isinstance(r, LoopDesign) and not isinstance(l, PropellerDesign)
    assert r != l and l != r


# ------------------------------------------------------------------ loop pins


def test_loop_design_accepts_a_well_formed_hub_loop():
    d = _loop()
    assert d.topology in BLADE_TOPOLOGIES
    assert len(d.station_s) == len(d.chord_m) == len(d.twist_rad)


def test_all_three_topology_names_are_well_formed_at_contract_level():
    """Names are well-formed here; BUILDABILITY is the geometry layer's call
    and is reported there with SolverError, preserving the ValueError (malformed)
    vs SolverError (infeasible) layering."""
    for t in BLADE_TOPOLOGIES:
        # non-loop topologies do not carry the hub-endpoint requirement
        _loop(topology=t)
    with pytest.raises(ValueError, match="topology must be one of"):
        _loop(topology="moebius")


def test_station_s_must_be_strictly_increasing_within_unit_interval():
    with pytest.raises(ValueError, match="strictly increasing"):
        _loop(station_s=(0.0, 0.5, 0.5), chord_m=(0.01,) * 3,
              twist_rad=(0.0,) * 3)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        _loop(station_s=(0.0, 0.5, 1.5), chord_m=(0.01,) * 3,
              twist_rad=(0.0,) * 3)


def test_loop_requires_at_least_two_stations():
    with pytest.raises(ValueError, match=">= 2 stations"):
        _loop(station_s=(0.0,), chord_m=(0.01,), twist_rad=(0.0,))


def test_loop_chord_positive_and_twist_finite():
    with pytest.raises(ValueError, match="chord_m"):
        _loop(chord_m=(0.012, 0.0, 0.012))
    with pytest.raises(ValueError, match="twist_rad"):
        _loop(twist_rad=(0.1, float("nan"), 0.1))


def test_loop_hub_radius_must_be_nonneg_and_inside_radius():
    with pytest.raises(ValueError, match="hub_radius_m"):
        _loop(hub_radius_m=-0.001)
    with pytest.raises(ValueError, match="hub_radius_m must be <"):
        _loop(hub_radius_m=0.2)


def test_cubic_spine_needs_four_control_points():
    with pytest.raises(ValueError, match=">= 4 control points"):
        _loop(spine_ctrl_m=((0.01, 0.0, 0.0), (0.05, 0.02, 0.01),
                            (0.0, 0.0, 0.01)))


def test_envelope_and_hub_endpoint_guards_fire_for_hand_built_designs():
    """Unreachable from a genome (loop_spec pins/box-bounds r), retained as a
    guard for hand-built designs."""
    with pytest.raises(ValueError, match="outside radius_m"):
        _loop(spine_ctrl_m=((0.01, 0.0, 0.0), (0.5, 0.02, 0.01),
                            (0.05, -0.02, 0.06), (0.0, 0.0, 0.01)))
    with pytest.raises(ValueError, match="on the hub cylinder"):
        _loop(spine_ctrl_m=((0.02, 0.0, 0.0), (0.05, 0.02, 0.01),
                            (0.05, -0.02, 0.06), (0.0, 0.0, 0.01)))


def test_degenerate_rmf_seed_is_a_value_error():
    """The twist datum is measured from the transported seed normal, so a seed
    that cannot be formed leaves the geometry undefined - malformedness, not
    infeasibility. The degeneracy is now parallelism with the ROTATION AXIS,
    not with the radial direction: the radial version was degenerate exactly at
    the hub-leaving tangent `loop_hub` mandates, and refused a straight radial
    blade outright."""
    # start tangent parallel to the ROTATION AXIS (both control points share
    # their x and z, so P1 - P0 is purely axial)
    with pytest.raises(ValueError, match="parallel to the ROTATION AXIS"):
        _loop(topology="open",
              spine_ctrl_m=((0.02, 0.0, 0.0), (0.02, 0.04, 0.0),
                            (0.06, 0.02, 0.01), (0.07, 0.0, 0.01)))
    # first two control points coincident
    with pytest.raises(ValueError, match="coincide"):
        _loop(topology="open",
              spine_ctrl_m=((0.01, 0.0, 0.0), (0.01, 0.0, 0.0),
                            (0.05, -0.02, 0.06), (0.0, 0.0, 0.01)))


def test_a_tangent_parallel_to_the_RADIAL_direction_is_now_ACCEPTED():
    """The regression that motivated moving the seed onto the axis: a spine
    leaving the hub radially is the canonical blade, and the previous seed
    rejected it as malformed."""
    _loop(topology="open",
          spine_ctrl_m=((0.01, 0.0, 0.0), (0.05, 0.0, 0.0),
                        (0.06, 0.02, 0.01), (0.07, 0.0, 0.01)))


def test_loop_hub_requires_a_positive_hub_radius():
    """Both feet are pinned to the hub cylinder, and a zero-radius cylinder is
    the rotation axis, so the blade would have no azimuth. Checked only in the
    loop_hub branch: other topologies do not pin their endpoints."""
    with pytest.raises(ValueError, match="loop_hub needs hub_radius_m > 0"):
        _loop(topology="loop_hub", hub_radius_m=0.0,
              spine_ctrl_m=((0.0, 0.0, 0.0), (0.05, 0.02, 0.01),
                            (0.05, -0.02, 0.06), (0.0, 0.0, 0.0)))
