"""The rotation-sense cross-check that did not exist when it was needed.

A propeller running BACKWARDS still produces thrust toward its suction side, so
the thrust SIGN looks correct and nothing downstream flags it. It cost a full
Phase-4b slice: every CFD number produced before this guard was invalid
(measured CT -44% / CP -13% against UIUC, the classic signature of a reversed
section - lift collapses, drag does not).

Nothing else in propopt carries a rotation sense: BEMT uses |Omega r| and the
structural path uses centrifugal magnitude. The MRF dictionary is the only
place the sense exists, so it is the only place it can be wrong, and it had no
cross-check at all.
"""
import numpy as np
import pytest

from propopt.contracts import SolverError
from propopt.validation.uiuc import design_from_geometry, load_geometry


def _has_gmsh() -> bool:
    try:
        import gmsh  # noqa: F401
        return True
    except ImportError:
        return False


needs_gmsh = pytest.mark.skipif(not _has_gmsh(), reason="gmsh not installed")

UIUC_GEOM = "data/uiuc/apcsf_10x4.7_geom.txt"


def _design(rpm=4590.0, n=20):
    from pathlib import Path
    return design_from_geometry(load_geometry(Path(UIUC_GEOM)), 0.254, 2, rpm,
                                n_stations=n)


@needs_gmsh
def test_blade_advances_leading_edge_first_at_every_station():
    """cos(chord LE->TE, direction of travel) must be NEGATIVE everywhere.

    EVERY station, not just mid-span: a twist-SIGN error would leave a correct
    mid-span reading while reversing the root or the tip."""
    from propopt.hf.hover_case import HoverCaseConfig, leading_edge_first_cos
    cos = leading_edge_first_cos(_design(), HoverCaseConfig(rpm=4590.0))
    assert cos.size >= 20
    assert np.all(cos < 0.0), f"trailing edge leads at {np.flatnonzero(cos >= 0)}"


@needs_gmsh
def test_cos_tracks_the_local_twist():
    """The check must be measuring geometry, not returning a constant. With the
    chord mostly tangential, cos ~ -cos(twist), so it must vary monotonically
    from the high-twist root to the low-twist tip."""
    from propopt.hf.hover_case import HoverCaseConfig, leading_edge_first_cos
    d = _design()
    cos = leading_edge_first_cos(d, HoverCaseConfig(rpm=4590.0))
    twist = np.asarray(d.twist_rad)
    # root is the most twisted -> cos closest to 0; tip least -> closest to -1
    assert cos[0] > cos[-1]
    assert abs(cos[len(cos) // 2]) == pytest.approx(
        float(np.cos(twist[len(twist) // 2])), rel=0.05)


@needs_gmsh
def test_positive_omega_is_rejected():
    """THE regression. A positive omega about +y drives the blade
    trailing-edge-first and must be refused, loudly."""
    import dataclasses

    from propopt.hf.hover_case import (HoverCaseConfig,
                                       assert_leading_edge_first,
                                       leading_edge_first_cos)

    class Backwards(HoverCaseConfig):
        def omega_signed(self):           # the pre-fix behaviour
            return +self.omega_rad_s()

    bad = Backwards(rpm=4590.0)
    assert np.all(leading_edge_first_cos(_design(), bad) > 0.0)
    with pytest.raises(SolverError, match="ROTATION IS BACKWARDS"):
        assert_leading_edge_first(_design(), bad)


@needs_gmsh
def test_signed_omega_is_negative_and_matches_rpm():
    from propopt.hf.hover_case import HoverCaseConfig
    c = HoverCaseConfig(rpm=4590.0)
    assert c.omega_signed() == pytest.approx(-c.omega_rad_s())
    assert c.omega_rad_s() == pytest.approx(4590.0 * 2 * np.pi / 60.0)


@needs_gmsh
def test_build_refuses_a_backwards_case():
    """The guard must fire from build_hover_case, before any file is written —
    not merely be available as a helper nobody calls."""
    import inspect

    from propopt.hf import hover_case
    src = inspect.getsource(hover_case.build_hover_case)
    assert "assert_leading_edge_first" in src, (
        "build_hover_case no longer calls the rotation guard")


@needs_gmsh
def test_incidence_is_positive_at_every_station():
    """LE-ordering is NOT sufficient: a blade can lead with its nose and still
    sit at negative incidence. Angle of attack must be positive everywhere, and
    for a static rotor with no inflow it should equal the local twist."""
    from propopt.hf.hover_case import HoverCaseConfig, signed_incidence_deg
    d = _design()
    a = signed_incidence_deg(d, HoverCaseConfig(rpm=4590.0))
    assert np.all(a > 0.0), f"negative incidence at {np.flatnonzero(a <= 0)}"
    twist = np.degrees(np.asarray(d.twist_rad))
    assert a[len(a) // 2] == pytest.approx(twist[len(twist) // 2], abs=0.2)


@needs_gmsh
def test_backwards_omega_fails_the_incidence_guard_too():
    """The known-bad configuration must fail BOTH guards, not just the
    LE-ordering one — that is what makes the incidence check load-bearing."""
    from propopt.hf.hover_case import (HoverCaseConfig,
                                       assert_positive_incidence,
                                       signed_incidence_deg)

    class Backwards(HoverCaseConfig):
        def omega_signed(self):
            return +self.omega_rad_s()

    bad = Backwards(rpm=4590.0)
    assert np.all(signed_incidence_deg(_design(), bad) < 0.0)
    with pytest.raises(SolverError, match="NEGATIVE INCIDENCE"):
        assert_positive_incidence(_design(), bad)


@needs_gmsh
def test_suction_normal_is_perpendicular_to_the_chord():
    """The sign of the incidence depends entirely on sampling the upper and
    lower surface at the SAME chordwise index. Mismatched indices give a vector
    nearly parallel to the chord and a sign-flipped alpha — that error caused a
    wrong diagnosis and a killed run, so it is asserted, not assumed."""
    import numpy as np

    from propopt.geometry import DEFAULT_SECTION
    from propopt.hf.occ_geometry import station_rings
    rings = station_rings(_design(), DEFAULT_SECTION, 25, 2.0e-4)
    k, p = 25, 2 * 25 - 1
    for st in rings:
        chord = (0.5 * (st[k - 1] + st[k]) - st[0])[:2]
        chord = chord / np.linalg.norm(chord)
        j = k // 2
        n = st[p - j][:2] - st[j][:2]
        n = n / np.linalg.norm(n)
        assert abs(float(np.dot(chord, n))) < 1e-6
