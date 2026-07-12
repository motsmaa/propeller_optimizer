"""Phase 1: BEMT aero solver — sanity, momentum-theory bound, SolverError paths,
and a trend-based validation anchor (placeholder for real published-prop data)."""
import numpy as np
import pytest

from propopt.contracts import PropellerDesign, SolverError
from propopt.solvers.bemt import BEMTAeroSolver, analytic_polar
from propopt.units import RHO_AIR

N_STATIONS = 6


def make_design(rpm: float = 5000.0, n_blades: int = 2, radius_m: float = 0.127,
                hub_radius_m: float = 0.0127, pitch: float = 0.35) -> PropellerDesign:
    """A representative small prop: tapered chord, roughly constant-pitch twist."""
    r_norm = np.linspace(0.5 / N_STATIONS, 1 - 0.5 / N_STATIONS, N_STATIONS)
    chord = tuple(0.025 - 0.013 * ri for ri in r_norm)
    twist = tuple(float(np.arctan(pitch / (np.pi * ri))) for ri in r_norm)
    return PropellerDesign(radius_m, hub_radius_m, n_blades, rpm, chord, twist)


def test_reasonable_design_is_sane():
    ev = BEMTAeroSolver().evaluate(make_design())
    assert ev.thrust_N > 0.0
    assert 0.0 < ev.figure_of_merit < 1.0
    assert np.isfinite(ev.torque_Nm) and ev.torque_Nm > 0.0
    assert np.isfinite(ev.power_W) and ev.power_W > 0.0
    omega = make_design().rpm * 2.0 * np.pi / 60.0
    assert ev.power_W == pytest.approx(ev.torque_Nm * omega, rel=1e-12)


def test_never_beats_actuator_disk_ideal():
    """Static power must be >= ideal momentum-theory induced power,
    i.e. FoM <= 1, across a sweep of rpm / pitch / blade count."""
    solver = BEMTAeroSolver()
    for rpm in (3000.0, 6000.0, 10000.0):
        for pitch in (0.15, 0.35, 0.55):
            for n_blades in (2, 3):
                d = make_design(rpm=rpm, n_blades=n_blades, pitch=pitch)
                ev = solver.evaluate(d)
                if ev.thrust_N <= 0.0:
                    continue
                p_ideal = ev.thrust_N ** 1.5 / np.sqrt(2.0 * RHO_AIR * np.pi * d.radius_m ** 2)
                assert ev.power_W >= p_ideal
                assert ev.figure_of_merit < 1.0


def test_degenerate_hub_raises_solver_error():
    d = make_design(hub_radius_m=0.127)  # hub == tip: zero-span blade
    with pytest.raises(SolverError):
        BEMTAeroSolver().evaluate(d)


def test_excessive_tip_mach_raises_solver_error():
    d = make_design(rpm=60000.0)  # tip Mach ~2.3 on a 0.127 m radius
    with pytest.raises(SolverError):
        BEMTAeroSolver().evaluate(d)


def test_non_convergence_raises_solver_error():
    with pytest.raises(SolverError, match="converge"):
        BEMTAeroSolver(max_iter=2).evaluate(make_design())


def test_polar_seam_is_swappable():
    """A zero-drag polar through the seam must raise FoM vs the default polar."""
    def frictionless_polar(alpha, reynolds):
        cl, _ = analytic_polar(alpha, reynolds)
        return cl, np.zeros_like(np.asarray(alpha, dtype=float))

    base = BEMTAeroSolver().evaluate(make_design())
    slick = BEMTAeroSolver(polar=frictionless_polar).evaluate(make_design())
    assert slick.figure_of_merit > base.figure_of_merit


def test_validation_anchor_trends():
    """SUPERSEDED as the validation anchor by tests/test_validation_uiuc.py,
    which compares against measured UIUC APC Slow Flyer 10x4.7 static data.
    Kept as a fast, data-free trend check (runs even without data/uiuc/):
      - static thrust scales with rpm^2 (Re/Mach effects absent in this polar);
      - a ~10-inch 2-blade prop at 5000 rpm produces O(1) N of static thrust
        and a hover FoM in the usual 0.3-0.85 band;
      - more blades produce more static thrust at fixed geometry and rpm.
    """
    solver = BEMTAeroSolver()
    e1 = solver.evaluate(make_design(rpm=4000.0))
    e2 = solver.evaluate(make_design(rpm=8000.0))
    assert e2.thrust_N / e1.thrust_N == pytest.approx(4.0, rel=1e-3)

    ev = solver.evaluate(make_design(rpm=5000.0))
    assert 1.0 < ev.thrust_N < 8.0
    assert 0.3 < ev.figure_of_merit < 0.85

    e3 = solver.evaluate(make_design(rpm=5000.0, n_blades=3))
    assert e3.thrust_N > ev.thrust_N
