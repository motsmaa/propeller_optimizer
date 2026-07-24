"""Phase-2c anchors: the motor model + torque-equilibrium root-find against
closed form. The equilibrium is validated with an ANALYTIC prop law (Q = k N^2)
so it is proven independently of BEMT."""
import math

import numpy as np
import pytest

from propopt.contracts import SolverError
from propopt.motor import DEFAULT_MOTOR, KT_CONST, MotorSpec, solve_torque_equilibrium
from propopt.problem_spec import radial_spec
from propopt.solvers.beam import BeamStructSolver
from propopt.solvers.bemt import BEMTAeroSolver

M = MotorSpec("test", kv_rpm_per_v=1000.0, rm_ohm=0.10, i0_a=0.5, v_max_v=12.0, i_max_a=25.0)


def _analytic_equilibrium(motor, throttle, k):
    """Positive root of k N^2 = Q_motor(N): k N^2 + (Kt/(KV Rm)) N - Q_stall = 0."""
    kt = motor.kt_nm_per_a
    b = kt / (motor.kv_rpm_per_v * motor.rm_ohm)
    q_stall = motor.stall_torque(throttle)
    return (-b + math.sqrt(b ** 2 + 4.0 * k * q_stall)) / (2.0 * k)


def test_kt_constant():
    assert KT_CONST == pytest.approx(9.5493, rel=1e-4)
    assert M.kt_nm_per_a == pytest.approx(9.5493 / M.kv_rpm_per_v, rel=1e-4)


def test_stall_and_noload_closed_form():
    thr = 0.8
    v = thr * M.v_max_v
    assert M.stall_torque(thr) == pytest.approx(M.kt_nm_per_a * (v / M.rm_ohm - M.i0_a), rel=1e-12)
    assert M.no_load_speed(thr) == pytest.approx(M.kv_rpm_per_v * (v - M.i0_a * M.rm_ohm), rel=1e-12)
    # torque vanishes exactly at no-load speed
    assert M.torque(M.no_load_speed(thr), thr) == pytest.approx(0.0, abs=1e-12)


def test_no_load_equilibrium_with_zero_prop_torque():
    """With Q_prop == 0 the equilibrium is exactly the no-load speed."""
    for thr in (0.3, 0.6, 1.0):
        n = solve_torque_equilibrium(M, thr, lambda rpm: 0.0)
        assert n == pytest.approx(M.no_load_speed(thr), rel=1e-6)


def test_equilibrium_matches_analytic_quadratic():
    for thr in (0.4, 0.7, 1.0):
        for k in (1e-9, 5e-9, 2e-8):
            n = solve_torque_equilibrium(M, thr, lambda rpm: k * rpm ** 2)
            assert n == pytest.approx(_analytic_equilibrium(M, thr, k), rel=1e-6)
            # torque residual is tiny vs the torque scale (rtol bounds rpm, so
            # the residual is g'-scaled — measure it against stall torque)
            assert abs(k * n ** 2 - M.torque(n, thr)) < 1e-6 * M.stall_torque(thr)


def test_motor_efficiency_below_one():
    thr = 1.0
    n0 = M.no_load_speed(thr)
    for n in np.linspace(0.1 * n0, 0.95 * n0, 15):
        omega = n * 2.0 * math.pi / 60.0
        eta = M.torque(n, thr) * omega / (M.voltage(thr) * M.current(n, thr))
        assert 0.0 < eta < 1.0


def test_monotonic_throttle_and_blade():
    k = 1e-8
    n_lo_thr = solve_torque_equilibrium(M, 0.4, lambda rpm: k * rpm ** 2)
    n_hi_thr = solve_torque_equilibrium(M, 0.9, lambda rpm: k * rpm ** 2)
    assert n_hi_thr > n_lo_thr  # more throttle -> higher equilibrium speed

    n_small = solve_torque_equilibrium(M, 0.8, lambda rpm: 5e-9 * rpm ** 2)
    n_big = solve_torque_equilibrium(M, 0.8, lambda rpm: 5e-8 * rpm ** 2)
    assert n_big < n_small  # more aggressive/larger blade -> lower speed


def test_no_equilibrium_raises_solver_error():
    # blade so heavy the motor cannot spin it up (Q_prop > Q_motor near stall)
    with pytest.raises(SolverError):
        solve_torque_equilibrium(M, 1.0, lambda rpm: 1.0 * rpm ** 2)
    # throttle too low to turn at all (no-load speed <= floor)
    with pytest.raises(SolverError):
        solve_torque_equilibrium(M, 1e-4, lambda rpm: 1e-9 * rpm ** 2)


def test_overcurrent_is_recoverable_via_current():
    """A heavy prop within a low current limit reaches an equilibrium (not a
    crash), but the recovered operating current exceeds I_max — the case-(b)
    signal the problem_spec constraint turns into infeasibility."""
    weak = MotorSpec("weak", 1000.0, 0.10, 0.5, 12.0, i_max_a=8.0)
    k = 4e-8
    n = solve_torque_equilibrium(weak, 1.0, lambda rpm: k * rpm ** 2)   # exists
    q_eq = k * n ** 2
    assert weak.current_from_torque(q_eq) > weak.i_max_a                # over limit


def test_current_recovered_from_prop_torque_matches_direct():
    """The constraint recovers I from ev.torque_Nm alone: at equilibrium that
    must equal the motor's direct I(N)."""
    thr, k = 0.85, 1e-8
    n = solve_torque_equilibrium(M, thr, lambda rpm: k * rpm ** 2)
    # convergence-limited: the discrepancy is the root-find residual g(n)/Kt,
    # bounded by the bracket width, not an identity error
    assert M.current_from_torque(k * n ** 2) == pytest.approx(M.current(n, thr), rel=1e-6)


def test_radial_spec_motor_mode_decodes_feasible_rpm():
    """Integration: motor-mode decode returns a solved rpm and a well-formed
    4-constraint spec whose current constraint is finite."""
    spec = radial_spec(motor=DEFAULT_MOTOR)
    assert [c.name for c in spec.constraints][-1] == "motor_current<=Imax"
    x = 0.5 * (spec.xl + spec.xu)  # mid-box genome, throttle 0.6
    design = spec.decode(x)
    assert 100.0 < design.rpm < DEFAULT_MOTOR.no_load_speed(1.0)
    g = spec.eval_constraints(BEMTAeroSolver().evaluate(design),
                              BeamStructSolver().evaluate(design))
    assert np.all(np.isfinite(g)) and len(g) == 4
