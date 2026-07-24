"""Phase 2c demonstrator: NSGA-II with the motor-propeller torque equilibrium.

rpm is no longer in the genome — throttle is the operating variable, and each
design's rpm is solved from Q_motor(N) == Q_prop(N) (motor.py) inside decode.
All four objectives are physical (BEMT thrust/FoM, lofted-solid mass, beam SF).

Reports the THROTTLE distribution across the front (the A2 relief-valve
analogue of the pre-2c free-rpm diagnostic) and the resulting equilibrium-rpm
distribution, and contrasts the motor-coupled front with the free-rpm front.
One shared BEMTAeroSolver is bound into decode, aero, and the beam so the
equilibrium root-find's final solve is reused (see the memo in bemt.py)."""
import numpy as np

from propopt.contracts import SolverError
from propopt.motor import DEFAULT_MOTOR
from propopt.optimize import optimize
from propopt.problem_spec import radial_spec
from propopt.solvers.beam import BeamStructSolver
from propopt.solvers.bemt import BEMTAeroSolver


def _summary(res, spec):
    for i, name in enumerate(spec.obj_names):
        col = res.F[:, i]
        print(f"    {name:16s} min={col.min():10.4g}  max={col.max():10.4g}")


def main() -> None:
    aero = BEMTAeroSolver()                 # shared -> memo reuses the root-find solve
    beam = BeamStructSolver(aero=aero)
    spec = radial_spec(motor=DEFAULT_MOTOR, aero_for_rpm=aero)
    res = optimize(spec, aero, beam, pop_size=60, n_gen=40, seed=0)

    print(f"MOTOR-COUPLED front ({DEFAULT_MOTOR.name}):")
    print(f"  Pareto points: {len(res)}   failed evals: {res.n_failed}")
    _summary(res, spec)

    throttles, rpms, currents = [], [], []
    for x in res.X:
        design = spec.decode(x)             # re-solve equilibrium for reporting
        try:
            ev = aero.evaluate(design)
        except SolverError:
            continue
        throttles.append(float(x[-1]))
        rpms.append(design.rpm)
        currents.append(DEFAULT_MOTOR.current_from_torque(ev.torque_Nm))
    throttles, rpms, currents = map(np.array, (throttles, rpms, currents))
    full = int(np.sum(throttles > 0.99))
    print(f"  throttle: min={throttles.min():.2f} median={np.median(throttles):.2f} "
          f"max={throttles.max():.2f};  {full}/{len(throttles)} at full throttle "
          f"({len(throttles) - full} de-rated)")
    print(f"  equilibrium rpm: {rpms.min():.0f}-{rpms.max():.0f} "
          f"(median {np.median(rpms):.0f});  current {currents.min():.1f}-"
          f"{currents.max():.1f} A (I_max={DEFAULT_MOTOR.i_max_a})")

    # contrast: pre-2c free-rpm front (rpm is a genome variable)
    free = radial_spec(motor=None)
    res_free = optimize(free, BEMTAeroSolver(), BeamStructSolver(),
                        pop_size=60, n_gen=40, seed=0)
    print(f"\nFREE-RPM front (pre-2c, rpm in genome): {len(res_free)} points, "
          f"{res_free.n_failed} failed")
    _summary(res_free, free)
    print(f"  thrust max {res_free.F[:, 0].max():.2f} N vs motor-coupled "
          f"{res.F[:, 0].max():.2f} N")


if __name__ == "__main__":
    main()
