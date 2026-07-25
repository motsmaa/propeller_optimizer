"""Phase 3: front post-processing (constraint activity) + convergence
characterization (hypervolume vs generation, seed variance).

Uses the frozen PropellerProblem from optimize.py directly for the
convergence runs (with a callback + save-free HV tracking) — optimize.py and
contracts.py are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import OBJECTIVES, RunConfig, build_run
from .motor import MotorSpec
from .optimize import ParetoResult
from .problem_spec import ProblemSpec

# --- hypervolume -------------------------------------------------------------

def normalized_losses(f_phys: np.ndarray, cfg: RunConfig) -> np.ndarray:
    """Map physical objectives (OBJECTIVES order) to a common [0,1] loss,
    0 = ideal, 1 = nadir, using the config's fixed HV bounds. Works for max-
    and min-objectives alike: ideal/nadir bracket the value with a consistent
    sign, so |val-ideal|/|nadir-ideal| in [0,1]."""
    ideal = np.asarray(cfg.hv_ideal, dtype=float)
    nadir = np.asarray(cfg.hv_nadir, dtype=float)
    f = np.atleast_2d(np.asarray(f_phys, dtype=float))
    return np.clip(np.abs(f - ideal) / np.abs(nadir - ideal), 0.0, 1.0)


def hypervolume(f_phys: np.ndarray, cfg: RunConfig) -> float:
    """Normalized hypervolume of a physical front against the fixed nadir
    (ref point = 1 in normalized loss space). Relative metric for convergence,
    not a physical quantity."""
    from pymoo.indicators.hv import HV
    f = np.atleast_2d(np.asarray(f_phys, dtype=float))
    if f.size == 0:
        return 0.0
    losses = normalized_losses(f, cfg)
    return float(HV(ref_point=np.ones(losses.shape[1]))(losses))


# --- constraint activity -----------------------------------------------------

def constraint_scales(spec: ProblemSpec, cfg: RunConfig) -> np.ndarray:
    """Per-constraint threshold scale for the binding tolerance. Each g is
    normalized differently, so a single absolute tolerance is wrong — each
    constraint gets the magnitude of its own threshold:
      safety_factor>=1  -> 1.0  (the SF threshold)
      thrust>=min       -> min_thrust_N
      resonance>=min    -> resonance_margin_min
      current<=Imax     -> 1.0  (g = I/Imax - 1 is already threshold-normalized)
    """
    scales = []
    for c in spec.constraints:
        n = c.name
        if "safety_factor" in n:
            scales.append(1.0)
        elif "thrust" in n:
            scales.append(cfg.min_thrust_N)
        elif "resonance" in n:
            scales.append(cfg.resonance_margin_min)
        elif "current" in n:
            scales.append(1.0)
        else:
            scales.append(1.0)
    return np.asarray(scales, dtype=float)


def classify(g: np.ndarray, scales: np.ndarray, tol_frac: float) -> list[str]:
    """g <= 0 feasible (pymoo convention). Within tol_frac*scale of 0 -> the
    constraint is ACTIVE (binding); comfortably negative -> slack; positive
    beyond tol -> violated."""
    tol = tol_frac * scales
    out = []
    for gi, ti in zip(np.atleast_1d(g), tol):
        if gi > ti:
            out.append("violated")
        elif gi >= -ti:
            out.append("binding")
        else:
            out.append("slack")
    return out


@dataclass(frozen=True)
class DesignRecord:
    x: tuple[float, ...]
    objectives: dict[str, float]      # physical, OBJECTIVES order
    rpm: float
    throttle: float | None            # None in free-rpm mode
    current_a: float | None           # None if no motor
    g: dict[str, float]               # constraint name -> g value
    activity: dict[str, str]          # constraint name -> binding|slack|violated
    feasible: bool


def analyze_front(res: ParetoResult, cfg: RunConfig,
                  motor: MotorSpec | None) -> tuple[list[DesignRecord], dict]:
    """Re-evaluate every Pareto design to recover its constraint g-vector,
    activity classification, and operating point. Returns (records, summary)
    where summary has per-constraint activity counts across the front and the
    answer to 'what caps throttle'."""
    spec, aero, struct = build_run(cfg)
    names = [c.name for c in spec.constraints]
    scales = constraint_scales(spec, cfg)
    records: list[DesignRecord] = []
    counts = {n: {"binding": 0, "slack": 0, "violated": 0} for n in names}

    for x in np.atleast_2d(res.X):
        try:
            design = spec.decode(x)
            ev = aero.evaluate(design)
            st = struct.evaluate(design)
        except Exception:  # a re-eval failure shouldn't be on a feasible front
            continue
        g = spec.eval_constraints(ev, st)
        acts = classify(g, scales, cfg.binding_tol_frac)
        for n, a in zip(names, acts):
            counts[n][a] += 1
        records.append(DesignRecord(
            x=tuple(float(v) for v in x),
            objectives={
                "thrust_N": ev.thrust_N, "figure_of_merit": ev.figure_of_merit,
                "mass_kg": st.mass_kg, "safety_factor": st.safety_factor},
            rpm=design.rpm,
            throttle=float(x[-1]) if motor is not None else None,
            current_a=(motor.current_from_torque(ev.torque_Nm)
                       if motor is not None else None),
            g={n: float(gi) for n, gi in zip(names, g)},
            activity={n: a for n, a in zip(names, acts)},
            feasible=bool(np.all(g <= 0.0)),
        ))

    binding_share = {n: counts[n]["binding"] / max(len(records), 1) for n in names}
    summary = {
        "n_designs": len(records),
        "constraint_counts": counts,
        "binding_share": binding_share,
        # 'what caps throttle': the most-frequently-binding constraint(s)
        "dominant_binding": sorted(names, key=lambda n: -counts[n]["binding"]),
    }
    return records, summary


# --- convergence -------------------------------------------------------------

def _non_dominated_mask(p: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated rows of a minimization set p (n, m)."""
    n = len(p)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = np.all(p <= p[i], axis=1) & np.any(p < p[i], axis=1)
        dominated[i] = False
        mask[i] = not np.any(dominated)
    return mask


def hv_history(cfg: RunConfig, seed: int) -> np.ndarray:
    """Hypervolume of the CUMULATIVE feasible non-dominated archive per
    generation for one seed. Accumulating the best-so-far archive (not the
    current population's front) makes the curve monotonic non-decreasing by
    construction — the correct convergence metric. Runs the frozen
    PropellerProblem with a callback; no save_history (memory-light)."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.callback import Callback
    from pymoo.optimize import minimize

    from .optimize import PropellerProblem

    spec, aero, struct = build_run(cfg)

    class _HV(Callback):
        def __init__(self):
            super().__init__()
            self.hv: list[float] = []
            self.archive: np.ndarray | None = None  # normalized losses (m, k)

        def notify(self, algorithm):
            opt = algorithm.opt
            f = None if opt is None else opt.get("F")
            if f is not None and len(f):
                f = np.atleast_2d(f)
                g = opt.get("G")
                feas = (np.all(np.atleast_2d(g) <= 1e-9, axis=1)
                        if g is not None else np.ones(len(f), bool))
                f = f[feas]
                if len(f):
                    losses = normalized_losses(spec.to_physical(f), cfg)
                    self.archive = (losses if self.archive is None
                                    else np.vstack([self.archive, losses]))
                    self.archive = self.archive[_non_dominated_mask(self.archive)]
            if self.archive is None or not len(self.archive):
                self.hv.append(0.0)
                return
            from pymoo.indicators.hv import HV
            self.hv.append(float(HV(ref_point=np.ones(self.archive.shape[1]))(self.archive)))

    cb = _HV()
    minimize(PropellerProblem(spec, aero, struct),
             NSGA2(pop_size=cfg.pop_size), ("n_gen", cfg.n_gen),
             seed=seed, callback=cb, verbose=False)
    return np.asarray(cb.hv, dtype=float)


def aggregate_results(results: list[ParetoResult], spec: ProblemSpec) -> ParetoResult:
    """Merge per-seed feasible fronts into ONE cross-seed non-dominated front:
    union all designs, then keep the Pareto set across the union. Pure merge
    (no optimization) so tests can reuse already-run per-seed results. HV can
    only rise: the aggregate weakly dominates every constituent front."""
    xs = [np.atleast_2d(r.X) for r in results if len(r)]
    fs = [np.atleast_2d(r.F) for r in results if len(r)]
    n_failed = sum(r.n_failed for r in results)
    if not fs:
        return ParetoResult(np.empty((0, spec.n_var)), np.empty((0, spec.n_obj)),
                            spec.obj_names, n_failed)
    x, f = np.vstack(xs), np.vstack(fs)
    mask = _non_dominated_mask(f * spec.senses)   # physical -> min-form -> ND
    return ParetoResult(x[mask], f[mask], spec.obj_names, n_failed)


def aggregate_front(cfg: RunConfig, seeds: tuple[int, ...] | None = None) -> ParetoResult:
    """Decision-mode front: run the configured seeds (same optimize path as the
    convergence study) and aggregate their feasible fronts. Reuses run() +
    aggregate_results — no parallel optimizer."""
    from .config import run
    seeds = cfg.seeds if seeds is None else seeds
    spec, _, _ = build_run(cfg)
    return aggregate_results([run(cfg, seed=s) for s in seeds], spec)


def seed_variance(cfg: RunConfig, seeds: tuple[int, ...]) -> dict:
    """Run several seeds; report HV-vs-gen per seed and final-HV spread."""
    histories = {s: hv_history(cfg, s) for s in seeds}
    finals = np.array([h[-1] for h in histories.values()])
    mean = float(finals.mean())
    return {
        "seeds": list(seeds),
        "histories": {s: h.tolist() for s, h in histories.items()},
        "final_hv": finals.tolist(),
        "final_hv_mean": mean,
        "final_hv_std": float(finals.std()),
        "final_hv_spread": float(finals.max() - finals.min()),
        "final_hv_rel_spread": float((finals.max() - finals.min()) / mean) if mean else 0.0,
        # monotonic non-decreasing within each seed (allow tiny numerical dips)
        "monotonic": {s: bool(np.all(np.diff(h) >= -1e-9)) for s, h in histories.items()},
    }
