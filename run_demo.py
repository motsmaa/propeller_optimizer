"""Phase 3 entry point: run a config, export the front + plots, print the
constraint-activity summary. Fully reproducible from config + seed.

    python run_demo.py                     # default config, seed 0
    python run_demo.py --seed 2 --out runs/s2
    python run_demo.py --convergence       # + HV-vs-gen seed-variance study (slow)

Plotting needs the optional 'plot' extra (matplotlib); it degrades gracefully
with a message if absent. Nothing here is a runtime dependency beyond
numpy+pymoo.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import dataclasses

from propopt.analysis import aggregate_front, analyze_front, seed_variance
from propopt.config import run
from propopt.configs.default import decision_config, exploration_config
from propopt.export import export_front_json, export_selected_stl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("exploration", "decision"),
                    default="exploration",
                    help="exploration = single seed (demos); decision = "
                         "aggregate >=5 seeds into one cross-seed front")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the exploration seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="override the decision-mode / convergence seed list")
    ap.add_argument("--out", type=Path, default=Path("runs/default"), help="output dir")
    ap.add_argument("--convergence", action="store_true",
                    help="run the multi-seed HV-vs-generation study (slow)")
    args = ap.parse_args()

    cfg = decision_config() if args.mode == "decision" else exploration_config()
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)
    if args.seeds is not None:
        cfg = dataclasses.replace(cfg, seeds=tuple(args.seeds))
    args.out.mkdir(parents=True, exist_ok=True)

    if cfg.mode == "decision":
        res = aggregate_front(cfg, cfg.seeds)
        origin = f"decision, {len(cfg.seeds)} seeds {list(cfg.seeds)}"
    else:
        res = run(cfg, seed=cfg.seed)
        origin = f"exploration, seed {cfg.seed}"
    records, summary = analyze_front(res, cfg, cfg.motor)
    print(f"Front: {len(records)} designs ({origin}, failed evals {res.n_failed})")
    for i, name in enumerate(("thrust_N", "figure_of_merit", "mass_kg", "safety_factor")):
        col = res.F[:, i]
        print(f"  {name:16s} min={col.min():10.4g}  max={col.max():10.4g}")

    print("\nConstraint activity across the front "
          f"(binding = |g| within {cfg.binding_tol_frac:.0%} of the constraint scale):")
    for n, c in summary["constraint_counts"].items():
        print(f"  {n:22s} binding={c['binding']:3d}  slack={c['slack']:3d}  "
              f"violated={c['violated']:3d}")
    if cfg.motor is not None:
        thr = [r.throttle for r in records]
        interior = sum(1 for r in records if all(v != "binding" for v in r.activity.values()))
        print(f"  throttle {min(thr):.2f}-{max(thr):.2f}; "
              f"{interior}/{len(records)} designs are constraint-interior "
              "(throttle set by the objective trade-off, not a hard cap).")
        print(f"  most-binding constraint(s): {summary['dominant_binding'][:2]}")

    convergence = None
    if args.convergence:
        print(f"\nConvergence study, seeds {list(cfg.seeds)} "
              f"(pop {cfg.pop_size} x gen {cfg.n_gen})...")
        convergence = seed_variance(cfg, cfg.seeds)
        print(f"  final-HV mean={convergence['final_hv_mean']:.4f} "
              f"rel-spread={convergence['final_hv_rel_spread']:.1%}  "
              f"monotonic={all(convergence['monotonic'].values())}")

    json_path = export_front_json(args.out / "front.json", cfg, records, summary, convergence)
    stls = export_selected_stl(args.out / "stl", cfg, records)
    print(f"\nWrote {json_path} and {len(stls)} STL(s) under {args.out / 'stl'}")

    try:
        from propopt.plotting import hv_convergence, parallel_coordinates, scatter_matrix
        scatter_matrix(records, args.out / "scatter_matrix.png")
        parallel_coordinates(records, args.out / "parallel_coords.png")
        if convergence is not None:
            hv_convergence(convergence, args.out / "hv_convergence.png")
        print(f"Wrote plots under {args.out}")
    except Exception as exc:  # PlottingUnavailable or any backend issue
        print(f"Plots skipped: {exc}")


if __name__ == "__main__":
    main()
