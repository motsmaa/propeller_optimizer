"""Phase-5 slice 5a GATE: watertight-and-meshable pass rate over random loop genomes.

The deliverable is one number, but a bare rate cannot distinguish "the topology
is wrong" from "the sampler is bad", and those have different fallbacks. So
every candidate is classified into exactly one bucket and the distribution is
reported alongside the rate.

STAGES, in order; a candidate stops at its first failure:
  decode        genome -> LoopDesign (seed conditioning, unimplemented topology)
  feasibility   the geometric penalties, evaluated but NOT gating (recorded so
                the correlation between infeasibility and sweep failure is
                visible; a sweep that succeeds on a self-intersecting spine is
                a silently wrong solid, which matters more than a failure)
  sweep         OCC addThruSections -> single solid, positive kernel volume
  watertight    trimesh, INDEPENDENTLY: is_watertight, is_winding_consistent,
                and volume agreement with the OCC kernel. The generator does
                not grade its own homework.
  meshable      3-D tet mesh; Phase 4a got gamma_min = 0.000 from a watertight
                surface with a knife edge, so watertight is not meshable.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from propopt.contracts import SolverError
from propopt.problem_spec import loop_spec

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=240)
ap.add_argument("--seed", type=int, default=20260908)
ap.add_argument("--n-ctrl", type=int, default=5)
ap.add_argument("--loft", type=int, default=40)
ap.add_argument("--mesh-every", type=int, default=1)
ap.add_argument("--target", type=float, default=1.5e-3)
ap.add_argument("--out", default="runs/loop_gate.json")
# DIAGNOSTIC, not the headline. Rejection-samples until the geometric penalties
# are satisfied, so the pass rate is CONDITIONAL on feasibility. This does not
# tune any bound: the box is unchanged and the generator is unchanged; it only
# asks what the sweep does on the designs an optimizer's penalties would steer
# it toward, which is the question the headline rate cannot answer.
ap.add_argument("--require-feasible", action="store_true")
ap.add_argument("--max-draws", type=int, default=4000)
A = ap.parse_args()

try:
    import trimesh
    HAVE_TRIMESH = True
except ImportError:
    HAVE_TRIMESH = False

from propopt.hf.loop_sweep import mesh_swept_blade, sweep_blade  # noqa: E402

SPEC = loop_spec(n_ctrl=A.n_ctrl)
RNG = np.random.default_rng(A.seed)
GEOM_NAMES = SPEC.geom_names

rows = []
buckets: Counter[str] = Counter()
decode_reasons: Counter[str] = Counter()
sweep_reasons: Counter[str] = Counter()
mesh_reasons: Counter[str] = Counter()
wt_reasons: Counter[str] = Counter()
tmp = Path(tempfile.mkdtemp(prefix="loopgate_"))


def short(e: Exception, n: int = 60) -> str:
    s = str(e).replace("\n", " ")
    for key in ("ill-conditioned: the start tangent is within",
                "ill-conditioned: |P1-P0|", "cusp / zero tangent",
                "reverses direction", "not implemented in slice 5a",
                "non-positive-volume tets", "volume drifted",
                "produced no order-1 tetrahedra", "3-D meshing failed",
                "surface meshing failed", "OCC sweep failed",
                "expected 1", "non-physical volume"):
        if key in s:
            return key
    return s[:n]


draws = 0
i = 0
while i < A.n:
    x = SPEC.xl + (SPEC.xu - SPEC.xl) * RNG.random(SPEC.n_var)
    draws += 1
    if draws > A.max_draws:
        print(f"  stopped: {A.max_draws} draws yielded only {i} accepted")
        break
    rec: dict = {"i": i}
    try:
        design = SPEC.decode(x)
    except SolverError as e:
        if A.require_feasible:
            continue
        i += 1
        buckets["decode"] += 1
        decode_reasons[short(e)] += 1
        rows.append({**rec, "stage": "decode", "why": short(e)})
        continue

    try:
        g = SPEC.eval_geom(design)
        rec["geom"] = {n: float(v) for n, v in zip(GEOM_NAMES, g)}
        rec["geom_feasible"] = bool(np.all(g <= 0.0))
    except SolverError as e:
        if A.require_feasible:
            continue
        i += 1
        buckets["feasibility"] += 1
        rows.append({**rec, "stage": "feasibility", "why": short(e)})
        continue
    if A.require_feasible and not rec["geom_feasible"]:
        continue
    i += 1

    try:
        blade = sweep_blade(design, n_loft_sections=A.loft, surface=HAVE_TRIMESH)
        rec.update(volume=blade.volume_m3, length=blade.length_m,
                   min_gap=blade.min_section_gap_m)
    except SolverError as e:
        buckets["sweep"] += 1
        sweep_reasons[short(e)] += 1
        rows.append({**rec, "stage": "sweep", "why": short(e)})
        continue

    if HAVE_TRIMESH and blade.surface_nodes is not None:
        # process=False: no vertex merging, no repair — the same convention
        # tests/test_geometry.py uses for the triangle loft.
        m = trimesh.Trimesh(vertices=blade.surface_nodes,
                            faces=blade.surface_tris, process=False)
        rec["tm_watertight"] = bool(m.is_watertight)
        rec["tm_winding_ok"] = bool(m.is_winding_consistent)
        rec["tm_volume"] = float(m.volume)
        rec["tm_vol_rel"] = (float(m.volume) / blade.volume_m3 - 1.0
                             if blade.volume_m3 else float("nan"))
        why = None
        if not m.is_watertight:
            why = "trimesh: not watertight"
        elif not m.is_winding_consistent:
            why = "trimesh: inconsistent winding"
        elif abs(rec["tm_vol_rel"]) > 0.02:
            why = "trimesh: volume disagrees with OCC >2%%"
        if why:
            buckets["not_watertight"] += 1
            wt_reasons[why] += 1
            rows.append({**rec, "stage": "watertight", "why": why})
            continue

    if i % A.mesh_every == 0:
        try:
            mesh = mesh_swept_blade(design, A.target, n_loft_sections=A.loft,
                                    reference_volume_m3=blade.volume_m3)
            rec.update(gamma_min=mesh.quality_min, gamma_p01=mesh.quality_p01,
                       gamma_mean=mesh.quality_mean, n_tets=mesh.n_tets,
                       vol_drift=mesh.volume_rel_error)
        except SolverError as e:
            buckets["unmeshable"] += 1
            mesh_reasons[short(e)] += 1
            rows.append({**rec, "stage": "unmeshable", "why": short(e)})
            continue

    buckets["pass"] += 1
    rows.append({**rec, "stage": "pass"})
    if buckets["pass"] % 10 == 0:
        print(f"  ... {i}/{A.n} evaluated, {buckets['pass']} passed", flush=True)

# ------------------------------------------------------------------ report ---
n = max(i, 1)
print()
print("=" * 74)
print(f"PHASE 5a GATE - {n} random in-bounds loop genomes, seed {A.seed}")
print(f"n_ctrl={A.n_ctrl}  loft sections={A.loft}  target={A.target * 1e3:.2f} mm"
      f"  trimesh={'yes' if HAVE_TRIMESH else 'ABSENT'}")
if A.require_feasible:
    print(f"DIAGNOSTIC MODE: rejection-sampled to geometric feasibility "
          f"({A.n} accepted from {draws} draws = "
          f"{100.0 * A.n / max(draws, 1):.2f}% acceptance). NOT the headline "
          f"rate; bounds and generator are unchanged.")
print("=" * 74)
order = ["pass", "decode", "feasibility", "sweep", "not_watertight", "unmeshable"]
for k in order:
    v = buckets.get(k, 0)
    print(f"  {k:16s} {v:5d}   {100.0 * v / n:6.2f}%")
print(f"\n  HEADLINE watertight-and-meshable pass rate: "
      f"{100.0 * buckets.get('pass', 0) / n:.1f}%")

for title, c in (("decode rejections", decode_reasons),
                 ("sweep failures", sweep_reasons),
                 ("watertightness failures", wt_reasons),
                 ("meshing failures", mesh_reasons)):
    if c:
        print(f"\n  {title}:")
        for k, v in c.most_common():
            print(f"    {v:5d}  {k}")

ok = [r for r in rows if r["stage"] == "pass"]
if ok:
    q = {k: np.array([r[k] for r in ok if k in r])
         for k in ("gamma_min", "gamma_p01", "gamma_mean", "n_tets")}
    if q["gamma_min"].size:
        print(f"\n  MESH QUALITY over {q['gamma_min'].size} meshed passes:")
        for k in ("gamma_min", "gamma_p01", "gamma_mean"):
            a = q[k]
            print(f"    {k:10s} min={a.min():.4f}  median={np.median(a):.4f}  "
                  f"max={a.max():.4f}")
        print(f"    {'n_tets':10s} min={int(q['n_tets'].min())}  "
              f"median={int(np.median(q['n_tets']))}  max={int(q['n_tets'].max())}")
    if HAVE_TRIMESH:
        vr = np.array([r["tm_vol_rel"] for r in ok if "tm_vol_rel" in r])
        if vr.size:
            print(f"\n  TRIMESH vs OCC volume over {vr.size} passes: "
                  f"max |rel| = {np.abs(vr).max():.3e}, "
                  f"median = {np.median(np.abs(vr)):.3e}")

feas = [r for r in rows if "geom_feasible" in r]
if feas:
    nf = sum(1 for r in feas if r["geom_feasible"])
    print(f"\n  GEOMETRIC FEASIBILITY (recorded, NOT gating): {nf}/{len(feas)} "
          f"= {100.0 * nf / len(feas):.1f}% satisfy all penalties")
    for gname in GEOM_NAMES:
        viol = sum(1 for r in feas if r["geom"].get(gname, -1) > 0)
        print(f"    {gname:24s} violated by {viol:4d} "
              f"({100.0 * viol / len(feas):5.1f}%)")
    swept_ok = [r for r in feas if r["stage"] in ("pass", "unmeshable",
                                                  "watertight")]
    bad_but_swept = [r for r in swept_ok if not r["geom_feasible"]]
    print(f"    of {len(swept_ok)} that swept successfully, "
          f"{len(bad_but_swept)} were geometrically INFEASIBLE "
          "(OCC does not check self-intersection, which is why the penalties "
          "exist and why watertight != valid)")

    # DOES THE PENALTY PREDICT THE FAILURE? This is the question that separates
    # "the sweep generator is broken" from "the sampler hands it broken spines".
    print("")
    print("  PENALTY vs OUTCOME (is infeasibility what actually fails?)")
    for gname in GEOM_NAMES:
        v_pass = sum(1 for r in feas
                     if r["stage"] == "pass" and r["geom"][gname] > 0)
        v_fail = sum(1 for r in feas
                     if r["stage"] != "pass" and r["geom"][gname] > 0)
        n_pass = sum(1 for r in feas if r["stage"] == "pass")
        n_fail = len(feas) - n_pass
        print(f"    {gname:24s} violated in {v_pass:3d}/{n_pass:3d} passes, "
              f"{v_fail:3d}/{n_fail:3d} failures")
    sev = {"pass": [], "fail": []}
    for r in feas:
        sev["pass" if r["stage"] == "pass" else "fail"].append(
            max(r["geom"][g] for g in GEOM_NAMES))
    for k, a in sev.items():
        if a:
            a = np.array(a)
            print(f"    worst penalty among {k:4s}: median {np.median(a):8.3f}  "
                  f"p90 {np.percentile(a, 90):8.3f}")

    # geometry that the SAMPLER, not the generator, is responsible for
    curv = np.array([r["geom"]["curvature_vs_section"] for r in feas])
    print("")
    print("  SAMPLER DIAGNOSIS: curvature_vs_section = "
          "section_radius/min_curvature_radius - 1")
    print(f"    median {np.median(curv):.2f}, p10 {np.percentile(curv, 10):.2f}, "
          f"min {curv.min():.2f}   (<= 0 means the tube fits its own bend)")

Path(A.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({"n": n, "seed": A.seed, "n_ctrl": A.n_ctrl, "loft": A.loft,
           "buckets": dict(buckets), "rows": rows,
           "decode_reasons": dict(decode_reasons),
           "sweep_reasons": dict(sweep_reasons),
           "mesh_reasons": dict(mesh_reasons),
           "wt_reasons": dict(wt_reasons)}, open(A.out, "w"), indent=1)
print(f"\n  wrote {A.out}")
