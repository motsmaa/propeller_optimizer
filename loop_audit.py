"""Pre-flight audit for the Phase-5 loop path.

Same standing rule as `preflight_audit.py`: an unexercised config field is an
unvalidated config field, and the class fails SILENTLY (the `forces` binData
sub-dict in Phase 4b was accepted without warning and simply did nothing). Every
`loop_spec` field is varied once, cheaply, through decode -> frames ->
feasibility -> sweep, plus expected-refusal cases for the two topologies slice
5a does not implement.

Cheap on purpose: 8 loft sections and no 3-D mesh. This is a case-writer smoke
test, not the gate.
"""
import numpy as np
from propopt.contracts import SolverError
from propopt.problem_spec import loop_spec
from propopt.hf.loop_sweep import sweep_blade

BASE = dict(n_ctrl=5, n_stations=6)
CASES = [
    ("baseline",                 {}, "ok"),
    ("n_ctrl=4(min cubic)",      dict(n_ctrl=4), "ok"),
    ("n_ctrl=7",                 dict(n_ctrl=7), "ok"),
    ("n_stations=2(min)",        dict(n_stations=2), "ok"),
    ("n_stations=12",            dict(n_stations=12), "ok"),
    ("radius_m=0.20",            dict(radius_m=0.20), "ok"),
    ("hub_radius_m=0.02",        dict(hub_radius_m=0.02), "ok"),
    ("n_blades=3",               dict(n_blades=3), "ok"),
    ("n_blades=1",               dict(n_blades=1), "ok"),
    ("chord_bounds tight",       dict(chord_bounds=(0.004, 0.008)), "ok"),
    ("twist_bounds wide",        dict(twist_bounds=(-0.8, 0.8)), "ok"),
    ("rpm_bounds",               dict(rpm_bounds=(500.0, 3000.0)), "ok"),
    ("axial_bound_radii=0.2",    dict(axial_bound_radii=0.2), "ok"),
    ("axial_bound_radii=1.2",    dict(axial_bound_radii=1.2), "ok"),
    ("te_thickness_m=4e-4",      dict(te_thickness_m=4.0e-4), "ok"),
    # EXPECTED REFUSALS: both are well-formed NAMES at the contract level, so
    # the refusal must come from decode as a SolverError (a feasibility
    # outcome), not from the contract as a ValueError (which would abort a run).
    ("topology=loop_neighbour",  dict(topology="loop_neighbour"), "refuse"),
    ("topology=open",            dict(topology="open"), "refuse"),
    # malformed FACTORY arguments are ValueError: a programming error, not a
    # candidate the optimizer should penalize
    ("topology=moebius",         dict(topology="moebius"), "valueerror"),
    ("n_ctrl=3(<cubic)",         dict(n_ctrl=3), "valueerror"),
    ("n_stations=1",             dict(n_stations=1), "valueerror"),
]

results = []
for name, over, expect in CASES:
    stage = "factory"
    try:
        spec = loop_spec(**{**BASE, **over})
        rng = np.random.default_rng(7)
        n_ok = 0
        last = None
        for _ in range(6):
            x = spec.xl + (spec.xu - spec.xl) * rng.random(spec.n_var)
            stage = "decode"
            d = spec.decode(x)
            stage = "geom"
            spec.eval_geom(d)
            stage = "sweep"
            try:
                sweep_blade(d, n_loft_sections=8,
                            te_thickness_m=over.get("te_thickness_m", 2.0e-4))
                n_ok += 1
            except SolverError as e:
                last = str(e)[:60]
        got = "ok" if n_ok else "no-sweep"
        results.append((name, "OK" if expect == "ok" and n_ok else
                        ("FAIL(not refused)" if expect != "ok" else
                         f"FAIL(0/6 swept: {last})"), f"{n_ok}/6 swept"))
    except SolverError as e:
        ok = expect == "refuse" and stage == "decode" and "not implemented" in str(e)
        results.append((name, "OK(refused)" if ok else f"FAIL@{stage}",
                        "" if ok else str(e)[:70]))
    except ValueError as e:
        ok = expect == "valueerror"
        results.append((name, "OK(rejected)" if ok else f"FAIL@{stage}(ValueError)",
                        "" if ok else str(e)[:70]))
    print(f"{results[-1][0]:28s} {results[-1][1]:24s} {results[-1][2]}", flush=True)

print("\n=== SUMMARY ===")
bad = [r for r in results if not r[1].startswith("OK")]
print(f"{len(results) - len(bad)}/{len(results)} passed")
for nm, st, msg in bad:
    print(f"  {nm:28s} {st:24s} {msg}")
