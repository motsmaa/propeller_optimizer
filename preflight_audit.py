"""Pre-flight: exercise every never-varied HoverCaseConfig field.

Build the case + mesh + 2 solver iterations on a deliberately tiny mesh. This is
the check that would have caught the three missing `epsilon` dictionary entries
in seconds instead of two diverged 2-hour runs.
"""
import shutil, sys, traceback
from pathlib import Path
from propopt.hf.hover_case import HoverCaseConfig, build_hover_case
from propopt.hf.openfoam import make_case, run_foam
from propopt.validation.uiuc import design_from_geometry, load_geometry

geom = load_geometry(Path("data/uiuc/apcsf_10x4.7_geom.txt"))
design = design_from_geometry(geom, 0.254, 2, 4590.0, n_stations=20)
BASE = dict(rpm=4590.0, periodic=False, background_cells_per_span=10,
            core_radii=1.3, core_cell_frac=0.55, surface_level=(1, 2),
            rotor_region_level=0, n_layers=0, n_iterations=2, n_proc=4,
            far_field_radii=3.0)

CASES = [
    ("baseline",            {}),
    ("turbulence=kOmega",   dict(turbulence_model="kOmega")),
    ("turbulence=SpalartAllmaras", dict(turbulence_model="SpalartAllmaras")),
    ("turbulence=kEpsilon", dict(turbulence_model="kEpsilon")),
    ("turbulence=realizableKE", dict(turbulence_model="realizableKE")),
    ("far_field_bc=inletOutlet", dict(far_field_bc="inletOutlet")),
    ("nut=nutUSpaldingWallFunction", dict(nut_wall_function="nutUSpaldingWallFunction")),
    ("core_radii=2.0",      dict(core_radii=2.0)),
    ("core_cell_frac=0.35", dict(core_cell_frac=0.35)),
    ("stl_size_m=8e-4",     dict(stl_size_m=8.0e-4)),
    ("n_chord=41",          dict(n_chord=41)),
    ("te_thickness=3e-4",   dict(te_thickness_m=3.0e-4)),
    ("write_interval=1",    dict(write_interval=1)),
    ("rho=1.100",           dict(rho=1.100)),
    ("nu=1.8e-5",           dict(nu=1.8e-5)),
    ("mrf_zone_radii=1.8",  dict(mrf_zone_radii=1.8)),
    ("mrf_half_height=0.9", dict(mrf_zone_half_height_radii=0.9)),
    ("n_layers=3(rel)",     dict(n_layers=3)),
    # EXPECTED REFUSAL: the 180-degree sector is geometrically degenerate
    # (coplanar cyclic pair meeting on the rotation axis). The guard must
    # raise; silently producing a bad mesh is the failure mode being tested.
    ("periodic=True(180)",  dict(periodic=True), "refuse"),
]

results = []
for entry in CASES:
    name, over = entry[0], entry[1]
    expect_refusal = len(entry) > 2 and entry[2] == "refuse"
    src = Path("runs/audit_case")
    if src.exists(): shutil.rmtree(src)
    stage = "config"
    try:
        cfg = HoverCaseConfig(**{**BASE, **over})
        case = make_case("audit", src)
        stage = "build_hover_case"
        build_hover_case(case, design, cfg)
        case.sync_in()
        for app, args, npx in (("blockMesh", (), None),
                               ("snappyHexMesh", ("-overwrite",), None),
                               ("topoSet", (), None),
                               ("simpleFoam", (), None)):
            stage = app
            run_foam(app, case, args, n_proc=npx, timeout=1800, measure_rss=False)
        results.append((name, "FAIL(not refused)" if expect_refusal else "OK", ""))
    except Exception as exc:
        msg = str(exc).replace("\n", " ")
        for key in ("not found in dictionary", "FOAM FATAL", "cannot find",
                    "Unknown", "not implemented"):
            i = msg.find(key)
            if i >= 0:
                msg = msg[i:i+110]; break
        else:
            msg = msg[:110]
        ok = expect_refusal and stage == "build_hover_case" and "REFUSED" in str(exc)
        results.append((name, "OK(refused)" if ok else "FAIL@" + stage,
                        "" if ok else msg))
    print(f"{results[-1][0]:32s} {results[-1][1]:22s} {results[-1][2]}", flush=True)

print("\n=== SUMMARY ===")
bad = [r for r in results if not r[1].startswith("OK")]
print(f"{len(results)-len(bad)}/{len(results)} passed")
for n, s, m in bad:
    print(f"  {n:32s} {s:22s} {m}")
