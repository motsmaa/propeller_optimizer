"""Phase 3: front export (JSON + STL) and plotting graceful-degradation."""
import dataclasses
import json

import pytest

from propopt.analysis import analyze_front
from propopt.config import run
from propopt.configs.default import default_config
from propopt.export import export_front_json, export_selected_stl, select_representative


@pytest.fixture(scope="module")
def front():
    cfg = dataclasses.replace(default_config(), pop_size=20, n_gen=4)
    res = run(cfg, seed=0)
    records, summary = analyze_front(res, cfg, cfg.motor)
    return cfg, records, summary


def test_json_export_roundtrips(front, tmp_path):
    cfg, records, summary = front
    path = export_front_json(tmp_path / "front.json", cfg, records, summary)
    data = json.loads(path.read_text())
    assert set(data) >= {"config", "constraint_activity", "designs"}
    assert data["config"]["seed"] == cfg.seed
    assert len(data["designs"]) == len(records)
    d0 = data["designs"][0]
    assert set(d0) >= {"x", "objectives", "rpm", "throttle", "current_a", "g", "activity"}
    assert set(d0["objectives"]) == {"thrust_N", "figure_of_merit", "mass_kg", "safety_factor"}
    # every constraint appears in both g and activity maps
    assert set(d0["g"]) == set(d0["activity"])


def test_stl_export_is_valid_ascii(front, tmp_path):
    cfg, records, _ = front
    paths = export_selected_stl(tmp_path / "stl", cfg, records)
    assert 1 <= len(paths) <= 4
    for p in paths:
        text = p.read_text()
        assert text.startswith("solid ") and text.strip().endswith("endsolid " + p.stem.replace("blade_", ""))
        n_facets = text.count("facet normal")
        assert n_facets > 0
        assert text.count("vertex") == 3 * n_facets      # triangles
        assert text.count("outer loop") == n_facets


def test_select_representative_picks_corners(front):
    _, records, _ = front
    idx = select_representative(records)
    assert set(idx) == {"max_thrust", "max_fom", "min_mass", "max_safety"}
    # the max_thrust pick really is the max-thrust design
    assert records[idx["max_thrust"]].objectives["thrust_N"] == max(
        r.objectives["thrust_N"] for r in records)


def test_plotting_writes_or_skips(front, tmp_path):
    pytest.importorskip("matplotlib")   # optional 'plot' extra
    from propopt.plotting import parallel_coordinates, scatter_matrix
    _, records, _ = front
    p1 = scatter_matrix(records, tmp_path / "sm.png")
    p2 = parallel_coordinates(records, tmp_path / "pc.png")
    assert p1.exists() and p1.stat().st_size > 0
    assert p2.exists() and p2.stat().st_size > 0
