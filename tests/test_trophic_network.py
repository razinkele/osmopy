from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from osmose.trophic_network import (
    _read_diet_matrix,
    _split_species,
    available_times,
    diet_network_at,
    make_trophic_network_html,
    network_node_universe,
    species_layout,
)
from tests._data_guards import require_eec_output


def _write_diet(path, rows, cols):
    # rows: list of dicts with Time, Prey, <predator cols>; cols: predator column names
    df = pd.DataFrame(rows, columns=["Time", "Prey", *cols])
    df.to_csv(path, index=False)  # clean header, no preamble


def test_split_species():
    assert _split_species("cod in [10.000000, 30.000000[") == "cod"
    assert _split_species("Diatoms") == "Diatoms"


def test_read_diet_matrix_wildcard(tmp_path):
    d = tmp_path / "output" / "Trophic"
    d.mkdir(parents=True)
    _write_diet(
        d / "eec_dietMatrix_Simu0.csv",
        [{"Time": 1.0, "Prey": "herring", "cod in [0, 50[": 30.0}],
        ["cod in [0, 50["],
    )
    wide = _read_diet_matrix(tmp_path / "output")  # wildcard finds it under Trophic/
    assert list(wide.columns) == ["Time", "Prey", "cod in [0, 50["]


def test_read_diet_matrix_missing(tmp_path):
    (tmp_path / "output").mkdir()
    with pytest.raises(FileNotFoundError):
        _read_diet_matrix(tmp_path / "output")


def test_available_times(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _write_diet(
        d / "x_dietMatrix.csv",
        [
            {"Time": 2.0, "Prey": "a", "p in [0, 1[": 1.0},
            {"Time": 1.0, "Prey": "a", "p in [0, 1[": 1.0},
        ],
        ["p in [0, 1["],
    )
    assert available_times(d) == [1.0, 2.0]


def test_network_node_universe(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _write_diet(
        d / "x_dietMatrix.csv",
        [{"Time": 1.0, "Prey": "herring in [0, 5[", "cod in [0, 5[": 10.0, "cod in [5, 9[": 20.0}],
        ["cod in [0, 5[", "cod in [5, 9["],
    )
    assert network_node_universe(d, "species") == ["cod", "herring"]
    assert network_node_universe(d, "stage") == ["cod in [0, 5[", "cod in [5, 9[", "herring"]


def test_read_diet_matrix_eec_real():
    require_eec_output("*dietMatrix*")
    wide = _read_diet_matrix(Path("data/eec_full/output"))
    assert "Time" in wide.columns and "Prey" in wide.columns
    assert wide["Time"].nunique() == 70


def _diet_fixture(path):
    # herring has a DEAD [30,inf[ stage (all 0) — exercises dead-stage exclusion (NOT cod).
    # predator cols sum to ~100 per live stage. Includes a self-loop (cod eats cod) + a NaN.
    rows = [
        # prey-species "cod" split into 2 stages summed to species within a predator col
        {
            "Time": 1.0,
            "Prey": "cod in [0, 10[",
            "cod in [0, 50[": 5.0,
            "herring in [0, 10[": 0.0,
            "herring in [10, 30[": 0.0,
            "herring in [30, inf[": 0.0,
        },
        {
            "Time": 1.0,
            "Prey": "cod in [10, inf[",
            "cod in [0, 50[": 15.0,
            "herring in [0, 10[": 0.0,
            "herring in [10, 30[": 0.0,
            "herring in [30, inf[": 0.0,
        },
        {
            "Time": 1.0,
            "Prey": "herring in [0, 5[",
            "cod in [0, 50[": 80.0,
            "herring in [0, 10[": 60.0,
            "herring in [10, 30[": 40.0,
            "herring in [30, inf[": 0.0,
        },
        {
            "Time": 1.0,
            "Prey": "Diatoms",
            "cod in [0, 50[": float("nan"),
            "herring in [0, 10[": 40.0,
            "herring in [10, 30[": 60.0,
            "herring in [30, inf[": 0.0,
        },
    ]
    cols = ["cod in [0, 50[", "herring in [0, 10[", "herring in [10, 30[", "herring in [30, inf["]
    pd.DataFrame(rows, columns=["Time", "Prey", *cols]).to_csv(path, index=False)


def test_diet_network_species_prey_sum_and_dead_stage(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _diet_fixture(d / "x_dietMatrix.csv")
    net = diet_network_at(d, time=1.0, threshold=0.0)
    m = {(r.predator, r.prey): r.proportion for r in net.itertuples()}
    # prey "cod" stages SUM within cod-predator: 5+15 = 20 (exact)
    assert m[("cod", "cod")] == pytest.approx(20.0)
    # herring predator: live stages are [0,10[ and [10,30[ (the [30,inf[ is all-zero=dead, excluded).
    # herring-on-Diatoms = mean(40, 60) over the 2 LIVE stages = 50 (NOT /3 incl. the dead stage)
    assert m[("herring", "Diatoms")] == pytest.approx(50.0)
    # herring-on-herring = mean(60, 40) = 50
    assert m[("herring", "herring")] == pytest.approx(50.0)


def test_diet_network_threshold_and_nan(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _diet_fixture(d / "x_dietMatrix.csv")
    net = diet_network_at(d, time=1.0, threshold=30.0)
    # cod->cod is 20 -> filtered out at threshold 30; herring->Diatoms (50) kept
    assert ("cod", "cod") not in {(r.predator, r.prey) for r in net.itertuples()}
    assert (net["proportion"] >= 30.0).all()
    # cod->Diatoms was NaN -> dropped entirely
    assert ("cod", "Diatoms") not in {(r.predator, r.prey) for r in net.itertuples()}


def test_diet_network_stage_level(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _diet_fixture(d / "x_dietMatrix.csv")
    net = diet_network_at(d, time=1.0, threshold=0.0, predator_level="stage")
    preds = set(net["predator"])
    assert "cod in [0, 50[" in preds  # predator kept at stage granularity
    assert "cod" not in preds


def test_diet_network_bad_time(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _diet_fixture(d / "x_dietMatrix.csv")
    with pytest.raises(ValueError):
        diet_network_at(d, time=99.0)


def test_diet_network_eec_real():
    require_eec_output("*dietMatrix*")
    net = diet_network_at(Path("data/eec_full/output"), time=1.0)  # no prefix (wildcard)
    assert list(net.columns) == ["predator", "prey", "proportion"]
    assert len(net) > 0 and (net["proportion"] >= 0).all()
    # species-level: no size suffix in node names
    assert not any(" in [" in s for s in set(net["predator"]) | set(net["prey"]))


def test_species_layout_deterministic():
    a = species_layout(["cod", "herring", "sprat"])
    b = species_layout(["sprat", "cod", "herring"])
    assert set(a) == {"cod", "herring", "sprat"}
    assert a["cod"] == b["cod"]  # deterministic (fixed seed), order-independent
    assert all(isinstance(v, tuple) and len(v) == 2 for v in a.values())


def test_make_trophic_network_html_self_contained_fixed_layout():
    pytest.importorskip("pyvis")
    import pandas as pd

    df = pd.DataFrame(
        {
            "predator": ["cod", "herring", "cod"],
            "prey": ["herring", "cod", "cod"],  # mutual cycle + self-loop
            "proportion": [70.0, 10.0, 20.0],
        }
    )
    pos = species_layout(["cod", "herring"])
    html = make_trophic_network_html(df, positions=pos, threshold=0.0)
    assert 'src="lib/' not in html  # self-contained (cdn_resources='in_line')
    assert '"physics"' in html and "false" in html  # physics disabled
    assert '"x"' in html and '"y"' in html  # fixed coords emitted
    assert "cod" in html and "herring" in html


def test_results_has_trophic_network_wiring():
    src = (Path(__file__).resolve().parent.parent / "ui" / "pages" / "results.py").read_text()
    assert "Trophic Network" in src  # the nav_panel
    assert "trophic_network" in src  # the output id / render fn
    assert "trophic_time" in src  # the time slider
    assert "make_trophic_network_html" in src  # the builder is used
    assert "update_slider" in src  # slider populated on load
    assert "_dietMatrix" not in src  # reads via the helper, not a hardcoded glob here


# ── Python-engine diet-matrix format (Time, <pred>_<prey> columns; species-level) ──
# The engine writes `Time, <predator>_<prey>` (predator-major, biomass eaten), NOT the
# Java `Time, Prey, <predator-stage cols>` layout. The trophic reader must accept both
# or it raises KeyError 'Prey' on every Python-engine run.


def _write_engine_diet(path, rows):
    """Engine format: comma-separated `Time, <pred>_<prey>` columns (no Prey column)."""
    pd.DataFrame(rows).to_csv(path, index=False)


def test_engine_format_node_universe(tmp_path):
    _write_engine_diet(
        tmp_path / "osm_dietMatrix_Simu0.csv",
        [{"Time": 0.0, "cod_herring": 10.0, "cod_sprat": 30.0, "herring_sprat": 5.0}],
    )
    assert set(network_node_universe(tmp_path)) == {"cod", "herring", "sprat"}
    # 'stage' falls back to species for engine output (no size-stages) — no error.
    assert set(network_node_universe(tmp_path, predator_level="stage")) == {
        "cod",
        "herring",
        "sprat",
    }


def test_engine_format_diet_network_at(tmp_path):
    # cod ate 40 t (herring 10, sprat 30) -> 25% / 75%; herring ate 5 t sprat -> 100%.
    _write_engine_diet(
        tmp_path / "osm_dietMatrix_Simu0.csv",
        [
            {
                "Time": 0.0,
                "cod_herring": 10.0,
                "cod_sprat": 30.0,
                "herring_sprat": 5.0,
                "herring_cod": 0.0,
            }
        ],
    )
    net = diet_network_at(tmp_path, time=0.0, threshold=1.0)
    cod = net[net["predator"] == "cod"].set_index("prey")["proportion"]
    assert cod["herring"] == pytest.approx(25.0)
    assert cod["sprat"] == pytest.approx(75.0)
    herring = net[net["predator"] == "herring"].set_index("prey")["proportion"]
    assert herring["sprat"] == pytest.approx(100.0)
    # 'stage' falls back to species (identical edges) for engine output.
    net_stage = diet_network_at(tmp_path, time=0.0, threshold=1.0, predator_level="stage")
    assert set(zip(net_stage["predator"], net_stage["prey"])) == set(
        zip(net["predator"], net["prey"])
    )


def test_engine_format_available_times(tmp_path):
    _write_engine_diet(
        tmp_path / "osm_dietMatrix_Simu0.csv",
        [{"Time": 0.0, "cod_herring": 1.0}, {"Time": 1.0, "cod_herring": 2.0}],
    )
    assert available_times(tmp_path) == [0.0, 1.0]
