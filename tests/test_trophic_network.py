from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from osmose.trophic_network import (
    _read_diet_matrix,
    _split_species,
    available_times,
    diet_network_at,
    network_node_universe,
)


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
    net = diet_network_at(Path("data/eec_full/output"), time=1.0)  # no prefix (wildcard)
    assert list(net.columns) == ["predator", "prey", "proportion"]
    assert len(net) > 0 and (net["proportion"] >= 0).all()
    # species-level: no size suffix in node names
    assert not any(" in [" in s for s in set(net["predator"]) | set(net["prey"]))
