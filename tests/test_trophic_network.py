from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from osmose.trophic_network import (
    _read_diet_matrix,
    _split_species,
    available_times,
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
