from __future__ import annotations

import pandas as pd
import pytest

from osmose.size_spectrum import (
    _community_long,
    _infer_bin_width,
    _read_community_by_size,
    _window_by_time,
)


def _write_community_csv(path, rows):
    """rows: list of (Time, Size, sp1, sp2). Clean header, NO preamble."""
    pd.DataFrame(rows, columns=["Time", "Size", "sp1", "sp2"]).to_csv(path, index=False)


def test_read_community_by_size_finds_and_reads(tmp_path):
    d = tmp_path / "output" / "Indicators"
    d.mkdir(parents=True)
    _write_community_csv(
        d / "osm_biomassDistribBySize_Simu0.csv", [(1.0, 0.0, 2.0, 3.0), (1.0, 10.0, 1.0, 1.0)]
    )
    wide = _read_community_by_size(tmp_path / "output", "biomassDistribBySize", "osm")
    assert list(wide.columns) == ["Time", "Size", "sp1", "sp2"]
    assert len(wide) == 2


def test_read_community_by_size_missing_raises(tmp_path):
    (tmp_path / "output").mkdir()
    with pytest.raises(FileNotFoundError):
        _read_community_by_size(tmp_path / "output", "biomassDistribBySize", "osm")


def test_read_community_by_size_empty_file_raises(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    (d / "osm_biomassDistribBySize_Simu0.csv").write_text("")  # 0-content
    with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
        _read_community_by_size(d, "biomassDistribBySize", "osm")


def test_community_long_sums_species():
    wide = pd.DataFrame(
        {"Time": [1.0, 1.0], "Size": [0.0, 10.0], "sp1": [2.0, 1.0], "sp2": [3.0, 4.0]}
    )
    long = _community_long(wide)
    assert list(long.columns) == ["time", "size", "value"]
    assert long.loc[long["size"] == 0.0, "value"].iloc[0] == 5.0
    assert long.loc[long["size"] == 10.0, "value"].iloc[0] == 5.0
    assert long["size"].dtype == float


def test_window_by_time_selects_years_not_rows():
    df = pd.DataFrame({"time": [1.0, 1.0, 2.0, 3.0], "size": [0, 10, 0, 0], "value": [1, 1, 1, 1]})
    w = _window_by_time(df, "time", 1)
    assert sorted(w["time"].unique()) == [3.0]


def test_window_by_time_rejects_nonpositive():
    df = pd.DataFrame({"time": [1.0], "size": [0.0], "value": [1.0]})
    with pytest.raises(ValueError):
        _window_by_time(df, "time", 0)


def test_infer_bin_width():
    assert _infer_bin_width([0.0, 10.0, 20.0, 30.0]) == 10.0
