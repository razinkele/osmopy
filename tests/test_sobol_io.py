"""Unit tests for osmose.calibration.sobol_io."""

from __future__ import annotations

import math

import numpy as np
import pytest

from osmose.calibration.sobol_io import (
    influential_keys,
    list_sobol_results,
    load_sobol_result,
    rank_rows,
    rows_to_csv,
    save_sobol_result,
)


def _result_1d():
    return {
        "param_names": ["a", "b", "c"],
        "S1": np.array([0.4, 0.1, 0.25]),
        "ST": np.array([0.5, 0.15, 0.3]),
        "S1_conf": np.array([0.05, 0.02, 0.03]),
        "ST_conf": np.array([0.06, 0.03, 0.04]),
    }


def _result_2d():
    return {
        "param_names": ["a", "b"],
        "n_objectives": 2,
        "objective_names": ["o0", "o1"],
        "S1": np.array([[0.4, 0.1], [0.2, 0.7]]),
        "ST": np.array([[0.5, 0.15], [0.25, 0.8]]),
        "S1_conf": np.array([[0.05, 0.02], [0.03, 0.04]]),
        "ST_conf": np.array([[0.06, 0.03], [0.04, 0.05]]),
    }


def test_save_load_round_trip_1d(tmp_path):
    p = save_sobol_result(
        _result_1d(),
        metadata={
            "source": "test",
            "n_base": 16,
            "param_bounds": [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
            "objective_names": ["RMSE"],
            "timestamp": "2026-06-14T08:00:00",
        },
        directory=tmp_path,
    )
    assert p.exists()
    d = load_sobol_result("2026-06-14T08:00:00", directory=tmp_path)
    assert d["param_names"] == ["a", "b", "c"]
    assert d["S1"] == [0.4, 0.1, 0.25]  # numpy stored as list
    assert d["n_objectives"] == 1
    assert d["objective_names"] == ["RMSE"]
    # param_bounds round-trips as list-of-lists (tuples serialize to lists)
    assert d["param_bounds"] == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert d["source"] == "test" and d["n_base"] == 16


def test_save_load_round_trip_2d(tmp_path):
    save_sobol_result(
        _result_2d(),
        metadata={"source": "s", "timestamp": "2026-06-14T09:00:00"},
        directory=tmp_path,
    )
    d = load_sobol_result("2026-06-14T09:00:00", directory=tmp_path)
    assert d["n_objectives"] == 2
    assert d["S1"] == [[0.4, 0.1], [0.2, 0.7]]


def test_save_collision_safe(tmp_path):
    md = {"source": "s", "timestamp": "2026-06-14T10:00:00"}
    p1 = save_sobol_result(_result_1d(), metadata=md, directory=tmp_path)
    p2 = save_sobol_result(_result_1d(), metadata=md, directory=tmp_path)
    assert p1 != p2  # second got a -1 suffix; neither overwritten
    assert p1.exists() and p2.exists()


def test_list_newest_first_and_skips_corrupt(tmp_path):
    save_sobol_result(
        _result_1d(),
        metadata={"source": "s", "timestamp": "2026-06-14T01:00:00"},
        directory=tmp_path,
    )
    save_sobol_result(
        _result_1d(),
        metadata={"source": "s", "timestamp": "2026-06-14T02:00:00"},
        directory=tmp_path,
    )
    (tmp_path / "sobol_broken.json").write_text("{ not json")
    out = list_sobol_results(directory=tmp_path)
    assert [s["timestamp"] for s in out] == ["2026-06-14T02:00:00", "2026-06-14T01:00:00"]
    assert out[0]["n_params"] == 3 and out[0]["n_objectives"] == 1


def test_list_tolerates_none_objective_names(tmp_path):
    r = _result_1d()
    save_sobol_result(
        r, metadata={"source": "s", "timestamp": "2026-06-14T03:00:00"}, directory=tmp_path
    )
    out = list_sobol_results(directory=tmp_path)
    assert out[0]["objective_names"] is None  # 1-D save with no objective_names provided


def test_rank_rows_sort_and_1d():
    rows = rank_rows(_result_1d(), sort="ST")
    assert [r["param"] for r in rows] == ["a", "c", "b"]  # ST desc: 0.5, 0.3, 0.15
    rows_s1 = rank_rows(_result_1d(), sort="S1")
    assert [r["param"] for r in rows_s1] == ["a", "c", "b"]
    rows_name = rank_rows(_result_1d(), sort="name")
    assert [r["param"] for r in rows_name] == ["a", "b", "c"]


def test_rank_rows_1d_ignores_objective_idx_even_with_objective_names():
    # The live-run shape: 1-D S1 but objective_names present (n_objectives=1).
    r = _result_1d()
    r["objective_names"] = ["RMSE"]
    r["n_objectives"] = 1
    rows = rank_rows(r, objective_idx=5, sort="ST")  # idx ignored for 1-D
    assert rows[0]["param"] == "a" and rows[0]["st"] == 0.5


def test_rank_rows_2d_selects_objective_and_clamps():
    rows0 = rank_rows(_result_2d(), objective_idx=0, sort="ST")
    assert rows0[0]["param"] == "a" and rows0[0]["st"] == 0.5
    rows1 = rank_rows(_result_2d(), objective_idx=1, sort="ST")
    assert rows1[0]["param"] == "b" and rows1[0]["st"] == 0.8
    rows_clamp = rank_rows(_result_2d(), objective_idx=99, sort="ST")  # clamped to 1
    assert rows_clamp[0]["param"] == "b"


def test_rank_rows_accepts_lists():
    r = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in _result_1d().items()}
    rows = rank_rows(r, sort="ST")
    assert rows[0]["param"] == "a"


def test_rank_rows_nan_sinks_to_bottom():
    r = _result_1d()
    r["ST"] = np.array([0.5, float("nan"), 0.3])
    rows = rank_rows(r, sort="ST")
    assert rows[-1]["param"] == "b" and math.isnan(rows[-1]["st"])


def test_influential_keys_boundary_and_nan():
    rows = rank_rows(_result_1d(), sort="ST")
    assert influential_keys(rows, 0.3) == ["a", "c"]  # st == 0.3 included
    r = _result_1d()
    r["ST"] = np.array([0.5, float("nan"), 0.3])
    assert "b" not in influential_keys(rank_rows(r, sort="ST"), 0.0)  # NaN excluded


def test_rows_to_csv():
    csv = rows_to_csv(rank_rows(_result_1d(), sort="ST"))
    assert csv.splitlines()[0] == "param,S1,S1_conf,ST,ST_conf"
    assert csv.splitlines()[1].startswith("a,")


def test_load_rejects_unsafe_timestamp(tmp_path):
    with pytest.raises(ValueError):
        load_sobol_result("../x", directory=tmp_path)
    with pytest.raises(ValueError):
        load_sobol_result("/abs", directory=tmp_path)


def test_load_round_trips_colon_timestamp(tmp_path):
    save_sobol_result(
        _result_1d(),
        metadata={"source": "s", "timestamp": "2026-06-14T11:22:33"},
        directory=tmp_path,
    )
    # stored file uses '-' but load is given the ':' form
    assert (tmp_path / "sobol_2026-06-14T11-22-33.json").exists()
    d = load_sobol_result("2026-06-14T11:22:33", directory=tmp_path)
    assert d["param_names"] == ["a", "b", "c"]
