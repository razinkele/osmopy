"""Tests for osmose.calibration.pareto — Pareto-front solution selection helpers."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.calibration.pareto import (
    apply_solution_overrides,
    nondominated_indices,
    select_solution,
    solution_overrides_csv,
)


def test_nondominated_indices_simple_front():
    # rows 0 and 2 are non-dominated; row 1 is dominated by row 0.
    F = np.array([[1.0, 4.0], [2.0, 5.0], [4.0, 1.0]])
    idx = nondominated_indices(F)
    assert set(idx.tolist()) == {0, 2}


def test_nondominated_indices_all_on_front():
    F = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    assert set(nondominated_indices(F).tolist()) == {0, 1, 2}


def test_nondominated_indices_single_row():
    assert nondominated_indices(np.array([[5.0, 5.0]])).tolist() == [0]


def test_select_solution_maps_keys_to_values():
    X = np.array([[10.0, 0.5], [20.0, 0.8]])
    F = np.array([[1.0, 9.0], [9.0, 1.0]])
    keys = ["species.linf.sp0", "species.k.sp0"]
    sol = select_solution(X, F, keys, 1)
    assert sol["index"] == 1
    assert sol["params"] == {"species.linf.sp0": 20.0, "species.k.sp0": 0.8}
    assert sol["objectives"] == [9.0, 1.0]


def test_select_solution_out_of_range_raises():
    X = np.array([[1.0]])
    F = np.array([[1.0]])
    with pytest.raises(IndexError):
        select_solution(X, F, ["a"], 5)


def test_select_solution_key_count_mismatch_raises():
    X = np.array([[1.0, 2.0]])
    F = np.array([[1.0, 1.0]])
    with pytest.raises(ValueError):
        select_solution(X, F, ["only_one_key"], 0)


def test_solution_overrides_csv_format():
    csv = solution_overrides_csv({"species.linf.sp0": 20.0, "species.k.sp0": 0.8})
    lines = csv.strip().split("\n")
    assert lines == ["species.linf.sp0 ; 20.0", "species.k.sp0 ; 0.8"]


def test_solution_overrides_csv_empty():
    assert solution_overrides_csv({}) == ""


def test_apply_solution_overrides_merges_and_counts():
    cfg = {"a": "1", "b": "2"}
    new, n = apply_solution_overrides(cfg, {"a": 5.0, "c": 3.5})
    assert new == {"a": "5.0", "b": "2", "c": "3.5"}
    assert n == 2  # "a" changed, "c" added
    assert cfg == {"a": "1", "b": "2"}  # input not mutated


def test_apply_solution_overrides_unchanged_value_not_counted():
    new, n = apply_solution_overrides({"a": "5.0"}, {"a": 5.0})
    assert new == {"a": "5.0"} and n == 0


def test_apply_solution_overrides_empty_params():
    cfg = {"a": "1"}
    new, n = apply_solution_overrides(cfg, {})
    assert new == cfg and n == 0


def test_apply_solution_overrides_matches_csv_rendering():
    """Apply and Download must render identical string values for the same params."""
    params = {"x.y": 0.125, "z": 3.0}
    new, _ = apply_solution_overrides({}, params)
    csv_vals = dict(
        line.split(" ; ") for line in solution_overrides_csv(params).strip().split("\n")
    )
    assert new == csv_vals
