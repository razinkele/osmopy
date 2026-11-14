"""summarize() in scripts/baltic_cell_occupancy.py: empty steps and empty windows."""

from __future__ import annotations

import json

from scripts.baltic_cell_occupancy import BINS, summarize


def _empty_step(step: int) -> dict:
    return {
        "step": step,
        "schools": 0,
        "cells": 0,
        "mean": None,
        "median": None,
        "max": 0,
        "scanned": 0,
        "eligible": 0,
        "bins": {k: 0 for k, _, _ in BINS},
        "n_over_100": 0,
    }


def _occupied_step(step: int) -> dict:
    # Two cells holding 3 and 1 schools: 3**2 + 1**2 = 10 pairs scanned, 4 of them eligible.
    return {
        "step": step,
        "schools": 4,
        "cells": 2,
        "mean": 2.0,
        "median": 2.0,
        "max": 3,
        "scanned": 10,
        "eligible": 4,
        "bins": {"le15": 10, "16_50": 0, "51_100": 0, "gt100": 0},
        "n_over_100": 0,
    }


def test_empty_steps_count_as_zero_in_the_per_step_means():
    s = summarize([_occupied_step(0), _empty_step(1)], "w", subdt=10)
    assert s["steps"] == 2 and s["steps_with_schools"] == 1
    assert s["schools_per_step_mean"] == 2  # (4 + 0) / 2
    assert s["occupied_cells_mean"] == 1  # (2 + 0) / 2
    assert s["schools_per_cell_mean"] == 2.0  # from the occupied step only
    assert s["schools_per_cell_max"] == 3
    assert s["eligible_fraction"] == 0.4
    assert s["pairs_scanned_x_subdt"] == 100


def test_a_window_with_no_occupied_step_reports_null_not_nan():
    s = summarize([_empty_step(0), _empty_step(1)], "w", subdt=10)
    assert s["steps"] == 2 and s["steps_with_schools"] == 0
    assert s["schools_per_step_mean"] == 0
    for key in ("schools_per_cell_mean", "schools_per_cell_median", "schools_per_cell_max"):
        assert s[key] is None, key
    assert s["eligible_fraction"] is None
    json.dumps(s, allow_nan=False)  # raises on NaN


def test_a_window_with_no_step_at_all_reports_null_not_nan():
    s = summarize([], "w", subdt=10)
    assert s["steps"] == 0
    assert s["schools_per_step_mean"] is None and s["cells_over_100_per_step_mean"] is None
    json.dumps(s, allow_nan=False)
