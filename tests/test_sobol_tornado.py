"""Unit tests for make_sobol_tornado (pure Plotly figure builder)."""

from __future__ import annotations

from ui.pages.calibration_charts import make_sobol_tornado

_ROWS = [
    {"param": "a", "s1": 0.4, "s1_conf": 0.05, "st": 0.5, "st_conf": 0.06},
    {"param": "b", "s1": 0.1, "s1_conf": 0.02, "st": 0.15, "st_conf": 0.03},
]


def test_both_yields_two_traces():
    fig = make_sobol_tornado(_ROWS, indices="Both")
    assert len(fig.data) == 2


def test_s1_only_one_trace():
    fig = make_sobol_tornado(_ROWS, indices="S1")
    assert len(fig.data) == 1
    assert fig.data[0].name.startswith("S1")


def test_st_only_one_trace():
    fig = make_sobol_tornado(_ROWS, indices="ST")
    assert len(fig.data) == 1
    assert fig.data[0].name.startswith("ST")


def test_horizontal_and_error_x():
    fig = make_sobol_tornado(_ROWS, indices="Both")
    for tr in fig.data:
        assert tr.orientation == "h"
        assert tuple(tr.error_x.array) != ()


def test_threshold_highlights_influential_on_st_bar():
    fig = make_sobol_tornado(_ROWS, indices="ST", threshold=0.3)
    st_bar = fig.data[0]
    # rows order is as given (a: st=0.5 influential, b: st=0.15 not)
    colors = list(st_bar.marker.color)
    assert colors[0] != colors[1]  # influential vs muted differ


def test_empty_rows_no_bars():
    fig = make_sobol_tornado([], indices="Both")
    assert len(fig.data) == 0
