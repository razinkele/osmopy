"""Tests for results page chart generation functions."""

import types

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import xarray as xr

from ui.pages import results as rp
from ui.pages.results import (
    make_diet_heatmap,
    make_timeseries_chart,
)
from ui.pages.grid_helpers import make_spatial_map


class _FakeResults:
    """Mimics OsmoseResults' wide biomass()/yield_biomass()/abundance() accessors."""

    def __init__(self, frame):
        self._f = frame

    def biomass(self, species=None):
        return self._f

    def yield_biomass(self, species=None):
        return self._f

    def abundance(self, species=None):
        return self._f


def _wide(**species_to_series):
    n = len(next(iter(species_to_series.values())))
    d = {"Time": list(range(1, n + 1))}
    d.update(species_to_series)
    d["species"] = ["all"] * n
    return pd.DataFrame(d)


def _patch_osmose_results(monkeypatch, frames_by_dir):
    """Patch osmose.results.OsmoseResults so each output_dir maps to a fixed wide frame."""

    def _factory(output_dir, prefix="osm", strict=True):
        return _FakeResults(frames_by_dir[str(output_dir)])

    monkeypatch.setattr("osmose.results.OsmoseResults", _factory)


def test_delta_for_selected_ranks_and_signs(monkeypatch):
    frames = {
        "base": _wide(cod=[100.0, 100.0], herring=[50.0, 50.0]),
        "var": _wide(cod=[110.0, 110.0], herring=[100.0, 100.0]),
    }
    _patch_osmose_results(monkeypatch, frames)
    recs = [types.SimpleNamespace(output_dir="base"), types.SimpleNamespace(output_dir="var")]
    deltas = rp._delta_for_selected(recs, metric="biomass", window_years=2)
    by = {d.species: d for d in deltas}
    assert by["cod"].pct_delta == pytest.approx(0.10)
    assert by["herring"].pct_delta == pytest.approx(1.0)
    assert [d.species for d in deltas][0] == "herring"  # biggest |Δ%| first


def test_delta_for_selected_swap_flips_sign(monkeypatch):
    frames = {
        "base": _wide(cod=[100.0, 100.0]),
        "var": _wide(cod=[110.0, 110.0]),
    }
    _patch_osmose_results(monkeypatch, frames)
    recs = [types.SimpleNamespace(output_dir="base"), types.SimpleNamespace(output_dir="var")]
    fwd = rp._delta_for_selected(recs, metric="biomass", window_years=2)[0]
    rev = rp._delta_for_selected(list(reversed(recs)), metric="biomass", window_years=2)[0]
    assert fwd.pct_delta == pytest.approx(0.10)  # base→var: +10%
    assert rev.pct_delta == pytest.approx(-10 / 110)  # var→base: -9.09%


def test_delta_for_selected_requires_two(monkeypatch):
    recs = [types.SimpleNamespace(output_dir="only")]
    with pytest.raises(ValueError):
        rp._delta_for_selected(recs, metric="biomass", window_years=2)


def test_results_ui_builds():
    # results_ui() must construct without error after the Compare Runs additions.
    from ui.pages.results import results_ui

    tag = results_ui()
    assert tag is not None
    html = str(tag)
    assert "compare_window_years" in html  # the new slider is wired
    assert "run_delta_chart" in html  # the new chart output
    assert "run_delta_table" in html  # the new table output


def test_make_timeseries_chart_biomass():
    df = pd.DataFrame(
        {
            "time": [0, 1, 2, 0, 1, 2],
            "biomass": [100, 200, 300, 50, 100, 150],
            "species": ["Anchovy", "Anchovy", "Anchovy", "Sardine", "Sardine", "Sardine"],
        }
    )
    fig = make_timeseries_chart(df, "biomass", "Biomass")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Two species traces


def test_make_timeseries_chart_empty():
    df = pd.DataFrame()
    fig = make_timeseries_chart(df, "biomass", "Biomass")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_make_timeseries_chart_with_species_filter():
    df = pd.DataFrame(
        {
            "time": [0, 1, 0, 1],
            "biomass": [100, 200, 50, 100],
            "species": ["Anchovy", "Anchovy", "Sardine", "Sardine"],
        }
    )
    fig = make_timeseries_chart(df, "biomass", "Biomass", species="Anchovy")
    assert len(fig.data) == 1


def test_make_timeseries_chart_wide_form_from_engine():
    """OsmoseResults.biomass() returns wide-form with capital 'Time', one column
    per species, and a constant 'species'='all' column. The chart helper must
    detect and melt this shape before plotting."""
    df = pd.DataFrame(
        {
            "Time": [0, 1, 2],
            "cod": [100.0, 110.0, 120.0],
            "sprat": [1000.0, 1100.0, 1050.0],
            "stickleback": [500.0, 480.0, 460.0],
            "species": ["all", "all", "all"],
        }
    )
    fig = make_timeseries_chart(df, "biomass", "Biomass")
    assert isinstance(fig, go.Figure)
    # 3 species columns → 3 traces after the internal melt
    assert len(fig.data) == 3
    trace_names = sorted(t.name for t in fig.data)
    assert trace_names == ["cod", "sprat", "stickleback"]


def test_make_timeseries_chart_wide_form_species_filter():
    """Species filter still works on wide-form input after the internal melt."""
    df = pd.DataFrame(
        {
            "Time": [0, 1, 2],
            "cod": [100.0, 110.0, 120.0],
            "sprat": [1000.0, 1100.0, 1050.0],
            "species": ["all", "all", "all"],
        }
    )
    fig = make_timeseries_chart(df, "biomass", "Biomass", species="cod")
    assert len(fig.data) == 1
    assert fig.data[0].name == "cod"


def test_make_timeseries_chart_no_time_column():
    """A DataFrame without any 'time' / 'Time' column returns an empty figure
    rather than raising — the chart can't show a time-series without time."""
    df = pd.DataFrame({"cod": [100, 200], "species": ["all", "all"]})
    fig = make_timeseries_chart(df, "biomass", "Biomass")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_make_timeseries_chart_species_all_sentinel():
    """The UI passes `species="all"` from the default select option to mean
    'show all species'. The chart helper must treat 'all' as no filter, not
    as a literal species name to filter by."""
    df = pd.DataFrame(
        {
            "Time": [0, 1, 2],
            "cod": [100.0, 110.0, 120.0],
            "sprat": [1000.0, 1100.0, 1050.0],
            "species": ["all", "all", "all"],
        }
    )
    fig = make_timeseries_chart(df, "biomass", "Biomass", species="all")
    # 2 species columns → 2 traces; the "all" sentinel doesn't drop them
    assert len(fig.data) == 2
    trace_names = sorted(t.name for t in fig.data)
    assert trace_names == ["cod", "sprat"]


def test_make_diet_heatmap():
    df = pd.DataFrame(
        {
            "time": [0, 0],
            "species": ["Anchovy", "Anchovy"],
            "prey_Sardine": [0.6, 0.5],
            "prey_Plankton": [0.4, 0.5],
        }
    )
    fig = make_diet_heatmap(df)
    assert isinstance(fig, go.Figure)


def test_make_diet_heatmap_empty():
    df = pd.DataFrame()
    fig = make_diet_heatmap(df)
    assert isinstance(fig, go.Figure)


def test_make_spatial_map():
    ds = xr.Dataset(
        {
            "biomass": xr.DataArray(
                np.random.rand(3, 5, 5),
                dims=["time", "lat", "lon"],
                coords={
                    "time": range(3),
                    "lat": np.linspace(43, 48, 5),
                    "lon": np.linspace(-5, 0, 5),
                },
            )
        }
    )
    fig = make_spatial_map(ds, "biomass", time_idx=0)
    assert isinstance(fig, go.Figure)


def test_make_spatial_map_with_title():
    ds = xr.Dataset(
        {
            "biomass": xr.DataArray(
                np.random.rand(1, 3, 3),
                dims=["time", "lat", "lon"],
                coords={"time": [0], "lat": [43, 44, 45], "lon": [-3, -2, -1]},
            )
        }
    )
    fig = make_spatial_map(ds, "biomass", time_idx=0, title="Biomass t=0")
    assert fig.layout.title.text == "Biomass t=0"


def test_make_spatial_map_multiple_timesteps():
    """Spatial map renders correctly at different time indices."""
    data = np.random.rand(5, 4, 6)
    ds = xr.Dataset(
        {"biomass": (["time", "lat", "lon"], data)},
        coords={
            "time": range(5),
            "lat": np.linspace(40, 50, 4),
            "lon": np.linspace(-5, 5, 6),
        },
    )
    for t in range(5):
        fig = make_spatial_map(ds, "biomass", time_idx=t, title=f"t={t}")
        assert fig is not None
        assert f"t={t}" in fig.layout.title.text


def test_compare_run_choices_builds_label_map():
    import types

    from ui.pages.results import _compare_run_choices

    runs = [
        types.SimpleNamespace(timestamp="2026-06-03T12:00:00", duration_sec=42.0),
        types.SimpleNamespace(timestamp="2026-06-04T08:30:15", duration_sec=7.4),
    ]
    assert _compare_run_choices(runs) == {
        "2026-06-03T12:00:00": "2026-06-03T12:00:00 (42s)",
        "2026-06-04T08:30:15": "2026-06-04T08:30:15 (7s)",
    }


def test_compare_run_choices_empty():
    from ui.pages.results import _compare_run_choices

    assert _compare_run_choices([]) == {}


def test_compare_runs_readers_have_no_output_dir_guard():
    """The 4 Compare Runs readers must not gate on input.output_dir() — they read
    run history from default_run_history(), not the active output dir. Asserts the
    two exact removed guard-return forms are gone (the surviving _load_results
    notification at ~:444 uses a different string and is intentionally untouched)."""
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parent.parent / "ui" / "pages" / "results.py"
    ).read_text()
    assert src.count('return go.Figure().update_layout(title="Invalid output directory"') == 0
    assert src.count('ui.div("Invalid output directory.")') == 0


def test_compare_runs_selector_populated_independently_of_output_dir():
    """The selector is populated by a nav-triggered effect reading run history,
    not by _do_load_results. Assert the new wiring exists and the old populate
    block (its distinctive comment) is gone from _do_load_results."""
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parent.parent / "ui" / "pages" / "results.py"
    ).read_text()
    assert "_populate_compare_runs" in src
    assert "_last_compare_choices" in src
    assert "Populate run comparison choices from history" not in src


def test_scenario_diff_tab_wired_into_results():
    """The Scenario Diff tab + sub-server are embedded in the Results page."""
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parent.parent / "ui" / "pages" / "results.py"
    ).read_text()
    assert "scenario_diff_nav_panel" in src
    assert "scenario_diff_server" in src


def test_scenario_diff_config_panel_wired():
    """The config-diff panel (accordion + output) is emitted in the Scenario Diff tab body."""
    from ui.pages.scenario_diff import scenario_diff_nav_panel

    # str(NavPanel) is only a repr and .tagify() raises outside a navset; render the BODY.
    html = str(scenario_diff_nav_panel().content)
    assert "diff_config_table" in html
    assert "Config differences" in html
