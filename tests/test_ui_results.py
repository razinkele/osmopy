"""Tests for results page chart generation functions."""

import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go

from ui.pages.results import (
    make_timeseries_chart,
    make_diet_heatmap,
)
from ui.pages.grid_helpers import make_spatial_map


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
