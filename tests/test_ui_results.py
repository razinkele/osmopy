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
