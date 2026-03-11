"""Tests for osmose.plotting — pure chart functions returning go.Figure."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from osmose.plotting import (
    make_ci_timeseries,
    make_growth_curves,
    make_mortality_breakdown,
    make_predation_ranges,
    make_size_spectrum_plot,
    make_stacked_area,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_long_df():
    """Empty DataFrame with the columns expected by stacked-area charts."""
    return pd.DataFrame(columns=["time", "species", "bin", "value"])


def _sample_long_df(n_times=5, species="Anchovy", bins=None):
    """Create a small long-format DataFrame for stacked-area tests."""
    bins = bins or ["0-5", "5-10", "10-15"]
    rows = []
    for t in range(n_times):
        for b in bins:
            rows.append({"time": t, "species": species, "bin": b, "value": float(t + 1)})
    return pd.DataFrame(rows)


def _mortality_df(n_times=5, species="Anchovy"):
    rows = []
    for t in range(n_times):
        rows.append(
            {
                "time": t,
                "predation": 0.3,
                "starvation": 0.1,
                "fishing": 0.4,
                "natural": 0.2,
                "species": species,
            }
        )
    return pd.DataFrame(rows)


def _size_abundance_df():
    sizes = [1, 2, 5, 10, 20, 50]
    abundances = [1000, 500, 100, 50, 10, 2]
    return pd.DataFrame({"size": sizes, "abundance": abundances})


# ---------------------------------------------------------------------------
# make_stacked_area
# ---------------------------------------------------------------------------


class TestMakeStackedArea:
    def test_empty_df_returns_figure_with_title(self):
        fig = make_stacked_area(_empty_long_df(), "Empty Chart")
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Empty Chart"
        assert len(fig.data) == 0

    def test_normal_input_creates_traces_per_bin(self):
        df = _sample_long_df(bins=["0-5", "5-10"])
        fig = make_stacked_area(df, "Biomass by Age")
        assert len(fig.data) == 2
        assert fig.layout.title.text == "Biomass by Age"

    def test_species_filter(self):
        df1 = _sample_long_df(species="Anchovy")
        df2 = _sample_long_df(species="Sardine")
        df = pd.concat([df1, df2], ignore_index=True)
        fig = make_stacked_area(df, "Filtered", species="Anchovy")
        # Only bins for Anchovy
        assert len(fig.data) == 3

    def test_uses_osmose_template(self):
        df = _sample_long_df()
        fig = make_stacked_area(df, "T")
        assert fig.layout.template.layout.paper_bgcolor is not None or "osmose" in str(
            fig.layout.template
        )

    def test_traces_are_stacked(self):
        df = _sample_long_df()
        fig = make_stacked_area(df, "Stacked")
        for trace in fig.data:
            assert trace.stackgroup == "one"


# ---------------------------------------------------------------------------
# make_mortality_breakdown
# ---------------------------------------------------------------------------


class TestMakeMortalityBreakdown:
    def test_empty_df(self):
        df = pd.DataFrame(
            columns=["time", "predation", "starvation", "fishing", "natural", "species"]
        )
        fig = make_mortality_breakdown(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_creates_four_traces(self):
        df = _mortality_df()
        fig = make_mortality_breakdown(df)
        assert len(fig.data) == 4

    def test_species_filter(self):
        df = pd.concat(
            [_mortality_df(species="Anchovy"), _mortality_df(species="Sardine")],
            ignore_index=True,
        )
        fig = make_mortality_breakdown(df, species="Sardine")
        # Should still be 4 traces, just filtered data
        assert len(fig.data) == 4
        for trace in fig.data:
            assert len(trace.x) == 5

    def test_correct_colors(self):
        df = _mortality_df()
        fig = make_mortality_breakdown(df)
        colors = {trace.name: trace.fillcolor for trace in fig.data}
        assert colors["predation"] == "#e74c3c"
        assert colors["starvation"] == "#f39c12"
        assert colors["fishing"] == "#3498db"
        assert colors["natural"] == "#95a5a6"


# ---------------------------------------------------------------------------
# make_size_spectrum_plot
# ---------------------------------------------------------------------------


class TestMakeSizeSpectrumPlot:
    def test_empty_df(self):
        df = pd.DataFrame(columns=["size", "abundance"])
        fig = make_size_spectrum_plot(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_normal_creates_scatter_and_regression(self):
        df = _size_abundance_df()
        fig = make_size_spectrum_plot(df)
        # One scatter + one regression line
        assert len(fig.data) == 2

    def test_log_scale_axes(self):
        df = _size_abundance_df()
        fig = make_size_spectrum_plot(df)
        assert fig.layout.xaxis.type == "log"
        assert fig.layout.yaxis.type == "log"

    def test_has_slope_annotation(self):
        df = _size_abundance_df()
        fig = make_size_spectrum_plot(df)
        assert len(fig.layout.annotations) >= 1
        annotation_text = fig.layout.annotations[0].text
        assert "slope" in annotation_text.lower()

    def test_single_point_no_crash(self):
        df = pd.DataFrame({"size": [10], "abundance": [100]})
        fig = make_size_spectrum_plot(df)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# make_ci_timeseries
# ---------------------------------------------------------------------------


class TestMakeCITimeseries:
    def test_empty_arrays(self):
        fig = make_ci_timeseries([], [], [], [], "Empty", "Y")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_normal_input(self):
        time = list(range(10))
        mean = [float(x) for x in range(10)]
        lower = [m - 1.0 for m in mean]
        upper = [m + 1.0 for m in mean]
        fig = make_ci_timeseries(time, mean, lower, upper, "Biomass", "tons")
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Biomass"
        assert fig.layout.yaxis.title.text == "tons"

    def test_has_band_and_line(self):
        time = list(range(5))
        mean = [1.0, 2.0, 3.0, 4.0, 5.0]
        lower = [0.5, 1.5, 2.5, 3.5, 4.5]
        upper = [1.5, 2.5, 3.5, 4.5, 5.5]
        fig = make_ci_timeseries(time, mean, lower, upper, "T", "Y")
        # Should have at least 2 traces: CI band and mean line
        assert len(fig.data) >= 2

    def test_numpy_arrays_accepted(self):
        time = np.arange(5)
        mean = np.ones(5)
        lower = np.zeros(5)
        upper = np.ones(5) * 2
        fig = make_ci_timeseries(time, mean, lower, upper, "NP", "val")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2


# ---------------------------------------------------------------------------
# make_growth_curves
# ---------------------------------------------------------------------------


class TestMakeGrowthCurves:
    def test_empty_list(self):
        fig = make_growth_curves([])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_single_species(self):
        params = [{"name": "Anchovy", "linf": 20, "k": 0.5, "t0": -0.2, "lifespan": 5}]
        fig = make_growth_curves(params)
        assert len(fig.data) == 1
        assert fig.data[0].name == "Anchovy"

    def test_multiple_species(self):
        params = [
            {"name": "Anchovy", "linf": 20, "k": 0.5, "t0": -0.2, "lifespan": 5},
            {"name": "Sardine", "linf": 30, "k": 0.3, "t0": -0.5, "lifespan": 8},
        ]
        fig = make_growth_curves(params)
        assert len(fig.data) == 2

    def test_values_clipped_positive(self):
        params = [{"name": "Fish", "linf": 20, "k": 0.5, "t0": 1.0, "lifespan": 5}]
        fig = make_growth_curves(params)
        y_vals = fig.data[0].y
        assert all(v >= 0 for v in y_vals)

    def test_title_and_labels(self):
        params = [{"name": "Fish", "linf": 20, "k": 0.5, "t0": 0, "lifespan": 5}]
        fig = make_growth_curves(params)
        assert "growth" in fig.layout.title.text.lower() or "von" in fig.layout.title.text.lower()


# ---------------------------------------------------------------------------
# make_predation_ranges
# ---------------------------------------------------------------------------


class TestMakePredationRanges:
    def test_empty_list(self):
        fig = make_predation_ranges([])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_single_species(self):
        params = [{"name": "Anchovy", "size_ratio_min": 0.01, "size_ratio_max": 0.5}]
        fig = make_predation_ranges(params)
        assert len(fig.data) == 1

    def test_multiple_species(self):
        params = [
            {"name": "Anchovy", "size_ratio_min": 0.01, "size_ratio_max": 0.5},
            {"name": "Sardine", "size_ratio_min": 0.05, "size_ratio_max": 0.8},
        ]
        fig = make_predation_ranges(params)
        assert len(fig.data) == 2

    def test_horizontal_orientation(self):
        params = [{"name": "Fish", "size_ratio_min": 0.1, "size_ratio_max": 0.9}]
        fig = make_predation_ranges(params)
        assert fig.data[0].orientation == "h"

    def test_bar_widths_correct(self):
        params = [{"name": "Fish", "size_ratio_min": 0.1, "size_ratio_max": 0.9}]
        fig = make_predation_ranges(params)
        # The bar should represent the range width
        trace = fig.data[0]
        # x value should be the range width (0.8)
        assert abs(trace.x[0] - 0.8) < 1e-6


# ---------------------------------------------------------------------------
# Column guard tests
# ---------------------------------------------------------------------------

import pytest


def test_make_stacked_area_rejects_missing_columns():
    df = pd.DataFrame({"time": [1], "wrong": [10]})
    with pytest.raises(ValueError, match="missing columns"):
        make_stacked_area(df, "Title")


def test_make_mortality_breakdown_rejects_missing_columns():
    df = pd.DataFrame({"time": [1], "predation": [0.1]})
    with pytest.raises(ValueError, match="missing columns"):
        make_mortality_breakdown(df)


def test_make_size_spectrum_plot_rejects_missing_columns():
    df = pd.DataFrame({"size": [1, 2], "wrong": [10, 20]})
    with pytest.raises(ValueError, match="missing columns"):
        make_size_spectrum_plot(df)
