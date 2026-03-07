"""Pure chart functions for OSMOSE outputs — all return go.Figure."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

TEMPLATE = "osmose"


def _ensure_template() -> None:
    """Register the osmose template if not already present."""
    if TEMPLATE in pio.templates:
        return
    pio.templates[TEMPLATE] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(15, 25, 35, 0)",
            plot_bgcolor="rgba(15, 25, 35, 0.6)",
            font=dict(family="Plus Jakarta Sans, -apple-system, sans-serif", color="#e2e8f0"),
            colorway=[
                "#e8a838",
                "#38c9b1",
                "#3498db",
                "#e74c3c",
                "#9b59b6",
                "#2ecc71",
                "#f39c12",
                "#1abc9c",
            ],
            legend=dict(bgcolor="rgba(0, 0, 0, 0.2)"),
            margin=dict(l=50, r=20, t=40, b=40),
        ),
    )


_ensure_template()

# Mortality source colors
_MORT_COLORS = {
    "predation": "#e74c3c",
    "starvation": "#f39c12",
    "fishing": "#3498db",
    "natural": "#95a5a6",
}


def _empty_figure(title: str) -> go.Figure:
    """Return an empty figure with the given title and osmose template."""
    return go.Figure().update_layout(title=dict(text=title), template=TEMPLATE)


# ---------------------------------------------------------------------------
# 1. Stacked area (ByAge / BySize / ByTL)
# ---------------------------------------------------------------------------


def make_stacked_area(
    df: pd.DataFrame,
    title: str,
    species: str | None = None,
) -> go.Figure:
    """Stacked area chart from long-format DataFrame.

    Columns: time, species, bin, value.
    One trace per unique bin value.
    """
    if df.empty:
        return _empty_figure(title)

    if species is not None:
        df = df[df["species"] == species]
        if df.empty:
            return _empty_figure(title)

    fig = go.Figure()
    for bin_label in df["bin"].unique():
        subset = df[df["bin"] == bin_label]
        fig.add_trace(
            go.Scatter(
                x=subset["time"],
                y=subset["value"],
                name=str(bin_label),
                mode="lines",
                stackgroup="one",
            )
        )
    fig.update_layout(title=dict(text=title), template=TEMPLATE)
    return fig


# ---------------------------------------------------------------------------
# 2. Mortality breakdown
# ---------------------------------------------------------------------------


def make_mortality_breakdown(
    df: pd.DataFrame,
    species: str | None = None,
) -> go.Figure:
    """Stacked area of mortality by source.

    Columns: time, predation, starvation, fishing, natural, species.
    """
    if df.empty:
        return _empty_figure("Mortality Breakdown")

    if species is not None:
        df = df[df["species"] == species]
        if df.empty:
            return _empty_figure("Mortality Breakdown")

    fig = go.Figure()
    sources = ["predation", "starvation", "fishing", "natural"]
    for source in sources:
        fig.add_trace(
            go.Scatter(
                x=df["time"].values,
                y=df[source].values,
                name=source,
                mode="lines",
                stackgroup="one",
                fillcolor=_MORT_COLORS[source],
                line=dict(color=_MORT_COLORS[source]),
            )
        )
    fig.update_layout(title=dict(text="Mortality Breakdown"), template=TEMPLATE)
    return fig


# ---------------------------------------------------------------------------
# 3. Size spectrum (log-log scatter + regression)
# ---------------------------------------------------------------------------


def make_size_spectrum_plot(df: pd.DataFrame) -> go.Figure:
    """Log-log scatter with linear regression line and slope annotation."""
    title = "Size Spectrum"
    if df.empty:
        return _empty_figure(title)

    # Filter out non-positive values for log transform
    df = df[(df["size"] > 0) & (df["abundance"] > 0)].copy()
    if df.empty:
        return _empty_figure(title)

    fig = go.Figure()

    # Scatter on log-log
    fig.add_trace(
        go.Scatter(
            x=df["size"],
            y=df["abundance"],
            mode="markers",
            name="Observed",
        )
    )

    # Linear regression in log-log space
    log_size = np.log10(df["size"].values.astype(float))
    log_abund = np.log10(df["abundance"].values.astype(float))

    if len(log_size) >= 2:
        coeffs = np.polyfit(log_size, log_abund, 1)
        slope, intercept = coeffs
        fitted = 10 ** (slope * log_size + intercept)
        fig.add_trace(
            go.Scatter(
                x=df["size"],
                y=fitted,
                mode="lines",
                name="Regression",
                line=dict(dash="dash"),
            )
        )
        fig.add_annotation(
            text=f"Slope = {slope:.2f}",
            xref="paper",
            yref="paper",
            x=0.95,
            y=0.95,
            showarrow=False,
            font=dict(size=12),
        )

    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(type="log", title="Size"),
        yaxis=dict(type="log", title="Abundance"),
        template=TEMPLATE,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. CI timeseries
# ---------------------------------------------------------------------------


def make_ci_timeseries(
    time: Sequence,
    mean: Sequence,
    lower: Sequence,
    upper: Sequence,
    title: str,
    y_label: str,
) -> go.Figure:
    """Time series with confidence-interval band."""
    time = list(time)
    mean = list(mean)
    lower = list(lower)
    upper = list(upper)

    if len(time) == 0:
        return _empty_figure(title)

    fig = go.Figure()

    # CI band as filled area (upper forward, lower reversed)
    fig.add_trace(
        go.Scatter(
            x=time + time[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(232, 168, 56, 0.2)",
            line=dict(color="rgba(255, 255, 255, 0)"),
            name="CI",
            showlegend=False,
        )
    )

    # Mean line
    fig.add_trace(
        go.Scatter(
            x=time,
            y=mean,
            mode="lines",
            name="Mean",
            line=dict(color="#e8a838", width=2),
        )
    )

    fig.update_layout(
        title=dict(text=title),
        yaxis=dict(title=dict(text=y_label)),
        template=TEMPLATE,
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Von Bertalanffy growth curves
# ---------------------------------------------------------------------------


def make_growth_curves(species_params: list[dict]) -> go.Figure:
    """Von Bertalanffy L(t) growth curves.

    Each dict: name, linf, k, t0, lifespan.
    L = linf * (1 - exp(-k * (t - t0))), clipped to >= 0.
    """
    title = "Von Bertalanffy Growth Curves"
    if not species_params:
        return _empty_figure(title)

    fig = go.Figure()
    for sp in species_params:
        t = np.linspace(0, sp["lifespan"], 200)
        length = sp["linf"] * (1 - np.exp(-sp["k"] * (t - sp["t0"])))
        length = np.clip(length, 0, None)
        fig.add_trace(
            go.Scatter(
                x=t,
                y=length,
                mode="lines",
                name=sp["name"],
            )
        )

    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title="Age (years)"),
        yaxis=dict(title="Length (cm)"),
        template=TEMPLATE,
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Predation size-ratio ranges (horizontal bar)
# ---------------------------------------------------------------------------


def make_predation_ranges(species_params: list[dict]) -> go.Figure:
    """Horizontal bar chart of predation size-ratio ranges.

    Each dict: name, size_ratio_min, size_ratio_max.
    """
    title = "Predation Size-Ratio Ranges"
    if not species_params:
        return _empty_figure(title)

    fig = go.Figure()
    for sp in species_params:
        width = sp["size_ratio_max"] - sp["size_ratio_min"]
        fig.add_trace(
            go.Bar(
                x=[width],
                y=[sp["name"]],
                orientation="h",
                base=[sp["size_ratio_min"]],
                name=sp["name"],
            )
        )

    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title="Prey/Predator Size Ratio"),
        template=TEMPLATE,
    )
    return fig
