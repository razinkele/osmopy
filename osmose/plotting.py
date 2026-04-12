"""Pure chart functions for OSMOSE outputs — all return go.Figure."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from osmose.plotly_theme import PLOTLY_TEMPLATE as TEMPLATE, ensure_templates

ensure_templates()


def _require_columns(df: pd.DataFrame, *cols: str, context: str = "") -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{context}: missing columns {sorted(missing)}, got {sorted(df.columns)}")


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

    _require_columns(df, "time", "bin", "value", context="make_stacked_area")

    if species is not None:
        df = df[df["species"] == species]  # type: ignore[assignment]
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

    _require_columns(
        df,
        "time",
        "predation",
        "starvation",
        "fishing",
        "natural",
        context="make_mortality_breakdown",
    )

    if species is not None:
        df = df[df["species"] == species]  # type: ignore[assignment]
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

    _require_columns(df, "size", "abundance", context="make_size_spectrum_plot")

    # Filter out non-positive values for log transform
    df = df[(df["size"] > 0) & (df["abundance"] > 0)].copy()  # type: ignore[assignment]
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


# ---------------------------------------------------------------------------
# 7. Food web Sankey diagram
# ---------------------------------------------------------------------------


def make_food_web(
    diet_df: pd.DataFrame,
    threshold: float = 0.01,
) -> go.Figure:
    """Sankey diagram of predator-prey relationships.

    Columns: predator, prey, proportion.
    Filters links below threshold.
    """
    title = "Food Web"
    if diet_df.empty:
        return _empty_figure(title)

    _require_columns(diet_df, "predator", "prey", "proportion", context="make_food_web")

    # Filter weak links
    diet_df = diet_df[diet_df["proportion"] >= threshold]  # type: ignore[assignment]
    if diet_df.empty:
        return _empty_figure(title)

    # Build node list
    all_species = sorted(set(diet_df["predator"]) | set(diet_df["prey"]))
    node_idx = {sp: i for i, sp in enumerate(all_species)}

    fig = go.Figure(
        go.Sankey(
            node=dict(label=all_species, pad=15, thickness=20),
            link=dict(
                source=[node_idx[p] for p in diet_df["predator"]],
                target=[node_idx[p] for p in diet_df["prey"]],
                value=diet_df["proportion"].tolist(),
            ),
        )
    )
    fig.update_layout(title=dict(text=title), template=TEMPLATE)
    return fig


# ---------------------------------------------------------------------------
# 8. Run comparison grouped bar chart
# ---------------------------------------------------------------------------


def make_run_comparison(
    records: list,
    metrics: list[str] | None = None,
) -> go.Figure:
    """Grouped bar chart comparing summary stats across runs.

    Args:
        records: List of RunRecord objects with .summary and .timestamp attrs.
        metrics: Which keys to compare. None = all keys from first record.
    """
    title = "Run Comparison"
    if not records:
        return _empty_figure(title)

    if metrics is None:
        metrics = sorted(records[0].summary.keys()) if records[0].summary else []
    if not metrics:
        return _empty_figure(title)

    fig = go.Figure()
    for record in records:
        label = record.timestamp[:19] if hasattr(record, "timestamp") else "Run"
        values = [record.summary.get(m, 0) for m in metrics]
        fig.add_trace(go.Bar(name=label, x=metrics, y=values))

    fig.update_layout(
        title=dict(text=title),
        barmode="group",
        template=TEMPLATE,
    )
    return fig


# ---------------------------------------------------------------------------
# 9. Species dashboard (small multiples)
# ---------------------------------------------------------------------------


def make_species_dashboard(
    biomass_df: pd.DataFrame,
    yield_df: pd.DataFrame,
) -> go.Figure:
    """Small multiples: one row per species with biomass + yield lines.

    biomass_df columns: time, species, biomass
    yield_df columns: time, species, yield
    """
    title = "Species Dashboard"
    if biomass_df.empty:
        return _empty_figure(title)

    _require_columns(biomass_df, "time", "species", "biomass", context="make_species_dashboard")

    species_list = sorted(biomass_df["species"].unique())
    n = len(species_list)
    if n == 0:
        return _empty_figure(title)

    fig = make_subplots(rows=n, cols=1, subplot_titles=species_list, shared_xaxes=True)

    for i, sp in enumerate(species_list, 1):
        sp_bio = biomass_df[biomass_df["species"] == sp]
        fig.add_trace(
            go.Scatter(x=sp_bio["time"], y=sp_bio["biomass"], name=f"{sp} biomass", mode="lines"),
            row=i,
            col=1,
        )
        if not yield_df.empty and "species" in yield_df.columns:
            sp_yield = yield_df[yield_df["species"] == sp]
            if not sp_yield.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sp_yield["time"],
                        y=sp_yield["yield"],
                        name=f"{sp} yield",
                        mode="lines",
                        line=dict(dash="dash"),
                    ),
                    row=i,
                    col=1,
                )

    fig.update_layout(title=dict(text=title), template=TEMPLATE, height=300 * n)
    return fig
