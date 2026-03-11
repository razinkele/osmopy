"""Canonical Plotly theme definitions for OSMOSE charts."""

import plotly.graph_objects as go
import plotly.io as pio

# Dark theme colorway
OSMOSE_COLORS = [
    "#e8a838",  # amber
    "#38c9b1",  # teal
    "#3498db",  # blue
    "#e74c3c",  # red
    "#9b59b6",  # purple
    "#2ecc71",  # green
    "#f39c12",  # orange
    "#1abc9c",  # turquoise
]

# Light theme colorway
OSMOSE_COLORS_LIGHT = [
    "#d4942e",  # amber (darkened for light bg)
    "#2ba89e",  # teal
    "#2980b9",  # blue
    "#c0392b",  # red
    "#8e44ad",  # purple
    "#27ae60",  # green
    "#e67e22",  # orange
    "#16a085",  # turquoise
]

PLOTLY_TEMPLATE = "osmose"
PLOTLY_TEMPLATE_LIGHT = "osmose-light"


def ensure_templates() -> None:
    """Register both osmose Plotly templates if not already present."""
    if "osmose" in pio.templates:
        return

    pio.templates["osmose"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(15, 25, 35, 0)",
            plot_bgcolor="rgba(15, 25, 35, 0.6)",
            font=dict(
                family="Plus Jakarta Sans, -apple-system, sans-serif",
                color="#e2e8f0",
                size=12,
            ),
            title=dict(
                font=dict(size=15, color="#e2e8f0"),
                x=0.02,
                xanchor="left",
            ),
            xaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.06)",
                linecolor="rgba(255, 255, 255, 0.1)",
                zerolinecolor="rgba(255, 255, 255, 0.08)",
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.06)",
                linecolor="rgba(255, 255, 255, 0.1)",
                zerolinecolor="rgba(255, 255, 255, 0.08)",
                tickfont=dict(size=11),
            ),
            colorway=OSMOSE_COLORS,
            legend=dict(
                bgcolor="rgba(0, 0, 0, 0.2)",
                bordercolor="rgba(255, 255, 255, 0.06)",
                borderwidth=1,
                font=dict(size=11),
            ),
            margin=dict(l=50, r=20, t=40, b=40),
        ),
    )

    pio.templates["osmose-light"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(255, 255, 255, 0)",
            plot_bgcolor="rgba(240, 244, 248, 0.6)",
            font=dict(
                family="Plus Jakarta Sans, -apple-system, sans-serif",
                color="#1a2a3a",
                size=12,
            ),
            title=dict(
                font=dict(size=15, color="#1a2a3a"),
                x=0.02,
                xanchor="left",
            ),
            xaxis=dict(
                gridcolor="rgba(0, 0, 0, 0.06)",
                linecolor="rgba(0, 0, 0, 0.1)",
                zerolinecolor="rgba(0, 0, 0, 0.08)",
                tickfont=dict(size=11, color="#4a5a6a"),
            ),
            yaxis=dict(
                gridcolor="rgba(0, 0, 0, 0.06)",
                linecolor="rgba(0, 0, 0, 0.1)",
                zerolinecolor="rgba(0, 0, 0, 0.08)",
                tickfont=dict(size=11, color="#4a5a6a"),
            ),
            colorway=OSMOSE_COLORS_LIGHT,
            legend=dict(
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.08)",
                borderwidth=1,
                font=dict(size=11, color="#4a5a6a"),
            ),
            margin=dict(l=50, r=20, t=40, b=40),
        ),
    )


def get_plotly_template(theme_mode: str = "dark") -> str:
    """Return the appropriate Plotly template name for the given theme."""
    return PLOTLY_TEMPLATE_LIGHT if theme_mode == "light" else PLOTLY_TEMPLATE
