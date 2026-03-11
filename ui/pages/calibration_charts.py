# ui/pages/calibration_charts.py
"""Chart/visualization helper functions for the calibration page."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def make_convergence_chart(history: list[float], tmpl: str = "osmose") -> go.Figure:
    """Line chart of best objective value per generation."""
    if not history:
        return go.Figure().update_layout(title="Convergence", template=tmpl)
    import plotly.express as px

    fig = px.line(x=list(range(len(history))), y=history, title="Convergence")
    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="Best Objective",
        template=tmpl,
    )
    return fig


def make_pareto_chart(F: np.ndarray, obj_names: list[str], tmpl: str = "osmose") -> go.Figure:
    """Scatter plot of Pareto front (2 objectives)."""
    import plotly.express as px

    fig = px.scatter(x=F[:, 0], y=F[:, 1], title="Pareto Front")
    fig.update_layout(
        xaxis_title=obj_names[0] if len(obj_names) > 0 else "Obj 1",
        yaxis_title=obj_names[1] if len(obj_names) > 1 else "Obj 2",
        template=tmpl,
    )
    return fig


def make_sensitivity_chart(result: dict, tmpl: str = "osmose") -> go.Figure:
    """Bar chart of Sobol sensitivity indices."""
    names = result["param_names"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="S1 (First-order)", x=names, y=result["S1"]))
    fig.add_trace(go.Bar(name="ST (Total-order)", x=names, y=result["ST"]))
    fig.update_layout(
        title="Sobol Sensitivity Indices",
        barmode="group",
        template=tmpl,
    )
    return fig
