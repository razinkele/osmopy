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
    """Scatter plot of Pareto front (2+ objectives) or histogram (1 objective)."""
    import plotly.express as px

    if F.shape[1] < 2:
        fig = px.histogram(x=F[:, 0], title="Objective Distribution")
        fig.update_layout(
            xaxis_title=obj_names[0] if obj_names else "Objective",
            yaxis_title="Count",
            template=tmpl,
        )
        return fig
    fig = px.scatter(x=F[:, 0], y=F[:, 1], title="Pareto Front")
    fig.update_layout(
        xaxis_title=obj_names[0] if len(obj_names) > 0 else "Obj 1",
        yaxis_title=obj_names[1] if len(obj_names) > 1 else "Obj 2",
        template=tmpl,
    )
    return fig


def make_sensitivity_chart(
    result: dict,
    tmpl: str = "osmose",
    selected_objective: int = 0,
) -> go.Figure:
    """Bar chart of Sobol sensitivity indices (1D or multi-objective)."""
    if "objective_names" in result:
        s1 = result["S1"][selected_objective]
        st = result["ST"][selected_objective]
        obj_name = result["objective_names"][selected_objective]
        title = f"Sobol Sensitivity — {obj_name}"
    else:
        s1 = result["S1"]
        st = result["ST"]
        title = "Sobol Sensitivity Indices"

    names = result["param_names"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="S1 (First-order)", x=names, y=s1))
    fig.add_trace(go.Bar(name="ST (Total-order)", x=names, y=st))
    fig.update_layout(title=title, barmode="group", template=tmpl)
    return fig


def make_correlation_chart(
    X: np.ndarray,
    F: np.ndarray,
    param_names: list[str],
    tmpl: str = "osmose",
) -> go.Figure:
    """Parallel coordinates plot of Pareto candidates."""
    import pandas as pd
    import plotly.express as px

    if X is None or len(X) == 0:
        return go.Figure().update_layout(
            title="Parameter Correlations (run calibration first)", template=tmpl
        )
    df = pd.DataFrame(X, columns=param_names)
    df["objective"] = F[:, 0] if F.shape[1] == 1 else np.sum(F, axis=1)
    fig = px.parallel_coordinates(
        df,
        color="objective",
        dimensions=param_names,
        color_continuous_scale="Viridis_r",
    )
    fig.update_layout(template=tmpl)
    return fig
