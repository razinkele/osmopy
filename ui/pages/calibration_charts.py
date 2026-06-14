# ui/pages/calibration_charts.py
"""Chart/visualization helper functions for the calibration page."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go


def make_convergence_chart(
    history: list[float],
    tmpl: str = "osmose",
    optimizer: str | None = None,
    phase: str | None = None,
) -> go.Figure:
    """Line chart of best objective value per generation.

    When ``optimizer`` AND ``phase`` are both given, queries
    :func:`osmose.calibration.history.list_runs` for prior matching runs and
    adds a horizontal dashed reference line at the minimum ``best_objective``
    with annotation ``"best ever: f=<X.XXX>"``. Existing callers that pass
    only ``history`` and ``tmpl`` are unaffected — no reference line drawn.
    """
    if not history:
        return go.Figure().update_layout(title="Convergence", template=tmpl)
    import plotly.express as px

    fig = px.line(x=list(range(len(history))), y=history, title="Convergence")
    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="Best Objective",
        template=tmpl,
    )
    if optimizer is not None and phase is not None:
        from osmose.calibration import history as hist_mod

        try:
            runs = hist_mod.list_runs()
        except Exception:  # noqa: BLE001 — defensive; history is optional context
            runs = []
        matching = [r for r in runs if r.get("algorithm") == optimizer and r.get("phase") == phase]
        finite = [
            r["best_objective"]
            for r in matching
            if r.get("best_objective") not in (None, float("inf"))
        ]
        if finite:
            best_ever = min(finite)
            fig.add_hline(
                y=best_ever,
                line_dash="dash",
                annotation_text=f"best ever: f={best_ever:.3f}",
                annotation_position="top left",
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


def make_sobol_tornado(
    rows: list[dict],
    *,
    indices: str = "Both",
    threshold: float = 0.05,
    template: str = "osmose",
) -> go.Figure:
    """Horizontal tornado of Sobol indices from pre-ranked ``rows``.

    ``rows`` is the output of ``sobol_io.rank_rows`` (already objective-selected and
    sorted). ``indices`` in {"Both","S1","ST"} picks which bars to draw. ST bars are
    colored by influence (``st >= threshold``); a dashed reference line marks the
    threshold. Does no 1-D/2-D dispatch and no I/O.
    """
    if not rows:
        return go.Figure().update_layout(title="Sobol sensitivity", template=template)
    params = [r["param"] for r in rows]
    fig = go.Figure()
    if indices in ("Both", "S1"):
        fig.add_trace(
            go.Bar(
                name="S1 (First-order)",
                y=params,
                x=[r["s1"] for r in rows],
                orientation="h",
                error_x={"type": "data", "array": [r["s1_conf"] for r in rows]},
            )
        )
    if indices in ("Both", "ST"):
        colors = [
            "#d62728" if (not math.isnan(r["st"]) and r["st"] >= threshold) else "#7f7f7f"
            for r in rows
        ]
        fig.add_trace(
            go.Bar(
                name="ST (Total-order)",
                y=params,
                x=[r["st"] for r in rows],
                orientation="h",
                marker={"color": colors},
                error_x={"type": "data", "array": [r["st_conf"] for r in rows]},
            )
        )
    fig.update_layout(barmode="group", title="Sobol sensitivity", template=template)
    fig.update_yaxes(autorange="reversed")  # top-ranked param at the top
    fig.add_vline(x=threshold, line_dash="dash", line_color="#888")
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
    df = pd.DataFrame(X, columns=pd.Index(param_names))
    df["objective"] = F[:, 0] if F.shape[1] == 1 else np.sum(F, axis=1)
    fig = px.parallel_coordinates(
        df,
        color="objective",
        dimensions=param_names,
        color_continuous_scale="Viridis_r",
    )
    fig.update_layout(template=tmpl)
    return fig
