# ui/pages/calibration.py
"""Calibration page - multi-objective optimization and sensitivity analysis."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shiny.render import DataGrid
from shinywidgets import output_widget, render_plotly

from ui.pages.calibration_charts import (
    make_convergence_chart,
    make_correlation_chart,
    make_pareto_chart,
    make_sensitivity_chart,
)
from ui.pages.calibration_handlers import (
    _make_progress_callback,
    _resolve_optimum_weights,
    build_free_params,
    collect_selected_params,
    get_calibratable_params,
    register_calibration_handlers,
)
from ui.pages.run import copy_data_files
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.state import get_theme_mode
from ui.styles import STYLE_EMPTY, STYLE_HINT_BLOCK

# Re-export for backward compatibility (tests import these from calibration)
__all__ = [
    "calibration_ui",
    "calibration_server",
    "get_calibratable_params",
    "collect_selected_params",
    "build_free_params",
    "make_convergence_chart",
    "make_correlation_chart",
    "make_pareto_chart",
    "make_sensitivity_chart",
    "_make_progress_callback",
]


# -- Pure module-level helpers for surrogate-optimum rendering ----------------


def _obj_labels_for_surrogate(optimum: dict | None) -> list[str]:
    """Best-effort objective labels for the weights row and tables.

    ``find_optimum`` (``surrogate.py:123``) does NOT currently populate an
    ``objective_labels`` key — the helper falls back to ``obj_{i}`` for
    every position. The ``objective_labels`` branch exists so a caller
    can inject friendlier labels by adding that key to the payload
    before ``post_surrogate_optimum`` (out of scope for Phase 3;
    reserved for a future follow-up).
    """
    if optimum is None:
        return []
    labels = optimum.get("objective_labels")
    if labels:
        return list(labels)
    obj = optimum.get("predicted_objectives")
    n = int(getattr(obj, "__len__", lambda: 0)()) if obj is not None else 0
    return [f"obj_{i}" for i in range(n)]


def _render_weights_inputs(mode: str, optimum: dict | None):
    """Pure helper used by the ``weights_inputs`` @render.ui delegate."""
    if mode != "weighted":
        return ui.TagList()  # empty — Pareto mode shows no weights
    if optimum is None:
        return ui.help_text(
            "Run the surrogate workflow first — weights match the fitted surrogate's objectives.",
            class_="small text-muted",
        )
    labels = _obj_labels_for_surrogate(optimum)
    n_obj = len(labels)
    rows = []
    for i in range(n_obj):
        rows.append(
            ui.input_numeric(
                f"cal_weight_{i}",
                f"w[{labels[i]}]",
                value=1.0,
                min=0.0,
                step=0.1,
            )
        )
    return ui.TagList(*rows)


def _render_pareto_scatter(optimum: dict | None, tmpl: str = "osmose"):
    """Pure helper for the ``surrogate_pareto_scatter`` delegate.

    Returns an empty ``plotly.graph_objects.Figure`` for missing data
    OR for n_obj >= 3 (the scatter is not meaningful past 2 objectives).
    """
    if optimum is None or "pareto" not in optimum:
        return go.Figure()
    F = np.asarray(optimum["pareto"]["objectives"])
    if F.ndim != 2 or F.shape[1] >= 3:
        return go.Figure()
    labels = _obj_labels_for_surrogate(optimum)
    return make_pareto_chart(F, labels, tmpl=tmpl)


def _render_pareto_table(optimum: dict | None):
    """Pure helper for the ``surrogate_pareto_table`` delegate.

    Returns a ``shiny.render.DataGrid`` wrapping an empty DataFrame
    when no Pareto set is present, otherwise the full Pareto set
    laid out as obj_*, ±obj_*, param_* columns.
    """
    if optimum is None or "pareto" not in optimum:
        return DataGrid(pd.DataFrame(), selection_mode="row")
    F = np.asarray(optimum["pareto"]["objectives"])
    U = np.asarray(optimum["pareto"]["uncertainty"])
    P = np.asarray(optimum["pareto"]["params"])
    labels = _obj_labels_for_surrogate(optimum)
    n_obj = F.shape[1] if F.ndim == 2 else 0
    n_params = P.shape[1] if P.ndim == 2 else 0
    data: dict[str, np.ndarray] = {}
    for i in range(n_obj):
        data[labels[i] if i < len(labels) else f"obj_{i}"] = F[:, i]
    for i in range(n_obj):
        data[f"\u00b1{labels[i] if i < len(labels) else f'obj_{i}'}"] = U[:, i]
    for j in range(n_params):
        data[f"param_{j}"] = P[:, j]
    return DataGrid(pd.DataFrame(data), selection_mode="row")


def _render_weighted_summary(optimum: dict | None, weights: list[float] | None):
    """Pure helper for the ``surrogate_weighted_summary`` delegate.

    When ``weights`` is None/empty the weighted-sum scalar is omitted.
    Otherwise the summary line shows ``w·means = X`` per the spec.
    """
    if optimum is None or "pareto" in optimum:
        return ui.help_text(
            "Switch to Weighted sum and run the surrogate workflow "
            "to see the single weighted-optimum point.",
            class_="small text-muted",
        )
    obj = np.asarray(optimum["predicted_objectives"])
    unc = np.asarray(optimum["predicted_uncertainty"])
    params = np.asarray(optimum["params"])
    scalar_line = None
    if weights:
        w = np.asarray(weights, dtype=float)
        w_dot_means = float(w @ obj)
        scalar_line = ui.tags.span(
            f"weighted sum (w\u00b7means) = {w_dot_means:.4g}",
            class_="fw-semibold",
        )
    rows = [
        ui.tags.strong("Weighted optimum"),
        ui.br(),
    ]
    if scalar_line is not None:
        rows += [scalar_line, ui.br()]
    rows += [
        ui.tags.span(
            f"objectives = {np.round(obj, 4).tolist()} (\u00b1{np.round(unc, 4).tolist()})"
        ),
        ui.br(),
        ui.tags.span(f"parameters = {np.round(params, 4).tolist()}"),
    ]
    return ui.div(*rows, class_="p-3 border rounded bg-light")


# -- Shiny UI / Server --------------------------------------------------------


def calibration_ui():
    return ui.div(
        expand_tab("Calibration Setup", "calibration"),
        ui.layout_columns(
            # Left: Configuration
            ui.card(
                collapsible_card_header("Calibration Setup", "calibration"),
                ui.input_select(
                    "cal_algorithm",
                    "Algorithm",
                    choices={
                        "nsga2": "NSGA-II (Direct)",
                        "surrogate": "GP Surrogate",
                    },
                ),
                ui.input_numeric("cal_pop_size", "Population size", value=50, min=10, max=500),
                ui.input_numeric("cal_generations", "Generations", value=100, min=10, max=1000),
                ui.input_numeric("cal_n_parallel", "Parallel workers", value=4, min=1, max=32),
                ui.hr(),
                ui.h5("Free Parameters"),
                ui.p("Select parameters to optimize:", style=STYLE_HINT_BLOCK),
                ui.output_ui("free_param_selector"),
                ui.hr(),
                ui.h5("Objectives"),
                ui.input_file("observed_biomass", "Upload observed biomass CSV", accept=[".csv"]),
                ui.input_file("observed_diet", "Upload observed diet matrix CSV", accept=[".csv"]),
                # Banded loss section
                ui.hr(),
                ui.input_checkbox("cal_banded_loss_enabled", "Enable Banded Loss Objective"),
                ui.panel_conditional(
                    "input.cal_banded_loss_enabled",
                    ui.input_select(
                        "cal_banded_source",
                        "Targets source",
                        choices={
                            "baltic": "Baltic defaults",
                            "upload": "Upload custom...",
                        },
                    ),
                    ui.panel_conditional(
                        "input.cal_banded_source === 'upload'",
                        ui.input_file(
                            "cal_banded_targets_file",
                            "Upload targets CSV",
                            accept=[".csv"],
                        ),
                    ),
                    ui.input_numeric(
                        "cal_w_stability", "Stability weight", value=5.0, min=0.0, max=100.0
                    ),
                    ui.input_numeric(
                        "cal_w_worst", "Worst-species weight", value=0.5, min=0.0, max=10.0
                    ),
                ),
                ui.hr(),
                ui.input_checkbox(
                    "cal_preflight_enabled",
                    "Pre-flight screening (recommended)",
                    value=True,
                ),
                ui.input_numeric(
                    "cal_preflight_workers",
                    "Workers",
                    value=1,
                    min=1,
                    max=max(1, os.cpu_count() or 1),
                    step=1,
                ),
                ui.help_text(
                    "Parallel evaluators for preflight sample runs. 1 = sequential (default).",
                    class_="small text-muted",
                ),
                ui.layout_columns(
                    ui.input_action_button(
                        "btn_start_cal", "Start Calibration", class_="btn-success w-100"
                    ),
                    ui.input_action_button("btn_stop_cal", "Stop", class_="btn-danger w-100"),
                    col_widths=[8, 4],
                ),
            ),
            # Right: Results — two-level grouped navigation
            ui.navset_card_pill(
                ui.nav_panel(
                    "Run",
                    ui.div(
                        ui.output_text("cal_status"),
                        ui.output_ui("run_header"),
                        ui.output_ui("ices_proxy_table"),
                        output_widget("convergence_chart"),
                    ),
                ),
                ui.nav_panel(
                    "Results",
                    ui.navset_tab(
                        ui.nav_panel(
                            "Pareto Front",
                            ui.div(
                                ui.input_radio_buttons(
                                    "cal_optimum_mode",
                                    "Optimum",
                                    choices={
                                        "pareto": "Pareto front",
                                        "weighted": "Weighted sum",
                                    },
                                    selected="pareto",
                                    inline=True,
                                ),
                                ui.output_ui("weights_inputs"),
                                ui.help_text(
                                    "Weights are raw non-negative floats. "
                                    "Scaling is irrelevant for ranking within "
                                    "a single search — [0.3, 0.7] and [3, 7] "
                                    "pick the same point.",
                                    class_="small text-muted",
                                ),
                                output_widget("pareto_chart"),
                                ui.panel_conditional(
                                    "input.cal_optimum_mode == 'pareto'",
                                    ui.layout_columns(
                                        output_widget("surrogate_pareto_scatter"),
                                        ui.output_data_frame("surrogate_pareto_table"),
                                        col_widths=[6, 6],
                                    ),
                                ),
                                ui.panel_conditional(
                                    "input.cal_optimum_mode == 'weighted'",
                                    ui.output_ui("surrogate_weighted_summary"),
                                ),
                            ),
                        ),
                        ui.nav_panel(
                            "Correlations",
                            ui.div(output_widget("correlation_chart")),
                        ),
                        ui.nav_panel(
                            "Best Parameters",
                            ui.div(
                                ui.output_ui("best_params_table"),
                                ui.hr(),
                                ui.h6("Multi-Seed Validation"),
                                ui.layout_columns(
                                    ui.input_numeric(
                                        "cal_val_top_n", "Top N", value=5, min=1, max=50
                                    ),
                                    ui.input_numeric(
                                        "cal_val_seeds", "Seeds", value=5, min=2, max=20
                                    ),
                                    ui.input_action_button(
                                        "btn_validate",
                                        "Validate",
                                        class_="btn-info w-100",
                                    ),
                                    col_widths=[4, 4, 4],
                                ),
                                ui.output_ui("validation_table"),
                            ),
                        ),
                        ui.nav_panel(
                            "Sensitivity",
                            ui.div(
                                ui.input_action_button(
                                    "btn_sensitivity",
                                    "Run Sensitivity Analysis",
                                    class_="btn-info w-100",
                                ),
                                ui.output_ui("sensitivity_objective_selector"),
                                output_widget("sensitivity_chart"),
                            ),
                        ),
                        id="cal_results_tabs",
                    ),
                ),
                ui.nav_panel(
                    "History",
                    ui.div(
                        ui.output_ui("history_banner"),
                        ui.output_ui("history_list"),
                    ),
                ),
                id="cal_groups",
            ),
            col_widths=[4, 8],
        ),
        class_="osm-split-layout",
        id="split_calibration",
    )


def calibration_server(input, output, session, state):
    cal_history = reactive.value([])
    cal_F: reactive.value[np.ndarray | None] = reactive.value(None)
    cal_X: reactive.value[np.ndarray | None] = reactive.value(None)
    sensitivity_result = reactive.value(None)
    cal_thread = reactive.value(None)
    surrogate_status = reactive.value("")
    validation_result = reactive.value(None)
    cal_param_names = reactive.value([])
    history_banner_text = reactive.value("")
    history_trigger = reactive.value(0)
    preflight_result = reactive.value(None)
    surrogate_optimum = reactive.value(None)

    def _tmpl() -> str:
        mode = get_theme_mode(input)
        return "osmose" if mode == "dark" else "osmose-light"

    # Register event handlers
    register_calibration_handlers(
        input=input,
        output=output,
        session=session,
        state=state,
        cal_history=cal_history,
        cal_F=cal_F,
        cal_X=cal_X,
        sensitivity_result=sensitivity_result,
        cal_thread=cal_thread,
        surrogate_status=surrogate_status,
        copy_data_files=copy_data_files,
        validation_result=validation_result,
        cal_param_names=cal_param_names,
        preflight_result=preflight_result,
        history_banner_text=history_banner_text,
        history_trigger=history_trigger,
        surrogate_optimum=surrogate_optimum,
    )

    @render.ui
    def weights_inputs():
        return _render_weights_inputs(input.cal_optimum_mode(), surrogate_optimum.get())

    @render_plotly
    def surrogate_pareto_scatter():
        return _render_pareto_scatter(surrogate_optimum.get(), tmpl=_tmpl())

    @render.data_frame
    def surrogate_pareto_table():
        return _render_pareto_table(surrogate_optimum.get())

    @render.ui
    def surrogate_weighted_summary():
        optimum = surrogate_optimum.get()
        n_obj = len(_obj_labels_for_surrogate(optimum))
        weights = _resolve_optimum_weights(input, n_obj) if optimum else None
        return _render_weighted_summary(optimum, weights)

    @render.text
    def cal_status():
        surr = surrogate_status.get()
        if surr:
            return surr
        hist = cal_history.get()
        if not hist:
            return "Ready. Configure parameters and objectives, then click Start."
        return f"Generation {len(hist)} — Best: {hist[-1]:.4f}"

    @render.ui
    def free_param_selector():
        state.load_trigger.get()
        with reactive.isolate():
            cfg = state.config.get()
        n_str = cfg.get("simulation.nspecies", "3")
        try:
            n_species = int(float(n_str or "3"))
        except (ValueError, TypeError):
            n_species = 3
        params = get_calibratable_params(state.registry, n_species)
        checkboxes = [
            ui.input_checkbox(
                f"cal_param_{p['key'].replace('.', '_')}",
                p["label"],
                value=False,
            )
            for p in params
        ]
        return ui.div(*checkboxes)

    @render_plotly
    def convergence_chart():
        # Read the active checkpoint to derive (optimizer, phase) for the
        # best-ever reference line. _scan_results_dir is module-level and
        # importable; _live_snapshot is a @reactive.poll local to
        # register_calibration_handlers and CANNOT be imported.
        from ui.pages.calibration_handlers import _scan_results_dir
        try:
            snap = _scan_results_dir()
            if snap.active.kind == "ok":
                opt = snap.active.checkpoint.optimizer
                ph = snap.active.checkpoint.phase
            else:
                opt = ph = None
        except Exception:  # noqa: BLE001 — defensive fallback; should never fire
            opt = ph = None
        return make_convergence_chart(
            cal_history.get(), tmpl=_tmpl(), optimizer=opt, phase=ph,
        )

    @render_plotly
    def pareto_chart():
        F = cal_F.get()
        if F is None:
            return go.Figure().update_layout(
                title="Pareto Front (run calibration first)", template=_tmpl()
            )
        obj_labels = [f"Obj {i + 1}" for i in range(F.shape[1])]
        return make_pareto_chart(F, obj_labels, tmpl=_tmpl())

    @render_plotly
    def correlation_chart():
        X = cal_X.get()
        F = cal_F.get()
        names = cal_param_names.get()
        if X is None or F is None or not names:
            return go.Figure().update_layout(
                title="Parameter Correlations (run calibration first)", template=_tmpl()
            )
        return make_correlation_chart(X, F, names, tmpl=_tmpl())

    @render.ui
    def best_params_table():
        X = cal_X.get()
        F = cal_F.get()
        if X is None or F is None:
            return ui.div("Run calibration to see best parameters.", style=STYLE_EMPTY)
        order = np.argsort(F.sum(axis=1))[:10]
        names = cal_param_names.get()
        if not names:
            with reactive.isolate():
                cfg = state.config.get()
            selected = collect_selected_params(input, state, config=cfg)
            names = [p["key"].split(".")[-1] for p in selected]
        headers = [ui.tags.th(n) for n in names]
        headers.append(ui.tags.th("Total Obj"))
        rows = []
        for idx in order:
            cells = [ui.tags.td(f"{v:.4f}") for v in X[idx]]
            cells.append(ui.tags.td(f"{F[idx].sum():.4f}"))
            rows.append(ui.tags.tr(*cells))
        return ui.tags.table(
            ui.tags.thead(ui.tags.tr(*headers)),
            ui.tags.tbody(*rows),
            class_="table table-sm table-striped",
        )

    @render.ui
    def validation_table():
        result = validation_result.get()
        if result is None:
            return ui.div()
        rankings = result["rankings"]
        scores = result["scores"]
        headers = [
            ui.tags.th("Rank"),
            ui.tags.th("Mean"),
            ui.tags.th("Std"),
            ui.tags.th("CV"),
            ui.tags.th("Worst"),
        ]
        rows = []
        for i, idx in enumerate(rankings):
            s = scores[idx]
            rows.append(
                ui.tags.tr(
                    ui.tags.td(str(i + 1)),
                    ui.tags.td(f"{s['mean']:.4f}"),
                    ui.tags.td(f"{s['std']:.4f}"),
                    ui.tags.td(f"{s['cv']:.1%}"),
                    ui.tags.td(f"{s['worst_value']:.4f}"),
                )
            )
        return ui.tags.table(
            ui.tags.thead(ui.tags.tr(*headers)),
            ui.tags.tbody(*rows),
            class_="table table-sm table-striped",
        )

    @render.ui
    def sensitivity_objective_selector():
        result = sensitivity_result.get()
        if result is None or "objective_names" not in result:
            return ui.div()
        choices = {str(i): name for i, name in enumerate(result["objective_names"])}
        return ui.input_select("cal_sens_objective", "Objective", choices=choices)

    @render_plotly
    def sensitivity_chart():
        result = sensitivity_result.get()
        if result is None:
            return go.Figure().update_layout(title="Sensitivity (click Run)", template=_tmpl())
        selected = 0
        if "objective_names" in result:
            try:
                selected = int(input.cal_sens_objective())
            except (AttributeError, ValueError, TypeError):
                selected = 0
        return make_sensitivity_chart(result, tmpl=_tmpl(), selected_objective=selected)

    @render.ui
    def history_banner():
        text = history_banner_text.get()
        if not text:
            return ui.div()
        return ui.div(
            ui.tags.em(text),
            class_="alert alert-info py-1 px-2 mb-2",
        )

    @render.ui
    def history_list():
        history_trigger.get()  # Re-render when history changes (save/delete)
        from osmose.calibration.history import list_runs

        runs = list_runs()
        if not runs:
            return ui.div("No calibration history yet.", style=STYLE_EMPTY)
        cards = []
        for i, run in enumerate(runs):
            ts = run["timestamp"]
            algo = run["algorithm"].upper()
            best = run["best_objective"]
            n_p = run["n_params"]
            dur = run["duration_seconds"]
            dur_str = f"{dur // 60}m {dur % 60}s" if dur >= 60 else f"{dur}s"
            cards.append(
                ui.div(
                    ui.div(
                        ui.tags.strong(f"{ts} — {algo}"),
                        ui.span(
                            f" Best: {best:.3f} | {n_p} params | {dur_str}",
                            style="margin-left: 8px; opacity: 0.8;",
                        ),
                        ui.div(
                            ui.input_action_button(
                                f"btn_load_run_{i}",
                                "Load",
                                class_="btn-sm btn-outline-primary me-1",
                            ),
                            ui.input_action_button(
                                f"btn_delete_run_{i}",
                                "Delete",
                                class_="btn-sm btn-outline-danger",
                            ),
                            style="display: inline-block; float: right;",
                        ),
                        style="padding: 8px;",
                    ),
                    class_="border rounded mb-2",
                )
            )
        return ui.div(
            ui.h6(f"Calibration History ({len(runs)} runs)"),
            *cards,
        )
