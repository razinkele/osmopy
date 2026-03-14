# ui/pages/calibration.py
"""Calibration page - multi-objective optimization and sensitivity analysis."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shinywidgets import output_widget, render_plotly

from ui.pages.calibration_charts import (
    make_convergence_chart,
    make_pareto_chart,
    make_sensitivity_chart,
)
from ui.pages.calibration_handlers import (
    _make_progress_callback,
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
    "make_pareto_chart",
    "make_sensitivity_chart",
    "_make_progress_callback",
]


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
            ui.p(
                "Select parameters to optimize:",
                style=STYLE_HINT_BLOCK,
            ),
            ui.output_ui("free_param_selector"),
            ui.hr(),
            ui.h5("Objectives"),
            ui.input_file("observed_biomass", "Upload observed biomass CSV", accept=[".csv"]),
            ui.input_file("observed_diet", "Upload observed diet matrix CSV", accept=[".csv"]),
            ui.hr(),
            ui.layout_columns(
                ui.input_action_button(
                    "btn_start_cal", "Start Calibration", class_="btn-success w-100"
                ),
                ui.input_action_button("btn_stop_cal", "Stop", class_="btn-danger w-100"),
                col_widths=[8, 4],
            ),
        ),
        # Right: Results
        ui.navset_card_tab(
            ui.nav_panel(
                "Progress",
                ui.div(
                    ui.output_text("cal_status"),
                    output_widget("convergence_chart"),
                ),
            ),
            ui.nav_panel(
                "Pareto Front",
                ui.div(
                    output_widget("pareto_chart"),
                ),
            ),
            ui.nav_panel(
                "Best Parameters",
                ui.div(
                    ui.output_ui("best_params_table"),
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
                    output_widget("sensitivity_chart"),
                ),
            ),
        ),
        col_widths=[4, 8],
        ),
        class_="osm-split-layout",
        id="split_calibration",
    )


def calibration_server(input, output, session, state):
    cal_history = reactive.value([])
    cal_F = reactive.value(None)
    cal_X = reactive.value(None)
    sensitivity_result = reactive.value(None)
    cal_thread = reactive.value(None)
    surrogate_status = reactive.value("")

    def _tmpl() -> str:
        mode = get_theme_mode(input)
        return "osmose" if mode == "dark" else "osmose-light"

    # Register event handlers (start/stop calibration, sensitivity)
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
    )

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
        cfg = state.config.get()
        n_str = cfg.get("simulation.nspecies", "3")
        n_species = int(n_str) if n_str else 3
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
        return make_convergence_chart(cal_history.get(), tmpl=_tmpl())

    @render_plotly
    def pareto_chart():
        F = cal_F.get()
        if F is None:
            return go.Figure().update_layout(
                title="Pareto Front (run calibration first)", template=_tmpl()
            )
        return make_pareto_chart(F, ["Biomass RMSE", "Diet Distance"], tmpl=_tmpl())

    @render.ui
    def best_params_table():
        X = cal_X.get()
        F = cal_F.get()
        if X is None or F is None:
            return ui.div(
                "Run calibration to see best parameters.",
                style=STYLE_EMPTY,
            )
        order = np.argsort(F.sum(axis=1))[:10]  # type: ignore[union-attr]
        with reactive.isolate():
            cfg = state.config.get()
        selected = collect_selected_params(input, state, config=cfg)
        headers = [ui.tags.th(p["key"].split(".")[-1]) for p in selected]
        headers.append(ui.tags.th("Total Obj"))
        rows = []
        for idx in order:  # type: ignore[union-attr]
            cells = [ui.tags.td(f"{v:.4f}") for v in X[idx]]  # type: ignore[index]
            cells.append(ui.tags.td(f"{F[idx].sum():.4f}"))
            rows.append(ui.tags.tr(*cells))
        return ui.tags.table(
            ui.tags.thead(ui.tags.tr(*headers)),
            ui.tags.tbody(*rows),
            class_="table table-sm table-striped",
        )

    @render_plotly
    def sensitivity_chart():
        result = sensitivity_result.get()
        if result is None:
            return go.Figure().update_layout(title="Sensitivity (click Run)", template=_tmpl())
        return make_sensitivity_chart(result, tmpl=_tmpl())
