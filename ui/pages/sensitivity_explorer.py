"""Parameter Sensitivity Explorer — browse persisted Sobol results.

Loads a persisted Sobol artifact (osmose.calibration.sobol_io) and renders a ranked
S1/ST tornado, a table, and two exports. Read-only; no engine/config dependency.
"""

from __future__ import annotations

import plotly.graph_objects as go
from shiny import reactive, render, ui
from shiny.types import SilentException
from shinywidgets import output_widget, render_plotly

from osmose.calibration.sobol_io import (
    influential_keys,
    list_sobol_results,
    load_sobol_result,
    rank_rows,
    rows_to_csv,
)
from osmose.logging import setup_logging
from ui.pages.calibration_charts import make_sobol_tornado
from ui.styles import STYLE_MONO_KEY

_log = setup_logging("osmose.sensitivity_explorer")

_EMPTY_MSG = (
    "No sensitivity result to display — pick a saved run above, or run one from "
    "Calibration → Results → Sensitivity."
)


def _tpl(input) -> str:
    from ui.state import get_theme_mode

    return "osmose" if get_theme_mode(input) == "dark" else "osmose-light"


def _run_choices(summaries) -> dict[str, str]:
    return {
        s[
            "timestamp"
        ]: f"{s['timestamp'][:19]} ({s.get('source', 'unknown')}, n_base={s.get('n_base')})"
        for s in summaries
    }


def sensitivity_explorer_ui():
    return ui.div(
        ui.h3("Parameter Sensitivity Explorer"),
        ui.layout_columns(
            ui.input_select("sens_run", "Result", choices={}),
            ui.output_ui("sens_objective_ui"),
            ui.input_radio_buttons(
                "sens_index", "Indices", choices=["Both", "S1", "ST"], selected="Both", inline=True
            ),
            ui.input_select(
                "sens_sort",
                "Sort by",
                choices={"ST": "Total (ST)", "S1": "First-order (S1)", "name": "Name"},
            ),
            ui.input_slider(
                "sens_threshold", "Influence threshold (ST)", min=0, max=1, value=0.05, step=0.01
            ),
            col_widths=[3, 3, 2, 2, 2],
        ),
        output_widget("sens_tornado"),
        ui.output_ui("sens_table"),
        ui.div(
            ui.download_button("sens_export_csv", "Download ranked CSV"),
            ui.download_button("sens_export_keys", "Download influential keys"),
            class_="d-flex gap-2 mt-2",
        ),
    )


def sensitivity_explorer_server(input, output, session, state):
    # `state` is unused (page is config-independent) but kept for the standard
    # *_server(input, output, session, state) call signature.
    _last_choices: reactive.Value[dict] = reactive.Value({})

    def _safe(getter, default=None):
        try:
            return getter()
        except (SilentException, AttributeError):
            return default

    @reactive.effect
    def _populate_runs():
        if input.main_nav() != "sensitivity":
            return
        try:
            summaries = list_sobol_results()
        except Exception:  # noqa: BLE001 — never crash the page on a discovery error
            return
        choices = _run_choices(summaries)
        with reactive.isolate():
            if choices == _last_choices.get():
                return
        _last_choices.set(choices)
        ui.update_select("sens_run", choices=choices)

    @reactive.calc
    def _result():
        ts = _safe(input.sens_run)
        if not ts:
            return None
        try:
            return load_sobol_result(ts)
        except Exception:  # noqa: BLE001 — degrade to empty state on a bad/missing artifact
            _log.warning("Failed to load sobol result %r", ts, exc_info=True)
            return None

    def _obj_idx(r) -> int:
        n = int(r.get("n_objectives", 1))
        return max(0, min(int(_safe(input.sens_objective, "0") or 0), n - 1))

    @reactive.calc
    def _rows():
        r = _result()
        if r is None:
            return []
        return rank_rows(r, objective_idx=_obj_idx(r), sort=_safe(input.sens_sort, "ST") or "ST")

    @render.ui
    def sens_objective_ui():
        r = _result()
        if r is None or int(r.get("n_objectives", 1)) <= 1:
            return ui.div()
        names = r.get("objective_names")
        n = int(r.get("n_objectives", 1))
        choices = {str(i): (names[i] if names and i < len(names) else f"obj_{i}") for i in range(n)}
        return ui.input_select("sens_objective", "Objective", choices=choices, selected="0")

    @render_plotly
    def sens_tornado():
        if _result() is None:
            return go.Figure().update_layout(title=_EMPTY_MSG, template=_tpl(input))
        return make_sobol_tornado(
            _rows(),
            indices=_safe(input.sens_index, "Both") or "Both",
            threshold=float(_safe(input.sens_threshold, 0.05) or 0.05),
            template=_tpl(input),
        )

    @render.ui
    def sens_table():
        if _result() is None:
            return ui.p(_EMPTY_MSG, class_="text-muted")
        rows = _rows()
        thr = float(_safe(input.sens_threshold, 0.05) or 0.05)
        infl = set(influential_keys(rows, thr))
        body = [
            ui.tags.tr(
                ui.tags.td(r["param"], style=STYLE_MONO_KEY),
                ui.tags.td(f"{r['s1']:.3g}"),
                ui.tags.td(f"{r['st']:.3g}"),
                ui.tags.td(
                    ui.tags.span("influential", class_="badge bg-success")
                    if r["param"] in infl
                    else ""
                ),
            )
            for r in rows
        ]
        return ui.tags.table(
            ui.tags.thead(
                ui.tags.tr(
                    ui.tags.th("Param"),
                    ui.tags.th("S1"),
                    ui.tags.th("ST"),
                    ui.tags.th("Influential"),
                )
            ),
            ui.tags.tbody(*body),
            class_="table table-sm table-striped",
            style="font-size: 13px;",
        )

    @render.download(filename="sensitivity_ranked.csv")
    def sens_export_csv():
        yield rows_to_csv(_rows())

    @render.download(filename="influential_keys.txt")
    def sens_export_keys():
        thr = float(_safe(input.sens_threshold, 0.05) or 0.05)
        yield "\n".join(influential_keys(_rows(), thr))
