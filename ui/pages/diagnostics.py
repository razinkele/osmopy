"""Diagnostics page — Python engine performance and runtime metrics."""

from shiny import render, ui

from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.styles import STYLE_EMPTY


def diagnostics_ui():
    return ui.div(
        expand_tab("Engine Diagnostics", "diagnostics"),
        ui.layout_columns(
            # Left: Timing breakdown
            ui.card(
                collapsible_card_header("Engine Diagnostics", "diagnostics"),
                ui.div(
                    ui.h5("Performance Dashboard"),
                    ui.p(
                        "After running the Python engine, timing breakdowns, "
                        "Numba JIT status, and memory usage will appear here.",
                        style=STYLE_EMPTY,
                    ),
                    ui.hr(),
                    ui.h5("Process Timing"),
                    ui.output_ui("diag_timing"),
                    ui.hr(),
                    ui.h5("Numba JIT Status"),
                    ui.output_ui("diag_numba"),
                    ui.hr(),
                    ui.h5("Memory Profile"),
                    ui.output_ui("diag_memory"),
                ),
            ),
            # Right: Comparison
            ui.card(
                ui.card_header("Engine Comparison"),
                ui.div(
                    ui.p(
                        "Run both Java and Python engines on the same config "
                        "to see a side-by-side timing comparison.",
                        style=STYLE_EMPTY,
                    ),
                    ui.output_ui("diag_comparison"),
                ),
            ),
            col_widths=[7, 5],
        ),
        class_="osm-split-layout",
        id="split_diagnostics",
    )


def diagnostics_server(input, output, session, state):
    @render.ui
    def diag_timing():
        if state.engine_mode.get() != "python":
            return ui.p("Switch to Python engine to view diagnostics.", style=STYLE_EMPTY)
        result = state.run_result.get()
        if result is None:
            return ui.p("No run results yet. Run the Python engine first.", style=STYLE_EMPTY)
        timing = getattr(result, "timing", None)
        if timing is None:
            return ui.p("No timing data available for this run.", style=STYLE_EMPTY)
        rows = []
        for process, seconds in sorted(timing.items()):
            rows.append(ui.tags.tr(
                ui.tags.td(process, style="font-weight: 500;"),
                ui.tags.td(f"{seconds:.3f}s"),
            ))
        return ui.tags.table(
            ui.tags.thead(ui.tags.tr(ui.tags.th("Process"), ui.tags.th("Time"))),
            ui.tags.tbody(*rows),
            class_="table table-sm table-striped",
        )

    @render.ui
    def diag_numba():
        return ui.p("Numba JIT compilation info will appear after a Python engine run.",
                     style=STYLE_EMPTY)

    @render.ui
    def diag_memory():
        return ui.p("Memory usage profiling will appear after a Python engine run.",
                     style=STYLE_EMPTY)

    @render.ui
    def diag_comparison():
        return ui.p("Comparison data will appear after running both engines.",
                     style=STYLE_EMPTY)
