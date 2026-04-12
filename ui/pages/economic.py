"""Economic page — fleet economics and market configuration (Python engine only)."""

from shiny import render, ui

from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.styles import STYLE_EMPTY


def economic_ui():
    return ui.div(
        expand_tab("Economic Configuration", "economic"),
        ui.layout_columns(
            ui.card(
                collapsible_card_header("Economic Configuration", "economic"),
                ui.output_ui("economic_content"),
            ),
            col_widths=[12],
        ),
        class_="osm-split-layout",
        id="split_economic",
    )


def economic_server(input, output, session, state):
    @render.ui
    def economic_content():
        if state.engine_mode.get() != "python":
            return ui.p(
                "Switch to Python engine to access Economic module.", style=STYLE_EMPTY
            )
        return ui.div(
            ui.h5("Economic Module"),
            ui.p(
                "Configure fleet economics, market dynamics, and quota "
                "management. This module couples economic decision-making "
                "with ecological simulation.",
            ),
            ui.hr(),
            ui.p(
                "Fleet cost structures, market prices, and quota parameters "
                "will be available here once the economic engine module is "
                "implemented.",
                style=STYLE_EMPTY,
            ),
            ui.tags.ul(
                ui.tags.li("Fleet cost structures (fuel, labour, maintenance)"),
                ui.tags.li("Market prices and demand curves"),
                ui.tags.li("Quota management and allocation rules"),
                ui.tags.li("Effort dynamics and fleet behaviour"),
                style="color: var(--osm-text-muted); font-size: 0.82rem;",
            ),
        )
