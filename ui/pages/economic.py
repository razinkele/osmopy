"""Economic page — fleet economics and market configuration (Python engine only)."""

from shiny import render

from ui.components.placeholder import engine_mode_hint, placeholder_content, placeholder_ui
from ui.state import AppState


def economic_ui():
    return placeholder_ui("economic", "Economic Configuration", "economic_content")


def economic_server(input, output, session, state: AppState):
    @render.ui
    def economic_content():
        if state.engine_mode.get() != "python":
            return engine_mode_hint("Economic module")
        return placeholder_content(
            heading="Economic Module",
            intro=(
                "Configure fleet economics, market dynamics, and quota "
                "management. This module couples economic decision-making "
                "with ecological simulation."
            ),
            note=(
                "Fleet cost structures, market prices, and quota parameters "
                "will be available here once the economic engine module is "
                "implemented."
            ),
            bullets=[
                "Fleet cost structures (fuel, labour, maintenance)",
                "Market prices and demand curves",
                "Quota management and allocation rules",
                "Effort dynamics and fleet behaviour",
            ],
        )
