"""Genetics page — Ev-OSMOSE evolutionary genetics configuration (Python engine only)."""

from shiny import render

from ui.components.placeholder import engine_mode_hint, placeholder_content, placeholder_ui
from ui.state import AppState


def genetics_ui():
    return placeholder_ui("genetics", "Genetics Configuration", "genetics_content")


def genetics_server(input, output, session, state: AppState):
    @render.ui
    def genetics_content():
        if state.engine_mode.get() != "python":
            return engine_mode_hint("Genetics")
        return placeholder_content(
            heading="Ev-OSMOSE Genetics Module",
            intro=(
                "Configure evolutionary genetics parameters for species traits. "
                "This module enables heritable trait variation, mutation, and "
                "natural selection across generations."
            ),
            note=(
                "Trait heritability, mutation rates, and selection pressure "
                "parameters will be available here once the Ev-OSMOSE engine "
                "module is implemented."
            ),
            bullets=[
                "Trait heritability coefficients per species",
                "Mutation rate and variance",
                "Selection pressure functions",
                "Genetic diversity metrics",
            ],
        )
