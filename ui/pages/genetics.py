"""Genetics page — Ev-OSMOSE evolutionary genetics configuration (Python engine only)."""

from shiny import ui

from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.styles import STYLE_EMPTY


def genetics_ui():
    return ui.div(
        expand_tab("Genetics Configuration", "genetics"),
        ui.layout_columns(
            ui.card(
                collapsible_card_header("Genetics Configuration", "genetics"),
                ui.div(
                    ui.h5("Ev-OSMOSE Genetics Module"),
                    ui.p(
                        "Configure evolutionary genetics parameters for species traits. "
                        "This module enables heritable trait variation, mutation, and "
                        "natural selection across generations.",
                    ),
                    ui.hr(),
                    ui.p(
                        "Trait heritability, mutation rates, and selection pressure "
                        "parameters will be available here once the Ev-OSMOSE engine "
                        "module is implemented.",
                        style=STYLE_EMPTY,
                    ),
                    ui.tags.ul(
                        ui.tags.li("Trait heritability coefficients per species"),
                        ui.tags.li("Mutation rate and variance"),
                        ui.tags.li("Selection pressure functions"),
                        ui.tags.li("Genetic diversity metrics"),
                        style="color: var(--osm-text-muted); font-size: 0.82rem;",
                    ),
                ),
            ),
            col_widths=[12],
        ),
        class_="osm-split-layout",
        id="split_genetics",
    )


def genetics_server(input, output, session, state):
    pass
