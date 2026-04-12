"""Fishing configuration page."""

from shiny import ui, reactive, render

from osmose.schema.fishing import FISHING_FIELDS
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.components.param_form import render_field
from ui.pages._helpers import collect_resolved_keys
from ui.state import sync_inputs

FISHING_GLOBAL_KEYS: list[str] = [f.key_pattern for f in FISHING_FIELDS if not f.indexed]


def fishing_ui():
    return ui.div(
        expand_tab("Fisheries Module", "fishing"),
        ui.layout_columns(
            ui.card(
                collapsible_card_header("Fisheries Module", "fishing"),
                ui.output_ui("fishing_global_fields"),
                ui.hr(),
                ui.input_numeric("n_fisheries", "Number of fisheries", value=1, min=0, max=20),
                ui.output_ui("fishery_panels"),
            ),
            ui.card(
                ui.card_header("Marine Protected Areas"),
                ui.input_numeric("n_mpas", "Number of MPAs", value=0, min=0, max=10),
                ui.output_ui("mpa_panels"),
            ),
            col_widths=[8, 4],
        ),
        class_="osm-split-layout",
        id="split_fishing",
    )


def fishing_server(input, output, session, state):
    global_fields = [f for f in FISHING_FIELDS if not f.indexed]

    @render.ui
    def fishing_global_fields():
        state.load_trigger.get()
        with reactive.isolate():
            cfg = state.config.get()
        return ui.div(*[render_field(f, config=cfg) for f in global_fields])

    @render.ui
    def fishery_panels():
        state.load_trigger.get()
        n = input.n_fisheries()
        with reactive.isolate():
            cfg = state.config.get()
        fishery_fields = [f for f in FISHING_FIELDS if f.indexed and "fsh" in f.key_pattern]
        panels = []
        for i in range(n):
            card = ui.card(
                ui.card_header(f"Fishery {i}"),
                *[render_field(f, species_idx=i, config=cfg) for f in fishery_fields],
            )
            panels.append(card)
        return ui.div(*panels)

    @render.ui
    def mpa_panels():
        state.load_trigger.get()
        n = input.n_mpas()
        with reactive.isolate():
            cfg = state.config.get()
        mpa_fields = [f for f in FISHING_FIELDS if f.indexed and "mpa" in f.key_pattern]
        panels = []
        for i in range(n):
            card = ui.card(
                ui.card_header(f"MPA {i}"),
                *[render_field(f, species_idx=i, config=cfg) for f in mpa_fields],
            )
            panels.append(card)
        return ui.div(*panels)

    @reactive.effect
    def sync_fishing_inputs():
        sync_inputs(input, state, FISHING_GLOBAL_KEYS)

    @reactive.effect
    def sync_fishery_inputs():
        n = input.n_fisheries()
        fishery_fields = [f for f in FISHING_FIELDS if f.indexed and "fsh" in f.key_pattern]
        sync_inputs(input, state, collect_resolved_keys(fishery_fields, n))

    @reactive.effect
    def sync_mpa_inputs():
        n = input.n_mpas()
        mpa_fields = [f for f in FISHING_FIELDS if f.indexed and "mpa" in f.key_pattern]
        sync_inputs(input, state, collect_resolved_keys(mpa_fields, n))
