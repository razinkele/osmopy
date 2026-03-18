"""Movement / spatial distribution page."""

from shiny import ui, reactive, render

from osmose.schema.movement import MOVEMENT_FIELDS
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.components.param_form import render_field
from ui.state import sync_inputs

MOVEMENT_GLOBAL_KEYS: list[str] = [f.key_pattern for f in MOVEMENT_FIELDS if not f.indexed]


def movement_ui():
    return ui.div(
        expand_tab("Movement Settings", "movement"),
        ui.layout_columns(
            ui.card(
                collapsible_card_header("Movement Settings", "movement"),
                ui.output_ui("movement_global_fields"),
                ui.hr(),
                ui.h5("Per-Species Distribution Method"),
                ui.output_ui("species_movement_panels"),
            ),
            ui.card(
                ui.card_header("Distribution Maps"),
                ui.input_numeric("n_maps", "Number of distribution maps", value=1, min=0, max=50),
                ui.output_ui("map_panels"),
            ),
            col_widths=[5, 7],
        ),
        class_="osm-split-layout",
        id="split_movement",
    )


def movement_server(input, output, session, state):
    global_fields = [f for f in MOVEMENT_FIELDS if not f.indexed]

    @render.ui
    def movement_global_fields():
        state.load_trigger.get()
        with reactive.isolate():
            cfg = state.config.get()
        return ui.div(*[render_field(f, config=cfg) for f in global_fields if not f.advanced])

    @render.ui
    def species_movement_panels():
        state.load_trigger.get()
        per_species = [f for f in MOVEMENT_FIELDS if f.indexed and "map" not in f.key_pattern]
        with reactive.isolate():
            cfg = state.config.get()
            try:
                n_species = int(float(cfg.get("simulation.nspecies", "3") or "3"))
            except (ValueError, TypeError):
                n_species = 3
        panels = []
        for i in range(n_species):
            panels.extend([render_field(f, species_idx=i, config=cfg) for f in per_species])
        return ui.div(*panels)

    @render.ui
    def map_panels():
        state.load_trigger.get()
        n = input.n_maps()
        with reactive.isolate():
            cfg = state.config.get()
        map_fields = [f for f in MOVEMENT_FIELDS if f.indexed and "map" in f.key_pattern]
        panels = []
        for i in range(n):
            card = ui.card(
                ui.card_header(f"Map {i}"),
                *[render_field(f, species_idx=i, config=cfg) for f in map_fields],
            )
            panels.append(card)
        return ui.div(*panels)

    @reactive.effect
    def sync_movement_inputs():
        sync_inputs(input, state, MOVEMENT_GLOBAL_KEYS)

    @reactive.effect
    def sync_species_movement_inputs():
        per_species = [f for f in MOVEMENT_FIELDS if f.indexed and "map" not in f.key_pattern]
        with reactive.isolate():
            _ns_raw = state.config.get().get("simulation.nspecies", "3")
            try:
                n_species = int(float(_ns_raw or "3"))
            except (ValueError, TypeError):
                n_species = 3
        for i in range(n_species):
            keys = [f.resolve_key(i) for f in per_species]
            sync_inputs(input, state, keys)

    @reactive.effect
    def sync_map_inputs():
        n = input.n_maps()
        map_fields = [f for f in MOVEMENT_FIELDS if f.indexed and "map" in f.key_pattern]
        for i in range(n):
            keys = [f.resolve_key(i) for f in map_fields]
            sync_inputs(input, state, keys)
