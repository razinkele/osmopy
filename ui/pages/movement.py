"""Movement / spatial distribution page."""

from shiny import ui, reactive, render

from osmose.schema.movement import MOVEMENT_FIELDS
from ui.components.param_form import render_field
from ui.state import sync_inputs

MOVEMENT_GLOBAL_KEYS: list[str] = [f.key_pattern for f in MOVEMENT_FIELDS if not f.indexed]


def movement_ui():
    global_fields = [f for f in MOVEMENT_FIELDS if not f.indexed]

    return ui.layout_columns(
        ui.card(
            ui.card_header("Movement Settings"),
            *[render_field(f) for f in global_fields if not f.advanced],
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
    )


def movement_server(input, output, session, state):
    @render.ui
    def species_movement_panels():
        per_species = [f for f in MOVEMENT_FIELDS if f.indexed and "map" not in f.key_pattern]
        with reactive.isolate():
            n_species = int(state.config.get().get("simulation.nspecies", "3"))
        panels = []
        for i in range(n_species):
            panels.extend([render_field(f, species_idx=i) for f in per_species])
        return ui.div(*panels)

    @render.ui
    def map_panels():
        n = input.n_maps()
        map_fields = [f for f in MOVEMENT_FIELDS if f.indexed and "map" in f.key_pattern]
        panels = []
        for i in range(n):
            card = ui.card(
                ui.card_header(f"Map {i}"),
                *[render_field(f, species_idx=i) for f in map_fields],
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
            n_species = int(state.config.get().get("simulation.nspecies", "3"))
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
