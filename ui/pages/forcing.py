"""Environmental forcing / LTL configuration page."""

from shiny import ui, reactive, render

from osmose.schema.bioenergetics import BIOENERGETICS_FIELDS
from osmose.schema.ltl import LTL_FIELDS
from ui.components.param_form import render_field
from ui.state import sync_inputs

FORCING_GLOBAL_KEYS: list[str] = [f.key_pattern for f in LTL_FIELDS if not f.indexed]
_TEMP_KEYS: list[str] = [
    f.key_pattern
    for f in BIOENERGETICS_FIELDS
    if f.key_pattern.startswith("temperature.") and not f.indexed
]


def forcing_ui():
    return ui.layout_columns(
        ui.card(
            ui.card_header("Lower Trophic Level (Plankton)"),
            ui.output_ui("forcing_global_fields"),
            ui.hr(),
            ui.input_numeric("n_resources", "Number of resource groups", value=3, min=0, max=20),
            ui.output_ui("resource_panels"),
        ),
        ui.card(
            ui.card_header("Environmental Forcing"),
            ui.output_ui("forcing_temp_fields"),
            ui.hr(),
            ui.p(
                "Upload NetCDF forcing data for spatially-varying temperature, "
                "oxygen, or other environmental variables."
            ),
        ),
        col_widths=[7, 5],
    )


def forcing_server(input, output, session, state):
    global_ltl = [f for f in LTL_FIELDS if not f.indexed]
    temp_fields = [f for f in BIOENERGETICS_FIELDS if f.key_pattern.startswith("temperature.")]

    @render.ui
    def forcing_global_fields():
        state.load_trigger.get()
        with reactive.isolate():
            cfg = state.config.get()
        return ui.div(
            ui.h5("Global LTL Settings"),
            *[render_field(f, config=cfg) for f in global_ltl],
        )

    @render.ui
    def forcing_temp_fields():
        state.load_trigger.get()
        with reactive.isolate():
            cfg = state.config.get()
        return ui.div(
            ui.h5("Temperature"),
            *[render_field(f, config=cfg) for f in temp_fields if not f.advanced],
        )

    @render.ui
    def resource_panels():
        state.load_trigger.get()
        n = input.n_resources()
        with reactive.isolate():
            cfg = state.config.get()
        panels = []
        for i in range(n):
            resource_fields = [f for f in LTL_FIELDS if f.indexed]
            card = ui.card(
                ui.card_header(f"Resource Group {i}"),
                *[render_field(f, species_idx=i, config=cfg) for f in resource_fields],
            )
            panels.append(card)
        return ui.div(*panels)

    @reactive.effect
    def sync_forcing_inputs():
        sync_inputs(input, state, FORCING_GLOBAL_KEYS + _TEMP_KEYS)

    @reactive.effect
    def sync_resource_inputs():
        n = input.n_resources()
        indexed_fields = [f for f in LTL_FIELDS if f.indexed]
        for i in range(n):
            keys = [f.resolve_key(i) for f in indexed_fields]
            sync_inputs(input, state, keys)
