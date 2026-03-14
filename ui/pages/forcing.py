"""Environmental forcing / LTL configuration page."""

from shiny import ui, reactive, render

from osmose.schema.bioenergetics import BIOENERGETICS_FIELDS
from osmose.schema.ltl import LTL_FIELDS
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.components.param_form import render_field, render_species_table
from ui.state import sync_inputs

FORCING_GLOBAL_KEYS: list[str] = [f.key_pattern for f in LTL_FIELDS if not f.indexed]
_TEMP_KEYS: list[str] = [
    f.key_pattern
    for f in BIOENERGETICS_FIELDS
    if f.key_pattern.startswith("temperature.") and not f.indexed
]


def forcing_ui():
    return ui.div(
        expand_tab("Lower Trophic Level (Plankton)", "forcing"),
        ui.layout_columns(
            ui.card(
                collapsible_card_header("Lower Trophic Level (Plankton)", "forcing"),
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
        ),
        class_="osm-split-layout",
        id="split_forcing",
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
        try:
            n_focal = int(float(cfg.get("simulation.nspecies", "0") or "0"))
        except (ValueError, TypeError):
            n_focal = 0
        indexed_fields = [f for f in LTL_FIELDS if f.indexed]
        names = [cfg.get(f"species.name.sp{n_focal + i}", f"Resource {i}") for i in range(n)]
        return render_species_table(
            indexed_fields, n_species=n, species_names=names,
            start_idx=n_focal, config=cfg,
        )

    @reactive.effect
    def sync_forcing_inputs():
        sync_inputs(input, state, FORCING_GLOBAL_KEYS + _TEMP_KEYS)

    @reactive.effect
    def sync_resource_inputs():
        n = input.n_resources()
        with reactive.isolate():
            if state.loading.get():
                return
            cfg = state.config.get()
        try:
            n_focal = int(float(cfg.get("simulation.nspecies", "0") or "0"))
        except (ValueError, TypeError):
            n_focal = 0
        indexed_fields = [f for f in LTL_FIELDS if f.indexed]
        for i in range(n):
            sp_idx = n_focal + i
            for field in indexed_fields:
                config_key = field.resolve_key(sp_idx)
                base_key = field.key_pattern.replace(".sp{idx}", "").replace("{idx}", "").replace(".", "_")
                input_id = f"spt_{base_key}_{sp_idx}"
                try:
                    val = getattr(input, input_id)()
                except (AttributeError, TypeError):
                    continue
                if val is not None:
                    state.update_config(config_key, str(val))
