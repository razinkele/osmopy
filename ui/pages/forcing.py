"""Environmental forcing / LTL configuration page."""

from collections.abc import Callable
from typing import Any

from shiny import reactive, render, ui
from shiny.types import SilentException

from osmose.config.multivalue import is_multi_value
from osmose.schema.bioenergetics import BIOENERGETICS_FIELDS
from osmose.schema.ltl import LTL_FIELDS
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.components.param_form import copy_species0_to_all, render_field, render_species_table
from ui.pages._helpers import parse_nspecies
from ui.state import REGISTRY, AppState, same_config_value, sync_inputs, untouched_unset

FORCING_GLOBAL_KEYS: list[str] = [f.key_pattern for f in LTL_FIELDS if not f.indexed]
_TEMP_KEYS: list[str] = [
    f.key_pattern
    for f in BIOENERGETICS_FIELDS
    if f.key_pattern.startswith("temperature.") and not f.indexed
]


def collect_resource_table_updates(
    n_resources: int,
    n_focal: int,
    config: dict[str, str],
    read_input: Callable[[str], Any],
) -> dict[str, str]:
    """Config updates from the resource table's inputs (what ``sync_resource_inputs`` applies).

    A per-stage array cell has no input (``render_species_table`` shows it read-only) and is
    skipped, decided from the config: Shiny keeps the last value of an input removed from the
    page, so a stale input would otherwise write a scalar over the array.
    """
    updates: dict[str, str] = {}
    for i in range(n_resources):
        sp_idx = n_focal + i
        for field in (f for f in LTL_FIELDS if f.indexed):
            config_key = field.resolve_key(sp_idx)
            if is_multi_value(config.get(config_key), field):
                continue
            base_key = (
                field.key_pattern.replace(".sp{idx}", "").replace("{idx}", "").replace(".", "_")
            )
            try:
                val = read_input(f"spt_{base_key}_{sp_idx}")
            except (AttributeError, SilentException):
                continue
            if val is not None and not untouched_unset(field, config_key, config, str(val)):
                updates[config_key] = str(val)
    return updates


def forcing_ui():
    return ui.div(
        expand_tab("Lower Trophic Level (Plankton)", "forcing"),
        ui.layout_columns(
            ui.card(
                collapsible_card_header("Lower Trophic Level (Plankton)", "forcing"),
                ui.output_ui("forcing_global_fields"),
                ui.hr(),
                ui.input_numeric(
                    "n_resources", "Number of resource groups", value=3, min=0, max=20
                ),
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


def forcing_server(input, output, session, state: AppState):
    global_ltl = [f for f in LTL_FIELDS if not f.indexed]
    temp_fields = [f for f in BIOENERGETICS_FIELDS if f.key_pattern.startswith("temperature.")]

    @render.ui
    def forcing_global_fields():
        state.load_trigger.get()
        with reactive.isolate():
            cfg = state.config.get()
        gen_input = state.mark_rendered("forcing.global")
        content = ui.div(
            ui.h5("Global LTL Settings"),
            *[render_field(f, config=cfg) for f in global_ltl],
        )
        return ui.TagList(gen_input, content)  # see mark_rendered

    @render.ui
    def forcing_temp_fields():
        state.load_trigger.get()
        with reactive.isolate():
            cfg = state.config.get()
        gen_input = state.mark_rendered("forcing.temp")
        content = ui.div(
            ui.h5("Temperature"),
            *[render_field(f, config=cfg) for f in temp_fields if not f.advanced],
        )
        return ui.TagList(gen_input, content)  # see mark_rendered

    @render.ui
    def resource_panels():
        state.load_trigger.get()
        n = input.n_resources()
        with reactive.isolate():
            cfg = state.config.get()
        n_focal = parse_nspecies(cfg)
        indexed_fields = [f for f in LTL_FIELDS if f.indexed]
        names = [cfg.get(f"species.name.sp{n_focal + i}", f"Resource {i}") for i in range(n)]
        gen_input = state.mark_rendered("forcing.resources")
        content = render_species_table(
            indexed_fields,
            n_species=n,
            species_names=names,
            start_idx=n_focal,
            config=cfg,
            copy_event="copy_rsc0_to_all",
        )
        return ui.TagList(gen_input, content)  # see mark_rendered

    @reactive.effect
    def sync_forcing_inputs():
        sync_inputs(input, state, FORCING_GLOBAL_KEYS, rendered="forcing.global")

    @reactive.effect
    def sync_temperature_inputs():
        # Its own sync, gated by its own render: the two sit in different cards, and a collapsed
        # LTL panel suspends forcing_global_fields, which would block temperature edits.
        sync_inputs(input, state, _TEMP_KEYS, rendered="forcing.temp")

    @reactive.effect
    def sync_resource_inputs():
        """Auto-sync resource table cells to state.config.

        Collects all per-resource parameter updates into a single dict, then
        calls state.config.set() once instead of once per parameter.
        """
        n = input.n_resources()
        with reactive.isolate():
            cfg = dict(state.config.get())
        n_focal = parse_nspecies(cfg)

        # Collect all updates before touching reactive state.
        updates = collect_resource_table_updates(
            n, n_focal, cfg, lambda input_id: getattr(input, input_id)()
        )

        # Gate only the WRITE: the cell reads above keep this effect subscribed to them.
        # The render generation the browser reports is read here too, before the gates (#283).
        current = state.rendered_current("forcing.resources", input=input)
        with reactive.isolate():
            if state.busy.get():
                return
        if not current:
            return  # the browser's cells are not from the current render (yet)

        # Apply all updates in a single config.set() call.
        actual_changes = {
            k: v
            for k, v in updates.items()
            if not same_config_value(cfg.get(k), v, REGISTRY.match_field(k))
        }
        if actual_changes:
            cfg.update(actual_changes)
            with reactive.isolate():
                state.config.set(cfg)
                state.dirty.set(True)

    @reactive.effect
    @reactive.event(input.copy_rsc0_to_all)
    def handle_copy_sp0_resources():
        n = input.n_resources()
        if n < 2 or not state.rendered_current("forcing.resources", input=input):
            return  # never copy from inputs rendered for an earlier config load
        state.busy.set("Copying resource 0 parameters to all resources…")
        try:
            with reactive.isolate():
                cfg = dict(state.config.get())
            n_focal = parse_nspecies(cfg)
            indexed_fields = [f for f in LTL_FIELDS if f.indexed]
            count = copy_species0_to_all(indexed_fields, n, cfg, input, session, start_idx=n_focal)
            state.config.set(cfg)
            state.dirty.set(True)
            ui.notification_show(
                f"Copied {count} parameters from resource 0 to {n - 1} resources.",
                type="message",
                duration=3,
            )
        finally:
            state.busy.set(None)
