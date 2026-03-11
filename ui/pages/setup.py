"""Species & Simulation setup page."""

from pathlib import Path

from shiny import ui, reactive, render

from osmose.demo import list_demos, migrate_config
from osmose.logging import setup_logging
from osmose.schema.simulation import SIMULATION_FIELDS
from osmose.schema.species import SPECIES_FIELDS
from ui.components.param_form import render_category, render_species_params
from ui.state import sync_inputs

_log = setup_logging("osmose.setup")

# Keys for non-indexed simulation fields (synced automatically)
SETUP_GLOBAL_KEYS: list[str] = [f.key_pattern for f in SIMULATION_FIELDS if not f.advanced]


def get_species_keys(species_idx: int, show_advanced: bool = False) -> list[str]:
    """Return resolved OSMOSE keys for one species."""
    keys = []
    for f in SPECIES_FIELDS:
        if f.advanced and not show_advanced:
            continue
        keys.append(f.resolve_key(species_idx))
    return keys


def setup_ui():
    demo_choices = {
        "": "— Select example —",
        **{d: d.replace("_", " ").title() for d in list_demos()},
    }
    return ui.layout_columns(
        # Left column: Simulation settings
        ui.card(
            ui.card_header("Simulation Settings"),
            ui.div(
                ui.input_select(
                    "load_example",
                    "Load bundled example",
                    choices=demo_choices,
                    selected="",
                ),
            ),
            ui.hr(),
            render_category(
                [f for f in SIMULATION_FIELDS if not f.advanced],
            ),
        ),
        # Right column: Species configuration (dynamic)
        ui.card(
            ui.card_header("Species Configuration"),
            ui.input_numeric("n_species", "Number of focal species", value=3, min=1, max=20),
            ui.input_switch("show_advanced_species", "Show advanced parameters", value=False),
            ui.output_ui("species_panels"),
        ),
        col_widths=[4, 8],
    )


def setup_server(input, output, session, state):

    @render.ui
    def species_panels():
        n = input.n_species()
        show_adv = input.show_advanced_species()
        panels = []
        # Isolate config read to avoid reactive loop:
        # sync_species_inputs writes config → config.set() would
        # re-trigger this render → new inputs → re-trigger sync → ∞
        with reactive.isolate():
            cfg = state.config.get()
        for i in range(n):
            name = cfg.get(f"species.name.sp{i}", f"Species {i}")
            panels.append(
                render_species_params(
                    SPECIES_FIELDS,
                    species_idx=i,
                    species_name=name,
                    show_advanced=show_adv,
                    config=cfg,
                )
            )
        return ui.div(*panels)

    @reactive.effect
    def handle_load_example():
        """Load a bundled example config when dropdown selection changes."""
        from osmose.config.reader import OsmoseConfigReader
        from osmose.schema.base import ParamType

        example = input.load_example()
        if not example:
            return

        examples_dir = Path(__file__).parent.parent.parent / "data" / "examples"
        master = examples_dir / "osm_all-parameters.csv"
        if not master.exists():
            ui.notification_show(f"Example not found: {master}", type="error", duration=5)
            return

        # Guard: prevent sync effects from overwriting config while we push values
        state.loading.set(True)

        try:
            reader = OsmoseConfigReader()
            cfg = migrate_config(reader.read(master))
            state.config.set(cfg)
            state.config_dir.set(examples_dir)

            # Update species count
            n_species = int(cfg.get("simulation.nspecies", "3"))
            ui.update_numeric("n_species", value=n_species)

            # Push simulation-level values into existing UI inputs
            updated = 0
            for key, val in cfg.items():
                if key.startswith("osmose.configuration."):
                    continue
                field = state.registry.match_field(key)
                input_id = key.replace(".", "_")
                if field is None:
                    continue
                try:
                    if field.param_type in (ParamType.FLOAT, ParamType.INT):
                        numeric_val = (
                            float(val) if field.param_type == ParamType.FLOAT else int(val)
                        )
                        ui.update_numeric(input_id, value=numeric_val)
                    elif field.param_type == ParamType.BOOL:
                        ui.update_switch(input_id, value=val.lower() in ("true", "1", "yes"))
                    elif field.param_type == ParamType.ENUM:
                        ui.update_select(input_id, selected=val)
                    else:
                        ui.update_text(input_id, value=val)
                    updated += 1
                except Exception as exc:
                    _log.debug("Could not update input %s: %s", input_id, exc)

            ui.notification_show(
                f"Loaded '{example}' ({updated} parameters applied).",
                type="message",
                duration=3,
            )
            state.dirty.set(False)
        finally:
            state.loading.set(False)

    @reactive.effect
    def sync_simulation_inputs():
        """Auto-sync simulation fields to state.config."""
        sync_inputs(input, state, SETUP_GLOBAL_KEYS)

    @reactive.effect
    def sync_species_inputs():
        """Auto-sync species fields to state.config."""
        if state.loading.get():
            return
        n = input.n_species()
        show_adv = input.show_advanced_species()
        # Update nspecies in config
        state.update_config("simulation.nspecies", str(n))
        # Sync each species' fields
        for i in range(n):
            keys = get_species_keys(i, show_adv)
            sync_inputs(input, state, keys)
