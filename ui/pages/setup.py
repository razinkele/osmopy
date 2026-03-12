"""Species & Simulation setup page."""

from pathlib import Path

from shiny import ui, reactive, render

from osmose.demo import list_demos, migrate_config, osmose_demo
from osmose.logging import setup_logging
from osmose.schema.simulation import SIMULATION_FIELDS
from osmose.schema.species import SPECIES_FIELDS
from ui.components.param_form import render_category, render_species_table
from ui.state import sync_inputs

_log = setup_logging("osmose.setup")

# Keys for non-indexed simulation fields (synced automatically)
SETUP_GLOBAL_KEYS: list[str] = [f.key_pattern for f in SIMULATION_FIELDS if not f.advanced]


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
                ui.layout_columns(
                    ui.input_select(
                        "load_example",
                        "Example configuration",
                        choices=demo_choices,
                        selected="",
                    ),
                    ui.input_action_button(
                        "btn_load_example", "Load", class_="btn-primary mt-4"
                    ),
                    col_widths=[8, 4],
                ),
            ),
            ui.hr(),
            ui.output_ui("simulation_fields"),
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
    def simulation_fields():
        state.load_trigger.get()
        with reactive.isolate():
            cfg = state.config.get()
        return render_category(
            [f for f in SIMULATION_FIELDS if not f.advanced],
            config=cfg,
        )

    @render.ui
    def species_panels():
        state.load_trigger.get()
        n = input.n_species()
        show_adv = input.show_advanced_species()
        with reactive.isolate():
            cfg = state.config.get()
        names = [cfg.get(f"species.name.sp{i}", f"Species {i}") for i in range(n)]
        return render_species_table(
            SPECIES_FIELDS, n_species=n, species_names=names,
            show_advanced=show_adv, config=cfg,
        )

    @reactive.effect
    @reactive.event(input.btn_load_example)
    def handle_load_example():
        """Load a bundled example config when Load button is clicked."""
        import tempfile

        from osmose.config.reader import OsmoseConfigReader

        example = input.load_example()
        if not example:
            ui.notification_show("Select an example first.", type="warning", duration=3)
            return

        try:
            tmp = Path(tempfile.mkdtemp(prefix="osmose_demo_"))
            result = osmose_demo(example, tmp)
        except ValueError as exc:
            ui.notification_show(str(exc), type="error", duration=5)
            return

        master = result["config_file"]
        if not master.exists():
            ui.notification_show(f"Example not found: {master}", type="error", duration=5)
            return

        config_dir = master.parent

        state.loading.set(True)
        try:
            reader = OsmoseConfigReader()
            cfg = migrate_config(reader.read(master))
            state.config.set(cfg)
            state.config_dir.set(config_dir)
            state.config_name.set(example.replace("_", " ").title())

            # Extract species names
            n_species = int(cfg.get("simulation.nspecies", "0"))
            names = [cfg.get(f"species.name.sp{i}", f"Species {i}") for i in range(n_species)]
            state.species_names.set(names)

            ui.update_numeric("n_species", value=n_species)

            with reactive.isolate():
                state.load_trigger.set(state.load_trigger.get() + 1)

            ui.notification_show(
                f"Loaded '{example}' ({len(cfg)} parameters).",
                type="message",
                duration=3,
            )
            state.dirty.set(False)
            # Do NOT reset dropdown — keep selection visible
        finally:
            state.loading.set(False)

    @reactive.effect
    def sync_simulation_inputs():
        """Auto-sync simulation fields to state.config."""
        sync_inputs(input, state, SETUP_GLOBAL_KEYS)

    @reactive.effect
    def sync_species_inputs():
        """Auto-sync species table cells to state.config."""
        with reactive.isolate():
            if state.loading.get():
                return
        n = input.n_species()
        show_adv = input.show_advanced_species()
        state.update_config("simulation.nspecies", str(n))

        visible = [f for f in SPECIES_FIELDS if f.indexed and (show_adv or not f.advanced)]
        for i in range(n):
            for field in visible:
                config_key = field.resolve_key(i)
                base_key = field.key_pattern.replace(".sp{idx}", "").replace("{idx}", "").replace(".", "_")
                input_id = f"spt_{base_key}_{i}"
                try:
                    val = getattr(input, input_id)()
                except (AttributeError, TypeError):
                    continue
                if val is not None:
                    state.update_config(config_key, str(val))
