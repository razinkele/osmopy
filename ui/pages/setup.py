"""Species & Simulation setup page."""

from shiny import ui, reactive, render
from shiny.types import SilentException

from osmose.logging import setup_logging
from osmose.schema.simulation import SIMULATION_FIELDS
from osmose.schema.species import SPECIES_FIELDS
from osmose.config.validator import summarize_config_validation
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.components.fishbase_bootstrap import fishbase_bootstrap_ui, fishbase_bootstrap_server
from ui.components.param_form import (
    copy_species0_to_all,
    render_category,
    render_species_table,
)
from ui.state import AppState, sync_inputs

_log = setup_logging("osmose.setup")

# Keys for non-indexed simulation fields (synced automatically)
SETUP_GLOBAL_KEYS: list[str] = [f.key_pattern for f in SIMULATION_FIELDS if not f.advanced]


def setup_ui():
    return ui.div(
        # Full-width live validation summary, above the split layout. Kept OUTSIDE the
        # osm-split-layout flex row so it spans full width and does not break the
        # expand_tab→layout_columns adjacency the collapse CSS depends on.
        ui.output_ui("config_validation"),
        ui.div(
            expand_tab("Simulation Settings", "setup"),
            ui.layout_columns(
                # Left column: Simulation settings
                ui.card(
                    collapsible_card_header("Simulation Settings", "setup"),
                    ui.output_ui("simulation_fields"),
                ),
                # Right column: Species configuration (dynamic)
                ui.card(
                    ui.card_header("Species Configuration"),
                    ui.input_numeric(
                        "n_species", "Number of focal species", value=3, min=1, max=20
                    ),
                    ui.input_switch(
                        "show_advanced_species", "Show advanced parameters", value=False
                    ),
                    ui.output_ui("species_panels"),
                    fishbase_bootstrap_ui(),
                ),
                col_widths=[4, 8],
            ),
            class_="osm-split-layout",
            id="split_setup",
        ),
        class_="osm-setup-root d-flex flex-column h-100",
    )


def setup_server(input, output, session, state: AppState):
    @reactive.calc
    def _config_validation():
        """(loaded, errors, warnings). Recomputes on every edit AND on load,
        since every loader sets state.config directly. Never raises — a half-entered
        config must never crash the Setup tab."""
        config = state.config.get()
        config_dir = state.config_dir.get()
        loaded = bool(config) and "simulation.nspecies" in config
        if not loaded:
            return (False, [], [])
        try:
            errors, warnings = summarize_config_validation(config, state.registry, config_dir)
        except Exception as exc:  # noqa: BLE001 - panel must degrade, never crash the tab
            return (True, [], [f"validation unavailable: {exc}"])
        return (True, errors, warnings)

    @render.ui
    def config_validation():
        loaded, errors, warnings = _config_validation()

        def _issue_lines(items, css, cap=10):
            lines = [
                ui.div(f"• {m}", class_=css, style="overflow-wrap:anywhere") for m in items[:cap]
            ]
            if len(items) > cap:
                lines.append(ui.div(f"… and {len(items) - cap} more", class_="small text-muted"))
            return lines

        if not loaded:
            inner = ui.div("No configuration loaded", class_="small text-muted")
        elif not errors and not warnings:
            inner = ui.div("✓ Configuration valid", class_="small text-success")
        else:
            badge_css = "fw-bold text-danger" if errors else "fw-bold text-warning"
            inner = ui.div(
                ui.div(
                    f"{len(errors)} error(s) · {len(warnings)} warning(s)",
                    class_=badge_css,
                ),
                *_issue_lines(errors, "small text-danger"),
                *_issue_lines(warnings, "small text-warning"),
            )

        return ui.card(
            ui.div(inner, **{"aria-live": "polite", "aria-atomic": "true"}),
            class_="mb-2",
        )

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
            SPECIES_FIELDS,
            n_species=n,
            species_names=names,
            show_advanced=show_adv,
            config=cfg,
        )

    @reactive.effect
    def sync_simulation_inputs():
        """Auto-sync simulation fields to state.config."""
        sync_inputs(input, state, SETUP_GLOBAL_KEYS)

    @reactive.effect
    def sync_species_inputs():
        """Auto-sync species table cells to state.config.

        Collects all per-species parameter updates into a single dict, then
        calls state.config.set() once instead of once per parameter.  With
        20 species × 30+ fields this avoids 600+ individual reactive updates.
        """
        with reactive.isolate():
            if state.busy.get():
                return
        n = input.n_species()
        show_adv = input.show_advanced_species()

        # Collect all updates before touching reactive state.
        updates: dict[str, str] = {"simulation.nspecies": str(n)}

        visible = [f for f in SPECIES_FIELDS if f.indexed and (show_adv or not f.advanced)]
        for i in range(n):
            for field in visible:
                config_key = field.resolve_key(i)
                base_key = (
                    field.key_pattern.replace(".sp{idx}", "").replace("{idx}", "").replace(".", "_")
                )
                input_id = f"spt_{base_key}_{i}"
                try:
                    val = getattr(input, input_id)()
                except (AttributeError, SilentException):
                    continue
                if val is not None:
                    updates[config_key] = str(val)

        # Apply all updates in a single config.set() call.
        with reactive.isolate():
            cfg = dict(state.config.get())
        actual_changes = {k: v for k, v in updates.items() if cfg.get(k) != v}
        if actual_changes:
            cfg.update(actual_changes)
            state.config.set(cfg)
            state.dirty.set(True)

        # Update global species names list (read from freshly updated cfg).
        names = [cfg.get(f"species.name.sp{i}", f"Species {i}") for i in range(n)]
        state.species_names.set(names)

    @reactive.effect
    @reactive.event(input.copy_sp0_to_all)
    def handle_copy_sp0():
        n = input.n_species()
        if n < 2:
            return
        state.busy.set("Copying species 0 parameters to all species…")
        try:
            with reactive.isolate():
                cfg = dict(state.config.get())
            show_adv = input.show_advanced_species()
            count = copy_species0_to_all(
                SPECIES_FIELDS, n, cfg, input, session, show_advanced=show_adv
            )
            state.config.set(cfg)
            state.dirty.set(True)
            ui.notification_show(
                f"Copied {count} parameters from species 0 to {n - 1} species.",
                type="message",
                duration=3,
            )
        finally:
            state.busy.set(None)

    fishbase_bootstrap_server(input, output, session, state)
