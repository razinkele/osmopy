"""Scenario management page."""

import zipfile
from pathlib import Path

from shiny import reactive, render, ui

from osmose.logging import setup_logging
from osmose.scenarios import Scenario, ScenarioManager
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.styles import STYLE_DIFF_ROW, STYLE_EMPTY

_log = setup_logging("osmose.scenarios_ui")


def scenarios_ui():
    return ui.div(
        expand_tab("Save Scenario", "scenarios"),
        ui.layout_columns(
            # Left: Save & manage
            ui.card(
                collapsible_card_header("Save Scenario", "scenarios"),
                ui.input_text("scenario_name", "Scenario name"),
                ui.input_text("scenario_desc", "Description"),
                ui.input_text("scenario_tags", "Tags (comma-separated)"),
                ui.input_action_button(
                    "btn_save_scenario", "Save Current Config", class_="btn-success w-100"
                ),
            ),
            # Middle: Scenario list
            ui.card(
                ui.card_header("Saved Scenarios"),
                ui.output_ui("scenario_list"),
                ui.hr(),
                ui.layout_columns(
                    ui.input_action_button("btn_load_scenario", "Load", class_="btn-primary w-100"),
                    ui.input_action_button("btn_fork_scenario", "Fork", class_="btn-info w-100"),
                    ui.input_action_button(
                        "btn_delete_scenario", "Delete", class_="btn-danger w-100"
                    ),
                    col_widths=[4, 4, 4],
                ),
            ),
            # Right: Compare
            ui.card(
                ui.card_header("Compare Scenarios"),
                ui.input_select("compare_a", "Scenario A", choices={}),
                ui.input_select("compare_b", "Scenario B", choices={}),
                ui.input_action_button("btn_compare", "Compare", class_="btn-warning w-100"),
                ui.hr(),
                ui.output_ui("compare_results"),
            ),
            col_widths=[3, 5, 4],
        ),
        ui.card(
            ui.card_header("Bulk Operations"),
            ui.download_button(
                "export_all_scenarios",
                "Export All (ZIP)",
                class_="btn-primary w-100",
            ),
            ui.input_file(
                "import_scenarios_zip",
                "Import Scenarios (ZIP)",
                accept=[".zip"],
            ),
        ),
        class_="osm-split-layout",
        id="split_scenarios",
    )


def scenarios_server(input, output, session, state):
    mgr = ScenarioManager(state.scenarios_dir)
    refresh_trigger = reactive.value(0)

    def _bump():
        """Increment the refresh trigger to force re-render of scenario list."""
        with reactive.isolate():
            current = refresh_trigger.get()
        refresh_trigger.set(current + 1)

    def _scenario_names() -> list[str]:
        """Return a sorted list of scenario names."""
        return [s["name"] for s in mgr.list_scenarios()]

    # --- Scenario list (radio buttons) ---

    @render.ui
    def scenario_list():
        refresh_trigger.get()  # depend on trigger
        scenarios = mgr.list_scenarios()
        if not scenarios:
            return ui.div(
                "No scenarios saved yet.",
                style=STYLE_EMPTY,
            )
        choices = {s["name"]: f"{s['name']}  ({s.get('description', '')})" for s in scenarios}
        return ui.input_radio_buttons("selected_scenario", None, choices=choices)

    # --- Save ---

    pending_save: reactive.Value[Scenario | None] = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.btn_save_scenario)
    def handle_save():
        name = input.scenario_name().strip()
        if not name:
            ui.notification_show("Enter a scenario name.", type="warning", duration=5)
            return
        tags_raw = input.scenario_tags().strip()
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []
        scenario = Scenario(
            name=name,
            description=input.scenario_desc().strip(),
            config=dict(state.config.get()),
            tags=tags,
            key_case_map=dict(state.key_case_map.get()),
        )
        # Check if scenario already exists
        existing = {s["name"] for s in mgr.list_scenarios()}
        if name in existing:
            pending_save.set(scenario)
            ui.modal_show(
                ui.modal(
                    ui.p(f"Scenario '{name}' already exists. Overwrite?"),
                    title="Confirm Overwrite",
                    easy_close=True,
                    footer=ui.div(
                        ui.input_action_button(
                            "btn_confirm_overwrite", "Overwrite", class_="btn-warning"
                        ),
                        ui.tags.button(
                            "Cancel",
                            class_="btn btn-secondary",
                            **{"data-bs-dismiss": "modal"},
                        ),
                    ),
                )
            )
            return
        _do_save(scenario)

    @reactive.effect
    @reactive.event(input.btn_confirm_overwrite)
    def handle_confirm_overwrite():
        scenario = pending_save.get()
        if scenario is None:
            return
        ui.modal_remove()
        _do_save(scenario)
        pending_save.set(None)

    def _do_save(scenario: Scenario):
        try:
            mgr.save(scenario)
        except (OSError, ValueError) as exc:
            _log.error("Failed to save scenario: %s", exc, exc_info=True)
            ui.notification_show(
                "Failed to save scenario. Check server logs for details.",
                type="error",
                duration=15,
            )
            return
        state.dirty.set(False)
        _bump()
        ui.notification_show(f"Scenario '{scenario.name}' saved.", type="message", duration=3)

    # --- Load ---

    @reactive.effect
    @reactive.event(input.btn_load_scenario)
    def handle_load():
        selected = input.selected_scenario()
        if not selected:
            return
        loaded = mgr.load(selected)
        state.loading.set(True)
        try:
            state.config.set(loaded.config)
            state.config_name.set(selected)
            state.key_case_map.set(dict(loaded.key_case_map))
            state.dirty.set(False)

            try:
                n_species = int(float(loaded.config.get("simulation.nspecies", "3") or "3"))
            except (ValueError, TypeError):
                n_species = 3
            names = [
                loaded.config.get(f"species.name.sp{i}", f"Species {i}") for i in range(n_species)
            ]
            state.species_names.set(names)
            ui.update_numeric("n_species", value=n_species)

            with reactive.isolate():
                state.load_trigger.set(state.load_trigger.get() + 1)

            ui.notification_show(
                f"Loaded scenario '{selected}' ({len(loaded.config)} parameters).",
                type="message",
                duration=3,
            )
        finally:
            state.loading.set(False)

    # --- Fork ---

    @reactive.effect
    @reactive.event(input.btn_fork_scenario)
    def handle_fork():
        selected = input.selected_scenario()
        if not selected:
            return
        new_name = f"{selected}_fork"
        try:
            mgr.fork(selected, new_name)
        except (OSError, ValueError, FileNotFoundError) as exc:
            _log.error("Failed to fork scenario: %s", exc, exc_info=True)
            ui.notification_show(
                "Failed to fork scenario. Check server logs for details.",
                type="error",
                duration=15,
            )
            return
        _bump()
        ui.notification_show(f"Forked as '{new_name}'.", type="message", duration=3)

    # --- Delete ---

    pending_delete: reactive.Value[str | None] = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.btn_delete_scenario)
    def handle_delete():
        selected = input.selected_scenario()
        if not selected:
            return
        pending_delete.set(selected)
        ui.modal_show(
            ui.modal(
                ui.p(f"Are you sure you want to delete scenario '{selected}'?"),
                ui.p("This action cannot be undone.", style="color: #e74c3c; font-size: 0.9em;"),
                title="Confirm Delete",
                easy_close=True,
                footer=ui.div(
                    ui.input_action_button(
                        "btn_confirm_delete", "Delete", class_="btn-danger"
                    ),
                    ui.tags.button(
                        "Cancel",
                        class_="btn btn-secondary",
                        **{"data-bs-dismiss": "modal"},
                    ),
                ),
            )
        )

    @reactive.effect
    @reactive.event(input.btn_confirm_delete)
    def handle_confirm_delete():
        name = pending_delete.get()
        if not name:
            return
        ui.modal_remove()
        try:
            mgr.delete(name)
        except (OSError, FileNotFoundError) as exc:
            _log.error("Failed to delete scenario: %s", exc, exc_info=True)
            ui.notification_show(
                "Failed to delete scenario. Check server logs for details.",
                type="error",
                duration=15,
            )
            return
        pending_delete.set(None)
        _bump()
        ui.notification_show("Scenario deleted.", type="message", duration=3)

    # --- Update compare dropdowns when scenario list changes ---

    @reactive.effect
    def update_compare_choices():
        refresh_trigger.get()  # depend on trigger
        names = _scenario_names()
        choices = {n: n for n in names}
        ui.update_select("compare_a", choices=choices, session=session)
        ui.update_select("compare_b", choices=choices, session=session)

    # --- Compare ---

    compare_diffs = reactive.value([])

    @reactive.effect
    @reactive.event(input.btn_compare)
    def handle_compare():
        a = input.compare_a()
        b = input.compare_b()
        if not a or not b or a == b:
            compare_diffs.set([])
            return
        diffs = mgr.compare(a, b)
        compare_diffs.set(diffs)

    @render.ui
    def compare_results():
        diffs = compare_diffs.get()
        if not diffs:
            return ui.div(
                "Select two scenarios and click Compare.",
                style=STYLE_EMPTY,
            )
        rows = []
        for d in diffs:
            rows.append(
                ui.tags.tr(
                    ui.tags.td(d.key, style="font-weight: bold;"),
                    ui.tags.td(str(d.value_a) if d.value_a is not None else "(missing)"),
                    ui.tags.td(str(d.value_b) if d.value_b is not None else "(missing)"),
                    style=STYLE_DIFF_ROW,
                )
            )
        return ui.tags.table(
            ui.tags.thead(
                ui.tags.tr(
                    ui.tags.th("Parameter"),
                    ui.tags.th("Value A"),
                    ui.tags.th("Value B"),
                )
            ),
            ui.tags.tbody(*rows),
            class_="table table-sm table-bordered",
        )

    # --- Bulk Export ---

    @render.download(filename="osmose_scenarios.zip")
    def export_all_scenarios():
        import atexit
        import shutil
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp(prefix="osmose_export_"))
        zip_path = tmp_dir / "osmose_scenarios.zip"
        mgr.export_all(zip_path)
        # Schedule cleanup after Shiny finishes serving the download
        atexit.register(shutil.rmtree, str(tmp_dir), True)
        return str(zip_path)

    # --- Bulk Import ---

    @reactive.effect
    @reactive.event(input.import_scenarios_zip)
    def handle_import_scenarios():
        file_info = input.import_scenarios_zip()
        if not file_info:
            return
        zip_path = Path(file_info[0]["datapath"])
        try:
            mgr.import_all(zip_path)
        except (OSError, zipfile.BadZipFile, ValueError, KeyError) as exc:
            _log.error("Failed to import scenarios: %s", exc, exc_info=True)
            ui.notification_show(
                "Failed to import scenarios. Check server logs for details.",
                type="error",
                duration=15,
            )
            return
        _bump()
        ui.notification_show("Scenarios imported.", type="message", duration=3)

    # --- Keyboard shortcut: Ctrl+S / Cmd+S → save scenario ---

    @reactive.effect
    @reactive.event(input.shortcut_save)
    def handle_shortcut_save():
        name = input.scenario_name().strip()
        if not name:
            ui.notification_show(
                "Enter a scenario name on the Scenarios tab to save.",
                type="warning",
                duration=5,
            )
            return
        tags_raw = input.scenario_tags().strip()
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []
        scenario = Scenario(
            name=name,
            description=input.scenario_desc().strip(),
            config=dict(state.config.get()),
            tags=tags,
            key_case_map=dict(state.key_case_map.get()),
        )
        _do_save(scenario)

