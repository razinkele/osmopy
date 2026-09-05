"""Advanced raw config editor page."""

import atexit
import shutil
import tempfile
from pathlib import Path

from shiny import reactive, render, ui

from osmose.config.reader import OsmoseConfigReader
from osmose.config.writer import OsmoseConfigWriter
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.state import REGISTRY, AppState
from ui.styles import (
    COLOR_DANGER,
    COLOR_MUTED,
    COLOR_SUCCESS,
    STYLE_EMPTY,
    STYLE_MONO_KEY,
    STYLE_SCROLL_TABLE,
)


def compute_import_diff(
    current: dict[str, str], incoming: dict[str, str]
) -> list[dict[str, str | None]]:
    """Compute diff between current config and incoming import.

    Returns list of dicts with keys: key, old, new (only changed/new keys).
    """
    diff = []
    for key, new_val in sorted(incoming.items()):
        old_val = current.get(key)
        if old_val != new_val:
            diff.append({"key": key, "old": old_val, "new": new_val})
    return diff


def multi_value_keys(cfg: dict[str, str]) -> list[str]:
    """Config keys whose value is a ``;``-separated array (e.g. per-stage parameters).

    The schema-driven form tabs render these read-only (see ``render_field``);
    this page is the one place they can be edited, so it needs the same test.
    """
    return sorted(k for k, v in cfg.items() if isinstance(v, str) and ";" in v)


def advanced_ui():
    categories = ["all"] + REGISTRY.categories()

    return ui.div(
        expand_tab("Config I/O", "advanced"),
        ui.layout_columns(
            # Controls
            ui.card(
                collapsible_card_header("Config I/O", "advanced"),
                ui.input_file(
                    "import_config", "Import OSMOSE config", accept=[".csv", ".properties"]
                ),
                ui.output_ui("import_preview"),
                ui.download_button(
                    "export_config", "Export Current Config", class_="btn-primary w-100"
                ),
            ),
            ui.card(
                ui.card_header("Filters"),
                ui.input_select(
                    "adv_category",
                    "Category",
                    choices={c: c.title() for c in categories},
                ),
                ui.input_text("adv_search", "Search parameters", placeholder="Type to filter..."),
                ui.p(
                    f"Total parameters in registry: {len(REGISTRY.all_fields())}",
                    style=COLOR_MUTED + " font-size: 12px;",
                ),
            ),
            col_widths=[4, 8],
        ),
        ui.card(
            ui.card_header("All Parameters"),
            ui.output_ui("multi_value_editor"),
            ui.output_ui("param_table"),
        ),
        class_="osm-split-layout",
        id="split_advanced",
    )


def advanced_server(input, output, session, state: AppState):
    import_pending = reactive.value({})
    pending_case_map = reactive.value({})

    @reactive.effect
    @reactive.event(input.import_config)
    def handle_import():
        file_info = input.import_config()
        if not file_info:
            return
        filepath = Path(file_info[0]["datapath"])
        reader = OsmoseConfigReader()
        try:
            new_cfg = reader.read_file(filepath)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            ui.notification_show(f"Failed to parse config file: {exc}", type="error", duration=15)
            return
        loaded = new_cfg
        # Stage for preview instead of merging directly
        import_pending.set(loaded)
        pending_case_map.set(dict(reader.key_case_map))

    @render.ui
    def import_preview():
        pending = import_pending.get()
        if not pending:
            return ui.div()

        with reactive.isolate():
            current_cfg = state.config.get()
        diff = compute_import_diff(current_cfg, pending)
        if not diff:
            return ui.div(
                ui.p("No changes detected in imported file.", style=COLOR_MUTED),
            )

        rows = []
        for d in diff:
            old_display = d["old"] if d["old"] is not None else "(new)"
            rows.append(
                ui.tags.tr(
                    ui.tags.td(d["key"], style=STYLE_MONO_KEY),
                    ui.tags.td(
                        str(old_display),
                        style=COLOR_DANGER if d["old"] is not None else COLOR_MUTED,
                    ),
                    ui.tags.td(str(d["new"]), style=COLOR_SUCCESS),
                )
            )

        return ui.div(
            ui.h6(f"Import Preview: {len(diff)} change(s) detected"),
            ui.tags.div(
                ui.tags.table(
                    ui.tags.thead(
                        ui.tags.tr(
                            ui.tags.th("Key"),
                            ui.tags.th("Current"),
                            ui.tags.th("New Value"),
                        )
                    ),
                    ui.tags.tbody(*rows),
                    class_="table table-striped table-sm",
                ),
                style=STYLE_SCROLL_TABLE,
            ),
            ui.input_action_button(
                "confirm_import", "Confirm Import", class_="btn-success w-100 mt-2"
            ),
        )

    @reactive.effect
    @reactive.event(import_pending)
    def _clear_empty_import():
        """Clear import_pending when diff is empty (moved out of render)."""
        pending = import_pending.get()
        if not pending:
            return
        with reactive.isolate():
            current_cfg = state.config.get()
        diff = compute_import_diff(current_cfg, pending)
        if not diff:
            import_pending.set({})

    @reactive.effect
    @reactive.event(input.confirm_import)
    def confirm_import():
        pending = import_pending.get()
        if not pending:
            return
        state.busy.set("Importing advanced parameters…")
        try:
            with reactive.isolate():
                cfg = dict(state.config.get())
            cfg.update(pending)
            state.config.set(cfg)
            state.dirty.set(True)  # Mark as dirty so unsaved-changes warning fires
            import_pending.set({})

            with reactive.isolate():
                cm = dict(state.key_case_map.get())
            cm.update(pending_case_map.get())
            state.key_case_map.set(cm)
            pending_case_map.set({})

            with reactive.isolate():
                state.load_trigger.set(state.load_trigger.get() + 1)

            ui.notification_show(
                f"Imported {len(pending)} parameter(s).",
                type="message",
                duration=3,
            )
        finally:
            state.busy.set(None)

    # -- H12: multi-value (";"-array) editor ---------------------------------
    # The form tabs show these entries read-only and point users here. The
    # banner lists every such key and offers a single text editor that writes
    # the array back verbatim, so per-stage parameters never have to be edited
    # by re-importing a CSV.
    _mv_selected = reactive.value("")

    @render.ui
    def multi_value_editor():
        # Same re-render contract as param_table: explicit loads/imports only.
        state.load_trigger.get()
        with reactive.isolate():
            cfg = state.config.get()
            remembered = _mv_selected.get()
        keys = multi_value_keys(cfg)
        if not keys:
            return ui.div()
        selected = remembered if remembered in keys else keys[0]
        return ui.div(
            ui.tags.strong(f"{len(keys)} multi-value parameter(s)"),
            ui.p(
                "';'-separated arrays (e.g. per-stage values) are shown read-only in the "
                "form tabs. Edit them here — the full array is written back verbatim.",
                class_="mb-2 small",
            ),
            ui.layout_columns(
                ui.input_select(
                    "adv_mv_key", "Parameter", choices={k: k for k in keys}, selected=selected
                ),
                ui.input_text("adv_mv_value", "Value (';'-separated)", value=cfg[selected]),
                ui.div(
                    ui.input_action_button("adv_mv_apply", "Apply", class_="btn-primary w-100"),
                    style="display: flex; align-items: flex-end; height: 100%;",
                ),
                col_widths=[5, 5, 2],
            ),
            class_="alert alert-info py-2 mb-2",
        )

    @reactive.effect
    @reactive.event(input.adv_mv_key)
    def _sync_multi_value_field():
        key = input.adv_mv_key()
        if not key:
            return
        _mv_selected.set(key)
        ui.update_text("adv_mv_value", value=state.get_config_value(key))

    @reactive.effect
    @reactive.event(input.adv_mv_apply)
    def _apply_multi_value():
        key = input.adv_mv_key()
        raw = (input.adv_mv_value() or "").strip()
        if not key:
            return
        if ";" not in raw:
            ui.notification_show(
                "Value must remain a ';'-separated array — use the form tabs for scalars.",
                type="warning",
                duration=6,
            )
            return
        state.update_config(key, raw)
        # Refresh the form tabs and the table below so they show the new array.
        with reactive.isolate():
            state.load_trigger.set(state.load_trigger.get() + 1)
        ui.notification_show(f"Updated {key}.", type="message", duration=3)

    @render.download(filename="osm_all-parameters.csv")
    def export_config():
        work_dir = Path(tempfile.mkdtemp(prefix="osmose_export_"))
        atexit.register(shutil.rmtree, str(work_dir), True)
        writer = OsmoseConfigWriter()
        writer.write(state.config.get(), work_dir, key_case_map=state.key_case_map.get())
        master = work_dir / "osm_all-parameters.csv"
        return str(master)

    @render.ui
    def param_table():
        # Use load_trigger as the reactive dependency so the table only
        # re-renders on explicit config loads, not on every keystroke.
        # Filter inputs (category, search) still re-render normally because
        # they are read directly above.  Config is read in isolate() to avoid
        # taking a full dependency on every config change.
        state.load_trigger.get()
        category = input.adv_category()
        search = input.adv_search().lower() if input.adv_search() else ""

        if category == "all":
            fields = REGISTRY.all_fields()
        else:
            fields = REGISTRY.fields_by_category(category)

        if search:
            fields = [
                f
                for f in fields
                if search in f.key_pattern.lower() or search in f.description.lower()
            ]

        if not fields:
            return ui.div("No parameters match your filter.", style=STYLE_EMPTY)

        # Show current config values — read with isolate() to avoid re-rendering
        # on every incremental config change (e.g. during species sync).
        with reactive.isolate():
            cfg = state.config.get()

        rows = []
        for f in fields[:100]:
            # For indexed fields key_pattern contains literal "{idx}" which
            # never appears in the config dict.  Show the sp0 value instead.
            if f.indexed:
                current_val = cfg.get(f.resolve_key(0), "-")
            else:
                current_val = cfg.get(f.key_pattern, "-")
            rows.append(
                ui.tags.tr(
                    ui.tags.td(f.key_pattern, style=STYLE_MONO_KEY),
                    ui.tags.td(f.param_type.value),
                    ui.tags.td(str(current_val)),
                    ui.tags.td(f.category),
                    ui.tags.td(
                        f.description[:60] + "..." if len(f.description) > 60 else f.description
                    ),
                )
            )

        return ui.tags.div(
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("Key"),
                        ui.tags.th("Type"),
                        ui.tags.th("Current Value"),
                        ui.tags.th("Category"),
                        ui.tags.th("Description"),
                    )
                ),
                ui.tags.tbody(*rows),
                class_="table table-striped table-hover table-sm",
            ),
            style=STYLE_SCROLL_TABLE,
        )
