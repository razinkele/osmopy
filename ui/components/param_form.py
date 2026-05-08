"""Auto-generate Shiny input widgets from OSMOSE schema fields."""

from __future__ import annotations

from html import escape as _esc
from typing import TYPE_CHECKING, Any

from shiny import ui
from osmose.logging import setup_logging
from osmose.schema.base import OsmoseField, ParamType

if TYPE_CHECKING:
    from shiny.session import Session

_log = setup_logging("osmose.param_form")


def input_id_for_key(key: str, prefix: str = "") -> str:
    """Convert a config key to its Shiny input ID.

    Shiny input IDs cannot contain dots, so the canonical encoding is
    `key.replace(".", "_")` plus an optional prefix. This helper exists
    so that `render_field` (which writes the input) and `sync_inputs`
    (which reads it back) can call exactly the same function rather
    than duplicating the encoding inline. M13 (2026-05-06).
    """
    return f"{prefix}{key}".replace(".", "_")


def input_id_for_field(field: OsmoseField, idx: int | None = None, prefix: str = "") -> str:
    """Convert an OsmoseField (+ optional species/fishery/map index) to its Shiny input ID.

    For an indexed field with `idx` provided, resolves the placeholder
    (e.g. `species.linf.sp{idx}` -> `species.linf.sp3` -> `species_linf_sp3`).
    For an indexed field without `idx`, drops the placeholder for use as a
    generic-form input id (e.g. `species_linf_sp`). For a non-indexed field
    the key is used as-is.
    """
    if field.indexed and idx is not None:
        return input_id_for_key(field.resolve_key(idx), prefix)
    if field.indexed:
        return input_id_for_key(field.key_pattern, prefix).replace("{idx}", "")
    return input_id_for_key(field.key_pattern, prefix)


def _tooltip_content(field: OsmoseField) -> str:
    """Build tooltip HTML content from field metadata."""
    parts = [f"<strong>{_esc(field.description)}</strong>"]
    if field.min_val is not None or field.max_val is not None:
        range_str = constraint_hint(field)
        if range_str:
            parts.append(f"<br>{_esc(range_str)}")
    if field.default is not None:
        parts.append(f"<br>Default: {_esc(str(field.default))}")
    parts.append(f"<br><code>{_esc(field.key_pattern)}</code>")
    return "".join(parts)


def _wrap_with_tooltip(label: str, field: OsmoseField) -> ui.Tag:
    """Wrap a label string with a (?) tooltip icon."""
    content = _tooltip_content(field)
    return ui.tags.span(
        label,
        " ",
        ui.tags.span(
            "(?)",
            class_="osm-tooltip-icon",
            tabindex="0",
            **{
                "data-bs-toggle": "popover",
                "data-bs-trigger": "hover focus click",
                "data-bs-html": "true",
                "data-bs-content": content,
                "data-bs-placement": "top",
                "aria-label": f"Help: {field.description}",
            },
        ),
    )


def constraint_hint(field: OsmoseField) -> str:
    """Generate a constraint hint string for a field.

    Returns text like 'Range: 1.0 — 200.0 cm' or empty string if no constraints.
    """
    parts: list[str] = []
    if field.min_val is not None and field.max_val is not None:
        parts.append(f"Range: {field.min_val} — {field.max_val}")
    elif field.min_val is not None:
        parts.append(f"Min: {field.min_val}")
    elif field.max_val is not None:
        parts.append(f"Max: {field.max_val}")
    if field.unit and parts:
        parts[0] = f"{parts[0]} {field.unit}"
    return " | ".join(parts)


def render_field(
    field: OsmoseField,
    species_idx: int | None = None,
    prefix: str = "",
    config: dict[str, str] | None = None,
) -> ui.Tag:
    """Generate a Shiny input widget from an OsmoseField.

    Args:
        field: The schema field definition.
        species_idx: Species index for indexed fields (required if field.indexed).
        prefix: Optional prefix for the input ID (for namespacing).
        config: Optional config dict to read initial values from (overrides field.default).

    Returns:
        A Shiny UI element (input widget).
    """
    # Build unique input ID and config key (M13: use the canonical helper)
    if field.indexed and species_idx is not None:
        config_key = field.resolve_key(species_idx)
    else:
        config_key = field.key_pattern
    input_id = input_id_for_field(field, idx=species_idx, prefix=prefix)

    # H12: detect multi-value entries (";"-separated arrays like "2.3;1.8"
    # for per-stage parameters). The single-value inputs below cannot
    # represent these — render a visibly read-only text input with a label
    # directing the user to edit via the Advanced tab. The reverse-side
    # guard at ui/state.py:sync_inputs already preserves the original
    # array if a single-value form somehow tries to overwrite it.
    if config is not None and config_key in config:
        cfg_val = config[config_key]
        if isinstance(cfg_val, str) and ";" in cfg_val:
            base_label = field.description or field.key_pattern
            label = ui.tags.span(
                f"{base_label} (multi-value — edit in Advanced)",
                class_="osm-multivalue-readonly-label",
            )
            return ui.input_text(
                input_id,
                label,
                value=cfg_val,
                placeholder="multi-value array (read-only here)",
            )

    # Resolve initial value: config takes priority over field.default
    default = field.default
    if config is not None and config_key in config:
        raw = config[config_key]
        if field.param_type == ParamType.FLOAT:
            try:
                default = float(raw)
            except (ValueError, TypeError):
                _log.warning(
                    "Config value for %s is not a valid float: %r, using default %s",
                    config_key,
                    raw,
                    field.default,
                )
                default = field.default
        elif field.param_type == ParamType.INT:
            try:
                default = int(raw)
            except (ValueError, TypeError):
                _log.warning(
                    "Config value for %s is not a valid int: %r, using default %s",
                    config_key,
                    raw,
                    field.default,
                )
                default = field.default
        elif field.param_type == ParamType.BOOL:
            default = str(raw).lower() in ("true", "1", "yes")
        else:
            default = raw

    label_text = field.description or field.key_pattern
    if field.unit:
        label_text = f"{label_text} ({field.unit})"
    label = _wrap_with_tooltip(label_text, field)

    match field.param_type:
        case ParamType.FLOAT:
            return ui.input_numeric(
                input_id,
                label,
                value=default if default is not None else 0.0,  # type: ignore[arg-type]
                min=field.min_val,
                max=field.max_val,
                step=_guess_step(field),
            )
        case ParamType.INT:
            return ui.input_numeric(
                input_id,
                label,
                value=default if default is not None else 0,  # type: ignore[arg-type]
                min=int(field.min_val) if field.min_val is not None else None,
                max=int(field.max_val) if field.max_val is not None else None,
                step=1,
            )
        case ParamType.BOOL:
            return ui.input_switch(
                input_id,
                label,
                value=bool(default) if default is not None else False,
            )
        case ParamType.STRING:
            return ui.input_text(
                input_id,
                label,
                value=str(default) if default is not None else "",
            )
        case ParamType.ENUM:
            # When choice_labels is set, show friendly labels in the dropdown
            # while keeping Shiny's underlying value equal to the engine-readable
            # choice string (e.g. selectivity '0' shown as 'knife-edge').
            if field.choice_labels:
                choices = {c: field.choice_labels.get(c, c) for c in (field.choices or [])}
            else:
                choices = {c: c for c in (field.choices or [])}
            return ui.input_select(
                input_id,
                label,
                choices=choices,
                selected=default,  # type: ignore[arg-type]
            )
        case ParamType.FILE_PATH:
            return ui.input_file(
                input_id,
                label,
                accept=[".csv", ".nc", ".properties"],
            )
        case ParamType.MATRIX:
            # Matrix editing is handled by a separate component
            return ui.input_file(
                input_id,
                f"{label} (CSV matrix)",
                accept=[".csv"],
            )
        case _:
            return ui.input_text(input_id, label, value=str(default or ""))


def render_category(
    fields: list[OsmoseField],
    species_idx: int | None = None,
    prefix: str = "",
    show_advanced: bool = False,
    config: dict[str, str] | None = None,
) -> ui.Tag:
    """Generate a form section for a group of fields.

    Args:
        fields: List of OsmoseField objects to render.
        species_idx: Species index for indexed fields.
        prefix: Input ID prefix.
        show_advanced: Whether to include advanced fields.
        config: Optional config dict to read initial values from.

    Returns:
        A Shiny UI div containing all the input widgets.
    """
    widgets = []
    for field in fields:
        if field.advanced and not show_advanced:
            continue
        widgets.append(render_field(field, species_idx, prefix, config=config))
    return ui.div(*widgets)


def render_species_params(
    fields: list[OsmoseField],
    species_idx: int,
    species_name: str,
    show_advanced: bool = False,
    config: dict[str, str] | None = None,
) -> ui.Tag:
    """Render parameters for a single species inside an accordion panel.

    Args:
        fields: All species-level OsmoseField objects.
        species_idx: The species index (0-based).
        species_name: Display name of the species.
        show_advanced: Whether to include advanced fields.
        config: Optional config dict to read initial values from.
    """
    # Group fields by category
    categories: dict[str, list[OsmoseField]] = {}
    for f in fields:
        cat = f.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(f)

    panels = []
    for cat_name, cat_fields in categories.items():
        filtered = [f for f in cat_fields if show_advanced or not f.advanced]
        if filtered:
            panel_content = render_category(
                filtered,
                species_idx,
                show_advanced=show_advanced,
                config=config,
            )
            panels.append(
                ui.accordion_panel(
                    cat_name.replace("_", " ").title(),
                    panel_content,
                )
            )

    return ui.card(
        ui.card_header(f"Species {species_idx}: {species_name}"),
        ui.accordion(*panels, id=f"species_{species_idx}_accordion", open=False),
    )


def render_species_table(
    fields: list[OsmoseField],
    n_species: int,
    species_names: list[str],
    start_idx: int = 0,
    show_advanced: bool = False,
    config: dict[str, str] | None = None,
) -> ui.Tag:
    """Render a spreadsheet-style table: params as rows, species as columns.

    Args:
        fields: Schema fields to display (must be indexed).
        n_species: Number of species columns.
        species_names: Display names for each species.
        start_idx: Starting species index (for LTL resources offset by nspecies).
        show_advanced: Whether to include advanced fields.
        config: Optional config dict for initial values.
    """
    if n_species == 0:
        return ui.div(
            "Load a configuration to view species parameters.",
            style="padding: 20px; text-align: center; color: #5a6a7a;",
        )

    # Filter to indexed, non-advanced fields
    visible = [f for f in fields if f.indexed and (show_advanced or not f.advanced)]

    # Group by category
    categories: dict[str, list[OsmoseField]] = {}
    for f in visible:
        cat = f.category or "other"
        categories.setdefault(cat, []).append(f)

    # Build header row: Parameter | Species0 | Species1 | ...
    header_cells = [
        ui.tags.th(
            "Parameter",
            style=(
                "position: sticky; left: 0; z-index: 2;"
                " background: var(--osm-bg-card, #162232);"
                " min-width: 200px; padding: 8px 12px;"
            ),
        )
    ]
    for i, name in enumerate(species_names):
        th_content: list[ui.Tag | str] = [name]
        if i == 0 and n_species > 1:
            th_content.append(
                ui.tags.button(
                    "📋 Copy to all",
                    class_="btn btn-sm btn-outline-warning mt-1",
                    style="font-size: 9px; padding: 1px 5px; display: block; margin: 2px auto 0;",
                    onclick=(
                        "Shiny.setInputValue('copy_sp0_to_all', Date.now(),"
                        " {priority: 'event'})"
                    ),
                    title="Copy all species 0 values to other species",
                )
            )
        header_cells.append(
            ui.tags.th(*th_content, style="text-align: center; min-width: 90px; padding: 8px;")
        )
    header = ui.tags.thead(
        ui.tags.tr(*header_cells, style="border-bottom: 2px solid var(--osm-border, #2d3d50);")
    )

    # Build body rows grouped by category
    rows = []
    for cat_name, cat_fields in categories.items():
        display_cat = cat_name.replace("_", " ").title()
        n_fields = len(cat_fields)
        # Category group header row (collapsible via JS)
        cat_id = f"spt_cat_{cat_name}"
        rows.append(
            ui.tags.tr(
                ui.tags.td(
                    ui.tags.span(
                        f"\u25bc {display_cat} ",
                        ui.tags.span(
                            f"({n_fields} params)",
                            style="color: #5a6a7a; font-weight: 400; font-size: 10px;",
                        ),
                        style="cursor: pointer;",
                    ),
                    colspan=str(n_species + 1),
                    style=(
                        "padding: 6px 12px; font-weight: 700;"
                        " color: #d4a017;"
                        " background: var(--osm-bg-section, #1a2a3a);"
                    ),
                ),
                **{"data-spt-cat": cat_id, "onclick": f"toggleSptCategory('{cat_id}')"},
                style="cursor: pointer;",
            )
        )

        # Parameter rows
        for field in cat_fields:
            label = field.description or field.key_pattern
            unit_text = f" ({field.unit})" if field.unit else ""
            tooltip_html = _tooltip_content(field)
            param_cell = ui.tags.td(
                ui.tags.span(label),
                ui.tags.span(unit_text, style="color: #5a6a7a;"),
                " ",
                ui.tags.span(
                    "(?)",
                    class_="osm-tooltip-icon",
                    tabindex="0",
                    **{
                        "data-bs-toggle": "popover",
                        "data-bs-trigger": "hover focus click",
                        "data-bs-html": "true",
                        "data-bs-content": tooltip_html,
                        "data-bs-placement": "right",
                        "aria-label": f"Help: {field.description}",
                    },
                ),
                style=(
                    "padding: 5px 12px; position: sticky;"
                    " left: 0; z-index: 1;"
                    " background: var(--osm-bg-card, #0f1923);"
                ),
            )

            value_cells = []
            for i in range(n_species):
                sp_idx = start_idx + i
                config_key = field.resolve_key(sp_idx)
                # Input ID: spt_{key_without_sp_idx}_{species_idx}
                base_key = (
                    field.key_pattern.replace(".sp{idx}", "").replace("{idx}", "").replace(".", "_")
                )
                input_id = f"spt_{base_key}_{sp_idx}"

                # Resolve value from config or default
                val = field.default
                if config and config_key in config:
                    raw = config[config_key]
                    if field.param_type in (ParamType.FLOAT, ParamType.INT):
                        try:
                            val = float(raw) if field.param_type == ParamType.FLOAT else int(raw)
                        except (ValueError, TypeError):
                            val = field.default
                    elif field.param_type == ParamType.BOOL:
                        val = str(raw).lower() in ("true", "1", "yes")
                    else:
                        val = raw

                cell_style = "text-align: center; padding: 4px;"
                if field.param_type in (ParamType.FLOAT, ParamType.INT):
                    widget = ui.input_numeric(
                        input_id,
                        "",
                        value=float(val) if val is not None else 0,
                        min=field.min_val,
                        max=field.max_val,
                        step=_guess_step(field) if field.param_type == ParamType.FLOAT else 1,
                        width="90px",
                    )
                elif field.param_type == ParamType.BOOL:
                    widget = ui.input_switch(
                        input_id, "", value=bool(val) if val is not None else False
                    )
                elif field.param_type == ParamType.ENUM:
                    # See render_field's ENUM branch — choice_labels keeps
                    # the stored value (engine-readable) while showing a
                    # friendly label in the dropdown.
                    if field.choice_labels:
                        choices = {c: field.choice_labels.get(c, c) for c in (field.choices or [])}
                    else:
                        choices = {c: c for c in (field.choices or [])}
                    widget = ui.input_select(
                        input_id, "", choices=choices, selected=str(val) if val is not None else None, width="90px"
                    )
                elif field.param_type in (ParamType.FILE_PATH, ParamType.MATRIX):
                    widget = ui.tags.span("file", style="color: #5a6a7a; font-size: 11px;")
                else:
                    widget = ui.input_text(input_id, "", value=str(val or ""), width="90px")

                value_cells.append(ui.tags.td(widget, style=cell_style))

            rows.append(
                ui.tags.tr(
                    param_cell,
                    *value_cells,
                    **{"data-spt-group": cat_id},
                    style="border-bottom: 1px solid var(--osm-border-dim, #1a2a3a);",
                )
            )

    body = ui.tags.tbody(*rows)
    table = ui.tags.table(
        header,
        body,
        class_="table table-sm",
        style="width: 100%; border-collapse: collapse; font-size: 12px;",
    )

    # Client-side JS for collapsing categories (no server round-trip)
    collapse_js = ui.tags.script("""
    function toggleSptCategory(catId) {
        var rows = document.querySelectorAll('[data-spt-group="' + catId + '"]');
        var header = document.querySelector('[data-spt-cat="' + catId + '"]');
        var visible = rows.length > 0 && rows[0].style.display !== 'none';
        rows.forEach(function(r) { r.style.display = visible ? 'none' : ''; });
        var span = header.querySelector('span');
        if (span) {
            var text = span.textContent;
            span.textContent = visible
                ? text.replace('\u25bc', '\u25b6')
                : text.replace('\u25b6', '\u25bc');
        }
    }
    """)

    return ui.div(
        ui.div(table, style="max-height: 600px; overflow: auto;"),
        collapse_js,
    )


def _guess_step(field: OsmoseField) -> float:
    """Guess an appropriate step value for a numeric input."""
    if field.max_val is not None and field.min_val is not None:
        range_val = field.max_val - field.min_val
        if range_val <= 1:
            return 0.01
        elif range_val <= 10:
            return 0.1
        elif range_val <= 100:
            return 1.0
        else:
            return 10.0
    return 0.1


def copy_species0_to_all(
    fields: list[OsmoseField],
    n_species: int,
    config: dict[str, str],
    input: Any,
    session: Session | None,
    start_idx: int = 0,
    show_advanced: bool = False,
) -> int:
    """Copy all species 0 input values to all other species.

    Returns the number of parameters copied.
    """
    visible = [f for f in fields if f.indexed and (show_advanced or not f.advanced)]
    copied = 0
    for field in visible:
        if field.param_type in (ParamType.FILE_PATH, ParamType.MATRIX):
            continue
        base_key = (
            field.key_pattern.replace(".sp{idx}", "").replace("{idx}", "").replace(".", "_")
        )
        src_id = f"spt_{base_key}_{start_idx}"
        try:
            src_val = getattr(input, src_id)()
        except Exception:
            continue
        if src_val is None:
            continue

        for i in range(1, n_species):
            sp_idx = start_idx + i
            config_key = field.resolve_key(sp_idx)
            dst_id = f"spt_{base_key}_{sp_idx}"

            config[config_key] = str(src_val)

            if field.param_type in (ParamType.FLOAT, ParamType.INT):
                ui.update_numeric(dst_id, value=float(src_val), session=session)
            elif field.param_type == ParamType.BOOL:
                ui.update_switch(dst_id, value=bool(src_val), session=session)
            elif field.param_type == ParamType.ENUM:
                ui.update_select(dst_id, selected=str(src_val), session=session)
            else:
                ui.update_text(dst_id, value=str(src_val), session=session)
        copied += 1
    return copied
