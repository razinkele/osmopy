"""Auto-generate Shiny input widgets from OSMOSE schema fields."""

from __future__ import annotations

import logging

from shiny import ui
from osmose.schema.base import OsmoseField, ParamType
from ui.styles import STYLE_HINT

_log = logging.getLogger("osmose.param_form")


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
    # Build unique input ID and config key
    if field.indexed and species_idx is not None:
        config_key = field.resolve_key(species_idx)
        input_id = f"{prefix}{config_key}".replace(".", "_")
    else:
        config_key = field.key_pattern
        input_id = f"{prefix}{config_key}".replace(".", "_").replace("{idx}", "")

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

    label = field.description or field.key_pattern
    if field.unit:
        label = f"{label} ({field.unit})"

    match field.param_type:
        case ParamType.FLOAT:
            widget = ui.input_numeric(
                input_id,
                label,
                value=default if default is not None else 0.0,  # type: ignore[arg-type]
                min=field.min_val,
                max=field.max_val,
                step=_guess_step(field),
            )
            hint = constraint_hint(field)
            if hint:
                return ui.div(
                    widget,
                    ui.tags.small(hint, style=STYLE_HINT),
                )
            return widget
        case ParamType.INT:
            widget = ui.input_numeric(
                input_id,
                label,
                value=default if default is not None else 0,  # type: ignore[arg-type]
                min=int(field.min_val) if field.min_val is not None else None,
                max=int(field.max_val) if field.max_val is not None else None,
                step=1,
            )
            hint = constraint_hint(field)
            if hint:
                return ui.div(
                    widget,
                    ui.tags.small(hint, style=STYLE_HINT),
                )
            return widget
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
            style="position: sticky; left: 0; z-index: 2; background: var(--osm-bg-card, #162232); min-width: 200px; padding: 8px 12px;",
        )
    ]
    for i, name in enumerate(species_names):
        header_cells.append(ui.tags.th(name, style="text-align: center; min-width: 90px; padding: 8px;"))
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
                    style="padding: 6px 12px; font-weight: 700; color: #d4a017; background: var(--osm-bg-section, #1a2a3a);",
                ),
                **{"data-spt-cat": cat_id, "onclick": f"toggleSptCategory('{cat_id}')"},
                style="cursor: pointer;",
            )
        )

        # Parameter rows
        for field in cat_fields:
            label = field.description or field.key_pattern
            unit_text = f" ({field.unit})" if field.unit else ""
            param_cell = ui.tags.td(
                ui.tags.span(label),
                ui.tags.span(unit_text, style="color: #5a6a7a;"),
                style="padding: 5px 12px; position: sticky; left: 0; z-index: 1; background: var(--osm-bg-card, #0f1923);",
            )

            value_cells = []
            for i in range(n_species):
                sp_idx = start_idx + i
                config_key = field.resolve_key(sp_idx)
                # Input ID: spt_{key_without_sp_idx}_{species_idx}
                base_key = (
                    field.key_pattern.replace(".sp{idx}", "")
                    .replace("{idx}", "")
                    .replace(".", "_")
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
                        value=val if val is not None else 0,
                        min=field.min_val,
                        max=field.max_val,
                        step=_guess_step(field) if field.param_type == ParamType.FLOAT else 1,
                        width="90px",
                    )
                elif field.param_type == ParamType.BOOL:
                    widget = ui.input_switch(input_id, "", value=bool(val) if val is not None else False)
                elif field.param_type == ParamType.ENUM:
                    choices = {c: c for c in (field.choices or [])}
                    widget = ui.input_select(input_id, "", choices=choices, selected=val, width="90px")
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
            span.textContent = visible ? text.replace('\u25bc', '\u25b6') : text.replace('\u25b6', '\u25bc');
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
