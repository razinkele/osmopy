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
