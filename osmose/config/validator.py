"""Validate OSMOSE configuration against schema."""

from __future__ import annotations

from pathlib import Path

from osmose.schema.base import ParamType


def validate_config(config: dict[str, str], registry) -> tuple[list[str], list[str]]:
    """Validate config values against registry schema.

    Returns (errors, warnings) -- lists of human-readable messages.
    """
    errors: list[str] = []
    warnings: list[str] = []

    for key, value in config.items():
        field = registry.match_field(key)
        if field is None:
            continue

        # Skip null/none/empty values — these are optional unset params
        if not value or value.lower() in ("null", "none"):
            continue

        # Skip multi-value params (semicolon-separated stage values)
        if ";" in value:
            continue

        # Type check
        if field.param_type in (ParamType.FLOAT, ParamType.INT):
            try:
                num = float(value)
            except (ValueError, TypeError):
                errors.append(f"{key}: expected number, got '{value}'")
                continue

            # Bounds check
            if field.min_val is not None and num < field.min_val:
                errors.append(f"{key}: value {num} below minimum {field.min_val}")
            if field.max_val is not None and num > field.max_val:
                errors.append(f"{key}: value {num} above maximum {field.max_val}")

        elif field.param_type == ParamType.BOOL:
            if value.lower() not in ("true", "false", "0", "1"):
                errors.append(f"{key}: expected boolean, got '{value}'")

        elif field.param_type == ParamType.ENUM:
            if field.choices and value not in field.choices:
                errors.append(
                    f"Invalid value for '{key}': '{value}' "
                    f"(expected one of {field.choices})"
                )

    return errors, warnings


def validate_field(key: str, value: str, field) -> str | None:
    """Validate a single field value. Returns error message or None."""
    from osmose.schema.base import ParamType

    if field.param_type in (ParamType.FLOAT, ParamType.INT):
        try:
            num = float(value)
        except (ValueError, TypeError):
            return f"Expected number, got '{value}'"
        if field.min_val is not None and num < field.min_val:
            return f"Value {num} below minimum {field.min_val}"
        if field.max_val is not None and num > field.max_val:
            return f"Value {num} above maximum {field.max_val}"
    elif field.param_type == ParamType.BOOL:
        if value.lower() not in ("true", "false", "0", "1"):
            return f"Expected boolean, got '{value}'"
    return None


def check_file_references(
    config: dict[str, str],
    base_dir: str,
    registry=None,
) -> list[str]:
    """Check that file-referencing parameters point to existing files."""
    missing = []
    base = Path(base_dir)
    for key, value in config.items():
        is_file_param = False
        if registry is not None:
            field = registry.match_field(key)
            if field is not None and field.param_type == ParamType.FILE_PATH:
                is_file_param = True
        else:
            is_file_param = "file" in key.lower()
        if not is_file_param:
            continue
        if not value or value.lower() in ("null", "none"):
            continue
        ref = Path(value)
        if not ref.is_absolute():
            ref = base / ref
        if not ref.exists():
            missing.append(f"File not found for '{key}': {ref}")
    return missing


def check_species_consistency(config: dict[str, str]) -> list[str]:
    """Check that species.name keys exist for all declared species and resources."""
    warnings = []
    try:
        nspecies = int(float(config.get("simulation.nspecies", "0")))
    except (ValueError, TypeError):
        warnings.append("simulation.nspecies has non-numeric value")
        return warnings
    try:
        nresource = int(float(config.get("simulation.nresource", "0")))
    except (ValueError, TypeError):
        nresource = 0
    for i in range(nspecies):
        key = f"species.name.sp{i}"
        if key not in config:
            warnings.append(f"Missing focal species name: {key}")
    for i in range(nresource):
        idx = nspecies + i
        key = f"species.name.sp{idx}"
        if key not in config:
            warnings.append(f"Missing resource species name: {key}")
    return warnings
