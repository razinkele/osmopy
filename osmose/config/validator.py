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

    return errors, warnings


def check_file_references(config: dict[str, str], base_dir: str) -> list[str]:
    """Check that all file-referencing parameters point to existing files.

    Returns list of error messages for missing files.
    """
    missing: list[str] = []
    file_keys = [k for k in config if "file" in k.lower()]
    base = Path(base_dir)

    for key in file_keys:
        value = config[key]
        if not value or value.lower() in ("null", "none", ""):
            continue
        path = Path(value)
        if not path.is_absolute():
            path = base / path
        if not path.exists():
            missing.append(f"{key}: file not found: {path}")

    return missing


def check_species_consistency(config: dict[str, str]) -> list[str]:
    """Check that nspecies matches the number of indexed species params."""
    warnings: list[str] = []
    nspecies_str = config.get("simulation.nspecies", "0")
    try:
        nspecies = int(nspecies_str)
    except ValueError:
        return [f"simulation.nspecies is not a number: {nspecies_str}"]

    # Check species.name.spN exists for all N
    for i in range(nspecies):
        key = f"species.name.sp{i}"
        if key not in config:
            warnings.append(f"Missing {key} (expected {nspecies} species)")

    return warnings
