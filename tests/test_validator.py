"""Tests for config validation."""

import pytest
from osmose.config.validator import (
    validate_config,
    check_file_references,
    check_species_consistency,
)


@pytest.fixture
def registry():
    from osmose.schema import build_registry

    return build_registry()


def test_validate_config_valid(registry):
    config = {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "100",
        "simulation.nspecies": "1",
        "species.name.sp0": "Anchovy",
        "species.linf.sp0": "19.5",
    }
    errors, warnings = validate_config(config, registry)
    assert len(errors) == 0


def test_validate_config_bad_type(registry):
    config = {
        "simulation.time.ndtperyear": "not_a_number",
        "simulation.nspecies": "1",
    }
    errors, warnings = validate_config(config, registry)
    assert any("ndtperyear" in e for e in errors)


def test_validate_config_out_of_bounds(registry):
    config = {
        "simulation.time.ndtperyear": "9999",
        "simulation.nspecies": "1",
    }
    errors, warnings = validate_config(config, registry)
    assert any("ndtperyear" in e for e in errors)


def test_check_species_consistency():
    config = {
        "simulation.nspecies": "2",
        "species.name.sp0": "Anchovy",
        # Missing sp1
    }
    warnings = check_species_consistency(config)
    assert len(warnings) > 0


def test_check_file_references(tmp_path):
    # Create a file that exists
    (tmp_path / "grid.csv").write_text("0,0\n")
    config = {
        "grid.mask.file": str(tmp_path / "grid.csv"),
        "reproduction.season.file.sp0": str(tmp_path / "missing.csv"),
    }
    missing = check_file_references(config, str(tmp_path))
    assert len(missing) == 1
    assert "missing.csv" in missing[0]


def test_check_file_references_uses_schema(tmp_path):
    """check_file_references should use FILE_PATH schema type, not string heuristic."""
    from osmose.config.validator import check_file_references
    from osmose.schema import build_registry

    registry = build_registry()
    config = {
        "reproduction.season.file.sp0": "nonexistent.csv",
        "grid.netcdf.file": "nonexistent.nc",
        "species.name.sp0": "Anchovy",
    }
    missing = check_file_references(config, str(tmp_path), registry)
    assert len(missing) == 2
    assert any("reproduction.season.file.sp0" in m for m in missing)
    assert any("grid.netcdf.file" in m for m in missing)


def test_check_species_consistency_checks_resources():
    from osmose.config.validator import check_species_consistency

    config = {
        "simulation.nspecies": "2",
        "simulation.nresource": "2",
        "species.name.sp0": "Anchovy",
        "species.name.sp1": "Sardine",
    }
    warnings = check_species_consistency(config)
    assert len(warnings) == 2
    assert any("sp2" in w for w in warnings)
    assert any("sp3" in w for w in warnings)


def test_validate_config_checks_enum_values():
    from osmose.config.validator import validate_config
    from osmose.schema import build_registry

    registry = build_registry()
    config = {"species.type.sp0": "invalid_type"}
    errors, _ = validate_config(config, registry)
    assert len(errors) >= 1
    assert any("invalid_type" in e for e in errors)


def test_validate_config_accepts_valid_enum():
    from osmose.config.validator import validate_config
    from osmose.schema import build_registry

    registry = build_registry()
    config = {"species.type.sp0": "focal"}
    errors, _ = validate_config(config, registry)
    assert errors == []


def test_check_species_consistency_nonnumeric():
    from osmose.config.validator import check_species_consistency

    config = {"simulation.nspecies": "three"}
    warnings = check_species_consistency(config)
    assert any("nspecies" in w.lower() or "non-numeric" in w.lower() for w in warnings)


def test_check_species_consistency_float():
    from osmose.config.validator import check_species_consistency

    config = {"simulation.nspecies": "3.0", "simulation.nresource": "1.0"}
    warnings = check_species_consistency(config)
    assert isinstance(warnings, list)  # should not crash
