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
