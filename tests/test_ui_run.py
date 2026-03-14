"""Tests for run page logic -- config writing, override parsing, status flow."""

from osmose.config.validator import validate_config
from osmose.schema import build_registry
from ui.pages.run import _inject_random_movement_ncell, parse_overrides, write_temp_config


def test_parse_overrides_empty():
    assert parse_overrides("") == {}


def test_parse_overrides_single():
    assert parse_overrides("simulation.nspecies=5") == {"simulation.nspecies": "5"}


def test_parse_overrides_multiple():
    text = "simulation.nspecies=5\nspecies.k.sp0=0.3"
    result = parse_overrides(text)
    assert result == {"simulation.nspecies": "5", "species.k.sp0": "0.3"}


def test_parse_overrides_skips_blank_lines():
    text = "a=1\n\nb=2\n"
    assert parse_overrides(text) == {"a": "1", "b": "2"}


def test_parse_overrides_strips_whitespace():
    text = "  a = 1  \n  b=2"
    assert parse_overrides(text) == {"a": "1", "b": "2"}


def test_write_temp_config(tmp_path):
    config = {"simulation.nspecies": "3", "species.k.sp0": "0.2"}
    config_path = write_temp_config(config, tmp_path)
    assert config_path.exists()
    assert config_path.name == "osm_all-parameters.csv"
    content = config_path.read_text()
    assert "simulation.nspecies" in content


def test_inject_ncell_for_random_movement():
    config = {
        "grid.nlon": "15",
        "grid.nlat": "12",
        "movement.distribution.method.sp0": "random",
        "movement.distribution.method.sp1": "maps",
        "movement.distribution.method.sp2": "random",
    }
    _inject_random_movement_ncell(config)
    assert config["movement.distribution.ncell.sp0"] == "180"
    assert "movement.distribution.ncell.sp1" not in config
    assert config["movement.distribution.ncell.sp2"] == "180"


def test_inject_ncell_preserves_existing():
    config = {
        "grid.nlon": "15",
        "grid.nlat": "12",
        "movement.distribution.method.sp0": "random",
        "movement.distribution.ncell.sp0": "50",
    }
    _inject_random_movement_ncell(config)
    assert config["movement.distribution.ncell.sp0"] == "50"


def test_prerun_validation_blocks_on_errors():
    registry = build_registry()
    config = {"species.linf.sp0": "not_a_number", "simulation.nspecies": "2"}
    errors, warnings = validate_config(config, registry)
    assert len(errors) > 0


def test_prerun_validation_passes_valid_config():
    registry = build_registry()
    config = {"simulation.nspecies": "3", "species.linf.sp0": "50.0"}
    errors, warnings = validate_config(config, registry)
    assert len(errors) == 0
