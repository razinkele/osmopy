"""Tests for demo generation and config migration."""

import pytest

from osmose.demo import osmose_demo, migrate_config


def test_osmose_demo_bay_of_biscay(tmp_path):
    result = osmose_demo("bay_of_biscay", tmp_path)
    assert "config_file" in result
    assert result["config_file"].exists()
    assert result["output_dir"].exists()
    # Should have master config + sub-files
    csv_files = list(result["config_file"].parent.glob("*.csv"))
    assert len(csv_files) >= 2


def test_osmose_demo_unknown_scenario(tmp_path):
    with pytest.raises(ValueError, match="Unknown scenario"):
        osmose_demo("nonexistent", tmp_path)


def test_osmose_demo_list():
    from osmose.demo import list_demos

    demos = list_demos()
    assert "bay_of_biscay" in demos


def test_migrate_config_noop():
    """Config already at target version should be unchanged."""
    config = {"osmose.version": "4.3.0", "simulation.nspecies": "3"}
    result = migrate_config(config, target_version="4.3.0")
    assert result == config


def test_migrate_config_renames():
    """Config migration should rename deprecated keys."""
    config = {"simulation.nplankton": "2"}
    result = migrate_config(config, target_version="4.3.0")
    assert "simulation.nresource" in result
    assert "simulation.nplankton" not in result
