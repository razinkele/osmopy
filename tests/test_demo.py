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
    config = {"simulation.nplankton": "2", "grid.ncolumn": "20", "grid.nline": "20"}
    result = migrate_config(config, target_version="4.3.0")
    assert "simulation.nresource" in result
    assert "simulation.nplankton" not in result
    # grid.ncolumn -> grid.nlon, grid.nline -> grid.nlat
    assert "grid.nlon" in result
    assert "grid.ncolumn" not in result
    assert "grid.nlat" in result
    assert "grid.nline" not in result


def test_osmose_demo_copies_support_dirs(tmp_path):
    """Demo must copy subdirectories (reproduction, grid, predation, ltl)."""
    result = osmose_demo("bay_of_biscay", tmp_path)
    config_dir = result["config_file"].parent
    assert (config_dir / "reproduction" / "reprod_anchovy.csv").exists()
    assert (config_dir / "grid" / "bay_of_biscay_mask.csv").exists()
    assert (config_dir / "predation" / "accessibility_matrix.csv").exists()
    assert (config_dir / "ltl").exists()


def test_osmose_demo_config_is_loadable(tmp_path):
    """Generated demo config must load and validate correctly."""
    from osmose.config.reader import OsmoseConfigReader

    result = osmose_demo("bay_of_biscay", tmp_path)
    reader = OsmoseConfigReader()
    config = reader.read(result["config_file"])
    assert int(config["simulation.nspecies"]) == 8
    assert config["species.name.sp0"] == "Anchovy"


def test_migrate_from_pre_3_2():
    config = {
        "osmose.version": "3.1.0",
        "population.initialization.biomass.sp0": "1000",
        "population.initialization.biomass.sp1": "2000",
    }
    result = migrate_config(config, target_version="4.3.3")
    assert "population.seeding.biomass.sp0" in result
    assert "population.seeding.biomass.sp1" in result
    assert "population.initialization.biomass.sp0" not in result
    assert result["osmose.version"] == "4.3.3"


def test_migrate_from_pre_4_2_3():
    config = {
        "osmose.version": "4.2.2",
        "simulation.nplankton": "6",
        "plankton.name.plk0": "SmallPhyto",
        "plankton.tl.plk0": "1.0",
    }
    result = migrate_config(config, target_version="4.3.3")
    assert "simulation.nresource" in result
    assert "simulation.nplankton" not in result
    assert "resource.name.plk0" in result
    assert "resource.tl.plk0" in result
    assert "plankton.name.plk0" not in result
    assert result["osmose.version"] == "4.3.3"


def test_migrate_from_pre_4_2_5():
    config = {
        "osmose.version": "4.2.4",
        "mortality.natural.rate.sp0": "0.8",
    }
    result = migrate_config(config, target_version="4.3.3")
    assert "mortality.additional.rate.sp0" in result
    assert "mortality.natural.rate.sp0" not in result


def test_migrate_from_pre_3_3_3():
    config = {
        "osmose.version": "3.3.0",
        "grid.ncolumn": "20",
        "grid.nline": "20",
    }
    result = migrate_config(config, target_version="4.3.3")
    assert "grid.nlon" in result
    assert "grid.nlat" in result
    assert "grid.ncolumn" not in result
    assert "grid.nline" not in result


def test_migrate_sequential_application():
    config = {
        "osmose.version": "3.0.0",
        "population.initialization.biomass.sp0": "1000",
        "grid.ncolumn": "20",
        "grid.nline": "20",
        "simulation.nplankton": "6",
        "mortality.natural.rate.sp0": "0.8",
    }
    result = migrate_config(config, target_version="4.3.3")
    assert "population.seeding.biomass.sp0" in result
    assert "grid.nlon" in result
    assert "grid.nlat" in result
    assert "simulation.nresource" in result
    assert "mortality.additional.rate.sp0" in result
    assert "population.initialization.biomass.sp0" not in result
    assert "grid.ncolumn" not in result
    assert "simulation.nplankton" not in result
    assert "mortality.natural.rate.sp0" not in result
    assert result["osmose.version"] == "4.3.3"


def test_migrate_already_at_target():
    config = {"osmose.version": "4.3.3", "simulation.nspecies": "8"}
    result = migrate_config(config, target_version="4.3.3")
    assert result == config


def test_migrate_no_version_key():
    config = {
        "population.initialization.biomass.sp0": "1000",
        "grid.ncolumn": "20",
    }
    result = migrate_config(config, target_version="4.3.3")
    assert "population.seeding.biomass.sp0" in result
    assert "grid.nlon" in result
