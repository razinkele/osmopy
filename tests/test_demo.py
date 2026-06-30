"""Tests for demo generation and config migration."""

import pytest

from osmose.demo import DEMO_INFO, demo_info, migrate_config, osmose_demo, list_demos


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


def test_osmose_demo_honors_data_dir_env(tmp_path, monkeypatch):
    """OSMOSE_DATA_DIR overrides the bundled-data location (wheel/Docker installs
    where data/ is not a sibling of the installed osmose package)."""
    custom = tmp_path / "custom_data" / "minimal"
    custom.mkdir(parents=True)
    (custom / "osm_all-parameters.csv").write_text("simulation.nspecies ; 2\nMARKER ; from_env\n")
    monkeypatch.setenv("OSMOSE_DATA_DIR", str(tmp_path / "custom_data"))
    result = osmose_demo("minimal", tmp_path / "out")
    # Config came from the env-pointed dir, not the repo bundle.
    assert "MARKER ; from_env" in result["config_file"].read_text()


def test_osmose_demo_warns_when_data_missing(tmp_path, monkeypatch, caplog):
    """A missing data bundle warns loudly (not a silent stub) but still produces a file."""
    import logging

    monkeypatch.setenv("OSMOSE_DATA_DIR", str(tmp_path / "empty"))  # no subdirs present
    with caplog.at_level(logging.WARNING, logger="osmose.demo"):
        result = osmose_demo("minimal", tmp_path / "out")
    assert result["config_file"].exists()  # stub still written — no crash
    assert any("Bundled demo data not found" in r.message for r in caplog.records)


def test_osmose_demo_list():
    from osmose.demo import list_demos

    demos = list_demos()
    assert "bay_of_biscay" in demos
    assert "eec" in demos
    assert "eec_full" in demos
    assert "minimal" in demos


def test_osmose_demo_eec(tmp_path):
    result = osmose_demo("eec", tmp_path)
    assert result["config_file"].exists()
    assert result["output_dir"].exists()
    csv_files = list(result["config_file"].parent.glob("*.csv"))
    assert len(csv_files) >= 2


def test_osmose_demo_eec_config_is_loadable(tmp_path):
    from osmose.config.reader import OsmoseConfigReader

    result = osmose_demo("eec", tmp_path)
    reader = OsmoseConfigReader()
    config = reader.read(result["config_file"])
    assert int(config["simulation.nspecies"]) == 6
    assert config["species.name.sp0"] == "Herring"
    assert config["species.name.sp5"] == "Cod"


def test_osmose_demo_eec_copies_support_dirs(tmp_path):
    result = osmose_demo("eec", tmp_path)
    config_dir = result["config_file"].parent
    assert (config_dir / "reproduction" / "reprod_herring.csv").exists()
    assert (config_dir / "reproduction" / "reprod_cod.csv").exists()
    assert (config_dir / "grid" / "eec_mask.csv").exists()
    assert (config_dir / "predation" / "accessibility_matrix.csv").exists()


def test_osmose_demo_eec_full(tmp_path):
    result = osmose_demo("eec_full", tmp_path)
    assert result["config_file"].exists()
    assert result["output_dir"].exists()
    assert result["config_file"].name == "eec_all-parameters.csv"


def test_osmose_demo_eec_full_config_is_loadable(tmp_path):
    from osmose.config.reader import OsmoseConfigReader

    result = osmose_demo("eec_full", tmp_path)
    reader = OsmoseConfigReader()
    config = reader.read(result["config_file"])
    assert int(config["simulation.nspecies"]) == 14
    assert int(config["simulation.nresource"]) == 10
    assert config["species.name.sp0"] == "lesserSpottedDogfish"
    assert config["species.name.sp13"] == "squids"


def test_osmose_demo_eec_full_has_netcdf_and_maps(tmp_path):
    result = osmose_demo("eec_full", tmp_path)
    config_dir = result["config_file"].parent
    assert (config_dir / "eec_grid-mask.nc").exists()
    assert (config_dir / "eec_ltlbiomassTons.nc").exists()
    assert (config_dir / "maps").is_dir()
    map_files = list((config_dir / "maps").glob("*.csv"))
    assert len(map_files) >= 35
    assert (config_dir / "reproduction").is_dir()
    reprod_files = list((config_dir / "reproduction").glob("*.csv"))
    assert len(reprod_files) == 14


def test_osmose_demo_minimal(tmp_path):
    result = osmose_demo("minimal", tmp_path)
    assert result["config_file"].exists()
    assert result["output_dir"].exists()
    csv_files = list(result["config_file"].parent.glob("*.csv"))
    assert len(csv_files) >= 2


def test_osmose_demo_minimal_config_is_loadable(tmp_path):
    from osmose.config.reader import OsmoseConfigReader

    result = osmose_demo("minimal", tmp_path)
    reader = OsmoseConfigReader()
    config = reader.read(result["config_file"])
    assert int(config["simulation.nspecies"]) == 2
    assert config["species.name.sp0"] == "Anchovy"
    assert config["species.name.sp1"] == "Hake"


def test_osmose_demo_minimal_has_predation(tmp_path):
    result = osmose_demo("minimal", tmp_path)
    config_dir = result["config_file"].parent
    assert (config_dir / "predation" / "accessibility_matrix.csv").exists()


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


_REQUIRED = ("title", "region", "species", "resources", "engine", "summary")


def test_demo_info_covers_all_demos_with_full_fields():
    for name in list_demos():
        assert name in DEMO_INFO, f"DEMO_INFO missing {name}"
        entry = DEMO_INFO[name]
        for field in _REQUIRED:
            assert entry.get(field), f"{name}.{field} empty"


def test_demo_info_accessor_and_engine_facts():
    # baltic + eec_full run on the Java 4.4.1 jar too (sub-projects A/C1/C2), not Python-only
    assert demo_info("baltic")["engine"] == "Java 4.4.1 + Python"
    assert demo_info("eec_full")["engine"] == "Java 4.4.1 + Python"
    assert demo_info("bay_of_biscay")["engine"].startswith("Java")
    assert demo_info("eec")["title"] == "Eastern English Channel"
    assert demo_info("unknown") is None
