import tempfile
from pathlib import Path
from osmose.config.writer import OsmoseConfigWriter
from osmose.config.reader import OsmoseConfigReader


def test_write_creates_master_file():
    config = {
        "simulation.time.ndtperyear": 12,
        "simulation.nspecies": 2,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(config, Path(tmpdir))
        master = Path(tmpdir) / "osm_all-parameters.csv"
        assert master.exists()


def test_write_creates_species_subfile():
    config = {
        "simulation.nspecies": 2,
        "species.name.sp0": "Anchovy",
        "species.linf.sp0": 19.5,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(config, Path(tmpdir))
        species_file = Path(tmpdir) / "osm_param-species.csv"
        assert species_file.exists()
        content = species_file.read_text()
        assert "species.name.sp0" in content
        assert "Anchovy" in content


def test_master_references_subfiles():
    config = {
        "simulation.nspecies": 2,
        "species.name.sp0": "Anchovy",
        "grid.nlon": 30,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(config, Path(tmpdir))
        master_content = (Path(tmpdir) / "osm_all-parameters.csv").read_text()
        assert "osmose.configuration.species" in master_content
        assert "osmose.configuration.grid" in master_content


def test_roundtrip_basic():
    """Write config, read it back, verify values match."""
    config = {
        "simulation.time.ndtperyear": "12",
        "simulation.nspecies": "2",
        "species.name.sp0": "Anchovy",
        "species.linf.sp0": "19.5",
        "grid.nlon": "30",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(config, Path(tmpdir))
        reader = OsmoseConfigReader()
        result = reader.read(Path(tmpdir) / "osm_all-parameters.csv")
        for key, value in config.items():
            assert result[key] == str(value), f"Mismatch: {key}: {result.get(key)} != {value}"


def test_write_uses_semicolon_separator():
    config = {"simulation.nspecies": "2"}
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(config, Path(tmpdir))
        content = (Path(tmpdir) / "osm_all-parameters.csv").read_text()
        assert ";" in content


def test_write_empty_config():
    """An empty config should still create a master file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write({}, Path(tmpdir))
        assert (Path(tmpdir) / "osm_all-parameters.csv").exists()


def test_routing_fishing_params():
    config = {
        "fisheries.enabled": "true",
        "fisheries.name.fsh0": "Trawl",
        "mortality.fishing.rate.sp0": "0.3",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(config, Path(tmpdir))
        fishing = Path(tmpdir) / "osm_param-fishing.csv"
        assert fishing.exists()
        content = fishing.read_text()
        assert "fisheries.enabled" in content
        assert "fisheries.name.fsh0" in content
        assert "mortality.fishing.rate.sp0" in content


def test_routing_output_params():
    config = {
        "output.biomass.enabled": "true",
        "output.dir.path": "output",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(config, Path(tmpdir))
        output_file = Path(tmpdir) / "osm_param-output.csv"
        assert output_file.exists()
        content = output_file.read_text()
        assert "output.biomass.enabled" in content


def test_only_creates_subfiles_with_content():
    """Don't create sub-files if there are no params for that category."""
    config = {
        "simulation.nspecies": "1",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(config, Path(tmpdir))
        assert (Path(tmpdir) / "osm_all-parameters.csv").exists()
        assert not (Path(tmpdir) / "osm_param-grid.csv").exists()
        assert not (Path(tmpdir) / "osm_param-species.csv").exists()
