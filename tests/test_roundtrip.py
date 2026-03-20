"""Comprehensive round-trip tests for OSMOSE config write -> read cycle.

Builds a full config covering ALL parameter categories, writes it to disk
with OsmoseConfigWriter, reads it back with OsmoseConfigReader, and
verifies every key-value pair survives the round-trip unchanged.
"""

import tempfile
from pathlib import Path

from osmose.config.reader import OsmoseConfigReader
from osmose.config.writer import OsmoseConfigWriter

FIXTURES = Path(__file__).parent / "fixtures"


def _full_config() -> dict[str, str]:
    """Return a config dict covering every parameter category."""
    return {
        # ---- simulation ----
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "30",
        "simulation.nspecies": "2",
        "simulation.ncpu": "4",
        "simulation.nsimulation": "1",
        # ---- species 0 ----
        "species.name.sp0": "Anchovy",
        "species.linf.sp0": "19.5",
        "species.k.sp0": "0.364",
        "species.t0.sp0": "-0.7",
        "species.maturity.size.sp0": "12.0",
        "species.sexratio.sp0": "0.5",
        "species.lifespan.sp0": "4",
        "species.lw.condition.factor.sp0": "0.006",
        "species.lw.allpower.sp0": "3.06",
        # ---- species 1 ----
        "species.name.sp1": "Sardine",
        "species.linf.sp1": "23.0",
        "species.k.sp1": "0.28",
        "species.t0.sp1": "-0.9",
        "species.maturity.size.sp1": "15.0",
        "species.sexratio.sp1": "0.5",
        "species.lifespan.sp1": "5",
        "species.lw.condition.factor.sp1": "0.007",
        "species.lw.allpower.sp1": "3.10",
        # ---- mortality rates (master-level) ----
        "mortality.natural.rate.sp0": "0.2",
        "mortality.natural.rate.sp1": "0.18",
        "mortality.starvation.rate.max.sp0": "0.5",
        # ---- fishing mortality (fishing sub-file) ----
        "mortality.fishing.rate.sp0": "0.3",
        "mortality.fishing.rate.sp1": "0.25",
        # ---- seeding biomass (species sub-file) ----
        "population.seeding.biomass.sp0": "10000",
        "population.seeding.biomass.sp1": "15000",
        # ---- grid ----
        "grid.nlon": "30",
        "grid.nlat": "20",
        "grid.upleft.lat": "44.0",
        "grid.upleft.lon": "-2.0",
        "grid.lowright.lat": "43.0",
        "grid.lowright.lon": "-1.0",
        # ---- predation ----
        "predation.accessibility.file": "accessibility.csv",
        "predation.accessibility.stage.structure": "age",
        # ---- fishing ----
        "fisheries.enabled": "true",
        "fisheries.name.fsh0": "Trawl",
        # ---- movement ----
        "movement.distribution.method.sp0": "maps",
        "movement.map0.species": "sp0",
        "movement.map0.file": "maps/sp0_season0.csv",
        # ---- ltl ----
        "ltl.java.classname": "fr.ird.osmose.ltl.LTLFastForcing",
        "ltl.netcdf.file": "ltl_data.nc",
        "ltl.nstep": "365",
        # ---- output ----
        "output.dir.path": "output",
        "output.file.prefix": "osmose_run",
        "output.biomass.enabled": "true",
        "output.abundance.enabled": "true",
        # ---- bioenergetics ----
        "temperature.filename": "temperature.nc",
        "temperature.varname": "temp",
        # ---- economics ----
        "economy.enabled": "false",
    }


def test_full_roundtrip():
    """Every key-value pair from the original config survives write+read."""
    config = _full_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(config, Path(tmpdir))

        reader = OsmoseConfigReader()
        result = reader.read(Path(tmpdir) / "osm_all-parameters.csv")

        for key, value in config.items():
            assert key in result, f"Key missing after roundtrip: {key}"
            assert result[key] == value, (
                f"Value mismatch for '{key}': expected '{value}', got '{result[key]}'"
            )


def test_roundtrip_all_values_are_strings():
    """After read-back every value is a string."""
    config = _full_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(config, Path(tmpdir))

        reader = OsmoseConfigReader()
        result = reader.read(Path(tmpdir) / "osm_all-parameters.csv")

        for key, value in result.items():
            assert isinstance(value, str), (
                f"Value for '{key}' is {type(value).__name__}, expected str"
            )


def test_roundtrip_subfile_count():
    """The correct number of sub-files are created for the full config."""
    config = _full_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(config, Path(tmpdir))

        expected_subfiles = {
            "osm_param-species.csv",
            "osm_param-grid.csv",
            "osm_param-predation.csv",
            "osm_param-fishing.csv",
            "osm_param-movement.csv",
            "osm_param-ltl.csv",
            "osm_param-output.csv",
            "osm_param-bioenergetics.csv",
            "osm_param-economics.csv",
        }

        for fname in expected_subfiles:
            assert (Path(tmpdir) / fname).exists(), f"Expected sub-file missing: {fname}"


def test_roundtrip_master_has_configuration_references():
    """The master file contains osmose.configuration.* keys for each sub-file."""
    config = _full_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(config, Path(tmpdir))

        reader = OsmoseConfigReader()
        master_only = reader.read_file(Path(tmpdir) / "osm_all-parameters.csv")

        config_refs = {
            k: v for k, v in master_only.items() if k.startswith("osmose.configuration.")
        }

        expected_suffixes = {
            "species",
            "grid",
            "predation",
            "fishing",
            "movement",
            "ltl",
            "output",
            "bioenergetics",
            "economics",
        }
        actual_suffixes = {k.split("osmose.configuration.")[1] for k in config_refs}
        assert actual_suffixes == expected_suffixes, (
            f"Config references mismatch: expected {expected_suffixes}, got {actual_suffixes}"
        )


def test_roundtrip_preserves_species_names_with_spaces():
    """A species name containing spaces survives the roundtrip."""
    config = {
        "simulation.nspecies": "1",
        "species.name.sp0": "King Mackerel",
        "species.linf.sp0": "100.0",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(config, Path(tmpdir))

        reader = OsmoseConfigReader()
        result = reader.read(Path(tmpdir) / "osm_all-parameters.csv")

        assert result["species.name.sp0"] == "King Mackerel"


def test_roundtrip_preserves_boolean_values():
    """String booleans 'true' and 'false' survive as-is."""
    config = {
        "simulation.nspecies": "1",
        "fisheries.enabled": "true",
        "economy.enabled": "false",
        "output.biomass.enabled": "true",
        "output.abundance.enabled": "false",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(config, Path(tmpdir))

        reader = OsmoseConfigReader()
        result = reader.read(Path(tmpdir) / "osm_all-parameters.csv")

        assert result["fisheries.enabled"] == "true"
        assert result["economy.enabled"] == "false"
        assert result["output.biomass.enabled"] == "true"
        assert result["output.abundance.enabled"] == "false"


def test_roundtrip_preserves_float_precision():
    """Float string values like '19.5' don't drift to '19.50000001'."""
    config = {
        "simulation.nspecies": "1",
        "species.name.sp0": "Test",
        "species.linf.sp0": "19.5",
        "species.k.sp0": "0.364",
        "species.t0.sp0": "-0.7",
        "grid.upleft.lat": "44.123456789",
        "grid.lowright.lon": "-1.987654321",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(config, Path(tmpdir))

        reader = OsmoseConfigReader()
        result = reader.read(Path(tmpdir) / "osm_all-parameters.csv")

        assert result["species.linf.sp0"] == "19.5"
        assert result["species.k.sp0"] == "0.364"
        assert result["species.t0.sp0"] == "-0.7"
        assert result["grid.upleft.lat"] == "44.123456789"
        assert result["grid.lowright.lon"] == "-1.987654321"


def test_fixture_roundtrip():
    """Read existing test fixtures, write them out, read again, verify equality."""
    reader = OsmoseConfigReader()
    original = reader.read(FIXTURES / "osm_all-parameters.csv")

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(original, Path(tmpdir))

        roundtripped = reader.read(Path(tmpdir) / "osm_all-parameters.csv")

        # Every key from the original (excluding osmose.configuration.* meta-keys
        # which may differ in path) should be present and equal.
        for key, value in original.items():
            if key.startswith("osmose.configuration.") or key.startswith("_osmose."):
                continue
            assert key in roundtripped, f"Key lost after fixture roundtrip: {key}"
            assert roundtripped[key] == value, (
                f"Value mismatch for '{key}': expected '{value}', got '{roundtripped[key]}'"
            )
