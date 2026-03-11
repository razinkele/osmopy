"""End-to-end integration tests for the OSMOSE config pipeline.

Tests the full flow: load example config -> validate against schema -> write
config -> read back -> compare with original. Does NOT require Java.
"""

import tempfile
from pathlib import Path

from osmose.config.reader import OsmoseConfigReader
from osmose.config.writer import OsmoseConfigWriter
from osmose.schema import build_registry
from osmose.schema.base import ParamType

EXAMPLES = Path(__file__).parent.parent / "data" / "examples"


# --------------------------------------------------------------------------
# 1. Load example config
# --------------------------------------------------------------------------


class TestLoadExampleConfig:
    """Verify that the example config files load correctly."""

    def test_example_master_exists(self):
        master = EXAMPLES / "osm_all-parameters.csv"
        assert master.exists(), f"Example master config not found: {master}"

    def test_example_subfiles_exist(self):
        expected = [
            "osm_param-species.csv",
            "osm_param-grid.csv",
            "osm_param-output.csv",
            "osm_param-predation.csv",
            "osm_param-fishing.csv",
            "osm_param-ltl.csv",
            "osm_param-movement.csv",
        ]
        for fname in expected:
            assert (EXAMPLES / fname).exists(), f"Example sub-config missing: {fname}"

    def test_reader_loads_all_keys(self):
        reader = OsmoseConfigReader()
        config = reader.read(EXAMPLES / "osm_all-parameters.csv")
        # Master-level simulation keys
        assert "simulation.time.ndtperyear" in config
        assert "simulation.time.nyear" in config
        assert "simulation.nspecies" in config
        # Species keys from sub-file
        assert "species.name.sp0" in config
        assert "species.linf.sp0" in config
        assert "species.name.sp1" in config
        assert "species.name.sp2" in config
        # Grid keys
        assert "grid.nlon" in config
        assert "grid.nlat" in config
        # Output keys
        assert "output.dir.path" in config
        # Fishing keys
        assert "mortality.fishing.rate.sp0" in config
        # Movement keys
        assert "movement.distribution.method.sp0" in config
        # LTL keys
        assert "ltl.name.rsc0" in config

    def test_reader_loads_correct_values(self):
        reader = OsmoseConfigReader()
        config = reader.read(EXAMPLES / "osm_all-parameters.csv")
        assert config["simulation.nspecies"] == "8"
        assert config["species.name.sp0"] == "Anchovy"
        assert config["species.name.sp1"] == "Sardine"
        assert config["species.name.sp5"] == "Hake"
        assert config["species.name.sp7"] == "BlueWhiting"
        assert config["species.linf.sp0"] == "19.5"
        assert config["species.linf.sp5"] == "110.0"
        assert config["grid.nlon"] == "20"
        assert config["grid.upleft.lat"] == "48.0"
        assert config["mortality.fishing.rate.sp5"] == "0.50"

    def test_reader_loads_expected_key_count(self):
        reader = OsmoseConfigReader()
        config = reader.read(EXAMPLES / "osm_all-parameters.csv")
        # Exclude osmose.configuration.* meta-keys for content count
        content_keys = {
            k: v for k, v in config.items() if not k.startswith("osmose.configuration.")
        }
        # Expect at least 80 non-meta keys from all sub-files combined
        assert len(content_keys) >= 80, (
            f"Expected at least 80 content keys, got {len(content_keys)}"
        )


# --------------------------------------------------------------------------
# 2. Schema validation / matching
# --------------------------------------------------------------------------


class TestSchemaMapping:
    """Verify that example config keys can be matched to schema fields."""

    def test_known_keys_match_schema(self):
        reader = OsmoseConfigReader()
        reader.read(EXAMPLES / "osm_all-parameters.csv")
        registry = build_registry()

        # These keys must be recognized by the schema
        must_match = [
            "simulation.time.ndtperyear",
            "simulation.time.nyear",
            "simulation.nspecies",
            "species.linf.sp0",
            "species.k.sp0",
            "grid.upleft.lat",
            "grid.upleft.lon",
            "output.dir.path",
            "mortality.subdt",
        ]

        for key in must_match:
            field = registry.match_field(key)
            assert field is not None, f"Schema field not found for key: {key}"

    def test_species_indexed_fields_match(self):
        registry = build_registry()
        # All sp0/sp1/sp2 species keys should match an indexed field
        for idx in range(3):
            field = registry.match_field(f"species.linf.sp{idx}")
            assert field is not None, f"No match for species.linf.sp{idx}"
            assert field.indexed is True

    def test_validation_passes_for_numeric_keys(self):
        reader = OsmoseConfigReader()
        config = reader.read(EXAMPLES / "osm_all-parameters.csv")
        registry = build_registry()

        # Build a typed subset for validation (only numeric keys with known schema)
        typed_config = {}
        for key, value in config.items():
            field = registry.match_field(key)
            if field is not None:
                if field.param_type == ParamType.INT:
                    try:
                        typed_config[key] = int(value)
                    except ValueError:
                        pass
                elif field.param_type == ParamType.FLOAT:
                    try:
                        typed_config[key] = float(value)
                    except ValueError:
                        pass

        errors = registry.validate(typed_config)
        assert errors == [], f"Validation errors for example config: {errors}"

    def test_registry_covers_all_categories(self):
        registry = build_registry()
        categories = registry.categories()
        expected = {"simulation", "growth", "grid", "output"}
        # At minimum these categories should be present
        for cat in expected:
            assert cat in categories, f"Missing category: {cat}"


# --------------------------------------------------------------------------
# 3. Write config -> read back -> compare (full roundtrip)
# --------------------------------------------------------------------------


class TestFullRoundtrip:
    """Load example, write to new dir, read back, compare."""

    def test_roundtrip_preserves_all_content_keys(self):
        reader = OsmoseConfigReader()
        original = reader.read(EXAMPLES / "osm_all-parameters.csv")

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = OsmoseConfigWriter()
            # Strip osmose.configuration.* meta-keys before writing
            # (the writer generates its own references)
            content = {
                k: v for k, v in original.items() if not k.startswith("osmose.configuration.")
            }
            writer.write(content, Path(tmpdir))

            roundtripped = reader.read(Path(tmpdir) / "osm_all-parameters.csv")

            for key, value in content.items():
                assert key in roundtripped, f"Key lost after roundtrip: {key}"
                assert roundtripped[key] == value, (
                    f"Value mismatch for '{key}': expected '{value}', got '{roundtripped[key]}'"
                )

    def test_roundtrip_generates_correct_subfiles(self):
        reader = OsmoseConfigReader()
        original = reader.read(EXAMPLES / "osm_all-parameters.csv")

        with tempfile.TemporaryDirectory() as tmpdir:
            content = {
                k: v for k, v in original.items() if not k.startswith("osmose.configuration.")
            }
            writer = OsmoseConfigWriter()
            writer.write(content, Path(tmpdir))

            # Check that expected sub-files were created
            expected_subfiles = [
                "osm_param-species.csv",
                "osm_param-grid.csv",
                "osm_param-output.csv",
                "osm_param-predation.csv",
                "osm_param-fishing.csv",
                "osm_param-ltl.csv",
                "osm_param-movement.csv",
            ]
            for fname in expected_subfiles:
                assert (Path(tmpdir) / fname).exists(), f"Expected sub-file not generated: {fname}"

    def test_roundtrip_master_has_references(self):
        reader = OsmoseConfigReader()
        original = reader.read(EXAMPLES / "osm_all-parameters.csv")

        with tempfile.TemporaryDirectory() as tmpdir:
            content = {
                k: v for k, v in original.items() if not k.startswith("osmose.configuration.")
            }
            writer = OsmoseConfigWriter()
            writer.write(content, Path(tmpdir))

            master = reader.read_file(Path(tmpdir) / "osm_all-parameters.csv")
            refs = {k for k in master if k.startswith("osmose.configuration.")}

            expected_refs = {
                "osmose.configuration.species",
                "osmose.configuration.grid",
                "osmose.configuration.output",
                "osmose.configuration.predation",
                "osmose.configuration.fishing",
                "osmose.configuration.ltl",
                "osmose.configuration.movement",
            }
            assert refs == expected_refs, (
                f"Master references mismatch: expected {expected_refs}, got {refs}"
            )

    def test_roundtrip_preserves_species_data(self):
        reader = OsmoseConfigReader()
        original = reader.read(EXAMPLES / "osm_all-parameters.csv")

        with tempfile.TemporaryDirectory() as tmpdir:
            content = {
                k: v for k, v in original.items() if not k.startswith("osmose.configuration.")
            }
            writer = OsmoseConfigWriter()
            writer.write(content, Path(tmpdir))

            result = reader.read(Path(tmpdir) / "osm_all-parameters.csv")

            # All 8 species should survive
            assert result["species.name.sp0"] == "Anchovy"
            assert result["species.name.sp1"] == "Sardine"
            assert result["species.name.sp5"] == "Hake"
            assert result["species.name.sp7"] == "BlueWhiting"
            assert result["species.linf.sp0"] == "19.5"
            assert result["species.linf.sp1"] == "23.0"
            assert result["species.linf.sp5"] == "110.0"
            assert result["species.k.sp0"] == "0.364"
            assert result["species.k.sp1"] == "0.280"
            assert result["species.k.sp5"] == "0.106"

    def test_roundtrip_preserves_grid_data(self):
        reader = OsmoseConfigReader()
        original = reader.read(EXAMPLES / "osm_all-parameters.csv")

        with tempfile.TemporaryDirectory() as tmpdir:
            content = {
                k: v for k, v in original.items() if not k.startswith("osmose.configuration.")
            }
            writer = OsmoseConfigWriter()
            writer.write(content, Path(tmpdir))

            result = reader.read(Path(tmpdir) / "osm_all-parameters.csv")

            assert result["grid.nlon"] == "20"
            assert result["grid.nlat"] == "20"
            assert result["grid.upleft.lat"] == "48.0"
            assert result["grid.lowright.lon"] == "-1.0"

    def test_roundtrip_key_count_preserved(self):
        reader = OsmoseConfigReader()
        original = reader.read(EXAMPLES / "osm_all-parameters.csv")

        with tempfile.TemporaryDirectory() as tmpdir:
            content = {
                k: v for k, v in original.items() if not k.startswith("osmose.configuration.")
            }
            writer = OsmoseConfigWriter()
            writer.write(content, Path(tmpdir))

            result = reader.read(Path(tmpdir) / "osm_all-parameters.csv")
            result_content = {
                k: v for k, v in result.items() if not k.startswith("osmose.configuration.")
            }

            assert len(result_content) == len(content), (
                f"Key count changed: original={len(content)}, roundtripped={len(result_content)}"
            )


# --------------------------------------------------------------------------
# 4. Cross-validation: schema + config coherence
# --------------------------------------------------------------------------


class TestSchemaConfigCoherence:
    """Ensure schema and example config are consistent with each other."""

    def test_simulation_nspecies_matches_species_count(self):
        reader = OsmoseConfigReader()
        config = reader.read(EXAMPLES / "osm_all-parameters.csv")

        nspecies = int(config["simulation.nspecies"])
        # Count focal species (sp0..sp{n-1}); resources also have species.name keys
        focal_names = [f"species.name.sp{i}" for i in range(nspecies)]
        for key in focal_names:
            assert key in config, f"Missing focal species key: {key}"

    def test_all_species_have_required_params(self):
        reader = OsmoseConfigReader()
        config = reader.read(EXAMPLES / "osm_all-parameters.csv")

        nspecies = int(config["simulation.nspecies"])
        required_prefixes = [
            "species.name",
            "species.linf",
            "species.k",
            "species.lifespan",
        ]
        for i in range(nspecies):
            for prefix in required_prefixes:
                key = f"{prefix}.sp{i}"
                assert key in config, f"Required species param missing: {key}"

    def test_grid_coordinates_are_valid(self):
        reader = OsmoseConfigReader()
        config = reader.read(EXAMPLES / "osm_all-parameters.csv")

        lat_up = float(config["grid.upleft.lat"])
        lat_low = float(config["grid.lowright.lat"])
        lon_left = float(config["grid.upleft.lon"])
        lon_right = float(config["grid.lowright.lon"])

        assert lat_up > lat_low, "Upper-left lat must be > lower-right lat"
        assert lon_right > lon_left, "Lower-right lon must be > upper-left lon"
        assert -90 <= lat_low <= 90
        assert -90 <= lat_up <= 90
        assert -180 <= lon_left <= 180
        assert -180 <= lon_right <= 180

    def test_output_dir_is_set(self):
        reader = OsmoseConfigReader()
        config = reader.read(EXAMPLES / "osm_all-parameters.csv")
        assert config.get("output.dir.path"), "output.dir.path must be set"

    def test_fishing_rates_are_plausible(self):
        reader = OsmoseConfigReader()
        config = reader.read(EXAMPLES / "osm_all-parameters.csv")

        nspecies = int(config["simulation.nspecies"])
        for i in range(nspecies):
            key = f"mortality.fishing.rate.sp{i}"
            if key in config:
                rate = float(config[key])
                assert 0 <= rate <= 2.0, f"Implausible fishing rate for sp{i}: {rate}"
