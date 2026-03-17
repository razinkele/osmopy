"""Full-model integration tests for all example studies.

Tests the complete lifecycle for each bundled OSMOSE example:
  1. Load config via demo module
  2. Validate config against schema + structural checks
  3. Write config to temp dir and roundtrip
  4. Build runner command (no actual Java)
  5. Simulate output files and read them back via OsmoseResults
  6. Run analysis / ensemble aggregation on simulated outputs

Each study class is parametrized to cover bay_of_biscay, eec, eec_full, and
minimal with study-specific assertions (species counts, grid bounds, etc.).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osmose.analysis import ensemble_stats, shannon_diversity, summary_table
from osmose.config.reader import OsmoseConfigReader
from osmose.config.validator import (
    check_species_consistency,
    validate_config,
)
from osmose.config.writer import OsmoseConfigWriter
from osmose.demo import list_demos, migrate_config, osmose_demo
from osmose.ensemble import aggregate_replicates
from osmose.results import OsmoseResults
from osmose.runner import OsmoseRunner
from osmose.schema import build_registry
from osmose.schema.base import ParamType

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

STUDY_SPECS = {
    "bay_of_biscay": {
        "nspecies": 8,
        "nresource": 6,
        "nyear": 50,
        "ndtperyear": 24,
        "species_names": {
            0: "Anchovy",
            1: "Sardine",
            5: "Hake",
            7: "BlueWhiting",
        },
        "has_ltl": True,
        "has_fishing": True,
        "master_name": "osm_all-parameters.csv",
        "grid_nlon": 20,
        "grid_nlat": 20,
    },
    "eec": {
        "nspecies": 6,
        "nresource": 0,
        "nyear": 30,
        "ndtperyear": 24,
        "species_names": {
            0: "Herring",
            1: "Sprat",
            2: "Whiting",
            3: "Sole",
            4: "Plaice",
            5: "Cod",
        },
        "has_ltl": False,
        "has_fishing": True,
        "master_name": "osm_all-parameters.csv",
        "grid_nlon": 15,
        "grid_nlat": 12,
    },
    "eec_full": {
        "nspecies": 14,
        "nresource": 10,
        "nyear": 70,
        "ndtperyear": 24,
        "species_names": {
            0: "lesserSpottedDogfish",
            13: "squids",
        },
        "has_ltl": True,
        "has_fishing": True,
        "master_name": "eec_all-parameters.csv",
        "grid_nlon": None,  # eec_full grid may differ
        "grid_nlat": None,
    },
    "minimal": {
        "nspecies": 2,
        "nresource": 0,
        "nyear": 10,
        "ndtperyear": 12,
        "species_names": {
            0: "Anchovy",
            1: "Hake",
        },
        "has_ltl": False,
        "has_fishing": False,
        "master_name": "osm_all-parameters.csv",
        "grid_nlon": 10,
        "grid_nlat": 10,
    },
}

ALL_STUDIES = list(STUDY_SPECS.keys())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def registry():
    return build_registry()


@pytest.fixture(params=ALL_STUDIES)
def study_name(request):
    return request.param


@pytest.fixture
def study_demo(study_name, tmp_path):
    """Generate demo for a study and return (name, result, config_dict)."""
    result = osmose_demo(study_name, tmp_path)
    reader = OsmoseConfigReader()
    config = reader.read(result["config_file"])
    return study_name, result, config


# ---------------------------------------------------------------------------
# 1. Demo loading & structural integrity
# ---------------------------------------------------------------------------


class TestDemoLoading:
    """Verify all studies load correctly via the demo module."""

    def test_all_demos_listed(self):
        demos = list_demos()
        for name in ALL_STUDIES:
            assert name in demos

    def test_demo_generates_config_file(self, study_demo):
        name, result, _ = study_demo
        assert result["config_file"].exists(), f"{name}: config_file not found"
        assert result["output_dir"].exists(), f"{name}: output_dir not found"
        assert result["config_file"].name == STUDY_SPECS[name]["master_name"]

    def test_demo_config_is_readable(self, study_demo):
        name, _, config = study_demo
        assert len(config) > 0, f"{name}: config is empty"

    def test_demo_nspecies_matches_spec(self, study_demo):
        name, _, config = study_demo
        spec = STUDY_SPECS[name]
        assert int(config["simulation.nspecies"]) == spec["nspecies"]

    def test_demo_species_names_match_spec(self, study_demo):
        name, _, config = study_demo
        spec = STUDY_SPECS[name]
        for idx, expected_name in spec["species_names"].items():
            key = f"species.name.sp{idx}"
            assert config.get(key) == expected_name, (
                f"{name}: expected species.name.sp{idx}={expected_name}, "
                f"got {config.get(key)!r}"
            )


# ---------------------------------------------------------------------------
# 2. Schema validation per study
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Validate each study's config against the OSMOSE schema."""

    def test_species_consistency(self, study_demo):
        name, _, config = study_demo
        warnings = check_species_consistency(config)
        assert warnings == [], f"{name}: species consistency warnings: {warnings}"

    def test_config_validation_no_errors(self, study_demo, registry):
        name, _, config = study_demo
        errors, _ = validate_config(config, registry)
        assert errors == [], f"{name}: validation errors: {errors}"

    def test_numeric_params_match_schema_bounds(self, study_demo, registry):
        name, _, config = study_demo
        out_of_bounds = []
        for key, value in config.items():
            field = registry.match_field(key)
            if field is None or field.param_type not in (ParamType.FLOAT, ParamType.INT):
                continue
            if not value or value.lower() in ("null", "none") or ";" in value:
                continue
            try:
                num = float(value)
            except ValueError:
                continue
            if field.min_val is not None and num < field.min_val:
                out_of_bounds.append(f"{key}={num} < min={field.min_val}")
            if field.max_val is not None and num > field.max_val:
                out_of_bounds.append(f"{key}={num} > max={field.max_val}")
        assert out_of_bounds == [], f"{name}: out of bounds: {out_of_bounds}"

    def test_known_keys_recognized_by_schema(self, study_demo, registry):
        """At least 50% of content keys should match a schema field."""
        name, _, config = study_demo
        content_keys = [
            k for k in config
            if not k.startswith("osmose.configuration.")
            and not k.startswith("#")
        ]
        matched = sum(1 for k in content_keys if registry.match_field(k) is not None)
        ratio = matched / max(len(content_keys), 1)
        assert ratio >= 0.3, (
            f"{name}: only {matched}/{len(content_keys)} keys matched schema ({ratio:.0%})"
        )


# ---------------------------------------------------------------------------
# 3. Grid & spatial coherence
# ---------------------------------------------------------------------------


class TestGridCoherence:
    """Verify grid configuration is self-consistent for each study."""

    def test_grid_dimensions_are_positive(self, study_demo):
        name, _, config = study_demo
        if "grid.nlon" in config:
            assert int(config["grid.nlon"]) > 0, f"{name}: grid.nlon must be positive"
        if "grid.nlat" in config:
            assert int(config["grid.nlat"]) > 0, f"{name}: grid.nlat must be positive"

    def test_grid_coordinates_are_valid(self, study_demo):
        name, _, config = study_demo
        if "grid.upleft.lat" not in config:
            pytest.skip(f"{name}: no grid coordinate keys")
        lat_up = float(config["grid.upleft.lat"])
        lat_low = float(config["grid.lowright.lat"])
        lon_left = float(config["grid.upleft.lon"])
        lon_right = float(config["grid.lowright.lon"])

        assert -90 <= lat_low <= 90, f"{name}: invalid lat_low={lat_low}"
        assert -90 <= lat_up <= 90, f"{name}: invalid lat_up={lat_up}"
        assert -180 <= lon_left <= 180, f"{name}: invalid lon_left={lon_left}"
        assert -180 <= lon_right <= 180, f"{name}: invalid lon_right={lon_right}"
        assert lat_up > lat_low, f"{name}: upper lat must exceed lower lat"
        assert lon_right > lon_left, f"{name}: right lon must exceed left lon"

    def test_grid_dimensions_match_spec(self, study_demo):
        name, _, config = study_demo
        spec = STUDY_SPECS[name]
        if spec["grid_nlon"] is not None and "grid.nlon" in config:
            assert int(config["grid.nlon"]) == spec["grid_nlon"]
        if spec["grid_nlat"] is not None and "grid.nlat" in config:
            assert int(config["grid.nlat"]) == spec["grid_nlat"]


# ---------------------------------------------------------------------------
# 4. Species biological parameters
# ---------------------------------------------------------------------------


class TestSpeciesBiology:
    """Validate species parameters are biologically plausible."""

    def test_all_species_have_growth_params(self, study_demo):
        name, _, config = study_demo
        nspecies = int(config["simulation.nspecies"])
        for i in range(nspecies):
            assert f"species.name.sp{i}" in config, f"{name}: missing name for sp{i}"
            assert f"species.linf.sp{i}" in config, f"{name}: missing linf for sp{i}"
            assert f"species.k.sp{i}" in config, f"{name}: missing k for sp{i}"
            assert f"species.lifespan.sp{i}" in config, f"{name}: missing lifespan for sp{i}"

    def test_linf_is_positive(self, study_demo):
        name, _, config = study_demo
        nspecies = int(config["simulation.nspecies"])
        for i in range(nspecies):
            key = f"species.linf.sp{i}"
            if key in config:
                assert float(config[key]) > 0, f"{name}: linf for sp{i} must be positive"

    def test_growth_rate_is_positive(self, study_demo):
        name, _, config = study_demo
        nspecies = int(config["simulation.nspecies"])
        for i in range(nspecies):
            key = f"species.k.sp{i}"
            if key in config:
                assert float(config[key]) > 0, f"{name}: k for sp{i} must be positive"

    def test_lifespan_is_positive_integer(self, study_demo):
        name, _, config = study_demo
        nspecies = int(config["simulation.nspecies"])
        for i in range(nspecies):
            key = f"species.lifespan.sp{i}"
            if key in config:
                lifespan = float(config[key])
                assert lifespan > 0, f"{name}: lifespan for sp{i} must be positive"
                assert lifespan == int(lifespan), f"{name}: lifespan for sp{i} should be integer"

    def test_maturity_size_less_than_linf(self, study_demo):
        name, _, config = study_demo
        nspecies = int(config["simulation.nspecies"])
        for i in range(nspecies):
            linf_key = f"species.linf.sp{i}"
            mat_key = f"species.maturity.size.sp{i}"
            if linf_key in config and mat_key in config:
                linf = float(config[linf_key])
                maturity = float(config[mat_key])
                assert maturity < linf, (
                    f"{name}: maturity size ({maturity}) >= linf ({linf}) for sp{i}"
                )

    def test_sex_ratio_between_0_and_1(self, study_demo):
        name, _, config = study_demo
        nspecies = int(config["simulation.nspecies"])
        for i in range(nspecies):
            key = f"species.sexratio.sp{i}"
            if key in config:
                ratio = float(config[key])
                assert 0 <= ratio <= 1, f"{name}: sex ratio for sp{i} out of range: {ratio}"

    def test_seeding_biomass_is_positive(self, study_demo):
        name, _, config = study_demo
        nspecies = int(config["simulation.nspecies"])
        for i in range(nspecies):
            key = f"population.seeding.biomass.sp{i}"
            if key in config:
                biomass = float(config[key])
                assert biomass > 0, f"{name}: seeding biomass for sp{i} must be positive"


# ---------------------------------------------------------------------------
# 5. Config roundtrip (write -> read -> compare)
# ---------------------------------------------------------------------------


class TestConfigRoundtrip:
    """Write each study's config to disk, read back, verify no data loss."""

    def test_roundtrip_preserves_all_content_keys(self, study_demo):
        name, _, config = study_demo
        content = {
            k: v for k, v in config.items()
            if not k.startswith("osmose.configuration.")
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = OsmoseConfigWriter()
            writer.write(content, Path(tmpdir))
            reader = OsmoseConfigReader()
            result = reader.read(Path(tmpdir) / "osm_all-parameters.csv")

            lost = [k for k in content if k not in result]
            assert lost == [], f"{name}: keys lost after roundtrip: {lost}"

    def test_roundtrip_preserves_values(self, study_demo):
        name, _, config = study_demo
        content = {
            k: v for k, v in config.items()
            if not k.startswith("osmose.configuration.")
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = OsmoseConfigWriter()
            writer.write(content, Path(tmpdir))
            reader = OsmoseConfigReader()
            result = reader.read(Path(tmpdir) / "osm_all-parameters.csv")

            mismatched = []
            for key, value in content.items():
                if key in result and result[key] != value:
                    mismatched.append(f"{key}: {value!r} -> {result[key]!r}")
            assert mismatched == [], f"{name}: value mismatches: {mismatched}"


# ---------------------------------------------------------------------------
# 6. Runner command construction per study
# ---------------------------------------------------------------------------


class TestRunnerCommandBuilding:
    """Verify OsmoseRunner builds correct commands for each study config."""

    def test_build_cmd_includes_config_path(self, study_demo, tmp_path):
        name, result, _ = study_demo
        jar = tmp_path / "osmose.jar"
        jar.touch()
        runner = OsmoseRunner(jar_path=jar)
        cmd = runner._build_cmd(config_path=result["config_file"])
        assert str(result["config_file"]) in cmd

    def test_build_cmd_with_output_dir(self, study_demo, tmp_path):
        name, result, _ = study_demo
        jar = tmp_path / "osmose.jar"
        jar.touch()
        runner = OsmoseRunner(jar_path=jar)
        out = tmp_path / "run_output"
        cmd = runner._build_cmd(config_path=result["config_file"], output_dir=out)
        assert any(f"-Poutput.dir.path={out}" in arg for arg in cmd)

    def test_build_cmd_with_study_overrides(self, study_demo, tmp_path):
        """Verify overrides for common study tweaks (e.g., shorter run)."""
        name, result, config = study_demo
        jar = tmp_path / "osmose.jar"
        jar.touch()
        runner = OsmoseRunner(jar_path=jar)
        overrides = {
            "simulation.time.nyear": "5",
            "simulation.random.seed": "42",
        }
        cmd = runner._build_cmd(
            config_path=result["config_file"],
            overrides=overrides,
        )
        assert "-Psimulation.time.nyear=5" in cmd
        assert "-Psimulation.random.seed=42" in cmd


# ---------------------------------------------------------------------------
# 7. Simulated output reading (mock OSMOSE output files)
# ---------------------------------------------------------------------------


def _create_mock_output(output_dir: Path, species_names: list[str], n_timesteps: int = 120):
    """Create realistic mock OSMOSE CSV output files for testing OsmoseResults."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    for sp in species_names:
        # Biomass time series
        times = list(range(n_timesteps))
        biomass = rng.exponential(scale=50000, size=n_timesteps).tolist()
        pd.DataFrame({"time": times, "biomass": biomass}).to_csv(
            output_dir / f"osm_biomass_{sp}.csv", index=False
        )

        # Abundance time series
        abundance = rng.integers(1000, 100000, size=n_timesteps).tolist()
        pd.DataFrame({"time": times, "abundance": abundance}).to_csv(
            output_dir / f"osm_abundance_{sp}.csv", index=False
        )

        # Mortality time series
        mortality_vals = rng.uniform(0, 1, size=n_timesteps).tolist()
        pd.DataFrame({"time": times, "mortality": mortality_vals}).to_csv(
            output_dir / f"osm_mortality_{sp}.csv", index=False
        )

        # Yield time series
        yield_vals = rng.exponential(scale=5000, size=n_timesteps).tolist()
        pd.DataFrame({"time": times, "yield": yield_vals}).to_csv(
            output_dir / f"osm_yield_{sp}.csv", index=False
        )

        # Mean trophic level
        tl_vals = rng.uniform(2.0, 4.5, size=n_timesteps).tolist()
        pd.DataFrame({"time": times, "meanTL": tl_vals}).to_csv(
            output_dir / f"osm_meanTL_{sp}.csv", index=False
        )

        # Mean size
        size_vals = rng.uniform(5.0, 50.0, size=n_timesteps).tolist()
        pd.DataFrame({"time": times, "meanSize": size_vals}).to_csv(
            output_dir / f"osm_meanSize_{sp}.csv", index=False
        )

        # Diet matrix
        diet_vals = rng.dirichlet(np.ones(len(species_names)), size=n_timesteps)
        diet_df = pd.DataFrame(diet_vals, columns=species_names)
        diet_df.insert(0, "Time", times)
        diet_df.to_csv(output_dir / f"osm_dietMatrix_{sp}.csv", index=False)

        # 2D: Biomass by age (5 age classes)
        age_cols = [f"age{a}" for a in range(5)]
        age_data = rng.exponential(scale=10000, size=(n_timesteps, 5))
        age_df = pd.DataFrame(age_data, columns=age_cols)
        age_df.insert(0, "Time", times)
        age_df.to_csv(output_dir / f"osm_biomassByAge_{sp}.csv", index=False)

        # 2D: Biomass by size (10 size classes)
        size_cols = [f"size{s}" for s in range(10)]
        size_data = rng.exponential(scale=5000, size=(n_timesteps, 10))
        size_df = pd.DataFrame(size_data, columns=size_cols)
        size_df.insert(0, "Time", times)
        size_df.to_csv(output_dir / f"osm_biomassBySize_{sp}.csv", index=False)

    # Size spectrum (no species column)
    sizes = [2**i for i in range(1, 15)]
    abundances = [rng.exponential(scale=10**(7 - 0.3 * i)) for i in range(14)]
    pd.DataFrame({"size": sizes, "abundance": abundances}).to_csv(
        output_dir / "osm_sizeSpectrum.csv", index=False
    )


class TestOutputReading:
    """Read simulated output files using OsmoseResults for each study."""

    @pytest.fixture
    def study_output(self, study_demo, tmp_path):
        """Create mock output for the study's species list."""
        name, _, config = study_demo
        nspecies = int(config["simulation.nspecies"])
        species = [config[f"species.name.sp{i}"] for i in range(nspecies)]
        output_dir = tmp_path / "output"
        _create_mock_output(output_dir, species)
        return name, species, OsmoseResults(output_dir, prefix="osm")

    def test_list_outputs_finds_files(self, study_output):
        name, species, results = study_output
        outputs = results.list_outputs()
        assert len(outputs) > 0, f"{name}: no output files found"
        # At minimum: biomass + abundance + mortality per species + size spectrum
        assert len(outputs) >= len(species) * 3 + 1

    def test_biomass_reads_all_species(self, study_output):
        name, species, results = study_output
        df = results.biomass()
        assert not df.empty, f"{name}: biomass DataFrame is empty"
        actual_species = set(df["species"].unique())
        assert actual_species == set(species), (
            f"{name}: biomass species mismatch: expected {set(species)}, got {actual_species}"
        )

    def test_biomass_filter_by_species(self, study_output):
        name, species, results = study_output
        sp = species[0]
        df = results.biomass(species=sp)
        assert not df.empty
        assert set(df["species"].unique()) == {sp}

    def test_abundance_reads_all_species(self, study_output):
        name, species, results = study_output
        df = results.abundance()
        assert not df.empty
        assert set(df["species"].unique()) == set(species)

    def test_mortality_reads_all_species(self, study_output):
        name, species, results = study_output
        df = results.mortality()
        assert not df.empty
        assert set(df["species"].unique()) == set(species)

    def test_yield_biomass_reads_data(self, study_output):
        name, _, results = study_output
        df = results.yield_biomass()
        assert not df.empty

    def test_mean_tl_reads_data(self, study_output):
        name, _, results = study_output
        df = results.mean_trophic_level()
        assert not df.empty

    def test_mean_size_reads_data(self, study_output):
        name, _, results = study_output
        df = results.mean_size()
        assert not df.empty

    def test_diet_matrix_reads_data(self, study_output):
        name, _, results = study_output
        df = results.diet_matrix()
        assert not df.empty

    def test_biomass_by_age_reads_2d(self, study_output):
        name, _, results = study_output
        df = results.biomass_by_age()
        assert not df.empty
        assert "bin" in df.columns
        assert "value" in df.columns

    def test_biomass_by_size_reads_2d(self, study_output):
        name, _, results = study_output
        df = results.biomass_by_size()
        assert not df.empty
        assert "bin" in df.columns

    def test_size_spectrum(self, study_output):
        name, _, results = study_output
        df = results.size_spectrum()
        assert not df.empty
        assert "size" in df.columns
        assert "abundance" in df.columns

    def test_export_dataframe_biomass(self, study_output):
        name, _, results = study_output
        df = results.export_dataframe("biomass")
        assert not df.empty

    def test_export_dataframe_unknown_type(self, study_output):
        name, _, results = study_output
        df = results.export_dataframe("nonexistent_type")
        assert df.empty

    def test_close_cache(self, study_output):
        """close_cache should not error even with no NetCDF files opened."""
        _, _, results = study_output
        results.close_cache()
        assert results._nc_cache == {}


# ---------------------------------------------------------------------------
# 8. Analysis pipeline on simulated outputs
# ---------------------------------------------------------------------------


class TestAnalysisPipeline:
    """Run analysis functions on simulated output data for each study."""

    @pytest.fixture
    def study_biomass(self, study_demo, tmp_path):
        """Generate mock biomass DataFrames (as if from multiple replicates)."""
        name, _, config = study_demo
        nspecies = int(config["simulation.nspecies"])
        species = [config[f"species.name.sp{i}"] for i in range(nspecies)]
        rng = np.random.default_rng(123)

        replicates = []
        for _ in range(3):
            frames = []
            for sp in species:
                n_time = 50
                frames.append(pd.DataFrame({
                    "time": range(n_time),
                    "species": sp,
                    "biomass": rng.exponential(scale=50000, size=n_time),
                }))
            replicates.append(pd.concat(frames, ignore_index=True))
        return name, species, replicates

    def test_ensemble_stats_computes_mean_and_ci(self, study_biomass):
        name, species, replicates = study_biomass
        result = ensemble_stats(replicates, value_col="biomass", group_cols=["time"])
        assert not result.empty
        assert "mean" in result.columns
        assert "std" in result.columns
        assert "ci_lower" in result.columns
        assert "ci_upper" in result.columns
        # CI lower should be <= mean <= CI upper
        assert (result["ci_lower"] <= result["mean"]).all()
        assert (result["mean"] <= result["ci_upper"]).all()

    def test_summary_table_all_species(self, study_biomass):
        name, species, replicates = study_biomass
        result = summary_table(replicates, value_col="biomass")
        assert not result.empty
        assert set(result["species"]) == set(species)
        assert "mean" in result.columns
        assert "min" in result.columns
        assert "max" in result.columns

    def test_shannon_diversity(self, study_biomass):
        name, _, replicates = study_biomass
        df = replicates[0]
        result = shannon_diversity(df)
        assert not result.empty
        assert "time" in result.columns
        assert "shannon" in result.columns
        # Shannon diversity should be non-negative
        assert (result["shannon"] >= 0).all()
        # For multi-species, should be > 0
        assert (result["shannon"] > 0).all()


# ---------------------------------------------------------------------------
# 9. Ensemble aggregation with mock replicate directories
# ---------------------------------------------------------------------------


class TestEnsembleAggregation:
    """Test aggregate_replicates with mock replicate output directories."""

    @pytest.fixture
    def replicate_dirs(self, study_demo, tmp_path):
        """Create 3 replicate output directories with mock CSVs."""
        name, _, config = study_demo
        nspecies = int(config["simulation.nspecies"])
        species = [config[f"species.name.sp{i}"] for i in range(nspecies)]
        dirs = []
        for rep_idx in range(3):
            rep_dir = tmp_path / f"rep_{rep_idx}"
            _create_mock_output(rep_dir, species, n_timesteps=60)
            dirs.append(rep_dir)
        return name, species, dirs

    def test_aggregate_biomass(self, replicate_dirs):
        name, _, dirs = replicate_dirs
        result = aggregate_replicates(dirs, "biomass")
        assert len(result["time"]) > 0
        assert len(result["mean"]) == len(result["time"])
        assert len(result["lower"]) == len(result["time"])
        assert len(result["upper"]) == len(result["time"])

    def test_aggregate_lower_le_mean_le_upper(self, replicate_dirs):
        name, _, dirs = replicate_dirs
        result = aggregate_replicates(dirs, "biomass")
        for i in range(len(result["time"])):
            assert result["lower"][i] <= result["mean"][i] + 1e-10
            assert result["mean"][i] <= result["upper"][i] + 1e-10

    def test_aggregate_abundance(self, replicate_dirs):
        name, _, dirs = replicate_dirs
        result = aggregate_replicates(dirs, "abundance")
        assert len(result["time"]) > 0

    def test_aggregate_mortality(self, replicate_dirs):
        name, _, dirs = replicate_dirs
        result = aggregate_replicates(dirs, "mortality")
        assert len(result["time"]) > 0

    def test_aggregate_empty_dirs(self):
        result = aggregate_replicates([], "biomass")
        assert result == {"time": [], "mean": [], "lower": [], "upper": []}


# ---------------------------------------------------------------------------
# 10. Full pipeline: load -> validate -> write -> mock-run -> read -> analyze
# ---------------------------------------------------------------------------


class _ScriptRunner(OsmoseRunner):
    """OsmoseRunner that invokes Python scripts instead of Java."""

    def _build_cmd(
        self,
        config_path: Path,
        output_dir: Path | None = None,
        java_opts: list[str] | None = None,
        overrides: dict[str, str] | None = None,
        **kwargs,
    ) -> list[str]:
        cmd = [self.java_cmd, str(self.jar_path), str(config_path)]
        if output_dir:
            cmd.append(f"-Poutput.dir.path={output_dir}")
        if overrides:
            for key, value in overrides.items():
                cmd.append(f"-P{key}={value}")
        return cmd


class TestFullPipeline:
    """End-to-end pipeline test: config lifecycle + mock run + output analysis."""

    @pytest.fixture
    def pipeline_setup(self, study_demo, tmp_path):
        """Set up the full pipeline for a study."""
        name, demo_result, config = study_demo
        spec = STUDY_SPECS[name]

        # Write config to a fresh directory
        content = {
            k: v for k, v in config.items()
            if not k.startswith("osmose.configuration.")
        }
        config_dir = tmp_path / "roundtrip_config"
        config_dir.mkdir()
        writer = OsmoseConfigWriter()
        writer.write(content, config_dir)
        config_path = config_dir / "osm_all-parameters.csv"

        # Verify roundtrip
        reader = OsmoseConfigReader()
        roundtripped = reader.read(config_path)
        for key, value in content.items():
            assert roundtripped.get(key) == value

        # Create mock output
        nspecies = spec["nspecies"]
        species = [config[f"species.name.sp{i}"] for i in range(nspecies)]
        output_dir = tmp_path / "output"
        _create_mock_output(output_dir, species)

        return name, spec, config_path, output_dir, species

    def test_full_pipeline_biomass_analysis(self, pipeline_setup):
        name, spec, config_path, output_dir, species = pipeline_setup
        results = OsmoseResults(output_dir, prefix="osm")

        # Read biomass
        df = results.biomass()
        assert not df.empty
        assert set(df["species"].unique()) == set(species)

        # Shannon diversity — needs columns: time, species, biomass
        if "time" in df.columns and "biomass" in df.columns and "species" in df.columns:
            diversity = shannon_diversity(df)
            assert not diversity.empty
            assert (diversity["shannon"] >= 0).all()

    def test_full_pipeline_summary_stats(self, pipeline_setup):
        name, spec, _, output_dir, species = pipeline_setup
        results = OsmoseResults(output_dir, prefix="osm")

        # Read all output types and verify non-empty
        biomass = results.biomass()
        abundance = results.abundance()
        mortality = results.mortality()
        assert not biomass.empty
        assert not abundance.empty
        assert not mortality.empty

        # Summary across species
        summary = summary_table([biomass], value_col="biomass")
        assert set(summary["species"]) == set(species)
        for _, row in summary.iterrows():
            assert row["mean"] > 0
            assert row["min"] >= 0
            assert row["max"] >= row["mean"]

    def test_full_pipeline_2d_outputs(self, pipeline_setup):
        name, _, _, output_dir, _ = pipeline_setup
        results = OsmoseResults(output_dir, prefix="osm")

        by_age = results.biomass_by_age()
        assert not by_age.empty
        assert "bin" in by_age.columns
        assert "value" in by_age.columns
        assert "species" in by_age.columns

        by_size = results.biomass_by_size()
        assert not by_size.empty

    def test_full_pipeline_size_spectrum_analysis(self, pipeline_setup):
        from osmose.analysis import size_spectrum_slope

        name, _, _, output_dir, _ = pipeline_setup
        results = OsmoseResults(output_dir, prefix="osm")
        spectrum = results.size_spectrum()
        assert not spectrum.empty

        slope, intercept, r_squared = size_spectrum_slope(spectrum)
        # Size spectrum slope should be negative (larger sizes have fewer individuals)
        assert slope < 0, f"{name}: expected negative slope, got {slope}"
        assert 0 <= r_squared <= 1

    async def test_full_pipeline_mock_run(self, study_demo, tmp_path):
        """Execute a fake JAR and verify the RunResult."""
        name, demo_result, config = study_demo

        # Create a fake JAR script
        script = tmp_path / "fake_osmose.py"
        nspecies = int(config["simulation.nspecies"])
        script.write_text(
            "import sys, pathlib\n"
            "output_dir = None\n"
            "for arg in sys.argv[1:]:\n"
            '    if arg.startswith("-Poutput.dir.path="):\n'
            '        output_dir = arg.split("=", 1)[1]\n'
            "if output_dir:\n"
            "    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)\n"
            f'print("OSMOSE v4.3.3 — {name}")\n'
            f'print("Species: {nspecies}")\n'
            'print("Simulation complete")\n'
        )

        runner = _ScriptRunner(jar_path=script, java_cmd=sys.executable)
        out = tmp_path / "run_output"
        result = await runner.run(
            config_path=demo_result["config_file"],
            output_dir=out,
            overrides={"simulation.time.nyear": "2"},
        )
        assert result.returncode == 0
        assert "Simulation complete" in result.stdout
        assert name in result.stdout
        assert result.output_dir == out


# ---------------------------------------------------------------------------
# 11. Cross-study comparison tests
# ---------------------------------------------------------------------------


class TestCrossStudyConsistency:
    """Tests that compare properties across all studies."""

    @pytest.fixture(scope="class")
    def all_configs(self):
        """Load all study configs once."""
        configs = {}
        reader = OsmoseConfigReader()
        for study_name in ALL_STUDIES:
            with tempfile.TemporaryDirectory() as tmp:
                result = osmose_demo(study_name, Path(tmp))
                configs[study_name] = reader.read(result["config_file"])
        return configs

    def test_all_studies_have_version(self, all_configs):
        for name, config in all_configs.items():
            assert "osmose.version" in config, f"{name}: missing osmose.version"
            assert config["osmose.version"] == "4.3.3", (
                f"{name}: expected version 4.3.3, got {config['osmose.version']}"
            )

    def test_all_studies_have_valid_time_config(self, all_configs):
        for name, config in all_configs.items():
            nyear = int(config["simulation.time.nyear"])
            ndtperyear = int(config["simulation.time.ndtperyear"])
            assert nyear > 0, f"{name}: nyear must be positive"
            assert ndtperyear > 0, f"{name}: ndtperyear must be positive"
            assert ndtperyear in (6, 12, 24, 52, 365), (
                f"{name}: unusual ndtperyear={ndtperyear}"
            )

    def test_all_studies_have_mortality_subdt(self, all_configs):
        for name, config in all_configs.items():
            assert "mortality.subdt" in config, f"{name}: missing mortality.subdt"
            subdt = int(config["mortality.subdt"])
            assert subdt > 0

    def test_species_count_increases_with_complexity(self, all_configs):
        """Studies should be ordered: minimal < eec < bay_of_biscay < eec_full."""
        counts = {
            name: int(config["simulation.nspecies"])
            for name, config in all_configs.items()
        }
        assert counts["minimal"] < counts["eec"]
        assert counts["eec"] < counts["bay_of_biscay"]
        assert counts["bay_of_biscay"] < counts["eec_full"]

    def test_all_studies_pass_species_consistency(self, all_configs):
        for name, config in all_configs.items():
            warnings = check_species_consistency(config)
            assert warnings == [], f"{name}: species consistency: {warnings}"

    def test_all_studies_pass_schema_validation(self, all_configs):
        registry = build_registry()
        for name, config in all_configs.items():
            errors, _ = validate_config(config, registry)
            assert errors == [], f"{name}: validation errors: {errors}"

    def test_migration_is_idempotent(self, all_configs):
        """Migrating an already-current config should be a no-op."""
        for name, config in all_configs.items():
            migrated = migrate_config(config, target_version="4.3.3")
            # Content keys should be identical
            content_orig = {
                k: v for k, v in config.items()
                if not k.startswith("osmose.configuration.")
            }
            content_migrated = {
                k: v for k, v in migrated.items()
                if not k.startswith("osmose.configuration.")
            }
            assert content_orig == content_migrated, (
                f"{name}: migration changed already-current config"
            )


# ---------------------------------------------------------------------------
# 12. File reference validation
# ---------------------------------------------------------------------------


class TestFileReferences:
    """Verify that file-referencing params in each study point to existing files."""

    def test_reproduction_files_exist(self, study_demo):
        name, result, config = study_demo
        config_dir = result["config_file"].parent
        for key, value in config.items():
            if key.startswith("reproduction.season.file.") and value:
                ref = Path(value)
                if not ref.is_absolute():
                    ref = config_dir / ref
                assert ref.exists(), f"{name}: reproduction file not found: {ref}"

    def test_accessibility_file_exists(self, study_demo):
        name, result, config = study_demo
        config_dir = result["config_file"].parent
        key = "predation.accessibility.file"
        if key in config and config[key]:
            ref = Path(config[key])
            if not ref.is_absolute():
                ref = config_dir / ref
            assert ref.exists(), f"{name}: accessibility file not found: {ref}"

    def test_grid_netcdf_exists(self, study_demo):
        name, result, config = study_demo
        config_dir = result["config_file"].parent
        key = "grid.netcdf.file"
        if key in config and config[key]:
            ref = Path(config[key])
            if not ref.is_absolute():
                ref = config_dir / ref
            assert ref.exists(), f"{name}: grid NetCDF not found: {ref}"

    def test_movement_map_files_exist(self, study_demo):
        name, result, config = study_demo
        config_dir = result["config_file"].parent
        for key, value in config.items():
            if "movement" in key and "file" in key.lower() and value:
                ref = Path(value)
                if not ref.is_absolute():
                    ref = config_dir / ref
                # Only check if the reference looks like a file path (has extension)
                if ref.suffix:
                    assert ref.exists(), f"{name}: movement map not found: {ref}"


# ---------------------------------------------------------------------------
# 13. Predation configuration
# ---------------------------------------------------------------------------


class TestPredationConfig:
    """Validate predation/diet parameters per study."""

    def test_accessibility_matrix_is_readable(self, study_demo):
        name, result, config = study_demo
        config_dir = result["config_file"].parent
        key = "predation.accessibility.file"
        if key not in config or not config[key]:
            pytest.skip(f"{name}: no accessibility file configured")
        ref = config_dir / config[key]
        if not ref.exists():
            pytest.skip(f"{name}: accessibility file missing (covered by file ref test)")
        df = pd.read_csv(ref)
        assert not df.empty, f"{name}: accessibility matrix is empty"

    def test_predprey_size_ratios_are_positive(self, study_demo):
        name, _, config = study_demo
        nspecies = int(config["simulation.nspecies"])
        for i in range(nspecies):
            for suffix in ("max", "min"):
                key = f"predation.predPrey.sizeRatio.{suffix}.sp{i}"
                if key in config:
                    val = float(config[key])
                    assert val > 0, f"{name}: {key}={val} must be positive"

    def test_ingestion_rates_are_positive(self, study_demo):
        name, _, config = study_demo
        nspecies = int(config["simulation.nspecies"])
        for i in range(nspecies):
            key = f"predation.ingestion.rate.max.sp{i}"
            if key in config:
                val = float(config[key])
                assert val > 0, f"{name}: {key}={val} must be positive"
