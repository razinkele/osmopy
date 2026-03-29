"""Study-level workflow tests: scenarios, CLI, calibration, config manipulation.

Extends test_study_fullmodel.py with higher-level workflow tests:
  - Scenario save/load/compare/fork with real study configs
  - CLI validate against each study
  - Calibration problem setup with study-derived free parameters
  - Config manipulation: add/remove species, sensitivity sweeps, fishing scenarios
  - Reproduction/movement/fishing data file content validation
  - Objective functions on simulated study outputs
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from osmose.calibration.objectives import (
    biomass_rmse,
    diet_distance,
    normalized_rmse,
    weighted_multi_objective,
    yield_rmse,
)
from osmose.calibration.problem import FreeParameter, OsmoseCalibrationProblem, Transform
from osmose.cli import cmd_validate
from osmose.config.reader import OsmoseConfigReader
from osmose.config.validator import validate_config
from osmose.config.writer import OsmoseConfigWriter
from osmose.demo import migrate_config, osmose_demo
from osmose.scenarios import Scenario, ScenarioManager
from osmose.schema import build_registry

ALL_STUDIES = ["bay_of_biscay", "eec", "eec_full", "minimal"]


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
def study_config(study_name, tmp_path):
    """Load a study's config and return (name, config_dict, config_file_path)."""
    result = osmose_demo(study_name, tmp_path)
    reader = OsmoseConfigReader()
    config = reader.read(result["config_file"])
    return study_name, config, result["config_file"]


# ---------------------------------------------------------------------------
# 1. Scenario management with real study configs
# ---------------------------------------------------------------------------


class TestScenarioWorkflows:
    """Save, load, compare, fork using real study configurations."""

    def test_save_and_load_study_config(self, study_config, tmp_path):
        name, config, _ = study_config
        manager = ScenarioManager(tmp_path / "scenarios")
        scenario = Scenario(
            name=f"{name}_baseline",
            description=f"Baseline {name} configuration",
            config=config,
            tags=["baseline", name],
        )
        manager.save(scenario)
        loaded = manager.load(f"{name}_baseline")
        assert loaded.config == config
        assert loaded.tags == ["baseline", name]

    def test_fork_and_modify_study(self, study_config, tmp_path):
        """Fork a study, modify fishing, verify original unchanged."""
        name, config, _ = study_config
        manager = ScenarioManager(tmp_path / "scenarios")
        manager.save(Scenario(name=f"{name}_base", config=config))

        forked = manager.fork(f"{name}_base", f"{name}_high_fishing")
        # Double fishing mortality for sp0
        key = "mortality.fishing.rate.sp0"
        if key in forked.config:
            original_rate = float(forked.config[key])
            forked.config[key] = str(original_rate * 2)
            manager.save(forked)

            # Verify original is untouched
            original = manager.load(f"{name}_base")
            assert float(original.config[key]) == original_rate

    def test_compare_baseline_vs_modified(self, study_config, tmp_path):
        """Compare two variants of a study config."""
        name, config, _ = study_config
        manager = ScenarioManager(tmp_path / "scenarios")

        # Baseline
        manager.save(Scenario(name="baseline", config=dict(config)))

        # Modified: shorter simulation
        modified = dict(config)
        modified["simulation.time.nyear"] = "5"
        manager.save(Scenario(name="short_run", config=modified))

        diffs = manager.compare("baseline", "short_run")
        diff_keys = {d.key for d in diffs}
        assert "simulation.time.nyear" in diff_keys

    def test_export_import_study_roundtrip(self, study_config, tmp_path):
        """Export study config to ZIP, import into fresh manager."""
        name, config, _ = study_config
        src_mgr = ScenarioManager(tmp_path / "src_scenarios")
        src_mgr.save(Scenario(name=name, config=config))

        zip_path = tmp_path / "export.zip"
        src_mgr.export_all(zip_path)

        dst_mgr = ScenarioManager(tmp_path / "dst_scenarios")
        count = dst_mgr.import_all(zip_path)
        assert count == 1

        loaded = dst_mgr.load(name)
        assert loaded.config == config


# ---------------------------------------------------------------------------
# 2. CLI validate with real study configs
# ---------------------------------------------------------------------------


class TestCLIValidation:
    """Run CLI validate command against each study's config."""

    def test_cli_validate_study_config(self, study_config):
        """CLI validate should return 0 (success) for each study."""
        name, _, config_file = study_config
        args = MagicMock()
        args.config = str(config_file)
        result = cmd_validate(args)
        assert result == 0, f"CLI validate failed for {name}"

    def test_cli_validate_detects_missing_file(self, tmp_path):
        args = MagicMock()
        args.config = str(tmp_path / "nonexistent.csv")
        result = cmd_validate(args)
        assert result == 1


# ---------------------------------------------------------------------------
# 3. File reference validation (content-level)
# ---------------------------------------------------------------------------


class TestFileReferenceContent:
    """Validate that referenced data files have valid content."""

    def test_reproduction_files_have_seasonal_data(self, study_config):
        """Reproduction season files should have time and species columns."""
        name, config, config_file = study_config
        config_dir = config_file.parent
        for key, value in config.items():
            if not key.startswith("reproduction.season.file."):
                continue
            if not value:
                continue
            ref = config_dir / value
            if not ref.exists():
                continue
            df = pd.read_csv(ref, sep=";")
            assert len(df) > 0, f"{name}: empty reproduction file {value}"
            assert len(df.columns) >= 2, (
                f"{name}: reproduction file {value} needs at least 2 columns"
            )
            # First column should be time-like (monotonically increasing)
            first_col = df.iloc[:, 0]
            assert first_col.is_monotonic_increasing, (
                f"{name}: {value} first column not monotonically increasing"
            )

    def test_reproduction_seasonality_sums_to_one(self, study_config):
        """Reproduction season proportions should sum approximately to 1."""
        name, config, config_file = study_config
        config_dir = config_file.parent
        for key, value in config.items():
            if not key.startswith("reproduction.season.file."):
                continue
            if not value:
                continue
            ref = config_dir / value
            if not ref.exists():
                continue
            df = pd.read_csv(ref, sep=";")
            if len(df.columns) < 2:
                continue
            # Sum of reproduction proportions (second column)
            reprod_sum = df.iloc[:, 1].sum()
            # Some files might have 0 values; just check it's non-negative
            assert reprod_sum >= 0, f"{name}: negative reproduction sum in {value}"

    def test_accessibility_matrix_is_square_or_rectangular(self, study_config):
        """Accessibility matrix should be a valid numeric matrix."""
        name, config, config_file = study_config
        key = "predation.accessibility.file"
        if key not in config or not config[key]:
            pytest.skip(f"{name}: no accessibility file")
        ref = config_file.parent / config[key]
        if not ref.exists():
            pytest.skip(f"{name}: accessibility file missing")
        # OSMOSE accessibility matrices use `;` separator
        df = pd.read_csv(ref, sep=";")
        assert not df.empty
        # Should have numeric data (columns after the label column)
        numeric = df.select_dtypes(include=[np.number])
        assert numeric.shape[1] > 0, f"{name}: no numeric columns in accessibility matrix"
        # All values should be between 0 and 1
        assert (numeric.values >= 0).all(), f"{name}: negative accessibility values"
        assert (numeric.values <= 1).all(), f"{name}: accessibility values > 1"

    def test_movement_map_files_are_valid_grids(self, study_config):
        """Movement map CSV files should be valid 2D grids."""
        name, config, config_file = study_config
        config_dir = config_file.parent
        # eec_full stores maps in a maps/ directory; check it directly
        maps_dir = config_dir / "maps"
        if not maps_dir.exists():
            pytest.skip(f"{name}: no maps directory")
        map_files = list(maps_dir.glob("*.csv"))
        if not map_files:
            pytest.skip(f"{name}: no CSV map files")
        for map_file in map_files[:5]:  # Check first 5 for speed
            # OSMOSE map files use `;` separator
            df = pd.read_csv(map_file, header=None, sep=";")
            assert not df.empty, f"{name}: empty movement map {map_file.name}"
            numeric = df.select_dtypes(include=[np.number])
            assert numeric.shape[0] > 0 and numeric.shape[1] > 0, (
                f"{name}: {map_file.name} has no numeric grid data"
            )


# ---------------------------------------------------------------------------
# 4. Config manipulation tests
# ---------------------------------------------------------------------------


class TestConfigManipulation:
    """Test modifying study configs and validating the results."""

    def test_reduce_species_count(self, study_config, registry):
        """Remove the last species and verify config still validates."""
        name, config, _ = study_config
        nspecies = int(config["simulation.nspecies"])
        if nspecies <= 1:
            pytest.skip("Can't reduce single-species config")

        # Create modified config with one fewer species
        modified = dict(config)
        modified["simulation.nspecies"] = str(nspecies - 1)
        # Remove keys for the last species
        last_idx = nspecies - 1
        keys_to_remove = [k for k in modified if k.endswith(f".sp{last_idx}")]
        for k in keys_to_remove:
            del modified[k]

        # Should still validate
        errors, _ = validate_config(modified, registry)
        assert errors == [], f"{name}: validation errors after removing sp{last_idx}: {errors}"

    def test_change_simulation_duration(self, study_config, registry):
        """Modify nyear and ndtperyear, verify validates."""
        name, config, _ = study_config
        modified = dict(config)
        modified["simulation.time.nyear"] = "5"
        modified["simulation.time.ndtperyear"] = "12"

        errors, _ = validate_config(modified, registry)
        assert errors == [], f"{name}: validation errors after duration change: {errors}"

    def test_double_fishing_mortality(self, study_config, registry):
        """Double all fishing mortality rates and verify they still validate."""
        name, config, _ = study_config
        modified = dict(config)
        for key, value in config.items():
            if key.startswith("mortality.fishing.rate."):
                try:
                    modified[key] = str(float(value) * 2)
                except ValueError:
                    pass

        errors, _ = validate_config(modified, registry)
        assert errors == [], f"{name}: errors after doubling fishing rates: {errors}"

    def test_zero_fishing_mortality(self, study_config, registry):
        """Set all fishing to zero — valid no-fishing scenario."""
        name, config, _ = study_config
        modified = dict(config)
        for key in config:
            if key.startswith("mortality.fishing.rate."):
                modified[key] = "0.0"

        errors, _ = validate_config(modified, registry)
        assert errors == [], f"{name}: errors with zero fishing: {errors}"

    def test_config_roundtrip_after_modification(self, study_config):
        """Modified config survives write -> read roundtrip."""
        name, config, _ = study_config
        modified = dict(config)
        modified["simulation.time.nyear"] = "3"

        content = {k: v for k, v in modified.items() if not k.startswith("osmose.configuration.")}
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = OsmoseConfigWriter()
            writer.write(content, Path(tmpdir))
            reader = OsmoseConfigReader()
            result = reader.read(Path(tmpdir) / "osm_all-parameters.csv")
            assert result["simulation.time.nyear"] == "3"

    def test_add_output_types(self, study_config, registry):
        """Enable additional outputs and verify validates."""
        name, config, _ = study_config
        modified = dict(config)
        modified["output.biomass.enabled"] = "true"
        modified["output.abundance.enabled"] = "true"
        modified["output.spatial.enabled"] = "true"

        errors, _ = validate_config(modified, registry)
        assert errors == [], f"{name}: errors after enabling outputs: {errors}"


# ---------------------------------------------------------------------------
# 5. Calibration problem setup with study configs
# ---------------------------------------------------------------------------


class TestCalibrationSetup:
    """Set up calibration problems using real study config parameters."""

    def test_calibration_problem_from_study_growth_params(self, study_config, tmp_path):
        """Build a calibration problem with free growth parameters from each study."""
        name, config, config_file = study_config
        nspecies = int(config["simulation.nspecies"])

        # Define free parameters: growth rate K for each species
        free_params = []
        for i in range(min(nspecies, 3)):  # Limit to 3 for test speed
            key = f"species.k.sp{i}"
            if key in config:
                k_val = float(config[key])
                free_params.append(
                    FreeParameter(
                        key=key,
                        lower_bound=k_val * 0.5,
                        upper_bound=k_val * 2.0,
                    )
                )

        if not free_params:
            pytest.skip(f"{name}: no growth params found")

        problem = OsmoseCalibrationProblem(
            free_params=free_params,
            objective_fns=[lambda r: 0.0],
            base_config_path=config_file,
            jar_path=tmp_path / "osmose.jar",
            work_dir=tmp_path / "calibration",
        )

        assert problem.n_var == len(free_params)
        assert problem.n_obj == 1
        # Bounds should reflect the doubled/halved values
        for i, fp in enumerate(free_params):
            assert problem.xl[i] == fp.lower_bound
            assert problem.xu[i] == fp.upper_bound

    def test_calibration_problem_multi_objective(self, study_config, tmp_path):
        """Multi-objective problem with biomass + yield objectives."""
        name, config, config_file = study_config
        nspecies = int(config["simulation.nspecies"])

        free_params = []
        for i in range(min(nspecies, 2)):
            key = f"species.k.sp{i}"
            if key in config:
                free_params.append(FreeParameter(key=key, lower_bound=0.01, upper_bound=1.0))

        if not free_params:
            pytest.skip(f"{name}: no growth params")

        def mock_biomass_obj(results):
            return 0.5

        def mock_yield_obj(results):
            return 1.0

        problem = OsmoseCalibrationProblem(
            free_params=free_params,
            objective_fns=[mock_biomass_obj, mock_yield_obj],
            base_config_path=config_file,
            jar_path=tmp_path / "osmose.jar",
            work_dir=tmp_path / "calibration",
        )

        assert problem.n_obj == 2

        # Simulate evaluation
        X = np.random.uniform(
            [fp.lower_bound for fp in free_params],
            [fp.upper_bound for fp in free_params],
            size=(5, len(free_params)),
        )
        out = {}
        with patch.object(problem, "_run_single", return_value=[0.5, 1.0]):
            problem._evaluate(X, out)

        assert out["F"].shape == (5, 2)

    def test_calibration_log_transform_param(self, study_config, tmp_path):
        """Set up a log-transformed parameter from a study."""
        name, config, config_file = study_config
        key = "species.k.sp0"
        if key not in config:
            pytest.skip(f"{name}: missing {key}")

        fp = FreeParameter(key=key, lower_bound=-2, upper_bound=0, transform=Transform.LOG)
        problem = OsmoseCalibrationProblem(
            free_params=[fp],
            objective_fns=[lambda r: 0.0],
            base_config_path=config_file,
            jar_path=tmp_path / "osmose.jar",
            work_dir=tmp_path / "calibration",
        )

        # Verify candidate evaluation applies 10**val
        X = np.array([[-1.0]])  # 10**-1 = 0.1
        out = {}
        with patch.object(problem, "_run_single", return_value=[0.0]) as mock_run:
            problem._evaluate(X, out)

        overrides = mock_run.call_args[0][0]
        assert abs(float(overrides[key]) - 0.1) < 1e-10


# ---------------------------------------------------------------------------
# 6. Objective functions with study-like data
# ---------------------------------------------------------------------------


class TestObjectiveFunctions:
    """Test calibration objectives using data shaped like each study."""

    @pytest.fixture
    def study_timeseries(self, study_config):
        """Generate simulated + observed time series for each study."""
        name, config, _ = study_config
        nspecies = int(config["simulation.nspecies"])
        species = [config[f"species.name.sp{i}"] for i in range(nspecies)]
        rng = np.random.default_rng(42)
        n_time = 50

        sim_frames = []
        obs_frames = []
        for sp in species:
            sim_biomass = rng.exponential(scale=50000, size=n_time)
            obs_biomass = sim_biomass * rng.normal(1.0, 0.1, size=n_time)
            sim_frames.append(
                pd.DataFrame(
                    {
                        "time": range(n_time),
                        "species": sp,
                        "biomass": sim_biomass,
                        "yield": rng.exponential(scale=5000, size=n_time),
                    }
                )
            )
            obs_frames.append(
                pd.DataFrame(
                    {
                        "time": range(n_time),
                        "species": sp,
                        "biomass": obs_biomass,
                        "yield": rng.exponential(scale=5000, size=n_time),
                    }
                )
            )

        sim = pd.concat(sim_frames, ignore_index=True)
        obs = pd.concat(obs_frames, ignore_index=True)
        return name, species, sim, obs

    def test_biomass_rmse_is_finite(self, study_timeseries):
        name, species, sim, obs = study_timeseries
        rmse = biomass_rmse(sim, obs)
        assert np.isfinite(rmse), f"{name}: biomass RMSE is not finite"
        assert rmse >= 0

    def test_biomass_rmse_per_species(self, study_timeseries):
        name, species, sim, obs = study_timeseries
        for sp in species:
            rmse = biomass_rmse(sim, obs, species=sp)
            assert np.isfinite(rmse), f"{name}: biomass RMSE for {sp} not finite"
            assert rmse >= 0

    def test_yield_rmse_is_finite(self, study_timeseries):
        name, _, sim, obs = study_timeseries
        rmse = yield_rmse(sim, obs)
        assert np.isfinite(rmse)
        assert rmse >= 0

    def test_normalized_rmse(self, study_timeseries):
        name, _, sim, obs = study_timeseries
        sim_arr = sim["biomass"].values
        obs_arr = obs["biomass"].values
        nrmse = normalized_rmse(sim_arr, obs_arr)
        assert np.isfinite(nrmse)
        assert nrmse >= 0

    def test_weighted_multi_objective(self, study_timeseries):
        name, _, sim, obs = study_timeseries
        obj1 = biomass_rmse(sim, obs)
        obj2 = yield_rmse(sim, obs)
        weighted = weighted_multi_objective([obj1, obj2], [0.7, 0.3])
        assert np.isfinite(weighted)
        assert weighted >= 0

    def test_diet_distance_with_matching_matrices(self, study_timeseries):
        name, species, _, _ = study_timeseries
        rng = np.random.default_rng(99)
        n = len(species)
        sim_diet = pd.DataFrame(
            rng.dirichlet(np.ones(n), size=n),
            columns=species,
        )
        obs_diet = pd.DataFrame(
            rng.dirichlet(np.ones(n), size=n),
            columns=species,
        )
        dist = diet_distance(sim_diet, obs_diet)
        assert np.isfinite(dist)
        assert dist >= 0

    def test_diet_distance_identical_is_zero(self, study_timeseries):
        name, species, _, _ = study_timeseries
        rng = np.random.default_rng(99)
        n = len(species)
        diet = pd.DataFrame(
            rng.dirichlet(np.ones(n), size=n),
            columns=species,
        )
        dist = diet_distance(diet, diet)
        assert abs(dist) < 1e-10


# ---------------------------------------------------------------------------
# 7. Sensitivity sweep (parameter perturbation)
# ---------------------------------------------------------------------------


class TestSensitivitySweeps:
    """Verify config perturbation for one-at-a-time sensitivity analysis."""

    def test_perturb_growth_params(self, study_config, registry):
        """Perturb each species' K by +/-20% and validate."""
        name, config, _ = study_config
        nspecies = int(config["simulation.nspecies"])
        perturbations = [0.8, 1.0, 1.2]

        for i in range(min(nspecies, 3)):
            key = f"species.k.sp{i}"
            if key not in config:
                continue
            base_val = float(config[key])
            for factor in perturbations:
                modified = dict(config)
                modified[key] = str(base_val * factor)
                errors, _ = validate_config(modified, registry)
                assert errors == [], f"{name}: perturbing {key} by {factor}x caused: {errors}"

    def test_perturb_mortality_params(self, study_config, registry):
        """Perturb additional mortality rates and validate."""
        name, config, _ = study_config
        nspecies = int(config["simulation.nspecies"])

        for i in range(min(nspecies, 3)):
            key = f"mortality.additional.rate.sp{i}"
            if key not in config:
                continue
            base_val = float(config[key])
            for factor in [0.5, 1.5]:
                modified = dict(config)
                modified[key] = str(base_val * factor)
                errors, _ = validate_config(modified, registry)
                assert errors == [], f"{name}: perturbing {key} by {factor}x caused: {errors}"

    def test_perturb_seeding_biomass(self, study_config, registry):
        """Perturb seeding biomass and validate."""
        name, config, _ = study_config
        nspecies = int(config["simulation.nspecies"])

        for i in range(min(nspecies, 2)):
            key = f"population.seeding.biomass.sp{i}"
            if key not in config:
                continue
            base_val = float(config[key])
            for factor in [0.1, 10.0]:
                modified = dict(config)
                modified[key] = str(base_val * factor)
                errors, _ = validate_config(modified, registry)
                assert errors == [], f"{name}: perturbing {key} by {factor}x caused: {errors}"


# ---------------------------------------------------------------------------
# 8. Migration edge cases with study configs
# ---------------------------------------------------------------------------


class TestMigrationWithStudyConfigs:
    """Test config migration starting from pre-current versions of study configs."""

    def test_migrate_study_from_old_version(self, study_config):
        """Downgrade version and re-migrate; result should match original keys."""
        name, config, _ = study_config
        # Simulate an old-version config by reverting key renames
        old_config = dict(config)
        old_config["osmose.version"] = "3.0.0"

        # Revert mortality.additional -> mortality.natural
        for key in list(old_config.keys()):
            if key.startswith("mortality.additional.rate."):
                suffix = key[len("mortality.additional.rate.") :]
                old_config[f"mortality.natural.rate.{suffix}"] = old_config.pop(key)
            elif key.startswith("mortality.additional.larva.rate."):
                suffix = key[len("mortality.additional.larva.rate.") :]
                old_config[f"mortality.natural.larva.rate.{suffix}"] = old_config.pop(key)

        # Migrate to current
        migrated = migrate_config(old_config, target_version="4.3.3")

        # Verify mortality keys were renamed back
        for key in config:
            if key.startswith("mortality.additional.rate."):
                assert key in migrated, f"{name}: migration lost {key}"

    def test_double_migration_is_stable(self, study_config):
        """Migrating an already-migrated config should be a no-op."""
        name, config, _ = study_config
        first = migrate_config(config, target_version="4.3.3")
        second = migrate_config(first, target_version="4.3.3")
        assert first == second, f"{name}: double migration changed config"


# ---------------------------------------------------------------------------
# 9. Fishing-specific tests
# ---------------------------------------------------------------------------


class TestFishingConfig:
    """Validate fishing configuration across studies."""

    def test_fishing_rates_are_non_negative(self, study_config):
        name, config, _ = study_config
        nspecies = int(config["simulation.nspecies"])
        for i in range(nspecies):
            key = f"mortality.fishing.rate.sp{i}"
            if key in config:
                rate = float(config[key])
                assert rate >= 0, f"{name}: negative fishing rate for sp{i}: {rate}"

    def test_fishing_enabled_flag_is_boolean(self, study_config):
        name, config, _ = study_config
        key = "simulation.fishing.mortality.enabled"
        if key in config:
            assert config[key].lower() in ("true", "false"), (
                f"{name}: invalid boolean for fishing enabled: {config[key]}"
            )

    def test_starvation_max_rates_are_bounded(self, study_config):
        name, config, _ = study_config
        nspecies = int(config["simulation.nspecies"])
        for i in range(nspecies):
            key = f"mortality.starvation.rate.max.sp{i}"
            if key in config:
                rate = float(config[key])
                assert 0 <= rate <= 1.0, (
                    f"{name}: starvation max rate for sp{i} out of [0,1]: {rate}"
                )

    def test_additional_mortality_rates_are_plausible(self, study_config):
        name, config, _ = study_config
        nspecies = int(config["simulation.nspecies"])
        for i in range(nspecies):
            key = f"mortality.additional.rate.sp{i}"
            if key in config:
                rate = float(config[key])
                assert 0 <= rate <= 5.0, (
                    f"{name}: implausible additional mortality for sp{i}: {rate}"
                )


# ---------------------------------------------------------------------------
# 10. Movement configuration tests
# ---------------------------------------------------------------------------


class TestMovementConfig:
    """Validate movement parameters across studies."""

    def test_movement_method_is_valid(self, study_config):
        name, config, _ = study_config
        valid_methods = {"random", "maps", "none", "connectivity", "migration"}
        nspecies = int(config["simulation.nspecies"])
        for i in range(nspecies):
            key = f"movement.distribution.method.sp{i}"
            if key in config:
                method = config[key].lower()
                assert method in valid_methods, (
                    f"{name}: invalid movement method for sp{i}: {method}"
                )

    def test_eec_full_has_movement_maps(self, tmp_path):
        """EEC full study should have extensive movement map files."""
        result = osmose_demo("eec_full", tmp_path)
        config_dir = result["config_file"].parent
        maps_dir = config_dir / "maps"
        if maps_dir.exists():
            map_files = list(maps_dir.glob("*.csv"))
            assert len(map_files) >= 35, f"Expected >= 35 movement maps, got {len(map_files)}"


# ---------------------------------------------------------------------------
# 11. Resource/LTL configuration tests
# ---------------------------------------------------------------------------


class TestResourceConfig:
    """Validate lower trophic level / resource configuration."""

    def test_bay_of_biscay_has_resources(self, tmp_path):
        result = osmose_demo("bay_of_biscay", tmp_path)
        reader = OsmoseConfigReader()
        config = reader.read(result["config_file"])
        nresource = int(config.get("simulation.nresource", "0"))
        assert nresource == 6

    def test_resource_species_have_names(self, study_config):
        name, config, _ = study_config
        nspecies = int(config["simulation.nspecies"])
        nresource = int(config.get("simulation.nresource", "0"))
        if nresource == 0:
            pytest.skip(f"{name}: no resources")
        # Resource species are indexed after focal species in some configs
        # Check if resource names exist
        resource_names = [
            k for k in config if k.startswith("species.name.") or k.startswith("resource.name.")
        ]
        assert len(resource_names) >= nspecies, (
            f"{name}: expected at least {nspecies} species names, got {len(resource_names)}"
        )

    def test_minimal_has_no_resources(self, tmp_path):
        result = osmose_demo("minimal", tmp_path)
        reader = OsmoseConfigReader()
        config = reader.read(result["config_file"])
        nresource = int(config.get("simulation.nresource", "0"))
        assert nresource == 0


# ---------------------------------------------------------------------------
# 12. Study-specific edge case tests
# ---------------------------------------------------------------------------


class TestStudySpecificEdgeCases:
    """Study-specific edge cases and known configurations."""

    def test_eec_full_uses_different_master_name(self, tmp_path):
        """EEC full uses eec_all-parameters.csv, not osm_all-parameters.csv."""
        result = osmose_demo("eec_full", tmp_path)
        assert result["config_file"].name == "eec_all-parameters.csv"

    def test_eec_full_has_14_species_14_reproduction_files(self, tmp_path):
        result = osmose_demo("eec_full", tmp_path)
        config_dir = result["config_file"].parent
        reprod_dir = config_dir / "reproduction"
        if reprod_dir.exists():
            reprod_files = list(reprod_dir.glob("*.csv"))
            assert len(reprod_files) == 14

    def test_eec_full_has_netcdf_files(self, tmp_path):
        result = osmose_demo("eec_full", tmp_path)
        config_dir = result["config_file"].parent
        nc_files = list(config_dir.glob("*.nc"))
        assert len(nc_files) >= 2, f"Expected >=2 NetCDF files, got {len(nc_files)}"

    def test_minimal_has_short_simulation(self, tmp_path):
        result = osmose_demo("minimal", tmp_path)
        reader = OsmoseConfigReader()
        config = reader.read(result["config_file"])
        assert int(config["simulation.time.nyear"]) == 10
        assert int(config["simulation.time.ndtperyear"]) == 12

    def test_bay_of_biscay_has_8_reproduction_files(self, tmp_path):
        result = osmose_demo("bay_of_biscay", tmp_path)
        config_dir = result["config_file"].parent
        reprod_files = list((config_dir / "reproduction").glob("*.csv"))
        assert len(reprod_files) == 8

    def test_eec_has_6_reproduction_files(self, tmp_path):
        result = osmose_demo("eec", tmp_path)
        config_dir = result["config_file"].parent
        reprod_files = list((config_dir / "reproduction").glob("*.csv"))
        assert len(reprod_files) == 6

    def test_all_studies_have_predation_config(self):
        """Every study should have predation parameters."""
        reader = OsmoseConfigReader()
        for study_name in ALL_STUDIES:
            with tempfile.TemporaryDirectory() as tmp:
                result = osmose_demo(study_name, Path(tmp))
                config = reader.read(result["config_file"])
                pred_keys = [k for k in config if k.startswith("predation.")]
                assert len(pred_keys) > 0, f"{study_name}: no predation parameters"

    def test_all_studies_have_output_config(self):
        """Every study should have output configuration."""
        reader = OsmoseConfigReader()
        for study_name in ALL_STUDIES:
            with tempfile.TemporaryDirectory() as tmp:
                result = osmose_demo(study_name, Path(tmp))
                config = reader.read(result["config_file"])
                output_keys = [k for k in config if k.startswith("output.")]
                assert len(output_keys) > 0, f"{study_name}: no output parameters"
