"""Tests for Phase 5: Output system."""

import numpy as np

from tests.helpers import _make_school  # noqa: F401 — used in tests below
from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import StepOutput, simulate
from osmose.engine.state import MortalityCause


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_config(n_sp: int = 1, n_dt: int = 12) -> dict[str, str]:
    """Minimal config dict for phase 5 tests."""
    cfg: dict[str, str] = {
        "simulation.time.ndtperyear": str(n_dt),
        "simulation.time.nyear": "2",
        "simulation.nspecies": str(n_sp),
        "mortality.subdt": "10",
    }
    names = ["FishA", "FishB", "FishC"]
    for i in range(n_sp):
        cfg.update(
            {
                f"simulation.nschool.sp{i}": "5",
                f"species.name.sp{i}": names[i],
                f"species.linf.sp{i}": "30.0",
                f"species.k.sp{i}": "0.3",
                f"species.t0.sp{i}": "-0.1",
                f"species.egg.size.sp{i}": "0.1",
                f"species.length2weight.condition.factor.sp{i}": "0.006",
                f"species.length2weight.allometric.power.sp{i}": "3.0",
                f"species.lifespan.sp{i}": "5",
                f"species.vonbertalanffy.threshold.age.sp{i}": "1.0",
                f"predation.ingestion.rate.max.sp{i}": "3.5",
                f"predation.efficiency.critical.sp{i}": "0.57",
                f"species.sexratio.sp{i}": "0.5",
                f"species.relativefecundity.sp{i}": "800",
                f"species.maturity.size.sp{i}": "12.0",
                f"population.seeding.biomass.sp{i}": "50000",
            }
        )
    return cfg


# ===========================================================================
# 5.1 — Output Recording Frequency
# ===========================================================================


class TestOutputRecordingFrequency:
    def test_record_frequency_parsed(self):
        """output.recordfrequency.ndt should be parsed into config."""
        cfg = _base_config(n_sp=1, n_dt=12)
        cfg["output.recordfrequency.ndt"] = "6"
        config = EngineConfig.from_dict(cfg)
        assert config.output_record_frequency == 6

    def test_record_frequency_defaults_to_every_step(self):
        """Default recording frequency should be 1 (every step)."""
        cfg = _base_config(n_sp=1, n_dt=12)
        config = EngineConfig.from_dict(cfg)
        assert config.output_record_frequency == 1

    def test_simulation_output_length_with_frequency(self):
        """With record frequency, output count should be n_steps / frequency."""
        cfg = _base_config(n_sp=1, n_dt=4)
        cfg["simulation.time.nyear"] = "2"
        cfg["output.recordfrequency.ndt"] = "2"
        config = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(3, 3)
        rng = np.random.default_rng(42)
        outputs = simulate(config, grid, rng)
        # n_steps = 4*2 = 8, frequency = 2, so 8/2 = 4 outputs
        assert len(outputs) == 4

    def test_simulation_output_length_default(self):
        """Without custom frequency (default=1), output count should be n_steps."""
        cfg = _base_config(n_sp=1, n_dt=4)
        cfg["simulation.time.nyear"] = "2"
        config = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(3, 3)
        rng = np.random.default_rng(42)
        outputs = simulate(config, grid, rng)
        # n_steps = 8, default frequency = 1, so 8 outputs
        assert len(outputs) == 8


# ===========================================================================
# 5.2 — Yield/Catches Output
# ===========================================================================


class TestYieldOutput:
    def test_yield_tracked_in_step_output(self):
        """StepOutput should include yield_by_species field."""
        cfg = _base_config(n_sp=1, n_dt=4)
        config = EngineConfig.from_dict(cfg)
        n_total = config.n_species + config.n_background
        # Create a StepOutput with yield data
        out = StepOutput(
            step=0,
            biomass=np.zeros(n_total),
            abundance=np.zeros(n_total),
            mortality_by_cause=np.zeros((config.n_species, len(MortalityCause))),
            yield_by_species=np.array([100.0]),
        )
        assert out.yield_by_species[0] == 100.0

    def test_yield_csv_written(self, tmp_path):
        """write_outputs should produce a yield CSV file."""
        from osmose.engine.output import write_outputs

        cfg = _base_config(n_sp=1, n_dt=4)
        config = EngineConfig.from_dict(cfg)
        outputs = [
            StepOutput(
                step=i,
                biomass=np.array([100.0]),
                abundance=np.array([500.0]),
                mortality_by_cause=np.zeros((1, len(MortalityCause))),
                yield_by_species=np.array([10.0 * i]),
            )
            for i in range(4)
        ]
        write_outputs(outputs, tmp_path, config, prefix="test")
        yield_path = tmp_path / "test_yield_Simu0.csv"
        assert yield_path.exists()
        # Read and verify content
        lines = yield_path.read_text().splitlines()
        assert len(lines) >= 3  # header description + column header + data rows


# ===========================================================================
# 5.5 — Diet Composition Output Per Step
# ===========================================================================


class TestDietCompositionOutput:
    def test_diet_config_parsed(self):
        """output.diet.composition.enabled should be parsed into config."""
        cfg = _base_config(n_sp=1)
        cfg["output.diet.composition.enabled"] = "true"
        config = EngineConfig.from_dict(cfg)
        assert config.diet_output_enabled is True

    def test_diet_config_defaults_false(self):
        """Diet output should default to disabled."""
        cfg = _base_config(n_sp=1)
        config = EngineConfig.from_dict(cfg)
        assert config.diet_output_enabled is False


# ===========================================================================
# 5.6 — NetCDF Output Format
# ===========================================================================


class TestNetCDFOutput:
    def test_netcdf_written(self, tmp_path):
        """write_outputs_netcdf should produce a valid NetCDF file."""
        from osmose.engine.output import write_outputs_netcdf

        cfg = _base_config(n_sp=2, n_dt=4)
        config = EngineConfig.from_dict(cfg)
        outputs = [
            StepOutput(
                step=i,
                biomass=np.array([100.0 * (i + 1), 200.0 * (i + 1)]),
                abundance=np.array([500.0, 1000.0]),
                mortality_by_cause=np.zeros((2, len(MortalityCause))),
                yield_by_species=np.zeros(2),
            )
            for i in range(8)
        ]
        out_path = tmp_path / "test_output.nc"
        write_outputs_netcdf(outputs, out_path, config)
        assert out_path.exists()

        import xarray as xr

        ds = xr.open_dataset(out_path)
        assert "biomass" in ds
        assert "abundance" in ds
        assert ds["biomass"].dims == ("time", "species")
        assert ds["biomass"].shape == (8, 2)
        ds.close()

    def test_netcdf_correct_values(self, tmp_path):
        """NetCDF biomass values should match input StepOutputs."""
        from osmose.engine.output import write_outputs_netcdf

        cfg = _base_config(n_sp=1, n_dt=4)
        config = EngineConfig.from_dict(cfg)
        outputs = [
            StepOutput(
                step=0,
                biomass=np.array([42.0]),
                abundance=np.array([100.0]),
                mortality_by_cause=np.zeros((1, len(MortalityCause))),
                yield_by_species=np.zeros(1),
            )
        ]
        out_path = tmp_path / "test_values.nc"
        write_outputs_netcdf(outputs, out_path, config)

        import xarray as xr

        ds = xr.open_dataset(out_path)
        np.testing.assert_allclose(ds["biomass"].values[0, 0], 42.0)
        np.testing.assert_allclose(ds["abundance"].values[0, 0], 100.0)
        ds.close()


# ===========================================================================
# 5.7 — Initial State Output (output.step0.include)
# ===========================================================================


class TestInitialStateOutput:
    def test_step0_config_parsed(self):
        """output.step0.include should be parsed into config."""
        cfg = _base_config(n_sp=1, n_dt=4)
        cfg["output.step0.include"] = "true"
        config = EngineConfig.from_dict(cfg)
        assert config.output_step0_include is True

    def test_step0_defaults_false(self):
        """output.step0.include should default to False."""
        cfg = _base_config(n_sp=1, n_dt=4)
        config = EngineConfig.from_dict(cfg)
        assert config.output_step0_include is False

    def test_simulation_includes_initial_state(self):
        """When step0 is enabled, outputs should include an extra initial-state record."""
        cfg = _base_config(n_sp=1, n_dt=4)
        cfg["simulation.time.nyear"] = "1"
        cfg["output.step0.include"] = "true"
        config = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(3, 3)
        rng = np.random.default_rng(42)
        outputs = simulate(config, grid, rng)
        # n_steps = 4, frequency = 1, so 4 outputs + 1 initial state = 5
        assert len(outputs) == 5
        # First output should be step -1 (initial state)
        assert outputs[0].step == -1


def test_write_outputs_skips_diet_csv_when_disabled(tmp_path):
    import numpy as np
    from osmose.engine.output import write_outputs
    from osmose.engine.simulate import StepOutput
    from tests.helpers import make_minimal_engine_config

    cfg = make_minimal_engine_config(diet_output_enabled=False)
    n_pred = cfg.n_species
    n_prey = len(cfg.all_species_names)
    step_out = StepOutput(
        step=23,
        biomass=np.zeros(n_pred + cfg.n_background),
        abundance=np.zeros(n_pred + cfg.n_background),
        mortality_by_cause=np.zeros((n_pred, 8)),
        diet_by_species=np.ones((n_pred, n_prey), dtype=np.float64),
    )
    # write_outputs signature: (outputs, output_dir, config, prefix="osm")
    write_outputs([step_out], tmp_path, cfg, prefix="run")
    assert not (tmp_path / "run_dietMatrix_Simu0.csv").exists()
