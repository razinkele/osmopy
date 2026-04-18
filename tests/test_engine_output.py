"""Tests for engine output writer — CSV format matching Java."""

import numpy as np
import pandas as pd
import xarray as xr

from osmose.engine.config import EngineConfig
from osmose.engine.output import write_outputs, write_outputs_netcdf
from osmose.engine.simulate import StepOutput
from osmose.engine.state import MortalityCause
from tests.helpers import make_minimal_engine_config

_N_CAUSES = len(MortalityCause)


def _make_output_config() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "2",
        "simulation.nschool.sp0": "5",
        "simulation.nschool.sp1": "5",
        "species.name.sp0": "Anchovy",
        "species.name.sp1": "Hake",
        "species.linf.sp0": "19.5",
        "species.linf.sp1": "110.0",
        "species.k.sp0": "0.364",
        "species.k.sp1": "0.106",
        "species.t0.sp0": "-0.70",
        "species.t0.sp1": "-0.17",
        "species.egg.size.sp0": "0.1",
        "species.egg.size.sp1": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.condition.factor.sp1": "0.005",
        "species.length2weight.allometric.power.sp0": "3.06",
        "species.length2weight.allometric.power.sp1": "3.14",
        "species.lifespan.sp0": "4",
        "species.lifespan.sp1": "12",
        "species.vonbertalanffy.threshold.age.sp0": "0",
        "species.vonbertalanffy.threshold.age.sp1": "0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.ingestion.rate.max.sp1": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "predation.efficiency.critical.sp1": "0.57",
    }


def _step_output(step, biomass, abundance, mortality_by_cause=None):
    """Helper to create StepOutput with default zero mortality."""
    n_sp = len(biomass)
    if mortality_by_cause is None:
        mortality_by_cause = np.zeros((n_sp, _N_CAUSES), dtype=np.float64)
    return StepOutput(
        step=step,
        biomass=biomass,
        abundance=abundance,
        mortality_by_cause=mortality_by_cause,
    )


class TestWriteOutputs:
    def test_creates_biomass_csv(self, tmp_path):
        cfg = EngineConfig.from_dict(_make_output_config())
        outputs = [
            _step_output(0, np.array([100.0, 200.0]), np.array([1000.0, 500.0])),
            _step_output(1, np.array([110.0, 190.0]), np.array([1100.0, 480.0])),
        ]
        write_outputs(outputs, tmp_path, cfg)
        biomass_file = tmp_path / "osm_biomass_Simu0.csv"
        assert biomass_file.exists()

    def test_biomass_csv_format(self, tmp_path):
        cfg = EngineConfig.from_dict(_make_output_config())
        outputs = [
            _step_output(0, np.array([100.0, 200.0]), np.array([1000.0, 500.0])),
        ]
        write_outputs(outputs, tmp_path, cfg)
        biomass_file = tmp_path / "osm_biomass_Simu0.csv"
        lines = biomass_file.read_text().splitlines()
        # First line is description
        assert "biomass" in lines[0].lower()
        # Second line is header with species names
        assert "Anchovy" in lines[1]
        assert "Hake" in lines[1]
        assert "Time" in lines[1]

    def test_biomass_csv_readable_by_pandas(self, tmp_path):
        cfg = EngineConfig.from_dict(_make_output_config())
        outputs = [
            _step_output(i, np.array([100.0 + i, 200.0 - i]), np.array([1000.0, 500.0]))
            for i in range(12)
        ]
        write_outputs(outputs, tmp_path, cfg)
        df = pd.read_csv(tmp_path / "osm_biomass_Simu0.csv", skiprows=1)
        assert len(df) == 12
        assert "Anchovy" in df.columns
        assert "Hake" in df.columns
        assert "Time" in df.columns

    def test_creates_abundance_csv(self, tmp_path):
        cfg = EngineConfig.from_dict(_make_output_config())
        outputs = [
            _step_output(0, np.array([100.0, 200.0]), np.array([1000.0, 500.0])),
        ]
        write_outputs(outputs, tmp_path, cfg)
        assert (tmp_path / "osm_abundance_Simu0.csv").exists()

    def test_time_in_years(self, tmp_path):
        """Time column should be in fractional years."""
        cfg = EngineConfig.from_dict(_make_output_config())
        outputs = [
            _step_output(0, np.array([100.0, 200.0]), np.array([1000.0, 500.0])),
            _step_output(6, np.array([110.0, 190.0]), np.array([1100.0, 480.0])),
        ]
        write_outputs(outputs, tmp_path, cfg)
        df = pd.read_csv(tmp_path / "osm_biomass_Simu0.csv", skiprows=1)
        np.testing.assert_allclose(df["Time"].iloc[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(df["Time"].iloc[1], 0.5, atol=1e-6)


class TestMortalityOutput:
    def test_creates_mortality_directory(self, tmp_path):
        cfg = EngineConfig.from_dict(_make_output_config())
        outputs = [
            _step_output(0, np.array([100.0, 200.0]), np.array([1000.0, 500.0])),
        ]
        write_outputs(outputs, tmp_path, cfg)
        assert (tmp_path / "Mortality").is_dir()

    def test_creates_per_species_files(self, tmp_path):
        cfg = EngineConfig.from_dict(_make_output_config())
        mort = np.zeros((2, _N_CAUSES))
        mort[0, MortalityCause.PREDATION] = 5.0
        outputs = [
            _step_output(
                0, np.array([100.0, 200.0]), np.array([1000.0, 500.0]), mortality_by_cause=mort
            ),
        ]
        write_outputs(outputs, tmp_path, cfg)
        assert (tmp_path / "Mortality" / "osm_mortalityRate-Anchovy_Simu0.csv").exists()
        assert (tmp_path / "Mortality" / "osm_mortalityRate-Hake_Simu0.csv").exists()

    def test_mortality_csv_has_cause_columns(self, tmp_path):
        cfg = EngineConfig.from_dict(_make_output_config())
        mort = np.zeros((2, _N_CAUSES))
        mort[0, MortalityCause.FISHING] = 10.0
        outputs = [
            _step_output(
                0, np.array([100.0, 200.0]), np.array([1000.0, 500.0]), mortality_by_cause=mort
            ),
        ]
        write_outputs(outputs, tmp_path, cfg)
        df = pd.read_csv(tmp_path / "Mortality" / "osm_mortalityRate-Anchovy_Simu0.csv", skiprows=1)
        assert "Fishing" in df.columns
        assert "Predation" in df.columns
        assert df["Fishing"].iloc[0] == 10.0


class TestPythonEngineWritesOutput:
    def test_engine_run_creates_output_files(self, tmp_path):
        """PythonEngine.run() should create output CSV files."""
        from osmose.engine import PythonEngine

        config = {
            **_make_output_config(),
            "population.seeding.biomass.sp0": "50000",
            "population.seeding.biomass.sp1": "60000",
        }
        engine = PythonEngine()
        result = engine.run(config=config, output_dir=tmp_path, seed=42)
        assert result.returncode == 0
        assert (tmp_path / "osm_biomass_Simu0.csv").exists()
        assert (tmp_path / "osm_abundance_Simu0.csv").exists()
        # Read and verify non-empty
        df = pd.read_csv(tmp_path / "osm_biomass_Simu0.csv", skiprows=1)
        assert len(df) == 12  # 12 timesteps for 1 year


# ---------------------------------------------------------------------------
# NetCDF output tests (SP-4 Task 2)
# ---------------------------------------------------------------------------

# Extra raw config keys to build a valid 2-species EngineConfig.
_CFG_2SP: dict[str, str] = {
    "simulation.nspecies": "2",
    "simulation.nschool.sp0": "5",
    "simulation.nschool.sp1": "5",
    "species.name.sp1": "sp1",
    "species.lifespan.sp1": "10",
    "species.linf.sp1": "80.0",
    "species.k.sp1": "0.1",
    "species.t0.sp1": "0.0",
    "species.egg.size.sp1": "0.1",
    "species.length2weight.condition.factor.sp1": "0.01",
    "species.length2weight.allometric.power.sp1": "3.0",
    "species.vonbertalanffy.threshold.age.sp1": "0.0",
    "predation.ingestion.rate.max.sp1": "3.5",
    "predation.efficiency.critical.sp1": "0.57",
}


def _make_step_with_age(step: int, n_sp: int, bins_by_sp: dict | None = None) -> StepOutput:
    biomass_by_age = None
    abundance_by_age = None
    if bins_by_sp is not None:
        biomass_by_age = {
            sp: np.arange(bins_by_sp[sp], dtype=np.float64)
            for sp in range(n_sp)
            if sp in bins_by_sp
        }
        abundance_by_age = {
            sp: np.arange(bins_by_sp[sp], dtype=np.float64) * 100.0
            for sp in range(n_sp)
            if sp in bins_by_sp
        }
    return StepOutput(
        step=step,
        biomass=np.full(n_sp, 100.0 * (step + 1)),
        abundance=np.full(n_sp, 1000.0 * (step + 1)),
        mortality_by_cause=np.arange(n_sp * 8, dtype=np.float64).reshape(n_sp, 8),
        biomass_by_age=biomass_by_age,
        abundance_by_age=abundance_by_age,
    )


def test_netcdf_contains_biomass_by_age_when_enabled(tmp_path):
    cfg = make_minimal_engine_config(
        extra_cfg=_CFG_2SP,
        output_biomass_byage_netcdf=True,
    )
    outputs = [_make_step_with_age(t, 2, {0: 3, 1: 3}) for t in (23, 47, 71)]
    write_outputs_netcdf(outputs, tmp_path / "run_Simu0.nc", cfg)
    ds = xr.open_dataset(tmp_path / "run_Simu0.nc")
    assert "biomass_by_age" in ds.data_vars
    assert ds["biomass_by_age"].dims == ("time", "species", "age_bin")
    assert ds["biomass_by_age"].shape == (3, 2, 3)
    np.testing.assert_array_equal(ds["biomass_by_age"].values[0, 0, :], [0, 1, 2])


def test_netcdf_contains_mortality_by_cause(tmp_path):
    cfg = make_minimal_engine_config(extra_cfg=_CFG_2SP, output_mortality_netcdf=True)
    outputs = [_make_step_with_age(t, 2) for t in (23, 47)]
    write_outputs_netcdf(outputs, tmp_path / "run_Simu0.nc", cfg)
    ds = xr.open_dataset(tmp_path / "run_Simu0.nc")
    assert "mortality_by_cause" in ds.data_vars
    assert ds["mortality_by_cause"].dims == ("time", "species", "cause")
    assert list(ds.coords["cause"].values) == [
        "Predation",
        "Starvation",
        "Additional",
        "Fishing",
        "Out",
        "Foraging",
        "Discards",
        "Aging",
    ]  # capitalize() to match existing CSV writer at output.py:161


def test_netcdf_not_written_when_every_toggle_disabled(tmp_path):
    cfg = make_minimal_engine_config(
        output_biomass_netcdf=False,
        output_abundance_netcdf=False,
        output_yield_biomass_netcdf=False,
        output_biomass_byage_netcdf=False,
        output_abundance_byage_netcdf=False,
        output_biomass_bysize_netcdf=False,
        output_abundance_bysize_netcdf=False,
        output_mortality_netcdf=False,
    )
    path = tmp_path / "run_Simu0.nc"
    write_outputs_netcdf([_make_step_with_age(23, 1)], path, cfg)
    assert not path.exists()


def test_netcdf_pads_ragged_age_bins_with_nan(tmp_path):
    cfg = make_minimal_engine_config(
        extra_cfg=_CFG_2SP,
        output_biomass_byage_netcdf=True,
    )
    outputs = [_make_step_with_age(23, 2, {0: 4, 1: 2})]
    write_outputs_netcdf(outputs, tmp_path / "run_Simu0.nc", cfg)
    ds = xr.open_dataset(tmp_path / "run_Simu0.nc")
    assert ds["biomass_by_age"].shape == (1, 2, 4)
    np.testing.assert_array_equal(
        np.isnan(ds["biomass_by_age"].values[0, 1, :]),
        [False, False, True, True],
    )


def test_netcdf_pads_ragged_size_bins_with_nan(tmp_path):
    cfg = make_minimal_engine_config(
        extra_cfg=_CFG_2SP,
        output_biomass_bysize_netcdf=True,
    )
    step = StepOutput(
        step=23,
        biomass=np.array([100.0, 200.0]),
        abundance=np.array([1000.0, 2000.0]),
        mortality_by_cause=np.zeros((2, 8)),
        biomass_by_size={
            0: np.array([1.0, 2.0, 3.0, 4.0]),
            1: np.array([10.0, 20.0]),
        },
    )
    write_outputs_netcdf([step], tmp_path / "run_Simu0.nc", cfg)
    ds = xr.open_dataset(tmp_path / "run_Simu0.nc")
    assert ds["biomass_by_size"].shape == (1, 2, 4)
    np.testing.assert_array_equal(
        np.isnan(ds["biomass_by_size"].values[0, 1, :]),
        [False, False, True, True],
    )


def test_netcdf_cf_conventions_attr(tmp_path):
    cfg = make_minimal_engine_config(
        n_species=1,
        output_biomass_byage_netcdf=True,
    )
    outputs = [_make_step_with_age(23, 1, {0: 2})]
    write_outputs_netcdf(outputs, tmp_path / "run_Simu0.nc", cfg)
    ds = xr.open_dataset(tmp_path / "run_Simu0.nc")
    assert ds.attrs.get("Conventions") == "CF-1.8"
    # _FillValue surfaces on encoding after round-trip
    fv = ds["biomass_by_age"].encoding.get("_FillValue")
    assert fv is not None and np.isnan(fv)
