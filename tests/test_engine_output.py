"""Tests for engine output writer — CSV format matching Java."""

import numpy as np
import pandas as pd

from osmose.engine.config import EngineConfig
from osmose.engine.output import write_outputs
from osmose.engine.simulate import StepOutput
from osmose.engine.state import MortalityCause

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
        biomass_file = tmp_path / "osmose_biomass_Simu0.csv"
        assert biomass_file.exists()

    def test_biomass_csv_format(self, tmp_path):
        cfg = EngineConfig.from_dict(_make_output_config())
        outputs = [
            _step_output(0, np.array([100.0, 200.0]), np.array([1000.0, 500.0])),
        ]
        write_outputs(outputs, tmp_path, cfg)
        biomass_file = tmp_path / "osmose_biomass_Simu0.csv"
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
        df = pd.read_csv(tmp_path / "osmose_biomass_Simu0.csv", skiprows=1)
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
        assert (tmp_path / "osmose_abundance_Simu0.csv").exists()

    def test_time_in_years(self, tmp_path):
        """Time column should be in fractional years."""
        cfg = EngineConfig.from_dict(_make_output_config())
        outputs = [
            _step_output(0, np.array([100.0, 200.0]), np.array([1000.0, 500.0])),
            _step_output(6, np.array([110.0, 190.0]), np.array([1100.0, 480.0])),
        ]
        write_outputs(outputs, tmp_path, cfg)
        df = pd.read_csv(tmp_path / "osmose_biomass_Simu0.csv", skiprows=1)
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
        assert (tmp_path / "Mortality" / "osmose_mortalityRate-Anchovy_Simu0.csv").exists()
        assert (tmp_path / "Mortality" / "osmose_mortalityRate-Hake_Simu0.csv").exists()

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
        df = pd.read_csv(
            tmp_path / "Mortality" / "osmose_mortalityRate-Anchovy_Simu0.csv", skiprows=1
        )
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
        assert (tmp_path / "osmose_biomass_Simu0.csv").exists()
        assert (tmp_path / "osmose_abundance_Simu0.csv").exists()
        # Read and verify non-empty
        df = pd.read_csv(tmp_path / "osmose_biomass_Simu0.csv", skiprows=1)
        assert len(df) == 12  # 12 timesteps for 1 year
