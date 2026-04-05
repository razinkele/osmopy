"""Tests for age/size distribution CSV output writers."""

import numpy as np
import pandas as pd

from osmose.engine.config import EngineConfig
from osmose.engine.output import write_outputs
from osmose.engine.simulate import StepOutput
from osmose.engine.state import MortalityCause

_N_CAUSES = len(MortalityCause)


def _make_config(extra: dict | None = None) -> dict[str, str]:
    base = {
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
    if extra:
        base.update(extra)
    return base


def _step_output(step, biomass, abundance, mortality_by_cause=None, **kwargs):
    n_sp = len(biomass)
    if mortality_by_cause is None:
        mortality_by_cause = np.zeros((n_sp, _N_CAUSES), dtype=np.float64)
    return StepOutput(
        step=step,
        biomass=biomass,
        abundance=abundance,
        mortality_by_cause=mortality_by_cause,
        **kwargs,
    )


def _make_age_dist(n_sp: int, n_age_bins: int) -> dict[int, np.ndarray]:
    return {sp: np.arange(n_age_bins, dtype=np.float64) * (sp + 1) for sp in range(n_sp)}


def _make_size_dist(n_sp: int, n_size_bins: int) -> dict[int, np.ndarray]:
    return {sp: np.ones(n_size_bins, dtype=np.float64) * (sp + 1) for sp in range(n_sp)}


class TestDistributionCSVOutput:
    def test_biomass_by_age_csv_written(self, tmp_path):
        """write_outputs should create biomassByAge CSVs when data is present."""
        cfg = EngineConfig.from_dict(_make_config())
        n_age_bins = 5
        age_dist = _make_age_dist(2, n_age_bins)
        outputs = [
            _step_output(
                t,
                np.array([100.0, 200.0]),
                np.array([1000.0, 500.0]),
                biomass_by_age=age_dist,
                abundance_by_age=age_dist,
            )
            for t in range(3)
        ]
        write_outputs(outputs, tmp_path, cfg)
        assert (tmp_path / "osmose_biomassByAge_Anchovy_Simu0.csv").exists()
        assert (tmp_path / "osmose_biomassByAge_Hake_Simu0.csv").exists()

    def test_abundance_by_age_csv_written(self, tmp_path):
        """write_outputs should create abundanceByAge CSVs when data is present."""
        cfg = EngineConfig.from_dict(_make_config())
        n_age_bins = 5
        age_dist = _make_age_dist(2, n_age_bins)
        outputs = [
            _step_output(
                t,
                np.array([100.0, 200.0]),
                np.array([1000.0, 500.0]),
                abundance_by_age=age_dist,
            )
            for t in range(2)
        ]
        write_outputs(outputs, tmp_path, cfg)
        assert (tmp_path / "osmose_abundanceByAge_Anchovy_Simu0.csv").exists()
        assert (tmp_path / "osmose_abundanceByAge_Hake_Simu0.csv").exists()

    def test_csv_has_correct_headers_age(self, tmp_path):
        """biomassByAge CSV should have Time column + integer age bin columns."""
        cfg = EngineConfig.from_dict(_make_config())
        n_age_bins = 4
        age_dist = _make_age_dist(2, n_age_bins)
        outputs = [
            _step_output(
                t,
                np.array([100.0, 200.0]),
                np.array([1000.0, 500.0]),
                biomass_by_age=age_dist,
            )
            for t in range(2)
        ]
        write_outputs(outputs, tmp_path, cfg)
        df = pd.read_csv(tmp_path / "osmose_biomassByAge_Anchovy_Simu0.csv")
        assert "Time" in df.columns
        # Age bin columns should be "0", "1", "2", "3"
        for i in range(n_age_bins):
            assert str(i) in df.columns

    def test_csv_has_correct_headers_size(self, tmp_path):
        """biomassBySize CSV should have Time + size-edge float columns."""
        cfg = EngineConfig.from_dict(
            _make_config(
                {
                    "output.distrib.bySize.min": "0.0",
                    "output.distrib.bySize.max": "5.0",
                    "output.distrib.bySize.incr": "1.0",
                }
            )
        )
        n_size_bins = 5
        size_dist = _make_size_dist(2, n_size_bins)
        outputs = [
            _step_output(
                t,
                np.array([100.0, 200.0]),
                np.array([1000.0, 500.0]),
                biomass_by_size=size_dist,
                abundance_by_size=size_dist,
            )
            for t in range(2)
        ]
        write_outputs(outputs, tmp_path, cfg)
        df = pd.read_csv(tmp_path / "osmose_biomassBySize_Anchovy_Simu0.csv")
        assert "Time" in df.columns
        # Size columns should be formatted as floats e.g. "0.0", "1.0", ...
        assert "0.0" in df.columns
        assert "10.0" in df.columns  # second size bin edge

    def test_csv_data_values_correct(self, tmp_path):
        """CSV data values should match the distribution arrays."""
        cfg = EngineConfig.from_dict(_make_config())
        # Species 0 gets [0, 10, 20], species 1 gets [0, 20, 40]
        age_dist_t0 = {0: np.array([0.0, 10.0, 20.0]), 1: np.array([0.0, 20.0, 40.0])}
        age_dist_t1 = {0: np.array([1.0, 11.0, 21.0]), 1: np.array([1.0, 21.0, 41.0])}
        outputs = [
            _step_output(
                0,
                np.array([100.0, 200.0]),
                np.array([1000.0, 500.0]),
                biomass_by_age=age_dist_t0,
            ),
            _step_output(
                1,
                np.array([110.0, 190.0]),
                np.array([1100.0, 480.0]),
                biomass_by_age=age_dist_t1,
            ),
        ]
        write_outputs(outputs, tmp_path, cfg)
        df = pd.read_csv(tmp_path / "osmose_biomassByAge_Anchovy_Simu0.csv")
        assert len(df) == 2
        np.testing.assert_allclose(df["1"].iloc[0], 10.0)
        np.testing.assert_allclose(df["1"].iloc[1], 11.0)

    def test_time_column_in_years(self, tmp_path):
        """Time column should be in fractional years."""
        cfg = EngineConfig.from_dict(_make_config())
        n_age_bins = 2
        age_dist = _make_age_dist(2, n_age_bins)
        outputs = [
            _step_output(
                step,
                np.array([100.0, 200.0]),
                np.array([1000.0, 500.0]),
                biomass_by_age=age_dist,
            )
            for step in [0, 6, 12]
        ]
        write_outputs(outputs, tmp_path, cfg)
        df = pd.read_csv(tmp_path / "osmose_biomassByAge_Anchovy_Simu0.csv")
        np.testing.assert_allclose(df["Time"].iloc[0], 0.0, atol=1e-9)
        np.testing.assert_allclose(df["Time"].iloc[1], 0.5, atol=1e-9)
        np.testing.assert_allclose(df["Time"].iloc[2], 1.0, atol=1e-9)

    def test_no_csv_when_disabled(self, tmp_path):
        """No distribution CSVs written when distribution dicts are None."""
        cfg = EngineConfig.from_dict(_make_config())
        outputs = [
            _step_output(t, np.array([100.0, 200.0]), np.array([1000.0, 500.0])) for t in range(3)
        ]
        write_outputs(outputs, tmp_path, cfg)
        # None of the distribution files should exist
        for label in ["biomassByAge", "abundanceByAge", "biomassBySize", "abundanceBySize"]:
            for sp in ["Anchovy", "Hake"]:
                assert not (tmp_path / f"osmose_{label}_{sp}_Simu0.csv").exists()

    def test_biomass_by_size_csv_written(self, tmp_path):
        """write_outputs should create biomassBySize CSVs when data is present."""
        cfg = EngineConfig.from_dict(_make_config())
        n_size_bins = 6
        size_dist = _make_size_dist(2, n_size_bins)
        outputs = [
            _step_output(
                t,
                np.array([100.0, 200.0]),
                np.array([1000.0, 500.0]),
                biomass_by_size=size_dist,
            )
            for t in range(2)
        ]
        write_outputs(outputs, tmp_path, cfg)
        assert (tmp_path / "osmose_biomassBySize_Anchovy_Simu0.csv").exists()
        assert (tmp_path / "osmose_biomassBySize_Hake_Simu0.csv").exists()

    def test_mixed_none_and_data_steps(self, tmp_path):
        """Steps with None distribution should contribute zeros to the matrix."""
        cfg = EngineConfig.from_dict(_make_config())
        age_dist = {0: np.array([5.0, 10.0, 15.0]), 1: np.array([2.0, 4.0, 6.0])}
        outputs = [
            _step_output(
                0,
                np.array([100.0, 200.0]),
                np.array([1000.0, 500.0]),
                biomass_by_age=age_dist,
            ),
            _step_output(
                1,
                np.array([110.0, 190.0]),
                np.array([1100.0, 480.0]),
                biomass_by_age=None,  # no data for this step
            ),
        ]
        write_outputs(outputs, tmp_path, cfg)
        df = pd.read_csv(tmp_path / "osmose_biomassByAge_Anchovy_Simu0.csv")
        assert len(df) == 2
        # Second row should be zeros (None data contributes zeros)
        np.testing.assert_allclose(df["0"].iloc[1], 0.0)
        np.testing.assert_allclose(df["1"].iloc[1], 0.0)


class TestDistributionCollectOutputs:
    """Integration tests: _collect_outputs populates distribution fields."""

    def _make_engine_config(self, extra: dict | None = None) -> EngineConfig:
        base = _make_config(extra)
        return EngineConfig.from_dict(base)

    def test_collect_outputs_age_disabled_by_default(self):
        """Distribution dicts should be None when flags are off."""
        from osmose.engine.simulate import _collect_outputs
        from osmose.engine.state import SchoolState

        cfg = self._make_engine_config()
        state = SchoolState.create(n_schools=0, species_id=np.array([], dtype=np.int32))
        out = _collect_outputs(state, cfg, 0)
        assert out.biomass_by_age is None
        assert out.abundance_by_age is None
        assert out.biomass_by_size is None
        assert out.abundance_by_size is None

    def test_collect_outputs_age_enabled(self):
        """Distribution dicts should be populated when byage flags are on."""
        from osmose.engine.simulate import _collect_outputs
        from osmose.engine.state import SchoolState

        cfg = self._make_engine_config(
            {
                "output.biomass.byage.enabled": "true",
                "output.abundance.byage.enabled": "true",
            }
        )
        # Empty state — distributions should be all-zero dicts
        state = SchoolState.create(n_schools=0, species_id=np.array([], dtype=np.int32))
        out = _collect_outputs(state, cfg, 0)
        assert out.biomass_by_age is not None
        assert out.abundance_by_age is not None
        assert 0 in out.biomass_by_age
        assert 1 in out.biomass_by_age
        np.testing.assert_array_equal(out.biomass_by_age[0], 0.0)

    def test_collect_outputs_size_enabled(self):
        """Size distribution dicts should be populated when bysize flags are on."""
        from osmose.engine.simulate import _collect_outputs
        from osmose.engine.state import SchoolState

        cfg = self._make_engine_config(
            {
                "output.biomass.bysize.enabled": "true",
                "output.abundance.bysize.enabled": "true",
            }
        )
        state = SchoolState.create(n_schools=0, species_id=np.array([], dtype=np.int32))
        out = _collect_outputs(state, cfg, 0)
        assert out.biomass_by_size is not None
        assert out.abundance_by_size is not None
        assert 0 in out.biomass_by_size
