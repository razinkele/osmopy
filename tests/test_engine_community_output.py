"""Tests for Python-engine community outputs: DistribBySize + meanTL."""

import numpy as np
import pandas as pd
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.output import (
    _build_distrib_bysize_community_dataframes,
    _build_meantl_dataframe,
    write_outputs,
)
from osmose.engine.simulate import StepOutput, _collect_mean_tl
from osmose.engine.state import SchoolState


def _base_config(extra: dict | None = None) -> dict[str, str]:
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


def test_output_meantl_flag_defaults_false():
    config = EngineConfig.from_dict(_base_config())
    assert config.output_meantl is False


def test_output_meantl_flag_enabled():
    # NOTE: the key is LOWERCASE. The real OsmoseConfigReader lowercases every config key
    # (reader.py:157), and config.py reads the flag with the lowercase literal. _base_config
    # builds a hand dict that bypasses the reader, so it MUST use the lowercase key here.
    config = EngineConfig.from_dict(_base_config({"output.meantl.enabled": "true"}))
    assert config.output_meantl is True


def test_collect_mean_tl_biomass_weighted_excludes_zero_tl():
    config = EngineConfig.from_dict(_base_config())  # 2 focal species, no cutoff age
    # sp0 two schools with UNEQUAL per-school weight so abundance- and biomass-weighting DIFFER
    # (this pins biomass-weighting; an equal-weight fixture would pass under either and lock nothing):
    #   school A: biomass 30, TL 4.0 ; school B: biomass 50, TL 2.0
    #   biomass-weighted  = (4*30 + 2*50) / (30+50) = 220/80 = 2.75   <-- Java convention (asserted)
    #   abundance-weighted would be (4*30 + 2*10)/40 = 3.5            <-- must NOT be the result
    # sp1 one school TL 0.0 (unfed/egg) -> excluded -> sp1 absent from the dict.
    state = SchoolState.create(3).replace(
        species_id=np.array([0, 0, 1], dtype=np.int32),
        abundance=np.array([30.0, 10.0, 50.0], dtype=np.float64),
        biomass=np.array([30.0, 50.0, 50.0], dtype=np.float64),
        trophic_level=np.array([4.0, 2.0, 0.0], dtype=np.float64),
        age_dt=np.array([12, 12, 12], dtype=np.int32),
    )
    out = _collect_mean_tl(state, config)
    assert out[0] == pytest.approx(2.75)  # biomass-weighted (NOT 3.5 abundance-weighted)
    assert 1 not in out  # sp1's only school has TL 0 -> excluded


def test_collect_mean_tl_empty_state():
    config = EngineConfig.from_dict(_base_config())
    assert _collect_mean_tl(SchoolState.create(0), config) == {}


def _step_output(step, biomass, abundance, **kwargs):
    n_sp = len(biomass)
    from osmose.engine.state import MortalityCause

    return StepOutput(
        step=step,
        biomass=biomass,
        abundance=abundance,
        mortality_by_cause=np.zeros((n_sp, len(MortalityCause)), dtype=np.float64),
        **kwargs,
    )


def test_build_meantl_dataframe_wide():
    config = EngineConfig.from_dict(_base_config({"output.meantl.enabled": "true"}))
    outputs = [
        _step_output(0, np.array([1.0, 1.0]), np.array([1.0, 1.0]), mean_tl={0: 3.5, 1: 4.2}),
        _step_output(12, np.array([1.0, 1.0]), np.array([1.0, 1.0]), mean_tl={0: 3.6}),
    ]
    dfs = _build_meantl_dataframe(outputs, config)
    df = dfs["meanTL"]
    assert list(df.columns) == ["Time", "Anchovy", "Hake"]
    assert df["Anchovy"].tolist() == pytest.approx([3.5, 3.6])
    assert df["Hake"][0] == pytest.approx(4.2)
    assert np.isnan(df["Hake"][1])  # absent that step -> NaN


def test_write_meantl_csv_gated(tmp_path):
    config_off = EngineConfig.from_dict(_base_config())  # flag off
    outputs = [_step_output(0, np.array([1.0, 1.0]), np.array([1.0, 1.0]), mean_tl={0: 3.5})]
    write_outputs(outputs, tmp_path, config_off, prefix="osm")
    assert not (tmp_path / "osm_meanTL_Simu0.csv").exists()

    config_on = EngineConfig.from_dict(_base_config({"output.meantl.enabled": "true"}))
    write_outputs(outputs, tmp_path, config_on, prefix="osm")
    written = pd.read_csv(tmp_path / "osm_meanTL_Simu0.csv")
    assert "Anchovy" in written.columns and written["Anchovy"][0] == pytest.approx(3.5)


def test_build_distrib_bysize_community():
    config = EngineConfig.from_dict(
        _base_config(
            {
                "output.biomass.bysize.enabled": "true",
                "output.distrib.bysize.min": "0",
                "output.distrib.bysize.incr": "10",
            }
        )
    )
    # 1 step, 3 size bins; sp0 = [1,2,3], sp1 = [4,5,6].
    outputs = [
        _step_output(
            0,
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            biomass_by_size={0: np.array([1.0, 2.0, 3.0]), 1: np.array([4.0, 5.0, 6.0])},
            abundance_by_size={0: np.array([1.0, 2.0, 3.0]), 1: np.array([4.0, 5.0, 6.0])},
        )
    ]
    dfs = _build_distrib_bysize_community_dataframes(outputs, config)
    df = dfs["biomassDistribBySize"]
    assert list(df.columns) == ["Time", "Size", "Anchovy", "Hake"]
    assert df["Size"].tolist() == pytest.approx([0.0, 10.0, 20.0])
    assert df["Anchovy"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert df["Hake"].tolist() == pytest.approx([4.0, 5.0, 6.0])
    # abundance flag is off in this config -> only the biomass community file is built
    assert "abundanceDistribBySize" not in dfs


def test_distrib_bysize_community_written_and_readable(tmp_path):
    from osmose.size_spectrum import _read_community_by_size

    config = EngineConfig.from_dict(_base_config({"output.biomass.bysize.enabled": "true"}))
    outputs = [
        _step_output(
            0,
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            biomass_by_size={0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])},
            abundance_by_size={0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])},
        )
    ]
    write_outputs(outputs, tmp_path, config, prefix="osm")
    wide = _read_community_by_size(tmp_path, "biomassDistribBySize", "osm")
    assert list(wide.columns) == ["Time", "Size", "Anchovy", "Hake"]
