"""Tests for Python-engine community outputs: DistribBySize + meanTL."""

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.simulate import _collect_mean_tl
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
