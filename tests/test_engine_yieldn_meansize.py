import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.simulate import _collect_mean_size, _collect_yield_n
from osmose.engine.state import MortalityCause, SchoolState


def _base_cfg() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "20.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "1",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "movement.distribution.method.sp0": "random",
        "movement.randomwalk.range.sp0": "1",
    }


def test_config_parses_yieldn_meansize_flags():
    cfg = EngineConfig.from_dict(
        {
            **_base_cfg(),
            "output.yield.abundance.enabled": "true",
            "output.size.enabled": "true",
            "output.yield.abundance.netcdf.enabled": "true",
            "output.size.netcdf.enabled": "true",
        }
    )
    assert cfg.output_yield_abundance is True
    assert cfg.output_mean_size is True
    assert cfg.output_yield_abundance_netcdf is True
    assert cfg.output_mean_size_netcdf is True


def test_config_yieldn_meansize_flags_default_false():
    cfg = EngineConfig.from_dict(_base_cfg())
    assert cfg.output_yield_abundance is False
    assert cfg.output_mean_size is False
    assert cfg.output_yield_abundance_netcdf is False
    assert cfg.output_mean_size_netcdf is False


def _two_school_state() -> SchoolState:
    # 2 focal schools of species 0: lengths 10 & 30, abundance 100 & 300,
    # fishing deaths 4 & 6.
    s = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
    s = s.replace(
        length=np.array([10.0, 30.0]),
        abundance=np.array([100.0, 300.0]),
        weight=np.array([0.01, 0.27]),
        biomass=np.array([1.0, 81.0]),
        age_dt=np.array([12, 24], dtype=np.int32),
    )
    nd = s.n_dead.copy()
    nd[:, int(MortalityCause.FISHING)] = np.array([4.0, 6.0])
    return s.replace(n_dead=nd)


class _Cfg:
    n_species = 1
    n_dt_per_year = 12
    output_cutoff_age = None


def test_collect_yield_n_is_fishing_deaths_in_numbers():
    yn = _collect_yield_n(_two_school_state(), _Cfg())
    assert yn.shape == (1,)
    assert yn[0] == pytest.approx(10.0)  # 4 + 6 deaths, NO weight


def test_collect_mean_size_abundance_weighted():
    ms = _collect_mean_size(_two_school_state(), _Cfg())
    # (100*10 + 300*30) / (100+300) = 10000/400 = 25.0
    assert ms[0] == pytest.approx(25.0)


def test_collect_mean_size_applies_cutoff_and_omits_empty():
    class CutCfg(_Cfg):
        output_cutoff_age = np.array(
            [1.5]
        )  # cutoff 1.5 yr; age_dt=12 is 1.0 yr (excluded), age_dt=24 is 2.0 yr (included)

    ms = _collect_mean_size(_two_school_state(), CutCfg())
    # only the age_dt=24 (2 yr) school survives → mean length = 30
    assert ms[0] == pytest.approx(30.0)
    # a state with no qualifying school → species omitted
    empty = _collect_mean_size(SchoolState.create(0), _Cfg())
    assert empty == {}
