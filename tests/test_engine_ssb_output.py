import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.simulate import _collect_ssb
from osmose.engine.state import SchoolState


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
        "species.maturity.size.sp0": "12.0",
    }


def test_config_parses_ssb_flags():
    cfg = EngineConfig.from_dict(
        {**_base_cfg(), "output.ssb.enabled": "true", "output.ssb.netcdf.enabled": "true"}
    )
    assert cfg.output_ssb is True
    assert cfg.output_ssb_netcdf is True
    assert EngineConfig.from_dict(_base_cfg()).output_ssb is False


def test_collect_ssb_uses_maturity_conjunction():
    # 3 schools of sp0, maturity_size=12, maturity_age_dt=0 (size-only):
    # lengths 8/15/20 (school0 immature by size), abundance 100 each, weight 0.01/0.05/0.1
    s = SchoolState.create(n_schools=3, species_id=np.zeros(3, dtype=np.int32))
    s = s.replace(
        length=np.array([8.0, 15.0, 20.0]),
        abundance=np.array([100.0, 100.0, 100.0]),
        weight=np.array([0.01, 0.05, 0.1]),
        age_dt=np.array([6, 12, 24], dtype=np.int32),
    )

    class Cfg:
        n_species = 1
        maturity_size = np.array([12.0])
        maturity_age_dt = np.array([0], dtype=np.int32)

    ssb = _collect_ssb(s, Cfg())
    # mature = length>=12 AND age_dt>=0 AND abundance>0 → schools 1,2: 100*0.05 + 100*0.1 = 15.0
    assert ssb[0] == pytest.approx(15.0)
