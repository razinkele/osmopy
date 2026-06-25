"""Tests for osmose.validation.fmsy_sweep — mode detection + species->fishery map."""

import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig
from osmose.validation.fmsy_sweep import fishing_override

BALTIC = "data/baltic/baltic_all-parameters.csv"


def _baltic_cfg() -> dict[str, str]:
    raw = dict(OsmoseConfigReader().read(BALTIC))
    raw["simulation.time.nyear"] = "2"
    return raw


def test_fisheries_mode_override_actually_changes_fishing_rate():
    raw = _baltic_cfg()
    cfg = EngineConfig.from_dict(dict(raw))
    key, base = fishing_override(raw, cfg, 0)
    assert key.startswith("fisheries.rate.base.fsh")  # baltic is v4 fisheries-mode
    assert base == pytest.approx(cfg.fishing_rate[0])
    # overriding the returned key MUST move fishing_rate[0] (the no-op-trap guard)
    bumped = dict(raw)
    bumped[key] = "9.0"
    assert EngineConfig.from_dict(bumped).fishing_rate[0] == pytest.approx(9.0)
    # and ONLY species 0 if 1:1 (baltic is 1:1)
    assert EngineConfig.from_dict(bumped).fishing_rate[1] == pytest.approx(cfg.fishing_rate[1])


def test_legacy_mode_override():
    raw = {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "Fish",
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
        "mortality.fishing.rate.method.sp0": "constant",
        "mortality.fishing.rate.sp0": "0.2",
    }
    cfg = EngineConfig.from_dict(dict(raw))
    key, base = fishing_override(raw, cfg, 0)
    assert key == "mortality.fishing.rate.sp0"
    bumped = dict(raw)
    bumped[key] = "0.9"
    assert EngineConfig.from_dict(bumped).fishing_rate[0] == pytest.approx(0.9)
