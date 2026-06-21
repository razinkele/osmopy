"""Item 1: the interleaved mortality loop knife-edges fishing selectivity types
2 (Gaussian) / 3 (log-normal). EngineConfig must warn rather than silently
diverge. (Warn, not wire -- consistent with the PR-1 reject-not-wire decision.)
"""

import logging

from osmose.engine import config as config_module
from osmose.engine.config import EngineConfig

# A minimal valid 1-species config (adapted from tests/test_engine_config_validation.py
# ::test_from_dict_still_works). The two simulation.time.* keys are REQUIRED -- from_dict
# reads simulation.time.ndtperyear unconditionally and raises KeyError without them.
_MINIMAL = {
    "simulation.time.ndtperyear": "24",
    "simulation.time.nyear": "1",
    "simulation.nspecies": "1",
    "simulation.nschool.sp0": "20",
    "species.name.sp0": "Anchovy",
    "species.linf.sp0": "15.0",
    "species.k.sp0": "0.4",
    "species.t0.sp0": "-0.1",
    "species.egg.size.sp0": "0.1",
    "species.length2weight.condition.factor.sp0": "0.006",
    "species.length2weight.allometric.power.sp0": "3.0",
    "species.lifespan.sp0": "3",
    "species.vonbertalanffy.threshold.age.sp0": "1.0",
    "mortality.subdt": "10",
    "predation.ingestion.rate.max.sp0": "3.5",
    "predation.efficiency.critical.sp0": "0.57",
}


def _fresh_config() -> EngineConfig:
    config_module._WARNED_UNSUPPORTED_MORTALITY.clear()
    return EngineConfig.from_dict(_MINIMAL)


def test_warns_on_selectivity_type_2(caplog):
    cfg = _fresh_config()
    cfg.fishing_selectivity_type[0] = 2  # Gaussian
    with caplog.at_level(logging.WARNING, logger="osmose.engine.config"):
        cfg._warn_unsupported_mortality_features()
    assert any("selectivity type 2" in r.getMessage() for r in caplog.records)


def test_warns_on_selectivity_type_3(caplog):
    cfg = _fresh_config()
    cfg.fishing_selectivity_type[0] = 3  # log-normal
    with caplog.at_level(logging.WARNING, logger="osmose.engine.config"):
        cfg._warn_unsupported_mortality_features()
    assert any(
        "selectivity type" in r.getMessage() and "3" in r.getMessage() for r in caplog.records
    )


def test_no_selectivity_warning_for_types_0_and_1(caplog):
    cfg = _fresh_config()
    cfg.fishing_selectivity_type[0] = 1  # logistic -- supported
    with caplog.at_level(logging.WARNING, logger="osmose.engine.config"):
        cfg._warn_unsupported_mortality_features()
    assert not any("selectivity type" in r.getMessage() for r in caplog.records)
