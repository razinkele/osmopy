"""#120: the Python engine warns (does not silently ignore) when a config requests restart."""

import logging

import pytest

from osmose.engine.config import _WARNED_UNSUPPORTED_RESTART
from osmose.engine.config import EngineConfig


@pytest.fixture(autouse=True)
def _clear_restart_warning_cache():
    """Dedup set is process-global; clear before each test so assertions aren't suppressed."""
    _WARNED_UNSUPPORTED_RESTART.clear()
    yield


def _base_cfg() -> dict[str, str]:
    # A fresh raw dict: NO osmose.version, NO native simulation.restart.* — so the old-spelling
    # test (below) actually renames + fires. (data/minimal can't be reused: it stamps
    # osmose.version=4.4.1 AND sets a native simulation.restart.enabled=false, either of which
    # suppresses the old-spelling warning — verified in the design spec.)
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


def _restart_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and "restart" in r.getMessage().lower()
    ]


def test_restart_file_warns_resume(caplog):
    cfg = _base_cfg() | {"simulation.restart.file": "snap.nc"}
    with caplog.at_level(logging.WARNING):
        EngineConfig.from_dict(cfg)
    msgs = _restart_warnings(caplog)
    assert any("cold-start" in m.lower() and "snap.nc" in m for m in msgs), msgs


def test_restart_enabled_warns_write(caplog):
    cfg = _base_cfg() | {"simulation.restart.enabled": "true"}
    with caplog.at_level(logging.WARNING):
        EngineConfig.from_dict(cfg)
    msgs = _restart_warnings(caplog)
    assert any("output" in m.lower() for m in msgs), msgs


def test_both_keys_warn_distinctly(caplog):
    cfg = _base_cfg() | {"simulation.restart.file": "snap.nc", "simulation.restart.enabled": "true"}
    with caplog.at_level(logging.WARNING):
        EngineConfig.from_dict(cfg)
    msgs = _restart_warnings(caplog)
    assert len(set(msgs)) == 2, msgs  # two distinct messages


def test_no_restart_no_warning(caplog):
    with caplog.at_level(logging.WARNING):
        EngineConfig.from_dict(_base_cfg())
    assert _restart_warnings(caplog) == []


def test_restart_enabled_false_and_null_file_no_warning(caplog):
    cfg = _base_cfg() | {"simulation.restart.enabled": "false", "simulation.restart.file": "null"}
    with caplog.at_level(logging.WARNING):
        EngineConfig.from_dict(cfg)
    assert _restart_warnings(caplog) == []


def test_old_spelling_output_restart_enabled_warns(caplog):
    # output.restart.enabled -> (canonicalize) -> simulation.restart.enabled. The post-canonicalize
    # check catches it. Uses the fresh dict (no version, no native key) so the rename fires.
    cfg = _base_cfg() | {"output.restart.enabled": "true"}
    with caplog.at_level(logging.WARNING):
        EngineConfig.from_dict(cfg)
    assert any("output" in m.lower() for m in _restart_warnings(caplog))


def test_dedup_warns_once(caplog):
    cfg = _base_cfg() | {"simulation.restart.file": "snap.nc"}
    with caplog.at_level(logging.WARNING):
        EngineConfig.from_dict(cfg)
        EngineConfig.from_dict(cfg)
    # dedup NOT cleared between the two calls (autouse clears only before the test) -> once
    assert len(_restart_warnings(caplog)) == 1
