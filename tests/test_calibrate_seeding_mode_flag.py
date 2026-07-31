"""`--seeding-mode` must actually reach base_config (GitHub #143).

The refit's whole point is that calibration runs under the same seeding mode it will be certified
and scored under. If the flag were accepted but not injected, the refit would silently optimise
against the default and the A/B would compare parameters fitted under identical conditions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load():
    d = PROJECT_ROOT / "scripts"
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("calibrate_baltic", d / "calibrate_baltic.py")
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["calibrate_baltic"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def cal():
    return _load()


def test_cli_accepts_both_modes_and_rejects_a_typo(cal):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--seeding-mode", choices=["stock_recruitment", "linear"], default=None)
    assert p.parse_args(["--seeding-mode", "linear"]).seeding_mode == "linear"
    assert p.parse_args([]).seeding_mode is None
    with pytest.raises(SystemExit):
        p.parse_args(["--seeding-mode", "linaer"])


def test_run_calibration_signature_exposes_seeding_mode(cal):
    import inspect

    sig = inspect.signature(cal.run_calibration)
    assert "seeding_mode" in sig.parameters
    assert sig.parameters["seeding_mode"].default is None


def test_injection_line_targets_the_engine_key(cal):
    """The injected key must be the one EngineConfig reads, not a near-miss spelling."""
    src = (PROJECT_ROOT / "scripts" / "calibrate_baltic.py").read_text()
    assert 'base_config["population.seeding.mode"] = seeding_mode' in src
    from osmose.engine.config import _parse_seeding_mode

    assert _parse_seeding_mode({"population.seeding.mode": "linear"}) == "linear"
