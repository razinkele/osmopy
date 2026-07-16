"""Smoke test for the depensation placement harness (SP1 Unit 2). The full grid sweep is a
multi-decade real-engine run (Task 8, hours) — these only check the plumbing imports and that a
short SSB series comes back, CI-skipped where the engine runs."""

import os
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))


def test_gate_overrides_shape():
    from calibrate_depensation_bistability import gate_overrides

    ov = gate_overrides(90_000.0, 4.0)
    assert ov["reproduction.depensation.gate.enabled"] == "true"
    assert ov["reproduction.depensation.gate.species.enabled.sp0"] == "true"
    assert ov["reproduction.depensation.gate.s50.sp0"] == "90000.0"
    assert ov["reproduction.depensation.gate.theta.sp0"] == "4.0"
    assert ov["output.ssb.enabled"] == "true"


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="multi-decade real-engine run")
def test_cod_ssb_series_runs_short():
    from calibrate_depensation_bistability import (
        cod_ssb_series,
        read_base_config,
        read_base_larva_rates,
    )

    base = read_base_config()
    s = cod_ssb_series(base, read_base_larva_rates(base), 0.85, True, 90_000.0, 4.0, 0, n_year=5)
    assert len(s) > 0 and float(s.min()) >= 0.0
