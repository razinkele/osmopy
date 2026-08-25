"""CI-safe helper tests for the C1 thermal-knob A/B harness (no engine runs).

Covers write_arm_series, expected_factors, and arm_overrides in isolation —
the machinery the full A/B (scripts/baltic_c1_knob_ab.py::run_ab, Task 6, not
exercised here) is built on. Loaded via importlib since scripts/ is not a
package (matches the F1 harness precedent's test style).
"""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "baltic_c1_knob_ab",
    Path(__file__).resolve().parent.parent / "scripts" / "baltic_c1_knob_ab.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_arm_series_layout(tmp_path):
    p = tmp_path / "arm.csv"
    m.write_arm_series(p, trefs={0: 16.0, 1: 8.0}, dT=2.0)
    lines = p.read_text().splitlines()
    assert lines[0] == "year,temp_sp0,temp_sp1"
    assert lines[1] == "1974,16.0,8.0"  # spin-up at tref in EVERY arm
    assert lines[19] == "1992,16.0,8.0"
    assert lines[20] == "1993,18.0,10.0"  # +dT on the historical block only
    assert len(lines) == 51 and "#" not in p.read_text()


def test_expected_factors():
    f = m.expected_factors(beta=-0.51, dT=2.0)
    assert (f[:19] == 1.0).all()
    assert np.allclose(f[19:], np.exp(-1.02))
    assert (m.expected_factors(beta=-0.51, dT=0.0) == 1.0).all()  # identity arm


def test_arm_overrides_identity_vs_scenario(tmp_path):
    p = tmp_path / "arm.csv"
    m.write_arm_series(p, {1: 8.0}, dT=0.0)
    base = m.arm_overrides("knob", str(p), trefs={1: 8.0}, betas={1: -0.51}, enabled=(1,))
    assert base["reproduction.thermal.gate.enabled"] == "true"
    assert base["reproduction.thermal.gate.response"] == "exponential"
    assert base["reproduction.thermal.gate.beta.sp1"] == "-0.51"
    assert base["reproduction.thermal.gate.tref.sp1"] == "8.0"
    off = m.arm_overrides("off", str(p), {1: 8.0}, {1: -0.51}, (1,))
    assert "reproduction.thermal.gate.enabled" not in off
