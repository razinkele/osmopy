# tests/test_calibrate_baltic_a2r.py
"""Bounded recalibration phase a2r: 6 params only, gate-species objective, depletion base.

Spec 2026-08-08 §4 Phase 1 item 4: regrowth + zooplanktivore availabilities ONLY —
this file is the guard that the phase stays bounded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import calibrate_baltic as cal  # noqa: E402

GATE_SPECIES = {"cod_west", "cod_east", "herring", "sprat", "flounder", "perch", "stickleback"}


def test_a2r_params_exactly_six():
    keys, bounds, x0 = cal.get_a2r_params()
    assert keys == [
        "species.regrowth.rate.zoo3",
        "species.regrowth.rate.benthos",
        "species.accessibility2fish.sp9",
        "species.accessibility2fish.sp11",
        "species.accessibility2fish.sp12",
        "species.accessibility2fish.sp13",
    ]
    assert len(bounds) == len(x0) == 6
    # zoo3: 0.05..2.0 per step in log10; x0 at the carried-over prior
    assert np.isclose(bounds[0][0], np.log10(0.05)) and np.isclose(bounds[0][1], np.log10(2.0))
    assert np.isclose(x0[0], np.log10(0.911553421016705))
    # benthos: 0.01..0.316 per step in log10; x0 at the literature rate
    assert np.isclose(bounds[1][0], -2.0) and np.isclose(bounds[1][1], -0.5)
    assert np.isclose(x0[1], np.log10(0.03))
    # accessibilities: x0 at the current config value 0.8
    for i in (2, 3, 4, 5):
        assert np.isclose(x0[i], np.log10(0.8))


def test_a2r_sentinels_expand_to_current_layout():
    keys, _, x0 = cal.get_a2r_params()
    overrides = cal.expand_param_overrides(keys, x0)
    assert set(overrides) == {
        "species.regrowth.rate.sp11",
        "species.regrowth.rate.sp12",
        "species.regrowth.rate.sp13",
        "species.regrowth.rate.sp14",
        "species.accessibility2fish.sp9",
        "species.accessibility2fish.sp11",
        "species.accessibility2fish.sp12",
        "species.accessibility2fish.sp13",
    }
    # zoo3 must NOT touch benthos; benthos sentinel owns sp14
    assert np.isclose(float(overrides["species.regrowth.rate.sp11"]), 0.911553421016705)
    assert np.isclose(float(overrides["species.regrowth.rate.sp14"]), 0.03)
    # legacy sentinel untouched: still expands sp11..sp14 as before
    legacy = cal.expand_param_overrides(["species.regrowth.rate.zoo"], [np.log10(0.5)])
    assert set(legacy) == {f"species.regrowth.rate.sp{i}" for i in (11, 12, 13, 14)}


def test_a2r_targets_exclude_indicative_overshoots():
    targets = cal.get_a2r_targets()
    names = {t.species for t in targets}
    assert names == GATE_SPECIES


def test_a2r_guard_rejects_hybrid():
    """Verify that run_calibration guards against --a2 + --phase a2r hybrid (silent bug fix)."""
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        cal.run_calibration(phase="a2r", a2=True)
    assert "--a2" in str(exc_info.value)
    assert "a2r" in str(exc_info.value)
