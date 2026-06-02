"""Fast unit tests for the FR process diagnostic + shepherd-fr eval mode.

The full simulation-based diagnostic (scripts/fr_process_diagnostic.py) is
exercised by a manual smoke run, not in CI (too slow). Here we cover the pure
helpers + the config injection, which are fast and deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def test_realized_mortality_basic():
    """M_q = eaten_q / biomass_q / n_years, elementwise."""
    from fr_process_diagnostic import realized_mortality

    eaten = np.array([100.0, 50.0, 0.0])
    biomass = np.array([1000.0, 500.0, 200.0])
    mort = realized_mortality(eaten, biomass, n_years=10.0)
    # 100/1000/10 = 0.01 ; 50/500/10 = 0.01 ; 0/200/10 = 0.0
    np.testing.assert_allclose(mort, [0.01, 0.01, 0.0])


def test_realized_mortality_zero_biomass_is_zero():
    """Prey with non-positive biomass yields 0.0 (no defined rate, no div-by-0)."""
    from fr_process_diagnostic import realized_mortality

    eaten = np.array([100.0, 100.0])
    biomass = np.array([0.0, -5.0])
    mort = realized_mortality(eaten, biomass, n_years=5.0)
    np.testing.assert_array_equal(mort, [0.0, 0.0])


def test_realized_mortality_n_years_scales():
    """Doubling the window halves the per-year rate."""
    from fr_process_diagnostic import realized_mortality

    eaten = np.array([100.0])
    biomass = np.array([1000.0])
    m1 = realized_mortality(eaten, biomass, n_years=1.0)
    m2 = realized_mortality(eaten, biomass, n_years=2.0)
    np.testing.assert_allclose(m2, m1 / 2.0)


def test_predator_slot_map():
    """The 4 FR predators map to the documented runtime diet-row slots."""
    from fr_process_diagnostic import PREDATOR_SLOTS

    # cod sp0 -> 0, pikeperch sp5 -> 5, GreySeal sp14 -> 8, Cormorant sp15 -> 9.
    assert PREDATOR_SLOTS == {0: 0, 5: 5, 14: 8, 15: 9}


def test_shepherd_fr_mode_injects_type3_and_default_k():
    """--mode shepherd-fr sets shepherd SR on all 8 species + type3 FR on the 4
    predators, defaulting halfsat to 1.0 when the params JSON lacks the key."""
    from evaluate_calibration_vs_ices import FR_PREDATOR_SP, _apply_mode

    cfg: dict[str, str] = {}
    _apply_mode(cfg, "shepherd-fr", params={})  # no halfsat keys -> default K=1.0

    # Shepherd SR injected on all 8 focal species + cod ssb_half pin.
    for i in range(8):
        assert cfg[f"stock.recruitment.type.sp{i}"] == "shepherd"
    assert cfg["stock.recruitment.ssbhalf.sp0"] == "120000"

    # type3 FR on each of the 4 predators with default halfsat.
    for sp in FR_PREDATOR_SP:
        assert cfg[f"predation.functional.response.shape.sp{sp}"] == "type3"
        assert cfg[f"predation.functional.response.halfsat.sp{sp}"] == "1.0"


def test_shepherd_fr_mode_reads_halfsat_from_params():
    """When the params JSON carries halfsat keys, shepherd-fr uses those values."""
    from evaluate_calibration_vs_ices import _apply_mode

    params = {
        "predation.functional.response.halfsat.sp0": 2.5,
        "predation.functional.response.halfsat.sp5": 0.3,
        "predation.functional.response.halfsat.sp14": 1.1,
        "predation.functional.response.halfsat.sp15": 0.9,
    }
    cfg: dict[str, str] = {}
    _apply_mode(cfg, "shepherd-fr", params=params)
    assert cfg["predation.functional.response.halfsat.sp0"] == "2.5"
    assert cfg["predation.functional.response.halfsat.sp5"] == "0.3"
    assert cfg["predation.functional.response.shape.sp0"] == "type3"


def test_shepherd_mode_unchanged_no_fr():
    """Plain shepherd mode must NOT inject any FR keys (regression guard)."""
    from evaluate_calibration_vs_ices import _apply_mode

    cfg: dict[str, str] = {}
    _apply_mode(cfg, "shepherd")
    assert not any("functional.response" in k for k in cfg)
    assert cfg["stock.recruitment.type.sp0"] == "shepherd"


def test_unknown_mode_raises():
    from evaluate_calibration_vs_ices import _apply_mode

    with pytest.raises(ValueError, match="unknown mode"):
        _apply_mode({}, "nonsense")
