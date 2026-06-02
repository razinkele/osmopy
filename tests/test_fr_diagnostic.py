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


# ---------------------------------------------------------------------------
# BUG-1 regression: resolve_halfsat preserves distinct per-predator K values
# ---------------------------------------------------------------------------


def test_resolve_halfsat_returns_per_predator_values():
    """Each predator gets its OWN K from params — not a uniform value."""
    from fr_process_diagnostic import resolve_halfsat
    from evaluate_calibration_vs_ices import FR_PREDATOR_SP

    params = {
        "predation.functional.response.halfsat.sp0": 2.0,
        "predation.functional.response.halfsat.sp5": 3.0,
        "predation.functional.response.halfsat.sp14": 4.0,
        "predation.functional.response.halfsat.sp15": 0.5,
    }
    result = resolve_halfsat(params, FR_PREDATOR_SP, default_k=99.0)

    assert result[0] == 2.0, "sp0 should get its own K=2.0"
    assert result[5] == 3.0, "sp5 should get its own K=3.0"
    assert result[14] == 4.0, "sp14 should get its own K=4.0"
    assert result[15] == 0.5, "sp15 should get its own K=0.5"

    # Values must be distinct — confirming none were clobbered to a uniform value.
    values = list(result.values())
    assert len(set(values)) == 4, "all four K values should be distinct"


def test_resolve_halfsat_fallback_for_absent_key():
    """A predator absent from params gets the default_k fallback."""
    from fr_process_diagnostic import resolve_halfsat
    from evaluate_calibration_vs_ices import FR_PREDATOR_SP

    # Only sp0 is present; sp5/14/15 are absent.
    params = {
        "predation.functional.response.halfsat.sp0": 7.5,
    }
    result = resolve_halfsat(params, FR_PREDATOR_SP, default_k=1.0)

    assert result[0] == 7.5, "present key should be used"
    assert result[5] == 1.0, "absent sp5 should fall back to default_k"
    assert result[14] == 1.0, "absent sp14 should fall back to default_k"
    assert result[15] == 1.0, "absent sp15 should fall back to default_k"


def test_resolve_halfsat_uniform_when_no_params():
    """With an empty params dict every predator gets the default_k."""
    from fr_process_diagnostic import resolve_halfsat
    from evaluate_calibration_vs_ices import FR_PREDATOR_SP

    result = resolve_halfsat({}, FR_PREDATOR_SP, default_k=2.5)
    assert all(v == 2.5 for v in result.values()), "all should equal default_k=2.5"


def test_build_base_config_fr_on_preserves_per_predator_k(monkeypatch):
    """FR-ON config has each predator's OWN K — not clobbered to a uniform value.

    We mock OsmoseConfigReader.read to avoid needing the real Baltic CSV files,
    and inject a fake params dict with distinct per-predator halfsat values.
    """
    # Ensure scripts/ is on the path (already done at module level, but be safe).
    from fr_process_diagnostic import _build_base_config
    from evaluate_calibration_vs_ices import FR_PREDATOR_SP

    # Params with distinct K per predator.
    params = {
        "predation.functional.response.halfsat.sp0": 2.0,
        "predation.functional.response.halfsat.sp5": 3.0,
        "predation.functional.response.halfsat.sp14": 4.0,
        "predation.functional.response.halfsat.sp15": 0.5,
        # Include some non-FR params so the param-override pass exercises both code paths.
        "stock.recruitment.r.sp0": 0.8,
    }

    # Monkeypatch OsmoseConfigReader.read to return a minimal base config dict
    # rather than reading real CSV files from disk.
    import osmose.config.reader as _reader_mod

    def _fake_read(self, path):
        # Return the bare minimum keys that _apply_mode and the loops need.
        return {
            "simulation.time.nyear": "5",
            "stock.recruitment.type.sp0": "bh",
        }

    monkeypatch.setattr(_reader_mod.OsmoseConfigReader, "read", _fake_read)

    cfg = _build_base_config(params, fr_on=True, k=99.0)

    # Each predator must have its OWN K, not the uniform fallback 99.0.
    assert cfg["predation.functional.response.halfsat.sp0"] == "2.0", "sp0 K clobbered"
    assert cfg["predation.functional.response.halfsat.sp5"] == "3.0", "sp5 K clobbered"
    assert cfg["predation.functional.response.halfsat.sp14"] == "4.0", "sp14 K clobbered"
    assert cfg["predation.functional.response.halfsat.sp15"] == "0.5", "sp15 K clobbered"

    # All shapes must be type3 for FR-ON.
    for sp in FR_PREDATOR_SP:
        assert cfg[f"predation.functional.response.shape.sp{sp}"] == "type3"


def test_build_base_config_fr_on_fallback_for_absent_key(monkeypatch):
    """A predator absent from params falls back to --k in the FR-ON config."""
    import osmose.config.reader as _reader_mod
    from fr_process_diagnostic import _build_base_config

    params = {
        "predation.functional.response.halfsat.sp0": 7.5,
        # sp5, sp14, sp15 absent
    }

    def _fake_read(self, path):
        return {"simulation.time.nyear": "5"}

    monkeypatch.setattr(_reader_mod.OsmoseConfigReader, "read", _fake_read)

    cfg = _build_base_config(params, fr_on=True, k=1.0)

    assert cfg["predation.functional.response.halfsat.sp0"] == "7.5", "present key should be used"
    assert cfg["predation.functional.response.halfsat.sp5"] == "1.0", "absent sp5 fallback to k"
    assert cfg["predation.functional.response.halfsat.sp14"] == "1.0", "absent sp14 fallback to k"
    assert cfg["predation.functional.response.halfsat.sp15"] == "1.0", "absent sp15 fallback to k"


def test_build_base_config_fr_off_has_no_halfsat_keys(monkeypatch):
    """FR-OFF config must not contain any halfsat keys (regression guard)."""
    import osmose.config.reader as _reader_mod
    from fr_process_diagnostic import _build_base_config
    from evaluate_calibration_vs_ices import FR_PREDATOR_SP

    params = {
        "predation.functional.response.halfsat.sp0": 2.0,
        "predation.functional.response.halfsat.sp5": 3.0,
    }

    def _fake_read(self, path):
        return {"simulation.time.nyear": "5"}

    monkeypatch.setattr(_reader_mod.OsmoseConfigReader, "read", _fake_read)

    cfg = _build_base_config(params, fr_on=False, k=1.0)

    for sp in FR_PREDATOR_SP:
        hs_key = f"predation.functional.response.halfsat.sp{sp}"
        assert hs_key not in cfg, f"FR-OFF must not have {hs_key}"
        assert cfg[f"predation.functional.response.shape.sp{sp}"] == "type1"
