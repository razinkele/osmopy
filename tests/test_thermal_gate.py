import numpy as np
import pytest
from types import SimpleNamespace

from osmose.engine.processes.thermal_gate import (
    logistic_response,
    normalize_factor,
    thermal_gate_factor,
)


# ── Task 1: pure response + normalization ───────────────────────────────────
def test_logistic_is_half_at_t50():
    r = logistic_response(np.array([18.5]), t50=18.5, slope=1.5)
    assert r[0] == pytest.approx(0.5)


def test_logistic_monotone_increasing():
    r = logistic_response(np.array([10.0, 15.0, 18.5, 22.0]), t50=18.5, slope=1.5)
    assert np.all(np.diff(r) > 0)


def test_thermal_cap_clips_to_unit_and_floor():
    r = np.array([0.01, 0.5, 0.99])
    out = normalize_factor(r, mode="thermal_cap", r_ref=0.5, window_idx=[0, 1, 2], floor=0.05)
    assert out[0] == pytest.approx(0.05)  # floored
    assert out[1] == pytest.approx(1.0)  # 0.5/0.5 = 1.0
    assert out[2] == pytest.approx(1.0)  # clipped to 1
    assert np.all(out <= 1.0) and np.all(out >= 0.05)


def test_mean_preserving_has_unit_mean_over_window():
    r = np.array([0.2, 0.4, 0.6, 0.8])
    out = normalize_factor(r, mode="mean_preserving", r_ref=0.0, window_idx=[0, 1, 2, 3], floor=0.0)
    assert np.mean(out) == pytest.approx(1.0)


def test_mean_preserving_rejects_positive_floor():
    # review finding 4: floor>0 would break the unit-mean invariant.
    r = np.array([0.2, 0.4, 0.6, 0.8])
    with pytest.raises(ValueError, match="floor"):
        normalize_factor(r, mode="mean_preserving", r_ref=0.0, window_idx=[0, 1, 2, 3], floor=0.1)


def test_bad_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        normalize_factor(np.array([0.5]), mode="bogus", r_ref=0.5, window_idx=[0], floor=0.0)


# ── Task 2: per-step helper ─────────────────────────────────────────────────
def _stub(factor, enabled, offset=0, n_species=6, n_dt=24):
    return SimpleNamespace(
        n_species=n_species,
        n_dt_per_year=n_dt,
        thermal_gate_factor_by_index=factor,
        thermal_gate_enabled=enabled,
        thermal_gate_offset=offset,
    )


def test_factor_all_ones_when_off():
    out = thermal_gate_factor(_stub(None, None), step=0)
    assert np.array_equal(out, np.ones(6))


def test_factor_applies_only_to_enabled_species_for_current_year():
    factor = np.array([[1.0, 1.0, 1.0, 1.0, 0.3, 0.7], [1.0, 1.0, 1.0, 1.0, 0.9, 0.8]])
    enabled = np.array([False, False, False, False, True, True])
    out = thermal_gate_factor(_stub(factor, enabled), step=0)
    assert out[4] == pytest.approx(0.3) and out[5] == pytest.approx(0.7)
    assert out[0] == 1.0 and out[3] == 1.0
    out1 = thermal_gate_factor(_stub(factor, enabled), step=24)
    assert out1[4] == pytest.approx(0.9)


def test_factor_year_index_wraps_around_series():
    factor = np.array([[1, 1, 1, 1, 0.3, 0.3], [1, 1, 1, 1, 0.9, 0.9]], dtype=float)
    enabled = np.array([False, False, False, False, True, True])
    out = thermal_gate_factor(_stub(factor, enabled), step=48)  # year 2, 2-row series -> idx 0
    assert out[4] == pytest.approx(0.3)
