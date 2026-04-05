"""Tests for numerical edge cases in growth and predation."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine.processes.growth import expected_length_vb
from osmose.engine.processes.predation import compute_size_overlap


class TestGrowthEdgeCases:
    """Numerical edge cases for Von Bertalanffy growth."""

    def test_vb_zero_k(self):
        """k=0 (no growth) should produce finite results, not NaN or inf."""
        result = expected_length_vb(
            age_dt=np.array([0, 12, 48, 120], dtype=np.int32),
            linf=np.array([30.0, 30.0, 30.0, 30.0]),
            k=np.array([0.0, 0.0, 0.0, 0.0]),
            t0=np.array([-0.1, -0.1, -0.1, -0.1]),
            egg_size=np.array([0.1, 0.1, 0.1, 0.1]),
            vb_threshold_age=np.array([1.0, 1.0, 1.0, 1.0]),
            n_dt_per_year=24,
        )
        assert np.isfinite(result).all(), f"Non-finite values with k=0: {result}"

    def test_vb_positive_t0(self):
        """Positive t0 (unusual but valid) should produce finite results.

        When t0 > threshold_age the VB formula yields negative l_thres, which
        propagates into the linear interpolation phase and can produce negative
        intermediate lengths.  The test documents this known limitation: results
        are finite (no NaN/inf) even if some intermediate ages are negative.
        Age==0 always returns egg_size and ages above threshold follow the VB
        curve, both of which are correct.
        """
        result = expected_length_vb(
            age_dt=np.array([0, 6, 12, 48, 120], dtype=np.int32),
            linf=np.array([30.0, 30.0, 30.0, 30.0, 30.0]),
            k=np.array([0.3, 0.3, 0.3, 0.3, 0.3]),
            t0=np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
            egg_size=np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
            vb_threshold_age=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            n_dt_per_year=24,
        )
        # Must be finite — no NaN or inf regardless of inputs
        assert np.isfinite(result).all(), f"Non-finite values with t0=2.0: {result}"
        # Age 0 always returns egg_size
        assert result[0] == pytest.approx(0.1), f"Age-0 should equal egg_size, got {result[0]}"
        # Ages beyond threshold (age_dt=48 -> 2 years > threshold 1 year) follow VB
        age_years_48 = 48.0 / 24
        expected_vb = 30.0 * (1 - np.exp(-0.3 * (age_years_48 - 2.0)))
        assert result[3] == pytest.approx(expected_vb, abs=1e-10)

    def test_vb_large_k(self):
        """Very large k (fast growth) should not overflow."""
        result = expected_length_vb(
            age_dt=np.array([0, 24, 48], dtype=np.int32),
            linf=np.array([100.0, 100.0, 100.0]),
            k=np.array([10.0, 10.0, 10.0]),
            t0=np.array([-0.1, -0.1, -0.1]),
            egg_size=np.array([0.1, 0.1, 0.1]),
            vb_threshold_age=np.array([1.0, 1.0, 1.0]),
            n_dt_per_year=24,
        )
        assert np.isfinite(result).all(), f"Non-finite values with k=10: {result}"
        # Should approach linf but not exceed it significantly
        assert (result <= 100.0 + 1e-10).all(), f"Length exceeds linf with k=10: {result}"

    def test_vb_zero_threshold_age(self):
        """threshold_age=0 triggers the safe fallback branch; results should be finite."""
        result = expected_length_vb(
            age_dt=np.array([0, 12, 48], dtype=np.int32),
            linf=np.array([30.0, 30.0, 30.0]),
            k=np.array([0.3, 0.3, 0.3]),
            t0=np.array([-0.1, -0.1, -0.1]),
            egg_size=np.array([0.1, 0.1, 0.1]),
            vb_threshold_age=np.array([0.0, 0.0, 0.0]),
            n_dt_per_year=24,
        )
        assert np.isfinite(result).all(), f"Non-finite values with threshold_age=0: {result}"


class TestPredationEdgeCases:
    """Numerical edge cases for predation size overlap."""

    def test_size_overlap_zero_prey_length(self):
        """Zero-length prey must return False without raising ZeroDivisionError."""
        result = compute_size_overlap(
            pred_length=25.0,
            prey_length=0.0,
            ratio_min=1.0,
            ratio_max=3.5,
        )
        assert result is False

    def test_size_overlap_negative_prey_length(self):
        """Negative prey length (invalid data) must return False gracefully."""
        result = compute_size_overlap(
            pred_length=25.0,
            prey_length=-5.0,
            ratio_min=1.0,
            ratio_max=3.5,
        )
        assert result is False

    def test_size_overlap_zero_pred_length(self):
        """Zero predator length: ratio=0, below any reasonable r_min, so False."""
        result = compute_size_overlap(
            pred_length=0.0,
            prey_length=10.0,
            ratio_min=1.0,
            ratio_max=3.5,
        )
        assert result is False

    def test_size_overlap_equal_lengths(self):
        """Predator == prey length gives ratio=1.0; included when r_min=1.0."""
        result = compute_size_overlap(
            pred_length=10.0,
            prey_length=10.0,
            ratio_min=1.0,
            ratio_max=3.5,
        )
        assert result is True

    def test_size_overlap_returns_bool(self):
        """Return type must be bool regardless of input."""
        result = compute_size_overlap(25.0, 10.0, 1.0, 3.5)
        assert isinstance(result, bool)
