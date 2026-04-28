"""Unit tests for the pure stock-recruitment helper."""

import numpy as np
import pytest

from osmose.engine.processes.reproduction import apply_stock_recruitment


class TestApplyStockRecruitment:
    def test_none_returns_input_unchanged(self):
        linear = np.array([1000.0, 2000.0])
        ssb = np.array([10.0, 20.0])
        ssb_half = np.array([0.0, 0.0])
        types = ["none", "none"]
        out = apply_stock_recruitment(linear, ssb, ssb_half, types)
        np.testing.assert_array_equal(out, linear)

    def test_beverton_holt_low_ssb_approaches_linear(self):
        """At SSB << ssb_half, B-H ≈ linear (within 1%)."""
        linear = np.array([1000.0])
        ssb = np.array([1.0])
        ssb_half = np.array([1000.0])
        types = ["beverton_holt"]
        out = apply_stock_recruitment(linear, ssb, ssb_half, types)
        # 1000 / (1 + 1/1000) = 999.0...
        assert abs(out[0] - linear[0]) / linear[0] < 0.01

    def test_beverton_holt_at_half_saturation(self):
        """At SSB == ssb_half, B-H = linear / 2."""
        linear = np.array([1000.0])
        ssb = np.array([500.0])
        ssb_half = np.array([500.0])
        types = ["beverton_holt"]
        out = apply_stock_recruitment(linear, ssb, ssb_half, types)
        np.testing.assert_allclose(out, [500.0])

    def test_beverton_holt_asymptote(self):
        """At SSB >> ssb_half, B-H plateaus at linear * (ssb_half/ssb)."""
        linear = np.array([1_000_000.0])
        ssb = np.array([100_000.0])
        ssb_half = np.array([100.0])
        types = ["beverton_holt"]
        out = apply_stock_recruitment(linear, ssb, ssb_half, types)
        # 1e6 / (1 + 1e5/100) = 1e6 / 1001 ≈ 999
        assert out[0] < linear[0] * 0.01

    def test_ricker_at_peak(self):
        """Ricker peaks at SSB = ssb_half (where d eggs / d SSB = 0).

        At SSB = ssb_half, the multiplier is exp(-1) ≈ 0.368.
        """
        linear = np.array([1000.0])
        ssb = np.array([500.0])
        ssb_half = np.array([500.0])
        types = ["ricker"]
        out = apply_stock_recruitment(linear, ssb, ssb_half, types)
        np.testing.assert_allclose(out, [1000.0 * np.exp(-1.0)], rtol=1e-6)

    def test_ricker_high_ssb_collapses(self):
        """Ricker recruitment goes to ~0 at very high SSB / ssb_half ratios."""
        linear = np.array([1000.0])
        ssb = np.array([10000.0])
        ssb_half = np.array([100.0])
        types = ["ricker"]
        out = apply_stock_recruitment(linear, ssb, ssb_half, types)
        assert out[0] < 1e-30

    def test_mixed_types_per_species(self):
        """Different SR types can coexist across species in one call."""
        linear = np.array([1000.0, 1000.0, 1000.0])
        ssb = np.array([500.0, 500.0, 500.0])
        ssb_half = np.array([0.0, 500.0, 500.0])
        types = ["none", "beverton_holt", "ricker"]
        out = apply_stock_recruitment(linear, ssb, ssb_half, types)
        np.testing.assert_allclose(out[0], 1000.0)
        np.testing.assert_allclose(out[1], 500.0)
        np.testing.assert_allclose(out[2], 1000.0 * np.exp(-1.0), rtol=1e-6)

    def test_zero_ssb_returns_zero(self):
        """SSB=0 with any SR type returns 0 (linear is already 0)."""
        linear = np.array([0.0, 0.0])
        ssb = np.array([0.0, 0.0])
        ssb_half = np.array([100.0, 100.0])
        types = ["beverton_holt", "ricker"]
        out = apply_stock_recruitment(linear, ssb, ssb_half, types)
        np.testing.assert_array_equal(out, [0.0, 0.0])

    def test_unknown_type_raises(self):
        linear = np.array([1000.0])
        ssb = np.array([100.0])
        ssb_half = np.array([100.0])
        with pytest.raises(ValueError, match="unknown stock-recruitment type"):
            apply_stock_recruitment(linear, ssb, ssb_half, ["sigmoid"])
