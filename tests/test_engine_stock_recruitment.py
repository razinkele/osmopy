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
        """At SSB = ssb_half, the Ricker multiplier is exp(-1) ≈ 0.368."""
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

    def test_zero_ssb_with_zero_ssb_half_does_not_divide_by_zero(self):
        """The ssb<=0 guard must short-circuit before B-H divides by ssb_half=0.

        Without the guard, B-H computes 1000 / (1 + 0/0) which yields NaN. The
        guard preserves the linear_eggs value (which would be 0 in production
        because SSB=0 -> linear=0, but here we use linear>0 to exercise the
        contract directly).
        """
        linear = np.array([1000.0, 1000.0])
        ssb = np.array([0.0, 0.0])
        ssb_half = np.array([0.0, 0.0])
        types = ["beverton_holt", "ricker"]
        out = apply_stock_recruitment(linear, ssb, ssb_half, types)
        np.testing.assert_array_equal(out, linear)
        assert not np.any(np.isnan(out)), "guard must prevent NaN"

    def test_unknown_type_raises(self):
        linear = np.array([1000.0])
        ssb = np.array([100.0])
        ssb_half = np.array([100.0])
        with pytest.raises(ValueError, match="unknown stock-recruitment type"):
            apply_stock_recruitment(linear, ssb, ssb_half, ["sigmoid"])

    def test_shepherd_beta_one_equals_beverton_holt(self):
        """Shepherd at beta=1 is identically Beverton-Holt (correctness anchor)."""
        linear = np.array([1000.0, 2000.0])
        ssb = np.array([500.0, 1500.0])
        ssb_half = np.array([500.0, 1000.0])
        bh = apply_stock_recruitment(
            linear, ssb, ssb_half, ["beverton_holt", "beverton_holt"]
        )
        shep = apply_stock_recruitment(
            linear, ssb, ssb_half, ["shepherd", "shepherd"], np.array([1.0, 1.0])
        )
        np.testing.assert_array_equal(shep, bh)

    def test_shepherd_low_ssb_approaches_linear(self):
        """At SSB << ssb_half, Shepherd ≈ linear for any beta."""
        linear = np.array([1000.0])
        ssb = np.array([1.0])
        ssb_half = np.array([1000.0])
        out = apply_stock_recruitment(
            linear, ssb, ssb_half, ["shepherd"], np.array([2.0])
        )
        assert abs(out[0] - linear[0]) / linear[0] < 0.01

    def test_shepherd_high_beta_overcompensates(self):
        """beta>1: with linear ∝ ssb, recruitment turns down at very high SSB."""
        alpha = 1.0
        ssb_half = np.array([500.0])
        beta = np.array([3.0])
        r_peak = apply_stock_recruitment(
            np.array([alpha * 500.0]), np.array([500.0]), ssb_half, ["shepherd"], beta
        )
        r_high = apply_stock_recruitment(
            np.array([alpha * 5000.0]), np.array([5000.0]), ssb_half, ["shepherd"], beta
        )
        assert r_high[0] < r_peak[0]

    def test_shepherd_low_beta_gentler_than_bh(self):
        """beta<1 caps less aggressively than B-H at the same high SSB."""
        linear = np.array([1000.0])
        ssb = np.array([2000.0])
        ssb_half = np.array([500.0])
        bh = apply_stock_recruitment(linear, ssb, ssb_half, ["beverton_holt"])
        shep = apply_stock_recruitment(
            linear, ssb, ssb_half, ["shepherd"], np.array([0.5])
        )
        assert shep[0] > bh[0]

    def test_shepherd_defaults_beta_one_when_array_omitted(self):
        """If shepherd_beta is not passed, beta defaults to 1.0 (≡ B-H)."""
        linear = np.array([1000.0])
        ssb = np.array([500.0])
        ssb_half = np.array([500.0])
        out = apply_stock_recruitment(linear, ssb, ssb_half, ["shepherd"])
        np.testing.assert_allclose(out, [500.0])
