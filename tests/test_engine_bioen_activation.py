"""Tests for bioen process activation (SP-5)."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine.processes.foraging_mortality import foraging_rate


class TestForagingMortality:
    """ForagingMortality.getRate() -- two modes."""

    def test_constant_mode(self) -> None:
        """Without genetics: F = k_for / ndt_per_year."""
        k_for = np.array([0.24])
        ndt_per_year = 24
        rate = foraging_rate(k_for=k_for, ndt_per_year=ndt_per_year)
        assert rate[0] == pytest.approx(0.01)  # 0.24 / 24

    def test_constant_mode_clamped(self) -> None:
        """Negative result clamped to 0."""
        k_for = np.array([-0.1])
        rate = foraging_rate(k_for=k_for, ndt_per_year=24)
        assert rate[0] == 0.0

    def test_genetic_mode(self) -> None:
        """With genetics: F = k1 * exp(k2 * (imax_trait - I_max)) / ndt."""
        k1 = np.array([0.1])
        k2 = np.array([2.0])
        imax_trait = np.array([5.0])
        I_max = np.array([5.0])  # trait == baseline -> exp(0) = 1
        ndt_per_year = 24
        rate = foraging_rate(
            k_for=None,
            ndt_per_year=ndt_per_year,
            k1_for=k1,
            k2_for=k2,
            imax_trait=imax_trait,
            I_max=I_max,
        )
        assert rate[0] == pytest.approx(0.1 / 24)  # k1 * exp(0) / 24

    def test_genetic_mode_penalty(self) -> None:
        """Trait below baseline -> exponential penalty increases rate."""
        k1 = np.array([0.1])
        k2 = np.array([1.0])
        imax_trait = np.array([3.0])  # below baseline
        I_max = np.array([5.0])
        rate = foraging_rate(
            k_for=None,
            ndt_per_year=24,
            k1_for=k1,
            k2_for=k2,
            imax_trait=imax_trait,
            I_max=I_max,
        )
        # exp(1.0 * (3 - 5)) = exp(-2) ~ 0.135
        expected = 0.1 * np.exp(-2.0) / 24
        assert rate[0] == pytest.approx(expected, rel=1e-6)
