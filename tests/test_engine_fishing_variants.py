"""Tests for fishing system variants and selectivity types."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine.processes.selectivity import (
    gaussian,
    log_normal,
    sigmoid,
)


class TestSigmoidSelectivity:
    """Type 1: Logistic sigmoid parameterized by L50/L75 (Java formula)."""

    def test_half_at_l50(self) -> None:
        """Selectivity = 0.5 at L50."""
        sel = sigmoid(np.array([20.0]), l50=20.0, l75=25.0)
        assert sel[0] == pytest.approx(0.5, rel=1e-6)

    def test_75pct_at_l75(self) -> None:
        """Selectivity = 0.75 at L75."""
        sel = sigmoid(np.array([25.0]), l50=20.0, l75=25.0)
        assert sel[0] == pytest.approx(0.75, rel=1e-6)


class TestGaussianSelectivity:
    """Type 2: Normal distribution, normalized by peak."""

    def test_peak_at_l50(self) -> None:
        """Maximum selectivity (1.0) at L50."""
        length = np.array([20.0])
        sel = gaussian(length, l50=20.0, l75=25.0)
        assert sel[0] == pytest.approx(1.0)

    def test_symmetric_bell(self) -> None:
        """Symmetric around L50."""
        l50, l75 = 20.0, 25.0
        lengths = np.array([15.0, 25.0])  # equidistant from l50
        sel = gaussian(lengths, l50=l50, l75=l75)
        assert sel[0] == pytest.approx(sel[1], rel=1e-6)

    def test_decreases_from_peak(self) -> None:
        """Selectivity decreases away from L50."""
        lengths = np.array([10.0, 20.0, 30.0])
        sel = gaussian(lengths, l50=20.0, l75=25.0)
        assert sel[1] > sel[0]
        assert sel[1] > sel[2]

    def test_at_l75_approx_75pct(self) -> None:
        """At L75, selectivity should be approximately 0.7978 of peak.

        Java: sd = (L75 - L50) / qnorm(0.75), so at L75 the normal PDF
        ratio = density(L75)/density(L50) = exp(-0.5 * 0.6745**2) ~ 0.7978.
        """
        sel = gaussian(np.array([25.0]), l50=20.0, l75=25.0)
        assert sel[0] == pytest.approx(0.7978, rel=0.01)


class TestLogNormalSelectivity:
    """Type 3: Log-normal distribution, normalized by mode."""

    def test_peak_at_mode(self) -> None:
        """Maximum selectivity at the mode = exp(mean - sd**2)."""
        l50, l75 = 20.0, 30.0
        mean = np.log(l50)
        q75 = 0.674489750196082
        sd = np.log(l75 / l50) / q75
        mode = np.exp(mean - sd**2)
        sel = log_normal(np.array([mode]), l50=l50, l75=l75)
        assert sel[0] == pytest.approx(1.0, rel=1e-4)

    def test_asymmetric(self) -> None:
        """Log-normal is right-skewed -- more selectivity above mode than below."""
        l50, l75 = 20.0, 30.0
        lengths = np.array([5.0, 10.0, 20.0, 30.0, 50.0])
        sel = log_normal(lengths, l50=l50, l75=l75)
        # Should be highest near mode, asymmetric
        peak_idx = np.argmax(sel)
        assert sel[peak_idx] > sel[0]
        assert sel[peak_idx] > sel[-1]
