"""Tests for dissolved oxygen dose-response function."""
import numpy as np
import pytest
from osmose.engine.processes.oxygen_function import f_o2


class TestFO2:
    C1 = 1.0   # maximum response coefficient
    C2 = 5.0   # half-saturation concentration (mmol/m³)

    def test_fo2_at_zero_is_zero(self):
        """f_O2(0) = 0 because numerator is zero."""
        result = f_o2(np.array([0.0]), self.C1, self.C2)
        assert result[0] == pytest.approx(0.0, abs=1e-15)

    def test_fo2_saturates_at_high_oxygen(self):
        """f_O2 approaches C1 as O2 -> infinity."""
        very_high = np.array([1e9])
        result = f_o2(very_high, self.C1, self.C2)
        assert result[0] == pytest.approx(self.C1, rel=1e-6), \
            "f_O2 should approach C1 at very high O2 concentrations"

    def test_fo2_half_saturation(self):
        """f_O2(C2) = C1 * 0.5 by definition of the Michaelis-Menten form."""
        result = f_o2(np.array([self.C2]), self.C1, self.C2)
        assert result[0] == pytest.approx(self.C1 * 0.5, rel=1e-12)

    def test_fo2_vectorized(self):
        """Array input returns array of same shape with expected properties."""
        o2 = np.linspace(0.0, 100.0, 500)
        result = f_o2(o2, self.C1, self.C2)
        assert result.shape == (500,)
        # Monotonically increasing
        assert np.all(np.diff(result) >= 0), "f_O2 must be monotonically non-decreasing"
        # Bounded by [0, C1]
        assert result[0] == pytest.approx(0.0, abs=1e-15)
        assert np.all(result <= self.C1 + 1e-12)

    def test_fo2_scales_with_c1(self):
        """Doubling C1 doubles the output."""
        o2 = np.array([10.0])
        r1 = f_o2(o2, 1.0, self.C2)
        r2 = f_o2(o2, 2.0, self.C2)
        assert r2[0] == pytest.approx(2.0 * r1[0], rel=1e-12)

    def test_fo2_known_value(self):
        """Spot-check: f_O2(10, C1=1, C2=5) = 1*10/(10+5) = 2/3."""
        result = f_o2(np.array([10.0]), 1.0, 5.0)
        assert result[0] == pytest.approx(10.0 / 15.0, rel=1e-12)
