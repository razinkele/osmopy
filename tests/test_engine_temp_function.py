"""Tests for Johnson thermal performance curve and Arrhenius function."""

import numpy as np
import pytest
from osmose.engine.processes.temp_function import phi_t, arrhenius


class TestPhiT:
    # Representative parameters: e_m=0.6 eV, e_d=2.0 eV, t_p=15°C
    E_M = 0.6
    E_D = 2.0
    T_P = 15.0

    def test_phi_t_at_peak_is_one(self):
        """phi_T(T_P) == 1.0 by normalization."""
        result = phi_t(np.array([self.T_P]), self.E_M, self.E_D, self.T_P)
        assert result.shape == (1,)
        assert abs(result[0] - 1.0) < 1e-12

    def test_phi_t_decreases_away_from_peak(self):
        """Values below and above peak are strictly less than 1."""
        temps = np.array([5.0, 25.0])
        result = phi_t(temps, self.E_M, self.E_D, self.T_P)
        assert result[0] < 1.0, "phi_T(5°C) should be < 1 when peak is 15°C"
        assert result[1] < 1.0, "phi_T(25°C) should be < 1 when peak is 15°C"

    def test_phi_t_vectorized(self):
        """Array of 100 temperatures all produce values in (0, 1]."""
        temps = np.linspace(0.0, 30.0, 100)
        result = phi_t(temps, self.E_M, self.E_D, self.T_P)
        assert result.shape == (100,)
        assert np.all(result > 0.0), "phi_T must be positive everywhere"
        assert np.all(result <= 1.0 + 1e-10), "phi_T must not exceed 1 (normalized at peak)"

    def test_phi_t_symmetric_around_peak(self):
        """Both sides of peak return values < 1 (curve need not be symmetric)."""
        below = phi_t(np.array([self.T_P - 10.0]), self.E_M, self.E_D, self.T_P)
        above = phi_t(np.array([self.T_P + 10.0]), self.E_M, self.E_D, self.T_P)
        assert below[0] < 1.0
        assert above[0] < 1.0

    def test_phi_t_scalar_input(self):
        """Scalar array input works without error."""
        result = phi_t(np.array([self.T_P]), self.E_M, self.E_D, self.T_P)
        assert float(result[0]) == pytest.approx(1.0, abs=1e-12)


class TestArrhenius:
    E_M = 0.6  # eV activation energy

    def test_arrhenius_increases_with_temp(self):
        """Higher temperature produces higher Arrhenius value."""
        low = arrhenius(np.array([10.0]), self.E_M)
        high = arrhenius(np.array([20.0]), self.E_M)
        assert high[0] > low[0], "Arrhenius value should increase with temperature"

    def test_arrhenius_positive(self):
        """Arrhenius values are always strictly positive."""
        temps = np.linspace(-2.0, 40.0, 50)
        result = arrhenius(temps, self.E_M)
        assert np.all(result > 0.0), "Arrhenius must be positive for all temperatures"

    def test_arrhenius_vectorized(self):
        """Array input returns array of same shape."""
        temps = np.linspace(0.0, 30.0, 200)
        result = arrhenius(temps, self.E_M)
        assert result.shape == (200,)

    def test_arrhenius_known_value(self):
        """Spot-check against manual formula: exp(-e_m / (K_B * T_K))."""
        K_B = 8.62e-5
        t_c = 20.0
        t_k = t_c + 273.15
        expected = np.exp(-self.E_M / (K_B * t_k))
        result = arrhenius(np.array([t_c]), self.E_M)
        assert result[0] == pytest.approx(expected, rel=1e-10)
