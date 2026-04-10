"""Tests for numerical stability guards in bioenergetic functions."""

import numpy as np
import pytest

from osmose.engine.processes.oxygen_function import f_o2
from osmose.engine.processes.temp_function import phi_t


class TestF_O2Guards:
    def test_zero_o2_and_c2(self):
        """f_o2 with o2=0 and c2=0 should return 0, not NaN."""
        result = f_o2(np.array([0.0]), c1=1.0, c2=0.0)
        assert np.isfinite(result).all()
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_negative_o2(self):
        """f_o2 with negative o2 should not produce NaN."""
        result = f_o2(np.array([-1.0]), c1=1.0, c2=0.5)
        assert np.isfinite(result).all()

    def test_normal_values_unchanged(self):
        """Normal values should produce the same result as before."""
        result = f_o2(np.array([5.0]), c1=0.8, c2=2.0)
        expected = 0.8 * 5.0 / (5.0 + 2.0)
        assert result[0] == pytest.approx(expected, rel=1e-12)


class TestPhiTGuards:
    def test_equal_activation_energies(self):
        """phi_t with e_d == e_m should not produce NaN/inf."""
        result = phi_t(np.array([15.0]), e_m=0.5, e_d=0.5, t_p=20.0)
        assert np.isfinite(result).all()

    def test_normal_values_unchanged(self):
        """Normal values should produce the same result as before."""
        result = phi_t(np.array([20.0]), e_m=0.5, e_d=1.2, t_p=20.0)
        assert result[0] == pytest.approx(1.0, rel=1e-6)
