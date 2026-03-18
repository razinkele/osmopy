"""Tests for fishing selectivity."""

import numpy as np

from osmose.engine.processes.selectivity import knife_edge, sigmoid


class TestKnifeEdge:
    def test_below_l50(self):
        lengths = np.array([5.0, 8.0, 9.9])
        result = knife_edge(lengths, l50=10.0)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_above_l50(self):
        lengths = np.array([10.0, 15.0, 20.0])
        result = knife_edge(lengths, l50=10.0)
        np.testing.assert_array_equal(result, [1.0, 1.0, 1.0])


class TestSigmoid:
    def test_at_l50_is_half(self):
        lengths = np.array([10.0])
        result = sigmoid(lengths, l50=10.0, slope=1.0)
        np.testing.assert_allclose(result, [0.5])

    def test_well_above_l50_near_one(self):
        lengths = np.array([20.0])
        result = sigmoid(lengths, l50=10.0, slope=1.0)
        assert result[0] > 0.99

    def test_well_below_l50_near_zero(self):
        lengths = np.array([0.0])
        result = sigmoid(lengths, l50=10.0, slope=1.0)
        assert result[0] < 0.01
