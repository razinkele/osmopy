"""Tests for bioen_starvation: energy-deficit starvation with gonad buffer."""
import numpy as np
import pytest

from osmose.engine.processes.bioen_starvation import bioen_starvation


class TestBioenStarvation:
    def test_positive_enet_no_starvation(self):
        """No deaths when E_net >= 0."""
        n_dead, new_gonad = bioen_starvation(
            e_net=np.array([0.1]), gonad_weight=np.array([0.001]),
            weight=np.array([0.01]), eta=1.5, n_subdt=10)
        assert n_dead[0] == 0.0
        assert new_gonad[0] == pytest.approx(0.001)

    def test_gonad_absorbs_deficit(self):
        """Sufficient gonad: no deaths, gonad decreases."""
        n_dead, new_gonad = bioen_starvation(
            e_net=np.array([-0.001]), gonad_weight=np.array([0.01]),
            weight=np.array([0.01]), eta=1.5, n_subdt=1)
        assert n_dead[0] == 0.0
        assert new_gonad[0] < 0.01
        assert new_gonad[0] == pytest.approx(0.01 - 1.5 * 0.001)

    def test_gonad_insufficient_causes_death(self):
        """No gonad buffer: full deficit -> deaths."""
        n_dead, new_gonad = bioen_starvation(
            e_net=np.array([-10.0]), gonad_weight=np.array([0.0]),
            weight=np.array([0.01]), eta=1.5, n_subdt=1)
        assert n_dead[0] > 0
        assert new_gonad[0] == 0.0

    def test_partial_gonad_buffer(self):
        """Some gonad: partial buffer, remaining deficit -> fewer deaths."""
        n_dead_no_gonad, _ = bioen_starvation(
            e_net=np.array([-1.0]), gonad_weight=np.array([0.0]),
            weight=np.array([0.01]), eta=1.5, n_subdt=1)
        n_dead_with_gonad, _ = bioen_starvation(
            e_net=np.array([-1.0]), gonad_weight=np.array([0.5]),
            weight=np.array([0.01]), eta=1.5, n_subdt=1)
        assert n_dead_with_gonad[0] < n_dead_no_gonad[0]

    def test_vectorized(self):
        """Multiple schools with different conditions."""
        e_net = np.array([0.1, -0.01, -10.0])
        gonad = np.array([0.0, 0.1, 0.0])
        weight = np.array([0.01, 0.01, 0.01])
        n_dead, new_gonad = bioen_starvation(e_net, gonad, weight, eta=1.5, n_subdt=1)
        assert n_dead[0] == 0.0  # positive E_net
        assert n_dead[1] == 0.0  # gonad sufficient
        assert n_dead[2] > 0     # no gonad, big deficit
