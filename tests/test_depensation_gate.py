import numpy as np
import pytest
from osmose.engine.processes.depensation_gate import depensation_factor


def test_half_at_s50():
    # A(S50) = 0.5 exactly, any theta
    f = depensation_factor(
        np.array([50_000.0]), np.array([50_000.0]), np.array([4.0]), np.array([True])
    )
    assert f[0] == pytest.approx(0.5)


def test_approaches_zero_at_low_ssb():
    f = depensation_factor(
        np.array([1_000.0]), np.array([50_000.0]), np.array([4.0]), np.array([True])
    )
    assert 0.0 < f[0] < 1e-4


def test_zero_at_ssb_zero():
    f = depensation_factor(np.array([0.0]), np.array([50_000.0]), np.array([4.0]), np.array([True]))
    assert f[0] == 0.0


def test_approaches_one_at_high_ssb():
    f = depensation_factor(
        np.array([5_000_000.0]), np.array([50_000.0]), np.array([4.0]), np.array([True])
    )
    assert f[0] == pytest.approx(1.0, abs=1e-3)


def test_disabled_is_one():
    f = depensation_factor(
        np.array([1_000.0]), np.array([50_000.0]), np.array([4.0]), np.array([False])
    )
    assert f[0] == 1.0


def test_theta_one_boundary():
    # theta=1: A = SSB/(S50+SSB); at S50 still 0.5, still <1 at low SSB
    f = depensation_factor(
        np.array([50_000.0]), np.array([50_000.0]), np.array([1.0]), np.array([True])
    )
    assert f[0] == pytest.approx(0.5)


def test_multi_species_isolation():
    # only enabled species differ from 1.0
    ssb = np.array([1_000.0, 1_000.0])
    s50 = np.array([50_000.0, 50_000.0])
    theta = np.array([4.0, 4.0])
    enabled = np.array([True, False])
    f = depensation_factor(ssb, s50, theta, enabled)
    assert f[0] < 1e-4
    assert f[1] == 1.0
