"""Tests for the UQ sampler (SamplerResult summaries, dimension cap, DynestySampler)."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.calibration.uq.sampler import (
    MAX_NOMINAL_DIM,
    SamplerResult,
    check_dimension,
)


def test_check_dimension_ok_at_cap():
    check_dimension(MAX_NOMINAL_DIM)  # exactly at the cap is allowed; must not raise


def test_check_dimension_raises_above_cap():
    with pytest.raises(ValueError, match="exceeds"):
        check_dimension(MAX_NOMINAL_DIM + 1)


def _result(samples, weights):
    return SamplerResult(
        samples=samples, weights=weights, logz=0.0, logz_err=0.1, ess=100.0, converged=True
    )


def test_sampler_result_posterior_mean_is_weighted():
    r = _result(np.array([[0.0, 0.0], [2.0, 2.0]]), np.array([0.25, 0.75]))
    assert np.allclose(r.posterior_mean(), [1.5, 1.5])


def test_sampler_result_credible_interval_ordered_and_brackets_mean():
    x = np.linspace(0.0, 10.0, 101)
    r = _result(np.column_stack([x, x]), np.ones_like(x))
    lo, hi = r.credible_interval(0.9)
    assert lo.shape == (2,) and hi.shape == (2,)
    assert np.all(lo < hi)
    m = r.posterior_mean()
    assert np.all(lo < m) and np.all(m < hi)


def test_sampler_result_correlation_and_marginal_sd_weighted():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    r = _result(np.column_stack([x, -x]), np.ones_like(x))  # perfectly anti-correlated
    assert r.correlation()[0, 1] == pytest.approx(-1.0)
    assert np.all(r.marginal_sd() > 0.0)
