"""Tests for the UQ sampler (SamplerResult summaries, dimension cap, DynestySampler)."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.calibration.problem import FreeParameter, Transform
from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.posterior import make_log_posterior
from osmose.calibration.uq.sampler import (
    MAX_NOMINAL_DIM,
    DynestySampler,
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


_THETA_STAR = np.array([0.3, 0.6])
_SIG_SEED, _EMU_VAR = 0.02, 0.01


class _AnalyticEmulator:
    def __init__(self, w, b, var):
        self.w = np.asarray(w, float)
        self.b = b
        self.var = var

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return X @ self.w + self.b, np.full(len(X), self.var)


def _fp2():
    return [
        FreeParameter("a.sp0", 0.0, 1.0, Transform.LINEAR),
        FreeParameter("b.sp0", 0.0, 1.0, Transform.LINEAR),
    ]


def _identifiable_log_post():
    emus = {
        "A_biomass_mean": _AnalyticEmulator([1.0, 0.0], 2.0, _EMU_VAR),
        "B_biomass_mean": _AnalyticEmulator([0.0, 1.0], 1.0, _EMU_VAR),
        "C_biomass_mean": _AnalyticEmulator([1.0, 1.0], 0.5, _EMU_VAR),
    }
    targets = []
    for key, emu in emus.items():
        mu_star, _ = emu.predict(_THETA_STAR.reshape(1, -1))
        value = float(np.exp(mu_star[0] + 0.5 * _SIG_SEED))
        targets.append(
            BiomassTarget(
                species=key.split("_")[0],
                target=value,
                lower=value * 0.8,
                upper=value * 1.2,
                reference_point_type="biomass",
            )
        )
    return make_log_posterior(
        emus, targets, _fp2(), sigma_seed_sq_by_key={k: _SIG_SEED for k in emus}
    )


def test_dynesty_recovers_theta_star():
    result = DynestySampler().sample(_identifiable_log_post(), _fp2(), seed=0)
    lo, hi = result.credible_interval(0.9)
    assert np.all((lo <= _THETA_STAR) & (_THETA_STAR <= hi))  # 90% CI covers theta*
    assert np.allclose(result.posterior_mean(), _THETA_STAR, atol=0.1)  # generous
    assert result.converged
    assert result.ess > 100.0


def test_dynesty_result_carries_weights_and_evidence():
    result = DynestySampler().sample(_identifiable_log_post(), _fp2(), seed=0)
    assert result.samples.shape[1] == 2
    assert result.weights.shape[0] == result.samples.shape[0]
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(result.logz) and np.isfinite(result.logz_err)


def test_dynesty_dimension_cap_aborts_before_sampling():
    # max_dim=1 with 2 params: the cap must raise, and it must do so WITHOUT sampling
    # (a sentinel log_posterior that would fail if ever called).
    def _never(theta):
        raise AssertionError("log_posterior must not be called when the cap trips")

    with pytest.raises(ValueError, match="exceeds"):
        DynestySampler(max_dim=1).sample(_never, _fp2(), seed=0)
