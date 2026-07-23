"""Tests for posterior composition + emulator fitting."""

from __future__ import annotations

import math

import numpy as np
import pytest

from osmose.calibration.problem import FreeParameter, Transform
from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.design import DesignResult
from osmose.calibration.uq.likelihood import gaussian_log_biomass
from osmose.calibration.uq.posterior import fit_emulators, make_log_posterior


class _AnalyticEmulator:
    """Injected emulator matching GPEmulator.predict: X (n,d) -> (mean (n,), var (n,))."""

    def __init__(self, w, b, var):
        self.w = np.asarray(w, float)
        self.b = b
        self.var = var

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        mean = X @ self.w + self.b
        return mean, np.full(len(mean), self.var)


def _fp2():
    return [
        FreeParameter("a.sp0", 0.0, 1.0, Transform.LINEAR),
        FreeParameter("b.sp0", 0.0, 1.0, Transform.LINEAR),
    ]


def _target(species, rpt, target, lower, upper):
    return BiomassTarget(
        species=species, target=target, lower=lower, upper=upper, reference_point_type=rpt
    )


def test_fit_emulators_one_per_key_with_enough_points():
    n = 6
    X = np.linspace(0, 1, n).reshape(-1, 1)
    Y = {"cod_biomass_mean": np.log(np.full(n, 100.0)) + 0.01 * np.arange(n)}
    alpha = {"cod_biomass_mean": np.full(n, 1e-3)}
    design = DesignResult(X=X, keys=["cod_biomass_mean"], Y=Y, alpha=alpha)
    emus = fit_emulators(design)
    assert set(emus) == {"cod_biomass_mean"}
    mean, var = emus["cod_biomass_mean"].predict(np.array([[0.5]]))
    assert mean.shape == (1,) and var.shape == (1,)


def test_fit_emulators_skips_insufficient_points():
    X = np.array([[0.0], [1.0]])
    Y = {"k": np.array([np.log(100.0), np.nan])}  # only 1 valid point
    alpha = {"k": np.array([1e-3, np.nan])}
    design = DesignResult(X=X, keys=["k"], Y=Y, alpha=alpha)
    assert fit_emulators(design) == {}


def test_make_log_posterior_sums_prior_and_likelihoods():
    emu = _AnalyticEmulator([1.0, 0.0], 2.0, 0.01)
    emus = {"cod_biomass_mean": emu}
    tgt = _target("cod", "biomass", 20.0, 16.0, 24.0)
    theta = np.array([0.3, 0.5])
    logp = make_log_posterior(emus, [tgt], _fp2(), sigma_seed_sq_by_key={"cod_biomass_mean": 0.02})(
        theta
    )
    mu, var = emu.predict(theta.reshape(1, -1))
    expected = 0.0 + gaussian_log_biomass(
        float(mu[0]), float(var[0]), 20.0, 16.0, 24.0, sigma_seed_sq=0.02, sigma_disc_sq=0.0, k=1.0
    )
    assert abs(logp - expected) < 1e-9


def test_make_log_posterior_prior_gates_out_of_box():
    emus = {"cod_biomass_mean": _AnalyticEmulator([1.0, 0.0], 2.0, 0.01)}
    tgt = _target("cod", "biomass", 20.0, 16.0, 24.0)
    logp = make_log_posterior(emus, [tgt], _fp2(), sigma_seed_sq_by_key={"cod_biomass_mean": 0.02})
    assert logp(np.array([1.5, 0.5])) == -math.inf


def test_make_log_posterior_missing_emulator_key_raises():
    tgt = _target("cod", "ssb", 20.0, 16.0, 24.0)  # key cod_ssb_mean
    with pytest.raises(KeyError, match="cod_ssb_mean"):
        make_log_posterior({}, [tgt], _fp2(), sigma_seed_sq_by_key={})
