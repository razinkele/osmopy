"""Tests for the UQ end-to-end orchestrator (run_surrogate_bayes) + helpers."""

from __future__ import annotations

import numpy as np

from osmose.calibration.uq.design import DesignResult
from osmose.calibration.uq.run import UQResult, derive_sigma_seed_sq


def test_derive_sigma_seed_sq_pools_per_key():
    # alpha = s^2 / n_seeds, so n_seeds * mean(alpha) recovers the pooled s^2.
    n_seeds = 5
    X = np.linspace(0, 1, 4).reshape(-1, 1)
    alpha = {"cod_biomass_mean": np.full(4, 0.02 / n_seeds)}  # s^2 = 0.02
    Y = {"cod_biomass_mean": np.zeros(4)}
    design = DesignResult(X=X, keys=["cod_biomass_mean"], Y=Y, alpha=alpha)
    out = derive_sigma_seed_sq(design, ["cod_biomass_mean"], n_seeds)
    assert out["cod_biomass_mean"] == np.float64(0.02)


def test_derive_sigma_seed_sq_uses_valid_points_only():
    # A censored (NaN) point must not enter the mean.
    n_seeds = 4
    X = np.array([[0.0], [1.0]])
    alpha = {"k": np.array([0.04 / n_seeds, np.nan])}
    Y = {"k": np.array([0.0, np.nan])}
    design = DesignResult(X=X, keys=["k"], Y=Y, alpha=alpha)
    out = derive_sigma_seed_sq(design, ["k"], n_seeds)
    assert out["k"] == np.float64(0.04)  # only the one valid point


def test_uqresult_gate_failed_has_no_posterior():
    design = DesignResult(
        X=np.zeros((1, 1)), keys=["k"], Y={"k": np.array([1.0])}, alpha={"k": np.array([0.1])}
    )
    r = UQResult(status="gate_failed", gate_reports={}, design=design, n_censored={"k": 0})
    assert r.sampler_result is None
    assert r.posterior_mean is None
