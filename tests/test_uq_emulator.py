"""Tests for the UQ GP emulator (fit/predict, alpha co-scaling)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from osmose.calibration.uq.emulator import GPEmulator


@pytest.fixture()
def design():
    """40-point 3-D design with a smooth log-space target and tiny noise."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(40, 3))
    Y = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] - X[:, 2]
    alpha = np.full(40, 1e-3)
    return X, Y, alpha


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        GPEmulator().predict(np.zeros((1, 3)))


def test_fit_predict_returns_mean_and_variance(design):
    X, Y, alpha = design
    emu = GPEmulator(n_restarts_optimizer=0).fit(X, Y, alpha)
    mean, var = emu.predict(X)
    assert mean.shape == (40,)
    assert var.shape == (40,)
    assert np.all(var >= 0.0)
    # Low-noise GP ~interpolates its training points.
    assert np.sqrt(np.mean((mean - Y) ** 2)) < 0.1


def test_fit_accepts_scalar_alpha(design):
    X, Y, _ = design
    emu = GPEmulator(n_restarts_optimizer=0).fit(X, Y, 1e-3)
    mean, var = emu.predict(X[:5])
    assert mean.shape == (5,)
    assert var.shape == (5,)


def test_length_scale_invariant_under_y_rescaling(design):
    """Co-scaled alpha makes the standardized-space fit identical under Y->c*Y,
    so the ARD length scales (which live in X-space) are Y-unit invariant."""
    X, Y, alpha = design
    c = 10.0
    emu1 = GPEmulator(n_restarts_optimizer=0).fit(X, Y, alpha)
    # Rescaling Y by c scales its noise variance by c^2, hence raw alpha by c^2.
    emu2 = GPEmulator(n_restarts_optimizer=0).fit(X, c * Y, (c**2) * alpha)
    ls1 = np.atleast_1d(emu1.gp.kernel_.length_scale)
    ls2 = np.atleast_1d(emu2.gp.kernel_.length_scale)
    assert np.allclose(ls1, ls2, rtol=1e-3)


def test_predictive_variance_scales_with_y_units(design):
    """predict() variance is in Y-units: under Y->c*Y it scales by c^2 (correct
    covariant behavior), NOT invariant. Guards the spec's 'invariant' misphrasing."""
    X, Y, alpha = design
    c = 10.0
    emu1 = GPEmulator(n_restarts_optimizer=0).fit(X, Y, alpha)
    emu2 = GPEmulator(n_restarts_optimizer=0).fit(X, c * Y, (c**2) * alpha)
    Xt = np.array([[0.3, 0.4, 0.5], [0.1, 0.9, 0.2]])
    _, var1 = emu1.predict(Xt)
    _, var2 = emu2.predict(Xt)
    assert np.allclose(var2, (c**2) * var1, rtol=1e-3)


def test_without_coscaling_length_scales_drift(design):
    """Negative control: a plain GP fed RAW alpha at Y vs c*Y drifts — this is
    the failure mode co-scaling fixes."""
    X, Y, alpha = design
    c = 10.0

    def _raw_fit(y, a):
        kernel = Matern(length_scale=np.ones(3), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=a, normalize_y=True, n_restarts_optimizer=0, random_state=42
        )
        gp.fit(X, y)
        return np.atleast_1d(gp.kernel_.length_scale)

    ls1 = _raw_fit(Y, alpha)  # raw alpha, no co-scaling
    ls2 = _raw_fit(c * Y, (c**2) * alpha)  # raw alpha at rescaled Y
    assert not np.allclose(ls1, ls2, rtol=1e-2)


def test_cross_validate_returns_per_fold_variances(design):
    X, Y, alpha = design
    cv = GPEmulator(n_restarts_optimizer=0).cross_validate(X, Y, alpha, k_folds=5, seed=0)
    n = len(X)
    assert cv["y_true"].shape == (n,)
    assert cv["y_pred"].shape == (n,)
    assert cv["pred_var"].shape == (n,)
    assert np.all(cv["pred_var"] >= 0.0)
    assert len(cv["fold_rmse"]) == 5
    assert len(cv["fold_r2"]) == 5
    assert isinstance(cv["mean_rmse"], float)
    assert isinstance(cv["mean_r2"], float)
    # A smooth low-noise target should cross-validate well.
    assert cv["mean_r2"] > 0.8


def test_cross_validate_accepts_scalar_alpha(design):
    X, Y, _ = design
    cv = GPEmulator(n_restarts_optimizer=0).cross_validate(X, Y, 1e-3, k_folds=4, seed=1)
    assert cv["pred_var"].shape == (len(X),)


def test_cross_validate_too_few_samples_raises():
    X = np.zeros((3, 2))
    Y = np.zeros(3)
    alpha = np.full(3, 1e-3)
    with pytest.raises(ValueError, match="k_folds"):
        GPEmulator().cross_validate(X, Y, alpha, k_folds=5)
