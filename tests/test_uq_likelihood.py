"""Tests for the UQ likelihoods (Gaussian split-normal + BandFaithful)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from osmose.calibration.uq.likelihood import (
    _log_interval_prob,
    band_faithful,
    gaussian_log_biomass,
)


def test_gaussian_normalizer_integrates_to_one():
    # For fixed band/var, the density over the residual (varying mu_emu) is proper.
    grid = np.linspace(-40.0, 40.0, 200001)
    vals = np.array(
        [
            gaussian_log_biomass(
                m, 0.05, 100.0, 60.0, 130.0, sigma_seed_sq=0.02, sigma_disc_sq=0.0, k=1.96
            )
            for m in grid
        ]
    )
    integral = np.trapezoid(np.exp(vals), grid)
    assert abs(integral - 1.0) < 1e-3


def test_gaussian_continuous_at_r_zero():
    # r = 0 when mu_emu = ln(target) - 0.5*sigma_seed_sq. Both branches must agree.
    mu0 = np.log(100.0) - 0.5 * 0.02
    left = gaussian_log_biomass(
        mu0 - 1e-7, 0.03, 100.0, 70.0, 140.0, sigma_seed_sq=0.02, sigma_disc_sq=0.0, k=1.0
    )
    right = gaussian_log_biomass(
        mu0 + 1e-7, 0.03, 100.0, 70.0, 140.0, sigma_seed_sq=0.02, sigma_disc_sq=0.0, k=1.0
    )
    assert abs(left - right) < 1e-9


def test_gaussian_symmetric_band_peaks_at_jensen_corrected_target():
    # Symmetric (in log) band around target; the max over mu_emu sits at the
    # Jensen-corrected point mu_emu = ln(target) - 0.5*sigma_seed_sq.
    target, sig_seed = 100.0, 0.02
    lower, upper = target / 1.3, target * 1.3  # log-symmetric
    grid = np.linspace(np.log(50.0), np.log(200.0), 5001)
    vals = [
        gaussian_log_biomass(
            m, 0.01, target, lower, upper, sigma_seed_sq=sig_seed, sigma_disc_sq=0.0, k=1.0
        )
        for m in grid
    ]
    mode = grid[int(np.argmax(vals))]
    assert abs(mode - (np.log(target) - 0.5 * sig_seed)) < 1e-2


def test_gaussian_var_floor_degenerate_band_is_finite():
    # lower == upper (zero-width band) + sigma_disc=0 + tiny var must not blow up.
    v = gaussian_log_biomass(
        np.log(100.0), 1e-15, 100.0, 100.0, 100.0, sigma_seed_sq=0.0, sigma_disc_sq=0.0, k=1.0
    )
    assert np.isfinite(v)


def test_gaussian_asymmetric_band_uses_correct_side():
    # Prediction above target uses the upper sigma; below uses the lower. With a
    # tight upper band and wide lower band, an over-prediction is penalized harder
    # than an equal-magnitude under-prediction.
    target = 100.0
    lower, upper = 40.0, 110.0  # wide below, tight above
    lt = np.log(target)
    over = gaussian_log_biomass(
        lt + 0.2, 0.0, target, lower, upper, sigma_seed_sq=0.0, sigma_disc_sq=0.0, k=1.0
    )
    under = gaussian_log_biomass(
        lt - 0.2, 0.0, target, lower, upper, sigma_seed_sq=0.0, sigma_disc_sq=0.0, k=1.0
    )
    assert over < under


def test_band_faithful_high_inside_band():
    # Prediction centered in the band with tiny variance -> P ~ 1 -> loglik ~ 0.
    v = band_faithful(
        np.log(100.0),
        1e-4,
        100.0,
        80.0,
        120.0,
        sigma_seed_sq=0.0,
        sigma_disc_sq=0.0,
    )
    assert v > -0.05


def test_band_faithful_decays_and_stays_finite_far_outside():
    inside = band_faithful(
        np.log(100.0),
        0.01,
        100.0,
        80.0,
        120.0,
        sigma_seed_sq=0.0,
        sigma_disc_sq=0.0,
    )
    far = band_faithful(
        np.log(1e6),
        0.01,
        100.0,
        80.0,
        120.0,
        sigma_seed_sq=0.0,
        sigma_disc_sq=0.0,
    )
    very_far = band_faithful(
        np.log(1e-6),
        0.01,
        100.0,
        80.0,
        120.0,
        sigma_seed_sq=0.0,
        sigma_disc_sq=0.0,
    )
    assert far < inside
    assert np.isfinite(far) and np.isfinite(very_far)


def test_log_interval_prob_matches_naive_for_mild_case():
    # Mild case (band around the center) where the naive difference is stable.
    lo_z, hi_z = -1.0, 1.5
    naive = np.log(norm.cdf(hi_z) - norm.cdf(lo_z))
    assert abs(_log_interval_prob(lo_z, hi_z) - naive) < 1e-9


def test_log_interval_prob_reflection_symmetry():
    # log(Phi(b)-Phi(a)) == log(Phi(-a)-Phi(-b)) by symmetry of the normal.
    assert abs(_log_interval_prob(2.0, 4.0) - _log_interval_prob(-4.0, -2.0)) < 1e-9
