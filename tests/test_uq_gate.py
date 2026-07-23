"""Tests for the UQ calibration gate (coverage/PIT/MSSR decision)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats.qmc import LatinHypercube

from osmose.calibration.uq.gate import GateReport, evaluate_emulator_calibration

# The misspecified case intentionally feeds the GP an un-fittable step function,
# so its optimizer legitimately hits length-scale bounds. Silence that expected
# noise so test output stays pristine; the pass/fail assertions still guard the
# decision.
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def _synthetic_design(
    misspecified: bool, n: int = 60, s: int = 8, sigma: float = 0.15, seed: int = 0
):
    """Natural-log-mean design with Gaussian seed noise; well-specified is
    GP-friendly (smooth), misspecified is a step discontinuity a stationary GP
    cannot fit. Returns (X, Y=mean-of-logs, alpha=var(logs, ddof=1)/s)."""
    X = LatinHypercube(d=2, seed=seed).random(n=n)
    Y = np.empty(n)
    alpha = np.empty(n)
    for i in range(n):
        logs = np.empty(s)
        for k in range(s):
            rng = np.random.default_rng(1000 + i * s + k)
            if misspecified:
                mean_log = 5.0 if X[i, 0] > 0.5 else 1.0
            else:
                mean_log = 2.0 + np.sin(3.0 * X[i, 0]) + 0.5 * np.cos(4.0 * X[i, 1])
            logs[k] = mean_log + rng.normal(0.0, sigma)
        Y[i] = float(np.mean(logs))
        alpha[i] = float(np.var(logs, ddof=1)) / s
    return X, Y, alpha


def test_gate_passes_calibrated_synthetic():
    X, Y, alpha = _synthetic_design(misspecified=False)
    report = evaluate_emulator_calibration(X, Y, alpha, key="cod_biomass_mean")
    assert report.passed is True, report.reasons
    assert report.key == "cod_biomass_mean"
    assert report.n == 60


def test_gate_fails_miscalibrated_synthetic():
    X, Y, alpha = _synthetic_design(misspecified=True)
    report = evaluate_emulator_calibration(X, Y, alpha)
    assert report.passed is False
    # The step discontinuity blows up the standardized residuals.
    assert report.mssr > 2.5


def test_gate_insufficient_points_not_calibratable():
    X = np.linspace(0, 1, 5).reshape(-1, 1)
    Y = np.arange(5, dtype=float)
    alpha = np.full(5, 0.01)
    report = evaluate_emulator_calibration(X, Y, alpha)
    assert report.passed is False
    assert any("insufficient" in r.lower() for r in report.reasons)


def test_gate_report_is_dataclass_with_metrics():
    X, Y, alpha = _synthetic_design(misspecified=False, n=40)
    report = evaluate_emulator_calibration(X, Y, alpha)
    assert isinstance(report, GateReport)
    for field in ("coverage", "mssr", "pit_pvalue", "r2", "r2_ceiling"):
        assert isinstance(getattr(report, field), float)


def test_gate_loo_band_runs_and_returns_finite_metrics():
    # n=12 is in the LOO band (MIN_GATE_POINTS <= n <= LOO_MAX): gate uses
    # leave-one-out CV. Assert it executes and yields finite metrics, not a
    # brittle pass/fail decision.
    X, Y, alpha = _synthetic_design(misspecified=False, n=12)
    report = evaluate_emulator_calibration(X, Y, alpha)
    assert report.n == 12
    assert isinstance(report.passed, bool)
    for field in ("coverage", "mssr", "pit_pvalue", "r2", "r2_ceiling"):
        value = getattr(report, field)
        assert np.isfinite(value), f"{field} is not finite: {value}"
