"""Tests for the UQ end-to-end orchestrator (run_surrogate_bayes) + helpers."""

from __future__ import annotations

import numpy as np

from osmose.calibration.problem import FreeParameter, Transform
from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.design import DesignResult
from osmose.calibration.uq.gate import GateReport
from osmose.calibration.uq.run import UQResult, derive_sigma_seed_sq, run_surrogate_bayes


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


_THETA_STAR = np.array([0.3, 0.6])
_SIG_SEED = 0.03

# Monotonic over [0,1] (identifiable) AND smooth (GP-calibratable).
_MEANS = {
    "A_biomass_mean": lambda t: 2.0 + np.sin(1.5 * t[0]),
    "B_biomass_mean": lambda t: 1.0 + np.sin(1.5 * t[1]),
    "C_biomass_mean": lambda t: 0.5 + np.sin(1.2 * (t[0] + t[1])),
}


def _fp2():
    return [
        FreeParameter("a.sp0", 0.0, 1.0, Transform.LINEAR),
        FreeParameter("b.sp0", 0.0, 1.0, Transform.LINEAR),
    ]


def _evaluator(x, seed):
    rng = np.random.default_rng(int(seed))
    return {k: float(np.exp(f(x) + rng.normal(0.0, _SIG_SEED))) for k, f in _MEANS.items()}


def _targets():
    ts = []
    for key, f in _MEANS.items():
        value = float(np.exp(f(_THETA_STAR) + 0.5 * _SIG_SEED**2))
        ts.append(
            BiomassTarget(
                species=key.split("_")[0],
                target=value,
                lower=value * 0.8,
                upper=value * 1.2,
                reference_point_type="biomass",
            )
        )
    return ts


def _pass_gate(X, Y, alpha, **kw):
    return GateReport(len(X), 0.95, 1.0, 0.5, 0.9, 0.95, True, [])


def _fail_gate(X, Y, alpha, **kw):
    return GateReport(len(X), 0.5, 9.0, 0.001, 0.5, 0.9, False, ["synthetic-fail"])


def test_run_surrogate_bayes_recovers_theta_star():
    # Injected always-pass gate: tests the full grow->fit->sigma_seed->posterior->
    # sample composition recovers theta* WITHOUT hinging on a synthetic clearing
    # the real gate. Recovery is through a FITTED emulator (looser than Phase 2b).
    result = run_surrogate_bayes(
        _evaluator,
        _fp2(),
        _targets(),
        n_seeds=6,
        n0=40,
        increment=20,
        n_max=100,
        seed=0,
        gate_fn=_pass_gate,
    )
    assert result.status == "ok"
    assert np.allclose(result.posterior_mean, _THETA_STAR, atol=0.12)
    assert result.sampler_result.converged


def test_run_surrogate_bayes_result_fields_populated():
    result = run_surrogate_bayes(
        _evaluator,
        _fp2(),
        _targets(),
        n_seeds=6,
        n0=40,
        increment=20,
        n_max=100,
        seed=0,
        gate_fn=_pass_gate,
    )
    assert set(result.gate_reports) == set(_MEANS)
    assert set(result.n_censored) == set(_MEANS)
    assert result.sampler_result is not None
    assert result.design.X.shape[1] == 2
