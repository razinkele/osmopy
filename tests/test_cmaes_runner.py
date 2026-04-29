"""Smoke tests for the CMA-ES runner against analytical objectives."""
from __future__ import annotations

import numpy as np
import pytest

from osmose.calibration.cmaes_runner import run_cmaes


def _rosenbrock(x: np.ndarray) -> float:
    """Classic 2-D Rosenbrock: minimum at (1, 1) with value 0."""
    return float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)


def _sphere(x: np.ndarray) -> float:
    """N-D sphere: minimum at origin with value 0."""
    return float(np.sum(x ** 2))


def test_rosenbrock_2d_converges_near_optimum():
    result = run_cmaes(
        _rosenbrock,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        x0=[-2.0, 2.0],
        sigma0=0.5,
        maxiter=100,
        tol=1e-6,
        seed=42,
    )
    # Rosenbrock has a curved valley — CMA-ES should find (1,1) within tolerance
    assert result["fun"] < 1e-3, f"got fun={result['fun']}"
    assert np.allclose(result["x"], [1.0, 1.0], atol=0.05), f"x={result['x']}"
    assert result["nfev"] < 5000  # easy problem; should converge fast
    assert len(result["history"]) >= 1
    assert result["history"][-1]["best"] <= result["history"][0]["best"]


def test_sphere_10d_converges_to_origin():
    n_dim = 10
    result = run_cmaes(
        _sphere,
        bounds=[(-5.0, 5.0)] * n_dim,
        x0=[2.0] * n_dim,
        sigma0=0.3,
        maxiter=200,
        tol=1e-6,
        seed=42,
    )
    assert result["fun"] < 1e-4
    assert np.linalg.norm(result["x"]) < 0.05


def test_seed_reproducibility():
    """Same seed → same trajectory."""
    a = run_cmaes(_sphere, [(-5, 5)] * 3, x0=[1.0, 1.0, 1.0], maxiter=20, seed=7)
    b = run_cmaes(_sphere, [(-5, 5)] * 3, x0=[1.0, 1.0, 1.0], maxiter=20, seed=7)
    assert a["nfev"] == b["nfev"]
    assert pytest.approx(a["fun"], rel=1e-12) == b["fun"]


def test_nan_objective_does_not_stall():
    """If a candidate eval returns NaN, CMA-ES must keep going (penalty replacement)."""
    call_count = {"n": 0}

    def flaky(x):
        call_count["n"] += 1
        if call_count["n"] % 7 == 0:
            return float("nan")
        return _sphere(x)

    result = run_cmaes(
        flaky, [(-5, 5)] * 3, x0=[1.0, 1.0, 1.0],
        maxiter=30, seed=42,
    )
    # Just verify it terminated and got near the optimum despite the NaNs
    assert result["nfev"] > 0
    assert result["fun"] < 1.0


def test_all_nan_first_generation_terminates_gracefully():
    """Every candidate NaN'd → cma sees flat fitness and stops cleanly.

    This is correct behaviour: a totally broken objective should not be silently
    "calibrated" with random output. Verify the run doesn't crash and returns
    a sensible result dict.
    """
    def all_nan(_x):
        return float("nan")

    result = run_cmaes(
        all_nan, [(-5, 5)] * 3, x0=[1.0, 1.0, 1.0],
        maxiter=30, seed=42,
    )
    # Run terminated without exception, returned the expected schema
    assert "x" in result and "fun" in result and "success" in result
    assert "message" in result
    assert result["nfev"] > 0
    # Flat-fitness stop or similar; not a "real" success
    assert result["success"] is False


def test_partial_nan_uses_running_worst_penalty():
    """Penalty should be strictly worse than any finite value seen across all gens."""
    seen_finite = []

    def occasional_nan(x):
        v = _sphere(x)
        seen_finite.append(v)
        # NaN every 5th call; with popsize ~7 that's roughly 1-2 NaNs per gen
        if len(seen_finite) % 5 == 0:
            return float("nan")
        return v

    result = run_cmaes(
        occasional_nan, [(-5, 5)] * 3, x0=[1.0, 1.0, 1.0],
        maxiter=20, seed=42,
    )
    # Despite the NaNs, the optimization makes meaningful progress
    assert result["fun"] < 1.0
    # Best objective from history monotone non-increasing (no penalty injection
    # spuriously becoming the best)
    bests = [h["best"] for h in result["history"]]
    for prev, nxt in zip(bests[:-1], bests[1:]):
        assert nxt <= prev + 1e-9


def test_success_false_when_maxiter_hit():
    """maxiter exhaustion must NOT set success=True."""
    # Use a tiny maxiter so even a trivial sphere doesn't converge in time
    result = run_cmaes(
        _sphere, [(-5, 5)] * 5, x0=[2.0] * 5,
        sigma0=0.3, maxiter=2, tol=1e-12, seed=42,
    )
    assert result["success"] is False, (
        f"maxiter stop should not be success; got message={result['message']}"
    )
    assert "maxiter" in result["message"]


def test_success_true_on_genuine_convergence():
    """tolfun-triggered stop should set success=True."""
    result = run_cmaes(
        _sphere, [(-5, 5)] * 3, x0=[2.0] * 3,
        sigma0=0.3, maxiter=200, tol=1e-6, seed=42,
    )
    assert result["success"] is True
    assert result["fun"] < 1e-4


def test_x0_outside_bounds_is_clipped_with_warning():
    """Out-of-bounds x0 must be clipped (not crash cma), with a warning."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_cmaes(
            _sphere, [(-1, 1)] * 3, x0=[5.0, -3.0, 0.5],
            sigma0=0.1, maxiter=20, seed=42,
        )
    assert any("clipped into bounds" in str(w.message) for w in caught), (
        f"expected clip warning; caught: {[str(w.message) for w in caught]}"
    )
    # Returned x is in bounds
    assert all(-1.0 <= xi <= 1.0 for xi in result["x"])


def test_returned_best_x_is_in_bounds():
    """Defensive clip: result['x'] is always in bounds even if cma's internal best wasn't."""
    result = run_cmaes(
        _sphere, [(-2, 2)] * 4, x0=[0.5] * 4,
        sigma0=0.3, maxiter=30, seed=42,
    )
    assert all(-2.0 <= xi <= 2.0 for xi in result["x"])
