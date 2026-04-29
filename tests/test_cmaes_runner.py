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
