"""Smoke tests for the GP surrogate-assisted DE runner."""
from __future__ import annotations

import numpy as np
import pytest

from osmose.calibration.surrogate_de import (
    _select_topk_lcb,
    surrogate_assisted_de,
)


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def _rosenbrock(x: np.ndarray) -> float:
    return float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)


def test_sphere_5d_converges_below_threshold():
    """5-D sphere: should be near-zero with a small total real-eval budget."""
    result = surrogate_assisted_de(
        _sphere,
        bounds=[(-2.0, 2.0)] * 5,
        n_initial=25,
        n_iterations=4,
        n_topk=15,
        seed=42,
    )
    # Total real evals ≤ 25 + 4*15 = 85
    assert result["nfev"] <= 85
    assert result["fun"] < 0.1, f"got fun={result['fun']}"
    # History should record monotonic non-increase in best objective
    bests = [h["best"] for h in result["history"]]
    for prev, nxt in zip(bests[:-1], bests[1:]):
        assert nxt <= prev + 1e-9, f"best regressed: {prev} → {nxt}"


def test_rosenbrock_2d_progress():
    """Rosenbrock is harder; we don't require convergence to 0, just clear progress."""
    result = surrogate_assisted_de(
        _rosenbrock,
        bounds=[(-3.0, 3.0)] * 2,
        n_initial=20,
        n_iterations=5,
        n_topk=15,
        seed=42,
    )
    # Initial best vs final best: substantial improvement expected
    initial_best = result["history"][0]["best"]
    final_best = result["history"][-1]["best"]
    assert final_best < initial_best * 0.1, (
        f"insufficient improvement: {initial_best} → {final_best}"
    )


def test_warm_start_x0_is_first_sample():
    """x0 must override the first LHS slot, not get clipped or ignored."""
    result = surrogate_assisted_de(
        _sphere,
        bounds=[(-2.0, 2.0)] * 3,
        x0=[1.5, -1.5, 0.5],
        n_initial=10,
        n_iterations=1,
        n_topk=5,
        seed=42,
    )
    # The first row of X_train is the LHS init (with x0 overriding slot 0).
    # After 1 iteration we have init + topk candidates appended; the original
    # x0 still lives at row 0.
    assert np.allclose(result["X_train"][0], [1.5, -1.5, 0.5])


def test_lcb_acquisition_picks_low_predicted_mean():
    """Sanity-check LCB ranks candidates with low GP mean ahead of high mean."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(40, 2))
    # Train on a clear bowl: y = ||x||^2 + small noise
    y = np.sum(X ** 2, axis=1) + rng.normal(scale=0.01, size=40)
    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True, random_state=0)
    gp.fit(X, y)

    bounds_arr = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    picks = _select_topk_lcb(gp, bounds_arr, k=5, n_candidates=2000, kappa=2.0, rng=rng)

    # Top-5 LCB picks should cluster near origin (the optimum)
    norms = np.linalg.norm(picks, axis=1)
    assert np.mean(norms) < 1.0, f"LCB picks not near optimum: mean norm {np.mean(norms)}"


def test_all_nan_initial_raises():
    """If every initial eval returns NaN, fail loudly rather than silently degrade."""
    def bad(_x):
        return float("nan")

    with pytest.raises(RuntimeError, match="All initial evaluations returned NaN"):
        surrogate_assisted_de(
            bad,
            bounds=[(-1.0, 1.0)] * 2,
            n_initial=5,
            n_iterations=1,
            n_topk=3,
            seed=42,
        )
