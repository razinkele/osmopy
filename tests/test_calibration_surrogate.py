"""Tests for SurrogateCalibrator cross-validation and fit_score_."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.calibration.surrogate import SurrogateCalibrator


@pytest.fixture()
def sample_data():
    """Simple 1D function: y = x^2 with noise."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 10, size=(50, 1))
    y = X[:, 0] ** 2 + rng.normal(0, 1, size=50)
    return X, y


class TestFitScore:
    def test_fit_score_none_before_fit(self) -> None:
        cal = SurrogateCalibrator(param_bounds=[(0, 10)])
        assert cal.fit_score_ is None

    def test_fit_score_set_after_fit(self, sample_data) -> None:
        X, y = sample_data
        cal = SurrogateCalibrator(param_bounds=[(0, 10)], n_restarts_optimizer=0)
        cal.fit(X, y)
        assert cal.fit_score_ is not None
        assert isinstance(cal.fit_score_, float)

    def test_fit_score_near_one_for_gp(self, sample_data) -> None:
        """GP is an exact interpolator; in-sample R² should be very high."""
        X, y = sample_data
        cal = SurrogateCalibrator(param_bounds=[(0, 10)], n_restarts_optimizer=0)
        cal.fit(X, y)
        assert cal.fit_score_ > 0.99


class TestCrossValidate:
    def test_returns_expected_keys(self, sample_data) -> None:
        X, y = sample_data
        cal = SurrogateCalibrator(param_bounds=[(0, 10)], n_restarts_optimizer=0)
        result = cal.cross_validate(X, y, k_folds=3)
        assert "fold_rmse" in result
        assert "fold_r2" in result
        assert "mean_rmse" in result
        assert "mean_r2" in result
        assert "std_rmse" in result
        assert "std_r2" in result

    def test_fold_count_matches_k(self, sample_data) -> None:
        X, y = sample_data
        cal = SurrogateCalibrator(param_bounds=[(0, 10)], n_restarts_optimizer=0)
        result = cal.cross_validate(X, y, k_folds=5)
        assert len(result["fold_rmse"]) == 5
        assert len(result["fold_r2"]) == 5

    def test_mean_is_average_of_folds(self, sample_data) -> None:
        X, y = sample_data
        cal = SurrogateCalibrator(param_bounds=[(0, 10)], n_restarts_optimizer=0)
        result = cal.cross_validate(X, y, k_folds=3)
        assert result["mean_rmse"] == pytest.approx(np.mean(result["fold_rmse"]))
        assert result["mean_r2"] == pytest.approx(np.mean(result["fold_r2"]))

    def test_reasonable_r2_on_smooth_function(self) -> None:
        """Linear function + small noise is smooth — GP should generalize well."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 5, size=(50, 1))
        y = 2.0 * X[:, 0] + rng.normal(0, 0.1, size=50)
        cal = SurrogateCalibrator(param_bounds=[(0, 5)], n_restarts_optimizer=1)
        result = cal.cross_validate(X, y, k_folds=5)
        assert result["mean_r2"] > 0.8

    def test_raises_if_too_few_samples(self) -> None:
        X = np.array([[1.0], [2.0]])
        y = np.array([1.0, 4.0])
        cal = SurrogateCalibrator(param_bounds=[(0, 10)], n_restarts_optimizer=0)
        with pytest.raises(ValueError, match="k_folds"):
            cal.cross_validate(X, y, k_folds=5)


class TestFindOptimumMultiObjective:
    def test_find_optimum_multi_objective_returns_pareto_without_weights(self) -> None:
        """With multiple objectives and no weights, find_optimum returns the
        non-dominated (Pareto) set rather than an unweighted scalar sum."""
        rng = np.random.default_rng(7)
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        sc = SurrogateCalibrator(
            param_bounds=bounds, n_objectives=2, n_restarts_optimizer=0
        )

        X = rng.uniform(0, 1, size=(40, 2))
        # Competing objectives: obj0 prefers x0 small, obj1 prefers x0 large.
        y = np.stack([X[:, 0], 1.0 - X[:, 0]], axis=1)
        sc.fit(X, y)

        result = sc.find_optimum(n_candidates=500, seed=1)
        assert "pareto" in result, "must return a Pareto set key when no weights are given"
        pareto = result["pareto"]
        assert pareto["params"].shape[0] >= 2, "Pareto front has at least two points"
        assert pareto["objectives"].shape[1] == 2

    def test_find_optimum_multi_objective_with_weights_returns_single_point(self) -> None:
        """When explicit weights are supplied, a scalarized best single point
        must be returned (argmin of weighted-sum of posterior means)."""
        rng = np.random.default_rng(11)
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        sc = SurrogateCalibrator(
            param_bounds=bounds, n_objectives=2, n_restarts_optimizer=0
        )
        X = rng.uniform(0, 1, size=(40, 2))
        y = np.stack([X[:, 0], 1.0 - X[:, 0]], axis=1)
        sc.fit(X, y)

        result = sc.find_optimum(n_candidates=500, seed=1, weights=[1.0, 0.0])
        # With weight only on obj0 (prefers small x0), best x0 should be near 0.
        assert result["params"][0] < 0.2

    def test_find_optimum_weights_must_match_n_objectives(self) -> None:
        sc = SurrogateCalibrator(
            param_bounds=[(0.0, 1.0)], n_objectives=2, n_restarts_optimizer=0
        )
        X = np.array([[0.1], [0.5], [0.9]])
        y = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        sc.fit(X, y)
        with pytest.raises(ValueError, match="weights"):
            sc.find_optimum(weights=[0.5])  # wrong length
