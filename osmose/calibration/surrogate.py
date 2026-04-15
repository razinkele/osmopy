# osmose/calibration/surrogate.py
"""Gaussian Process surrogate model for fast OSMOSE calibration."""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore[import-untyped]
from sklearn.gaussian_process.kernels import Matern  # type: ignore[import-untyped]
from scipy.stats.qmc import LatinHypercube


class SurrogateCalibrator:
    """GP surrogate model that emulates OSMOSE for fast optimization.

    Workflow:
    1. generate_samples() -- Latin hypercube sampling of parameter space
    2. (user runs OSMOSE for each sample and collects results)
    3. fit(X_train, y_train) -- Fit GP model
    4. predict(X) -- Cheap prediction on new points
    5. optimize() -- Find optimal parameters on the surrogate
    """

    def __init__(
        self,
        param_bounds: list[tuple[float, float]],
        n_objectives: int = 1,
        n_restarts_optimizer: int = 2,
    ):
        self.param_bounds = param_bounds
        self.n_objectives = n_objectives
        self.n_params = len(param_bounds)
        self.models: list[GaussianProcessRegressor] = []
        self._is_fitted = False
        self._n_restarts_optimizer = n_restarts_optimizer
        self.fit_score_: float | None = None

    def generate_samples(self, n_samples: int = 500, seed: int = 42) -> np.ndarray:
        """Generate Latin hypercube samples in the parameter space.

        Returns:
            Array of shape (n_samples, n_params) with values in parameter bounds.
        """
        sampler = LatinHypercube(d=self.n_params, seed=seed)  # type: ignore[call-arg]
        unit_samples = sampler.random(n=n_samples)
        # Scale to parameter bounds
        lower = np.array([b[0] for b in self.param_bounds])
        upper = np.array([b[1] for b in self.param_bounds])
        return unit_samples * (upper - lower) + lower

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit GP models to training data.

        Args:
            X: Training inputs of shape (n_samples, n_params).
            y: Training outputs of shape (n_samples, n_objectives).
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.models = []
        for obj_idx in range(y.shape[1]):
            kernel = Matern(nu=2.5)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=self._n_restarts_optimizer,
                random_state=42,
            )
            gp.fit(X, y[:, obj_idx])
            self.models.append(gp)

        self.n_objectives = y.shape[1]
        self._is_fitted = True

        # Compute in-sample R² as a sanity check
        # Note: y has already been reshaped to 2D earlier in fit(), so y[:, 0] is safe
        means, _ = self.predict(X)
        y_col = y[:, 0]
        ss_res = float(np.sum((y_col - means[:, 0]) ** 2))
        ss_tot = float(np.sum((y_col - np.mean(y_col)) ** 2))
        self.fit_score_ = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict objective values and uncertainties.

        Returns:
            (means, stds) -- each of shape (n_samples, n_objectives).
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before predict()")

        means = np.zeros((X.shape[0], self.n_objectives))
        stds = np.zeros((X.shape[0], self.n_objectives))

        for i, model in enumerate(self.models):
            m, s = model.predict(X, return_std=True)
            means[:, i] = m
            stds[:, i] = s

        return means, stds

    def find_optimum(self, n_candidates: int = 10000, seed: int = 123) -> dict:
        """Find the optimum on the surrogate by evaluating many random candidates.

        Returns dict with 'params', 'predicted_objectives', 'predicted_uncertainty'.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before find_optimum()")

        candidates = self.generate_samples(n_candidates, seed=seed)
        means, stds = self.predict(candidates)

        # For single objective: find minimum
        # For multi-objective: return Pareto front candidates
        if self.n_objectives == 1:
            best_idx = np.argmin(means[:, 0])
            return {
                "params": candidates[best_idx],
                "predicted_objectives": means[best_idx],
                "predicted_uncertainty": stds[best_idx],
            }
        else:
            # Return top candidates by sum of objectives (simple aggregation)
            scores = np.sum(means, axis=1)
            best_idx = np.argmin(scores)
            return {
                "params": candidates[best_idx],
                "predicted_objectives": means[best_idx],
                "predicted_uncertainty": stds[best_idx],
            }

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k_folds: int = 5,
        seed: int = 42,
    ) -> dict:
        """K-fold cross-validation of the surrogate.

        Raises ValueError if len(X) < k_folds.
        """
        if len(X) < k_folds:
            raise ValueError(
                f"Need at least k_folds={k_folds} samples, got {len(X)}"
            )

        from sklearn.model_selection import KFold  # type: ignore[import-untyped]

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        fold_rmse: list[float] = []
        fold_r2: list[float] = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            fold_cal = SurrogateCalibrator(
                param_bounds=self.param_bounds,
                n_objectives=y.shape[1],
                n_restarts_optimizer=self._n_restarts_optimizer,
            )
            fold_cal.fit(X_train, y_train)
            means, _ = fold_cal.predict(X_test)

            y_col = y_test[:, 0]
            pred_col = means[:, 0]
            rmse = float(np.sqrt(np.mean((y_col - pred_col) ** 2)))
            ss_res = float(np.sum((y_col - pred_col) ** 2))
            ss_tot = float(np.sum((y_col - np.mean(y_col)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            fold_rmse.append(rmse)
            fold_r2.append(r2)

        return {
            "fold_rmse": fold_rmse,
            "fold_r2": fold_r2,
            "mean_rmse": float(np.mean(fold_rmse)),
            "mean_r2": float(np.mean(fold_r2)),
            "std_rmse": float(np.std(fold_rmse)),
            "std_r2": float(np.std(fold_r2)),
        }
