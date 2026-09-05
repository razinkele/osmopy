"""Single-output GP emulator for one OSMOSE output stat.

Builds its OWN sklearn GP — it never touches the shared ``SurrogateCalibrator``
(the UI and ``find_optimum`` depend on that untouched). ARD ``Matern(2.5)``
(per-dimension length scales), per-point heteroscedastic noise via ``alpha``,
``normalize_y=True``, fit on natural-log outputs.
"""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore[import-untyped]
from sklearn.gaussian_process.kernels import Matern  # type: ignore[import-untyped]
from sklearn.model_selection import KFold  # type: ignore[import-untyped]


class SupportsPredict(Protocol):
    """The duck type ``make_log_posterior`` and the predictive diagnostic accept.

    Those callers take INJECTED emulators so synthetic tests can substitute an
    analytic one (see posterior.py's module docstring) — they must not require
    ``GPEmulator`` itself. They were previously annotated ``Mapping[str, object]``,
    which expressed "anything" rather than "anything with predict()" and made
    every ``.predict`` call a type error. This states the actual contract.
    """

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class GPEmulator:
    """One GP emulating a single natural-log output stat over sampling-space X.

    ``predict`` returns the LATENT (noise-free) posterior mean and variance in
    the same natural-log units as ``Y``; it does not add the training noise
    ``alpha`` back. Phase 1's calibration gate adds the held-out seed-mean noise
    ``s²/S`` itself when standardizing residuals.
    """

    def __init__(self, n_restarts_optimizer: int = 2, random_state: int = 42) -> None:
        self._n_restarts_optimizer = n_restarts_optimizer
        self._random_state = random_state
        self.gp: GaussianProcessRegressor | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray, alpha: np.ndarray | float) -> "GPEmulator":
        """Fit the GP. ``X`` is (n, d); ``Y`` is (n,) natural-log; ``alpha`` is
        per-point noise variance ``s²/S`` (scalar broadcast allowed)."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).ravel()
        alpha_arr = np.asarray(alpha, dtype=float)
        if alpha_arr.ndim == 0:
            alpha_arr = np.full(Y.shape[0], float(alpha_arr))

        # sklearn adds `alpha` to the kernel diagonal UNSCALED, but normalize_y
        # standardizes Y by its std. Co-scale the raw noise into standardized-Y
        # units so it is correct post-normalization. ddof=0 matches sklearn's
        # internal np.std(ddof=0); ddof=1 would inject an n/(n-1) error.
        var_Y = float(np.var(Y, ddof=0))
        alpha_scaled = alpha_arr / var_Y if var_Y > 0.0 else alpha_arr

        # Both ignores below are sklearn stub deficiencies, not loose typing:
        # `length_scale` is declared `float` but an array is exactly how sklearn
        # selects the ARD kernel (one length scale per dimension — the whole point
        # here, see the module docstring), and `alpha` is declared `float` but an
        # (n,) array is sklearn's documented per-point heteroscedastic noise. Both
        # array forms are load-bearing; do not "fix" these by passing scalars.
        kernel = Matern(
            length_scale=np.ones(X.shape[1]),  # type: ignore[arg-type]
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha_scaled,  # type: ignore[arg-type]
            normalize_y=True,
            n_restarts_optimizer=self._n_restarts_optimizer,
            random_state=self._random_state,
        )
        gp.fit(X, Y)
        self.gp = gp
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, var)`` in Y-units; ``var`` is latent (noise-free)."""
        if self.gp is None:
            raise RuntimeError("Must call fit() before predict()")
        # sklearn's `predict` is overloaded on `return_std`; the stub collapses the
        # return to a union including the bare-ndarray (return_std=False) form, so
        # unpacking and `std**2` both look invalid. return_std=True always yields
        # the 2-tuple at runtime — state that rather than suppress the symptom.
        mean, std = cast(
            "tuple[np.ndarray, np.ndarray]",
            self.gp.predict(np.asarray(X, dtype=float), return_std=True),
        )
        return mean, std**2

    def cross_validate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        alpha: np.ndarray | float,
        k_folds: int = 5,
        seed: int = 42,
    ) -> dict:
        """K-fold CV returning per-fold predictive variances for the Phase 1 gate.

        Fits a fresh emulator per fold and predicts on the held-out points.
        Returns ``y_true``/``y_pred``/``pred_var`` (each concatenated in fold
        order, not input order) plus per-fold and mean RMSE/R². ``pred_var`` is
        the latent predictive variance; the gate adds held-out seed-mean noise.
        Also returns ``test_idx``, the held-out row indices in fold-concatenation
        order (a permutation of ``range(len(X))``).
        Raises ``ValueError`` if ``len(X) < k_folds``.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).ravel()
        alpha_arr = np.asarray(alpha, dtype=float)
        if alpha_arr.ndim == 0:
            alpha_arr = np.full(Y.shape[0], float(alpha_arr))
        if len(X) < k_folds:
            raise ValueError(f"Need at least k_folds={k_folds} samples, got {len(X)}")

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        y_true: list[np.ndarray] = []
        y_pred: list[np.ndarray] = []
        pred_var: list[np.ndarray] = []
        test_idx: list[np.ndarray] = []
        fold_rmse: list[float] = []
        fold_r2: list[float] = []

        for train_idx, test_index in kf.split(X):
            fold = GPEmulator(self._n_restarts_optimizer, self._random_state)
            fold.fit(X[train_idx], Y[train_idx], alpha_arr[train_idx])
            mean, var = fold.predict(X[test_index])
            truth = Y[test_index]

            y_true.append(truth)
            y_pred.append(mean)
            pred_var.append(var)
            test_idx.append(test_index)
            fold_rmse.append(float(np.sqrt(np.mean((truth - mean) ** 2))))
            ss_res = float(np.sum((truth - mean) ** 2))
            ss_tot = float(np.sum((truth - np.mean(truth)) ** 2))
            fold_r2.append(1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0)

        return {
            "y_true": np.concatenate(y_true),
            "y_pred": np.concatenate(y_pred),
            "pred_var": np.concatenate(pred_var),
            "test_idx": np.concatenate(test_idx),
            "fold_rmse": fold_rmse,
            "fold_r2": fold_r2,
            "mean_rmse": float(np.mean(fold_rmse)),
            "mean_r2": float(np.mean(fold_r2)),
        }
