"""Single-output GP emulator for one OSMOSE output stat.

Builds its OWN sklearn GP — it never touches the shared ``SurrogateCalibrator``
(the UI and ``find_optimum`` depend on that untouched). ARD ``Matern(2.5)``
(per-dimension length scales), per-point heteroscedastic noise via ``alpha``,
``normalize_y=True``, fit on natural-log outputs.
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore[import-untyped]
from sklearn.gaussian_process.kernels import Matern  # type: ignore[import-untyped]


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

        kernel = Matern(
            length_scale=np.ones(X.shape[1]),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha_scaled,
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
        mean, std = self.gp.predict(np.asarray(X, dtype=float), return_std=True)
        return mean, std**2
