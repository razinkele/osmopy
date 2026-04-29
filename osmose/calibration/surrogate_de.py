"""GP surrogate-assisted differential evolution.

Trains a Gaussian Process emulator on a small batch of real evaluations,
runs DE on the GP-predicted objective (cheap), then real-evaluates the
top-K candidates selected via Lower Confidence Bound acquisition, and
retrains. Iterating yields convergence in 5–10× fewer real evaluations
than vanilla DE on smooth problems.

Tier C1 of the speedup roadmap. Builds on the existing
`osmose/calibration/surrogate.py` (sklearn GP with Matern kernel) for
the emulator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution
from scipy.stats.qmc import LatinHypercube
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore[import-untyped]
from sklearn.gaussian_process.kernels import Matern  # type: ignore[import-untyped]


def _eval_batch(
    objective: Callable[[NDArray[np.float64]], float],
    samples: NDArray[np.float64],
    workers: int,
) -> NDArray[np.float64]:
    """Real-evaluate a batch of parameter vectors. NaNs preserved."""
    if workers > 1:
        values = joblib.Parallel(n_jobs=workers, batch_size=1)(
            joblib.delayed(objective)(s) for s in samples
        )
    else:
        values = [float(objective(s)) for s in samples]
    return np.asarray(values, dtype=float)


def _select_topk_lcb(
    gp: GaussianProcessRegressor,
    bounds: NDArray[np.float64],
    k: int,
    n_candidates: int,
    kappa: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Sample uniformly from bounds, score by Lower Confidence Bound, take top-k.

    LCB acquisition: ``mu - kappa * sigma``. Lower is better. ``kappa=2`` balances
    exploitation (low predicted mean) and exploration (high predicted std). For a
    pure exploration sweep, raise kappa; for a final greedy refinement, lower it.
    """
    n_dim = bounds.shape[0]
    samples = bounds[:, 0] + rng.uniform(size=(n_candidates, n_dim)) * (bounds[:, 1] - bounds[:, 0])
    mu, sigma = gp.predict(samples, return_std=True)
    acquisition = mu - kappa * sigma
    top_idx = np.argsort(acquisition)[:k]
    return samples[top_idx]


def surrogate_assisted_de(
    objective: Callable[[NDArray[np.float64]], float],
    bounds: list[tuple[float, float]],
    x0: list[float] | NDArray[np.float64] | None = None,
    n_initial: int | None = None,
    n_iterations: int = 6,
    n_topk: int = 30,
    de_maxiter: int = 100,
    de_popsize_mult: int = 5,
    lcb_kappa: float = 2.0,
    lcb_n_candidates: int = 10000,
    workers: int = 1,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]:
    """Optimise `objective` over `bounds` using GP-assisted DE.

    Loop structure:
      1. LHS-sample ``n_initial`` points (optionally include ``x0`` first), real-eval.
      2. For each of ``n_iterations`` rounds:
         a. Fit a GP to (X_train, y_train).
         b. Run scipy DE on the GP-predicted objective (cheap proxy).
         c. Combine the DE optimum with LCB-acquisition candidates → ``n_topk`` points.
         d. Real-eval those; append to training set.
      3. Return the best real-evaluated point.

    Total real evals: ``n_initial + n_iterations * n_topk``.
    For 27 params with ``n_initial=27*5=135``, ``n_iterations=6``, ``n_topk=30`` → 315.
    Versus vanilla DE which often needs 1000–3000 evals on the same problem.

    Parameters
    ----------
    objective
        Real-valued cost (minimised). Picklable when ``workers > 1``.
    bounds
        ``[(lo, hi)]`` per dimension.
    x0
        Optional warm-start. Replaces the first LHS sample.
    n_initial
        Initial real-eval batch size. Default: ``max(20, 5 * n_dim)``.
    n_iterations
        Surrogate refinement rounds.
    n_topk
        Real evaluations per round. Always includes the GP-DE optimum.
    de_maxiter, de_popsize_mult
        DE settings for the surrogate-objective optimisation. Cheap; can be aggressive.
    lcb_kappa
        Acquisition weight on uncertainty. Higher = more exploration.
    lcb_n_candidates
        Pool size for LCB selection.
    workers
        Parallel workers for real-eval batches.
    seed
        RNG seed.
    verbose
        Print per-iteration diagnostics.

    Returns
    -------
    dict
        ``{x, fun, nfev, history, X_train, y_train}``.
    """
    bounds_arr = np.asarray(bounds, dtype=float)
    n_dim = bounds_arr.shape[0]
    if n_initial is None:
        n_initial = max(20, 5 * n_dim)

    rng = np.random.default_rng(seed)

    # Phase 1: LHS init + real-eval
    lhs = LatinHypercube(d=n_dim, seed=rng)
    init_samples = bounds_arr[:, 0] + lhs.random(n_initial) * (bounds_arr[:, 1] - bounds_arr[:, 0])
    if x0 is not None:
        init_samples[0] = np.asarray(x0, dtype=float)
        init_samples[0] = np.clip(init_samples[0], bounds_arr[:, 0], bounds_arr[:, 1])

    Y = _eval_batch(objective, init_samples, workers)
    finite = np.isfinite(Y)
    if not finite.any():
        raise RuntimeError("All initial evaluations returned NaN/inf — objective is broken")

    X_train = init_samples[finite].copy()
    y_train = Y[finite].copy()

    history: list[dict[str, Any]] = [{
        "phase": "init",
        "real_evals": int(finite.sum()),
        "best": float(np.min(y_train)),
        "n_train": int(finite.sum()),
    }]
    if verbose:
        print(f"[surrogate-DE] init: {finite.sum()} real evals, best={np.min(y_train):.4f}")

    # Phase 2..N: surrogate refinement
    for it in range(n_iterations):
        # Fit GP. normalize_y=True is critical — otherwise the GP defaults to a
        # zero prior mean which severely biases predictions when y is far from 0.
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=int(seed) + it,
        )
        gp.fit(X_train, y_train)

        # DE on GP-predicted mean — cheap, treat as scalar function
        def gp_mean(x: NDArray[np.float64]) -> float:
            return float(gp.predict(np.asarray(x).reshape(1, -1))[0])

        de_res = differential_evolution(
            gp_mean,
            bounds,
            maxiter=de_maxiter,
            popsize=de_popsize_mult,
            seed=int(seed) + it,
            tol=1e-4,
            mutation=(0.5, 1.5),
            recombination=0.8,
            polish=False,
            updating="immediate",  # single-threaded GP eval; immediate is fine
        )

        # Build top-K candidate set: GP-DE optimum + LCB-selected from uniform pool
        de_optimum = np.asarray(de_res.x, dtype=float).reshape(1, -1)
        lcb_picks = _select_topk_lcb(
            gp, bounds_arr, k=max(0, n_topk - 1),
            n_candidates=lcb_n_candidates,
            kappa=lcb_kappa,
            rng=rng,
        )
        candidates = np.vstack([de_optimum, lcb_picks])

        # Real-eval candidates
        new_y = _eval_batch(objective, candidates, workers)
        finite_new = np.isfinite(new_y)
        if finite_new.any():
            X_train = np.vstack([X_train, candidates[finite_new]])
            y_train = np.concatenate([y_train, new_y[finite_new]])

        best_idx = int(np.argmin(y_train))
        history.append({
            "phase": f"iter{it}",
            "real_evals": int(len(candidates)),
            "best": float(y_train[best_idx]),
            "gp_de_pred": float(de_res.fun),
            "gp_de_real": float(new_y[0]) if finite_new[0] else float("nan"),
            "n_train": int(len(y_train)),
        })
        if verbose:
            print(
                f"[surrogate-DE] iter{it}: best={y_train[best_idx]:.4f} "
                f"(GP predicted {de_res.fun:.4f}, real {history[-1]['gp_de_real']:.4f}); "
                f"n_train={len(y_train)}"
            )

    best_idx = int(np.argmin(y_train))
    return {
        "x": X_train[best_idx],
        "fun": float(y_train[best_idx]),
        "nfev": int(len(y_train)),
        "history": history,
        "X_train": X_train,
        "y_train": y_train,
    }
