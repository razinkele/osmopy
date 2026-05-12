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

import logging
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution
from scipy.stats.qmc import LatinHypercube
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore[import-untyped]
from sklearn.gaussian_process.kernels import Matern, WhiteKernel  # type: ignore[import-untyped]

from osmose.calibration.checkpoint import _write_progress_checkpoint


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
    *,
    param_keys: list[str] | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 0,
    phase: str = "unknown",
    evaluator: Any = None,
    banded_targets: dict[str, tuple[float, float]] | None = None,
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

    logger = logging.getLogger("osmose.calibration.surrogate_de")
    _checkpoint_state = {
        "gen": 0,
        "best_fun_seen": float("inf"),
        "gens_since_improvement": 0,
        "start_time": time.time(),
        "persistence_failure_notified": False,
    }

    # Phase 1: LHS init + real-eval
    lhs = LatinHypercube(d=n_dim, seed=rng)
    init_samples = bounds_arr[:, 0] + lhs.random(n_initial) * (bounds_arr[:, 1] - bounds_arr[:, 0])
    if x0 is not None:
        x0_arr = np.asarray(x0, dtype=float)
        x0_clipped = np.clip(x0_arr, bounds_arr[:, 0], bounds_arr[:, 1])
        if not np.allclose(x0_clipped, x0_arr):
            # Silent mutation of a warm-start would mislead callers comparing
            # the result against the seed they passed in. Surface it.
            warnings.warn(
                f"x0 clipped into bounds: "
                f"{int((x0_clipped != x0_arr).sum())} of {n_dim} components changed",
                stacklevel=2,
            )
        init_samples[0] = x0_clipped

    # Track real-eval budget separately from training-set size — NaN evals are
    # dropped from training but DID consume budget; nfev must reflect that.
    total_real_evals = 0

    Y = _eval_batch(objective, init_samples, workers)
    total_real_evals += int(len(Y))
    finite = np.isfinite(Y)
    if not finite.any():
        raise RuntimeError("All initial evaluations returned NaN/inf — objective is broken")

    X_train = init_samples[finite].copy()
    y_train = Y[finite].copy()

    history: list[dict[str, Any]] = [{
        "phase": "init",
        "real_evals": int(len(Y)),
        "real_evals_finite": int(finite.sum()),
        "best": float(np.min(y_train)),
        "n_train": int(finite.sum()),
    }]
    if verbose:
        print(
            f"[surrogate-DE] init: {finite.sum()}/{len(Y)} finite evals, "
            f"best={np.min(y_train):.4f}"
        )

    # Phase 2..N: surrogate refinement
    for it in range(n_iterations):
        # Fit GP. normalize_y=True keeps the prior mean tracking the data; the
        # WhiteKernel adds a learnable noise floor so near-duplicate training
        # points (a real risk when many evals NaN and survivors cluster) do not
        # produce a singular covariance matrix.
        kernel = Matern(nu=2.5) + WhiteKernel(
            noise_level=1e-3, noise_level_bounds=(1e-7, 1.0)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=int(seed) + it,
        )
        try:
            gp.fit(X_train, y_train)
        except (np.linalg.LinAlgError, ValueError) as exc:
            # Singular covariance or other fit failure — fall back: skip the
            # surrogate-DE step this iteration and sample fresh LHS candidates
            # so we keep accumulating training data instead of crashing.
            warnings.warn(
                f"surrogate-DE: GP fit failed at iter {it} ({type(exc).__name__}: {exc}); "
                f"falling back to LHS exploration this iteration",
                stacklevel=2,
            )
            fallback_lhs = LatinHypercube(d=n_dim, seed=rng)
            candidates = bounds_arr[:, 0] + fallback_lhs.random(n_topk) * (
                bounds_arr[:, 1] - bounds_arr[:, 0]
            )
            new_y = _eval_batch(objective, candidates, workers)
            total_real_evals += int(len(new_y))
            finite_new = np.isfinite(new_y)
            if finite_new.any():
                X_train = np.vstack([X_train, candidates[finite_new]])
                y_train = np.concatenate([y_train, new_y[finite_new]])
            best_idx = int(np.argmin(y_train))
            history.append({
                "phase": f"iter{it}_lhs_fallback",
                "real_evals": int(len(candidates)),
                "best": float(y_train[best_idx]),
                "gp_de_pred": float("nan"),
                "gp_de_real": float("nan"),
                "n_train": int(len(y_train)),
            })

            _checkpoint_state["gen"] += 1
            _best_x = X_train[best_idx]
            _best_fun = float(y_train[best_idx])
            _prior_best = _checkpoint_state["best_fun_seen"]
            if _best_fun < _prior_best:
                _checkpoint_state["best_fun_seen"] = _best_fun
                _checkpoint_state["gens_since_improvement"] = 0
            else:
                _checkpoint_state["gens_since_improvement"] += 1

            if (
                checkpoint_path is not None
                and checkpoint_every > 0
                and param_keys is not None
                and _checkpoint_state["gen"] % checkpoint_every == 0
            ):
                _write_progress_checkpoint(
                    checkpoint_path=checkpoint_path,
                    state=_checkpoint_state,
                    best_x=_best_x,
                    best_fun=_best_fun,
                    optimizer="surrogate-de",
                    phase=phase,
                    generation_budget=n_iterations,
                    param_keys=param_keys,
                    bounds=bounds,
                    evaluator=evaluator,
                    banded_targets=banded_targets,
                    logger=logger,
                )
            continue

        # DE on GP-predicted mean — cheap, treat as scalar function. The closure
        # captures `gp` by reference; this is correct because the inner DE call
        # is synchronous and `gp` is not reassigned until the next loop iteration.
        # Do NOT pass this closure to `_eval_batch` (which serialises across
        # worker processes); only the real `objective` should be parallelised.
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
        total_real_evals += int(len(new_y))
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

        _checkpoint_state["gen"] += 1
        _best_x = X_train[best_idx]
        _best_fun = float(y_train[best_idx])
        _prior_best = _checkpoint_state["best_fun_seen"]
        if _best_fun < _prior_best:
            _checkpoint_state["best_fun_seen"] = _best_fun
            _checkpoint_state["gens_since_improvement"] = 0
        else:
            _checkpoint_state["gens_since_improvement"] += 1

        if (
            checkpoint_path is not None
            and checkpoint_every > 0
            and param_keys is not None
            and _checkpoint_state["gen"] % checkpoint_every == 0
        ):
            _write_progress_checkpoint(
                checkpoint_path=checkpoint_path,
                state=_checkpoint_state,
                best_x=_best_x,
                best_fun=_best_fun,
                optimizer="surrogate-de",
                phase=phase,
                generation_budget=n_iterations,
                param_keys=param_keys,
                bounds=bounds,
                evaluator=evaluator,
                banded_targets=banded_targets,
                logger=logger,
            )

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
        # nfev = total real evaluations actually consumed (including NaN'd
        # ones), not just the size of the final training set. Users budget
        # against real time spent.
        "nfev": int(total_real_evals),
        "history": history,
        "X_train": X_train,
        "y_train": y_train,
    }
