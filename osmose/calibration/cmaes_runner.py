"""Covariance Matrix Adaptation Evolution Strategy (CMA-ES) wrapper.

CMA-ES typically converges in 2–3× fewer evaluations than scipy's
differential evolution on continuous noisy black-box problems in 20–50
dimensions. Use this when DE is converging slowly and the objective is
reasonably smooth (no hard discontinuities).

Tier C2 of the speedup roadmap. The `cma` package is already in the venv.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cma  # type: ignore[import-untyped]
import joblib  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray


def run_cmaes(
    objective: Callable[[NDArray[np.float64]], float],
    bounds: list[tuple[float, float]],
    x0: list[float] | NDArray[np.float64],
    sigma0: float = 0.3,
    popsize: int | None = None,
    maxiter: int = 200,
    tol: float = 0.005,
    seed: int = 42,
    workers: int = 1,
    verbose: bool = False,
) -> dict[str, Any]:
    """Optimise `objective` over `bounds` via CMA-ES.

    Parameters
    ----------
    objective
        Maps a length-D parameter vector to a real-valued cost (minimised).
        Must be picklable when ``workers > 1``.
    bounds
        ``[(lo, hi)]`` per dimension. Same conventions as scipy.optimize.
    x0
        Initial mean of the search distribution.
    sigma0
        Initial step size as a fraction of the (uniform) bound width. CMA-ES
        adapts this internally; 0.3 is a common starting choice.
    popsize
        Population (lambda) per generation. ``None`` uses CMA's default
        ``4 + int(3 * log(D))`` which is sample-efficient for D ≤ 50.
    maxiter
        Maximum number of generations.
    tol
        ``tolfun`` — convergence triggers when the per-generation objective
        std drops below this for ``cma.options['tolflatfitness']``
        consecutive gens.
    seed
        RNG seed for the search distribution. Determinism also requires the
        objective itself to be deterministic at fixed seed.
    workers
        ``joblib.Parallel(n_jobs=workers)`` evaluates each generation's
        candidates in parallel. ``1`` = sequential.
    verbose
        Forward CMA's per-generation diagnostic to stdout.

    Returns
    -------
    dict
        ``{x, fun, success, message, nfev, history}``. ``history`` is a list
        of per-generation ``{gen, best, mean, std}``.
    """
    bounds_arr = np.asarray(bounds, dtype=float)
    n_dim = bounds_arr.shape[0]
    if popsize is None:
        popsize = 4 + int(3 * np.log(n_dim))

    cma_opts = {
        "bounds": [bounds_arr[:, 0].tolist(), bounds_arr[:, 1].tolist()],
        "popsize": popsize,
        "tolfun": tol,
        "maxiter": maxiter,
        "seed": seed,
        "verbose": 1 if verbose else -9,
    }
    es = cma.CMAEvolutionStrategy(np.asarray(x0, dtype=float).tolist(), sigma0, cma_opts)

    history: list[dict[str, Any]] = []

    while not es.stop():
        candidates = es.ask()
        if workers > 1:
            values = joblib.Parallel(n_jobs=workers, batch_size=1)(
                joblib.delayed(objective)(np.asarray(c, dtype=float)) for c in candidates
            )
        else:
            values = [float(objective(np.asarray(c, dtype=float))) for c in candidates]

        # Replace any NaN/inf with a large but finite penalty so CMA's
        # mean/cov updates stay well-defined; non-finite values cause cma
        # to refuse the gen and silently stall.
        values_arr = np.asarray(values, dtype=float)
        nan_mask = ~np.isfinite(values_arr)
        if nan_mask.any():
            penalty = float(np.nanmax(values_arr[~nan_mask])) + 1.0 if (~nan_mask).any() else 1e9
            values_arr[nan_mask] = penalty
            values = values_arr.tolist()

        es.tell(candidates, values)
        history.append({
            "gen": int(es.countiter),
            "best": float(es.best.f),
            "mean": float(np.mean(values_arr)),
            "std": float(np.std(values_arr)),
        })

    return {
        "x": np.asarray(es.best.x, dtype=float),
        "fun": float(es.best.f),
        "success": True,
        "message": str(es.stop()),
        "nfev": int(es.countevals),
        "history": history,
    }
