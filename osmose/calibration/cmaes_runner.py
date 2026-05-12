"""Covariance Matrix Adaptation Evolution Strategy (CMA-ES) wrapper.

CMA-ES typically converges in 2–3× fewer evaluations than scipy's
differential evolution on continuous noisy black-box problems in 20–50
dimensions. Use this when DE is converging slowly and the objective is
reasonably smooth (no hard discontinuities).

Tier C2 of the speedup roadmap. The `cma` package is already in the venv.

Bound handling: cma 4.x defaults to ``BoundaryHandler=BoundTransform`` when
``bounds`` is set — a smooth bijective transform that maps R^D into the
bounded hypercube, so the internal distribution adapts in unbounded space
and candidates returned by ``es.ask()`` are always feasible. We rely on this
default rather than the alternative ``BoundPenalty`` (clip-and-penalize),
which would let the distribution mean drift outside bounds in high dim.

Determinism note: the ``seed`` argument seeds cma's internal RNG, so
single-worker runs are reproducible. With ``workers > 1`` joblib's task
scheduling can interact with worker-side numpy RNG state in non-deterministic
ways, so reproducibility is best-effort. For strict bit-for-bit
reproducibility, run with ``workers=1`` or ensure the objective seeds its
own RNG explicitly per call.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cma  # type: ignore[import-untyped]
import joblib  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from osmose.calibration.checkpoint import _write_progress_checkpoint


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
    *,
    param_keys: list[str] | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 0,
    phase: str = "unknown",
    evaluator: Any = None,
    banded_targets: dict[str, tuple[float, float]] | None = None,
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

    # Clip x0 into bounds before handing it to cma — cma's BoundTransform
    # rejects out-of-bounds initial means with ValueError ("argument of
    # inverse must be within the given bounds"). Warn on silent mutation.
    x0_arr = np.asarray(x0, dtype=float)
    x0_clipped = np.clip(x0_arr, bounds_arr[:, 0], bounds_arr[:, 1])
    if not np.allclose(x0_clipped, x0_arr):
        warnings.warn(
            f"x0 clipped into bounds before CMA-ES init: "
            f"{(x0_clipped != x0_arr).sum()} of {n_dim} components changed",
            stacklevel=2,
        )

    cma_opts = {
        "bounds": [bounds_arr[:, 0].tolist(), bounds_arr[:, 1].tolist()],
        "popsize": popsize,
        "tolfun": tol,
        "maxiter": maxiter,
        "seed": seed,
        "verbose": 1 if verbose else -9,
    }
    es = cma.CMAEvolutionStrategy(x0_clipped.tolist(), sigma0, cma_opts)

    logger = logging.getLogger("osmose.calibration.cmaes_runner")
    _checkpoint_state = {
        "gen": 0,
        "best_fun_seen": float("inf"),
        "gens_since_improvement": 0,
        "start_time": time.time(),
        "persistence_failure_notified": False,
    }

    history: list[dict[str, Any]] = []
    # Track the largest finite cost ever observed across all generations so
    # NaN replacements stay strictly worse than any real datum — preventing a
    # partial-NaN gen from injecting an artificially low "penalty" that would
    # corrupt cma's covariance update.
    worst_ever: float = -np.inf

    while not es.stop():
        candidates = es.ask()
        if workers > 1:
            values = joblib.Parallel(n_jobs=workers, batch_size=1)(
                joblib.delayed(objective)(np.asarray(c, dtype=float)) for c in candidates
            )
        else:
            values = [float(objective(np.asarray(c, dtype=float))) for c in candidates]

        values_arr = np.asarray(values, dtype=float)
        finite_mask = np.isfinite(values_arr)
        if finite_mask.any():
            worst_ever = max(worst_ever, float(np.max(values_arr[finite_mask])))

        nan_mask = ~finite_mask
        if nan_mask.any():
            # Penalty is strictly worse than any finite value seen so far,
            # across all generations. If we've never seen a finite value,
            # use a large fallback so cma can still proceed.
            if np.isfinite(worst_ever):
                penalty = worst_ever * 1.1 + 1.0
            else:
                penalty = 1e9
            values_arr[nan_mask] = penalty
            values = values_arr.tolist()

        es.tell(candidates, values)
        history.append({
            "gen": int(es.countiter),
            "best": float(es.best.f),
            "mean": float(np.mean(values_arr)),
            "std": float(np.std(values_arr)),
        })

        _checkpoint_state["gen"] += 1
        _best_x = np.clip(
            np.asarray(es.best.x, dtype=float),
            bounds_arr[:, 0], bounds_arr[:, 1],
        )
        _best_fun = float(es.best.f)
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
                optimizer="cmaes",
                phase=phase,
                generation_budget=maxiter,
                param_keys=param_keys,
                bounds=bounds,
                evaluator=evaluator,
                banded_targets=banded_targets,
                logger=logger,
            )

    # Map cma's stop reason to a meaningful success flag. Budget-exhausted or
    # degenerate-state stops are NOT successful convergence. Also: if every
    # single eval returned NaN (worst_ever stays -inf), cma stops via tolfun
    # because all penalty values are equal — that's not real convergence.
    stop_reasons = es.stop()
    non_convergence = {"maxiter", "maxfevals", "conditioncov",
                       "noeffectaxis", "noeffectcoord"}
    success = bool(
        stop_reasons
        and not (set(stop_reasons.keys()) & non_convergence)
        and np.isfinite(worst_ever)  # at least one real datum seen
    )

    # Defensive clip on best.x — under BoundTransform es.best.x is always in
    # bounds, but if a future cma version or option change breaks that
    # invariant, a downstream caller should never receive an infeasible point.
    best_x = np.clip(np.asarray(es.best.x, dtype=float),
                     bounds_arr[:, 0], bounds_arr[:, 1])

    return {
        "x": best_x,
        "fun": float(es.best.f),
        "success": success,
        "message": str(stop_reasons),
        "nfev": int(es.countevals),
        "history": history,
    }
