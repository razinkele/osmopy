"""Seeded LHS design executed through the Python engine, reduced for the UQ emulator.

The design runs through an INJECTABLE evaluator — a ``(point, seed) -> stat-dict``
callable — so the whole pipeline is testable without OSMOSE runs. The real
evaluator (``make_engine_evaluator``) runs the Python engine with SSB output
enabled; tests pass a synthetic function.

Two transforms live here and must not be conflated: base-10 ``10**val`` at the
simulator-input boundary (``point_to_overrides``, mirroring problem.py:263) and
natural ``np.log`` of the linear stat when forming the GP target Y (``run_design``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.stats.qmc import LatinHypercube

from osmose.calibration.problem import FreeParameter, Transform
from osmose.calibration.uq.gate import GateReport, evaluate_emulator_calibration
from osmose.calibration.uq.output_stats import compute_uq_stats


def point_to_overrides(x: np.ndarray, free_params: list[FreeParameter]) -> dict[str, str]:
    """One sampling-space point -> OSMOSE override dict.

    Applies base-10 ``10**val`` for ``Transform.LOG`` params (the simulator-input
    transform, matching osmose/calibration/problem.py:263) and stringifies every
    value. NOT the natural-log GP-target transform, which is separate.
    """
    overrides: dict[str, str] = {}
    for j, fp in enumerate(free_params):
        val = float(x[j])
        if fp.transform == Transform.LOG:
            val = 10.0**val
        overrides[fp.key] = str(val)
    return overrides


def lhs_design(free_params: list[FreeParameter], n_points: int, seed: int) -> np.ndarray:
    """Seeded Latin-hypercube design, ``(n_points, d)``, scaled to sampling-space bounds.

    No transform is applied — the design lives in sampling space; the emulator
    trains on sampling-space X and ``point_to_overrides`` transforms only at the
    simulator-input boundary.
    """
    d = len(free_params)
    unit = LatinHypercube(d=d, seed=seed).random(n=n_points)
    lower = np.array([fp.lower_bound for fp in free_params])
    upper = np.array([fp.upper_bound for fp in free_params])
    return unit * (upper - lower) + lower


Evaluator = Callable[[np.ndarray, int], dict[str, float]]


@dataclass
class DesignResult:
    """LHS design with per-targeted-key natural-log seed-mean targets and noise.

    ``Y[key]`` and ``alpha[key]`` are (n,) arrays over the shared design ``X``;
    NaN marks a point censored for that key (a species extinct at that point).
    Censoring is per-key: a point dropped for ``cod_ssb_mean`` still trains
    ``herring_biomass_mean``.
    """

    X: np.ndarray
    keys: list[str]
    Y: dict[str, np.ndarray]
    alpha: dict[str, np.ndarray]

    def valid(self, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rows where ``key`` is not censored: (X_valid, Y_valid, alpha_valid)."""
        mask = ~np.isnan(self.Y[key])
        return self.X[mask], self.Y[key][mask], self.alpha[key][mask]

    def n_censored(self, key: str) -> int:
        """Number of points censored (NaN) for ``key``."""
        return int(np.isnan(self.Y[key]).sum())


def run_design(
    evaluator: Evaluator,
    free_params: list[FreeParameter],
    target_keys: Sequence[str],
    n_points: int,
    n_seeds: int,
    *,
    seed: int = 0,
    seed_offset: int = 0,
    X: np.ndarray | None = None,
) -> DesignResult:
    """Run an LHS design through ``evaluator`` over ``n_seeds`` seeds; reduce per key.

    For each design point and targeted key, collect the linear stat over seeds;
    if any seed is missing or <= 0 the point is CENSORED for that key (mean-of-logs
    undefined). Otherwise ``Y = mean(log values)`` (natural log) and
    ``alpha = var(log values, ddof=1) / n_seeds`` (unbiased per-run variance of the
    seed mean — ddof=1 here is a different quantity from the emulator's ddof=0).

    Run seeds are ``seed_offset + i*n_seeds + k`` for point ``i``, seed ``k`` —
    deterministic, so a re-run reproduces X, Y, alpha. ``X`` may be supplied to
    re-run a fixed design (used by the growth loop's appended batches).
    """
    if n_seeds < 2:
        raise ValueError(f"n_seeds must be >= 2 to estimate seed variance, got {n_seeds}")
    if X is None:
        X = lhs_design(free_params, n_points, seed)
    n = len(X)
    keys = list(target_keys)
    Y = {k: np.full(n, np.nan) for k in keys}
    alpha = {k: np.full(n, np.nan) for k in keys}

    # Flat (task_id, point-input, run-seed) list over every (point, seed); the run
    # seed is deterministic per task, so results are order-independent and a batch
    # evaluator's parallel output is bit-identical to the serial loop.
    tasks = [
        (i * n_seeds + k, X[i], seed_offset + i * n_seeds + k)
        for i in range(n)
        for k in range(n_seeds)
    ]
    if hasattr(evaluator, "evaluate_batch"):
        results = evaluator.evaluate_batch(tasks)  # aligned to task order
    else:
        results = [evaluator(x, s) for _, x, s in tasks]

    for i in range(n):
        per_seed = results[i * n_seeds : (i + 1) * n_seeds]
        for key in keys:
            vals = [d.get(key) for d in per_seed]
            if any(v is None or v <= 0.0 for v in vals):
                continue  # censor: mean-of-logs undefined
            logs = np.log(np.asarray(vals, dtype=float))
            Y[key][i] = float(np.mean(logs))
            alpha[key][i] = float(np.var(logs, ddof=1)) / n_seeds

    return DesignResult(X=X, keys=keys, Y=Y, alpha=alpha)


_WORKER_EVALUATOR: Evaluator | None = None


def _worker_init(factory: Callable[[], Evaluator]) -> None:
    """ProcessPoolExecutor initializer: build the (serial) evaluator once per worker."""
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = factory()


def _worker_eval(task_id: int, x: np.ndarray, seed: int) -> tuple[int, dict[str, float]]:
    """Evaluate one (point, seed) task in a worker; return ``(task_id, stats)``."""
    assert _WORKER_EVALUATOR is not None  # set by _worker_init
    return task_id, _WORKER_EVALUATOR(x, int(seed))


class _ParallelEngineEvaluator:
    """Batch-capable evaluator over a persistent process pool.

    Each worker rebuilds the serial evaluator once from a picklable ``factory``
    (typically ``partial(make_engine_evaluator, ..., n_workers=1)``); only
    ``(task_id, x, seed)`` cross the process boundary. Results are scattered back
    by ``task_id``, so batch output is aligned to input order and bit-identical to
    the serial loop regardless of completion order. Also callable as
    ``(x, seed) -> dict`` (a one-task batch) so serial callers still work. The pool
    is lazy-started and reused across design-growth rounds; the caller owns its
    lifecycle via ``close()`` / the context-manager protocol.
    """

    def __init__(self, factory: Callable[[], Evaluator], n_workers: int) -> None:
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self._factory = factory
        self._n_workers = int(n_workers)
        self._pool: ProcessPoolExecutor | None = None

    def _ensure_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            # spawn (not fork): the parent is multi-threaded (numpy/BLAS) and the
            # engine uses numba — forking a locked threadpool risks a child deadlock.
            # Spawn's per-worker startup cost amortizes over the long design run.
            self._pool = ProcessPoolExecutor(
                max_workers=self._n_workers,
                mp_context=get_context("spawn"),
                initializer=_worker_init,
                initargs=(self._factory,),
            )
        return self._pool

    def evaluate_batch(
        self, tasks: Sequence[tuple[int, np.ndarray, int]]
    ) -> list[dict[str, float]]:
        """Run every task in the pool; return stats aligned to input task order."""
        pool = self._ensure_pool()
        futures = {pool.submit(_worker_eval, tid, x, s): tid for tid, x, s in tasks}
        by_id: dict[int, dict[str, float]] = {}
        for fut in as_completed(futures):
            tid, stats = fut.result()
            by_id[tid] = stats
        return [by_id[tid] for tid, _, _ in tasks]

    def __call__(self, x: np.ndarray, seed: int) -> dict[str, float]:
        return self.evaluate_batch([(0, x, int(seed))])[0]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> _ParallelEngineEvaluator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def make_engine_evaluator(
    free_params: list[FreeParameter],
    base_config_path: Path,
    species_names: Sequence[str],
    *,
    enable_ssb: bool = True,
    nyear: int | None = None,
    n_workers: int = 1,
) -> Evaluator:
    """Build the real Python-engine evaluator: point+seed -> per-species stat dict.

    Reads the base config once, then per call injects ``output.ssb.enabled='true'``
    (when ``enable_ssb``; the CSV flag — the netcdf flag does not make in-memory
    ``.ssb()`` readable), optionally overrides ``simulation.time.nyear``, applies
    ``point_to_overrides``, runs the engine, and reduces with ``compute_uq_stats``.
    ``PythonEngine``/``OsmoseConfigReader`` are lazy-imported to keep design.py light.

    ``n_workers > 1`` returns a batch-capable ``_ParallelEngineEvaluator`` that runs
    design points across a process pool (each worker rebuilds this serial evaluator
    once); ``n_workers == 1`` returns the serial closure unchanged.
    """
    if n_workers > 1:
        factory = partial(
            make_engine_evaluator,
            free_params,
            base_config_path,
            species_names,
            enable_ssb=enable_ssb,
            nyear=nyear,
            n_workers=1,
        )
        return _ParallelEngineEvaluator(factory, n_workers)

    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine

    base_cfg = OsmoseConfigReader().read(base_config_path)
    species = list(species_names)

    def evaluate(x: np.ndarray, seed: int) -> dict[str, float]:
        cfg = dict(base_cfg)
        if enable_ssb:
            cfg["output.ssb.enabled"] = "true"
        if nyear is not None:
            cfg["simulation.time.nyear"] = str(nyear)
        cfg.update(point_to_overrides(x, free_params))
        results = PythonEngine().run_in_memory(cfg, seed=int(seed))
        return compute_uq_stats(results, species)

    return evaluate


@dataclass
class GrowthResult:
    """Outcome of the bounded design-growth loop."""

    design: DesignResult
    reports: dict[str, GateReport]
    status: str  # "calibrated" | "aborted_n_max"
    rounds: int


def _merge_designs(a: DesignResult, b: DesignResult) -> DesignResult:
    """Concatenate two designs over the same targeted keys (a's keys, in order)."""
    X = np.vstack([a.X, b.X])
    Y = {k: np.concatenate([a.Y[k], b.Y[k]]) for k in a.keys}
    alpha = {k: np.concatenate([a.alpha[k], b.alpha[k]]) for k in a.keys}
    return DesignResult(X=X, keys=list(a.keys), Y=Y, alpha=alpha)


def grow_until_calibrated(
    evaluator: Evaluator,
    free_params: list[FreeParameter],
    target_keys: Sequence[str],
    n_seeds: int,
    *,
    n0: int,
    increment: int,
    n_max: int,
    seed: int = 0,
    gate_fn: Callable[..., GateReport] | None = None,
) -> GrowthResult:
    """Build N0, gate every key, and grow by ``increment`` until all keys pass or
    ``n_max`` aborts.

    ``gate_fn(X, Y, alpha, key=...)`` defaults to the real
    ``evaluate_emulator_calibration`` and is injectable for deterministic tests.
    Each appended batch uses a distinct deterministic seed offset so re-runs
    reproduce. ``n_max`` is a HARD safety ceiling, not a target — the loop never
    grows past it and aborts with the last reports.
    """
    if n0 <= 0 or increment <= 0:
        raise ValueError(f"n0 and increment must be positive, got n0={n0}, increment={increment}")
    if n0 > n_max:
        raise ValueError(f"n0 ({n0}) must not exceed n_max ({n_max})")
    keys = list(target_keys)
    if not keys:
        raise ValueError("target_keys must be non-empty")

    gate = gate_fn if gate_fn is not None else evaluate_emulator_calibration

    def _gate_all(result: DesignResult) -> dict[str, GateReport]:
        reports = {}
        for key in keys:
            Xv, Yv, av = result.valid(key)
            reports[key] = gate(Xv, Yv, av, key=key)
        return reports

    result = run_design(evaluator, free_params, keys, n0, n_seeds, seed=seed, seed_offset=0)
    rounds = 0
    while True:
        reports = _gate_all(result)
        if all(r.passed for r in reports.values()):
            return GrowthResult(design=result, reports=reports, status="calibrated", rounds=rounds)
        if len(result.X) + increment > n_max:
            return GrowthResult(
                design=result, reports=reports, status="aborted_n_max", rounds=rounds
            )
        rounds += 1
        batch = run_design(
            evaluator,
            free_params,
            keys,
            increment,
            n_seeds,
            seed=seed + rounds,
            seed_offset=rounds * 1_000_000,
        )
        result = _merge_designs(result, batch)
