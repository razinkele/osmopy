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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats.qmc import LatinHypercube

from osmose.calibration.problem import FreeParameter, Transform
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

    for i in range(n):
        per_seed = [evaluator(X[i], seed_offset + i * n_seeds + k) for k in range(n_seeds)]
        for key in keys:
            vals = [d.get(key) for d in per_seed]
            if any(v is None or v <= 0.0 for v in vals):
                continue  # censor: mean-of-logs undefined
            logs = np.log(np.asarray(vals, dtype=float))
            Y[key][i] = float(np.mean(logs))
            alpha[key][i] = float(np.var(logs, ddof=1)) / n_seeds

    return DesignResult(X=X, keys=keys, Y=Y, alpha=alpha)


def make_engine_evaluator(
    free_params: list[FreeParameter],
    base_config_path: Path,
    species_names: Sequence[str],
    *,
    enable_ssb: bool = True,
    nyear: int | None = None,
) -> Evaluator:
    """Build the real Python-engine evaluator: point+seed -> per-species stat dict.

    Reads the base config once, then per call injects ``output.ssb.enabled='true'``
    (when ``enable_ssb``; the CSV flag — the netcdf flag does not make in-memory
    ``.ssb()`` readable), optionally overrides ``simulation.time.nyear``, applies
    ``point_to_overrides``, runs the engine, and reduces with ``compute_uq_stats``.
    ``PythonEngine``/``OsmoseConfigReader`` are lazy-imported to keep design.py light.
    """
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
