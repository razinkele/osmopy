"""Posterior-predictive diagnostic (Goal 2): per-species marginal emulator-predictive
ranges from the posterior over theta.

DIAGNOSTIC, not calibrated against reality. The joint draws are computed internally
and DISCARDED — only per-species marginal ranges + a theta-mediated cross-species
correlation are returned, so totals/ratios/P(total>X) (invalid for a conditionally-
independent emulator) cannot be derived from the result. y is a single-run
log-biomass with mean mu_emu (NO Jensen shift): the predictive median biomass is
the GEOMETRIC mean, below the arithmetic target by exp(-0.5*sigma_seed_sq).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm  # type: ignore[import-untyped]

from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.keying import target_to_output_key
from osmose.calibration.uq.sampler import SamplerResult


@dataclass
class EmulatorPredictiveRanges:
    """Per-species MARGINAL emulator-predictive ranges (a labeled diagnostic).

    Emulator-coverage-measured, NOT calibrated against reality; marginal-only (no
    joint draws are exposed). ``log_ranges``/``biomass_ranges`` are ``(lo, median,
    hi)`` at ``level``; ``median`` is the geometric mean (below the arithmetic
    target by the Jensen factor). ``cross_species_correlation`` is theta-mediated
    only — NOT a trophic posterior-predictive check.
    """

    keys: list[str]
    log_ranges: dict[str, tuple[float, float, float]]
    biomass_ranges: dict[str, tuple[float, float, float]]
    cross_species_correlation: np.ndarray
    level: float


def posterior_predictive(
    sampler_result: SamplerResult,
    emulators: Mapping[str, object],
    target_keys: Sequence[str],
    sigma_seed_sq_by_key: Mapping[str, float],
    *,
    n_draws: int = 4000,
    seed: int = 0,
    level: float = 0.9,
) -> EmulatorPredictiveRanges:
    """Genuine per-theta mixture -> per-species marginal predictive ranges.

    Resamples theta BY IMPORTANCE WEIGHT (dynesty samples are weighted dead points,
    not equal-weight draws), predicts each emulator, adds single-run seed noise
    ``N(0, emulator_var(theta) + sigma_seed_sq[key])`` (no Jensen shift), and reduces
    to marginal quantiles. The joint draw array is internal and discarded.
    """
    rng = np.random.default_rng(seed)
    weights = np.asarray(sampler_result.weights, dtype=float)
    p = weights / weights.sum()
    n = len(sampler_result.samples)
    idx = rng.choice(n, size=n_draws, p=p)  # resample by weight -- the load-bearing step
    thetas = np.asarray(sampler_result.samples)[idx]

    keys = list(target_keys)
    lo_q, hi_q = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    joint_log = np.empty((n_draws, len(keys)))  # internal; discarded (marginal-only guard)
    log_ranges: dict[str, tuple[float, float, float]] = {}
    biomass_ranges: dict[str, tuple[float, float, float]] = {}

    for j, key in enumerate(keys):
        mean, var = emulators[key].predict(thetas)
        sd = np.sqrt(np.asarray(var) + sigma_seed_sq_by_key[key])
        y = np.asarray(mean) + sd * rng.standard_normal(
            n_draws
        )  # single-run log-biomass, no Jensen
        joint_log[:, j] = y
        q = np.quantile(y, [lo_q, 0.5, hi_q])
        log_ranges[key] = (float(q[0]), float(q[1]), float(q[2]))
        biomass_ranges[key] = (float(np.exp(q[0])), float(np.exp(q[1])), float(np.exp(q[2])))

    corr = np.atleast_2d(np.corrcoef(joint_log, rowvar=False))
    return EmulatorPredictiveRanges(
        keys=keys,
        log_ranges=log_ranges,
        biomass_ranges=biomass_ranges,
        cross_species_correlation=corr,
        level=level,
    )


def marginal_coverage(
    ranges: EmulatorPredictiveRanges,
    targets: Sequence[BiomassTarget],
) -> dict[str, bool]:
    """Per-species marginal coverage: does each target fall within its predictive
    biomass ``[lo, hi]``? The one honest, cheap posterior-predictive check available
    now — a genuine trophic (joint) PPC needs per-seed joint design outputs, which
    Phase 1 discarded.
    """
    out: dict[str, bool] = {}
    for target in targets:
        key = target_to_output_key(target)
        lo, _median, hi = ranges.biomass_ranges[key]
        out[key] = lo <= target.target <= hi
    return out


def emulator_holdout_coverage(
    emulators: Mapping[str, object],
    holdout_X: np.ndarray,
    holdout_Y: Mapping[str, np.ndarray],
    holdout_alpha: Mapping[str, np.ndarray],
    *,
    level: float = 0.95,
) -> dict[str, float]:
    """Per-key fraction of held-out engine points inside the emulator's predictive interval.

    The load-bearing method-validation metric: run fresh (out-of-design) points
    through the engine, then check whether each key's log seed-mean ``holdout_Y``
    falls within ``mu +/- z*sqrt(var + alpha)`` (z at ``level``; ``var`` the GP
    latent variance, ``alpha`` the held-out seed-mean noise s²/S) — the SAME
    standardization the calibration gate cross-validates, but on genuinely
    out-of-design points. Coverage ~= ``level`` for a calibrated emulator.
    NaN held-out entries (censored points) are excluded per key.
    """
    z = float(norm.ppf(0.5 + level / 2.0))
    out: dict[str, float] = {}
    X = np.atleast_2d(np.asarray(holdout_X, dtype=float))
    for key, emu in emulators.items():
        y = np.asarray(holdout_Y[key], dtype=float).ravel()
        a = np.asarray(holdout_alpha[key], dtype=float).ravel()
        valid = ~np.isnan(y)
        if not valid.any():
            out[key] = float("nan")
            continue
        mean, var = emu.predict(X[valid])  # type: ignore[attr-defined]
        sd = np.sqrt(np.asarray(var, dtype=float) + a[valid])
        inside = np.abs(y[valid] - np.asarray(mean, dtype=float)) <= z * sd
        out[key] = float(np.mean(inside))
    return out
