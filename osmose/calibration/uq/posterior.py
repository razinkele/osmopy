"""Compose prior + per-target likelihood into log_posterior(theta).

The posterior takes INJECTED emulators (duck-typed predict(X)->(mean,var)), so
synthetic tests inject an analytic emulator and recovery is exact. fit_emulators
is the production path that builds real GPs from a DesignResult. Cross-target
independence (the log-likelihoods are summed) is a documented overconfidence
source: trophically-coupled species are treated as conditionally independent.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from osmose.calibration.problem import FreeParameter
from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.design import DesignResult
from osmose.calibration.uq.emulator import GPEmulator
from osmose.calibration.uq.keying import target_to_output_key
from osmose.calibration.uq.likelihood import band_faithful, gaussian_log_biomass
from osmose.calibration.uq.prior import log_prior

DEFAULT_K_BY_TYPE = {"biomass": 1.0, "ssb": 1.0, "catch": 1.0}
_LIKELIHOODS = {"gaussian": gaussian_log_biomass, "band": band_faithful}


def fit_emulators(design: DesignResult, min_points: int = 2) -> dict[str, GPEmulator]:
    """Fit one GP per key with at least ``min_points`` valid (uncensored) points.

    Keys with too few valid points are omitted (a GP needs >=2). Fits on the
    per-key valid slice — the natural-log seed-mean targets and their noise.
    """
    emulators: dict[str, GPEmulator] = {}
    for key in design.keys:
        X, Y, alpha = design.valid(key)
        if len(X) >= min_points:
            emulators[key] = GPEmulator().fit(X, Y, alpha)
    return emulators


def make_log_posterior(
    emulators: Mapping[str, object],
    targets: Sequence[BiomassTarget],
    free_params: list[FreeParameter],
    *,
    sigma_seed_sq_by_key: Mapping[str, float],
    sigma_disc_sq: float = 0.0,
    k_by_type: Mapping[str, float] | None = None,
    likelihood: str = "gaussian",
) -> Callable[[np.ndarray], float]:
    """Return ``log_post(theta) -> float`` = log_prior + sum of per-target log-likelihoods.

    Emulators are injected (duck-typed ``predict(X)->(mean,var)``). Every target's
    key is validated against ``emulators`` at construction (raises ``KeyError``).
    ``sigma_seed_sq`` and ``k`` are resolved per target once, up front.
    """
    k_by_type = k_by_type if k_by_type is not None else DEFAULT_K_BY_TYPE
    like_fn = _LIKELIHOODS[likelihood]

    resolved = []
    for t in targets:
        if not (t.lower < t.upper):
            raise ValueError(
                f"target for species {t.species!r} has lower ({t.lower}) >= upper "
                f"({t.upper}); a band requires lower < upper"
            )
        key = target_to_output_key(t)
        if key not in emulators:
            raise KeyError(f"no emulator for target key {key!r}")
        resolved.append((t, key, sigma_seed_sq_by_key[key], k_by_type[t.reference_point_type]))

    def log_post(theta: np.ndarray) -> float:
        lp = log_prior(theta, free_params)
        if not math.isfinite(lp):
            return lp
        theta_2d = np.atleast_2d(theta)
        for t, key, seed_sq, k in resolved:
            mean, var = emulators[key].predict(theta_2d)
            lp += like_fn(
                float(mean[0]),
                float(var[0]),
                t.target,
                t.lower,
                t.upper,
                sigma_seed_sq=seed_sq,
                sigma_disc_sq=sigma_disc_sq,
                k=k,
            )
        return lp

    return log_post
