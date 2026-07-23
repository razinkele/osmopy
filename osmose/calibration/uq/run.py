"""End-to-end surrogate-Bayesian UQ orchestrator.

run_surrogate_bayes composes the layer: grow_until_calibrated (Phase 1) ->
fit_emulators + make_log_posterior (Phase 2a) -> DynestySampler (Phase 2b).
It takes the same injectable evaluator the layer uses (default: the real
make_engine_evaluator), so the pipeline is testable synthetically; real engine
runs are production usage. Cheap-and-fatal misconfiguration (over-dimension,
malformed targets) raises BEFORE the expensive design; the gate-failed case
short-circuits without sampling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from osmose.calibration.uq.design import DesignResult
from osmose.calibration.uq.gate import GateReport
from osmose.calibration.uq.sampler import SamplerResult


@dataclass
class UQResult:
    """Bundle of a surrogate-Bayesian UQ run.

    ``status`` is one of ``"ok"`` (calibrated + converged), ``"gate_failed"``
    (design never calibrated -> no posterior), ``"sampled_not_converged"``
    (sampled but the convergence flag is False). ``sampler_result`` /
    ``posterior_mean`` are ``None`` when the gate failed.
    """

    status: str
    gate_reports: dict[str, GateReport]
    design: DesignResult
    n_censored: dict[str, int]
    sampler_result: SamplerResult | None = None
    posterior_mean: np.ndarray | None = None


def derive_sigma_seed_sq(
    design: DesignResult,
    target_keys: Sequence[str],
    n_seeds: int,
) -> dict[str, float]:
    """Per-key pooled seed variance ``s²`` = ``n_seeds * mean(alpha[key]_valid)``.

    ``alpha = s²/n_seeds`` (variance of the seed mean), so this recovers the mean
    per-run variance. Assumes ``n_seeds`` is constant across the design (a
    per-point-S future would break this); pooled-per-key is the Phase 2
    simplification. Censored (NaN) points are excluded via ``design.valid``.
    """
    out: dict[str, float] = {}
    for key in target_keys:
        _, _, alpha = design.valid(key)
        out[key] = n_seeds * float(np.mean(alpha)) if len(alpha) else 0.0
    return out
