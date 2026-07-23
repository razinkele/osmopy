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

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from osmose.calibration.problem import FreeParameter
from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.design import DesignResult, Evaluator, grow_until_calibrated
from osmose.calibration.uq.gate import GateReport
from osmose.calibration.uq.keying import target_to_output_key
from osmose.calibration.uq.posterior import fit_emulators, make_log_posterior
from osmose.calibration.uq.predictive import EmulatorPredictiveRanges, posterior_predictive
from osmose.calibration.uq.sampler import DynestySampler, SamplerResult, check_dimension


@dataclass
class UQResult:
    """Bundle of a surrogate-Bayesian UQ run.

    ``status`` is one of ``"ok"`` (calibrated + converged), ``"gate_failed"``
    (design never calibrated -> no posterior), ``"sampled_not_converged"``
    (sampled but the convergence flag is False). ``sampler_result`` /
    ``posterior_mean`` are ``None`` when the gate failed. ``predictive_ranges``
    is ``None`` on the gate-failed path and when ``include_predictive=False``.
    """

    status: str
    gate_reports: dict[str, GateReport]
    design: DesignResult
    n_censored: dict[str, int]
    sampler_result: SamplerResult | None = None
    posterior_mean: np.ndarray | None = None
    predictive_ranges: EmulatorPredictiveRanges | None = None


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


def run_surrogate_bayes(
    evaluator: Evaluator,
    free_params: list[FreeParameter],
    targets: Sequence[BiomassTarget],
    n_seeds: int,
    *,
    n0: int,
    increment: int,
    n_max: int,
    seed: int = 0,
    gate_fn: Callable[..., GateReport] | None = None,
    likelihood: str = "gaussian",
    sigma_disc_sq: float = 0.0,
    k_by_type: dict[str, float] | None = None,
    sampler: DynestySampler | None = None,
    include_predictive: bool = True,
) -> UQResult:
    """Run the full surrogate-Bayesian UQ pipeline; return a UQResult.

    Fails loud up front on caller misconfiguration (over-dimension, malformed
    target) BEFORE the expensive design. Grows a calibrated design (real gate by
    default; ``gate_fn`` injectable); short-circuits to ``"gate_failed"`` if it
    never calibrates; else fits emulators, derives per-key σ_seed², builds the
    posterior, samples it, and (by default) computes per-key predictive ranges.
    """
    # Fail-fast validation, BEFORE grow (potentially thousands of engine runs).
    check_dimension(len(free_params))
    for t in targets:
        if not (t.lower < t.upper):
            raise ValueError(
                f"target for species {t.species!r} has lower ({t.lower}) >= "
                f"upper ({t.upper}); a band requires lower < upper"
            )
    keys = [target_to_output_key(t) for t in targets]  # raises on unknown reference_point_type

    growth = grow_until_calibrated(
        evaluator,
        free_params,
        keys,
        n_seeds,
        n0=n0,
        increment=increment,
        n_max=n_max,
        seed=seed,
        gate_fn=gate_fn,
    )
    n_censored = {k: growth.design.n_censored(k) for k in keys}
    if growth.status != "calibrated":
        return UQResult(
            status="gate_failed",
            gate_reports=growth.reports,
            design=growth.design,
            n_censored=n_censored,
        )

    emulators = fit_emulators(growth.design)
    sigma_seed_sq = derive_sigma_seed_sq(growth.design, keys, n_seeds)
    log_posterior = make_log_posterior(
        emulators,
        targets,
        free_params,
        sigma_seed_sq_by_key=sigma_seed_sq,
        sigma_disc_sq=sigma_disc_sq,
        k_by_type=k_by_type,
        likelihood=likelihood,
    )
    sampler = sampler if sampler is not None else DynestySampler()
    sampler_result = sampler.sample(log_posterior, free_params, seed=seed)
    status = "ok" if sampler_result.converged else "sampled_not_converged"
    predictive_ranges = None
    if include_predictive:
        predictive_ranges = posterior_predictive(
            sampler_result, emulators, keys, sigma_seed_sq, seed=seed
        )
    return UQResult(
        status=status,
        gate_reports=growth.reports,
        design=growth.design,
        n_censored=n_censored,
        sampler_result=sampler_result,
        posterior_mean=sampler_result.posterior_mean(),
        predictive_ranges=predictive_ranges,
    )
