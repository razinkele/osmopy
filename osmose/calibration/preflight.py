"""Pre-flight sensitivity analysis for OSMOSE calibration.

Validates calibration parameters before optimization runs using Morris screening
and optional Sobol analysis to detect negligible parameters, blow-ups, flat
objectives, and tight bounds.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
from SALib.sample import morris as morris_sample  # type: ignore[import-untyped]
from SALib.analyze import morris as morris_analyze  # type: ignore[import-untyped]

from osmose.engine import PythonEngine
from osmose.results import OsmoseResults
from osmose.calibration.problem import FreeParameter, Transform

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class IssueCategory(Enum):
    """Category of a pre-flight issue."""

    NEGLIGIBLE = "negligible"
    BLOWUP = "blowup"
    FLAT_OBJECTIVE = "flat_objective"
    BOUND_TIGHT = "bound_tight"
    ALL_NEGLIGIBLE = "all_negligible"


class IssueSeverity(Enum):
    """Severity level of a pre-flight issue."""

    WARNING = "warning"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ParameterScreening:
    """Morris screening result for a single parameter.

    Attributes:
        key: Parameter key (e.g. ``species.linf.sp0``).
        mu_star: Modified mean of elementary effects (|EE|).  Must be >= 0.
        sigma: Standard deviation of elementary effects.
        mu_star_conf: 95 % bootstrap confidence interval for mu_star.
        influential: True when parameter exceeds the negligible threshold.
    """

    key: str
    mu_star: float
    sigma: float
    mu_star_conf: float
    influential: bool

    def __post_init__(self) -> None:
        if self.mu_star < 0:
            raise ValueError(f"mu_star must be >= 0, got {self.mu_star!r} for key {self.key!r}")


@dataclass
class PreflightIssue:
    """A single issue detected by the pre-flight analysis.

    Attributes:
        category: The category of the issue (see :class:`IssueCategory`).
        severity: WARNING or ERROR.
        param_key: The affected parameter key, or None for global issues.
        message: Human-readable description.
        suggestion: Actionable remediation hint.
        auto_fixable: Whether the issue can be fixed automatically (e.g. parameter removal).
    """

    category: IssueCategory
    severity: IssueSeverity
    param_key: str | None
    message: str
    suggestion: str
    auto_fixable: bool


@dataclass
class PreflightResult:
    """Aggregated result of a pre-flight analysis run.

    Attributes:
        screening: Per-parameter Morris screening results.
        sobol: Optional Sobol indices dict (from :class:`~osmose.calibration.sensitivity.SensitivityAnalyzer`).
        issues: Detected issues sorted by severity (errors first).
        survivors: Parameter keys that passed all filters.
        elapsed_seconds: Wall-clock time for the analysis.
    """

    screening: list[ParameterScreening]
    sobol: dict | None
    issues: list[PreflightIssue]
    survivors: list[str]
    elapsed_seconds: float


class PreflightEvalError(RuntimeError):
    """Raised when a preflight run fails so many samples that results are
    unusable. The caller should review the evaluation_fn rather than trust
    degenerate sensitivity indices.
    """


# ---------------------------------------------------------------------------
# Morris Screening
# ---------------------------------------------------------------------------


def run_morris_screening(
    param_names: list[str],
    param_bounds: list[tuple[float, float]],
    eval_fn: object,
    *,
    n_trajectories: int = 10,
    num_levels: int = 4,
    negligible_threshold: float = 0.1,
    seed: int | None = None,
) -> list[ParameterScreening]:
    """Run Morris elementary-effects screening.

    Parameters
    ----------
    param_names:
        Names of the free parameters.
    param_bounds:
        ``(lower, upper)`` bounds for each parameter.
    eval_fn:
        Callable ``(X: np.ndarray) -> np.ndarray`` that accepts an
        ``(N, k)`` sample matrix and returns either a 1-D array of
        shape ``(N,)`` (single objective) or a 2-D array of shape
        ``(N, n_obj)`` (multiple objectives).
    n_trajectories:
        Number of Morris trajectories (``r`` in SALib).
    num_levels:
        Number of grid levels for the Morris design.
    negligible_threshold:
        A parameter is deemed *negligible* when
        ``(mu_star + mu_star_conf) < negligible_threshold * max(mu_star)``.
    seed:
        Random seed forwarded to :func:`SALib.sample.morris.sample`.

    Returns
    -------
    list[ParameterScreening]
        One entry per parameter, ordered to match *param_names*.
    """
    problem = {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": list(param_bounds),
    }

    sample_kwargs: dict = {"num_levels": num_levels}
    if seed is not None:
        sample_kwargs["seed"] = seed

    X = morris_sample.sample(problem, n_trajectories, **sample_kwargs)
    Y_raw = np.asarray(eval_fn(X))  # type: ignore[operator]

    if Y_raw.ndim == 1:
        Y_raw = Y_raw[:, np.newaxis]

    n_obj = Y_raw.shape[1]
    k = len(param_names)

    agg_mu_star = np.zeros(k)
    agg_conf = np.zeros(k)
    agg_sigma = np.zeros(k)

    for obj_idx in range(n_obj):
        Y_col = Y_raw[:, obj_idx]
        result = morris_analyze.analyze(
            problem, X, Y_col, num_levels=num_levels, print_to_console=False
        )
        mu_star_obj = np.asarray(result["mu_star"])
        conf_obj = np.asarray(result["mu_star_conf"])
        sigma_obj = np.asarray(result["sigma"])
        # Aggregate across objectives via max (worst-case sensitivity)
        agg_mu_star = np.maximum(agg_mu_star, mu_star_obj)
        agg_conf = np.maximum(agg_conf, conf_obj)
        agg_sigma = np.maximum(agg_sigma, sigma_obj)

    max_mu_star = float(np.max(agg_mu_star)) if np.max(agg_mu_star) > 0 else 1.0

    results: list[ParameterScreening] = []
    for j, name in enumerate(param_names):
        influential = (agg_mu_star[j] + agg_conf[j]) >= negligible_threshold * max_mu_star
        results.append(
            ParameterScreening(
                key=name,
                mu_star=float(agg_mu_star[j]),
                sigma=float(agg_sigma[j]),
                mu_star_conf=float(agg_conf[j]),
                influential=influential,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Issue Detection
# ---------------------------------------------------------------------------


def _issues_blowup(blowup_params: list[str] | None) -> list[PreflightIssue]:
    """Emit ERROR per parameter that caused a numerical blow-up during Morris."""
    return [
        PreflightIssue(
            category=IssueCategory.BLOWUP,
            severity=IssueSeverity.ERROR,
            param_key=key,
            message=f"Parameter '{key}' caused numerical blow-up during Morris sampling.",
            suggestion="Tighten the parameter bounds or check the model for instability.",
            auto_fixable=False,
        )
        for key in blowup_params or []
    ]


def _issues_negligible(screening: list[ParameterScreening]) -> list[PreflightIssue]:
    """Emit WARNING per parameter that Morris flagged as non-influential."""
    return [
        PreflightIssue(
            category=IssueCategory.NEGLIGIBLE,
            severity=IssueSeverity.WARNING,
            param_key=ps.key,
            message=f"Parameter '{ps.key}' has negligible influence on the objective(s).",
            suggestion="Consider removing from the free-parameter set to reduce cost.",
            auto_fixable=True,
        )
        for ps in screening
        if not ps.influential
    ]


def _issues_all_negligible(screening: list[ParameterScreening]) -> list[PreflightIssue]:
    """Emit a single ERROR when every screened parameter is negligible."""
    if not screening or any(ps.influential for ps in screening):
        return []
    return [
        PreflightIssue(
            category=IssueCategory.ALL_NEGLIGIBLE,
            severity=IssueSeverity.ERROR,
            param_key=None,
            message="All parameters are negligible — the objective is insensitive to every free parameter.",
            suggestion="Review the objective function, parameter bounds, and model configuration.",
            auto_fixable=False,
        )
    ]


def _issues_flat(sobol_result: dict | None) -> list[PreflightIssue]:
    """Emit ERROR per objective whose Sobol ``sum(max(0, S1))`` is below 0.05."""
    if sobol_result is None or "S1" not in sobol_result:
        return []
    S1 = np.asarray(sobol_result["S1"])
    out: list[PreflightIssue] = []
    if S1.ndim == 1:
        total_variance = float(np.sum(np.maximum(0.0, S1)))
        if total_variance < 0.05:
            out.append(
                PreflightIssue(
                    category=IssueCategory.FLAT_OBJECTIVE,
                    severity=IssueSeverity.ERROR,
                    param_key=None,
                    message=f"Objective is effectively flat (sum(S1)={total_variance:.4f} < 0.05).",
                    suggestion="Check for output clipping, constant targets, or degenerate runs.",
                    auto_fixable=False,
                )
            )
        return out
    # Multi-objective: SensitivityAnalyzer convention is shape (n_obj, n_params).
    for obj_idx in range(S1.shape[0]):
        row = S1[obj_idx]
        total_variance = float(np.sum(np.maximum(0.0, row)))
        if total_variance < 0.05:
            obj_names = sobol_result.get("objective_names", None)
            obj_label = obj_names[obj_idx] if obj_names else f"obj_{obj_idx}"
            out.append(
                PreflightIssue(
                    category=IssueCategory.FLAT_OBJECTIVE,
                    severity=IssueSeverity.ERROR,
                    param_key=None,
                    message=(
                        f"Objective '{obj_label}' is effectively flat "
                        f"(sum(S1)={total_variance:.4f} < 0.05)."
                    ),
                    suggestion="Check for output clipping, constant targets, or degenerate runs.",
                    auto_fixable=False,
                )
            )
    return out


def _issues_bound_tight(
    screening: list[ParameterScreening],
    sobol_result: dict | None,
) -> list[PreflightIssue]:
    """Emit WARNING when Sobol total-order ST > 0.3 AND Morris sigma/mu* > 1.5.

    Jointly these two signals indicate a parameter whose influence is strong
    but highly non-monotone — typically a symptom of bounds being too wide.
    """
    if sobol_result is None or "ST" not in sobol_result:
        return []
    ST = np.asarray(sobol_result["ST"])
    ST_agg = np.max(ST, axis=0) if ST.ndim > 1 else ST
    param_names_sobol: list[str] = sobol_result.get("param_names", [])
    screening_map = {ps.key: ps for ps in screening}
    out: list[PreflightIssue] = []
    for j, key in enumerate(param_names_sobol):
        if j >= len(ST_agg):
            continue
        st_val = float(ST_agg[j])
        ps = screening_map.get(key)
        if ps is None:
            continue
        ratio = ps.sigma / ps.mu_star if ps.mu_star > 0 else 0.0
        if st_val > 0.3 and ratio > 1.5:
            out.append(
                PreflightIssue(
                    category=IssueCategory.BOUND_TIGHT,
                    severity=IssueSeverity.WARNING,
                    param_key=key,
                    message=(
                        f"Parameter '{key}' has high total-order sensitivity (ST={st_val:.3f}) "
                        f"and high nonlinearity (sigma/mu*={ratio:.2f}). "
                        "Bounds may be too wide or the response is non-monotone."
                    ),
                    suggestion="Consider tightening parameter bounds or applying a log transform.",
                    auto_fixable=False,
                )
            )
    return out


def detect_issues(
    screening: list[ParameterScreening],
    sobol_result: dict | None = None,
    blowup_params: list[str] | None = None,
) -> list[PreflightIssue]:
    """Classify preflight screening + Sobol indices into actionable issues.

    Delegates to per-category detectors (``_issues_blowup``,
    ``_issues_negligible``, ``_issues_all_negligible``, ``_issues_flat``,
    ``_issues_bound_tight``) and preserves the legacy errors-first ordering.

    Parameters
    ----------
    screening:
        Per-parameter Morris screening results.
    sobol_result:
        Optional dict returned by
        :meth:`~osmose.calibration.sensitivity.SensitivityAnalyzer.analyze`.
        Supports 1-D (single-objective) and 2-D (``(n_obj, n_params)``) shapes.
    blowup_params:
        Parameter keys that caused numerical blow-ups during sampling.

    Returns
    -------
    list[PreflightIssue]
        Errors first, then warnings.
    """
    issues: list[PreflightIssue] = []
    issues.extend(_issues_blowup(blowup_params))
    issues.extend(_issues_negligible(screening))
    issues.extend(_issues_all_negligible(screening))
    issues.extend(_issues_flat(sobol_result))
    issues.extend(_issues_bound_tight(screening, sobol_result))
    issues.sort(key=lambda i: 0 if i.severity is IssueSeverity.ERROR else 1)
    return issues


# ---------------------------------------------------------------------------
# Two-stage Orchestrator
# ---------------------------------------------------------------------------


def _run_morris_stage(
    *,
    param_names: list[str],
    param_bounds: list[tuple[float, float]],
    evaluation_fn: Callable[[np.ndarray], np.ndarray],
    n_trajectories: int,
    num_levels: int,
    negligible_threshold: float,
    seed: int | None,
    cancel_event: threading.Event | None = None,
) -> tuple[list[ParameterScreening], float, np.ndarray, set[str]]:
    """Morris sampling + per-objective analysis (max-aggregated across objectives).

    Returns ``(screening, failure_rate, Y_clean, blowup_params_set)``. Raises
    :class:`PreflightEvalError` when >50 % of Morris samples fail — sensitivity
    indices at that point would be uninformative noise.

    If ``cancel_event`` is set mid-stage the function returns an empty result
    ``([], 0.0, np.zeros((0, 1)), set())`` and the caller is responsible for
    short-circuiting.
    """
    k = len(param_names)
    problem = {"num_vars": k, "names": param_names, "bounds": list(param_bounds)}

    sample_kwargs: dict = {"num_levels": num_levels}
    if seed is not None:
        sample_kwargs["seed"] = seed
    X = morris_sample.sample(problem, n_trajectories, **sample_kwargs)

    if cancel_event is not None and cancel_event.is_set():
        return [], 0.0, np.zeros((0, 1)), set()

    Y_raw = np.asarray(evaluation_fn(X))
    if Y_raw.ndim == 1:
        Y_raw = Y_raw[:, np.newaxis]

    n_samples = X.shape[0]
    blowup_sample_flags = np.any(~np.isfinite(Y_raw), axis=1)

    traj_size = k + 1
    blowup_params_set: set[str] = set()
    for traj_idx in range(n_trajectories):
        base = traj_idx * traj_size
        for step in range(1, traj_size):
            row = base + step
            if row >= n_samples:
                break
            if blowup_sample_flags[row]:
                diff = np.abs(X[row] - X[row - 1])
                if diff.max() > 0:
                    blowup_params_set.add(param_names[int(np.argmax(diff))])

    n_failed = int(np.sum(blowup_sample_flags))
    failure_rate = n_failed / max(n_samples, 1)

    _MAJORITY_FAILURE = 0.5
    if failure_rate > _MAJORITY_FAILURE:
        raise PreflightEvalError(
            f"Morris stage failure rate {failure_rate:.0%} exceeds "
            f"{_MAJORITY_FAILURE:.0%}; check evaluation_fn — sensitivity "
            "indices would be meaningless."
        )

    Y_clean = np.where(np.isfinite(Y_raw), Y_raw, 1e6)
    n_obj = Y_clean.shape[1]
    agg_mu_star = np.zeros(k)
    agg_conf = np.zeros(k)
    agg_sigma = np.zeros(k)

    for obj_idx in range(n_obj):
        result = morris_analyze.analyze(
            problem, X, Y_clean[:, obj_idx], num_levels=num_levels,
            print_to_console=False,
        )
        agg_mu_star = np.maximum(agg_mu_star, np.asarray(result["mu_star"]))
        agg_conf = np.maximum(agg_conf, np.asarray(result["mu_star_conf"]))
        agg_sigma = np.maximum(agg_sigma, np.asarray(result["sigma"]))

    max_mu_star = float(np.max(agg_mu_star)) if np.max(agg_mu_star) > 0 else 1.0
    screening: list[ParameterScreening] = []
    for j, name in enumerate(param_names):
        influential = (agg_mu_star[j] + agg_conf[j]) >= negligible_threshold * max_mu_star
        screening.append(
            ParameterScreening(
                key=name,
                mu_star=float(agg_mu_star[j]),
                sigma=float(agg_sigma[j]),
                mu_star_conf=float(agg_conf[j]),
                influential=influential,
            )
        )
    return screening, failure_rate, Y_clean, blowup_params_set


def _maybe_run_sobol_stage(
    *,
    screening: list[ParameterScreening],
    param_names: list[str],
    param_bounds: list[tuple[float, float]],
    evaluation_fn: Callable[[np.ndarray], np.ndarray],
    sobol_n_base: int,
    sobol_failure_threshold: float,
    seed: int | None,
    cancel_event: threading.Event | None = None,
) -> tuple[dict | None, list[str]]:
    """Refine with Sobol on the non-negligible survivors.

    Non-surviving parameters are pinned to their midpoint during sampling so
    the evaluation_fn still receives full-length vectors. With fewer than two
    survivors the stage is skipped (``(None, [])``) — a 1-dim Sobol is
    degenerate and would report ``ST == 1.0`` by construction.

    Returns ``(sobol_result_or_None, additional_blowup_keys)``. Survivors are
    added to ``additional_blowup_keys`` when the Sobol failure rate exceeds
    the threshold.
    """
    from osmose.calibration.sensitivity import SensitivityAnalyzer

    survivor_idx = [i for i, s in enumerate(screening) if s.influential]
    if len(survivor_idx) < 2:
        return None, []

    survivors = [screening[i].key for i in survivor_idx]
    survivor_bounds = [param_bounds[param_names.index(sv)] for sv in survivors]
    midpoints = np.array([(lo + hi) / 2.0 for lo, hi in param_bounds])

    analyzer = SensitivityAnalyzer(param_names=survivors, param_bounds=survivor_bounds)

    from SALib.sample import sobol as sobol_sample_mod  # type: ignore[import-untyped]
    sobol_kwargs: dict = {}
    if seed is not None:
        sobol_kwargs["seed"] = seed
    sobol_problem = {
        "num_vars": len(survivors),
        "names": survivors,
        "bounds": survivor_bounds,
    }
    X_sobol_reduced = sobol_sample_mod.sample(sobol_problem, sobol_n_base, **sobol_kwargs)

    n_sobol = X_sobol_reduced.shape[0]
    X_sobol_full = np.tile(midpoints, (n_sobol, 1))
    for new_col, sv in enumerate(survivors):
        X_sobol_full[:, param_names.index(sv)] = X_sobol_reduced[:, new_col]

    if cancel_event is not None and cancel_event.is_set():
        return None, []

    Y_sobol_raw = np.asarray(evaluation_fn(X_sobol_full))
    if Y_sobol_raw.ndim == 1:
        Y_sobol_raw = Y_sobol_raw[:, np.newaxis]

    sobol_failed = np.any(~np.isfinite(Y_sobol_raw), axis=1)
    sobol_failure_rate = float(np.sum(sobol_failed)) / max(n_sobol, 1)
    additions = list(survivors) if sobol_failure_rate > sobol_failure_threshold else []

    Y_sobol_clean = np.where(np.isfinite(Y_sobol_raw), Y_sobol_raw, 1e6)
    if Y_sobol_clean.shape[1] == 1:
        sobol_result = analyzer.analyze(Y_sobol_clean[:, 0])
    else:
        sobol_result = analyzer.analyze(Y_sobol_clean)
    return sobol_result, additions


def run_preflight(
    param_names: list[str],
    param_bounds: list[tuple[float, float]],
    evaluation_fn: Callable[[np.ndarray], np.ndarray],
    *,
    n_trajectories: int = 10,
    num_levels: int = 4,
    negligible_threshold: float = 0.1,
    blowup_threshold: float = 0.30,
    sobol_n_base: int = 64,
    sobol_failure_threshold: float = 0.10,
    seed: int | None = None,
    cancel_event: threading.Event | None = None,
) -> PreflightResult:
    """Run two-stage pre-flight sensitivity analysis (Morris + Sobol).

    See :func:`_run_morris_stage` and :func:`_maybe_run_sobol_stage` for the
    per-stage implementations. Aborts with :class:`PreflightEvalError` if
    Morris fails majority-of-samples; otherwise assembles a
    :class:`PreflightResult` with detected issues sorted errors-first.

    ``cancel_event.set()`` before or during the run short-circuits and yields
    a minimal result without raising.
    """
    t_start = time.monotonic()

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def _empty_result() -> PreflightResult:
        return PreflightResult(
            screening=[], sobol=None, issues=[], survivors=[],
            elapsed_seconds=time.monotonic() - t_start,
        )

    if _cancelled():
        return _empty_result()

    screening, failure_rate, _Y_clean, blowup_params_set = _run_morris_stage(
        param_names=param_names,
        param_bounds=param_bounds,
        evaluation_fn=evaluation_fn,
        n_trajectories=n_trajectories,
        num_levels=num_levels,
        negligible_threshold=negligible_threshold,
        seed=seed,
        cancel_event=cancel_event,
    )
    if not screening:
        # Morris was cancelled during sampling.
        return _empty_result()

    blowup_params: list[str] = sorted(blowup_params_set)
    if failure_rate > blowup_threshold and not blowup_params:
        blowup_params = list(param_names)

    survivors = [ps.key for ps in screening if ps.influential]

    if _cancelled():
        issues = detect_issues(screening, blowup_params=blowup_params)
        return PreflightResult(
            screening=screening, sobol=None, issues=issues,
            survivors=survivors, elapsed_seconds=time.monotonic() - t_start,
        )

    sobol_result, sobol_additions = _maybe_run_sobol_stage(
        screening=screening,
        param_names=param_names,
        param_bounds=param_bounds,
        evaluation_fn=evaluation_fn,
        sobol_n_base=sobol_n_base,
        sobol_failure_threshold=sobol_failure_threshold,
        seed=seed,
        cancel_event=cancel_event,
    )
    if sobol_additions:
        blowup_params = sorted(set(blowup_params) | set(sobol_additions))

    issues = detect_issues(screening, sobol_result=sobol_result, blowup_params=blowup_params)
    return PreflightResult(
        screening=screening,
        sobol=sobol_result,
        issues=issues,
        survivors=survivors,
        elapsed_seconds=time.monotonic() - t_start,
    )


# ---------------------------------------------------------------------------
# Evaluation Function Factory
# ---------------------------------------------------------------------------


def make_preflight_eval_fn(
    free_params: list[FreeParameter],
    base_config: dict[str, str],
    output_dir: Path,
    objective_fns: list[Callable],
    *,
    run_years: int | None = None,
    n_workers: int = 1,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create an evaluation function suitable for ``run_preflight()``.

    The returned callable accepts a sample matrix ``X`` of shape ``(N, k)``
    and returns a ``(N, n_obj)`` array of objective values. Failed rows are
    left as ``inf`` (SALib then treats them as blow-ups) and the exception is
    logged at WARNING level.

    When ``run_years`` is None (default), the preflight run length is clamped
    to ``min(5, base_config["simulation.time.nyear"])`` — the legacy contract.
    When explicitly set, the caller's value is used verbatim (useful for
    fast-failure tests and when the caller already decided run length).

    The returned callable exposes two diagnostic attributes:

    * ``samples`` — number of sample rows evaluated across all calls
    * ``failures`` — number of sample rows that raised an exception
    """
    n_obj = len(objective_fns)

    if run_years is None:
        configured_years = int(base_config.get("simulation.time.nyear", "10"))
        effective_years = min(5, configured_years)
    else:
        effective_years = int(run_years)

    class _EvalFn:
        def __init__(self) -> None:
            self.samples = 0
            self.failures = 0

        def _one(self, i: int, X: np.ndarray) -> tuple[int, np.ndarray | None, Exception | None]:
            config = dict(base_config)
            config["simulation.time.nyear"] = str(effective_years)
            for j, fp in enumerate(free_params):
                val = float(X[i, j])
                if fp.transform is Transform.LOG:
                    val = 10.0**val
                config[fp.key] = str(val)
            try:
                engine = PythonEngine()
                out_i = Path(output_dir) / f"preflight_{i}"
                out_i.mkdir(parents=True, exist_ok=True)
                engine.run(config, out_i)
                osmose_results = OsmoseResults(out_i)
                row = np.array([float(fn(osmose_results)) for fn in objective_fns])
                return i, row, None
            except Exception as exc:  # noqa: BLE001 — preflight is best-effort
                return i, None, exc

        def __call__(self, X: np.ndarray) -> np.ndarray:
            n_samples = X.shape[0]
            results_matrix = np.full((n_samples, n_obj), np.inf)

            def _record(i: int, row: np.ndarray | None, err: Exception | None) -> None:
                if err is not None:
                    _log.warning(
                        "preflight sample %d failed (%s: %s); row left as inf",
                        i,
                        type(err).__name__,
                        err,
                    )
                    self.failures += 1
                else:
                    results_matrix[i] = row
                self.samples += 1

            if n_workers <= 1:
                for i in range(n_samples):
                    _record(*self._one(i, X))
            else:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    for i, row, err in pool.map(lambda i: self._one(i, X), range(n_samples)):
                        _record(i, row, err)

            return results_matrix

    return _EvalFn()
