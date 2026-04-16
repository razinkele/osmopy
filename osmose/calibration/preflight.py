"""Pre-flight sensitivity analysis for OSMOSE calibration.

Validates calibration parameters before optimization runs using Morris screening
and optional Sobol analysis to detect negligible parameters, blow-ups, flat
objectives, and tight bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from SALib.sample import morris as morris_sample  # type: ignore[import-untyped]
from SALib.analyze import morris as morris_analyze  # type: ignore[import-untyped]

if TYPE_CHECKING:
    pass


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


def detect_issues(
    screening: list[ParameterScreening],
    sobol_result: dict | None = None,
    blowup_params: list[str] | None = None,
) -> list[PreflightIssue]:
    """Detect pre-flight issues from screening results and optional Sobol indices.

    Parameters
    ----------
    screening:
        Per-parameter Morris screening results (from :func:`run_morris_screening`).
    sobol_result:
        Optional dict returned by
        :meth:`~osmose.calibration.sensitivity.SensitivityAnalyzer.analyze`.
        Supports both 1-D (single objective) and 2-D (multi-objective) shapes.
    blowup_params:
        Parameter keys that caused numerical blow-ups during sampling.

    Returns
    -------
    list[PreflightIssue]
        Issues ordered errors-first, then warnings.
    """
    issues: list[PreflightIssue] = []

    # --- BLOWUP -----------------------------------------------------------------
    for key in blowup_params or []:
        issues.append(
            PreflightIssue(
                category=IssueCategory.BLOWUP,
                severity=IssueSeverity.ERROR,
                param_key=key,
                message=f"Parameter '{key}' caused numerical blow-up during Morris sampling.",
                suggestion="Tighten the parameter bounds or check the model for instability.",
                auto_fixable=False,
            )
        )

    # --- NEGLIGIBLE (per-parameter) --------------------------------------------
    negligible_keys = [ps.key for ps in screening if not ps.influential]
    for key in negligible_keys:
        issues.append(
            PreflightIssue(
                category=IssueCategory.NEGLIGIBLE,
                severity=IssueSeverity.WARNING,
                param_key=key,
                message=f"Parameter '{key}' has negligible influence on the objective(s).",
                suggestion="Consider removing from the free-parameter set to reduce cost.",
                auto_fixable=True,
            )
        )

    # --- ALL_NEGLIGIBLE --------------------------------------------------------
    if screening and all(not ps.influential for ps in screening):
        issues.append(
            PreflightIssue(
                category=IssueCategory.ALL_NEGLIGIBLE,
                severity=IssueSeverity.ERROR,
                param_key=None,
                message="All parameters are negligible — the objective is insensitive to every free parameter.",
                suggestion="Review the objective function, parameter bounds, and model configuration.",
                auto_fixable=False,
            )
        )

    # --- FLAT_OBJECTIVE (from Sobol S1) ----------------------------------------
    if sobol_result is not None and "S1" in sobol_result:
        S1 = np.asarray(sobol_result["S1"])
        if S1.ndim == 1:
            # Single objective: shape (n_params,)
            total_variance = float(np.sum(np.maximum(0.0, S1)))
            if total_variance < 0.05:
                issues.append(
                    PreflightIssue(
                        category=IssueCategory.FLAT_OBJECTIVE,
                        severity=IssueSeverity.ERROR,
                        param_key=None,
                        message=f"Objective is effectively flat (sum(S1)={total_variance:.4f} < 0.05).",
                        suggestion="Check for output clipping, constant targets, or degenerate runs.",
                        auto_fixable=False,
                    )
                )
        else:
            # Multi-objective: shape (n_params, n_obj) or (n_obj, n_params)
            # Convention from SensitivityAnalyzer: shape is (n_obj, n_params)
            for obj_idx in range(S1.shape[0]):
                row = S1[obj_idx]
                total_variance = float(np.sum(np.maximum(0.0, row)))
                if total_variance < 0.05:
                    obj_names = sobol_result.get("objective_names", None)
                    obj_label = obj_names[obj_idx] if obj_names else f"obj_{obj_idx}"
                    issues.append(
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

    # --- BOUND_TIGHT (from Sobol ST + Morris sigma/mu_star) --------------------
    if sobol_result is not None and "ST" in sobol_result:
        ST = np.asarray(sobol_result["ST"])
        if ST.ndim > 1:
            # Aggregate across objectives via max
            ST_agg = np.max(ST, axis=0)
        else:
            ST_agg = ST

        param_names_sobol: list[str] = sobol_result.get("param_names", [])
        screening_map = {ps.key: ps for ps in screening}

        for j, key in enumerate(param_names_sobol):
            if j >= len(ST_agg):
                continue
            st_val = float(ST_agg[j])
            ps = screening_map.get(key)
            if ps is None:
                continue
            ratio = ps.sigma / ps.mu_star if ps.mu_star > 0 else 0.0
            if st_val > 0.3 and ratio > 1.5:
                issues.append(
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

    # Sort: errors first, then warnings
    issues.sort(key=lambda i: 0 if i.severity is IssueSeverity.ERROR else 1)
    return issues
