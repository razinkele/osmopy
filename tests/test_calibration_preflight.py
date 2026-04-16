"""Tests for pre-flight sensitivity analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.calibration.preflight import (
    IssueCategory,
    IssueSeverity,
    ParameterScreening,
    PreflightIssue,
    PreflightResult,
    run_morris_screening,
    detect_issues,  # noqa: F401 — used in TestIssueDetection below
)


class TestDataModel:
    def test_issue_category_is_enum(self) -> None:
        from enum import Enum

        assert issubclass(IssueCategory, Enum)
        assert IssueCategory.NEGLIGIBLE
        assert IssueCategory.BLOWUP
        assert IssueCategory.FLAT_OBJECTIVE
        assert IssueCategory.BOUND_TIGHT
        assert IssueCategory.ALL_NEGLIGIBLE

    def test_issue_severity_is_enum(self) -> None:
        from enum import Enum

        assert issubclass(IssueSeverity, Enum)
        assert IssueSeverity.WARNING
        assert IssueSeverity.ERROR

    def test_parameter_screening_valid(self) -> None:
        ps = ParameterScreening(
            key="sp0.linf", mu_star=1.5, sigma=0.3, mu_star_conf=0.2, influential=True
        )
        assert ps.key == "sp0.linf"
        assert ps.mu_star == 1.5
        assert ps.sigma == 0.3
        assert ps.mu_star_conf == 0.2
        assert ps.influential is True

    def test_parameter_screening_negative_mu_star_raises(self) -> None:
        with pytest.raises(ValueError, match="mu_star"):
            ParameterScreening(
                key="sp0.linf", mu_star=-0.1, sigma=0.3, mu_star_conf=0.2, influential=False
            )

    def test_preflight_issue_valid(self) -> None:
        issue = PreflightIssue(
            category=IssueCategory.NEGLIGIBLE,
            severity=IssueSeverity.WARNING,
            param_key="sp0.linf",
            message="Parameter has negligible influence.",
            suggestion="Consider removing from free parameters.",
            auto_fixable=True,
        )
        assert issue.category is IssueCategory.NEGLIGIBLE
        assert issue.severity is IssueSeverity.WARNING
        assert issue.param_key == "sp0.linf"
        assert issue.auto_fixable is True

    def test_preflight_result_valid(self) -> None:
        screening = [
            ParameterScreening(key="a", mu_star=2.0, sigma=0.5, mu_star_conf=0.1, influential=True)
        ]
        result = PreflightResult(
            screening=screening,
            sobol=None,
            issues=[],
            survivors=["a"],
            elapsed_seconds=1.23,
        )
        assert result.screening == screening
        assert result.sobol is None
        assert result.survivors == ["a"]
        assert result.elapsed_seconds == pytest.approx(1.23)


class TestMorrisScreening:
    """Tests for run_morris_screening()."""

    def test_ranking_correctness(self) -> None:
        """y = 3*a + 0*b + 0.5*c  →  a highest mu_star, b negligible, c influential."""

        def eval_fn(X: np.ndarray) -> np.ndarray:
            return 3.0 * X[:, 0] + 0.0 * X[:, 1] + 0.5 * X[:, 2]

        results = run_morris_screening(
            param_names=["a", "b", "c"],
            param_bounds=[(0, 1), (0, 1), (0, 1)],
            eval_fn=eval_fn,
            n_trajectories=20,
            seed=42,
        )
        by_key = {ps.key: ps for ps in results}
        assert by_key["a"].mu_star > by_key["c"].mu_star > by_key["b"].mu_star
        assert by_key["a"].influential
        assert by_key["c"].influential
        assert not by_key["b"].influential

    def test_multi_objective_aggregation(self) -> None:
        """obj1 = 3*a, obj2 = 3*b  →  both a and b influential after max aggregation."""

        def eval_fn(X: np.ndarray) -> np.ndarray:
            obj1 = 3.0 * X[:, 0]
            obj2 = 3.0 * X[:, 1]
            return np.column_stack([obj1, obj2])

        results = run_morris_screening(
            param_names=["a", "b"],
            param_bounds=[(0, 1), (0, 1)],
            eval_fn=eval_fn,
            n_trajectories=20,
            seed=42,
        )
        by_key = {ps.key: ps for ps in results}
        assert by_key["a"].influential
        assert by_key["b"].influential
