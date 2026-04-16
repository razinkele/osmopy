"""Tests for pre-flight sensitivity analysis module."""

from __future__ import annotations

import pytest

from osmose.calibration.preflight import (
    IssueCategory,
    IssueSeverity,
    ParameterScreening,
    PreflightIssue,
    PreflightResult,
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
