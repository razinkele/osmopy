"""Tests for pre-flight UI integration."""

from __future__ import annotations

from osmose.calibration.preflight import (
    IssueCategory,
    IssueSeverity,
    ParameterScreening,
    PreflightIssue,
    PreflightResult,
)
from ui.pages.calibration_handlers import apply_preflight_fixes, build_preflight_modal
from osmose.calibration.problem import FreeParameter


class TestBuildPreflightModal:
    def test_modal_contains_issue_checkboxes(self) -> None:
        result = PreflightResult(
            screening=[
                ParameterScreening("a", 1.0, 0.1, 0.05, True),
                ParameterScreening("b", 0.001, 0.0, 0.001, False),
            ],
            sobol=None,
            issues=[
                PreflightIssue(
                    IssueCategory.NEGLIGIBLE,
                    IssueSeverity.WARNING,
                    "b",
                    "Negligible",
                    "Remove b",
                    auto_fixable=True,
                ),
            ],
            survivors=["a"],
            elapsed_seconds=10.0,
        )
        modal = build_preflight_modal(result)
        html = str(modal)
        assert "Negligible" in html
        assert "preflight_fix_0" in html

    def test_non_fixable_shows_as_text(self) -> None:
        result = PreflightResult(
            screening=[ParameterScreening("a", 1.0, 0.1, 0.05, True)],
            sobol=None,
            issues=[
                PreflightIssue(
                    IssueCategory.FLAT_OBJECTIVE,
                    IssueSeverity.WARNING,
                    None,
                    "Objective flat",
                    "Check data",
                    auto_fixable=False,
                ),
            ],
            survivors=["a"],
            elapsed_seconds=5.0,
        )
        modal = build_preflight_modal(result)
        html = str(modal)
        assert "Objective flat" in html
        assert "preflight_fix_0" not in html

    def test_no_issues_returns_none(self) -> None:
        result = PreflightResult(
            screening=[ParameterScreening("a", 1.0, 0.1, 0.05, True)],
            sobol=None,
            issues=[],
            survivors=["a"],
            elapsed_seconds=5.0,
        )
        assert build_preflight_modal(result) is None


class TestApplyPreflightFixes:
    def test_negligible_removed(self) -> None:
        free_params = [FreeParameter("a", 0.1, 1.0), FreeParameter("b", 0.1, 1.0)]
        issues = [
            PreflightIssue(
                IssueCategory.NEGLIGIBLE,
                IssueSeverity.WARNING,
                "b",
                "Negligible",
                "Remove b",
                auto_fixable=True,
            )
        ]
        updated = apply_preflight_fixes(free_params, issues, [True])
        assert len(updated) == 1
        assert updated[0].key == "a"

    def test_bound_adjusted(self) -> None:
        free_params = [FreeParameter("a", 0.1, 1.0)]
        issues = [
            PreflightIssue(
                IssueCategory.BOUND_TIGHT,
                IssueSeverity.WARNING,
                "a",
                "Tight bounds",
                "Widen by 20%",
                auto_fixable=True,
            )
        ]
        updated = apply_preflight_fixes(free_params, issues, [True])
        assert len(updated) == 1
        assert updated[0].lower_bound < 0.1
        assert updated[0].upper_bound > 1.0

    def test_unchecked_fix_not_applied(self) -> None:
        free_params = [FreeParameter("a", 0.1, 1.0), FreeParameter("b", 0.1, 1.0)]
        issues = [
            PreflightIssue(
                IssueCategory.NEGLIGIBLE,
                IssueSeverity.WARNING,
                "b",
                "Negligible",
                "Remove b",
                auto_fixable=True,
            )
        ]
        updated = apply_preflight_fixes(free_params, issues, [False])
        assert len(updated) == 2

    def test_blowup_tightens_bounds(self) -> None:
        free_params = [FreeParameter("a", 0.0, 10.0)]
        issues = [
            PreflightIssue(
                IssueCategory.BLOWUP,
                IssueSeverity.WARNING,
                "a",
                "Blowup",
                "Tighten bounds",
                auto_fixable=True,
            )
        ]
        updated = apply_preflight_fixes(free_params, issues, [True])
        assert len(updated) == 1
        assert updated[0].lower_bound > 0.0
        assert updated[0].upper_bound < 10.0
