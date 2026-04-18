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


class TestIssueDetection:
    """Tests for detect_issues()."""

    def _make_screening(
        self, influential: bool, key: str = "a", mu_star: float = 1.0, sigma: float = 0.2
    ) -> ParameterScreening:
        return ParameterScreening(
            key=key, mu_star=mu_star, sigma=sigma, mu_star_conf=0.05, influential=influential
        )

    def test_negligible_detected(self) -> None:
        screening = [
            self._make_screening(influential=True, key="a"),
            self._make_screening(influential=False, key="b"),
        ]
        issues = detect_issues(screening)
        negligible = [i for i in issues if i.category is IssueCategory.NEGLIGIBLE]
        assert len(negligible) == 1
        assert negligible[0].param_key == "b"
        assert negligible[0].severity is IssueSeverity.WARNING
        assert negligible[0].auto_fixable is True

    def test_all_negligible_detected(self) -> None:
        screening = [
            self._make_screening(influential=False, key="a"),
            self._make_screening(influential=False, key="b"),
        ]
        issues = detect_issues(screening)
        all_neg = [i for i in issues if i.category is IssueCategory.ALL_NEGLIGIBLE]
        assert len(all_neg) == 1
        assert all_neg[0].severity is IssueSeverity.ERROR
        assert all_neg[0].auto_fixable is False

    def test_flat_objective_detected(self) -> None:
        screening = [self._make_screening(influential=True, key="a")]
        S1 = np.zeros((2, 1))
        sobol_result = {
            "S1": S1,
            "ST": np.ones((2, 1)) * 0.5,
            "param_names": ["a"],
            "objective_names": ["biomass", "yield"],
        }
        issues = detect_issues(screening, sobol_result=sobol_result)
        flat = [i for i in issues if i.category is IssueCategory.FLAT_OBJECTIVE]
        assert len(flat) == 2

    def test_bound_tight_detected(self) -> None:
        screening = [
            ParameterScreening(key="a", mu_star=1.0, sigma=2.0, mu_star_conf=0.05, influential=True)
        ]
        sobol_result = {
            "S1": np.array([0.4]),
            "ST": np.array([0.5]),
            "param_names": ["a"],
        }
        issues = detect_issues(screening, sobol_result=sobol_result)
        bound_tight = [i for i in issues if i.category is IssueCategory.BOUND_TIGHT]
        assert len(bound_tight) == 1
        assert bound_tight[0].param_key == "a"
        assert bound_tight[0].severity is IssueSeverity.WARNING

    def test_no_issues_when_clean(self) -> None:
        screening = [
            ParameterScreening(
                key="a", mu_star=2.0, sigma=0.3, mu_star_conf=0.05, influential=True
            ),
            ParameterScreening(
                key="b", mu_star=1.5, sigma=0.2, mu_star_conf=0.05, influential=True
            ),
        ]
        sobol_result = {
            "S1": np.array([0.4, 0.35]),
            "ST": np.array([0.45, 0.4]),
            "param_names": ["a", "b"],
        }
        issues = detect_issues(screening, sobol_result=sobol_result)
        assert issues == []


class TestRunPreflight:
    """Tests for run_preflight() two-stage orchestrator."""

    def test_happy_path(self) -> None:
        """y = 3*a + 0*b + 0.5*c  ->  a,c survive, b filtered, sobol populated."""
        from osmose.calibration.preflight import run_preflight

        def evaluation_fn(X: np.ndarray) -> np.ndarray:
            return 3.0 * X[:, 0] + 0.0 * X[:, 1] + 0.5 * X[:, 2]

        result = run_preflight(
            param_names=["a", "b", "c"],
            param_bounds=[(0, 1), (0, 1), (0, 1)],
            evaluation_fn=evaluation_fn,
            n_trajectories=20,
            seed=42,
        )

        assert isinstance(result.screening, list)
        assert len(result.screening) == 3
        assert result.elapsed_seconds > 0

        by_key = {ps.key: ps for ps in result.screening}
        assert by_key["a"].influential
        assert by_key["c"].influential
        assert not by_key["b"].influential

        assert "a" in result.survivors
        assert "c" in result.survivors
        assert "b" not in result.survivors

        assert result.sobol is not None
        assert "S1" in result.sobol
        assert "ST" in result.sobol

    def test_failure_abort(self) -> None:
        """50% failure rate triggers blowup issues."""
        from osmose.calibration.preflight import run_preflight, IssueCategory

        def evaluation_fn(X: np.ndarray) -> np.ndarray:
            Y = np.ones(X.shape[0])
            Y[::2] = np.nan
            return Y

        result = run_preflight(
            param_names=["a", "b"],
            param_bounds=[(0, 1), (0, 1)],
            evaluation_fn=evaluation_fn,
            n_trajectories=10,
            blowup_threshold=0.30,
            seed=1,
        )

        blowup_issues = [i for i in result.issues if i.category is IssueCategory.BLOWUP]
        assert len(blowup_issues) > 0

    def test_single_parameter(self) -> None:
        """k=1 works correctly."""
        from osmose.calibration.preflight import run_preflight

        def evaluation_fn(X: np.ndarray) -> np.ndarray:
            return 2.0 * X[:, 0]

        result = run_preflight(
            param_names=["a"],
            param_bounds=[(0, 1)],
            evaluation_fn=evaluation_fn,
            n_trajectories=10,
            seed=0,
        )

        assert len(result.screening) == 1
        assert result.screening[0].key == "a"
        assert result.elapsed_seconds >= 0

    def test_cancellation(self) -> None:
        """cancel_event.set() before call returns quickly with empty result."""
        import threading
        from osmose.calibration.preflight import run_preflight

        cancel_event = threading.Event()
        cancel_event.set()

        def evaluation_fn(X: np.ndarray) -> np.ndarray:
            raise RuntimeError("Should not be called")

        result = run_preflight(
            param_names=["a", "b"],
            param_bounds=[(0, 1), (0, 1)],
            evaluation_fn=evaluation_fn,
            cancel_event=cancel_event,
        )

        assert result.screening == []
        assert result.survivors == []
        assert result.issues == []
        assert result.elapsed_seconds >= 0


class TestMakePreflightEvalFn:
    """Tests for make_preflight_eval_fn() factory."""

    def test_sim_years_clamped_to_5(self) -> None:
        """Configured 30yr -> run uses 5."""
        from unittest.mock import MagicMock, patch
        from osmose.calibration.problem import FreeParameter
        from osmose.calibration.preflight import make_preflight_eval_fn
        from pathlib import Path

        free_params = [FreeParameter(key="sp.linf", lower_bound=0.0, upper_bound=1.0)]
        base_config = {"simulation.time.nyear": "30", "sp.linf": "0.5"}
        output_dir = Path("/tmp/test_osmose_output")

        captured_configs = []

        def mock_objective(results):
            return 1.0

        with patch("osmose.calibration.preflight.PythonEngine") as MockEngine:
            with patch("osmose.calibration.preflight.OsmoseResults") as MockResults:
                mock_engine_instance = MagicMock()
                MockEngine.return_value = mock_engine_instance
                MockResults.return_value = MagicMock()

                def capture_run(config, out_dir, seed=0):
                    captured_configs.append(dict(config))
                    return MagicMock()

                mock_engine_instance.run.side_effect = capture_run

                fn = make_preflight_eval_fn(
                    free_params=free_params,
                    base_config=base_config,
                    output_dir=output_dir,
                    objective_fns=[mock_objective],
                )
                X = np.array([[0.5]])
                fn(X)

        assert len(captured_configs) == 1
        assert captured_configs[0]["simulation.time.nyear"] == "5"

    def test_sim_years_keeps_short(self) -> None:
        """Configured 3yr -> run keeps 3."""
        from unittest.mock import MagicMock, patch
        from osmose.calibration.problem import FreeParameter
        from osmose.calibration.preflight import make_preflight_eval_fn
        from pathlib import Path

        free_params = [FreeParameter(key="sp.linf", lower_bound=0.0, upper_bound=1.0)]
        base_config = {"simulation.time.nyear": "3", "sp.linf": "0.5"}
        output_dir = Path("/tmp/test_osmose_output")

        captured_configs = []

        def mock_objective(results):
            return 1.0

        with patch("osmose.calibration.preflight.PythonEngine") as MockEngine:
            with patch("osmose.calibration.preflight.OsmoseResults") as MockResults:
                mock_engine_instance = MagicMock()
                MockEngine.return_value = mock_engine_instance
                MockResults.return_value = MagicMock()

                def capture_run(config, out_dir, seed=0):
                    captured_configs.append(dict(config))
                    return MagicMock()

                mock_engine_instance.run.side_effect = capture_run

                fn = make_preflight_eval_fn(
                    free_params=free_params,
                    base_config=base_config,
                    output_dir=output_dir,
                    objective_fns=[mock_objective],
                )
                X = np.array([[0.5]])
                fn(X)

        assert len(captured_configs) == 1
        assert captured_configs[0]["simulation.time.nyear"] == "3"

    def test_log_transform_applied(self) -> None:
        """-1.0 with LOG transform -> config value 0.1."""
        from unittest.mock import MagicMock, patch
        from osmose.calibration.problem import FreeParameter, Transform
        from osmose.calibration.preflight import make_preflight_eval_fn
        from pathlib import Path

        free_params = [
            FreeParameter(key="sp.linf", lower_bound=-2.0, upper_bound=2.0, transform=Transform.LOG)
        ]
        base_config = {"simulation.time.nyear": "3", "sp.linf": "1.0"}
        output_dir = Path("/tmp/test_osmose_output")

        captured_configs = []

        def mock_objective(results):
            return 1.0

        with patch("osmose.calibration.preflight.PythonEngine") as MockEngine:
            with patch("osmose.calibration.preflight.OsmoseResults") as MockResults:
                mock_engine_instance = MagicMock()
                MockEngine.return_value = mock_engine_instance
                MockResults.return_value = MagicMock()

                def capture_run(config, out_dir, seed=0):
                    captured_configs.append(dict(config))
                    return MagicMock()

                mock_engine_instance.run.side_effect = capture_run

                fn = make_preflight_eval_fn(
                    free_params=free_params,
                    base_config=base_config,
                    output_dir=output_dir,
                    objective_fns=[mock_objective],
                )
                X = np.array([[-1.0]])
                fn(X)

        assert len(captured_configs) == 1
        actual_val = float(captured_configs[0]["sp.linf"])
        assert abs(actual_val - 0.1) < 1e-9


def test_preflight_eval_fn_logs_and_counts_failures(monkeypatch, caplog, tmp_path) -> None:
    """Engine exceptions must be logged, counted, and leave the row as inf."""
    import logging

    from osmose.calibration import preflight as pre
    from osmose.calibration.problem import FreeParameter

    class _FakeEngine:
        def run(self, config, output_dir):  # noqa: ARG002
            raise RuntimeError("synthetic blow-up")

    class _FakeResults:
        def __init__(self, *a, **kw) -> None: ...

    monkeypatch.setattr(pre, "PythonEngine", lambda: _FakeEngine())
    monkeypatch.setattr(pre, "OsmoseResults", _FakeResults)

    fp_spec = [FreeParameter(key="predation.efficiency.sp0",
                             lower_bound=0.1, upper_bound=0.9)]
    fn = pre.make_preflight_eval_fn(
        free_params=fp_spec,
        base_config={"simulation.time.nyear": "1"},
        output_dir=tmp_path,
        objective_fns=[lambda r: 0.0],
        run_years=1,
    )

    X = np.array([[0.5]])
    with caplog.at_level(logging.WARNING, logger="osmose.calibration.preflight"):
        Y = fn(X)
    assert not np.isfinite(Y[0, 0])
    assert any("synthetic blow-up" in rec.getMessage() for rec in caplog.records), \
        "Exception message must appear in logs (no silent pass)"
    assert fn.failures == 1
    assert fn.samples == 1


def test_run_preflight_aborts_when_failure_rate_exceeds_threshold(tmp_path) -> None:
    """If >50% of Morris samples fail, run_preflight must raise PreflightEvalError."""
    from osmose.calibration import preflight as pre

    def always_fails(X):
        return np.full((X.shape[0], 1), np.inf)

    with pytest.raises(pre.PreflightEvalError, match="failure rate"):
        pre.run_preflight(
            param_names=["a", "b"],
            param_bounds=[(0.0, 1.0), (0.0, 1.0)],
            evaluation_fn=always_fails,
            n_trajectories=3,
            num_levels=4,
        )


def test_preflight_eval_fn_parallel_matches_serial(monkeypatch, tmp_path) -> None:
    """With per-sample output_dir isolation, n_workers must not change output."""
    from pathlib import Path

    from osmose.calibration import preflight as pre
    from osmose.calibration.problem import FreeParameter

    class _Engine:
        def run(self, config, output_dir):
            (Path(output_dir) / "value.txt").write_text(str(config["species.k.sp0"]))

    class _Results:
        def __init__(self, output_dir, *a, **kw) -> None:
            self.value = float((Path(output_dir) / "value.txt").read_text())

    monkeypatch.setattr(pre, "PythonEngine", lambda: _Engine())
    monkeypatch.setattr(pre, "OsmoseResults", _Results)

    fp = [FreeParameter("species.k.sp0", 0.1, 0.9)]

    def build(n_workers):
        return pre.make_preflight_eval_fn(
            free_params=fp,
            base_config={},
            output_dir=tmp_path,
            objective_fns=[lambda r: r.value * 2.0],
            run_years=1,
            n_workers=n_workers,
        )

    X = np.linspace(0.1, 0.9, 12).reshape(-1, 1)
    Y_serial = build(1)(X)
    Y_parallel = build(4)(X)
    np.testing.assert_allclose(Y_serial, Y_parallel)


def test_run_morris_stage_returns_screening_and_failure_rate() -> None:
    """Extracted Morris stage is a pure sampler+analyzer with signature
    (screening, failure_rate, Y_clean, blowup_params_set)."""
    from osmose.calibration import preflight as pre

    param_names = ["a", "b"]
    param_bounds = [(0.0, 1.0), (0.0, 1.0)]

    def eval_fn(X):
        return np.column_stack([X[:, 0], X[:, 1]])

    screening, failure_rate, Y_clean, blowup_params = pre._run_morris_stage(
        param_names=param_names,
        param_bounds=param_bounds,
        evaluation_fn=eval_fn,
        n_trajectories=4,
        num_levels=4,
        negligible_threshold=0.1,
        seed=1,
    )
    assert len(screening) == 2
    assert {s.key for s in screening} == {"a", "b"}
    assert failure_rate == 0.0
    assert Y_clean.shape[1] == 2
    assert blowup_params == set()


def test_maybe_run_sobol_stage_skips_when_few_survivors() -> None:
    """< 2 survivors → Sobol stage returns (None, [])."""
    from osmose.calibration import preflight as pre

    screening = [
        pre.ParameterScreening(
            key="a", mu_star=0.5, sigma=0.1, mu_star_conf=0.01, influential=True,
        ),
    ]
    result, additions = pre._maybe_run_sobol_stage(
        screening=screening,
        param_names=["a"],
        param_bounds=[(0.0, 1.0)],
        evaluation_fn=lambda X: np.zeros((X.shape[0], 1)),
        sobol_n_base=16,
        sobol_failure_threshold=0.5,
        seed=1,
    )
    assert result is None
    assert additions == []


# ---------------------------------------------------------------------------
# Task 8: detect_issues split into per-category helpers
# ---------------------------------------------------------------------------


def _screen(key, mu_star=1.0, sigma=0.1, influential=True):
    return ParameterScreening(
        key=key, mu_star=mu_star, sigma=sigma, mu_star_conf=0.01,
        influential=influential,
    )


def test_issues_blowup_emits_error_per_key() -> None:
    from osmose.calibration.preflight import _issues_blowup
    issues = _issues_blowup(["a", "b"])
    assert len(issues) == 2
    assert all(i.category is IssueCategory.BLOWUP for i in issues)
    assert all(i.severity is IssueSeverity.ERROR for i in issues)
    assert {i.param_key for i in issues} == {"a", "b"}


def test_issues_negligible_emits_warning_per_non_influential() -> None:
    from osmose.calibration.preflight import _issues_negligible
    screening = [_screen("a", influential=False), _screen("b", influential=True)]
    issues = _issues_negligible(screening)
    assert len(issues) == 1
    assert issues[0].category is IssueCategory.NEGLIGIBLE
    assert issues[0].severity is IssueSeverity.WARNING
    assert issues[0].param_key == "a"


def test_issues_all_negligible_fires_when_every_param_is_negligible() -> None:
    from osmose.calibration.preflight import _issues_all_negligible
    screening = [_screen("a", influential=False), _screen("b", influential=False)]
    issues = _issues_all_negligible(screening)
    assert len(issues) == 1
    assert issues[0].category is IssueCategory.ALL_NEGLIGIBLE
    assert issues[0].severity is IssueSeverity.ERROR
    assert issues[0].param_key is None


def test_issues_flat_fires_on_low_S1_sum() -> None:
    from osmose.calibration.preflight import _issues_flat
    sobol = {"S1": [0.01, 0.01]}  # sum = 0.02 < 0.05
    issues = _issues_flat(sobol)
    assert any(i.category is IssueCategory.FLAT_OBJECTIVE for i in issues)


def test_issues_bound_tight_fires_on_high_ST_and_high_sigma_ratio() -> None:
    from osmose.calibration.preflight import _issues_bound_tight
    screening = [_screen("a", mu_star=0.2, sigma=0.4, influential=True)]  # sigma/mu* = 2.0
    sobol = {"ST": [0.5], "param_names": ["a"]}  # ST > 0.3
    issues = _issues_bound_tight(screening, sobol)
    assert any(i.category is IssueCategory.BOUND_TIGHT for i in issues)


def test_detect_issues_preserves_error_first_sort() -> None:
    """Orchestrator must sort errors before warnings (legacy behavior)."""
    screening = [_screen("a", influential=False)]  # WARNING
    issues = detect_issues(screening, blowup_params=["b"])  # ERROR from blowup
    assert issues[0].severity is IssueSeverity.ERROR
    assert issues[-1].severity is IssueSeverity.WARNING
