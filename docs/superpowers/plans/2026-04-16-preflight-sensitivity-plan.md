# Pre-flight Sensitivity Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a two-stage pre-flight sensitivity analysis (Morris screening + targeted Sobol) that runs automatically before calibration to validate parameter influence and problem well-posedness.

**Architecture:** New `osmose/calibration/preflight.py` contains all library logic (data model, Morris screening, issue detection, orchestrator, eval function factory). UI changes are minimal: a checkbox in `calibration.py`, a new message type in `CalibrationMessageQueue`, and a modal + handler in `calibration_handlers.py`.

**Tech Stack:** SALib (Morris + Sobol), NumPy, existing `SensitivityAnalyzer`, Shiny reactive + threading patterns.

**Spec:** `docs/superpowers/specs/2026-04-16-preflight-sensitivity-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `osmose/calibration/preflight.py` | **create** | Data model (enums, dataclasses), Morris screening, issue detection, `run_preflight()` orchestrator, `make_preflight_eval_fn()` factory |
| `osmose/calibration/__init__.py` | **modify** | Export new public API symbols |
| `ui/pages/calibration.py` | **modify** | Add checkbox + `preflight_result` reactive value |
| `ui/pages/calibration_handlers.py` | **modify** | Add `post_preflight()` to message queue, pre-flight thread, modal builder, fix-application handler |
| `tests/test_calibration_preflight.py` | **create** | Unit tests for library: Morris screening, issue detection, orchestrator, eval factory |
| `tests/test_ui_calibration_preflight.py` | **create** | UI tests: checkbox, modal rendering, fix application |

---

### Task 1: Data Model — Enums and Dataclasses

**Files:**
- Create: `osmose/calibration/preflight.py`
- Test: `tests/test_calibration_preflight.py`

- [ ] **Step 1: Write failing tests for data model**

Create `tests/test_calibration_preflight.py`:

```python
"""Tests for pre-flight sensitivity analysis."""

from __future__ import annotations

import enum

import pytest

from osmose.calibration.preflight import (
    IssueCategory,
    IssueSeverity,
    ParameterScreening,
    PreflightIssue,
    PreflightResult,
)


class TestDataModel:
    """Enums and dataclasses."""

    def test_issue_category_is_enum(self) -> None:
        assert issubclass(IssueCategory, enum.Enum)
        assert hasattr(IssueCategory, "NEGLIGIBLE")
        assert hasattr(IssueCategory, "BLOWUP")
        assert hasattr(IssueCategory, "FLAT_OBJECTIVE")
        assert hasattr(IssueCategory, "BOUND_TIGHT")
        assert hasattr(IssueCategory, "ALL_NEGLIGIBLE")

    def test_issue_severity_is_enum(self) -> None:
        assert issubclass(IssueSeverity, enum.Enum)
        assert hasattr(IssueSeverity, "WARNING")
        assert hasattr(IssueSeverity, "ERROR")

    def test_parameter_screening_valid(self) -> None:
        ps = ParameterScreening(
            key="species.k.sp0", mu_star=0.5, sigma=0.1, mu_star_conf=0.05, influential=True
        )
        assert ps.key == "species.k.sp0"
        assert ps.influential is True

    def test_parameter_screening_negative_mu_star_raises(self) -> None:
        with pytest.raises(ValueError, match="mu_star"):
            ParameterScreening(
                key="x", mu_star=-0.1, sigma=0.1, mu_star_conf=0.05, influential=True
            )

    def test_preflight_issue_valid(self) -> None:
        issue = PreflightIssue(
            category=IssueCategory.NEGLIGIBLE,
            severity=IssueSeverity.WARNING,
            param_key="species.k.sp0",
            message="Negligible influence",
            suggestion="Remove parameter",
            auto_fixable=True,
        )
        assert issue.category == IssueCategory.NEGLIGIBLE
        assert issue.auto_fixable is True

    def test_preflight_result_valid(self) -> None:
        result = PreflightResult(
            screening=[], sobol=None, issues=[], survivors=[], elapsed_seconds=0.0
        )
        assert result.survivors == []
        assert result.sobol is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py::TestDataModel -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.calibration.preflight'`

- [ ] **Step 3: Implement data model**

Create `osmose/calibration/preflight.py`:

```python
# osmose/calibration/preflight.py
"""Pre-flight sensitivity analysis for OSMOSE calibration.

Two-stage procedure (Morris screening + targeted Sobol) that validates
parameter influence and problem well-posedness before optimization.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class IssueCategory(enum.Enum):
    """Category of pre-flight issue."""

    NEGLIGIBLE = "negligible"
    BLOWUP = "blowup"
    FLAT_OBJECTIVE = "flat_objective"
    BOUND_TIGHT = "bound_tight"
    ALL_NEGLIGIBLE = "all_negligible"


class IssueSeverity(enum.Enum):
    """Severity of pre-flight issue."""

    WARNING = "warning"
    ERROR = "error"


@dataclass
class ParameterScreening:
    """Morris screening result for one parameter."""

    key: str
    mu_star: float
    sigma: float
    mu_star_conf: float
    influential: bool

    def __post_init__(self) -> None:
        if self.mu_star < 0:
            raise ValueError(f"mu_star must be >= 0, got {self.mu_star}")


@dataclass
class PreflightIssue:
    """A single issue found during pre-flight screening."""

    category: IssueCategory
    severity: IssueSeverity
    param_key: str | None
    message: str
    suggestion: str
    auto_fixable: bool


@dataclass
class PreflightResult:
    """Complete result of pre-flight screening."""

    screening: list[ParameterScreening]
    sobol: dict | None
    issues: list[PreflightIssue]
    survivors: list[str]
    elapsed_seconds: float
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py::TestDataModel -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/calibration/preflight.py tests/test_calibration_preflight.py
git commit -m "feat(calibration): add preflight data model — enums and dataclasses"
```

---

### Task 2: Morris Screening Logic

**Files:**
- Modify: `osmose/calibration/preflight.py`
- Test: `tests/test_calibration_preflight.py`

- [ ] **Step 1: Write failing tests for Morris screening**

Append to `tests/test_calibration_preflight.py`:

```python
import numpy as np

from osmose.calibration.preflight import run_morris_screening


class TestMorrisScreening:
    """Morris stage in isolation."""

    def test_ranking_correctness(self) -> None:
        """y = 3*a + 0*b + 0.5*c → a highest, b negligible, c influential."""
        names = ["a", "b", "c"]
        bounds = [(0, 1), (0, 1), (0, 1)]

        def eval_fn(x: np.ndarray) -> np.ndarray:
            return np.array([3 * x[0] + 0 * x[1] + 0.5 * x[2]])

        screening = run_morris_screening(names, bounds, eval_fn, morris_n=10)
        by_key = {s.key: s for s in screening}

        assert by_key["a"].influential is True
        assert by_key["b"].influential is False
        assert by_key["c"].influential is True
        assert by_key["a"].mu_star > by_key["c"].mu_star

    def test_multi_objective_aggregation(self) -> None:
        """obj1 depends on a only, obj2 depends on b only → both influential."""
        names = ["a", "b"]
        bounds = [(0, 1), (0, 1)]

        def eval_fn(x: np.ndarray) -> np.ndarray:
            return np.array([3 * x[0], 3 * x[1]])

        screening = run_morris_screening(names, bounds, eval_fn, morris_n=10)
        by_key = {s.key: s for s in screening}

        assert by_key["a"].influential is True
        assert by_key["b"].influential is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py::TestMorrisScreening -v`
Expected: FAIL — `cannot import name 'run_morris_screening'`

- [ ] **Step 3: Implement Morris screening**

Add to `osmose/calibration/preflight.py`:

```python
import numpy as np
from SALib.sample import morris as morris_sample  # type: ignore[import-untyped]
from SALib.analyze import morris as morris_analyze  # type: ignore[import-untyped]

from typing import Callable


def run_morris_screening(
    param_names: list[str],
    param_bounds: list[tuple[float, float]],
    eval_fn: Callable[[np.ndarray], np.ndarray],
    morris_n: int = 10,
    negligible_threshold: float = 0.01,
) -> list[ParameterScreening]:
    """Run Morris elementary effects screening.

    Args:
        param_names: Parameter names.
        param_bounds: (lower, upper) for each parameter.
        eval_fn: Maps parameter vector → objective array (1D).
        morris_n: Number of Morris trajectories.
        negligible_threshold: Fraction of max(mu_star) below which a param is negligible.

    Returns:
        List of ParameterScreening, one per parameter.
    """
    problem = {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": param_bounds,
    }

    X = morris_sample.sample(problem, morris_n)

    # Evaluate all samples
    n_samples = X.shape[0]
    first_y = eval_fn(X[0])
    n_obj = len(first_y)
    Y = np.zeros((n_samples, n_obj))
    Y[0] = first_y
    for i in range(1, n_samples):
        Y[i] = eval_fn(X[i])

    # Per-objective Morris analysis, aggregate via max
    k = len(param_names)
    mu_star_all = np.zeros((n_obj, k))
    sigma_all = np.zeros((n_obj, k))
    mu_star_conf_all = np.zeros((n_obj, k))

    for col in range(n_obj):
        result = morris_analyze.analyze(problem, X, Y[:, col])
        mu_star_all[col] = result["mu_star"]
        sigma_all[col] = result["sigma"]
        mu_star_conf_all[col] = result["mu_star_conf"]

    # Aggregate: max across objectives
    agg_mu_star = np.max(mu_star_all, axis=0)
    agg_sigma = np.max(sigma_all, axis=0)
    agg_conf = np.max(mu_star_conf_all, axis=0)

    # Classify
    max_mu = np.max(agg_mu_star) if np.max(agg_mu_star) > 0 else 1.0
    threshold = negligible_threshold * max_mu

    screening = []
    for j in range(k):
        influential = (agg_mu_star[j] + agg_conf[j]) >= threshold
        screening.append(
            ParameterScreening(
                key=param_names[j],
                mu_star=float(agg_mu_star[j]),
                sigma=float(agg_sigma[j]),
                mu_star_conf=float(agg_conf[j]),
                influential=influential,
            )
        )

    return screening
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py::TestMorrisScreening -v`
Expected: Both tests PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/calibration/preflight.py tests/test_calibration_preflight.py
git commit -m "feat(calibration): add Morris screening with multi-objective aggregation"
```

---

### Task 3: Issue Detection Logic

**Files:**
- Modify: `osmose/calibration/preflight.py`
- Test: `tests/test_calibration_preflight.py`

- [ ] **Step 1: Write failing tests for issue detection**

Append to `tests/test_calibration_preflight.py`:

```python
from osmose.calibration.preflight import (
    detect_issues,
    IssueCategory,
    IssueSeverity,
    ParameterScreening,
)


class TestIssueDetection:
    """Issue diagnosis logic."""

    def test_negligible_detected(self) -> None:
        screening = [
            ParameterScreening("a", mu_star=1.0, sigma=0.1, mu_star_conf=0.05, influential=True),
            ParameterScreening("b", mu_star=0.001, sigma=0.0, mu_star_conf=0.001, influential=False),
        ]
        issues = detect_issues(screening, sobol_result=None)
        negligible = [i for i in issues if i.category == IssueCategory.NEGLIGIBLE]
        assert len(negligible) == 1
        assert negligible[0].param_key == "b"
        assert negligible[0].auto_fixable is True
        assert negligible[0].severity == IssueSeverity.WARNING

    def test_all_negligible_detected(self) -> None:
        screening = [
            ParameterScreening("a", mu_star=0.001, sigma=0.0, mu_star_conf=0.0, influential=False),
            ParameterScreening("b", mu_star=0.002, sigma=0.0, mu_star_conf=0.0, influential=False),
        ]
        issues = detect_issues(screening, sobol_result=None)
        all_neg = [i for i in issues if i.category == IssueCategory.ALL_NEGLIGIBLE]
        assert len(all_neg) == 1
        assert all_neg[0].severity == IssueSeverity.ERROR
        assert all_neg[0].auto_fixable is False

    def test_flat_objective_detected(self) -> None:
        screening = [
            ParameterScreening("a", mu_star=1.0, sigma=0.1, mu_star_conf=0.05, influential=True),
        ]
        sobol_result = {
            "S1": np.array([[0.01, 0.02]]),  # (n_obj=2, n_params=1) — obj1 flat-ish
            "ST": np.array([[0.05, 0.3]]),
            "param_names": ["a"],
            "objective_names": ["obj_flat", "obj_responsive"],
            "n_objectives": 2,
        }
        issues = detect_issues(screening, sobol_result=sobol_result)
        flat = [i for i in issues if i.category == IssueCategory.FLAT_OBJECTIVE]
        assert len(flat) == 1
        assert "obj_flat" in flat[0].message

    def test_bound_tight_detected(self) -> None:
        # ST > 0.3 AND sigma/mu_star > 1.5
        screening = [
            ParameterScreening("a", mu_star=0.5, sigma=1.0, mu_star_conf=0.05, influential=True),
        ]
        sobol_result = {
            "S1": np.array([0.4]),
            "ST": np.array([0.5]),  # > 0.3
            "param_names": ["a"],
        }
        issues = detect_issues(screening, sobol_result=sobol_result)
        tight = [i for i in issues if i.category == IssueCategory.BOUND_TIGHT]
        assert len(tight) == 1
        assert tight[0].param_key == "a"
        assert tight[0].auto_fixable is True

    def test_no_issues_when_clean(self) -> None:
        screening = [
            ParameterScreening("a", mu_star=1.0, sigma=0.1, mu_star_conf=0.05, influential=True),
        ]
        sobol_result = {
            "S1": np.array([0.8]),
            "ST": np.array([0.9]),
            "param_names": ["a"],
        }
        issues = detect_issues(screening, sobol_result=sobol_result)
        assert issues == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py::TestIssueDetection -v`
Expected: FAIL — `cannot import name 'detect_issues'`

- [ ] **Step 3: Implement issue detection**

Add to `osmose/calibration/preflight.py`:

```python
def detect_issues(
    screening: list[ParameterScreening],
    sobol_result: dict | None,
    blowup_params: list[str] | None = None,
) -> list[PreflightIssue]:
    """Diagnose issues from screening and Sobol results.

    Args:
        screening: Morris screening results.
        sobol_result: SensitivityAnalyzer output dict, or None if Sobol was skipped.
        blowup_params: Parameter keys where evaluations returned inf/NaN.

    Returns:
        List of PreflightIssue objects.
    """
    issues: list[PreflightIssue] = []

    # Check for blowups
    for key in blowup_params or []:
        issues.append(
            PreflightIssue(
                category=IssueCategory.BLOWUP,
                severity=IssueSeverity.ERROR,
                param_key=key,
                message=f"Objective returned inf/NaN when perturbing '{key}'",
                suggestion=f"Tighten bounds on '{key}'",
                auto_fixable=True,
            )
        )

    # Check for negligible parameters
    negligible_keys = [s.key for s in screening if not s.influential]
    for s in screening:
        if not s.influential:
            issues.append(
                PreflightIssue(
                    category=IssueCategory.NEGLIGIBLE,
                    severity=IssueSeverity.WARNING,
                    param_key=s.key,
                    message=f"'{s.key}' has negligible influence (mu*={s.mu_star:.4f})",
                    suggestion=f"Remove '{s.key}' from calibration",
                    auto_fixable=True,
                )
            )

    # Check if ALL negligible
    if len(negligible_keys) == len(screening) and len(screening) > 0:
        issues.append(
            PreflightIssue(
                category=IssueCategory.ALL_NEGLIGIBLE,
                severity=IssueSeverity.ERROR,
                param_key=None,
                message="All parameters have negligible influence on objectives",
                suggestion="Review parameter selection and bounds",
                auto_fixable=False,
            )
        )

    if sobol_result is None:
        return issues

    param_names = sobol_result["param_names"]
    S1 = sobol_result["S1"]
    ST = sobol_result["ST"]
    screening_by_key = {s.key: s for s in screening}

    # Handle 1D (single objective) vs 2D (multi-objective) Sobol results
    if S1.ndim == 1:
        # Single objective: check flat
        total_s1 = float(np.sum(np.maximum(S1, 0)))
        if total_s1 < 0.05:
            obj_name = sobol_result.get("objective_names", ["objective"])[0] if "objective_names" in sobol_result else "objective"
            issues.append(
                PreflightIssue(
                    category=IssueCategory.FLAT_OBJECTIVE,
                    severity=IssueSeverity.WARNING,
                    param_key=None,
                    message=f"Objective '{obj_name}' is unresponsive (total S1={total_s1:.3f})",
                    suggestion="Check objective function or target data",
                    auto_fixable=False,
                )
            )
        # Check bound_tight per param
        for j, name in enumerate(param_names):
            s = screening_by_key.get(name)
            if s and s.mu_star > 0 and ST[j] > 0.3 and (s.sigma / s.mu_star) > 1.5:
                issues.append(
                    PreflightIssue(
                        category=IssueCategory.BOUND_TIGHT,
                        severity=IssueSeverity.WARNING,
                        param_key=name,
                        message=f"'{name}' may have overly tight bounds (ST={ST[j]:.2f}, sigma/mu*={s.sigma / s.mu_star:.1f})",
                        suggestion=f"Widen bounds on '{name}' by 20%",
                        auto_fixable=True,
                    )
                )
    else:
        # Multi-objective: S1 has shape (n_obj, n_params)
        n_obj = S1.shape[0]
        obj_names = sobol_result.get("objective_names", [f"obj_{i}" for i in range(n_obj)])
        for col in range(n_obj):
            total_s1 = float(np.sum(np.maximum(S1[col], 0)))
            if total_s1 < 0.05:
                issues.append(
                    PreflightIssue(
                        category=IssueCategory.FLAT_OBJECTIVE,
                        severity=IssueSeverity.WARNING,
                        param_key=None,
                        message=f"Objective '{obj_names[col]}' is unresponsive (total S1={total_s1:.3f})",
                        suggestion="Check objective function or target data",
                        auto_fixable=False,
                    )
                )
        # bound_tight: use max ST across objectives
        max_ST = np.max(ST, axis=0)
        for j, name in enumerate(param_names):
            s = screening_by_key.get(name)
            if s and s.mu_star > 0 and max_ST[j] > 0.3 and (s.sigma / s.mu_star) > 1.5:
                issues.append(
                    PreflightIssue(
                        category=IssueCategory.BOUND_TIGHT,
                        severity=IssueSeverity.WARNING,
                        param_key=name,
                        message=f"'{name}' may have overly tight bounds (ST={max_ST[j]:.2f}, sigma/mu*={s.sigma / s.mu_star:.1f})",
                        suggestion=f"Widen bounds on '{name}' by 20%",
                        auto_fixable=True,
                    )
                )

    return issues
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py::TestIssueDetection -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/calibration/preflight.py tests/test_calibration_preflight.py
git commit -m "feat(calibration): add preflight issue detection logic"
```

---

### Task 4: `run_preflight()` Orchestrator

**Files:**
- Modify: `osmose/calibration/preflight.py`
- Test: `tests/test_calibration_preflight.py`

- [ ] **Step 1: Write failing tests for orchestrator**

Append to `tests/test_calibration_preflight.py`:

```python
import threading
from osmose.calibration.preflight import run_preflight, PreflightResult, IssueCategory


class TestRunPreflight:
    """Orchestrator end-to-end with synthetic eval functions."""

    def test_happy_path(self) -> None:
        """Clear signal → survivors, Sobol populated, minimal issues."""
        names = ["a", "b", "c"]
        bounds = [(0, 1), (0, 1), (0, 1)]

        def eval_fn(x: np.ndarray) -> np.ndarray:
            return np.array([3 * x[0] + 0 * x[1] + 0.5 * x[2]])

        result = run_preflight(names, bounds, eval_fn, morris_n=10, sobol_n_base=64)

        assert isinstance(result, PreflightResult)
        assert "a" in result.survivors
        assert "c" in result.survivors
        assert "b" not in result.survivors
        assert result.sobol is not None
        assert result.elapsed_seconds > 0

    def test_failure_abort(self) -> None:
        """>30% Morris evals return inf → early abort, no Sobol."""
        names = ["a", "b"]
        bounds = [(0, 1), (0, 1)]
        call_count = [0]

        def eval_fn(x: np.ndarray) -> np.ndarray:
            call_count[0] += 1
            if call_count[0] % 2 == 0:  # 50% failure rate
                return np.array([float("inf")])
            return np.array([x[0] + x[1]])

        result = run_preflight(names, bounds, eval_fn, morris_n=10, sobol_n_base=64)

        # Should have aborted — blowup issues present
        blowup_or_error = [
            i for i in result.issues
            if i.category == IssueCategory.BLOWUP or i.severity.value == "error"
        ]
        assert len(blowup_or_error) > 0
        # Sobol may or may not have run depending on which evals failed

    def test_single_parameter(self) -> None:
        """Morris with k=1 still works."""
        names = ["a"]
        bounds = [(0, 1)]

        def eval_fn(x: np.ndarray) -> np.ndarray:
            return np.array([x[0] ** 2])

        result = run_preflight(names, bounds, eval_fn, morris_n=10, sobol_n_base=64)
        assert isinstance(result, PreflightResult)
        assert len(result.screening) == 1

    def test_cancellation(self) -> None:
        """Cancel flag stops execution cleanly."""
        names = ["a", "b"]
        bounds = [(0, 1), (0, 1)]
        cancel = threading.Event()
        cancel.set()  # Cancel immediately

        def eval_fn(x: np.ndarray) -> np.ndarray:
            return np.array([x[0]])

        result = run_preflight(
            names, bounds, eval_fn, morris_n=10, sobol_n_base=64, cancel_event=cancel
        )
        # Should return quickly with partial/empty result
        assert isinstance(result, PreflightResult)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py::TestRunPreflight -v`
Expected: FAIL — `cannot import name 'run_preflight'`

- [ ] **Step 3: Implement orchestrator**

Add to `osmose/calibration/preflight.py`:

```python
import time
import threading

from osmose.calibration.sensitivity import SensitivityAnalyzer


def run_preflight(
    param_names: list[str],
    param_bounds: list[tuple[float, float]],
    eval_fn: Callable[[np.ndarray], np.ndarray],
    objective_names: list[str] | None = None,
    morris_n: int = 10,
    sobol_n_base: int = 64,
    negligible_threshold: float = 0.01,
    on_progress: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> PreflightResult:
    """Run two-stage pre-flight: Morris screening + targeted Sobol.

    Args:
        param_names: Parameter names (matching FreeParameter.key).
        param_bounds: (lower, upper) for each parameter.
        eval_fn: Maps parameter vector (1D array) → objective values (1D array).
        objective_names: Labels for objectives.
        morris_n: Number of Morris trajectories.
        sobol_n_base: Base sample size for Sobol on survivors.
        negligible_threshold: Fraction of max(mu_star) for negligible cutoff.
        on_progress: Optional progress callback.
        cancel_event: Optional threading.Event for cancellation.

    Returns:
        PreflightResult with screening, Sobol results, and issues.
    """
    t0 = time.monotonic()

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    if _cancelled():
        return PreflightResult([], None, [], [], time.monotonic() - t0)

    # --- Stage 1: Morris screening ---
    if on_progress:
        on_progress("Pre-flight: Screening parameters (1/2)...")

    problem = {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": param_bounds,
    }

    X_morris = morris_sample.sample(problem, morris_n)
    n_samples = X_morris.shape[0]

    first_y = eval_fn(X_morris[0])
    n_obj = len(first_y)
    Y = np.zeros((n_samples, n_obj))
    Y[0] = first_y
    n_failures = int(np.any(~np.isfinite(first_y)))
    blowup_params: list[str] = []

    k = len(param_names)
    traj_size = k + 1

    for i in range(1, n_samples):
        if _cancelled():
            return PreflightResult([], None, [], [], time.monotonic() - t0)
        Y[i] = eval_fn(X_morris[i])
        if np.any(~np.isfinite(Y[i])):
            n_failures += 1
            # Identify which param was perturbed in this trajectory step
            traj_idx = i // traj_size
            step_in_traj = i % traj_size
            if step_in_traj > 0:
                prev = X_morris[i - 1]
                diff = np.abs(X_morris[i] - prev)
                changed = int(np.argmax(diff))
                if param_names[changed] not in blowup_params:
                    blowup_params.append(param_names[changed])

    # Check failure threshold (30%)
    failure_rate = n_failures / n_samples if n_samples > 0 else 0
    if failure_rate > 0.3:
        issues = detect_issues([], None, blowup_params=blowup_params)
        issues.append(
            PreflightIssue(
                category=IssueCategory.BLOWUP,
                severity=IssueSeverity.ERROR,
                param_key=None,
                message=f"Too many evaluations failed ({n_failures}/{n_samples}, {failure_rate:.0%})",
                suggestion="Check configuration and parameter bounds",
                auto_fixable=False,
            )
        )
        return PreflightResult([], None, issues, [], time.monotonic() - t0)

    # Morris analysis per objective
    mu_star_all = np.zeros((n_obj, k))
    sigma_all = np.zeros((n_obj, k))
    conf_all = np.zeros((n_obj, k))

    # Replace inf/NaN with large penalty for Morris analysis
    Y_clean = np.copy(Y)
    Y_clean[~np.isfinite(Y_clean)] = 1e6

    for col in range(n_obj):
        result = morris_analyze.analyze(problem, X_morris, Y_clean[:, col])
        mu_star_all[col] = result["mu_star"]
        sigma_all[col] = result["sigma"]
        conf_all[col] = result["mu_star_conf"]

    agg_mu_star = np.max(mu_star_all, axis=0)
    agg_sigma = np.max(sigma_all, axis=0)
    agg_conf = np.max(conf_all, axis=0)

    max_mu = np.max(agg_mu_star) if np.max(agg_mu_star) > 0 else 1.0
    threshold = negligible_threshold * max_mu

    screening = []
    for j in range(k):
        influential = (agg_mu_star[j] + agg_conf[j]) >= threshold
        screening.append(
            ParameterScreening(
                key=param_names[j],
                mu_star=float(agg_mu_star[j]),
                sigma=float(agg_sigma[j]),
                mu_star_conf=float(agg_conf[j]),
                influential=influential,
            )
        )

    survivors = [s.key for s in screening if s.influential]

    # Check all negligible
    if not survivors:
        issues = detect_issues(screening, None, blowup_params=blowup_params)
        return PreflightResult(screening, None, issues, [], time.monotonic() - t0)

    if _cancelled():
        issues = detect_issues(screening, None, blowup_params=blowup_params)
        return PreflightResult(screening, None, issues, survivors, time.monotonic() - t0)

    # --- Stage 2: Targeted Sobol on survivors ---
    if on_progress:
        on_progress("Pre-flight: Validating bounds (2/2)...")

    survivor_indices = [param_names.index(s) for s in survivors]
    survivor_bounds = [param_bounds[i] for i in survivor_indices]
    analyzer = SensitivityAnalyzer(survivors, survivor_bounds)
    sobol_X = analyzer.generate_samples(n_base=sobol_n_base)

    n_sobol = sobol_X.shape[0]
    Y_sobol = np.zeros((n_sobol, n_obj))
    n_sobol_failures = 0

    for i in range(n_sobol):
        if _cancelled():
            issues = detect_issues(screening, None, blowup_params=blowup_params)
            return PreflightResult(screening, None, issues, survivors, time.monotonic() - t0)

        # Map survivor params back to full param vector (non-survivors at midpoint)
        full_x = np.array([(b[0] + b[1]) / 2 for b in param_bounds])
        for si, idx in enumerate(survivor_indices):
            full_x[idx] = sobol_X[i, si]
        Y_sobol[i] = eval_fn(full_x)
        if np.any(~np.isfinite(Y_sobol[i])):
            n_sobol_failures += 1

    # Check Sobol failure threshold (10%)
    sobol_failure_rate = n_sobol_failures / n_sobol if n_sobol > 0 else 0
    if sobol_failure_rate > 0.1:
        issues = detect_issues(screening, None, blowup_params=blowup_params)
        issues.append(
            PreflightIssue(
                category=IssueCategory.BLOWUP,
                severity=IssueSeverity.ERROR,
                param_key=None,
                message=f"Too many Sobol evaluations failed ({n_sobol_failures}/{n_sobol}, {sobol_failure_rate:.0%})",
                suggestion="Check configuration and parameter bounds",
                auto_fixable=False,
            )
        )
        return PreflightResult(screening, None, issues, survivors, time.monotonic() - t0)

    # Replace inf/NaN for Sobol analysis
    Y_sobol_clean = np.copy(Y_sobol)
    Y_sobol_clean[~np.isfinite(Y_sobol_clean)] = 1e6

    if n_obj == 1:
        sobol_result = analyzer.analyze(Y_sobol_clean[:, 0])
    else:
        sobol_result = analyzer.analyze(Y_sobol_clean, objective_names=objective_names)

    issues = detect_issues(screening, sobol_result, blowup_params=blowup_params)

    return PreflightResult(
        screening=screening,
        sobol=sobol_result,
        issues=issues,
        survivors=survivors,
        elapsed_seconds=time.monotonic() - t0,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py::TestRunPreflight -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Run full test file**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py -v`
Expected: All 15 tests PASS

- [ ] **Step 6: Commit**

```bash
git add osmose/calibration/preflight.py tests/test_calibration_preflight.py
git commit -m "feat(calibration): add run_preflight() two-stage orchestrator"
```

---

### Task 5: Evaluation Function Factory

**Files:**
- Modify: `osmose/calibration/preflight.py`
- Test: `tests/test_calibration_preflight.py`

- [ ] **Step 1: Write failing tests for eval factory**

Append to `tests/test_calibration_preflight.py`:

```python
from unittest.mock import MagicMock, patch
from pathlib import Path
from osmose.calibration.preflight import make_preflight_eval_fn
from osmose.calibration.problem import FreeParameter, Transform


class TestMakePreflightEvalFn:
    """Evaluation function factory."""

    def test_sim_years_clamped_to_5(self) -> None:
        """Configured 30yr → factory uses 5yr."""
        config = {"simulation.time.nyear": "30", "_osmose.config.dir": "/tmp"}
        free_params = [FreeParameter("species.k.sp0", 0.1, 1.0)]

        with patch("osmose.calibration.preflight.PythonEngine") as MockEngine:
            mock_instance = MagicMock()
            mock_result = MagicMock(returncode=0)
            mock_instance.run.return_value = mock_result
            MockEngine.return_value = mock_instance

            with patch("osmose.calibration.preflight.OsmoseResults") as MockResults:
                mock_results_inst = MagicMock()
                MockResults.return_value = mock_results_inst
                mock_results_inst.close = MagicMock()

                eval_fn = make_preflight_eval_fn(
                    config, free_params, [lambda r: 1.0], Path("/tmp/work")
                )
                eval_fn(np.array([0.5]))

                # Check the config passed to engine.run had nyear=5
                call_args = mock_instance.run.call_args
                cfg_used = call_args[0][0]
                assert cfg_used["simulation.time.nyear"] == "5"

    def test_sim_years_keeps_short(self) -> None:
        """Configured 3yr → factory uses 3yr (no clamping up)."""
        config = {"simulation.time.nyear": "3", "_osmose.config.dir": "/tmp"}
        free_params = [FreeParameter("species.k.sp0", 0.1, 1.0)]

        with patch("osmose.calibration.preflight.PythonEngine") as MockEngine:
            mock_instance = MagicMock()
            mock_result = MagicMock(returncode=0)
            mock_instance.run.return_value = mock_result
            MockEngine.return_value = mock_instance

            with patch("osmose.calibration.preflight.OsmoseResults") as MockResults:
                mock_results_inst = MagicMock()
                MockResults.return_value = mock_results_inst
                mock_results_inst.close = MagicMock()

                eval_fn = make_preflight_eval_fn(
                    config, free_params, [lambda r: 1.0], Path("/tmp/work")
                )
                eval_fn(np.array([0.5]))

                cfg_used = mock_instance.run.call_args[0][0]
                assert cfg_used["simulation.time.nyear"] == "3"

    def test_log_transform_applied(self) -> None:
        """LOG transform: config value = 10^x."""
        config = {"simulation.time.nyear": "5", "_osmose.config.dir": "/tmp"}
        free_params = [FreeParameter("species.k.sp0", -2.0, 0.0, transform=Transform.LOG)]

        with patch("osmose.calibration.preflight.PythonEngine") as MockEngine:
            mock_instance = MagicMock()
            mock_result = MagicMock(returncode=0)
            mock_instance.run.return_value = mock_result
            MockEngine.return_value = mock_instance

            with patch("osmose.calibration.preflight.OsmoseResults") as MockResults:
                mock_results_inst = MagicMock()
                MockResults.return_value = mock_results_inst
                mock_results_inst.close = MagicMock()

                eval_fn = make_preflight_eval_fn(
                    config, free_params, [lambda r: 1.0], Path("/tmp/work")
                )
                eval_fn(np.array([-1.0]))  # 10^(-1) = 0.1

                cfg_used = mock_instance.run.call_args[0][0]
                assert float(cfg_used["species.k.sp0"]) == pytest.approx(0.1, rel=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py::TestMakePreflightEvalFn -v`
Expected: FAIL — `cannot import name 'make_preflight_eval_fn'`

- [ ] **Step 3: Implement eval factory**

Add to `osmose/calibration/preflight.py`:

```python
from pathlib import Path

from osmose.calibration.problem import FreeParameter, Transform


def make_preflight_eval_fn(
    config: dict[str, str],
    free_params: list[FreeParameter],
    objective_fns: list[Callable],
    work_dir: Path,
    sim_years: int | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create an evaluation function for pre-flight screening.

    Args:
        config: Ready-to-use OSMOSE config dict (with _osmose.config.dir set).
        free_params: Parameters being calibrated.
        objective_fns: Objective functions (each takes OsmoseResults, returns float).
        work_dir: Directory for temporary engine output.
        sim_years: Override simulation years. Defaults to min(5, configured).

    Returns:
        Function mapping parameter vector (1D array) → objective values (1D array).
    """
    from osmose.engine import PythonEngine
    from osmose.results import OsmoseResults

    cfg = dict(config)
    configured_years = int(cfg.get("simulation.time.nyear", "10"))
    effective_years = sim_years if sim_years is not None else min(5, configured_years)
    cfg["simulation.time.nyear"] = str(effective_years)

    output_dir = work_dir / "preflight_eval"

    def eval_fn(x: np.ndarray) -> np.ndarray:
        run_cfg = dict(cfg)
        for j, fp in enumerate(free_params):
            val = x[j]
            if fp.transform == Transform.LOG:
                val = 10**val
            run_cfg[fp.key] = str(val)

        try:
            engine = PythonEngine()
            engine.run(run_cfg, output_dir=output_dir, seed=0)
            results = OsmoseResults(output_dir, strict=False)
            obj_values = np.array([fn(results) for fn in objective_fns])
            results.close()
            return obj_values
        except Exception:
            return np.full(len(objective_fns), float("inf"))

    return eval_fn
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py::TestMakePreflightEvalFn -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/calibration/preflight.py tests/test_calibration_preflight.py
git commit -m "feat(calibration): add make_preflight_eval_fn() factory"
```

---

### Task 6: Export Public API

**Files:**
- Modify: `osmose/calibration/__init__.py`

- [ ] **Step 1: Update `__init__.py` exports**

Add the new symbols to `osmose/calibration/__init__.py`. Add these imports after the existing `SensitivityAnalyzer` import:

```python
from osmose.calibration.preflight import (
    IssueCategory,
    IssueSeverity,
    ParameterScreening,
    PreflightIssue,
    PreflightResult,
    run_preflight,
    make_preflight_eval_fn,
)
```

Add these names to the `__all__` list:

```python
    "IssueCategory",
    "IssueSeverity",
    "ParameterScreening",
    "PreflightIssue",
    "PreflightResult",
    "run_preflight",
    "make_preflight_eval_fn",
```

- [ ] **Step 2: Verify imports work**

Run: `.venv/bin/python -c "from osmose.calibration import run_preflight, PreflightResult, IssueCategory; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run full library test suite**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py -v`
Expected: All 18 tests PASS

- [ ] **Step 4: Commit**

```bash
git add osmose/calibration/__init__.py
git commit -m "feat(calibration): export preflight API from __init__"
```

---

### Task 7: UI — Checkbox and Message Queue

**Files:**
- Modify: `ui/pages/calibration.py:100-108` — add checkbox before Start button
- Modify: `ui/pages/calibration_handlers.py:24-58` — add `post_preflight()` to message queue
- Modify: `ui/pages/calibration.py:184-217` — add `preflight_result` reactive value

- [ ] **Step 1: Add `post_preflight()` to `CalibrationMessageQueue`**

In `ui/pages/calibration_handlers.py`, add after the `post_history_saved` method (line 49):

```python
    def post_preflight(self, result) -> None:
        self._q.put(("preflight", result))
```

- [ ] **Step 2: Add checkbox to calibration UI**

In `ui/pages/calibration.py`, insert before the `ui.layout_columns` containing the Start/Stop buttons (before line 102):

```python
                ui.input_checkbox(
                    "cal_preflight_enabled",
                    "Pre-flight screening (recommended)",
                    value=True,
                ),
```

- [ ] **Step 3: Add `preflight_result` reactive value to `calibration_server`**

In `ui/pages/calibration.py`, in `calibration_server()`, add after line 188 (`cal_param_names = reactive.value([])`):

```python
    preflight_result = reactive.value(None)
```

Pass it to `register_calibration_handlers`:

```python
    register_calibration_handlers(
        ...
        preflight_result=preflight_result,
    )
```

- [ ] **Step 4: Update `register_calibration_handlers` signature**

In `ui/pages/calibration_handlers.py`, add `preflight_result` to the function signature (after `cal_param_names`):

```python
def register_calibration_handlers(
    ...
    cal_param_names,
    preflight_result,
    history_banner_text,
    history_trigger,
):
```

- [ ] **Step 5: Add `"preflight"` handler to the message poll**

In `ui/pages/calibration_handlers.py`, in `_poll_cal_messages()`, add after the `"validation"` handler:

```python
            elif kind == "preflight":
                preflight_result.set(payload)
```

- [ ] **Step 6: Verify app starts**

Run: `.venv/bin/python -c "from ui.pages.calibration import calibration_ui; print('OK')"`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add ui/pages/calibration.py ui/pages/calibration_handlers.py
git commit -m "feat(ui): add preflight checkbox, message queue type, and reactive value"
```

---

### Task 8: UI — Pre-flight Thread and Modal

**Files:**
- Modify: `ui/pages/calibration_handlers.py` — pre-flight thread in `handle_start_cal`, modal builder, fix-application handler
- Test: `tests/test_ui_calibration_preflight.py`

- [ ] **Step 1: Write failing UI tests**

Create `tests/test_ui_calibration_preflight.py`:

```python
"""Tests for pre-flight UI integration."""

from __future__ import annotations

from shiny import reactive

from osmose.calibration.preflight import (
    IssueCategory,
    IssueSeverity,
    ParameterScreening,
    PreflightIssue,
    PreflightResult,
)
from ui.pages.calibration_handlers import build_preflight_modal, apply_preflight_fixes
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
                    IssueCategory.NEGLIGIBLE, IssueSeverity.WARNING,
                    "b", "Negligible", "Remove b", auto_fixable=True,
                ),
            ],
            survivors=["a"],
            elapsed_seconds=10.0,
        )
        modal = build_preflight_modal(result)
        html = str(modal)
        assert "Negligible" in html
        assert "preflight_fix_0" in html  # Checkbox ID

    def test_non_fixable_shows_as_text(self) -> None:
        result = PreflightResult(
            screening=[ParameterScreening("a", 1.0, 0.1, 0.05, True)],
            sobol=None,
            issues=[
                PreflightIssue(
                    IssueCategory.FLAT_OBJECTIVE, IssueSeverity.WARNING,
                    None, "Objective flat", "Check data", auto_fixable=False,
                ),
            ],
            survivors=["a"],
            elapsed_seconds=5.0,
        )
        modal = build_preflight_modal(result)
        html = str(modal)
        assert "Objective flat" in html
        # No checkbox for non-fixable
        assert "preflight_fix_0" not in html

    def test_no_issues_returns_none(self) -> None:
        result = PreflightResult(
            screening=[ParameterScreening("a", 1.0, 0.1, 0.05, True)],
            sobol=None,
            issues=[],
            survivors=["a"],
            elapsed_seconds=5.0,
        )
        modal = build_preflight_modal(result)
        assert modal is None


class TestApplyPreflightFixes:
    def test_negligible_removed(self) -> None:
        free_params = [
            FreeParameter("a", 0.1, 1.0),
            FreeParameter("b", 0.1, 1.0),
        ]
        issues = [
            PreflightIssue(
                IssueCategory.NEGLIGIBLE, IssueSeverity.WARNING,
                "b", "Negligible", "Remove b", auto_fixable=True,
            ),
        ]
        checked = [True]
        updated = apply_preflight_fixes(free_params, issues, checked)
        assert len(updated) == 1
        assert updated[0].key == "a"

    def test_bound_adjusted(self) -> None:
        free_params = [FreeParameter("a", 0.1, 1.0)]
        issues = [
            PreflightIssue(
                IssueCategory.BOUND_TIGHT, IssueSeverity.WARNING,
                "a", "Tight bounds", "Widen by 20%", auto_fixable=True,
            ),
        ]
        checked = [True]
        updated = apply_preflight_fixes(free_params, issues, checked)
        assert len(updated) == 1
        # Bounds widened by 20%: lower * 0.8, upper * 1.2
        assert updated[0].lower_bound < 0.1
        assert updated[0].upper_bound > 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ui_calibration_preflight.py -v`
Expected: FAIL — `cannot import name 'build_preflight_modal'`

- [ ] **Step 3: Implement modal builder and fix-application**

Add to `ui/pages/calibration_handlers.py` (after the existing helper functions, before `register_calibration_handlers`):

```python
from osmose.calibration.preflight import (
    PreflightResult,
    PreflightIssue,
    IssueCategory,
)


def build_preflight_modal(result: PreflightResult):
    """Build a Shiny modal for pre-flight results. Returns None if no issues."""
    if not result.issues:
        return None

    n_influential = sum(1 for s in result.screening if s.influential)
    n_negligible = len(result.screening) - n_influential

    header = ui.div(
        ui.h5("Pre-flight Screening Results"),
        ui.p(
            f"Screened {len(result.screening)} parameters in "
            f"{result.elapsed_seconds:.0f}s. "
            f"{n_influential} influential, {n_negligible} negligible."
        ),
    )

    issue_rows = []
    fixable_idx = 0
    for issue in result.issues:
        if issue.auto_fixable:
            issue_rows.append(
                ui.div(
                    ui.input_checkbox(
                        f"preflight_fix_{fixable_idx}",
                        f"{issue.suggestion}",
                        value=True,
                    ),
                    ui.p(f"({issue.message})", style="margin-left: 24px; opacity: 0.7; margin-top: -8px;"),
                    class_="mb-2",
                )
            )
            fixable_idx += 1
        else:
            icon = "⚠" if issue.severity.value == "warning" else "✕"
            issue_rows.append(
                ui.div(
                    ui.p(f"{icon} {issue.message}"),
                    ui.p(f"Suggestion: {issue.suggestion}", style="opacity: 0.7; margin-left: 16px;"),
                    class_="mb-2",
                )
            )

    return ui.modal(
        header,
        ui.h6(f"Issues ({len(result.issues)}):"),
        *issue_rows,
        title="Pre-flight Screening",
        easy_close=False,
        footer=ui.div(
            ui.input_action_button(
                "btn_preflight_apply", "Apply Selected & Start", class_="btn-success"
            ),
            ui.tags.button(
                "Cancel",
                class_="btn btn-secondary ms-2",
                **{"data-bs-dismiss": "modal"},
            ),
        ),
    )


def apply_preflight_fixes(
    free_params: list,
    issues: list[PreflightIssue],
    checked: list[bool],
) -> list:
    """Apply user-selected fixes and return updated free_params.

    Args:
        free_params: Original FreeParameter list.
        issues: Only the auto_fixable issues (in order matching checked).
        checked: Whether each fixable issue's checkbox was checked.

    Returns:
        Updated FreeParameter list.
    """
    from osmose.calibration.problem import FreeParameter

    remove_keys: set[str] = set()
    bound_adjustments: dict[str, tuple[float, float]] = {}

    for issue, is_checked in zip(issues, checked):
        if not is_checked:
            continue
        if issue.category == IssueCategory.NEGLIGIBLE and issue.param_key:
            remove_keys.add(issue.param_key)
        elif issue.category == IssueCategory.BOUND_TIGHT and issue.param_key:
            # Find current bounds and widen by 20%
            for fp in free_params:
                if fp.key == issue.param_key:
                    span = fp.upper_bound - fp.lower_bound
                    new_lower = fp.lower_bound - 0.1 * span
                    new_upper = fp.upper_bound + 0.1 * span
                    bound_adjustments[fp.key] = (new_lower, new_upper)
                    break
        elif issue.category == IssueCategory.BLOWUP and issue.param_key:
            # Tighten by 20%
            for fp in free_params:
                if fp.key == issue.param_key:
                    span = fp.upper_bound - fp.lower_bound
                    new_lower = fp.lower_bound + 0.1 * span
                    new_upper = fp.upper_bound - 0.1 * span
                    if new_lower < new_upper:
                        bound_adjustments[fp.key] = (new_lower, new_upper)
                    break

    updated = []
    for fp in free_params:
        if fp.key in remove_keys:
            continue
        if fp.key in bound_adjustments:
            lo, hi = bound_adjustments[fp.key]
            updated.append(FreeParameter(fp.key, lo, hi, fp.transform))
        else:
            updated.append(fp)

    return updated
```

- [ ] **Step 4: Run UI tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ui_calibration_preflight.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ui/pages/calibration_handlers.py tests/test_ui_calibration_preflight.py
git commit -m "feat(ui): add preflight modal builder and fix-application logic"
```

---

### Task 9: UI — Wire Pre-flight into Start Calibration Handler

**Files:**
- Modify: `ui/pages/calibration_handlers.py` — modify `handle_start_cal` and add `handle_preflight_apply`

This is the integration task. The existing `handle_start_cal` in `calibration_handlers.py` (starting at line 229) needs to check the checkbox and either run the pre-flight thread or go directly to optimization.

- [ ] **Step 1: Refactor `handle_start_cal` — extract optimization into helpers**

In `register_calibration_handlers`, extract the optimization logic (everything after parameter/objective validation, starting from line 340 `cal_param_names.set(...)`) into two new inner functions:

1. `_start_optimization(selected, free_params, objective_fns, obj_names, banded_enabled, obs_bio, obs_diet)` — the full path including config writing, problem creation, and thread launch. Called when pre-flight is skipped.

2. `_start_optimization_with_params(updated_params)` — a lighter version that uses pre-captured objective_fns/config from the pre-flight invocation (stored in a closure or nonlocal variable) and accepts already-fixed-up `free_params`. Called from `handle_preflight_apply`.

Both share the same core logic (NSGA-II/surrogate thread). The refactoring makes `free_params` a parameter rather than computed inside, so the pre-flight can pass fixed-up params.

**Note:** `_start_optimization_from_preflight(free_params, payload)` referenced in Step 3 below is the same as `_start_optimization_with_params` — it starts calibration with the original params when pre-flight found no issues. Store `objective_fns`, `obj_names`, and config state in nonlocal variables during the pre-flight branch so these helpers can access them.

- [ ] **Step 2: Add pre-flight branch to `handle_start_cal`**

After the existing validation checks in `handle_start_cal` (after line 339 where `objective_fns` is validated), add:

```python
        # Check pre-flight checkbox
        preflight_enabled = False
        try:
            preflight_enabled = bool(input.cal_preflight_enabled())
        except (SilentException, AttributeError):
            pass

        if not preflight_enabled:
            _start_optimization(selected, free_params, objective_fns, obj_names, banded_enabled, obs_bio, obs_diet)
            return

        # Run pre-flight in background thread
        preflight_config = dict(current_config)
        preflight_work = Path(tempfile.mkdtemp(prefix="osmose_preflight_"))

        from osmose.calibration.preflight import run_preflight, make_preflight_eval_fn

        preflight_eval = make_preflight_eval_fn(
            preflight_config, free_params, objective_fns, preflight_work
        )

        cancel_event.clear()

        def run_preflight_thread():
            try:
                param_names = [fp.key for fp in free_params]
                param_bounds = [(fp.lower_bound, fp.upper_bound) for fp in free_params]
                result = run_preflight(
                    param_names,
                    param_bounds,
                    preflight_eval,
                    objective_names=obj_names,
                    on_progress=msg_queue.post_status,
                    cancel_event=cancel_event,
                )
                msg_queue.post_preflight(result)
            except Exception as exc:
                _log.error("Pre-flight failed: %s", exc, exc_info=True)
                msg_queue.post_error(f"Pre-flight failed: {exc}")

        thread = threading.Thread(target=run_preflight_thread, daemon=True)
        thread.start()
        cal_thread.set(thread)
```

- [ ] **Step 3: Handle pre-flight result in message poll**

Update the `"preflight"` handler in `_poll_cal_messages` to show the modal or start calibration:

```python
            elif kind == "preflight":
                preflight_result.set(payload)
                modal = build_preflight_modal(payload)
                if modal is None:
                    # No issues — start calibration immediately
                    surrogate_status.set("Pre-flight passed — starting calibration...")
                    # Re-read current state and start optimization
                    selected = collect_selected_params(input, state)
                    free_params = build_free_params(selected)
                    # Use the stored preflight result's survivors to filter
                    _start_optimization_from_preflight(free_params, payload)
                else:
                    ui.modal_show(modal)
```

- [ ] **Step 4: Add `handle_preflight_apply` handler**

Inside `register_calibration_handlers`, add a new reactive handler:

```python
    @reactive.effect
    @reactive.event(input.btn_preflight_apply)
    def handle_preflight_apply():
        result = preflight_result.get()
        if result is None:
            return
        ui.modal_remove()

        selected = collect_selected_params(input, state)
        free_params = build_free_params(selected)

        # Collect which fixable issues are checked
        fixable_issues = [i for i in result.issues if i.auto_fixable]
        checked = []
        for idx in range(len(fixable_issues)):
            try:
                checked.append(bool(getattr(input, f"preflight_fix_{idx}")()))
            except (SilentException, AttributeError):
                checked.append(False)

        updated_params = apply_preflight_fixes(free_params, fixable_issues, checked)

        if not updated_params:
            ui.notification_show(
                "All parameters removed by pre-flight fixes. Adjust selection.",
                type="warning",
                duration=5,
            )
            return

        # Store updated param names
        cal_param_names.set([fp.key.split(".")[-1] for fp in updated_params])

        # Start optimization with fixed-up params
        _start_optimization_with_params(updated_params)
```

- [ ] **Step 5: Test the full flow manually**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py tests/test_ui_calibration_preflight.py -v`
Expected: All tests PASS

- [ ] **Step 6: Run lint**

Run: `.venv/bin/ruff check osmose/calibration/preflight.py ui/pages/calibration.py ui/pages/calibration_handlers.py`
Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add ui/pages/calibration_handlers.py ui/pages/calibration.py
git commit -m "feat(ui): wire preflight into start calibration handler with modal flow"
```

---

### Task 10: Integration Test and Final Verification

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest -v`
Expected: All existing tests still pass + new preflight tests pass

- [ ] **Step 2: Run lint on all modified files**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`
Expected: No errors

- [ ] **Step 3: Run format check**

Run: `.venv/bin/ruff format --check osmose/ ui/ tests/`
Expected: No formatting changes needed

- [ ] **Step 4: Verify import from top-level**

Run: `.venv/bin/python -c "from osmose.calibration import run_preflight, PreflightResult, IssueCategory, make_preflight_eval_fn; print('All exports OK')"`
Expected: `All exports OK`

- [ ] **Step 5: Final commit if any fixups needed**

```bash
git add -A
git commit -m "fix: lint and formatting fixes for preflight feature"
```

(Skip if no changes needed.)
