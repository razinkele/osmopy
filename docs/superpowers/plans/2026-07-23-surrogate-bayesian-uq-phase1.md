# Surrogate-Bayesian UQ — Phase 1: Design + Calibration Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the machinery that answers "can we build a *calibrated* emulator?" — a seeded LHS design executed through the Python engine (SSB enabled) reduced to per-output `(X, Y=log-seed-mean, alpha=s²/S)`, a probabilistic coverage/PIT calibration gate, and a bounded design-growth loop — validated end-to-end on synthetic data with zero OSMOSE runs.

**Architecture:** Two new modules under the Phase 0 `osmose/calibration/uq/` subpackage. `gate.py` is pure numpy/scipy: it cross-validates a `GPEmulator` and standardizes held-out residuals to test coverage, mean-standardized-squared-residual (MSSR), and PIT uniformity. `design.py` runs the design through an **injectable evaluator** — a `(point, seed) -> stat-dict` callable that defaults to the real Python engine but is a synthetic function in tests — so the whole pipeline (design → gate → growth loop) is CI-testable and only one thin test actually runs the engine. Phase 0's `GPEmulator` gains a test-index return so the gate can align each fold's predictive variance with that point's seed-mean noise.

**Tech Stack:** Python 3.12+, NumPy, SciPy (`scipy.stats`, `scipy.stats.qmc.LatinHypercube`), scikit-learn (`KFold`) — all already core dependencies. Phase 0's `GPEmulator`, `compute_uq_stats`, `target_to_output_key`. pytest.

## Global Constraints

- **Python 3.12+**; type hints on all public APIs. **Ruff line length 100** — run `.venv/bin/ruff format` then `.venv/bin/ruff check` on new/changed files before each commit and fix findings.
- **Test runner is `.venv/bin/python -m pytest`** (system `python` may not exist).
- **No new dependencies.** The gate is pure numpy/scipy; `emcee`/`dynesty`/`arviz` are Phase 2. Do not add a `[uq]` extra in this phase.
- **`osmose/calibration/uq/__init__.py` stays thin** — no re-exports, no eager heavy imports; not added to `osmose/calibration/__init__.py`. Tests/callers import submodules directly. `PythonEngine`/`OsmoseConfigReader` are **lazy-imported inside the evaluator factory**, never at `design.py` module top.
- **Two distinct transforms, never conflated:**
  - **Simulator-input boundary:** base-10 `10**val` for `Transform.LOG` params (mirrors `osmose/calibration/problem.py:263`). Lives in `point_to_overrides`.
  - **GP training target:** **natural** `np.log` of the linear-scale stat. Lives in `run_design`'s reduction.
- **Two distinct `ddof`, never conflated:** `alpha_i = np.var(log-values-over-seeds, ddof=1) / S` — **ddof=1** (unbiased per-run variance estimate; this is a *different* quantity from Phase 0's emulator `np.var(Y, ddof=0)`, which matched sklearn's `normalize_y` standardization). A comment must state this so nobody "fixes" one to match the other.
- **`GPEmulator.predict()` returns LATENT variance;** the gate adds the held-out seed-mean noise `alpha` itself when standardizing residuals (`std = sqrt(pred_var + alpha_test)`).
- **SSB enabling requires the CSV flag:** set `config["output.ssb.enabled"] = "true"` (NOT `output.ssb.netcdf.enabled`) for in-memory `.ssb()` to work. The example fixture yields all-zero SSB at short horizons — never assert nonzero SSB values from it.
- **Determinism:** fixed seeds → reproducible `X`, `Y`, `alpha`. Per-point/per-seed run seeds and per-round batch seeds are derived deterministically from the design seed.
- **Do NOT mutate `SurrogateCalibrator`, `problem.py`, or any Phase 0 file except the one additive change to `emulator.py` in Task 1.**

## Pinned knobs (adjust here at plan-review; grounded in the spec + a verified gate prototype)

| Knob | Value | Rationale |
|---|---|---|
| `n_seeds` (S) | 10 (caller-set; ≥2 required) | Spec envelope; tests use 2–8. |
| `n0` (N₀) | `20 * d` | Spec initial design size. |
| `increment` | `10 * d` | Half N₀ per growth round (proposed). |
| `n_max` | `100 * d` | **Safety ceiling, not a target** — at d=17,S=10 that is ~17k runs; it exists to abort a non-converging design, not to be reached. |
| `COVERAGE_MIN` | 0.85 | Fraction of the 95% predictive interval that must cover; below-nominal band tolerates small-N sampling noise. |
| `MSSR_LOW, MSSR_HIGH` | 0.4, 2.5 | Mean standardized-squared-residual band; the **primary** discriminator (prototype: calibrated ≈ 0.8–1.3, miscalibrated 5–12). |
| `PIT_P_MIN` | 0.02 | KS-test p-value floor for PIT uniformity. |
| `MIN_GATE_POINTS` | 8 | Below this many valid (uncensored) points the gate reports not-calibratable rather than trusting the statistics. |
| `LOO_MAX`, `K_DEFAULT` | 20, 10 | Gate uses LOO CV when n ≤ 20, else 10-fold. |

These constants were validated against the real `GPEmulator`: a GP-friendly synthetic (`2 + sin(3·x₀) + 0.5·cos(4·x₁)`, natural-log-mean + Gaussian seed noise) passes 6/6 across seeds at N=60; a step discontinuity (`5 if x₀>0.5 else 1`) fails 6/6.

---

### Task 1: `GPEmulator.cross_validate` returns the test-index mapping

The gate must standardize each held-out residual by `pred_var(θᵢ) + alpha(θᵢ)`. `cross_validate` currently concatenates `y_true`/`y_pred`/`pred_var` in fold order but drops which input row each entry came from, so the caller cannot line up `alpha`. Add the concatenated test-index array.

**Files:**
- Modify: `osmose/calibration/uq/emulator.py` (add `test_idx` to `cross_validate`'s return dict)
- Test: `tests/test_uq_emulator.py` (append one case)

**Interfaces:**
- Consumes: existing `GPEmulator.cross_validate` from Phase 0.
- Produces: `cross_validate(...)` return dict gains `"test_idx": np.ndarray` — held-out row indices in the same fold-concatenation order as `y_true`/`y_pred`/`pred_var`, a permutation of `range(len(X))`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_uq_emulator.py`:

```python
def test_cross_validate_returns_aligned_test_idx(design):
    X, Y, alpha = design
    cv = GPEmulator(n_restarts_optimizer=0).cross_validate(X, Y, alpha, k_folds=5, seed=0)
    idx = cv["test_idx"]
    n = len(X)
    assert idx.shape == (n,)
    # A permutation of range(n): every row held out exactly once.
    assert sorted(idx.tolist()) == list(range(n))
    # Alignment: y_true equals Y indexed by test_idx (same fold-concat order).
    assert np.allclose(cv["y_true"], Y[idx])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_uq_emulator.py::test_cross_validate_returns_aligned_test_idx -q`
Expected: FAIL — `KeyError: 'test_idx'`.

- [ ] **Step 3: Add `test_idx` to `cross_validate`**

In `osmose/calibration/uq/emulator.py`, inside `cross_validate`, add a `test_idx` accumulator list alongside the existing ones and populate it per fold. Locate the loop and the return dict and change them to:

```python
        y_true: list[np.ndarray] = []
        y_pred: list[np.ndarray] = []
        pred_var: list[np.ndarray] = []
        test_idx: list[np.ndarray] = []
        fold_rmse: list[float] = []
        fold_r2: list[float] = []

        for train_idx, test_index in kf.split(X):
            fold = GPEmulator(self._n_restarts_optimizer, self._random_state)
            fold.fit(X[train_idx], Y[train_idx], alpha_arr[train_idx])
            mean, var = fold.predict(X[test_index])
            truth = Y[test_index]

            y_true.append(truth)
            y_pred.append(mean)
            pred_var.append(var)
            test_idx.append(test_index)
            fold_rmse.append(float(np.sqrt(np.mean((truth - mean) ** 2))))
            ss_res = float(np.sum((truth - mean) ** 2))
            ss_tot = float(np.sum((truth - np.mean(truth)) ** 2))
            fold_r2.append(1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0)

        return {
            "y_true": np.concatenate(y_true),
            "y_pred": np.concatenate(y_pred),
            "pred_var": np.concatenate(pred_var),
            "test_idx": np.concatenate(test_idx),
            "fold_rmse": fold_rmse,
            "fold_r2": fold_r2,
            "mean_rmse": float(np.mean(fold_rmse)),
            "mean_r2": float(np.mean(fold_r2)),
        }
```

(The only changes vs Phase 0: the loop variable is renamed `test_index`, a `test_idx` list accumulates `test_index`, and the return dict gains `"test_idx"`. Update the method docstring's returned-keys list to mention `test_idx`.)

- [ ] **Step 4: Run the emulator suite to verify all pass**

Run: `.venv/bin/python -m pytest tests/test_uq_emulator.py -q`
Expected: PASS — 10 passed.

- [ ] **Step 5: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/emulator.py tests/test_uq_emulator.py
.venv/bin/ruff check osmose/calibration/uq/emulator.py tests/test_uq_emulator.py
git add osmose/calibration/uq/emulator.py tests/test_uq_emulator.py
git commit -m "feat(uq): return aligned test_idx from cross_validate for the calibration gate (Phase 1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: design helpers — `point_to_overrides` + `lhs_design`

Two pure functions that the design executor and growth loop build on. `point_to_overrides` is the single home of the base-10 simulator-input transform and is unit-tested in isolation.

**Files:**
- Create: `osmose/calibration/uq/design.py`
- Test: `tests/test_uq_design.py`

**Interfaces:**
- Consumes: `osmose.calibration.problem.FreeParameter`, `Transform`; `scipy.stats.qmc.LatinHypercube`.
- Produces:
  - `point_to_overrides(x: np.ndarray, free_params: list[FreeParameter]) -> dict[str, str]` — one sampling-space point → OSMOSE override dict; applies base-10 `10**val` for `Transform.LOG`, stringifies every value.
  - `lhs_design(free_params: list[FreeParameter], n_points: int, seed: int) -> np.ndarray` — `(n_points, d)` seeded LHS scaled to each parameter's `[lower_bound, upper_bound]` in **sampling space** (no transform applied here).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_uq_design.py`:

```python
"""Tests for the UQ design executor (helpers, seed reduction, engine evaluator)."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.calibration.problem import FreeParameter, Transform
from osmose.calibration.uq.design import lhs_design, point_to_overrides


def _params():
    return [
        FreeParameter("mortality.fishing.rate.sp0", 0.0, 2.0, Transform.LINEAR),
        FreeParameter("species.larva.mortality.rate.sp0", -3.0, 0.0, Transform.LOG),
    ]


def test_point_to_overrides_linear_passthrough():
    ov = point_to_overrides(np.array([1.5, -1.0]), _params())
    assert ov["mortality.fishing.rate.sp0"] == "1.5"


def test_point_to_overrides_log_is_base10():
    ov = point_to_overrides(np.array([1.5, -2.0]), _params())
    # LOG param: 10**(-2.0) == 0.01
    assert float(ov["species.larva.mortality.rate.sp0"]) == pytest.approx(0.01)


def test_point_to_overrides_all_keys_stringified():
    ov = point_to_overrides(np.array([0.3, -1.0]), _params())
    assert set(ov) == {"mortality.fishing.rate.sp0", "species.larva.mortality.rate.sp0"}
    assert all(isinstance(v, str) for v in ov.values())


def test_lhs_design_shape_and_bounds():
    X = lhs_design(_params(), n_points=25, seed=0)
    assert X.shape == (25, 2)
    assert np.all(X[:, 0] >= 0.0) and np.all(X[:, 0] <= 2.0)
    assert np.all(X[:, 1] >= -3.0) and np.all(X[:, 1] <= 0.0)


def test_lhs_design_deterministic():
    a = lhs_design(_params(), n_points=25, seed=7)
    b = lhs_design(_params(), n_points=25, seed=7)
    assert np.array_equal(a, b)
    c = lhs_design(_params(), n_points=25, seed=8)
    assert not np.array_equal(a, c)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_design.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.calibration.uq.design'`.

- [ ] **Step 3: Create `design.py` with the two helpers**

Create `osmose/calibration/uq/design.py`:

```python
"""Seeded LHS design executed through the Python engine, reduced for the UQ emulator.

The design runs through an INJECTABLE evaluator — a ``(point, seed) -> stat-dict``
callable — so the whole pipeline is testable without OSMOSE runs. The real
evaluator (``make_engine_evaluator``) runs the Python engine with SSB output
enabled; tests pass a synthetic function.

Two transforms live here and must not be conflated: base-10 ``10**val`` at the
simulator-input boundary (``point_to_overrides``, mirroring problem.py:263) and
natural ``np.log`` of the linear stat when forming the GP target Y (``run_design``).
"""

from __future__ import annotations

import numpy as np
from scipy.stats.qmc import LatinHypercube

from osmose.calibration.problem import FreeParameter, Transform


def point_to_overrides(x: np.ndarray, free_params: list[FreeParameter]) -> dict[str, str]:
    """One sampling-space point -> OSMOSE override dict.

    Applies base-10 ``10**val`` for ``Transform.LOG`` params (the simulator-input
    transform, matching osmose/calibration/problem.py:263) and stringifies every
    value. NOT the natural-log GP-target transform, which is separate.
    """
    overrides: dict[str, str] = {}
    for j, fp in enumerate(free_params):
        val = float(x[j])
        if fp.transform == Transform.LOG:
            val = 10.0**val
        overrides[fp.key] = str(val)
    return overrides


def lhs_design(free_params: list[FreeParameter], n_points: int, seed: int) -> np.ndarray:
    """Seeded Latin-hypercube design, ``(n_points, d)``, scaled to sampling-space bounds.

    No transform is applied — the design lives in sampling space; the emulator
    trains on sampling-space X and ``point_to_overrides`` transforms only at the
    simulator-input boundary.
    """
    d = len(free_params)
    unit = LatinHypercube(d=d, seed=seed).random(n=n_points)
    lower = np.array([fp.lower_bound for fp in free_params])
    upper = np.array([fp.upper_bound for fp in free_params])
    return unit * (upper - lower) + lower
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_design.py -q`
Expected: PASS — 5 passed.

- [ ] **Step 5: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/design.py tests/test_uq_design.py
.venv/bin/ruff check osmose/calibration/uq/design.py tests/test_uq_design.py
git add osmose/calibration/uq/design.py tests/test_uq_design.py
git commit -m "feat(uq): add design helpers point_to_overrides + lhs_design (Phase 1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `gate.py` — coverage/PIT calibration gate

The heart of Phase 1: cross-validate the emulator, standardize held-out residuals by `pred_var + alpha`, and decide calibration from coverage, MSSR, and PIT uniformity. Pure numpy/scipy, fully synthetic-testable.

**Files:**
- Create: `osmose/calibration/uq/gate.py`
- Test: `tests/test_uq_gate.py`

**Interfaces:**
- Consumes: `GPEmulator` (with Task 1's `test_idx`); `scipy.stats` (`norm.cdf`, `kstest`).
- Produces:
  - Module constants `COVERAGE_MIN=0.85`, `MSSR_LOW=0.4`, `MSSR_HIGH=2.5`, `PIT_P_MIN=0.02`, `MIN_GATE_POINTS=8`, `LOO_MAX=20`, `K_DEFAULT=10`.
  - `@dataclass GateReport` with fields `n: int`, `coverage: float`, `mssr: float`, `pit_pvalue: float`, `r2: float`, `r2_ceiling: float`, `passed: bool`, `reasons: list[str]`, `key: str | None`.
  - `evaluate_emulator_calibration(X, Y, alpha, *, key=None, seed=0) -> GateReport` — the gate. Uses LOO CV when `len(X) <= LOO_MAX`, else `K_DEFAULT`-fold. Returns `passed=False` with a reason when `len(X) < MIN_GATE_POINTS`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_uq_gate.py`:

```python
"""Tests for the UQ calibration gate (coverage/PIT/MSSR decision)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats.qmc import LatinHypercube

from osmose.calibration.uq.gate import GateReport, evaluate_emulator_calibration

# The misspecified case intentionally feeds the GP an un-fittable step function,
# so its optimizer legitimately hits length-scale bounds. Silence that expected
# noise so test output stays pristine; the pass/fail assertions still guard the
# decision.
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def _synthetic_design(misspecified: bool, n: int = 60, s: int = 8, sigma: float = 0.15, seed: int = 0):
    """Natural-log-mean design with Gaussian seed noise; well-specified is
    GP-friendly (smooth), misspecified is a step discontinuity a stationary GP
    cannot fit. Returns (X, Y=mean-of-logs, alpha=var(logs, ddof=1)/s)."""
    X = LatinHypercube(d=2, seed=seed).random(n=n)
    Y = np.empty(n)
    alpha = np.empty(n)
    for i in range(n):
        logs = np.empty(s)
        for k in range(s):
            rng = np.random.default_rng(1000 + i * s + k)
            if misspecified:
                mean_log = 5.0 if X[i, 0] > 0.5 else 1.0
            else:
                mean_log = 2.0 + np.sin(3.0 * X[i, 0]) + 0.5 * np.cos(4.0 * X[i, 1])
            logs[k] = mean_log + rng.normal(0.0, sigma)
        Y[i] = float(np.mean(logs))
        alpha[i] = float(np.var(logs, ddof=1)) / s
    return X, Y, alpha


def test_gate_passes_calibrated_synthetic():
    X, Y, alpha = _synthetic_design(misspecified=False)
    report = evaluate_emulator_calibration(X, Y, alpha, key="cod_biomass_mean")
    assert report.passed is True, report.reasons
    assert report.key == "cod_biomass_mean"
    assert report.n == 60


def test_gate_fails_miscalibrated_synthetic():
    X, Y, alpha = _synthetic_design(misspecified=True)
    report = evaluate_emulator_calibration(X, Y, alpha)
    assert report.passed is False
    # The step discontinuity blows up the standardized residuals.
    assert report.mssr > 2.5


def test_gate_insufficient_points_not_calibratable():
    X = np.linspace(0, 1, 5).reshape(-1, 1)
    Y = np.arange(5, dtype=float)
    alpha = np.full(5, 0.01)
    report = evaluate_emulator_calibration(X, Y, alpha)
    assert report.passed is False
    assert any("insufficient" in r.lower() for r in report.reasons)


def test_gate_report_is_dataclass_with_metrics():
    X, Y, alpha = _synthetic_design(misspecified=False, n=40)
    report = evaluate_emulator_calibration(X, Y, alpha)
    assert isinstance(report, GateReport)
    for field in ("coverage", "mssr", "pit_pvalue", "r2", "r2_ceiling"):
        assert isinstance(getattr(report, field), float)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_gate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.calibration.uq.gate'`.

- [ ] **Step 3: Create `gate.py`**

Create `osmose/calibration/uq/gate.py`:

```python
"""Probabilistic emulator-calibration gate (Bastos & O'Hagan 2009 style).

Cross-validates a GPEmulator and standardizes held-out residuals by the latent
predictive variance plus the held-out seed-mean noise ``alpha`` (= s²/S). A
calibrated emulator has ~95% interval coverage, mean standardized-squared-residual
(MSSR) near 1, and PIT-uniform residuals. MSSR is the primary discriminator;
coverage and PIT are secondary. Certifies emulator fidelity only — NOT the
≤~20-effective-param sampler envelope, which Phase 2 enforces separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from osmose.calibration.uq.emulator import GPEmulator

COVERAGE_MIN = 0.85
MSSR_LOW, MSSR_HIGH = 0.4, 2.5
PIT_P_MIN = 0.02
MIN_GATE_POINTS = 8
LOO_MAX = 20
K_DEFAULT = 10


@dataclass
class GateReport:
    """Calibration verdict + metrics for one emulator output."""

    n: int
    coverage: float
    mssr: float
    pit_pvalue: float
    r2: float
    r2_ceiling: float
    passed: bool
    reasons: list[str] = field(default_factory=list)
    key: str | None = None


def _k_folds(n: int) -> int:
    """LOO for small designs, else K_DEFAULT-fold."""
    return n if n <= LOO_MAX else min(K_DEFAULT, n)


def evaluate_emulator_calibration(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: np.ndarray,
    *,
    key: str | None = None,
    seed: int = 0,
) -> GateReport:
    """Cross-validate the emulator and decide whether it is calibrated.

    ``X`` (n, d) sampling-space inputs; ``Y`` (n,) natural-log seed-mean targets;
    ``alpha`` (n,) per-point seed-mean noise s²/S. Returns a GateReport; a design
    with fewer than MIN_GATE_POINTS valid points is reported not-calibratable
    rather than trusting the statistics.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).ravel()
    alpha = np.asarray(alpha, dtype=float).ravel()
    n = len(X)

    if n < MIN_GATE_POINTS:
        return GateReport(
            n=n, coverage=float("nan"), mssr=float("nan"), pit_pvalue=float("nan"),
            r2=float("nan"), r2_ceiling=float("nan"), passed=False, key=key,
            reasons=[f"insufficient valid points ({n} < {MIN_GATE_POINTS})"],
        )

    cv = GPEmulator().cross_validate(X, Y, alpha, k_folds=_k_folds(n), seed=seed)
    y_true, y_pred = cv["y_true"], cv["y_pred"]
    total_var = cv["pred_var"] + alpha[cv["test_idx"]]
    resid = (y_true - y_pred) / np.sqrt(total_var)

    coverage = float(np.mean(np.abs(resid) < 1.96))
    mssr = float(np.mean(resid**2))
    pit_pvalue = float(stats.kstest(stats.norm.cdf(resid), "uniform").pvalue)

    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot if ss_tot > 0 else 0.0
    var_Y = float(np.var(Y, ddof=0))
    r2_ceiling = 1.0 - float(np.mean(alpha)) / var_Y if var_Y > 0 else 0.0

    reasons: list[str] = []
    if coverage < COVERAGE_MIN:
        reasons.append(f"coverage {coverage:.3f} < {COVERAGE_MIN}")
    if not (MSSR_LOW <= mssr <= MSSR_HIGH):
        reasons.append(f"MSSR {mssr:.3f} outside [{MSSR_LOW}, {MSSR_HIGH}]")
    if pit_pvalue <= PIT_P_MIN:
        reasons.append(f"PIT KS p={pit_pvalue:.3f} <= {PIT_P_MIN}")

    return GateReport(
        n=n, coverage=coverage, mssr=mssr, pit_pvalue=pit_pvalue, r2=r2,
        r2_ceiling=r2_ceiling, passed=not reasons, reasons=reasons, key=key,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_gate.py -q`
Expected: PASS — 4 passed. (The two synthetic-decision tests build real GPs and take a few seconds; do not loosen the thresholds if a test fails — increase N or reduce sigma in the fixture, since the thresholds are validated against the prototype.)

- [ ] **Step 5: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/gate.py tests/test_uq_gate.py
.venv/bin/ruff check osmose/calibration/uq/gate.py tests/test_uq_gate.py
git add osmose/calibration/uq/gate.py tests/test_uq_gate.py
git commit -m "feat(uq): add coverage/PIT emulator-calibration gate (Phase 1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `design.py` core — `DesignResult`, `run_design`, `make_engine_evaluator`

The design executor: run each LHS point through the evaluator over S seeds, reduce per targeted key to `(Y=mean-of-logs, alpha=var(logs, ddof=1)/S)` with per-key extinction censoring, and provide the real Python-engine evaluator.

**Files:**
- Modify: `osmose/calibration/uq/design.py` (add `DesignResult`, `run_design`, `make_engine_evaluator`, and the `Evaluator` alias)
- Test: `tests/test_uq_design.py` (append)

**Interfaces:**
- Consumes: `point_to_overrides`, `lhs_design` (Task 2); `compute_uq_stats` (Phase 0); lazily `PythonEngine`, `OsmoseConfigReader`.
- Produces:
  - `Evaluator = Callable[[np.ndarray, int], dict[str, float]]`.
  - `@dataclass DesignResult` with `X: np.ndarray`, `keys: list[str]`, `Y: dict[str, np.ndarray]`, `alpha: dict[str, np.ndarray]` (per-key arrays, NaN = censored), and methods `valid(key) -> (X, Y, alpha)` (rows where Y is not NaN) and `n_censored(key) -> int`.
  - `run_design(evaluator, free_params, target_keys, n_points, n_seeds, *, seed=0, seed_offset=0, X=None) -> DesignResult` — requires `n_seeds >= 2`.
  - `make_engine_evaluator(free_params, base_config_path, species_names, *, enable_ssb=True, nyear=None) -> Evaluator`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_uq_design.py`:

```python
from pathlib import Path

from osmose.calibration.uq.design import DesignResult, make_engine_evaluator, run_design

_EXAMPLE_CONFIG = Path(__file__).parent.parent / "data" / "examples" / "osm_all-parameters.csv"


def _linear_fp():
    return [FreeParameter("mortality.fishing.rate.sp0", 0.0, 1.0, Transform.LINEAR)]


def _const_evaluator(value_by_key, extinct_points=()):
    """Evaluator returning fixed per-key values with a tiny seed-dependent wobble,
    optionally forcing a species to 0 (extinction) at given point indices."""
    def ev(x, seed):
        rng = np.random.default_rng(int(seed))
        out = {}
        for k, v in value_by_key.items():
            out[k] = float(v * np.exp(rng.normal(0.0, 0.05)))
        return out
    return ev


def test_run_design_reduces_log_mean_and_ddof1_alpha():
    # Deterministic evaluator: value depends only on seed, so we can hand-check.
    def ev(x, seed):
        return {"cod_biomass_mean": float(np.exp(0.1 * int(seed)))}

    res = run_design(ev, _linear_fp(), ["cod_biomass_mean"], n_points=3, n_seeds=4,
                     seed=0, seed_offset=0)
    # Point 0 uses run seeds 0..3 -> logs = [0, 0.1, 0.2, 0.3].
    logs = np.array([0.0, 0.1, 0.2, 0.3])
    assert res.Y["cod_biomass_mean"][0] == pytest.approx(logs.mean())
    assert res.alpha["cod_biomass_mean"][0] == pytest.approx(logs.var(ddof=1) / 4)


def test_run_design_censors_extinction_per_point():
    def ev(x, seed):
        # Point index is encoded via x[0]; extinct (0.0) only at x[0] < 0.5.
        val = 0.0 if x[0] < 0.5 else 10.0
        return {"cod_biomass_mean": float(val)}

    X = np.array([[0.2], [0.8], [0.9]])
    res = run_design(ev, _linear_fp(), ["cod_biomass_mean"], n_points=3, n_seeds=2, X=X)
    assert np.isnan(res.Y["cod_biomass_mean"][0])  # extinct -> censored
    assert not np.isnan(res.Y["cod_biomass_mean"][1])
    assert res.n_censored("cod_biomass_mean") == 1


def test_run_design_per_key_independent_censoring():
    def ev(x, seed):
        # cod extinct everywhere; herring healthy everywhere.
        return {"cod_ssb_mean": 0.0, "herring_biomass_mean": 100.0}

    res = run_design(ev, _linear_fp(), ["cod_ssb_mean", "herring_biomass_mean"],
                     n_points=4, n_seeds=2, seed=1)
    assert res.n_censored("cod_ssb_mean") == 4
    assert res.n_censored("herring_biomass_mean") == 0
    Xv, Yv, av = res.valid("herring_biomass_mean")
    assert len(Xv) == 4 and not np.any(np.isnan(Yv))


def test_run_design_reproducible():
    ev = _const_evaluator({"cod_biomass_mean": 10.0})
    a = run_design(ev, _linear_fp(), ["cod_biomass_mean"], n_points=5, n_seeds=3, seed=2)
    b = run_design(ev, _linear_fp(), ["cod_biomass_mean"], n_points=5, n_seeds=3, seed=2)
    assert np.array_equal(a.X, b.X)
    assert np.allclose(a.Y["cod_biomass_mean"], b.Y["cod_biomass_mean"], equal_nan=True)
    assert np.allclose(a.alpha["cod_biomass_mean"], b.alpha["cod_biomass_mean"], equal_nan=True)


def test_run_design_requires_two_seeds():
    ev = _const_evaluator({"cod_biomass_mean": 10.0})
    with pytest.raises(ValueError, match="n_seeds"):
        run_design(ev, _linear_fp(), ["cod_biomass_mean"], n_points=3, n_seeds=1)


def test_design_result_valid_filters_censored_rows():
    Y = {"k": np.array([1.0, np.nan, 3.0])}
    alpha = {"k": np.array([0.1, np.nan, 0.3])}
    res = DesignResult(X=np.arange(3).reshape(-1, 1).astype(float), keys=["k"], Y=Y, alpha=alpha)
    Xv, Yv, av = res.valid("k")
    assert Xv.shape == (2, 1)
    assert np.array_equal(Yv, np.array([1.0, 3.0]))
    assert np.array_equal(av, np.array([0.1, 0.3]))


def test_engine_evaluator_emits_biomass_and_ssb_keys():
    from osmose.config import OsmoseConfigReader

    cfg = OsmoseConfigReader().read(_EXAMPLE_CONFIG)
    n_sp = int(cfg.get("simulation.nspecies", "0"))
    species = [cfg.get(f"species.name.sp{i}") for i in range(n_sp)]
    ev = make_engine_evaluator(_linear_fp(), _EXAMPLE_CONFIG, species, enable_ssb=True, nyear=1)
    stats = ev(np.array([0.3]), seed=1)
    # Biomass is always collected and non-zero.
    biomass_keys = [k for k in stats if k.endswith("_biomass_mean")]
    assert biomass_keys and all(stats[k] >= 0.0 for k in biomass_keys)
    # SSB plumbing: enabling output.ssb.enabled makes .ssb() readable, so _ssb_mean
    # keys are emitted. Values are 0.0 on this fixture at nyear=1 — assert presence,
    # NOT magnitude.
    assert any(k.endswith("_ssb_mean") for k in stats)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_design.py -q -k "run_design or design_result or engine_evaluator"`
Expected: FAIL — `ImportError: cannot import name 'run_design'` (and `DesignResult`, `make_engine_evaluator`).

- [ ] **Step 3: Add the core to `design.py`**

Add to `osmose/calibration/uq/design.py`. First extend the imports at the top:

```python
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from osmose.calibration.uq.output_stats import compute_uq_stats
```

Then append the type alias, dataclass, and functions:

```python
Evaluator = Callable[[np.ndarray, int], dict[str, float]]


@dataclass
class DesignResult:
    """LHS design with per-targeted-key natural-log seed-mean targets and noise.

    ``Y[key]`` and ``alpha[key]`` are (n,) arrays over the shared design ``X``;
    NaN marks a point censored for that key (a species extinct at that point).
    Censoring is per-key: a point dropped for ``cod_ssb_mean`` still trains
    ``herring_biomass_mean``.
    """

    X: np.ndarray
    keys: list[str]
    Y: dict[str, np.ndarray]
    alpha: dict[str, np.ndarray]

    def valid(self, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rows where ``key`` is not censored: (X_valid, Y_valid, alpha_valid)."""
        mask = ~np.isnan(self.Y[key])
        return self.X[mask], self.Y[key][mask], self.alpha[key][mask]

    def n_censored(self, key: str) -> int:
        """Number of points censored (NaN) for ``key``."""
        return int(np.isnan(self.Y[key]).sum())


def run_design(
    evaluator: Evaluator,
    free_params: list[FreeParameter],
    target_keys: Sequence[str],
    n_points: int,
    n_seeds: int,
    *,
    seed: int = 0,
    seed_offset: int = 0,
    X: np.ndarray | None = None,
) -> DesignResult:
    """Run an LHS design through ``evaluator`` over ``n_seeds`` seeds; reduce per key.

    For each design point and targeted key, collect the linear stat over seeds;
    if any seed is missing or <= 0 the point is CENSORED for that key (mean-of-logs
    undefined). Otherwise ``Y = mean(log values)`` (natural log) and
    ``alpha = var(log values, ddof=1) / n_seeds`` (unbiased per-run variance of the
    seed mean — ddof=1 here is a different quantity from the emulator's ddof=0).

    Run seeds are ``seed_offset + i*n_seeds + k`` for point ``i``, seed ``k`` —
    deterministic, so a re-run reproduces X, Y, alpha. ``X`` may be supplied to
    re-run a fixed design (used by the growth loop's appended batches).
    """
    if n_seeds < 2:
        raise ValueError(f"n_seeds must be >= 2 to estimate seed variance, got {n_seeds}")
    if X is None:
        X = lhs_design(free_params, n_points, seed)
    n = len(X)
    keys = list(target_keys)
    Y = {k: np.full(n, np.nan) for k in keys}
    alpha = {k: np.full(n, np.nan) for k in keys}

    for i in range(n):
        per_seed = [
            evaluator(X[i], seed_offset + i * n_seeds + k) for k in range(n_seeds)
        ]
        for key in keys:
            vals = [d.get(key) for d in per_seed]
            if any(v is None or v <= 0.0 for v in vals):
                continue  # censor: mean-of-logs undefined
            logs = np.log(np.asarray(vals, dtype=float))
            Y[key][i] = float(np.mean(logs))
            alpha[key][i] = float(np.var(logs, ddof=1)) / n_seeds

    return DesignResult(X=X, keys=keys, Y=Y, alpha=alpha)


def make_engine_evaluator(
    free_params: list[FreeParameter],
    base_config_path: Path,
    species_names: Sequence[str],
    *,
    enable_ssb: bool = True,
    nyear: int | None = None,
) -> Evaluator:
    """Build the real Python-engine evaluator: point+seed -> per-species stat dict.

    Reads the base config once, then per call injects ``output.ssb.enabled='true'``
    (when ``enable_ssb``; the CSV flag — the netcdf flag does not make in-memory
    ``.ssb()`` readable), optionally overrides ``simulation.time.nyear``, applies
    ``point_to_overrides``, runs the engine, and reduces with ``compute_uq_stats``.
    ``PythonEngine``/``OsmoseConfigReader`` are lazy-imported to keep design.py light.
    """
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine

    base_cfg = OsmoseConfigReader().read(base_config_path)
    species = list(species_names)

    def evaluate(x: np.ndarray, seed: int) -> dict[str, float]:
        cfg = dict(base_cfg)
        if enable_ssb:
            cfg["output.ssb.enabled"] = "true"
        if nyear is not None:
            cfg["simulation.time.nyear"] = str(nyear)
        cfg.update(point_to_overrides(x, free_params))
        results = PythonEngine().run_in_memory(cfg, seed=int(seed))
        return compute_uq_stats(results, species)

    return evaluate
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_design.py -q`
Expected: PASS — 12 passed. (The `test_engine_evaluator_*` case runs one short engine simulation, a few seconds.)

- [ ] **Step 5: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/design.py tests/test_uq_design.py
.venv/bin/ruff check osmose/calibration/uq/design.py tests/test_uq_design.py
git add osmose/calibration/uq/design.py tests/test_uq_design.py
git commit -m "feat(uq): add design executor with injectable evaluator + per-key censoring (Phase 1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: `grow_until_calibrated` — bounded design-growth loop + synthetic acceptance

Compose design + gate: build N₀, gate every targeted key, and on failure append a fresh seeded-LHS batch and re-gate, until all keys pass or the hard `n_max` ceiling aborts. Validate end-to-end on the well-specified and misspecified synthetics.

**Files:**
- Modify: `osmose/calibration/uq/design.py` (add `GrowthResult`, `_merge_designs`, `grow_until_calibrated`)
- Test: `tests/test_uq_growth.py`

**Interfaces:**
- Consumes: `run_design`, `DesignResult` (Task 4); `evaluate_emulator_calibration`, `GateReport` (Task 3).
- Produces:
  - `@dataclass GrowthResult` with `design: DesignResult`, `reports: dict[str, GateReport]`, `status: str` (`"calibrated"` or `"aborted_n_max"`), `rounds: int`.
  - `grow_until_calibrated(evaluator, free_params, target_keys, n_seeds, *, n0, increment, n_max, seed=0, gate_fn=None) -> GrowthResult` — `gate_fn` defaults to `evaluate_emulator_calibration` and is injectable for deterministic tests.
  - `_merge_designs(a: DesignResult, b: DesignResult) -> DesignResult`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_uq_growth.py`:

```python
"""Tests for the bounded design-growth loop + synthetic acceptance."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.calibration.problem import FreeParameter, Transform
from osmose.calibration.uq.design import (
    DesignResult,
    _merge_designs,
    grow_until_calibrated,
)
from osmose.calibration.uq.gate import GateReport

# The misspecified acceptance case drives the GP to its length-scale bounds by
# design; silence the expected ConvergenceWarning so output stays pristine.
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def _fp2():
    return [
        FreeParameter("a.sp0", 0.0, 1.0, Transform.LINEAR),
        FreeParameter("b.sp0", 0.0, 1.0, Transform.LINEAR),
    ]


def _report(passed):
    return GateReport(n=10, coverage=0.95, mssr=1.0, pit_pvalue=0.5, r2=0.9,
                      r2_ceiling=0.95, passed=passed, reasons=[])


def test_merge_designs_concatenates_x_and_per_key_arrays():
    a = DesignResult(X=np.zeros((2, 2)), keys=["k"], Y={"k": np.array([1.0, 2.0])},
                     alpha={"k": np.array([0.1, 0.2])})
    b = DesignResult(X=np.ones((3, 2)), keys=["k"], Y={"k": np.array([3.0, 4.0, 5.0])},
                     alpha={"k": np.array([0.3, 0.4, 0.5])})
    m = _merge_designs(a, b)
    assert m.X.shape == (5, 2)
    assert np.array_equal(m.Y["k"], np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert np.array_equal(m.alpha["k"], np.array([0.1, 0.2, 0.3, 0.4, 0.5]))


def test_growth_aborts_at_n_max_when_gate_always_fails():
    ev = lambda x, seed: {"k": 10.0}  # noqa: E731
    gate_fn = lambda X, Y, alpha, **kw: _report(False)  # noqa: E731
    result = grow_until_calibrated(ev, _fp2(), ["k"], n_seeds=2,
                                   n0=10, increment=10, n_max=25, seed=0, gate_fn=gate_fn)
    assert result.status == "aborted_n_max"
    assert len(result.design.X) <= 25
    assert result.rounds >= 1


def test_growth_returns_calibrated_after_one_append():
    ev = lambda x, seed: {"k": 10.0}  # noqa: E731
    calls = {"n": 0}

    def gate_fn(X, Y, alpha, **kw):
        calls["n"] += 1
        return _report(calls["n"] >= 2)  # fail first gate, pass the second

    result = grow_until_calibrated(ev, _fp2(), ["k"], n_seeds=2,
                                   n0=10, increment=10, n_max=100, seed=0, gate_fn=gate_fn)
    assert result.status == "calibrated"
    assert result.rounds == 1  # one append happened
    assert len(result.design.X) == 20


def _synthetic_evaluator(misspecified):
    def ev(x, seed):
        rng = np.random.default_rng(int(seed))
        if misspecified:
            mean_log = 5.0 if x[0] > 0.5 else 1.0
        else:
            mean_log = 2.0 + np.sin(3.0 * x[0]) + 0.5 * np.cos(4.0 * x[1])
        return {"cod_biomass_mean": float(np.exp(mean_log + rng.normal(0.0, 0.15)))}
    return ev


def test_growth_well_specified_synthetic_calibrates():
    result = grow_until_calibrated(
        _synthetic_evaluator(misspecified=False), _fp2(), ["cod_biomass_mean"],
        n_seeds=8, n0=60, increment=30, n_max=120, seed=0)
    assert result.status == "calibrated"
    assert result.reports["cod_biomass_mean"].passed


def test_growth_misspecified_synthetic_aborts_loudly():
    result = grow_until_calibrated(
        _synthetic_evaluator(misspecified=True), _fp2(), ["cod_biomass_mean"],
        n_seeds=8, n0=60, increment=30, n_max=60, seed=0)  # n_max == n0: one gate, then abort
    assert result.status == "aborted_n_max"
    assert not result.reports["cod_biomass_mean"].passed
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uq_growth.py -q -k "merge or aborts or one_append"`
Expected: FAIL — `ImportError: cannot import name '_merge_designs'` (and `grow_until_calibrated`).

- [ ] **Step 3: Add the growth loop to `design.py`**

Add to `osmose/calibration/uq/design.py`. Extend the imports:

```python
from osmose.calibration.uq.gate import GateReport, evaluate_emulator_calibration
```

Append:

```python
@dataclass
class GrowthResult:
    """Outcome of the bounded design-growth loop."""

    design: DesignResult
    reports: dict[str, GateReport]
    status: str  # "calibrated" | "aborted_n_max"
    rounds: int


def _merge_designs(a: DesignResult, b: DesignResult) -> DesignResult:
    """Concatenate two designs over the same targeted keys (a's keys, in order)."""
    X = np.vstack([a.X, b.X])
    Y = {k: np.concatenate([a.Y[k], b.Y[k]]) for k in a.keys}
    alpha = {k: np.concatenate([a.alpha[k], b.alpha[k]]) for k in a.keys}
    return DesignResult(X=X, keys=list(a.keys), Y=Y, alpha=alpha)


def grow_until_calibrated(
    evaluator: Evaluator,
    free_params: list[FreeParameter],
    target_keys: Sequence[str],
    n_seeds: int,
    *,
    n0: int,
    increment: int,
    n_max: int,
    seed: int = 0,
    gate_fn: Callable[..., GateReport] | None = None,
) -> GrowthResult:
    """Build N0, gate every key, and grow by ``increment`` until all keys pass or
    ``n_max`` aborts.

    ``gate_fn(X, Y, alpha, key=...)`` defaults to the real
    ``evaluate_emulator_calibration`` and is injectable for deterministic tests.
    Each appended batch uses a distinct deterministic seed offset so re-runs
    reproduce. ``n_max`` is a HARD safety ceiling, not a target — the loop never
    grows past it and aborts with the last reports.
    """
    gate = gate_fn if gate_fn is not None else evaluate_emulator_calibration
    keys = list(target_keys)

    def _gate_all(result: DesignResult) -> dict[str, GateReport]:
        reports = {}
        for key in keys:
            Xv, Yv, av = result.valid(key)
            reports[key] = gate(Xv, Yv, av, key=key)
        return reports

    result = run_design(evaluator, free_params, keys, n0, n_seeds, seed=seed, seed_offset=0)
    rounds = 0
    while True:
        reports = _gate_all(result)
        if all(r.passed for r in reports.values()):
            return GrowthResult(design=result, reports=reports, status="calibrated", rounds=rounds)
        if len(result.X) + increment > n_max:
            return GrowthResult(
                design=result, reports=reports, status="aborted_n_max", rounds=rounds
            )
        rounds += 1
        batch = run_design(
            evaluator, free_params, keys, increment, n_seeds,
            seed=seed + rounds, seed_offset=rounds * 1_000_000,
        )
        result = _merge_designs(result, batch)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uq_growth.py -q`
Expected: PASS — 5 passed. (The two real-gate synthetic acceptance tests build GPs and take several seconds each.)

- [ ] **Step 5: Lint/format and commit**

```bash
.venv/bin/ruff format osmose/calibration/uq/design.py tests/test_uq_growth.py
.venv/bin/ruff check osmose/calibration/uq/design.py tests/test_uq_growth.py
git add osmose/calibration/uq/design.py tests/test_uq_growth.py
git commit -m "feat(uq): add bounded design-growth loop + synthetic acceptance (Phase 1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Phase 1 Done Check

After Task 5:

```bash
.venv/bin/python -m pytest tests/test_uq_emulator.py tests/test_uq_output_stats.py tests/test_uq_design.py tests/test_uq_gate.py tests/test_uq_growth.py -q
.venv/bin/python -m pytest tests/test_calibration_surrogate.py tests/test_calibration_problem.py -q
```

Expected: all pass. The second command confirms Phase 1 did not disturb the existing calibration harness (`problem.py` is untouched).

Milestone (from the spec): "can we build a *calibrated* emulator" — a seeded design executed through the engine (SSB enabled), reduced to per-output `(X, Y, alpha)` with per-key extinction censoring, a coverage/PIT gate that passes a well-specified synthetic and fails a misspecified one, and a bounded design-growth loop — with `problem.py`/`SurrogateCalibrator` untouched and no new dependencies.

## Phase 2 Watch-Points (carry forward)

- **The gate certifies emulator fidelity only, NOT the ≤~20-effective-param sampler envelope.** Phase 2 must add the sampler-adequacy diagnostic + hard nominal-dimension cap that aborts regardless of a gate pass (spec's trust-gate section).
- **`n_max` as ceiling, not target:** if the growth loop routinely aborts, that is a signal the chunk has fallen out of the trustworthy regime — surface it to the user rather than silently returning an uncalibrated design.
- **SSB-zero fixture:** the example config yields all-zero SSB at short horizons, so an SSB-targeted design there censors every point. Real Baltic configs at full horizon produce nonzero SSB; the integration test deliberately asserts SSB *plumbing*, not values.
- **`total_var` has no epsilon floor (from the Phase 1 whole-branch review).** The gate standardizes by `sqrt(pred_var + alpha[test_idx])`. On the real path this is always positive (seed noise `alpha>0`, held-out latent variance `>0`), so no floor is needed now. But when Phase 2's sampler leans on this quantity, add a small epsilon floor as cheap insurance against a division by ~0 in a degenerate (zero-seed-variance + near-zero LOO variance) corner.
- **Growth-loop seed collision:** batch seed offset is `rounds * 1_000_000`; with `increment * n_seeds` per batch far below 1e6 this cannot collide, but if `n_max`/`increment`/`n_seeds` ever grow past that budget, widen the offset.

## Deliberate Phase 1 simplifications (recorded, not dropped)

The spec's "Extinction / regime handling" section lists three checks; Phase 1 implements only the first. These are deferred because the other two need signals `compute_uq_stats` does not yet expose (it emits only means, per the Phase 0 review), so they are a coherent later increment — but they must not be silently forgotten:

- **Extinction censoring — DONE** (any seed ≤ 0 → point censored per key, mean-of-logs undefined).
- **Non-equilibrium point rejection — DEFERRED.** The spec wants points rejected/flagged via existing CV/trend stability signals. `compute_uq_stats` returns only means; add CV/trend to it (or compute them in `run_design`) and reject non-equilibrated points before this design feeds a posterior.
- **Non-stationarity diagnostic — DEFERRED.** The spec wants a flag when `s²` (per-point seed variance) varies strongly across the design or residuals correlate with location. `run_design` already computes per-point `alpha = s²/S`; a diagnostic over that array is a small follow-up.
- **Acceptance-test determinism vs CPU arch:** the two real-gate synthetic tests are deterministic (fixed LHS seed + fixed run-seed scheme) and were verified to pass on this machine with margin (well-specified MSSR 1.735 vs the 2.5 ceiling; misspecified 6.31). GP marginal-likelihood optimization is far more numerically stable across CPU architectures than the chaotic ecosystem simulation that caused the earlier trophic-cascade CI divergence, so the risk of an arch-dependent flip is low; if one ever does flip, increase the fixture's `n_points` (tightens MSSR toward 1) rather than loosening the gate thresholds.

## Self-Review (completed during authoring)

- **Spec coverage (Phase 1 rows):** `design.py` seed loop + SSB enabled → Task 4 (`run_design`, `make_engine_evaluator`); probabilistic coverage/PIT gate → Task 3; bounded design-growth loop → Task 5; well-specified synthetic validation → Task 3 + Task 5; the `cross_validate` variance-return the gate needs → Task 1; natural-log Y / ddof=1 alpha / per-key censoring / two-transform separation → Global Constraints + Task 4.
- **Deferred to later phases (correctly):** prior/likelihood/posterior/sampler/predictive (`prior.py`, `likelihood.py`, `posterior.py`, `sampler.py`, `predictive.py`, `run.py`), the `[uq]` extra deps, the sampler-adequacy + dimension-cap envelope enforcement, δ(θ) — all Phase 2/3.
- **Type consistency:** `Evaluator` signature `(np.ndarray, int) -> dict[str, float]` is identical across `run_design`, `make_engine_evaluator`, and the growth loop; `GateReport`/`DesignResult`/`GrowthResult` field names match between producer and test; `cross_validate`'s new `test_idx` key is consumed by the gate under that exact name.
- **Placeholder scan:** none — every code step contains complete, runnable content. Gate thresholds and synthetic parameters are the verified prototype values.

## Execution Handoff

Two execution options:
1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks. Note Tasks 3–5 include real-GP synthetic tests (a few seconds each) and Task 4 runs one short engine simulation.
2. **Inline Execution** — execute tasks in this session with checkpoints.
