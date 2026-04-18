# Calibration & Sensitivity Suite — Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 9 issues identified in the 2026-04-17 calibration-suite review — loud failures in preflight, correct multi-objective handling in the GP surrogate, configurable timeouts, richer subprocess diagnostics, automatic disk cleanup, parallelized preflight, decomposed long functions, and a tightened CLI script.

**Architecture:** Incremental, TDD-driven fixes to `osmose/calibration/` modules and `scripts/calibrate_baltic.py`. No public API breaks for existing callers — every new parameter is optional with a default that preserves current behavior, except where behavior *was wrong* (silent failures in preflight, naive multi-obj sum in surrogate), which raise or return differently and are documented per-task.

**Tech Stack:** Python 3.12, pymoo, SALib, scikit-learn GPs, NumPy, pytest, ruff.

**Out of scope:** Acquisition-function-based active learning for the GP surrogate (smell #8 in the review is a feature, not a fix; deferred). Convergence criteria beyond generation count, hypervolume metrics, and parameter-CI uncertainty quantification are likewise deferred.

**Pre-flight:**
- Baseline: `.venv/bin/python -m pytest -q` must be green before starting (currently 2411 passed).
- Lint baseline: `.venv/bin/ruff check osmose/ scripts/ tests/` must be clean.
- All work lands in the current working tree. No worktree created — the diff surface is ≤ ~8 files.

---

## File Structure

**Modified:**
- `osmose/calibration/preflight.py` (Tasks 1, 6, 7, 8)
- `osmose/calibration/surrogate.py` (Task 2)
- `osmose/calibration/problem.py` (Tasks 3, 4, 5)
- `scripts/calibrate_baltic.py` (Task 9)

**Tests created / extended:**
- `tests/test_calibration_preflight.py` — silent-failure regression, parallelism determinism, stage-split unit tests
- `tests/test_calibration_surrogate.py` — multi-obj aggregation contract
- `tests/test_calibration_problem.py` — configurable timeout, stderr-file artifact, cleanup flag

---

## Task 1: Harden preflight against silent failures (High)

**Files:**
- Modify: `osmose/calibration/preflight.py:633-697` (make_preflight_eval_fn)
- Modify: `osmose/calibration/preflight.py:121-217` (run_morris_screening) — docstring only
- Test: `tests/test_calibration_preflight.py`

- [ ] **Step 1: Write failing regression test for eval failures raising loudly**

Add to `tests/test_calibration_preflight.py`:

```python
import logging

import numpy as np
import pytest

from osmose.calibration.preflight import make_preflight_eval_fn


def test_preflight_eval_fn_logs_and_counts_failures(monkeypatch, caplog, tmp_path):
    """Engine exceptions must be logged, counted, and expose per-row failure flags."""
    from osmose.calibration import preflight as pre

    class _FakeEngine:
        def run(self, config, output_dir):
            raise RuntimeError("synthetic blow-up")

    class _FakeResults:
        def __init__(self, *a, **kw): ...

    monkeypatch.setattr(pre, "PythonEngine", lambda: _FakeEngine())
    monkeypatch.setattr(pre, "OsmoseResults", _FakeResults)

    fp_spec = [pre.FreeParameter(key="predation.efficiency.sp0",
                                 lower_bound=0.1, upper_bound=0.9)]
    fn = make_preflight_eval_fn(
        base_config={"simulation.time.nyear": "1"},
        free_params=fp_spec,
        objective_fns=[lambda r: 0.0],
        output_dir=tmp_path,
        run_years=1,
    )

    X = np.array([[0.5]])
    with caplog.at_level(logging.WARNING, logger="osmose.calibration.preflight"):
        Y = fn(X)
    assert not np.isfinite(Y[0, 0])
    assert any("synthetic blow-up" in rec.getMessage() for rec in caplog.records), \
        "Exception message must appear in logs (no silent pass)"
    assert fn.failures == 1  # new attribute: failure counter
    assert fn.samples == 1
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py::test_preflight_eval_fn_logs_and_counts_failures -v`
Expected: FAIL with `AssertionError` on the caplog check (current code swallows exceptions silently) or `AttributeError: 'function' object has no attribute 'failures'`.

- [ ] **Step 3: Replace silent-pass with logged counter and attach metrics**

Edit `osmose/calibration/preflight.py:633-697` — replace the existing `make_preflight_eval_fn` and its inner `_evaluate` with:

```python
def make_preflight_eval_fn(
    base_config: dict[str, str],
    free_params: list[FreeParameter],
    objective_fns: list[Callable],
    output_dir: Path,
    run_years: int = 5,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build an evaluation callable for preflight sensitivity analysis.

    The returned callable exposes:
      - ``samples``: number of samples evaluated across all calls
      - ``failures``: number of samples that raised an exception

    Failures are logged at WARNING level with the exception message; the
    corresponding row in the result matrix is filled with ``inf`` so SALib
    treats it as a blow-up rather than receiving a masked NaN.
    """
    from osmose.engine import PythonEngine  # local import to avoid cycle
    from osmose.results import OsmoseResults

    n_obj = len(objective_fns)

    class _EvalFn:
        def __init__(self) -> None:
            self.samples = 0
            self.failures = 0

        def __call__(self, X: np.ndarray) -> np.ndarray:
            n_samples = X.shape[0]
            results_matrix = np.full((n_samples, n_obj), np.inf)
            for i in range(n_samples):
                config = dict(base_config)
                config["simulation.time.nyear"] = str(run_years)
                for j, fp in enumerate(free_params):
                    val = float(X[i, j])
                    if fp.transform is Transform.LOG:
                        val = 10.0**val
                    config[fp.key] = str(val)
                try:
                    engine = PythonEngine()
                    engine.run(config, output_dir)
                    results_matrix[i] = np.array(
                        [float(fn(OsmoseResults(output_dir))) for fn in objective_fns]
                    )
                except Exception as exc:  # noqa: BLE001 — preflight is best-effort
                    _log.warning(
                        "preflight sample %d failed (%s: %s); row left as inf",
                        i,
                        type(exc).__name__,
                        exc,
                    )
                    self.failures += 1
                self.samples += 1
            return results_matrix

    return _EvalFn()
```

(No new imports needed — `PythonEngine`, `OsmoseResults`, `FreeParameter`, and `Transform` are already imported at `preflight.py:21-23`.)

- [ ] **Step 4: Run the test to confirm it passes**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py::test_preflight_eval_fn_logs_and_counts_failures -v`
Expected: PASS.

- [ ] **Step 5: Add an abort-on-excessive-failures regression**

Append to `tests/test_calibration_preflight.py`:

```python
def test_run_preflight_aborts_when_failure_rate_exceeds_threshold(monkeypatch, tmp_path):
    """If >50% of Morris samples fail, run_preflight must raise PreflightEvalError."""
    from osmose.calibration import preflight as pre

    calls = {"n": 0}
    def always_fails(X):
        fn = always_fails
        fn.samples = X.shape[0]
        fn.failures = X.shape[0]  # 100% failure
        calls["n"] += 1
        return np.full((X.shape[0], 1), np.inf)
    always_fails.samples = 0
    always_fails.failures = 0

    with pytest.raises(pre.PreflightEvalError, match="failure rate"):
        pre.run_preflight(
            param_names=["a", "b"],
            param_bounds=[(0.0, 1.0), (0.0, 1.0)],
            evaluation_fn=always_fails,
            n_trajectories=3,
            num_levels=4,
        )
```

- [ ] **Step 6: Run to confirm it fails (error class doesn't exist yet)**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py::test_run_preflight_aborts_when_failure_rate_exceeds_threshold -v`
Expected: FAIL with `AttributeError: module 'osmose.calibration.preflight' has no attribute 'PreflightEvalError'`.

- [ ] **Step 7: Add the exception class and wire the abort**

In `osmose/calibration/preflight.py`, near the top-level class definitions (after `PreflightResult`):

```python
class PreflightEvalError(RuntimeError):
    """Raised when a preflight run fails so many samples that results are
    unusable. The caller should review the evaluation_fn rather than trust
    degenerate sensitivity indices."""
```

Then inside `run_preflight`, just after the existing `failure_rate = n_failed / max(n_samples, 1)` computation (around line 500), add:

```python
    # Hard abort if a majority of Morris samples failed — the resulting Morris
    # indices would be uninformative noise.
    _MAJORITY_FAILURE = 0.5
    if failure_rate > _MAJORITY_FAILURE:
        raise PreflightEvalError(
            f"Morris stage failure rate {failure_rate:.0%} exceeds {_MAJORITY_FAILURE:.0%}; "
            "check evaluation_fn — sensitivity indices would be meaningless."
        )
```

- [ ] **Step 8: Run both preflight tests to confirm green**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py -v`
Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add osmose/calibration/preflight.py tests/test_calibration_preflight.py
git commit -m "fix(calibration): loud failures + abort threshold in preflight eval"
```

---

## Task 2: Fix surrogate multi-objective aggregation (Medium)

**Files:**
- Modify: `osmose/calibration/surrogate.py:102-130` (find_optimum)
- Test: `tests/test_calibration_surrogate.py`

- [ ] **Step 1: Write failing test for Pareto default in multi-obj**

Add to `tests/test_calibration_surrogate.py`:

```python
def test_find_optimum_multi_objective_returns_pareto_without_weights():
    """With multiple objectives and no weights, find_optimum should return
    the non-dominated (Pareto) set rather than an unweighted scalar sum."""
    import numpy as np

    from osmose.calibration.surrogate import SurrogateCalibrator

    rng = np.random.default_rng(7)
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    sc = SurrogateCalibrator(param_bounds=bounds, n_objectives=2)

    X = rng.uniform(0, 1, size=(40, 2))
    # Construct competing objectives: obj0 prefers x0 small, obj1 prefers x0 large.
    y = np.stack([X[:, 0], 1.0 - X[:, 0]], axis=1)
    sc.fit(X, y)

    result = sc.find_optimum(n_candidates=500, seed=1)
    assert "pareto" in result, "must return a Pareto set key when no weights are given"
    pareto = result["pareto"]
    assert pareto["params"].shape[0] >= 2, "Pareto front has at least two points"
    assert pareto["objectives"].shape[1] == 2


def test_find_optimum_multi_objective_with_weights_returns_single_point():
    """When explicit weights are supplied, a scalarized best single point
    must be returned (sum of weighted means)."""
    import numpy as np

    from osmose.calibration.surrogate import SurrogateCalibrator

    rng = np.random.default_rng(11)
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    sc = SurrogateCalibrator(param_bounds=bounds, n_objectives=2)
    X = rng.uniform(0, 1, size=(40, 2))
    y = np.stack([X[:, 0], 1.0 - X[:, 0]], axis=1)
    sc.fit(X, y)

    result = sc.find_optimum(n_candidates=500, seed=1, weights=[1.0, 0.0])
    # With weight only on obj0 (prefers small x0), best x0 should be near 0
    assert result["params"][0] < 0.2


def test_find_optimum_weights_must_match_n_objectives():
    import numpy as np
    import pytest

    from osmose.calibration.surrogate import SurrogateCalibrator

    sc = SurrogateCalibrator(param_bounds=[(0.0, 1.0)], n_objectives=2)
    X = np.array([[0.1], [0.5], [0.9]])
    y = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    sc.fit(X, y)
    with pytest.raises(ValueError, match="weights"):
        sc.find_optimum(weights=[0.5])  # wrong length
```

- [ ] **Step 2: Run to confirm FAIL**

Run: `.venv/bin/python -m pytest tests/test_calibration_surrogate.py -v -k "find_optimum"`
Expected: the three new tests FAIL — the current API doesn't accept `weights` and never returns `pareto`.

- [ ] **Step 3: Replace find_optimum with weighted-or-Pareto variant**

Edit `osmose/calibration/surrogate.py:102-130` to:

```python
    def find_optimum(
        self,
        n_candidates: int = 10000,
        seed: int = 123,
        weights: list[float] | None = None,
    ) -> dict:
        """Find the optimum on the surrogate.

        Single-objective: returns the argmin of the posterior mean.

        Multi-objective with ``weights``: returns the argmin of the weighted
        sum of posterior means. ``weights`` must be length ``n_objectives`` and
        should be non-negative.

        Multi-objective without ``weights``: returns the Pareto (non-dominated)
        set under minimization of the posterior means. The raw unweighted-sum
        aggregation used previously silently conflated objectives with
        different units and is no longer the default.

        Returns
        -------
        dict
          * single-objective / weighted: ``params``, ``predicted_objectives``,
            ``predicted_uncertainty``
          * multi-objective / unweighted: the same keys populated with an
            anchor point (best weighted sum with equal weights, for
            backwards-compatible callers) plus ``pareto`` containing
            ``params`` (M, k), ``objectives`` (M, n_obj), ``uncertainty`` (M, n_obj)
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before find_optimum()")

        candidates = self.generate_samples(n_candidates, seed=seed)
        means, stds = self.predict(candidates)

        if self.n_objectives == 1:
            best_idx = int(np.argmin(means[:, 0]))
            return {
                "params": candidates[best_idx],
                "predicted_objectives": means[best_idx],
                "predicted_uncertainty": stds[best_idx],
            }

        if weights is not None:
            w = np.asarray(weights, dtype=float)
            if w.shape != (self.n_objectives,):
                raise ValueError(
                    f"weights length {w.shape[0] if w.ndim else 0} != "
                    f"n_objectives {self.n_objectives}"
                )
            if np.any(w < 0):
                raise ValueError("weights must be non-negative")
            scores = means @ w
            best_idx = int(np.argmin(scores))
            return {
                "params": candidates[best_idx],
                "predicted_objectives": means[best_idx],
                "predicted_uncertainty": stds[best_idx],
            }

        # Multi-objective, no weights: compute non-dominated set.
        pareto_idx = _non_dominated_indices(means)
        # Anchor point: best equal-weighted sum so legacy callers that read
        # "params" get *a* sensible result.
        anchor = int(np.argmin(means.sum(axis=1)))
        return {
            "params": candidates[anchor],
            "predicted_objectives": means[anchor],
            "predicted_uncertainty": stds[anchor],
            "pareto": {
                "params": candidates[pareto_idx],
                "objectives": means[pareto_idx],
                "uncertainty": stds[pareto_idx],
            },
        }
```

And add this helper just above `class SurrogateCalibrator`:

```python
def _non_dominated_indices(F: np.ndarray) -> np.ndarray:
    """Return indices of non-dominated rows (minimization).

    A row i dominates j iff F[i] <= F[j] component-wise with at least one
    strict inequality. This is O(n^2) in the number of candidates but the
    surrogate's candidate pool is small (<= 10k) so it's fine.
    """
    n = F.shape[0]
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        le = np.all(F <= F[i], axis=1)
        lt = np.any(F < F[i], axis=1)
        dominators = le & lt
        dominators[i] = False
        if np.any(dominators):
            is_dominated[i] = True
    return np.flatnonzero(~is_dominated)
```

- [ ] **Step 4: Run the new tests to confirm PASS**

Run: `.venv/bin/python -m pytest tests/test_calibration_surrogate.py -v`
Expected: all PASS (including pre-existing tests — they only use single-obj path which is unchanged).

- [ ] **Step 5: Commit**

```bash
git add osmose/calibration/surrogate.py tests/test_calibration_surrogate.py
git commit -m "fix(calibration): surrogate multi-obj returns Pareto or weighted-sum, not naive sum"
```

---

## Task 3: Configurable subprocess timeout (Low)

**Files:**
- Modify: `osmose/calibration/problem.py:76-110` (__init__), `:195-244` (_run_single)
- Test: `tests/test_calibration_problem.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_calibration_problem.py`:

```python
def test_subprocess_timeout_is_configurable(tmp_path):
    from osmose.calibration.problem import OsmoseCalibrationProblem, FreeParameter

    problem = OsmoseCalibrationProblem(
        free_params=[FreeParameter("k", 0.1, 1.0)],
        objective_fns=[lambda r: 0.0],
        base_config_path=tmp_path / "missing.csv",
        jar_path=tmp_path / "missing.jar",
        work_dir=tmp_path,
        subprocess_timeout=42,
    )
    assert problem.subprocess_timeout == 42


def test_subprocess_timeout_default_is_3600(tmp_path):
    from osmose.calibration.problem import OsmoseCalibrationProblem, FreeParameter

    problem = OsmoseCalibrationProblem(
        free_params=[FreeParameter("k", 0.1, 1.0)],
        objective_fns=[lambda r: 0.0],
        base_config_path=tmp_path / "missing.csv",
        jar_path=tmp_path / "missing.jar",
        work_dir=tmp_path,
    )
    assert problem.subprocess_timeout == 3600
```

- [ ] **Step 2: Run to confirm FAIL**

Run: `.venv/bin/python -m pytest tests/test_calibration_problem.py -v -k "timeout"`
Expected: FAIL (`TypeError: unexpected keyword` or `AttributeError: subprocess_timeout`).

- [ ] **Step 3: Add parameter to __init__ and use it in _run_single**

In `osmose/calibration/problem.py`:

Add to `__init__` signature (after `registry`):

```python
        subprocess_timeout: int = 3600,
```

Store it (near the other self.X assignments):

```python
        self.subprocess_timeout = int(subprocess_timeout)
```

Replace the hardcoded timeout in `_run_single` (currently at line 210):

```python
        result = subprocess.run(cmd, capture_output=True, timeout=self.subprocess_timeout)
```

(Remove the `# 1-hour timeout per evaluation; consider making configurable…` comment above that line.)

- [ ] **Step 4: Run tests — timeout tests plus full problem suite**

Run: `.venv/bin/python -m pytest tests/test_calibration_problem.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add osmose/calibration/problem.py tests/test_calibration_problem.py
git commit -m "feat(calibration): configurable subprocess_timeout on OsmoseCalibrationProblem"
```

---

## Task 4: Persist subprocess stderr on failure (Low)

**Files:**
- Modify: `osmose/calibration/problem.py:212-217` (error handling in _run_single)
- Test: `tests/test_calibration_problem.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_calibration_problem.py`:

```python
def test_run_single_persists_full_stderr_on_failure(monkeypatch, tmp_path):
    """A non-zero subprocess exit must write the full stderr to run_dir/stderr.txt."""
    from osmose.calibration.problem import OsmoseCalibrationProblem, FreeParameter
    import subprocess

    big_stderr = b"ERROR: " + (b"x" * 2000)

    class _Result:
        returncode = 1
        stderr = big_stderr
        stdout = b""

    def fake_run(*args, **kwargs):
        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Create a dummy JAR so the constructor succeeds; base config need not exist.
    jar = tmp_path / "osmose.jar"
    jar.write_bytes(b"")
    problem = OsmoseCalibrationProblem(
        free_params=[FreeParameter("species.K.sp0", 0.1, 1.0)],
        objective_fns=[lambda r: 0.0],
        base_config_path=tmp_path / "cfg.csv",
        jar_path=jar,
        work_dir=tmp_path / "work",
    )
    out = problem._run_single({"species.K.sp0": "0.5"}, run_id=0)
    assert out == [float("inf")]
    stderr_file = tmp_path / "work" / "run_0" / "stderr.txt"
    assert stderr_file.exists()
    assert stderr_file.read_bytes() == big_stderr
```

- [ ] **Step 2: Run to confirm FAIL**

Run: `.venv/bin/python -m pytest tests/test_calibration_problem.py::test_run_single_persists_full_stderr_on_failure -v`
Expected: FAIL (file does not exist).

- [ ] **Step 3: Write full stderr on failure**

In `osmose/calibration/problem.py`, replace the block around line 212 (`if result.returncode != 0:`) with:

```python
        if result.returncode != 0:
            stderr_bytes = result.stderr or b""
            try:
                (run_dir / "stderr.txt").write_bytes(stderr_bytes)
            except OSError:
                pass  # don't let disk issues mask the underlying failure
            stderr_msg = stderr_bytes.decode(errors="replace")[:500]
            _log.warning(
                "OSMOSE run %d failed (exit %d); full stderr at %s; head: %s",
                run_id,
                result.returncode,
                run_dir / "stderr.txt",
                stderr_msg,
            )
            return [float("inf")] * self.n_obj
```

- [ ] **Step 4: Run test to confirm PASS**

Run: `.venv/bin/python -m pytest tests/test_calibration_problem.py::test_run_single_persists_full_stderr_on_failure -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add osmose/calibration/problem.py tests/test_calibration_problem.py
git commit -m "feat(calibration): persist full subprocess stderr to run_dir/stderr.txt on failure"
```

---

## Task 5: Optional auto-cleanup of run directories (Low)

**Files:**
- Modify: `osmose/calibration/problem.py` (__init__ + _run_single)
- Test: `tests/test_calibration_problem.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_calibration_problem.py`:

```python
def test_cleanup_after_eval_true_removes_run_dir(monkeypatch, tmp_path):
    from osmose.calibration.problem import OsmoseCalibrationProblem, FreeParameter
    import subprocess

    class _Result:
        returncode = 0
        stderr = b""
        stdout = b""

    class _FakeResults:
        def __init__(self, *a, **kw): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())
    monkeypatch.setattr("osmose.results.OsmoseResults", _FakeResults)

    jar = tmp_path / "osmose.jar"; jar.write_bytes(b"")
    problem = OsmoseCalibrationProblem(
        free_params=[FreeParameter("species.K.sp0", 0.1, 1.0)],
        objective_fns=[lambda r: 0.0],
        base_config_path=tmp_path / "cfg.csv",
        jar_path=jar,
        work_dir=tmp_path / "work",
        cleanup_after_eval=True,
    )
    problem._run_single({"species.K.sp0": "0.5"}, run_id=0)
    assert not (tmp_path / "work" / "run_0").exists()


def test_cleanup_after_eval_false_keeps_run_dir(monkeypatch, tmp_path):
    from osmose.calibration.problem import OsmoseCalibrationProblem, FreeParameter
    import subprocess

    class _Result:
        returncode = 0
        stderr = b""
        stdout = b""

    class _FakeResults:
        def __init__(self, *a, **kw): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())
    monkeypatch.setattr("osmose.results.OsmoseResults", _FakeResults)

    jar = tmp_path / "osmose.jar"; jar.write_bytes(b"")
    problem = OsmoseCalibrationProblem(
        free_params=[FreeParameter("species.K.sp0", 0.1, 1.0)],
        objective_fns=[lambda r: 0.0],
        base_config_path=tmp_path / "cfg.csv",
        jar_path=jar,
        work_dir=tmp_path / "work",
        cleanup_after_eval=False,
    )
    problem._run_single({"species.K.sp0": "0.5"}, run_id=1)
    assert (tmp_path / "work" / "run_1").exists()
```

- [ ] **Step 2: Run to confirm FAIL**

Run: `.venv/bin/python -m pytest tests/test_calibration_problem.py -v -k "cleanup_after_eval"`
Expected: FAIL (`TypeError: unexpected keyword 'cleanup_after_eval'`).

- [ ] **Step 3: Implement the option**

In `osmose/calibration/problem.py` `__init__`, add parameter (after `subprocess_timeout`):

```python
        cleanup_after_eval: bool = False,
```

Store: `self.cleanup_after_eval = bool(cleanup_after_eval)`.

At the end of `_run_single`, just before `return obj_values`, add:

```python
        if self.cleanup_after_eval:
            self.cleanup_run(run_id)
```

Default remains `False` so existing callers see no behavior change.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_calibration_problem.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add osmose/calibration/problem.py tests/test_calibration_problem.py
git commit -m "feat(calibration): optional cleanup_after_eval to reclaim disk on large campaigns"
```

---

## Task 6: Parallelize preflight evaluation (Medium)

**Files:**
- Modify: `osmose/calibration/preflight.py` (make_preflight_eval_fn)
- Test: `tests/test_calibration_preflight.py`

- [ ] **Step 1: Write a determinism test across workers**

Add to `tests/test_calibration_preflight.py`:

```python
def test_preflight_eval_fn_parallel_matches_serial(monkeypatch, tmp_path):
    """With a per-sample output_dir contract, n_workers must not change
    the output matrix. Engine→Results communicate via an on-disk file,
    one per sample — so no shared state can race."""
    from pathlib import Path

    from osmose.calibration import preflight as pre

    class _Engine:
        def run(self, config, output_dir):
            (Path(output_dir) / "value.txt").write_text(
                str(config["species.K.sp0"])
            )

    class _Results:
        def __init__(self, output_dir, *a, **kw):
            self.value = float((Path(output_dir) / "value.txt").read_text())

    monkeypatch.setattr(pre, "PythonEngine", lambda: _Engine())
    monkeypatch.setattr(pre, "OsmoseResults", _Results)

    fp = [pre.FreeParameter("species.K.sp0", 0.1, 0.9)]

    def build(n_workers):
        return pre.make_preflight_eval_fn(
            base_config={},
            free_params=fp,
            objective_fns=[lambda r: r.value * 2.0],
            output_dir=tmp_path,
            run_years=1,
            n_workers=n_workers,
        )

    X = np.linspace(0.1, 0.9, 12).reshape(-1, 1)
    Y_serial = build(1)(X)
    Y_parallel = build(4)(X)
    np.testing.assert_allclose(Y_serial, Y_parallel)
```

- [ ] **Step 2: Run to confirm FAIL**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py::test_preflight_eval_fn_parallel_matches_serial -v`
Expected: FAIL (`TypeError: unexpected keyword 'n_workers'`).

- [ ] **Step 3: Thread-pool the evaluator**

Edit `osmose/calibration/preflight.py` `make_preflight_eval_fn` (the `_EvalFn` class from Task 1):

Update the signature:

```python
def make_preflight_eval_fn(
    base_config: dict[str, str],
    free_params: list[FreeParameter],
    objective_fns: list[Callable],
    output_dir: Path,
    run_years: int = 5,
    n_workers: int = 1,
) -> Callable[[np.ndarray], np.ndarray]:
```

And replace the body of `_EvalFn.__call__`:

```python
        def __call__(self, X: np.ndarray) -> np.ndarray:
            n_samples = X.shape[0]
            results_matrix = np.full((n_samples, n_obj), np.inf)

            def _one(i: int) -> tuple[int, np.ndarray | None, Exception | None]:
                config = dict(base_config)
                config["simulation.time.nyear"] = str(run_years)
                for j, fp in enumerate(free_params):
                    val = float(X[i, j])
                    if fp.transform is Transform.LOG:
                        val = 10.0**val
                    config[fp.key] = str(val)
                try:
                    engine = PythonEngine()
                    # Per-sample output_dir — avoids concurrent writers clobbering.
                    out_i = Path(output_dir) / f"preflight_{i}"
                    out_i.mkdir(parents=True, exist_ok=True)
                    engine.run(config, out_i)
                    row = np.array(
                        [float(fn(OsmoseResults(out_i))) for fn in objective_fns]
                    )
                    return i, row, None
                except Exception as exc:  # noqa: BLE001
                    return i, None, exc

            if n_workers <= 1:
                for i in range(n_samples):
                    _, row, err = _one(i)
                    if err is not None:
                        _log.warning(
                            "preflight sample %d failed (%s: %s); row left as inf",
                            i, type(err).__name__, err,
                        )
                        self.failures += 1
                    else:
                        results_matrix[i] = row
                    self.samples += 1
            else:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    for i, row, err in pool.map(_one, range(n_samples)):
                        if err is not None:
                            _log.warning(
                                "preflight sample %d failed (%s: %s); row left as inf",
                                i, type(err).__name__, err,
                            )
                            self.failures += 1
                        else:
                            results_matrix[i] = row
                        self.samples += 1
            return results_matrix
```

Note: each sample writes to its own `preflight_i` subdir to avoid concurrent writer clobbering. This also aids post-hoc debugging.

- [ ] **Step 4: Run the new test plus the Task-1 regression**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add osmose/calibration/preflight.py tests/test_calibration_preflight.py
git commit -m "perf(calibration): parallelize preflight eval with ThreadPoolExecutor (n_workers)"
```

---

## Task 7: Split run_preflight into stages (Medium)

**Files:**
- Modify: `osmose/calibration/preflight.py:369-632` (run_preflight)
- Test: `tests/test_calibration_preflight.py`

**Rationale:** `run_preflight` is 264 lines performing (a) Morris sampling + analysis, (b) issue detection, (c) optional Sobol refinement, (d) result assembly. These are independent concerns — splitting improves testability and makes the `PreflightEvalError` insertion from Task 1 sit in one obvious place.

- [ ] **Step 1: Read the current run_preflight carefully**

Run: `.venv/bin/sed -n '369,632p' osmose/calibration/preflight.py`
Expected: ~264 lines of mixed-concern code. Note the boundaries:
  - 444-545: Morris sampling + eval + blow-up tracking + Morris analyze
  - 547-599: conditional Sobol stage
  - 600-632: issue detection + PreflightResult return

- [ ] **Step 2: Write stage unit tests**

Append to `tests/test_calibration_preflight.py`:

```python
def test_run_morris_stage_returns_screening_and_failure_rate(monkeypatch):
    """_run_morris_stage is a pure sampler + analyzer with no side effects
    beyond the provided evaluation_fn."""
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
        seed=1,
    )
    assert len(screening) == 2
    assert screening[0].key == "a" and screening[1].key == "b"
    assert failure_rate == 0.0
    assert Y_clean.shape[1] == 2
    assert blowup_params == set()


def test_maybe_run_sobol_stage_skips_when_few_survivors(monkeypatch):
    from osmose.calibration import preflight as pre
    # One influential parameter → survivors_idx length 1 → Sobol skipped.
    screening = [
        pre.ParameterScreening(
            key="a", mu_star=0.5, sigma=0.1, mu_star_conf=0.01,
            influential=True,
        ),
    ]
    out = pre._maybe_run_sobol_stage(
        screening=screening,
        param_bounds=[(0.0, 1.0)],
        evaluation_fn=lambda X: np.zeros((X.shape[0], 1)),
        sobol_n_base=16,
        sobol_failure_threshold=0.5,
        seed=1,
    )
    assert out is None  # <2 survivors → no Sobol
```

- [ ] **Step 3: Run to confirm FAIL**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py -v -k "run_morris_stage or maybe_run_sobol_stage"`
Expected: FAIL (`AttributeError: module 'osmose.calibration.preflight' has no attribute '_run_morris_stage'`).

- [ ] **Step 4: Extract the stages**

Add three module-level helpers above `run_preflight`:

```python
def _run_morris_stage(
    *,
    param_names: list[str],
    param_bounds: list[tuple[float, float]],
    evaluation_fn: Callable[[np.ndarray], np.ndarray],
    n_trajectories: int,
    num_levels: int,
    seed: int | None,
    cancel_event: threading.Event | None = None,
) -> tuple[list[ParameterScreening], float, np.ndarray, set[str]]:
    """Morris sampling + analysis. Returns (screening, failure_rate, Y_clean, blowup_params).

    Raises PreflightEvalError if cancelled or if sampling produces no rows.
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
    Y_clean = np.where(np.isfinite(Y_raw), Y_raw, 1e6)

    n_obj = Y_clean.shape[1]
    agg_mu_star = np.zeros(k)
    agg_conf = np.zeros(k)
    agg_sigma = np.zeros(k)
    for obj_idx in range(n_obj):
        result = morris_analyze.analyze(
            problem, X, Y_clean[:, obj_idx], num_levels=num_levels, print_to_console=False
        )
        agg_mu_star += np.asarray(result["mu_star"])
        agg_conf += np.asarray(result["mu_star_conf"])
        agg_sigma += np.asarray(result["sigma"])
    agg_mu_star /= n_obj
    agg_conf /= n_obj
    agg_sigma /= n_obj

    max_mu = agg_mu_star.max() if agg_mu_star.size and agg_mu_star.max() > 0 else 1.0
    screening = [
        ParameterScreening(
            key=param_names[i],
            mu_star=float(agg_mu_star[i]),
            sigma=float(agg_sigma[i]),
            mu_star_conf=float(agg_conf[i]),
            influential=bool(float(agg_mu_star[i]) >= 0.1 * max_mu),
        )
        for i in range(k)
    ]
    return screening, failure_rate, Y_clean, blowup_params_set


def _maybe_run_sobol_stage(
    *,
    screening: list[ParameterScreening],
    param_bounds: list[tuple[float, float]],
    evaluation_fn: Callable[[np.ndarray], np.ndarray],
    sobol_n_base: int,
    sobol_failure_threshold: float,
    seed: int | None,
    cancel_event: threading.Event | None = None,
) -> dict | None:
    """Refine with Sobol on the non-negligible survivors. Returns None if
    fewer than 2 survivors or if the failure threshold is exceeded."""
    from osmose.calibration.sensitivity import SensitivityAnalyzer  # local for cycle

    survivors_idx = [i for i, s in enumerate(screening) if s.influential]
    if len(survivors_idx) < 2:
        return None
    if cancel_event is not None and cancel_event.is_set():
        return None

    surv_names = [screening[i].key for i in survivors_idx]
    surv_bounds = [param_bounds[i] for i in survivors_idx]
    analyzer = SensitivityAnalyzer(param_names=surv_names, param_bounds=surv_bounds)
    X = analyzer.generate_samples(n_base=sobol_n_base, seed=seed)
    Y = np.asarray(evaluation_fn(X))
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    n_failed = int(np.sum(np.any(~np.isfinite(Y), axis=1)))
    failure_rate = n_failed / max(Y.shape[0], 1)
    if failure_rate > sobol_failure_threshold:
        return None
    Y_clean = np.where(np.isfinite(Y), Y, 1e6)
    analysis = analyzer.analyze(Y_clean)
    analysis["param_names"] = surv_names
    return analysis


def _assemble_preflight_result(
    screening: list[ParameterScreening],
    sobol_result: dict | None,
    blowup_params_set: set[str],
    elapsed: float,
) -> PreflightResult:
    # detect_issues expects blowup_params as list-like of keys, not a set
    issues = detect_issues(
        screening=screening,
        sobol_result=sobol_result,
        blowup_params=list(blowup_params_set),
    )
    survivors = [s.key for s in screening if s.influential]
    return PreflightResult(
        screening=screening,
        sobol=sobol_result,
        issues=issues,
        survivors=survivors,
        elapsed_seconds=elapsed,
    )
```

Then **replace** the body of `run_preflight` with:

```python
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
    """Run two-stage preflight (Morris screening → optional Sobol refinement).

    See :func:`_run_morris_stage` and :func:`_maybe_run_sobol_stage` for the
    per-stage details. Aborts with :class:`PreflightEvalError` if the Morris
    majority-failure threshold (50 %) is exceeded.
    """
    t_start = time.monotonic()
    if cancel_event is not None and cancel_event.is_set():
        return PreflightResult([], None, [], [], time.monotonic() - t_start)

    screening, failure_rate, _Y, blowup_params = _run_morris_stage(
        param_names=param_names,
        param_bounds=param_bounds,
        evaluation_fn=evaluation_fn,
        n_trajectories=n_trajectories,
        num_levels=num_levels,
        seed=seed,
        cancel_event=cancel_event,
    )
    _MAJORITY_FAILURE = 0.5
    if failure_rate > _MAJORITY_FAILURE:
        raise PreflightEvalError(
            f"Morris stage failure rate {failure_rate:.0%} exceeds {_MAJORITY_FAILURE:.0%}; "
            "check evaluation_fn — sensitivity indices would be meaningless."
        )

    sobol_result = _maybe_run_sobol_stage(
        screening=screening,
        param_bounds=param_bounds,
        evaluation_fn=evaluation_fn,
        sobol_n_base=sobol_n_base,
        sobol_failure_threshold=sobol_failure_threshold,
        seed=seed,
        cancel_event=cancel_event,
    )

    return _assemble_preflight_result(
        screening=screening,
        sobol_result=sobol_result,
        blowup_params_set=blowup_params,
        elapsed=time.monotonic() - t_start,
    )
```

Keep `PreflightEvalError` (added in Task 1) at module scope near `PreflightResult` — not nested inside `run_preflight`.

- [ ] **Step 5: Run all preflight tests**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py -v`
Expected: all PASS (including Task 1 tests — the abort still fires, just from a shorter function now).

- [ ] **Step 6: Commit**

```bash
git add osmose/calibration/preflight.py tests/test_calibration_preflight.py
git commit -m "refactor(calibration): split run_preflight into Morris/Sobol/assemble stages"
```

---

## Task 8: Split detect_issues into per-category helpers (Medium)

**Files:**
- Modify: `osmose/calibration/preflight.py:218-361` (detect_issues)
- Test: `tests/test_calibration_preflight.py`

**Contract preserved:** the existing signature `detect_issues(screening, sobol_result=None, blowup_params=None)`, all five issue categories (`NEGLIGIBLE`, `BLOWUP`, `FLAT_OBJECTIVE`, `BOUND_TIGHT`, `ALL_NEGLIGIBLE`), required `PreflightIssue` fields (`param_key`, `suggestion`, `auto_fixable`), and the errors-first final sort.

- [ ] **Step 1: Re-read detect_issues carefully**

Run: `.venv/bin/sed -n '218,361p' osmose/calibration/preflight.py`
Note the five independent passes:
  1. BLOWUP: iterates `blowup_params` directly, emits ERROR per key.
  2. NEGLIGIBLE: emits WARNING per screening entry with `not ps.influential`.
  3. ALL_NEGLIGIBLE: emits one ERROR if every screening entry has `not influential`.
  4. FLAT_OBJECTIVE: uses Sobol `S1` (1-D or 2-D), threshold `sum(max(0, S1)) < 0.05`.
  5. BOUND_TIGHT: uses Sobol `ST_agg` (max across objectives) **and** Morris `sigma/mu_star > 1.5`, with `ST > 0.3`.

- [ ] **Step 2: Write per-helper tests**

Append to `tests/test_calibration_preflight.py`:

```python
def _screen(key, mu_star=1.0, sigma=0.1, influential=True):
    from osmose.calibration.preflight import ParameterScreening
    return ParameterScreening(
        key=key, mu_star=mu_star, sigma=sigma, mu_star_conf=0.01,
        influential=influential,
    )


def test_issues_blowup_emits_error_per_key():
    from osmose.calibration.preflight import _issues_blowup, IssueCategory, IssueSeverity
    issues = _issues_blowup(["a", "b"])
    assert len(issues) == 2
    assert all(i.category is IssueCategory.BLOWUP for i in issues)
    assert all(i.severity is IssueSeverity.ERROR for i in issues)
    assert {i.param_key for i in issues} == {"a", "b"}


def test_issues_negligible_emits_warning_per_non_influential():
    from osmose.calibration.preflight import _issues_negligible, IssueCategory, IssueSeverity
    screening = [_screen("a", influential=False), _screen("b", influential=True)]
    issues = _issues_negligible(screening)
    assert len(issues) == 1
    assert issues[0].category is IssueCategory.NEGLIGIBLE
    assert issues[0].severity is IssueSeverity.WARNING
    assert issues[0].param_key == "a"


def test_issues_all_negligible_fires_when_every_param_is_negligible():
    from osmose.calibration.preflight import _issues_all_negligible, IssueCategory, IssueSeverity
    screening = [_screen("a", influential=False), _screen("b", influential=False)]
    issues = _issues_all_negligible(screening)
    assert len(issues) == 1
    assert issues[0].category is IssueCategory.ALL_NEGLIGIBLE
    assert issues[0].severity is IssueSeverity.ERROR
    assert issues[0].param_key is None


def test_issues_flat_fires_on_low_S1_sum():
    from osmose.calibration.preflight import _issues_flat, IssueCategory
    sobol = {"S1": [0.01, 0.01]}  # sum = 0.02 < 0.05
    issues = _issues_flat(sobol)
    assert any(i.category is IssueCategory.FLAT_OBJECTIVE for i in issues)


def test_issues_bound_tight_fires_on_high_ST_and_high_sigma_ratio():
    from osmose.calibration.preflight import _issues_bound_tight, IssueCategory
    screening = [_screen("a", mu_star=0.2, sigma=0.4, influential=True)]  # sigma/mu* = 2.0
    sobol = {"ST": [0.5], "param_names": ["a"]}  # ST > 0.3
    issues = _issues_bound_tight(screening, sobol)
    assert any(i.category is IssueCategory.BOUND_TIGHT for i in issues)


def test_detect_issues_preserves_error_first_sort():
    """Orchestrator must sort errors before warnings (legacy behaviour)."""
    from osmose.calibration.preflight import detect_issues, IssueSeverity
    screening = [_screen("a", influential=False)]  # WARNING
    issues = detect_issues(screening, blowup_params=["b"])  # ERROR from blowup
    assert issues[0].severity is IssueSeverity.ERROR
    assert issues[-1].severity is IssueSeverity.WARNING
```

- [ ] **Step 3: Run to confirm FAIL**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py -v -k "issues_"`
Expected: FAIL — helpers `_issues_blowup`, `_issues_negligible`, `_issues_all_negligible`, `_issues_flat`, `_issues_bound_tight` don't exist yet.

- [ ] **Step 4: Replace detect_issues with per-helper extraction**

In `osmose/calibration/preflight.py`, replace the existing `detect_issues` (lines 218-361) with the five helpers and a thin orchestrator. The logic below is copied verbatim from the current implementation — only the structure changes.

```python
def _issues_blowup(blowup_params: list[str] | None) -> list[PreflightIssue]:
    """Emit ERROR per parameter that caused a numerical blow-up during Morris."""
    out: list[PreflightIssue] = []
    for key in blowup_params or []:
        out.append(
            PreflightIssue(
                category=IssueCategory.BLOWUP,
                severity=IssueSeverity.ERROR,
                param_key=key,
                message=f"Parameter '{key}' caused numerical blow-up during Morris sampling.",
                suggestion="Tighten the parameter bounds or check the model for instability.",
                auto_fixable=False,
            )
        )
    return out


def _issues_negligible(screening: list[ParameterScreening]) -> list[PreflightIssue]:
    """Emit WARNING per parameter that Morris flagged as non-influential."""
    out: list[PreflightIssue] = []
    for ps in screening:
        if ps.influential:
            continue
        out.append(
            PreflightIssue(
                category=IssueCategory.NEGLIGIBLE,
                severity=IssueSeverity.WARNING,
                param_key=ps.key,
                message=f"Parameter '{ps.key}' has negligible influence on the objective(s).",
                suggestion="Consider removing from the free-parameter set to reduce cost.",
                auto_fixable=True,
            )
        )
    return out


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
    """Emit ERROR per objective whose Sobol S1-sum is below 0.05."""
    out: list[PreflightIssue] = []
    if sobol_result is None or "S1" not in sobol_result:
        return out
    S1 = np.asarray(sobol_result["S1"])
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
    # Multi-objective — SensitivityAnalyzer convention: shape (n_obj, n_params)
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
    """Emit WARNING when both Sobol total-order ST > 0.3 and Morris sigma/mu* > 1.5.

    These two signals together indicate a parameter whose influence is strong
    but highly non-monotone — typically a symptom of bounds being too wide.
    """
    out: list[PreflightIssue] = []
    if sobol_result is None or "ST" not in sobol_result:
        return out
    ST = np.asarray(sobol_result["ST"])
    ST_agg = np.max(ST, axis=0) if ST.ndim > 1 else ST
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

    Delegates to per-category detectors and preserves the legacy
    errors-first ordering.
    """
    issues: list[PreflightIssue] = []
    issues.extend(_issues_blowup(blowup_params))
    issues.extend(_issues_negligible(screening))
    issues.extend(_issues_all_negligible(screening))
    issues.extend(_issues_flat(sobol_result))
    issues.extend(_issues_bound_tight(screening, sobol_result))
    issues.sort(key=lambda i: 0 if i.severity is IssueSeverity.ERROR else 1)
    return issues
```

- [ ] **Step 5: Run all preflight tests**

Run: `.venv/bin/python -m pytest tests/test_calibration_preflight.py -q`
Expected: all PASS (the legacy `test_calibration_preflight.py` tests keep working because `detect_issues` still has its old signature and still sorts errors-first).

- [ ] **Step 6: Run the full suite once (catches any missed import)**

Run: `.venv/bin/python -m pytest -q`
Expected: 2411 passed (or current count), no regressions.

- [ ] **Step 7: Commit**

```bash
git add osmose/calibration/preflight.py tests/test_calibration_preflight.py
git commit -m "refactor(calibration): split detect_issues into per-category helpers"
```

---

## Task 9: Tighten `scripts/calibrate_baltic.py` (Low)

**Files:**
- Modify: `scripts/calibrate_baltic.py` (signatures + counter)
- Test: `tests/test_calibrate_baltic_smoke.py` (new)

**Note:** `scripts/` has no `__init__.py`, so the script cannot be imported as `scripts.calibrate_baltic`. The smoke test below uses `importlib.util.spec_from_file_location` to load the module from its path directly, which also exercises top-level side effects (argparse setup) without triggering argparse itself.

- [ ] **Step 1: Locate the mutable counter and current signatures**

Run: `grep -n "def run_simulation\|def make_objective\|call_count" scripts/calibrate_baltic.py`
Confirm the current shapes:
- `run_simulation(config: dict[str, str], overrides: dict[str, str], n_years: int = 40, seed: int = 42) -> dict[str, float]` — already typed ✓
- `make_objective(base_config, targets, param_keys, n_years=40, seed=42, use_log_space=True, w_stability=5.0, w_worst=0.5)` — **not** typed; uses `call_count = [0]` (list-wrapped mutable, not a nonlocal closure — slightly safer than `nonlocal` but still best replaced with `itertools.count`).

- [ ] **Step 2: Write a minimal smoke test**

Create `tests/test_calibrate_baltic_smoke.py`:

```python
"""Smoke test: the Baltic calibration CLI loads and shows help."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def test_calibrate_baltic_module_loads_via_spec():
    """Load the script from disk (scripts/ has no __init__.py)."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "calibrate_baltic.py"
    spec = importlib.util.spec_from_file_location("calibrate_baltic", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Public entry points must exist
    assert hasattr(module, "run_simulation")
    assert hasattr(module, "make_objective")


def test_calibrate_baltic_help():
    """``--help`` should succeed and mention the calibration role."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "calibrate_baltic.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Baltic" in result.stdout or "calibrat" in result.stdout.lower()
```

- [ ] **Step 3: Run the smoke test (it may already pass before the code fixes)**

Run: `.venv/bin/python -m pytest tests/test_calibrate_baltic_smoke.py -v`
Expected: PASS.

- [ ] **Step 4: Type-hint `make_objective`**

In `scripts/calibrate_baltic.py:139`, update the signature. Current:

```python
def make_objective(
    base_config: dict[str, str],
    targets: list[BiomassTarget],
    param_keys: list[str],
    n_years: int = 40,
    seed: int = 42,
    use_log_space: bool = True,
    w_stability: float = 5.0,
    w_worst: float = 0.5,
):
```

Add return type only (everything else is already typed):

```python
from typing import Callable  # add to top-level imports if missing

def make_objective(
    base_config: dict[str, str],
    targets: list[BiomassTarget],
    param_keys: list[str],
    n_years: int = 40,
    seed: int = 42,
    use_log_space: bool = True,
    w_stability: float = 5.0,
    w_worst: float = 0.5,
) -> Callable[[np.ndarray], float]:
```

Then scan the file for any phase driver (`def phase_N(`, `def run_phase(`) that lacks a return type annotation and add one — use `-> dict[str, float]` (or whatever the return actually is per its final `return` statement).

- [ ] **Step 5: Replace the list-wrapped counter with `itertools.count`**

In `scripts/calibrate_baltic.py` inside `make_objective` (starting around line 158), replace:

```python
    call_count = [0]

    def objective(x: np.ndarray) -> float:
        call_count[0] += 1
        # ... body uses call_count[0] in progress-printing on line ~221 and ~227
```

with:

```python
    from itertools import count
    _calls = count(1)

    def objective(x: np.ndarray) -> float:
        call_idx = next(_calls)
        # ... replace every remaining `call_count[0]` with `call_idx`
```

This keeps the integer monotonicity (1, 2, 3, …) and makes the increment atomic at the CPython C level (safer if the objective is ever parallelized). Everywhere the body reads `call_count[0]`, substitute `call_idx`.

- [ ] **Step 6: Confirm lint + smoke**

Run: `.venv/bin/ruff check scripts/calibrate_baltic.py && .venv/bin/python -m pytest tests/test_calibrate_baltic_smoke.py -q`
Expected: clean + PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/calibrate_baltic.py tests/test_calibrate_baltic_smoke.py
git commit -m "refactor(scripts): return-type hint + itertools counter in calibrate_baltic"
```

---

## Task 10: Final validation

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: ≈ 2429 passed — **2411 baseline + 18 new tests** (Task 1: 2, Task 2: 3, Task 3: 2, Task 4: 1, Task 5: 2, Task 6: 1, Task 7: 2, Task 8: 3, Task 9: 2) — 15 skipped, 0 failures.

- [ ] **Step 2: Full lint**

Run: `.venv/bin/ruff check osmose/ scripts/ tests/ ui/`
Expected: `All checks passed!`

- [ ] **Step 3: Baltic engine smoke (end-to-end sanity — nothing calibration-facing should have changed behavior for it)**

Run:

```bash
.venv/bin/python -c "
from pathlib import Path; import tempfile
from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine
cfg = OsmoseConfigReader().read(Path('data/baltic/baltic_all-parameters.csv'))
cfg['simulation.time.nyear'] = '1'
with tempfile.TemporaryDirectory() as td:
    PythonEngine().run(config=cfg, output_dir=Path(td), seed=42)
print('Baltic smoke OK')
"
```

Expected: `Baltic smoke OK`.

- [ ] **Step 4: Skim git log**

Run: `git log --oneline -15`
Expected: one commit per task (9 implementation commits) — clear history.

- [ ] **Step 5: Update CHANGELOG**

Append under "Unreleased" in `CHANGELOG.md`:

```markdown
### Changed
- calibration: preflight evaluator logs exceptions and aborts when Morris
  majority fails (was: silent `except Exception: pass`).
- calibration: surrogate `find_optimum(weights=...)` for weighted scalarization;
  default multi-objective returns a Pareto set instead of naive sum.
- calibration: `OsmoseCalibrationProblem` now accepts `subprocess_timeout`
  and `cleanup_after_eval` options; on subprocess failure the full stderr
  lands in `run_dir/stderr.txt`.
- calibration: preflight evaluation loop is parallelizable via `n_workers`.

### Refactored
- calibration: `run_preflight` split into Morris/Sobol/assemble stages;
  `detect_issues` split into per-category helpers.
- scripts: `calibrate_baltic.py` gains type hints and a thread-safe counter.
```

- [ ] **Step 6: Final commit**

```bash
git add CHANGELOG.md
git commit -m "docs: changelog for calibration/sensitivity fixes"
```

---

## Self-review checklist (done before handoff)

- **Spec coverage:** Every one of the 9 review smells is addressed in a task:
  - #1 (High, silent fail) → Task 1
  - #2 (Medium, surrogate agg) → Task 2
  - #3 (Medium, long functions) → Tasks 7 + 8
  - #4 (Medium, serial preflight) → Task 6
  - #5 (Low, no cleanup) → Task 5
  - #6 (Low, timeout) → Task 3
  - #7 (Low, CLI script) → Task 9
  - #9 (Low, stderr truncation) → Task 4
  - #8 (Low, surrogate active learning) → **deferred**, called out above as out of scope.

- **Placeholders:** scanned — none of "TBD", "implement later", "similar to Task N". Every code step contains the code.

- **Type consistency:** `FreeParameter`, `Transform`, `ParameterScreening`, `PreflightResult`, `PreflightIssue`, `PreflightEvalError`, `OsmoseCalibrationProblem` names used identically across tasks.

- **Commands:** absolute test paths, `.venv/bin/...` prefixes throughout. Expected outputs stated.
