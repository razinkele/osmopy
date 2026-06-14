# NSGA-II ProcessPoolExecutor speedup — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `parallel_backend="process"` to the NSGA-II calibration so per-candidate OSMOSE sims run in separate processes (GIL-free), recovering more than the current 3.02× thread speedup.

**Architecture:** Picklable objective functors (biomass/diet) replace the UI lambdas; a `forkserver` `ProcessPoolExecutor` evaluates candidates via an initializer that reconstructs the problem per worker (registry rebuilt iff the parent had one), reused across generations with broken-pool rebuild/retry; selected at the NSGA-II launch sites with auto thread-fallback when objectives aren't picklable.

**Tech Stack:** Python 3.12, pymoo, `concurrent.futures.ProcessPoolExecutor` (forkserver), pytest. ruff + pyright gates.

**Spec:** `docs/superpowers/specs/2026-06-14-nsga2-process-pool-design.md`

---

## File Structure

- **Modify** `osmose/calibration/objectives.py` — add `BiomassRMSEObjective`/`DietDistanceObjective` (Task 1)
- **Modify** `tests/test_objectives.py` — functor tests (Task 1)
- **Modify** `osmose/calibration/problem.py` — `_EvalSpec`, worker fns, `parallel_backend`, process branch, pool lifecycle (Task 2)
- **Create** `tests/test_nsga2_parallel.py` — parity / broken-pool / worker-count / lifecycle (Task 2)
- **Modify** `ui/pages/calibration_handlers.py` — functors + backend pick + `shutdown_pool()` finally (Task 3)
- **Modify** `scripts/benchmark_calibration.py` — `--backend` switch + finally (Task 3)
- **Modify** `tests/test_app_structure.py` (or a wiring test) + `CHANGELOG.md` (Task 3)

Per-task gate: `.venv/bin/ruff check osmose/ ui/ tests/`, `.venv/bin/ruff format osmose/ ui/ tests/` (**not** app.py), `.venv/bin/pyright <touched files>` (`--pythonpath .venv/bin/python` when app.py is among them).

---

### Task 1: Picklable objective functors

**Files:** Modify `osmose/calibration/objectives.py`; Modify `tests/test_objectives.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_objectives.py`:

```python
import pickle

import pandas as pd

from osmose.calibration.objectives import BiomassRMSEObjective, DietDistanceObjective


class _FakeResults:
    def __init__(self, biomass_df, diet_df):
        self._b, self._d = biomass_df, diet_df

    def biomass(self):
        return self._b

    def diet_matrix(self):
        return self._d


def test_biomass_objective_matches_function_and_pickles():
    obs = pd.DataFrame({"time": [0, 1], "species": ["a", "a"], "biomass": [1.0, 2.0]})
    sim = pd.DataFrame({"time": [0, 1], "species": ["a", "a"], "biomass": [1.5, 2.5]})
    obj = BiomassRMSEObjective(obs)
    from osmose.calibration.objectives import biomass_rmse

    assert obj(_FakeResults(sim, None)) == biomass_rmse(sim, obs)
    assert pickle.loads(pickle.dumps(obj))(_FakeResults(sim, None)) == obj(_FakeResults(sim, None))


def test_diet_objective_matches_function_and_pickles():
    obs = pd.DataFrame({"pred": ["a"], "x": [1.0]})
    sim = pd.DataFrame({"pred": ["a"], "x": [2.0]})
    obj = DietDistanceObjective(obs)
    from osmose.calibration.objectives import diet_distance

    assert obj(_FakeResults(None, sim)) == diet_distance(sim, obs)
    assert pickle.loads(pickle.dumps(obj)) is not None  # round-trips
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_objectives.py -q`
Expected: FAIL (`ImportError: cannot import name 'BiomassRMSEObjective'`).

- [ ] **Step 3: Implement the functors**

Append to `osmose/calibration/objectives.py`:

```python
class BiomassRMSEObjective:
    """Picklable biomass-RMSE objective (wraps `biomass_rmse`; holds the observed frame).

    A module-level class instead of a lambda so it can cross a ProcessPoolExecutor boundary.
    """

    def __init__(self, observed: pd.DataFrame, species: str | None = None):
        self.observed = observed
        self.species = species

    def __call__(self, results) -> float:
        return biomass_rmse(results.biomass(), self.observed, self.species)


class DietDistanceObjective:
    """Picklable diet-distance objective (wraps `diet_distance`; holds the observed matrix)."""

    def __init__(self, observed: pd.DataFrame):
        self.observed = observed

    def __call__(self, results) -> float:
        return diet_distance(results.diet_matrix(), self.observed)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_objectives.py -q`
Expected: PASS.

- [ ] **Step 5: Lint / format / type-check**

`.venv/bin/ruff check osmose/ ui/ tests/`; `.venv/bin/ruff format osmose/ ui/ tests/`; `.venv/bin/pyright osmose/calibration/objectives.py tests/test_objectives.py` → 0 errors.

- [ ] **Step 6: Commit**

```bash
git add osmose/calibration/objectives.py tests/test_objectives.py
git commit -m "feat(calibration): picklable BiomassRMSE/DietDistance objective functors"
```

---

### Task 2: ProcessPoolExecutor backend in `problem.py`

**Files:** Modify `osmose/calibration/problem.py`; Create `tests/test_nsga2_parallel.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nsga2_parallel.py`:

```python
"""Process backend parity + robustness for NSGA-II calibration."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.calibration.objectives import BiomassRMSEObjective
from osmose.calibration.problem import (
    FreeParameter,
    OsmoseCalibrationProblem,
    _resolve_worker_count,
)

_MINIMAL = pytest.importorskip("pathlib").Path(__file__).resolve().parent.parent / "data" / "minimal" / "osm_all-parameters.csv"


def _baseline_obs(work_dir):
    """Run the engine once on the unmodified minimal config → use its biomass as 'observed'
    so the objective is finite (no >50% inf abort)."""
    p = OsmoseCalibrationProblem(
        free_params=[FreeParameter("predation.efficiency.critical", 0.3, 0.5)],
        objective_fns=[lambda r: 0.0],
        base_config_path=_MINIMAL,
        work_dir=work_dir,
    )
    res = p._run_python_engine({}, run_id=0)
    assert res is not None
    with res as r:
        return r.biomass()


def _make_problem(work_dir, obs, backend):
    return OsmoseCalibrationProblem(
        free_params=[FreeParameter("predation.efficiency.critical", 0.3, 0.5)],
        objective_fns=[BiomassRMSEObjective(obs)],
        base_config_path=_MINIMAL,
        work_dir=work_dir,
        n_parallel=2,
        parallel_backend=backend,
    )


@pytest.mark.skipif(not _MINIMAL.exists(), reason="minimal config absent")
def test_thread_process_parity(tmp_path):
    obs = _baseline_obs(tmp_path / "base")
    X = np.array([[0.35], [0.45]])
    pt = _make_problem(tmp_path / "t", obs, "thread")
    pp = _make_problem(tmp_path / "p", obs, "process")
    out_t, out_p = {}, {}
    try:
        pt._evaluate(X, out_t)
        pp._evaluate(X, out_p)
    finally:
        pp.shutdown_pool()
    np.testing.assert_allclose(out_t["F"], out_p["F"])


def test_resolve_worker_count(monkeypatch):
    monkeypatch.delenv("OSMOSE_NSGA2_WORKERS", raising=False)
    assert _resolve_worker_count(4) == 4
    monkeypatch.setenv("OSMOSE_NSGA2_WORKERS", "8")
    assert _resolve_worker_count(4) == 8
    monkeypatch.setenv("OSMOSE_NSGA2_WORKERS", "999")
    assert _resolve_worker_count(4) == 32
    monkeypatch.setenv("OSMOSE_NSGA2_WORKERS", "0")
    assert _resolve_worker_count(4) == 1


def test_shutdown_pool_idempotent(tmp_path):
    p = _make_problem(tmp_path, None, "thread")
    p.shutdown_pool()  # no pool created yet → no-op
    p.shutdown_pool()


def test_broken_pool_recovers(tmp_path, monkeypatch):
    """A submit that raises BrokenProcessPool must not propagate; the candidate is retried
    on a rebuilt pool and ends finite (guards against widening the inner except)."""
    obs = _baseline_obs(tmp_path / "base")
    p = _make_problem(tmp_path / "p", obs, "process")

    real_ensure = p._ensure_pool
    calls = {"n": 0}

    class _BrokenOnce:
        def submit(self, *a, **k):
            raise __import__("concurrent.futures.process", fromlist=["BrokenProcessPool"]).BrokenProcessPool("boom")

    def _ensure():
        calls["n"] += 1
        return _BrokenOnce() if calls["n"] == 1 else real_ensure()

    monkeypatch.setattr(p, "_ensure_pool", _ensure)
    out = {}
    try:
        p._evaluate(np.array([[0.4]]), out)
    finally:
        p.shutdown_pool()
    assert np.isfinite(out["F"]).all()  # rebuilt + retried → finite, not inf
    assert calls["n"] >= 2
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_nsga2_parallel.py -q`
Expected: FAIL (`ImportError: cannot import name '_resolve_worker_count'` / `parallel_backend` unexpected kwarg).

- [ ] **Step 3: Add imports + module-level worker machinery to `problem.py`**

At the top of `osmose/calibration/problem.py`, extend the imports (currently `from concurrent.futures import ThreadPoolExecutor` and `from typing import TYPE_CHECKING, Callable`):

```python
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import TYPE_CHECKING, Callable, Literal

from osmose.schema import build_registry
```

After `_python_engine_errors = ...` (line ~43), add the worker machinery:

```python
@dataclass(frozen=True)
class _EvalSpec:
    """Picklable subset of an OsmoseCalibrationProblem needed to evaluate one candidate in a worker."""

    free_params: list[FreeParameter]
    objective_fns: list[Callable]
    base_config_path: Path
    work_dir: Path
    jar_path: Path | None
    java_cmd: str
    enable_cache: bool
    cache_dir: Path | None
    subprocess_timeout: int
    cleanup_after_eval: bool
    use_java_engine: bool
    use_registry: bool


_WORKER_PROBLEM: "OsmoseCalibrationProblem | None" = None


def _worker_init(spec: _EvalSpec) -> None:
    """ProcessPoolExecutor initializer: reconstruct the problem once per worker."""
    global _WORKER_PROBLEM
    reg = build_registry() if spec.use_registry else None
    _WORKER_PROBLEM = OsmoseCalibrationProblem(
        free_params=spec.free_params,
        objective_fns=spec.objective_fns,
        base_config_path=spec.base_config_path,
        work_dir=spec.work_dir,
        jar_path=spec.jar_path,
        java_cmd=spec.java_cmd,
        n_parallel=1,
        parallel_backend="thread",  # workers must never nest a pool
        enable_cache=spec.enable_cache,
        cache_dir=spec.cache_dir,
        registry=reg,
        subprocess_timeout=spec.subprocess_timeout,
        cleanup_after_eval=spec.cleanup_after_eval,
        use_java_engine=spec.use_java_engine,
    )


def _worker_eval(run_id: int, params: np.ndarray) -> list[float]:
    """Evaluate one candidate in the worker; never raise into the pool."""
    assert _WORKER_PROBLEM is not None
    try:
        return _WORKER_PROBLEM._evaluate_candidate(run_id, params)
    except Exception:  # noqa: BLE001 — one bad candidate must not kill the generation
        return [float("inf")] * _WORKER_PROBLEM.n_obj


def _resolve_worker_count(n_parallel: int) -> int:
    """Worker count from OSMOSE_NSGA2_WORKERS (fallback n_parallel), clamped to [1, 32]."""
    try:
        n = int(os.environ.get("OSMOSE_NSGA2_WORKERS", n_parallel))
    except ValueError:
        n = n_parallel
    return max(1, min(32, n))
```

- [ ] **Step 4: Add `parallel_backend` + pool fields to `__init__`**

In `OsmoseCalibrationProblem.__init__`, add a keyword-only param (after `use_java_engine`) and an attribute:

```python
        parallel_backend: Literal["thread", "process"] = "thread",
```

and in the body (e.g. after `self.use_java_engine = ...`):

```python
        self.parallel_backend = parallel_backend
        self._executor: ProcessPoolExecutor | None = None
```

- [ ] **Step 5: Rewrite `_evaluate`'s branch structure (backend-aware) + add the process path**

Replace the `if self.n_parallel > 1: ... else: ...` block in `_evaluate` with:

```python
        if self.parallel_backend == "process" and self.n_parallel > 1:
            self._evaluate_process(X, F)
        elif self.n_parallel > 1:
            with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
                futures = {
                    executor.submit(self._evaluate_candidate, i, params): i
                    for i, params in enumerate(X)
                }
                for future in futures:
                    i = futures[future]
                    try:
                        objectives = future.result()
                        for k, obj_val in enumerate(objectives):
                            F[i, k] = obj_val
                    except _expected_errors as exc:
                        _log.warning("Candidate %d failed (expected): %s", i, exc)
        else:
            for i, params in enumerate(X):
                try:
                    objectives = self._evaluate_candidate(i, params)
                    for k, obj_val in enumerate(objectives):
                        F[i, k] = obj_val
                except _expected_errors as exc:
                    _log.warning("Candidate %d failed (expected): %s", i, exc)
```

(The `>50% inf` abort + `out["F"] = F` after this block stay unchanged.)

Add the process-eval + pool methods to the class:

```python
    def _evaluate_process(self, X, F) -> None:
        """Evaluate the population via the persistent process pool, recovering from a dead worker.

        Preserves already-scored candidates across a BrokenProcessPool: only the still-unscored
        ones are retried on a rebuilt pool (re-running is safe — seed=run_id is deterministic).
        """
        pending = {i: params for i, params in enumerate(X)}
        for _attempt in range(3):  # 1 initial + 2 rebuild retries
            if not pending:
                break
            pool = self._ensure_pool()
            try:
                # submit INSIDE the try: a pool that broke idle between gens raises
                # BrokenProcessPool from submit() itself, which must hit the handler below.
                futures = {pool.submit(_worker_eval, i, p): i for i, p in pending.items()}
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        for k, v in enumerate(fut.result()):
                            F[i, k] = v
                        del pending[i]
                    except _expected_errors as exc:
                        # MUST stay narrow _expected_errors, NOT _python_engine_errors:
                        # BrokenProcessPool subclasses RuntimeError, so the broader set would
                        # swallow worker-death here and defeat the rebuild/retry below.
                        _log.warning("Candidate %d failed (expected): %s", i, exc)
                        del pending[i]
            except BrokenProcessPool as exc:
                _log.warning(
                    "Process pool broke (worker died): %s — rebuilding, retrying %d candidates",
                    exc,
                    len(pending),
                )
                self.shutdown_pool()  # finished results stay in F; pending retried on a fresh pool
        # any still-pending after the retry cap stay inf; the >50% guard then applies

    def _eval_spec(self) -> _EvalSpec:
        return _EvalSpec(
            free_params=self.free_params,
            objective_fns=self.objective_fns,
            base_config_path=self.base_config_path,
            work_dir=self.work_dir,
            jar_path=self.jar_path,
            java_cmd=self.java_cmd,
            enable_cache=self._enable_cache,
            cache_dir=self._cache_dir,
            subprocess_timeout=self.subprocess_timeout,
            cleanup_after_eval=self.cleanup_after_eval,
            use_java_engine=self.use_java_engine,
            use_registry=self._registry is not None,
        )

    def _ensure_pool(self) -> ProcessPoolExecutor:
        # Discard a pool that broke while idle between generations.
        if self._executor is not None and getattr(self._executor, "_broken", False):
            self.shutdown_pool()
        if self._executor is None:
            spec = self._eval_spec()
            try:
                pickle.dumps(spec)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Objective/spec is not picklable; use parallel_backend='thread'"
                ) from exc
            self._executor = ProcessPoolExecutor(
                max_workers=_resolve_worker_count(self.n_parallel),
                mp_context=multiprocessing.get_context("forkserver"),
                initializer=_worker_init,
                initargs=(spec,),
            )
        return self._executor

    def shutdown_pool(self, cancel_futures: bool = False) -> None:
        if self._executor is not None:
            self._executor.shutdown(cancel_futures=cancel_futures)
            self._executor = None
```

- [ ] **Step 6: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_nsga2_parallel.py -q`
Expected: PASS (4 passed; a few seconds for the forkserver + Numba warm).

- [ ] **Step 7: Lint / format / type-check + no regressions**

`.venv/bin/ruff check osmose/ ui/ tests/`; `.venv/bin/ruff format osmose/ ui/ tests/`;
`.venv/bin/pyright osmose/calibration/problem.py tests/test_nsga2_parallel.py` → 0 errors.
`.venv/bin/python -m pytest tests/test_calibration_problem_python_engine.py -q` → still passes (additive kwarg).

- [ ] **Step 8: Commit**

```bash
git add osmose/calibration/problem.py tests/test_nsga2_parallel.py
git commit -m "feat(calibration): ProcessPoolExecutor backend for NSGA-II (forkserver, broken-pool recovery)"
```

---

### Task 3: Wire at the NSGA-II launch sites + benchmark + CHANGELOG

**Files:** Modify `ui/pages/calibration_handlers.py`, `scripts/benchmark_calibration.py`, `tests/test_app_structure.py`, `CHANGELOG.md`

- [ ] **Step 1: Write the failing wiring test**

Append to `tests/test_app_structure.py`:

```python
def test_nsga2_process_backend_wired():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    handlers = (root / "ui" / "pages" / "calibration_handlers.py").read_text()
    assert "BiomassRMSEObjective" in handlers and "DietDistanceObjective" in handlers
    assert "parallel_backend" in handlers and "shutdown_pool" in handlers
    bench = (root / "scripts" / "benchmark_calibration.py").read_text()
    assert "--backend" in bench
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_app_structure.py::test_nsga2_process_backend_wired -q`
Expected: FAIL.

- [ ] **Step 3: Swap the UI lambdas for functors**

In `ui/pages/calibration_handlers.py`, add to the imports (near the other `from osmose.calibration...`):

```python
from osmose.calibration.objectives import BiomassRMSEObjective, DietDistanceObjective
```

Replace the two lambda objective appends (currently at `calibration_handlers.py:1386,1389`):

```python
            objective_fns.append(lambda r, df=obs_bio_df: biomass_rmse(r.biomass(), df))
```
→
```python
            objective_fns.append(BiomassRMSEObjective(obs_bio_df))
```
and
```python
            objective_fns.append(lambda r, df=obs_diet_df: diet_distance(r.diet_matrix(), df))
```
→
```python
            objective_fns.append(DietDistanceObjective(obs_diet_df))
```

(The banded objective is left unchanged — it falls back to thread via the picklability check below.)

- [ ] **Step 4: Add the backend picker + pass it to the optimization problem**

In `calibration_handlers.py`, add a module-level helper (near the other helpers):

```python
def _all_picklable(objective_fns: list) -> bool:
    import pickle

    try:
        pickle.dumps(objective_fns)
        return True
    except Exception:  # noqa: BLE001
        _log.info("Some objectives are not picklable — NSGA-II falls back to the thread backend")
        return False
```

At the **NSGA-II optimization** problem construction (`OsmoseCalibrationProblem(` at `calibration_handlers.py:1096` — the one whose `problem` is passed to `minimize()` in `run_optimization`), add the kwarg:

```python
            parallel_backend=("process" if _all_picklable(_shared_objective_fns) else "thread"),
```

(Use the objective list in scope at that construction — `_shared_objective_fns`. Leave the preflight ctor at `:1590` and the sensitivity ctor at `:1751` unchanged — sensitivity has its own pool; preflight is cheap and stays thread/serial.)

- [ ] **Step 5: Shut the pool down in a `finally`**

In `run_optimization` (the `try:` wrapping `minimize(...)` near `calibration_handlers.py:1257`), add a `finally` that releases the pool (covers normal end, exception, and cancel):

```python
                    finally:
                        problem.shutdown_pool(cancel_futures=True)
```

(If `run_optimization` already has a `try/except`, add/extend a `finally`; `shutdown_pool` is a no-op for thread-backend runs.)

- [ ] **Step 6: Add `--backend` to the benchmark**

In `scripts/benchmark_calibration.py`: thread a `backend` parameter into the run function that builds the problem (`OsmoseCalibrationProblem(` at `:42`) — add `parallel_backend=backend` to that ctor; wrap the `minimize(...)` (`:58`) so `problem.shutdown_pool()` runs in a `finally`. Add the CLI arg:

```python
    parser.add_argument("--backend", choices=["thread", "process"], default="thread")
```

and pass `args.backend` through to the run function.

- [ ] **Step 7: Run the wiring test + gates**

`.venv/bin/python -m pytest tests/test_app_structure.py::test_nsga2_process_backend_wired -q` → PASS.
`.venv/bin/ruff check osmose/ ui/ tests/`; `.venv/bin/ruff format osmose/ ui/ tests/` (NOT app.py — none touched here);
`.venv/bin/pyright ui/pages/calibration_handlers.py scripts/benchmark_calibration.py tests/test_app_structure.py` → 0 errors.

- [ ] **Step 8: CHANGELOG**

Under `## [Unreleased]` → `### Added` in `CHANGELOG.md`:

```markdown
- **calibration (perf):** an opt-in ProcessPoolExecutor backend for NSGA-II
  (`parallel_backend="process"`) that evaluates candidates in separate processes (GIL-free),
  recovering more than the prior ~3.02× thread-bound speedup. Objectives are now picklable functors
  (`BiomassRMSEObjective`/`DietDistanceObjective`); the dashboard selects the process backend
  automatically when objectives are picklable (banded loss falls back to threads). Worker count via
  `OSMOSE_NSGA2_WORKERS`; `scripts/benchmark_calibration.py --backend {thread,process}` compares them.
```

- [ ] **Step 9: Commit**

```bash
git add ui/pages/calibration_handlers.py scripts/benchmark_calibration.py tests/test_app_structure.py CHANGELOG.md
git commit -m "feat(calibration): select process backend at NSGA-II sites + benchmark switch"
```

---

## Final verification (after all tasks)

- [ ] Full non-e2e suite: `.venv/bin/python -m pytest -m 'not e2e' -n auto -q`
- [ ] `.venv/bin/ruff check osmose/ ui/ tests/` + `.venv/bin/ruff format --check osmose/ ui/ tests/` clean
- [ ] `.venv/bin/pyright --pythonpath .venv/bin/python osmose/calibration/objectives.py osmose/calibration/problem.py ui/pages/calibration_handlers.py scripts/benchmark_calibration.py tests/test_objectives.py tests/test_nsga2_parallel.py tests/test_app_structure.py` → 0 errors
- [ ] Demonstrate the speedup (not CI): `.venv/bin/python scripts/benchmark_calibration.py --backend thread` vs `--backend process` on the multi-core box; record the ratio.
- [ ] Final whole-implementation review before finishing the branch.

## Self-Review (plan author)

- **Spec coverage:** functors (Task 1) ↔ spec §1; `_EvalSpec`/worker fns/`parallel_backend`/process branch/pool lifecycle (Task 2) ↔ spec §2–3; launch-site wiring + benchmark (Task 3) ↔ spec §4; tests (Tasks 1–2) ↔ spec Testing 1–5; CHANGELOG (Task 3) ↔ spec note. The forkserver context, broken-pool rebuild/retry, `use_registry=(self._registry is not None)`, narrow inner `_expected_errors`, and the `data/minimal/` parity config are all carried verbatim from the converged spec.
- **Contract consistency:** functor signatures match the call sites; `_EvalSpec` fields ↔ `_worker_init` kwargs ↔ `OsmoseCalibrationProblem.__init__`; `_resolve_worker_count`/`shutdown_pool`/`_ensure_pool` names identical across problem.py, tests, and the launch sites; the inner per-future `except` is `_expected_errors` (NOT `_python_engine_errors`) and the broken-pool test asserts finite-after-retry to lock that in.
- **Additive/ordering:** `parallel_backend` is keyword-only with a default → no existing ctor call breaks; Task 1 (functors) precedes Task 2 (which the worker uses) precedes Task 3 (wiring).
- **No placeholders:** every code step shows complete code; commands have expected output.
