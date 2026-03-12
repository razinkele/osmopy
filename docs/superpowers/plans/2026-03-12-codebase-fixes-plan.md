# Codebase Fixes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all Critical and High severity findings from the 2026-03-12 codebase analysis, plus selected Medium fixes, hardening the OSMOSE Python app for production use.

**Architecture:** Fixes are grouped into 5 phases by impact. Each phase produces a working, tested codebase. Phases 1-2 address all Critical/High issues. Phases 3-5 handle test quality and Medium improvements.

**Tech Stack:** Python 3.12, Shiny for Python, pytest, asyncio, threading, queue

**Spec:** `docs/superpowers/specs/2026-03-12-codebase-analysis-findings.md`

**Deferred findings (Phase 5 — address opportunistically):**
M2 (override key injection), M9/M12 (grid.py extraction), M11 (error handling style), M13 (lazy results loading), M15 (docstrings), M16 (unparseable config lines), M18 (csv_maps_to_netcdf silent failure), M22 (integration test scope), H16 (reactive UI integration tests — high complexity, requires Playwright/ShinyTestClient infrastructure), and all Low findings.

---

## Chunk 1: Phase 1 — Critical Safety (Tasks 1-6)

### Task 1: Thread-safe calibration communication (C1, H5)

**Findings:** C1 (thread-unsafe reactive writes), H5 (no error notification for thread failures)

**Files:**
- Modify: `ui/pages/calibration_handlers.py:99-353`
- Modify: `ui/pages/calibration.py:145-160` (cal_status render)
- Test: `tests/test_ui_calibration_handlers.py`

The core problem: `run_surrogate()` and `run_optimization()` run on `threading.Thread` but call `.get()` and `.set()` on Shiny `reactive.value` objects, which are not thread-safe.

**Solution:** Replace direct reactive writes with a `queue.Queue` polled by `reactive.poll()`.

- [ ] **Step 1: Write the failing test for thread-safe message relay**

```python
# tests/test_ui_calibration_handlers.py — add at end of file
import threading


def test_calibration_message_queue():
    """Thread-safe message queue relays updates without reactive writes."""
    from ui.pages.calibration_handlers import CalibrationMessageQueue

    q = CalibrationMessageQueue()

    # Simulate thread posting messages
    def worker():
        q.post_status("Fitting GP model...")
        q.post_history_append(0.5)
        q.post_history_append(0.3)
        q.post_results(X=[[1, 2]], F=[[0.5]])
        q.post_error("Something broke")
        q.post_sensitivity({"S1": [0.5]})

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    msgs = q.drain()
    assert len(msgs) == 6
    assert msgs[0] == ("status", "Fitting GP model...")
    assert msgs[1] == ("history_append", 0.5)
    assert msgs[2] == ("history_append", 0.3)
    assert msgs[3][0] == "results"
    assert msgs[4] == ("error", "Something broke")
    assert msgs[5][0] == "sensitivity"


def test_calibration_message_queue_drain_empties():
    from ui.pages.calibration_handlers import CalibrationMessageQueue

    q = CalibrationMessageQueue()
    q.post_status("hello")
    assert len(q.drain()) == 1
    assert len(q.drain()) == 0  # second drain is empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ui_calibration_handlers.py::test_calibration_message_queue -v`
Expected: FAIL — `CalibrationMessageQueue` not defined

- [ ] **Step 3: Implement CalibrationMessageQueue**

Add at top of `ui/pages/calibration_handlers.py` (after existing imports):

```python
import queue as _queue_mod
import time


class CalibrationMessageQueue:
    """Thread-safe message queue for calibration thread -> UI communication."""

    def __init__(self):
        self._q: _queue_mod.Queue = _queue_mod.Queue()

    def post_status(self, msg: str) -> None:
        self._q.put(("status", msg))

    def post_history_append(self, value: float) -> None:
        self._q.put(("history_append", value))

    def post_results(self, X, F) -> None:
        self._q.put(("results", (X, F)))

    def post_error(self, msg: str) -> None:
        self._q.put(("error", msg))

    def post_sensitivity(self, result) -> None:
        self._q.put(("sensitivity", result))

    def drain(self) -> list[tuple]:
        msgs = []
        while True:
            try:
                msgs.append(self._q.get_nowait())
            except _queue_mod.Empty:
                break
        return msgs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ui_calibration_handlers.py::test_calibration_message_queue tests/test_ui_calibration_handlers.py::test_calibration_message_queue_drain_empties -v`
Expected: PASS

- [ ] **Step 5: Refactor `register_calibration_handlers` to use message queue**

In `ui/pages/calibration_handlers.py`, inside `register_calibration_handlers` (line 99):

1. Create shared queue and cancel event at the start of the function body:

```python
    msg_queue = CalibrationMessageQueue()
    cancel_event = threading.Event()
```

2. Add a `reactive.poll` to drain and apply messages every 500ms. Place this before the event handlers:

```python
    @reactive.poll(lambda: time.time(), interval_secs=0.5)
    def _poll_cal_messages():
        msgs = msg_queue.drain()
        for kind, payload in msgs:
            if kind == "status":
                surrogate_status.set(payload)
            elif kind == "history_append":
                current = cal_history.get()
                cal_history.set(current + [payload])
            elif kind == "results":
                X, F = payload
                cal_X.set(X)
                cal_F.set(F)
            elif kind == "error":
                surrogate_status.set(f"Failed: {payload}")
                ui.notification_show(f"Calibration error: {payload}", type="error", duration=10)
            elif kind == "sensitivity":
                sensitivity_result.set(payload)
```

3. In `handle_start_cal` (line 117): replace `cancel_flag.set(False)` with `cancel_event.clear()`.

4. In `handle_stop_cal` (line 280): replace `cancel_flag.set(True)` with `cancel_event.set()`.

5. Replace ALL reactive writes in `run_surrogate()` (lines 186-233):
   - Every `surrogate_status.set(...)` → `msg_queue.post_status(...)`
   - Every `cancel_flag.get()` → `cancel_event.is_set()`
   - `cal_X.set(samples)` + `cal_F.set(Y)` → `msg_queue.post_results(X=samples, F=Y)`
   - `cal_history.set(history)` → loop: `for val in history: msg_queue.post_history_append(val)`

6. Replace ALL reactive writes in `run_optimization()` (lines 240-272):
   - Replace `append_history` function body: `msg_queue.post_history_append(val)`
   - `cancel_check=cancel_flag.get` → `cancel_check=cancel_event.is_set`
   - `cal_F.set(res.F)` + `cal_X.set(res.X)` → `msg_queue.post_results(X=res.X, F=res.F)`
   - `surrogate_status.set(f"Calibration failed: {exc}")` → `msg_queue.post_error(str(exc))`

7. Replace ALL reactive writes in `run_sensitivity()` (lines 317-349):
   - `sensitivity_result.set(sens_result)` → `msg_queue.post_sensitivity(sens_result)`
   - `surrogate_status.set(f"Sensitivity failed: {exc}")` → `msg_queue.post_error(f"Sensitivity: {exc}")`

- [ ] **Step 6: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add ui/pages/calibration_handlers.py tests/test_ui_calibration_handlers.py
git commit -m "fix: thread-safe calibration communication via message queue

Replace direct reactive.value writes from background threads with a
thread-safe queue polled by reactive.poll(). Fixes C1, H5.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Fix scenario load and config import state guards (C2, C5)

**Findings:** C2 (scenario load bypasses guards), C5 (config import bypasses guards)

**Files:**
- Modify: `ui/pages/scenarios.py:120-128`
- Modify: `ui/pages/advanced.py:155-165`
- Test: `tests/test_ui_load_scenarios.py`

- [ ] **Step 1: Write failing test for scenario load state management**

This tests the handler's behavior by verifying that after loading a scenario, the config contains the loaded values (not stale values). The test simulates the broken behavior where sync effects overwrite loaded config.

```python
# tests/test_ui_load_scenarios.py — add at end
def test_scenario_load_updates_all_metadata(tmp_path):
    """Loading a scenario must update config_name, species_names, and bump load_trigger."""
    from unittest.mock import MagicMock, patch
    from osmose.scenarios import Scenario, ScenarioManager

    mgr = ScenarioManager(tmp_path / "scenarios")
    config = {
        "simulation.nspecies": "2",
        "species.name.sp0": "Anchovy",
        "species.name.sp1": "Sardine",
    }
    mgr.save(Scenario(name="test_scenario", config=config))
    loaded = mgr.load("test_scenario")

    # Verify the data layer returns correct data
    assert loaded.config["simulation.nspecies"] == "2"
    assert loaded.config["species.name.sp0"] == "Anchovy"
    assert loaded.config["species.name.sp1"] == "Sardine"
    assert loaded.name == "test_scenario"

    # The fix must set these state fields (verify by code inspection):
    # - state.loading = True during update
    # - state.config_name = selected scenario name
    # - state.species_names = extracted species names
    # - state.load_trigger incremented
    # - state.loading = False in finally block
    # These cannot be unit-tested without Shiny session, but we verify
    # the handler code contains them in test_scenario_load_has_state_guards.


def test_scenario_load_has_state_guards():
    """Verify handle_load in scenarios.py sets loading flag and bumps trigger."""
    import inspect
    from ui.pages.scenarios import scenarios_server

    source = inspect.getsource(scenarios_server)
    # After the fix, handle_load must contain these patterns:
    assert "state.loading.set(True)" in source, "handle_load must set loading flag"
    assert "state.load_trigger.set(" in source, "handle_load must bump load_trigger"
    assert "finally:" in source, "handle_load must use try/finally for loading flag"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ui_load_scenarios.py::test_scenario_load_has_state_guards -v`
Expected: FAIL — `state.loading.set(True)` not found in source

- [ ] **Step 3: Fix `handle_load` in scenarios.py**

Replace lines 120-128 of `ui/pages/scenarios.py`:

```python
    @reactive.effect
    @reactive.event(input.btn_load_scenario)
    def handle_load():
        selected = input.selected_scenario()
        if not selected:
            return
        loaded = mgr.load(selected)
        state.loading.set(True)
        try:
            state.config.set(loaded.config)
            state.config_name.set(selected)
            state.dirty.set(False)

            n_species = int(loaded.config.get("simulation.nspecies", "3"))
            names = [
                loaded.config.get(f"species.name.sp{i}", f"Species {i}")
                for i in range(n_species)
            ]
            state.species_names.set(names)
            ui.update_numeric("n_species", value=n_species)

            with reactive.isolate():
                state.load_trigger.set(state.load_trigger.get() + 1)

            ui.notification_show(
                f"Loaded scenario '{selected}' ({len(loaded.config)} parameters).",
                type="message",
                duration=3,
            )
        finally:
            state.loading.set(False)
```

- [ ] **Step 4: Fix `confirm_import` in advanced.py**

Replace lines 155-165 of `ui/pages/advanced.py`:

```python
    @reactive.effect
    @reactive.event(input.confirm_import)
    def confirm_import():
        pending = import_pending.get()
        if not pending:
            return
        state.loading.set(True)
        try:
            with reactive.isolate():
                cfg = dict(state.config.get())
            cfg.update(pending)
            state.config.set(cfg)
            import_pending.set({})

            with reactive.isolate():
                state.load_trigger.set(state.load_trigger.get() + 1)

            ui.notification_show(
                f"Imported {len(pending)} parameter(s).",
                type="message",
                duration=3,
            )
        finally:
            state.loading.set(False)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ui_load_scenarios.py -v`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add ui/pages/scenarios.py ui/pages/advanced.py tests/test_ui_load_scenarios.py
git commit -m "fix: add state guards to scenario load and config import

Mirror handle_load_example pattern: set loading flag, update metadata,
bump load_trigger, clear loading in finally. Fixes C2, C5.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Add top-level exception handler to surrogate thread (C3)

**Findings:** C3 (surrogate thread dies silently)

**Files:**
- Modify: `ui/pages/calibration_handlers.py:186-233` (run_surrogate function)
- Test: `tests/test_ui_calibration_handlers.py`

- [ ] **Step 1: Write test for surrogate error relay**

```python
# tests/test_ui_calibration_handlers.py — add
def test_surrogate_error_relayed_to_queue():
    """If surrogate calibration raises, error must be posted to queue."""
    from ui.pages.calibration_handlers import CalibrationMessageQueue

    q = CalibrationMessageQueue()

    def fake_surrogate_with_handler():
        try:
            raise RuntimeError("GP fit singular matrix")
        except Exception as exc:
            q.post_error(f"Surrogate calibration failed: {exc}")

    import threading
    t = threading.Thread(target=fake_surrogate_with_handler)
    t.start()
    t.join()

    msgs = q.drain()
    assert len(msgs) == 1
    assert msgs[0][0] == "error"
    assert "singular matrix" in msgs[0][1]
```

- [ ] **Step 2: Run test to verify it passes (validates the queue pattern)**

Run: `.venv/bin/python -m pytest tests/test_ui_calibration_handlers.py::test_surrogate_error_relayed_to_queue -v`
Expected: PASS (this tests the queue pattern; the actual fix is in production code)

- [ ] **Step 3: Wrap `run_surrogate()` body in try/except**

After the Task 1 refactor, `run_surrogate()` posts to the message queue. Add a top-level try/except around the entire body:

```python
            def run_surrogate():
                try:
                    from osmose.calibration.surrogate import SurrogateCalibrator
                    # ... entire existing body unchanged ...
                except Exception as exc:
                    _log.error("Surrogate calibration failed: %s", exc, exc_info=True)
                    msg_queue.post_error(f"Surrogate calibration failed: {exc}")
```

This matches the pattern already used by `run_optimization()`.

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add ui/pages/calibration_handlers.py tests/test_ui_calibration_handlers.py
git commit -m "fix: add top-level exception handler to surrogate calibration thread

Prevents silent thread death when SurrogateCalibrator.fit() or
find_optimum() raises. Error relayed to UI via message queue. Fixes C3.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Narrow calibration exception handling + failure tracking (C4, H12)

**Findings:** C4 (broad except masks all failures), H12 (inf corrupts Sobol)

**Files:**
- Modify: `osmose/calibration/problem.py:68-100`
- Modify: `ui/pages/calibration_handlers.py:328-343` (sensitivity loop)
- Test: `tests/test_calibration_problem.py`

- [ ] **Step 1: Write test for failure tracking in calibration**

```python
# tests/test_calibration_problem.py — add at end
import pytest


def test_evaluate_aborts_on_majority_failures(tmp_path):
    """Optimization should raise when >50% of candidates fail unexpectedly."""
    import numpy as np
    from unittest.mock import patch
    from osmose.calibration.problem import OsmoseCalibrationProblem, FreeParameter

    problem = OsmoseCalibrationProblem(
        free_params=[FreeParameter(key="test.param", lower_bound=0, upper_bound=1)],
        objective_fns=[lambda r: 1.0],
        base_config_path=tmp_path / "config.csv",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path,
    )

    X = np.array([[0.1], [0.5], [0.9]])
    out = {}

    # All candidates raise TypeError (unexpected exception) — should propagate
    with patch.object(problem, "_evaluate_candidate", side_effect=TypeError("bad objective")):
        with pytest.raises(TypeError):
            problem._evaluate(X, out)


def test_evaluate_tolerates_expected_failures(tmp_path):
    """Expected failures (OSError, etc.) are scored as inf, not propagated."""
    import numpy as np
    from unittest.mock import patch
    from osmose.calibration.problem import OsmoseCalibrationProblem, FreeParameter

    problem = OsmoseCalibrationProblem(
        free_params=[FreeParameter(key="test.param", lower_bound=0, upper_bound=1)],
        objective_fns=[lambda r: 1.0],
        base_config_path=tmp_path / "config.csv",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path,
    )

    X = np.array([[0.1], [0.5], [0.9]])
    out = {}

    # OSError is expected — should be handled gracefully
    with patch.object(problem, "_evaluate_candidate", side_effect=OSError("disk full")):
        problem._evaluate(X, out)  # should not raise

    assert np.all(np.isinf(out["F"]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_problem.py::test_evaluate_aborts_on_majority_failures tests/test_calibration_problem.py::test_evaluate_tolerates_expected_failures -v`
Expected: FAIL — TypeError not propagated, OSError test may pass already

- [ ] **Step 3: Implement narrowed exception handling in `_evaluate`**

Replace lines 68-100 of `osmose/calibration/problem.py`:

```python
    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate a population of candidates.

        X has shape (pop_size, n_var). Each row is a candidate.
        Expected failures (OSError, timeout, file not found) score as inf.
        Unexpected failures (TypeError, etc.) propagate immediately.
        """
        _log.info("Evaluating %d candidates (parallel=%d)", X.shape[0], self.n_parallel)
        F = np.full((X.shape[0], self.n_obj), np.inf)
        _expected_errors = (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
            OSError,
        )

        if self.n_parallel > 1:
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

        out["F"] = F
```

Unexpected exceptions (TypeError, ImportError, etc.) now propagate naturally — they are no longer caught.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_problem.py::test_evaluate_aborts_on_majority_failures tests/test_calibration_problem.py::test_evaluate_tolerates_expected_failures -v`
Expected: PASS

- [ ] **Step 5: Add sensitivity failure threshold**

In `ui/pages/calibration_handlers.py`, inside `run_sensitivity()`, after the sample evaluation loop (after the `for idx, row in enumerate(samples)` block, before `sens_result = analyzer.analyze(Y)`), add:

```python
                n_inf = int(np.isinf(Y).sum())
                if n_inf > len(Y) * 0.1:
                    msg_queue.post_error(
                        f"Sensitivity aborted: {n_inf}/{len(Y)} samples failed "
                        f"(>10% threshold)"
                    )
                    return
```

- [ ] **Step 6: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add osmose/calibration/problem.py ui/pages/calibration_handlers.py tests/test_calibration_problem.py
git commit -m "fix: narrow calibration exceptions — propagate unexpected errors

Expected errors (OSError, timeout) score as inf. Unexpected errors
(TypeError, etc.) propagate immediately. Sensitivity aborts when >10%
of samples fail. Fixes C4, H12.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Path traversal protection (H1, M3, M5)

**Findings:** H1 (scenario save/delete path traversal), M3 (history path traversal), M5 (no name sanitization)

**Files:**
- Modify: `osmose/scenarios.py:47-136`
- Modify: `osmose/history.py:51-57`
- Test: `tests/test_scenarios.py`
- Test: `tests/test_history.py`

- [ ] **Step 1: Write failing tests for path traversal**

```python
# tests/test_scenarios.py — add at end
import pytest


def test_save_rejects_path_traversal(tmp_path):
    from osmose.scenarios import Scenario, ScenarioManager

    mgr = ScenarioManager(tmp_path / "scenarios")
    scenario = Scenario(name="../../etc/evil", config={})
    with pytest.raises(ValueError, match="[Uu]nsafe"):
        mgr.save(scenario)


def test_delete_rejects_path_traversal(tmp_path):
    from osmose.scenarios import ScenarioManager

    mgr = ScenarioManager(tmp_path / "scenarios")
    with pytest.raises(ValueError, match="[Uu]nsafe"):
        mgr.delete("../../etc")


def test_load_rejects_path_traversal(tmp_path):
    from osmose.scenarios import ScenarioManager

    mgr = ScenarioManager(tmp_path / "scenarios")
    with pytest.raises(ValueError, match="[Uu]nsafe"):
        mgr.load("../../etc/passwd")


def test_fork_rejects_path_traversal(tmp_path):
    from osmose.scenarios import ScenarioManager

    mgr = ScenarioManager(tmp_path / "scenarios")
    with pytest.raises(ValueError, match="[Uu]nsafe"):
        mgr.fork("../../etc/passwd", "new_name")


def test_save_rejects_empty_name(tmp_path):
    from osmose.scenarios import Scenario, ScenarioManager

    mgr = ScenarioManager(tmp_path / "scenarios")
    scenario = Scenario(name="", config={})
    with pytest.raises(ValueError, match="[Ii]nvalid"):
        mgr.save(scenario)
```

```python
# tests/test_history.py — add at end
import pytest


def test_load_run_rejects_path_traversal(tmp_path):
    from osmose.history import RunHistory

    history = RunHistory(tmp_path / "history")
    with pytest.raises(ValueError, match="[Uu]nsafe"):
        history.load_run("../../etc/passwd")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scenarios.py::test_save_rejects_path_traversal tests/test_scenarios.py::test_delete_rejects_path_traversal tests/test_scenarios.py::test_fork_rejects_path_traversal tests/test_history.py::test_load_run_rejects_path_traversal -v`
Expected: FAIL — no ValueError raised

- [ ] **Step 3: Add `_validate_path` helper to ScenarioManager**

In `osmose/scenarios.py`, add method to `ScenarioManager` after `__init__` (after line 52):

```python
    def _validate_path(self, name: str) -> Path:
        """Validate a scenario name resolves within storage_dir."""
        if not name or not name.strip():
            raise ValueError(f"Invalid scenario name: {name!r}")
        target = (self.storage_dir / name).resolve()
        if not target.is_relative_to(self.storage_dir.resolve()):
            raise ValueError(f"Unsafe scenario name: {name!r}")
        return target
```

Then update these methods to use it:

**`save()` (line 57):** replace `target = self.storage_dir / scenario.name` with:
```python
        target = self._validate_path(scenario.name)
```

**`load()` (line 83):** replace `path = self.storage_dir / name / "scenario.json"` with:
```python
        target = self._validate_path(name)
        path = target / "scenario.json"
```

**`delete()` (line 108):** replace `path = self.storage_dir / name` with:
```python
        path = self._validate_path(name)
```

**`fork()` (line 125):** add validation at the start of the method body:
```python
    def fork(self, source_name: str, new_name: str, description: str = "") -> Scenario:
        """Create a new scenario based on an existing one."""
        self._validate_path(source_name)
        self._validate_path(new_name)
        source = self.load(source_name)
        # ... rest unchanged ...
```

- [ ] **Step 4: Add path validation to RunHistory**

In `osmose/history.py`, replace `load_run` (lines 51-57):

```python
    def load_run(self, timestamp: str) -> RunRecord:
        """Load a specific run record by timestamp."""
        safe_ts = timestamp.replace(":", "-")
        path = (self.history_dir / f"run_{safe_ts}.json").resolve()
        if not path.is_relative_to(self.history_dir.resolve()):
            raise ValueError(f"Unsafe timestamp: {timestamp!r}")
        with open(path) as f:
            data = json.load(f)
        return RunRecord(**data)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scenarios.py::test_save_rejects_path_traversal tests/test_scenarios.py::test_delete_rejects_path_traversal tests/test_scenarios.py::test_load_rejects_path_traversal tests/test_scenarios.py::test_fork_rejects_path_traversal tests/test_scenarios.py::test_save_rejects_empty_name tests/test_history.py::test_load_run_rejects_path_traversal -v`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add osmose/scenarios.py osmose/history.py tests/test_scenarios.py tests/test_history.py
git commit -m "fix: add path traversal protection to scenarios and history

Validate that resolved paths stay within storage_dir/history_dir using
resolve() + is_relative_to(). Reject empty names. Fixes H1, M3, M5.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Fix vacuous test assertion (C6)

**Findings:** C6 (or True makes assertion always pass)

**Files:**
- Modify: `tests/test_param_form.py:77-92`

- [ ] **Step 1: Fix the assertion**

Replace lines 77-93 of `tests/test_param_form.py` with:

```python
def test_render_category_filters_advanced():
    fields = [
        OsmoseField(key_pattern="field.basic", param_type=ParamType.FLOAT, default=1.0, advanced=False),
        OsmoseField(key_pattern="field.secret", param_type=ParamType.FLOAT, default=2.0, advanced=True),
    ]
    # Without advanced — only basic field should be rendered
    result = render_category(fields, show_advanced=False)
    html = str(result)
    assert "field_basic" in html  # input ID uses underscores
    assert "field_secret" not in html

    # With advanced — both fields should be present
    result_adv = render_category(fields, show_advanced=True)
    html_adv = str(result_adv)
    assert "field_basic" in html_adv
    assert "field_secret" in html_adv
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_param_form.py::test_render_category_filters_advanced -v`
Expected: PASS (if it fails, the filtering logic itself has a bug — investigate and fix)

- [ ] **Step 3: Commit**

```bash
git add tests/test_param_form.py
git commit -m "fix: replace vacuous 'or True' assertion with real check

Test now verifies advanced fields are excluded when show_advanced=False
and included when True. Fixes C6.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 2: Phase 2 — High-Priority Hardening (Tasks 7-13)

### Task 7: HTML escape in report template (H2)

**Files:**
- Modify: `osmose/reporting.py:75-76`
- Test: `tests/test_reporting.py`

- [ ] **Step 1: Write test for HTML escaping**

```python
# tests/test_reporting.py — add
def test_summary_table_escapes_html(tmp_path):
    """Species names with HTML should be escaped in report output."""
    import pandas as pd
    from osmose.reporting import generate_report

    class FakeResults:
        def biomass(self):
            return pd.DataFrame({
                "time": [0, 1],
                "species": ["<script>alert(1)</script>", "<script>alert(1)</script>"],
                "biomass": [100.0, 200.0],
            })
        def mean_trophic_level(self):
            return pd.DataFrame()
        def mortality(self):
            return pd.DataFrame()

    config = {"simulation.nspecies": "1", "simulation.time.nyear": "1"}
    out = tmp_path / "report.html"
    generate_report(FakeResults(), config, out)

    html = out.read_text()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_reporting.py::test_summary_table_escapes_html -v`
Expected: FAIL — `<script>` found in HTML

- [ ] **Step 3: Fix `to_html` call**

In `osmose/reporting.py:75-76`, change to explicitly escape cell content. The `| safe` filter is needed for the `<table>` HTML structure, so keep it but ensure cell values are escaped:

```python
    summary_html = (
        table.to_html(index=False, classes="table", escape=True) if not table.empty else "<p>No data</p>"
    )
```

Note: pandas `to_html` has `escape=True` as default in most versions, but setting it explicitly ensures safety regardless of version.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_reporting.py::test_summary_table_escapes_html -v`
Expected: PASS

- [ ] **Step 5: Run full test suite, commit**

Run: `.venv/bin/python -m pytest -x -q`

```bash
git add osmose/reporting.py tests/test_reporting.py
git commit -m "fix: escape HTML in report summary table to prevent XSS

Add escape=True to DataFrame.to_html(). Species names with HTML
characters are now properly escaped. Fixes H2.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Fix calibration checkbox reactive dependency (H3)

**Files:**
- Modify: `ui/pages/calibration.py:162-176`

- [ ] **Step 1: Fix `free_param_selector` to use load_trigger**

Replace lines 162-176 of `ui/pages/calibration.py`:

```python
    @render.ui
    def free_param_selector():
        state.load_trigger.get()  # only re-render on config load
        with reactive.isolate():
            cfg = state.config.get()
        n_str = cfg.get("simulation.nspecies", "3")
        n_species = int(n_str) if n_str else 3
        params = get_calibratable_params(state.registry, n_species)
        checkboxes = [
            ui.input_checkbox(
                f"cal_param_{p['key'].replace('.', '_')}",
                p["label"],
                value=False,
            )
            for p in params
        ]
        return ui.div(*checkboxes)
```

- [ ] **Step 2: Run full test suite, commit**

Run: `.venv/bin/python -m pytest -x -q`

```bash
git add ui/pages/calibration.py
git commit -m "fix: use load_trigger for calibration checkbox rendering

Prevents checkbox selections from being destroyed on every config
change. Checkboxes now only re-render when a config is loaded. Fixes H3.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 9: Fix results loading race (H4, M8)

**Files:**
- Modify: `ui/pages/results.py:342-346`
- Modify: `ui/pages/run.py` (add explicit reset)

Note: Tasks 9, 11, 12, 22, 25 all modify `ui/pages/results.py`. Apply in order to avoid line drift issues.

- [ ] **Step 1: Remove the problematic `_reset_results_loaded` effect**

Delete lines 342-346 of `ui/pages/results.py`:

```python
    # DELETE this entire block:
    @reactive.effect
    def _reset_results_loaded():
        """Reset loaded flag when output directory changes."""
        state.output_dir.get()  # take dependency
        state.results_loaded.set(False)
```

- [ ] **Step 2: Add explicit reset in handle_run**

In `ui/pages/run.py`, inside `handle_run()`, before the runner starts (after the line `state.busy.set("Running OSMOSE...")`), add:

```python
        state.results_loaded.set(False)
```

- [ ] **Step 3: Run full test suite, commit**

Run: `.venv/bin/python -m pytest -x -q`

```bash
git add ui/pages/results.py ui/pages/run.py
git commit -m "fix: remove reactive results_loaded reset to prevent double-load

Replace automatic reset effect with explicit reset when a new run starts.
Prevents race where results_loaded toggles False->True->False. Fixes H4, M8.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 10: Add user notifications to grid loading failures (H7-H10)

**Files:**
- Modify: `ui/pages/grid.py:58-67, 216-228, 466-468, 529-531, 688-710`

- [ ] **Step 1: Narrow exceptions in all 4 grid loading functions**

In `ui/pages/grid.py`, replace the broad `except Exception` in each function:

**`_load_mask` (line 65):**
```python
    except (OSError, ValueError, KeyError) as exc:
        _log.warning("Failed to load mask %s: %s", full_path, exc)
        return None
```

**`_load_netcdf_grid` (line ~226):**
```python
    except (OSError, ValueError, KeyError) as exc:
        _log.warning("Failed to load NetCDF grid %s: %s", full_path, exc)
        return None
```

**`_load_csv_overlay` (line ~466):**
```python
    except (OSError, ValueError, KeyError) as exc:
        _log.warning("Failed to load CSV overlay %s: %s", file_path, exc)
        return None
```

**`_load_netcdf_overlay` (line ~529):**
```python
    except (OSError, ValueError, KeyError) as exc:
        _log.warning("Failed to load overlay %s: %s", file_path, exc)
        return None
```

- [ ] **Step 2: Add notifications at loader call sites in `update_grid_map`**

In `ui/pages/grid.py`, in the `update_grid_map` reactive effect (line 688), add notifications after failed loads:

After line 699 (`nc_data = _load_netcdf_grid(cfg, config_dir=cfg_dir) if is_ncgrid else None`):
```python
        if is_ncgrid and nc_data is None:
            ui.notification_show("Could not load NetCDF grid file.", type="warning", duration=5)
```

After line 707 (`mask = _load_mask(cfg, config_dir=cfg_dir)`):
```python
        if mask is None and cfg.get("grid.mask.file"):
            ui.notification_show("Could not load grid mask file.", type="warning", duration=5)
```

For overlay notifications, find the overlay loading call sites in the grid server (search for `_load_csv_overlay` and `_load_netcdf_overlay` calls) and add similar notifications.

- [ ] **Step 3: Run full test suite, commit**

Run: `.venv/bin/python -m pytest -x -q`

```bash
git add ui/pages/grid.py
git commit -m "fix: narrow grid file exceptions and add user notifications

Replace broad except Exception with specific (OSError, ValueError, KeyError).
Show notification when grid files fail to load. Fixes H7-H10.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 11: Add error handling to results loading (H6)

**Files:**
- Modify: `ui/pages/results.py:253-318` (_do_load_results)

- [ ] **Step 1: Wrap `_do_load_results` body in try/except**

Add logging import at the top of `ui/pages/results.py` if not present:
```python
from osmose.logging import setup_logging
_log = setup_logging("osmose.results.ui")
```

Wrap the body of `_do_load_results`:

```python
    def _do_load_results(out_dir: Path):
        try:
            from osmose.results import OsmoseResults

            # Close previous results to release file handles
            prev = results_obj.get()
            if prev is not None:
                try:
                    prev.close()
                except (AttributeError, OSError):
                    pass

            res = OsmoseResults(out_dir)
            # ... existing body unchanged through line 318 ...
            state.results_loaded.set(True)
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            _log.error("Failed to load results from %s: %s", out_dir, exc, exc_info=True)
            ui.notification_show(
                f"Failed to load results: {exc}", type="error", duration=10
            )
```

Note: This also addresses Task 22 (close NetCDF cache) — the `prev.close()` call releases previous file handles. The `OsmoseResults.close()` method exists at `osmose/results.py:344`.

- [ ] **Step 2: Run full test suite, commit**

Run: `.venv/bin/python -m pytest -x -q`

```bash
git add ui/pages/results.py
git commit -m "fix: add error handling and NetCDF cleanup to results loading

Wrap _do_load_results in try/except for user notification on failures.
Close previous OsmoseResults to release file handles. Fixes H6, M14.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 12: Narrow ensemble mode exception (H11)

**Files:**
- Modify: `ui/pages/results.py:405-409`

- [ ] **Step 1: Replace broad except**

Replace lines 405-409 of `ui/pages/results.py`:

```python
        ensemble_on = False
        try:
            ensemble_on = bool(input.ensemble_mode()) and bool(rep_dirs.get())
        except (AttributeError, TypeError):
            pass
```

- [ ] **Step 2: Run full test suite, commit**

Run: `.venv/bin/python -m pytest -x -q`

```bash
git add ui/pages/results.py
git commit -m "fix: narrow ensemble mode exception to AttributeError/TypeError

Prevents swallowing framework errors while still handling missing input
gracefully. Fixes H11.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 13: Fix run history logging level (H13)

**Files:**
- Modify: `ui/pages/run.py:292-293`

- [ ] **Step 1: Fix logging level and narrow exception**

Add `import json` to the imports at the top of `ui/pages/run.py` if not already present.

Replace lines 292-293:

```python
            except (OSError, json.JSONDecodeError) as exc:
                _log.warning("Failed to save run history: %s", exc)
```

- [ ] **Step 2: Run full test suite, commit**

Run: `.venv/bin/python -m pytest -x -q`

```bash
git add ui/pages/run.py
git commit -m "fix: upgrade run history save error from debug to warning

Narrow exception from bare Exception to OSError/JSONDecodeError.
Change log level from debug to warning so operators see failures. Fixes H13.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 3: Phase 3 — Test Quality + Minor Fixes (Tasks 14-17)

### Task 14: Add NaN/malformed input edge case tests (H14, H15)

**Files:**
- Modify: `tests/test_analysis.py` (already imports `import pandas as pd` at top)
- Modify: `tests/test_config_reader.py`

- [ ] **Step 1: Add NaN tests for analysis functions**

```python
# tests/test_analysis.py — add at end
import numpy as np


def test_ensemble_stats_with_nan():
    """ensemble_stats should handle NaN values gracefully."""
    from osmose.analysis import ensemble_stats

    df = pd.DataFrame({
        "time": [0, 1, 2, 0, 1, 2],
        "species": ["A", "A", "A", "A", "A", "A"],
        "biomass": [100.0, np.nan, 300.0, 150.0, 200.0, np.nan],
        "replicate": [0, 0, 0, 1, 1, 1],
    })
    result = ensemble_stats(df, "biomass")
    # Should produce a result with mean/std columns, NaN handled by nanmean
    assert "mean" in result.columns or result.empty


def test_shannon_diversity_with_zeros():
    """shannon_diversity should handle zero biomass (extinct species)."""
    from osmose.analysis import shannon_diversity

    df = pd.DataFrame({
        "species": ["A", "B", "C"],
        "biomass": [100.0, 0.0, 0.0],
    })
    result = shannon_diversity(df)
    assert np.isfinite(result)  # should not be NaN or inf


def test_size_spectrum_slope_with_nan():
    """size_spectrum_slope should handle NaN abundances."""
    from osmose.analysis import size_spectrum_slope

    df = pd.DataFrame({
        "size": [1.0, 2.0, 3.0, 4.0],
        "abundance": [100.0, np.nan, 30.0, 10.0],
    })
    # Should either handle NaN gracefully or raise a clear ValueError
    try:
        result = size_spectrum_slope(df)
        assert np.isfinite(result)
    except ValueError:
        pass  # acceptable: clear error on bad data
```

- [ ] **Step 2: Add malformed input tests for config reader**

```python
# tests/test_config_reader.py — add at end
def test_read_line_with_no_separator(tmp_path):
    """Config lines with no separator should be skipped without crashing."""
    from osmose.config.reader import OsmoseConfigReader

    config_file = tmp_path / "test.csv"
    config_file.write_text("just_a_key_no_value\nreal.key;real_value\n")
    reader = OsmoseConfigReader()
    result = reader.read_file(config_file)
    assert "real.key" in result
    assert "just_a_key_no_value" not in result


def test_read_file_with_bom(tmp_path):
    """Config files with UTF-8 BOM should be parsed correctly."""
    from osmose.config.reader import OsmoseConfigReader

    config_file = tmp_path / "bom.csv"
    config_file.write_bytes(b"\xef\xbb\xbfspecies.name.sp0;Anchovy\n")
    reader = OsmoseConfigReader()
    result = reader.read_file(config_file)
    # Key should not have BOM character prefix
    assert any("species.name.sp0" in k for k in result)


def test_read_file_with_multiple_separators(tmp_path):
    """Config line with multiple separators should split on first only."""
    from osmose.config.reader import OsmoseConfigReader

    config_file = tmp_path / "multi.csv"
    config_file.write_text("output.dir.path;/path/to/dir;extra\n")
    reader = OsmoseConfigReader()
    result = reader.read_file(config_file)
    assert result.get("output.dir.path") == "/path/to/dir;extra"
```

- [ ] **Step 3: Run the new tests**

Run: `.venv/bin/python -m pytest tests/test_analysis.py::test_ensemble_stats_with_nan tests/test_analysis.py::test_shannon_diversity_with_zeros tests/test_analysis.py::test_size_spectrum_slope_with_nan tests/test_config_reader.py::test_read_line_with_no_separator tests/test_config_reader.py::test_read_file_with_bom tests/test_config_reader.py::test_read_file_with_multiple_separators -v`

Expected: Some may fail — fix the underlying code if they reveal real bugs.

- [ ] **Step 4: Fix any bugs revealed, run full suite, commit**

Run: `.venv/bin/python -m pytest -x -q`

```bash
git add tests/test_analysis.py tests/test_config_reader.py
git commit -m "test: add NaN/malformed input edge case tests

Cover NaN biomass, zero diversity, BOM config files, and missing
separators. Fixes H14, H15.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 15: Fix brittle source inspection test (M21)

**Files:**
- Modify: `tests/test_sync_config_pages.py:41-47`

- [ ] **Step 1: Replace source inspection with behavioral test**

Replace lines 41-47:

```python
def test_movement_uses_dynamic_species_count():
    """Movement sync should work for arbitrary species counts, not hardcoded 3."""
    from ui.pages.movement import MOVEMENT_FIELDS

    # Verify movement has indexed fields (the ones that need dynamic species count)
    indexed_fields = [f for f in MOVEMENT_FIELDS if f.indexed]
    assert len(indexed_fields) > 0, "Movement should have species-indexed fields"
    # Verify keys use sp{idx} pattern (resolvable for any species count)
    for f in indexed_fields:
        assert "{idx}" in f.key_pattern or "sp{idx}" in f.key_pattern, (
            f"Field {f.key_pattern} should use sp{{idx}} for dynamic species indexing"
        )
```

- [ ] **Step 2: Run test, commit**

Run: `.venv/bin/python -m pytest tests/test_sync_config_pages.py::test_movement_uses_dynamic_species_count -v`

```bash
git add tests/test_sync_config_pages.py
git commit -m "test: replace brittle source inspection with behavioral test

Check field patterns instead of grepping source code. Fixes M21.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 16: Add parallel calibration test (M20)

**Files:**
- Modify: `tests/test_calibration_problem.py`

- [ ] **Step 1: Add test with real parallel execution**

```python
# tests/test_calibration_problem.py — add at end
import time as _time


def test_evaluate_parallel_real_execution(tmp_path):
    """Parallel evaluation should run candidates concurrently."""
    import numpy as np
    from unittest.mock import patch
    from osmose.calibration.problem import OsmoseCalibrationProblem, FreeParameter

    problem = OsmoseCalibrationProblem(
        free_params=[FreeParameter(key="p", lower_bound=0, upper_bound=1)],
        objective_fns=[lambda r: 1.0],
        base_config_path=tmp_path / "c.csv",
        jar_path=tmp_path / "f.jar",
        work_dir=tmp_path,
        n_parallel=4,
    )

    def slow_evaluate(i, params):
        _time.sleep(0.1)
        return [float(params[0])]

    X = np.array([[0.1], [0.2], [0.3], [0.4]])
    out = {}

    with patch.object(problem, "_evaluate_candidate", side_effect=slow_evaluate):
        start = _time.time()
        problem._evaluate(X, out)
        elapsed = _time.time() - start

    # 4 tasks at 0.1s each: serial = 0.4s, parallel should be ~0.1s
    assert elapsed < 0.35, f"Parallel execution too slow: {elapsed:.2f}s (expected <0.35s)"
    assert out["F"].shape == (4, 1)
```

- [ ] **Step 2: Run test, commit**

Run: `.venv/bin/python -m pytest tests/test_calibration_problem.py::test_evaluate_parallel_real_execution -v`

```bash
git add tests/test_calibration_problem.py
git commit -m "test: add real parallel execution test for calibration

Verifies ThreadPoolExecutor concurrency with timing assertion. Fixes M20.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 17: Narrow `get_theme_mode` exception

**Files:**
- Modify: `ui/state.py:119-128`

- [ ] **Step 1: Narrow the exception**

Replace lines 119-128 of `ui/state.py`:

```python
def get_theme_mode(input: object) -> str:
    """Safely read theme_mode from Shiny input, defaulting to 'light'."""
    try:
        mode = input.theme_mode()  # type: ignore[attr-defined]
        return mode if mode in ("dark", "light") else "light"
    except (AttributeError, TypeError):
        return "light"
```

- [ ] **Step 2: Run full test suite, commit**

Run: `.venv/bin/python -m pytest -x -q`

```bash
git add ui/state.py
git commit -m "fix: narrow get_theme_mode exception to AttributeError/TypeError

Prevents masking framework errors while still handling missing input.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 4: Phase 4 — Medium-Priority Improvements (Tasks 18-25)

### Task 18: Validate java_opts against allowlist (M1)

**Files:**
- Modify: `ui/pages/run.py:246-247`
- Test: `tests/test_ui_run.py`

- [ ] **Step 1: Write test for java_opts validation**

```python
# tests/test_ui_run.py — add at end
import pytest


def test_validate_java_opts_allows_memory():
    from ui.pages.run import validate_java_opts

    assert validate_java_opts("-Xmx4g -Xms1g") == ["-Xmx4g", "-Xms1g"]


def test_validate_java_opts_rejects_agent():
    from ui.pages.run import validate_java_opts

    with pytest.raises(ValueError, match="[Ff]orbidden"):
        validate_java_opts("-agentlib:jdwp=transport=dt_socket,server=y")


def test_validate_java_opts_rejects_javaagent():
    from ui.pages.run import validate_java_opts

    with pytest.raises(ValueError, match="[Ff]orbidden"):
        validate_java_opts("-javaagent:/path/to/evil.jar")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ui_run.py::test_validate_java_opts_allows_memory -v`
Expected: FAIL — `validate_java_opts` not defined

- [ ] **Step 3: Implement validation function**

Add to `ui/pages/run.py` (before the server function):

```python
_FORBIDDEN_JAVA_OPTS = ("-agent", "-javaagent", "-XX:OnOutOfMemoryError", "-XX:OnError")


def validate_java_opts(opts_text: str) -> list[str]:
    """Parse and validate Java options, rejecting dangerous flags."""
    if not opts_text.strip():
        return []
    opts = opts_text.split()
    for opt in opts:
        lower = opt.lower()
        for forbidden in _FORBIDDEN_JAVA_OPTS:
            if lower.startswith(forbidden.lower()):
                raise ValueError(f"Forbidden Java option: {opt}")
    return opts
```

Then update the handler (line 246-247) to use it:

```python
        java_opts_text = input.java_opts() or ""
        try:
            java_opts = validate_java_opts(java_opts_text) or None
        except ValueError as exc:
            ui.notification_show(str(exc), type="error", duration=5)
            return
```

- [ ] **Step 4: Run tests, commit**

Run: `.venv/bin/python -m pytest tests/test_ui_run.py -v`

```bash
git add ui/pages/run.py tests/test_ui_run.py
git commit -m "fix: validate java_opts against forbidden flag allowlist

Reject -agent, -javaagent, -XX:OnOutOfMemoryError flags that could
open debug ports or execute commands. Fixes M1.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 19: Config reader sub-file path validation (M4)

**Files:**
- Modify: `osmose/config/reader.py:44-48`
- Test: `tests/test_config_reader.py`

- [ ] **Step 1: Write test for sub-file path traversal**

```python
# tests/test_config_reader.py — add at end
def test_read_rejects_subfile_path_traversal(tmp_path):
    """Sub-file references must not escape the config directory."""
    from osmose.config.reader import OsmoseConfigReader

    main = tmp_path / "main.csv"
    main.write_text("osmose.configuration.evil;../../../etc/passwd\n")
    reader = OsmoseConfigReader()
    result = reader.read_file(main)
    # Should not have read /etc/passwd content
    assert "root" not in str(result.values())
```

- [ ] **Step 2: Add path validation to reader**

In `osmose/config/reader.py`, replace the sub-file handling block (around line 44-48):

```python
            if key.startswith("osmose.configuration."):
                sub_path = filepath.parent / value.strip()
                sub_resolved = sub_path.resolve()
                config_root = filepath.parent.resolve()
                if not sub_resolved.is_relative_to(config_root):
                    _log.warning(
                        "Skipping sub-file outside config directory: %s -> %s",
                        key, sub_path,
                    )
                    continue
                if sub_path.exists():
                    self._read_recursive(sub_path, flat, _seen)
```

- [ ] **Step 3: Run tests, commit**

Run: `.venv/bin/python -m pytest tests/test_config_reader.py -v`

```bash
git add osmose/config/reader.py tests/test_config_reader.py
git commit -m "fix: validate sub-file paths stay within config directory

Prevent config reader from following osmose.configuration.* references
that escape the config file's parent directory. Fixes M4.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 20: Batch species sync (M7)

**Files:**
- Modify: `ui/pages/setup.py:147-176`

- [ ] **Step 1: Refactor sync_species_inputs to batch updates**

Replace lines 147-176 of `ui/pages/setup.py`:

```python
    @reactive.effect
    def sync_species_inputs():
        """Auto-sync species table cells to state.config (batched)."""
        with reactive.isolate():
            if state.loading.get():
                return
        n = input.n_species()
        show_adv = input.show_advanced_species()

        # Batch all changes into a single config.set()
        with reactive.isolate():
            cfg = dict(state.config.get())

        changed = False
        if cfg.get("simulation.nspecies") != str(n):
            cfg["simulation.nspecies"] = str(n)
            changed = True

        visible = [f for f in SPECIES_FIELDS if f.indexed and (show_adv or not f.advanced)]
        for i in range(n):
            for field in visible:
                config_key = field.resolve_key(i)
                base_key = field.key_pattern.replace(".sp{idx}", "").replace("{idx}", "").replace(".", "_")
                input_id = f"spt_{base_key}_{i}"
                try:
                    val = getattr(input, input_id)()
                except (AttributeError, TypeError):
                    continue
                if val is not None and cfg.get(config_key) != str(val):
                    cfg[config_key] = str(val)
                    changed = True

        if changed:
            state.config.set(cfg)
            state.dirty.set(True)

        # Update global species names list
        names = [cfg.get(f"species.name.sp{i}", f"Species {i}") for i in range(n)]
        state.species_names.set(names)
```

- [ ] **Step 2: Run full test suite, commit**

Run: `.venv/bin/python -m pytest -x -q`

```bash
git add ui/pages/setup.py
git commit -m "perf: batch species input sync into single config.set()

Replace 60+ individual update_config calls with one batched write.
Reduces reactive invalidation overhead. Fixes M7.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 21: Fix param_table reactivity in Advanced page (M6)

**Files:**
- Modify: `ui/pages/advanced.py:175-228`

- [ ] **Step 1: Use load_trigger instead of direct config dependency**

Replace `param_table` function (lines 175-228) — add `state.load_trigger.get()` and wrap config read in `reactive.isolate()`:

```python
    @render.ui
    def param_table():
        state.load_trigger.get()  # re-render on config load only
        category = input.adv_category()
        search = input.adv_search().lower() if input.adv_search() else ""

        if category == "all":
            fields = REGISTRY.all_fields()
        else:
            fields = REGISTRY.fields_by_category(category)

        if search:
            fields = [
                f
                for f in fields
                if search in f.key_pattern.lower() or search in f.description.lower()
            ]

        if not fields:
            return ui.div("No parameters match your filter.", style=STYLE_EMPTY)

        with reactive.isolate():
            cfg = state.config.get()

        rows = []
        for f in fields[:100]:
            current_val = cfg.get(f.key_pattern, "-")
            rows.append(
                ui.tags.tr(
                    ui.tags.td(f.key_pattern, style=STYLE_MONO_KEY),
                    ui.tags.td(f.param_type.value),
                    ui.tags.td(str(current_val)),
                    ui.tags.td(f.category),
                    ui.tags.td(
                        f.description[:60] + "..." if len(f.description) > 60 else f.description
                    ),
                )
            )

        return ui.tags.div(
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("Key"),
                        ui.tags.th("Type"),
                        ui.tags.th("Current Value"),
                        ui.tags.th("Category"),
                        ui.tags.th("Description"),
                    )
                ),
                ui.tags.tbody(*rows),
                class_="table table-striped table-hover table-sm",
            ),
            style=STYLE_SCROLL_TABLE,
        )
```

- [ ] **Step 2: Run full test suite, commit**

Run: `.venv/bin/python -m pytest -x -q`

```bash
git add ui/pages/advanced.py
git commit -m "perf: use load_trigger for param_table to avoid constant re-renders

Read config inside reactive.isolate() so the table only re-renders on
config load, category change, or search. Fixes M6.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 22: Resilient scenario listing — skip corrupt JSON (M17)

**Files:**
- Modify: `osmose/scenarios.py:88-104`
- Test: `tests/test_scenarios.py`

- [ ] **Step 1: Write test for corrupt JSON handling**

```python
# tests/test_scenarios.py — add at end
def test_list_scenarios_skips_corrupt_json(tmp_path):
    """One corrupt scenario.json should not block listing others."""
    from osmose.scenarios import Scenario, ScenarioManager

    mgr = ScenarioManager(tmp_path / "scenarios")
    mgr.save(Scenario(name="good", config={"a": "1"}))

    bad_dir = tmp_path / "scenarios" / "bad"
    bad_dir.mkdir()
    (bad_dir / "scenario.json").write_text("{invalid json")

    scenarios = mgr.list_scenarios()
    names = [s["name"] for s in scenarios]
    assert "good" in names
    assert "bad" not in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_scenarios.py::test_list_scenarios_skips_corrupt_json -v`
Expected: FAIL — `json.JSONDecodeError` crashes the whole listing

- [ ] **Step 3: Fix list_scenarios**

Replace lines 88-104 of `osmose/scenarios.py`:

```python
    def list_scenarios(self) -> list[dict[str, str]]:
        """List all saved scenarios with basic metadata."""
        results = []
        for d in sorted(self.storage_dir.iterdir()):
            json_path = d / "scenario.json"
            if d.is_dir() and json_path.exists():
                try:
                    with open(json_path) as f:
                        data = json.load(f)
                    results.append(
                        {
                            "name": data["name"],
                            "description": data.get("description", ""),
                            "modified_at": data.get("modified_at", ""),
                            "tags": data.get("tags", []),
                        }
                    )
                except (json.JSONDecodeError, KeyError) as exc:
                    _log.warning("Skipping corrupt scenario in %s: %s", d.name, exc)
        return results
```

- [ ] **Step 4: Run test, full suite, commit**

Run: `.venv/bin/python -m pytest -x -q`

```bash
git add osmose/scenarios.py tests/test_scenarios.py
git commit -m "fix: skip corrupt scenario JSON instead of blocking all listing

One malformed scenario.json no longer prevents listing other scenarios.
Logs warning for corrupt entries. Fixes M17.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 23: Standardize logging initialization (M10)

**Files:**
- Modify: `osmose/scenarios.py:15` — uses `logging.getLogger`
- Modify: `ui/pages/run.py:14` — uses `logging.getLogger` (verify)
- Modify: `ui/components/param_form.py:12` — uses `logging.getLogger` (verify)

- [ ] **Step 1: Search for all stdlib logger usage**

Run: `.venv/bin/python -c "import subprocess; subprocess.run(['grep', '-rn', 'logging.getLogger', 'osmose/', 'ui/'], cwd='.')"`

Identify all files using `logging.getLogger` instead of `setup_logging`.

- [ ] **Step 2: Replace each instance**

For each file found, replace:
```python
import logging
_log = logging.getLogger("osmose.MODULE_NAME")
```
with:
```python
from osmose.logging import setup_logging
_log = setup_logging("osmose.MODULE_NAME")
```

Remove the `import logging` line if no other references to `logging` exist in the file.

Known files to update:
- `osmose/scenarios.py:15` — `_log = logging.getLogger("osmose.scenarios")`

- [ ] **Step 3: Run full test suite, commit**

Run: `.venv/bin/python -m pytest -x -q`

```bash
git add osmose/scenarios.py
git commit -m "refactor: standardize logging to use setup_logging everywhere

Replace bare logging.getLogger() calls with osmose.logging.setup_logging
for consistent formatted output. Fixes M10.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 24: Download handler user feedback (M19)

**Files:**
- Modify: `ui/pages/results.py:581-600`

- [ ] **Step 1: Add notifications for download failures**

In `ui/pages/results.py`, update the `download_results_csv` function (lines 581-600):

Replace lines 585-587:
```python
        out_dir = Path(input.output_dir())
        if not out_dir.is_dir():
            ui.notification_show("Output directory not found.", type="error", duration=5)
            return
```

Replace lines 594-595:
```python
        if df.empty:
            ui.notification_show("No data available for download.", type="warning", duration=3)
            return
```

- [ ] **Step 2: Run full test suite, commit**

Run: `.venv/bin/python -m pytest -x -q`

```bash
git add ui/pages/results.py
git commit -m "fix: add user notifications when download handler fails

Show clear error/warning when output directory is missing or data is
empty, instead of silently returning nothing. Fixes M19.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 25: Config reader sub-file path validation (M4)

Covered in Task 19 above.

---

## Verification

After all tasks are complete:

- [ ] **Run full test suite:** `.venv/bin/python -m pytest -v`
- [ ] **Run linter:** `.venv/bin/ruff check osmose/ ui/ tests/`
- [ ] **Run formatter:** `.venv/bin/ruff format osmose/ ui/ tests/`
- [ ] **Verify findings addressed:**
  - Critical: C1 (Task 1), C2 (Task 2), C3 (Task 3), C4 (Task 4), C5 (Task 2), C6 (Task 6)
  - High: H1 (Task 5), H2 (Task 7), H3 (Task 8), H4 (Task 9), H5 (Task 1), H6 (Task 11), H7-H10 (Task 10), H11 (Task 12), H12 (Task 4), H13 (Task 13), H14-H15 (Task 14)
  - Medium: M1 (Task 18), M3 (Task 5), M4 (Task 19), M5 (Task 5), M6 (Task 21), M7 (Task 20), M8 (Task 9), M10 (Task 23), M14 (Task 11), M17 (Task 22), M19 (Task 24), M20 (Task 16), M21 (Task 15)
  - Deferred to Phase 5: M2, M9, M11, M12, M13, M15, M16, M18, M22, H16, all Low findings
