# Python Run-Progress Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a Python-engine run visibly report progress — a live progress bar + an in-place console line, the live map auto-enabled for spatial configs, and the dead `py_verbosity`/`py_threads` inputs resolved.

**Architecture:** Reuse the engine's existing per-step `step_observer` hook (no engine change) via a pure `make_run_observer` that fans out always-on progress + the optional live snapshot. Progress flows through the same queue→reactive-poll plumbing the live map already uses. A pure `format_progress_label` keeps the year/step math unit-testable.

**Tech Stack:** Python 3.12, Shiny for Python, numba (`set_num_threads`), pytest + Playwright (e2e). No new dependencies.

---

## File Structure

- **Modify:** `osmose/live_movement.py` — add pure `make_run_observer`, `config_is_spatial`, `format_progress_label`.
- **Modify:** `ui/pages/run.py` — progress queue/poll/`_progress`, `run_progress` render, console progress line, compose run observer in `handle_run`, wire `py_threads`→`numba.set_num_threads`, remove `py_verbosity`, auto-enable live switch.
- **Create:** `tests/test_run_observer.py` — unit tests for the three pure helpers.
- **Modify:** `tests/test_ui_run_capability.py` — drop `py_verbosity` from the input-id assertion.
- **Modify:** `tests/test_e2e_live_movement.py`, `tests/test_e2e_baltic.py` — remove the manual `#live_movement_view` clicks (auto-on for Baltic); add a plain-run progress assertion.

Run tests with `.venv/bin/python -m pytest`. Lint: `.venv/bin/ruff check` + `--format check`.

---

## Task 1: Pure helpers in `osmose/live_movement.py`

**Files:**
- Modify: `osmose/live_movement.py`
- Test: `tests/test_run_observer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_observer.py
import queue
import types

import pytest

from osmose.live_movement import config_is_spatial, format_progress_label, make_run_observer


def _cfg(n_steps):
    return types.SimpleNamespace(n_steps=n_steps)


def test_run_observer_pushes_one_based_done_every_step():
    q: queue.Queue = queue.Queue(maxsize=1)
    obs = make_run_observer(q)
    obs(0, None, None, _cfg(24))  # step 0 -> done 1
    done, n, elapsed = q.get_nowait()
    assert done == 1 and n == 24 and elapsed >= 0.0
    obs(5, None, None, _cfg(24))  # step 5 -> done 6
    done, n, elapsed = q.get_nowait()
    assert done == 6 and n == 24 and elapsed >= 0.0


def test_run_observer_delegates_to_live_observer():
    q: queue.Queue = queue.Queue(maxsize=1)
    seen = []
    obs = make_run_observer(q, live_observer=lambda s, st, g, c: seen.append(s))
    obs(3, "st", "g", _cfg(10))
    assert seen == [3]
    assert q.get_nowait()[0] == 4  # done = step+1


def test_run_observer_without_live_observer_still_pushes_progress():
    q: queue.Queue = queue.Queue(maxsize=1)
    obs = make_run_observer(q, live_observer=None)
    obs(0, None, None, _cfg(10))
    assert q.get_nowait() == (1, 10, pytest.approx(0.0, abs=5.0))


def test_run_observer_swallows_live_observer_exception():
    q: queue.Queue = queue.Queue(maxsize=1)

    def boom(s, st, g, c):
        raise RuntimeError("live boom")

    obs = make_run_observer(q, live_observer=boom)
    obs(0, None, None, _cfg(10))  # must NOT raise
    assert q.get_nowait()[0] == 1  # progress still pushed (pushed before delegate)


def test_config_is_spatial():
    baltic = {
        "grid.nlon": "50", "grid.nlat": "40",
        "grid.upleft.lat": "66", "grid.upleft.lon": "10",
        "grid.lowright.lat": "54", "grid.lowright.lon": "30",
    }
    assert config_is_spatial(baltic) is True
    assert config_is_spatial({"grid.netcdf.file": "eec_grid-mask.nc"}) is False
    assert config_is_spatial({}) is False


def test_format_progress_label_year_off_by_one():
    # 1-year run, final tick: done == ndt must give "Year 1/1" (NOT "Year 2/1")
    assert "Year 1/1" in format_progress_label(24, 24, 24)
    # first step of year 2 in a 2-year run
    assert format_progress_label(25, 48, 24).startswith("Year 2/2")
    # first step
    assert format_progress_label(1, 24, 24).startswith("Year 1/1")
    # ndt <= 0 -> step-only label, no ZeroDivisionError, no "Year"
    s = format_progress_label(3, 10, 0)
    assert s.startswith("Step 3/10") and "Year" not in s
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_run_observer.py -v`
Expected: FAIL — `ImportError: cannot import name 'make_run_observer'`

- [ ] **Step 3: Write the implementation**

Add to `osmose/live_movement.py`. Add the import near the top (with the existing imports):

```python
from osmose.maps.builder import GridSpec
```

Then add the three functions (after `make_step_observer`):

```python
def make_run_observer(
    progress_q: "queue.Queue[tuple[int, int, float]]",
    live_observer: "Callable[[int, object, object, object], None] | None" = None,
    *,
    now: Callable[[], float] = time.monotonic,
) -> Callable[[int, object, object, object], None]:
    """Step-observer that pushes (done, n_steps, elapsed_s) to progress_q every step
    (done = step + 1, 1-based) and delegates to live_observer when given.

    Never raises into the engine. Progress is pushed BEFORE delegating, so a failing
    live_observer cannot suppress progress. Drop-oldest on a maxsize-1 queue.
    """
    start: list[float | None] = [None]

    def observer(step: int, state, grid, config) -> None:
        try:
            if start[0] is None:
                start[0] = now()
            done = step + 1
            n = int(config.n_steps)
            elapsed = now() - start[0]
            try:
                progress_q.put_nowait((done, n, elapsed))
            except queue.Full:
                try:
                    progress_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    progress_q.put_nowait((done, n, elapsed))
                except queue.Full:
                    pass
            if live_observer is not None:
                live_observer(step, state, grid, config)
        except Exception:  # noqa: BLE001 — never crash the running simulation
            _log.warning("run observer failed at step %s", step, exc_info=True)

    return observer


def config_is_spatial(config: dict[str, str]) -> bool:
    """True when the config has a regular grid that yields live-movement frames
    (GridSpec.from_config succeeds). NcGrid configs (only grid.netcdf.file) -> False.
    """
    try:
        GridSpec.from_config(config)
        return True
    except (KeyError, ValueError, TypeError):
        return False


def format_progress_label(done: int, n_steps: int, ndt: int) -> str:
    """Human progress label from 1-based completed-step count `done`.

    ndt > 0 -> 'Year y/ny · step done/n · pct%' (year bucket uses (done-1)//ndt to
    convert the 1-based done back to a 0-based index, so a 1-year run's final tick
    done==ndt gives Year 1/1, not Year 2/1). ndt <= 0 -> step-only label (no division).
    """
    pct = round(done / n_steps * 100) if n_steps else 0
    if ndt and ndt > 0:
        year = (done - 1) // ndt + 1
        n_years = -(-n_steps // ndt)  # ceil division
        return f"Year {year}/{n_years} · step {done}/{n_steps} · {pct}%"
    return f"Step {done}/{n_steps} · {pct}%"
```

(Confirm `time`, `queue`, `Callable`, and `_log` are already imported in `live_movement.py` — they are, used by `make_step_observer`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_run_observer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/live_movement.py tests/test_run_observer.py
git commit -m "feat(run): pure run-observer + spatial predicate + progress label"
```

---

## Task 2: Progress plumbing, bar, and console line (`ui/pages/run.py`)

**Files:**
- Modify: `ui/pages/run.py`

Add the progress queue/value/poll, the `run_progress` render, the console progress line, and compose the run observer in `handle_run`. The Python branch currently passes `live_observer` to the thread; it must pass the composed observer and reset/clear `_progress`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ui_run_capability.py
import ui.pages.run as _run_page  # noqa: E402  (module already imported in this file)


def test_run_page_has_progress_machinery():
    text = open(_run_page.__file__, encoding="utf-8").read()
    assert 'output_ui("run_progress")' in text
    assert "make_run_observer" in text
    assert "_progress" in text
    assert "format_progress_label" in text
```

(If `import ui.pages.run as run_page` already exists at the top of the file, reuse that name instead of re-importing.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ui_run_capability.py::test_run_page_has_progress_machinery -v`
Expected: FAIL — machinery absent.

- [ ] **Step 3: Make the change**

(a) Update the import at the top of `ui/pages/run.py`:

```python
from osmose.live_movement import (
    config_is_spatial,
    format_progress_label,
    make_run_observer,
    make_step_observer,
)
```

(b) In `run_ui()`, add the progress bar output in the Run Status block (after `ui.h5("Run Status")`, before/after `ui.output_text("run_status")`):

```python
                ui.h5("Run Status"),
                ui.output_ui("run_progress"),
                ui.output_text("run_status"),
```

(c) In `run_server()`, near the other live-movement reactive values (after `_run_config_cell`), add:

```python
    _progress_q: queue.Queue = queue.Queue(maxsize=1)  # (done, n_steps, elapsed_s)
    _progress: reactive.Value = reactive.Value(None)  # None | (done, n_steps, elapsed_s)
```

(d) Add the progress poll + consumer (next to `_drain_live_queue`/`_consume_live_poll`):

```python
    @reactive.poll(lambda: time.time(), interval_secs=0.2)
    def _drain_progress():
        latest = None
        while True:
            try:
                latest = _progress_q.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            _progress.set(latest)

    @reactive.effect
    def _consume_progress():
        _drain_progress()
```

(e) Add the `run_progress` render (next to `run_status`):

```python
    @render.ui
    def run_progress():
        prog = _progress.get()
        if prog is None:
            return None
        done, n, _elapsed = prog
        try:
            ndt = int(float(state.config.get().get("simulation.time.ndtperyear", "0") or "0"))
        except (TypeError, ValueError):
            ndt = 0
        label = format_progress_label(done, n, ndt)
        pct = round(done / n * 100) if n else 0
        return ui.div(
            ui.div(
                f"{pct}%",
                class_="progress-bar",
                role="progressbar",
                style=f"width: {pct}%",
            ),
            ui.tags.small(label, class_="text-muted"),
            class_="progress mb-1",
            style="height: auto;",
        )
```

(f) Update `run_console` to append the in-place progress line while running:

```python
    @render.ui
    def run_console():
        lines = run_log.get()
        prog = _progress.get()
        text = "\n".join(lines[-200:]) if lines else ""
        if prog is not None:
            done, n, _elapsed = prog
            pct = round(done / n * 100) if n else 0
            prog_line = f"running · step {done}/{n} ({pct}%)"
            text = f"{text}\n{prog_line}" if text else prog_line
        if not text:
            text = "No output yet. Click 'Start Run' to begin."
        return ui.tags.pre(text, style=STYLE_CONSOLE)
```

(g) In `handle_run`, reset `_progress` at the very TOP (before all early returns):

```python
    async def handle_run():
        _progress.set(None)
        engine_mode = state.engine_mode.get()
        ...
```

(h) In the Python branch of `handle_run`, build the composed observer and pass it to the thread (replace the bare `live_observer` argument):

```python
            run_observer = make_run_observer(_progress_q, live_observer)
            ...
            threading.Thread(
                target=_python_engine_thread,
                args=(run_config, output_dir, cancel_token, run_observer, _run_done_q),
                daemon=True,
            ).start()
```

(i) In `_drain_run_done`, clear `_progress` on every terminal outcome (add near where it sets `_live_status_val`/buttons):

```python
        _progress.set(None)
```

- [ ] **Step 4: Run test + import smoke**

Run: `.venv/bin/python -m pytest tests/test_ui_run_capability.py::test_run_page_has_progress_machinery -v && .venv/bin/python -c "import app"`
Expected: PASS and clean import.

- [ ] **Step 5: Commit**

```bash
git add ui/pages/run.py tests/test_ui_run_capability.py
git commit -m "feat(run): Python progress bar + in-place console line (reuses step_observer)"
```

---

## Task 3: Wire `py_threads`, remove `py_verbosity`

**Files:**
- Modify: `ui/pages/run.py`
- Modify: `tests/test_ui_run_capability.py`

- [ ] **Step 1: Update the existing assertion test (failing first)**

In `tests/test_ui_run_capability.py`, the `test_run_page_uses_panel_conditional_for_engine_settings` tuple includes `"py_verbosity"` (~line 23). Remove `"py_verbosity"` from that tuple. Add a positive assertion that `py_threads` is wired:

```python
def test_py_threads_wired_and_verbosity_removed():
    text = open(run_page.__file__, encoding="utf-8").read()
    assert "py_verbosity" not in text          # widget removed
    assert "set_num_threads" in text           # py_threads now wired
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ui_run_capability.py::test_py_threads_wired_and_verbosity_removed -v`
Expected: FAIL — `py_verbosity` still present, `set_num_threads` absent.

- [ ] **Step 3: Make the change**

(a) In `run_ui()` Python `panel_conditional`, change the `py_threads` widget and DELETE the `py_verbosity` select:

```python
                ui.panel_conditional(
                    "input.engine_mode === 'python'",
                    ui.input_numeric(
                        "py_threads",
                        "Threads (Numba; 0 = auto/all cores)",
                        value=0,
                        min=0,
                        max=32,
                    ),
                    ui.input_text_area(
                        "py_param_overrides",
                        "Parameter overrides (key=value, one per line)",
                        rows=4,
                    ),
                ),
```

(b) Add the import at the top of `ui/pages/run.py`:

```python
import numba  # type: ignore[import-untyped]
```

(c) In the Python branch of `handle_run`, before starting the thread, wire threads:

```python
            try:
                n_threads = int(input.py_threads() or 0)
                if n_threads >= 1:
                    numba.set_num_threads(min(n_threads, numba.config.NUMBA_NUM_THREADS))
                else:
                    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)  # Auto = all cores
            except Exception:  # noqa: BLE001 — never block a run on a bad thread count
                _log.warning("could not apply py_threads; using Numba default", exc_info=True)
```

- [ ] **Step 4: Run test + import smoke**

Run: `.venv/bin/python -m pytest tests/test_ui_run_capability.py -v && .venv/bin/python -c "import app"`
Expected: PASS (all capability tests incl. the updated tuple) and clean import.

- [ ] **Step 5: Commit**

```bash
git add ui/pages/run.py tests/test_ui_run_capability.py
git commit -m "feat(run): wire py_threads -> numba.set_num_threads (0=auto), drop dead py_verbosity"
```

---

## Task 4: Auto-enable the live map for spatial configs

**Files:**
- Modify: `ui/pages/run.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ui_run_capability.py
def test_run_page_auto_enables_live_for_spatial():
    text = open(run_page.__file__, encoding="utf-8").read()
    assert "config_is_spatial" in text
    assert 'update_switch("live_movement_view"' in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ui_run_capability.py::test_run_page_auto_enables_live_for_spatial -v`
Expected: FAIL — auto-enable effect absent.

- [ ] **Step 3: Make the change**

In `run_server()`, add a changed-only auto-enable effect (mirrors the `_last_live_species` guard pattern). Near the other effects, add a plain mutable cell and the effect:

```python
    _last_spatial: list[bool | None] = [None]  # changed-only guard for auto-enable

    @reactive.effect
    def _auto_enable_live_for_spatial():
        config = state.config.get()
        if not config:
            return
        spatial = config_is_spatial(config)
        if spatial == _last_spatial[0]:
            return
        _last_spatial[0] = spatial
        ui.update_switch("live_movement_view", value=spatial, session=session)
```

This fires only when the loaded config's spatial-ness *changes*, so it auto-sets on config-load and does not stomp a user's manual toggle within the same config. `handle_run`'s gate (`if input.live_movement_view() and engine_mode == "python"`) is unchanged.

- [ ] **Step 4: Run test + import smoke**

Run: `.venv/bin/python -m pytest tests/test_ui_run_capability.py::test_run_page_auto_enables_live_for_spatial -v && .venv/bin/python -c "import app"`
Expected: PASS and clean import.

- [ ] **Step 5: Commit**

```bash
git add ui/pages/run.py
git commit -m "feat(run): auto-enable live movement for spatial (regular-grid) configs"
```

---

## Task 5: Update e2e (remove manual toggle clicks, add plain-run progress)

**Files:**
- Modify: `tests/test_e2e_live_movement.py`, `tests/test_e2e_baltic.py`

With auto-enable, Baltic is spatial → the live switch is already on; the existing manual `.click()` would turn it OFF. Remove those clicks and add a plain-run progress assertion.

- [ ] **Step 1: Remove the manual toggle clicks**

- `tests/test_e2e_live_movement.py:58` and `:96` — delete `page.locator("#live_movement_view").click()` (the switch is auto-on for Baltic). Keep the rest.
- `tests/test_e2e_baltic.py:57` — delete `page.locator("#live_movement_view").click()`.

- [ ] **Step 2: Add a plain-run progress assertion**

Append to `tests/test_e2e_live_movement.py`:

```python
def test_run_progress_shows_during_python_run(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
    dismiss_changelog_modal(page)
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.select_option("#load_example", "baltic")
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=_LOAD_TIMEOUT)
    page.locator(".nav-pills .nav-link[data-value='run']").click()
    page.locator("#engineBtnPython").click()
    py_overrides = page.locator("#py_param_overrides")
    expect(py_overrides).to_be_visible(timeout=_LOAD_TIMEOUT)
    py_overrides.fill("simulation.time.nyear=1")
    # Do NOT touch the live toggle — progress must appear regardless.
    page.locator("#btn_run").click()
    # The progress bar appears and the console shows a 'step N/' line during the run.
    expect(page.locator("#run_progress")).to_contain_text("step", timeout=_RUN_TIMEOUT)
    expect(page.locator("#run_console")).to_contain_text("step", timeout=_RUN_TIMEOUT)
    expect(page.locator("#run_status")).to_contain_text("Complete", timeout=_RUN_TIMEOUT)
```

- [ ] **Step 3: Run the e2e**

Run: `.venv/bin/python -m pytest tests/test_e2e_live_movement.py tests/test_e2e_baltic.py -m e2e -v`
Expected: PASS (all cases incl. the new progress test; the two/one de-clicked cases still go running→done because the switch is auto-on).
If Playwright/browser is unavailable, note it and fall back to confirming the source edits.

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e_live_movement.py tests/test_e2e_baltic.py
git commit -m "test(e2e): drop manual live-toggle clicks (auto-on); assert plain-run progress"
```

---

## Task 6: Full suite + lint/format/pyright

**Files:** none (verification only)

- [ ] **Step 1: Full suite**

Run: `.venv/bin/python -m pytest -q -n auto`
Expected: all pass (new run-observer tests + capability tests + existing suite; e2e excluded by default `-m 'not e2e'`).

- [ ] **Step 2: Lint + format (matches CI)**

Run: `.venv/bin/ruff check osmose/ ui/ tests/ && .venv/bin/ruff format --check osmose/ ui/ tests/`
Expected: clean. If format fails, `.venv/bin/ruff format osmose/ ui/ tests/` and re-commit.

- [ ] **Step 3: Pyright on the changed code**

Run: `.venv/bin/python -m pyright --pythonpath .venv/bin/python osmose/live_movement.py ui/pages/run.py`
Expected: 0 errors. (Don't run bare `pyright` — CI-pyright-reproduction gotcha.)

- [ ] **Step 4: e2e smoke (the real proof)**

Run: `.venv/bin/python -m pytest tests/test_e2e_live_movement.py -m e2e -v`
Expected: PASS — confirms a plain Python run shows progress without touching the toggle.

- [ ] **Step 5: Commit any fixups**

```bash
git add -A
git commit -m "chore(run): lint/format/pyright fixups" || echo "nothing to commit"
```

---

## Self-Review notes (applied)

- **Spec coverage:** pure helpers (Task 1) → progress plumbing/bar/console (Task 2) → threads/verbosity (Task 3) → auto-enable (Task 4) → e2e (Task 5) → gates (Task 6). All spec components covered.
- **1-based `done` convention** is used consistently: `make_run_observer` pushes `done = step+1`; `_progress`, `run_progress`, `run_console`, and `format_progress_label` all treat the first element as 1-based; the year math is `(done-1)//ndt+1` (regression-locked in Task 1's `test_format_progress_label_year_off_by_one`).
- **Lifecycle:** `_progress` reset at top of `handle_run` (before all early returns) + cleared in `_drain_run_done`; Java path never feeds `_progress` → `run_progress` is Python-only.
- **ndtperyear** read with the lowercase in-memory key (mirrors grid.py:392), with a step-only fallback when `ndt<=0` (no division-by-zero).
- **py_threads** idempotent (always `set_num_threads`; `n<1`→all cores), default flipped to `0`/min `0` to avoid the single-thread slowdown; **process-wide side effect** noted.
- **Out of scope:** Java-path changes, engine internals, per-step ecology stats.
