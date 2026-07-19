# Surface Python-engine Warnings in the UI Console — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On a UI Python-engine run, stream the engine's `WARNING+` `osmose` logs (chiefly the #120/#123 warnings) into the run Console Output panel — live — instead of only to the server's stderr, so UI users actually see them.

**Architecture:** A `logging.Handler` (`_QueueLogHandler`) attached to the `osmose` logger for the duration of one Python run, filtered to that run's thread ident, posts formatted `WARNING+` records to the existing `_run_log_q` (which `_drain_run_log` already streams into the console). A new `osmose.engine.reset_run_warnings()` clears the three process-global warning-dedup sets at each UI run start so every run re-emits its warnings.

**Tech Stack:** Python 3.12/3.13, stdlib `logging`/`threading`/`queue`, pytest, Shiny (UI). No new dependencies.

## Global Constraints

Copied verbatim from `docs/superpowers/specs/2026-07-19-ui-surface-python-warnings-design.md`. Every task implicitly includes these.

- **Isolation is by thread, not just time.** `_QueueLogHandler` forwards ONLY records whose `record.thread` equals the ident captured (via `threading.get_ident()`) inside `_python_engine_thread` — so a concurrent Python run in another Shiny session (shared global `osmose` logger, separate `_run_log_q`) cannot leak into this console.
- **Capture `WARNING+` only.** Handler level `logging.WARNING`. Format `"%(levelname)s: %(message)s"`.
- **Attach the handler as the FIRST statement inside the `try`; detach in `finally`.** No handler may leak onto the process-global `osmose` logger.
- **`reset_run_warnings()` clears all THREE dedup sets:** `config_validation._WARNED_JAVA_ONLY_KEYS`, `config._WARNED_UNSUPPORTED_RESTART`, `config._WARNED_UNSUPPORTED_MORTALITY`. Called at UI run start (main thread) ONLY — CLI/batch must not call it (their once-per-process dedup stays).
- **Do NOT add `run_log.set([])` at the Python launch site.** The Python path already sets a fresh console at `run.py:889-894` (validation block or `[]`); a launch-clear would wipe it. Streamed warnings append below it via `_drain_run_log`.
- **`ui/pages/run.py` needs stdlib `import logging` added** (it currently imports only `from osmose.logging import setup_logging`).
- Final gate = full suite. Known pre-existing flake `test_trophic_cascade_visible` (skipif CI; fails on base too) is NOT a blocker.

---

### Task 1: `reset_run_warnings()` in the engine package

Clears the three process-global warning-dedup sets so the next run re-emits. Engine-only, no UI — independently testable.

**Files:**
- Modify: `osmose/engine/__init__.py` (add a module-level `reset_run_warnings()` function)
- Test: `tests/test_reset_run_warnings.py` (new)

**Interfaces:**
- Produces: `osmose.engine.reset_run_warnings() -> None` — clears `config_validation._WARNED_JAVA_ONLY_KEYS`, `config._WARNED_UNSUPPORTED_RESTART`, `config._WARNED_UNSUPPORTED_MORTALITY`. Task 2 calls it at the UI launch site.

- [ ] **Step 1: Write the failing test**

Create `tests/test_reset_run_warnings.py`:

```python
"""reset_run_warnings clears all three engine warning-dedup sets so each UI run re-emits."""


def test_reset_run_warnings_clears_all_three_dedup_sets():
    from osmose.engine import reset_run_warnings
    from osmose.engine import config as cfg_mod
    from osmose.engine import config_validation as cv

    cv._WARNED_JAVA_ONLY_KEYS.add("fingerprint-a")
    cfg_mod._WARNED_UNSUPPORTED_RESTART.add("restart-msg")
    cfg_mod._WARNED_UNSUPPORTED_MORTALITY.add("mortality-msg")

    reset_run_warnings()

    assert cv._WARNED_JAVA_ONLY_KEYS == set()
    assert cfg_mod._WARNED_UNSUPPORTED_RESTART == set()
    assert cfg_mod._WARNED_UNSUPPORTED_MORTALITY == set()


def test_reset_run_warnings_lets_a_warning_re_emit():
    import logging

    from osmose.engine import reset_run_warnings
    from osmose.engine.config_validation import warn_unread_java_only_keys

    cfg = {"simulation.ncpu": "8"}  # a java-only key -> #123 warning

    caplog_records = []
    h = logging.Handler()
    h.emit = lambda r: caplog_records.append(r.getMessage())
    logging.getLogger("osmose.config").addHandler(h)
    logging.getLogger("osmose.config").setLevel(logging.WARNING)
    try:
        warn_unread_java_only_keys(cfg)          # first emit populates the dedup set
        first = sum("issue #123" in m for m in caplog_records)
        warn_unread_java_only_keys(cfg)          # deduped -> no second emit
        deduped = sum("issue #123" in m for m in caplog_records)
        reset_run_warnings()                     # clear -> next call re-emits
        warn_unread_java_only_keys(cfg)
        after_reset = sum("issue #123" in m for m in caplog_records)
    finally:
        logging.getLogger("osmose.config").removeHandler(h)

    assert first == 1
    assert deduped == 1          # the 2nd call was suppressed by dedup
    assert after_reset == 2      # reset let it re-emit
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/razinka/osmopy && python -m pytest tests/test_reset_run_warnings.py -q`
Expected: FAIL — `cannot import name 'reset_run_warnings' from 'osmose.engine'`.

- [ ] **Step 3: Implement `reset_run_warnings`**

In `osmose/engine/__init__.py`, add at module level (after the existing imports/`PythonEngine` class; a top-level function is fine):

```python
def reset_run_warnings() -> None:
    """Clear the engine's per-process warning-dedup caches so the next run re-emits its warnings.

    UI-run-scoped: the interactive UI calls this at each run start; CLI/batch do NOT, keeping their
    once-per-process dedup (which prevents spamming a many-sim calibration). Enumerates every engine
    WARNING-dedup set — add new ones here.
    """
    from osmose.engine import config as _config
    from osmose.engine import config_validation as _cv

    _cv._WARNED_JAVA_ONLY_KEYS.clear()
    _config._WARNED_UNSUPPORTED_RESTART.clear()
    _config._WARNED_UNSUPPORTED_MORTALITY.clear()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/razinka/osmopy && python -m pytest tests/test_reset_run_warnings.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/razinka/osmopy && git add osmose/engine/__init__.py tests/test_reset_run_warnings.py
git commit -m "feat: reset_run_warnings() clears the engine warning-dedup caches (per-UI-run)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `_QueueLogHandler` + wire it into the Python run path

The UI log bridge: a thread-filtered handler streaming `osmose` `WARNING+` into the run console, attached for the duration of a Python run, plus the launch-site wiring.

**Files:**
- Modify: `ui/pages/run.py` — add `import logging`; add `_QueueLogHandler`; add a `log_q` param to `_python_engine_thread` and attach/detach the handler; at the launch site (`run.py:944`) pass `_run_log_q` and call `reset_run_warnings()`; update `_drain_run_log`'s stale docstring.
- Test: `tests/test_ui_python_warnings_console.py` (new)

**Interfaces:**
- Consumes: `osmose.engine.reset_run_warnings()` (Task 1); the existing module-globals `_run_log_q` (`run.py:509`) and `_drain_run_log` (`run.py:585`).
- Produces: `ui.pages.run._QueueLogHandler(log_q, thread_id, level=logging.WARNING)`; `_python_engine_thread(run_config, output_dir, cancel_token, step_observer, done_q, n_threads=0, log_q=None)` (new trailing `log_q` param).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ui_python_warnings_console.py`:

```python
"""The Python run path streams osmose WARNING+ logs into the run console queue (thread-isolated)."""

import logging
import queue
import tempfile
import threading
from pathlib import Path


def test_queue_log_handler_forwards_warning_on_matching_thread():
    from ui.pages.run import _QueueLogHandler

    q: queue.Queue = queue.Queue()
    h = _QueueLogHandler(q, threading.get_ident())
    log = logging.getLogger("osmose.config")
    log.addHandler(h)
    log.setLevel(logging.WARNING)
    try:
        log.warning("hello from the run")
        log.info("info is below WARNING")  # must be dropped by level
    finally:
        log.removeHandler(h)

    lines = []
    while not q.empty():
        lines.append(q.get_nowait())
    assert lines == ["WARNING: hello from the run"]  # exactly the WARNING, formatted, no INFO


def test_queue_log_handler_drops_records_from_a_different_thread():
    from ui.pages.run import _QueueLogHandler

    q: queue.Queue = queue.Queue()
    h = _QueueLogHandler(q, threading.get_ident())  # bound to THIS thread
    log = logging.getLogger("osmose.config")
    log.addHandler(h)
    log.setLevel(logging.WARNING)
    try:
        # emit from a DIFFERENT thread -> record.thread differs -> must be dropped (no leak)
        t = threading.Thread(target=lambda: log.warning("from another session's run"))
        t.start()
        t.join()
        log.warning("from this run")  # same thread -> forwarded
    finally:
        log.removeHandler(h)

    lines = []
    while not q.empty():
        lines.append(q.get_nowait())
    assert lines == ["WARNING: from this run"]  # the other thread's warning was filtered out


def test_queue_log_handler_never_raises_on_broken_queue():
    from ui.pages.run import _QueueLogHandler

    class _Full:
        def put_nowait(self, _):
            raise RuntimeError("queue exploded")

    h = _QueueLogHandler(_Full(), threading.get_ident())
    rec = logging.LogRecord("osmose.config", logging.WARNING, __file__, 1, "x", None, None)
    h.emit(rec)  # must NOT raise (a log line can't break a run)


def test_python_engine_thread_streams_120_and_123_and_detaches(tmp_path):
    from osmose.config.reader import OsmoseConfigReader
    from ui.pages.run import _QueueLogHandler, _python_engine_thread

    cfg = dict(OsmoseConfigReader().read("data/minimal/osm_all-parameters.csv"))
    cfg["_osmose.config.dir"] = "data/minimal"     # so the engine resolves the bundled data files
    cfg["simulation.ncpu"] = "8"                   # java-only -> #123
    cfg["oxygen.factor"] = "1.0"                   # java-only -> #123
    cfg["simulation.restart.file"] = "snap.nc"     # -> #120
    cfg["simulation.time.nyear"] = "1"             # keep the run short

    out = tmp_path / "output"
    out.mkdir()
    log_q: queue.Queue = queue.Queue()
    done_q: queue.Queue = queue.Queue()

    # Runs synchronously here (a thread in prod). The #120/#123 warnings fire in _prepare_run on
    # THIS thread, so the thread-filtered handler forwards them regardless of the run's outcome.
    _python_engine_thread(cfg, out, threading.Event(), None, done_q, 0, log_q)

    lines = []
    while not log_q.empty():
        lines.append(log_q.get_nowait())
    assert any("see issue #123" in ln for ln in lines), f"no #123 warning streamed; got {lines}"
    assert any("see issue #120" in ln for ln in lines), f"no #120 warning streamed; got {lines}"

    # handler was detached in the finally -> no leak on the global osmose logger
    assert not any(
        isinstance(h, _QueueLogHandler) for h in logging.getLogger("osmose").handlers
    ), "the _QueueLogHandler leaked onto the osmose logger"


def test_python_engine_thread_without_log_q_attaches_no_handler(tmp_path):
    from osmose.config.reader import OsmoseConfigReader
    from ui.pages.run import _QueueLogHandler, _python_engine_thread

    cfg = dict(OsmoseConfigReader().read("data/minimal/osm_all-parameters.csv"))
    cfg["_osmose.config.dir"] = "data/minimal"
    cfg["simulation.time.nyear"] = "1"
    done_q: queue.Queue = queue.Queue()
    out = tmp_path / "out"
    out.mkdir()

    _python_engine_thread(cfg, out, threading.Event(), None, done_q, 0, None)  # log_q=None

    assert not any(
        isinstance(h, _QueueLogHandler) for h in logging.getLogger("osmose").handlers
    ), "no handler should be attached when log_q is None"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/razinka/osmopy && python -m pytest tests/test_ui_python_warnings_console.py -q`
Expected: FAIL — `cannot import name '_QueueLogHandler' from 'ui.pages.run'` (and `_python_engine_thread` has no `log_q` param).

- [ ] **Step 3: Add `import logging` and the `_QueueLogHandler` class**

In `ui/pages/run.py`, add `import logging` to the stdlib import block near the top (beside `import queue` at line 3 / `import threading` at line 6). Then add the handler class near the other module-level run helpers (e.g. just above `_python_engine_thread` at line 307):

```python
class _QueueLogHandler(logging.Handler):
    """Bridge osmose WARNING+ logs from ONE run's thread into that run's console queue (live, like
    the Java jar console). Thread-filtered so a concurrent session's run cannot leak in."""

    def __init__(self, log_q: "queue.Queue", thread_id: int, level: int = logging.WARNING) -> None:
        super().__init__(level)
        self._log_q = log_q
        self._thread_id = thread_id  # only records emitted on THIS run's thread are forwarded
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self._thread_id:
            return  # a different session's run (shared global 'osmose' logger) — not ours
        try:
            self._log_q.put_nowait(self.format(record))
        except Exception:  # noqa: BLE001 — a log line must never break a run
            self.handleError(record)
```

- [ ] **Step 4: Add the `log_q` param + attach/detach in `_python_engine_thread`**

Replace `_python_engine_thread` (`run.py:307-339`) with this (adds `log_q=None`; attaches the handler as the FIRST statement inside the `try`, detaches in `finally`; the `except` bodies are unchanged from the current code):

```python
def _python_engine_thread(run_config, output_dir, cancel_token, step_observer, done_q,
                          n_threads=0, log_q=None):
    """Run the Python engine in a background thread; post the outcome to ``done_q``.

    Fire-and-forget (the calibration-dashboard pattern): runs OFF the main thread so the event
    handler that launched it returns immediately, letting the reactive poll flush live movement
    frames AND run_log/status during the run.

    When ``log_q`` is given, a thread-filtered ``_QueueLogHandler`` is attached to the ``osmose``
    logger for the run's duration, so the engine's WARNING+ logs (the #120/#123 warnings) stream
    live into the run console via ``_drain_run_log`` — mirroring the Java jar-console stream.

    Touches NO reactive state. Posts ``(kind, result_or_None, message)``.
    """
    apply_single_run_threads(n_threads)
    osmose_logger = logging.getLogger("osmose")
    handler = None
    try:
        if log_q is not None:
            handler = _QueueLogHandler(log_q, threading.get_ident())
            osmose_logger.addHandler(handler)  # first, so the finally always covers it
        engine = PythonEngine()
        result = engine.run(
            run_config,
            output_dir,
            seed=0,
            cancel_token=cancel_token,
            step_observer=step_observer,
        )
        done_q.put(("done", result, ""))
    except SimulationCancelled as exc:
        _log.info("Python engine cancelled: %s", exc)
        done_q.put(("cancelled", None, str(exc) or "user cancelled"))
    except Exception as exc:  # noqa: BLE001
        _log.error("Python engine failed: %s", exc, exc_info=True)
        done_q.put(("failed", None, str(exc)))
    finally:
        if handler is not None:
            osmose_logger.removeHandler(handler)
```

- [ ] **Step 5: Wire the launch site — pass `_run_log_q` and reset warnings**

At the Python launch branch (`run.py:938-948`), (a) add `reset_run_warnings()` right before the thread launch, and (b) add `log_q=_run_log_q` to the thread args. Do NOT add `run_log.set([])`. Concretely, change the block that currently reads:

```python
            run_observer = make_run_observer(_progress_q, live_observer)
            n_threads = int(input.py_threads() or 0)
            threading.Thread(
                target=_python_engine_thread,
                args=(run_config, output_dir, cancel_token, run_observer, _run_done_q, n_threads),
                daemon=True,
            ).start()
```

to:

```python
            run_observer = make_run_observer(_progress_q, live_observer)
            n_threads = int(input.py_threads() or 0)
            reset_run_warnings()  # per-UI-run: clear the engine warning-dedup so this run re-emits
            threading.Thread(
                target=_python_engine_thread,
                args=(run_config, output_dir, cancel_token, run_observer, _run_done_q, n_threads,
                      _run_log_q),
                daemon=True,
            ).start()
```

Add `reset_run_warnings` to the existing top-of-file engine import at `run.py:22`, which currently reads `from osmose.engine import PythonEngine, SimulationCancelled` → change it to `from osmose.engine import PythonEngine, SimulationCancelled, reset_run_warnings`. (No circular import: `osmose.engine` does not import `ui`.)

- [ ] **Step 6: Update the stale `_drain_run_log` docstring**

In `run.py:585-586`, change the docstring from "Stream the Java run's console lines (posted off-thread) into run_log on the main thread." to reflect that it now streams both engines' lines:

```python
    def _drain_run_log():
        """Stream a run's console lines (Java jar console, or Python engine WARNING+ logs posted
        by _QueueLogHandler) from _run_log_q into run_log on the main thread."""
```

- [ ] **Step 7: Run the new tests to verify they pass**

Run: `cd /home/razinka/osmopy && python -m pytest tests/test_ui_python_warnings_console.py -v`
Expected: all five tests PASS (handler forwards WARNING / drops INFO / drops other-thread / never-raises; `_python_engine_thread` streams #120+#123 and detaches; `log_q=None` attaches nothing).

- [ ] **Step 8: Run the whole suite + lint**

Run: `cd /home/razinka/osmopy && python -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -20`
Expected: green except the known pre-existing flake `test_trophic_cascade_visible` (fails on base too). Any OTHER failure — especially in `tests/test_ui_run.py` or `tests/test_java_engine_thread.py` (they import `_python_engine_thread` / run.py) — must be understood: the new trailing `log_q=None` param is back-compatible, so existing positional callers are unaffected; if something breaks, STOP and report rather than paper over.

Run: `cd /home/razinka/osmopy && ruff check osmose/ ui/ tests/ && ruff format --check osmose/ ui/ tests/`
Expected: clean. If `ruff format --check` reports diffs, run `ruff format osmose/ ui/ tests/` and re-check.

- [ ] **Step 9: Commit**

```bash
cd /home/razinka/osmopy && git add ui/pages/run.py tests/test_ui_python_warnings_console.py
git commit -m "feat: stream Python-engine WARNING+ logs into the UI run console

_QueueLogHandler (thread-filtered) attached to the osmose logger during a Python
run bridges the #120/#123 warnings into _run_log_q -> the live console, instead of
only stderr. reset_run_warnings() at launch makes each run re-emit. Mirrors the
Java jar-console stream; the handler is scoped to the run and its thread (no
cross-session leakage).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Test environment:** `ui.pages.run` imports `shiny_deckgl`, which is present in the project venv (existing tests like `tests/test_java_engine_thread.py` / `tests/test_ui_run.py` import `ui.pages.run` fine) but NOT in an ad-hoc `python3`. Run the tests through the project's pytest, not a bare interpreter.
- **The integration test does not depend on the run succeeding.** The #120/#123 warnings fire in `_prepare_run` before the simulation loop, so they are on `log_q` even if `engine.run` later raises on an env artifact. Assert on the queue, not on `done_q`'s outcome.
- **`reset_run_warnings` import placement:** put `from osmose.engine import reset_run_warnings` wherever `run.py` imports its other engine symbols; if `_python_engine_thread` reaches `PythonEngine` via a top-of-file import, match that. Do not introduce a circular import (osmose.engine does not import ui).
- The `except` bodies in `_python_engine_thread` are copied verbatim from the current code — do not alter the cancel/failed behaviour.
