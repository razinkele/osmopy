# Surface Python-engine warnings in the UI Console — design

**Date:** 2026-07-19
**Branch:** `feat/ui-surface-python-warnings`
**Origin:** Follow-up to #120 (PR #125) / #123 (PR #126). Those features emit `log.warning` on a
Python-engine run, but the warnings go only to the server's **stderr** (systemd journal) — the UI
Console Output panel never shows them, so a UI user running a Python simulation never sees them.
CLI/script/notebook users DO see them (stderr is their console); this fixes the UI gap.

## Problem (verified by execution)

- The Python engine runs **in-process** on a background thread (`ui/pages/run.py::_python_engine_thread`,
  launched at `run.py:944`). That thread posts only the final outcome to `_run_done_q` — it has **no
  `log_q`**, unlike `_java_engine_thread` (`run.py:395`), which streams the jar console to `_run_log_q`
  → `_drain_run_log` (a 0.2s reactive poll, `run.py:585`) → the `run_console` panel.
- `osmose` loggers write to `sys.stderr` via a `StreamHandler` (`osmose/logging.py:25`). No handler
  routes them to the UI queue, and `_drain_run_done` (`run.py:531`) only appends cancel/error/result
  lines — never the warnings.
- So on a UI **Python** run, the #120 (`osmose.engine.config`) and #123 (`osmose.config`) warnings
  fire correctly but are invisible to the user.

## Goal

On a UI Python-engine run, stream the engine's `WARNING+` `osmose` logs (chiefly the #120/#123
warnings) into the run Console Output panel — live, using the existing `_run_log_q` / `_drain_run_log`
machinery — and make the warnings re-emit on each UI run (not once per server process).

## Approach — chosen: A (log-bridge handler scoped to the run)

Rejected: **B** a permanent global handler installed at startup (captures ALL osmose logging —
calibration, other pages, background tasks — into the run console: noise + cross-page leakage);
**C** collecting logs into `RunResult` for post-run display (post-run not live; couples the engine's
return contract to UI display). A keeps isolation tight — the handler exists only for the run's
duration — and reuses the proven Java-streaming path.

## Architecture — three small units

### 1. `_QueueLogHandler(logging.Handler)` — the log→queue bridge (`ui/pages/run.py`)

Beside the other run helpers. Formats each record and non-blockingly posts it to a `queue.Queue`;
level `WARNING`; never raises out of `emit` (a log line must not break a run).

```python
class _QueueLogHandler(logging.Handler):
    """Bridge osmose WARNING+ logs into the run console queue (live, like the Java jar console)."""
    def __init__(self, log_q: "queue.Queue", level: int = logging.WARNING) -> None:
        super().__init__(level)
        self._log_q = log_q
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._log_q.put_nowait(self.format(record))
        except Exception:  # noqa: BLE001 — a log line must never break a run
            self.handleError(record)
```

Format is `LEVEL: message` (e.g. `WARNING: 18 config key(s) are valid OSMOSE keys …`) — the console
panel already frames it; no timestamp/logger-name noise needed. (The stderr handler keeps the full
`asctime [name] LEVEL` format for the journal — unchanged.)

### 2. Wire it into `_python_engine_thread` (`ui/pages/run.py`)

- Add a `log_q` parameter (mirroring `_java_engine_thread`'s signature).
- Inside, attach a `_QueueLogHandler(log_q)` to the **`osmose`** logger before `engine.run()`, and
  detach it in a `finally` (so no handler leaks across runs). One handler on the `osmose` parent
  catches both warnings via propagation: #123 logs to `osmose.config`, #120 to `osmose.engine.config`,
  both children of `osmose` with default `propagate=True`.
- At the launch site (`run.py:944`), pass `_run_log_q` in the thread args and clear the console
  (`run_log.set([])`) exactly as the Java branch does at `run.py:978`.

```python
def _python_engine_thread(run_config, output_dir, cancel_token, step_observer, done_q,
                          n_threads=0, log_q=None):
    apply_single_run_threads(n_threads)
    osmose_logger = logging.getLogger("osmose")
    handler = _QueueLogHandler(log_q) if log_q is not None else None
    if handler is not None:
        osmose_logger.addHandler(handler)
    engine = PythonEngine()
    try:
        result = engine.run(run_config, output_dir, seed=0,
                            cancel_token=cancel_token, step_observer=step_observer)
        done_q.put(("done", result, ""))
    except SimulationCancelled as exc:
        ...
    except Exception as exc:  # noqa: BLE001
        ...
    finally:
        if handler is not None:
            osmose_logger.removeHandler(handler)
```

`log_q=None` keeps the function usable without a UI (defensive; the launch site always passes
`_run_log_q`).

### 3. `reset_run_warnings()` — per-UI-run dedup reset (`osmose/engine/__init__.py`)

The two warning dedup sets are process-global (`config_validation._WARNED_JAVA_ONLY_KEYS`,
`config._WARNED_UNSUPPORTED_RESTART`) so a long-lived Shiny server would warn only on a config's
FIRST run. `reset_run_warnings()` clears both (lazy imports — `__init__.py` is the package home that
already imports both in `_prepare_run`, and `config` imports `config_validation` one-way, so
function-local imports avoid any cycle). The UI calls it at run start; CLI/batch never call it, so
their per-process dedup (which prevents spamming a 1000-sim calibration) is untouched.

```python
def reset_run_warnings() -> None:
    """Clear the engine's per-process warning-dedup caches so the next run re-emits its warnings.
    UI-run-scoped: the interactive UI calls this at each run start; CLI/batch do NOT, keeping their
    once-per-process dedup."""
    from osmose.engine import config_validation
    from osmose.engine import config as _config
    config_validation._WARNED_JAVA_ONLY_KEYS.clear()
    _config._WARNED_UNSUPPORTED_RESTART.clear()
```

Called on the main thread in `handle_run`'s Python branch, right before launching the thread (after
`run_log.set([])`), so the cleared state is in place before `engine.run()` fires the warnings.

## Data flow

UI Run (Python) → `reset_run_warnings()` + `run_log.set([])` + launch `_python_engine_thread(…,
log_q=_run_log_q)` → handler attached to `osmose` → `engine.run()` → `_prepare_run` → #123/#120
`log.warning` → propagate to `osmose` → `_QueueLogHandler.emit` → `_run_log_q.put_nowait` →
`_drain_run_log` (0.2s poll) → `run_console` panel shows it **live** → thread `finally` detaches the
handler.

## Error handling

- `_QueueLogHandler.emit` never raises (wraps in try/except → `handleError`).
- The handler is removed in a `finally`, so a crashing run cannot leak a handler onto the global
  `osmose` logger (which would then post stray lines into a later run's console).
- `_run_log_q` is unbounded; `put_nowait` never blocks the engine thread.
- `_drain_run_log` already caps `run_log` at 500 lines (`run.py:597`); no change.

## Testing

- **`_QueueLogHandler`:** an `osmose.config` `WARNING` record → the formatted line lands on the
  queue; an `INFO` record does NOT (level filter); an `emit` whose `put_nowait` raises does not
  propagate out of `emit` (a broken queue can't break a run).
- **`reset_run_warnings()`:** after a warning has populated `_WARNED_JAVA_ONLY_KEYS` /
  `_WARNED_UNSUPPORTED_RESTART`, calling it empties both, and a second `warn_unread_java_only_keys` /
  restart-warn re-emits (proves per-run re-emission). No cross-module import cycle on import.
- **Integration (`_python_engine_thread` with `log_q`):** run it on the bundled minimal config plus a
  java-only key and a `simulation.restart.file`; assert the passed `log_q` receives BOTH the #123
  summary line and the #120 restart line, and that after the thread finishes the `osmose` logger has
  **no** `_QueueLogHandler` attached (detached cleanly). Use the real `data/minimal` config; no mocks.
- **No-log_q back-compat:** `_python_engine_thread(..., log_q=None)` runs and attaches no handler
  (existing callers/tests unaffected).
- **Whole-suite guard:** the existing suite stays green (the engine's stderr logging, the CLI dedup,
  and `from_dict`'s silence are all unchanged). Known pre-existing flake `test_trophic_cascade_visible`
  is unrelated. If a `ui/` visual baseline covers the run console, refresh it in CI (advisory gate).

## Success criteria

- On a UI Python-engine run whose config sets java-only keys and/or restart, the #123 summary and
  #120 restart warning appear in the run Console Output panel, live, verified by the integration test
  (warning on the queue) — not only on stderr.
- Each UI run re-emits its warnings (dedup reset per run); CLI/batch dedup is unchanged.
- The handler is scoped to the run — never attached outside a run, always detached after, no
  cross-page or cross-run leakage.
- Java runs, the CLI path, `from_dict`'s silence, and INFO-level engine logs are all unaffected.
