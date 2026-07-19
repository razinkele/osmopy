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

## Approach — chosen: A (log-bridge handler scoped to the run, filtered to the run's thread)

Rejected: **B** a permanent global handler installed at startup (captures ALL osmose logging —
calibration, other pages, background tasks — into the run console: noise + cross-page leakage);
**C** collecting logs into `RunResult` for post-run display (post-run not live; couples the engine's
return contract to UI display). A keeps isolation tight and reuses the proven Java-streaming path.

**Isolation is by thread, not just by time (round-1 review fix).** Time-boxing alone is insufficient:
the handler lives on the *process-global* `osmose` logger, so during its window a **concurrent Python
run in a different Shiny session** (each session has its own `_run_log_q` but they share the one
global `osmose` logger) would leak its warnings into this session's console — verified reproducible,
and the exact leakage B was rejected for. The handler therefore filters by the engine thread's ident:
each `_python_engine_thread` runs on its own `threading.Thread`, the #120/#123 warnings fire on that
thread (both fire in `_prepare_run` — #123 via `warn_unread_java_only_keys` at `__init__.py:78`, #120
via `EngineConfig.from_dict` called at `__init__.py:81` → `config.py:2040-2052`), and the handler posts only records whose
`record.thread` equals the ident captured when it was created in that thread. A concurrent session's
warnings (different thread) are dropped — genuine per-run isolation, not merely time-boxing.

## Architecture — three small units

### 1. `_QueueLogHandler(logging.Handler)` — the log→queue bridge (`ui/pages/run.py`)

Beside the other run helpers. Formats each record and non-blockingly posts it to a `queue.Queue`;
level `WARNING`; never raises out of `emit` (a log line must not break a run).

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

`thread_id` is captured as `threading.get_ident()` inside `_python_engine_thread` (so it is the
engine thread's ident); `record.thread` is the ident of the thread that emitted the record, which for
#120/#123 is that same engine thread (they fire in `_prepare_run`). Format is `LEVEL: message` (e.g.
`WARNING: 18 config key(s) are valid OSMOSE keys …`) — the console panel already frames it; no
timestamp/logger-name noise needed. (The stderr handler keeps the full `asctime [name] LEVEL` format
for the journal — unchanged.)

### 2. Wire it into `_python_engine_thread` (`ui/pages/run.py`)

- Add a `log_q` parameter (mirroring `_java_engine_thread`'s signature).
- Inside, attach a thread-filtered `_QueueLogHandler(log_q, threading.get_ident())` to the **`osmose`**
  logger as the **first statement of the `try`** (so the `finally` always covers it — no leak even if
  a later line raises), and detach in the `finally`. One handler on the `osmose` parent catches both
  warnings via propagation: #123 logs to `osmose.config`, #120 to `osmose.engine.config`, both
  children of `osmose` with default `propagate=True` (round-1-verified).
- **Do NOT add `run_log.set([])` at the launch site.** Unlike the Java branch (which clears at
  `run.py:978`), the Python path already sets a **fresh** console just before dispatch at
  `run.py:889-894` — either the pre-run validation-warnings block (`--- WARNINGS (continuing anyway)
  ---`) or `[]`. Clearing again at launch would **wipe that validation block**. The streamed #120/#123
  warnings instead APPEND to it (via `_drain_run_log`, which does `run_log.get() + new_lines`), so the
  console shows the pre-run validation block followed by the live run warnings — both preserved. At
  the launch site (`run.py:944`) only add `log_q=_run_log_q` to the thread args.

```python
def _python_engine_thread(run_config, output_dir, cancel_token, step_observer, done_q,
                          n_threads=0, log_q=None):
    apply_single_run_threads(n_threads)
    osmose_logger = logging.getLogger("osmose")
    handler = None
    try:
        if log_q is not None:
            handler = _QueueLogHandler(log_q, threading.get_ident())
            osmose_logger.addHandler(handler)  # first, so `finally` always covers it
        engine = PythonEngine()
        result = engine.run(run_config, output_dir, seed=0,
                            cancel_token=cancel_token, step_observer=step_observer)
        done_q.put(("done", result, ""))
    except SimulationCancelled as exc:
        ...  # (unchanged) post ("cancelled", None, msg)
    except Exception as exc:  # noqa: BLE001
        ...  # (unchanged) post ("failed", None, msg)
    finally:
        if handler is not None:
            osmose_logger.removeHandler(handler)
```

`log_q=None` keeps the function usable without a UI (defensive; the launch site always passes
`_run_log_q`). Also update `_drain_run_log`'s now-stale docstring (`run.py:586`, "Stream the Java
run's console lines") — it streams Python engine warnings too.

### 3. `reset_run_warnings()` — per-UI-run dedup reset (`osmose/engine/__init__.py`)

The engine's warning dedup sets are process-global, so a long-lived Shiny server would warn only on a
config's FIRST run. There are **three** such sets (round-1-verified — the handler captures ALL osmose
WARNING+, so resetting only two would leave the mortality warning inconsistent: #120/#123 re-emit but
it wouldn't): `config_validation._WARNED_JAVA_ONLY_KEYS` (#123), `config._WARNED_UNSUPPORTED_RESTART`
(#120), and `config._WARNED_UNSUPPORTED_MORTALITY` (parsed-but-unapplied mortality features,
`config.py:1977`). `reset_run_warnings()` clears all three (lazy imports — `__init__.py` is the
package home that already imports both modules in `_prepare_run`, and `config` imports
`config_validation` one-way, so function-local imports avoid any cycle). The UI calls it at run start;
CLI/batch never call it, so their per-process dedup (which prevents spamming a 1000-sim calibration)
is untouched.

```python
def reset_run_warnings() -> None:
    """Clear the engine's per-process warning-dedup caches so the next run re-emits its warnings.
    UI-run-scoped: the interactive UI calls this at each run start; CLI/batch do NOT, keeping their
    once-per-process dedup. Enumerates every engine WARNING-dedup set (add new ones here)."""
    from osmose.engine import config_validation
    from osmose.engine import config as _config
    config_validation._WARNED_JAVA_ONLY_KEYS.clear()
    _config._WARNED_UNSUPPORTED_RESTART.clear()
    _config._WARNED_UNSUPPORTED_MORTALITY.clear()
```

Called on the main thread in `handle_run`'s Python branch, right before launching the engine thread
(after the fresh-console block at `run.py:889-894`), so the cleared state is in place before
`engine.run()` fires the warnings.

## Data flow

UI Run (Python) → `reset_run_warnings()` (main thread, before launch) → console already freshly set
at `run.py:889-894` (validation block or `[]`) → launch `_python_engine_thread(…, log_q=_run_log_q)`
→ thread attaches thread-filtered `_QueueLogHandler` to `osmose` → `engine.run()` → `_prepare_run` →
#123/#120 `log.warning` (on the engine thread) → propagate to `osmose` → `_QueueLogHandler.emit`
(thread matches) → `_run_log_q.put_nowait` → `_drain_run_log` (0.2s poll) APPENDS to `run_log` →
`run_console` panel shows it **live**, below the pre-run validation block → thread `finally` detaches
the handler.

`reset_run_warnings()` is called on the **main thread before** the engine thread starts, so the
cleared state strictly precedes the warning (which fires later, on the engine thread) — no race.

## Error handling

- `_QueueLogHandler.emit` never raises (wraps in try/except → `handleError`).
- The handler is removed in a `finally`, so a crashing run cannot leak a handler onto the global
  `osmose` logger (which would then post stray lines into a later run's console).
- `_run_log_q` is unbounded; `put_nowait` never blocks the engine thread.
- `_drain_run_log` already caps `run_log` at 500 lines (`run.py:597`); no change.
- **Required import:** `ui/pages/run.py` currently imports only `from osmose.logging import
  setup_logging`, NOT stdlib `logging`. Add `import logging` (the `_QueueLogHandler` code uses
  `logging.Handler`/`LogRecord`/`WARNING`/`Formatter`/`getLogger`). `threading` is already imported.
- **Failed-run behaviour (round-2 note, intended):** on a failed Python run the except block logs
  `_log.error("Python engine failed", exc_info=True)` on `osmose.run` (a child of `osmose`) while the
  handler is still attached and on the engine thread — so the console ALSO shows `ERROR: Python
  engine failed: …` plus the traceback, in addition to the `--- ERROR ---` summary line
  `_drain_run_done` appends (`run.py:552`). This is acceptable and useful (the traceback aids
  debugging); it is NOT pure duplication (summary line vs. full traceback). Documented so it is not a
  surprise. The cancel path uses `_log.info` (below WARNING) and is correctly not captured.

## Concurrency note (round-2)

`reset_run_warnings()` clears process-global sets on the serialized main thread, but the
dedup check-then-emit runs on the engine thread. In the rare case of **two concurrent sessions
running configs with an identical warning fingerprint**, if both resets land before either emit, the
first engine thread to reach the warn adds the fingerprint and the second suppresses it — so one
session's console may omit that warning (it is still on stderr once). This is benign (a missing
*informational* warning, never wrong data) and **cannot leak** (the thread filter still prevents one
session's warning from appearing in another's console). Making the dedup thread-local would fix it
but is over-engineering for this rare, harmless case — accepted as a known limitation. The
single-session ordering (reset → launch → warn) has no race.

## Testing

- **`_QueueLogHandler`:** a `WARNING` record emitted on the handler's `thread_id` → the formatted
  line lands on the queue; an `INFO` record does NOT (level filter); a record with a **different
  `thread` ident** does NOT (thread filter — pins the cross-session-leakage fix); an `emit` whose
  `put_nowait` raises does not propagate out of `emit` (a broken queue can't break a run).
- **`reset_run_warnings()`:** after warnings have populated all three sets
  (`_WARNED_JAVA_ONLY_KEYS`, `_WARNED_UNSUPPORTED_RESTART`, `_WARNED_UNSUPPORTED_MORTALITY`), calling
  it empties **all three**, and a second `warn_unread_java_only_keys` / restart-warn re-emits (proves
  per-run re-emission). No cross-module import cycle on import.
- **Integration (`_python_engine_thread` with `log_q`):** run it on the bundled `data/minimal` config
  plus **`simulation.ncpu` + `oxygen.factor`** (confirmed java-only, trigger #123) and
  `simulation.restart.file` (triggers #120); assert the passed `log_q` receives BOTH the #123 summary
  line (`"see issue #123"`) and the #120 restart line (`"see issue #120"`), and that after the thread
  finishes the `osmose` logger has **no** `_QueueLogHandler` attached (detached cleanly). Real config,
  no mocks. (`engine.run` may raise later on env artifacts — irrelevant; the warnings fire in
  `_prepare_run` before the loop, as round-1 e2e confirmed.)
- **No-log_q back-compat:** `_python_engine_thread(..., log_q=None)` runs and attaches no handler
  (existing callers/tests unaffected).
- **Whole-suite guard:** the existing suite stays green (the engine's stderr logging, the CLI dedup,
  and `from_dict`'s silence are all unchanged). Known pre-existing flake `test_trophic_cascade_visible`
  is unrelated. If a `ui/` visual baseline covers the run console, refresh it in CI (advisory gate).

## Success criteria

- On a UI Python-engine run whose config sets java-only keys and/or restart, the #123 summary and
  #120 restart warning appear in the run Console Output panel, live, below the pre-run validation
  block, verified by the integration test (warning on the queue) — not only on stderr.
- Each UI run re-emits its warnings: `reset_run_warnings()` clears all three engine warning-dedup
  sets per run; CLI/batch dedup is unchanged (they never call it).
- The handler is scoped to the run AND to the run's thread — never attached outside a run, always
  detached after, and a **concurrent Python run in another Shiny session cannot leak** into this
  console (thread-ident filter), verified by the handler's thread-filter test.
- Java runs, the CLI path, `from_dict`'s silence, INFO-level engine logs, and the pre-run validation
  block are all unaffected.
