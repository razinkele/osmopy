# Python Run-Progress Feedback — Design

**Date:** 2026-06-21
**Status:** Approved (brainstorming complete)

## Context

Running a simulation on the **Python engine** gives almost no visible feedback (diagnosed via systematic debugging — not a regression; the e2e and a plain-run probe confirm the handlers work and prod is healthy):

- **Console stays blank.** `PythonEngine.run()` has no log/progress callback — only the *Java* path streams to the console via `on_progress` (run.py). The large Console card shows *"No output yet. Click 'Start Run' to begin."* for the entire Python run.
- **Live map is opt-in, off by default.** The "Stream movement during run" switch (`live_movement_view`, default `value=False`) must be flipped *before* running; otherwise no animation.
- **Status updates but is minimal.** `run_status` does go Idle → "Running (Python engine)..." → "Complete" (probe-verified), but it's one small line with no progress indicator, visually drowned by the blank console + blank map.
- **Dead controls.** `py_threads` and `py_verbosity` are rendered on the Python panel but never read in `handle_run` — the "Verbosity" dropdown is a trap given the "no output" complaint.

This was a deliberately deferred gap ("Python progress streaming to the console" was out-of-scope in the engine-capability work). This spec closes it.

## Goal

Make a Python run **visibly report progress**: a live progress bar + an in-place console line, the live map auto-enabled for spatial configs, and the dead inputs resolved — so a user immediately sees the run is happening and how far along it is.

## Non-goals

- No Java-path changes (it already streams to the console).
- No engine-internal/simulation changes beyond reusing the existing `step_observer` hook (no new engine parameter).
- No per-step biomass/ecology stats in the console (progress only).
- Not changing the live-map rendering itself (only when it auto-enables).

## Key facts (verified)

- `simulate()` calls `step_observer(step, state, grid, config)` **every step** (simulate.py:1627) inside `for step in range(config.n_steps)`; the observer already receives `config.n_steps`, so **progress is derivable from the existing hook** — no new engine parameter.
- `make_step_observer` (osmose/live_movement.py:109) builds a `MovementSnapshot` and is self-throttling + never raises into the engine.
- Numba parallelism is `@njit(parallel=True)` + `prange` (mortality.py:1433); thread count is process-wide via `numba.set_num_threads(n)` (verified available; capped at `numba.config.NUMBA_NUM_THREADS`). The `py_threads` input defaults to **1** — wiring it naively would force single-threaded and **slow the default run** (Numba currently uses all cores).
- Spatial predicate: `GridSpec.from_config(cfg)` (osmose/maps/builder.py) needs `grid.nlon/nlat/upleft.lat/upleft.lon/lowright.lat/lowright.lon`. Baltic has all six (regular grid → spatial); eec_full has only `grid.netcdf.file` (NcGrid → not spatial). So `from_config` succeeding is a correct spatial predicate.

## Components

### 1. `osmose/live_movement.py` — `make_run_observer` (pure, unit-testable)

A composing step-observer factory that fans out the engine's single `step_observer` call into **always-on progress** plus an **optional live snapshot**:

```python
def make_run_observer(
    progress_q: "queue.Queue[tuple[int, int, float]]",
    live_observer: "Callable[[int, object, object, object], None] | None" = None,
    *,
    now: Callable[[], float] = time.monotonic,
) -> Callable[[int, object, object, object], None]:
    """Return a step-observer that (a) pushes (step+1, n_steps, elapsed_s) to
    progress_q every step (drop-oldest), and (b) delegates to live_observer when
    given. Never raises into the engine."""
```

- Progress push every step: `(done, int(config.n_steps), now() - start)` where **`done = step + 1`** is the **1-based count of completed steps** (ranges 1..n_steps), and `start` is captured at first call. Drop-oldest on a maxsize-1 queue. (The tuple element is the 1-based `done`, NOT the 0-based loop index — the render formulas below depend on this convention.)
- If `live_observer` is not None, call it after the progress push (it self-throttles).
- Wrapped in try/except so it never raises into the running simulation (preserves the existing guarantee).

### 2. `osmose/live_movement.py` (or a small helper) — `config_is_spatial`

```python
def config_is_spatial(config: dict[str, str]) -> bool:
    """True when the config has a regular grid that yields live-movement frames
    (GridSpec.from_config succeeds). NcGrid configs (eec_full) return False."""
```

Implemented by attempting `GridSpec.from_config(config)` and returning False on `KeyError`/`ValueError`. Pure, unit-testable.

### 3. `ui/pages/run.py` — progress plumbing + auto live map + thread wiring

**Progress state + poll (Python runs):**
- New `_progress_q: queue.Queue` (maxsize 1, drop-oldest) and `_progress: reactive.Value` holding `None | (done, n_steps, elapsed_s)` where `done` is the 1-based completed-step count. (The existing `_live_queue` is maxsize 4 drained-to-latest; progress uses maxsize-1 drop-oldest — both coalesce to the latest tuple, fine.)
- New `_drain_progress` `@reactive.poll(lambda: time.time(), interval_secs=0.2)` + `_consume_progress` effect (mirrors `_drain_live_queue`/`_consume_live_poll`), setting `_progress` from the latest queued tuple.
- **Reset `_progress` to `None` at the very TOP of `handle_run`** (before ALL early returns — Java-block, validation-error, etc.), symmetric with how `run_log` is managed, so a blocked/early-return run never leaves a stale progress bar. The `_drain_run_done` terminal handler also clears `_progress` to `None` on done/failed/cancelled.
- **Python-only invariant:** the progress queue/observer is fed only by the Python engine. The Java path (`_run_java_engine`) streams to `run_log` via `on_progress` and never touches `_progress`, so `run_progress` renders nothing during a Java run. `run_progress` renders only when `_progress` is set (i.e. a Python run is active).

**Progress bar render:**
- `run_progress` `@render.ui`: when `_progress` is set, render a Bootstrap progress bar (`ui.div(ui.div(class_="progress-bar", style=f"width:{pct}%"), class_="progress")`) with a label; when `None`, render nothing. Placed in `run_ui` in the always-visible Run Status block (NOT inside the python `panel_conditional`).
- **Year derivation (defensive, using the 1-based `done`):** the progress tuple carries `(done, n_steps, elapsed)` with `done` = completed steps (1..n). The render reads steps-per-year from `state.config` using the **lowercase canonical key** `simulation.time.ndtperyear` (the reader lowercases keys — the raw file's `ndtPerYear` is stored lowercase in `state.config`; mirror grid.py:392 / map_builder.py:266: `int(float(cfg.get("simulation.time.ndtperyear", "0") or "0"))`). Let `pct = round(done / n * 100)`. If `ndt > 0` → label `f"Year {(done - 1) // ndt + 1}/{ceil(n / ndt)} · step {done}/{n} · {pct}%"` (note `(done - 1) // ndt` converts the 1-based `done` back to a 0-based index for the year bucket — so a 1-year run's final tick `done == ndt` gives `(ndt-1)//ndt + 1 == 1` → "Year 1/1", not "Year 2/1"); if `ndt <= 0` (missing/unparseable) → fall back to a step-only label `f"Step {done}/{n} · {pct}%"` (no division). Never raises.
- **Extract this into a pure helper** `format_progress_label(done: int, n_steps: int, ndt: int) -> str` (in `osmose/live_movement.py`) so the off-by-one year math is unit-testable independent of Shiny; the `run_progress` render and the console line both call it.

**Console in-place line:**
- `run_console` (a `@render.ui` currently reading `run_log.get()`) is recomputed each tick from `run_log.get()` AND `_progress.get()` as: the existing `run_log` lines (warnings/errors), then — when `_progress` is set — **one** trailing progress line. Because the render is recomputed (not appended to `run_log`), the progress line overwrites in place every tick (no flood, bounded history). When idle and `run_log` is empty, the existing placeholder still shows. To avoid literal duplication with the progress-bar label two inches away, keep the console line **terse** — e.g. `running · step {done}/{n} ({pct}%)` (same 1-based `done`) — leaving the Year/elapsed detail to the bar label.

**handle_run (Python branch) changes:**
- Always create the progress queue + `make_run_observer(progress_q, live_observer)`; pass the composed observer to the engine thread (replacing the bare `live_observer`).
- `live_observer` is built only when streaming is active (see auto-enable below).
- Read `input.py_threads()` before dispatch and **always** call `numba.set_num_threads` (so the setting is idempotent across successive runs in one session — `set_num_threads` is process-wide and sticky): `n = int(input.py_threads() or 0)`; if `n >= 1` → `numba.set_num_threads(min(n, numba.config.NUMBA_NUM_THREADS))`; if `n < 1` (Auto, including a negative typed value) → `numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)` (restore all cores). Wrap in try/except so an unexpected value can't abort the run. NOTE (process-wide side effect): this is a global numba setting; acceptable here, but the plan should note it could affect a concurrent in-process calibration run.

**Auto-enable live map (spatial configs):**
- The `live_movement_view` switch **default flips to on when a spatial config is loaded**, off otherwise. Implementation: an effect on `state.config` that sets the switch via `ui.update_switch("live_movement_view", value=config_is_spatial(config))`. The toggle remains a user override; `handle_run`'s existing gate (`if input.live_movement_view() and engine_mode == "python"`) is unchanged.
- **Changed-only guard** (mirror the existing `_last_live_species` pattern, run.py:423): the effect fires the `update_switch` only when the config's spatial-ness *changes* (track the last applied value in a plain mutable cell), so it auto-sets on config-LOAD only and does not re-stomp a user's manual toggle on every unrelated `state.config` write.

**Dead inputs:**
- `py_threads`: **lower the input's `min` from 1 to 0** and set its default `value` to **0 = Auto (all cores)** (current widget is `value=1, min=1, max=32` at run.py:210 — change to `value=0, min=0`), labeled "Threads (Numba; 0 = auto/all cores)"; wired to `numba.set_num_threads` per the handle_run rule above.
- `py_verbosity`: **remove** the `ui.input_select("py_verbosity", …)` widget entirely (it is never read).

### Data flow

```
engine step loop ──step_observer(step,state,grid,config)──> make_run_observer
   ├─ progress_q ──_drain_progress (0.2s poll)──> _progress ──> run_progress bar + console line
   └─ live_observer (only if streaming) ──_live_queue──> _drain_live_queue ──> live map   [unchanged]
```

## Error handling

- `make_run_observer` wraps its body in try/except and never raises into the engine (existing live-observer guarantee preserved). Progress is best-effort (drop-oldest queue).
- `numba.set_num_threads` is clamped to a valid range; an invalid/edge value falls back to the current default (no crash).
- `config_is_spatial` returns False on any `KeyError`/`ValueError` from `GridSpec.from_config` (never raises).
- No new failure paths in the run lifecycle; `_drain_run_done` clears `_progress` on done/failed/cancelled.

## Testing

**Pure (`tests/test_live_movement.py` or a new `tests/test_run_observer.py`):**
- `make_run_observer`: a fake observer call sequence pushes `(step+1, n_steps, elapsed)` to the queue each step; elapsed is monotonic-nondecreasing; delegates to a provided `live_observer` (spy called with the same args); with `live_observer=None` only progress is pushed; an exception in the live observer is swallowed (never propagates).
- **Progress independent of live frames:** with `live_observer=None` (non-spatial config), `make_run_observer` still pushes a progress tuple every step — progress works even when the live map produces no frames.
- `config_is_spatial`: Baltic config (regular-grid keys) → True; eec_full-like config (only `grid.netcdf.file`) → False; empty/partial grid → False.
- `format_progress_label`: **off-by-one regression lock** — `format_progress_label(done=ndt, n_steps=ndt, ndt=ndt)` (1-year run, final tick) → contains `"Year 1/1"` (NOT "Year 2/1"); a mid-year tick → correct year/step/pct; `ndt=0` (or missing) → step-only label `"Step {done}/{n}"` with no "Year" and no ZeroDivisionError.

**Page/render (`tests/test_ui_run*.py`):**
- `import app` clean after removing `py_verbosity` and adding `run_progress`.
- The source exposes `run_progress` and no longer references `py_verbosity`; `py_threads` is wired (handle_run reads it + calls `numba.set_num_threads`).
- **Edit `tests/test_ui_run_capability.py`:** the existing `test_run_page_uses_panel_conditional_for_engine_settings` asserts a tuple of per-engine input ids that includes `"py_verbosity"` (lines ~22-23). Remove `"py_verbosity"` from that tuple (the widget is deleted); keep `"py_threads"`.

**e2e (`tests/test_e2e_live_movement.py`):**
- Add/adjust a case for a **plain Python run without manually toggling** the live switch: load Baltic → switch to Python → set `nyear` → Start Run → assert `#run_progress` appears and the Console shows a `step N/` progress line during the run, and the live map auto-streams (toggle auto-on for Baltic).
- **Required fix to THREE existing Baltic+Python cases that manually click the switch ON** — with auto-enable, Baltic is spatial so the switch is ALREADY on, and the click turns it OFF, breaking the `#live_movement_status` "running"/"done" assertions. Remove the manual click (or assert-it's-on instead) in all three:
  - `tests/test_e2e_live_movement.py:58` (`test_live_movement_renders_during_python_run`)
  - `tests/test_e2e_live_movement.py:96` (`test_live_movement_cancel_path`)
  - `tests/test_e2e_baltic.py:57` (Baltic+Python run asserting live_movement_status running/done at :62-63)
  (`grep -rn "live_movement_view" tests/` confirms these are the only manual-click sites; `tests/test_ui_run.py:109` merely asserts the id exists in source — the switch is kept, so no change there. The e2e does NOT reference `py_verbosity` — the earlier "update e2e if it relied on py_verbosity" was incorrect.)

## Files

- **Modify:** `osmose/live_movement.py` (add `make_run_observer`, `config_is_spatial`, `format_progress_label`).
- **Modify:** `ui/pages/run.py` (progress queue/poll/`_progress`, `run_progress` render, console progress line, compose run observer, wire `py_threads` → `numba.set_num_threads`, remove `py_verbosity`, auto-enable switch effect).
- **Modify/Add tests:** `tests/test_run_observer.py` (or extend `tests/test_live_movement.py`), `tests/test_ui_run_capability.py` (drop `py_verbosity` from the input-id assertion), `tests/test_e2e_live_movement.py` (remove the manual `#live_movement_view` clicks at lines 58/96; add the plain-run progress assertion), `tests/test_e2e_baltic.py` (remove the manual `#live_movement_view` click at line 57).

## Reused infrastructure

The existing `step_observer` engine hook, the `_live_queue`/`_drain_live_queue` reactive-poll pattern, `make_step_observer`, `GridSpec.from_config`, and `osmose.logging`.
