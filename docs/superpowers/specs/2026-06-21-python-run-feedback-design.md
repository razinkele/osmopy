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

- Progress push every step: `(step + 1, int(config.n_steps), now() - start)` where `start` is captured at first call. Drop-oldest on a maxsize-1 queue (same pattern as the live queue).
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
- New `_progress_q: queue.Queue` (maxsize 1) and `_progress: reactive.Value` holding `None | (current_step, n_steps, elapsed_s)`.
- New `_drain_progress` `@reactive.poll(lambda: time.time(), interval_secs=0.2)` + `_consume_progress` effect (mirrors `_drain_live_queue`/`_consume_live_poll`), setting `_progress` from the latest queued tuple.
- Reset `_progress` to `None` at run start and on completion (the `_drain_run_done` terminal handler clears it).

**Progress bar render:**
- `run_progress` `@render.ui`: when `_progress` is set, render a Bootstrap progress bar with a label `f"Year {y}/{ny} · step {s}/{n} · {pct}%"` (year derived from step and `ndtPerYear`); when `None`, render nothing. Placed in `run_ui` near the Run Status block.

**Console in-place line:**
- `run_console` is recomputed each tick as: the existing `run_log` lines (warnings/errors), then — when `_progress` is set — **one** trailing progress line derived from `_progress`. Because the render is recomputed (not appended to `run_log`), the progress line overwrites in place every tick (no flood, bounded history). When idle and `run_log` is empty, the existing placeholder still shows.

**handle_run (Python branch) changes:**
- Always create the progress queue + `make_run_observer(progress_q, live_observer)`; pass the composed observer to the engine thread (replacing the bare `live_observer`).
- `live_observer` is built only when streaming is active (see auto-enable below).
- Read `input.py_threads()` before dispatch: when `n >= 1`, call `numba.set_num_threads(min(n, numba.config.NUMBA_NUM_THREADS))`; when `n == 0` (Auto), do **not** call it (Numba keeps its all-cores default — avoids the default-1 slowdown). Wrapped so an invalid value falls back to Auto.

**Auto-enable live map (spatial configs):**
- The `live_movement_view` switch **default flips to on when a spatial config is loaded**, off otherwise. Implementation: an effect that, when `state.config` changes, sets the switch via `ui.update_switch("live_movement_view", value=config_is_spatial(config))`. The toggle remains a user override; `handle_run`'s existing gate (`if input.live_movement_view() and engine_mode == "python"`) is unchanged.

**Dead inputs:**
- `py_threads`: redefault to **0 = Auto (all cores)** (min stays 0), labeled "Threads (Numba; 0 = auto/all cores)"; wired to `numba.set_num_threads` per the handle_run rule above.
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
- `config_is_spatial`: Baltic config (regular-grid keys) → True; eec_full-like config (only `grid.netcdf.file`) → False; empty/partial grid → False.

**Page/render (`tests/test_ui_run*.py`):**
- `import app` clean after removing `py_verbosity` and adding `run_progress`.
- The source exposes `run_progress` and no longer references `py_verbosity`; `py_threads` is wired (handle_run reads it).

**e2e (`tests/test_e2e_live_movement.py` or a new case):**
- A **plain Python run without manually toggling** the live switch: load Baltic → switch to Python → set `nyear` → Start Run → assert `#run_progress` appears and the Console shows a `step N/` progress line during the run, and the live map auto-streams (toggle auto-on for Baltic). Update the existing e2e if it relied on `py_verbosity`.

## Files

- **Modify:** `osmose/live_movement.py` (add `make_run_observer`, `config_is_spatial`).
- **Modify:** `ui/pages/run.py` (progress queue/poll/`_progress`, `run_progress` render, console progress line, compose run observer, wire `py_threads` → `numba.set_num_threads`, remove `py_verbosity`, auto-enable switch effect).
- **Modify/Add tests:** `tests/test_run_observer.py` (or extend `tests/test_live_movement.py`), `tests/test_ui_run*.py`, `tests/test_e2e_live_movement.py`.

## Reused infrastructure

The existing `step_observer` engine hook, the `_live_queue`/`_drain_live_queue` reactive-poll pattern, `make_step_observer`, `GridSpec.from_config`, and `osmose.logging`.
