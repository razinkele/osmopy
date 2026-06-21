# Run-Page Rework: Python default, compact UI, live-stream crash fix — Design

**Date:** 2026-06-21
**Status:** Approved (brainstorming complete)

## Context

Three user-reported problems with the Run page, all addressed in one rework:

1. **Java is the UI default engine**, but the Python engine is OSMOPY's primary, validated engine. A fresh user lands on Java with a one-version-stale jar. (Also the "Java is the UI default" nuance flagged by the jar-swap deep review.)
2. **The Run page wastes screen space** — the "Running simulation (Python)…" busy overlay is large, and the Live Movement controls/map occupy the page even when not in use.
3. **The app crashed** during a live run (dots mode, cod filter, step 1037/1200, snapshot truncated 5000-of-14846). **Root cause established by systematic debugging** (direct prod stack trace + server-side elimination):
   - **Client trigger:** sustained live **dots** streaming over a long run exhausts the browser tab's WebGL/memory. Dots is the heaviest renderer (per-point `ScatterplotLayer` geometry, ≤5000 pts/frame, every ~0.2s for minutes) vs. heatmap (one aggregated texture). The server-side snapshot/layer build is **proven robust** (1200-frame stress: 0 exceptions, 4.5 MB peak) — the failure is in the unbounded client stream. The live view has no cap on cumulative streaming load/duration.
   - **Server fragility (the log cascade):** when the tab/session died mid-run, the reactive polls (`_drain_run_done`/`_drain_live_queue`/`_drain_progress`), the async `_render_live_map`, and the fire-and-forget daemon engine thread kept firing against the destroyed session → `DestroyedReactiveError: '_live_status_val' has been destroyed` + `CancelledError` cascade (prod log, 12:50:22). Nothing links session teardown to cancelling the run; `_render_live_map` has no error guard. (`NRestarts=0` — the *server* never died; the *session* did.)

## Goals

A — make **Python the default engine**.
B — **reclaim screen space**: compact busy indicator; Live Movement collapsed by default with its controls inside the collapsible body.
C — **make the live-stream crash impossible**: bound client load, fall back to heatmap when dots would be heavy, and harden the server against session teardown so a dropped session cancels the run instead of cascading errors.

## Non-goals

- No change to the engine/runner execution or the Python engine itself.
- No new live-view features (playback, scrubbing, per-cell detail).
- No change to the Java live path (Java has no live movement).

## Component A — Python default engine

- `ui/state.py:64` `engine_mode` default `reactive.Value("java")` → `reactive.Value("python")`.
- `app.py`: the client engine init reads `localStorage.getItem('osmose-engine') || 'java'` (line ~480) → default `'python'`; the initial active-button state and `setEngineMode` logic already follow the resolved mode, so only the fallback string changes. The Java/Python toggle buttons (`app.py:279-286`) are unchanged — the user can still pick Java.
- Tests: `tests/test_state.py` engine-default assertion updated to `"python"`; any test asserting the `'java'` localStorage fallback updated.

## Component B — Reclaim screen space

### B1. Compact busy indicator
`app.py:632-640` `loading_overlay` renders a full `osm-loading-overlay` (large centered spinner + message). Change it to a **slim, fixed-position corner indicator** (small spinner + message in a compact pill, bottom-right), not a page-dominating block. Markup change in `loading_overlay` + a new/adjusted CSS rule in `www/osmose.css` (`osm-loading-overlay` → compact variant). No behavior change — still driven by `state.busy`.

### B2. Live Movement collapsed by default
The Live Movement card already supports body-collapse (`body_collapse_header("Live Movement (Python engine)", "run_live_movement")`). Make it **collapsed by default**: the `toggleCardBody` restore logic in `app.py` (~line 215) reads localStorage `osm-body-collapsed:<panel_id>`; default the `run_live_movement` panel to collapsed when no stored preference exists (so a first-time user sees it collapsed; their explicit choice persists). The mode/species controls already live in the card body, so collapsing hides them too.

## Component C — Live-stream crash fix (the core)

### C1. The card-expanded state becomes the server-readable stream gate (replaces the switch)
Today streaming is gated by the `live_movement_view` switch (`input.live_movement_view()`), with an auto-enable-for-spatial effect. That model both (a) streams while the card is collapsed/unwatched and (b) duplicates the collapse control. Replace it:

- **Remove** the `ui.input_switch("live_movement_view", …)` widget and the `_auto_enable_live_for_spatial` effect.
- **Add a server-readable "live card expanded" input.** Extend `toggleCardBody` (app.py) so that for the `run_live_movement` card it also calls `Shiny.setInputValue('live_view_expanded', <expanded:bool>)`, and set the initial value on page init (matching the collapsed-by-default state → `false`). (Generic-but-targeted: only this card publishes the input, keyed by `panel_id === 'run_live_movement'`.)
- **Gate streaming on `input.live_view_expanded()`**, evaluated at run start in `handle_run` (the Python branch): build + attach the live snapshot observer **only if the live card is expanded when the run starts** (`make_run_observer(_progress_q, live_observer)` with `live_observer` built only when expanded; otherwise `make_run_observer(_progress_q, None)` — progress still streams). This means: to watch live movement, expand the card *before* starting the run (same ergonomics as the old "flip the switch first", now unified with the collapse control). Expanding mid-run is out of scope (documented).
- `_render_live_map` keeps rendering whatever snapshots arrive; with no live observer attached, none arrive and it idles.

### C2. Bound the client load
- **Lower `dot_cap`** in `build_snapshot` from `5000` to **`2000`** (osmose/live_movement.py:63 default). 2000 points is ample for a visual and ~2.5× less per-frame geometry. Update the docstring + the live-status "showing N of M" text reads the new cap automatically.
- **Throttle the live render harder.** `make_step_observer`'s `throttle_s` is `0.2`; raise the live-snapshot throttle to **`0.5`s** (≤2 fps) — smooth enough for a live view, far less cumulative client churn over a long run. (Progress updates keep their own cadence; this only affects the map.)
- **Auto-fall-back to heatmap when dots would be heavy.** In `_render_live_map`: if `mode == "dots"` but the filtered point count exceeds a threshold (**`1500`**), render the **heatmap** layer instead and surface a one-line note ("Too many schools for dots — showing heatmap"). Dots remains exact for small/filtered sets (the common useful case); the browser-killing large-dots case can't happen. Pure helper `choose_live_layer(snap, species_filter, mode, *, dots_max=1500) -> (layer, note)` in `ui/pages/live_movement_render.py` so the threshold logic is unit-testable.

### C3. Harden the server against session teardown
- **`session.on_ended`** registered in `run_server`: on session end, (a) `set()` the run cancel token if a run is in flight (`state.run_cancel_token`, exists at `ui/state.py:57`) so the daemon engine thread sees `SimulationCancelled` and exits instead of running to completion against a dead session; (b) set a plain mutable flag `_session_alive[0] = False`.
- **Guard the reactive consumers.** Wrap the bodies of `_drain_run_done`, `_drain_live_queue`, `_drain_progress`, and `_render_live_map`: **early-return when `not _session_alive[0]`** (the primary, version-independent guard — these polls fire on a timer independent of the session), AND wrap the reactive access in a broad `try/except Exception` (debug-log only) as defense-in-depth against the check→access teardown race. **Do NOT import `DestroyedReactiveError`** — it is not publicly exported (`shiny.reactive` does not expose it; only the private `shiny.reactive._reactives` path works, which is fragile across versions). These consumers are best-effort UI updaters, so swallowing any post-teardown exception is correct.

## Data flow (after)

```
expand live card ──Shiny.setInputValue('live_view_expanded', true)──┐
                                                                     ▼
handle_run (python): live_observer = make_step_observer(...) ONLY if live_view_expanded
   make_run_observer(progress_q, live_observer)  ── engine thread
        ├─ progress_q  → _drain_progress → bar/console            [always]
        └─ _live_queue → _drain_live_queue → _render_live_map      [only if expanded]
                                              └─ choose_live_layer (dots≤1500 else heatmap)
session.on_ended → cancel token (stop thread) + _session_alive=False (polls no-op)
```

## Error handling

- `_render_live_map` + all three poll consumers: `_session_alive` early-return + `try/except DestroyedReactiveError` → no cascade on teardown.
- `session.on_ended` cancels the run → the daemon engine thread exits promptly (no runaway compute against a dead session).
- `choose_live_layer` never raises (pure); empty filtered set → empty layer (already handled).
- Heatmap fallback is a render-time choice; no new failure path.

## Testing

**Pure (`tests/test_live_movement_render.py` or extend existing):**
- `choose_live_layer`: `mode="dots"` with ≤1500 filtered points → scatterplot layer, no note; >1500 → heatmap layer + note; `mode="heatmap"` always heatmap; empty filtered set → empty layer, no crash.
- `build_snapshot` `dot_cap` default is 2000; a 14846-school state truncates to 2000 with `truncated=True`, `n_total=14846`.

**Page/state (`tests/test_state.py`, `tests/test_ui_run*.py`):**
- engine default is `"python"`.
- `run.py` source: `live_movement_view` switch removed; `live_view_expanded` gate present; `session.on_ended` registered; the three polls + `_render_live_map` reference `_session_alive` (early-return guard).
- `import app` clean.

**e2e (`tests/test_e2e_live_movement.py`, `tests/test_e2e_baltic.py`):**
- Replace the `#live_movement_view` interactions with **expanding the live card** (click its collapse toggle to expand) before Start Run; assert live status goes running→done. The plain-run progress test (no live view) is unaffected (progress is independent of the live gate). Update any assertion referencing the removed switch.

## Files

- **Modify:** `ui/state.py` (engine default), `app.py` (localStorage default, `loading_overlay` compaction, `toggleCardBody` → `live_view_expanded` input + collapsed-by-default for `run_live_movement`), `www/osmose.css` (compact busy indicator), `ui/pages/run.py` (remove switch + auto-enable effect; `live_view_expanded` gate in `handle_run`; `session.on_ended`; `_session_alive` + `DestroyedReactiveError` guards in the 3 polls + `_render_live_map`; use `choose_live_layer`), `osmose/live_movement.py` (`dot_cap` 5000→2000; live throttle 0.2→0.5), `ui/pages/live_movement_render.py` (`choose_live_layer` helper).
- **Tests:** `tests/test_live_movement_render.py` (new/extended), `tests/test_state.py`, `tests/test_ui_run_capability.py`, `tests/test_e2e_live_movement.py`, `tests/test_e2e_baltic.py`.

## Reused infrastructure

`state.busy` (busy indicator), `body_collapse_header`/`toggleCardBody` (collapse), `make_step_observer`/`make_run_observer` (the live/progress observers), `state.run_cancel_token` (`ui/state.py:57`, for the on-ended cancel), `session.on_ended` (verified available), the `_drain_*` reactive-poll pattern.
