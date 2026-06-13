# Live-During-Run Movement Visualization — Design

**Date:** 2026-06-13
**Status:** Approved (brainstorming) — in-loop review round 1 incorporated
**Backlog item:** UI / Shiny → "Live-during-run movement visualization"

## Goal

Stream living schools' spatial positions onto a deck.gl map on the Run page **while a
Python-engine simulation runs**, so the user watches the ecosystem move in real time —
either as a biomass-weighted density **heatmap** or as individual **dots**, toggled.

Distinct from the existing Grid-page "Movement Animation" overlay, which only animates
the *configured input* movement distribution maps (`movement.file.map{N}`,
`ui/pages/grid.py`), not live engine state.

## Engine scope (and why)

**Python engine only.** It runs **in-process** in an executor thread
(`ui/pages/run.py:257-272` — `run_in_executor(None, lambda: engine.run(..., cancel_token=...))`),
so an in-process step callback can stream positions. The **Java engine is a prebuilt
JAR** (`osmose-java/osmose_4.3.3-jar-with-dependencies.jar`, no source in the repo) run
as a **subprocess** (`osmose/runner.py`) emitting only stdout progress text, and "the
Java engine stays unchanged" is a documented tenet
(`docs/plans/2026-02-21-osmose-python-port-design.md`). Live Java streaming would require
forking/rebuilding Java OSMOSE — out of scope. Java runs / toggle-off show a "live view
available for the Python engine" note. The rendering layer stays engine-neutral.

## Non-goals (YAGNI)

- No Java live streaming.
- No movement *trails* in v1 (`trips_layer` + the built-in `trips_animation_ui/server`
  make this a natural follow-up; not in scope).
- No persistence/replay of the live stream to disk (post-run spatial output already has
  the Spatial Results page).
- No sub-cell physics — schools carry only integer `cell_x/cell_y`; dots render at cell
  centers with deterministic jitter.

## Coordinate reality (important)

`Grid.lat`/`Grid.lon` are `(ny,)`/`(nx,)` and **optional** (`osmose/engine/grid.py:32-33`).
They are populated **only** for NetCDF-backed grids (`Grid.from_netcdf`,
`grid.py:106-116`); rectangular `Grid.from_dimensions` grids leave them `None`
(`grid.py:119-122`). The post-run spatial-NetCDF writer, when they are `None`, falls back
to **`np.arange(ny)` / `np.arange(nx)` integer cell indices** (`osmose/engine/output.py:723-744`)
— there is **no** config-bounds (`grid.upleft/lowright`) derivation anywhere in the engine.

Therefore `resolve_grid_latlon(grid)` returns `grid.lat`/`grid.lon` when present, else
`np.arange(ny, dtype=np.float64)` / `np.arange(nx, dtype=np.float64)` — **byte-identical to
the writer's fallback** (`output.py:726,731` use the same `dtype=np.float64`, so live and
post-run coordinates agree). Consequence, stated plainly: rectangular grids render at
index coordinates `0..n` (not geographic), exactly like the existing post-run spatial
path; true geographic rendering requires a NetCDF-backed grid — the **same limitation the
Map View / Spatial Results already have**. (All shipped real configs — minimal, Baltic,
EEC — set `grid.netcdf.file`, so `grid.lat/lon` are real for them.)

## Architecture & module boundaries

Pure logic in `osmose/`, rendering/reactives in `ui/`. Transport is an **in-memory
bounded queue + `reactive.poll`** — the pattern proven by the calibration dashboard
(`ui/pages/calibration_handlers.py:996-1047` polls a thread-safe queue and `.set()`s
reactive values) and by the user's CENOP/`~/cenjas` project.

1. **Engine hook (core, UI-independent)** — `osmose/engine/simulate.py` (`simulate`,
   `:1220`) + `osmose/engine/__init__.py` (`PythonEngine.run`, `:87`): add an optional
   **keyword-only** `step_observer` callable, invoked **immediately after**
   `accumulated.append(step_out)` (`simulate.py:1583`) as
   `if step_observer is not None: step_observer(step, state, grid, config)`. Default
   `None` → loop byte-for-byte unchanged, **zero** overhead when off, and parity-safe by
   construction (existing callers pass args positionally — `test_engine_parity.py:87`).
   It fires only for loop steps `range(config.n_steps)` (`simulate.py:1343`), never for
   the optional pre-loop `step=-1` init snapshot (`simulate.py:1340-1341`).
   `PythonEngine.run` threads `step_observer` through to `simulate`. (`run_in_memory`/
   `run_ensemble` untouched.) No `step_out` is passed (the snapshot never needs it).

2. **Snapshot + transport (new, core, UI-independent)** — `osmose/live_movement.py`
   (no UI imports; `queue.Queue` is stdlib, so the transport helper stays here and is
   unit-testable):
   - `MovementSnapshot` (dataclass): `step:int`, `n_steps:int`, `status:str`
     ("running"|"done"|"cancelled"), `species:list[str]`, parallel 1-D numpy arrays
     `sp_id`, `lon`, `lat`, `biomass`, `truncated:bool`, `n_total:int`, grid extent
     `lon_min,lon_max,lat_min,lat_max:float` (from `resolve_grid_latlon`, so the UI can
     frame the view from the first snapshot — see the framing note in step 5), and grid
     cell spacing `lon_step,lat_step:float` (median diff of the full grid coord array; 0
     if < 2 cells; drives dot jitter so it works even when all schools share one cell).
   - `resolve_grid_latlon(grid) -> tuple[np.ndarray, np.ndarray]`: `grid.lat`/`grid.lon`
     if present, else `np.arange(grid.ny, dtype=np.float64)`/`np.arange(grid.nx,
     dtype=np.float64)` (see "Coordinate reality").
   - `build_snapshot(step, state, grid, config, *, status="running", dot_cap=5000) ->
     MovementSnapshot`: pure; `n_steps = config.n_steps`. Selection mask = **focal +
     located + living**:
     `(state.species_id < config.n_species) & ~state.is_out & (state.cell_x >= 0) &
     (state.cell_y >= 0) & (state.biomass > 0)` — the codebase-canonical focal test
     (`simulate.py:943-959`) plus the located guard that drops freshly-spawned eggs at
     `(-1,-1)` (`reproduction.py:183-185`), which would otherwise map to `lon[-1]` (a
     phantom corner blob). (`~is_background` is *not* needed — background schools are
     already stripped at `simulate.py:1499`, before the hook.) Maps survivors to
     `lon[cell_x]`, `lat[cell_y]`, carries `biomass`. If count > `dot_cap`, deterministic
     stride-sample and set `truncated=True` (`n_total` = pre-cap count). Empty selection →
     empty arrays, no error. No jitter here (render-side).

3. **Transport (core helper + single persistent queue)** — `make_step_observer(queue, *,
   dot_cap, now=time.monotonic) -> callable` in `osmose/live_movement.py`. The queue
   itself is a **single persistent `queue.Queue(maxsize=4)` created once** in
   `run_server` (mirroring `calibration_handlers.py:797`, which makes one
   `CalibrationMessageQueue` per session, not per run) — a once-per-session poll closure
   can only safely reference one stable queue object. "Per-run reset" is achieved by
   **draining-and-discarding** any stale items at run start, not by recreating the queue.
   The observer:
   - Builds a snapshot and `put_nowait`s it; on `queue.Full`, frees a slot with a
     **narrow** `try: q.get_nowait() except queue.Empty: pass`, then `put_nowait` — so the
     **newest frame is retained** and the engine thread never blocks.
   - Throttles by wall-clock (≥ ~0.2 s via the injectable `now`) but **always** emits the
     first loop step (index 0) and the final step (`config.n_steps - 1`).
   - Wrapped in try/except around snapshot-build so a failure is logged and swallowed —
     it must never crash the running simulation.

4. **Render helpers (new, UI)** — `ui/pages/live_movement_render.py` (new module:
   `grid_helpers.py` is Plotly/config-page-scoped and 1254 LOC, and its map helpers build
   Plotly `imshow`, not deck.gl; the only deck.gl layer code today is inline at
   `spatial_results.py:570-598` — that is the template). **API matches the proven
   convention** (`spatial_results.py:570-598`, CENOP `map_layers.py:224-236`): first
   positional `id`, **camelCase** deck.gl props, `@@=d.field` accessor strings, and
   **row-dict** data.
   - `_points_to_rows(snapshot, species_filter) -> list[dict]`: zips the snapshot arrays
     into `[{"position": [lon, lat], "weight": biomass, "fill": [r,g,b,a]}, ...]`.
     **Species filter resolution:** `species_filter` is a name or `None`; a name is
     resolved to its index in `snapshot.species` (== `sp_id`, since
     `MovementSnapshot.species = config.species_names` in `sp_id` order — verified
     `output.py:161,219`) and rows are masked by `sp_id`. Colors come from a **local**
     deterministic `species_color(sp_id)` (cycles a categorical RGBA palette by species
     index — `shiny_deckgl.SPECIES_COLORS` is a 3-entry *seal* palette, `ibm.py:48-52`,
     and must NOT be used for fish). The **heatmap** path uses this shared `_points_to_rows`
     builder (and ignores its `fill`); the **dots** path builds rows inline instead, since
     it adds per-row jitter + a biomass-scaled radius the shared builder doesn't carry.
   - `heatmap_layer_from_points(snapshot, species_filter)` →
     `heatmap_layer("live_movement", data=rows, getPosition="@@=d.position",
     getWeight="@@=d.weight", colorRange=color_range(palette=PALETTE_THERMAL))`
     (`PALETTE_THERMAL` + `color_range` from `shiny_deckgl.colors`). Uses un-jittered
     cell-center rows.
   - `dots_layer_from_points(snapshot, species_filter)` →
     `scatterplot_layer("live_movement", data=rows_with_jitter, getPosition="@@=d.position",
     getFillColor="@@=d.fill", getRadius="@@=d.radius", radiusMinPixels=2, pickable=True)`.
     Deterministic per-school jitter (seeded by school index, **no RNG**) bounded to ±¼ of
     the **grid cell spacing carried in the snapshot** (`lon_step`/`lat_step`) — so it
     works even when every school is in the same cell (a per-occupied-coord estimate would
     collapse to 0 there) and needs no `cell_w/cell_h` params. (`*_step == 0`, a 1-cell
     grid → no jitter; `radiusMinPixels` still separates dots visually.)
   - Both use the **same stable layer id** `"live_movement"` so `partial_update` patches
     it in place. Empty rows → a valid empty layer.

5. **UI wiring** — `ui/pages/run.py`. Persistent session state (created once in
   `run_server`): the single `queue.Queue(maxsize=4)`, a `_latest_snapshot:
   reactive.Value[MovementSnapshot | None]`, and a **plain (non-reactive) main-thread
   mutable** `_framed` flag (e.g. a one-element list / closure cell) — NOT a
   `reactive.Value`, because the render effect both reads and writes it, and a reactive
   flag would make the effect re-trigger on its own write.
   Controls: live-view toggle (default **off**), heatmap/dots mode radio, species
   selector. A `MapWidget` panel beside the console (`output_widget`, declared statically
   like `spatial_results.py:147`).

   **On a Python run start with the toggle on:** drain-and-discard any stale queue items;
   `_latest_snapshot.set(None)`; reset the `_framed` flag to False; do ONE initial **full** `await
   map.update(session, layers=[empty layer], view_state=<neutral world default>,
   widgets=[zoom/compass/scale])` to install widgets + basemap (no `Grid` exists yet at
   run-start — it is built inside `engine.run` via `_prepare_run`, `engine/__init__.py:98`
   — so the view is **framed later**, on the first snapshot, not here). Then build the
   observer over the persistent queue and pass `step_observer=observer` to `engine.run`.
   Reset `.set()` and the `await map.update` must be **back-to-back with no awaitable
   between them** so a nested flush can't render before the basemap exists.

   **Consuming the queue (two-part, per `calibration_handlers.py:996-1051`):**
   `@reactive.poll(lambda: time.time(), interval_secs=0.2)` decorates a drain function
   that drains the queue to the latest snapshot and `.set()`s `_latest_snapshot` +
   progress label; a **separate** `@reactive.effect` calls that poll function to drive it
   (a bare `@reactive.poll` is a lazy calc and never runs on its own — the consuming
   effect at `calibration_handlers.py:1049-1051` is mandatory). The always-on 0.2 s poll
   draining an empty queue is a cheap no-op (matches the calibration dashboard's
   always-on 0.5 s poll).

   **Render effect** — `@reactive.effect` on an **`async def`** depending on
   `_latest_snapshot`, `input.live_mode`, `input.species_filter`, `get_theme_mode(input)`:
   guard `snapshot is None → return`. First, swap the **basemap** if the theme changed —
   `style = CARTO_DARK if get_theme_mode(input)=="dark" else CARTO_POSITRON; if style !=
   map.style: map.style = style; await map.set_style(session, style)` (per
   `spatial_results.py:501-505`) — because `partial_update` patches layers only and cannot
   change the deck-level basemap style. Then build the active-mode layer; if `not _framed`
   → `await map.update(session, layers=[layer], view_state=<framed from snapshot
   lon_min/max/lat_min/max>)` (a **full** update — `update` serializes `view_state`,
   `map_widget.py:341-342`, while `partial_update` sends only `id`+`layers`,
   `map_widget.py:384-387`) then set the `_framed` flag True; else `await
   map.partial_update(session, layers=[layer])`. Mirror the async-effect precedent
   `spatial_results.py:498-505,630`. Toggle-off / Java → static map + note.

## Data flow & lifecycle

- **Start (main thread):** toggle on + Python run → drain-and-discard stale items from
  the persistent queue, `_latest_snapshot.set(None)` + reset the `_framed` flag, then (no
  awaitable between the resets and this `await`) an initial full `map.update(...)` that
  installs widgets + basemap with a neutral default `view_state` (the grid doesn't exist
  yet — framing happens on the first snapshot), then `engine.run(..., step_observer=observer)`
  in the existing executor.
- **Producer (executor thread):** each step after `accumulated.append(step_out)`, the
  observer throttles → `build_snapshot` → `put_nowait` (drop-oldest, newest retained).
  Never blocks, never raises into the loop.
- **Consumer (main thread):** a `@reactive.poll` (0.2 s) drain-calc + a separate
  consuming `@reactive.effect` that drives it (both required) drain the queue to the
  most-recent snapshot and `.set()` `_latest_snapshot` + progress. The async render
  effect frames the view on the **first** snapshot (full `update` with `view_state` from
  the snapshot's grid bounds, then `_framed=True`) and uses `partial_update` of only the
  moving layer thereafter.
- **Completion / cancel:** after `engine.run` returns / `SimulationCancelled` / error,
  the handler is back on the **main thread** (post-`await` at `run.py:270-273`); it builds
  the terminal `MovementSnapshot(status="done"|"cancelled")` and calls
  `_latest_snapshot.set(terminal)` **directly** (NOT via the queue — the poll could stop
  before draining it), then marks the run idle. The final frame stays on screen.
- **Threading rule (load-bearing):** reactive `.set()` happens **only** on the main
  thread (in the poll body and the post-await handler); the observer touches only the
  thread-safe queue.
- **Cleanup:** the queue is a **single persistent object** (created once in `run_server`,
  not recreated per run); "per-run reset" = drain-and-discard stale items + reset
  `_latest_snapshot`/`_framed` at run start. The poll closure references that one stable
  queue. No disk.

## Error handling & edge cases

| Case | Behavior |
|---|---|
| Toggle off / Java engine | `step_observer=None`; loop untouched; static map + "live view available for the Python engine" note. |
| Snapshot/`put` raises | Observer logs + swallows; frame skipped; sim continues. |
| Queue full (UI lag) | Free one slot (narrow `queue.Empty` catch) + `put_nowait` newest; engine never blocks. |
| No located living focal schools at a step | Empty arrays → empty layer (basemap only). |
| Unlocated eggs (cell=-1) | Excluded by the `cell_x/cell_y >= 0` mask (never placed at `lon[-1]`). |
| `> dot_cap` schools | Deterministic stride sample; `truncated=True` → "showing N of M". |
| Rectangular (non-NetCDF) grid | `resolve_grid_latlon` returns index coords `0..n` (matches the post-run spatial path); render is non-geographic — documented limitation. |
| Run cancelled | `cancel_token` raises `SimulationCancelled`; handler sets a `cancelled` terminal snapshot directly; final frame retained. |
| Run errors | Existing run-error path; handler sets a terminal snapshot directly; last frame retained; error via the existing run-log/notification. |
| Toggle mode/species after run | Async render effect re-renders the retained `_latest_snapshot` (it depends on `input.live_mode`/`input.species_filter`). |
| Theme switch | Render effect depends on `get_theme_mode(input)`; palette re-themes via the layer, basemap via `await map.set_style(...)` guarded by `style != map.style`. |
| Rapid successive runs | Drain-and-discard stale items from the persistent queue + reset `_latest_snapshot`/`_framed`; stale frames discarded. |

## Testing

- **Unit — `osmose/live_movement.py` (pure):** `build_snapshot` maps `cell_x/cell_y` →
  cell-center `lon/lat`; carries biomass; selection includes only focal+located+living
  schools and **drops cell=-1 eggs** (assert a -1 school is absent, not placed at
  `lon[-1]`); excludes `is_out` and zero-biomass; sets step/`n_steps`(=config.n_steps)/
  status. `dot_cap` exceeded → length==cap, `truncated`, correct `n_total`, deterministic
  sample. Empty selection → empty arrays. `resolve_grid_latlon` returns grid arrays when
  present and `np.arange` indices when None.
- **Unit — `make_step_observer` (no Shiny, injected `now`):** always emits step 0 +
  final; throttles intermediate; on a `maxsize=1` queue rapid puts keep only the newest
  and never block/raise; a snapshot exception is swallowed.
- **Unit — render helpers:** `_points_to_rows` produces row dicts with `position`/
  `weight`/`fill`; `species_color` gives every focal species a distinct defined RGBA;
  `heatmap_layer_from_points` uses `getWeight`/`colorRange` with a stable id and species
  filter reduces rows; `dots_layer_from_points` jitter is deterministic (same index→same
  offset, bounded to ±¼ of the snapshot's `lon_step`/`lat_step` grid spacing, so two
  schools in the SAME cell still get distinct positions; `*_step==0` → no jitter), empty
  rows → valid empty layer.
- **Engine hook (parity, concrete):** with the same seed, run
  `out_a = simulate(cfg, grid, default_rng(s))` and
  `out_b = simulate(cfg, grid, default_rng(s), step_observer=spy)`;
  `np.testing.assert_array_equal` each StepOutput field (biomass/abundance) across all
  steps → identical (no-overhead/parity); assert `spy` was called exactly `config.n_steps`
  times with monotonic `step` and `config` carrying the right `n_steps`; cancel still
  works with an observer attached.
- **e2e (Playwright):** substrate = **`data/baltic`** with `simulation.time.nyear=1`
  (data/baltic is `ndtPerYear=24`, so nyear=1 → **24 steps**; runs in a few seconds warm
  after JIT warmup; maps movement → 100% located; NetCDF grid → real lat/lon).
  **NOT `data/minimal`** (random movement leaves schools unlocated → renders nothing).
  Live toggle on → Run → live map panel appears, progress advances, a frame renders
  (heatmap); toggle to dots renders; species filter changes; on completion the final
  frame + "complete" shows; Java/toggle-off shows the note. Also exercise the **cancel
  path**: cancel mid-run → the retained frame stays and a "cancelled" status shows (covers
  the terminal-snapshot-direct-set, whose cancel/error branches have no unit test). Widget
  content is shadow-DOM → assert container + screenshot (per the scenario-diff / CENOP e2e
  precedent).

## Build order (TDD)

1. `osmose/live_movement.py`: `MovementSnapshot`, `resolve_grid_latlon`, `build_snapshot`,
   `make_step_observer` + unit tests.
2. `ui/pages/live_movement_render.py`: `_points_to_rows`, `species_color`,
   `heatmap_layer_from_points`, `dots_layer_from_points` + unit tests.
3. Engine hook: keyword-only `step_observer` in `simulate` + `PythonEngine.run` + parity/
   fires-per-step/cancel tests.
4. `ui/pages/run.py`: add `import time, queue` (currently imported by neither); persistent
   queue + `_latest_snapshot` reactive + `_framed` plain flag; controls + MapWidget panel; initial
   full update (widgets + neutral view); the two-part `reactive.poll` drain-calc +
   consuming effect; async render effect (frame-on-first-snapshot then partial_update);
   terminal-snapshot-direct-set on completion/cancel/error.
5. e2e Playwright validation against `data/baltic` (nyear=1).
