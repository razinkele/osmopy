# Scenario Diff View — Design

**Date:** 2026-06-13
**Status:** Approved (brainstorming) — in-loop review converged (rounds 1–3; clean)
**Backlog item:** UI / Shiny → "Scenario diff view (side-by-side biomass + spatial maps)"

## Goal

Let a user visually compare two completed runs ("scenarios") — a **baseline (A)**
and a **variant (B)** — showing both the per-species biomass trajectories and the
spatial fields side-by-side, plus a computed **B−A difference map** that highlights
*where* a config change moved the ecosystem.

This complements, rather than replaces, the existing **Compare Runs** tab, which
does N-run numeric biomass deltas. Scenario Diff is a focused 2-run *visual*
side-by-side that adds the spatial dimension.

## Non-goals (YAGNI)

- No N-way (>2) comparison — that is what Compare Runs already does.
- No new run-execution path; operates only on runs already in history.
- No spatial overlay on a basemap / deck.gl polygon layer — flat plotly `imshow`
  maps only (matches the existing Spatial Results "Flat" view).
- No diffing of configs/text (that is a separate backlog item, "Config diff tool").

## Architecture & placement

Follows the codebase split: **pure data logic in `osmose/`**, **rendering and
reactives in `ui/pages/`**.

- **New tab:** `ui.nav_panel("Scenario Diff")` inside the existing
  `navset_card_tab` in `ui/pages/results.py` (the same navset that holds
  "Compare Runs" at `ui/pages/results.py:286-336`). This honors the chosen
  placement (a tab in Results, not a new top-level page).
- **New module** `ui/pages/scenario_diff.py` (keeps `results.py` lean), exporting:
  - `scenario_diff_nav_panel()` → returns the `ui.nav_panel(...)` to embed in the
    Results navset.
  - `scenario_diff_server(input, output, session, state)` → wires reactives/outputs;
    called once from inside `results_server` (`ui/pages/results.py:345`), which
    already receives `state`.
  - **Note — this is a NEW pattern.** No existing page nests a `*_nav_panel()`
    builder in another page's navset, and `results_server` currently calls no
    sub-server (every other page, e.g. `spatial_results` at `app.py:272,533`, is a
    top-level page with its own `*_ui()`/`*_server()`). The sub-server call is
    deliberately introduced here to keep the already-500+-line `results_server`
    from growing; it is valid (the closure receives the same `input/output/
    session/state`) but should be documented as new, not described as mirroring
    existing factoring.
- **New pure function** in `osmose/spatial_series.py`:
  `spatial_diff_2d(ds_a, ds_b, variable, *, time_a=0, time_b=0, species=None, reduce="sum")`
  → returns `(B − A)` as a 2-D `np.ndarray` (NaN where either side is land/missing).
  - Reuses `spatial_slice_2d` for each side; `species` and `reduce` carry the exact
    same semantics as in `spatial_slice_2d` (`reduce` only applies when
    `species is None`).
  - **Coordinate alignment, not shape equality.** Before subtracting, it asserts the
    two grids correspond cell-to-cell. Check **shape equality first** (an
    `np.allclose`/broadcast check alone would let `lat=[55]` broadcast against
    `lat=[55,55]`), then assert **exact** coordinate equality with
    `np.array_equal(lat_a, lat_b)` and `np.array_equal(lon_a, lon_b)`. Exact (not
    `np.allclose`) is correct here because the engine writes `lat`/`lon` as plain
    float64 with no packing (`osmose/engine/output.py:747-752`), so identical grids
    roundtrip bit-exactly; `np.allclose`'s default `rtol=1e-5` would instead tolerate
    ~60 m of drift at Baltic latitudes and pass two genuinely different near-aligned
    grids. Locate the coord via the existing dim-name sets (don't hardcode `"lat"` —
    OSMOSE output uses `lat`/`lon`, but `spatial_slice_2d` already tolerates
    `latitude`/`y`/`x`). Raise `ValueError` on any mismatch. Equal *shape* alone is
    insufficient — two runs can share `(ny, nx)` but differ in coordinate values or
    lat orientation (`spatial_slice_2d` returns a coordinate-stripped `np.ndarray`,
    so positional subtraction would silently compare different geographic cells).
  - **Species must be passed by NAME**, not integer index. `spatial_slice_2d`
    selects a single species by `.sel(name)` for strings but `.isel(int)` for ints
    (`osmose/spatial_series.py:222-225`); index 2 can denote a different species in
    each run. `spatial_diff_2d` therefore accepts only a species *name* or `None`,
    and the int-index branch is not used for diffing.
  - **Summing an explicit subset:** `spatial_slice_2d` only supports `species=None`
    (all) or a single name (`osmose/spatial_series.py:221-230`); it cannot sum an
    arbitrary list. For the "All (summed)" path over the common species subset, the
    caller (or `spatial_diff_2d` internally) first narrows each dataset with
    `ds.sel({species_dim: common_names})`, then calls `spatial_slice_2d(...,
    species=None, reduce="sum")` on the narrowed dataset.
  - Separate `time_a`/`time_b` indices let the caller align on real time (see Data
    flow).
- **New thin renderer** in `ui/pages/grid_helpers.py`:
  `make_diff_map(diff_array, lat, lon, *, var_name, title=None, template="osmose")`
  → plotly `imshow` with a **diverging RdBu colorscale, `zmid=0`** and a **symmetric
  range** `zmin=-h, zmax=+h` where `h = max(|finite values|.max(), EPS)`.
  - It is array-input (the precomputed diff), unlike `make_spatial_map` which is
    dataset-input (`ui/pages/grid_helpers.py:1166`) — so it is a sibling, not a
    drop-in. The shared non-finite→`None` z-serialization (currently inline at
    `ui/pages/grid_helpers.py:1188`, required because shinywidgets serialises with
    `json(allow_nan=False)`) is factored into a small shared helper used by both.
  - **Degenerate guards:** if no finite cells (all land / all-NaN) → return an
    empty-state figure (do NOT call `np.nanmax` on an all-NaN array). If all finite
    cells are 0 (identical runs) → the `EPS` floor keeps a valid, flat colorbar.
  - **Coord source:** the server passes `lat`/`lon` extracted via the **same**
    dim-name lookup used by `spatial_diff_2d` (do not copy `make_spatial_map`'s
    hardcoded `ds["lat"]`/`ds["lon"]` at `ui/pages/grid_helpers.py:1190-1191`), so the
    diff array and its axes are guaranteed to agree.
- **New pure plotting function** in `osmose/plotting.py`:
  `make_biomass_overlay(long_a, long_b, species, *, template="osmose")` → one figure,
  two traces per selected species (A solid, B dashed). Placed beside the existing
  `make_run_comparison`/`make_species_dashboard` (NOT inline in the reactive), to
  match the codebase split. (`make_run_comparison` at `osmose/plotting.py:444` is a
  grouped *bar chart of summary scalars* — it does NOT render per-species
  trajectories, so the overlay is genuinely new.) Its inputs are **already-normalized
  long-form frames** (`time, species, value`) produced by the shared
  `biomass_long` helper below — not raw `OsmoseResults.biomass()` output — so the
  overlay never re-derives the WIDE/LONG shape itself.

## Data flow

- **Run sources:** both selectors are populated from
  `_compare_run_choices(default_run_history().list_runs())` — the same plumbing the
  Compare Runs selector uses (`ui/pages/results.py:76`). Choices are **keyed by
  `RunRecord.timestamp`**; resolve a selected value to a directory via
  `default_run_history().load_run(ts).output_dir` →
  `OsmoseResults(output_dir, strict=False)` (the resolution path Compare Runs uses
  at `ui/pages/results.py:744,194-195`).
- **Biomass (shape-aware — do NOT iterate columns naively):**
  `OsmoseResults.biomass()` (`osmose/results.py:392`) concatenates per-species
  output files and **adds a constant `species` column**, yielding a frame that the
  rest of the codebase treats with an explicit WIDE/LONG discrimination
  (`osmose/analysis.py:183-219`, keyed on whether a `value` column exists, excluding
  `_NON_SPECIES_COLS = {"Time","time","species"}` at `osmose/analysis.py:169`).
  the overlay MUST reuse that same normalization rather than treating raw columns as
  species. No reusable long-normalizer exists today (the WIDE/LONG logic is inline in
  `_per_species_window_mean` and returns a dict, not a frame), so introduce a shared
  `biomass_long(results: OsmoseResults) -> pd.DataFrame` (columns `time, species,
  value`) — calling `results.biomass()` internally — and use it for BOTH the overlay
  (which receives the resulting long frames) and as the single place the WIDE/LONG
  discrimination lives. The species list for `diff_species` derives from this long
  frame's `species` column, never from raw `biomass()` columns.
- **Caption (mean B−A over a trailing window):** reuse `_delta_for_selected(records,
  metric, window_years)` (`ui/pages/results.py:181`), which wraps `run_delta`
  (`osmose/analysis.py:232`, returns `list[SpeciesDelta]` with
  `abs_delta = variant_mean − baseline_mean`, i.e. B−A). `window_years` must be ≥1
  (it raises otherwise, `osmose/analysis.py:197-199`). If `_delta_for_selected` must
  be shared across modules, lift it to a shared helper rather than importing across
  pages.
- **Spatial availability:** a run "has spatial output" iff `res.list_outputs()`
  contains a `.nc` file.
- **NetCDF handle lifecycle (single handle per run, not per render):** follow the
  established Spatial Results pattern — hold each run's open dataset in its own
  `reactive.Value` (`_diff_ds_a`, `_diff_ds_b`), opened when the selected run changes
  and **closed-on-swap** (mirroring `ui/pages/spatial_results.py:167,193-244`). Every
  render (the `@render.ui` slider and the three maps) reads from these shared
  handles. Do NOT open/close per render — with four render functions touching the
  data that is up to 4 opens × 2 files per reactive cycle, and overlapping render
  passes re-introduce the exact HDF5 locking error the Cell Series open-dataset fix
  exists to avoid.
- **Spatial species set (common subset):** the maps' species selector
  (`diff_spatial_species`) lists the **intersection** of both runs' species names.
  "All (summed)" sums that common subset in *each* run (pass the explicit common
  name list to each side), so the summed A, B, and B−A maps are comparable. Two runs
  with disjoint or differently-sized species sets would otherwise produce
  non-comparable sums.
- **Time alignment (by value, not index):** the engine writes `time` as fractional
  years *from each run's own simulation start* at *each run's cadence*
  (`osmose/engine/output.py:759`), monotonic non-decreasing (`o.step` increases,
  `osmose/engine/output.py:715`). Index `i` is therefore NOT guaranteed to be the
  same real time across two runs. The diff slider ranges over the **overlapping
  time-coordinate interval** `[max(t_a[0], t_b[0]), min(t_a[-1], t_b[-1])]`; for the
  chosen value `v` the code picks the **nearest index in each run independently** via
  `int(np.abs(times - v).argmin())` (no interpolation) → `time_a`, `time_b`, and
  shows each run's actual `time` value in its map title. If the two runs' cadence
  differs (compare `ds.attrs["n_dt_per_year"]`, written at
  `osmose/engine/output.py:757`), surface a one-line warning.

## UI

Tab sidebar controls:

- `diff_run_a` — baseline selector.
- `diff_run_b` — variant selector.
- `diff_species` — multi-select for the biomass overlay (species common to both
  runs, derived via the normalized species list, not raw columns).
- `diff_spatial_species` — single-select for the maps over the **intersection** of
  both runs' species names; includes an "All (summed)" option that sums that common
  subset in each run. Build each run's species list the way Spatial Results does
  (`[str(s) for s in ds["species"].values]`, `ui/pages/spatial_results.py:305`), then
  intersect. Selection is always by name (never index).
- `diff_time` — a slider rendered **dynamically via `@render.ui` returning
  `ui.input_slider(...)`** (the pattern `spatial_results.py:321-342` uses), keyed off
  both selected runs' datasets, ranging over the overlapping time interval. Do NOT
  use imperative `ui.update_slider` against a static slider — that hits a
  reactive-ordering hazard when the two datasets resolve.

Outputs:

1. **Biomass overlay** (always renders — every run has CSV biomass): one plotly
   figure via `make_biomass_overlay`; for each selected species two traces — A
   solid, B dashed. A short caption shows the mean B−A over a trailing window. The
   "always renders" guarantee is load-bearing on `OsmoseResults(output_dir,
   strict=False)` + lazy init: a missing dir → `list_outputs() == []` and
   `biomass()` returns an empty frame rather than raising
   (`osmose/results.py:292-295,339,658-661`). Do NOT add an `output_dir.exists()`
   guard that raises — degrade to the per-run empty-state instead.
2. **Spatial row** (graceful degrade): three plotly maps — `A` and `B` via
   `make_spatial_map`, and `B−A` via `make_diff_map`. Rendered only when **both**
   runs expose a `.nc`; otherwise a single empty-state card: *"No spatial output —
   enable `output.spatial` in both configs."*

## Error handling & edge cases

| Case | Behavior |
|---|---|
| Same run for A and B | Allowed; diff map all-zero (EPS floor keeps a valid colorbar); caption notes "identical runs". No crash. |
| Spatial grid coordinate mismatch | `spatial_diff_2d` raises `ValueError`; UI catches → empty-state "Grids differ — cannot diff spatially." Biomass overlay still renders. |
| Different time cadence / start | Slider over overlapping interval; nearest-index per run; map titles show each run's real time; one-line cadence-mismatch warning. |
| Only one run has NetCDF | "Spatial unavailable" → spatial empty-state (need both). |
| Missing/empty biomass CSV for a run | Per-run empty-state inside the overlay; the other run still plots. |
| NetCDF re-open ("HDF error") | One handle per run held in a `reactive.Value`, opened on run-change and closed-on-swap; renders share it (no per-render open/close). |
| Disjoint species sets between runs | Spatial selector and "All (summed)" use the common-name intersection; overlay species likewise. |
| Land cells (NaN) | Shared z-builder maps non-finite → `None` for both map types. |
| Diff array all-NaN (both land) | `make_diff_map` returns an empty-state figure (no `nanmax` on all-NaN). |
| No runs / one run in history | Selectors empty/short; both outputs show "Select two runs to compare." |

## Testing

**Unit** — add to `tests/test_spatial_series.py`, covering `spatial_diff_2d`, using
the existing `_make_spatial_nc(path, *, n_time, species, ny, nx, land_cell)` fixture
(`tests/test_spatial_series.py:16`):

- matching grids subtract correctly (verify B−A sign convention).
- land NaN propagates: result is NaN if *either* side is land/missing.
- `species=` single-select vs `reduce="sum"` / `reduce="mean"`.
- `time_a`/`time_b` select the correct step on each side independently.
- **lat/lon coordinate mismatch (same shape, different coords) → `ValueError`.**
- A == B → all-zero array (finite cells 0.0, land cells NaN).

Plus unit tests for `make_diff_map` (all-NaN → empty-state fig; all-zero → valid
symmetric range) and `make_biomass_overlay` (correct trace count, dashed B,
shape-aware species selection on a constant-`species`-column frame).

**Live UI validation (Playwright).** The existing suite has **no** fixture that
injects runs into history, and `RUN_HISTORY_DIR` is a fixed path
(`osmose/history.py:15`) read by the app subprocess — so monkeypatching is not
possible. The mechanism is therefore: a test/dev harness **writes synthetic
`run_*.json` records (with `output_dir` pointing at a synthetic
`(time,species,lat,lon)` NetCDF + minimal real biomass CSVs in a repo-internal dir)
into the real `data/history/` directory before `page.goto`, with teardown that
removes them**. This is new infrastructure, acknowledged as such. Live checks:

- select A & B → biomass overlay shows traces for both runs.
- three spatial maps render (verify via screenshot; plotly `output_widget` content
  is not reliably DOM-queryable — existing e2e tests assert deck.gl via
  `#grid_map canvas`, not plotly internals).
- a run lacking NetCDF → spatial empty-state shows, biomass overlay still renders.

A lightweight **structure test** (mirroring `tests/test_ui_results.py:296-305`'s
source-string assertions) confirms the tab + server are wired, independent of the
heavier live harness.

## Build order (TDD)

1. `spatial_diff_2d` in `osmose/spatial_series.py` (coord-aligned) + unit tests.
2. `make_diff_map` (+ shared z-builder helper) in `ui/pages/grid_helpers.py` + unit
   tests.
3. `make_biomass_overlay` + the shared biomass-normalization helper in
   `osmose/plotting.py` / `osmose/analysis.py` + unit tests.
4. `ui/pages/scenario_diff.py` — nav panel + server.
5. Wire panel + sub-server into `ui/pages/results.py`; structure test.
6. Playwright validation against the synthetic `data/history/` substrate.

## Substrate note

No shipped config sets `output.spatial.enabled=true`, so the spatial half is
validated with a synthetic `(time, species, lat, lon)` NetCDF plus real minimal
biomass CSVs in a repo-internal directory, surfaced to the app by writing synthetic
run records into `data/history/`. The biomass half works against any two real runs
in history.
