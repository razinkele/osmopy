# Map-Based Scenario Builder — Design

> A Shiny tool to author OSMOSE spatial grid maps (species distribution/movement maps, the land mask, or generic zones) by **drawing on a georeferenced map** — polygon-fill + cell-brush — and save them as the `;`-separated grid CSVs the engine reads, with type-aware config wiring. Closes a real gap: today these maps are hand-crafted offline.

## Goal

A new "Map Builder" UI page where a user draws regions/cells on the loaded config's georeferenced grid and the tool writes a standard map CSV + wires it into the config (a distribution `movement.file.map{N}` with applicability, the `grid.mask.file`, or a bare zone CSV). Works for both engines (the output is a plain config map; no engine change).

## Scope

In scope: a **general grid-map painter** for all OSMOSE grid-CSV map types (distribution/movement, land mask, generic zone); **polygon-draw + cell-brush**; **numeric paint value (default 1) + an editable land mask**; **save CSV + type-aware config wiring**. Built on `shiny_deckgl` (approach A — native interactive map).

Out of scope (YAGNI): spatial-fishing/MPA effort mechanics beyond a generic zone CSV; multi-map batch editing; undo/redo history beyond a single clear/reset; authoring the grid bounds themselves (the grid is taken from the loaded config); 4.4.0-specific map features.

## Background (verified)

- OSMOSE spatial maps are georeferenced grid CSVs: `nlat` rows × `nlon` cols, `;`-separated, `-99` = land/masked, otherwise a presence/probability value (e.g. Baltic = 40×50; `data/baltic/maps/cod_adult.csv` is all `1`/`-99`). The grid is georeferenced via `grid.upleft.{lat,lon}` / `grid.lowright.{lat,lon}` + `grid.nlon`/`grid.nlat`; land from `grid.mask.file`.
- Distribution maps are wired via `movement.file.map{idx}` + per-map applicability (`movement.map{idx}.species/age.min/age.max/season/years`, per `osmose/schema/movement.py`).
- `shiny_deckgl` supports drawing natively: `DrawMode.{DRAW_POLYGON,DRAW_RECTANGLE,...}`, `pickable` layers, `info_widget` (pick) — it is interactive, not render-only. `ui/pages/grid_helpers.py::build_grid_layers` already maps a grid → per-cell lat/lon polygons (georeferencing solved); `load_mask` reads the base mask. `ui/pages/map_viewer.py` shows the established render pattern + view-state.

## Architecture

Two new units + reuse of existing spatial infra, splitting all browser-independent logic into a pure core:

- **`osmose/maps/builder.py`** — pure, Shiny-free, fully unit-testable: grid geometry, rasterization, paint/erase/mask ops, CSV (de)serialization, validation, and the config-wiring key computation.
- **`ui/pages/map_builder.py`** — the Shiny page: interactive deck.gl map (draw + brush), controls, reactive grid state, save handler. Thin; delegates to the core.
- **Reuse:** `ui/pages/grid_helpers.py` (`build_grid_layers`, `load_mask`), `ui/state.py` `AppState`, `osmose/schema/movement.py` field defs, the `map_viewer` view-state conventions.

### `osmose/maps/builder.py` — interface

- `GridSpec` (dataclass) — from `state.config`: `nlon, nlat, upleft_lat, upleft_lon, lowright_lat, lowright_lon`. Methods/derived: `cell_centers() -> (lat_2d, lon_2d)`, `cell_polygons() -> list[[lon,lat]×4]` (corner rectangles; share the math `grid_helpers` uses).
- `MapGrid` (wraps an `np.ndarray` shape `(nlat, nlon)`, land = `-99`):
  - `apply_cells(cells: Iterable[(row,col)], value: float) -> None`
  - `apply_polygon(grid_spec, polygon_lonlat, value, *, mask_edit=False) -> None`
  - `erase(cells) -> None` (→ `0`)
  - `set_mask(cells, masked: bool) -> None` (→ `-99` or `0`)
  - `array -> np.ndarray`
- `rasterize_polygon(grid_spec, polygon_lonlat, mask, *, mask_edit=False) -> list[(row,col)]` — `matplotlib.path.Path.contains_points` over cell centers; excludes `-99` cells unless `mask_edit`.
- `to_csv_text(map_grid) -> str` — `;`-separated rows, `-99` land, integer-or-float values; matches the engine's map CSV format + the reader's separator.
- `from_csv_text(text, grid_spec) -> MapGrid` — parse + validate dims == `(nlat, nlon)` (raises `ValueError` on mismatch).
- `validate(map_grid, grid_spec) -> list[str]` — returns problems: dim mismatch; mask cells inconsistent with the grid mask (for mask-type saves).
- `wire_map_into_config(config, map_type, rel_path, *, applicability=None) -> (new_config, summary)` — pure: computes the config keys to add. Distribution → next free `movement.file.map{N}` index + `movement.map{N}.species/age.min/age.max/season/years` from `applicability`; mask → `grid.mask.file = rel_path`; zone → no keys. Returns the updated config dict + a human summary. (UI applies it via `AppState`.)

### `ui/pages/map_builder.py` — page

- New nav entry "Map Builder" (navset_pill_list, per the UI architecture). On entry: read grid from `state.config` → `GridSpec` + base mask (`load_mask`). No grid → hint + disabled.
- **Start controls:** "New blank map" (`-99` mask, else `0`) | "Load existing map…" (select a `movement.file.map{N}`/`grid.mask.file` from the config → `from_csv_text`).
- **Tool mode:** Polygon-draw | Brush | Eraser | Mask-edit. **Paint value:** numeric (default `1`).
- **Map-type selector:** Distribution | Land mask | Generic zone — drives Save.
- **Distribution applicability form** (shown for Distribution type): species (required), age min/max, season steps, years — reuse `schema/movement.py` field renders.
- **Interactive map:** `build_grid_layers`-rendered cells over the basemap (view-state from grid bounds), colored by current value (empty transparent, value colored, land gray); `DrawMode.DRAW_POLYGON`/`RECTANGLE` in polygon mode; pickable click/drag in brush/eraser/mask modes.
- **Reactive state:** `reactive.Value[np.ndarray]`. Draw event → `rasterize_polygon`→`apply_polygon`; brush → `apply_cells`; mask-edit → `set_mask`; each updates the array → live re-render.
- **Save:** filename input → `to_csv_text` → write to `<config maps dir>/<name>.csv` (the working/run config location the app uses for edits, NOT the read-only demo source) → `wire_map_into_config` → `AppState.load_config`/update → reload. Toast on success.

## Data flow

`state.config` (grid bounds + mask) → `GridSpec` + base mask → init `MapGrid` (blank or loaded) → `reactive.Value` → [polygon-draw → `rasterize_polygon`→`apply_polygon`; brush → `apply_cells`; mask-edit → `set_mask`] → reactive array → deck.gl re-render → Save → `to_csv_text` → write file + `wire_map_into_config` + reload `state.config`.

## Error handling / edge cases

- No grid / malformed bounds → builder disabled, hint "Load a config with a defined grid first."
- Paint a value on a land (`-99`) cell → blocked unless Mask-edit mode (the explicit land/sea change path).
- Mask save → `validate` dims == `grid.nlon×nlat` (don't produce a mask inconsistent with the configured grid).
- Polygon entirely outside the grid → 0 cells matched → info toast, no-op.
- Distribution map saved with no species selected → block with a clear message (an unapplied map is a silent footgun).
- Overwriting an existing map file → confirm.
- Writing into a loaded demo config (read-only source) → write into the working/run `maps/` dir (mirror how the app already persists config edits via `state.config` + the run-time writer), never the original demo dir. **KEY INTEGRATION DETAIL the plan MUST pin (by reading `ui/pages/run.py::write_temp_config` + how `movement.file.map{N}` relative paths resolve at run time):** a newly-drawn map referenced as `maps/<name>.csv` must actually be found by the engine when it runs. Determine where map files live relative to the config the engine consumes (does `write_temp_config` copy referenced map files into the run dir? are paths config-dir-relative or absolute?) and write the new CSV + register a path that resolves for BOTH the in-UI render and the run. If demo `maps/` is read-only, the resolution is a writable working/config dir the app controls (or an absolute path in the key). Resolve this before the UI save path is finalized — it is the one real integration risk.

## Testing strategy

- **Pure core (CI — the bulk):** `rasterize_polygon` (concave polygons, cells on polygon edges, Baltic-style bounds, polygon outside grid); `apply_cells`/`erase`/`set_mask`; `to_csv_text`/`from_csv_text` round-trip (exact match to the engine map CSV format + separator); `validate` (dim mismatch, mask consistency); `GridSpec` cell-center geometry vs the grid bounds. Property test (Hypothesis): the rasterized cell set equals exactly the cells whose center lies inside the polygon. `wire_map_into_config` (next-free map index; correct `movement.map{N}.*` keys for distribution; `grid.mask.file` for mask; no keys for zone).
- **UI (no browser):** `import ui.pages.map_builder` clean; the save handler writes a CSV + applies the wiring keys (mirror existing ui page tests, e.g. `test_ui_*`).
- **deck.gl draw/pick event binding** (the one hard-to-unit-test seam) → covered by the e2e/visual Playwright harness + manual; all event→cell logic lives in the pure core, tested independently of the widget.
- Full suite green; ruff + pyright clean.

## Out of scope (future)

Spatial-fishing effort mechanics; undo/redo stack; grid-bounds authoring; importing a map from a raster/shapefile; multi-map batch tools.
