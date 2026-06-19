# Map-Based Scenario Builder — Design

> A Shiny tool to author OSMOSE spatial grid maps (species distribution/movement maps, the land mask, or generic zones) by **drawing on a georeferenced map** — polygon-fill + cell-brush — and save them as the `;`-separated grid CSVs the engine reads, with type-aware config wiring. Closes a real gap: today these maps are hand-crafted offline.

## Goal

A new "Map Builder" UI page where a user draws regions/cells on the loaded config's georeferenced grid and the tool writes a standard map CSV + wires it into the config (a distribution `movement.file.map{N}` with applicability, the `grid.mask.file`, or a bare zone CSV). Works for both engines (the output is a plain config map; no engine change).

## Scope

In scope: a **general grid-map painter** for all OSMOSE grid-CSV map types (distribution/movement, land mask, generic zone); **polygon-draw + cell-brush**; **numeric paint value (default 1) + an editable land mask**; **save CSV + type-aware config wiring**. Built on `shiny_deckgl` (approach A — native interactive map).

Out of scope (YAGNI): spatial-fishing/MPA effort mechanics beyond a generic zone CSV; multi-map batch editing; undo/redo history beyond a single clear/reset; authoring the grid bounds themselves (the grid is taken from the loaded config); 4.4.0-specific map features.

## Background (verified)

- OSMOSE spatial maps are georeferenced grid CSVs: `nlat` rows × `nlon` cols, `;`-separated, `-99` = land/masked, otherwise a presence/probability value (e.g. Baltic = 40×50; `data/baltic/maps/cod_adult.csv` is all `1`/`-99`). The grid is georeferenced via `grid.upleft.{lat,lon}` / `grid.lowright.{lat,lon}` + `grid.nlon`/`grid.nlat`; land from `grid.mask.file`.
- Distribution maps are wired via `movement.file.map{idx}` + per-map applicability — the REAL keys (verified in `osmose/schema/movement.py` + read at `osmose/engine/movement_maps.py:128-178`): `movement.species.map{idx}` (single species name), `movement.steps.map{idx}`, `movement.initialage.map{idx}`/`movement.lastage.map{idx}`, `movement.initialyear.map{idx}`/`movement.lastyear.map{idx}` (lastyear inclusive) or `movement.years.map{idx}`. (NOT `movement.map{idx}.*` — that inverted form is a known no-op bug the schema warns about.) Maps are discovered by iterating `movement.species.map{N}`. CSV maps are stored **south-row-0** on disk; the reader flips (`np.flipud` / `ny-1-csv_row_idx`).
- `shiny_deckgl` supports drawing natively: `DrawMode.{DRAW_POLYGON,DRAW_RECTANGLE,...}`, `pickable` layers, `info_widget` (pick) — it is interactive, not render-only. `ui/pages/grid_helpers.py::build_grid_layers` already maps a grid → per-cell lat/lon polygons (georeferencing solved); `load_mask` reads the base mask. `ui/pages/map_viewer.py` shows the established render pattern + view-state.

## Architecture

Two new units + reuse of existing spatial infra, splitting all browser-independent logic into a pure core:

- **`osmose/maps/builder.py`** — pure, Shiny-free, fully unit-testable: grid geometry, rasterization, paint/erase/mask ops, CSV (de)serialization, validation, and the config-wiring key computation.
- **`ui/pages/map_builder.py`** — the Shiny page: interactive deck.gl map (draw + brush), controls, reactive grid state, save handler. Thin; delegates to the core.
- **Reuse:** `ui/pages/grid_helpers.py` (`build_grid_layers`, `load_mask`), `ui/state.py` `AppState`, `osmose/schema/movement.py` field defs, the `map_viewer` view-state conventions.

### `osmose/maps/builder.py` — interface

- `GridSpec` (dataclass) — from `state.config`: `nlon, nlat, upleft_lat, upleft_lon, lowright_lat, lowright_lon`. Derived (MUST match `grid_helpers.build_grid_layers` exactly: `dx=(lr_lon-ul_lon)/nlon`, `dy=(ul_lat-lr_lat)/nlat`): `cell_centers() -> (lat_2d, lon_2d)` with center of `(r,c)` = `(ul_lat-(r+0.5)*dy, ul_lon+(c+0.5)*dx)`; `cell_polygons() -> list[[lon,lat]×4]` (corner rectangles). **Row/orientation convention: the in-memory array is NORTH-row-0** (row 0 = `upleft`/northernmost), matching `build_grid_layers`'s deck.gl layout. A test asserts `cell_polygons()` matches `build_grid_layers` cell-for-cell.
- `MapGrid` (wraps an `np.ndarray` shape `(nlat, nlon)`, **north-row-0**, land = `-99`):
  - `apply_cells(cells: Iterable[(row,col)], value: float) -> None`
  - `apply_polygon(grid_spec, polygon_lonlat, value, *, mask_edit=False) -> None`
  - `erase(cells) -> None` (→ `0`)
  - `set_mask(cells, masked: bool) -> None` (→ `-99` or `0`)
  - `array -> np.ndarray`
- `rasterize_polygon(grid_spec, polygon_lonlat, mask, *, mask_edit=False) -> list[(row,col)]` — `matplotlib.path.Path.contains_points` over cell centers; excludes `-99` cells unless `mask_edit`. Center-in-polygon is the sole rule (no overlap-mode — YAGNI). A polygon containing no cell center → empty list (UI shows an info toast).
- `lonlat_to_cell(grid_spec, lon, lat) -> (row,col) | None` — maps a clicked basemap coordinate to a north-row-0 cell (for brush via `_map_click`); `None` if outside the grid.
- `to_csv_text(map_grid) -> str` / `from_csv_text(text, grid_spec) -> MapGrid` — **CRITICAL orientation:** the engine map-CSV file is SOUTH-row-0 (`movement_maps._load_csv_grid` does `grid_row = ny-1-csv_row_idx`; `load_mask`/`load_csv_overlay` `np.flipud`). The in-memory array is north-row-0, so `to_csv_text` must `np.flipud` (north-row-0 → south-row-0 file) and `from_csv_text` must `np.flipud` back. `;`-separated, `-99` land. `from_csv_text` validates dims == `(nlat, nlon)` (raises `ValueError`). The round-trip test asserts a written CSV, read by the ENGINE's `movement_maps._load_csv_grid`, reproduces the painted north-row-0 grid (NOT just self-round-trip — that would pass while flipped).
- `validate(map_grid, grid_spec) -> list[str]` — dim mismatch; for mask saves, mask cells inconsistent with the grid; for distribution/zone, WARN (not block) where a value is painted on a base-mask land (`-99`) cell (the engine treats those as absent regardless).
- `wire_map_into_config(config, map_type, rel_path, *, applicability=None) -> (new_config, summary)` — pure. Distribution → next free index across the WHOLE `map{N}` family (engine discovers maps via `movement.species.map{N}`) + the REAL keys from `applicability`: `movement.species.map{N}` (single species name — case-insensitively matched, `movement_maps.py:131`), `movement.file.map{N}=rel_path`, optional `movement.steps.map{N}` (`;`-sep season step indices), `movement.initialage.map{N}`/`movement.lastage.map{N}` (years, float), `movement.initialyear.map{N}`/`movement.lastyear.map{N}` (lastyear INCLUSIVE, `movement_maps.py:173-175`) or `movement.years.map{N}`. **One species per map** (the key is a single string, no list form). To apply one drawn file to N species, emit N `map{N}` blocks sharing `rel_path`. Mask → `grid.mask.file = rel_path`. Zone → no keys. Returns updated config + a human summary. (UI applies via `AppState`.)

### `ui/pages/map_builder.py` — page

- New nav entry "Map Builder" (navset_pill_list, per the UI architecture). On entry: read grid from `state.config` → `GridSpec` + base mask (`load_mask`). No grid → hint + disabled.
- **Start controls:** "New blank map" (`-99` mask, else `0`) | "Load existing map…" (select a `movement.file.map{N}`/`grid.mask.file` from the config → `from_csv_text`).
- **Tool mode:** Polygon-draw | Brush | Eraser | Mask-edit. **Paint value:** numeric (default `1`).
- **Map-type selector:** Distribution | Land mask | Generic zone — drives Save.
- **Distribution applicability form** (shown for Distribution type): species (required, **single** species per map — single string key), initial/last age, season steps, initial/last year (or years) — reuse `schema/movement.py` field renders. (Optional v1.x: multi-select species → emit N `map{N}` blocks sharing the file.)
- **Interactive map:** `build_grid_layers`-rendered cells over the basemap (view-state from grid bounds), colored by current value (empty transparent, value colored, land gray). **Draw** = MapLibre MapboxDraw enabled at runtime via `MapWidget.enable_draw(session, modes=["draw_polygon"])` (NOT a deck.gl layer prop) → drawn GeoJSON arrives reactively at `input.{id}_drawn_features()`. **Brush/eraser/mask** = `input.{id}_map_click()` (lon/lat on every click) → `lonlat_to_cell` → cell op. (`info_widget` is display-only — not used for read-back.)
- **Reactive state:** `reactive.Value[np.ndarray]`. `@reactive.event(input.{id}_drawn_features)` → `rasterize_polygon`→`apply_polygon`; `@reactive.event(input.{id}_map_click)` → `lonlat_to_cell` → `apply_cells`/`erase`/`set_mask` per tool mode; each updates the array → live re-render. Throttle/batch re-renders for responsiveness on larger grids (eec is bigger than Baltic's 40×50).
- **Save:** filename input → `to_csv_text` (flipped to south-row-0) → write to `state.config_dir/maps/<name>.csv` → `wire_map_into_config` (relative path `maps/<name>.csv`) → apply keys via `AppState` → reload. Toast on success.

## Data flow

`state.config` (grid bounds + mask) → `GridSpec` + base mask → init `MapGrid` (blank or loaded) → `reactive.Value` → [polygon-draw → `rasterize_polygon`→`apply_polygon`; brush → `apply_cells`; mask-edit → `set_mask`] → reactive array → deck.gl re-render → Save → `to_csv_text` → write file + `wire_map_into_config` + reload `state.config`.

## Error handling / edge cases

- No grid / malformed bounds → builder disabled, hint "Load a config with a defined grid first."
- Paint a value on a land (`-99`) cell → blocked unless Mask-edit mode (the explicit land/sea change path).
- Mask save → `validate` dims == `grid.nlon×nlat` (don't produce a mask inconsistent with the configured grid).
- Polygon entirely outside the grid → 0 cells matched → info toast, no-op.
- Distribution map saved with no species selected → block with a clear message (an unapplied map is a silent footgun).
- Overwriting an existing map file → confirm.
- **Map-file location + run-time resolution (RESOLVED by review — was the flagged risk):** demos are loaded into a WRITABLE per-session temp copy (`osmose/demo.py` copytrees `data/<config>` → `/tmp/osmose_demo_*/config/`), and `state.config_dir` points there — NOT the read-only repo. So write the CSV to `state.config_dir/maps/<name>.csv` and register the **relative** path `maps/<name>.csv` (never absolute; no `..` — `path_resolution.resolve_data_path` rejects `..`). This resolves for all three consumers: the in-UI overlay render (`grid.py` requires the file under `state.config_dir`), the Python engine (resolves `movement.file.map{N}` relative to `_osmose.config.dir` = `state.config_dir`), and the Java engine (`write_temp_config` copytrees the whole config dir — maps included — into the run temp dir before invoking the jar). The file MUST exist on disk under `state.config_dir` before a run (write_temp_config copies what's present; it doesn't fetch missing files).
- **`state.config_dir is None`** (config built from scratch / reset, never loaded from a demo) → the builder must create a session temp dir, `state.config_dir.set(it)`, and write there; or, if no writable dir can be established (`os.access` check fails), disable Save with a clear message.

## Testing strategy

- **Pure core (CI — the bulk):** `rasterize_polygon` (concave polygons, cells on polygon edges, Baltic-style bounds, polygon outside grid); `lonlat_to_cell` (incl. outside-grid → None); `apply_cells`/`erase`/`set_mask`; **`to_csv_text`→ENGINE `movement_maps._load_csv_grid` reproduces the painted north-row-0 grid** (asserts the flip is correct, not just self-round-trip) + `from_csv_text` round-trip; `validate` (dim mismatch, mask consistency, distribution-on-land warning); `GridSpec.cell_polygons()` matches `build_grid_layers` cell-for-cell. Property test (Hypothesis): the rasterized cell set equals exactly the cells whose center lies inside the polygon. `wire_map_into_config` — asserts the REAL keys: `movement.species.map{N}` (single string) + `movement.file.map{N}` + `movement.{initialage,lastage,initialyear,lastyear,steps,years}.map{N}` for distribution; next-free index across the whole `map{N}` family; `grid.mask.file` for mask; no keys for zone.
- **UI (no browser):** `import ui.pages.map_builder` clean; the save handler writes a CSV + applies the wiring keys (mirror existing ui page tests, e.g. `test_ui_*`).
- **deck.gl draw/pick event binding** (the one hard-to-unit-test seam) → covered by the e2e/visual Playwright harness + manual; all event→cell logic lives in the pure core, tested independently of the widget.
- Full suite green; ruff + pyright clean.

## Out of scope (future)

Spatial-fishing effort mechanics; undo/redo stack; grid-bounds authoring; importing a map from a raster/shapefile; multi-map batch tools.
