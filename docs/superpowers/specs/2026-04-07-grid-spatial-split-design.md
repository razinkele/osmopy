# Split Grid & Maps into Grid Config + Spatial Results

**Date:** 2026-04-07
**Status:** Draft

## Problem

The current "Grid & Maps" tab serves two purposes: configuring the simulation grid (a pre-run concern) and visualizing spatial data (useful both pre- and post-run). After a simulation completes, users must navigate back to the Results tab's small "Spatial Distribution" sub-tab to see spatial outputs as Plotly heatmaps — there is no way to view run results on the real deck.gl map.

## Solution

Split "Grid & Maps" into two distinct navigation items:

1. **"Grid"** (Configure section) — grid definition and config-time spatial overlays
2. **"Spatial Results"** (Execute section) — run output visualization on deck.gl map + Plotly heatmap

## Navigation Structure

### Before

```
Configure: Setup | Grid & Maps | Forcing | Fishing | Movement
Execute:   Run | Results
```

### After

```
Configure: Setup | Grid | Forcing | Fishing | Movement
Execute:   Run | Results | Spatial Results
```

"Spatial Results" is **disabled** (greyed-out, non-interactive) until `state.output_dir` is set by a completed run. Once a run finishes, the pill activates.

## Grid Page (Configure)

**File:** `ui/pages/grid.py` (existing, minimal changes)

Retains all current functionality:

- Grid type selector (OriginalGrid / NcGrid)
- Grid coordinate and dimension fields
- NetCDF grid settings (file, variables)
- Overlay selector: config-time spatial files (LTL biomass, background species, fishing distribution, MPA maps)
- Movement Animation controls (species, speed, time step slider)
- deck.gl MapWidget rendering grid cells and overlays
- Collapsible panel behavior

**Changes:**
- Tab label renamed from "Grid & Maps" to "Grid"
- No logic or layout changes

## Spatial Results Page (Execute)

**File:** `ui/pages/spatial_results.py` (new)

### Layout

Same split-layout pattern as other pages:

```
+---------------------------+------------------------------------------+
| Controls (col 5)          | Map / Heatmap (col 7)                    |
|                           |                                          |
| Output variable selector  |  navset_card_tab:                        |
|   - Biomass               |    Tab "Map View":                       |
|   - Abundance             |      deck.gl MapWidget                   |
|   - Yield                 |      (polygon_layer, color gradient)     |
|   - Size                  |                                          |
|   - LTL                   |    Tab "Flat View":                      |
|                           |      Plotly imshow heatmap               |
| Species filter            |      (relocated from Results page)       |
|   - All / individual      |                                          |
|                           |                                          |
| Time step slider          |                                          |
|   - Play/Pause animate    |                                          |
|   - Range from NetCDF     |                                          |
|                           |                                          |
| Color scale info          |                                          |
|   - Min/max display       |                                          |
|                           |                                          |
| Collapsible panel         |                                          |
+---------------------------+------------------------------------------+
```

### Controls Sidebar

- **Output variable selector** (`spatial_result_type`): Dropdown listing available spatial output types. Choices populated dynamically via `OsmoseResults(output_dir).list_outputs()` filtered to `.nc` files. Maps filenames to labels (e.g., `osm_biomass.nc` -> "Biomass"). Only files that actually exist in the output directory appear.
- **Species filter** (`spatial_species`): Dropdown with "All" + individual species. For multi-species NetCDF files, filters to the selected variable. For single-species files, auto-selects.
- **Time step slider** (`spatial_time`): Integer slider from 0 to `n_time - 1`, with play/pause animation. Range set dynamically from the loaded NetCDF dataset's time dimension.
- **Color scale display**: Read-only text showing current min/max values of the displayed variable.

### Map View (Primary Tab)

A new `MapWidget` instance (`spatial_map`) using the same shiny_deckgl polygon_layer approach as the Grid page:

- Grid cell geometry derived from `state.config` (same `_read_grid_values()` pattern or equivalent read from the run's config)
- For NcGrid configs: read lat/lon from the NetCDF grid mask
- Fill color: continuous gradient mapped to the spatial variable value at each cell
  - Color palette: sequential (e.g., Viridis or similar ocean-themed gradient)
  - Cells with zero/NaN values rendered transparent
- Legend: gradient color bar with min/max labels (using `layer_legend_widget` or `deck_legend_control`)
- Tooltip: shows cell coordinates and value on hover
- Map style: respects dark/light theme toggle (CARTO_DARK / CARTO_POSITRON)
- View state: auto-zoomed to fit the grid extent

### Flat View (Secondary Tab)

The existing Plotly `imshow` heatmap currently in the Results page "Spatial Distribution" tab, relocated here:

- Same `make_spatial_map()` function
- Same time slider (shared `spatial_time` input)
- Plotly template: `osmose` / `osmose-light` based on theme

### Data Flow

```
Run completes
  → state.output_dir set by run_server
  → "Spatial Results" nav pill activates (conditional UI)
  → User clicks "Spatial Results"
  → spatial_results_server reads state.output_dir
  → Creates OsmoseResults(output_dir)
  → Calls list_outputs(), filters to .nc files
  → Populates output variable dropdown with discovered NC filenames
  → User selects variable + species
  → Loads xarray Dataset via OsmoseResults.read_netcdf(filename)
  → Extracts data variable names from Dataset for species filter
  → Slices by variable name and time step
  → Map View: builds polygon_layer with fill_color from values
  → Flat View: renders Plotly imshow from same data slice
```

### Disabled State

Shiny for Python's `nav_panel` does not support a native `disabled` parameter. Implementation uses a reactive effect that injects JS to toggle the pill's CSS class:

1. The nav panel is always present in the nav structure:
   ```python
   ui.nav_panel("Spatial Results", spatial_results_ui(), value="spatial_results")
   ```

2. A reactive effect in `spatial_results_server` (or app-level server) watches `state.output_dir` and runs `session.send_custom_message` or `ui.insert_ui` with a `<script>` tag that adds/removes a `disabled` CSS class on the `.nav-link[data-value='spatial_results']` element.

3. CSS in `www/osmose.css`:
   ```css
   .nav-link.osm-disabled {
       opacity: 0.4;
       pointer-events: none;
       cursor: not-allowed;
   }
   ```

4. On page load, the pill starts disabled. When `state.output_dir` changes from None to a path, the effect removes the class. If a new config is loaded (resetting output_dir to None), the class is re-added.

## Results Page Changes

**File:** `ui/pages/results.py` (modified)

Remove the "Spatial Distribution" `nav_panel` and its associated server code:

- Remove: `spatial_time_idx` slider UI
- Remove: `spatial_chart` output widget and render function
- Remove: `spatial_ds` reactive value and its loading logic in `_do_load_results()`
- Move: `make_spatial_map()` function to `grid_helpers.py` (imported by both results.py if needed and spatial_results.py)

Keep: Time Series, Diet Composition, Compare Runs tabs.

## app.py Changes

- Import `spatial_results_ui`, `spatial_results_server` from `ui.pages.spatial_results`
- Rename "Grid & Maps" to "Grid" in the nav panel
- Add "Spatial Results" nav panel after "Results" in the Execute section
- Call `spatial_results_server()` with state in the server function
- Add JS/CSS for disabled pill state management

## Shared Code

The following functions/patterns are used by both Grid and Spatial Results:

| Function | Current location | Shared approach |
|----------|-----------------|-----------------|
| `build_grid_layers()` | `grid_helpers.py` | Import from grid_helpers (no change) |
| `build_netcdf_grid_layers()` | `grid_helpers.py` | Import from grid_helpers (no change) |
| `load_netcdf_grid()` | `grid_helpers.py` | Import from grid_helpers (no change) |
| `_zoom_for_span()` | `grid_helpers.py` | Import from grid_helpers (no change) |
| `_make_legend()` | `grid.py` | Move to grid_helpers.py for sharing |
| `make_spatial_map()` | `results.py` | Move to grid_helpers.py (it takes xarray Dataset + var_name, produces Plotly figure) |
| MapWidget patterns | `grid.py` | Duplicated in spatial_results.py (separate widget instance, `spatial_map` ID) |

## Testing

### Unit Tests

- Test that `spatial_results_ui()` returns expected UI structure
- Test output variable detection from mock output directories
- Test color mapping from xarray values to RGBA

### E2E Tests (Playwright)

New file: `tests/test_e2e_spatial_results.py`

- Verify "Spatial Results" pill is disabled before running simulation
- Verify pill activates after a run completes (requires running a short simulation or mocking output_dir)
- Verify output variable selector populates with available types
- Verify map renders with polygon layers after selecting a variable
- Verify time slider updates the visualization
- Verify Flat View tab shows Plotly heatmap

### Existing Test Updates

- Update `test_e2e_grid_maps.py`: rename "Grid & Maps" to "Grid" in nav link selectors and docstrings
- Update `test_e2e_grid_overlay.py`: rename "Grid & Maps" to "Grid" in nav link selectors and docstrings
- `test_e2e_reactive.py` and `test_app_structure.py` do not reference "Grid & Maps" — no changes needed
- Remove spatial distribution tests from results test files (if any)

## Files Changed Summary

| File | Action |
|------|--------|
| `ui/pages/grid.py` | Rename tab label only |
| `ui/pages/spatial_results.py` | **New** — deck.gl map + Plotly heatmap for run outputs |
| `ui/pages/results.py` | Remove Spatial Distribution tab and server code |
| `ui/pages/grid_helpers.py` | Add `_make_legend()` (moved from grid.py) |
| `app.py` | Add spatial_results import, rename grid tab, add conditional nav pill |
| `www/osmose.css` | Add disabled pill styling |
| `tests/test_e2e_spatial_results.py` | **New** — e2e tests for the new page |
| `tests/test_e2e_grid_maps.py` | Update "Grid & Maps" → "Grid" references |
| `tests/test_e2e_grid_overlay.py` | Update "Grid & Maps" → "Grid" references |
| `tests/test_e2e_reactive.py` | No changes needed (doesn't reference "Grid & Maps") |
| `tests/test_app_structure.py` | No changes needed (doesn't reference "Grid & Maps") |
