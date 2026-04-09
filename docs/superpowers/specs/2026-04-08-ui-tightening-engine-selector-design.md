# UI Tightening, Layer Control & Engine Selector Design

**Date:** 2026-04-08
**Status:** Approved

## Overview

Four coordinated UI changes: (1) tighten vertical/horizontal spacing across the top area, (2) add a layer control widget to the grid map, (3) introduce a global Java/Python engine selector in the header, and (4) create UI infrastructure for Python-only features (Ev-OSMOSE genetics, economic module, Python run config, diagnostics).

## 1. Top Bar & Spacing Tightening

Comprehensive reduction of spacing across the header, nav pills, and content gap.

### Header Bar (`osmose.css` — `.osmose-header`)
- Padding: `8px 16px` → `4px 12px`
- Internal flexbox gap: `16px` → `8px`
- Logo font size and badge: reduce proportionally
- Wave animation `::after` height: reduce

### Nav Pills (left sidebar)
- Pill item padding and font size: tighten
- Group heading margins: reduce
- Gap between nav header and first pill: reduce

### Content Gap
- `navset_pill_list` top padding: reduce
- `page_fillable` internal padding: reduce
- Card header padding within pages: reduce

**Target:** ~30-40% vertical space reduction in the top 200px of the app.

## 2. Layer Control Widget

The grid map already has a `layer_legend_widget()` with `show_checkbox=True` and layer IDs (`"grid-extent"`, `"grid-ocean"`, `"grid-land"`, `"grid-overlay"`, `"movement-{id}"`). This section ensures the existing layer control works correctly with shiny-deckgl 1.9.2 and covers any gaps.

### Existing State (already implemented)
- Layer IDs assigned in `build_grid_layers()` / `build_netcdf_grid_layers()` in `grid_helpers.py`
- Legend entries built dynamically at `grid.py:556-633` with `show_checkbox=True`
- `make_legend()` compat wrapper in `grid_helpers.py:1121` dispatches to `layer_legend_widget()`
- Passed via `widgets=` parameter of `MapWidget.update()` (deck.gl widget, not MapLibre control)

### What's Needed
- **Verify** existing legend checkboxes toggle layer visibility correctly with shiny-deckgl 1.9.2 (the widget handles toggles client-side — no server-side `set_layer_visibility()` call needed)
- **Ensure** `make_legend()` compat wrapper works with 1.9.2's `layer_legend_widget()` API (check for any parameter changes)
- **Test** that grid-extent can be toggled off/on independently of overlay layers

### Layers in Control
| Layer | ID | Always Present |
|-------|-----|----------------|
| Grid Extent | `"grid-extent"` | Yes |
| Ocean Cells | `"grid-ocean"` | Yes (regular grid) |
| Land Cells | `"grid-land"` | Yes (regular grid) |
| Active Overlay | `"grid-overlay"` | When overlay selected |
| Movement Maps | `"movement-{id}"` | When movement active |

### Files
- `ui/pages/grid.py`: verify/fix legend integration with 1.9.2
- `ui/pages/grid_helpers.py`: verify `make_legend()` compat wrapper

## 3. Java/Python Engine Selector

### Placement
Header bar, between the version badge and the action buttons (theme/about/help).

### Widget
Compact segmented toggle — two buttons `Java` / `Python`. Active state uses `var(--osm-accent)` (orange), inactive is muted. Styled inline with the header aesthetic.

### State
- New `engine_mode` field on `AppState` — values `"java"` or `"python"`, default `"java"`
- Reactive — all engine-dependent sections re-render on change
- Persisted in `localStorage` (like theme toggle), survives page refresh

### Global Effects
- **Java selected:** Python-only nav items (Genetics, Economic, Diagnostics) are disabled/greyed with tooltip "Requires Python engine"
- **Python selected:** Java-specific controls (JAR selector, JVM flags) are disabled on Run page's Java tab
- Run page: "Java" tab and "Python" tab — active engine's tab is default-selected, both accessible but inactive engine's tab controls are disabled

### Files
- `app.py`: header toggle widget, nav items with engine gating, localStorage JS
- `ui/state.py`: `AppState.engine_mode` reactive field
- `www/osmose.css`: toggle styling, `.nav-item-disabled` styling

## 4. Python-Only UI Infrastructure

All gated behind `state.engine_mode == "python"`. When Java is selected, these nav items appear greyed/disabled with a tooltip.

### 4a. Python Run Config — Run page, "Python" tab
- No JAR picker — Python engine runs in-process
- Fields: number of threads (Numba prange), output directory, verbosity level
- Run button triggers `PythonEngine.run()` directly
- Console output streams Python engine logs (same console widget, different source)
- **File:** `ui/pages/run.py` (add `navset_tab()` with Java/Python tabs)

### 4b. Ev-OSMOSE Genetics — new nav item "Genetics" under Configure
- Species-level genetics parameter form (trait heritability, mutation rates, selection pressure)
- Uses existing `render_field()` / `render_category()` param form infrastructure
- Config keys under `evolution.*` namespace (schema does not exist yet — page is a stub with descriptive placeholder UI until the Ev-OSMOSE engine module is implemented per `2026-04-05-ev-osmose-economic-design.md`)
- **File:** new `ui/pages/genetics.py`

### 4c. Economic Module — new nav item "Economic" under Configure
- Fleet economics parameters (cost structures, market prices, quota management)
- Same param form infrastructure
- Config keys under `economic.*` namespace (schema does not exist yet — page is a stub with descriptive placeholder UI until the economic engine module is implemented per `2026-04-05-ev-osmose-economic-design.md`)
- **File:** new `ui/pages/economic.py`

### 4d. Python Engine Diagnostics — new nav item "Diagnostics" under Execute
- After Python run completes, shows:
  - Timing breakdown per process (predation, movement, mortality, growth)
  - Numba JIT compilation status (first-run vs cached)
  - Memory usage profile
  - Comparison table: Python vs Java timing (if both run)
- Read-only dashboard, no config inputs
- **File:** new `ui/pages/diagnostics.py`

### Engine Nav Gate Pattern
Helper function `engine_nav_item(label, engine="python")` wraps nav items with conditional disable class + tooltip. CSS `.nav-item-disabled` applies greyed-out styling with cursor and opacity changes. Lives in `app.py` alongside the nav definition (single use site, no separate file needed).

## File Change Summary

| Change | Files |
|--------|-------|
| Top bar tightening | `www/osmose.css` |
| Engine selector widget | `app.py` (header), `ui/state.py` (AppState.engine_mode) |
| Engine selector JS/CSS | `www/osmose.css`, inline JS in `app.py` |
| Engine nav disable | `app.py` (nav items + helper function) |
| Layer control verify | `ui/pages/grid.py`, `ui/pages/grid_helpers.py` (verify with 1.9.2) |
| Python run tab | `ui/pages/run.py` |
| Ev-OSMOSE page | new `ui/pages/genetics.py` |
| Economic page | new `ui/pages/economic.py` |
| Diagnostics page | new `ui/pages/diagnostics.py` |
| Page registration | `app.py` (nav items + server wiring) |

**Total:** ~9 files touched, 3 new files (genetics.py, economic.py, diagnostics.py), no structural rewrites.

## Out of Scope
- Actual Ev-OSMOSE/Economic engine logic (has its own spec: `2026-04-05-ev-osmose-economic-design.md`)
- Python engine runner integration (already exists)
- shiny-deckgl upgrade to 1.9.2 (user-managed)
