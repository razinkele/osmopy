# UI Compaction + CSV Map Viewer

> Date: 2026-04-07 (revised after code review)

## Summary

Two changes: (1) tighten the overall UI layout by removing the model info bar, slimming the header, and compressing nav sidebar spacing; (2) add a standalone CSV Map Viewer tab under MANAGE for browsing and previewing spatial files from the loaded config.

## 1. Header Compaction

### Current state

The app has a two-row header:
- **Row 1** (`.osmose-header`, ~53px): logo, version badge, theme toggle, About, Help
- **Row 2** (`#config_header`, ~40px): "EEC Full 14 species • 1095 parameters [modified]"

Total vertical overhead: ~93px before content starts.

### Changes

**Remove the second row entirely.** Move model info into the main header as right-aligned plain text, before the About/Help buttons.

Layout after change:
```
[OSMOPY | Marine Ecosystem Simulator] [v0.6.0]    EEC Full  14 species • 1095 params  modified    [About] [Help]
```

Specific edits:

| Element | Current | New |
|---------|---------|-----|
| `.osmose-header` padding | `14px 24px` | `8px 16px` |
| `.osmose-logo` font-size | `1.4rem` | `1.1rem` |
| `.osmose-logo .subtitle` font-size | `0.8rem` | `0.7rem` |
| `config_header` output_ui | Separate row in `app.py` layout (line 182) | Removed from top-level; moved inside `.osmose-header-actions` div |
| Model info | Rendered by `@render.ui def config_header()` | Same renderer, but `output_ui("config_header")` placed inside `.osmose-header-actions` before theme/about/help buttons |

**Implementation approach:** Keep `config_header` as a `@render.ui` but move the `output_ui("config_header")` from its own row (app.py line 182) into the `.osmose-header-actions` div (app.py lines 155-178), before the theme toggle button. Remove `STYLE_CONFIG_HEADER` from `ui/styles.py`, and also remove its import at `app.py:12` (`from ui.styles import STYLE_CONFIG_HEADER`) and its usage at `app.py:341` (`style=STYLE_CONFIG_HEADER`). The inline flex layout is no longer needed since the element is now a child of `.osmose-header-actions` which already has `display: flex`.

Target header height: ~36px (down from 53px + 40px = 93px). Net saving: ~57px.

### Model info display format

```
config_name  N species • M params  [modified]
```

- Config name: `color: var(--osm-accent); font-weight: 600; font-size: 0.78rem`
- Stats: `color: var(--osm-text-muted); font-size: 0.7rem; margin-left: 6px`
- Modified: `color: #e67e22; font-size: 0.65rem; font-style: italic; margin-left: 4px`
- Hidden when no config loaded (empty div)

**Responsive overflow:** Config name span gets `overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 180px` to prevent long names (e.g., "Bay of Biscay Extended 24 Species") from overflowing. At `@media (max-width: 768px)`, hide the model info entirely (same as `.osmose-badge` is already hidden).

**Light mode note:** The existing `#5a6a7a` for stats text has low contrast (~2.8:1) against light backgrounds. Using `var(--osm-text-muted)` inherits this pre-existing issue. No change in behavior — fixing light-mode contrast is out of scope for this spec.

## 2. Nav Sidebar Tightening

### Current state

Nav pills use Bootstrap 5 padding overridden at `osmose.css:361` with `padding: 8px 14px !important`. Section labels (`.osmose-section-label`) have margin and `::before` divider lines.

### Changes

Override in `www/osmose.css` (must use `!important` to beat existing rule at line 361):

```css
/* Tighter nav pills */
.nav-pills .nav-link {
    padding: 4px 12px !important;  /* was 8px 14px !important at line 361 */
    font-size: 0.82rem;
}

/* Tighter section label spacing */
.osmose-section-label {
    margin-top: 8px;          /* was ~12-16px */
    margin-bottom: 2px;       /* was ~4-8px */
    font-size: 0.6rem;        /* was 0.65rem */
}
```

The existing responsive override at `osmose.css:1132` (`@media max-width: 768px`) uses `padding: 6px 10px !important` — leave it unchanged (already tighter than the new desktop value).

Keep the same visual hierarchy — just compress vertical rhythm.

## 3. CSV Map Viewer Page

### Location

New tab **"Map Viewer"** in the **MANAGE** section of `navset_pill_list`, after "Advanced" (app.py line 213).

### Layout

Split-panel (`osm-split-layout`), same pattern as Grid page:

```
┌─────────────────────┬──────────────────────────────┐
│  File List (5/12)   │  Map Preview (7/12)          │
│                     │                              │
│  MOVEMENT MAPS (32) │  cod: Nurseries              │
│  ▸ cod              │  maps/6cod_nurseries.csv     │
│    · Nurseries  ◄── │  22×45 · Age 0+ yr · 24 steps│
│    · 1Plus          │                              │
│    · Spawning       │  ┌────────────────────────┐  │
│  ▸ sole             │  │                        │  │
│    · Spawning       │  │    deck.gl map          │  │
│    · Nurseries      │  │    preview              │  │
│    · 1Plus          │  │                        │  │
│  ▸ whiting          │  └────────────────────────┘  │
│    ...              │                              │
│                     │  Ocean: 464  Land: 530       │
│  FISHING (1)        │  Values: [0.0, 1.0]          │
│    Fishing Distrib  │                              │
│                     │                              │
│  OTHER (2)          │                              │
│    LTL Biomass (NC) │                              │
│    Background Sp.   │                              │
└─────────────────────┴──────────────────────────────┘
```

**Empty state:** When no config is loaded, show a centered hint: "Load a configuration to browse spatial files." Same pattern as Grid page's `grid_hint()` (grid.py:461-468).

### File discovery — shared helper extraction

Currently, file discovery logic is inline inside `grid_overlay_selector()` (grid.py:150-252), which is a `@render.ui` reactive — it cannot be called directly by the Map Viewer.

**Extract a shared pure function** into `grid_helpers.py`:

```python
def discover_spatial_files(
    cfg: dict[str, str],
    cfg_dir: Path | None,
) -> dict[str, dict[str, list[dict]]]:
    """Discover all spatial files (CSV/NC) from an OSMOSE config.

    Returns a categorized dict:
    {
        "movement": {"cod": [{"path": Path, "label": "Nurseries", "age": "0+ yr", "steps": 24}, ...], ...},
        "fishing": [{"path": Path, "label": "Fishing Distribution"}, ...],
        "other": [{"path": Path, "label": "LTL Biomass"}, ...],
    }
    """
```

This function encapsulates the 4-pass discovery logic currently in `grid_overlay_selector`:
1. General pass: scan config for `.csv`/`.nc` refs, skip non-spatial keys, deduplicate by resolved path
2. Movement pass: `movement.file.mapN` keys, grouped by `movement.species.mapN`
3. Fishing pass: `fisheries.movement.file.mapN` keys
4. MPA directory scan

Both `grid_overlay_selector` and the Map Viewer's file list call this function. Note: `grid_overlay_selector` needs a flattening step to convert the categorized return value into its existing `{resolved_path: label}` flat dict format, plus inject the UI-specific sentinel entries (`"grid_extent"` and `"__movement_animation__"`) which are not part of the discovery function.

**Also move `_overlay_label`** from `grid.py` (line 63) to `grid_helpers.py` — it's a pure function with no Shiny dependencies, and now both pages need it. Update `grid.py` to import it from `grid_helpers`.

### File list panel

- Collapsible category groups (movement species are sub-groups within Movement Maps)
- Each entry shows: human-readable label (from `derive_map_label` or `_overlay_label`)
- Selected entry highlighted with accent color
- Show file count per category in header: "Movement Maps (32)"

### Map preview panel

- `MapWidget` instance with ID `"map_viewer_map"` (separate from Grid's `"grid_map"` and Spatial Results' `"spatial_map"`)
- On file selection:
  - CSV files: `load_csv_overlay()` with `nc_data` from grid config (if NcGrid)
  - NC files: `load_netcdf_overlay()` with variable/time controls
- Base layers: ocean/land grid cells (same as Grid page)
- Overlay: selected file data with same color palettes (amber for CSV, blue-cyan-yellow for NC)
- Metadata bar below map: filename, dimensions, value range, species, age range, time steps
- For NC files with time dimension: time step slider appears
- Guard `config_dir is None` before calling overlay loaders (return empty map with hint)

### Reactive flow

```
input.map_viewer_file (select from list)
    → load selected file via load_csv_overlay / load_netcdf_overlay
    → build layers (grid base + overlay)
    → _viewer_map.update(session, layers=..., view_state=...)
```

Grid data (`nc_data` or bounding box) read from `state.config` with `reactive.isolate()` — same pattern as Grid page.

### File structure

Single new file: `ui/pages/map_viewer.py`
- `map_viewer_ui()` — returns the split-panel layout
- `map_viewer_server(input, output, session, state)` — reactive effects for file list, map preview

Register in `app.py`:
- Add `ui.nav_panel("Map Viewer", map_viewer_ui(), value="map_viewer")` under MANAGE section
- Call `map_viewer_server(input, output, session, state)` in `server()`
- Add `'map_viewer'` to `pageIds` JS array (app.py line 119) for collapsible panel state restoration

### Reused infrastructure

From `grid_helpers.py` (existing):
- `load_csv_overlay`, `load_netcdf_overlay`, `load_netcdf_grid`, `load_mask`
- `build_grid_layers`, `build_netcdf_grid_layers`
- `list_nc_overlay_variables`, `derive_map_label`, `make_legend`, `_read_csv_auto_sep`
- `_zoom_for_span`, `_safe_resolve`

From `grid_helpers.py` (new — extracted):
- `discover_spatial_files()` — shared file discovery
- `_overlay_label()` — moved from `grid.py`

From `shiny_deckgl`:
- `MapWidget`, `polygon_layer`, `CARTO_POSITRON`, `CARTO_DARK`, zoom/compass/fullscreen/scale widgets

From `ui/state.py`: `get_theme_mode`
From `ui/components/collapsible.py`: `collapsible_card_header`, `expand_tab`

**Note:** `map_viewer.py` must NOT import `ui.charts` (would shadow `shiny.ui` — see CLAUDE.md).

## 4. Files Modified

| File | Change |
|------|--------|
| `app.py` | Move `output_ui("config_header")` into `.osmose-header-actions` div; remove from top-level layout; remove `STYLE_CONFIG_HEADER` import (line 12) and usage (line 341); add Map Viewer nav_panel + server call; add `'map_viewer'` to `pageIds` JS array (line 119) |
| `www/osmose.css` | Slim header padding/font-size; tighter nav pill padding (with `!important`); section label compression; responsive rule to hide model info at narrow viewports; config name `text-overflow: ellipsis` |
| `ui/pages/map_viewer.py` | **New file** — Map Viewer page UI + server |
| `ui/pages/grid_helpers.py` | Add `discover_spatial_files()` function; receive `_overlay_label()` moved from grid.py |
| `ui/pages/grid.py` | Remove `_overlay_label()` definition; import from `grid_helpers` (used at line 63 definition, line 201 in selector, AND line 736 in CSV overlay label); refactor `grid_overlay_selector` to call `discover_spatial_files()` then flatten result + inject sentinel entries |
| `ui/styles.py` | Remove `STYLE_CONFIG_HEADER` (no longer used) |
| `tests/test_overlay_display.py` | Update `_overlay_label` import path in OD7 tests |

## 5. Out of Scope

- Editing map files from the viewer (read-only)
- Uploading new CSV maps
- Comparing two maps side-by-side (future feature)
- Map viewer for non-spatial config files (accessibility matrices, season CSVs)
- Fixing light-mode contrast for stats text (pre-existing issue)
