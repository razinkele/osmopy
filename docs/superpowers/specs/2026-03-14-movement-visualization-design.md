# Movement Visualization Design

## Summary

Add animated movement map visualization to the Grid & Maps page. When the user selects "Movement Animation" from the existing overlay dropdown, species-specific distribution maps animate across time steps on the deck.gl map, showing seasonal migration patterns with distinct colored layers per map.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| UI entry point | Entry in existing overlay dropdown (Option C) | Minimal UI, uses existing pattern |
| Overlapping maps | Separate colored layers with legend (Option C) | Shows nurseries vs spawning at a glance |
| Playback speed | Configurable: 0.5x, 1x, 2x, 4x (Option B) | Flexibility for study vs overview |
| Technical approach | Hybrid server preload + client playback (Approach 3) | Good performance without custom JS |
| Partial updates | Skip `_map.update()` when active map set unchanged | Most steps have no visual change |
| Page scope | Grid & Maps only (Option A) | Avoids duplicating map widget |

## UI Layout

When "Movement Animation" is selected from the overlay dropdown in the Grid Type card, three controls appear inline below it:

1. **Species dropdown** — lists species that have movement maps defined, extracted from `movement.species.map{N}` config keys. Shows species names, not indices.
2. **Speed selector** — inline dropdown: `0.5x | 1x | 2x | 4x` (default 1x). Maps to `AnimationOptions` interval: 2000ms, 1000ms, 500ms, 250ms.
3. **Time step slider** — range 0 to `simulation.time.ndtPerYear - 1` (fallback: infer max step from cached maps, or default to 23), with Play/Pause/Loop via Shiny `AnimationOptions`. Label shows "Step N / Total".

These controls are hidden when any other overlay option is selected.

On the map, active movement maps render as separate colored polygon layers with a legend. Each map gets a distinct color and a label derived from the file name + age range.

## Config Key Format

**Important:** The OSMOSE config files use keys in the format `movement.species.map{N}`, `movement.file.map{N}`, `movement.steps.map{N}`, `movement.initialAge.map{N}`, `movement.lastAge.map{N}`. This differs from the schema definitions in `osmose/schema/movement.py` which use `movement.map{idx}.species`, etc. The `build_movement_cache` function scans **raw config dict keys** (not schema patterns), so it must use the actual config format: `movement.{field}.map{N}`.

## Data Flow

### Cache Build (on species selection)

1. Scan raw config dict for all keys matching `movement.species.map{N}` to find maps for the selected species.
2. For each matching map index N, read:
   - `movement.file.map{N}` — CSV file path. Skip if value is `None`, `""`, or `"null"`.
   - `movement.steps.map{N}` — active time steps. **Parsing:** split on `;`, strip whitespace, discard empty strings, convert to `set[int]`.
   - `movement.initialAge.map{N}` / `movement.lastAge.map{N}` — age range (floats, for legend label)
3. Pre-read all valid CSV files using `load_csv_overlay()` and store in a `reactive.Value` cache:

```python
{
    "map0": {
        "label": "Spawning",
        "steps": {0, 1, 2, ..., 11},
        "age_range": "1+ yr",
        "color": [220, 60, 60, 140],
        "cells": [...deck.gl polygon dicts...]
    },
    ...
}
```

4. Colors assigned per-species-subset (not globally), so each species's maps get colors 0..N from the palette.
5. Cache invalidated only on species change or config reload (`load_trigger`).

### Playback (on slider tick)

1. Compute active map IDs: `{id for id, m in cache.items() if step in m["steps"]}`
2. Compare with previous active set (stored in `reactive.Value(frozenset())`)
3. If identical — skip `_map.update()` call and return early. The server still processes the reactive effect but avoids the deck.gl roundtrip.
4. If changed — build polygon layers + legend entries, call `_map.update()`

## Color Palette

Eight distinct colors assigned stably at cache-build time per species subset. If a species has more than 8 maps, colors cycle (with a log warning noting potential ambiguity):

| Index | Color RGB | Name |
|-------|-----------|------|
| 0 | 30, 120, 200 | Blue |
| 1 | 220, 60, 60 | Red |
| 2 | 40, 180, 80 | Green |
| 3 | 240, 150, 30 | Orange |
| 4 | 160, 60, 200 | Purple |
| 5 | 30, 190, 200 | Cyan |
| 6 | 220, 100, 160 | Pink |
| 7 | 200, 200, 40 | Yellow |

Fill opacity: 140/255 (~55%) for overlap blending. No line strokes.

### Legend

Uses existing `deck_legend_control` with `show_checkbox=True`. Each entry: map label + age range (e.g., "Spawning (1+ yr)"). Checkboxes toggle individual maps on/off.

### Label Derivation

Algorithm to derive a human-readable label from the CSV filename:

1. Strip directory path and `.csv` extension
2. Split on the first underscore: `"6cod_spawning"` → `["6cod", "spawning"]`
3. Discard the first segment (numeric prefix + species name)
4. Join remaining segments, replace underscores with spaces, title case
5. If nothing remains after stripping (e.g., `"1Roussette_01"` → `"01"`), use `"Map {N}"` as fallback

Examples:
- `maps/6cod_spawning.csv` → "Spawning"
- `maps/6cod_1plus.csv` → "1plus"
- `maps/3tacaud_spawners_printemps.csv` → "Spawners Printemps"
- `maps/12Hareng_plus.csv` → "Plus"
- `maps/1Roussette_01.csv` → "Map 0" (fallback, "01" is not descriptive)

Regex: `r"^\d+[A-Za-z]+_(.+)\.csv$"` — capture group 1 is the label portion.

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| No movement maps in config | Show notification "No movement maps configured". Hide species/speed/slider controls. |
| No maps active at current step | Show base grid only, empty legend. Slider continues. |
| File value is `None`, `""`, or `"null"` | Skip that map entry, do not attempt file resolution. |
| Missing CSV file on disk | Log warning, skip that map. Other maps render. One-time notification. |
| `simulation.time.ndtPerYear` missing | Infer from max step value across cached maps + 1. Default to 24 if no maps loaded. |
| Config reload during animation | Invalidate cache, reset slider to 0, rebuild species list. Reset to first species if previous doesn't exist. |
| Examples without movement maps (Minimal, Bay of Biscay, Eec) | "Movement Animation" appears in dropdown but shows "no maps" notification when selected. Only Eec Full has maps. |
| Speed change during playback | Re-render slider with new `AnimationOptions` interval. Preserve current slider position by capturing value before re-render and restoring it. |

## Files Modified

| File | Changes |
|------|---------|
| `ui/pages/grid.py` | Add "Movement Animation" overlay choice. Add conditional species/speed/slider controls in `grid_overlay_selector`. Add movement cache `reactive.Value`. Modify `update_grid_map` to handle animation mode with partial updates. |
| `ui/pages/grid_helpers.py` | Add `build_movement_cache(cfg, config_dir, grid_params)` — scans config, pre-reads CSVs, assigns colors, returns cache dict. Add `MOVEMENT_PALETTE` constant. Add `derive_map_label(filename)` helper. |
| `www/osmose.css` | Style animation controls group (inline layout for speed selector, spacing for slider). |

No new files. No changes to `app.py`, `movement.py`, `state.py`, or schema files.

## Tests

New tests in `tests/test_grid_helpers.py`:

- `test_build_movement_cache_basic` — mock config with 3 maps, verify cache structure, labels, step sets
- `test_build_movement_cache_missing_file` — verify missing CSV skipped with warning
- `test_build_movement_cache_no_maps` — empty config returns empty dict
- `test_derive_map_label` — filename parsing edge cases
- `test_movement_step_filtering` — verify correct maps returned per time step
- `test_movement_cache_null_file` — verify maps with `"null"` file value are skipped
- `test_movement_cache_color_cycling` — verify palette cycles for species with 9+ maps
- `test_movement_steps_parsing` — verify semicolon splitting handles trailing separators and whitespace
