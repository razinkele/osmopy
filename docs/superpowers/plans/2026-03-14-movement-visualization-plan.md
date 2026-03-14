# Movement Visualization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add animated movement map visualization to the Grid & Maps page, showing species seasonal migration patterns with distinct colored layers on the deck.gl map.

**Architecture:** A "Movement Animation" entry in the overlay dropdown reveals species/speed/slider controls. On species selection, all movement CSV maps are pre-loaded into a reactive cache. On each slider tick, only the active map set is compared and pushed to deck.gl if changed.

**Tech Stack:** Python/Shiny, shiny_deckgl (deck.gl + MapLibre), existing `load_csv_overlay()` helper, Shiny `AnimationOptions` for playback control.

**Spec:** `docs/superpowers/specs/2026-03-14-movement-visualization-design.md`

---

## Chunk 1: Helper Functions & Tests

### Task 1: Add `derive_map_label` helper and tests

**Files:**
- Modify: `ui/pages/grid_helpers.py` (append new function)
- Modify: `tests/test_grid_helpers.py` (append new tests)

- [ ] **Step 1: Write failing tests for `derive_map_label`**

Add to `tests/test_grid_helpers.py`:

```python
import pytest


def test_derive_map_label_spawning():
    from ui.pages.grid_helpers import derive_map_label

    assert derive_map_label("maps/6cod_spawning.csv", 17) == "Spawning"


def test_derive_map_label_multiword():
    from ui.pages.grid_helpers import derive_map_label

    assert derive_map_label("maps/3tacaud_spawners_printemps.csv", 5) == "Spawners Printemps"


def test_derive_map_label_numeric_fallback():
    from ui.pages.grid_helpers import derive_map_label

    # "01" is purely numeric → fallback to "Map 0"
    assert derive_map_label("maps/1Roussette_01.csv", 0) == "Map 0"


def test_derive_map_label_no_underscore():
    from ui.pages.grid_helpers import derive_map_label

    # No underscore → fallback
    assert derive_map_label("maps/empty.csv", 3) == "Map 3"


def test_derive_map_label_1plus():
    from ui.pages.grid_helpers import derive_map_label

    assert derive_map_label("maps/6cod_1plus.csv", 16) == "1Plus"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_grid_helpers.py::test_derive_map_label_spawning -v`
Expected: FAIL with `ImportError` or `cannot import name 'derive_map_label'`

- [ ] **Step 3: Implement `derive_map_label`**

Add to the end of `ui/pages/grid_helpers.py`:

```python
import re

_LABEL_RE = re.compile(r"^\d+[A-Za-z]+_(.+)$")


def derive_map_label(filename: str, map_index: int) -> str:
    """Derive a human-readable label from an OSMOSE movement map filename.

    Algorithm: strip path + extension, split on first underscore to remove
    the numeric-prefix+species segment, title-case the remainder.
    Falls back to "Map {index}" if the result is purely numeric or empty.
    """
    stem = Path(filename).stem  # "6cod_spawning"
    m = _LABEL_RE.match(stem)
    if not m:
        return f"Map {map_index}"
    label_part = m.group(1).replace("_", " ").title()
    # Purely numeric labels (e.g. "01") are not descriptive
    if label_part.strip().isdigit():
        return f"Map {map_index}"
    return label_part
```

- [ ] **Step 4: Run all `derive_map_label` tests**

Run: `.venv/bin/python -m pytest tests/test_grid_helpers.py -k derive_map_label -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add ui/pages/grid_helpers.py tests/test_grid_helpers.py
git commit -m "feat: add derive_map_label helper for movement map filenames"
```

---

### Task 2: Add `parse_movement_steps` helper and tests

**Files:**
- Modify: `ui/pages/grid_helpers.py` (append)
- Modify: `tests/test_grid_helpers.py` (append)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_grid_helpers.py`:

```python
def test_parse_movement_steps_basic():
    from ui.pages.grid_helpers import parse_movement_steps

    assert parse_movement_steps("0;1;2;3") == {0, 1, 2, 3}


def test_parse_movement_steps_trailing_semicolon():
    from ui.pages.grid_helpers import parse_movement_steps

    assert parse_movement_steps("6;7;8;") == {6, 7, 8}


def test_parse_movement_steps_whitespace():
    from ui.pages.grid_helpers import parse_movement_steps

    assert parse_movement_steps(" 0 ; 1 ; 2 ") == {0, 1, 2}


def test_parse_movement_steps_empty():
    from ui.pages.grid_helpers import parse_movement_steps

    assert parse_movement_steps("") == set()


def test_parse_movement_steps_none():
    from ui.pages.grid_helpers import parse_movement_steps

    assert parse_movement_steps(None) == set()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_grid_helpers.py::test_parse_movement_steps_basic -v`
Expected: FAIL

- [ ] **Step 3: Implement `parse_movement_steps`**

Add to `ui/pages/grid_helpers.py`:

```python
def parse_movement_steps(raw: str | None) -> set[int]:
    """Parse a semicolon-separated list of time step indices into a set."""
    if not raw:
        return set()
    steps = set()
    for part in raw.split(";"):
        part = part.strip()
        if part:
            try:
                steps.add(int(part))
            except ValueError:
                pass
    return steps
```

- [ ] **Step 4: Run all `parse_movement_steps` tests**

Run: `.venv/bin/python -m pytest tests/test_grid_helpers.py -k parse_movement_steps -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add ui/pages/grid_helpers.py tests/test_grid_helpers.py
git commit -m "feat: add parse_movement_steps helper"
```

---

### Task 3: Add `MOVEMENT_PALETTE` constant and `build_movement_cache` function with tests

**Files:**
- Modify: `ui/pages/grid_helpers.py` (append)
- Modify: `tests/test_grid_helpers.py` (append)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_grid_helpers.py`:

```python
def test_build_movement_cache_basic(tmp_path):
    from ui.pages.grid_helpers import build_movement_cache

    # Create two CSV map files (3x3 grid)
    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / "1sp_nurseries.csv").write_text("0;1;0\n1;0;1\n0;1;0\n")
    (tmp_path / "maps" / "1sp_spawning.csv").write_text("1;0;1\n0;1;0\n1;0;1\n")

    cfg = {
        "movement.species.map0": "speciesA",
        "movement.file.map0": "maps/1sp_nurseries.csv",
        "movement.steps.map0": "0;1;2;3",
        "movement.initialAge.map0": "0",
        "movement.lastAge.map0": "1",
        "movement.species.map1": "speciesA",
        "movement.file.map1": "maps/1sp_spawning.csv",
        "movement.steps.map1": "4;5;6;7",
        "movement.initialAge.map1": "1",
        "movement.lastAge.map1": "5",
    }
    grid_params = (48.0, -6.0, 43.0, -1.0, 3, 3)  # ul_lat, ul_lon, lr_lat, lr_lon, nx, ny

    cache = build_movement_cache(cfg, tmp_path, grid_params, species="speciesA")
    assert len(cache) == 2
    assert "map0" in cache
    assert "map1" in cache
    assert cache["map0"]["steps"] == {0, 1, 2, 3}
    assert cache["map1"]["steps"] == {4, 5, 6, 7}
    assert cache["map0"]["label"] == "Nurseries"
    assert cache["map1"]["label"] == "Spawning"
    assert cache["map0"]["age_range"] == "0-1 yr"
    assert cache["map1"]["age_range"] == "1-5 yr"
    assert cache["map0"]["cells"] is not None
    assert cache["map1"]["cells"] is not None
    # Different colors assigned
    assert cache["map0"]["color"] != cache["map1"]["color"]


def test_build_movement_cache_no_maps():
    from ui.pages.grid_helpers import build_movement_cache

    cfg = {"simulation.nspecies": "3"}
    cache = build_movement_cache(cfg, None, (0, 0, 0, 0, 10, 10), species="cod")
    assert cache == {}


def test_build_movement_cache_missing_file(tmp_path):
    from ui.pages.grid_helpers import build_movement_cache

    cfg = {
        "movement.species.map0": "cod",
        "movement.file.map0": "maps/nonexistent.csv",
        "movement.steps.map0": "0;1",
        "movement.initialAge.map0": "0",
        "movement.lastAge.map0": "2",
    }
    cache = build_movement_cache(cfg, tmp_path, (48.0, -6.0, 43.0, -1.0, 3, 3), species="cod")
    assert cache == {}


def test_build_movement_cache_null_file(tmp_path):
    from ui.pages.grid_helpers import build_movement_cache

    cfg = {
        "movement.species.map0": "cod",
        "movement.file.map0": "null",
        "movement.steps.map0": "0;1",
        "movement.initialAge.map0": "0",
        "movement.lastAge.map0": "2",
    }
    cache = build_movement_cache(cfg, tmp_path, (48.0, -6.0, 43.0, -1.0, 3, 3), species="cod")
    assert cache == {}


def test_build_movement_cache_color_cycling(tmp_path):
    from ui.pages.grid_helpers import build_movement_cache, MOVEMENT_PALETTE

    # Create 9 maps (more than palette size of 8)
    (tmp_path / "maps").mkdir()
    cfg = {}
    for i in range(9):
        fname = f"maps/1sp_map{i}.csv"
        (tmp_path / fname).write_text("0;1;0\n1;0;1\n0;1;0\n")
        cfg[f"movement.species.map{i}"] = "cod"
        cfg[f"movement.file.map{i}"] = fname
        cfg[f"movement.steps.map{i}"] = str(i)
        cfg[f"movement.initialAge.map{i}"] = "0"
        cfg[f"movement.lastAge.map{i}"] = "10"

    cache = build_movement_cache(cfg, tmp_path, (48.0, -6.0, 43.0, -1.0, 3, 3), species="cod")
    assert len(cache) == 9
    # 9th map should cycle back to first palette color
    assert cache["map0"]["color"][:3] == cache["map8"]["color"][:3]


def test_build_movement_cache_filters_species(tmp_path):
    from ui.pages.grid_helpers import build_movement_cache

    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / "a.csv").write_text("0;1;0\n1;0;1\n0;1;0\n")
    (tmp_path / "maps" / "b.csv").write_text("1;0;1\n0;1;0\n1;0;1\n")

    cfg = {
        "movement.species.map0": "cod",
        "movement.file.map0": "maps/a.csv",
        "movement.steps.map0": "0;1",
        "movement.initialAge.map0": "0",
        "movement.lastAge.map0": "2",
        "movement.species.map1": "herring",
        "movement.file.map1": "maps/b.csv",
        "movement.steps.map1": "2;3",
        "movement.initialAge.map1": "0",
        "movement.lastAge.map1": "5",
    }
    cache = build_movement_cache(cfg, tmp_path, (48.0, -6.0, 43.0, -1.0, 3, 3), species="cod")
    assert len(cache) == 1
    assert "map0" in cache
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_grid_helpers.py::test_build_movement_cache_basic -v`
Expected: FAIL

- [ ] **Step 3: Implement `MOVEMENT_PALETTE` and `build_movement_cache`**

Add to `ui/pages/grid_helpers.py`:

```python
MOVEMENT_PALETTE: list[list[int]] = [
    [30, 120, 200, 140],   # Blue
    [220, 60, 60, 140],    # Red
    [40, 180, 80, 140],    # Green
    [240, 150, 30, 140],   # Orange
    [160, 60, 200, 140],   # Purple
    [30, 190, 200, 140],   # Cyan
    [220, 100, 160, 140],  # Pink
    [200, 200, 40, 140],   # Yellow
]


def _format_age_range(min_age: str | None, max_age: str | None) -> str:
    """Format age range for legend labels."""
    try:
        lo = float(min_age) if min_age else 0
        hi = float(max_age) if max_age else None
    except (ValueError, TypeError):
        return ""
    if hi is None:
        return f"{lo:.0f}+ yr"
    return f"{lo:.0f}-{hi:.0f} yr"


def build_movement_cache(
    cfg: dict[str, str],
    config_dir: Path | None,
    grid_params: tuple[float, float, float, float, int, int],
    species: str,
) -> dict[str, dict]:
    """Pre-read all movement maps for a species and return a cache dict.

    Parameters
    ----------
    cfg
        Raw config dict (key → value strings).
    config_dir
        Directory containing the config files (for resolving relative paths).
    grid_params
        Tuple of (ul_lat, ul_lon, lr_lat, lr_lon, nx, ny) for grid bounds.
    species
        Species name to filter maps for.

    Returns
    -------
    dict
        Map ID → {"label", "steps", "age_range", "color", "cells"} for each valid map.
    """
    ul_lat, ul_lon, lr_lat, lr_lon, nx, ny = grid_params

    # Find all map indices for this species
    map_indices: list[str] = []
    for key, val in cfg.items():
        if key.startswith("movement.species.map") and val == species:
            idx = key[len("movement.species.map"):]
            map_indices.append(idx)

    if map_indices and len(map_indices) > len(MOVEMENT_PALETTE):
        _log.warning(
            "Species %s has %d maps but palette has %d colors; colors will cycle",
            species, len(map_indices), len(MOVEMENT_PALETTE),
        )

    cache: dict[str, dict] = {}
    color_idx = 0
    for idx in sorted(map_indices, key=lambda x: int(x) if x.isdigit() else 0):
        file_val = cfg.get(f"movement.file.map{idx}", "")
        if not file_val or file_val in ("null", "None"):
            continue

        # Resolve file path
        if config_dir:
            file_path = (config_dir / file_val).resolve()
            if not file_path.is_relative_to(config_dir.resolve()):
                _log.warning("Path traversal in movement map: %s", file_val)
                continue
            if not file_path.exists():
                _log.warning("Movement map file not found: %s", file_val)
                continue
        else:
            continue

        steps = parse_movement_steps(cfg.get(f"movement.steps.map{idx}"))
        if not steps:
            continue

        cells = load_csv_overlay(file_path, ul_lat, ul_lon, lr_lat, lr_lon, nx, ny)
        if not cells:
            continue

        label = derive_map_label(file_val, int(idx) if idx.isdigit() else 0)
        age_range = _format_age_range(
            cfg.get(f"movement.initialAge.map{idx}"),
            cfg.get(f"movement.lastAge.map{idx}"),
        )

        cache[f"map{idx}"] = {
            "label": label,
            "steps": steps,
            "age_range": age_range,
            "color": list(MOVEMENT_PALETTE[color_idx % len(MOVEMENT_PALETTE)]),
            "cells": cells,
        }
        color_idx += 1

    return cache
```

- [ ] **Step 4: Run all `build_movement_cache` tests**

Run: `.venv/bin/python -m pytest tests/test_grid_helpers.py -k build_movement_cache -v`
Expected: 6 passed

- [ ] **Step 5: Run full test suite to check no regressions**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: all pass

- [ ] **Step 6: Lint**

Run: `.venv/bin/ruff check ui/pages/grid_helpers.py tests/test_grid_helpers.py`
Expected: All checks passed

- [ ] **Step 7: Commit**

```bash
git add ui/pages/grid_helpers.py tests/test_grid_helpers.py
git commit -m "feat: add build_movement_cache and MOVEMENT_PALETTE for movement animation"
```

---

### Task 4: Add `list_movement_species` helper and test

**Files:**
- Modify: `ui/pages/grid_helpers.py` (append)
- Modify: `tests/test_grid_helpers.py` (append)

- [ ] **Step 1: Write failing test**

Add to `tests/test_grid_helpers.py`:

```python
def test_list_movement_species():
    from ui.pages.grid_helpers import list_movement_species

    cfg = {
        "movement.species.map0": "cod",
        "movement.species.map1": "cod",
        "movement.species.map2": "herring",
        "movement.species.map3": "sole",
    }
    result = list_movement_species(cfg)
    assert result == ["cod", "herring", "sole"]


def test_list_movement_species_empty():
    from ui.pages.grid_helpers import list_movement_species

    assert list_movement_species({}) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_grid_helpers.py::test_list_movement_species -v`
Expected: FAIL

- [ ] **Step 3: Implement**

Add to `ui/pages/grid_helpers.py`:

```python
def list_movement_species(cfg: dict[str, str]) -> list[str]:
    """Return sorted list of unique species names that have movement maps defined."""
    species: set[str] = set()
    for key, val in cfg.items():
        if key.startswith("movement.species.map") and val:
            species.add(val)
    return sorted(species)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_grid_helpers.py -k list_movement_species -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add ui/pages/grid_helpers.py tests/test_grid_helpers.py
git commit -m "feat: add list_movement_species helper"
```

---

## Chunk 2: UI Controls & Animation Logic

### Task 5: Add "Movement Animation" to overlay dropdown and conditional controls

**Files:**
- Modify: `ui/pages/grid.py` — `grid_overlay_selector` render function and imports

- [ ] **Step 1: Update imports in `grid.py`**

Add `list_movement_species` and `build_movement_cache` to the import block in `ui/pages/grid.py`:

```python
from ui.pages.grid_helpers import (
    build_grid_layers,
    build_movement_cache,
    build_netcdf_grid_layers,
    list_movement_species,
    load_csv_overlay,
    load_mask,
    load_netcdf_grid,
    load_netcdf_overlay,
)
```

- [ ] **Step 2: Rewrite `grid_overlay_selector` to include movement animation entry and conditional controls**

Replace the existing `grid_overlay_selector` render function in `grid_server` with:

```python
    @render.ui
    def grid_overlay_selector():
        state.load_trigger.get()
        with reactive.isolate():
            cfg = state.config.get()
        choices: dict[str, str] = {"grid_extent": "Grid extent"}
        skip_prefixes = ("grid.", "osmose.configuration.", "simulation.restart")

        for key, val in sorted(cfg.items()):
            if not val or not isinstance(val, str):
                continue
            if not (val.endswith(".nc") or val.endswith(".csv")):
                continue
            if any(key.startswith(p) for p in skip_prefixes):
                continue
            label = key.replace(".", " ").replace("_", " ").title()
            choices[key] = label

        # Add Movement Animation entry
        choices["__movement_animation__"] = "Movement Animation"

        if len(choices) <= 1:
            return ui.div()

        elements = [
            ui.input_select(
                "grid_overlay", "Overlay data", choices=choices, selected="grid_extent"
            ),
        ]

        # Show animation controls only when Movement Animation is selected
        try:
            overlay_val = input.grid_overlay()
        except Exception:
            overlay_val = "grid_extent"

        if overlay_val == "__movement_animation__":
            species_list = list_movement_species(cfg)
            if species_list:
                species_choices = {s: s for s in species_list}
                speed_choices = {"2000": "0.5x", "1000": "1x", "500": "2x", "250": "4x"}

                # Read nsteps from config with fallback
                try:
                    nsteps = int(float(cfg.get("simulation.time.ndtPerYear", "24") or "24"))
                except (ValueError, TypeError):
                    nsteps = 24

                # Read current speed for interval
                try:
                    interval = int(input.movement_speed())
                except Exception:
                    interval = 1000

                # Preserve current slider position on re-render (e.g. speed change)
                try:
                    current_step = input.movement_step()
                except Exception:
                    current_step = 0

                elements.extend([
                    ui.input_select(
                        "movement_species", "Species",
                        choices=species_choices, selected=species_list[0],
                    ),
                    ui.input_select(
                        "movement_speed", "Speed",
                        choices=speed_choices, selected=str(interval),
                    ),
                    ui.input_slider(
                        "movement_step", "Time step",
                        min=0, max=nsteps - 1, value=current_step, step=1,
                        animate=ui.AnimationOptions(
                            interval=interval, loop=True,
                            play_button="Play", pause_button="Pause",
                        ),
                    ),
                ])
            else:
                elements.append(
                    ui.p(
                        "No movement maps configured. Define maps in the Movement tab.",
                        style="color: var(--osm-text-muted); font-size: 12px; margin-top: 8px;",
                    )
                )

        return ui.div(*elements, class_="osm-movement-controls")
```

- [ ] **Step 3: Verify the app loads without errors**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: all pass

- [ ] **Step 4: Lint**

Run: `.venv/bin/ruff check ui/pages/grid.py`
Expected: All checks passed

- [ ] **Step 5: Commit**

```bash
git add ui/pages/grid.py
git commit -m "feat: add Movement Animation entry and conditional controls to overlay dropdown"
```

---

### Task 6: Add movement cache and partial-update logic to `update_grid_map`

**Files:**
- Modify: `ui/pages/grid.py` — add reactive values and modify `update_grid_map`

- [ ] **Step 1: Add reactive cache values inside `grid_server`**

Add these lines inside `grid_server()`, after the `_map = MapWidget(...)` line:

```python
    # Movement animation state
    _movement_cache: reactive.Value[dict[str, dict]] = reactive.Value({})
    _prev_active_maps: reactive.Value[frozenset[str]] = reactive.Value(frozenset())
```

- [ ] **Step 2: Add cache-building reactive effect**

Add a new reactive effect inside `grid_server()`:

```python
    @reactive.effect
    def _rebuild_movement_cache():
        """Rebuild the movement map cache when species or config changes."""
        state.load_trigger.get()
        try:
            overlay = input.grid_overlay()
        except Exception:
            return
        if overlay != "__movement_animation__":
            _movement_cache.set({})
            _prev_active_maps.set(frozenset())
            return
        try:
            species = input.movement_species()
        except Exception:
            return
        if not species:
            return

        with reactive.isolate():
            cfg = state.config.get()
            cfg_dir = state.config_dir.get()

        ul_lat, ul_lon, lr_lat, lr_lon, nx, ny = _read_grid_values()
        grid_params = (ul_lat, ul_lon, lr_lat, lr_lon, nx, ny)

        cache = build_movement_cache(cfg, cfg_dir, grid_params, species=species)
        _movement_cache.set(cache)
        _prev_active_maps.set(frozenset())  # force redraw on next tick
```

- [ ] **Step 3: Modify `update_grid_map` to handle movement animation mode**

In the existing `update_grid_map` function, replace the overlay loading section (lines starting with `# Load overlay data if selected`) with logic that branches on the overlay type. After the legend entries loop for grid layers, replace everything from `# Load overlay data if selected` down to (but not including) the `widgets = [` line with:

```python
        # Load overlay data if selected
        overlay = input.grid_overlay() if hasattr(input, "grid_overlay") else None
        if not overlay:
            overlay = "grid_extent"

        if overlay == "__movement_animation__":
            # Movement animation mode — use cached maps
            cache = _movement_cache.get()
            if cache:
                try:
                    step = input.movement_step()
                except Exception:
                    step = 0
                active_ids = frozenset(
                    mid for mid, m in cache.items() if step in m["steps"]
                )
                prev = _prev_active_maps.get()
                if active_ids == prev and prev:
                    return  # skip update — no visual change
                _prev_active_maps.set(active_ids)

                for mid in sorted(active_ids):
                    m = cache[mid]
                    layer_id = f"movement-{mid}"
                    layers.append(polygon_layer(
                        layer_id,
                        data=m["cells"],
                        get_polygon="@@=d.polygon",
                        get_fill_color=m["color"],
                        get_line_color=[0, 0, 0, 0],
                        filled=True,
                        stroked=False,
                        pickable=True,
                    ))
                    age_suffix = f" ({m['age_range']})" if m["age_range"] else ""
                    legend_entries.append({
                        "layer_id": layer_id,
                        "label": f"{m['label']}{age_suffix}",
                        "color": m["color"][:3],
                        "shape": "rect",
                    })

        elif overlay != "grid_extent":
            overlay_path_str = cfg.get(overlay, "")
            if overlay_path_str and cfg_dir:
                overlay_file = (cfg_dir / overlay_path_str).resolve()
                if not overlay_file.is_relative_to(cfg_dir.resolve()):
                    _log.warning("Skipping path traversal in overlay: %s", overlay_path_str)
                elif not overlay_file.exists():
                    ui.notification_show(
                        f"File not found: {overlay_path_str}", type="warning", duration=3
                    )
                elif overlay_file.suffix == ".nc":
                    fb_lat = nc_data[0] if nc_data else None
                    fb_lon = nc_data[1] if nc_data else None
                    cells = load_netcdf_overlay(overlay_file, fb_lat, fb_lon)
                    if cells:
                        layers.append(polygon_layer(
                            "grid-overlay",
                            data=cells,
                            get_polygon="@@=d.polygon",
                            get_fill_color=[255, 140, 0, 150],
                            get_line_color=[0, 0, 0, 0],
                            filled=True,
                            stroked=False,
                            pickable=True,
                        ))
                        legend_entries.append({
                            "layer_id": "grid-overlay",
                            "label": "Overlay Data",
                            "color": [255, 140, 0],
                            "shape": "rect",
                        })
                elif overlay_file.suffix == ".csv":
                    csv_cells = load_csv_overlay(
                        overlay_file, ul_lat, ul_lon, lr_lat, lr_lon, nx, ny,
                        nc_data=nc_data,
                    )
                    if csv_cells:
                        layers.append(polygon_layer(
                            "grid-overlay",
                            data=csv_cells,
                            get_polygon="@@=d.polygon",
                            get_fill_color="@@=d.fill",
                            get_line_color=[0, 0, 0, 0],
                            filled=True,
                            stroked=False,
                            pickable=True,
                        ))
                        legend_entries.append({
                            "layer_id": "grid-overlay",
                            "label": "Overlay Data",
                            "color": [255, 140, 0],
                            "shape": "rect",
                        })
```

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: all pass

- [ ] **Step 5: Lint**

Run: `.venv/bin/ruff check ui/pages/grid.py`
Expected: All checks passed

- [ ] **Step 6: Commit**

```bash
git add ui/pages/grid.py
git commit -m "feat: wire movement cache and partial-update animation into grid map"
```

---

### Task 7: Add CSS styling for movement animation controls

**Files:**
- Modify: `www/osmose.css` (append before responsive section)

- [ ] **Step 1: Add movement controls CSS**

Add before the `/* ── Responsive / mobile */` section in `www/osmose.css`:

```css
/* ── Movement animation controls ─────────────────────────── */
.osm-movement-controls .shiny-input-container {
    margin-bottom: 8px;
}
.osm-movement-controls .irs {
    margin-top: 4px;
}
```

- [ ] **Step 2: Commit**

```bash
git add www/osmose.css
git commit -m "style: add movement animation controls CSS"
```

---

## Chunk 3: Integration Testing & Polish

### Task 8: Manual integration test with Eec Full example

This task verifies the feature works end-to-end in the running app.

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: all pass, no regressions

- [ ] **Step 2: Restart Shiny server and load Eec Full**

```bash
touch /srv/shiny-server/osmose/restart.txt
```

Open http://localhost:3838/osmose/, load "Eec Full", go to Grid & Maps.

- [ ] **Step 3: Verify Movement Animation dropdown entry**

Select "Movement Animation" from the Overlay data dropdown. Verify:
- Species dropdown appears with species names (lesserSpottedDogfish, redMullet, etc.)
- Speed selector appears with 0.5x/1x/2x/4x options
- Time step slider appears with Play/Pause button

- [ ] **Step 4: Test animation playback**

Select a species (e.g., "cod"), click Play. Verify:
- Maps render on the deck.gl map with distinct colors
- Legend shows map labels with age ranges and checkboxes
- Animation cycles through time steps, maps appear/disappear at seasonal boundaries
- Most ticks show no visual change (partial update optimization)

- [ ] **Step 5: Test speed change**

Switch speed to 2x while playing. Verify slider continues (may briefly pause).

- [ ] **Step 6: Test no-maps example**

Load "Minimal" example, go to Grid & Maps, select "Movement Animation". Verify notification message appears.

- [ ] **Step 7: Commit final state**

```bash
git add -A
git commit -m "feat: movement visualization — integration verified"
```

---

### Task 9: Run lint, full test suite, and push

- [ ] **Step 1: Final lint check**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`
Expected: All checks passed

- [ ] **Step 2: Final test run**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: all pass

- [ ] **Step 3: Push to origin**

```bash
git push origin master
```
