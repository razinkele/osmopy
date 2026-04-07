# UI Compaction + CSV Map Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten the UI layout (remove info bar row, slim header, compress nav) and add a standalone Map Viewer tab under MANAGE for browsing/previewing CSV/NC spatial files.

**Architecture:** Three independent changes: (1) CSS compaction of header + nav, (2) move model info into header + cleanup, (3) new Map Viewer page reusing existing grid_helpers infrastructure. A shared `discover_spatial_files()` function is extracted from `grid_overlay_selector` so both Grid and Map Viewer can use it.

**Tech Stack:** Shiny for Python, shiny_deckgl (MapWidget), NumPy, pandas, xarray, CSS

---

### Task 1: CSS compaction — header and nav sidebar

**Files:**
- Modify: `www/osmose.css:84` (header padding), `www/osmose.css:114` (logo font), `www/osmose.css:124` (subtitle font), `www/osmose.css:361` (nav pill padding), `www/osmose.css:329-335` (section label), `www/osmose.css:1118-1136` (responsive block)

- [ ] **Step 1: Slim the header padding and logo**

In `www/osmose.css`, change `.osmose-header` padding (line 84) and `.osmose-logo` font-size (line 114) and `.osmose-logo .subtitle` font-size (line 124):

```css
/* line 84: was padding: 14px 24px; */
padding: 8px 16px;

/* line 114: was font-size: 1.4rem; */
font-size: 1.1rem;

/* line 124: was font-size: 0.8rem; */
font-size: 0.7rem;
```

- [ ] **Step 2: Tighten nav pill padding**

In `www/osmose.css`, change `.nav-pills .nav-link` (line 361) from `8px 14px` to `4px 12px`, and reduce font-size (line 362):

```css
/* line 361: was padding: 8px 14px !important; */
padding: 4px 12px !important;
/* line 362: was font-size: 0.85rem; */
font-size: 0.82rem;
```

- [ ] **Step 3: Tighten section label spacing**

In `www/osmose.css`, change `.nav-pills .nav-item .osmose-section-label` (line 329-335), reduce the padding:

```css
/* line 335: was padding: 20px 14px 6px; */
padding: 12px 14px 4px;
```

And change `.osmose-section-label` font-size (line 330):

```css
/* line 330: was font-size: 0.65rem; */
font-size: 0.6rem;
```

- [ ] **Step 4: Add responsive rule to hide model info at narrow viewports**

In `www/osmose.css`, inside the `@media (max-width: 768px)` block (after line 1130), add:

```css
  .osm-config-info {
    display: none;
  }
```

- [ ] **Step 5: Verify the app loads without visual regression**

Run: `.venv/bin/shiny run app.py --host 0.0.0.0 --port 8000`

Check: header is slimmer, nav pills are tighter, no overlapping text. Stop the server.

- [ ] **Step 6: Commit**

```bash
git add www/osmose.css
git commit -m "style: compact header padding, nav pill spacing, and section labels"
```

---

### Task 2: Move model info into header bar

**Files:**
- Modify: `app.py:12` (remove import), `app.py:155-182` (move output_ui), `app.py:317-342` (restyle config_header renderer)
- Modify: `ui/styles.py:36-39` (remove STYLE_CONFIG_HEADER)

- [ ] **Step 1: Remove STYLE_CONFIG_HEADER from ui/styles.py**

In `ui/styles.py`, delete lines 35-39:

```python
# Delete these lines:
# Config header bar
STYLE_CONFIG_HEADER = (
    "display: flex; justify-content: space-between; align-items: center; "
    "padding: 8px 16px; border-bottom: 1px solid var(--osm-border, #2d3d50);"
)
```

- [ ] **Step 2: Remove STYLE_CONFIG_HEADER import from app.py**

In `app.py`, delete line 12:

```python
# Delete this line:
from ui.styles import STYLE_CONFIG_HEADER
```

- [ ] **Step 3: Move output_ui("config_header") into the header div**

In `app.py`, remove `ui.output_ui("config_header")` from line 182 (the standalone row). Insert it inside `.osmose-header-actions` (line 155-177), as the first child before the theme toggle button.

Change lines 155-182 from:

```python
        ui.div(
            ui.tags.button(
                ui.tags.span("\u2600\ufe0f", class_="icon-sun"),
                ui.tags.span("\u263e", class_="icon-moon"),
                class_="osmose-theme-toggle",
                id="themeToggle",
                title="Toggle light/dark theme",
                onclick="toggleTheme()",
                **{"aria-label": "Toggle light/dark theme"},
            ),
            ui.tags.a(
                "About",
                class_="osmose-header-btn",
                href="#",
                **{"data-bs-toggle": "modal", "data-bs-target": "#aboutModal"},
            ),
            ui.tags.a(
                "Help",
                class_="osmose-header-btn",
                href="#",
                **{"data-bs-toggle": "modal", "data-bs-target": "#helpModal"},
            ),
            class_="osmose-header-actions",
        ),
        class_="osmose-header",
    ),
    # ── Global loading overlay ───────────────────────────────────
    ui.output_ui("config_header"),
```

To:

```python
        ui.div(
            ui.output_ui("config_header"),
            ui.tags.button(
                ui.tags.span("\u2600\ufe0f", class_="icon-sun"),
                ui.tags.span("\u263e", class_="icon-moon"),
                class_="osmose-theme-toggle",
                id="themeToggle",
                title="Toggle light/dark theme",
                onclick="toggleTheme()",
                **{"aria-label": "Toggle light/dark theme"},
            ),
            ui.tags.a(
                "About",
                class_="osmose-header-btn",
                href="#",
                **{"data-bs-toggle": "modal", "data-bs-target": "#aboutModal"},
            ),
            ui.tags.a(
                "Help",
                class_="osmose-header-btn",
                href="#",
                **{"data-bs-toggle": "modal", "data-bs-target": "#helpModal"},
            ),
            class_="osmose-header-actions",
        ),
        class_="osmose-header",
    ),
    # ── Global loading overlay ───────────────────────────────────
```

- [ ] **Step 4: Restyle the config_header renderer**

In `app.py`, replace the `config_header` render function (lines 317-342) with:

```python
    @render.ui
    def config_header():
        name = state.config_name.get()
        if not name:
            return ui.div()
        cfg = state.config.get()
        try:
            n_species = int(float(cfg.get("simulation.nspecies", "0")))
        except (ValueError, TypeError):
            n_species = 0
        n_params = len(cfg)
        is_dirty = state.dirty.get()
        return ui.div(
            ui.tags.span(
                name,
                style=(
                    "color: var(--osm-accent); font-weight: 600; font-size: 0.78rem;"
                    " overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"
                    " max-width: 180px; display: inline-block; vertical-align: middle;"
                ),
            ),
            ui.tags.span(
                f" {n_species} species \u2022 {n_params} params",
                style="color: var(--osm-text-muted); font-size: 0.7rem; margin-left: 6px;",
            ),
            ui.tags.span(
                " modified" if is_dirty else "",
                style="color: #e67e22; font-size: 0.65rem; font-style: italic; margin-left: 4px;",
            ),
            class_="osm-config-info",
            style="display: flex; align-items: center; margin-right: 8px;",
        )
```

- [ ] **Step 5: Run existing tests to verify no breakage**

Run: `.venv/bin/python -m pytest tests/ -x -q --no-header`

Expected: ALL PASS, no import errors from removed `STYLE_CONFIG_HEADER`

- [ ] **Step 6: Commit**

```bash
git add app.py ui/styles.py
git commit -m "feat: move model info into header bar, remove second row"
```

---

### Task 3: Extract _overlay_label and discover_spatial_files into grid_helpers

**Files:**
- Modify: `ui/pages/grid_helpers.py` (add functions)
- Modify: `ui/pages/grid.py` (remove _overlay_label, refactor selector)
- Modify: `tests/test_overlay_display.py` (update imports)
- Test: `tests/test_discover_spatial_files.py` (new)

- [ ] **Step 1: Write tests for discover_spatial_files**

Create `tests/test_discover_spatial_files.py`:

```python
"""Tests for discover_spatial_files and _overlay_label in grid_helpers."""

import pathlib

import numpy as np
import pandas as pd
import pytest


_EEC_FULL_DIR = pathlib.Path(__file__).parent.parent / "data" / "eec_full"
_EEC_FULL_AVAILABLE = (_EEC_FULL_DIR / "eec_all-parameters.csv").exists()


class TestOverlayLabelMoved:
    """Verify _overlay_label is importable from grid_helpers."""

    def test_import_from_grid_helpers(self):
        from ui.pages.grid_helpers import _overlay_label

        assert callable(_overlay_label)

    def test_ltl_label(self):
        from ui.pages.grid_helpers import _overlay_label

        assert _overlay_label("eec_ltlbiomassTons.nc") == "LTL Biomass"

    def test_fishing_label(self):
        from ui.pages.grid_helpers import _overlay_label

        label = _overlay_label("fishing/fishing-distrib.csv")
        assert "ishing" in label


@pytest.mark.skipif(not _EEC_FULL_AVAILABLE, reason="EEC Full not found")
class TestDiscoverSpatialFiles:
    @pytest.fixture(autouse=True)
    def _load(self):
        from osmose.config.reader import OsmoseConfigReader

        reader = OsmoseConfigReader()
        self.cfg = reader.read(_EEC_FULL_DIR / "eec_all-parameters.csv")
        self.cfg_dir = _EEC_FULL_DIR

    def test_returns_three_categories(self):
        from ui.pages.grid_helpers import discover_spatial_files

        result = discover_spatial_files(self.cfg, self.cfg_dir)
        assert "movement" in result
        assert "fishing" in result
        assert "other" in result

    def test_movement_grouped_by_species(self):
        from ui.pages.grid_helpers import discover_spatial_files

        result = discover_spatial_files(self.cfg, self.cfg_dir)
        movement = result["movement"]
        assert isinstance(movement, dict)
        assert "cod" in movement
        assert len(movement["cod"]) >= 3  # cod has 4 maps (some share files)

    def test_movement_has_14_species(self):
        from ui.pages.grid_helpers import discover_spatial_files

        result = discover_spatial_files(self.cfg, self.cfg_dir)
        assert len(result["movement"]) == 14

    def test_fishing_has_entries(self):
        from ui.pages.grid_helpers import discover_spatial_files

        result = discover_spatial_files(self.cfg, self.cfg_dir)
        assert len(result["fishing"]) >= 1

    def test_other_has_ltl(self):
        from ui.pages.grid_helpers import discover_spatial_files

        result = discover_spatial_files(self.cfg, self.cfg_dir)
        labels = [e["label"] for e in result["other"]]
        assert any("LTL" in l for l in labels)

    def test_all_paths_exist(self):
        from ui.pages.grid_helpers import discover_spatial_files

        result = discover_spatial_files(self.cfg, self.cfg_dir)
        for entries in result["fishing"]:
            assert entries["path"].exists(), f"Missing: {entries['path']}"
        for entries in result["other"]:
            assert entries["path"].exists(), f"Missing: {entries['path']}"

    def test_empty_config_returns_empty(self):
        from ui.pages.grid_helpers import discover_spatial_files

        result = discover_spatial_files({}, None)
        assert result == {"movement": {}, "fishing": [], "other": []}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_discover_spatial_files.py -v`

Expected: FAIL (functions don't exist yet)

- [ ] **Step 3: Move _overlay_label to grid_helpers.py**

In `ui/pages/grid_helpers.py`, add `_overlay_label` before the `_zoom_for_span` function (around line 69). Copy the entire function from `grid.py:63-76`:

```python
def _overlay_label(rel_path: str) -> str:
    """Generate a human-readable label from an overlay file path."""
    stem = Path(rel_path).stem.lower()
    if "ltl" in stem or ("ltlbiomass" in stem.replace("_", "").replace("-", "")):
        return "LTL Biomass"
    if "backgroundspecies" in stem.replace("_", "").replace("-", ""):
        return "Background Species"
    if "mpa" in stem or ("marine" in stem and "protected" in stem):
        return stem.replace("_", " ").replace("-", " ").title()
    if "distrib" in stem or ("fishing" in stem and "distrib" in stem):
        return "Fishing Distribution"
    if "fishing" in stem:
        return "Fishing"
    return stem.replace("_", " ").replace("-", " ").title()
```

- [ ] **Step 4: Implement discover_spatial_files in grid_helpers.py**

Add after `_overlay_label` in `ui/pages/grid_helpers.py`:

```python
def discover_spatial_files(
    cfg: dict[str, str],
    cfg_dir: Path | None,
) -> dict[str, dict[str, list[dict]] | list[dict]]:
    """Discover all spatial files (CSV/NC) from an OSMOSE config.

    Returns ``{"movement": {species: [entries]}, "fishing": [entries], "other": [entries]}``
    where each entry is ``{"path": Path, "label": str, ...}``.
    """
    movement: dict[str, list[dict]] = {}
    fishing: list[dict] = []
    other: list[dict] = []

    if not cfg_dir or not cfg_dir.is_dir():
        return {"movement": movement, "fishing": fishing, "other": other}

    seen_paths: set[str] = set()

    skip_prefixes = (
        "grid.", "osmose.configuration.", "simulation.restart",
        "predation.accessibility", "fisheries.catchability", "fisheries.discards",
        "movement.file.map", "movement.species.map",
        "movement.initialAge.", "movement.lastAge.", "movement.steps.",
        "movement.distribution.",
    )

    # Pass 1: general spatial files (non-movement, non-fishing)
    for key, val in sorted(cfg.items()):
        if not val or not isinstance(val, str):
            continue
        if not (val.endswith(".nc") or val.endswith(".csv")):
            continue
        if key.startswith(skip_prefixes) or "season" in key:
            continue
        try:
            resolved = _safe_resolve(cfg_dir, val)
        except Exception:
            continue
        if resolved is None or not resolved.exists():
            continue
        path_id = str(resolved)
        if path_id in seen_paths:
            continue
        seen_paths.add(path_id)
        other.append({"path": resolved, "label": _overlay_label(val)})

    # Pass 2: movement maps grouped by species
    for key, val in sorted(cfg.items()):
        if not key.startswith("movement.file.map") or not val:
            continue
        if val.lower() in ("null", "none") or not val.endswith(".csv"):
            continue
        try:
            resolved = _safe_resolve(cfg_dir, val)
        except Exception:
            continue
        if resolved is None or not resolved.exists():
            continue
        path_id = str(resolved)
        if path_id in seen_paths:
            continue
        seen_paths.add(path_id)
        idx = key[len("movement.file.map"):]
        species = cfg.get(f"movement.species.map{idx}", "unknown")
        age_range = ""
        min_age = cfg.get(f"movement.initialAge.map{idx}")
        max_age = cfg.get(f"movement.lastAge.map{idx}")
        if min_age is not None:
            age_range = f"{min_age}+" if max_age is None else f"{min_age}-{max_age}"
            age_range += " yr"
        steps_raw = cfg.get(f"movement.steps.map{idx}", "")
        n_steps = len([s for s in steps_raw.split(";") if s.strip()]) if steps_raw else 0
        label = derive_map_label(val, int(idx) if idx.isdigit() else 0)
        movement.setdefault(species, []).append({
            "path": resolved, "label": label, "age": age_range, "steps": n_steps,
        })

    # Pass 3: fishing distribution maps
    for key, val in sorted(cfg.items()):
        if not key.startswith("fisheries.movement.file.map") or not val:
            continue
        if not val.endswith(".csv"):
            continue
        try:
            resolved = _safe_resolve(cfg_dir, val)
        except Exception:
            continue
        if resolved is None or not resolved.exists():
            continue
        path_id = str(resolved)
        if path_id in seen_paths:
            continue
        seen_paths.add(path_id)
        fishing.append({"path": resolved, "label": f"Fishing: {_overlay_label(val)}"})

    # Pass 4: MPA directory scan
    mpa_dir = cfg_dir / "mpa"
    if mpa_dir.is_dir():
        for mpa_file in sorted(mpa_dir.glob("*.csv")):
            path_id = str(mpa_file.resolve())
            if path_id not in seen_paths:
                seen_paths.add(path_id)
                other.append({
                    "path": mpa_file.resolve(),
                    "label": f"MPA: {mpa_file.stem.replace('_', ' ').title()}",
                })

    return {"movement": movement, "fishing": fishing, "other": other}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_discover_spatial_files.py -v`

Expected: ALL PASS

- [ ] **Step 6: Update grid.py — remove _overlay_label, import from grid_helpers**

In `ui/pages/grid.py`:

1. Delete the `_overlay_label` function definition (lines 63-76).
2. Add `_overlay_label` to the existing import from `grid_helpers` (line 23-35). Add it to the import list:

```python
from ui.pages.grid_helpers import (
    _overlay_label,
    build_grid_layers,
    ...
)
```

- [ ] **Step 7: Update tests/test_overlay_display.py imports**

In `tests/test_overlay_display.py`, find the OD7 test class `TestOverlayLabel`. There are six individual `from ui.pages.grid import _overlay_label` statements at lines 664, 668, 672, 677, 683, and 689. Change all of them from:

```python
from ui.pages.grid import _overlay_label
```

To:

```python
from ui.pages.grid_helpers import _overlay_label
```

- [ ] **Step 8: Run all overlay tests**

Run: `.venv/bin/python -m pytest tests/test_overlay_display.py tests/test_discover_spatial_files.py tests/test_csv_map_display.py -v`

Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add ui/pages/grid_helpers.py ui/pages/grid.py tests/test_discover_spatial_files.py tests/test_overlay_display.py
git commit -m "refactor: extract _overlay_label and discover_spatial_files into grid_helpers"
```

---

### Task 4: Create Map Viewer page

**Files:**
- Create: `ui/pages/map_viewer.py`
- Modify: `app.py` (register page)
- Test: `tests/test_map_viewer.py` (new)

- [ ] **Step 1: Write tests for the Map Viewer module**

Create `tests/test_map_viewer.py`:

```python
"""Tests for the Map Viewer page module."""

import pathlib

import pytest

_EEC_FULL_DIR = pathlib.Path(__file__).parent.parent / "data" / "eec_full"
_EEC_FULL_AVAILABLE = (_EEC_FULL_DIR / "eec_all-parameters.csv").exists()


class TestMapViewerUi:
    def test_map_viewer_ui_returns_div(self):
        from ui.pages.map_viewer import map_viewer_ui

        result = map_viewer_ui()
        html = str(result)
        assert "map_viewer_map" in html, "MapWidget ID must be present"
        assert "osm-split-layout" in html, "Must use split layout"

    def test_map_viewer_ui_has_file_list_output(self):
        from ui.pages.map_viewer import map_viewer_ui

        html = str(map_viewer_ui())
        assert "map_viewer_file_list" in html, "File list output_ui must be present"

    def test_map_viewer_ui_has_hint(self):
        from ui.pages.map_viewer import map_viewer_ui

        html = str(map_viewer_ui())
        assert "map_viewer_hint" in html, "Hint output_ui must be present"


@pytest.mark.skipif(not _EEC_FULL_AVAILABLE, reason="EEC Full not found")
class TestMapViewerFileList:
    """Test the file list builder used by the Map Viewer."""

    def test_build_file_list_choices_has_movement(self):
        from osmose.config.reader import OsmoseConfigReader
        from ui.pages.grid_helpers import discover_spatial_files

        reader = OsmoseConfigReader()
        cfg = reader.read(_EEC_FULL_DIR / "eec_all-parameters.csv")
        catalog = discover_spatial_files(cfg, _EEC_FULL_DIR)

        # Build flat choices like the Map Viewer does
        choices = {}
        for species, entries in sorted(catalog["movement"].items()):
            for e in entries:
                choices[str(e["path"])] = f"{species}: {e['label']}"
        assert len(choices) >= 20, f"Expected >=20 movement entries, got {len(choices)}"

    def test_build_file_list_choices_has_fishing(self):
        from osmose.config.reader import OsmoseConfigReader
        from ui.pages.grid_helpers import discover_spatial_files

        reader = OsmoseConfigReader()
        cfg = reader.read(_EEC_FULL_DIR / "eec_all-parameters.csv")
        catalog = discover_spatial_files(cfg, _EEC_FULL_DIR)
        assert len(catalog["fishing"]) >= 1
```

- [ ] **Step 2: Run tests to verify UI tests fail**

Run: `.venv/bin/python -m pytest tests/test_map_viewer.py -v`

Expected: FAIL (map_viewer module doesn't exist)

- [ ] **Step 3: Create ui/pages/map_viewer.py**

Create `ui/pages/map_viewer.py`:

```python
"""Map Viewer page — browse and preview CSV/NC spatial files."""

from pathlib import Path

import numpy as np
from shiny import ui, reactive, render
from shiny.types import SilentException

from shiny_deckgl import (
    MapWidget,
    polygon_layer,
    CARTO_POSITRON,
    CARTO_DARK,
    zoom_widget,
    compass_widget,
    fullscreen_widget,
    scale_widget,
)

from osmose.logging import setup_logging
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.pages.grid_helpers import (
    _overlay_label,
    _zoom_for_span,
    build_grid_layers,
    build_netcdf_grid_layers,
    discover_spatial_files,
    list_nc_overlay_variables,
    load_csv_overlay,
    load_mask,
    load_netcdf_grid,
    load_netcdf_overlay,
    make_legend,
)
from ui.state import get_theme_mode

_log = setup_logging("osmose.map_viewer")


def map_viewer_ui():
    viewer_map = MapWidget(
        "map_viewer_map",
        view_state={"latitude": 46.0, "longitude": -4.5, "zoom": 5, "pitch": 0, "bearing": 0},
        style=CARTO_POSITRON,
        tooltip={"html": "Value: {properties.value}", "style": {"fontSize": "12px"}},
        controls=[],
    )

    return ui.div(
        expand_tab("Map Viewer", "map_viewer"),
        ui.layout_columns(
            ui.card(
                collapsible_card_header("Map Viewer", "map_viewer"),
                ui.output_ui("map_viewer_file_list"),
            ),
            ui.div(
                ui.output_ui("map_viewer_hint"),
                ui.output_ui("map_viewer_nc_controls"),
                viewer_map.ui(height="100%"),
                ui.output_ui("map_viewer_metadata"),
                class_="osm-grid-map-container",
            ),
            col_widths=[5, 7],
        ),
        class_="osm-split-layout",
        id="split_map_viewer",
    )


def map_viewer_server(input, output, session, state):
    _map = MapWidget(
        "map_viewer_map",
        view_state={"latitude": 46.0, "longitude": -4.5, "zoom": 5},
        style=CARTO_POSITRON,
    )

    @render.ui
    def map_viewer_hint():
        state.load_trigger.get()
        with reactive.isolate():
            name = state.config_name.get()
        if not name:
            return ui.p(
                "Load a configuration to browse spatial files.",
                style="color: var(--osm-text-muted); text-align: center; padding: 40px 20px;",
            )
        return ui.div()

    @render.ui
    def map_viewer_file_list():
        state.load_trigger.get()
        with reactive.isolate():
            cfg = state.config.get()
            cfg_dir = state.config_dir.get()

        if not cfg or not cfg_dir:
            return ui.p(
                "No configuration loaded.",
                style="color: var(--osm-text-muted); font-size: 12px; padding: 8px;",
            )

        catalog = discover_spatial_files(cfg, cfg_dir)
        choices = {}

        # Movement maps grouped by species
        movement = catalog.get("movement", {})
        total_mvmt = sum(len(v) for v in movement.values())
        if movement:
            group_choices = {}
            for species in sorted(movement):
                for entry in movement[species]:
                    key = str(entry["path"])
                    suffix = f" ({entry['age']})" if entry.get("age") else ""
                    group_choices[key] = f"{species}: {entry['label']}{suffix}"
            choices[f"Movement Maps ({total_mvmt})"] = group_choices

        # Fishing
        fishing = catalog.get("fishing", [])
        if fishing:
            fish_choices = {}
            for entry in fishing:
                fish_choices[str(entry["path"])] = entry["label"]
            choices[f"Fishing ({len(fishing)})"] = fish_choices

        # Other
        other = catalog.get("other", [])
        if other:
            other_choices = {}
            for entry in other:
                other_choices[str(entry["path"])] = entry["label"]
            choices[f"Other ({len(other)})"] = other_choices

        if not choices:
            return ui.p(
                "No spatial files found in this configuration.",
                style="color: var(--osm-text-muted); font-size: 12px; padding: 8px;",
            )

        return ui.input_select(
            "map_viewer_file",
            "Select a file to preview",
            choices=choices,
            size=20,
        )

    @render.ui
    def map_viewer_nc_controls():
        """Show time slider for NC files with time dimension."""
        try:
            file_val = input.map_viewer_file()
        except SilentException:
            return ui.div()

        if not file_val:
            return ui.div()

        file_path = Path(file_val)
        if file_path.suffix != ".nc" or not file_path.exists():
            return ui.div()

        meta = list_nc_overlay_variables(str(file_path))
        if not meta:
            return ui.div()

        controls = []

        if len(meta) > 1:
            var_choices = {k: k.replace("_", " ").title() for k in meta}
            try:
                current_var = input.mv_nc_var()
            except SilentException:
                current_var = next(iter(meta))
            if current_var not in meta:
                current_var = next(iter(meta))
            controls.append(
                ui.input_select("mv_nc_var", "Variable", choices=var_choices, selected=current_var)
            )
            sel_var = current_var
        else:
            sole_var = next(iter(meta))
            controls.append(
                ui.div(
                    ui.input_select("mv_nc_var", "Variable", choices={sole_var: sole_var}, selected=sole_var),
                    style="display:none",
                )
            )
            sel_var = sole_var

        var_meta = meta.get(sel_var, {})
        n_time = var_meta.get("n_time", 1)
        if n_time > 1:
            try:
                current_step = int(input.mv_nc_time())
            except (SilentException, ValueError, TypeError):
                current_step = 0
            current_step = max(0, min(current_step, n_time - 1))
            controls.append(
                ui.input_slider("mv_nc_time", "Time step", min=0, max=n_time - 1, value=current_step, step=1)
            )

        return ui.div(*controls) if controls else ui.div()

    @render.ui
    def map_viewer_metadata():
        """Show file metadata below the map."""
        try:
            file_val = input.map_viewer_file()
        except SilentException:
            return ui.div()

        if not file_val:
            return ui.div()

        file_path = Path(file_val)
        if not file_path.exists():
            return ui.div()

        parts = [ui.tags.span(file_path.name, style="color: var(--osm-text-muted); font-size: 11px;")]
        return ui.div(
            *parts,
            style="padding: 4px 8px; font-size: 11px; color: var(--osm-text-muted);",
        )

    @reactive.effect
    async def update_map_viewer():
        try:
            file_val = input.map_viewer_file()
        except (SilentException, AttributeError):
            return

        if not file_val:
            return

        file_path = Path(file_val)
        if not file_path.exists():
            return

        is_dark = get_theme_mode(input) == "dark"

        with reactive.isolate():
            cfg = state.config.get()
            cfg_dir = state.config_dir.get()

        # Read bbox values unconditionally (needed for CSV overlay even when NcGrid)
        try:
            ul_lat = float(cfg.get("grid.upleft.lat", 0))
            ul_lon = float(cfg.get("grid.upleft.lon", 0))
            lr_lat = float(cfg.get("grid.lowright.lat", 0))
            lr_lon = float(cfg.get("grid.lowright.lon", 0))
            nx = int(float(cfg.get("grid.nlon", 0)))
            ny = int(float(cfg.get("grid.nlat", 0)))
        except (ValueError, TypeError):
            ul_lat = ul_lon = lr_lat = lr_lon = 0.0
            nx = ny = 0

        # Load grid base layers
        is_ncgrid = "NcGrid" in cfg.get("grid.java.classname", "")
        nc_data = load_netcdf_grid(cfg, config_dir=cfg_dir) if is_ncgrid else None

        if nc_data is not None:
            nc_lat, nc_lon, nc_mask = nc_data
            layers, view_state = build_netcdf_grid_layers(nc_lat, nc_lon, nc_mask, is_dark)
        else:
            mask = load_mask(cfg, config_dir=cfg_dir)
            layers = build_grid_layers(ul_lat, ul_lon, lr_lat, lr_lon, nx, ny, is_dark, mask)
            if ul_lat != 0 or ul_lon != 0 or lr_lat != 0 or lr_lon != 0:
                center_lat = (ul_lat + lr_lat) / 2
                center_lon = (ul_lon + lr_lon) / 2
                span = max(abs(ul_lat - lr_lat), abs(lr_lon - ul_lon))
                view_state = {"latitude": center_lat, "longitude": center_lon, "zoom": _zoom_for_span(span)}
            else:
                view_state = {"latitude": 46.0, "longitude": -4.5, "zoom": 5}

        # Load overlay
        legend_entries = []
        if file_path.suffix == ".nc":
            try:
                nc_var = input.mv_nc_var() or None
            except (SilentException, AttributeError):
                nc_var = None
            try:
                nc_time = max(0, int(input.mv_nc_time()))
            except (SilentException, ValueError, TypeError, AttributeError):
                nc_time = 0
            meta = list_nc_overlay_variables(str(file_path))
            nc_vmin = nc_vmax = None
            if meta:
                sel_meta = meta.get(nc_var or "") or next(iter(meta.values()), None)
                if sel_meta:
                    nc_vmin, nc_vmax = sel_meta["vmin"], sel_meta["vmax"]
            fb_lat = nc_data[0] if nc_data else None
            fb_lon = nc_data[1] if nc_data else None
            cells = load_netcdf_overlay(
                file_path, fb_lat, fb_lon,
                var_lat=cfg.get("grid.var.lat", "lat"),
                var_lon=cfg.get("grid.var.lon", "lon"),
                var_name=nc_var, time_step=nc_time, vmin=nc_vmin, vmax=nc_vmax,
            )
            if cells:
                layers.append(polygon_layer(
                    "viewer-overlay", data=cells,
                    get_polygon="@@=d.polygon", get_fill_color="@@=d.fill",
                    get_line_color=[0, 0, 0, 0], filled=True, stroked=False, pickable=True,
                ))
                label = (nc_var or "Overlay").replace("_", " ").title()
                legend_entries.append({"layer_id": "viewer-overlay", "label": label, "color": [0, 170, 180], "shape": "rect"})
        elif file_path.suffix == ".csv":
            if nc_data is not None:
                cells = load_csv_overlay(file_path, 0, 0, 0, 0, 0, 0, nc_data=nc_data)
            else:
                cells = load_csv_overlay(file_path, ul_lat, ul_lon, lr_lat, lr_lon, nx, ny)
            if cells:
                layers.append(polygon_layer(
                    "viewer-overlay", data=cells,
                    get_polygon="@@=d.polygon", get_fill_color="@@=d.fill",
                    get_line_color=[0, 0, 0, 0], filled=True, stroked=False, pickable=True,
                ))
                legend_entries.append({"layer_id": "viewer-overlay", "label": _overlay_label(file_path.name), "color": [255, 140, 0], "shape": "rect"})

        # Widgets
        style = CARTO_DARK if is_dark else CARTO_POSITRON
        if style != _map.style:
            _map.style = style
            await _map.set_style(session, style)

        widgets = [
            fullscreen_widget(placement="top-left"),
            zoom_widget(placement="top-right"),
            compass_widget(placement="top-right"),
            scale_widget(placement="bottom-right"),
        ]
        if legend_entries:
            widgets.append(make_legend(entries=legend_entries, placement="bottom-left", show_checkbox=False, collapsed=True, title="Layers"))

        await _map.update(session, layers=layers, view_state=view_state, transition_duration=800, widgets=widgets)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_map_viewer.py -v`

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add ui/pages/map_viewer.py tests/test_map_viewer.py
git commit -m "feat: add Map Viewer page with file list and deck.gl preview"
```

---

### Task 5: Register Map Viewer in app.py

> **Note:** Tasks 2 and 3 modify `app.py` before this task. Line numbers below reference the ORIGINAL file. After earlier tasks run, lines will have shifted by a few positions. Use content matching (search for the exact string) rather than relying on line numbers.

**Files:**
- Modify: `app.py` (import, nav_panel, server call, pageIds)

- [ ] **Step 1: Add import**

In `app.py`, after line 28 (`from ui.pages.advanced import advanced_ui, advanced_server`), add:

```python
from ui.pages.map_viewer import map_viewer_ui, map_viewer_server
```

- [ ] **Step 2: Add nav_panel under MANAGE**

In `app.py`, after line 213 (`ui.nav_panel("Advanced", advanced_ui(), value="advanced"),`), add:

```python
        ui.nav_panel("Map Viewer", map_viewer_ui(), value="map_viewer"),
```

- [ ] **Step 3: Add server call**

In `app.py`, after line 364 (`advanced_server(input, output, session, state)`), add:

```python
    map_viewer_server(input, output, session, state)
```

- [ ] **Step 4: Add 'map_viewer' to pageIds JS array**

In `app.py`, line 119-120, change:

```javascript
var pageIds = ['setup','grid','forcing','fishing','movement',
               'run','results','spatial_results','calibration','scenarios','advanced'];
```

To:

```javascript
var pageIds = ['setup','grid','forcing','fishing','movement',
               'run','results','spatial_results','calibration','scenarios','advanced','map_viewer'];
```

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q --no-header`

Expected: ALL PASS (no regressions)

- [ ] **Step 6: Manual verification**

Run: `.venv/bin/shiny run app.py --host 0.0.0.0 --port 8000`

Verify:
1. Header is compact — single row, model info right-aligned
2. Nav pills are tighter
3. Map Viewer tab appears under MANAGE
4. Load EEC Full example, navigate to Map Viewer
5. File list shows Movement Maps, Fishing, Other categories
6. Select a file — map preview renders
7. Stop server.

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "feat: register Map Viewer tab in app navigation"
```

---

### Task 6: E2E tests for Map Viewer

**Files:**
- Create: `tests/test_e2e_map_viewer.py`

- [ ] **Step 1: Write E2E Playwright tests**

Create `tests/test_e2e_map_viewer.py`:

```python
"""E2E Playwright tests for the Map Viewer page.

Run: .venv/bin/python -m pytest tests/test_e2e_map_viewer.py -v -m e2e
"""

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_TIMEOUT = 25_000


def _load_eec_full_and_goto_map_viewer(page: Page, app: ShinyAppProc) -> None:
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=15_000)
    page.select_option("#load_example", "eec_full")
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=15_000)
    page.locator(".nav-pills .nav-link[data-value='map_viewer']").click()
    page.wait_for_timeout(2_000)


def test_map_viewer_tab_accessible(page: Page, app: ShinyAppProc):
    """Map Viewer tab should be reachable after loading EEC Full."""
    _load_eec_full_and_goto_map_viewer(page, app)
    selector = page.locator("#map_viewer_file")
    expect(selector).to_be_visible(timeout=_TIMEOUT)


def test_map_viewer_file_list_populated(page: Page, app: ShinyAppProc):
    """File list should have movement and fishing entries with optgroup headers."""
    _load_eec_full_and_goto_map_viewer(page, app)
    page.wait_for_function(
        "(document.querySelector('#map_viewer_file')?.options?.length ?? 0) >= 5",
        timeout=_TIMEOUT,
    )
    count = page.evaluate("document.querySelector('#map_viewer_file')?.options?.length ?? 0")
    assert count >= 10, f"Expected >=10 file options, got {count}"

    # Verify optgroup headers exist (grouped select)
    optgroups = page.locator("#map_viewer_file optgroup").all()
    assert len(optgroups) >= 2, (
        f"Expected >=2 optgroup headers (Movement, Fishing/Other), got {len(optgroups)}"
    )
    labels = [og.get_attribute("label") or "" for og in optgroups]
    assert any("Movement" in l for l in labels), f"Expected 'Movement' optgroup, got: {labels}"


def test_map_viewer_renders_csv_overlay(page: Page, app: ShinyAppProc):
    """Selecting a CSV file should render a viewer-overlay layer."""
    _load_eec_full_and_goto_map_viewer(page, app)
    page.wait_for_function(
        "(document.querySelector('#map_viewer_file')?.options?.length ?? 0) >= 5",
        timeout=_TIMEOUT,
    )
    # Select the first valid option using Playwright's select_option (triggers Shiny binding)
    first_value = page.evaluate("""
        (() => {
            const sel = document.querySelector('#map_viewer_file');
            for (const opt of sel.options) {
                if (opt.value && !opt.disabled && !opt.parentElement.tagName === 'OPTGROUP') return opt.value;
                if (opt.value && !opt.disabled) return opt.value;
            }
            return '';
        })()
    """)
    assert first_value, "No selectable option found"
    page.select_option("#map_viewer_file", value=first_value)
    page.wait_for_function(
        """
        (() => {
            const inst = window.__deckgl_instances?.['map_viewer_map'];
            if (!inst || !inst.lastLayers) return false;
            return inst.lastLayers.some(l => l.id === 'viewer-overlay');
        })()
        """,
        timeout=_TIMEOUT,
    )


def test_map_viewer_empty_without_config(page: Page, app: ShinyAppProc):
    """Without loading a config, Map Viewer should show a hint."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=15_000)
    page.locator(".nav-pills .nav-link[data-value='map_viewer']").click()
    page.wait_for_timeout(2_000)
    hint = page.locator("text=Load a configuration")
    expect(hint).to_be_visible(timeout=_TIMEOUT)
```

- [ ] **Step 2: Run E2E tests**

Run: `.venv/bin/python -m pytest tests/test_e2e_map_viewer.py -v -m e2e`

Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e_map_viewer.py
git commit -m "test: add E2E Playwright tests for Map Viewer page"
```
