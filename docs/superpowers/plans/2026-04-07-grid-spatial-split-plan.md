# Grid / Spatial Results Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split "Grid & Maps" into a config-time "Grid" tab and a post-run "Spatial Results" tab with deck.gl map + Plotly heatmap views.

**Architecture:** The Grid page keeps all existing functionality with a label rename. A new `spatial_results.py` page reads run output NetCDF files via `OsmoseResults`, renders them as colored polygon layers on a deck.gl map (Map View) and as Plotly imshow heatmaps (Flat View). The Results page loses its Spatial Distribution sub-tab. A CSS-based disabled state hides the Spatial Results pill until a run completes.

**Tech Stack:** Shiny for Python, shiny_deckgl (MapWidget/polygon_layer), Plotly, xarray, NumPy

**Spec:** `docs/superpowers/specs/2026-04-07-grid-spatial-split-design.md`

---

### Task 1: Move `_make_legend` and `make_spatial_map` to grid_helpers.py

**Files:**
- Modify: `ui/pages/grid_helpers.py` (add two functions at bottom)
- Modify: `ui/pages/grid.py:81-91` (remove `_make_legend`, import from grid_helpers)
- Modify: `ui/pages/results.py:84-107` (remove `make_spatial_map`, import from grid_helpers)
- Test: `tests/test_grid_helpers.py`

- [ ] **Step 1: Write test for `make_spatial_map` import from grid_helpers**

Add to `tests/test_grid_helpers.py`:

```python
def test_make_spatial_map_importable():
    from ui.pages.grid_helpers import make_spatial_map
    assert callable(make_spatial_map)


def test_make_legend_importable():
    from ui.pages.grid_helpers import make_legend
    assert callable(make_legend)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_grid_helpers.py::test_make_spatial_map_importable tests/test_grid_helpers.py::test_make_legend_importable -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add `make_legend` to grid_helpers.py**

Append to the end of `ui/pages/grid_helpers.py`:

```python
def make_legend(entries: list[dict], **kwargs) -> dict:
    """Create a legend widget/control, adapting to shiny_deckgl version."""
    import shiny_deckgl as _sdgl  # type: ignore[import-untyped]

    if hasattr(_sdgl, "layer_legend_widget"):
        return _sdgl.layer_legend_widget(entries=entries, **kwargs)
    if hasattr(_sdgl, "deck_legend_control"):
        kw = dict(kwargs)
        if "placement" in kw:
            kw["position"] = kw.pop("placement")
        return _sdgl.deck_legend_control(entries=entries, **kw)
    return {}
```

- [ ] **Step 4: Add `make_spatial_map` to grid_helpers.py**

Append to `ui/pages/grid_helpers.py`:

```python
def make_spatial_map(ds, var_name: str, time_idx: int = 0,
                     title: str | None = None,
                     template: str = "osmose"):
    """Create a Plotly imshow heatmap from a spatial xarray Dataset."""
    import plotly.express as px

    data = ds[var_name].isel(time=time_idx).values
    lat = ds["lat"].values
    lon = ds["lon"].values
    fig = px.imshow(
        data,
        x=lon,
        y=lat,
        origin="lower",
        color_continuous_scale="Viridis",
        labels={"x": "Longitude", "y": "Latitude", "color": var_name},
        title=title or f"{var_name} (t={time_idx})",
    )
    fig.update_layout(template=template)
    return fig
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_grid_helpers.py::test_make_spatial_map_importable tests/test_grid_helpers.py::test_make_legend_importable -v`
Expected: PASS

- [ ] **Step 6: Update grid.py to import `make_legend` from grid_helpers**

In `ui/pages/grid.py`, remove the `_make_legend` function (lines 81-91) and add to the imports from grid_helpers:

```python
from ui.pages.grid_helpers import (
    build_grid_layers,
    build_movement_cache,
    build_netcdf_grid_layers,
    list_movement_species,
    list_nc_overlay_variables,
    load_csv_overlay,
    load_mask,
    load_netcdf_grid,
    load_netcdf_overlay,
    make_legend,
    _zoom_for_span,
)
```

Then replace all calls to `_make_legend(` with `make_legend(` in grid.py.

- [ ] **Step 7: Update results.py to import `make_spatial_map` from grid_helpers**

In `ui/pages/results.py`, remove the `make_spatial_map` function (lines 84-107) and add:

```python
from ui.pages.grid_helpers import make_spatial_map
```

- [ ] **Step 8: Run full test suite to verify nothing broke**

Run: `.venv/bin/python -m pytest tests/ -x -q --tb=short`
Expected: All pass (except pre-existing skips)

- [ ] **Step 9: Commit**

```
git add ui/pages/grid_helpers.py ui/pages/grid.py ui/pages/results.py tests/test_grid_helpers.py
git commit -m "refactor: move make_legend and make_spatial_map to grid_helpers for sharing"
```

---

### Task 2: Rename "Grid & Maps" to "Grid" in app.py and update tests

**Files:**
- Modify: `app.py:197`
- Modify: `tests/test_e2e_grid_maps.py` (4 occurrences)
- Modify: `tests/test_e2e_grid_overlay.py` (3 occurrences)

- [ ] **Step 1: Rename tab label in app.py**

In `app.py` line 197, change:

```python
ui.nav_panel("Grid & Maps", grid_ui(), value="grid"),
```

to:

```python
ui.nav_panel("Grid", grid_ui(), value="grid"),
```

- [ ] **Step 2: Update test_e2e_grid_maps.py**

Replace all occurrences of `'Grid & Maps'` with `'Grid'`:

- Line 1 docstring: `"""End-to-end tests for Grid page with the EEC Full dataset."""`
- Line 44 docstring: `"""Navigate to the Grid tab."""`
- Line 45 selector: `page.locator(".nav-pills .nav-link:has-text('Grid')").click()`

**Important:** The `:has-text('Grid')` selector will also match "Grid" in other contexts. Since the tab value is `grid`, use the more specific selector:

```python
page.locator(".nav-pills .nav-link[data-value='grid']").click()
```

Update `_goto_grid()` helper:

```python
def _goto_grid(page: Page) -> None:
    """Navigate to the Grid tab."""
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.wait_for_timeout(2_000)
```

- [ ] **Step 3: Update test_e2e_grid_overlay.py**

In `_load_eec_full_and_goto_grid()`, replace:

```python
    page.locator(".nav-pills .nav-link:has-text('Grid & Maps')").click()
```

with:

```python
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
```

Update docstrings similarly.

- [ ] **Step 4: Run unit tests to verify rename didn't break anything**

Run: `.venv/bin/python -m pytest tests/ -x -q --tb=short`
Expected: All pass

- [ ] **Step 5: Commit**

```
git add app.py tests/test_e2e_grid_maps.py tests/test_e2e_grid_overlay.py
git commit -m "refactor: rename Grid & Maps tab to Grid"
```

---

### Task 3: Add disabled pill CSS

**Files:**
- Modify: `www/osmose.css`

- [ ] **Step 1: Add disabled pill styles to osmose.css**

Append to `www/osmose.css`:

```css
/* Disabled nav pill — used for Spatial Results before a run completes */
.nav-pills .nav-link.osm-disabled {
  opacity: 0.4;
  pointer-events: none;
  cursor: not-allowed;
}
```

- [ ] **Step 2: Commit**

```
git add www/osmose.css
git commit -m "style: add osm-disabled class for greyed-out nav pills"
```

---

### Task 4: Remove Spatial Distribution tab from Results page

**Files:**
- Modify: `ui/pages/results.py`
- Test: existing tests

- [ ] **Step 1: Remove Spatial Distribution nav_panel from results_ui()**

In `ui/pages/results.py`, remove lines 202-219 (the entire `"Spatial Distribution"` nav_panel):

```python
            ui.nav_panel(
                "Spatial Distribution",
                ui.input_slider(
                    "spatial_time_idx",
                    "Time step",
                    min=0,
                    max=1,
                    value=0,
                    step=1,
                    animate=ui.AnimationOptions(
                        interval=1000,
                        loop=True,
                        play_button="Play",
                        pause_button="Pause",
                    ),
                ),
                output_widget("spatial_chart"),
            ),
```

- [ ] **Step 2: Remove `spatial_ds` reactive value from results_server()**

In `results_server()`, remove line 257:

```python
    spatial_ds: reactive.Value = reactive.Value(None)
```

- [ ] **Step 3: Remove spatial loading logic from `_do_load_results()`**

In `_do_load_results()`, remove lines 324-329:

```python
            # Look for NetCDF files for spatial data
            nc_files = [f for f in res.list_outputs() if f.endswith(".nc")]
            if nc_files:
                spatial_ds.set(res.read_netcdf(nc_files[0]))
                max_t = spatial_ds.get().sizes.get("time", 1) - 1
                ui.update_slider("spatial_time_idx", max=max(max_t, 0))
```

- [ ] **Step 4: Remove `spatial_chart` render function from results_server()**

Remove the entire `spatial_chart()` function (lines 553-573):

```python
    @render_plotly
    def spatial_chart():
        tmpl = _tpl(input)
        ds = spatial_ds.get()
        if ds is None:
            return go.Figure().update_layout(
                title="No spatial data loaded",
                template=tmpl,
            )
        time_idx = input.spatial_time_idx()
        # Find a suitable variable (prefer 'biomass')
        var_names = [v for v in ds.data_vars if "lat" in ds[v].dims and "lon" in ds[v].dims]
        if not var_names:
            return go.Figure().update_layout(
                title="No spatial variables found",
                template=tmpl,
            )
        var_name = "biomass" if "biomass" in var_names else var_names[0]
        max_t = ds.sizes.get("time", 1) - 1
        safe_idx = min(time_idx, max_t)
        return make_spatial_map(ds, var_name, time_idx=safe_idx, template=tmpl)
```

- [ ] **Step 5: Remove unused `xr` import if no longer needed**

Check if `xr` (xarray) is still used in results.py after the removal. If not, remove the import.

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q --tb=short`
Expected: All pass

- [ ] **Step 7: Commit**

```
git add ui/pages/results.py
git commit -m "refactor: remove Spatial Distribution tab from Results page

Spatial visualization moves to the new Spatial Results page."
```

---

### Task 5: Create `spatial_results.py` — UI function

**Files:**
- Create: `ui/pages/spatial_results.py`
- Test: `tests/test_spatial_results.py`

- [ ] **Step 1: Write test for spatial_results_ui structure**

Create `tests/test_spatial_results.py`:

```python
"""Unit tests for the Spatial Results page."""


def test_spatial_results_ui_returns_div():
    from ui.pages.spatial_results import spatial_results_ui

    result = spatial_results_ui()
    # Should return a Shiny Tag (div)
    assert hasattr(result, "attrs"), "spatial_results_ui should return a Shiny Tag"


def test_spatial_results_server_callable():
    from ui.pages.spatial_results import spatial_results_server

    assert callable(spatial_results_server)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_spatial_results.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create spatial_results.py with UI function**

Create `ui/pages/spatial_results.py`:

```python
"""Spatial Results page — post-run spatial output visualization."""

from pathlib import Path

from shiny import ui, reactive, render
from shiny.types import SilentException
from shinywidgets import output_widget, render_plotly  # type: ignore[import-untyped]

import numpy as np
import plotly.graph_objects as go

import shiny_deckgl as _sdgl  # type: ignore[import-untyped]
from shiny_deckgl import (  # type: ignore[import-untyped]
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
from osmose.results import OsmoseResults
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.pages.grid_helpers import (
    build_grid_layers,
    build_netcdf_grid_layers,
    load_netcdf_grid,
    make_legend,
    make_spatial_map,
    _zoom_for_span,
)
from ui.state import get_theme_mode

_log = setup_logging("osmose.spatial_results")

# Map NC filename patterns to human-readable labels
_NC_LABELS: dict[str, str] = {
    "biomass": "Biomass",
    "abundance": "Abundance",
    "yield": "Yield",
    "size": "Size",
    "ltl": "LTL",
    "meanTL": "Trophic Level",
}


def _nc_label(filename: str) -> str:
    """Derive a human-readable label from a NetCDF output filename."""
    stem = Path(filename).stem.lower()
    for key, label in _NC_LABELS.items():
        if key.lower() in stem:
            return label
    return Path(filename).stem.replace("_", " ").title()


def spatial_results_ui():
    spatial_map = MapWidget(
        "spatial_map",
        view_state={
            "latitude": 46.0,
            "longitude": -4.5,
            "zoom": 5,
            "pitch": 0,
            "bearing": 0,
        },
        style=CARTO_POSITRON,
        tooltip={
            "html": "({properties.row}, {properties.col})<br>Value: {properties.value}",
            "style": {"fontSize": "12px"},
        },
        controls=[],
    )

    return ui.div(
        expand_tab("Spatial Results", "spatial_results"),
        ui.layout_columns(
            # Controls sidebar
            ui.card(
                collapsible_card_header("Spatial Results", "spatial_results"),
                ui.output_ui("spatial_nc_selector"),
                ui.output_ui("spatial_var_selector"),
                ui.output_ui("spatial_time_controls"),
                ui.output_ui("spatial_scale_info"),
            ),
            # Map / Heatmap area
            ui.navset_card_tab(
                ui.nav_panel(
                    "Map View",
                    ui.div(
                        spatial_map.ui(height="100%"),
                        class_="osm-grid-map-container",
                    ),
                ),
                ui.nav_panel(
                    "Flat View",
                    output_widget("spatial_flat_chart"),
                ),
            ),
            col_widths=[5, 7],
        ),
        class_="osm-split-layout",
        id="split_spatial_results",
    )
```

- [ ] **Step 4: Add stub server function**

Append to `ui/pages/spatial_results.py`:

```python
def spatial_results_server(input, output, session, state):
    """Server logic for the Spatial Results page."""
    pass
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_spatial_results.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```
git add ui/pages/spatial_results.py tests/test_spatial_results.py
git commit -m "feat: add spatial_results page skeleton with UI layout"
```

---

### Task 6: Implement `spatial_results_server` — data loading and controls

**Files:**
- Modify: `ui/pages/spatial_results.py`
- Test: `tests/test_spatial_results.py`

- [ ] **Step 1: Write test for `_nc_label` helper**

Add to `tests/test_spatial_results.py`:

```python
def test_nc_label_biomass():
    from ui.pages.spatial_results import _nc_label

    assert _nc_label("osm_biomass_Sp0.nc") == "Biomass"


def test_nc_label_ltl():
    from ui.pages.spatial_results import _nc_label

    assert _nc_label("osm_ltlBiomass.nc") == "LTL"


def test_nc_label_unknown():
    from ui.pages.spatial_results import _nc_label

    label = _nc_label("osm_custom_output.nc")
    assert label == "Osm Custom Output"
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_spatial_results.py -v`
Expected: PASS (helper is already defined)

- [ ] **Step 3: Implement server function — reactive values and control renderers**

Replace the stub `spatial_results_server` in `ui/pages/spatial_results.py` with:

```python
def spatial_results_server(input, output, session, state):
    """Server logic for the Spatial Results page."""
    _spatial_ds = reactive.Value(None)  # xarray Dataset
    _spatial_nc_files = reactive.Value([])  # list of NC filenames
    _prev_output_dir = reactive.Value(None)

    # Lightweight handle for map updates
    _map = MapWidget(
        "spatial_map",
        view_state={"latitude": 46.0, "longitude": -4.5, "zoom": 5},
        style=CARTO_POSITRON,
    )

    # ── Pill disabled state ──────────────────────────────────────
    @reactive.effect
    async def _toggle_pill():
        out_dir = state.output_dir.get()
        has_results = out_dir is not None and out_dir.exists()
        action = "remove" if has_results else "add"
        await session.send_custom_message(
            "toggle-spatial-pill",
            {"action": action},
        )

    # ── Auto-load NC files when output_dir changes ───────────────
    @reactive.effect
    def _auto_load_spatial():
        out_dir = state.output_dir.get()
        if out_dir is None or not out_dir.exists():
            _spatial_ds.set(None)
            _spatial_nc_files.set([])
            return
        if out_dir == _prev_output_dir.get():
            return
        _prev_output_dir.set(out_dir)

        res = OsmoseResults(out_dir, strict=False)
        nc_files = [f for f in res.list_outputs() if f.endswith(".nc")]
        _spatial_nc_files.set(nc_files)

        if nc_files:
            _spatial_ds.set(res.read_netcdf(nc_files[0]))

    # ── NC file selector ─────────────────────────────────────────
    @render.ui
    def spatial_nc_selector():
        nc_files = _spatial_nc_files.get()
        if not nc_files:
            return ui.p(
                "No spatial outputs available. Run a simulation first.",
                style="color: var(--osm-text-muted); font-size: 12px; margin-top: 8px;",
            )
        choices = {f: _nc_label(f) for f in nc_files}
        return ui.input_select(
            "spatial_result_type", "Output type", choices=choices, selected=nc_files[0]
        )

    # ── Load dataset when NC file selection changes ──────────────
    @reactive.effect
    @reactive.event(input.spatial_result_type)
    def _load_selected_nc():
        out_dir = state.output_dir.get()
        if out_dir is None:
            return
        try:
            filename = input.spatial_result_type()
        except SilentException:
            return
        res = OsmoseResults(out_dir, strict=False)
        _spatial_ds.set(res.read_netcdf(filename))

    # ── Variable (species) selector ──────────────────────────────
    @render.ui
    def spatial_var_selector():
        ds = _spatial_ds.get()
        if ds is None:
            return ui.div()
        # Find data variables with lat/lon dims (spatial fields)
        var_names = [v for v in ds.data_vars if "lat" in ds[v].dims and "lon" in ds[v].dims]
        if len(var_names) <= 1:
            return ui.div()
        choices = {"__all__": "All (sum)"}
        choices.update({v: v.replace("_", " ").title() for v in var_names})
        return ui.input_select(
            "spatial_species", "Variable / Species", choices=choices, selected="__all__"
        )

    # ── Time step slider ─────────────────────────────────────────
    @render.ui
    def spatial_time_controls():
        ds = _spatial_ds.get()
        if ds is None:
            return ui.div()
        n_time = ds.sizes.get("time", 1)
        if n_time <= 1:
            return ui.div()
        return ui.input_slider(
            "spatial_time",
            "Time step",
            min=0,
            max=n_time - 1,
            value=0,
            step=1,
            animate=ui.AnimationOptions(
                interval=500,
                loop=True,
                play_button="Play",
                pause_button="Pause",
            ),
        )

    # ── Color scale info ─────────────────────────────────────────
    @render.ui
    def spatial_scale_info():
        ds = _spatial_ds.get()
        if ds is None:
            return ui.div()
        var_name = _get_var_name(ds, input)
        if var_name is None:
            return ui.div()
        vals = ds[var_name].values
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        return ui.div(
            ui.p(f"Range: {vmin:.4g} \u2014 {vmax:.4g}", style="font-size: 12px; color: var(--osm-text-muted);"),
        )

    # ── Flat View (Plotly heatmap) ───────────────────────────────
    @render_plotly
    def spatial_flat_chart():
        tmpl = "osmose-light" if get_theme_mode(input) == "light" else "osmose"
        ds = _spatial_ds.get()
        if ds is None:
            return go.Figure().update_layout(title="No spatial data loaded", template=tmpl)
        var_name = _get_var_name(ds, input)
        if var_name is None:
            return go.Figure().update_layout(title="No spatial variables found", template=tmpl)
        try:
            time_idx = int(input.spatial_time())
        except (SilentException, ValueError, TypeError):
            time_idx = 0
        max_t = ds.sizes.get("time", 1) - 1
        return make_spatial_map(ds, var_name, time_idx=min(time_idx, max_t), template=tmpl)

    # ── Map View (deck.gl polygon layer) ─────────────────────────
    @reactive.effect
    async def _update_spatial_map():
        ds = _spatial_ds.get()
        is_dark = get_theme_mode(input) == "dark"
        style = CARTO_DARK if is_dark else CARTO_POSITRON
        if style != _map.style:
            _map.style = style
            await _map.set_style(session, style)

        if ds is None:
            await _map.update(session, layers=[], view_state={"latitude": 46.0, "longitude": -4.5, "zoom": 5})
            return

        var_name = _get_var_name(ds, input)
        if var_name is None:
            return

        try:
            time_idx = int(input.spatial_time())
        except (SilentException, ValueError, TypeError):
            time_idx = 0
        max_t = ds.sizes.get("time", 1) - 1
        time_idx = min(time_idx, max_t)

        data_slice = ds[var_name].isel(time=time_idx).values  # (ny, nx)
        lat_raw = ds["lat"].values
        lon_raw = ds["lon"].values

        # Collapse 2D coordinate arrays to 1D (NcGrid outputs may have lat[y,x])
        lat = lat_raw[:, 0] if lat_raw.ndim == 2 else lat_raw
        lon = lon_raw[0, :] if lon_raw.ndim == 2 else lon_raw

        vmin = float(np.nanmin(data_slice))
        vmax = float(np.nanmax(data_slice))
        val_range = vmax - vmin if vmax > vmin else 1.0

        # Cell spacing — hoisted above loop (regular grid assumption)
        ny, nx = data_slice.shape
        dlat = abs(float(lat[1] - lat[0])) if ny > 1 else 1.0
        dlon = abs(float(lon[1] - lon[0])) if nx > 1 else 1.0

        # Build polygon cells with value-mapped colors (Viridis-like)
        cells = []
        for r in range(ny):
            la = float(lat[r])
            for c in range(nx):
                val = float(data_slice[r, c])
                if np.isnan(val) or val == 0:
                    continue
                lo = float(lon[c])
                # Viridis-inspired: blue (low) -> green (mid) -> yellow (high)
                t = (val - vmin) / val_range
                red = int(min(255, max(0, 68 + t * 187)))
                green = int(min(255, max(0, 1 + t * 204)))
                blue = int(min(255, max(0, 84 - t * 84)))
                alpha = 180
                cells.append({
                    "polygon": [
                        [lo - dlon / 2, la - dlat / 2],
                        [lo + dlon / 2, la - dlat / 2],
                        [lo + dlon / 2, la + dlat / 2],
                        [lo - dlon / 2, la + dlat / 2],
                    ],
                    "fill": [red, green, blue, alpha],
                    "value": round(val, 4),
                    "row": r,
                    "col": c,
                })

        layers = []
        if cells:
            layers.append(
                polygon_layer(
                    "spatial-data",
                    data=cells,
                    get_polygon="@@=d.polygon",
                    get_fill_color="@@=d.fill",
                    get_line_color=[0, 0, 0, 0],
                    filled=True,
                    stroked=False,
                    pickable=True,
                )
            )

        legend_entries = []
        if cells:
            legend_entries.append({
                "layer_id": "spatial-data",
                "label": f"{var_name} ({vmin:.2g}\u2013{vmax:.2g})",
                "color": [68, 100, 84],
                "shape": "rect",
            })

        widgets = [
            fullscreen_widget(placement="top-left"),
            zoom_widget(placement="top-right"),
            compass_widget(placement="top-right"),
            scale_widget(placement="bottom-left"),
        ]
        if legend_entries:
            widgets.append(make_legend(legend_entries, placement="bottom-right"))

        # Compute view state from lat/lon bounds
        center_lat = float(np.mean(lat))
        center_lon = float(np.mean(lon))
        lat_span = float(lat.max() - lat.min()) if len(lat) > 1 else 10.0
        lon_span = float(lon.max() - lon.min()) if len(lon) > 1 else 10.0
        span = max(lat_span, lon_span)
        zoom = _zoom_for_span(span)

        view_state = {"latitude": center_lat, "longitude": center_lon, "zoom": zoom}

        await _map.update(session, layers=layers, view_state=view_state, widgets=widgets)
```

- [ ] **Step 4: Add `_get_var_name` helper above the server function**

Add before `spatial_results_server`:

```python
def _get_var_name(ds, input) -> str | None:
    """Get the selected spatial variable name from the dataset."""
    var_names = [v for v in ds.data_vars if "lat" in ds[v].dims and "lon" in ds[v].dims]
    if not var_names:
        return None
    try:
        selected = input.spatial_species()
    except (SilentException, AttributeError):
        selected = "__all__"
    if selected == "__all__" or selected not in var_names:
        # Default: prefer 'biomass', else first spatial var
        return "biomass" if "biomass" in var_names else var_names[0]
    return selected
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_spatial_results.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```
git add ui/pages/spatial_results.py tests/test_spatial_results.py
git commit -m "feat: implement spatial_results_server with data loading and map rendering"
```

---

### Task 7: Wire up Spatial Results in app.py with disabled pill

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add import for spatial_results**

In `app.py`, after the `results` import (line 23), add:

```python
from ui.pages.spatial_results import spatial_results_ui, spatial_results_server
```

- [ ] **Step 2: Add Spatial Results nav panel**

In the navset_pill_list, after the "Results" nav panel (line 204), add:

```python
        ui.nav_panel("Spatial Results", spatial_results_ui(), value="spatial_results"),
```

- [ ] **Step 3: Add server call**

In the `server()` function, after `results_server` call (line 328), add:

```python
    spatial_results_server(input, output, session, state)
```

- [ ] **Step 4: Add JS message handler for pill toggle**

In the end-of-body script section of `app_ui` (after the popover init, before the closing tags), add a new `ui.tags.script` block. Find the last `ui.tags.script("""` in app_ui and add after it:

```python
    # Toggle Spatial Results pill disabled state
    ui.tags.script("""
    (function() {
        // Start disabled
        var poll = setInterval(function() {
            var pill = document.querySelector('.nav-link[data-value="spatial_results"]');
            if (!pill) return;
            clearInterval(poll);
            pill.classList.add('osm-disabled');
        }, 200);

        // Listen for server messages to toggle
        if (typeof Shiny !== 'undefined') {
            Shiny.addCustomMessageHandler('toggle-spatial-pill', function(msg) {
                var pill = document.querySelector('.nav-link[data-value="spatial_results"]');
                if (pill) {
                    if (msg.action === 'add') {
                        pill.classList.add('osm-disabled');
                    } else {
                        pill.classList.remove('osm-disabled');
                    }
                }
            });
        }
    })();
    """),
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q --tb=short`
Expected: All pass

- [ ] **Step 6: Commit**

```
git add app.py
git commit -m "feat: wire Spatial Results page into nav with disabled pill state"
```

---

### Task 8: Update E2E tests for the renamed Grid tab

**Files:**
- Modify: `tests/test_e2e_grid_maps.py`
- Modify: `tests/test_e2e_grid_overlay.py`

- [ ] **Step 1: Run existing e2e tests to establish baseline**

Run: `.venv/bin/python -m pytest tests/test_e2e_grid_maps.py tests/test_e2e_grid_overlay.py -v -m e2e`
Expected: All pass (the rename was done in Task 2, this verifies the data-value selectors work)

- [ ] **Step 2: Commit if any additional fixes needed**

If all pass, no commit needed — Task 2 already handled the rename.

---

### Task 9: Add E2E tests for Spatial Results page

**Files:**
- Create: `tests/test_e2e_spatial_results.py`

- [ ] **Step 1: Create E2E test file**

Create `tests/test_e2e_spatial_results.py`:

```python
"""End-to-end tests for the Spatial Results page.

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_spatial_results.py -v -m e2e

The Spatial Results pill is disabled until a simulation run completes.
These tests verify the disabled state and basic page functionality.
"""

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_LOAD_TIMEOUT = 15_000
_NAV_TIMEOUT = 10_000


def test_spatial_results_pill_disabled_initially(page: Page, app: ShinyAppProc):
    """Spatial Results pill should be disabled before any run."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)

    pill = page.locator(".nav-pills .nav-link[data-value='spatial_results']")
    expect(pill).to_be_visible(timeout=_NAV_TIMEOUT)

    # Should have osm-disabled class
    page.wait_for_timeout(1_000)
    classes = pill.get_attribute("class") or ""
    assert "osm-disabled" in classes, (
        f"Expected 'osm-disabled' in pill classes, got: '{classes}'"
    )


def test_spatial_results_pill_exists_in_execute_section(page: Page, app: ShinyAppProc):
    """Spatial Results pill should appear in the Execute section of navigation."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)

    # The pill should exist
    pill = page.locator(".nav-pills .nav-link[data-value='spatial_results']")
    expect(pill).to_be_visible(timeout=_NAV_TIMEOUT)
    assert pill.text_content().strip() == "Spatial Results"


def test_grid_tab_renamed_to_grid(page: Page, app: ShinyAppProc):
    """Grid tab should be labeled 'Grid' (not 'Grid & Maps')."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)

    grid_pill = page.locator(".nav-pills .nav-link[data-value='grid']")
    expect(grid_pill).to_be_visible(timeout=_NAV_TIMEOUT)
    assert grid_pill.text_content().strip() == "Grid"

    # Old name should not exist
    old_pills = page.locator(".nav-pills .nav-link:has-text('Grid & Maps')")
    assert old_pills.count() == 0, "Old 'Grid & Maps' pill should not exist"
```

- [ ] **Step 2: Run the E2E tests**

Run: `.venv/bin/python -m pytest tests/test_e2e_spatial_results.py -v -m e2e`
Expected: All 3 PASS

- [ ] **Step 3: Commit**

```
git add tests/test_e2e_spatial_results.py
git commit -m "test: add e2e tests for Spatial Results disabled pill and Grid rename"
```

---

### Task 10: Final integration test — run all tests

**Files:** None (verification only)

- [ ] **Step 1: Run full unit test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q --tb=short`
Expected: All pass (minus pre-existing skips)

- [ ] **Step 2: Run all E2E tests together**

Run: `.venv/bin/python -m pytest tests/test_e2e_grid_maps.py tests/test_e2e_grid_overlay.py tests/test_e2e_reactive.py tests/test_e2e_spatial_results.py -v -m e2e`
Expected: All pass

- [ ] **Step 3: Verify no lint issues**

Run: `.venv/bin/ruff check ui/pages/spatial_results.py ui/pages/grid.py ui/pages/results.py ui/pages/grid_helpers.py app.py`
Expected: Clean

- [ ] **Step 4: Final commit if any fixes needed**

Only commit if Step 1-3 revealed issues that needed fixing.
