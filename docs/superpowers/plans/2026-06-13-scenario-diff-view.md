# Scenario Diff View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Scenario Diff" tab to the Results page that shows two runs (baseline A, variant B) side-by-side — overlaid per-species biomass curves plus three spatial maps (A, B, and a B−A difference map).

**Architecture:** Pure data/plotting logic goes in `osmose/` (unit-tested, UI-independent); reactive wiring goes in a new `ui/pages/scenario_diff.py` whose nav-panel builder is embedded in the existing Results navset and whose server is called once from `results_server`. The spatial difference reuses `spatial_slice_2d`; the biomass overlay reuses the existing WIDE/LONG biomass normalization.

**Tech Stack:** Python, xarray/numpy (spatial NetCDF), pandas (biomass CSV), Plotly + shinywidgets (charts), Shiny for Python (reactives), pytest + Playwright (tests).

**Spec:** `docs/superpowers/specs/2026-06-13-scenario-diff-view-design.md`

---

## File Structure

- `osmose/spatial_series.py` — **modify**: add `grid_latlon(ds, variable)` and `spatial_diff_2d(...)`.
- `osmose/analysis.py` — **modify**: add `biomass_long(results)` (shared WIDE/LONG → long normalizer).
- `osmose/plotting.py` — **modify**: add `make_biomass_overlay(long_a, long_b, species, ...)`.
- `ui/pages/grid_helpers.py` — **modify**: add `_z_nan_to_none(data)` (shared serializer) + `make_diff_map(...)`; refactor `make_spatial_map` to use the shared serializer.
- `ui/pages/scenario_diff.py` — **create**: nav-panel builder + server.
- `ui/pages/results.py` — **modify**: embed the nav panel in the Results navset and call the sub-server.
- Tests: `tests/test_spatial_series.py`, `tests/test_analysis.py`, `tests/test_plotting.py`, `tests/test_grid_helpers.py`, `tests/test_ui_results.py`, `tests/test_e2e_scenario_diff.py` (new).

Run all unit tests with: `.venv/bin/python -m pytest <path> -v` (the repo's venv is `.venv`).

---

## Task 1: `spatial_diff_2d` + `grid_latlon` (core diff logic)

**Files:**
- Modify: `osmose/spatial_series.py`
- Test: `tests/test_spatial_series.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spatial_series.py`:

```python
# --- spatial_diff_2d -------------------------------------------------------

from osmose.spatial_series import grid_latlon, spatial_diff_2d  # noqa: E402


def _diff_ds(*, ny=3, nx=4, n_time=2, species=("cod", "sprat"), base=0.0, lat=None, land=None):
    """In-memory (time, species, lat, lon) dataset; cell value is identifiable."""
    lat = np.linspace(54.0, 55.0, ny) if lat is None else np.asarray(lat, dtype=float)
    lon = np.linspace(10.0, 12.0, nx)
    ns = len(species)
    data = np.fromfunction(
        lambda t, s, y, x: base + s * 1000.0 + t * 100.0 + y * 10.0 + x,
        (n_time, ns, ny, nx),
        dtype=float,
    )
    if land is not None:
        ly, lx = land
        data[:, :, ly, lx] = np.nan
    return xr.Dataset(
        {"spatial_biomass": (("time", "species", "lat", "lon"), data)},
        coords={
            "time": np.arange(n_time) / 12.0,
            "species": list(species),
            "lat": lat,
            "lon": lon,
        },
    )


def test_spatial_diff_2d_sum_over_species():
    a = _diff_ds(base=0.0)
    b = _diff_ds(base=7.0)  # every cell of B is A + 7 per species; 2 species -> +14 summed
    diff = spatial_diff_2d(a, b, "spatial_biomass", time_a=0, time_b=0)
    assert diff.shape == (3, 4)
    np.testing.assert_allclose(diff, np.full((3, 4), 14.0))


def test_spatial_diff_2d_land_nan_propagates():
    a = _diff_ds(land=(0, 0))
    b = _diff_ds(base=7.0)  # B has no land at (0,0); A does -> result NaN there
    diff = spatial_diff_2d(a, b, "spatial_biomass")
    assert np.isnan(diff[0, 0])
    assert np.isfinite(diff[1, 1])


def test_spatial_diff_2d_single_species_by_name():
    a = _diff_ds(base=0.0)
    b = _diff_ds(base=7.0)
    diff = spatial_diff_2d(a, b, "spatial_biomass", species="sprat")
    np.testing.assert_allclose(diff, np.full((3, 4), 7.0))  # one species -> +7


def test_spatial_diff_2d_time_indices_independent():
    a = _diff_ds(base=0.0)  # value includes t*100
    b = _diff_ds(base=0.0)
    # B at t=1 minus A at t=0, summed over 2 species: each species differs by +100 -> +200
    diff = spatial_diff_2d(a, b, "spatial_biomass", time_a=0, time_b=1)
    np.testing.assert_allclose(diff, np.full((3, 4), 200.0))


def test_spatial_diff_2d_identical_runs_all_zero():
    a = _diff_ds(land=(0, 0))
    diff = spatial_diff_2d(a, a, "spatial_biomass")
    assert np.isnan(diff[0, 0])
    finite = diff[np.isfinite(diff)]
    np.testing.assert_allclose(finite, 0.0)


def test_spatial_diff_2d_shape_mismatch_raises():
    a = _diff_ds(nx=4)
    b = _diff_ds(nx=5)
    with pytest.raises(ValueError, match="shape"):
        spatial_diff_2d(a, b, "spatial_biomass")


def test_spatial_diff_2d_coord_mismatch_raises():
    a = _diff_ds(lat=[54.0, 54.5, 55.0])
    b = _diff_ds(lat=[60.0, 60.5, 61.0])  # same shape, different coords
    with pytest.raises(ValueError, match="coordinate"):
        spatial_diff_2d(a, b, "spatial_biomass")


def test_spatial_diff_2d_int_species_rejected():
    a = _diff_ds()
    with pytest.raises(TypeError, match="name"):
        spatial_diff_2d(a, a, "spatial_biomass", species=1)


def test_grid_latlon_returns_coord_arrays():
    a = _diff_ds()
    lat, lon = grid_latlon(a, "spatial_biomass")
    np.testing.assert_allclose(lat, np.linspace(54.0, 55.0, 3))
    np.testing.assert_allclose(lon, np.linspace(10.0, 12.0, 4))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_spatial_series.py -k "diff_2d or grid_latlon" -v`
Expected: FAIL with `ImportError: cannot import name 'grid_latlon'` / `'spatial_diff_2d'`.

- [ ] **Step 3: Implement `grid_latlon` and `spatial_diff_2d`**

Append to `osmose/spatial_series.py` (after `spatial_slice_2d`):

```python
def grid_latlon(ds, variable):
    """Return ``(lat, lon)`` coordinate arrays for ``variable`` via the dim-name sets.

    Locates the lat/lon dims with the same alias-tolerant lookup ``spatial_slice_2d``
    uses, so callers (e.g. the map renderer) never hardcode ``"lat"``/``"lon"``.
    """
    dims = ds[variable].dims
    lat_dim = _find_dim(dims, _LAT_DIM_NAMES)
    lon_dim = _find_dim(dims, _LON_DIM_NAMES)
    if lat_dim is None or lon_dim is None:
        raise ValueError(f"variable {variable!r} has no lat/lon dims (have {tuple(dims)})")
    return np.asarray(ds[lat_dim].values), np.asarray(ds[lon_dim].values)


def spatial_diff_2d(ds_a, ds_b, variable, *, time_a=0, time_b=0, species=None, reduce="sum"):
    """``B − A`` as a 2-D ``(lat, lon)`` array; NaN where either side is land/missing.

    ``species`` and ``reduce`` mirror :func:`spatial_slice_2d`. For a diff, ``species``
    must be a name (str) or ``None`` — never an int index, since index *i* can denote a
    different species in each run. Grids must correspond cell-to-cell: shapes must match
    AND lat/lon coordinates must be exactly equal (the engine writes them as unpacked
    float64, so identical grids roundtrip bit-exactly).

    Raises
    ------
    TypeError
        ``species`` is a non-str, non-None value.
    ValueError
        Slice shapes differ, or lat/lon coordinates differ.
    """
    if species is not None and not isinstance(species, str):
        raise TypeError("species must be a name (str) or None for a diff (int index unsafe)")
    a = spatial_slice_2d(ds_a, variable, time_index=time_a, species=species, reduce=reduce)
    b = spatial_slice_2d(ds_b, variable, time_index=time_b, species=species, reduce=reduce)
    if a.shape != b.shape:
        raise ValueError(f"grid shapes differ: A {a.shape} vs B {b.shape}")
    lat_a, lon_a = grid_latlon(ds_a, variable)
    lat_b, lon_b = grid_latlon(ds_b, variable)
    if not (np.array_equal(lat_a, lat_b) and np.array_equal(lon_a, lon_b)):
        raise ValueError("grid coordinates differ — cannot diff cell-to-cell")
    return b - a
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_spatial_series.py -k "diff_2d or grid_latlon" -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add osmose/spatial_series.py tests/test_spatial_series.py
git commit -m "feat(spatial): add spatial_diff_2d + grid_latlon for scenario diff"
```

---

## Task 2: `make_diff_map` + shared NaN serializer

**Files:**
- Modify: `ui/pages/grid_helpers.py`
- Test: `tests/test_grid_helpers.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_grid_helpers.py`:

```python
# --- make_diff_map ---------------------------------------------------------

import numpy as np  # noqa: E402

from ui.pages.grid_helpers import make_diff_map  # noqa: E402


def test_make_diff_map_symmetric_range():
    data = np.array([[5.0, -3.0], [0.0, 2.0]])
    fig = make_diff_map(data, [54.0, 55.0], [10.0, 11.0], var_name="biomass")
    assert fig.layout.coloraxis.cmin == -5.0
    assert fig.layout.coloraxis.cmax == 5.0


def test_make_diff_map_all_nan_returns_empty_state():
    data = np.full((2, 2), np.nan)
    fig = make_diff_map(data, [54.0, 55.0], [10.0, 11.0], var_name="biomass")
    assert "no data" in fig.layout.title.text.lower()


def test_make_diff_map_all_zero_valid_range():
    data = np.zeros((2, 2))
    fig = make_diff_map(data, [54.0, 55.0], [10.0, 11.0], var_name="biomass")
    # EPS floor keeps a finite, symmetric range rather than zmin==zmax==0
    assert fig.layout.coloraxis.cmin < 0 < fig.layout.coloraxis.cmax
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_grid_helpers.py -k make_diff_map -v`
Expected: FAIL with `ImportError: cannot import name 'make_diff_map'`.

- [ ] **Step 3: Implement the shared serializer and `make_diff_map`; refactor `make_spatial_map`**

In `ui/pages/grid_helpers.py`, add this helper directly above `make_spatial_map`:

```python
def _z_nan_to_none(data):
    """Nested-list copy of a 2-D array with non-finite cells → ``None``.

    shinywidgets serialises figures with ``json(allow_nan=False)``, which rejects
    NaN/inf; plotly renders ``None`` as a gap, which is what we want for land cells.
    """
    import numpy as np

    return [[(float(v) if np.isfinite(v) else None) for v in row] for row in data]
```

In `make_spatial_map`, replace the inline z-builder line:

```python
    z = [[(float(v) if np.isfinite(v) else None) for v in row] for row in data]
```

with:

```python
    z = _z_nan_to_none(data)
```

Then add `make_diff_map` directly below `make_spatial_map`:

```python
def make_diff_map(
    diff_array,
    lat,
    lon,
    *,
    var_name: str,
    title: str | None = None,
    template: str = "osmose",
):
    """Diverging RdBu map of a precomputed ``B − A`` diff array, centered at zero.

    Array-input sibling of :func:`make_spatial_map` (which is dataset-input). The
    caller supplies ``lat``/``lon`` (use ``osmose.spatial_series.grid_latlon`` so the
    axes match the diff array). Returns an empty-state figure when no cell is finite;
    an EPS floor keeps a valid colorbar when every finite cell is 0 (identical runs).
    """
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    data = np.asarray(diff_array, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return go.Figure().update_layout(
            title=dict(text=title or f"Δ {var_name} (no data)"), template=template
        )
    half = max(float(np.abs(finite).max()), 1e-9)
    fig = px.imshow(
        _z_nan_to_none(data),
        x=np.asarray(lon),
        y=np.asarray(lat),
        origin="lower",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0.0,
        zmin=-half,
        zmax=half,
        labels={"x": "Longitude", "y": "Latitude", "color": f"Δ {var_name}"},
        title=title or f"Δ {var_name} (B−A)",
    )
    fig.update_layout(template=template)
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_grid_helpers.py -k "make_diff_map or make_spatial_map" -v`
Expected: PASS (the 3 new tests, and any existing `make_spatial_map` tests still pass after the refactor).

- [ ] **Step 5: Commit**

```bash
git add ui/pages/grid_helpers.py tests/test_grid_helpers.py
git commit -m "feat(ui): add make_diff_map + shared NaN→None serializer"
```

---

## Task 3: `biomass_long` + `make_biomass_overlay`

**Files:**
- Modify: `osmose/analysis.py`, `osmose/plotting.py`
- Test: `tests/test_analysis.py`, `tests/test_plotting.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_analysis.py`:

```python
# --- biomass_long ----------------------------------------------------------

import pandas as pd  # noqa: E402

from osmose.analysis import biomass_long  # noqa: E402


class _FakeResults:
    def __init__(self, df):
        self._df = df

    def biomass(self):
        return self._df


def test_biomass_long_from_wide():
    # WIDE: Time + per-species cols + a constant `species` column (="all"), as on disk
    wide = pd.DataFrame(
        {"Time": [0.0, 1.0], "cod": [10.0, 12.0], "sprat": [5.0, 6.0], "species": ["all", "all"]}
    )
    out = biomass_long(_FakeResults(wide))
    assert set(out.columns) == {"time", "species", "value"}
    assert set(out["species"]) == {"cod", "sprat"}
    cod = out[(out["species"] == "cod") & (out["time"] == 1.0)]
    assert cod["value"].iloc[0] == 12.0


def test_biomass_long_from_long_passthrough():
    long = pd.DataFrame(
        {"time": [0.0, 0.0], "species": ["cod", "sprat"], "value": [10.0, 5.0]}
    )
    out = biomass_long(_FakeResults(long))
    assert set(out.columns) == {"time", "species", "value"}
    assert len(out) == 2


def test_biomass_long_empty():
    out = biomass_long(_FakeResults(pd.DataFrame()))
    assert list(out.columns) == ["time", "species", "value"]
    assert len(out) == 0
```

Append to `tests/test_plotting.py`:

```python
# --- make_biomass_overlay --------------------------------------------------

import pandas as pd  # noqa: E402

from osmose.plotting import make_biomass_overlay  # noqa: E402


def _long(values_by_species):
    rows = []
    for sp, vals in values_by_species.items():
        for t, v in enumerate(vals):
            rows.append({"time": float(t), "species": sp, "value": float(v)})
    return pd.DataFrame(rows)


def test_make_biomass_overlay_trace_count_and_dash():
    a = _long({"cod": [10, 11], "sprat": [5, 6]})
    b = _long({"cod": [12, 13], "sprat": [4, 3]})
    fig = make_biomass_overlay(a, b, ["cod", "sprat"])
    # 2 species x (A solid + B dashed) = 4 traces
    assert len(fig.data) == 4
    dashed = [tr for tr in fig.data if tr.line.dash == "dash"]
    assert len(dashed) == 2  # the B traces


def test_make_biomass_overlay_empty_species_returns_empty_figure():
    a = _long({"cod": [10, 11]})
    fig = make_biomass_overlay(a, a, [])
    assert len(fig.data) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_analysis.py -k biomass_long tests/test_plotting.py -k biomass_overlay -v`
Expected: FAIL with `ImportError` for `biomass_long` / `make_biomass_overlay`.

- [ ] **Step 3a: Implement `biomass_long`**

Append to `osmose/analysis.py`:

```python
def biomass_long(results) -> pd.DataFrame:
    """Normalize a run's biomass output to long form: columns ``time, species, value``.

    Handles both on-disk shapes the same way :func:`_per_species_window_mean` does:
    WIDE (``Time`` + one column per species + a constant ``species`` column) and LONG
    (``time, species, value``). Returns an empty 3-column frame for a missing/empty run
    (e.g. ``OsmoseResults(dir, strict=False)`` with no files), so callers degrade to an
    empty state rather than raising.
    """
    df = results.biomass()
    empty = pd.DataFrame(columns=["time", "species", "value"])
    if df is None or len(df) == 0:
        return empty
    cols = set(df.columns)
    if "value" in cols and "species" in cols:  # LONG
        time_col = "time" if "time" in cols else "Time"
        return (
            df[[time_col, "species", "value"]]
            .rename(columns={time_col: "time"})
            .reset_index(drop=True)
        )
    # WIDE: species are the non-Time/non-species columns
    time_col = "Time" if "Time" in cols else "time"
    species_cols = [c for c in df.columns if c not in _NON_SPECIES_COLS]
    if not species_cols:
        return empty
    melted = df.melt(
        id_vars=[time_col], value_vars=species_cols, var_name="species", value_name="value"
    )
    return melted.rename(columns={time_col: "time"}).reset_index(drop=True)
```

- [ ] **Step 3b: Implement `make_biomass_overlay`**

Append to `osmose/plotting.py`:

```python
# Paired colors so each species' A (solid) and B (dashed) traces match.
_OVERLAY_PALETTE = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)


def make_biomass_overlay(
    long_a: pd.DataFrame,
    long_b: pd.DataFrame,
    species: Sequence[str],
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> go.Figure:
    """Overlay per-species biomass trajectories from two runs (A solid, B dashed).

    Inputs are already-normalized long frames (``time, species, value``) from
    :func:`osmose.analysis.biomass_long` — this never re-derives the WIDE/LONG shape.
    """
    title = "Scenario biomass: A vs B"
    if not species:
        return _empty_figure(title)
    fig = go.Figure()
    for i, sp in enumerate(species):
        color = _OVERLAY_PALETTE[i % len(_OVERLAY_PALETTE)]
        a = long_a[long_a["species"] == sp].sort_values("time")
        b = long_b[long_b["species"] == sp].sort_values("time")
        if len(a):
            fig.add_trace(
                go.Scatter(
                    x=a["time"], y=a["value"], mode="lines",
                    name=f"{sp} ({label_a})", line=dict(color=color),
                )
            )
        if len(b):
            fig.add_trace(
                go.Scatter(
                    x=b["time"], y=b["value"], mode="lines",
                    name=f"{sp} ({label_b})", line=dict(color=color, dash="dash"),
                )
            )
    fig.update_layout(
        title=dict(text=title), xaxis_title="time", yaxis_title="biomass", template=TEMPLATE
    )
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_analysis.py -k biomass_long tests/test_plotting.py -k biomass_overlay -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add osmose/analysis.py osmose/plotting.py tests/test_analysis.py tests/test_plotting.py
git commit -m "feat: add biomass_long normalizer + make_biomass_overlay chart"
```

---

## Task 4: `scenario_diff.py` — nav panel + server

**Files:**
- Create: `ui/pages/scenario_diff.py`
- Test: (covered by Task 5 structure test + Task 6 e2e)

This task wires the reactives. No new unit test is added here (the pure logic is already tested in Tasks 1–3; UI wiring is verified by the Task 5 structure test and Task 6 e2e). Build it, then verify the app imports cleanly.

- [ ] **Step 1: Create the module**

Create `ui/pages/scenario_diff.py`:

```python
"""Scenario Diff tab — side-by-side biomass + spatial maps for two runs.

Embedded as a tab in the Results page: ``scenario_diff_nav_panel()`` is added to the
Results navset and ``scenario_diff_server(...)`` is called once from ``results_server``.
This sub-server-in-a-page pattern is new (other pages are top-level), chosen to keep the
already-large ``results_server`` from growing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from shiny import reactive, render, ui
from shiny.types import SilentException
from shinywidgets import output_widget, render_plotly

from osmose.analysis import biomass_long, run_delta
from osmose.history import default_run_history
from osmose.logging import setup_logging
from osmose.plotting import make_biomass_overlay
from osmose.results import OsmoseResults
from osmose.spatial_series import grid_latlon, spatial_diff_2d
from ui.pages.grid_helpers import make_diff_map, make_spatial_map

_log = setup_logging("osmose.scenario_diff")

_SPATIAL_VAR_HINT = "spatial_biomass"  # preferred diff variable when present


def _run_choices(runs) -> dict[str, str]:
    return {r.timestamp: f"{r.timestamp[:19]} ({r.duration_sec:.0f}s)" for r in runs}


def _tpl(input) -> str:
    from ui.state import get_theme_mode

    return "osmose" if get_theme_mode(input) == "dark" else "osmose-light"


def scenario_diff_nav_panel():
    """The 'Scenario Diff' nav panel for embedding in the Results navset."""
    return ui.nav_panel(
        "Scenario Diff",
        ui.layout_columns(
            ui.div(
                ui.input_select("diff_run_a", "Baseline (A)", choices={}),
                ui.input_select("diff_run_b", "Variant (B)", choices={}),
                ui.input_selectize(
                    "diff_species", "Biomass species", choices={}, multiple=True
                ),
                ui.input_slider("diff_window_years", "Caption window (years)", min=1, max=30, value=10),
            ),
            col_widths=[12],
        ),
        output_widget("diff_biomass_chart"),
        ui.output_ui("diff_biomass_caption"),
        ui.hr(),
        ui.output_ui("diff_spatial_controls"),
        ui.output_ui("diff_cadence_warning"),
        ui.output_ui("diff_spatial_status"),
        # Map widgets are declared STATICALLY (shinywidgets binds on DOM insertion;
        # injecting output_widget inside a @render.ui leaves them blank). Each render
        # function returns an empty-state figure when its dataset is absent.
        ui.layout_columns(
            output_widget("diff_map_a"),
            output_widget("diff_map_b"),
            output_widget("diff_map_delta"),
            col_widths=[4, 4, 4],
        ),
    )


def scenario_diff_server(input, output, session, state):
    """Reactives for the Scenario Diff tab. Called once from results_server."""
    _ds_a: reactive.Value = reactive.Value(None)  # xarray Dataset | None
    _ds_b: reactive.Value = reactive.Value(None)
    _shared_ds: reactive.Value[bool] = reactive.Value(False)  # A and B are the SAME handle
    _last_choices: reactive.Value[dict] = reactive.Value({})
    _last_species: reactive.Value[dict] = reactive.Value({})

    # NB: render functions use BARE @render_plotly / @render.ui (no @output) to match
    # the host pages (ui/pages/results.py, ui/pages/spatial_results.py). The `output`
    # server param is unused but kept in the signature (the caller passes it positionally).

    def _safe(getter, default=None):
        # Catch AttributeError too: dynamic inputs (diff_spatial_species, diff_time)
        # are created inside a @render.ui, so a static map render can read them before
        # they register. This matches ui/pages/spatial_results.py:100,316,457.
        try:
            return getter()
        except (SilentException, AttributeError):
            return default

    # ── Populate run selectors from history when on the Results tab ──
    @reactive.effect
    def _populate_diff_runs():
        if input.main_nav() != "results":
            return
        try:
            runs = default_run_history().list_runs()
        except Exception:  # noqa: BLE001 — never crash the page on a history-read error
            return
        choices = _run_choices(runs)
        with reactive.isolate():
            if choices == _last_choices.get():
                return
        _last_choices.set(choices)
        ui.update_select("diff_run_a", choices=choices)
        ui.update_select("diff_run_b", choices=choices)

    # ── Resolve a selected timestamp → OsmoseResults ──
    def _results_for(ts):
        if not ts:
            return None
        try:
            rec = default_run_history().load_run(ts)
        except Exception:  # noqa: BLE001
            return None
        return OsmoseResults(Path(rec.output_dir), strict=False)

    # ── Spatial NetCDF lifecycle ──
    # ONE effect opens both runs, deduping the same-run case: opening the SAME .nc path
    # twice would trigger the documented HDF5-locking error, and selecting one run for
    # both A and B is a supported case (the "identical runs" caption). When A == B we
    # share a single handle and never double-close it. Handles are opened with
    # xr.open_dataset (the holder owns the only reference — no cache-then-close smell).
    def _close_one(ds):
        if ds is not None:
            try:
                ds.close()
            except Exception:  # noqa: BLE001
                _log.warning("Failed to close scenario-diff dataset", exc_info=True)

    def _close_handles():
        """Close held handles WITHOUT touching reactive values (safe at session end)."""
        a = _ds_a.get()
        _close_one(a)
        if not _shared_ds.get():
            _close_one(_ds_b.get())

    def _open_one(res):
        if res is None or res.output_dir is None:
            return None
        try:
            nc = [f for f in res.list_outputs() if f.endswith(".nc")]
        except (OSError, ValueError, KeyError):
            return None
        if not nc:
            return None
        try:
            return xr.open_dataset(Path(res.output_dir) / nc[0])
        except (OSError, ValueError, KeyError) as exc:
            _log.error("Failed to open spatial output: %s", exc, exc_info=True)
            return None

    @reactive.effect
    def _load_spatial():
        ts_a = _safe(input.diff_run_a)
        ts_b = _safe(input.diff_run_b)
        with reactive.isolate():  # close prior handles without depending on them
            _close_handles()
        ds_a = _open_one(_results_for(ts_a))
        if ts_a and ts_a == ts_b:
            ds_b = ds_a  # same run → share the single handle (no double-open)
            shared = True
        else:
            ds_b = _open_one(_results_for(ts_b))
            shared = False
        _shared_ds.set(shared)
        _ds_a.set(ds_a)
        _ds_b.set(ds_b)

    # Close handles when the session ends (we may hold two long-lived datasets).
    def _on_session_end():
        with reactive.isolate():
            _close_handles()

    session.on_ended(_on_session_end)

    # ── Biomass long frames (reactive) ──
    @reactive.calc
    def _long_a():
        res = _results_for(_safe(input.diff_run_a))
        return biomass_long(res) if res is not None else None

    @reactive.calc
    def _long_b():
        res = _results_for(_safe(input.diff_run_b))
        return biomass_long(res) if res is not None else None

    # ── Biomass species selector population (common to both runs) ──
    @reactive.effect
    def _populate_diff_species():
        la, lb = _long_a(), _long_b()
        if la is None or lb is None:
            return
        common = sorted(set(la["species"]) & set(lb["species"]))
        choices = {s: s for s in common}
        with reactive.isolate():
            # Changed-only guard: don't re-run update_selectize (and clobber the user's
            # manual selection) when the common species set hasn't changed.
            if choices == _last_species.get():
                return
            current = _safe(input.diff_species, ()) or ()
        _last_species.set(choices)
        keep = [s for s in current if s in choices]
        ui.update_selectize("diff_species", choices=choices, selected=keep or common[:3])

    # ── Biomass overlay chart ──
    @render_plotly
    def diff_biomass_chart():
        la, lb = _long_a(), _long_b()
        if la is None or lb is None:
            return go.Figure().update_layout(
                title="Select two runs to compare", template=_tpl(input)
            )
        species = list(_safe(input.diff_species, ()) or ())
        fig = make_biomass_overlay(la, lb, species)
        fig.update_layout(template=_tpl(input))
        return fig

    # ── Biomass caption (mean B−A over the trailing window) ──
    @render.ui
    def diff_biomass_caption():
        ts_a, ts_b = _safe(input.diff_run_a), _safe(input.diff_run_b)
        ra, rb = _results_for(ts_a), _results_for(ts_b)
        if ra is None or rb is None:
            return ui.div()
        if ts_a == ts_b:
            return ui.p("Identical runs (A = B).", class_="text-muted")
        try:
            deltas = run_delta(ra, rb, metric="biomass", window_years=int(input.diff_window_years()))
        except (ValueError, KeyError):
            return ui.div()
        species = set(_safe(input.diff_species, ()) or ())
        rows = [d for d in deltas if not species or d.species in species]
        if not rows:
            return ui.div()
        items = [
            ui.tags.li(f"{d.species}: ΔB = {d.abs_delta:+.3g}")
            for d in rows
        ]
        return ui.div(ui.p("Mean B−A over trailing window:"), ui.tags.ul(*items))

    # ── Spatial: common species + variable helpers ──
    def _has_latlon(ds, v):
        dims = {str(d) for d in ds[v].dims}
        return "lat" in dims and "lon" in dims

    def _spatial_var(ds):
        candidates = [v for v in ds.data_vars if _has_latlon(ds, v)]
        if _SPATIAL_VAR_HINT in candidates:
            return _SPATIAL_VAR_HINT
        return candidates[0] if candidates else None

    def _common_species(ds_a, ds_b):
        sa = {str(s) for s in ds_a["species"].values} if "species" in ds_a.coords else set()
        sb = {str(s) for s in ds_b["species"].values} if "species" in ds_b.coords else set()
        return sorted(sa & sb)

    def _has_species_dim(ds, var):
        return "species" in {str(d) for d in ds[var].dims}

    def _spatial_empty_reason(a, b):
        """Message if the spatial maps can't be rendered comparably, else None.

        Crucially guards the summed path when there is NO common species: feeding
        ``ds.sel(species=[])`` into ``spatial_slice_2d``'s ``sum(skipna=False)`` over a
        zero-length dim yields 0.0 everywhere (silently destroying the land-NaN mask),
        so we short-circuit to an empty state instead.
        """
        if a is None or b is None:
            return "No spatial output — enable output.spatial in both configs."
        var_a, var_b = _spatial_var(a), _spatial_var(b)
        if var_a is None or var_b is None:
            return "No spatial variable in one of the runs."
        if _spatial_species() is None:  # "All (summed)" path
            if (_has_species_dim(a, var_a) or _has_species_dim(b, var_b)) and not _common_species(a, b):
                return "No common species for spatial maps."
        return None

    # ── Spatial controls (species + time), rendered dynamically ──
    # (Dynamic input_select/input_slider inside @render.ui is fine — only shinywidgets
    # output_widget must be static; see the nav panel.)
    @render.ui
    def diff_spatial_controls():
        a, b = _ds_a.get(), _ds_b.get()
        if a is None or b is None:
            return ui.div()
        common = _common_species(a, b)
        choices = {"__sum__": "All (summed)"}
        choices.update({s: s for s in common})
        # Overlapping time range by VALUE; slider indexes into a shared 0..N-1 fraction
        n_a = int(a.sizes.get("time", 1))
        n_b = int(b.sizes.get("time", 1))
        n = min(n_a, n_b)
        return ui.div(
            ui.input_select("diff_spatial_species", "Map species", choices=choices, selected="__sum__"),
            ui.input_slider("diff_time", "Time step", min=0, max=max(n - 1, 0), value=0, step=1),
        )

    # ── Cadence-mismatch warning (spec: warn if n_dt_per_year differs) ──
    @render.ui
    def diff_cadence_warning():
        a, b = _ds_a.get(), _ds_b.get()
        if a is None or b is None:
            return ui.div()
        na = a.attrs.get("n_dt_per_year")
        nb = b.attrs.get("n_dt_per_year")
        if na is not None and nb is not None and na != nb:
            return ui.p(
                f"⚠ Runs have different time cadence ({na} vs {nb} steps/year); "
                "maps are aligned by nearest time value.",
                class_="text-warning",
            )
        return ui.div()

    # ── Spatial empty-state message (maps render empty figures alongside) ──
    @render.ui
    def diff_spatial_status():
        reason = _spatial_empty_reason(_ds_a.get(), _ds_b.get())
        return ui.p(reason, class_="text-muted") if reason else ui.div()

    def _spatial_species():
        sel = _safe(input.diff_spatial_species, "__sum__")
        return None if sel in ("__sum__", None) else sel

    def _time_indices(a, b):
        """Nearest indices in A and B for the chosen overlapping-time value."""
        ta = np.asarray(a["time"].values)
        tb = np.asarray(b["time"].values)
        lo = max(float(ta.min()), float(tb.min()))
        hi = min(float(ta.max()), float(tb.max()))
        idx = int(_safe(input.diff_time, 0) or 0)
        # Map the integer slider position onto the overlapping value range, then snap.
        n = min(len(ta), len(tb))
        frac = idx / max(n - 1, 1)
        v = lo + frac * (hi - lo)
        return int(np.abs(ta - v).argmin()), int(np.abs(tb - v).argmin())

    # ── Three spatial maps (static widgets; empty-state figure when no data) ──
    def _subset_for_sum(ds, var, common):
        """Narrow a dataset to the common species (so 'All (summed)' is comparable).

        Only reached when ``common`` is non-empty — ``_spatial_empty_reason`` short-
        circuits the empty case before any map computes, so ``ds.sel(species=[])``
        (which would zero the grid and destroy the land-NaN mask) never runs.
        """
        if _has_species_dim(ds, var):
            return ds.sel(species=common)
        return ds

    def _state_fig(msg):
        return go.Figure().update_layout(title=msg, template=_tpl(input))

    @render_plotly
    def diff_map_a():
        return _one_side_map(_ds_a.get(), _ds_b.get(), which="a")

    @render_plotly
    def diff_map_b():
        return _one_side_map(_ds_a.get(), _ds_b.get(), which="b")

    def _one_side_map(a, b, *, which):
        reason = _spatial_empty_reason(a, b)
        if reason:
            return _state_fig(reason)
        var = _spatial_var(a)
        common = _common_species(a, b)
        sp = _spatial_species()
        ti_a, ti_b = _time_indices(a, b)
        ds = a if which == "a" else b
        ti = ti_a if which == "a" else ti_b
        label = "A" if which == "a" else "B"
        if sp is None:
            ds = _subset_for_sum(ds, var, common)
        try:
            fig = make_spatial_map(ds, var, time_idx=ti, species=sp, title=f"{label}: {var}")
        except (ValueError, KeyError) as exc:
            return _state_fig(f"Cannot render {label}: {exc}")
        fig.update_layout(template=_tpl(input))
        return fig

    @render_plotly
    def diff_map_delta():
        a, b = _ds_a.get(), _ds_b.get()
        reason = _spatial_empty_reason(a, b)
        if reason:
            return _state_fig(reason)
        var = _spatial_var(a)
        common = _common_species(a, b)
        sp = _spatial_species()
        ti_a, ti_b = _time_indices(a, b)
        ds_a, ds_b = a, b
        if sp is None:
            ds_a = _subset_for_sum(a, var, common)
            ds_b = _subset_for_sum(b, var, common)
        try:
            diff = spatial_diff_2d(ds_a, ds_b, var, time_a=ti_a, time_b=ti_b, species=sp)
        except (ValueError, TypeError, KeyError) as exc:
            return _state_fig(f"Cannot diff: {exc}")
        lat, lon = grid_latlon(ds_a, var)
        return make_diff_map(diff, lat, lon, var_name=var, template=_tpl(input))
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `.venv/bin/python -c "import ui.pages.scenario_diff as m; print(m.scenario_diff_nav_panel is not None and m.scenario_diff_server is not None)"`
Expected: prints `True` (no ImportError).

- [ ] **Step 3: Commit**

```bash
git add ui/pages/scenario_diff.py
git commit -m "feat(ui): add scenario_diff tab module (nav panel + server)"
```

---

## Task 5: Wire the tab into Results + structure test

**Files:**
- Modify: `ui/pages/results.py`
- Test: `tests/test_ui_results.py`

- [ ] **Step 1: Write the failing structure test**

Append to `tests/test_ui_results.py`:

```python
def test_scenario_diff_tab_wired_into_results():
    """The Scenario Diff tab + sub-server are embedded in the Results page."""
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parent.parent / "ui" / "pages" / "results.py"
    ).read_text()
    assert "scenario_diff_nav_panel" in src
    assert "scenario_diff_server" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py -k scenario_diff_tab_wired -v`
Expected: FAIL (`scenario_diff_nav_panel` not in `results.py`).

- [ ] **Step 3: Add the import**

In `ui/pages/results.py`, after the existing `from ui.state import AppState` import (line ~17), add:

```python
from ui.pages.scenario_diff import scenario_diff_nav_panel, scenario_diff_server
```

- [ ] **Step 4: Embed the nav panel in the Results navset**

In `ui/pages/results.py`, the `navset_card_tab(...)` ends after the "Compare Runs" `ui.nav_panel(...)` (the closing `),` of that nav panel, immediately before the navset's own closing `),` at line ~336). Add the new tab as the last child of the navset — insert this line right after the "Compare Runs" nav_panel's closing `),`:

```python
            scenario_diff_nav_panel(),
```

The navset tail should read:

```python
                output_widget("run_delta_chart"),
                ui.output_ui("run_delta_table"),
            ),
            scenario_diff_nav_panel(),
        ),
    )
```

- [ ] **Step 5: Call the sub-server**

In `ui/pages/results.py`, at the very end of the `results_server(...)` function body, add:

```python
    scenario_diff_server(input, output, session, state)
```

Place it as the last statement inside `results_server` (same indentation as the other top-level statements in that function).

- [ ] **Step 6: Run the structure test + a broad import/smoke check**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py -k scenario_diff_tab_wired -v`
Expected: PASS.

Run: `.venv/bin/python -c "import app"`
Expected: no error (the full app, including the new tab + sub-server, imports).

- [ ] **Step 7: Run the full unit suite for touched modules**

Run: `.venv/bin/python -m pytest tests/test_spatial_series.py tests/test_grid_helpers.py tests/test_analysis.py tests/test_plotting.py tests/test_ui_results.py -q`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add ui/pages/results.py tests/test_ui_results.py
git commit -m "feat(ui): embed Scenario Diff tab + sub-server in Results page"
```

---

## Task 6: End-to-end Playwright validation

**Files:**
- Create: `tests/test_e2e_scenario_diff.py`

This test writes two synthetic run records into the real `data/history/` directory
(the app subprocess reads that fixed path — it cannot be monkeypatched), each pointing
at a synthetic output dir with a biomass CSV + spatial NetCDF, then drives the tab. The
fixture cleans up afterward. Run explicitly with `-m e2e`.

- [ ] **Step 1: Write the e2e test**

Create `tests/test_e2e_scenario_diff.py`:

```python
"""End-to-end test for the Scenario Diff tab.

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_scenario_diff.py -v -m e2e

Writes two synthetic runs into the real data/history/ directory (the app subprocess
reads that fixed path), each with a biomass CSV + a spatial NetCDF, then verifies the
biomass overlay and the three spatial maps render.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_REPO = Path(__file__).resolve().parent.parent
_HISTORY = _REPO / "data" / "history"
_SUBSTRATE = _REPO / "data" / "_scenario_diff_e2e"
_LOAD_TIMEOUT = 15_000


def _write_run(name: str, idx: int, *, base: float) -> tuple[Path, Path]:
    """Create a synthetic output dir (biomass CSV + spatial NetCDF) and history record."""
    out = _SUBSTRATE / name
    out.mkdir(parents=True, exist_ok=True)
    # WIDE biomass CSV: Time + per-species columns (osm_biomass_*.csv convention)
    times = np.arange(10) / 1.0
    pd.DataFrame(
        {"Time": times, "cod": base + times, "sprat": base + 0.5 * times}
    ).to_csv(out / "osm_biomass_Simu0.csv", index=False)
    # Spatial NetCDF (time, species, lat, lon)
    ny, nx, nt = 4, 5, 10
    data = np.fromfunction(
        lambda t, s, y, x: base + s * 10 + t + y + x * 0.1, (nt, 2, ny, nx), dtype=float
    )
    xr.Dataset(
        {"spatial_biomass": (("time", "species", "lat", "lon"), data)},
        coords={
            "time": times,
            "species": ["cod", "sprat"],
            "lat": np.linspace(54.0, 55.0, ny),
            "lon": np.linspace(10.0, 12.0, nx),
        },
    ).to_netcdf(out / "osm_spatial_biomass_Simu0.nc")
    # History record (timestamp drives the selector label)
    ts = f"2026-06-13T0{idx}:00:00"
    rec = {
        "timestamp": ts,
        "config_snapshot": {},
        "duration_sec": 1.0,
        "output_dir": str(out),
        "summary": {},
    }
    rec_path = _HISTORY / f"run_{ts.replace(':', '-')}.json"
    rec_path.write_text(json.dumps(rec))
    return rec_path, out


@pytest.fixture
def two_runs():
    _HISTORY.mkdir(parents=True, exist_ok=True)
    created = [_write_run("runA", 1, base=100.0), _write_run("runB", 2, base=130.0)]
    yield [ts for ts, _ in created]
    import shutil

    for rec_path, _ in created:
        rec_path.unlink(missing_ok=True)
    shutil.rmtree(_SUBSTRATE, ignore_errors=True)


def test_scenario_diff_renders_overlay_and_maps(page: Page, app: ShinyAppProc, two_runs):
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)

    # Go to Results page, then the Scenario Diff tab.
    page.locator(".nav-pills .nav-link[data-value='results']").click()
    page.get_by_role("tab", name="Scenario Diff").click()

    # Wait for the history-backed selectors to populate (the effect runs after the
    # Results tab activates) before selecting — avoids a race on an empty <select>.
    # Assert OUR option exists (robust to other pre-existing runs in data/history).
    expect(
        page.locator("#diff_run_a option[value='2026-06-13T02:00:00']")
    ).to_have_count(1, timeout=_LOAD_TIMEOUT)

    # Select baseline=runA (T01) and variant=runB (T02) by explicit timestamp value
    # (list_runs sorts DESC, so index order is not A-then-B).
    page.locator("#diff_run_a").select_option("2026-06-13T01:00:00")
    page.locator("#diff_run_b").select_option("2026-06-13T02:00:00")

    # Biomass overlay widget renders.
    expect(page.locator("#diff_biomass_chart")).to_be_visible(timeout=_LOAD_TIMEOUT)
    # Three spatial map widgets render once both NetCDFs are open.
    expect(page.locator("#diff_map_delta")).to_be_visible(timeout=_LOAD_TIMEOUT)

    # Screenshot for manual confirmation (plotly content is in shadow DOM).
    page.screenshot(path=str(_REPO / "screenshots" / "scenario_diff_e2e.png"))


def test_scenario_diff_same_run_for_a_and_b(page: Page, app: ShinyAppProc, two_runs):
    """Selecting the SAME run for A and B must not crash (the identical-runs case).

    Exercises the real on-disk handle path: the server shares one open dataset rather
    than opening the same .nc twice (which would risk the HDF5-locking error). Unit
    tests only pass the same in-memory object, so this is the only coverage of it.
    """
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
    page.locator(".nav-pills .nav-link[data-value='results']").click()
    page.get_by_role("tab", name="Scenario Diff").click()
    expect(
        page.locator("#diff_run_a option[value='2026-06-13T01:00:00']")
    ).to_have_count(1, timeout=_LOAD_TIMEOUT)

    # Same run for both sides → all-zero diff, single shared handle, no crash.
    page.locator("#diff_run_a").select_option("2026-06-13T01:00:00")
    page.locator("#diff_run_b").select_option("2026-06-13T01:00:00")

    expect(page.locator("#diff_biomass_chart")).to_be_visible(timeout=_LOAD_TIMEOUT)
    expect(page.locator("#diff_map_delta")).to_be_visible(timeout=_LOAD_TIMEOUT)
    # The identical-runs caption should appear.
    expect(page.locator("#diff_biomass_caption")).to_contain_text(
        "Identical runs", timeout=_LOAD_TIMEOUT
    )
```

- [ ] **Step 2: Run the e2e test**

Run: `.venv/bin/python -m pytest tests/test_e2e_scenario_diff.py -v -m e2e`
Expected: PASS. Inspect `screenshots/scenario_diff_e2e.png` to confirm the overlay shows A/B curves and the three maps render (A, B, and a red/blue difference map).

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e_scenario_diff.py
git commit -m "test(e2e): scenario diff tab renders overlay + spatial maps"
```

---

## Final verification

- [ ] Run the full suite for all touched areas:

Run: `.venv/bin/python -m pytest tests/test_spatial_series.py tests/test_grid_helpers.py tests/test_analysis.py tests/test_plotting.py tests/test_ui_results.py -q`
Expected: all PASS.

- [ ] Confirm lint is clean (CI runs both):

Run: `.venv/bin/ruff check osmose/ ui/ tests/ && .venv/bin/ruff format --check osmose/ ui/ tests/`
Expected: no errors. (If `ruff format --check` reports files, run `.venv/bin/ruff format osmose/ ui/ tests/` and re-commit.)

- [ ] Confirm types (CI runs pyright):

Run: `.venv/bin/pyright osmose/spatial_series.py osmose/analysis.py osmose/plotting.py ui/pages/scenario_diff.py ui/pages/grid_helpers.py`
Expected: no new errors.
