# Live-During-Run Movement Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream living schools' positions onto a deck.gl map on the Run page while a Python-engine simulation runs — as a biomass-weighted heatmap or individual dots, toggled.

**Architecture:** An optional in-process `step_observer` hook in the Python engine loop pushes a `MovementSnapshot` onto a bounded `queue.Queue`; the Run page polls the queue (`reactive.poll` + a consuming effect, the calibration-dashboard pattern), drains to the latest snapshot, and an async render effect draws it via `shiny_deckgl` `MapWidget.partial_update`. Python engine only (Java is a subprocess JAR — out of scope).

**Tech Stack:** Python, numpy, `queue` (stdlib), Shiny for Python (`reactive.poll`/`reactive.effect`), `shiny_deckgl` (`MapWidget`, `heatmap_layer`/`scatterplot_layer`), pytest + Playwright.

**Spec:** `docs/superpowers/specs/2026-06-13-live-movement-viz-design.md`

---

## File Structure

- `osmose/live_movement.py` — **create** (core, pure, no UI imports): `MovementSnapshot`, `resolve_grid_latlon`, `build_snapshot`, `make_step_observer`.
- `ui/pages/live_movement_render.py` — **create** (deck.gl layer builders): `species_color`, `_points_to_rows`, `heatmap_layer_from_points`, `dots_layer_from_points`.
- `osmose/engine/simulate.py` — **modify**: add keyword-only `step_observer` param + one call site.
- `osmose/engine/__init__.py` — **modify**: thread `step_observer` through `PythonEngine.run`.
- `ui/pages/run.py` — **modify**: MapWidget panel + controls + persistent queue/reactives + poll + async render effect + observer wiring + terminal status.
- Tests: `tests/test_live_movement.py`, `tests/test_live_movement_render.py`, `tests/test_engine_simulate.py` (append), `tests/test_ui_run.py` (append or create), `tests/test_e2e_live_movement.py` (create).

Run all unit tests with `.venv/bin/python -m pytest <path> -v`.

---

## Task 1: `osmose/live_movement.py` — snapshot + transport

**Files:**
- Create: `osmose/live_movement.py`
- Test: `tests/test_live_movement.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_live_movement.py`:

```python
"""Tests for osmose.live_movement (snapshot + transport for the live movement view)."""

from __future__ import annotations

import queue
import types

import numpy as np
import pytest

from osmose.engine.grid import Grid
from osmose.live_movement import (
    MovementSnapshot,
    build_snapshot,
    make_step_observer,
    resolve_grid_latlon,
)


def _state(species_id, cell_x, cell_y, biomass, is_out=None):
    """A lightweight stand-in exposing only the fields build_snapshot reads."""
    n = len(species_id)
    return types.SimpleNamespace(
        species_id=np.array(species_id, dtype=np.int32),
        cell_x=np.array(cell_x, dtype=np.int32),
        cell_y=np.array(cell_y, dtype=np.int32),
        biomass=np.array(biomass, dtype=np.float64),
        is_out=np.array(is_out if is_out is not None else [False] * n, dtype=bool),
    )


def _config(n_species=2, n_steps=12, names=("cod", "sprat")):
    return types.SimpleNamespace(
        n_species=n_species, n_steps=n_steps, species_names=list(names)
    )


def test_resolve_grid_latlon_uses_grid_arrays_when_present():
    g = Grid.from_dimensions(ny=3, nx=4)
    g.lat = np.array([54.0, 54.5, 55.0])
    g.lon = np.array([10.0, 10.5, 11.0, 11.5])
    lat, lon = resolve_grid_latlon(g)
    np.testing.assert_allclose(lat, [54.0, 54.5, 55.0])
    np.testing.assert_allclose(lon, [10.0, 10.5, 11.0, 11.5])
    assert lat.dtype == np.float64 and lon.dtype == np.float64


def test_resolve_grid_latlon_falls_back_to_float_indices():
    g = Grid.from_dimensions(ny=3, nx=4)  # lat/lon are None
    lat, lon = resolve_grid_latlon(g)
    np.testing.assert_array_equal(lat, np.arange(3, dtype=np.float64))
    np.testing.assert_array_equal(lon, np.arange(4, dtype=np.float64))
    assert lat.dtype == np.float64 and lon.dtype == np.float64


def test_build_snapshot_maps_cells_to_lonlat_and_carries_biomass():
    g = Grid.from_dimensions(ny=3, nx=4)
    g.lat = np.array([54.0, 54.5, 55.0])
    g.lon = np.array([10.0, 10.5, 11.0, 11.5])
    # two focal schools: cod at (cy=1,cx=2), sprat at (cy=0,cx=0)
    st = _state(species_id=[0, 1], cell_x=[2, 0], cell_y=[1, 0], biomass=[5.0, 3.0])
    snap = build_snapshot(0, st, g, _config())
    assert isinstance(snap, MovementSnapshot)
    assert snap.step == 0 and snap.n_steps == 12 and snap.status == "running"
    assert snap.species == ["cod", "sprat"]
    np.testing.assert_allclose(snap.lon, [11.0, 10.0])
    np.testing.assert_allclose(snap.lat, [54.5, 54.0])
    np.testing.assert_allclose(snap.biomass, [5.0, 3.0])
    np.testing.assert_array_equal(snap.sp_id, [0, 1])
    assert (snap.lon_min, snap.lon_max) == (10.0, 11.5)
    assert (snap.lat_min, snap.lat_max) == (54.0, 55.0)
    assert snap.lon_step == 0.5 and snap.lat_step == 0.5  # grid cell spacing
    assert snap.truncated is False and snap.n_total == 2


def test_build_snapshot_excludes_unlocated_eggs_background_dead_and_out():
    g = Grid.from_dimensions(ny=2, nx=2)
    st = _state(
        species_id=[0, 0, 0, 2, 0],  # idx3 sp_id=2 is background (>= n_species=2)
        cell_x=[0, -1, 1, 0, 1],     # idx1 cell_x=-1 (unlocated egg)
        cell_y=[0, 0, 1, 0, 1],
        biomass=[5.0, 9.0, 0.0, 9.0, 7.0],  # idx2 biomass=0 (dead)
        is_out=[False, False, False, False, True],  # idx4 out of domain
    )
    snap = build_snapshot(3, st, g, _config())
    # only idx0 survives the focal+located+living+in-domain mask
    np.testing.assert_array_equal(snap.sp_id, [0])
    np.testing.assert_allclose(snap.lon, [0.0])
    np.testing.assert_allclose(snap.lat, [0.0])
    assert snap.n_total == 1


def test_build_snapshot_dot_cap_samples_deterministically():
    g = Grid.from_dimensions(ny=10, nx=10)
    n = 100
    st = _state(
        species_id=[0] * n,
        cell_x=list(range(n % 10)) * 10 if False else [i % 10 for i in range(n)],
        cell_y=[i // 10 for i in range(n)],
        biomass=[1.0] * n,
    )
    snap = build_snapshot(0, st, g, _config(), dot_cap=10)
    assert snap.truncated is True
    assert snap.n_total == 100
    assert snap.sp_id.size == 10
    # deterministic: same inputs → same sample
    snap2 = build_snapshot(0, st, g, _config(), dot_cap=10)
    np.testing.assert_array_equal(snap.lon, snap2.lon)


def test_build_snapshot_empty_selection_is_safe():
    g = Grid.from_dimensions(ny=2, nx=2)
    st = _state(species_id=[0], cell_x=[0], cell_y=[0], biomass=[0.0])  # dead
    snap = build_snapshot(0, st, g, _config())
    assert snap.sp_id.size == 0 and snap.lon.size == 0 and snap.n_total == 0


def test_build_snapshot_status_passthrough():
    g = Grid.from_dimensions(ny=2, nx=2)
    st = _state(species_id=[0], cell_x=[0], cell_y=[0], biomass=[1.0])
    snap = build_snapshot(11, st, g, _config(), status="done")
    assert snap.status == "done"


def test_make_step_observer_emits_first_and_final_and_throttles():
    g = Grid.from_dimensions(ny=2, nx=2)
    st = _state(species_id=[0], cell_x=[0], cell_y=[0], biomass=[1.0])
    cfg = _config(n_steps=5)
    clock = {"t": 0.0}
    obs = make_step_observer(
        q := queue.Queue(maxsize=10), throttle_s=1.0, now=lambda: clock["t"]
    )
    obs(0, st, g, cfg)            # step 0 always emits
    clock["t"] = 0.1
    obs(1, st, g, cfg)            # throttled out (< 1.0s since last)
    clock["t"] = 2.0
    obs(2, st, g, cfg)            # emits (>= 1.0s)
    clock["t"] = 2.1
    obs(4, st, g, cfg)            # final step always emits
    steps = []
    while not q.empty():
        steps.append(q.get_nowait().step)
    assert steps == [0, 2, 4]


def test_make_step_observer_drops_oldest_keeps_newest_when_full():
    g = Grid.from_dimensions(ny=2, nx=2)
    st = _state(species_id=[0], cell_x=[0], cell_y=[0], biomass=[1.0])
    cfg = _config(n_steps=100)
    q = queue.Queue(maxsize=1)
    obs = make_step_observer(q, throttle_s=0.0, now=lambda: 0.0)
    obs(0, st, g, cfg)
    obs(50, st, g, cfg)  # queue was full → drop oldest, keep newest
    assert q.qsize() == 1
    assert q.get_nowait().step == 50


def test_make_step_observer_swallows_build_errors():
    cfg = _config()
    bad_grid = object()  # build_snapshot will raise on this
    q = queue.Queue(maxsize=2)
    obs = make_step_observer(q, throttle_s=0.0, now=lambda: 0.0)
    obs(0, object(), bad_grid, cfg)  # must not raise
    assert q.empty()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_live_movement.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'osmose.live_movement'`.

- [ ] **Step 3: Implement `osmose/live_movement.py`**

Create `osmose/live_movement.py`:

```python
"""Live-during-run movement snapshots + queue transport (Python engine only).

Pure core module — no UI imports. Produces a per-step ``MovementSnapshot`` of living
focal schools' positions for the Run-page live map, and a throttling queue observer the
engine step hook calls. ``queue.Queue`` is stdlib, so the transport helper stays here and
is unit-testable.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from osmose.logging import setup_logging

_log = setup_logging("osmose.live_movement")


@dataclass
class MovementSnapshot:
    """One frame of living focal schools' positions for the live map."""

    step: int
    n_steps: int
    status: str  # "running" | "done" | "cancelled"
    species: list[str]
    sp_id: NDArray[np.int32]
    lon: NDArray[np.float64]
    lat: NDArray[np.float64]
    biomass: NDArray[np.float64]
    truncated: bool
    n_total: int
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    lon_step: float  # grid cell spacing (median diff of full lon array); 0 if < 2 cells
    lat_step: float


def resolve_grid_latlon(grid) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(lat, lon)`` cell-coordinate arrays.

    Uses ``grid.lat``/``grid.lon`` when present (NetCDF-backed grids), else
    ``np.arange(ny|nx, dtype=float64)`` — byte-identical to the spatial-NetCDF writer's
    fallback (``osmose/engine/output.py:726,731``), so live and post-run coords agree.
    """
    lat = grid.lat if grid.lat is not None else np.arange(grid.ny, dtype=np.float64)
    lon = grid.lon if grid.lon is not None else np.arange(grid.nx, dtype=np.float64)
    return np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)


def build_snapshot(
    step: int, state, grid, config, *, status: str = "running", dot_cap: int = 5000
) -> MovementSnapshot:
    """Build a snapshot of focal + located + living schools at ``step`` (pure).

    Selection mask = focal (``species_id < n_species``) & in-domain (``~is_out``) &
    located (``cell_x/cell_y >= 0``, drops freshly-spawned eggs at ``-1``) & living
    (``biomass > 0``). Samples to ``dot_cap`` deterministically when exceeded.
    """
    lat_arr, lon_arr = resolve_grid_latlon(grid)
    mask = (
        (state.species_id < config.n_species)
        & ~state.is_out
        & (state.cell_x >= 0)
        & (state.cell_y >= 0)
        & (state.biomass > 0.0)
    )
    sp_id = state.species_id[mask]
    cx = state.cell_x[mask]
    cy = state.cell_y[mask]
    bm = state.biomass[mask]
    n_total = int(sp_id.size)
    truncated = n_total > dot_cap
    if truncated:
        idx = np.linspace(0, n_total - 1, dot_cap).astype(np.intp)
        sp_id, cx, cy, bm = sp_id[idx], cx[idx], cy[idx], bm[idx]
    lon_step = float(np.median(np.diff(lon_arr))) if lon_arr.size > 1 else 0.0
    lat_step = float(np.median(np.diff(lat_arr))) if lat_arr.size > 1 else 0.0
    return MovementSnapshot(
        step=int(step),
        n_steps=int(config.n_steps),
        status=status,
        species=list(config.species_names[: config.n_species]),
        sp_id=np.asarray(sp_id, dtype=np.int32),
        lon=lon_arr[cx].astype(np.float64),
        lat=lat_arr[cy].astype(np.float64),
        biomass=np.asarray(bm, dtype=np.float64),
        truncated=truncated,
        n_total=n_total,
        lon_min=float(lon_arr.min()),
        lon_max=float(lon_arr.max()),
        lat_min=float(lat_arr.min()),
        lat_max=float(lat_arr.max()),
        lon_step=lon_step,
        lat_step=lat_step,
    )


def make_step_observer(
    q: "queue.Queue[MovementSnapshot]",
    *,
    dot_cap: int = 5000,
    throttle_s: float = 0.2,
    now: Callable[[], float] = time.monotonic,
) -> Callable[[int, object, object, object], None]:
    """Return a step-observer that builds a snapshot and enqueues it (drop-oldest).

    Always emits step 0 and the final step (``config.n_steps - 1``); throttles the
    rest by wall-clock. Never blocks the engine thread and never raises into it.
    """
    last_emit: list[float | None] = [None]

    def observer(step: int, state, grid, config) -> None:
        n_steps = int(config.n_steps)
        is_edge = step == 0 or step == n_steps - 1
        t = now()
        if not is_edge and last_emit[0] is not None and (t - last_emit[0]) < throttle_s:
            return
        last_emit[0] = t
        try:
            snap = build_snapshot(step, state, grid, config, dot_cap=dot_cap)
        except Exception:  # noqa: BLE001 — never crash the running simulation
            _log.warning("live snapshot build failed at step %s", step, exc_info=True)
            return
        try:
            q.put_nowait(snap)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(snap)
            except queue.Full:
                pass

    return observer
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_live_movement.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Lint + commit**

Run: `.venv/bin/ruff check osmose/live_movement.py tests/test_live_movement.py && .venv/bin/ruff format osmose/live_movement.py tests/test_live_movement.py`

Write the commit message to a temp file (heredocs/`>` blocked) and `git commit -F /tmp/t1.txt`:
```
feat(live-movement): add MovementSnapshot + build_snapshot + queue observer

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```
Stage: `git add osmose/live_movement.py tests/test_live_movement.py`

---

## Task 2: `ui/pages/live_movement_render.py` — deck.gl layer builders

**Files:**
- Create: `ui/pages/live_movement_render.py`
- Test: `tests/test_live_movement_render.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_live_movement_render.py`:

```python
"""Tests for ui.pages.live_movement_render (deck.gl layer builders)."""

from __future__ import annotations

import numpy as np

from osmose.live_movement import MovementSnapshot
from ui.pages.live_movement_render import (
    dots_layer_from_points,
    heatmap_layer_from_points,
    species_color,
)


def _snap(sp_id, lon, lat, biomass, species=("cod", "sprat"), lon_step=1.0, lat_step=1.0):
    lo, la = list(lon), list(lat)
    return MovementSnapshot(
        step=0, n_steps=12, status="running", species=list(species),
        sp_id=np.array(sp_id, dtype=np.int32),
        lon=np.array(lon, dtype=np.float64),
        lat=np.array(lat, dtype=np.float64),
        biomass=np.array(biomass, dtype=np.float64),
        truncated=False, n_total=len(sp_id),
        lon_min=float(min(lo)) if lo else 0.0, lon_max=float(max(lo)) if lo else 0.0,
        lat_min=float(min(la)) if la else 0.0, lat_max=float(max(la)) if la else 0.0,
        lon_step=lon_step, lat_step=lat_step,
    )


def test_species_color_distinct_and_deterministic():
    c0, c1 = species_color(0), species_color(1)
    assert c0 != c1
    assert species_color(0) == c0  # deterministic
    assert len(c0) == 4 and all(0 <= v <= 255 for v in c0)


def test_heatmap_layer_structure():
    snap = _snap([0, 1], [10.0, 11.0], [54.0, 55.0], [5.0, 3.0])
    layer = heatmap_layer_from_points(snap, None)
    assert layer["id"] == "live_movement"
    assert layer["getPosition"] == "@@=d.position"
    assert layer["getWeight"] == "@@=d.weight"
    assert len(layer["data"]) == 2
    assert layer["data"][0]["position"] == [10.0, 54.0]
    assert layer["data"][0]["weight"] == 5.0
    assert isinstance(layer["colorRange"], list) and len(layer["colorRange"]) >= 2


def test_species_filter_reduces_rows():
    snap = _snap([0, 1, 0], [10.0, 11.0, 12.0], [54.0, 55.0, 56.0], [1.0, 2.0, 3.0])
    layer = heatmap_layer_from_points(snap, "cod")  # sp_id 0
    assert len(layer["data"]) == 2  # only the two cod rows


def test_dots_layer_structure_and_jitter_bounded_deterministic():
    snap = _snap([0, 0], [10.0, 10.0], [54.0, 54.0], [4.0, 9.0])  # same cell
    layer = dots_layer_from_points(snap, None)
    assert layer["id"] == "live_movement"
    assert layer["getFillColor"] == "@@=d.fill"
    assert layer["getRadius"] == "@@=d.radius"
    assert layer["pickable"] is True
    rows = layer["data"]
    assert len(rows) == 2
    # two schools in one cell get distinct jittered positions (deterministic)
    assert rows[0]["position"] != rows[1]["position"]
    layer2 = dots_layer_from_points(snap, None)
    assert [r["position"] for r in layer2["data"]] == [r["position"] for r in rows]
    # fill colored by species
    assert rows[0]["fill"] == list(species_color(0))


def test_empty_snapshot_yields_empty_layer():
    snap = _snap([], [], [], [])
    h = heatmap_layer_from_points(snap, None)
    d = dots_layer_from_points(snap, None)
    assert h["data"] == [] and d["data"] == []
    assert h["id"] == "live_movement" and d["id"] == "live_movement"
```

(The `_snap` helper above already guards empty lon/lat lists with `0.0`, so
`test_empty_snapshot_yields_empty_layer` constructs cleanly, and it carries `lon_step`/
`lat_step` so the jitter test has a non-zero scale.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_live_movement_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ui.pages.live_movement_render'`.

- [ ] **Step 3: Implement `ui/pages/live_movement_render.py`**

Create `ui/pages/live_movement_render.py`:

```python
"""deck.gl layer builders for the live movement view.

A new module rather than an addition to ui/pages/grid_helpers.py: that file is
Plotly/config-page-scoped (~1254 LOC) and builds Plotly figures, not deck.gl layers. The
only in-repo deck.gl layer code is inline at ui/pages/spatial_results.py:570-598 — this
module follows that convention (positional id, camelCase props, "@@=d.field" accessors,
row-dict data).
"""

from __future__ import annotations

import numpy as np
from shiny_deckgl import (  # type: ignore[import-untyped]
    PALETTE_THERMAL,
    color_range,
    heatmap_layer,
    scatterplot_layer,
)

from osmose.live_movement import MovementSnapshot

_LAYER_ID = "live_movement"

# Deterministic categorical RGBA palette (NOT shiny_deckgl.SPECIES_COLORS — that is a
# 3-entry seal palette unusable for fish).
_SPECIES_PALETTE: list[list[int]] = [
    [31, 119, 180, 200], [255, 127, 14, 200], [44, 160, 44, 200], [214, 39, 40, 200],
    [148, 103, 189, 200], [140, 86, 75, 200], [227, 119, 194, 200], [127, 127, 127, 200],
    [188, 189, 34, 200], [23, 190, 207, 200],
]


def species_color(sp_id: int) -> list[int]:
    """Deterministic RGBA for a species index (cycles the palette)."""
    return list(_SPECIES_PALETTE[int(sp_id) % len(_SPECIES_PALETTE)])


def _filter_mask(snap: MovementSnapshot, species_filter: str | None) -> np.ndarray:
    if species_filter is None:
        return np.ones(snap.sp_id.size, dtype=bool)
    try:
        target = snap.species.index(species_filter)  # name -> sp_id (species in sp_id order)
    except ValueError:
        return np.zeros(snap.sp_id.size, dtype=bool)
    return snap.sp_id == target


def _points_to_rows(snap: MovementSnapshot, species_filter: str | None) -> list[dict]:
    """Base rows: position + weight + fill. Heatmap ignores fill (one builder, both modes)."""
    m = _filter_mask(snap, species_filter)
    sp_id, lon, lat, bm = snap.sp_id[m], snap.lon[m], snap.lat[m], snap.biomass[m]
    return [
        {"position": [float(lo), float(la)], "weight": float(b), "fill": species_color(s)}
        for s, lo, la, b in zip(sp_id, lon, lat, bm)
    ]


def heatmap_layer_from_points(snap: MovementSnapshot, species_filter: str | None) -> dict:
    """Native deck.gl HeatmapLayer weighted by biomass, from un-jittered cell centers."""
    return heatmap_layer(
        _LAYER_ID,
        data=_points_to_rows(snap, species_filter),
        getPosition="@@=d.position",
        getWeight="@@=d.weight",
        colorRange=color_range(palette=PALETTE_THERMAL),
    )


def dots_layer_from_points(snap: MovementSnapshot, species_filter: str | None) -> dict:
    """ScatterplotLayer: one dot per school, colored by species, biomass-sized.

    Deterministic per-school in-cell jitter (seeded by row index, no RNG) bounded to ±¼ of
    the grid cell spacing carried in the snapshot (``lon_step``/``lat_step``) — so
    overlapping schools in one cell spread out even when every school is in the same cell
    (a per-occupied-coord estimate would collapse to 0 there). ``*_step == 0`` (a 1-cell
    grid) → no jitter; ``radiusMinPixels`` still separates dots visually.
    """
    m = _filter_mask(snap, species_filter)
    sp_id, lon, lat, bm = snap.sp_id[m], snap.lon[m], snap.lat[m], snap.biomass[m]
    jx = snap.lon_step * 0.25
    jy = snap.lat_step * 0.25
    bmax = float(bm.max()) if bm.size and bm.max() > 0 else 1.0
    rows = []
    for i, (s, lo, la, b) in enumerate(zip(sp_id, lon, lat, bm)):
        # Deterministic offsets in [-1, 1] from the row index (no RNG, reproducible).
        ox = ((i * 2654435761) % 1000 / 500.0 - 1.0) * jx
        oy = ((i * 40503) % 1000 / 500.0 - 1.0) * jy
        rows.append(
            {
                "position": [float(lo) + ox, float(la) + oy],
                "fill": species_color(s),
                "radius": 3.0 + 12.0 * float(np.sqrt(max(b, 0.0) / bmax)),
            }
        )
    return scatterplot_layer(
        _LAYER_ID,
        data=rows,
        getPosition="@@=d.position",
        getFillColor="@@=d.fill",
        getRadius="@@=d.radius",
        radiusUnits="pixels",
        radiusMinPixels=2,
        pickable=True,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_live_movement_render.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Lint + commit**

Run: `.venv/bin/ruff check ui/pages/live_movement_render.py tests/test_live_movement_render.py && .venv/bin/ruff format ui/pages/live_movement_render.py tests/test_live_movement_render.py`

Commit via temp file `/tmp/t2.txt`:
```
feat(live-movement): add deck.gl heatmap + dots layer builders

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```
Stage: `git add ui/pages/live_movement_render.py tests/test_live_movement_render.py`

---

## Task 3: Engine `step_observer` hook

**Files:**
- Modify: `osmose/engine/simulate.py` (signature ~`:1226`, call site after `:1583`)
- Modify: `osmose/engine/__init__.py` (`PythonEngine.run`, `:87-107`)
- Test: `tests/test_engine_simulate.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine_simulate.py`:

```python
def test_step_observer_fires_once_per_step(minimal_config):
    cfg = EngineConfig.from_dict(minimal_config)
    grid = Grid.from_dimensions(ny=3, nx=3)
    calls = []
    simulate(
        cfg,
        grid,
        np.random.default_rng(42),
        step_observer=lambda step, state, g, c: calls.append((step, c.n_steps)),
    )
    assert [s for s, _ in calls] == list(range(cfg.n_steps))
    assert all(n == cfg.n_steps for _, n in calls)


def test_step_observer_none_is_parity_safe(minimal_config):
    cfg = EngineConfig.from_dict(minimal_config)
    grid = Grid.from_dimensions(ny=3, nx=3)
    out_a = simulate(cfg, grid, np.random.default_rng(7))
    cfg2 = EngineConfig.from_dict(minimal_config)
    grid2 = Grid.from_dimensions(ny=3, nx=3)
    out_b = simulate(
        cfg2, grid2, np.random.default_rng(7), step_observer=lambda *a: None
    )
    assert len(out_a) == len(out_b)
    for a, b in zip(out_a, out_b):
        np.testing.assert_array_equal(a.biomass, b.biomass)
        np.testing.assert_array_equal(a.abundance, b.abundance)


def test_step_observer_survives_cancel(minimal_config):
    import threading

    from osmose.engine import SimulationCancelled

    cfg = EngineConfig.from_dict(minimal_config)
    grid = Grid.from_dimensions(ny=3, nx=3)
    token = threading.Event()
    calls = []

    def obs(step, state, g, c):
        calls.append(step)
        if step == 3:
            token.set()  # request cancel; loop checks the token at the next step's top

    with pytest.raises(SimulationCancelled):
        simulate(cfg, grid, np.random.default_rng(1), cancel_token=token, step_observer=obs)
    assert calls and max(calls) >= 3  # observer fired for the pre-cancel steps
```

(`pytest` is already imported at the top of `tests/test_engine_simulate.py`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_simulate.py -k step_observer -v`
Expected: FAIL with `TypeError: simulate() got an unexpected keyword argument 'step_observer'`.

- [ ] **Step 3a: Add the param to `simulate`**

In `osmose/engine/simulate.py`, in the `simulate(...)` signature (the keyword-only block after `*,` around line 1226-1229), add a parameter after `cancel_token`:

```python
    cancel_token: "threading.Event | None" = None,
    step_observer: "Callable[[int, object, object, object], None] | None" = None,
```

Ensure `Callable` is imported at the top of `simulate.py` — `simulate.py:14` currently has
only `from typing import TYPE_CHECKING, cast`, so **add `Callable`** to it:
`from typing import TYPE_CHECKING, Callable, cast`. This is **required for the pyright gate**
(it resolves the string forward-ref `"Callable[...]"`) even though `from __future__ import
annotations` stringifies it — do not omit it. (Same for `live_movement.py`: its
`from typing import Callable` is annotation-only but required for pyright; keep it.)

- [ ] **Step 3b: Add the call site**

In `osmose/engine/simulate.py`, find `accumulated.append(step_out)` (line ~1583) and add immediately after it (same indentation, inside the `for step` loop):

```python
        if step_observer is not None:
            step_observer(step, state, grid, config)
```

- [ ] **Step 3c: Thread it through `PythonEngine.run`**

In `osmose/engine/__init__.py`, `PythonEngine.run` (`:87`), add a keyword-only param and pass it to `simulate`:

```python
    def run(
        self,
        config: dict[str, str],
        output_dir: Path,
        seed: int = 0,
        *,
        cancel_token: "threading.Event | None" = None,
        step_observer=None,
    ) -> RunResult:
```

and in the `simulate(...)` call inside `run` (`:99-107`) add the argument:

```python
        outputs = simulate(
            engine_config,
            grid,
            rng,
            movement_rngs=movement_rngs,
            mortality_rngs=mortality_rngs,
            output_dir=output_dir,
            cancel_token=cancel_token,
            step_observer=step_observer,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_simulate.py -k step_observer -v`
Expected: PASS (2 tests).

Also run the full engine-simulate file + a parity smoke check for no regression:
`.venv/bin/python -m pytest tests/test_engine_simulate.py tests/test_engine_parity.py -q`
Expected: all PASS.

- [ ] **Step 5: Lint + commit**

Run: `.venv/bin/ruff check osmose/engine/simulate.py osmose/engine/__init__.py tests/test_engine_simulate.py && .venv/bin/ruff format osmose/engine/simulate.py osmose/engine/__init__.py tests/test_engine_simulate.py`

Commit via `/tmp/t3.txt`:
```
feat(engine): optional step_observer hook in the Python simulate loop

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```
Stage: `git add osmose/engine/simulate.py osmose/engine/__init__.py tests/test_engine_simulate.py`

---

## Task 4: Run-page wiring (controls + map + queue + poll + render)

**Files:**
- Modify: `ui/pages/run.py`
- Test: `tests/test_ui_run.py` (append; create if absent)

This task has no new unit test for the reactive wiring (covered by the Task 5 e2e); it
adds a **structure test** asserting the wiring symbols exist, and the heavy verification
is the e2e. READ the actual `run.py` around each anchor before editing.

- [ ] **Step 1: Write the failing structure test**

Append to `tests/test_ui_run.py` (create the file with this content if it does not exist):

```python
def test_live_movement_wired_into_run_page():
    """The live-movement view (toggle, map, queue, poll, observer) is wired into run.py."""
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parent.parent / "ui" / "pages" / "run.py"
    ).read_text()
    assert "make_step_observer" in src
    assert "live_movement_view" in src
    assert "heatmap_layer_from_points" in src
    assert "dots_layer_from_points" in src
    assert "partial_update" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ui_run.py -k live_movement_wired -v`
Expected: FAIL (symbols not yet in `run.py`).

- [ ] **Step 3a: Imports**

In `ui/pages/run.py`, after the existing imports (top of file, ~line 8-15), add:

```python
import queue
import time

from shiny_deckgl import (  # type: ignore[import-untyped]
    CARTO_DARK,
    CARTO_POSITRON,
    MapWidget,
    compass_widget,
    fullscreen_widget,
    scale_widget,
    zoom_widget,
)

from osmose.live_movement import make_step_observer
from ui.pages.live_movement_render import dots_layer_from_points, heatmap_layer_from_points
from ui.state import get_theme_mode
```

(`MapWidget.ui()` renders the map — no `output_widget` import is needed. If any of these
are already imported in `run.py`, do not duplicate — merge into the existing import.)

- [ ] **Step 3b: Controls + map panel in `run_ui`**

Two-handle MapWidget pattern: a UI handle in `run_ui` for `.ui()` + a server handle of the
same id in `run_server` for `update`/`partial_update` (exactly as `spatial_results.py:108`
UI handle and `:172` server handle).

**`run_ui()` (`:153-231`) currently returns a single expression:**
`ui.div(expand_tab("Run Configuration", "run"), ui.layout_columns(left, console_card,
col_widths=[4, 8]), class_="osm-split-layout", id="split_run")` — the **outer node is the
`ui.div`** (it carries `class_`/`id` and `expand_tab` as its first child); the
`layout_columns` is a child with only `col_widths=[4, 8]`. Refactor the body to create the
UI map handle, then **append the live-movement card as a new last child of that existing
`ui.div`** (preserving `expand_tab`, `class_`, `id` — do NOT wrap in a bare `TagList`,
which would drop them):

```python
def run_ui():
    live_map = MapWidget("live_map", style=CARTO_POSITRON)
    return ui.div(
        expand_tab("Run Configuration", "run"),
        ui.layout_columns(
            # ... the existing left config card + Console Output card, col_widths=[4, 8] ...
        ),
        ui.card(
            ui.card_header("Live Movement (Python engine)"),
            ui.input_switch("live_movement_view", "Stream movement during run", value=False),
            ui.input_radio_buttons(
                "live_movement_mode", "Mode",
                {"heatmap": "Heatmap", "dots": "Dots"}, selected="heatmap", inline=True,
            ),
            ui.input_select(
                "live_movement_species", "Species", choices={"__all__": "All species"}
            ),
            ui.output_ui("live_movement_status"),
            live_map.ui(height="420px"),
        ),
        class_="osm-split-layout",
        id="split_run",
    )
```

Keep the existing left/console cards + `col_widths` verbatim; only add the `live_map`
handle and the new `ui.card(...)`. Uses `ui.card_header` (matching the Console card at
`run.py:224`). The server creates its own `MapWidget("live_map")` handle (Step 3c).

- [ ] **Step 3c: Server state + map handle in `run_server`**

In `run_server(...)` (`:427`), near the other reactive declarations, add:

```python
    _live_map = MapWidget("live_map", style=CARTO_POSITRON)
    _live_queue: queue.Queue = queue.Queue(maxsize=4)
    _live_snapshot: reactive.Value = reactive.Value(None)  # MovementSnapshot | None
    _live_status_val: reactive.Value = reactive.Value("")   # "" | running | done | cancelled | failed
    _live_framed = [False]  # plain mutable flag (NOT reactive — render effect reads+writes it)
    _last_live_species = [None]  # plain flag for the species-selector changed-only guard
```

- [ ] **Step 3d: Poll (drain) + consuming effect + species selector population**

Add in `run_server`:

```python
    @reactive.poll(lambda: time.time(), interval_secs=0.2)
    def _drain_live_queue():
        latest = None
        while True:
            try:
                latest = _live_queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            _live_snapshot.set(latest)

    @reactive.effect
    def _consume_live_poll():
        _drain_live_queue()

    @reactive.effect
    def _populate_live_species():
        snap = _live_snapshot.get()
        if snap is None:
            return
        # Changed-only guard (snap.species is constant per run; _live_snapshot changes ~5x/s).
        # Without it, update_select re-runs every frame and resets the user's species choice
        # (mirrors the scenario_diff.py:108 / results.py:515 pattern).
        if snap.species == _last_live_species[0]:
            return
        _last_live_species[0] = list(snap.species)
        choices = {"__all__": "All species"}
        choices.update({name: name for name in snap.species})
        ui.update_select("live_movement_species", choices=choices)
```

- [ ] **Step 3e: Status output**

Add in `run_server`:

```python
    @render.ui
    def live_movement_status():
        status = _live_status_val.get()
        snap = _live_snapshot.get()
        if not status:
            if state.engine_mode.get() != "python":
                return ui.p("Live view available for the Python engine.", class_="text-muted")
            return ui.p("Enable the toggle before running to stream movement.", class_="text-muted")
        prog = f"step {snap.step + 1}/{snap.n_steps}" if snap is not None else ""
        extra = ""
        if snap is not None and snap.truncated:
            extra = f" — showing {snap.sp_id.size} of {snap.n_total} schools"
        return ui.p(f"{status} {prog}{extra}".strip())
```

(`run.py` render functions use **bare** `@render.ui` / `@render.text` with no `@output` —
verified at `run.py:432,455,459` — so no `@output` decorator is used here.)

- [ ] **Step 3f: Async render effect**

Add in `run_server`:

```python
    @reactive.effect
    async def _render_live_map():
        snap = _live_snapshot.get()
        mode = input.live_movement_mode()
        sel = input.live_movement_species()
        species_filter = None if sel in ("__all__", None) else sel
        # theme → basemap swap (partial_update cannot change deck-level style)
        style = CARTO_DARK if get_theme_mode(input) == "dark" else CARTO_POSITRON
        if style != _live_map.style:
            _live_map.style = style
            await _live_map.set_style(session, style)
        if snap is None:
            return
        layer = (
            dots_layer_from_points(snap, species_filter)
            if mode == "dots"
            else heatmap_layer_from_points(snap, species_filter)
        )
        if not _live_framed[0]:
            await _live_map.update(
                session,
                layers=[layer],
                view_state={
                    "latitude": (snap.lat_min + snap.lat_max) / 2,
                    "longitude": (snap.lon_min + snap.lon_max) / 2,
                    "zoom": 5,
                },
                widgets=[
                    fullscreen_widget(placement="top-left"),
                    zoom_widget(placement="top-right"),
                    compass_widget(placement="top-right"),
                    scale_widget(placement="bottom-left"),
                ],
            )
            _live_framed[0] = True
        else:
            await _live_map.partial_update(session, layers=[layer])
```

- [ ] **Step 3g: Wire the observer + terminal status into `handle_run`/`_run_python_engine`**

In `run_server`'s `handle_run` (`:470`), where it dispatches to the Python engine, before
the dispatch reset the live state and build the observer when the toggle is on:

```python
        live_observer = None
        if input.live_movement_view() and engine_mode == "python":
            # drain stale frames + reset per-run live state (single persistent queue)
            while True:
                try:
                    _live_queue.get_nowait()
                except queue.Empty:
                    break
            _live_snapshot.set(None)
            _live_framed[0] = False
            _last_live_species[0] = None  # re-populate the species selector for the new run
            _live_status_val.set("running")
            live_observer = make_step_observer(_live_queue)

        def _set_live_status(s: str) -> None:
            _live_status_val.set(s)
```

(Deliberate simplification vs the spec's "run-start initial full `update`": the basemap
renders client-side from the static `MapWidget("live_map", style=CARTO_POSITRON).ui()`, so
there is no "render before basemap exists" window — all widget/view_state setup happens in
the render effect's first-snapshot full `update`. If the e2e shows any empty/flash issue,
fall back to a run-start `await _live_map.update(session, layers=[], view_state=<neutral>,
widgets=[...])`.)

and pass `live_observer` + `_set_live_status` into the `_run_python_engine(...)` call.

Modify `_run_python_engine` (`ui/pages/run.py:234` — NOT the unrelated
`OsmoseCalibrationProblem._run_python_engine` at `osmose/calibration/problem.py:258`; edit
only the `run.py` async function) to accept the two new params and use them:

```python
async def _run_python_engine(
    input, state, session, config, work_dir, source_dir, run_log, status, runner_ref,
    *, step_observer=None, set_live_status=None,
):
    ...
        result = await loop.run_in_executor(
            None,
            lambda: engine.run(
                run_config, output_dir, seed=0, cancel_token=cancel_token,
                step_observer=step_observer,
            ),
        )
    except SimulationCancelled as exc:
        if set_live_status is not None:
            set_live_status("cancelled")
        ...
    except Exception as exc:
        if set_live_status is not None:
            set_live_status("failed")
        ...
    else:
        # No exception → success. (try / except / except / ELSE / finally order.)
        if set_live_status is not None:
            set_live_status("done")
    finally:
        ...  # existing busy/button reset, unchanged
```

**Placement is exact:** `_run_python_engine` (`run.py:268-311`) is
`try` (`:268`) / `except SimulationCancelled` (`:274`) / `except Exception` (`:289`) /
`finally` (`:306`) / then `_handle_result(...)` (`:311`). Insert the new `else:`
**between the end of `except Exception` (`:305`) and `finally:` (`:306`)** — Python
requires `else` before `finally`; placing it after `finally` is a SyntaxError. The
`except` blocks reassign `result` without re-raising, so `else` runs only on success, on
the **main thread** (post-`await`). Leave `finally` and `_handle_result(...)` unchanged.

- [ ] **Step 4: Verify**

Run: `.venv/bin/python -m pytest tests/test_ui_run.py -k live_movement_wired -v` → PASS.
Run: `.venv/bin/python -c "import app"` → no error (full app imports with the new wiring).
Run: `.venv/bin/ruff check ui/pages/run.py tests/test_ui_run.py && .venv/bin/ruff format ui/pages/run.py tests/test_ui_run.py`

- [ ] **Step 5: Commit**

Commit via `/tmp/t4.txt`:
```
feat(ui): live movement view on the Run page (map + queue + poll + render)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```
Stage: `git add ui/pages/run.py tests/test_ui_run.py`

---

## Task 5: End-to-end Playwright validation

**Files:**
- Create: `tests/test_e2e_live_movement.py`

Substrate = `data/baltic` with `nyear=1` (24 steps; maps movement → schools located;
NetCDF grid → real lat/lon). Playwright + chromium are installed.

- [ ] **Step 1: Write the e2e test**

Create `tests/test_e2e_live_movement.py`:

```python
"""End-to-end test for the live movement view on the Run page.

Run explicitly: .venv/bin/python -m pytest tests/test_e2e_live_movement.py -v -m e2e
"""

from __future__ import annotations

import pathlib
import re

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_REPO = pathlib.Path(__file__).resolve().parent.parent
_LOAD_TIMEOUT = 20_000
_RUN_TIMEOUT = 60_000


def test_live_movement_renders_during_python_run(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)

    # Load the Baltic example config via the Grid/Domain page loader (where #load_example
    # lives — grid.py:105; e2e precedent test_e2e_grid_maps.py:39-41).
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.select_option("#load_example", "baltic")  # "baltic" is a valid value (osmose/demo.py:71)
    page.click("#btn_load_example")
    # Wait for the load to settle before navigating (mirrors test_e2e_grid_maps.py:41) —
    # otherwise the run can start before state.config holds the Baltic config + movement maps.
    page.wait_for_selector(".shiny-notification", timeout=_LOAD_TIMEOUT)

    # Shorten the run: 1 year. (#py_param_overrides on the Run page — run.py:200.)
    page.locator(".nav-pills .nav-link[data-value='run']").click()
    page.locator("#py_param_overrides").fill("simulation.time.nyear=1")

    # The engine defaults to Java (ui/state.py:65), so clicking #engineBtnPython is
    # REQUIRED (not defensive) to enable the Python-only live view (engine toggle buttons
    # #engineBtnJava — app.py:195, #engineBtnPython — app.py:201).
    page.locator("#engineBtnPython").click()

    # Enable the live movement view, then run.
    page.locator("#live_movement_view").click()
    page.locator("#btn_run").click()

    # The live map container renders (note: #live_map is a static basemap, present as soon
    # as the Run page is active — it does NOT prove the run started).
    expect(page.locator("#live_map")).to_be_visible(timeout=_LOAD_TIMEOUT)
    # Guard against a validation-blocked run with a clear diagnostic. "Validation failed"
    # is set on #run_status (run.py:486), NOT #live_movement_status — assert the right one.
    expect(page.locator("#run_status")).not_to_contain_text(
        "Validation failed", timeout=_LOAD_TIMEOUT
    )
    # The status reads "running" as soon as the run starts (_live_status_val is set in
    # handle_run before the executor dispatch), then "done" on completion.
    expect(page.locator("#live_movement_status")).to_contain_text("running", timeout=_RUN_TIMEOUT)
    expect(page.locator("#live_movement_status")).to_contain_text("done", timeout=_RUN_TIMEOUT)

    # Toggle to dots mode (re-renders the retained final frame).
    page.locator("#live_movement_mode").get_by_text("Dots").click()
    page.screenshot(path=str(_REPO / "screenshots" / "live_movement_e2e.png"))


def test_live_movement_cancel_path(page: Page, app: ShinyAppProc):
    """Cancelling mid-run leaves the retained frame and shows a 'cancelled' status
    (covers the terminal-snapshot-direct-set cancel branch, which has no unit test)."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.select_option("#load_example", "baltic")
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=_LOAD_TIMEOUT)
    page.locator(".nav-pills .nav-link[data-value='run']").click()
    page.locator("#py_param_overrides").fill("simulation.time.nyear=10")  # ~10-14s warm; long cancel window
    page.locator("#engineBtnPython").click()
    page.locator("#live_movement_view").click()
    page.locator("#btn_run").click()
    # Gate the cancel on a REAL emitted frame — "running step N/M" appears only after the
    # observer has pushed a snapshot (bare "running" is set pre-dispatch and would not prove
    # the run is mid-flight; #live_map is a static basemap that resolves instantly).
    expect(page.locator("#live_movement_status")).to_contain_text(
        re.compile(r"running step \d+"), timeout=_RUN_TIMEOUT
    )
    page.locator("#btn_cancel").click()
    expect(page.locator("#live_movement_status")).to_contain_text("cancelled", timeout=_RUN_TIMEOUT)
    expect(page.locator("#live_map")).to_be_visible()
```

(Verify the exact loader control ids — `#load_example` / `#btn_load_example` /
`#py_param_overrides` — against `ui/pages/setup.py` and `run.py` while implementing; adapt
the selectors if they differ. The substrate requirement is fixed: Baltic, nyear=1.)

- [ ] **Step 2: Run the e2e LIVE**

Run (allow a few minutes): `.venv/bin/python -m pytest tests/test_e2e_live_movement.py -v -m e2e`
Expected: PASS. Inspect `screenshots/live_movement_e2e.png` — the map shows moving schools
(heatmap density / dots) and the status reads "done step 24/24".

**If it fails:** capture the failure (selector, console error, any `shiny-error-console`
overlay text) and report it — a live failure usually means a real wiring bug. Do not
weaken the assertions. Report BLOCKED with detail rather than papering over.

- [ ] **Step 3: Confirm default suite excludes it**

Run: `.venv/bin/python -m pytest tests/test_e2e_live_movement.py -q` → "no tests ran" / deselected (default `addopts = -m 'not e2e'`).

- [ ] **Step 4: Commit**

Commit via `/tmp/t5.txt`:
```
test(e2e): live movement view renders during a Python run

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```
Stage: `git add tests/test_e2e_live_movement.py`

---

## Final verification

- [ ] Touched-module unit suite:
`.venv/bin/python -m pytest tests/test_live_movement.py tests/test_live_movement_render.py tests/test_engine_simulate.py tests/test_ui_run.py -q` → all PASS.
- [ ] CI gates: `.venv/bin/ruff check osmose/ ui/ tests/ && .venv/bin/ruff format --check osmose/ ui/ tests/` → clean; `.venv/bin/pyright osmose/live_movement.py ui/pages/live_movement_render.py osmose/engine/simulate.py osmose/engine/__init__.py ui/pages/run.py` → 0 errors.
- [ ] `.venv/bin/python -c "import app"` → imports cleanly.
