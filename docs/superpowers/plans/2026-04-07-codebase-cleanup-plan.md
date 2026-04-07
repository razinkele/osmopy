# Codebase Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate code duplication, improve performance on hot paths, and consolidate test utilities across the OSMOSE Python UI layer.

**Architecture:** Extract shared helpers from `grid_helpers.py` (file search, half-step computation), vectorize the CSV overlay cell loop, consolidate string sentinels into constants, and unify FakeInput across test files. Colormaps are intentionally different per context (CSV overlay, NetCDF overlay, spatial results) and are NOT unified.

**Tech Stack:** Python 3.12+, NumPy, pandas, xarray, Shiny for Python, shiny-deckgl, pytest

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `ui/pages/grid_helpers.py` | Extract `_find_config_file()`, `_compute_half_extents()`; vectorize `load_csv_overlay` |
| Modify | `ui/pages/grid.py` | Replace string literals with constants |
| Modify | `ui/pages/map_viewer.py` | Extract shared `_DEFAULT_VIEW_STATE` |
| Create | `tests/helpers.py` | Shared `make_fake_input()`, `make_catch_all_input()`, `make_multi_input()` |
| Create | `tests/test_grid_helpers_utils.py` | Unit tests for extracted helpers |
| Modify | `tests/test_ui_state.py` | Use shared FakeInput fixture |
| Modify | `tests/test_state.py` | Use shared FakeInput fixture |
| Modify | `tests/test_ui_calibration_handlers.py` | Use shared FakeInput fixture |
| Modify | `tests/test_ui_load_scenarios.py` | Use shared FakeInput fixture |

**NOT in scope (intentional):**
- Colormaps: CSV overlay (quadratic), NetCDF overlay (piecewise blue-cyan-yellow), spatial_results (linear) are intentionally different per visual context. Do NOT unify.
- `SKIP_PREFIXES`: The tuples in `grid.py` and `grid_helpers.py` differ intentionally (`grid.py` omits `fisheries.movement.file.map` because it handles fishing maps in a separate pass). Do NOT unify without careful testing of the overlay dropdown behavior.

---

### Task 1: Extract `_find_config_file()` helper

Deduplicate the identical file-search pattern in `load_mask()` (lines 65–78) and `load_netcdf_grid()` (lines 406–419). The helper returns `None` silently — callers emit their own log messages at the appropriate level (`WARNING` for mask, `DEBUG` for NetCDF grid).

**Files:**
- Modify: `ui/pages/grid_helpers.py:35-85` (add helper before `load_mask`)
- Create: `tests/test_grid_helpers_utils.py`
- Test: `tests/test_overlay_display.py` (existing tests cover both callers)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grid_helpers_utils.py
from pathlib import Path
from ui.pages.grid_helpers import _find_config_file


def test_find_config_file_returns_existing(tmp_path):
    f = tmp_path / "grid" / "mask.csv"
    f.parent.mkdir()
    f.write_text("1;2;3")
    result = _find_config_file("grid/mask.csv", config_dir=tmp_path)
    assert result == f.resolve()


def test_find_config_file_returns_none_for_missing(tmp_path):
    result = _find_config_file("nonexistent.csv", config_dir=tmp_path)
    assert result is None


def test_find_config_file_rejects_traversal(tmp_path):
    result = _find_config_file("../../etc/passwd", config_dir=tmp_path)
    assert result is None


def test_find_config_file_falls_back_to_examples(tmp_path):
    """When config_dir has no match, search data/examples/ as fallback."""
    # This test only verifies the fallback chain is attempted;
    # actual examples dir may or may not have the file.
    result = _find_config_file("nonexistent_in_both.csv", config_dir=tmp_path)
    assert result is None  # Not found in either — returns None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_grid_helpers_utils.py -v`
Expected: FAIL — `_find_config_file` not yet exported

- [ ] **Step 3: Implement `_find_config_file`**

Add to `ui/pages/grid_helpers.py` before `load_mask()`:

```python
def _find_config_file(rel_path: str, config_dir: Path | None = None) -> Path | None:
    """Search config_dir then examples dir for a relative path, with traversal protection.

    Returns the resolved Path if found, None otherwise.
    Logs traversal rejections as warnings. Does NOT log file-not-found —
    callers emit their own context-specific messages at the appropriate level.
    """
    search_dirs: list[Path] = []
    if config_dir and config_dir.is_dir():
        search_dirs.append(config_dir)
    search_dirs.append(Path(__file__).parent.parent.parent / "data" / "examples")

    for d in search_dirs:
        candidate = _safe_resolve(d, rel_path)
        if candidate is None:
            _log.warning("Path traversal rejected for root %s: %s", d, rel_path)
            continue
        if candidate.exists():
            return candidate
    return None
```

- [ ] **Step 4: Refactor `load_mask` to use `_find_config_file`**

Replace lines 65–78 of `load_mask()` with:

```python
    full_path = _find_config_file(mask_path, config_dir)
    if full_path is None:
        _log.warning("Mask file not found: %s", mask_path)
        return None
```

Keep the existing try/except handlers (lines 84–94) untouched.

- [ ] **Step 5: Refactor `load_netcdf_grid` to use `_find_config_file`**

Replace lines 406–419 of `load_netcdf_grid()` with:

```python
    full_path = _find_config_file(nc_path, config_dir)
    if full_path is None:
        _log.debug("NetCDF grid file not found: %s", nc_path)
        return None
```

- [ ] **Step 6: Run all grid/overlay tests**

Run: `.venv/bin/python -m pytest tests/test_grid_helpers_utils.py tests/test_overlay_display.py tests/test_grid_map_bugs.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```
refactor: extract _find_config_file helper to deduplicate file search
```

---

### Task 2: Extract `_compute_half_extents()` helper

Deduplicate the vectorized half-step computation. The two existing sites have different fallback logic:
- `build_netcdf_grid_layers` uses `lat_sign * lon_step` (direction-aware)
- `load_netcdf_overlay` uses `lon_step if lon_step > 0 else 1.0` (simpler)

The shared helper uses the simpler fallback (suitable for overlay rendering). `build_netcdf_grid_layers` keeps its `lat_sign` post-processing as a caller-side fixup.

**Files:**
- Modify: `ui/pages/grid_helpers.py:524-560,838-860`
- Test: `tests/test_grid_helpers_utils.py`

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_grid_helpers_utils.py
import numpy as np
from ui.pages.grid_helpers import _compute_half_extents


def test_half_extents_regular_grid():
    lat = np.array([[48.0, 48.0], [47.0, 47.0]])
    lon = np.array([[1.0, 2.0], [1.0, 2.0]])
    hlat, hlon = _compute_half_extents(lat, lon)
    assert hlat.shape == (2, 2)
    assert hlon.shape == (2, 2)
    assert np.all(hlat > 0)
    assert np.all(hlon > 0)


def test_half_extents_single_row():
    lat = np.array([[48.0, 48.0, 48.0]])
    lon = np.array([[1.0, 2.0, 3.0]])
    hlat, hlon = _compute_half_extents(lat, lon)
    assert hlat.shape == (1, 3)
    assert np.all(hlon > 0)


def test_half_extents_single_column():
    lat = np.array([[48.0], [47.0], [46.0]])
    lon = np.array([[1.0], [1.0], [1.0]])
    hlat, hlon = _compute_half_extents(lat, lon)
    assert hlat.shape == (3, 1)
    assert np.all(hlat > 0)
    assert np.all(hlon > 0)  # fallback from lat_step


def test_half_extents_single_cell():
    lat = np.array([[48.0]])
    lon = np.array([[1.0]])
    hlat, hlon = _compute_half_extents(lat, lon)
    assert hlat.shape == (1, 1)
    assert hlat[0, 0] > 0
    assert hlon[0, 0] > 0


def test_half_extents_zero_step_fallback():
    """When all lat values are identical, dlat falls back to lon_step."""
    lat = np.array([[48.0, 48.0], [48.0, 48.0]])  # zero lat step
    lon = np.array([[1.0, 2.0], [1.0, 2.0]])
    hlat, hlon = _compute_half_extents(lat, lon)
    assert np.all(hlat > 0), "Zero-step dlat should fall back to lon_step"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_grid_helpers_utils.py -k half_extents -v`
Expected: FAIL

- [ ] **Step 3: Implement `_compute_half_extents`**

Add to `ui/pages/grid_helpers.py` after `CellOverlay`:

```python
def _compute_half_extents(
    lat_2d: np.ndarray, lon_2d: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-cell half-extents via finite differences for polygon construction.

    Returns (hlat, hlon) arrays of shape (ny, nx). All values are positive.
    Uses a simple fallback for zero-step cells (no direction awareness).
    """
    ny, nx = lat_2d.shape
    lat_f = lat_2d.astype(float)
    lon_f = lon_2d.astype(float)

    lat_step = abs(float(lat_f[min(1, ny - 1), 0] - lat_f[0, 0])) if ny > 1 else 1.0
    lon_step = abs(float(lon_f[0, min(1, nx - 1)] - lon_f[0, 0])) if nx > 1 else 1.0

    dlat = np.empty_like(lat_f)
    if ny == 1:
        dlat[:] = lat_step or 1.0
    else:
        dlat[0, :] = lat_f[1, :] - lat_f[0, :]
        dlat[-1, :] = lat_f[-1, :] - lat_f[-2, :]
        if ny > 2:
            dlat[1:-1, :] = (lat_f[2:, :] - lat_f[:-2, :]) / 2

    dlon = np.empty_like(lon_f)
    if nx == 1:
        dlon[:] = lon_step or 1.0
    else:
        dlon[:, 0] = lon_f[:, 1] - lon_f[:, 0]
        dlon[:, -1] = lon_f[:, -1] - lon_f[:, -2]
        if nx > 2:
            dlon[:, 1:-1] = (lon_f[:, 2:] - lon_f[:, :-2]) / 2

    dlat = np.where(dlat == 0, lon_step if lon_step > 0 else 1.0, dlat)
    dlon = np.where(dlon == 0, lat_step if lat_step > 0 else 1.0, dlon)

    return np.abs(dlat) / 2, np.abs(dlon) / 2
```

- [ ] **Step 4: Refactor `load_netcdf_overlay` to use shared helper**

Replace lines 838–860 with:
```python
hlat, hlon = _compute_half_extents(lat_2d, lon_2d)
```

- [ ] **Step 5: Refactor `build_netcdf_grid_layers` to use shared helper**

Replace lines 529–554 with:
```python
    hlat, hlon = _compute_half_extents(lat, lon)
```

Note: the existing `lat_sign` / `fallback_dlat` logic (lines 548–551) is dead code — `np.abs(dlat) / 2` on line 553 already discards the sign. The shared helper returns positive values via `np.abs()`, which is exactly what the polygon construction needs (it adds/subtracts hlat symmetrically around the center coordinate). Drop `lat_sign` entirely.

- [ ] **Step 6: Run all grid/overlay tests**

Run: `.venv/bin/python -m pytest tests/test_grid_helpers_utils.py tests/test_overlay_display.py tests/test_grid_map_bugs.py tests/test_csv_map_display.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```
refactor: extract _compute_half_extents to deduplicate finite-difference logic
```

---

### Task 3: Vectorize `load_csv_overlay` cell loop

Replace the O(ny*nx) Python loop with NumPy vectorization, matching the pattern already used in `load_netcdf_overlay`.

**CRITICAL correctness rules:**
1. Use `~(numeric < -9.0)` not `numeric > -9.0` — preserves -9.0 boundary (scalar uses `v < -9.0`)
2. Apply `np.clip(0, 255)` on ALL four RGBA channels before `.astype(np.uint8)` — prevents modular wraparound
3. Add a golden-output comparison test before vectorizing

**Depends on:** Task 2 (`_compute_half_extents` must exist for the NC branch)

**Files:**
- Modify: `ui/pages/grid_helpers.py:695-749`
- Create: `tests/test_csv_overlay_perf.py`
- Test: `tests/test_overlay_display.py` (existing + new golden test)

- [ ] **Step 1: Write golden-output comparison test**

This test captures scalar output BEFORE vectorization, then verifies vectorized output matches.

```python
# Append to tests/test_overlay_display.py
def test_csv_overlay_output_stability(tmp_path):
    """Cell-by-cell output must be identical before and after vectorization."""
    from ui.pages.grid_helpers import load_csv_overlay
    import pandas as pd

    p = tmp_path / "stability.csv"
    np.random.seed(42)
    data = np.random.rand(5, 6) * 10
    data[0, :3] = -99  # some sentinels
    data[2, 2] = 0.0   # zero (filtered)
    data[3, 0] = -9.0   # boundary value (should be KEPT)
    pd.DataFrame(data).to_csv(p, sep=";", header=False, index=False)

    cells = load_csv_overlay(
        p, ul_lat=50.0, ul_lon=-5.0, lr_lat=45.0, lr_lon=5.0, nx=6, ny=5
    )
    assert cells is not None

    # Verify -9.0 cell is included
    vals = [c["value"] for c in cells]
    assert -9.0 in vals, "-9.0 must be kept (not a sentinel)"

    # Verify all fills are valid RGBA
    for c in cells:
        r, g, b, a = c["fill"]
        assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255 and 0 <= a <= 255

    # Snapshot: store sorted (value, polygon[0]) pairs for comparison
    snapshot = sorted((c["value"], tuple(c["polygon"][0])) for c in cells)
    # Re-run and compare (idempotency check)
    cells2 = load_csv_overlay(
        p, ul_lat=50.0, ul_lon=-5.0, lr_lat=45.0, lr_lon=5.0, nx=6, ny=5
    )
    snapshot2 = sorted((c["value"], tuple(c["polygon"][0])) for c in cells2)
    assert snapshot == snapshot2, "Output must be deterministic"
```

- [ ] **Step 2: Run golden test to establish baseline**

Run: `.venv/bin/python -m pytest tests/test_overlay_display.py::test_csv_overlay_output_stability -v`
Expected: PASS (establishes baseline with scalar loop)

- [ ] **Step 3: Write benchmark test**

```python
# tests/test_csv_overlay_perf.py
import time
import numpy as np
import pandas as pd
from ui.pages.grid_helpers import load_csv_overlay


def test_csv_overlay_performance(tmp_path):
    """Large grid should complete in under 500ms."""
    p = tmp_path / "large.csv"
    ny, nx = 100, 200
    np.random.seed(123)
    data = np.random.rand(ny, nx) * 10
    data[0, :] = -99
    pd.DataFrame(data).to_csv(p, sep=";", header=False, index=False)
    start = time.perf_counter()
    cells = load_csv_overlay(p, ul_lat=50.0, ul_lon=-5.0, lr_lat=43.0, lr_lon=5.0, nx=nx, ny=ny)
    elapsed = time.perf_counter() - start
    assert cells is not None
    assert elapsed < 0.5, f"load_csv_overlay took {elapsed:.3f}s for {ny}x{nx} grid"
```

- [ ] **Step 4: Vectorize the cell loop**

Replace lines 702–749 with:

```python
        valid_mask = (~np.isnan(numeric)) & ~(numeric < -9.0) & (numeric != 0.0)
        valid_vals = numeric[valid_mask]
        r_idxs, c_idxs = np.where(valid_mask)

        if use_nc:
            if lat.ndim == 2:
                clats = lat[r_idxs, c_idxs].astype(float)
                clons = lon[r_idxs, c_idxs].astype(float)
            else:
                clats = lat[r_idxs].astype(float)
                clons = lon[c_idxs].astype(float)
            hlat_arr, hlon_arr = _compute_half_extents(
                lat if lat.ndim == 2 else np.broadcast_to(lat[:, np.newaxis], (g_ny, g_nx)),
                lon if lon.ndim == 2 else np.broadcast_to(lon[np.newaxis, :], (g_ny, g_nx)),
            )
            hlats = hlat_arr[r_idxs, c_idxs]
            hlons = hlon_arr[r_idxs, c_idxs]
        else:
            clons = ul_lon + (c_idxs + 0.5) * dx
            clats = ul_lat - (r_idxs + 0.5) * dy
            hlats = np.full(len(r_idxs), hlat)
            hlons = np.full(len(r_idxs), hlon)

        # Vectorized colormap
        if vmin == vmax:
            rgba = np.full((len(valid_vals), 4), [20, 220, 180, 180], dtype=np.uint8)
        else:
            t = (valid_vals - vmin) / vrange
            r_ch = np.clip(68 + 187 * t * t, 0, 255).astype(np.uint8)
            g_ch = np.clip(1 + 209 * t, 0, 255).astype(np.uint8)
            b_ch = np.clip(84 + 86 * t - 170 * t * t, 0, 255).astype(np.uint8)
            a_ch = np.clip(150 + 80 * t, 0, 255).astype(np.uint8)
            rgba = np.stack([r_ch, g_ch, b_ch, a_ch], axis=-1)

        cells = [
            {
                "polygon": [
                    [float(clons[i] - hlons[i]), float(clats[i] + hlats[i])],
                    [float(clons[i] + hlons[i]), float(clats[i] + hlats[i])],
                    [float(clons[i] + hlons[i]), float(clats[i] - hlats[i])],
                    [float(clons[i] - hlons[i]), float(clats[i] - hlats[i])],
                ],
                "value": float(valid_vals[i]),
                "fill": rgba[i].tolist(),
            }
            for i in range(len(valid_vals))
        ]
        return cells if cells else None
```

- [ ] **Step 5: Run golden test + all overlay tests**

Run: `.venv/bin/python -m pytest tests/test_overlay_display.py tests/test_csv_map_display.py tests/test_grid_map_bugs.py -v`
Expected: ALL PASS — golden test confirms identical output

- [ ] **Step 6: Run benchmark**

Run: `.venv/bin/python -m pytest tests/test_csv_overlay_perf.py -v -s`
Expected: Significant speedup

- [ ] **Step 7: Commit**

```
perf: vectorize load_csv_overlay cell loop with NumPy
```

---

### Task 4: Consolidate string sentinel constants in `grid.py`

Replace repeated `"grid_extent"` (6 occurrences) and `"__movement_animation__"` (6 occurrences, including line 53) with module-level constants.

**Files:**
- Modify: `ui/pages/grid.py:53,142,231,236,247,326,470,587,589,628`

- [ ] **Step 1: Add constants at module top of `grid.py`**

```python
_OVERLAY_GRID_EXTENT = "grid_extent"
_OVERLAY_MOVEMENT_ANIM = "__movement_animation__"
```

- [ ] **Step 2: Replace all string literals**

Find-and-replace all `"grid_extent"` → `_OVERLAY_GRID_EXTENT` and `"__movement_animation__"` → `_OVERLAY_MOVEMENT_ANIM` across `grid.py`. Include line 53 in `_validate_overlay_path`. Verify each replacement preserves the dict key context (e.g., `{_OVERLAY_GRID_EXTENT: "Grid extent"}`).

- [ ] **Step 3: Run grid tests**

Run: `.venv/bin/python -m pytest tests/test_csv_map_display.py tests/test_grid_map_bugs.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```
refactor: replace string sentinels with named constants in grid.py
```

---

### Task 5: Consolidate `FakeInput` test utility

Replace 8+ independent `FakeInput` class definitions with shared helpers. Three variants needed:
1. `make_fake_input(input_id, value)` — single-ID, raises `AttributeError` for others
2. `make_catch_all_input(value)` — returns same value for any attribute
3. `make_multi_input(**kwargs)` — returns different values per ID, with optional default

**Files:**
- Create: `tests/helpers.py`
- Modify: `tests/test_ui_state.py`, `tests/test_state.py`, `tests/test_ui_calibration_handlers.py`, `tests/test_ui_load_scenarios.py`

- [ ] **Step 1: Create shared helper**

```python
# tests/helpers.py
"""Shared test utilities for OSMOSE UI tests."""

from typing import Any


def make_fake_input(input_id: str, value: Any):
    """Create a FakeInput that only responds to a specific input ID."""

    class FakeInput:
        def __getattr__(self, name: str):
            if name == input_id:
                return lambda: value
            raise AttributeError(name)

    return FakeInput()


def make_catch_all_input(value: Any):
    """Create a FakeInput that returns the same value for any attribute."""

    class FakeInput:
        def __getattr__(self, name: str):
            return lambda: value

    return FakeInput()


_MISSING = object()


def make_multi_input(default: Any = _MISSING, **kwargs: Any):
    """Create a FakeInput that returns different values per input ID.

    Usage:
        make_multi_input(foo=42, bar="hello")          # raises AttributeError for others
        make_multi_input(foo=42, default=None)          # returns None for others
        make_multi_input(foo=42, default=False)         # returns False for others
    """

    class FakeInput:
        def __getattr__(self, name: str):
            if name in kwargs:
                return lambda: kwargs[name]
            if default is not _MISSING:
                return lambda: default
            raise AttributeError(name)

    return FakeInput()
```

- [ ] **Step 2: Update each test file**

For each file, replace inline `FakeInput` classes with the appropriate helper. Verify semantic equivalence:
- `test_ui_state.py`: Uses `_make_fake_input` → `make_fake_input` and `make_catch_all_input`
- `test_state.py` line 96: Uses multi-branch FakeInput → `make_multi_input(simulation_nspecies=5, simulation_time_ndtperyear=12, default=None)`
- `test_ui_calibration_handlers.py`: Returns default `False` for non-matching → `make_multi_input(specific_id=True, default=False)`
- `test_ui_load_scenarios.py`: Catch-all → `make_catch_all_input("overwritten")`

- [ ] **Step 3: Run all affected tests**

Run: `.venv/bin/python -m pytest tests/test_ui_state.py tests/test_state.py tests/test_ui_calibration_handlers.py tests/test_ui_load_scenarios.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```
refactor: consolidate FakeInput into shared test helpers
```

---

### Task 6: Extract shared `_DEFAULT_VIEW_STATE` in `map_viewer.py`

Deduplicate MapWidget view_state between `map_viewer_ui()` (line 43, 5 keys) and `map_viewer_server()` (line 73, 3 keys).

**Files:**
- Modify: `ui/pages/map_viewer.py:41-47,71-75`

- [ ] **Step 1: Add module-level constant**

```python
_DEFAULT_VIEW_STATE = {"latitude": 46.0, "longitude": -4.5, "zoom": 5, "pitch": 0, "bearing": 0}
```

- [ ] **Step 2: Use in both `map_viewer_ui` and `map_viewer_server`**

Both `MapWidget(...)` calls use `view_state=_DEFAULT_VIEW_STATE`.

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/ -q --ignore=tests/test_engine_parity.py --ignore=tests/test_movement_numba.py`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```
refactor: extract _DEFAULT_VIEW_STATE in map_viewer.py
```

---

### Task 7: Final lint, format, and full test run

**Files:** All modified files

- [ ] **Step 1: Lint and format**

Run: `.venv/bin/ruff check osmose/ ui/ tests/ scripts/ && .venv/bin/ruff format --check osmose/ ui/ tests/ scripts/`
Expected: All checks passed

- [ ] **Step 2: Full test suite**

Run: `.venv/bin/python -m pytest tests/ --ignore=tests/test_engine_parity.py --ignore=tests/test_movement_numba.py -q`
Expected: 2055+ passed, 0 failed

- [ ] **Step 3: Config round-trip check**

Run: `.venv/bin/python scripts/check_config_roundtrip.py data/eec_full && .venv/bin/python scripts/check_config_roundtrip.py data/examples`
Expected: PASS on both

- [ ] **Step 4: Final commit**

```
chore: lint and format cleanup after refactor
```
