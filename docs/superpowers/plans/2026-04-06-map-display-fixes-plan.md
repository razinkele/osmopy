# Map Display UI Fixes — Implementation Plan (revised after critic)

**Spec:** `docs/superpowers/specs/2026-04-06-map-display-review.md`  
**Baseline test command:** `cd osmose-python && .venv/bin/python -m pytest tests/test_grid_helpers.py tests/test_ui_grid.py -q --tb=short`  
**Baseline:** 39 passed (33 grid_helpers + 6 ui_grid)

---

## Critic corrections applied

1. **C1 fix corrected**: `abs(ul_lat - lr_lat) / ny` is still 0 for single-row because boundary is also degenerate. Correct fix: compute `lat_step`/`lon_step` BEFORE `_cell_polygon`, use `dlon` as `dlat` fallback and vice versa (1.0° last resort). Apply same fallback fix to `load_netcdf_overlay` (also missed by original plan).
2. **H1 description corrected**: `set_style()` IS called before early-return. Real bug: `_map.update()` skipped → polygon/legend/view colors don't update. Fix: include theme in the "no change" hash, not just `active_ids`.
3. **H2 guard corrected**: `is_relative_to(config_dir)` breaks valid example files. Correct: validate against each search root `d.resolve()` in the loop.

---

## Phase 1 — Critical Bugs

### C1: Zero-height polygons for single-row/col NetCDF grids

**File:** `ui/pages/grid_helpers.py`  
**Root cause:** `_cell_polygon` computes `dlat` from `lat[row+1,col] - lat[row-1,col]` which is 0 when `ny=1`. Edge-doubling still gives 0. The boundary polygon computation also gives `ul_lat == lr_lat` when `ny=1`, so the plan's fallback `abs(ul_lat-lr_lat)/ny` is also 0.

**Correct fix in `build_netcdf_grid_layers`:**
1. Compute lat/lon step BEFORE `_cell_polygon` definition and boundary polygon:
   ```python
   lat_step = abs(float(lat[0, 0] - lat[min(1, ny - 1), 0]))
   lon_step = abs(float(lon[0, 0] - lon[0, min(1, nx - 1)]))
   # Fallback for single-row/col: use the other dimension's step
   if lat_step == 0:
       lat_step = lon_step if lon_step > 0 else 1.0
   if lon_step == 0:
       lon_step = lat_step if lat_step > 0 else 1.0
   ```
2. Use `lat_step`/`lon_step` in boundary computation (already there, but now non-zero).
3. In `_cell_polygon`, add fallback after edge-doubling:
   ```python
   if dlat == 0:
       dlat = lon_step * (1 if lat[0,0] >= lat[min(1,ny-1), 0] else -1)
   if dlon == 0:
       dlon = lat_step
   ```
   (Sign matches existing convention.)

**Also fix `load_netcdf_overlay`:** same `dlat`/`dlon` computation, same degenerate-polygon bug for single-row overlays. Apply the same fallback logic.

**Also fix boundary polygon** when step is zero:
```python
ul_lat = float(lat.max()) + lat_step / 2
ul_lon = float(lon.min()) - lon_step / 2
lr_lat = float(lat.min()) - lat_step / 2
lr_lon = float(lon.max()) + lon_step / 2
```
This uses `lat_step` directly (already computed above) instead of recomputing from lat differences.

### C2: Animation stops itself — `movement_step` read without `reactive.isolate()`

**File:** `ui/pages/grid.py` — inside `movement_controls` (`@render.ui`)  
**Root cause:** `input.movement_step()` read without `reactive.isolate()` creates a reactive dependency. Every animation tick invalidates `movement_controls`, re-rendering the slider fresh (without animation state).

**Fix:** wrap in `reactive.isolate()`:
```python
with reactive.isolate():
    try:
        current_step = input.movement_step()
    except (SilentException, AttributeError):
        current_step = 0
```

### C3: `load_netcdf_overlay` hardcodes `"lat"`/`"lon"` variable names

**File:** `ui/pages/grid_helpers.py`  
**Fix:** add `var_lat: str = "lat"`, `var_lon: str = "lon"` params; try configured vars first:
```python
olat = ds[var_lat].values if var_lat in ds else (ds["lat"].values if "lat" in ds else fallback_lat)
olon = ds[var_lon].values if var_lon in ds else (ds["lon"].values if "lon" in ds else fallback_lon)
```
In `grid.py` caller, pass `var_lat=cfg.get("grid.var.lat", "lat")` and `var_lon=cfg.get("grid.var.lon", "lon")`.

---

## Phase 2 — High Security & Correctness

### H1: Theme change during animation skips polygon/legend color update

**File:** `ui/pages/grid.py`  
**Real bug (corrected from spec):** `set_style()` IS called before early-return, so basemap tile updates. But `_map.update(layers=...)` is skipped, so polygon fill colors (which depend on `is_dark`) and legend entries are NOT resent.

**Fix:** include `is_dark` in the "no visual change" check:
```python
prev_state = _prev_active_maps.get()  # change type to store (frozenset, bool)
if active_ids == prev_state[0] and prev_state[0] and is_dark == prev_state[1]:
    return  # nothing changed: same maps AND same theme
_prev_active_maps.set((active_ids, is_dark))
```
(Change `_prev_active_maps` from `reactive.Value[frozenset]` to `reactive.Value[tuple[frozenset, bool]]`, initialised to `(frozenset(), False)`)

### H2: Path traversal in `load_mask` and `load_netcdf_grid`

**File:** `ui/pages/grid_helpers.py`  
**Corrected fix:** validate against EACH search root, not just `config_dir`:
```python
for d in search_dirs:
    candidate = (d / mask_path).resolve()
    if not candidate.is_relative_to(d.resolve()):
        _log.warning("Path traversal rejected for root %s: %s", d, mask_path)
        continue
    if candidate.exists():
        full_path = candidate
        break
```
Apply the same pattern to `load_netcdf_grid`.

### H6: `"NULL"` uppercase not filtered in `build_movement_cache`

**File:** `ui/pages/grid_helpers.py:631`  
**Fix:** `if not file_val or file_val.lower() in ("null", "none"):`

### M1: `hasattr` guard for `input.grid_overlay` always True

**File:** `ui/pages/grid.py:~420`  
**Fix:**
```python
try:
    overlay = input.grid_overlay()
except SilentException:
    overlay = "grid_extent"
if not overlay:
    overlay = "grid_extent"
```

---

## Phase 3 — Performance

### H3: Vectorise `build_grid_layers` coordinate computation

**File:** `ui/pages/grid_helpers.py` (the non-NetCDF grid path)  
Replace nested for loop with NumPy meshgrid + list comprehension. Target: 100×100 in <15ms.

```python
dx = (lr_lon - ul_lon) / nx
dy = (ul_lat - lr_lat) / ny

# Vectorise cell corner computation
col_arr = np.arange(nx)
row_arr = np.arange(ny)
lo0 = ul_lon + col_arr * dx          # (nx,)
la0 = ul_lat - row_arr * dy          # (ny,)
lo0g, la0g = np.meshgrid(lo0, la0)  # (ny, nx)
lo1g = lo0g + dx
la1g = la0g - dy

# Mask
if mask is not None:
    mny = min(mask.shape[0], ny)
    mnx = min(mask.shape[1], nx)
    land_g = np.zeros((ny, nx), dtype=bool)
    land_g[:mny, :mnx] = mask[:mny, :mnx] <= 0
else:
    land_g = np.zeros((ny, nx), dtype=bool)

lo0f = lo0g.ravel(); lo1f = lo1g.ravel()
la0f = la0g.ravel(); la1f = la1g.ravel()
rows_f = np.repeat(row_arr, nx)
cols_f = np.tile(col_arr, ny)
land_f = land_g.ravel()

ocean_cells = [
    {"polygon": [[lo0f[i], la0f[i]], [lo1f[i], la0f[i]], [lo1f[i], la1f[i]], [lo0f[i], la1f[i]]],
     "row": int(rows_f[i]), "col": int(cols_f[i]), "type": "ocean"}
    for i in range(len(land_f)) if not land_f[i]
]
land_cells = [
    {"polygon": [[lo0f[i], la0f[i]], [lo1f[i], la0f[i]], [lo1f[i], la1f[i]], [lo0f[i], la1f[i]]],
     "row": int(rows_f[i]), "col": int(cols_f[i]), "type": "land"}
    for i in range(len(land_f)) if land_f[i]
]
```

---

## Phase 4 — Medium Fixes & DRY

### Extract `_zoom_for_span` (L4)
```python
def _zoom_for_span(span: float, default: float = 5.0) -> float:
    if span <= 0:
        return default
    return max(1.0, min(15.0, math.log2(360 / span) - 0.5))
```
Replace duplicated code in `grid.py` and `grid_helpers.py`.

### Init `n_maps` from loaded config in `movement.py` (L6)
```python
@reactive.effect
def sync_n_maps_from_config():
    state.load_trigger.get()
    with reactive.isolate():
        cfg = state.config.get()
    count = sum(1 for k in cfg if re.match(r"movement\.file\.map\d+", k) and cfg[k])
    if count > 0:
        ui.update_numeric("n_maps", value=count)
```

---

## Phase 5 — Test Coverage

### New test files:

**`tests/test_grid_map_bugs.py`** (regression tests for C1/C2/C3):
- `test_single_row_nonzero_height`: `ny=1`, assert polygon height > 0
- `test_single_col_nonzero_width`: `nx=1`, assert polygon width > 0  
- `test_netcdf_overlay_custom_var_names`: overlay with `"latitude"`/`"longitude"` vars
- `test_theme_change_during_animation`: H1 regression — polygon colors update with theme

**`tests/test_grid_security.py`** (path traversal):
- `test_load_mask_rejects_traversal`: `"../../etc/passwd"` → returns None
- `test_load_mask_allows_examples`: example dir file → returns data
- `test_load_netcdf_grid_rejects_traversal`

**`tests/test_grid_overlay.py`** (L1, L2):
- `test_csv_overlay_matching_dimensions`
- `test_csv_overlay_all_nan` → None
- `test_netcdf_overlay_custom_var_names`
- `test_netcdf_overlay_single_row_nonzero` (catches C1 in overlay path)
