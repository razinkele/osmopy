# UI Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all critical and important issues found by 3 independent code review agents across the OSMOSE Shiny UI (~5300 LOC).

**Architecture:** Each task targets one or two closely related issues. Tasks are ordered by severity (critical first) and grouped by file to minimize context switching. No new files are created — all changes are edits to existing files.

**Tech Stack:** Python Shiny (pyshiny), reactive programming, Bootstrap 5, pandas, xarray

---

## File Map

All changes are edits to existing files:

| File | Tasks | Issues Fixed |
|------|-------|-------------|
| `ui/state.py` | 1 | C1 (dirty flag), C6 (TypeError catch) |
| `ui/pages/run.py` | 2, 3 | C3 (on_progress), C4 (cancel button), C5 (off-by-one), I7 (warnings overwrite) |
| `ui/pages/results.py` | 4 | C2 (broad exception), I4 (temp dir leak) |
| `ui/pages/scenarios.py` | 5 | I8 (no error handling) |
| `ui/pages/advanced.py` | 6 | I9 (import error handling) |
| `ui/pages/calibration_handlers.py` | 7 | I5 (missing isolate) |
| `ui/pages/grid.py` | 8 | I3 (zeros sentinel), I11 (overlay notification), S6 (nsteps fallback) |
| `ui/pages/grid_helpers.py` | 9 | I10 (load_mask), I12 (xr context manager) |
| `ui/pages/fishing.py` | 10 | I6 (sync_inputs loop) |
| `ui/pages/movement.py` | 10 | I6 (sync_inputs loop) |
| `ui/tooltips.py` | 11 | I1 (dead code removal) |
| `ui/components/param_form.py` | 12 | S1 (XSS defense) |

---

### Task 1: Fix `sync_inputs` dirty flag and TypeError catch

**Issues:** C1 (dirty flag never set), C6 (TypeError silently caught)

**Files:**
- Modify: `ui/state.py:105-116`

- [ ] **Step 1: Add `state.dirty.set(True)` after config update in `sync_inputs`**

At line 116, after `state.config.set(cfg)`, add `state.dirty.set(True)`:

```python
        if actual_changes:
            cfg.update(actual_changes)
            state.config.set(cfg)
            state.dirty.set(True)
```

- [ ] **Step 2: Log warning on TypeError instead of silently continuing**

Replace line 105 to only catch `AttributeError` silently, and log `TypeError`:

```python
        try:
            val = getattr(input, input_id)()
        except AttributeError:
            continue
        except TypeError:
            _log.warning("sync_inputs: TypeError reading input '%s'", input_id)
            continue
```

Add at the top of the file (matching project convention):

```python
from osmose.logging import setup_logging
_log = setup_logging("osmose.state")
```

- [ ] **Step 3: Verify the app still runs**

Run: `.venv/bin/python -m pytest tests/ -x -q -k "not test_engine" --timeout=30`

- [ ] **Step 4: Commit**

```
git add ui/state.py
git commit -m "fix: sync_inputs sets dirty flag and logs TypeError instead of swallowing"
```

---

### Task 2: Fix run page — on_progress, cancel button, warnings overwrite

**Issues:** C3 (on_progress reactive read), C4 (cancel never re-disabled), I7 (warnings overwrite)

**Files:**
- Modify: `ui/pages/run.py:235-243,270-289`

- [ ] **Step 1: Fix on_progress to use isolate and cap list length**

Replace lines 270-273:

```python
        def on_progress(line: str):
            with reactive.isolate():
                lines = list(run_log.get())
            lines.append(line)
            if len(lines) > 500:
                lines = lines[-500:]
            run_log.set(lines)
```

- [ ] **Step 2: Re-disable cancel button in finally block**

Replace lines 287-289:

```python
        finally:
            state.busy.set(None)
            ui.update_action_button("btn_run", disabled=False, session=session)
            ui.update_action_button("btn_cancel", disabled=True, session=session)
```

- [ ] **Step 3: Don't overwrite pre-run warnings**

Replace lines 235-241. Keep warnings in the log when continuing:

```python
        if warnings:
            log_lines = ["--- WARNINGS (continuing anyway) ---"]
            log_lines.extend(warnings)
            run_log.set(log_lines)
        else:
            run_log.set([])

        status.set("Writing config...")
        ui.update_action_button("btn_run", disabled=True, session=session)
        ui.update_action_button("btn_cancel", disabled=False, session=session)
```

- [ ] **Step 4: Verify the app still runs**

Run: `.venv/bin/python -m pytest tests/ -x -q -k "not test_engine" --timeout=30`

- [ ] **Step 5: Commit**

```
git add ui/pages/run.py
git commit -m "fix: run page on_progress isolation, cancel button reset, preserve warnings"
```

---

### Task 3: Fix run page — off-by-one in ncell injection and history save exception

**Issues:** C5 (off-by-one), run.py:307 (broad exception in history save)

**Files:**
- Modify: `ui/pages/run.py:88,307`

- [ ] **Step 1: Fix off-by-one in `_inject_random_movement_ncell`**

Change line 88 from `total_cells = nlon * nlat` to:

```python
    total_cells = nlon * nlat - 1
```

- [ ] **Step 2: Narrow history save exception to specific types**

Replace lines 307-313:

```python
            except (OSError, ValueError) as exc:
                _log.warning("Failed to save run history: %s", exc)
                ui.notification_show(
                    f"Run completed but history could not be saved: {exc}",
                    type="warning",
                    duration=8,
                )
```

- [ ] **Step 3: Commit**

```
git add ui/pages/run.py
git commit -m "fix: correct ncell off-by-one, narrow history save exception types"
```

---

### Task 4: Fix results page — narrow exception and temp dir cleanup

**Issues:** C2 (broad exception), I4 (temp dir leak)

**Files:**
- Modify: `ui/pages/results.py:342,665-668`

- [ ] **Step 1: Narrow `_do_load_results` exception to expected types**

Replace line 342 (include ImportError/AttributeError since the function does dynamic imports):

```python
        except (OSError, ValueError, KeyError, ImportError, AttributeError, pd.errors.ParserError) as exc:
```

- [ ] **Step 2: Add cleanup for CSV export temp directory**

Replace lines 665-668:

```python
        tmp_dir = Path(tempfile.mkdtemp(prefix="osmose_export_"))
        atexit.register(shutil.rmtree, str(tmp_dir), True)
        csv_path = tmp_dir / "export.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)
```

Add `import atexit, shutil` at the top of the file if not already present.

- [ ] **Step 3: Commit**

```
git add ui/pages/results.py
git commit -m "fix: narrow results exception types, add temp dir cleanup for CSV export"
```

---

### Task 5: Add error handling to scenario operations

**Issues:** I8 (no error handling in save/load/fork/delete/import)

**Files:**
- Modify: `ui/pages/scenarios.py:101-175,245-257`

- [ ] **Step 1: Add try/except to `handle_save`**

Wrap `mgr.save(scenario)` (line 114) in error handling and add success notification:

```python
        try:
            mgr.save(scenario)
        except (OSError, ValueError) as exc:
            ui.notification_show(f"Failed to save scenario: {exc}", type="error", duration=8)
            return
        state.dirty.set(False)
        _bump()
        ui.notification_show(f"Scenario '{name}' saved.", type="message", duration=3)
```

- [ ] **Step 2: Add try/except to `handle_fork`**

Wrap `mgr.fork(selected, new_name)` (line 163):

```python
        try:
            mgr.fork(selected, new_name)
        except (OSError, ValueError, FileNotFoundError) as exc:
            ui.notification_show(f"Failed to fork scenario: {exc}", type="error", duration=8)
            return
        _bump()
        ui.notification_show(f"Forked as '{new_name}'.", type="message", duration=3)
```

- [ ] **Step 3: Add try/except to `handle_delete`**

Wrap `mgr.delete(selected)` (line 174):

```python
        try:
            mgr.delete(selected)
        except (OSError, FileNotFoundError) as exc:
            ui.notification_show(f"Failed to delete scenario: {exc}", type="error", duration=8)
            return
        _bump()
        ui.notification_show("Scenario deleted.", type="message", duration=3)
```

- [ ] **Step 4: Add try/except to `handle_import_scenarios`**

Wrap `mgr.import_all(zip_path)` (around line 253):

```python
        try:
            mgr.import_all(zip_path)
        except (OSError, zipfile.BadZipFile, ValueError, KeyError) as exc:
            ui.notification_show(f"Failed to import scenarios: {exc}", type="error", duration=8)
            return
```

Add `import zipfile` at the top if not present.

- [ ] **Step 5: Commit**

```
git add ui/pages/scenarios.py
git commit -m "fix: add error handling and success notifications to scenario operations"
```

---

### Task 6: Add error handling to config import

**Issues:** I9 (no error handling in read_file)

**Files:**
- Modify: `ui/pages/advanced.py:81-91`

- [ ] **Step 1: Wrap `reader.read_file(filepath)` in try/except**

Around line 86 where `reader.read_file(filepath)` is called, wrap it:

```python
            try:
                new_cfg = reader.read_file(filepath)
            except (OSError, ValueError, UnicodeDecodeError) as exc:
                ui.notification_show(
                    f"Failed to parse config file: {exc}", type="error", duration=8
                )
                return
```

- [ ] **Step 2: Commit**

```
git add ui/pages/advanced.py
git commit -m "fix: add error handling for config file import parsing"
```

---

### Task 7: Add reactive.isolate to calibration config reads

**Issues:** I5 (config.get without isolate)

**Files:**
- Modify: `ui/pages/calibration_handlers.py:205-208`

- [ ] **Step 1: Wrap config reads in reactive.isolate()**

Replace the bare `state.config.get()` calls around lines 205-208:

```python
        with reactive.isolate():
            current_config = state.config.get()
            source_dir = state.config_dir.get()
```

Then use `current_config` instead of `state.config.get()` in the subsequent code.

- [ ] **Step 2: Commit**

```
git add ui/pages/calibration_handlers.py
git commit -m "fix: add reactive.isolate to calibration config reads per project convention"
```

---

### Task 8: Fix grid page — zeros sentinel, overlay notifications, nsteps fallback

**Issues:** I3 (zeros sentinel), I11 (missing overlay notification), S6 (nsteps fallback)

**Files:**
- Modify: `ui/pages/grid.py:170-178,250-258,440-500`

- [ ] **Step 1: Fix zeros sentinel in `_read_grid_values`**

Replace lines 237-260. Use a flag to track whether inputs were available instead of the all-zeros sentinel (which breaks for equatorial grids at 0,0):

```python
        inputs_available = True
        try:
            ul_lat = float(input.grid_upleft_lat() or 0)
            ul_lon = float(input.grid_upleft_lon() or 0)
            lr_lat = float(input.grid_lowright_lat() or 0)
            lr_lon = float(input.grid_lowright_lon() or 0)
            nx = int(input.grid_nlon() or 0)
            ny = int(input.grid_nlat() or 0)
        except (SilentException, AttributeError):
            inputs_available = False
            ul_lat = ul_lon = lr_lat = lr_lon = 0.0
            nx = ny = 0

        # Fall back to config if inputs haven't populated yet
        if not inputs_available:
            with reactive.isolate():
                cfg = state.config.get()
            try:
                ul_lat = float(cfg.get("grid.upleft.lat", 0))
                ul_lon = float(cfg.get("grid.upleft.lon", 0))
                lr_lat = float(cfg.get("grid.lowright.lat", 0))
                lr_lon = float(cfg.get("grid.lowright.lon", 0))
                nx = int(float(cfg.get("grid.nlon", 0)))
                ny = int(float(cfg.get("grid.nlat", 0)))
            except (ValueError, TypeError):
                _log.warning("Invalid grid coordinate values in config")
                ul_lat = ul_lon = lr_lat = lr_lon = 0.0
                nx = ny = 0

        return ul_lat, ul_lon, lr_lat, lr_lon, nx, ny
```

- [ ] **Step 2: Add notification when overlay file exists but fails to load**

After the overlay loading block (around line 480), when `cells` is `None` but the file existed:

```python
            if cells is None:
                ui.notification_show(
                    f"Could not load overlay data from '{selected}'.",
                    type="warning",
                    duration=5,
                )
```

- [ ] **Step 3: Log warning for nsteps fallback**

Around line 175, add logging when the fallback triggers:

```python
        except (ValueError, TypeError):
            _log.warning(
                "Could not parse simulation.time.ndtperyear=%r, defaulting to 24",
                cfg.get("simulation.time.ndtperyear"),
            )
            nsteps = 24
```

- [ ] **Step 4: Commit**

```
git add ui/pages/grid.py
git commit -m "fix: grid zeros sentinel, overlay failure notification, nsteps fallback warning"
```

---

### Task 9: Fix grid_helpers — load_mask logging and xr context managers

**Issues:** I10 (load_mask), I12 (xr context manager)

**Files:**
- Modify: `ui/pages/grid_helpers.py:41,207-217,~478`

- [ ] **Step 1: Elevate load_mask file-not-found to warning**

Change line 41 from `_log.debug` to `_log.warning`:

```python
        _log.warning("Grid mask file not found: %s", mask_file)
```

- [ ] **Step 2: Use context manager for xr.open_dataset in `load_netcdf_grid`**

Replace the bare `xr.open_dataset` around line 207 with a `with` statement:

```python
    try:
        with xr.open_dataset(full_path) as ds:
            lat = ds[lat_var].values
            lon = ds[lon_var].values
            mask = ds[mask_var].values if mask_var in ds else None
    except (OSError, KeyError) as exc:
        _log.warning("Failed to load NetCDF grid %s: %s", full_path, exc)
        return None
```

- [ ] **Step 3: Apply same context manager fix to `load_netcdf_overlay` (~line 478)**

The `load_netcdf_overlay` function also uses bare `xr.open_dataset`. Apply the same `with` pattern.

- [ ] **Step 4: Commit**

```
git add ui/pages/grid_helpers.py
git commit -m "fix: elevate load_mask log to warning, use xr context managers"
```

---

### Task 10: Batch sync_inputs calls in fishing and movement pages

**Issues:** I6 (N reactive invalidations from loop)

**Files:**
- Modify: `ui/pages/fishing.py:80-100`
- Modify: `ui/pages/movement.py:85-105`

- [ ] **Step 1: Batch updates in fishing.py**

Replace the pattern of calling `sync_inputs` per species/fishery/MPA with a single batch. Collect all keys first, then call `sync_inputs` once:

```python
        all_keys = []
        # ... collect all keys from fishery/MPA loops
        all_keys.extend(fishery_keys)
        all_keys.extend(mpa_keys)
        sync_inputs(input, state, all_keys)
```

- [ ] **Step 2: Same pattern in movement.py**

Apply the same batching pattern to movement sync.

- [ ] **Step 3: Commit**

```
git add ui/pages/fishing.py ui/pages/movement.py
git commit -m "fix: batch sync_inputs calls to avoid N reactive invalidations"
```

---

### Task 11: Remove dead code `ui/tooltips.py`

**Issues:** I1 (dead module)

**Files:**
- Delete: `ui/tooltips.py`

- [ ] **Step 1: Verify no imports exist**

Run: `grep -r "tooltips" ui/ app.py --include="*.py"`

Expected: No imports of `ui.tooltips` or `MANUAL_TOOLTIPS`.

- [ ] **Step 2: Delete the file**

```
git rm ui/tooltips.py
```

- [ ] **Step 3: Commit**

```
git commit -m "chore: remove dead ui/tooltips.py module (never imported)"
```

---

### Task 12: Add html.escape to tooltip content

**Issues:** S1 (XSS defense)

**Files:**
- Modify: `ui/components/param_form.py:13-23`

- [ ] **Step 1: Escape field metadata in tooltip HTML**

In `_tooltip_content`, escape the field values before injecting into HTML:

```python
from html import escape

def _tooltip_content(field: OsmoseField) -> str:
    parts = []
    if field.description:
        parts.append(escape(field.description))
    parts.append(f"<br><small>Key: <code>{escape(field.key_pattern)}</code></small>")
    if field.default is not None:
        parts.append(f"<br><small>Default: {escape(str(field.default))}</small>")
    # ... rest of function
```

- [ ] **Step 2: Commit**

```
git add ui/components/param_form.py
git commit -m "fix: escape tooltip HTML content for XSS defense"
```

---

## Execution Order Summary

| Task | Severity | Est. Time | Dependencies |
|------|----------|-----------|-------------|
| 1 | Critical | 3 min | None |
| 2 | Critical | 5 min | None |
| 3 | Critical | 3 min | After Task 2 (same file) |
| 4 | Critical | 3 min | None |
| 5 | Important | 5 min | None |
| 6 | Important | 2 min | None |
| 7 | Important | 2 min | None |
| 8 | Important | 8 min | None |
| 9 | Important | 3 min | None |
| 10 | Important | 5 min | After Task 1 (depends on sync_inputs) |
| 11 | Important | 1 min | None |
| 12 | Suggestion | 2 min | None |

**Total estimated time:** ~42 minutes

**Parallelizable groups:**
- Group A (independent): Tasks 1, 2, 4, 5, 6, 7, 9, 11, 12
- Group B (depends on same-file changes): Task 3 after Task 2
- Group C (depends on Task 1): Task 10 after Task 1
- Group D: Task 8 (standalone)
