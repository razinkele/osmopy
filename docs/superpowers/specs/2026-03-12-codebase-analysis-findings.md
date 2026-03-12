# Codebase Analysis Findings

**Date:** 2026-03-12
**Scope:** All production code + test suite
**Priority lens:** Production readiness

## Severity Summary

| Severity | Count |
|----------|-------|
| Critical | 6 |
| High | 16 |
| Medium | 22 |
| Low | 25+ |

---

## CRITICAL Findings (6)

### C1. Thread-unsafe reactive value writes from calibration threads
**Source:** Agent 3 (Reactive), Agent 1 (Silent Failures)
**Files:** `ui/pages/calibration_handlers.py:186-236, 240-276`

`run_surrogate()` and `run_optimization()` run on background `threading.Thread` but call `.get()` and `.set()` on Shiny `reactive.value` objects. Shiny's reactive values are not thread-safe. This causes:
- Race conditions (partial updates visible to UI)
- Missed reactive invalidations (UI never updates)
- Potential crashes during concurrent reactive graph evaluation
- The `cal_history.get()` + `cal_history.set()` pattern is a TOCTOU race

**Fix:** Use a thread-safe queue + `reactive.poll()` or `reactive.invalidate_later()` to relay updates from worker threads to the main event loop.

### C2. Scenario load bypasses all state guards — effectively broken
**Source:** Agent 3 (Reactive)
**Files:** `ui/pages/scenarios.py:120-128`

`handle_load()` sets `state.config.set(loaded.config)` without:
- Setting `state.loading = True` (sync effects race and overwrite loaded config)
- Bumping `state.load_trigger` (forms don't re-render with new values)
- Updating metadata (`config_name`, `species_names`, `dirty`, `config_dir`)
- Updating `n_species` input

**Result:** Loading a scenario silently fails — forms show stale values, sync effects overwrite the loaded config.

**Fix:** Mirror the `handle_load_example` pattern from setup.py (set loading flag, update config, bump trigger, update metadata, clear loading in finally).

### C3. Surrogate calibration thread has no top-level exception handler
**Source:** Agent 1 (Silent Failures)
**Files:** `ui/pages/calibration_handlers.py:186-236`

If `SurrogateCalibrator.fit()` or `find_optimum()` throws, the thread dies silently. The UI freezes showing the last status message indefinitely. No error logged, no user notification.

**Fix:** Add top-level `try/except Exception` that logs at ERROR and sets `surrogate_status` to failure message (matching the NSGA-II path pattern).

### C4. Broad `except Exception` in calibration evaluation masks all failures
**Source:** Agent 1 (Silent Failures)
**Files:** `osmose/calibration/problem.py:85-98`

The entire candidate evaluation (subprocess, file I/O, objective function) is wrapped in `except Exception` that logs at WARNING and scores as `np.inf`. A misconfigured objective that always raises `TypeError` produces an entire population of `inf` — optimizer grinds through all generations with zero useful progress.

**Fix:** Narrow to expected failures (`subprocess.TimeoutExpired`, `CalledProcessError`, `FileNotFoundError`, `OSError`). Track failure count, abort if >50% of population fails. Propagate unexpected exceptions.

### C5. Config import (Advanced page) bypasses state guards
**Source:** Agent 3 (Reactive)
**Files:** `ui/pages/advanced.py:155-165`

`confirm_import()` sets config without `state.loading` guard or `load_trigger` bump — same broken pattern as C2.

**Fix:** Set `state.loading = True`, set config, bump `load_trigger`, set `state.loading = False`.

### C6. Vacuous test assertion (`or True`) gives false confidence
**Source:** Agent 5 (Test Quality)
**Files:** `tests/test_param_form.py:77-92`

```python
assert "a" in html.replace(".", "_").lower() or True  # always True
```

**Fix:** Remove `or True`, write proper assertion checking advanced field filtering.

---

## HIGH Findings (16)

### H1. Path traversal in scenario save/delete via user-supplied name
**Source:** Agent 2 (Security)
**Files:** `osmose/scenarios.py:57,108`

`scenario.name` from UI input is joined with `storage_dir` without path validation. Name like `../../etc` traverses outside. `delete()` with `shutil.rmtree()` could delete arbitrary directories.

**Fix:** Add `resolve()` + `is_relative_to()` check (same pattern already used in `import_all()`).

### H2. HTML injection via `| safe` filter in report template
**Source:** Agent 2 (Security)
**Files:** `osmose/templates/report.html:25`, `osmose/reporting.py:75-76`

`summary_html` from `DataFrame.to_html()` bypasses Jinja2 autoescape via `| safe`. Species names from output CSVs could contain `<script>` tags.

**Fix:** Use `table.to_html(escape=True)` explicitly.

### H3. Calibration checkbox UI resets on every config change
**Source:** Agent 3 (Reactive)
**Files:** `ui/pages/calibration.py:163-176`

`free_param_selector` takes full reactive dependency on `state.config`. Any parameter change anywhere re-renders the checkbox list, destroying selections.

**Fix:** Use `state.load_trigger.get()` as trigger, read config inside `reactive.isolate()`.

### H4. `results_loaded` race causes double-load
**Source:** Agent 3 (Reactive)
**Files:** `ui/pages/results.py:342-346`

`_reset_results_loaded` sets `results_loaded = False` when `output_dir` changes, but `_do_load_results` also sets both `output_dir` and `results_loaded = True`, creating a race condition.

**Fix:** Remove effect or add changed-value check with `reactive.isolate()`.

### H5. No error notification for calibration/sensitivity thread failures
**Source:** Agent 3 (Reactive), Agent 1 (Silent Failures)
**Files:** `ui/pages/calibration_handlers.py:270-272, 347-349`

Errors set `surrogate_status` from a background thread (unreliable per C1). Sensitivity errors go to wrong UI element.

**Fix:** Use thread-safe queue for error relay. Show errors on correct tab.

### H6. `_do_load_results` has no error handling
**Source:** Agent 3 (Reactive)
**Files:** `ui/pages/results.py:253-318`

If any `res.*()` call fails (corrupt CSV, missing files), the function crashes with no user notification.

**Fix:** Wrap in try/except, show `ui.notification_show()`.

### H7-H10. Grid page catches all exceptions in 4 file-loading functions
**Source:** Agent 1 (Silent Failures)
**Files:** `ui/pages/grid.py:65-67, 216-228, 466-468, 529-531`

`_load_mask`, `_load_netcdf_grid`, `_load_csv_overlay`, `_load_netcdf_overlay` all use `except Exception` → return None with no user notification. Grid preview silently shows nothing.

**Fix:** Narrow exceptions, add `ui.notification_show()` in grid server.

### H11. Empty `except Exception: pass` for ensemble mode
**Source:** Agent 1 (Silent Failures)
**Files:** `ui/pages/results.py:406-409`

**Fix:** Narrow to `except (AttributeError, TypeError)`.

### H12. Sensitivity failed samples corrupt Sobol analysis with `inf`
**Source:** Agent 1 (Silent Failures)
**Files:** `ui/pages/calibration_handlers.py:341-343`

`inf` values passed to SALib produce NaN Sobol indices.

**Fix:** Track failure count, abort if >10% of samples fail.

### H13. Run history save errors swallowed at DEBUG level
**Source:** Agent 1 (Silent Failures)
**Files:** `ui/pages/run.py:292-293`

**Fix:** Change to `_log.warning`, narrow exception, add user notification.

### H14. No NaN/Inf edge case tests for analysis functions
**Source:** Agent 5 (Test Quality)
**Files:** `tests/test_analysis.py`

OSMOSE outputs can contain NaN (species extinction). No tests cover this.

### H15. No malformed input tests for config reader
**Source:** Agent 5 (Test Quality)
**Files:** `tests/test_config_reader.py`

Missing: no separator, multiple separators, BOM, binary content.

### H16. No reactive UI integration tests
**Source:** Agent 5 (Test Quality)

UI tests only test extracted helper functions, not actual Shiny reactive behavior. The critical `reactive.isolate()` patterns have no automated verification.

---

## MEDIUM Findings (22)

### Security (5)
- **M1.** Unvalidated `java_opts` from UI — could open debug ports (`ui/pages/run.py:246`)
- **M2.** Override key injection in calibration subprocess commands (`osmose/calibration/problem.py:125`)
- **M3.** Path traversal in `RunHistory.load_run()` (`osmose/history.py:53`)
- **M4.** Config reader follows arbitrary sub-file paths (`osmose/config/reader.py:44`)
- **M5.** No scenario name sanitization before filesystem use (`ui/pages/scenarios.py:102`)

### Reactive/UI (3)
- **M6.** `param_table` in Advanced page re-renders on every config change (`ui/pages/advanced.py:196`)
- **M7.** `sync_species_inputs` calls `update_config` 60+ times unbatched (`ui/pages/setup.py:147-168`)
- **M8.** Auto-load results double-loads (consequence of H4) (`ui/pages/results.py:329-340`)

### Architecture (4)
- **M9.** `ui/pages/grid.py` is 854 lines mixing 3 concerns — extract helpers
- **M10.** Inconsistent logging initialization (3 different patterns across modules)
- **M11.** Inconsistent error handling style (raise vs return empty vs return sentinel)
- **M12.** Duplicate grid cell construction code (3 near-identical copies in grid.py)

### Performance (3)
- **M13.** `_do_load_results` loads all 16 output types eagerly (`ui/pages/results.py:253-277`)
- **M14.** `OsmoseResults._nc_cache` holds open xarray Datasets indefinitely — no eviction
- **M15.** Docstring coverage gaps on critical public functions (registry, calibration handlers)

### Silent Failures (4)
- **M16.** Unparseable config lines silently skipped (`osmose/config/reader.py`)
- **M17.** One corrupt JSON blocks all scenario listing (`osmose/scenarios.py:88-104`)
- **M18.** `csv_maps_to_netcdf` silently produces nothing on invalid input (`osmose/grid.py:97-100`)
- **M19.** Download handler returns None with no user feedback (`ui/pages/results.py:585-600`)

### Test Quality (3)
- **M20.** No parallel execution test for calibration (`tests/test_calibration_problem.py`)
- **M21.** Source code inspection test is brittle (`tests/test_sync_config_pages.py:41-47`)
- **M22.** Limited integration test scope — no runner→results→analysis pipeline test

---

## LOW Findings (25+)

Architecture: results.py inline charts should move to plotting.py, demo.py mixes concerns, redundant import in validator.py, `copy_data_files` cross-page import, duplicate `_require_columns` helper, `ensure_templates()` import side effects, ensemble O(N*T) loop.

Security: temp directories never cleaned up, path disclosure in validation errors.

Silent Failures: `get_theme_mode` catches all exceptions, `cancel()` missing ProcessLookupError handling, delete silently succeeds for non-existent scenarios, movement ncell workaround silently skipped, version tuple fallback over-migrates.

Test Quality: smoke-only assertions in plotting tests, exact color hex assertions, non-deterministic random seeds, temp file leaks, duplicate tests between scenario files.

---

## Recommended Fix Order (Implementation Plan)

### Phase 1: Critical Safety (blocks production use)
| # | Fix | Findings | Complexity |
|---|-----|----------|------------|
| 1 | Thread-safe calibration communication | C1, H5 | High |
| 2 | Scenario/import load state guards | C2, C5 | Medium |
| 3 | Surrogate thread exception handler | C3 | Low |
| 4 | Narrow calibration exception + failure tracking | C4, H12 | Medium |
| 5 | Path traversal protection (scenarios, history) | H1, M3, M5 | Low |
| 6 | Fix vacuous test assertion | C6 | Trivial |

### Phase 2: High-Priority Hardening
| # | Fix | Findings | Complexity |
|---|-----|----------|------------|
| 7 | HTML escape in report template | H2 | Trivial |
| 8 | Calibration checkbox reactive fix | H3 | Low |
| 9 | Results loading race fix | H4, M8 | Low |
| 10 | Add user notifications to grid loading failures | H7-H10 | Low |
| 11 | Add error handling to results loading | H6 | Low |
| 12 | Narrow ensemble mode exception | H11 | Trivial |
| 13 | Fix run history logging level | H13 | Trivial |

### Phase 3: Test Quality
| # | Fix | Findings | Complexity |
|---|-----|----------|------------|
| 14 | Add NaN/malformed input edge case tests | H14, H15 | Medium |
| 15 | Add reactive UI integration test | H16 | High |
| 16 | Fix brittle tests (source inspection, colors) | M21 | Low |
| 17 | Add parallel calibration test | M20 | Medium |

### Phase 4: Medium-Priority Improvements
| # | Fix | Findings | Complexity |
|---|-----|----------|------------|
| 18 | Validate java_opts against allowlist | M1 | Low |
| 19 | Config reader sub-file path validation | M4 | Low |
| 20 | Batch species sync, fix param_table reactivity | M6, M7 | Medium |
| 21 | Lazy results loading | M13 | Medium |
| 22 | Close NetCDF cache on directory switch | M14 | Low |
| 23 | Standardize logging initialization | M10 | Medium |
| 24 | Extract grid.py helpers | M9, M12 | Medium |
| 25 | Resilient scenario listing (skip corrupt) | M17 | Low |

### Phase 5: Low-Priority Polish
Remaining Low findings — address opportunistically during related work.
