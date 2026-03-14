# Codebase Analysis Findings

**Date:** 2026-03-12
**Scope:** All production code + test suite
**Priority lens:** Production readiness

## Severity Summary

| Severity | Count |
|----------|-------|
| Critical | 7 |
| High | 18 |
| Medium | 27 |
| Low | 25+ |

> **Updated 2026-03-14:** Added findings from second-round deep analysis (C7, H17-H18, M23-M27). Elevated M1 (java_opts) to C7.

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

### C7. Command injection via unvalidated `java_opts` (elevated from M1)
**Source:** Agent 2 (Security), 2026-03-14 deep analysis
**Files:** `ui/pages/run.py:247`, `osmose/runner.py:49`

User types arbitrary text into the "Java options" text field, which is split on whitespace and passed directly to `subprocess.create_subprocess_exec`. While not shell-invoked, unvalidated JVM flags like `-javaagent:/path/to/malicious.jar` or `-agentlib` enable arbitrary code execution on the server. On a server-deployed instance (via deploy.sh), any user with browser access can exploit this.

**Elevated from M1** because the app is server-deployed and accessible to non-technical users.

**Fix:** Validate `java_opts` against a whitelist of safe JVM flag patterns (memory: `-Xmx`, `-Xms`, `-Xss`; GC flags; system properties `-D`). Reject anything else.

---

## HIGH Findings (18)

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

### H17. `int()` crash on non-integer nspecies/nresource in validator and app header
**Source:** 2026-03-14 deep analysis
**Files:** `osmose/config/validator.py:109-110`, `app.py:283`

`int(config.get("simulation.nspecies", "0"))` crashes with `ValueError` if the config value is a float string like `"3.0"` or non-numeric. In `app.py:283` this crashes the render on every reactive update, leaving the UI permanently broken. In `validator.py` it crashes the validation function (whose purpose is to gracefully handle malformed input).

**Fix:** Use `int(float(...))` with try/except fallback.

### H18. `_build_netcdf_grid_layers` crashes on 1D lat/lon arrays
**Source:** 2026-03-14 deep analysis
**Files:** `ui/pages/grid.py:241`

`ny, nx = lat.shape` assumes `lat` is always 2D, but xarray returns 1D coordinate arrays for regularly-gridded NetCDF files. 1D `lat.shape` returns `(ny,)`, causing `ValueError: not enough values to unpack`. The `except Exception` in `update_grid_map` catches this silently, leaving the map empty with no explanation.

**Fix:** Check `lat.ndim` and broadcast 1D to 2D with `np.meshgrid`.

---

## MEDIUM Findings (27)

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

### Data Integrity (5)
- **M23.** Scenario backup not restored on save failure — if `os.rename(tmp_dir, target)` fails after backup was moved, original data left in backup only (`osmose/scenarios.py:65-77`)
- **M24.** Non-atomic config writes — `filepath.write_text()` can produce truncated files on crash/kill, especially dangerous during calibration (`osmose/config/writer.py:116`)
- **M25.** `_read_grid_values` missing `ValueError` catch — user typing "abc" into numeric grid field crashes the grid preview (`ui/pages/grid.py:647-654`)
- **M26.** Advanced param table shows "-" for all indexed fields — `cfg.get(f.key_pattern, "-")` looks up literal `{idx}` placeholder which never matches (`ui/pages/advanced.py:200`)
- **M27.** `export_dataframe` silently returns empty for unknown output types — typos invisible (`osmose/results.py:328-329`)

### Test Quality (3)
- **M20.** No parallel execution test for calibration (`tests/test_calibration_problem.py`)
- **M21.** Source code inspection test is brittle (`tests/test_sync_config_pages.py:41-47`)
- **M22.** Limited integration test scope — no runner→results→analysis pipeline test

### Test Coverage Gaps (3 — new 2026-03-14)
- **M28.** Path traversal rejection in `import_all` is untested — security-critical check has no test
- **M29.** CLI `cmd_run` and `cmd_report` have zero test coverage
- **M30.** 5 vacuous `hasattr`/`in _EXPORT_MAP` tests in `test_results.py` that can never fail

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
| 7 | Validate java_opts against allowlist | C7 (was M1) | Low |

### Phase 2: High-Priority Hardening
| # | Fix | Findings | Complexity |
|---|-----|----------|------------|
| 8 | HTML escape in report template | H2 | Trivial |
| 9 | Calibration checkbox reactive fix | H3 | Low |
| 10 | Results loading race fix | H4, M8 | Low |
| 11 | Add user notifications to grid loading failures | H7-H10 | Low |
| 12 | Add error handling to results loading | H6 | Low |
| 13 | Narrow ensemble mode exception | H11 | Trivial |
| 14 | Fix run history logging level | H13 | Trivial |
| 15 | Fix `int()` crash on non-integer nspecies | H17 | Trivial |
| 16 | Fix grid crash on 1D lat/lon arrays | H18 | Low |

### Phase 3: Test Quality
| # | Fix | Findings | Complexity |
|---|-----|----------|------------|
| 17 | Add NaN/malformed input edge case tests | H14, H15 | Medium |
| 18 | Add path traversal rejection test for import_all | M28 | Low |
| 19 | Add reactive UI integration test | H16 | High |
| 20 | Fix brittle tests (source inspection, colors) | M21 | Low |
| 21 | Add parallel calibration test | M20 | Medium |
| 22 | Replace vacuous hasattr tests with behavioral tests | M30 | Low |
| 23 | Add CLI cmd_run/cmd_report tests | M29 | Medium |

### Phase 4: Medium-Priority Improvements
| # | Fix | Findings | Complexity |
|---|-----|----------|------------|
| 24 | Config reader sub-file path validation | M4 | Low |
| 25 | Batch species sync, fix param_table reactivity | M6, M7 | Medium |
| 26 | Lazy results loading | M13 | Medium |
| 27 | Close NetCDF cache on directory switch | M14 | Low |
| 28 | Standardize logging initialization | M10 | Medium |
| 29 | Extract grid.py helpers | M9, M12 | Medium |
| 30 | Resilient scenario listing (skip corrupt) | M17 | Low |
| 31 | Fix scenario backup restore on save failure | M23 | Low |
| 32 | Atomic config file writes | M24 | Low |
| 33 | Fix missing ValueError catch in grid inputs | M25 | Trivial |
| 34 | Fix advanced param table for indexed fields | M26 | Low |
| 35 | Log warning for unknown export_dataframe types | M27 | Trivial |

### Phase 5: Low-Priority Polish
Remaining Low findings — address opportunistically during related work.
