# Deep-Review Latent-Item Hardening — Design

**Date:** 2026-06-21
**Status:** Approved (design)

## Goal

Close the four LATENT-only items deferred from the 2026-06-20 whole-codebase deep
review (`project_deep_review_remediation_2026_06`). None is hit by a bundled
config, so Java parity is unaffected; these are defensive/correctness hardening.

## Scope (4 items)

### Item 1 — Warn on fishing selectivity types 2/3 (silently knife-edged)

`osmose/engine/processes/mortality.py` applies fishing selectivity in the
interleaved per-cell mortality loop at TWO sites (single-school ~L206 and
vectorized ~L718). Both handle `sel_type == 0` (age knife-edge) and
`sel_type == 1` (logistic); every other value (including **2 = Gaussian** and
**3 = log-normal**) falls into the `else` branch and is silently treated as
length knife-edge. `osmose/engine/processes/fishing.py` *does* implement Gaussian
(type 2) / log-normal (type 3), but that path is not the interleaved production
loop — so a config using selectivity type 2 or 3 gets silently wrong fishing
mortality with no warning.

**Fix (warn, not wire — consistent with PR-1's reject-not-wire decision):**
extend `EngineConfig._warn_unsupported_mortality_features()`
(`osmose/engine/config.py:1519`) to detect any species whose
`self.fishing_selectivity_type[i]` is 2 or 3 and emit a throttled warning (reuse
the existing `_warn_once` + `_sp` helpers) telling the user the interleaved loop
knife-edges those types; use a logistic/knife-edge selectivity or the Java engine.
Wiring types 2/3 into the loop is deferred until a config needs it, so it can be
parity-validated as part of delivering it.

Iterate **focal species only** (`for i in range(self.n_species)`), consistent with
the sibling warns: `fishing_selectivity_type` is length n_total with trailing
background entries fixed at `-1` (`config.py:830-832`), and `_sp` keys off the
focal-only `species_names`. (`-1` never matches 2/3, so a full-array scan would
also be correct, but focal-only avoids any `_sp` index confusion.)

### Item 2 — Wizard demo tempdir leaks (atexit-only, unregistered prefix)

`ui/pages/scenarios.py:197` creates the wizard's demo source dir with
`tempfile.mkdtemp(prefix="osmose_wizard_")` and registers cleanup ONLY via
`atexit.register(shutil.rmtree, ...)`. On a long-running Shiny server `atexit`
effectively never fires, so each "New Scenario from demo" leaves a tempdir for the
process lifetime. And `osmose_wizard_` is **not** in `cleanup.py`'s
`_OSMOSE_PREFIXES` (`osmose_run_/cal_/sens_/export_/demo_`), so the periodic
`cleanup_old_temp_dirs()` sweep also misses it.

**IMPORTANT — this dir is session-lifetime state, NOT throwaway.** For demo
sources, `resolve_source` returns `config_dir = config_file.parent` = the
`osmose_wizard_` dir (`scenario_wizard.py:131`, docstring "caller-owned,
persistent"); `_do_wizard_create` publishes it via `state.config_dir.set(...)`
(`scenarios.py:259`); and it is consumed AFTER the create handler returns by
`handle_run` (`run.py:784`, both engines copy maps/data from it),
`calibration_handlers.py`, `map_builder.py`, and `map_viewer.py`. So it must
**not** be removed in the create handler — a prompt `rmtree` there would break
Run/Calibrate/Map for every demo-derived scenario.

**Two more unregistered prefixes leak the same way** (verified): `osmose_maps_`
(`map_builder.py:573`, also session-lifetime — assigned to `state.config_dir`)
and `osmose_val_` (`calibration_handlers.py:1595`, transient validation work_dir
whose per-run subdirs are cleaned but whose parent never is). The export tempdir
(`scenarios.py:542`) uses the registered `osmose_export_` prefix → already swept,
no change.

**Fix (sweep-membership only — no prompt deletion):** add `"osmose_wizard_"`,
`"osmose_maps_"`, and `"osmose_val_"` to `_OSMOSE_PREFIXES` in
`osmose/cleanup.py`, and keep the existing `atexit` registrations. These dirs are
then reclaimed by the age-gated periodic sweep (>24h) and the age-0 atexit sweep
at shutdown, exactly like `osmose_demo_` (which likewise backs a loaded demo's
`config_dir`). No change to the create handler.

### Item 3 — Run telemetry: `duration_sec` hardcoded to 0

`ui/pages/run.py:401` constructs `RunRecord(..., duration_sec=0, ...)`, so every
run persisted to run history records a zero duration and the Results / Compare-Runs
selectors render "(0s)" for every run (`results.py:84`, `scenario_diff.py:64`).
(Calibration history already times correctly via `time.time() - _t0`.)

`RunResult` (`runner.py:81-86`) carries NO timing field (only
returncode/output_dir/stdout/stderr/status/message), so the launch timestamp must
be threaded in — there is no duration to read off the result.

**Fix (thread the start time as a `_handle_result` parameter — NOT a single shared
cell).** The two engine paths cross the `_handle_result` boundary differently, so
a single cell would misfire on one of them: the **Java** path passes `config` as a
direct arg and calls `_handle_result(...)` synchronously (`run.py:377`); the
**Python** path is fire-and-forget, stashing `config` in `_run_config_cell`
(`run.py:820`) which the main-thread poll `_drain_run_done` reads at `run.py:503`.
Mirroring just the Python cell would leave the Java path reading `None`/a stale
start → wrong or double-counted duration on every Java run, shipping silently.

Concretely:
1. In `handle_run`, capture `t0 = time.monotonic()` ONCE before the
   `engine_mode` branch.
2. Add a `start_monotonic` parameter to `_handle_result`.
3. **Java path:** thread `t0` through `_run_java_engine(...)` to its synchronous
   `_handle_result(..., t0)` call.
4. **Python path:** add a `_run_start_cell` companion to `_run_config_cell`
   (set next to `run.py:820`), pass it at the poll site (`run.py:503`).
5. In `_handle_result`, compute
   `duration_sec = max(0.0, time.monotonic() - start_monotonic)` and pass it to
   `RunRecord(duration_sec=...)`, replacing the hardcoded `0` at `run.py:401`.

### Item 4 — numba pure-Python fallback never exercised in CI

`osmose/engine/processes/{mortality,predation,movement}.py` each guard numba with
`try: import numba … _HAS_NUMBA = True/False` and provide pure-Python fallbacks
behind `if not _HAS_NUMBA:`. `pyproject.toml [tool.coverage.run].omit` excludes
all three files. CI always installs numba (it is in `[dev]`). Today only the
**mortality** fallback dispatch is partly exercised — `test_engine_functional_response.py`
`mock.patch`es `mortality._HAS_NUMBA = False`. The **predation** and **movement**
fallback dispatch, and the true import-time `if not _HAS_NUMBA:` setup/symbol-skip
blocks in all three (which a runtime patch cannot reach), never execute in CI — a
regression there would ship undetected. (Confirmed safe to uninstall numba: every
`import numba` is `ImportError`-guarded, every `@njit` symbol is defined only under
`if _HAS_NUMBA:` and referenced only behind `_HAS_NUMBA` guards, and the numba-only
tests are `skipif`/`importorskip`/getattr-guarded, so collection will not crash.)

**Fix (numba-disabled CI leg):** add a dedicated job to `.github/workflows/ci.yml`
that installs `[dev]`, then `pip uninstall -y numba`, asserts numba is absent
(`python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('numba') is None else 1)"` — so the leg cannot silently pass on the JIT path),
then runs a bounded numba-absent-safe subset that drives all three fallbacks:
`tests/test_engine_functional_response.py tests/test_engine_predation_helpers.py
tests/test_movement_numba.py tests/test_jit_determinism.py` plus one short
end-to-end engine run (e.g. a small mortality/predation/movement sim). The
pure-Python path is slower, so keep the subset bounded (not the full suite).
Coverage on this leg is informational; the main `test` job keeps the 90% gate on
the numba path.

## Out of scope

- Wiring selectivity types 2/3 into the interleaved loop (deferred; needs parity
  validation against the Java engine).
- The `.env` credential rotation (separate USER-owned item).
- Any change to bundled-config behavior or Java parity.

## Testing

- **Item 1:** unit test — build an `EngineConfig` (or call the warn method on a
  minimal config) with a species set to `fishing_selectivity_type = 2` and assert
  the warning fires (caplog); assert NO warning for types 0/1. Reset the
  module-global `_WARNED_UNSUPPORTED_MORTALITY` throttle set in the test.
- **Item 2:** unit test — assert each of `"osmose_wizard_"`, `"osmose_maps_"`,
  `"osmose_val_"` is in `_OSMOSE_PREFIXES`, and that `cleanup_old_temp_dirs(0)`
  removes a freshly-made dir of each prefix. (No prompt-cleanup test — the dirs
  are intentionally session-lifetime; the sweep is the contract.)
- **Item 3:** unit test — call `_handle_result` with a known `start_monotonic`
  (monkeypatch `time.monotonic`) and assert the persisted `RunRecord.duration_sec`
  equals the expected elapsed (> 0). Cover BOTH engine paths' threading: assert
  the Java path passes `t0` through `_run_java_engine`, and the Python path via the
  `_run_start_cell` poll site (a value-passing test, not a stale/None read).
- **Item 4:** the CI leg itself is the test; locally, confirm the engine subset
  passes with numba uninstalled.

## Success criteria

1. A config using fishing selectivity type 2 or 3 emits a clear warning (no
   silent knife-edge); types 0/1 stay silent.
2. `osmose_wizard_`, `osmose_maps_`, and `osmose_val_` tempdirs are swept by
   `cleanup_old_temp_dirs` (registered in `_OSMOSE_PREFIXES`); no create-handler
   deletion of the session-lifetime `config_dir`.
3. Run history records real `duration_sec` on BOTH the Python and Java engine
   paths; Results/Compare show non-zero run durations.
4. CI has a numba-disabled leg that exercises the pure-Python engine fallbacks
   and asserts numba is absent.
5. Full suite + ruff + pyright stay green on the numba path; no Java-parity or
   bundled-config behavior change.
