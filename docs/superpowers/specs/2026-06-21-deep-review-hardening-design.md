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

### Item 2 — Wizard demo tempdir leaks (atexit-only, unregistered prefix)

`ui/pages/scenarios.py:197` creates the wizard's demo source dir with
`tempfile.mkdtemp(prefix="osmose_wizard_")` and registers cleanup ONLY via
`atexit.register(shutil.rmtree, ...)`. On a long-running Shiny server `atexit`
effectively never fires, so each "New Scenario from demo" leaks a tempdir for the
process lifetime. Worse, `osmose_wizard_` is **not** in `cleanup.py`'s
`_OSMOSE_PREFIXES` (`osmose_run_/cal_/sens_/export_/demo_`), so the periodic
`cleanup_old_temp_dirs()` sweep also misses it. (The export tempdir at
`scenarios.py:542` uses the registered `osmose_export_` prefix, so it is already
swept — no change needed there.)

**Fix (two parts):**
1. **Prompt cleanup:** the wizard copies/loads the demo config out of the
   tempdir, after which the tempdir is no longer needed — remove it promptly
   (`shutil.rmtree(dest, ignore_errors=True)`) once consumed, inside the wizard
   create handler, rather than relying on `atexit`. Keep an `atexit` registration
   as a fallback only if the dir must outlive the handler (it does not — verify
   during implementation; if it does, fall through to part 2 alone).
2. **Belt-and-suspenders:** add `"osmose_wizard_"` to `_OSMOSE_PREFIXES` in
   `osmose/cleanup.py` so any straggler wizard dirs are caught by the periodic /
   shutdown sweep like every other osmose temp dir.

### Item 3 — Run telemetry: `duration_sec` hardcoded to 0

`ui/pages/run.py:401` constructs `RunRecord(..., duration_sec=0, ...)`, so every
run persisted to run history records a zero duration and the Results / Compare-Runs
selectors render "(0s)" for every run (`results.py:84`, `scenario_diff.py:64`).
(Calibration history already times correctly via `time.time() - _t0`.)

**Fix:** capture a monotonic start time when a run is launched (in `handle_run`,
into a reactive/plain cell) and compute the real elapsed seconds in the result
handler, passing it to `RunRecord(duration_sec=...)`. Prefer a duration already
carried on the run-result object if one exists; otherwise thread the launch
timestamp. Use `time.monotonic()` for elapsed measurement.

### Item 4 — numba pure-Python fallback never exercised in CI

`osmose/engine/processes/{mortality,predation,movement}.py` each guard numba with
`try: import numba … _HAS_NUMBA = True/False` and provide pure-Python fallbacks
behind `if not _HAS_NUMBA:`. `pyproject.toml [tool.coverage.run].omit` excludes
all three files, and CI always installs numba (it is in `[dev]`), so the fallback
dispatch branches never execute in CI — a regression in a fallback path would ship
undetected.

**Fix (full numba-disabled CI leg):** add a dedicated job to
`.github/workflows/ci.yml` that installs `[dev]`, then `pip uninstall -y numba`,
then runs the engine test subset that exercises mortality/predation/movement so
the `_HAS_NUMBA = False` branches actually run. The leg asserts numba is absent
(so it cannot silently pass on the JIT path) and runs a bounded, representative
subset (not the full suite — the pure-Python path is slower). Coverage of these
files is informational on this leg; the existing main `test` job keeps the 90%
gate on the numba path.

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
- **Item 2:** unit test — invoke the wizard create path (or the extracted
  cleanup helper) and assert the `osmose_wizard_` dir is gone afterward; unit
  test that `"osmose_wizard_"` is in `_OSMOSE_PREFIXES` and that
  `cleanup_old_temp_dirs(0)` removes an `osmose_wizard_*` dir.
- **Item 3:** unit test — drive the result handler with a known elapsed and
  assert the persisted `RunRecord.duration_sec` is > 0 / equals the expected
  value (monkeypatch the clock).
- **Item 4:** the CI leg itself is the test; locally, confirm the engine subset
  passes with numba uninstalled.

## Success criteria

1. A config using fishing selectivity type 2 or 3 emits a clear warning (no
   silent knife-edge); types 0/1 stay silent.
2. The wizard no longer leaks `osmose_wizard_` tempdirs; the prefix is swept by
   `cleanup_old_temp_dirs`.
3. Run history records real `duration_sec`; Results/Compare show non-zero run
   durations.
4. CI has a numba-disabled leg that exercises the pure-Python engine fallbacks
   and asserts numba is absent.
5. Full suite + ruff + pyright stay green on the numba path; no Java-parity or
   bundled-config behavior change.
