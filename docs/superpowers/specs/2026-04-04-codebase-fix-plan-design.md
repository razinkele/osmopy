# Codebase Fix Plan — Deep Review Remediation

**Date:** 2026-04-04
**Scope:** 47 action items (from 7-agent deep review: CodeRabbit, Silent Failure Hunter, Type Design Analyzer, Engine Code Reviewer, Code Standards Reviewer, Comment Analyzer, Test Coverage Analyzer) + 5 deferred
**Strategy:** Moderate — batch safe fixes, isolate high-fan-out changes with individual verification gates
**Baseline:** 1761 tests passing, 14 skipped, 0 failures, lint clean

---

## Verification Protocol

Every phase ends with a verification gate. No phase begins until the prior gate passes.

- **Gate A (quick):** `.venv/bin/python -m pytest tests/ -x -q` — full suite
- **Gate B (parity):** `.venv/bin/python scripts/validate_engines.py --years 1` — Java output comparison
- **Gate C (lint):** `.venv/bin/ruff check osmose/ ui/ tests/` — zero warnings

All three gates must pass green after Phase 2 and Phase 3. Phase 1 and Phase 4 require only Gate A + Gate C.

---

## Phase 1 — Zero-Risk Quick Wins

**Risk:** Near-zero. All changes are 1-5 line edits. C4 and C5 fix silent-default bugs, so correctly-configured bioenergetics output and movement slider will start working as intended. No impact on simulation numerical output.
**Gate:** A + C
**Estimated items:** 15

### 1.1 Config key case fixes (C4, C5)

**C4** — `ui/pages/grid.py:186`
```python
# Before:
nsteps = int(float(cfg.get("simulation.time.ndtPerYear", "24") or "24"))
# After:
nsteps = int(float(cfg.get("simulation.time.ndtperyear", "24") or "24"))
```
**Why:** Config reader lowercases all keys at `reader.py:93`. Mixed-case lookup always returns the default. Movement slider range is always 0-23 regardless of actual ndtperyear.

**C5** — `osmose/engine/config.py:1336`
```python
# Before:
output_bioen_sizeinf=cfg.get("output.bioen.sizeInf.enabled", "false").lower() == "true",
# After:
output_bioen_sizeinf=cfg.get("output.bioen.sizeinf.enabled", "false").lower() == "true",
```
**Why:** Same root cause. Bioen size-infinity output is permanently disabled even when configured.

### 1.2 Calibration key pattern fix (C6)

**File:** `osmose/calibration/configure.py:11-12`
```python
# Before:
r"mortality\.natural\.rate\.sp\d+": (0.001, 2.0),
r"mortality\.natural\.larva\.rate\.sp\d+": (0.001, 10.0),
# After:
r"mortality\.additional\.rate\.sp\d+": (0.001, 2.0),
r"mortality\.additional\.larva\.rate\.sp\d+": (0.001, 10.0),
```
**Why:** Schema uses `mortality.additional.rate` (confirmed at `schema/species.py:313`). Engine uses `mortality.additional.rate` (confirmed at `engine/config.py:801`). The deprecated `mortality.natural.rate` keys never match any config entry, silently disabling auto-detection of mortality parameters for calibration.

### 1.3 Assert → if/raise for bioenergetics (C3)

**File:** `osmose/engine/simulate.py:151-164`

Replace 14 `assert config.bioen_X is not None` with:
```python
_BIOEN_REQUIRED = [
    "bioen_beta", "bioen_assimilation", "bioen_c_m", "bioen_eta",
    "bioen_r", "bioen_m0", "bioen_m1", "bioen_e_mobi", "bioen_e_d",
    "bioen_tp", "bioen_e_maint", "bioen_i_max", "bioen_theta", "bioen_c_rate",
]
for attr in _BIOEN_REQUIRED:
    if getattr(config, attr) is None:
        raise ValueError(f"Bioenergetics enabled but {attr} is None — check config")
```
**Why:** `assert` is stripped by `python -O`. These are critical safety checks that prevent NaN propagation through the entire bioenergetics pipeline.

### 1.4 Numba fallback: warnings.warn → _log.warning (H7)

**Files:** `osmose/engine/processes/mortality.py`, `movement.py`, `predation.py` (Numba ImportError handlers)

Replace `warnings.warn(..., ImportWarning, ...)` with `_log.warning(...)`. All 3 files currently lack a logger — add `from osmose.logging import setup_logging` and `_log = setup_logging("osmose.engine.processes.<module>")` to each, matching the project convention used in `runner.py`, `scenarios.py`, etc.
**Why:** `ImportWarning` is filtered by default in many Python environments. Users running via Shiny Server never see the 10-100x slowdown warning.

### 1.5 Missing calibration __all__ exports (H8)

**File:** `osmose/calibration/__init__.py`

Add imports and `__all__` entries for: `yield_rmse`, `catch_at_size_distance`, `size_at_age_rmse`, `weighted_multi_objective`.

### 1.6 Stale comments and docstring fixes (H9, M10-M14)

| ID | File | Change |
|----|------|--------|
| H9 | `simulate.py:49` | Remove stale "Stub process functions (replaced in Phase 2-7)" comment |
| M10 | `schema/species.py:88-93` | `"Java class implementing the growth model"` → `"Growth model type (Von Bertalanffy or Gompertz)"` |
| M10 | `schema/grid.py:8-10` | `"Java class implementing the grid"` → `"Grid implementation type"` |
| M10 | `schema/ltl.py:10-13` | `"Java class implementing the LTL forcing"` → `"LTL forcing method"` |
| M11 | `schema/species.py:176-180` | `"Egg size at hatching"` → `"Egg size (initial larval length)"` |
| M12 | `schema/species.py:387-390` | unit `"tons"` → `"tonnes"` |
| M13 | `simulate.py:121` | `"Apply Von Bertalanffy growth"` → `"Apply growth (Von Bertalanffy or Gompertz)"` |
| M14 | `predation.py:3-4` | Add note that interleaved mortality path is the primary codepath |

### 1.7 Missing type annotations (M15)

**File:** `osmose/config/validator.py`

Add type hints to `validate_config`, `validate_field`, `check_file_references` parameters.

---

## Phase 2 — Engine Correctness

**Risk:** Medium. Changes affect simulation output. Each fix verified against parity baselines.
**Gate:** A + B + C
**Estimated items:** 5

### 2.1 Division by zero guard in prey window (C1)

**Files:** `predation.py:409`, `mortality.py:327`, `mortality.py:778`

**Root cause:** `size_ratio_min_2d` is initialized with `np.zeros` (`config.py:958`). For background species with no feeding stages (`n_st == 0`), both `r_min` and `r_max` remain `0.0`. When both are zero, the overlap check at line 781 already skips resources (both bounds become `inf`). However, if only `r_min == 0` while `r_max > 0` (possible via malformed config bypassing schema validation, where schema enforces `min_val=1.0`), `pred_len / r_min` produces `+inf`, making resources incorrectly accessible. Focal species default to `1.0` (`config.py:890`) so this is a narrow edge case.

**Fix:** At each of the 3 resource-loop locations, add a guard before the division:
```python
if r_min_val <= 0:
    continue  # no valid prey window — skip resource
```

**Locations:** `predation.py:409`, `mortality.py:327`, `mortality.py:778`. Note: the school-on-school loop (`mortality.py:734`) uses `ratio < r_min` which is safe when `r_min == 0` (all positive ratios pass, but `r_max == 0` then skips via `ratio >= r_max`).

**Verification:** Parity test (Gate B). No parity change expected — test configs always have valid `sizeratio.min >= 1.0`. Add a targeted unit test with `size_ratio_min=0` to confirm the guard triggers.

### 2.2 CSV grid dimension validation (C7)

**File:** `osmose/engine/movement_maps.py:74-76`

**Root cause:** `row_values[:nx]` silently truncates. If CSV has fewer columns, remaining cells get uninitialized memory from `np.empty`.

**Fix:**
```python
if len(row_values) < nx:
    raise ValueError(
        f"Movement map CSV row {csv_row_idx} has {len(row_values)} columns, "
        f"expected {nx} (grid nx)"
    )
```

### 2.3 sync_inputs AttributeError fix (C8)

**File:** `ui/state.py:106-112`

**Root cause:** `getattr(input, input_id)()` — the `except AttributeError` catches both "input doesn't exist" and "accessor raised AttributeError internally." There is also a separate `except TypeError` at line 110 that must be preserved.

**Fix:**
```python
if not hasattr(input, input_id):
    continue
try:
    val = getattr(input, input_id)()
except TypeError:
    _log.warning("sync_inputs: TypeError reading input '%s'", input_id)
    continue
```

### 2.4 Empty DataFrame logging (H5)

**File:** `osmose/results.py:56-59` and similar locations

Add `_log.info("No files matching '%s' in %s", pattern, self.output_dir)` before returning empty DataFrame in non-strict mode.

### 2.5 ncell injection warning (from Silent Failure Finding 3)

**File:** `ui/pages/run.py:81-87`

Add `_log.warning(...)` before the early return when grid dimensions are invalid, so users know the Java off-by-one workaround was skipped.

---

## Phase 3 — Structural Hardening

**Risk:** High for some items. Changes touch files with 29-52 importers. Each sub-item gets its own test gate.
**Gate:** A + B + C after each sub-item
**Estimated items:** 7, each verified individually

### 3.1 `OsmoseField.__post_init__` validation (H3a)

**File:** `osmose/schema/base.py` (29 importers)

Add `__post_init__` to verify:
- If `indexed` then `"{idx}" in key_pattern` (and vice versa)
- If `choices` is set then `param_type == ENUM`
- If both `min_val` and `max_val` are set then `min_val <= max_val`

**Risk:** Any existing field that violates these invariants will crash at import time. Pre-check before committing:
```
.venv/bin/python -c "from osmose.schema import ALL_FIELDS; import itertools; fs=list(itertools.chain.from_iterable(ALL_FIELDS)); print(f'{len(fs)} fields validated')"
```
Must confirm all 215 fields pass before committing. (Verified: all 215 pass the proposed invariants as of 2026-04-04.)

### 3.2 `Grid.__init__` validation (H3b)

**File:** `osmose/engine/grid.py` (39 importers)

`Grid` is a regular class (not a dataclass). Add validation at the end of `__init__` to verify:
- `ocean_mask.shape == (self.ny, self.nx)`
- `ny >= 1` and `nx >= 1`
- `lat.shape == (ny,)` if lat is not None (and similarly for lon)

### 3.3 `SchoolState.__post_init__` shape check (H3c)

**File:** `osmose/engine/state.py` (52 importers)

Add lightweight length check: all 1D arrays have the same length, `n_dead` has shape `(n, len(MortalityCause))`.

**Risk:** Highest fan-out. Must verify the `create()`, `replace()`, `append()`, and `compact()` methods all produce valid states. Run full parity after this item.

### 3.4 NetCDF context managers (H2)

**Files:** `osmose/engine/physical_data.py:31`, `osmose/engine/background.py:296`, `osmose/engine/resources.py:207`

Wrap `xr.open_dataset()` in `with` blocks where data is immediately materialized. Note: `resources.py:207` stores `self._forcing_data` as a cached dataset — for this case, add a `close()` method to the class or defer to Phase 4 if the caching pattern is load-bearing. `physical_data.py` and `background.py` materialize data immediately and can use `with` blocks directly.

### 3.5 `OsmoseResults` context manager (H4)

**File:** `osmose/results.py` (18 importers)

Add `__enter__` / `__exit__` that calls `close_cache()`. Existing code continues to work unchanged; the context manager is opt-in.

### 3.6 Larvae flag fix (C2, latent)

**File:** `osmose/engine/simulate.py:171`

```python
# Before:
is_larvae = state.is_egg
# After:
is_larvae = state.age_dt < state.first_feeding_age_dt
```

**Why latent:** `first_feeding_age_dt` is currently hardcoded to 1 in `reproduction.py:90`, making `is_egg` equivalent. But the correct check is `age_dt < first_feeding_age_dt`, which will remain correct if the config is ever made configurable. Zero behavioral change with current configs.

### 3.7 Config key case map: eliminate module-level global (H1 partial — reader only)

**File:** `osmose/config/reader.py:14`

Remove `_last_key_case_map` module-level global. The reader already stores `self.key_case_map` as an instance attribute (`reader.py:33`). The global at line 42-43 copies from instance to module — just remove the `global` statement and the copy.

The consumer is `ui/pages/run.py:122-129`, which imports the global to restore original key case when writing temp config for Java. The reader instance is NOT in scope there (it's created in `setup.py:123` and `advanced.py:88`). **Fix:** Add a `key_case_map: dict[str, str]` attribute to `AppState` (`ui/state.py`). Populate it after each config load in `setup.py` and `advanced.py` via `state.key_case_map = reader.key_case_map`. Update `run.py:122-129` to read from `state.key_case_map` instead of the module global.

**Why partial:** Only fixing the reader global. The engine globals (`_config_dir`, `_diet_matrix`, `_tl_weighted_sum`) are not currently triggered by concurrent code and are a larger refactor deferred to a future engine architecture pass.

---

## Phase 4 — Tests + Polish

**Risk:** Low. New test files only; production code changes are isolated and low-fan-out.
**Gate:** A + C
**Estimated items:** 20

### 4.1 New tests for critical coverage gaps

| Test | File | What it verifies |
|------|------|-----------------|
| T1 | `test_engine_mortality.py` (add to existing) | `out_mortality` formula: `n_dead == N * (1 - exp(-rate/n_dt_per_year))` |
| T2 | `test_engine_mortality.py` (add to existing) | `additional_mortality_by_dt` time-varying rates, modular index |
| T3 | `test_engine_mortality_loop.py` (add to existing) | Mortality loop Numba vs Python parity (same pattern as `test_engine_predation_helpers.py`) |
| T4 | `test_engine_config.py` | `EngineConfig.from_dict({})` raises, non-numeric ndtperyear raises, `nspecies=0` handled |
| T5 | `test_config_writer.py` | Config roundtrip with semicolons in values |
| T7 | `test_engine_resources.py` | Resource depletion to zero biomass (no div-by-zero) |

### 4.2 Calibration abort on all-infinity (H6)

**File:** `osmose/calibration/problem.py`

Track failure count per generation. If >50% of candidates return `inf`, abort with a clear error message.

### 4.3 Remaining medium items

| ID | File | Change |
|----|------|--------|
| M4 | `engine/state.py` + `engine/processes/reproduction.py` | `append()` batch concatenation for reproduction loop. Must still call `SchoolState(...)` constructor so that `__post_init__` validation (added in 3.3) runs on the merged result. |
| M5 | `engine/processes/movement.py:145` | `list.pop(0)` → `collections.deque.popleft()` |
| M7 | `scenarios.py` | `Scenario.__post_init__` name validation (non-empty, no path separators) |
| M8 | `ui/pages/grid_helpers.py` | Surface mask loading failures to UI notification |
| M9 | `config/reader.py` | Track and return count of skipped unparseable lines |

### 4.4 Comment quality fixes from Comment Analyzer

| ID | File | Change |
|----|------|--------|
| — | `calibration/sensitivity.py:16,29` | "Saltelli sampling" → "Sobol sampling (Saltelli's extension)" |
| — | `mortality.py:57` | Update consumer list for `_tl_weighted_sum` |
| — | `growth.py:84` | Clarify weight conversion comment direction |
| — | `bioen_starvation.py:2` | Clarify "per-sub-timestep" → "internally loops over n_subdt" |
| — | `schema/species.py:65` | VB threshold description fix |
| — | `schema/species.py:188-193` | Egg weight unit clarification |
| — | `energy_budget.py:60,64` | Fix c_m and eta unit descriptions |
| — | `mortality.py:1-8` | Clarify interleaved mortality docstring |

---

## Items Explicitly Deferred

| Finding | Reason |
|---------|--------|
| H1 (engine globals: `_config_dir`, `_diet_matrix`, `_tl_weighted_sum`) | Not triggered by current code. ThreadPoolExecutor in calibration launches Java subprocesses, not Python engine. Requires engine architecture refactor. |
| M1 (EngineConfig 80+ fields → sub-configs) | Large refactor with 62 importers. Risk outweighs benefit for current stability. |
| M6 (standalone `fishing_mortality()` vectorization) | Legacy/test path only. Interleaved mortality path is already vectorized. |
| Scenario zip roundtrip tests (T6) | Medium effort, low priority. Scenarios work correctly; this is coverage completeness. |
| UI module smoke tests (T8) | Low impact. UI rendering tested manually via browser. |

---

## Success Criteria

After all 4 phases:
- All existing 1761 tests pass + ~10 new test functions pass (6 from Phase 4 + C1 guard test from Phase 2 + possible parametrized variants)
- Parity validation passes for both Bay of Biscay and EEC at year 1
- Ruff lint clean
- Zero behavioral change for correctly-configured simulations (C1, C7 only affect edge cases; C4, C5, C6 fix silent defaults)
