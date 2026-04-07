# Deep Codebase Review — osmose-python (2026-04-05)

**Scope**: Full audit of ~43K lines across 85+ production Python files  
**Agents**: 7 specialized reviewers run in parallel  
**Baseline**: Post-remediation (1766 tests passing, lint clean, parity PASS)

---

## Executive Summary

The codebase is well-structured with good separation of concerns, a clean schema-driven architecture, proper path traversal protections, and consistent naming conventions. The previous 47-item remediation addressed the most urgent issues. This review found **67 new findings** across 7 dimensions, concentrated in three themes:

1. **Thread safety**: Module-level mutable globals are unsafe under parallel calibration
2. **Silent failures**: Systematic tendency to return empty/default values instead of actionable errors
3. **Structural debt**: God objects (`EngineConfig`, `_collect_outputs`), triplicated Numba loops, and missing type invariants

| Category | Critical | High | Medium | Low | Total |
|----------|:--------:|:----:|:------:|:---:|:-----:|
| Silent Failures | 3 | 5 | 7 | — | 15 |
| Type Design | 2 | 3 | 5 | — | 10 |
| Code Quality | 3 | 6 | — | — | 9 |
| Code Simplification | 5 | 7 | 6 | — | 18 |
| Comment Quality | 3 | — | 8 | 5 | 16 |
| Test Coverage | — | 4 | 6 | — | 10 |
| Architecture | — | 8 | — | — | 8 |
| **Total** | **16** | **33** | **32** | **11** | **86** |

*Note: Many findings overlap across agents. Deduplicated unique findings: ~67.*

---

## Critical Findings (Must Fix)

### C1. Thread-unsafe module-level mutable globals

**Files**: `predation.py:45-65`, `mortality.py:59-61`, `config.py:75-81`  
**Agents**: Code Reviewer, Type Design, Architecture, Simplifier  
**Impact**: Data corruption when `ThreadPoolExecutor(n_parallel>1)` is used in calibration

Three module-level mutable globals create race conditions:
- `_diet_matrix`, `_diet_tracking_enabled` in `predation.py`
- `_tl_weighted_sum` in `mortality.py`  
- `_config_dir` in `config.py`

**Fix**: Encapsulate in a per-simulation `SimulationContext` dataclass passed through the call chain.

### C2. Division by zero in O2/temperature dose-response functions

**Files**: `oxygen_function.py:9`, `temp_function.py:27`  
**Agent**: Code Reviewer  
**Impact**: NaN/inf propagation through energy budget, corrupting biomass silently

`f_o2(o2, c1, c2)` divides by `o2 + c2` — when both are 0, produces NaN. Johnson `phi_t` has similar risk when `e_d == e_m`.

**Fix**: Add epsilon guards: `o2 + c2 + 1e-30`, `max(e_d - e_m, 1e-30)`.

### C3. Unclosed OsmoseResults in calibration hot path

**File**: `calibration/problem.py:182-186`  
**Agent**: Code Reviewer  
**Impact**: File handle leak — hundreds of unclosed NetCDF datasets during calibration

`OsmoseResults` caches opened xarray datasets. In calibration, `_evaluate_candidate` is called hundreds of times without closing.

**Fix**: `with OsmoseResults(output_dir) as results:` or `results.close()` in `finally`.

### C4. OsmoseResults returns empty DataFrames silently (non-strict mode)

**File**: `results.py:296-299`, `results.py:271-273`  
**Agent**: Silent Failure Hunter  
**Impact**: Empty charts with no explanation; calibration operates on empty data producing meaningless results

When output files are missing, returns `pd.DataFrame()` with only INFO-level logging. Calibration returns `float("inf")` per objective.

**Fix**: Default to `strict=True` for interactive use. Log at WARNING with directory path and patterns tried.

### C5. Calibration returns `[inf]` on OSMOSE failure without propagating details

**File**: `calibration/problem.py:170-177`  
**Agent**: Silent Failure Hunter  
**Impact**: Up to 50% of calibration runs can fail silently, producing meaningless results

Stderr truncated to 500 chars, logged at WARNING. User sees no indication that evaluations are failing.

**Fix**: Track failure count per generation. Surface common stderr patterns to calibration progress callback.

### C6. Stale "Phase 1 stub" docstring in engine processes `__init__.py`

**File**: `engine/processes/__init__.py:4`  
**Agent**: Comment Analyzer  
**Impact**: First thing a contributor reads — actively misleads into thinking implementations are incomplete

**Fix**: Replace with accurate description of the process architecture.

---

## High-Priority Findings (Should Fix)

### H1. EngineConfig is a 90-field god object

**File**: `config.py:522`, `from_dict()` spans ~700 lines  
**Agents**: Type Design, Simplifier, Architecture  

Split into `BioenConfig`, `FishingConfig`, `MovementConfig`, `OutputConfig`. This also structurally solves the "all bioen fields must be non-None when enabled" problem: `bioen: BioenConfig | None`.

### H2. Triplicated Numba mortality inner loop

**File**: `mortality.py:854`, `mortality.py:981`, `mortality.py:1132`  
**Agent**: Architecture  

The `if cause == 0/1/2/3` sequence is duplicated exactly 3 times (~100 lines each). Bug fixes or new mortality causes must be applied in 3 places. Comment `# SYNC: Inner loop logic duplicated` acknowledges this.

### H3. Duplicated predation logic between `predation.py` and `mortality.py`

**Files**: `predation.py` (entire), `mortality.py:238-431`  
**Agent**: Simplifier  

Both contain independent copies of accessibility-matrix lookup, resource-predation overlap, and diet-tracking. ~150 lines of duplicated logic.

**Fix**: Extract shared predation primitives into `predation_helpers.py`.

### H4. Background species injection allocates full state twice per step

**File**: `simulate.py:841-852`  
**Agent**: Architecture  

Every step: `state.append(bkg_schools)` → run mortality → `_strip_background(state, n_focal)`. Two O(n_schools) copy-all-25-arrays operations per step.

**Fix**: Keep focal and background in separate state objects.

### H5. `_collect_outputs` is 150+ lines of nested aggregation

**File**: `simulate.py:525-681`  
**Agent**: Simplifier  

Repeated species-mean pattern (`np.add.at` + safe divide) appears 4+ times.

**Fix**: Split into `_collect_biomass`, `_collect_mortality`, `_collect_yield`, `_collect_distributions`, `_collect_bioen`. Extract species-mean helper.

### H6. Path resolution duplicated 4 times

**Files**: `config.py:97`, `background.py:23`, `resources.py:167`, `movement_maps.py:20`  
**Agents**: Architecture, Type Design  

Four independent implementations of the same search-dir logic. Security fixes must be replicated to all.

**Fix**: Single `resolve_data_path()` in shared module.

### H7. `_version_tuple` returns (999,) on parse failure

**File**: `demo.py:56-61`  
**Agent**: Silent Failure Hunter  

Malformed version → `(999,)` → all migrations skipped → Java engine fails with cryptic errors.

**Fix**: Return `(0,)` (apply all migrations as safe default) and log warning.

### H8. `_do_load_results` catches AttributeError/KeyError (programming bugs)

**File**: `ui/pages/results.py:344-353`  
**Agent**: Silent Failure Hunter  

Over-broad except catches coding bugs and displays them as generic "Error loading results."

**Fix**: Narrow to `(OSError, pd.errors.ParserError)`. Let programming errors propagate.

### H9. SchoolState should be `frozen=True`

**File**: `state.py:31`  
**Agent**: Type Design  

The simulation already uses functional `replace()` pattern. Freezing prevents post-construction array corruption. Also add dtype validation for critical fields.

### H10. Missing `__post_init__` validation on 5 data types

**Files**: `StepOutput` (simulate.py:26), `MPAZone` (config.py:168), `CalibrationPhase` (multiphase.py:17), `ResourceSpeciesInfo` (resources.py:20), `BackgroundSpeciesInfo` (background.py:60)  
**Agent**: Type Design  

All have strong implicit invariants (bounds, length consistency) enforced nowhere.

### H11. Eliminate `raw_config` escape hatch from EngineConfig

**File**: `config.py:648`  
**Agent**: Type Design  

`ResourceState` and `BackgroundState` re-parse from `raw_config` instead of receiving typed data, defeating the typed config layer.

### H12. Per-school Python loop in fishing spatial maps

**File**: `fishing.py:87-103`  
**Agent**: Code Reviewer  

Pure Python per-school loop for spatial fishing map lookup. Should be vectorized with fancy indexing.

### H13. Output prefix mismatch: writer uses "osmose", reader expects "osm"

**File**: `output.py:24` vs `results.py:23`  
**Agent**: Comment Analyzer  

`write_outputs(prefix="osmose")` creates `osmose_biomass_*.csv` but `OsmoseResults(prefix="osm")` looks for `osm_biomass_*.csv`.

---

## Medium-Priority Findings

### Architecture & Simplification
- M1. Precompute species masks once per `_bioen_step()` instead of 6 times (simulate.py:275-380)
- M2. Calibration wraps Java subprocess, not the Python engine (problem.py)
- M3. `MultiPhaseCalibrator` uses scipy, `OsmoseCalibrationProblem` uses pymoo — disconnected (multiphase.py)
- M4. Mortality averaging: cumulative counts alongside mean biomass without documenting semantics (simulate.py:727)
- M5. `aggregate_replicates` returns empty dict silently for unsupported output types (ensemble.py:59)
- M6. Redundant wrapper `objective(x)` = `objective_fn(x)` in multiphase.py:91
- M7. Unused parameter `fixed_params` in `_optimize_phase` (multiphase.py:66)
- M8. 4 identical `spatial_*()` methods in results.py that all call `read_netcdf()`
- M9. `_require_columns` helper duplicated in plotting.py and analysis.py
- M10. `validate_config` and `validate_field` have duplicated type-check logic (validator.py:36-76)
- M11. Nested `np.where` in Gompertz growth — use `np.select` for readability (growth.py:133)

### Silent Failures & Error Handling
- M12. `sync_inputs` swallows TypeError — user input changes silently discarded (ui/state.py:110)
- M13. `_species_float_optional` silently uses defaults without logging (config.py:59)
- M14. `shutil.rmtree(ignore_errors=True)` in cleanup.py counts failed removals as removed
- M15. Movement Numba fallback (10-100x slower) with no user notification (movement.py:18)
- M16. Repeated `n_species = int(float(cfg.get("simulation.nspecies", "0")))` pattern in 8+ places
- M17. `read_csv` in results.py has no error handling for malformed CSVs (results.py:57)
- M18. `get_java_version` returns None for both "not installed" and "timed out" (runner.py:271)

### Comments
- M19. Config comment says "max discard rate" but code takes first nonzero (config.py:418)
- M20. 15°C "neutral for most species" is ecologically imprecise (simulate.py:262)
- M21. Inconsistent CI definitions between ensemble.py (percentiles) and analysis.py (parametric)
- M22. `demo.py` target_version default "4.3.3" but migration chain only covers to "4.3.0"
- M23. Boltzmann constant comment should note its role in Arrhenius/Johnson curves (temp_function.py:9)
- M24. `rng.py` docstring: "fixed" parameter name is confusing — expand
- M25. 3600s timeout magic number in calibration problem (problem.py:170)

---

## Test Coverage Gaps (Top 10)

| # | Gap | Criticality | File |
|---|-----|:-----------:|------|
| T1 | `_bioen_step()` orchestration untested directly | 9/10 | simulate.py:126-381 |
| T2 | Config reader error paths (circular refs, path escape, 10MB limit) | 8/10 | reader.py |
| T3 | Numerical edge cases: growth k=0, positive t0, zero-length prey | 8/10 | growth.py, mortality.py |
| T4 | 6 UI pages untested (1043 LOC) including help_modal.py (604 LOC) | 7/10 | ui/pages/ |
| T5 | `simulate()` multi-species interaction through full loop | 7/10 | simulate.py |
| T6 | Schema ENUM validation never tested | 6/10 | validator.py |
| T7 | `ui/state.py` (AppState, sync_inputs) — central UI state, no tests | 6/10 | ui/state.py |
| T8 | Ensemble aggregation edge cases (single replicate, NaN, mismatched times) | 6/10 | ensemble.py |
| T9 | Config round-trip with special chars, unicode, empty values | 5/10 | reader.py, writer.py |
| T10 | Movement map `_resolve_path()` fallback branches | 5/10 | movement_maps.py |

---

## Type Design Summary

| Type | Encap. | Invariants | Useful | Enforced | Priority |
|------|:------:|:----------:|:------:|:--------:|:--------:|
| MortalityCause | 5 | 5 | 5 | 5 | — |
| OsmoseField | 4 | 5 | 5 | 5 | — |
| Grid | 3 | 4 | 5 | 4 | Low |
| FreeParameter | 4 | 4 | 5 | 4 | Low |
| SchoolState | 2 | 3 | 5 | 3 | **High** |
| EngineConfig | 2 | 3 | 4 | 3 | **High** |
| StepOutput | 1 | 2 | 3 | 1 | **Medium** |
| MPAZone | 1 | 1 | 4 | 1 | **Medium** |
| CalibrationPhase | 1 | 1 | 3 | 1 | **Medium** |
| ResourceSpeciesInfo | 2 | 2 | 3 | 1 | **Medium** |
| BackgroundSpeciesInfo | 2 | 2 | 4 | 1 | **Medium** |

---

## Recommended Remediation Phases

### Phase 1: Safety & Correctness (C1-C5, H7, H8, M17)
- Encapsulate mutable globals in per-simulation context
- Add epsilon guards to O2/temp functions
- Close OsmoseResults in calibration
- Fix silent failure patterns (strict defaults, narrower catches)
- Fix version tuple fallback direction

### Phase 2: Structural Improvements (H1, H3, H5, H6, H9-H11)
- Split EngineConfig into sub-configs
- Extract shared predation helpers
- Split `_collect_outputs` and extract species-mean helper
- Consolidate path resolution
- Add `frozen=True` to SchoolState, add `__post_init__` to 5 types
- Remove `raw_config` escape hatch

### Phase 3: Performance & Dedup (H2, H4, H12, M1)
- Factor out common Numba mortality inner loop
- Separate background species state from focal
- Vectorize fishing spatial maps
- Precompute species masks in `_bioen_step`

### Phase 4: Tests & Polish (T1-T10, M19-M25)
- Add tests for bioen orchestration, config error paths, numerical edges
- Add tests for UI pages and AppState
- Fix stale/misleading comments
- Resolve TODOs or create tracking issues

---

*Generated by 7-agent deep review on 2026-04-05. All findings verified against current master (9431c75).*
