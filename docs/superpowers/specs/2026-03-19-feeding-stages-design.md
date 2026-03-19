# B2: Feeding Stages — Design Spec

> **Date:** 2026-03-19
> **Feature:** Per-stage predation size ratios based on age/size/weight/TL thresholds
> **Approach:** Pre-compute feeding stage per school, pass to predation kernel via 2D size ratio arrays

## Overview

Feeding stages allow predators to have multiple predation size windows that change as the organism grows. Each stage has independent min/max prey-to-predator size ratios. A predator's stage is determined by comparing a metric (age, size, weight, or trophic level) against per-species thresholds.

This is backward-compatible: species with no thresholds (the common case) have 1 stage and behave identically to the current implementation.

## Config Keys

```
predation.predprey.stage.structure = size                    # global default metric
predation.predprey.stage.structure.sp5 = age                 # per-species override (optional)
predation.predprey.stage.threshold.sp5 = 12                  # 1 threshold → 2 stages
predation.predprey.stage.threshold.sp0 = null                # no thresholds → 1 stage
predation.predprey.sizeratio.max.sp5 = 2.3;1.8              # one value per stage
predation.predprey.sizeratio.min.sp5 = 50;20                # one value per stage
```

**Discovery:** Threshold config keys use the **file index** (`sp{N}`), not the internal sequential index. Background species use their file index (e.g., `sp55`). Results are stored at sequential internal positions `[0, 1, 2...]` matching `EngineConfig` array indexing.

**`null` handling:** When `predation.predprey.stage.threshold.sp{N}` is `"null"`, absent, or empty → no thresholds → 1 stage. Matching Java's `isNull()` behavior.

**Size ratio validation:** If the parsed `sizeratio.max` value > parsed `sizeratio.min` value for the same stage, swap the stored values and warn (matching Java PredationMortality.java lines 127-136 — note: Java's warning message has a copy-paste bug naming `max` twice; Python should correctly name both keys). Validate that the number of size ratio values equals `n_stages` per species — raise `ValueError` on mismatch.

## Stage Computation Algorithm

Matches Java's `SchoolStage.getStage()` (lines 149-158):

```
stage = 0
for each threshold in thresholds[species]:
    if metric_value < threshold:   # strictly less-than
        break
    stage += 1
return stage
```

N thresholds produce N+1 stages. Boundary semantics: `T(s-1) <= value < T(s)`.

Example with thresholds `[12]` and metric = size:
- Length 8 cm → stage 0 (8 < 12)
- Length 12 cm → stage 1 (12 >= 12)
- Length 25 cm → stage 1 (25 >= 12)

## Metric Types

All four Java metrics supported, with correct unit conversions:

| Metric | Python source | Conversion | Java equivalent |
|--------|--------------|------------|-----------------|
| `"age"` | `state.age_dt` | `/ config.n_dt_per_year` (→ years) | `school.getAge()` |
| `"size"` | `state.length` | none (cm) | `school.getLength()` |
| `"weight"` | `state.weight` (per-individual, tonnes) | `* 1e6` (tonnes → grams) | `school.getWeight() * 1e6` |
| `"tl"` | `state.trophic_level` | none | `school.getTrophicLevel()` |

**CRITICAL:** The age metric uses years (float), not timesteps. The weight metric uses grams, not tonnes. Mismatching units would silently assign wrong stages.

## Data Model Changes

### `EngineConfig` changes

| Field | Before | After |
|-------|--------|-------|
| `size_ratio_min` | `(n_total,)` float64 | `(n_total, max_stages)` float64 |
| `size_ratio_max` | `(n_total,)` float64 | `(n_total, max_stages)` float64 |
| `feeding_stage_thresholds` | — | `list[NDArray[np.float64]]` per species |
| `feeding_stage_metric` | — | `list[str]` per species |
| `n_feeding_stages` | — | `NDArray[np.int32]` per species |

**Padding:** Species with fewer stages than `max_stages` get their last valid ratio repeated in trailing columns. This is safe for Numba (no NaN arithmetic issues) and prevents silent garbage values if stage computation has a bug.

**Background species:** Parsed from config using file index (e.g., `threshold.sp55`), stored at internal index `n_focal + bkg_idx`. Background species can have feeding stages — Java iterates `getPredatorIndex()` which includes both focal and background.

**Resource species:** Always stage 0 (empty thresholds). Not stored in feeding stage arrays since resources don't predate.

### `SchoolState.feeding_stage`

Already exists as `NDArray[np.int32]`, currently always 0. Populated fresh at the start of each mortality call by `compute_feeding_stages()`. NOT carried between timesteps — growth changes the metric values.

## Integration Points

### Predation Kernel

**Numba kernel `_predation_in_cell_numba`:**
- New parameter: `feeding_stage: NDArray[np.int32]`
- `size_ratio_min` and `size_ratio_max`: 1D → 2D
- Inside kernel: `r_min = size_ratio_min[sp_pred, feeding_stage[p_idx]]`

**Python fallback `_predation_in_cell_python`:**
- Same 2D lookup via `config.size_ratio_min[sp_pred, feeding_stage[p_idx]]`

**Resource predation `_predation_on_resources`:**
- Same change: use `feeding_stage[p_idx]` for the predator's size window when calculating resource prey overlap.

### Public `predation()` function

Before grouping schools by cell:
1. `feeding_stage = compute_feeding_stages(work_state, config)`
2. `work_state = work_state.replace(feeding_stage=feeding_stage)`
3. Pass `feeding_stage` array to kernel calls

### `compute_feeding_stages(state, config) -> NDArray[np.int32]`

New file: `osmose/engine/processes/feeding_stage.py`

Vectorized by metric type:
1. Group schools by their species' metric type
2. Extract metric values with correct unit conversion
3. Count thresholds exceeded per school
4. Return stage index array

The function uses `state.species_id` directly as the index into `config.feeding_stage_thresholds` and `config.feeding_stage_metric` — these arrays are indexed by internal sequential species ID, matching `SchoolState.species_id` values.

The inner threshold-counting loop is Python-speed but runs rarely (most species have 0-1 thresholds). Numba optimization deferred unless profiling shows need.

### Backward Compatibility

When all species have 1 stage:
- `max_stages = 1`
- 2D arrays are `(n_total, 1)`
- `feeding_stage` is all zeros
- Kernel does `size_ratio_min[sp_pred, feeding_stage[p_idx]]` = `size_ratio_min[sp_pred, 0]` — same scalar value as before
- No behavior change for configs without thresholds

**IMPORTANT:** The kernel indexing expression MUST be `size_ratio_min[sp_pred, feeding_stage[p_idx]]` even in the single-stage case — NOT `size_ratio_min[sp_pred]`, which on a 2D array would return a row (1D array) instead of a scalar, silently breaking the `ratio < r_min` comparison.

**Numba cache:** The signature change (1D→2D arrays + new parameter) invalidates the Numba cache. Old `.nbi`/`.nbc` files in `__pycache__/` must be cleared during development. The `cache=True` decorator will regenerate automatically on first call with the new types.

## Accessibility Matrix

**NOT part of this feature.** Java's `predation.accessibility.stage.*` keys control a separate `SchoolStage` instance for the accessibility matrix. B2 only affects predPrey size ratio lookups. The accessibility system is unchanged.

**Key distinction:** The accessibility stage system (`predation.accessibility.stage.structure`) only accepts `"age"` or `"size"` in Java (PredationMortality.java lines 89-97 — errors on other values). The predPrey stage system (`predation.predprey.stage.structure`) accepts all four metrics: `"age"`, `"size"`, `"weight"`, `"tl"`. These are independent `SchoolStage` instances.

## Java Parity Notes

### Size ratio naming inversion

Java's `predation.predPrey.sizeRatio.min` is the MAXIMUM ratio (largest ratio = smallest prey) and vice versa. The Python `config.py` already handles this — `size_ratio_min` stores the value from `sizeratio.min` key. The predation kernel check `ratio < r_min or ratio >= r_max` works correctly with this convention. No change needed for the naming.

### `getPredatorIndex()` = focal + background

Java iterates `getPredatorIndex()` (Configuration.java line 1680-1682) which concatenates `focalIndex` and `bkgIndex`. The Python equivalent iterates `range(n_species)` for focal species then uses `background_file_indices` for background species, storing at sequential internal positions.

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `osmose/engine/processes/feeding_stage.py` | Create | `compute_feeding_stages()` function |
| `osmose/engine/config.py` | Modify | 2D size ratios, threshold/metric/n_stages fields, swap validation |
| `osmose/engine/background.py` | Modify | Change `BackgroundSpeciesInfo.size_ratio_min/max` from `float` to `list[float]`; update `parse_background_species` to parse semicolon-separated multi-stage values |
| `osmose/engine/processes/predation.py` | Modify | 2D array indexing in Numba + Python kernels, pass feeding_stage |
| `tests/test_engine_feeding_stages.py` | Create | Full test suite |

## Testing Strategy (TDD)

1. **Config parsing:** Single-stage species (no thresholds) produces 1-element ratio arrays
2. **Config parsing:** Multi-stage species (e.g., threshold=12, 2 ratio values) produces correct 2D array
3. **Config parsing:** `"null"` string threshold treated as no thresholds (1 stage)
3b. **Config parsing:** Absent threshold key treated as no thresholds (1 stage)
4. **Config parsing:** Per-species metric override works
5. **Config parsing:** Size ratio swap validation (max > min → swap + warn)
6. **Config parsing:** Array length mismatch (2 stages but 1 ratio value) → ValueError
7. **Config parsing:** Background species get feeding stages via file index
8. **Stage computation:** Age metric converts timesteps to years
9. **Stage computation:** Size metric uses length directly
10. **Stage computation:** Weight metric applies `* 1e6` conversion
11. **Stage computation:** TL metric uses trophic level directly
12. **Stage computation:** Multiple thresholds produce correct stage assignments
13. **Stage computation:** Schools at exact threshold value → next stage (strictly less-than)
14. **Predation kernel:** Numba path uses correct stage-indexed ratio
15. **Predation kernel:** Python fallback uses correct stage-indexed ratio
16. **Predation kernel:** Resource predation uses stage-indexed ratio
17. **Backward compatibility:** Config with single-value ratios (no thresholds) behaves identically
18. **Integration:** Multi-stage predation produces different prey windows for juvenile vs adult predators
