# Feeding Stages (B2) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-stage predation size ratios to the Python OSMOSE engine, allowing juvenile and adult predators to have different prey size windows based on age, size, weight, or trophic level thresholds.

**Architecture:** Pre-compute feeding stage per school before predation, store in `SchoolState.feeding_stage`. Size ratio arrays become 2D `(n_total, max_stages)`. The Numba kernel gets one extra parameter (`feeding_stage`) and changes its size ratio indexing from `arr[sp]` to `arr[sp, stage]`. Backward-compatible: single-stage configs produce `(n_total, 1)` arrays with all zeros for `feeding_stage`.

**Tech Stack:** Python 3.12, NumPy, Numba, pytest

**Spec:** `docs/superpowers/specs/2026-03-19-feeding-stages-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `osmose/engine/processes/feeding_stage.py` | Create | `compute_feeding_stages()` function |
| `osmose/engine/config.py` | Modify | Parse multi-value size ratios into 2D arrays; add threshold/metric/n_stages fields; swap validation |
| `osmose/engine/background.py` | Modify | Change `BackgroundSpeciesInfo.size_ratio_min/max` from `float` to `list[float]`; parse multi-stage values |
| `osmose/engine/processes/predation.py` | Modify | 2D array indexing in Numba + Python kernels + resource predation; pass `feeding_stage`; compute stages before cell loop |
| `tests/test_engine_feeding_stages.py` | Create | Full test suite |

---

## Task 1: `compute_feeding_stages` + config parsing for thresholds/metrics

**Files:**
- Create: `osmose/engine/processes/feeding_stage.py`
- Modify: `osmose/engine/config.py`
- Create: `tests/test_engine_feeding_stages.py`

This task adds the threshold/metric parsing to `EngineConfig` and the `compute_feeding_stages()` function, but does NOT yet change size ratio arrays to 2D. The existing 1D size ratio arrays stay unchanged — Task 2 handles the 2D migration.

- [ ] **Step 1: Write failing tests for config parsing and stage computation**

```python
# tests/test_engine_feeding_stages.py
"""Tests for feeding stages (B2) — per-stage predation size ratios."""

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.processes.feeding_stage import compute_feeding_stages
from osmose.engine.state import SchoolState


def _make_base_config() -> dict[str, str]:
    """Minimal focal config (1 species) for feeding stage tests."""
    return {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.type.sp0": "focal",
        "species.name.sp0": "Anchovy",
        "species.linf.sp0": "20.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "predation.predprey.sizeratio.min.sp0": "50",
        "predation.predprey.sizeratio.max.sp0": "3",
    }


def _make_2stage_config() -> dict[str, str]:
    """Config with 2 species: sp0 has 1 stage, sp1 has 2 stages (size threshold=12)."""
    cfg = {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "2",
        "simulation.nschool.sp0": "5",
        "simulation.nschool.sp1": "5",
        "species.type.sp0": "focal",
        "species.type.sp1": "focal",
        "species.name.sp0": "Anchovy",
        "species.name.sp1": "Hake",
        "species.linf.sp0": "20.0",
        "species.linf.sp1": "80.0",
        "species.k.sp0": "0.3",
        "species.k.sp1": "0.15",
        "species.t0.sp0": "-0.1",
        "species.t0.sp1": "-0.2",
        "species.egg.size.sp0": "0.1",
        "species.egg.size.sp1": "0.2",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.condition.factor.sp1": "0.008",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.length2weight.allometric.power.sp1": "3.0",
        "species.lifespan.sp0": "3",
        "species.lifespan.sp1": "10",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "species.vonbertalanffy.threshold.age.sp1": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.ingestion.rate.max.sp1": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "predation.efficiency.critical.sp1": "0.57",
        # sp0: single stage (no threshold)
        "predation.predprey.sizeratio.min.sp0": "50",
        "predation.predprey.sizeratio.max.sp0": "3",
        "predation.predprey.stage.threshold.sp0": "null",
        # sp1: 2 stages (threshold at 12 cm)
        "predation.predprey.sizeratio.min.sp1": "50;20",
        "predation.predprey.sizeratio.max.sp1": "2.3;1.8",
        "predation.predprey.stage.threshold.sp1": "12",
        # Global structure
        "predation.predprey.stage.structure": "size",
    }
    return cfg


class TestConfigParsing:
    def test_single_stage_no_threshold(self):
        cfg = _make_base_config()
        ec = EngineConfig.from_dict(cfg)
        assert ec.n_feeding_stages[0] == 1
        assert len(ec.feeding_stage_thresholds[0]) == 0

    def test_multi_stage_threshold(self):
        cfg = _make_2stage_config()
        ec = EngineConfig.from_dict(cfg)
        assert ec.n_feeding_stages[0] == 1  # sp0: null → 1 stage
        assert ec.n_feeding_stages[1] == 2  # sp1: threshold 12 → 2 stages
        np.testing.assert_array_equal(ec.feeding_stage_thresholds[1], [12.0])

    def test_null_string_threshold(self):
        cfg = _make_base_config()
        cfg["predation.predprey.stage.threshold.sp0"] = "null"
        ec = EngineConfig.from_dict(cfg)
        assert ec.n_feeding_stages[0] == 1

    def test_absent_threshold_key(self):
        cfg = _make_base_config()
        # No predation.predprey.stage.threshold.sp0 key at all
        ec = EngineConfig.from_dict(cfg)
        assert ec.n_feeding_stages[0] == 1

    def test_global_metric_default(self):
        cfg = _make_2stage_config()
        ec = EngineConfig.from_dict(cfg)
        assert ec.feeding_stage_metric[0] == "size"
        assert ec.feeding_stage_metric[1] == "size"

    def test_per_species_metric_override(self):
        cfg = _make_2stage_config()
        cfg["predation.predprey.stage.structure.sp1"] = "age"
        ec = EngineConfig.from_dict(cfg)
        assert ec.feeding_stage_metric[0] == "size"  # global default
        assert ec.feeding_stage_metric[1] == "age"  # per-species override

    def test_absent_global_structure_defaults_to_size(self):
        cfg = _make_base_config()
        # No predation.predprey.stage.structure key
        ec = EngineConfig.from_dict(cfg)
        assert ec.feeding_stage_metric[0] == "size"

    def test_size_ratio_2d_single_stage(self):
        cfg = _make_base_config()
        ec = EngineConfig.from_dict(cfg)
        assert ec.size_ratio_min.ndim == 2
        assert ec.size_ratio_min.shape == (1, 1)
        np.testing.assert_allclose(ec.size_ratio_min[0, 0], 50.0)
        np.testing.assert_allclose(ec.size_ratio_max[0, 0], 3.0)

    def test_size_ratio_2d_multi_stage(self):
        cfg = _make_2stage_config()
        ec = EngineConfig.from_dict(cfg)
        # sp0: 1 stage → padded to max_stages=2
        # sp1: 2 stages
        assert ec.size_ratio_min.shape == (2, 2)
        np.testing.assert_allclose(ec.size_ratio_min[0, 0], 50.0)
        np.testing.assert_allclose(ec.size_ratio_min[0, 1], 50.0)  # padded with last value
        np.testing.assert_allclose(ec.size_ratio_min[1, 0], 50.0)  # stage 0
        np.testing.assert_allclose(ec.size_ratio_min[1, 1], 20.0)  # stage 1

    def test_size_ratio_swap_validation(self):
        cfg = _make_base_config()
        # Set max > min (inverted) — should swap
        cfg["predation.predprey.sizeratio.min.sp0"] = "2"
        cfg["predation.predprey.sizeratio.max.sp0"] = "50"
        ec = EngineConfig.from_dict(cfg)
        # After swap: min should be 50, max should be 2
        np.testing.assert_allclose(ec.size_ratio_min[0, 0], 50.0)
        np.testing.assert_allclose(ec.size_ratio_max[0, 0], 2.0)

    def test_stage_count_mismatch_raises(self):
        cfg = _make_2stage_config()
        # sp1 has 2 stages but give only 1 ratio value
        cfg["predation.predprey.sizeratio.min.sp1"] = "50"
        with pytest.raises(ValueError, match="mismatch"):
            EngineConfig.from_dict(cfg)

    def test_trailing_semicolon_handled(self):
        cfg = _make_2stage_config()
        cfg["predation.predprey.sizeratio.min.sp1"] = "50;20;"  # trailing semicolon
        cfg["predation.predprey.sizeratio.max.sp1"] = "2.3;1.8;"
        ec = EngineConfig.from_dict(cfg)
        assert ec.n_feeding_stages[1] == 2
        np.testing.assert_allclose(ec.size_ratio_min[1, 0], 50.0)
        np.testing.assert_allclose(ec.size_ratio_min[1, 1], 20.0)

    def test_unrecognized_metric_raises(self):
        cfg = _make_base_config()
        cfg["predation.predprey.stage.structure"] = "invalid"
        with pytest.raises(ValueError, match="metric"):
            EngineConfig.from_dict(cfg)


class TestComputeFeedingStages:
    def test_single_stage_all_zeros(self):
        cfg = _make_base_config()
        ec = EngineConfig.from_dict(cfg)
        state = SchoolState.create(n_schools=3, species_id=np.zeros(3, dtype=np.int32))
        state = state.replace(length=np.array([5.0, 10.0, 15.0]))
        stages = compute_feeding_stages(state, ec)
        np.testing.assert_array_equal(stages, [0, 0, 0])

    def test_size_metric_two_stages(self):
        cfg = _make_2stage_config()
        ec = EngineConfig.from_dict(cfg)
        # 3 schools of species 1 (Hake) with size threshold=12
        state = SchoolState.create(n_schools=3, species_id=np.ones(3, dtype=np.int32))
        state = state.replace(length=np.array([8.0, 12.0, 25.0]))
        stages = compute_feeding_stages(state, ec)
        np.testing.assert_array_equal(stages, [0, 1, 1])  # 8<12→0, 12>=12→1, 25>=12→1

    def test_age_metric_converts_to_years(self):
        cfg = _make_2stage_config()
        cfg["predation.predprey.stage.structure.sp1"] = "age"
        cfg["predation.predprey.stage.threshold.sp1"] = "2"  # 2 years
        ec = EngineConfig.from_dict(cfg)
        state = SchoolState.create(n_schools=3, species_id=np.ones(3, dtype=np.int32))
        # age_dt: 24 = 1 year, 48 = 2 years, 72 = 3 years (n_dt_per_year=24)
        state = state.replace(age_dt=np.array([24, 48, 72], dtype=np.int32))
        stages = compute_feeding_stages(state, ec)
        np.testing.assert_array_equal(stages, [0, 1, 1])  # 1<2→0, 2>=2→1, 3>=2→1

    def test_weight_metric_converts_to_grams(self):
        cfg = _make_2stage_config()
        cfg["predation.predprey.stage.structure.sp1"] = "weight"
        cfg["predation.predprey.stage.threshold.sp1"] = "1000"  # 1000 grams = 0.001 tonnes
        ec = EngineConfig.from_dict(cfg)
        state = SchoolState.create(n_schools=2, species_id=np.ones(2, dtype=np.int32))
        # weight in tonnes: 0.0005 = 500g, 0.002 = 2000g
        state = state.replace(weight=np.array([0.0005, 0.002]))
        stages = compute_feeding_stages(state, ec)
        np.testing.assert_array_equal(stages, [0, 1])  # 500<1000→0, 2000>=1000→1

    def test_tl_metric(self):
        cfg = _make_2stage_config()
        cfg["predation.predprey.stage.structure.sp1"] = "tl"
        cfg["predation.predprey.stage.threshold.sp1"] = "3.0"
        ec = EngineConfig.from_dict(cfg)
        state = SchoolState.create(n_schools=2, species_id=np.ones(2, dtype=np.int32))
        state = state.replace(trophic_level=np.array([2.5, 3.5]))
        stages = compute_feeding_stages(state, ec)
        np.testing.assert_array_equal(stages, [0, 1])

    def test_multiple_thresholds(self):
        cfg = _make_2stage_config()
        cfg["predation.predprey.stage.threshold.sp1"] = "10;20"  # 3 stages
        cfg["predation.predprey.sizeratio.min.sp1"] = "50;30;20"
        cfg["predation.predprey.sizeratio.max.sp1"] = "3;2;1.5"
        ec = EngineConfig.from_dict(cfg)
        state = SchoolState.create(n_schools=4, species_id=np.ones(4, dtype=np.int32))
        state = state.replace(length=np.array([5.0, 10.0, 15.0, 25.0]))
        stages = compute_feeding_stages(state, ec)
        np.testing.assert_array_equal(stages, [0, 1, 1, 2])  # 5<10→0, 10>=10→1, 15>=10 & <20→1, 25>=20→2

    def test_exact_threshold_value_goes_to_next_stage(self):
        cfg = _make_2stage_config()
        ec = EngineConfig.from_dict(cfg)
        state = SchoolState.create(n_schools=1, species_id=np.ones(1, dtype=np.int32))
        state = state.replace(length=np.array([12.0]))  # exactly at threshold
        stages = compute_feeding_stages(state, ec)
        assert stages[0] == 1  # 12 >= 12 → stage 1

    def test_mixed_species(self):
        cfg = _make_2stage_config()
        ec = EngineConfig.from_dict(cfg)
        # sp0 (1 stage) and sp1 (2 stages, threshold=12)
        state = SchoolState.create(
            n_schools=4, species_id=np.array([0, 0, 1, 1], dtype=np.int32)
        )
        state = state.replace(length=np.array([5.0, 15.0, 5.0, 15.0]))
        stages = compute_feeding_stages(state, ec)
        np.testing.assert_array_equal(stages, [0, 0, 0, 1])  # sp0 always 0; sp1: 5<12→0, 15>=12→1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_feeding_stages.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Create `feeding_stage.py`**

```python
# osmose/engine/processes/feeding_stage.py
"""Feeding stage computation for the OSMOSE Python engine.

Determines each school's feeding stage based on a species-specific metric
(age, size, weight, or trophic level) and threshold values. The stage
selects which predation size ratios apply.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_feeding_stages(state, config) -> NDArray[np.int32]:
    """Compute feeding stage index for each school.

    Uses the species' configured metric (age/size/weight/tl) and thresholds.
    Stage = number of thresholds the metric value meets or exceeds.
    Boundary: value < threshold → stop (strictly less-than, matching Java).

    Args:
        state: SchoolState with species_id, age_dt, length, weight, trophic_level
        config: EngineConfig with feeding_stage_thresholds, feeding_stage_metric,
                n_dt_per_year
    Returns:
        1D int32 array of stage indices, same length as state
    """
    n = len(state)
    if n == 0:
        return np.zeros(0, dtype=np.int32)

    stages = np.zeros(n, dtype=np.int32)
    sp = state.species_id

    # Process by metric type for efficiency
    for metric_type in ("age", "size", "weight", "tl"):
        # Find species using this metric
        species_uses = np.array(
            [m == metric_type for m in config.feeding_stage_metric], dtype=np.bool_
        )
        if not species_uses.any():
            continue

        # Find schools whose species uses this metric
        school_mask = species_uses[sp]
        if not school_mask.any():
            continue

        # Extract metric values with correct unit conversion
        if metric_type == "age":
            values = state.age_dt[school_mask].astype(np.float64) / config.n_dt_per_year
        elif metric_type == "size":
            values = state.length[school_mask]
        elif metric_type == "weight":
            values = state.weight[school_mask] * 1e6  # tonnes → grams
        else:  # tl
            values = state.trophic_level[school_mask]

        # Count thresholds exceeded per school
        masked_indices = np.nonzero(school_mask)[0]
        masked_sp = sp[school_mask]
        for i in range(len(values)):
            thresholds = config.feeding_stage_thresholds[masked_sp[i]]
            stage = 0
            for t in thresholds:
                if values[i] < t:
                    break
                stage += 1
            stages[masked_indices[i]] = stage

    return stages
```

- [ ] **Step 4: Modify `config.py` — add threshold/metric fields + 2D size ratios**

Changes to `osmose/engine/config.py`:

**Add new fields to `EngineConfig` dataclass:**
```python
    # Feeding stages
    feeding_stage_thresholds: list  # list[NDArray[np.float64]] per species
    feeding_stage_metric: list  # list[str] per species
    n_feeding_stages: NDArray[np.int32]
```

**Add a multi-value parser** (reuse pattern from `background.py`):
```python
def _parse_multi_float(value: str) -> list[float]:
    """Parse semicolon/comma-separated float values, filtering empty tokens."""
    import re
    parts = re.split(r"[;,]\s*", value.strip())
    return [float(p) for p in parts if p.strip()]
```

**In `from_dict`, replace the scalar size ratio parsing** with multi-value parsing:

1. Read the global structure key: `cfg.get("predation.predprey.stage.structure", "size")`
2. Validate metric is one of `"age"`, `"size"`, `"weight"`, `"tl"` (case-insensitive)
3. For each focal species `i in range(n_sp)`:
   - Read per-species structure override: `cfg.get(f"predation.predprey.stage.structure.sp{i}")`
   - Read threshold: `cfg.get(f"predation.predprey.stage.threshold.sp{i}", "")`
   - Handle "null"/absent/empty → empty threshold array
   - Read multi-value ratios via `_parse_multi_float`
   - Validate: `len(ratios) == n_stages`, else raise `ValueError`
   - Swap validation: if `max_val > min_val` for any stage, swap + warn
4. For background species: same logic using file indices from `background_file_indices`
5. Build 2D arrays: pad species with fewer stages by repeating their last valid value
6. Store: `size_ratio_min` as `(n_total, max_stages)`, `size_ratio_max` as `(n_total, max_stages)`

**CRITICAL:** The `size_ratio_min` and `size_ratio_max` fields change from 1D to 2D. This breaks the existing background extension code (lines 282-283) which concatenates 1D arrays. The extension code must be updated to work with 2D arrays.

**Also update `BackgroundSpeciesInfo`** in `background.py`:
- Change `size_ratio_min: float` → `size_ratio_min: list[float]`
- Change `size_ratio_max: float` → `size_ratio_max: list[float]`
- In `parse_background_species`, use `_parse_multi_float` for the ratio values

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_feeding_stages.py -v`
Expected: ALL PASS

- [ ] **Step 6: Run full test suite (config + stage tests only — predation tests deferred to Task 2)**

Run: `.venv/bin/python -m pytest tests/test_engine_feeding_stages.py -v`
Expected: ALL config parsing + stage computation tests PASS

**NOTE:** Do NOT run the full test suite yet — `config.size_ratio_min` is now 2D but the predation kernel still expects 1D. Task 2 updates the kernel atomically and runs the full suite.

- [ ] **Step 7: Commit config + feeding_stage.py changes (no predation.py yet)**

```bash
git add osmose/engine/processes/feeding_stage.py osmose/engine/config.py osmose/engine/background.py tests/test_engine_feeding_stages.py
git commit -m "feat(engine): add feeding stage config parsing and compute_feeding_stages()

NOTE: size_ratio arrays are now 2D — predation kernel update in next commit"
```

---

## Task 2: Update predation kernel for 2D size ratios

**Files:**
- Modify: `osmose/engine/processes/predation.py`
- Modify: `tests/test_engine_feeding_stages.py`

- [ ] **Step 1: Write failing tests for predation with stages**

Add to `tests/test_engine_feeding_stages.py`:

```python
from osmose.engine.processes.predation import predation


class TestPredationWithStages:
    def test_juvenile_uses_stage0_ratios(self):
        """Juvenile Hake (length=8, stage 0) uses sizeratio.min=50, max=2.3."""
        cfg = _make_2stage_config()
        ec = EngineConfig.from_dict(cfg)
        # Hake predator (sp1, length=8 < threshold 12 → stage 0)
        # Prey (sp0, length=2.5)
        # ratio = 8/2.5 = 3.2 → check: 3.2 < 50 (r_min) → True? No wait...
        # r_min = size_ratio_min[1,0] = 50, r_max = size_ratio_max[1,0] = 2.3
        # Check: ratio < r_min → 3.2 < 50 is True, so skip... that's wrong
        # Actually: the check is ratio < r_max or ratio >= r_min
        # r_max=2.3, r_min=50: 3.2 < 2.3 → False, 3.2 >= 50 → False → ACCEPTED
        # Wait, the kernel check is: if ratio < r_min or ratio >= r_max: continue
        # With r_min=50, r_max=2.3: 3.2 < 50 → True → SKIP. That's wrong.
        # The Java naming is inverted. In the Python kernel:
        #   r_min = config.size_ratio_min[sp] which stores sizeratio.min value (the LARGER number)
        #   r_max = config.size_ratio_max[sp] which stores sizeratio.max value (the SMALLER number)
        #   Check: ratio < r_min (50) → always skip big ratios?
        # Let me re-read predation.py line 89: if ratio < r_min or ratio >= r_max: continue
        # With r_min=50 from sizeratio.min, r_max=3 from sizeratio.max:
        #   ratio=6 → 6 < 50 is True → SKIP
        # This means the CURRENT predation code has r_min and r_max inverted from what the
        # comparison expects. Looking at the Java:
        #   preySizeMax = predLen / predPreySizesMax[iPred][iStage]  (smaller number → bigger prey window)
        #   preySizeMin = predLen / predPreySizesMin[iPred][iStage]  (bigger number → smaller prey window)
        #   prey.length >= preySizeMin && prey.length < preySizeMax
        # Java: prey is eligible if prey.length in [predLen/min, predLen/max)
        # With min=50, max=3: prey in [predLen/50, predLen/3)
        # For predLen=30: prey in [0.6, 10) → prey of length 5 is eligible (ratio=6, in [0.6,10))
        #
        # Python kernel: ratio = predLen/preyLen, check: ratio < r_min or ratio >= r_max → skip
        # With r_min stored from sizeratio.min=50, r_max stored from sizeratio.max=3:
        #   ratio=6 → 6 < 50 → True → SKIP. This is WRONG.
        #
        # Hmm, but the existing predation tests pass with the current code...
        # Let me check: in _make_base_config, sizeratio.min.sp0=50, sizeratio.max.sp0=3
        # But in EngineConfig parsing: size_ratio_min is read from sizeratio.min, size_ratio_max from sizeratio.max
        # The kernel uses r_min = size_ratio_min[sp] = 50, r_max = size_ratio_max[sp] = 3
        # Check: ratio < 50 → True for all reasonable ratios → always skip!
        #
        # The existing tests must be using different values. Let me use values that work
        # with the actual kernel logic.
        #
        # Actually reading config.py more carefully:
        # _species_float_optional(cfg, "predation.predprey.sizeratio.min.sp{i}", ..., default=1.0)
        # So size_ratio_min = 1.0 by default, size_ratio_max = 3.5 by default
        # Kernel: ratio < 1.0 or ratio >= 3.5 → skip
        # So ratio must be in [1.0, 3.5) to be eligible
        # This matches Java: prey in [predLen/3.5, predLen/1.0)
        #
        # The test configs in existing predation tests use defaults (1.0 and 3.5)
        # The eec_full config has sizeratio.min=50 which is the Java convention
        # but config.py reads it into size_ratio_min which the kernel checks as < r_min
        #
        # CONCLUSION: The kernel check `ratio < r_min or ratio >= r_max` works correctly
        # when r_min < r_max (e.g., 1.0 and 3.5). The Java convention where sizeratio.min=50
        # would make r_min=50 > r_max=3, which breaks the check.
        # BUT: the spec says there's a swap validation that swaps when max > min.
        # After swap: r_min=3, r_max=50. Check: ratio < 3 or ratio >= 50 → skip.
        # For ratio=6: 6 < 3 → False, 6 >= 50 → False → ACCEPTED. Correct!

        # Use explicit values that work with the kernel after swap validation
        cfg["predation.predprey.sizeratio.min.sp1"] = "100;50"   # after swap: these become r_max
        cfg["predation.predprey.sizeratio.max.sp1"] = "1;2"      # after swap: these become r_min
        ec = EngineConfig.from_dict(cfg)

        # Juvenile Hake (sp1, length=30, stage 0 since 30>=12... wait that's stage 1)
        # Let's use length=8 for juvenile (stage 0), prey at length=2
        # Stage 0 ratios after swap: r_min=1, r_max=100. ratio=8/2=4 → in [1,100) → ACCEPTED
        state = SchoolState.create(n_schools=2, species_id=np.array([1, 0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0, 1000.0]),
            weight=np.array([10.0, 1.0]),
            biomass=np.array([1000.0, 1000.0]),
            length=np.array([8.0, 2.0]),
            age_dt=np.array([10, 10], dtype=np.int32),
            first_feeding_age_dt=np.array([0, 0], dtype=np.int32),
            cell_x=np.array([0, 0], dtype=np.int32),
            cell_y=np.array([0, 0], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        new_state = predation(state, ec, rng, n_subdt=10, grid_ny=1, grid_nx=1)
        # Prey should be eaten
        assert new_state.abundance[1] < 1000.0

    def test_adult_uses_stage1_ratios(self):
        """Adult Hake (length=30, stage 1) uses different ratios than juvenile."""
        cfg = _make_2stage_config()
        # Stage 0: wide window [1, 100), Stage 1: narrow window [5, 10)
        cfg["predation.predprey.sizeratio.min.sp1"] = "100;10"
        cfg["predation.predprey.sizeratio.max.sp1"] = "1;5"
        ec = EngineConfig.from_dict(cfg)

        # Adult Hake (length=30 >= threshold 12 → stage 1)
        # Prey length=2, ratio=30/2=15 → stage 1 window [5, 10) → 15 >= 10 → SKIP
        state = SchoolState.create(n_schools=2, species_id=np.array([1, 0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0, 1000.0]),
            weight=np.array([10.0, 1.0]),
            biomass=np.array([1000.0, 1000.0]),
            length=np.array([30.0, 2.0]),
            age_dt=np.array([10, 10], dtype=np.int32),
            first_feeding_age_dt=np.array([0, 0], dtype=np.int32),
            cell_x=np.array([0, 0], dtype=np.int32),
            cell_y=np.array([0, 0], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        new_state = predation(state, ec, rng, n_subdt=10, grid_ny=1, grid_nx=1)
        # Prey should NOT be eaten (ratio 15 outside [5, 10))
        np.testing.assert_allclose(new_state.abundance[1], 1000.0)

    def test_backward_compat_single_stage(self):
        """Config with single-value ratios behaves identically to pre-stages code."""
        cfg = _make_base_config()
        ec = EngineConfig.from_dict(cfg)
        state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0, 1000.0]),
            weight=np.array([10.0, 1.0]),
            biomass=np.array([1000.0, 1000.0]),
            length=np.array([15.0, 7.0]),
            age_dt=np.array([10, 10], dtype=np.int32),
            first_feeding_age_dt=np.array([0, 0], dtype=np.int32),
            cell_x=np.array([0, 0], dtype=np.int32),
            cell_y=np.array([0, 0], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        new_state = predation(state, ec, rng, n_subdt=10, grid_ny=1, grid_nx=1)
        # ratio=15/7≈2.14. Default: r_min=1.0, r_max=3.5 → 2.14 in [1.0, 3.5) → ACCEPTED
        assert new_state.abundance[1] < 1000.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_feeding_stages.py::TestPredationWithStages -v`
Expected: FAIL — kernel expects 1D arrays, gets 2D

- [ ] **Step 3: Update predation kernel**

**Numba kernel `_predation_in_cell_numba`:**
1. Change `size_ratio_min` and `size_ratio_max` parameter types from 1D to 2D
2. Add `feeding_stage: NDArray[np.int32]` parameter
3. Change lines 66-67:
```python
# Before:
r_min = size_ratio_min[sp_pred]
r_max = size_ratio_max[sp_pred]
# After:
stage = feeding_stage[p_idx]
r_min = size_ratio_min[sp_pred, stage]
r_max = size_ratio_max[sp_pred, stage]
```

**Python fallback `_predation_in_cell_python`:**
Same change — use `state.feeding_stage[p_idx]` for 2D indexing:
```python
r_min = config.size_ratio_min[sp_pred, state.feeding_stage[p_idx]]
r_max = config.size_ratio_max[sp_pred, state.feeding_stage[p_idx]]
```

**Resource predation `_predation_on_resources`:**
Change lines 254-255:
```python
# Before:
r_min_val = config.size_ratio_min[sp_pred]
r_max_val = config.size_ratio_max[sp_pred]
# After:
stage = state.feeding_stage[p_idx]
r_min_val = config.size_ratio_min[sp_pred, stage]
r_max_val = config.size_ratio_max[sp_pred, stage]
```

**Public `predation()` function:**
After creating `work_state`, compute and store feeding stages:
```python
from osmose.engine.processes.feeding_stage import compute_feeding_stages

feeding_stage = compute_feeding_stages(work_state, config)
work_state = work_state.replace(feeding_stage=feeding_stage)
```

Update the Numba kernel call to pass `feeding_stage`:
```python
_predation_in_cell_numba(
    ...,
    config.size_ratio_min,   # now 2D
    config.size_ratio_max,   # now 2D
    ...,
    work_state.feeding_stage,  # NEW parameter
)
```

**Clear Numba cache** before testing:
```bash
find . -name "*.nbi" -o -name "*.nbc" | xargs rm -f
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_feeding_stages.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: ALL PASS including existing predation tests (backward compatible)

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/processes/predation.py tests/test_engine_feeding_stages.py
git commit -m "feat(engine): update predation kernel for 2D size ratios with feeding stages"
```

---

## Task 3: Background species + integration test

**Files:**
- Modify: `tests/test_engine_feeding_stages.py`

- [ ] **Step 1: Write tests for background species stages and integration**

```python
class TestBackgroundSpeciesStages:
    def test_background_species_get_thresholds(self):
        """Background species can have feeding stage thresholds."""
        cfg = _make_base_config()
        cfg.update({
            "species.type.sp10": "background",
            "species.name.sp10": "BkgSpecies",
            "species.nclass.sp10": "1",
            "species.length.sp10": "15",
            "species.size.proportion.sp10": "1.0",
            "species.trophic.level.sp10": "2",
            "species.age.sp10": "1",
            "species.length2weight.condition.factor.sp10": "0.006",
            "species.length2weight.allometric.power.sp10": "3.0",
            "predation.predprey.sizeratio.max.sp10": "2;1.5",
            "predation.predprey.sizeratio.min.sp10": "10;5",
            "predation.ingestion.rate.max.sp10": "3.5",
            "species.biomass.total.sp10": "1000",
            "simulation.nbackground": "1",
            "predation.predprey.stage.structure": "size",
            "predation.predprey.stage.threshold.sp10": "10",
        })
        ec = EngineConfig.from_dict(cfg)
        # Background is at internal index 1 (n_focal=1 + bkg_idx=0)
        assert ec.n_feeding_stages[1] == 2
        np.testing.assert_array_equal(ec.feeding_stage_thresholds[1], [10.0])


class TestFeedingStagesIntegration:
    def test_multi_stage_changes_prey_window(self):
        """Juvenile and adult predators eat different prey due to stage-specific ratios."""
        cfg = _make_2stage_config()
        # Stage 0 (juvenile, <12cm): wide window r_min=1, r_max=100
        # Stage 1 (adult, >=12cm): narrow window r_min=3, r_max=8
        cfg["predation.predprey.sizeratio.min.sp1"] = "100;8"
        cfg["predation.predprey.sizeratio.max.sp1"] = "1;3"
        cfg["simulation.time.nyear"] = "1"
        cfg["population.seeding.biomass.sp0"] = "100.0"
        cfg["population.seeding.biomass.sp1"] = "100.0"
        cfg["species.sexratio.sp0"] = "0.5"
        cfg["species.sexratio.sp1"] = "0.5"
        cfg["species.relativefecundity.sp0"] = "500"
        cfg["species.relativefecundity.sp1"] = "500"
        cfg["species.maturity.size.sp0"] = "10.0"
        cfg["species.maturity.size.sp1"] = "10.0"

        from osmose.engine.grid import Grid
        from osmose.engine.simulate import simulate

        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(ec, grid, rng)
        # Should complete without errors
        assert len(outputs) == ec.n_steps
        # Both species should have non-negative biomass
        for o in outputs:
            assert np.all(o.biomass >= 0)
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_feeding_stages.py -v`
Expected: ALL PASS

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: ALL PASS, zero regressions

- [ ] **Step 4: Commit**

```bash
git add tests/test_engine_feeding_stages.py
git commit -m "test(engine): add background species stage + integration tests for feeding stages"
```

---

## Task Summary

| Task | What | Tests | Files |
|------|------|-------|-------|
| 1 | Config parsing + `compute_feeding_stages()` + 2D size ratios in EngineConfig + background.py update | 22 tests | `feeding_stage.py`, `config.py`, `background.py`, `test_engine_feeding_stages.py` |
| 2 | Predation kernel update (Numba + Python + resource) | 3 tests | `predation.py`, `test_engine_feeding_stages.py` |
| 3 | Background species stages + integration | 2 tests | `test_engine_feeding_stages.py` |
| **Total** | | **27 tests** | |

### Spec Test Area Coverage

| Spec Area | Task |
|-----------|------|
| 1. Single-stage (no thresholds) | Task 1 |
| 2. Multi-stage (threshold, 2 ratio values) | Task 1 |
| 3. `"null"` string threshold | Task 1 |
| 3b. Absent threshold key | Task 1 |
| 4. Per-species metric override | Task 1 |
| 5. Size ratio swap validation | Task 1 |
| 6. Array length mismatch → ValueError | Task 1 |
| 7. Background species stages via file index | Task 3 |
| 8. Age metric (timesteps → years) | Task 1 |
| 9. Size metric (length) | Task 1 |
| 10. Weight metric (* 1e6) | Task 1 |
| 11. TL metric | Task 1 |
| 12. Multiple thresholds | Task 1 |
| 13. Exact threshold → next stage | Task 1 |
| 14. Numba path uses stage-indexed ratio | Task 2 |
| 15. Python fallback uses stage-indexed ratio | Task 2 |
| 16. Resource predation uses stage-indexed ratio | Task 2 (tested via integration) |
| 17. Backward compatibility | Task 2 |
| 18. Integration (different prey windows) | Task 3 |

---

## IMPORTANT: Review Corrections (from plan review)

### Correction 1: Size ratio test values in Task 2

The `TestPredationWithStages` tests in Task 2 Step 1 have a long reasoning block about size ratio naming conventions. The conclusions in the comments contain errors. **Implementers must use the following corrected approach:**

**Python kernel convention:** The kernel check `ratio < r_min or ratio >= r_max` expects `r_min < r_max`. The accepted range is `[r_min, r_max)`.

**Swap validation:** The new config parsing swaps values if `sizeratio.max value > sizeratio.min value` for any stage. This converts Java-convention configs (e.g., `sizeratio.min=50, sizeratio.max=3`) to Python convention (`r_min=3, r_max=50`).

**Correct test values:** Use Python convention directly in tests (r_min < r_max) to avoid confusion:
```python
# Stage 0: wide window [1.0, 10.0), Stage 1: narrow window [2.0, 5.0)
cfg["predation.predprey.sizeratio.min.sp1"] = "1.0;2.0"
cfg["predation.predprey.sizeratio.max.sp1"] = "10.0;5.0"
```

**Backward compatibility test:** Use `sizeratio.min=1.0, sizeratio.max=3.5` (matching existing test fixtures), NOT the Java convention values from `_make_base_config`.

**Java convention swap test:** Add a test verifying that Java-convention values (`sizeratio.min=50, sizeratio.max=3`) produce a working predation window after swap.

**Discard the reasoning block** (lines 467-521 in the plan) — it was exploratory analysis that led to wrong intermediate conclusions. The corrected tests above supersede it.

### Correction 2: Task 1 commit leaves predation tests broken

Task 1 changes `size_ratio_min/max` to 2D but does NOT update the predation kernel. This means the full test suite WILL fail after Task 1's commit. The plan explicitly notes this and instructs the implementer to only run `test_engine_feeding_stages.py` tests, not the full suite, until Task 2 is complete. **Task 2 must be committed immediately after Task 1 to restore the green state.**

### Correction 3: Background species 2D concatenation

When extending size ratio arrays with background species values in `config.py`, the focal 2D array `(n_focal, max_stages)` must be concatenated with a background 2D array `(n_bkg, max_stages)`. Both must have the same `max_stages` dimension. Compute `max_stages` across ALL species (focal + background) before building the arrays. Pad all species to `max_stages` by repeating their last valid ratio value.

### Correction 4: Reuse `_parse_array_float` from background.py

The spec instructs reusing `_parse_array_float()` (actually named `_parse_floats()` in `background.py`). Either import it or extract it to a shared utility. Do NOT create a duplicate `_parse_multi_float` in `config.py`.

### Correction 5: `state.weight` units confirmed

`state.weight` is in **tonnes per individual** (confirmed: `reproduction.py` line 53 converts via `* 1e6` for tonnes→grams). The `compute_feeding_stages` weight metric `* 1e6` conversion is correct.
