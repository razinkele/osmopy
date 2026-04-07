# Codebase Fix Plan — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 47 findings from 7-agent deep review across engine, config, schema, calibration, and UI layers without breaking parity or existing tests.

**Architecture:** 4-phase incremental remediation. Phase 1 batches safe 1-line fixes. Phase 2 fixes engine correctness with parity gates. Phase 3 hardens core types one at a time with individual gates. Phase 4 adds tests and polish.

**Tech Stack:** Python 3.12, NumPy, Numba, xarray, Shiny, pytest, ruff

**Spec:** `docs/superpowers/specs/2026-04-04-codebase-fix-plan-design.md`

---

## Phase 1 — Zero-Risk Quick Wins

### Task 1: Fix config key case bugs (C4, C5)

**Files:**
- Modify: `ui/pages/grid.py:186`
- Modify: `osmose/engine/config.py:1336`

- [ ] **Step 1: Fix C4 — lowercase ndtperyear key**

In `ui/pages/grid.py:186`, change:
```python
nsteps = int(float(cfg.get("simulation.time.ndtPerYear", "24") or "24"))
```
to:
```python
nsteps = int(float(cfg.get("simulation.time.ndtperyear", "24") or "24"))
```

- [ ] **Step 2: Fix C5 — lowercase sizeinf key**

In `osmose/engine/config.py:1336`, change:
```python
output_bioen_sizeinf=cfg.get("output.bioen.sizeInf.enabled", "false").lower() == "true",
```
to:
```python
output_bioen_sizeinf=cfg.get("output.bioen.sizeinf.enabled", "false").lower() == "true",
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed

- [ ] **Step 4: Commit**

```bash
git add ui/pages/grid.py osmose/engine/config.py
git commit -m "fix: lowercase config key lookups for ndtperyear and bioen sizeinf (C4, C5)"
```

---

### Task 2: Fix calibration key patterns (C6)

**Files:**
- Modify: `osmose/calibration/configure.py:11-12`

- [ ] **Step 1: Fix deprecated key patterns**

In `osmose/calibration/configure.py`, change lines 11-12:
```python
    r"mortality\.natural\.rate\.sp\d+": (0.001, 2.0),
    r"mortality\.natural\.larva\.rate\.sp\d+": (0.001, 10.0),
```
to:
```python
    r"mortality\.additional\.rate\.sp\d+": (0.001, 2.0),
    r"mortality\.additional\.larva\.rate\.sp\d+": (0.001, 10.0),
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed

- [ ] **Step 3: Commit**

```bash
git add osmose/calibration/configure.py
git commit -m "fix: use mortality.additional key pattern in calibration auto-detect (C6)"
```

---

### Task 3: Replace assert with if/raise for bioenergetics (C3)

**Files:**
- Modify: `osmose/engine/simulate.py:151-164`

- [ ] **Step 1: Replace 14 assert statements**

In `osmose/engine/simulate.py`, replace lines 151-164:
```python
    assert config.bioen_beta is not None
    assert config.bioen_assimilation is not None
    assert config.bioen_c_m is not None
    assert config.bioen_eta is not None
    assert config.bioen_r is not None
    assert config.bioen_m0 is not None
    assert config.bioen_m1 is not None
    assert config.bioen_e_mobi is not None
    assert config.bioen_e_d is not None
    assert config.bioen_tp is not None
    assert config.bioen_e_maint is not None
    assert config.bioen_i_max is not None
    assert config.bioen_theta is not None
    assert config.bioen_c_rate is not None
```
with:
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

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed

- [ ] **Step 3: Commit**

```bash
git add osmose/engine/simulate.py
git commit -m "fix: replace assert with if/raise for bioenergetics validation (C3)"
```

---

### Task 4: Numba fallback logging (H7)

**Files:**
- Modify: `osmose/engine/processes/mortality.py:12,42-48`
- Modify: `osmose/engine/processes/predation.py:12,27-33`
- Modify: `osmose/engine/processes/movement.py:5,19-25`

- [ ] **Step 1: Fix mortality.py**

In `osmose/engine/processes/mortality.py`, add after `import warnings` (line 12):
```python
from osmose.logging import setup_logging

_log = setup_logging("osmose.engine.processes.mortality")
```

Then replace lines 43-48:
```python
    warnings.warn(
        "Numba is not installed. Mortality will use pure Python fallback, "
        "which may be 10-100x slower. Install numba for optimal performance.",
        ImportWarning,
        stacklevel=2,
    )
```
with:
```python
    _log.warning(
        "Numba is not installed. Mortality will use pure Python fallback, "
        "which may be 10-100x slower. Install numba for optimal performance."
    )
```

- [ ] **Step 2: Fix predation.py**

In `osmose/engine/processes/predation.py`, add after `import warnings` (line 12):
```python
from osmose.logging import setup_logging

_log = setup_logging("osmose.engine.processes.predation")
```

Then replace lines 28-33:
```python
    warnings.warn(
        "Numba is not installed. Predation will use pure Python fallback, "
        "which may be 10-100x slower. Install numba for optimal performance.",
        ImportWarning,
        stacklevel=2,
    )
```
with:
```python
    _log.warning(
        "Numba is not installed. Predation will use pure Python fallback, "
        "which may be 10-100x slower. Install numba for optimal performance."
    )
```

- [ ] **Step 3: Fix movement.py**

In `osmose/engine/processes/movement.py`, add after `import warnings` (line 5):
```python
from osmose.logging import setup_logging

_log = setup_logging("osmose.engine.processes.movement")
```

Then replace lines 20-25:
```python
    warnings.warn(
        "Numba is not installed. Movement will use pure Python fallback, "
        "which may be 10-100x slower. Install numba for optimal performance.",
        ImportWarning,
        stacklevel=2,
    )
```
with:
```python
    _log.warning(
        "Numba is not installed. Movement will use pure Python fallback, "
        "which may be 10-100x slower. Install numba for optimal performance."
    )
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/mortality.py osmose/engine/processes/predation.py osmose/engine/processes/movement.py
git commit -m "fix: use _log.warning instead of warnings.warn for Numba fallback (H7)"
```

---

### Task 5: Fix calibration __all__ exports (H8)

**Files:**
- Modify: `osmose/calibration/__init__.py`

- [ ] **Step 1: Add missing imports and __all__ entries**

In `osmose/calibration/__init__.py`, change:
```python
from osmose.calibration.objectives import (
    biomass_rmse,
    abundance_rmse,
    diet_distance,
    normalized_rmse,
)
```
to:
```python
from osmose.calibration.objectives import (
    biomass_rmse,
    abundance_rmse,
    diet_distance,
    normalized_rmse,
    yield_rmse,
    catch_at_size_distance,
    size_at_age_rmse,
    weighted_multi_objective,
)
```

And change `__all__` to add the 4 missing entries after `"normalized_rmse"`:
```python
__all__ = [
    "biomass_rmse",
    "abundance_rmse",
    "diet_distance",
    "normalized_rmse",
    "yield_rmse",
    "catch_at_size_distance",
    "size_at_age_rmse",
    "weighted_multi_objective",
    "FreeParameter",
    "Transform",
    "OsmoseCalibrationProblem",
    "SurrogateCalibrator",
    "SensitivityAnalyzer",
]
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed

- [ ] **Step 3: Commit**

```bash
git add osmose/calibration/__init__.py
git commit -m "fix: add missing calibration objective exports to __all__ (H8)"
```

---

### Task 6: Stale comments and docstring fixes (H9, M10-M14)

**Files:**
- Modify: `osmose/engine/simulate.py:49,121`
- Modify: `osmose/schema/species.py:89,178,388`
- Modify: `osmose/schema/grid.py:11`
- Modify: `osmose/schema/ltl.py:12`
- Modify: `osmose/engine/processes/predation.py:1-8`

- [ ] **Step 1: Remove stale stub comment (H9)**

In `osmose/engine/simulate.py`, remove the 3-line comment block at lines 48-50:
```python
# ---------------------------------------------------------------------------
# Stub process functions (replaced in Phase 2-7)
# ---------------------------------------------------------------------------
```

- [ ] **Step 2: Fix growth docstring (M13)**

In `osmose/engine/simulate.py:121`, change:
```python
    """Apply Von Bertalanffy growth gated by predation success."""
```
to:
```python
    """Apply growth (Von Bertalanffy or Gompertz) gated by predation success."""
```

- [ ] **Step 3: Fix schema descriptions (M10)**

In `osmose/schema/species.py:89`, change:
```python
        description="Java class implementing the growth model",
```
to:
```python
        description="Growth model type (Von Bertalanffy or Gompertz)",
```

In `osmose/schema/grid.py:11`, change:
```python
        description="Java class implementing the grid",
```
to:
```python
        description="Grid implementation type",
```

In `osmose/schema/ltl.py:12`, change:
```python
        description="Java class implementing the LTL forcing",
```
to:
```python
        description="LTL forcing method",
```

- [ ] **Step 4: Fix egg description (M11) and unit (M12)**

In `osmose/schema/species.py:178`, change:
```python
        description="Egg size at hatching",
```
to:
```python
        description="Egg size (initial larval length)",
```

In `osmose/schema/species.py:388`, change:
```python
        unit="tons",
```
to:
```python
        unit="tonnes",
```

- [ ] **Step 5: Add predation legacy note (M14)**

In `osmose/engine/processes/predation.py`, change lines 1-8:
```python
"""Predation process for the OSMOSE Python engine.

Size-based opportunistic predation within grid cells. Predators are
processed sequentially in random order with asynchronous prey biomass updates.

Uses Numba JIT compilation for the inner cell loop when available,
falling back to pure Python otherwise.
"""
```
to:
```python
"""Predation process for the OSMOSE Python engine.

Size-based opportunistic predation within grid cells. Predators are
processed sequentially in random order with asynchronous prey biomass updates.

Note: The interleaved mortality path (mortality.py) is the primary predation
codepath used by simulate(). This standalone predation module is retained for
testing and backward compatibility.

Uses Numba JIT compilation for the inner cell loop when available,
falling back to pure Python otherwise.
"""
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/simulate.py osmose/schema/species.py osmose/schema/grid.py osmose/schema/ltl.py osmose/engine/processes/predation.py
git commit -m "docs: fix stale comments, schema descriptions, and docstrings (H9, M10-M14)"
```

---

### Task 7: Add type annotations to validator (M15)

**Files:**
- Modify: `osmose/config/validator.py:10,58,77`

- [ ] **Step 1: Add type hints**

In `osmose/config/validator.py`, change the existing import at line 7:
```python
from osmose.schema.base import ParamType
```
to:
```python
from osmose.schema.base import OsmoseField, ParamType
from osmose.schema.registry import ParameterRegistry
```

Also remove the redundant local import `from osmose.schema.base import ParamType` inside `validate_field` at line 60, since `ParamType` is now imported at module level.

Change line 10:
```python
def validate_config(config: dict[str, str], registry) -> tuple[list[str], list[str]]:
```
to:
```python
def validate_config(config: dict[str, str], registry: ParameterRegistry) -> tuple[list[str], list[str]]:
```

Change line 58:
```python
def validate_field(key: str, value: str, field) -> str | None:
```
to:
```python
def validate_field(key: str, value: str, field: OsmoseField) -> str | None:
```

Change line 77 (`check_file_references`) to add types to `config` and `registry`:
```python
def check_file_references(config: dict[str, str], base_dir: str, registry: ParameterRegistry | None = None) -> list[str]:
```

- [ ] **Step 2: Run tests and lint**

Run: `.venv/bin/python -m pytest tests/ -x -q && .venv/bin/ruff check osmose/ ui/ tests/`
Expected: 1761 passed, 0 lint errors

- [ ] **Step 3: Commit**

```bash
git add osmose/config/validator.py
git commit -m "refactor: add type annotations to config validator public API (M15)"
```

---

### Task 8: Phase 1 Gate

- [ ] **Step 1: Run Gate A (full test suite)**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed, 14 skipped

- [ ] **Step 2: Run Gate C (lint)**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`
Expected: 0 warnings

---

## Phase 2 — Engine Correctness

### Task 9: Division by zero guard in prey window (C1)

**Files:**
- Modify: `osmose/engine/processes/predation.py:407-409`
- Modify: `osmose/engine/processes/mortality.py:325-327`
- Modify: `osmose/engine/processes/mortality.py:776-778`

- [ ] **Step 1: Add guard in predation.py**

In `osmose/engine/processes/predation.py`, before line 408 (`prey_size_min = pred_len / r_max_val`), add:
```python
            if r_min_val <= 0 or r_max_val <= 0:
                continue
```
So lines 405-409 become:
```python
            # Size overlap: what fraction of the resource size range
            # falls within the predator's prey window?
            # Prey window: [L/r_max, L/r_min] (r_max > r_min by convention)
            if r_min_val <= 0 or r_max_val <= 0:
                continue
            prey_size_min = pred_len / r_max_val  # smallest prey this predator eats
            prey_size_max = pred_len / r_min_val  # largest prey this predator eats
```

- [ ] **Step 2: Add guard in mortality.py Python path**

In `osmose/engine/processes/mortality.py`, before line 326 (`prey_size_min = pred_len / r_max`), add:
```python
            if r_min <= 0 or r_max <= 0:
                continue
```
So lines 325-328 become:
```python
            if r_min <= 0 or r_max <= 0:
                continue
            prey_size_min = pred_len / r_max
            prey_size_max = pred_len / r_min
```

- [ ] **Step 3: Add guard in mortality.py Numba path**

In `osmose/engine/processes/mortality.py`, before line 777 (`prey_size_min = pred_len / r_max`), add:
```python
            if r_min <= 0 or r_max <= 0:
                continue
```
So lines 776-779 become:
```python
            if r_min <= 0 or r_max <= 0:
                continue
            prey_size_min = pred_len / r_max
            prey_size_max = pred_len / r_min
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/predation.py osmose/engine/processes/mortality.py
git commit -m "fix: guard against division by zero when size_ratio_min is 0 (C1)"
```

---

### Task 10: CSV grid dimension validation (C7)

**Files:**
- Modify: `osmose/engine/movement_maps.py:74-75`

- [ ] **Step 1: Add validation before grid assignment**

In `osmose/engine/movement_maps.py`, between line 73 (`row_values.append(float(p))`) and line 74 (`grid_row = ny - 1 - csv_row_idx`), add:
```python
        if len(row_values) < nx:
            raise ValueError(
                f"Movement map CSV row {csv_row_idx} has {len(row_values)} columns, "
                f"expected {nx} (grid nx)"
            )
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed

- [ ] **Step 3: Commit**

```bash
git add osmose/engine/movement_maps.py
git commit -m "fix: validate CSV grid dimensions in movement map loader (C7)"
```

---

### Task 11: Fix sync_inputs AttributeError (C8)

**Files:**
- Modify: `ui/state.py:106-112`

- [ ] **Step 1: Replace try/except with hasattr guard**

In `ui/state.py`, replace lines 106-112:
```python
        try:
            val = getattr(input, input_id)()
        except AttributeError:
            continue
        except TypeError:
            _log.warning("sync_inputs: TypeError reading input '%s'", input_id)
            continue
```
with:
```python
        if not hasattr(input, input_id):
            continue
        try:
            val = getattr(input, input_id)()
        except TypeError:
            _log.warning("sync_inputs: TypeError reading input '%s'", input_id)
            continue
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed

- [ ] **Step 3: Commit**

```bash
git add ui/state.py
git commit -m "fix: use hasattr guard instead of broad except AttributeError in sync_inputs (C8)"
```

---

### Task 12: Add logging for empty DataFrames (H5) and ncell warning (2.5)

**Files:**
- Modify: `osmose/results.py:56-59`
- Modify: `ui/pages/run.py:84-87`

- [ ] **Step 1: Add logging in results.py**

In `osmose/results.py`, in the `read_csv` method after `result = {}` and the loop, before `return result`, add logging when no files match. Find the pattern where `if not result:` or equivalent returns empty — add before each empty return:
```python
        if not result:
            _log.info("No files matching '%s' in %s", pattern, self.output_dir)
```

Do the same for `_read_species_output`, `_read_2d_output`, and `export_dataframe` methods wherever they return `pd.DataFrame()` in non-strict mode.

- [ ] **Step 2: Add ncell injection warning**

In `ui/pages/run.py`, after line 84 (`except ValueError:`) and before line 85 (`return`), add:
```python
        _log.warning("Cannot inject random movement ncell: grid dimensions invalid (nlon=%r, nlat=%r)",
                      config.get("grid.nlon"), config.get("grid.nlat"))
```

Similarly, after line 86 (`if nlon <= 0 or nlat <= 0:`) and before line 87 (`return`), add:
```python
        _log.warning("Cannot inject random movement ncell: grid dimensions non-positive (nlon=%d, nlat=%d)", nlon, nlat)
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed

- [ ] **Step 4: Commit**

```bash
git add osmose/results.py ui/pages/run.py
git commit -m "fix: add logging for empty DataFrame returns and ncell injection skip (H5, SF3)"
```

---

### Task 13: Phase 2 Gate

- [ ] **Step 1: Run Gate A (full test suite)**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed

- [ ] **Step 2: Run Gate B (parity validation)**

Run: `.venv/bin/python scripts/validate_engines.py --years 1`
Expected: All checks pass (Bay of Biscay + EEC)

- [ ] **Step 3: Run Gate C (lint)**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`
Expected: 0 warnings

---

## Phase 3 — Structural Hardening

### Task 14: OsmoseField.__post_init__ validation (H3a)

**Files:**
- Modify: `osmose/schema/base.py`

- [ ] **Step 1: Add __post_init__ to OsmoseField**

In `osmose/schema/base.py`, add after the field definitions (after line 53 `advanced: bool = False`):
```python
    def __post_init__(self) -> None:
        if self.indexed and "{idx}" not in self.key_pattern:
            raise ValueError(
                f"OsmoseField indexed=True but key_pattern has no {{idx}}: {self.key_pattern}"
            )
        if not self.indexed and "{idx}" in self.key_pattern:
            raise ValueError(
                f"OsmoseField indexed=False but key_pattern contains {{idx}}: {self.key_pattern}"
            )
        if self.choices is not None and self.param_type != ParamType.ENUM:
            raise ValueError(
                f"OsmoseField has choices but param_type is {self.param_type}, not ENUM: {self.key_pattern}"
            )
        if self.min_val is not None and self.max_val is not None and self.min_val > self.max_val:
            raise ValueError(
                f"OsmoseField min_val ({self.min_val}) > max_val ({self.max_val}): {self.key_pattern}"
            )
```

- [ ] **Step 2: Pre-check all 215 fields**

Run: `.venv/bin/python -c "from osmose.schema import ALL_FIELDS; import itertools; fs=list(itertools.chain.from_iterable(ALL_FIELDS)); print(f'{len(fs)} fields validated')"`
Expected: `215 fields validated`

- [ ] **Step 3: Run Gate A + B + C**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed

Run: `.venv/bin/python scripts/validate_engines.py --years 1`
Expected: Pass

Run: `.venv/bin/ruff check osmose/ ui/ tests/`
Expected: 0 warnings

- [ ] **Step 4: Commit**

```bash
git add osmose/schema/base.py
git commit -m "feat: add __post_init__ validation to OsmoseField (H3a)"
```

---

### Task 15: Grid.__init__ validation (H3b)

**Files:**
- Modify: `osmose/engine/grid.py:22-34`

- [ ] **Step 1: Add validation at end of __init__**

In `osmose/engine/grid.py`, after line 34 (`self.lon = lon`), add:
```python
        if ny < 1 or nx < 1:
            raise ValueError(f"Grid dimensions must be >= 1, got ny={ny}, nx={nx}")
        if ocean_mask.shape != (ny, nx):
            raise ValueError(
                f"ocean_mask shape {ocean_mask.shape} does not match (ny={ny}, nx={nx})"
            )
        if lat is not None and lat.shape != (ny,):
            raise ValueError(f"lat shape {lat.shape} does not match ny={ny}")
        if lon is not None and lon.shape != (nx,):
            raise ValueError(f"lon shape {lon.shape} does not match nx={nx}")
```

- [ ] **Step 2: Run Gate A + B + C**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed

Run: `.venv/bin/python scripts/validate_engines.py --years 1`
Expected: Pass

- [ ] **Step 3: Commit**

```bash
git add osmose/engine/grid.py
git commit -m "feat: add __init__ validation to Grid for shape invariants (H3b)"
```

---

### Task 16: SchoolState.__post_init__ shape check (H3c)

**Files:**
- Modify: `osmose/engine/state.py`

- [ ] **Step 1: Add __post_init__ after field definitions**

In `osmose/engine/state.py`, add after line 78 (`egg_retained: NDArray[np.float64]`) and before line 80 (`def __len__`):
```python
    def __post_init__(self) -> None:
        n = len(self.species_id)
        for f in fields(self):
            val = getattr(self, f.name)
            if f.name == "n_dead":
                if val.shape != (n, len(MortalityCause)):
                    raise ValueError(
                        f"n_dead shape {val.shape} != ({n}, {len(MortalityCause)})"
                    )
            elif val.ndim == 1 and len(val) != n:
                raise ValueError(
                    f"SchoolState.{f.name} length {len(val)} != {n}"
                )
```

- [ ] **Step 2: Run Gate A + B + C**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed

Run: `.venv/bin/python scripts/validate_engines.py --years 1`
Expected: Pass

- [ ] **Step 3: Commit**

```bash
git add osmose/engine/state.py
git commit -m "feat: add __post_init__ shape validation to SchoolState (H3c)"
```

---

### Task 17: NetCDF context managers (H2)

**Files:**
- Modify: `osmose/engine/physical_data.py:30-36`
- Modify: `osmose/engine/background.py:296-309`

- [ ] **Step 1: Fix physical_data.py**

In `osmose/engine/physical_data.py`, replace lines 30-36:
```python
        ds = xr.open_dataset(path)
        raw = ds[varname].values
        if raw.ndim == 2:
            raw = raw[np.newaxis, :, :]
        data = factor * (raw.astype(np.float64) + offset)
        return cls(data=data, constant=None, nsteps_year=nsteps_year)
```
with:
```python
        with xr.open_dataset(path) as ds:
            raw = ds[varname].values
            if raw.ndim == 2:
                raw = raw[np.newaxis, :, :]
            data = factor * (raw.astype(np.float64) + offset)
        return cls(data=data, constant=None, nsteps_year=nsteps_year)
```

- [ ] **Step 2: Fix background.py**

In `osmose/engine/background.py`, replace line 296:
```python
                    ds = xr.open_dataset(nc_path)
```
with:
```python
                    with xr.open_dataset(nc_path) as ds:
```
Then indent lines 297-310 (from `stripped = sp.name` through `raw = raw * sp.multiplier`) one level to be inside the `with` block. The `raw = da.values.astype(np.float64)` at line 310 materializes the data into a NumPy array, so the dataset can be safely closed when the `with` block exits. Lines 314+ (regrid, append) operate on the materialized `raw` array and stay outside the `with` block.

- [ ] **Step 3: Run Gate A + B + C**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed

- [ ] **Step 4: Commit**

```bash
git add osmose/engine/physical_data.py osmose/engine/background.py
git commit -m "fix: wrap xr.open_dataset in context managers to prevent file handle leaks (H2)"
```

---

### Task 18: OsmoseResults context manager (H4)

**Files:**
- Modify: `osmose/results.py`

- [ ] **Step 1: Add __enter__ and __exit__**

In `osmose/results.py`, add to the `OsmoseResults` class (after `__init__`):
```python
    def __enter__(self) -> OsmoseResults:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close_cache()
```

- [ ] **Step 2: Run Gate A + B + C**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed

- [ ] **Step 3: Commit**

```bash
git add osmose/results.py
git commit -m "feat: add context manager protocol to OsmoseResults (H4)"
```

---

### Task 19: Larvae flag fix (C2, latent)

**Files:**
- Modify: `osmose/engine/simulate.py:171`

- [ ] **Step 1: Fix larvae flag**

In `osmose/engine/simulate.py:171`, change:
```python
    is_larvae = state.is_egg  # larvae flag uses egg field (age 0 schools)
```
to:
```python
    is_larvae = state.age_dt < state.first_feeding_age_dt
```

- [ ] **Step 2: Run Gate A + B + C**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed

Run: `.venv/bin/python scripts/validate_engines.py --years 1`
Expected: Pass (no behavioral change — first_feeding_age_dt is currently hardcoded to 1)

- [ ] **Step 3: Commit**

```bash
git add osmose/engine/simulate.py
git commit -m "fix: use age_dt < first_feeding_age_dt for larvae check instead of is_egg (C2)"
```

---

### Task 20: Eliminate _last_key_case_map global (H1 partial)

**Files:**
- Modify: `osmose/config/reader.py:14,42-43`
- Modify: `ui/state.py:19-42`
- Modify: `ui/pages/setup.py` (where reader is created)
- Modify: `ui/pages/advanced.py` (where reader is created)
- Modify: `ui/pages/run.py:122-129`

- [ ] **Step 1: Remove global from reader.py**

In `osmose/config/reader.py`, delete line 14:
```python
_last_key_case_map: dict[str, str] = {}
```

In the `read()` method, remove lines 42-43:
```python
        global _last_key_case_map
        _last_key_case_map = dict(self.key_case_map)
```

- [ ] **Step 2: Add key_case_map to AppState**

In `ui/state.py`, inside `AppState.__init__`, add after line 42 (`self.results_loaded`):
```python
        self.key_case_map: dict[str, str] = {}
```

- [ ] **Step 3: Populate key_case_map after config load in setup.py**

In `ui/pages/setup.py`, find where `reader = OsmoseConfigReader()` is used and `reader.read(...)` is called. After the read, add:
```python
            state.key_case_map = dict(reader.key_case_map)
```

- [ ] **Step 4: Populate key_case_map after config load in advanced.py**

In `ui/pages/advanced.py`, find where `reader = OsmoseConfigReader()` is used and `reader.read(...)` is called. After the read, add:
```python
            state.key_case_map = dict(reader.key_case_map)
```

- [ ] **Step 5: Update write_temp_config signature and call site**

In `ui/pages/run.py`, change the `write_temp_config` function signature at line 97-99:
```python
def write_temp_config(
    config: dict[str, str], output_dir: Path, source_dir: Path | None = None
) -> Path:
```
to:
```python
def write_temp_config(
    config: dict[str, str], output_dir: Path, source_dir: Path | None = None,
    key_case_map: dict[str, str] | None = None,
) -> Path:
```

Then replace lines 122-129:
```python
    from osmose.config.reader import _last_key_case_map

    master = output_dir / "osm_all-parameters.csv"
    lines = []
    for key, value in sorted(config.items()):
        if key.startswith(("osmose.configuration.", "_")):
            continue
        original_key = _last_key_case_map.get(key, key)
        lines.append(f"{original_key} ; {value}\n")
```
with:
```python
    case_map = key_case_map or {}
    master = output_dir / "osm_all-parameters.csv"
    lines = []
    for key, value in sorted(config.items()):
        if key.startswith(("osmose.configuration.", "_")):
            continue
        original_key = case_map.get(key, key)
        lines.append(f"{original_key} ; {value}\n")
```

Update the call site at line 253:
```python
        config_path = write_temp_config(config, work_dir, source_dir)
```
to:
```python
        config_path = write_temp_config(config, work_dir, source_dir, key_case_map=state.key_case_map)
```

- [ ] **Step 6: Run Gate A + B + C**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 passed

Run: `.venv/bin/python scripts/validate_engines.py --years 1`
Expected: Pass

- [ ] **Step 7: Commit**

```bash
git add osmose/config/reader.py ui/state.py ui/pages/setup.py ui/pages/advanced.py ui/pages/run.py
git commit -m "refactor: move _last_key_case_map from module global to AppState (H1 partial)"
```

---

## Phase 4 — Tests + Polish

### Task 21: New tests for coverage gaps (T1, T2, T3, T4, T5, T7)

**Files:**
- Modify: `tests/test_engine_mortality.py`
- Modify: `tests/test_engine_mortality_loop.py`
- Modify: `tests/test_engine_config.py`
- Modify: `tests/test_config_writer.py`
- Modify: `tests/test_engine_resources.py`

- [ ] **Step 1: Add out_mortality formula test (T1)**

In `tests/test_engine_mortality.py`, add:
```python
def test_out_mortality_formula():
    """Verify out_mortality n_dead matches analytical formula."""
    from osmose.engine.processes.natural import out_mortality

    n_dt_per_year = 24
    rate = 1.0
    N = 1000.0
    expected_dead = N * (1 - np.exp(-rate / n_dt_per_year))

    # Create minimal state and config for out_mortality
    # (adapt from existing test fixtures in this file)
    state = _make_state(abundance=N)
    config = _make_config(out_mortality_rate=rate, n_dt_per_year=n_dt_per_year)
    result = out_mortality(state, config)
    actual_dead = result.n_dead[:, int(MortalityCause.OUT)].sum()
    np.testing.assert_allclose(actual_dead, expected_dead, rtol=1e-10)
```

Adapt `_make_state` and `_make_config` from existing helpers in the file.

- [ ] **Step 2: Add additional_mortality_by_dt test (T2)**

In `tests/test_engine_mortality.py`, add:
```python
def test_additional_mortality_by_dt_time_varying():
    """Verify time-varying additional mortality uses correct rate per step."""
    from osmose.engine.processes.natural import additional_mortality

    rates = [0.1, 0.5, 0.9]  # 3-step cycle
    for step in range(6):
        expected_rate = rates[step % len(rates)]
        state = _make_state(abundance=1000.0)
        config = _make_config(additional_mortality_by_dt=[rates])
        result = additional_mortality(state, config, step=step)
        # Verify the rate used matches rates[step % 3]
        actual_dead = result.n_dead[:, int(MortalityCause.ADDITIONAL)].sum()
        expected_dead = 1000.0 * (1 - np.exp(-expected_rate))
        np.testing.assert_allclose(actual_dead, expected_dead, rtol=1e-6)
```

Adapt fixtures from existing test patterns in the file.

- [ ] **Step 3: Add mortality Numba vs Python parity test (T3)**

In `tests/test_engine_mortality_loop.py`, add:
```python
def test_mortality_loop_numba_vs_python_parity():
    """Verify Numba and Python mortality paths produce identical results."""
    import osmose.engine.processes.mortality as mort

    # Save original flag
    original = mort._HAS_NUMBA

    # Run with Python fallback
    mort._HAS_NUMBA = False
    state_py = _run_mortality_loop()  # use existing helper that runs mortality()

    # Run with Numba
    mort._HAS_NUMBA = original
    if not original:
        pytest.skip("Numba not available")
    state_numba = _run_mortality_loop()

    # Compare outputs
    np.testing.assert_allclose(state_py.abundance, state_numba.abundance, rtol=1e-10)
    np.testing.assert_allclose(state_py.n_dead, state_numba.n_dead, rtol=1e-10)

    mort._HAS_NUMBA = original  # restore
```

Adapt `_run_mortality_loop` from existing fixtures in the file.

- [ ] **Step 4: Add EngineConfig error tests (T4)**

In `tests/test_engine_config.py`, add:
```python
def test_engine_config_from_empty_dict_raises():
    """EngineConfig.from_dict({}) should raise, not silently produce garbage."""
    from osmose.engine.config import EngineConfig
    with pytest.raises((KeyError, ValueError)):
        EngineConfig.from_dict({})


def test_engine_config_non_numeric_ndtperyear_raises():
    """Non-numeric ndtperyear should raise ValueError."""
    from osmose.engine.config import EngineConfig
    cfg = _minimal_config()
    cfg["simulation.time.ndtperyear"] = "abc"
    with pytest.raises(ValueError):
        EngineConfig.from_dict(cfg)
```

Adapt `_minimal_config` from existing test helpers.

- [ ] **Step 5: Add config writer semicolon test (T5)**

In `tests/test_config_writer.py`, add:
```python
def test_roundtrip_value_with_semicolon():
    """Values containing semicolons should survive write/read roundtrip or be rejected."""
    from osmose.config.writer import OsmoseConfigWriter
    from osmose.config.reader import OsmoseConfigReader

    config = {"species.name.sp0": "Test; with semicolon", "simulation.time.nyear": "10"}
    writer = OsmoseConfigWriter()
    writer.write(config, tmp_path / "output")

    reader = OsmoseConfigReader()
    result = reader.read(tmp_path / "output" / "osm_all-parameters.csv")
    # Either the value survives or the writer escapes/rejects it
    assert "species.name.sp0" in result
```

- [ ] **Step 6: Add resource depletion test (T7)**

In `tests/test_engine_resources.py`, add to `TestResourceState`:
```python
    def test_resource_depletion_to_zero_no_crash(self):
        """Depleting a resource to zero biomass should not cause division by zero."""
        grid = Grid.from_dimensions(ny=2, nx=2)
        # Build a minimal config with one resource species
        config = {
            "ltl.nsp": "1",
            "ltl.java.classname.0": "fr.ird.osmose.ltl.LTLFastForcing",
            "ltl.netcdf.var.biomass.0": "phyto",
            "species.name.sp0": "Phyto",
            "species.type.sp0": "resource",
            "species.size.min.sp0": "0.001",
            "species.size.max.sp0": "0.01",
            "species.trophiclevel.sp0": "1.0",
            "species.accessibility2fisheries.sp0": "0.05",
        }
        rs = ResourceState(config=config, grid=grid)
        if rs.n_resources == 0:
            pytest.skip("Resource creation requires NetCDF forcing data")
        # Deplete completely and verify no crash
        rs.deduct(species_idx=0, cell_y=0, cell_x=0, amount=rs.get_cell_biomass(0, 0, 0))
        assert rs.get_cell_biomass(0, 0, 0) >= 0.0
        # Deducting from zero should not crash
        rs.deduct(species_idx=0, cell_y=0, cell_x=0, amount=1.0)
        assert rs.get_cell_biomass(0, 0, 0) >= 0.0
```

- [ ] **Step 7: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 + new tests passed

- [ ] **Step 8: Commit**

```bash
git add tests/test_engine_mortality.py tests/test_engine_mortality_loop.py tests/test_engine_config.py tests/test_config_writer.py tests/test_engine_resources.py
git commit -m "test: add coverage for out_mortality formula, time-varying rates, Numba parity, config errors, resource depletion (T1-T7)"
```

---

### Task 22: Calibration abort on all-infinity (H6)

**Files:**
- Modify: `osmose/calibration/problem.py`

- [ ] **Step 1: Add failure tracking to _evaluate**

In `osmose/calibration/problem.py`, in the `_evaluate` method, after computing all objectives for a generation, add failure counting:

Find the section where `F[i, k] = obj_val` is set in the loop. After the loop completes (both parallel and sequential branches), add:
```python
        n_inf = np.all(np.isinf(F), axis=1).sum()
        if n_inf > len(F) * 0.5:
            raise RuntimeError(
                f"Calibration aborted: {n_inf}/{len(F)} candidates failed "
                f"(>50% returned inf). Check JAR path and config validity."
            )
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: All passed

- [ ] **Step 3: Commit**

```bash
git add osmose/calibration/problem.py
git commit -m "fix: abort calibration when >50% of candidates fail (H6)"
```

---

### Task 23: Remaining medium items (M4, M5, M7, M8, M9)

**Files:**
- Modify: `osmose/engine/processes/movement.py:141,145`
- Modify: `osmose/scenarios.py:19-29`
- Modify: `osmose/engine/processes/reproduction.py:104-106`
- Modify: `ui/pages/grid_helpers.py:44-54`
- Modify: `osmose/config/reader.py:85-101`

- [ ] **Step 1: Fix BFS queue (M5)**

In `osmose/engine/processes/movement.py`, add at the top imports:
```python
from collections import deque
```

At line 141, change:
```python
        queue: list[tuple[int, int]] = [(start_x, start_y)]
```
to:
```python
        queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
```

At line 145, change:
```python
            cx, cy = queue.pop(0)
```
to:
```python
            cx, cy = queue.popleft()
```

- [ ] **Step 2: Add Scenario name validation (M7)**

In `osmose/scenarios.py`, add to the `Scenario` dataclass after line 29:
```python
    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Scenario name must not be empty")
        if "/" in self.name or "\\" in self.name or ".." in self.name:
            raise ValueError(f"Scenario name contains invalid characters: {self.name!r}")
```

- [ ] **Step 3: Batch append in reproduction (M4)**

In `osmose/engine/processes/reproduction.py`, replace lines 104-106:
```python
    # Append new egg schools
    for new in new_schools_list:
        state = state.append(new)
```
with:
```python
    # Append all new egg schools in one batch
    if new_schools_list:
        merged_fields = {}
        for f in fields(state):
            existing = getattr(state, f.name)
            parts = [existing] + [getattr(s, f.name) for s in new_schools_list]
            merged_fields[f.name] = np.concatenate(parts)
        state = SchoolState(**merged_fields)
```

`SchoolState` is already imported at line 11. Add `from dataclasses import fields` to the imports at the top of the file.

- [ ] **Step 4: Surface mask loading failures to UI (M8)**

In `ui/pages/grid_helpers.py`, in the `load_mask` function, after the except blocks that return `None` (lines 44-54), the callers should check for `None` when a mask file WAS configured. Find the caller in the grid page that calls `load_mask()` and add a UI notification when it returns `None` but a mask path was configured:
```python
if mask_path and mask_data is None:
    ui.notification_show(
        f"Grid mask file configured but could not be loaded: {mask_path}. "
        "Grid preview may be inaccurate.",
        type="warning",
        duration=10,
    )
```

- [ ] **Step 5: Track skipped lines in config reader (M9)**

In `osmose/config/reader.py`, in `read_file()`, add a counter before the loop:
```python
        skipped = 0
```

In the `else` branch at line 100 (the unparseable line warning), increment:
```python
                    skipped += 1
```

After the loop, return the count by storing it on the instance:
```python
        self.skipped_lines += skipped
```

Add `self.skipped_lines: int = 0` to `__init__` and reset it to 0 at the start of `read()`.

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: All passed

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/processes/movement.py osmose/scenarios.py osmose/engine/processes/reproduction.py ui/pages/grid_helpers.py osmose/config/reader.py
git commit -m "refactor: BFS deque, Scenario validation, batch append, mask UI warning, reader skip count (M4-M9)"
```

---

### Task 24: Comment quality fixes (Phase 4.4)

**Files:**
- Modify: `osmose/calibration/sensitivity.py:16,29`
- Modify: `osmose/engine/processes/mortality.py:1-8,57`
- Modify: `osmose/engine/processes/growth.py:84`
- Modify: `osmose/engine/processes/bioen_starvation.py:2`
- Modify: `osmose/schema/species.py:65,188-193`
- Modify: `osmose/engine/processes/energy_budget.py:60,64`

- [ ] **Step 1: Fix sensitivity.py comments**

In `osmose/calibration/sensitivity.py:16`, change `Saltelli sampling` to `Sobol sampling (Saltelli's extension)`.
At line 29, change `"""Generate Saltelli samples for Sobol analysis."""` to `"""Generate Sobol samples for sensitivity analysis."""`.

- [ ] **Step 2: Fix mortality.py comments**

In `osmose/engine/processes/mortality.py:57`, change:
```python
# Module-level TL tracking accumulator (set by mortality(), used by _apply_predation_for_school)
```
to:
```python
# Module-level TL tracking accumulator (set by mortality(), used by predation helpers and Numba kernels)
```

In lines 1-8, change the module docstring `apply ONE cause to ONE school` to clarify: for each school slot i, all four mortality causes are applied (one per cause from that cause's independent shuffled sequence).

- [ ] **Step 3: Fix growth.py comment**

In `osmose/engine/processes/growth.py:84`, change:
```python
    # Update weight from new length: W = c * L^b, convert grams to tonnes
```
to:
```python
    # Compute weight from length: W_tonnes = c * L^b * 1e-6 (allometric formula gives grams, convert to tonnes)
```

- [ ] **Step 4: Fix bioen_starvation.py docstring**

In `osmose/engine/processes/bioen_starvation.py:2`, change:
```python
Matches Java BioenStarvationMortality: per-sub-timestep, gonad flushed before death toll.
```
to:
```python
Matches Java BioenStarvationMortality: internally loops over n_subdt, flushing gonad before computing death toll at each sub-step.
```

- [ ] **Step 5: Fix schema species descriptions**

In `osmose/schema/species.py:65`, change the description for `species.vonbertalanffy.threshold.age`:
```python
        description="Age threshold for switching growth models",
```
to:
```python
        description="Age below which Von Bertalanffy uses linear interpolation from egg size",
```

In `osmose/schema/species.py:188-193`, clarify egg weight description and unit.

- [ ] **Step 6: Fix energy_budget.py unit descriptions**

In `osmose/engine/processes/energy_budget.py:60`, change c_m description to:
```
c_m: Maintenance metabolic coefficient (energy_units * g^{-beta} * year^{-1}, modulated by Arrhenius temperature function).
```

At line 64, change eta description to:
```
eta: Energy density ratio (grams of energy per gram of gonad tissue, dimensionless in g-equivalent framework).
```

- [ ] **Step 7: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: All passed

- [ ] **Step 8: Commit**

```bash
git add osmose/calibration/sensitivity.py osmose/engine/processes/mortality.py osmose/engine/processes/growth.py osmose/engine/processes/bioen_starvation.py osmose/schema/species.py osmose/engine/processes/energy_budget.py
git commit -m "docs: fix comment quality issues from Comment Analyzer review"
```

---

### Task 25: Phase 4 Gate (Final)

- [ ] **Step 1: Run Gate A (full test suite)**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1761 + ~10 new tests passed

- [ ] **Step 2: Run Gate C (lint)**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`
Expected: 0 warnings

- [ ] **Step 3: Run Gate B (parity — final confirmation)**

Run: `.venv/bin/python scripts/validate_engines.py --years 1`
Expected: All checks pass
