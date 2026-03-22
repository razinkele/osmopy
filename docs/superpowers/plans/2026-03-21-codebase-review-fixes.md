# Codebase Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all critical bugs, silent failures, thread-safety issues, resource leaks, and dead code identified by the 6-agent codebase review.

**Architecture:** Fixes are grouped into 10 tasks ordered by severity: critical bugs first, then silent failures, then thread-safety, then cleanup. Each task is independent and produces a commit. All tests must pass after each task.

**Tech Stack:** Python 3.12, NumPy, Shiny for Python, pytest

---

## Task 1: Fix duplicate age/size distribution block in simulate.py

The function `_collect_outputs()` computes age and size distributions **twice**. Lines 556–608 use correct logic (age in years via `//`, size via `searchsorted`). Lines 610–652 re-declare the same variables, overwriting the correct results with incorrect logic (age in timesteps, size via `np.floor`). Delete the second block entirely.

**Files:**
- Modify: `osmose/engine/simulate.py:610-653`
- Test: `tests/test_engine_simulate.py`

- [ ] **Step 1: Write a test that verifies age distribution uses year-based bins**

Add to `tests/test_engine_simulate.py`:

```python
def test_age_distribution_uses_year_bins(minimal_config):
    """Regression: duplicate block used timestep bins instead of year bins.

    The correct first block bins ages as age_dt // n_dt_per_year (years).
    The duplicate block used age_dt directly (timesteps), producing wrong bin count.
    """
    cfg_dict = dict(minimal_config)
    cfg_dict["output.biomass.byage.enabled"] = "true"
    cfg_dict["output.abundance.byage.enabled"] = "true"

    cfg = EngineConfig.from_dict(cfg_dict)
    grid = Grid.from_dimensions(ny=3, nx=3)
    rng = np.random.default_rng(42)
    outputs = simulate(cfg, grid, rng)

    # With lifespan=3 years and year-based bins, we expect max_age_yr+1 = 4 bins
    # The duplicate block would produce lifespan_dt+1 = 37 bins (3*12+1)
    for out in outputs:
        if out.biomass_by_age is not None:
            n_bins = len(out.biomass_by_age[0])
            assert n_bins <= 5, (
                f"Got {n_bins} age bins — expected ~4 (year-based), "
                f"not ~37 (timestep-based). Duplicate block bug?"
            )
            break
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_simulate.py::test_age_distribution_uses_year_bins -v`
Expected: FAIL — the duplicate block produces ~37 bins (timestep-based) instead of ~4 (year-based).

- [ ] **Step 3: Delete the duplicate block (lines 610–652)**

In `osmose/engine/simulate.py`, delete the entire second block from line 610 (`# Compute age/size distributions when enabled`) through line 652 (the last `abundance_by_size[sp] = abs_`). Keep lines 556–608 intact — they are the correct implementation.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_simulate.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```
git -C /home/razinka/osmose/osmose-python add osmose/engine/simulate.py tests/test_engine_simulate.py
git -C /home/razinka/osmose/osmose-python commit -m "fix: remove duplicate age/size distribution block that overwrote correct results"
```

---

## Task 2: Fix fishing yield double 1e-6 conversion

`state.weight` is already in tonnes (computed as `condition_factor * L^b * 1e-6` in growth.py). Line 552 multiplies by `1e-6` again, making yield 10⁶ too small.

**Files:**
- Modify: `osmose/engine/simulate.py:552`
- Test: `tests/test_engine_simulate.py`

- [ ] **Step 1: Write a test that verifies yield magnitude**

Add to `tests/test_engine_simulate.py`:

```python
def test_fishing_yield_not_double_scaled(minimal_config):
    """Regression: yield was 1e6 too small due to double 1e-6 conversion."""
    from osmose.engine.simulate import _collect_outputs
    from osmose.engine.state import MortalityCause, SchoolState

    cfg = EngineConfig.from_dict(minimal_config)

    # Create a minimal state with known weight and fishing deaths
    state = SchoolState.create(
        n_schools=1,
        species_id=np.array([0], dtype=np.int32),
        age_dt=np.array([12], dtype=np.int32),
        length=np.array([10.0]),
        weight=np.array([0.001]),  # 0.001 tonnes = 1 kg (already in tonnes)
        abundance=np.array([100.0]),
        cell_x=np.array([0], dtype=np.int32),
        cell_y=np.array([0], dtype=np.int32),
        n_mortality_causes=len(MortalityCause),
    )
    state.n_dead[:, int(MortalityCause.FISHING)] = 100.0

    output = _collect_outputs(state, cfg, step=0)
    # Expected yield = 100 fish * 0.001 tonnes = 0.1 tonnes
    total_yield = output.yield_by_species.sum()
    assert total_yield > 0.01, f"Yield {total_yield} is suspiciously small (double 1e-6 bug?)"
    assert abs(total_yield - 0.1) < 1e-10, f"Expected 0.1 tonnes, got {total_yield}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_simulate.py::test_fishing_yield_not_double_scaled -v`
Expected: FAIL — yield will be 1e-7 instead of 0.1.

- [ ] **Step 3: Remove the extra `* 1e-6`**

In `osmose/engine/simulate.py` line 552, change:
```python
fishing_yield = fishing_dead * state.weight * 1e-6  # grams -> tonnes
```
to:
```python
fishing_yield = fishing_dead * state.weight  # weight already in tonnes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_simulate.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```
git -C /home/razinka/osmose/osmose-python add osmose/engine/simulate.py tests/test_engine_simulate.py
git -C /home/razinka/osmose/osmose-python commit -m "fix: remove double 1e-6 conversion on fishing yield"
```

---

## Task 3: Fix duplicate growth_class parsing in config.py

Lines 1077–1083 and 1086–1092 are identical `growth_class` list comprehensions. Delete the duplicate.

**Files:**
- Modify: `osmose/engine/config.py:1085-1092`

- [ ] **Step 1: Delete the duplicate growth_class block**

In `osmose/engine/config.py`, delete lines 1085–1092 (the second `# Growth class dispatch` comment and list comprehension). Keep lines 1077–1083.

- [ ] **Step 2: Run full test suite to verify nothing breaks**

Run: `.venv/bin/python -m pytest -x -q`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```
git -C /home/razinka/osmose/osmose-python add osmose/engine/config.py
git -C /home/razinka/osmose/osmose-python commit -m "fix: remove duplicate growth_class parsing block"
```

---

## Task 4: Narrow all `except Exception` blocks in UI sync/read code

Six locations catch `except Exception` where only `SilentException` and `AttributeError` are expected. This silently drops real errors. Narrow each to specific exceptions and add debug logging for unexpected errors.

**Files:**
- Modify: `ui/pages/results.py:462-466`
- Modify: `ui/pages/setup.py:181-184`
- Modify: `ui/pages/forcing.py:124-127`
- Modify: `ui/pages/calibration_handlers.py:118-122`
- Modify: `ui/pages/grid.py:237-247`
- Modify: `ui/state.py:124-129`

- [ ] **Step 1: Fix results.py ensemble mode (the worst offender)**

In `ui/pages/results.py`, replace lines 462–466:
```python
ensemble_on = False
try:
    ensemble_on = bool(input.ensemble_mode()) and bool(rep_dirs.get())
except Exception:
    pass
```
with:
```python
from shiny.types import SilentException

ensemble_on = False
try:
    ensemble_on = bool(input.ensemble_mode()) and bool(rep_dirs.get())
except (SilentException, AttributeError):
    pass
```

Ensure the `SilentException` import is at the top of the file (or reuse an existing import if present).

- [ ] **Step 2: Fix setup.py input sync**

In `ui/pages/setup.py`, replace lines 181–184:
```python
try:
    val = getattr(input, input_id)()
except Exception:
    continue
```
with:
```python
from shiny.types import SilentException

try:
    val = getattr(input, input_id)()
except (AttributeError, SilentException):
    continue
```

Ensure the import is at the top of the file.

- [ ] **Step 3: Fix forcing.py input sync**

In `ui/pages/forcing.py`, apply the same pattern as step 2 at lines 124–127.

- [ ] **Step 4: Fix calibration_handlers.py input sync**

In `ui/pages/calibration_handlers.py`, replace lines 118–122:
```python
try:
    if val():
        selected.append(p)
except Exception:
    continue
```
with:
```python
try:
    if val():
        selected.append(p)
except (SilentException, AttributeError):
    continue
```

- [ ] **Step 5: Fix grid.py coordinate read**

In `ui/pages/grid.py`, replace lines 244–247:
```python
except Exception:
    # SilentException raised when inputs aren't initialized yet
    ul_lat = ul_lon = lr_lat = lr_lon = 0.0
    nx = ny = 0
```
with:
```python
except (SilentException, AttributeError):
    ul_lat = ul_lon = lr_lat = lr_lon = 0.0
    nx = ny = 0
```

Add `from shiny.types import SilentException` at the top if not already imported.

- [ ] **Step 6: Fix state.py get_theme_mode**

In `ui/state.py`, replace lines 127–129:
```python
except Exception:
    # SilentException when input not initialized, AttributeError/TypeError otherwise
    return "light"
```
with:
```python
except (SilentException, AttributeError, TypeError):
    return "light"
```

Add `from shiny.types import SilentException` at the top if not already imported.

- [ ] **Step 7: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```
git -C /home/razinka/osmose/osmose-python add ui/pages/results.py ui/pages/setup.py ui/pages/forcing.py ui/pages/calibration_handlers.py ui/pages/grid.py ui/state.py
git -C /home/razinka/osmose/osmose-python commit -m "fix: narrow except Exception blocks to specific exceptions in UI sync code"
```

---

## Task 5: Fix silent fishing seasonality discard and movement map load

Two engine locations silently discard data on parse errors. Add warnings so users know their config data was rejected.

**Files:**
- Modify: `osmose/engine/config.py:313-320`
- Modify: `osmose/engine/movement_maps.py:221-228`

- [ ] **Step 1: Add warning for malformed fishing seasonality**

In `osmose/engine/config.py`, replace lines 319–320:
```python
except (ValueError, TypeError):
    pass
```
with:
```python
except (ValueError, TypeError) as exc:
    import warnings
    warnings.warn(
        f"Invalid fisheries.seasonality.fsh{fsh} value: {inline_val!r} — {exc}",
        stacklevel=2,
    )
```

- [ ] **Step 2: Narrow movement map load exception**

In `osmose/engine/movement_maps.py`, replace lines 223–228:
```python
except Exception as exc:
    warnings.warn(
        f"Failed to load movement map file {fp}: {exc}",
        stacklevel=2,
    )
    raw_grids[i] = None
```
with:
```python
except (FileNotFoundError, OSError, ValueError) as exc:
    logger.error("Failed to load movement map file %s: %s", fp, exc)
    raw_grids[i] = None
```

The file already has `logger = logging.getLogger(__name__)` at line 18. Use `logger` (not `_log`). Remove the `warnings` import if it was only used here.

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```
git -C /home/razinka/osmose/osmose-python add osmose/engine/config.py osmose/engine/movement_maps.py
git -C /home/razinka/osmose/osmose-python commit -m "fix: warn on malformed fishing seasonality, narrow movement map exception"
```

---

## Task 6: Fix runner.py returncode masking

`self._process.returncode or 0` maps `None` to `0`, masking a killed process as success.

**Files:**
- Modify: `osmose/runner.py:189`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Fix the returncode expression**

In `osmose/runner.py` line 189, change:
```python
returncode=self._process.returncode or 0,
```
to:
```python
returncode=self._process.returncode if self._process.returncode is not None else -1,
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_runner.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```
git -C /home/razinka/osmose/osmose-python add osmose/runner.py
git -C /home/razinka/osmose/osmose-python commit -m "fix: map None returncode to -1 instead of masking as success"
```

---

## Task 7: Fix temp directory leak in scenario export

`tempfile.mkdtemp()` is never cleaned up. Register with the existing cleanup module.

**Files:**
- Modify: `ui/pages/scenarios.py:231-238`

- [ ] **Step 1: Add deferred cleanup**

In `ui/pages/scenarios.py`, replace lines 231–238:
```python
@render.download(filename="osmose_scenarios.zip")
def export_all_scenarios():
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="osmose_export_"))
    zip_path = tmp_dir / "osmose_scenarios.zip"
    mgr.export_all(zip_path)
    return str(zip_path)
```
with:
```python
@render.download(filename="osmose_scenarios.zip")
def export_all_scenarios():
    import atexit
    import shutil
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="osmose_export_"))
    zip_path = tmp_dir / "osmose_scenarios.zip"
    mgr.export_all(zip_path)
    # Schedule cleanup after Shiny finishes serving the download
    atexit.register(shutil.rmtree, str(tmp_dir), True)
    return str(zip_path)
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_ui_scenarios.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```
git -C /home/razinka/osmose/osmose-python add ui/pages/scenarios.py
git -C /home/razinka/osmose/osmose-python commit -m "fix: register temp dir cleanup for scenario export downloads"
```

---

## Task 8: Fix config reader encoding and log levels

Two issues: (1) `open()` with no `encoding=` fails on Latin-1 configs, (2) unparseable lines logged at DEBUG instead of WARNING.

**Files:**
- Modify: `osmose/config/reader.py:67,80`
- Test: `tests/test_config_reader.py`

- [ ] **Step 1: Write test for accented characters in config**

Add to `tests/test_config_reader.py`:

```python
def test_reader_handles_latin1_characters(tmp_path):
    """Config files with accented species names (Latin-1) should not crash."""
    from osmose.config.reader import OsmoseConfigReader

    config_file = tmp_path / "test.csv"
    config_file.write_bytes("species.name.sp0 ; Sébaste\n".encode("latin-1"))

    reader = OsmoseConfigReader()
    result = reader.read_file(config_file)
    # Should not crash; value may have replacement chars but key must be present
    assert "species.name.sp0" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config_reader.py::test_reader_handles_latin1_characters -v`
Expected: FAIL with `UnicodeDecodeError`

- [ ] **Step 3: Fix encoding and log level**

In `osmose/config/reader.py` line 67, change:
```python
with open(filepath, "r") as f:
```
to:
```python
with open(filepath, "r", encoding="utf-8", errors="replace") as f:
```

In line 80, change:
```python
_log.debug("Skipping unparseable line in %s: %r", filepath.name, line)
```
to:
```python
_log.warning("Skipping unparseable line in %s: %r", filepath.name, line)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_config_reader.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```
git -C /home/razinka/osmose/osmose-python add osmose/config/reader.py tests/test_config_reader.py
git -C /home/razinka/osmose/osmose-python commit -m "fix: use utf-8 encoding with replace fallback, warn on unparseable lines"
```

---

## Task 9: Fix scenario load int() crash and calibration silent midpoint

Two minor silent failures: (1) `int("3.0")` crashes scenario load, (2) unparseable calibration value silently uses midpoint.

**Files:**
- Modify: `ui/pages/scenarios.py:133`
- Modify: `osmose/calibration/configure.py:37-40`
- Modify: `osmose/cleanup.py:66-67`

- [ ] **Step 1: Fix scenario load int conversion**

In `ui/pages/scenarios.py` line 133, change:
```python
n_species = int(loaded.config.get("simulation.nspecies", "3"))
```
to:
```python
try:
    n_species = int(float(loaded.config.get("simulation.nspecies", "3") or "3"))
except (ValueError, TypeError):
    n_species = 3
```

- [ ] **Step 2: Add logging for calibration midpoint fallback**

In `osmose/calibration/configure.py`, replace lines 37–40:
```python
try:
    guess = float(value)
except (ValueError, TypeError):
    guess = (lower + upper) / 2
```
with:
```python
try:
    guess = float(value)
except (ValueError, TypeError):
    import logging
    logging.getLogger(__name__).warning(
        "Config value for %r is not numeric: %r, using midpoint %.3f",
        key, value, (lower + upper) / 2,
    )
    guess = (lower + upper) / 2
```

- [ ] **Step 3: Add debug logging for cleanup failures**

In `osmose/cleanup.py`, replace lines 66–67:
```python
except OSError:
    pass
```
with:
```python
except OSError as exc:
    _log.debug("Could not stat/remove temp dir %s: %s", entry, exc)
```

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```
git -C /home/razinka/osmose/osmose-python add ui/pages/scenarios.py osmose/calibration/configure.py osmose/cleanup.py
git -C /home/razinka/osmose/osmose-python commit -m "fix: robust int parsing in scenario load, log calibration midpoint fallback"
```

---

## Task 10: Add test_engine_accessibility.py

The accessibility module (214 LOC) has **zero** tests despite being on the critical path for predation. Add a focused test file.

**Files:**
- Create: `tests/test_engine_accessibility.py`
- Reference: `osmose/engine/accessibility.py`

- [ ] **Step 1: Write core tests**

Create `tests/test_engine_accessibility.py`. The `AccessibilityMatrix` API is:
- `from_csv(csv_path, species_names)` — class method (semicolon-separated CSV)
- `prey_lookup` / `pred_lookup` — `dict[str, list[StageInfo]]` fields
- `get_index(species_name, age_years, role="prey")` — returns matrix index or -1
- `resolve_name(species_name)` — returns CSV label name or None
- `compute_school_indices(species_id, age_dt, n_dt_per_year, all_species_names, role)` — vectorized

Internal helpers: `_parse_label(label)` → `(name, threshold)` and `_parse_labels(labels)` → dict.

```python
"""Tests for osmose.engine.accessibility."""

import numpy as np
import pytest
from pathlib import Path

from osmose.engine.accessibility import (
    AccessibilityMatrix,
    StageInfo,
    _parse_label,
    _parse_labels,
)


class TestParseLabel:
    """Test the _parse_label helper directly."""

    def test_simple_name(self):
        name, threshold = _parse_label("cod")
        assert name == "cod"
        assert threshold == float("inf")

    def test_name_with_threshold(self):
        name, threshold = _parse_label("cod < 0.45")
        assert name == "cod"
        assert threshold == 0.45

    def test_whitespace_handling(self):
        name, threshold = _parse_label("  hake  < 1.5  ")
        assert name == "hake"
        assert threshold == 1.5


class TestParseLabels:
    """Test _parse_labels grouping and sorting."""

    def test_single_stage_species(self):
        result = _parse_labels(["cod", "hake"])
        assert len(result["cod"]) == 1
        assert result["cod"][0].threshold == float("inf")
        assert result["cod"][0].matrix_index == 0

    def test_two_stage_species(self):
        result = _parse_labels(["cod < 0.45", "cod"])
        stages = result["cod"]
        assert len(stages) == 2
        # Sorted: finite threshold first
        assert stages[0].threshold == 0.45
        assert stages[0].matrix_index == 0  # "cod < 0.45" was at index 0
        assert stages[1].threshold == float("inf")
        assert stages[1].matrix_index == 1  # "cod" was at index 1


class TestFromCsv:
    """Test loading from semicolon-separated CSV files."""

    def test_2x2_matrix_shape(self, tmp_path):
        csv = tmp_path / "access.csv"
        csv.write_text(";cod;hake\ncod;0.8;0.5\nhake;0.3;0.9\n")
        matrix = AccessibilityMatrix.from_csv(csv, species_names=["cod", "hake"])
        assert matrix.raw_matrix.shape == (2, 2)
        assert "cod" in matrix.prey_lookup
        assert "hake" in matrix.prey_lookup

    def test_matrix_values(self, tmp_path):
        csv = tmp_path / "access.csv"
        csv.write_text(";cod;hake\ncod;0.8;0.5\nhake;0.3;0.9\n")
        matrix = AccessibilityMatrix.from_csv(csv, species_names=["cod", "hake"])
        # Row 0 = cod (prey), Col 0 = cod (pred)
        assert matrix.raw_matrix[0, 0] == 0.8
        assert matrix.raw_matrix[1, 0] == 0.3

    def test_staged_labels(self, tmp_path):
        """CSV with threshold labels creates multi-stage lookups."""
        csv = tmp_path / "access.csv"
        csv.write_text(
            ";cod < 0.45;cod\ncod < 0.45;0.1;0.2\ncod;0.3;0.4\n"
        )
        matrix = AccessibilityMatrix.from_csv(csv, species_names=["cod"])
        assert len(matrix.prey_lookup["cod"]) == 2
        assert matrix.prey_lookup["cod"][0].threshold == 0.45


class TestGetIndex:
    """Test get_index for prey/pred role lookups."""

    def test_single_stage_any_age(self, tmp_path):
        csv = tmp_path / "access.csv"
        csv.write_text(";cod\ncod;1.0\n")
        matrix = AccessibilityMatrix.from_csv(csv, species_names=["cod"])
        idx = matrix.get_index("cod", age_years=5.0, role="prey")
        assert idx == 0

    def test_two_stage_young_vs_old(self, tmp_path):
        csv = tmp_path / "access.csv"
        csv.write_text(";cod < 0.45;cod\ncod < 0.45;0.1;0.2\ncod;0.3;0.4\n")
        matrix = AccessibilityMatrix.from_csv(csv, species_names=["cod"])
        young_idx = matrix.get_index("cod", age_years=0.2, role="prey")
        old_idx = matrix.get_index("cod", age_years=1.0, role="prey")
        assert young_idx != old_idx
        # Young (age 0.2 < 0.45) → first stage (index 0)
        assert young_idx == 0
        # Old (age 1.0 >= 0.45) → second stage (index 1)
        assert old_idx == 1

    def test_unknown_species_returns_minus_one(self, tmp_path):
        csv = tmp_path / "access.csv"
        csv.write_text(";cod\ncod;1.0\n")
        matrix = AccessibilityMatrix.from_csv(csv, species_names=["cod"])
        assert matrix.get_index("unknown_fish", age_years=1.0) == -1


class TestResolveName:
    """Test case-insensitive name resolution."""

    def test_case_insensitive(self, tmp_path):
        csv = tmp_path / "access.csv"
        csv.write_text(";Cod\nCod;1.0\n")
        matrix = AccessibilityMatrix.from_csv(csv, species_names=["cod"])
        assert matrix.resolve_name("cod") == "Cod"
        assert matrix.resolve_name("COD") is None  # only lowercase norm is stored

    def test_missing_species(self, tmp_path):
        csv = tmp_path / "access.csv"
        csv.write_text(";cod\ncod;1.0\n")
        matrix = AccessibilityMatrix.from_csv(csv, species_names=["cod"])
        assert matrix.resolve_name("hake") is None


class TestComputeSchoolIndices:
    """Test vectorized index computation."""

    def test_basic_vectorized(self, tmp_path):
        csv = tmp_path / "access.csv"
        csv.write_text(";cod;hake\ncod;0.8;0.5\nhake;0.3;0.9\n")
        matrix = AccessibilityMatrix.from_csv(csv, species_names=["cod", "hake"])
        indices = matrix.compute_school_indices(
            species_id=np.array([0, 1, 0], dtype=np.int32),
            age_dt=np.array([12, 24, 6], dtype=np.int32),
            n_dt_per_year=12,
            all_species_names=["cod", "hake"],
            role="prey",
        )
        assert indices.shape == (3,)
        # All should be valid (not -1) since both species are in the matrix
        assert (indices >= 0).all()
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_accessibility.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```
git -C /home/razinka/osmose/osmose-python add tests/test_engine_accessibility.py
git -C /home/razinka/osmose/osmose-python commit -m "test: add test_engine_accessibility.py for predation accessibility matrix"
```

---

## Summary

| Task | Severity | What it fixes |
|------|----------|---------------|
| 1 | Critical | Distribution output always None (duplicate block) |
| 2 | Critical | Fishing yield 10⁶ too small (double 1e-6) |
| 3 | Critical | Dead duplicate growth_class parsing |
| 4 | High | 6 locations of `except Exception` swallowing real errors |
| 5 | High | Silent data discard in fishing seasonality and movement maps |
| 6 | High | Runner returncode None → 0 masking |
| 7 | High | Temp dir leak on every scenario export |
| 8 | High | Config reader encoding failure + silent line skip |
| 9 | Medium | Scenario load crash + calibration silent midpoint |
| 10 | High | Zero test coverage for accessibility module |

**Not addressed in this plan** (deferred to separate plans due to scope):
- Thread-safety of module globals (`_config_dir`, `_tl_weighted_sum`, `_diet_matrix`) — requires architectural refactor
- Movement schema key format mismatch — requires schema redesign
- Species table `spt_` input IDs not syncing — requires sync mechanism redesign
- `EngineConfig.from_dict()` decomposition — large refactoring effort
- `_bioen_step` vectorization — performance optimization
