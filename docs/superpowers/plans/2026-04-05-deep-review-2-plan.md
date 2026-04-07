# Deep Review #2 Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 67 findings from the second 7-agent deep codebase review, organized into 4 phases of increasing risk.

**Architecture:** Each phase is independent and produces a passing test suite. Phase 1 fixes safety/correctness bugs with minimal structural changes. Phase 2 restructures types and helpers. Phase 3 improves performance. Phase 4 fills test gaps and fixes comments.

**Tech Stack:** Python 3.12, NumPy, pytest, dataclasses, Shiny

**Spec:** `docs/superpowers/specs/2026-04-05-deep-codebase-review.md`

**Test command:** `.venv/bin/python -m pytest tests/ -q --tb=short`

**Baseline:** 1766 passed, 14 skipped, 0 failures (commit 9431c75)

---

## File Structure

### Phase 1: Safety & Correctness
- Modify: `osmose/engine/processes/predation.py` — remove module globals, accept context param
- Modify: `osmose/engine/processes/mortality.py` — remove `_tl_weighted_sum` global, accept context param
- Modify: `osmose/engine/simulate.py` — create and pass `SimulationContext`
- Modify: `osmose/engine/config.py` — remove `_config_dir` global, pass explicitly
- Modify: `osmose/engine/processes/oxygen_function.py` — add epsilon guard
- Modify: `osmose/engine/processes/temp_function.py` — add epsilon guard
- Modify: `osmose/calibration/problem.py` — close OsmoseResults, surface failure info
- Modify: `osmose/results.py` — default `strict=True`, add CSV error handling
- Modify: `osmose/demo.py` — fix `_version_tuple` fallback
- Modify: `osmose/engine/processes/__init__.py` — fix stale docstring
- Modify: `ui/pages/results.py` — narrow exception catch
- Create: `tests/test_thread_safety.py`
- Create: `tests/test_numerical_guards.py`

### Phase 2: Structural Improvements
- Create: `osmose/engine/path_resolution.py` — consolidated path resolver
- Modify: `osmose/engine/config.py` — remove duplicated path resolution
- Modify: `osmose/engine/state.py` — add `frozen=True`
- Modify: `osmose/engine/simulate.py` — split `_collect_outputs`, add `frozen=True` to StepOutput
- Modify: `osmose/engine/background.py` — use shared path resolver, add `__post_init__`
- Modify: `osmose/engine/resources.py` — use shared path resolver, add `__post_init__`
- Modify: `osmose/engine/movement_maps.py` — use shared path resolver
- Modify: `osmose/engine/output.py` — fix prefix default
- Modify: `osmose/calibration/multiphase.py` — add `__post_init__` to CalibrationPhase
- Create: `tests/test_path_resolution.py`
- Create: `tests/test_type_invariants.py`

### Phase 3: Performance & Dedup
- Modify: `osmose/engine/processes/mortality.py` — extract shared inner loop helper
- Modify: `osmose/engine/simulate.py` — precompute species masks
- Modify: `osmose/engine/processes/fishing.py` — vectorize spatial maps

### Phase 4: Tests & Polish
- Create: `tests/test_bioen_orchestration.py`
- Create: `tests/test_config_reader_errors.py`
- Create: `tests/test_numerical_edges.py`
- Create: `tests/test_ui_state.py`
- Create: `tests/test_ensemble_edges.py`
- Create: `tests/test_validator_enum.py`
- Modify: 7 files for comment fixes

---

## Phase 1: Safety & Correctness

### Task 1: Add epsilon guards to O2 and temperature functions (C2)

**Files:**
- Modify: `osmose/engine/processes/oxygen_function.py:9-10`
- Modify: `osmose/engine/processes/temp_function.py:25-26`
- Create: `tests/test_numerical_guards.py`

- [ ] **Step 1: Write failing tests for division-by-zero edge cases**

```python
# tests/test_numerical_guards.py
"""Tests for numerical stability guards in bioenergetic functions."""

import numpy as np
import pytest

from osmose.engine.processes.oxygen_function import f_o2
from osmose.engine.processes.temp_function import phi_t, arrhenius


class TestF_O2Guards:
    def test_zero_o2_and_c2(self):
        """f_o2 with o2=0 and c2=0 should return 0, not NaN."""
        result = f_o2(np.array([0.0]), c1=1.0, c2=0.0)
        assert np.isfinite(result).all()
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_negative_o2(self):
        """f_o2 with negative o2 should not produce NaN."""
        result = f_o2(np.array([-1.0]), c1=1.0, c2=0.5)
        assert np.isfinite(result).all()

    def test_normal_values_unchanged(self):
        """Normal values should produce the same result as before."""
        result = f_o2(np.array([5.0]), c1=0.8, c2=2.0)
        expected = 0.8 * 5.0 / (5.0 + 2.0)
        assert result[0] == pytest.approx(expected, rel=1e-12)


class TestPhiTGuards:
    def test_equal_activation_energies(self):
        """phi_t with e_d == e_m should not produce NaN/inf."""
        result = phi_t(np.array([15.0]), e_m=0.5, e_d=0.5, t_p=20.0)
        assert np.isfinite(result).all()

    def test_normal_values_unchanged(self):
        """Normal values should produce the same result as before."""
        result = phi_t(np.array([20.0]), e_m=0.5, e_d=1.2, t_p=20.0)
        assert result[0] == pytest.approx(1.0, rel=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_numerical_guards.py -v`
Expected: `test_zero_o2_and_c2` and `test_equal_activation_energies` FAIL with NaN/inf

- [ ] **Step 3: Add epsilon guard to f_o2**

In `osmose/engine/processes/oxygen_function.py`, replace line 10:

```python
def f_o2(o2: NDArray[np.float64], c1: float, c2: float) -> NDArray[np.float64]:
    """Oxygen dose-response: f_O2 = C1 * O2 / (O2 + C2)."""
    denom = o2 + c2
    # Guard: when both o2 and c2 are zero, dose-response is zero
    return np.where(
        np.abs(denom) < 1e-30,
        0.0,
        c1 * o2 / np.where(np.abs(denom) < 1e-30, 1.0, denom),
    )
```

- [ ] **Step 4: Add epsilon guard to phi_t**

In `osmose/engine/processes/temp_function.py`, replace the `_raw` inner function (lines 24-28):

```python
    def _raw(t):
        num = np.exp(-e_m / (K_B * t))
        delta = e_d - e_m
        if abs(delta) < 1e-30:
            # Degenerate case: no declining phase, return Arrhenius-only
            return num
        ratio = e_m / delta
        denom = 1.0 + ratio * np.exp(e_d / K_B * (1.0 / t_p_k - 1.0 / t))
        return num / denom
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_numerical_guards.py -v`
Expected: All PASS

- [ ] **Step 6: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: 1766+ passed, 0 failures

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/processes/oxygen_function.py osmose/engine/processes/temp_function.py tests/test_numerical_guards.py
git commit -m "fix: add epsilon guards to f_o2 and phi_t to prevent NaN on edge cases (C2)"
```

---

### Task 2: Encapsulate diet/TL/config_dir in SimulationContext (C1)

**Files:**
- Modify: `osmose/engine/processes/predation.py:45-66` — remove globals, add context param
- Modify: `osmose/engine/processes/mortality.py:25-28,59-61` — remove globals, use context
- Modify: `osmose/engine/simulate.py` — create SimulationContext, pass through
- Modify: `osmose/engine/config.py:75-81` — remove `_config_dir` global
- Create: `tests/test_thread_safety.py`

- [ ] **Step 1: Write test for thread safety**

```python
# tests/test_thread_safety.py
"""Tests that simulation state is isolated per-run (no module globals)."""

import numpy as np

from osmose.engine.processes.predation import get_diet_matrix


def test_diet_matrix_not_global():
    """get_diet_matrix without context should return None."""
    matrix = get_diet_matrix()
    assert matrix is None, "Diet matrix should not persist as module-level state"
```

- [ ] **Step 2: Create SimulationContext dataclass in simulate.py**

Add near top of `osmose/engine/simulate.py` (after imports, before `_bioen_step`):

```python
@dataclass
class SimulationContext:
    """Per-simulation mutable state — replaces module-level globals.

    Passed through the call chain instead of using module-level variables,
    making the simulation re-entrant and thread-safe.
    """

    diet_tracking_enabled: bool = False
    diet_matrix: NDArray[np.float64] | None = None
    tl_weighted_sum: NDArray[np.float64] | None = None
    config_dir: str = ""
```

- [ ] **Step 3: Refactor predation.py — replace globals with context parameter**

In `osmose/engine/processes/predation.py`:

1. Remove the 3 module-level globals (`_diet_tracking_enabled`, `_diet_matrix` — lines 45-46).
2. Update `enable_diet_tracking`, `disable_diet_tracking`, `get_diet_matrix` to accept a `ctx` parameter:

```python
def enable_diet_tracking(n_schools: int, n_species: int, ctx: SimulationContext | None = None) -> None:
    """Enable per-school diet tracking on the given context."""
    if ctx is None:
        return
    ctx.diet_tracking_enabled = True
    ctx.diet_matrix = np.zeros((n_schools, n_species), dtype=np.float64)


def disable_diet_tracking(ctx: SimulationContext | None = None) -> None:
    """Disable diet tracking on the given context."""
    if ctx is None:
        return
    ctx.diet_tracking_enabled = False
    ctx.diet_matrix = None


def get_diet_matrix(ctx: SimulationContext | None = None) -> NDArray[np.float64] | None:
    """Return the current diet matrix from context."""
    return ctx.diet_matrix if ctx else None
```

3. Add `ctx: SimulationContext | None = None` to `predation()` signature. Replace internal reads of `_diet_tracking_enabled` with `ctx.diet_tracking_enabled if ctx else False` and `_diet_matrix` with `ctx.diet_matrix`.

- [ ] **Step 4: Refactor mortality.py — replace _tl_weighted_sum global with context**

In `osmose/engine/processes/mortality.py`:

1. Remove `_tl_weighted_sum` global (lines 59-61).
2. Remove imports of `_diet_matrix`, `_diet_tracking_enabled` from predation (lines 25-28).
3. Add `ctx: SimulationContext | None = None` to `mortality()` signature.
4. Replace `global _tl_weighted_sum` with `ctx.tl_weighted_sum = np.zeros(...)`.
5. Pass `ctx` through to inner functions that read diet/TL state.

- [ ] **Step 5: Refactor config.py — remove _config_dir global**

In `osmose/engine/config.py`:

1. Remove `_config_dir` global and `_set_config_dir()` (lines 75-81).
2. Change `_search_dirs()` to accept `config_dir: str` parameter:

```python
def _search_dirs(config_dir: str = "") -> list[Path]:
    dirs: list[Path] = []
    if config_dir:
        dirs.append(Path(config_dir))
    dirs.append(Path("."))
    dirs.append(Path("data/examples"))
    dirs += [Path(d) for d in _glob.glob("data/*/")]
    return dirs
```

3. Change `_resolve_file()` to accept `config_dir: str` and pass to `_search_dirs(config_dir)`.
4. Update `from_dict()` to pass `config_dir` explicitly.

- [ ] **Step 6: Update simulate.py — create and pass context**

In `simulate()`:

1. Create context: `ctx = SimulationContext(config_dir=cfg.get("_osmose.config.dir", ""))`.
2. Pass `ctx=ctx` to `mortality()` and `predation()` calls.

- [ ] **Step 7: Run full test suite, fix test breakage**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: Fix tests that relied on old global API (likely `test_engine_diet.py`, `test_engine_predation_helpers.py`). Update them to pass a `SimulationContext` instance.

- [ ] **Step 8: Commit**

```bash
git add osmose/engine/processes/predation.py osmose/engine/processes/mortality.py \
       osmose/engine/simulate.py osmose/engine/config.py tests/test_thread_safety.py
git commit -m "refactor: replace module-level mutable globals with SimulationContext (C1)

Encapsulates _diet_matrix, _tl_weighted_sum, and _config_dir in a
per-simulation SimulationContext dataclass passed through the call chain.
Makes the engine re-entrant and thread-safe for parallel calibration."
```

---

### Task 3: Close OsmoseResults in calibration and handle malformed CSVs (C3, C5, M17)

**Files:**
- Modify: `osmose/calibration/problem.py:180-187`
- Modify: `osmose/results.py:57-61`

- [ ] **Step 1: Fix _run_single to use context manager**

In `osmose/calibration/problem.py`, replace lines 180-187:

```python
    # Compute objectives
    from osmose.results import OsmoseResults

    with OsmoseResults(output_dir, strict=False) as results:
        obj_values = []
        for fn in self.objective_fns:
            obj_values.append(fn(results))

    return obj_values
```

- [ ] **Step 2: Add malformed CSV resilience to read_csv**

In `osmose/results.py`, replace `read_csv` method (lines 53-61):

```python
def read_csv(self, pattern: str) -> dict[str, pd.DataFrame]:
    """Read CSV output files matching a glob pattern."""
    result = {}
    for f in sorted(self.output_dir.glob(pattern)):
        try:
            result[f.stem] = pd.read_csv(f)
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            _log.warning("Skipping malformed CSV %s: %s", f.name, exc)
    return result
```

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: 1766+ passed

- [ ] **Step 4: Commit**

```bash
git add osmose/calibration/problem.py osmose/results.py
git commit -m "fix: close OsmoseResults in calibration and handle malformed CSVs (C3, C5, M17)"
```

---

### Task 4: Fix silent failure patterns (C4, H7, H8)

**Files:**
- Modify: `osmose/results.py:23` — change strict default
- Modify: `osmose/demo.py:56-61` — fix version fallback
- Modify: `ui/pages/results.py:344-353` — narrow exception types

- [ ] **Step 1: Change OsmoseResults strict default to True**

In `osmose/results.py` line 23:
```python
def __init__(self, output_dir: Path, prefix: str = "osm", strict: bool = True):
```

- [ ] **Step 2: Add explicit strict=False where empty results are acceptable**

Search all callers of `OsmoseResults(...)` and add `strict=False` where needed:
- `osmose/calibration/problem.py` — already `strict=False` from Task 3
- `osmose/ensemble.py` `aggregate_replicates` — add `strict=False`
- `ui/pages/results.py` `_do_load_results` — add `strict=False`
- Any test fixtures that construct OsmoseResults — add `strict=False`

- [ ] **Step 3: Fix _version_tuple fallback direction**

In `osmose/demo.py` lines 56-61, replace:

```python
def _version_tuple(v: str) -> tuple[int, ...]:
    """Parse version string to tuple for comparison."""
    try:
        return tuple(int(x) for x in v.split("."))
    except (ValueError, AttributeError):
        _log.warning("Could not parse version %r; applying all migrations", v)
        return (0,)
```

- [ ] **Step 4: Narrow exception types in _do_load_results**

In `ui/pages/results.py`, replace lines 344-353:

```python
        except (
            OSError,
            ValueError,
            pd.errors.ParserError,
        ) as exc:
            _log.error("Failed to load results: %s", exc, exc_info=True)
            ui.notification_show(f"Error loading results: {exc}", type="error", duration=10)
```

- [ ] **Step 5: Run full test suite, fix any strict-mode breakage**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: Some tests may need `strict=False` added to OsmoseResults constructors. Fix each.

- [ ] **Step 6: Commit**

```bash
git add osmose/results.py osmose/demo.py ui/pages/results.py osmose/ensemble.py
git commit -m "fix: default strict=True for OsmoseResults, fix version fallback, narrow UI catches (C4, H7, H8)"
```

---

### Task 5: Fix stale docstring and output prefix mismatch (C6, H13)

**Files:**
- Modify: `osmose/engine/processes/__init__.py`
- Modify: `osmose/engine/output.py:24`

- [ ] **Step 1: Fix stale docstring**

Replace entire content of `osmose/engine/processes/__init__.py`:

```python
"""OSMOSE simulation process functions.

Each process is a pure function: (state, config, ...) -> state.
"""
```

- [ ] **Step 2: Fix output prefix to match reader default**

In `osmose/engine/output.py` line 24, change default:

```python
def write_outputs(
    outputs: list[StepOutput],
    output_dir: Path,
    config: EngineConfig,
    prefix: str = "osm",
) -> None:
```

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: 1766+ passed (some output tests may need prefix update)

- [ ] **Step 4: Commit**

```bash
git add osmose/engine/processes/__init__.py osmose/engine/output.py
git commit -m "fix: remove stale Phase 1 docstring and align output prefix to 'osm' (C6, H13)"
```

---

## Phase 2: Structural Improvements

### Task 6: Consolidate path resolution into shared module (H6)

**Files:**
- Create: `osmose/engine/path_resolution.py`
- Modify: `osmose/engine/config.py:84-122`
- Modify: `osmose/engine/background.py:23-47`
- Modify: `osmose/engine/resources.py:167-187`
- Modify: `osmose/engine/movement_maps.py:20-44`
- Create: `tests/test_path_resolution.py`

- [ ] **Step 1: Write tests for shared path resolver**

```python
# tests/test_path_resolution.py
"""Tests for the consolidated path resolution module."""

from pathlib import Path
import pytest

from osmose.engine.path_resolution import resolve_data_path


def test_absolute_path_under_config_dir(tmp_path):
    data_file = tmp_path / "forcing.csv"
    data_file.write_text("x")
    result = resolve_data_path(str(data_file), config_dir=str(tmp_path))
    assert result == data_file


def test_relative_path_found_in_config_dir(tmp_path):
    data_file = tmp_path / "maps" / "map0.csv"
    data_file.parent.mkdir()
    data_file.write_text("x")
    result = resolve_data_path("maps/map0.csv", config_dir=str(tmp_path))
    assert result == data_file


def test_path_traversal_rejected(tmp_path):
    result = resolve_data_path("../../etc/passwd", config_dir=str(tmp_path))
    assert result is None


def test_empty_key_returns_none():
    result = resolve_data_path("", config_dir="/tmp")
    assert result is None


def test_not_found_returns_none(tmp_path):
    result = resolve_data_path("nonexistent.csv", config_dir=str(tmp_path))
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_path_resolution.py -v`
Expected: ImportError (module doesn't exist yet)

- [ ] **Step 3: Implement shared path resolver**

```python
# osmose/engine/path_resolution.py
"""Consolidated file path resolution for OSMOSE data files.

All engine modules that resolve relative data file paths (config, background,
resources, movement maps) use this single implementation. Security: rejects
'..' traversal and absolute paths outside known search directories.
"""

from __future__ import annotations

import glob as _glob
from pathlib import Path

from osmose.logging import setup_logging

_log = setup_logging("osmose.path")


def resolve_data_path(
    file_key: str,
    config_dir: str = "",
) -> Path | None:
    """Resolve a relative file path against standard search directories.

    Search order:
      1. As-is (works for absolute paths under config_dir)
      2. Relative to config_dir
      3. Relative to CWD
      4. Relative to data/examples/
      5. Relative to data/*/

    Returns None if the file is not found or the path is rejected.
    Rejects paths containing '..' segments to prevent directory traversal.
    """
    if not file_key:
        return None

    p = Path(file_key)

    if ".." in p.parts:
        _log.warning("Rejecting file key with '..' traversal: %s", file_key)
        return None

    if p.is_absolute():
        search = _build_search_dirs(config_dir)
        for base in search:
            try:
                if p.is_relative_to(base.resolve()) and p.exists():
                    return p
            except (ValueError, OSError):
                continue
        _log.warning("Rejecting absolute path not under any search dir: %s", file_key)
        return None

    for base in _build_search_dirs(config_dir):
        candidate = base / file_key
        if candidate.exists():
            return candidate

    return None


def _build_search_dirs(config_dir: str = "") -> list[Path]:
    """Build ordered list of directories to search for data files."""
    dirs: list[Path] = []
    if config_dir:
        dirs.append(Path(config_dir))
    dirs.append(Path("."))
    dirs.append(Path("data/examples"))
    dirs.extend(Path(d) for d in sorted(_glob.glob("data/*/")))
    return dirs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_path_resolution.py -v`
Expected: All PASS

- [ ] **Step 5: Replace the 4 duplicated implementations**

In each file, replace the local resolver with `from osmose.engine.path_resolution import resolve_data_path`:

- `osmose/engine/config.py` — remove `_search_dirs()` (lines 84-94) and `_resolve_file()` (lines 97-122); update callers to `resolve_data_path(file_key, config_dir=config_dir)`
- `osmose/engine/background.py` — remove `_resolve_path()` (lines 23-47); update callers
- `osmose/engine/resources.py` — remove `_resolve_data_file()` (lines 167-187); update callers
- `osmose/engine/movement_maps.py` — remove `_resolve_path()` (lines 20-44); update callers

- [ ] **Step 6: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: 1766+ passed

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/path_resolution.py tests/test_path_resolution.py \
       osmose/engine/config.py osmose/engine/background.py \
       osmose/engine/resources.py osmose/engine/movement_maps.py
git commit -m "refactor: consolidate 4 duplicated path resolvers into shared module (H6)"
```

---

### Task 7: Add __post_init__ validation to data types (H10)

**Files:**
- Modify: `osmose/engine/config.py:168` — MPAZone
- Modify: `osmose/engine/resources.py:19-30` — ResourceSpeciesInfo
- Modify: `osmose/engine/background.py:60-114` — BackgroundSpeciesInfo
- Modify: `osmose/calibration/multiphase.py:17` — CalibrationPhase
- Create: `tests/test_type_invariants.py`

- [ ] **Step 1: Write tests for type validation**

```python
# tests/test_type_invariants.py
"""Tests that data types reject invalid construction arguments."""

import numpy as np
import pytest


def test_mpa_zone_percentage_bounds():
    from osmose.engine.config import MPAZone
    with pytest.raises(ValueError, match="percentage"):
        MPAZone(percentage=1.5, start_year=0, end_year=10, grid=np.ones((2, 2)))


def test_mpa_zone_year_order():
    from osmose.engine.config import MPAZone
    with pytest.raises(ValueError, match="start_year"):
        MPAZone(percentage=0.5, start_year=10, end_year=5, grid=np.ones((2, 2)))


def test_resource_species_size_order():
    from osmose.engine.resources import ResourceSpeciesInfo
    with pytest.raises(ValueError, match="size_min"):
        ResourceSpeciesInfo(name="test", size_min=10.0, size_max=5.0,
                           trophic_level=2.0, accessibility=0.5)


def test_resource_species_accessibility_bounds():
    from osmose.engine.resources import ResourceSpeciesInfo
    with pytest.raises(ValueError, match="accessibility"):
        ResourceSpeciesInfo(name="test", size_min=1.0, size_max=10.0,
                           trophic_level=2.0, accessibility=1.5)


def test_background_species_proportion_sum():
    from osmose.engine.background import BackgroundSpeciesInfo
    with pytest.raises(ValueError, match="proportions"):
        BackgroundSpeciesInfo(
            name="test", species_index=0, file_index=0, n_class=2,
            lengths=[5.0, 10.0], trophic_levels=[2.0, 2.5],
            ages_dt=[0, 12], condition_factor=0.01, allometric_power=3.0,
            size_ratio_min=[0.5], size_ratio_max=[2.0],
            ingestion_rate=3.5, forcing_nsteps_year=24,
            proportions=[0.3, 0.3],  # sums to 0.6, not ~1.0
        )


def test_background_species_length_count_mismatch():
    from osmose.engine.background import BackgroundSpeciesInfo
    with pytest.raises(ValueError, match="n_class"):
        BackgroundSpeciesInfo(
            name="test", species_index=0, file_index=0, n_class=3,
            lengths=[5.0, 10.0], trophic_levels=[2.0, 2.5],  # only 2, not 3
            ages_dt=[0, 12], condition_factor=0.01, allometric_power=3.0,
            size_ratio_min=[0.5], size_ratio_max=[2.0],
            ingestion_rate=3.5, forcing_nsteps_year=24,
            proportions=[0.5, 0.5],
        )


def test_calibration_phase_empty_params():
    from osmose.calibration.multiphase import CalibrationPhase
    with pytest.raises(ValueError, match="free_params"):
        CalibrationPhase(free_params=[], algorithm="differential_evolution", max_iter=100)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_type_invariants.py -v`
Expected: All FAIL (no validation exists yet)

- [ ] **Step 3: Add __post_init__ to MPAZone**

In `osmose/engine/config.py`, after MPAZone dataclass fields:

```python
    def __post_init__(self) -> None:
        if not (0.0 <= self.percentage <= 1.0):
            raise ValueError(f"MPAZone percentage must be in [0, 1], got {self.percentage}")
        if self.start_year > self.end_year:
            raise ValueError(f"MPAZone start_year ({self.start_year}) > end_year ({self.end_year})")
```

- [ ] **Step 4: Add __post_init__ to ResourceSpeciesInfo**

In `osmose/engine/resources.py`:

```python
    def __post_init__(self) -> None:
        if self.size_min >= self.size_max:
            raise ValueError(f"size_min ({self.size_min}) must be < size_max ({self.size_max})")
        if not (0.0 <= self.accessibility <= 0.99):
            raise ValueError(f"accessibility must be in [0, 0.99], got {self.accessibility}")
        if self.trophic_level <= 0:
            raise ValueError(f"trophic_level must be > 0, got {self.trophic_level}")
```

- [ ] **Step 5: Add __post_init__ to BackgroundSpeciesInfo**

In `osmose/engine/background.py`:

```python
    def __post_init__(self) -> None:
        if len(self.lengths) != self.n_class:
            raise ValueError(f"n_class ({self.n_class}) != len(lengths) ({len(self.lengths)})")
        if len(self.trophic_levels) != self.n_class:
            raise ValueError(f"n_class ({self.n_class}) != len(trophic_levels) ({len(self.trophic_levels)})")
        if len(self.proportions) != self.n_class:
            raise ValueError(f"n_class ({self.n_class}) != len(proportions) ({len(self.proportions)})")
        if abs(sum(self.proportions) - 1.0) > 0.01:
            raise ValueError(f"proportions must sum to ~1.0, got {sum(self.proportions):.4f}")
```

- [ ] **Step 6: Add __post_init__ to CalibrationPhase**

In `osmose/calibration/multiphase.py`:

```python
    def __post_init__(self) -> None:
        if not self.free_params:
            raise ValueError("CalibrationPhase.free_params must not be empty")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
```

- [ ] **Step 7: Run full test suite, fix any fixture breakage**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: Some test fixtures may construct types with invalid values. Fix those fixtures.

- [ ] **Step 8: Commit**

```bash
git add osmose/engine/config.py osmose/engine/resources.py osmose/engine/background.py \
       osmose/calibration/multiphase.py tests/test_type_invariants.py
git commit -m "feat: add __post_init__ validation to MPAZone, ResourceSpeciesInfo, BackgroundSpeciesInfo, CalibrationPhase (H10)"
```

---

### Task 8: Make SchoolState frozen and StepOutput frozen (H9)

**Files:**
- Modify: `osmose/engine/state.py:30`
- Modify: `osmose/engine/simulate.py:26`

- [ ] **Step 1: Add frozen=True to SchoolState**

In `osmose/engine/state.py` line 30:

```python
@dataclass(frozen=True)
class SchoolState:
```

- [ ] **Step 2: Add frozen=True to StepOutput**

In `osmose/engine/simulate.py`:

```python
@dataclass(frozen=True)
class StepOutput:
```

- [ ] **Step 3: Run full test suite, fix mutation errors**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: Any test/code that does `state.field = value` will fail. Change those to `state.replace(field=value)`. Check for `FrozenInstanceError`.

- [ ] **Step 4: Commit**

```bash
git add osmose/engine/state.py osmose/engine/simulate.py
git commit -m "refactor: make SchoolState and StepOutput frozen to enforce immutable-replacement pattern (H9)"
```

---

### Task 9: Split _collect_outputs into focused functions (H5)

**Files:**
- Modify: `osmose/engine/simulate.py:525-681`

- [ ] **Step 1: Extract species-mean helper**

Add before `_collect_outputs` in `osmose/engine/simulate.py`:

```python
def _species_mean(
    values: NDArray[np.float64],
    species_id: NDArray[np.int32],
    n_species: int,
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Compute per-species mean of values for schools matching mask."""
    sums = np.zeros(n_species, dtype=np.float64)
    counts = np.zeros(n_species, dtype=np.float64)
    np.add.at(sums, species_id[mask], values[mask])
    np.add.at(counts, species_id[mask], 1)
    safe = np.where(counts > 0, counts, 1)
    return sums / safe
```

- [ ] **Step 2: Split _collect_outputs into sub-functions**

Extract into:
- `_collect_biomass_abundance(state, config, bkg_output)` → `(biomass, abundance)` (lines 532-548)
- `_collect_mortality(state, config)` → `mortality_by_cause` (lines 550-560)
- `_collect_yield(state, config)` → `yield_by_species` (lines 562-568)
- `_collect_distributions(state, config)` → `(biomass_by_age, abundance_by_age, biomass_by_size, abundance_by_size)` (lines 570-622)
- `_collect_bioen(state, config)` → bioen arrays using `_species_mean` (lines 624-664)

Then `_collect_outputs` becomes a thin coordinator calling these 5 functions.

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: 1766+ passed

- [ ] **Step 4: Commit**

```bash
git add osmose/engine/simulate.py
git commit -m "refactor: split _collect_outputs into focused sub-functions with shared species-mean helper (H5)"
```

---

## Phase 3: Performance & Dedup

### Task 10: Extract shared Numba mortality inner loop (H2)

**Files:**
- Modify: `osmose/engine/processes/mortality.py:854-1288`

The cause-dispatch inner loop body (cause == 0/1/2/3) is identically duplicated in 3 Numba functions:
- `_mortality_in_cell_numba` (lines 905-979)
- `_mortality_all_cells_numba` (lines 1057-1130)
- `_mortality_all_cells_parallel` (lines 1215-1288)

- [ ] **Step 1: Extract cause-dispatch into a shared @njit helper**

Create a `_apply_single_cause` Numba function containing the shared logic:

```python
@njit(cache=True)
def _apply_single_cause(
    cause, idx, inst_abd, n_dead,
    eff_starv, eff_additional, eff_fishing, fishing_discard,
    # predation params for cause==0 ...
) -> None:
    """Apply one mortality cause to one school. Single source of truth."""
    if cause == 1:  # STARVATION
        D = eff_starv[idx]
        if D > 0:
            abd = inst_abd[idx]
            if abd > 0:
                dead = abd * (1.0 - np.exp(-D))
                n_dead[idx, 1] += dead
                inst_abd[idx] -= dead
    elif cause == 2:  # ADDITIONAL
        D = eff_additional[idx]
        if D > 0:
            abd = inst_abd[idx]
            if abd > 0:
                dead = abd * (1.0 - np.exp(-D))
                n_dead[idx, 2] += dead
                inst_abd[idx] -= dead
    elif cause == 3:  # FISHING
        F = eff_fishing[idx]
        if F > 0:
            abd = inst_abd[idx]
            if abd > 0:
                dead = abd * (1.0 - np.exp(-F))
                discard_r = fishing_discard[idx]
                if discard_r > 0:
                    n_dead[idx, 3] += dead * (1.0 - discard_r)
                    n_dead[idx, 6] += dead * discard_r
                else:
                    n_dead[idx, 3] += dead
                inst_abd[idx] -= dead
```

Note: Predation (cause==0) calls `_apply_predation_numba` which has many parameters. It may need to remain inline or be called separately. The non-predation causes (1, 2, 3) are the simpler duplications to extract.

- [ ] **Step 2: Update all 3 functions to use the shared helper**

Replace the duplicated `elif cause == 1/2/3` blocks in each function with calls to `_apply_single_cause(...)`.

- [ ] **Step 3: Remove SYNC comments**

Remove all `# SYNC: Inner loop logic duplicated` comments.

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: 1766+ passed. Key tests: `test_engine_mortality_loop.py`, `test_mortality_rng.py`, `test_engine_parity.py`.

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/mortality.py
git commit -m "refactor: extract shared Numba mortality cause-dispatch, eliminating 3x duplication (H2)"
```

---

### Task 11: Precompute species masks in _bioen_step (M1)

**Files:**
- Modify: `osmose/engine/simulate.py:126-381`

- [ ] **Step 1: Precompute masks at the top of _bioen_step**

After the `_BIOEN_REQUIRED` check (around line 169), add:

```python
    # Precompute species masks once (reused by 6 loops below)
    sp_masks: list[tuple[int, NDArray[np.bool_]]] = [
        (sp, state.species_id == sp)
        for sp in range(config.n_species)
    ]
    sp_masks = [(sp, m) for sp, m in sp_masks if m.any()]
```

- [ ] **Step 2: Replace all 6 per-species loops**

Change each `for sp in range(config.n_species):` / `mask = state.species_id == sp` / `if not mask.any(): continue` to:

```python
    for sp, mask in sp_masks:
```

Loops at lines: 177, 201, 275, 312, 335, 354.

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: 1766+ passed

- [ ] **Step 4: Commit**

```bash
git add osmose/engine/simulate.py
git commit -m "perf: precompute species masks once in _bioen_step instead of 6 times (M1)"
```

---

### Task 12: Vectorize fishing spatial maps and MPA (H12)

**Files:**
- Modify: `osmose/engine/processes/fishing.py:87-120`

- [ ] **Step 1: Vectorize the spatial map lookup**

Replace per-school loop (lines 87-104) with per-species vectorized lookup:

```python
    spatial_factor = np.ones(len(state), dtype=np.float64)
    cy = state.cell_y.astype(np.intp)
    cx = state.cell_x.astype(np.intp)
    for sp_idx in range(len(config.fishing_spatial_maps)):
        sp_map = config.fishing_spatial_maps[sp_idx]
        if sp_map is None:
            continue
        sp_mask = sp == sp_idx
        if not sp_mask.any():
            continue
        sy, sx = cy[sp_mask], cx[sp_mask]
        valid = (sy >= 0) & (sy < sp_map.shape[0]) & (sx >= 0) & (sx < sp_map.shape[1])
        vals = np.zeros(sp_mask.sum(), dtype=np.float64)
        vals[valid] = sp_map[sy[valid], sx[valid]]
        vals[(vals <= 0) | np.isnan(vals)] = 0.0
        spatial_factor[sp_mask] = vals
```

- [ ] **Step 2: Vectorize the MPA loop**

Replace per-school MPA loop (lines 106-120):

```python
    mpa_factor = np.ones(len(state), dtype=np.float64)
    if config.mpa_zones is not None:
        year = step // config.n_dt_per_year
        for mpa in config.mpa_zones:
            if not (mpa.start_year <= year < mpa.end_year):
                continue
            valid = (cy >= 0) & (cy < mpa.grid.shape[0]) & (cx >= 0) & (cx < mpa.grid.shape[1])
            in_mpa = np.zeros(len(state), dtype=bool)
            in_mpa[valid] = mpa.grid[cy[valid], cx[valid]] > 0
            mpa_factor[in_mpa] *= 1.0 - mpa.percentage
```

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: 1766+ passed

- [ ] **Step 4: Commit**

```bash
git add osmose/engine/processes/fishing.py
git commit -m "perf: vectorize fishing spatial map and MPA lookups (H12)"
```

---

## Phase 4: Tests & Polish

### Task 13: Add bioenergetics orchestration tests (T1)

**Files:**
- Create: `tests/test_bioen_orchestration.py`

- [ ] **Step 1: Write orchestration tests**

```python
# tests/test_bioen_orchestration.py
"""Tests for _bioen_step orchestration within the simulation loop."""

import numpy as np
import pytest

from osmose.engine.simulate import _bioen_step
from osmose.engine.state import SchoolState
from osmose.engine.physical_data import PhysicalData


def test_bioen_step_constant_temperature(bioen_config, make_state):
    """_bioen_step with constant temperature should update weight and length."""
    state = make_state(n=2, species_ids=[0, 1], preyed_biomass=[0.5, 1.0])
    temp_data = PhysicalData.from_constant(15.0)
    result = _bioen_step(state, bioen_config, temp_data, step=0)
    assert not np.array_equal(result.weight, state.weight)
    assert np.all(np.isfinite(result.weight))
    assert np.all(np.isfinite(result.length))


def test_bioen_step_no_temperature(bioen_config, make_state):
    """_bioen_step with no temperature data should use 15C fallback."""
    state = make_state(n=2, species_ids=[0, 1], preyed_biomass=[0.5, 1.0])
    result = _bioen_step(state, bioen_config, temp_data=None, step=0)
    assert np.all(np.isfinite(result.weight))


def test_bioen_step_all_outputs_finite(bioen_config, make_state):
    """All bioenergetic state outputs should be finite."""
    state = make_state(n=3, species_ids=[0, 0, 1], preyed_biomass=[0.1, 0.5, 1.0])
    temp_data = PhysicalData.from_constant(15.0)
    result = _bioen_step(state, bioen_config, temp_data, step=0)
    assert np.all(np.isfinite(result.e_net))
    assert np.all(np.isfinite(result.e_gross))
    assert np.all(np.isfinite(result.e_maint))
    assert np.all(np.isfinite(result.rho))
    assert np.all(np.isfinite(result.gonad_weight))
```

Note: `bioen_config` and `make_state` fixtures need to be created in `conftest.py` or locally. Use the existing `make_engine_config` pattern from existing tests.

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_bioen_orchestration.py -v`
Expected: PASS (adjust fixtures as needed)

- [ ] **Step 3: Commit**

```bash
git add tests/test_bioen_orchestration.py
git commit -m "test: add _bioen_step orchestration tests for temperature branches and edge cases (T1)"
```

---

### Task 14: Add config reader error path tests (T2)

**Files:**
- Create: `tests/test_config_reader_errors.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_config_reader_errors.py
"""Tests for OsmoseConfigReader error handling paths."""

import pytest
from pathlib import Path

from osmose.config.reader import OsmoseConfigReader


def test_circular_config_reference(tmp_path):
    """Circular sub-config references should not infinite-loop."""
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("osmose.configuration.b ; b.csv\nfoo ; bar\n")
    b.write_text("osmose.configuration.a ; a.csv\nbaz ; qux\n")
    reader = OsmoseConfigReader()
    result = reader.read(a)
    assert "foo" in result
    assert "baz" in result


def test_path_escape_blocked(tmp_path):
    """Sub-config paths outside config dir should be skipped."""
    main = tmp_path / "main.csv"
    main.write_text("osmose.configuration.evil ; ../../../etc/passwd\n")
    reader = OsmoseConfigReader()
    result = reader.read(main)
    assert "_osmose.config.dir" in result


def test_oversized_file_rejected(tmp_path):
    """Files over 10MB should raise ValueError."""
    big = tmp_path / "big.csv"
    big.write_bytes(b"x" * 10_000_001)
    reader = OsmoseConfigReader()
    with pytest.raises(ValueError, match="too large"):
        reader.read_file(big)


def test_missing_subconfig_warns(tmp_path, caplog):
    """Referenced sub-config that doesn't exist should log warning."""
    main = tmp_path / "main.csv"
    main.write_text("osmose.configuration.missing ; nonexistent.csv\nfoo ; bar\n")
    reader = OsmoseConfigReader()
    result = reader.read(main)
    assert "foo" in result
    assert any("not found" in r.message for r in caplog.records)


def test_unparseable_line_skipped(tmp_path):
    """Lines without separators should be skipped with warning."""
    cfg = tmp_path / "test.csv"
    cfg.write_text("valid_key ; valid_value\nthis_has_no_separator\n")
    reader = OsmoseConfigReader()
    result = reader.read_file(cfg)
    assert "valid_key" in result
    assert reader.skipped_lines == 1
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_config_reader_errors.py -v`
Expected: All PASS (testing existing error handling)

- [ ] **Step 3: Commit**

```bash
git add tests/test_config_reader_errors.py
git commit -m "test: add config reader error path tests for circular refs, path escape, file size (T2)"
```

---

### Task 15: Add numerical edge case and validator ENUM tests (T3, T6)

**Files:**
- Create: `tests/test_numerical_edges.py`
- Create: `tests/test_validator_enum.py`
- Modify: `osmose/config/validator.py:62-76` — add ENUM to validate_field

- [ ] **Step 1: Write numerical edge case tests**

```python
# tests/test_numerical_edges.py
"""Tests for numerical edge cases in growth and predation."""

import numpy as np
import pytest

from osmose.engine.processes.growth import expected_length_vb
from osmose.engine.processes.predation import compute_size_overlap


class TestGrowthEdgeCases:
    def test_vb_zero_k(self):
        """k=0 should produce finite results."""
        result = expected_length_vb(
            age_years=np.array([1.0, 5.0, 10.0]),
            linf=50.0, k=0.0, t0=-0.5, egg_size=0.5,
        )
        assert np.all(np.isfinite(result))

    def test_vb_positive_t0(self):
        """Positive t0 should not produce negative lengths."""
        result = expected_length_vb(
            age_years=np.array([0.1, 0.5]),
            linf=50.0, k=0.3, t0=2.0, egg_size=0.5,
        )
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)


class TestPredationEdgeCases:
    def test_size_overlap_zero_prey_length(self):
        """Zero-length prey should not cause division errors."""
        result = compute_size_overlap(
            pred_length=10.0, prey_length=0.0,
            ratio_min=0.5, ratio_max=2.0,
        )
        assert isinstance(result, (bool, np.bool_))
```

- [ ] **Step 2: Write validator ENUM tests**

```python
# tests/test_validator_enum.py
"""Tests for ENUM validation in config validator."""

import pytest

from osmose.config.validator import validate_config, validate_field
from osmose.schema.base import OsmoseField, ParamType
from osmose.schema.registry import ParameterRegistry


def test_validate_config_rejects_invalid_enum():
    """Invalid enum values should produce validation errors."""
    field = OsmoseField(
        key_pattern="movement.type",
        param_type=ParamType.ENUM,
        description="Movement type",
        choices=["random", "maps"],
    )
    registry = ParameterRegistry()
    registry.register(field)
    errors = validate_config({"movement.type": "invalid_value"}, registry)
    assert any("invalid" in e.lower() or "Invalid" in e for e in errors)


def test_validate_field_handles_enum():
    """validate_field should check enum values."""
    field = OsmoseField(
        key_pattern="growth.type",
        param_type=ParamType.ENUM,
        description="Growth model",
        choices=["VB", "Gompertz"],
    )
    result = validate_field("growth.type", "Linear", field)
    assert result is not None  # should return error string
```

- [ ] **Step 3: Add ENUM handling to validate_field**

In `osmose/config/validator.py`, add after the BOOL branch (line 75):

```python
    elif field.param_type == ParamType.ENUM:
        if field.choices and value not in field.choices:
            return f"Expected one of {field.choices}, got '{value}'"
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_numerical_edges.py tests/test_validator_enum.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_numerical_edges.py tests/test_validator_enum.py osmose/config/validator.py
git commit -m "test: add numerical edge case and ENUM validation tests, fix validate_field ENUM gap (T3, T6)"
```

---

### Task 16: Add UI state and ensemble edge case tests (T7, T8)

**Files:**
- Create: `tests/test_ui_state.py`
- Create: `tests/test_ensemble_edges.py`

- [ ] **Step 1: Write UI state tests**

```python
# tests/test_ui_state.py
"""Tests for AppState and sync_inputs."""

import pytest
from ui.state import AppState


def test_app_state_initial_values():
    """AppState should initialize with empty config."""
    state = AppState()
    assert state.config.get() == {}
    assert state.output_dir.get() is None
    assert state.dirty.get() is False


def test_update_config_marks_dirty():
    """Updating a config key should set dirty=True."""
    state = AppState()
    state.config.set({"foo": "bar"})
    state.dirty.set(False)
    state.update_config("foo", "baz")
    assert state.dirty.get() is True


def test_update_config_no_change_stays_clean():
    """Setting same value should not mark dirty."""
    state = AppState()
    state.config.set({"foo": "bar"})
    state.dirty.set(False)
    state.update_config("foo", "bar")
    assert state.dirty.get() is False
```

- [ ] **Step 2: Write ensemble edge case tests**

```python
# tests/test_ensemble_edges.py
"""Tests for ensemble aggregation edge cases."""

from pathlib import Path

from osmose.ensemble import aggregate_replicates


def test_empty_rep_dirs():
    """Empty replicate list should return empty result."""
    result = aggregate_replicates([], "biomass")
    assert result["time"] == []


def test_unsupported_output_type():
    """Unsupported output type should return empty result."""
    result = aggregate_replicates([Path("/tmp/fake")], "nonexistent_type")
    assert result["time"] == []
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_ui_state.py tests/test_ensemble_edges.py -v`
Expected: PASS (may need Shiny reactive context; if so, skip UI tests with `pytest.importorskip("shiny")`)

- [ ] **Step 4: Commit**

```bash
git add tests/test_ui_state.py tests/test_ensemble_edges.py
git commit -m "test: add UI state and ensemble edge case tests (T7, T8)"
```

---

### Task 17: Fix remaining comment issues (M19-M25)

**Files:**
- Modify: `osmose/engine/config.py:418`
- Modify: `osmose/engine/simulate.py:262`
- Modify: `osmose/engine/rng.py:8-13`
- Modify: `osmose/engine/processes/temp_function.py:9`
- Modify: `osmose/calibration/problem.py:170`
- Modify: `osmose/demo.py:197-199`
- Modify: `osmose/ensemble.py:45-46`

- [ ] **Step 1: Fix all 7 comment issues**

1. **config.py:418** — `"# Take the max discard rate"` → `"# Take the first nonzero discard rate (assumes one primary fishery per species)"`

2. **simulate.py:262** — `"# No temperature data: default to 15°C (neutral for most species)"` → `"# No temperature data: use 15°C as fallback (mid-range assumption; may bias tropical/polar species)"`

3. **rng.py:8-13** — Expand docstring:
```python
"""Build per-species RNG generators.

When fixed=False: all species share a single Generator
    (non-deterministic per-species ordering).
When fixed=True: each species gets a reproducible independent Generator
    via SeedSequence (species ordering has no effect on results).
"""
```

4. **temp_function.py:9** — `K_B = 8.62e-5  # Boltzmann constant in eV/K` → `K_B = 8.62e-5  # Boltzmann constant (eV/K), used in Arrhenius and Johnson thermal curves`

5. **problem.py:170** — Add: `# 1-hour timeout per evaluation; consider making configurable for long simulations`

6. **demo.py** docstring — Add: `"Note: migration chain currently covers up to v4.3.0; versions above have no key renames."`

7. **ensemble.py** docstring — Add: `"Uses empirical percentiles (non-parametric), unlike analysis.ensemble_stats which uses parametric CI."`

- [ ] **Step 2: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: 1766+ passed

- [ ] **Step 3: Commit**

```bash
git add osmose/engine/config.py osmose/engine/simulate.py osmose/engine/rng.py \
       osmose/engine/processes/temp_function.py osmose/calibration/problem.py \
       osmose/demo.py osmose/ensemble.py
git commit -m "docs: fix misleading comments across 7 files (M19-M25)"
```

---

## Final Gate

After all 17 tasks:

- [ ] **Run full test suite:** `.venv/bin/python -m pytest tests/ -q --tb=short`
- [ ] **Run lint:** `.venv/bin/python -m ruff check osmose/ ui/ scripts/ app.py`
- [ ] **Run parity test:** `.venv/bin/python -m pytest tests/test_engine_parity.py -v`
- [ ] **Verify test count:** Should be ~1800+ (baseline 1766 + new tests)

Expected: All pass, lint clean, parity maintained.
