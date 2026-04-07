# Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the top 5 highest-impact findings from the deep codebase review — strict mode for OsmoseResults, Numba import warnings, FreeParameter type safety, path traversal guard, and EngineConfig validation.

**Architecture:** Each fix is independent with its own test. Changes are minimal (5-20 LOC each) and surgical — no refactoring, no new abstractions. TDD throughout.

**Tech Stack:** Python 3.12, pytest, NumPy, xarray, dataclasses, enums

---

### Task 1: Add `strict` mode to OsmoseResults

**Files:**
- Modify: `osmose/results.py:15-31` (constructor)
- Modify: `osmose/results.py:261-283` (`_read_species_output`)
- Modify: `osmose/results.py:223-259` (`_read_2d_output`)
- Modify: `osmose/results.py:189-201` (`size_spectrum`)
- Test: `tests/test_results_strict.py`

- [ ] **Step 1: Write failing tests for strict mode**

Create `tests/test_results_strict.py`:

```python
"""Tests for OsmoseResults strict mode."""

from pathlib import Path

import pytest

from osmose.results import OsmoseResults


def test_strict_raises_on_missing_dir(tmp_path: Path) -> None:
    """strict=True raises FileNotFoundError when output_dir does not exist."""
    r = OsmoseResults(tmp_path / "nonexistent", strict=True)
    with pytest.raises(FileNotFoundError, match="Output directory does not exist"):
        r.biomass()


def test_strict_raises_on_no_matching_files(tmp_path: Path) -> None:
    """strict=True raises FileNotFoundError when no files match the pattern."""
    r = OsmoseResults(tmp_path, strict=True)
    with pytest.raises(FileNotFoundError, match="No files matching"):
        r.biomass()


def test_strict_raises_on_no_spectrum_files(tmp_path: Path) -> None:
    """strict=True raises FileNotFoundError for size_spectrum when no files match."""
    r = OsmoseResults(tmp_path, strict=True)
    with pytest.raises(FileNotFoundError, match="No files matching"):
        r.size_spectrum()


def test_strict_raises_on_2d_missing(tmp_path: Path) -> None:
    """strict=True raises FileNotFoundError for 2D output methods."""
    r = OsmoseResults(tmp_path, strict=True)
    with pytest.raises(FileNotFoundError, match="No files matching"):
        r.biomass_by_age()


def test_non_strict_returns_empty_df(tmp_path: Path) -> None:
    """Default (strict=False) still returns empty DataFrame for backwards compat."""
    r = OsmoseResults(tmp_path / "nonexistent")
    df = r.biomass()
    assert df.empty


def test_strict_default_is_false(tmp_path: Path) -> None:
    """Verify strict defaults to False."""
    r = OsmoseResults(tmp_path / "nonexistent")
    assert r.strict is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_results_strict.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'strict'`

- [ ] **Step 3: Add `strict` parameter to OsmoseResults**

In `osmose/results.py`, modify the constructor to add the `strict` parameter:

```python
def __init__(self, output_dir: Path, prefix: str = "osm", strict: bool = False):
    self.output_dir = Path(output_dir)
    self.prefix = prefix
    self.strict = strict
    self._nc_cache: dict[str, xr.Dataset] = {}
```

Add a private helper method after `__init__` to avoid repeating the strict check logic:

```python
def _raise_if_strict(self, pattern: str) -> None:
    """Raise FileNotFoundError in strict mode when no files match."""
    if not self.strict:
        return
    if not self.output_dir.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {self.output_dir}"
        )
    raise FileNotFoundError(
        f"No files matching '{pattern}' in {self.output_dir}"
    )
```

Modify `_read_species_output` — replace the `if not frames:` block:

```python
        if not frames:
            self._raise_if_strict(pattern)
            return pd.DataFrame()
```

Modify `_read_2d_output` — replace the `if not frames:` block:

```python
        if not frames:
            self._raise_if_strict(pattern)
            return pd.DataFrame()
```

Modify `size_spectrum` — replace the `if not frames:` block:

```python
        if not frames:
            self._raise_if_strict(pattern)
            return pd.DataFrame()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_results_strict.py -v`
Expected: 6 passed

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1734+ passed (strict defaults to False, so existing code is unaffected)

- [ ] **Step 6: Commit**

```bash
git add tests/test_results_strict.py osmose/results.py
git commit -m "feat: add strict mode to OsmoseResults to raise on missing data"
```

---

### Task 2: Add Numba import warning

**Files:**
- Modify: `osmose/engine/processes/predation.py:10-25`
- Modify: `osmose/engine/processes/mortality.py:10-40`
- Modify: `osmose/engine/processes/movement.py:1-17`
- Test: `tests/test_numba_warning.py`

**Context:** None of the three files currently import `warnings` at the module level. The `import warnings` must be added to the top-level imports section (near `import numpy as np`), NOT inside the `except` block.

- [ ] **Step 1: Write failing test**

Create `tests/test_numba_warning.py`:

```python
"""Test that missing Numba triggers a warning."""

import importlib
import sys
import warnings


def test_predation_warns_without_numba() -> None:
    """Predation module emits ImportWarning when numba is unavailable."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Force re-import with numba blocked
        mods_to_remove = [k for k in sys.modules if k.startswith("osmose.engine.processes.predation")]
        for m in mods_to_remove:
            del sys.modules[m]
        orig = sys.modules.pop("numba", "NOT_SET")
        sys.modules["numba"] = None  # type: ignore[assignment]
        try:
            importlib.import_module("osmose.engine.processes.predation")
            numba_warnings = [x for x in w if "numba" in str(x.message).lower()]
            assert len(numba_warnings) >= 1, f"Expected numba warning, got: {[str(x.message) for x in w]}"
        finally:
            del sys.modules["numba"]
            if orig != "NOT_SET":
                sys.modules["numba"] = orig
            # Re-import clean version
            mods_to_remove = [k for k in sys.modules if k.startswith("osmose.engine.processes.predation")]
            for m in mods_to_remove:
                del sys.modules[m]


def test_mortality_warns_without_numba() -> None:
    """Mortality module emits ImportWarning when numba is unavailable."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mods_to_remove = [k for k in sys.modules if k.startswith("osmose.engine.processes.mortality")]
        for m in mods_to_remove:
            del sys.modules[m]
        orig = sys.modules.pop("numba", "NOT_SET")
        sys.modules["numba"] = None  # type: ignore[assignment]
        try:
            importlib.import_module("osmose.engine.processes.mortality")
            numba_warnings = [x for x in w if "numba" in str(x.message).lower()]
            assert len(numba_warnings) >= 1, f"Expected numba warning, got: {[str(x.message) for x in w]}"
        finally:
            del sys.modules["numba"]
            if orig != "NOT_SET":
                sys.modules["numba"] = orig
            mods_to_remove = [k for k in sys.modules if k.startswith("osmose.engine.processes.mortality")]
            for m in mods_to_remove:
                del sys.modules[m]


def test_movement_warns_without_numba() -> None:
    """Movement module emits ImportWarning when numba is unavailable."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mods_to_remove = [k for k in sys.modules if k.startswith("osmose.engine.processes.movement")]
        for m in mods_to_remove:
            del sys.modules[m]
        orig = sys.modules.pop("numba", "NOT_SET")
        sys.modules["numba"] = None  # type: ignore[assignment]
        try:
            importlib.import_module("osmose.engine.processes.movement")
            numba_warnings = [x for x in w if "numba" in str(x.message).lower()]
            assert len(numba_warnings) >= 1, f"Expected numba warning, got: {[str(x.message) for x in w]}"
        finally:
            del sys.modules["numba"]
            if orig != "NOT_SET":
                sys.modules["numba"] = orig
            mods_to_remove = [k for k in sys.modules if k.startswith("osmose.engine.processes.movement")]
            for m in mods_to_remove:
                del sys.modules[m]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_numba_warning.py -v`
Expected: FAIL — no warning captured (current code is silent)

- [ ] **Step 3: Add warnings to all three modules**

In `osmose/engine/processes/predation.py`, add `import warnings` after `import numpy as np` (line 12), then replace the try/except block (lines 20-25):

```python
try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    warnings.warn(
        "Numba is not installed. Predation will use pure Python fallback, "
        "which may be 10-100x slower. Install numba for optimal performance.",
        ImportWarning,
        stacklevel=2,
    )
```

In `osmose/engine/processes/mortality.py`, add `import warnings` after `import numpy as np` (line 12), then replace the try/except block (lines 35-40):

```python
try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    warnings.warn(
        "Numba is not installed. Mortality will use pure Python fallback, "
        "which may be 10-100x slower. Install numba for optimal performance.",
        ImportWarning,
        stacklevel=2,
    )
```

In `osmose/engine/processes/movement.py`, add `import warnings` after `import numpy as np` (line 5), then replace the try/except block (lines 13-17):

```python
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    warnings.warn(
        "Numba is not installed. Movement will use pure Python fallback, "
        "which may be 10-100x slower. Install numba for optimal performance.",
        ImportWarning,
        stacklevel=2,
    )
```

**Note:** `movement.py` already has `import warnings` at line 272 (inside a function body). Since `warnings` is now imported at module level, delete line 272 (`import warnings`) entirely — the `warnings.warn()` call on line 274 will use the module-level import.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_numba_warning.py -v`
Expected: 3 passed

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1734+ passed (Numba IS installed in dev env, so no warnings emitted)

- [ ] **Step 6: Commit**

```bash
git add tests/test_numba_warning.py osmose/engine/processes/predation.py osmose/engine/processes/mortality.py osmose/engine/processes/movement.py
git commit -m "feat: warn when Numba is not installed (10-100x performance impact)"
```

---

### Task 3: FreeParameter type safety — Enum transform + bounds validation

**Files:**
- Modify: `osmose/calibration/problem.py:1-8` (add `import enum`)
- Modify: `osmose/calibration/problem.py:30-37` (FreeParameter dataclass)
- Modify: `osmose/calibration/problem.py:117` (`_evaluate_candidate` transform check)
- Modify: `osmose/calibration/__init__.py:9,18` (export `Transform`)
- Modify: `tests/test_calibration_problem.py:11,137` (update existing tests)
- Modify: `tests/test_study_workflows.py:434` (update existing test)
- Test: `tests/test_calibration_types.py`

**Breaking change awareness:** The existing codebase uses `transform="log"` (string) in 3 test files and in `_evaluate_candidate`. ALL of these must be updated to use `Transform.LOG`. The `__init__.py` must also export `Transform` so downstream callers can import it.

- [ ] **Step 1: Write failing tests**

Create `tests/test_calibration_types.py`:

```python
"""Tests for FreeParameter type safety."""

import enum

import pytest

from osmose.calibration.problem import FreeParameter, Transform


def test_transform_is_enum() -> None:
    """Transform should be an enum, not a bare string."""
    assert issubclass(Transform, enum.Enum)
    assert hasattr(Transform, "LINEAR")
    assert hasattr(Transform, "LOG")


def test_default_transform_is_linear() -> None:
    """FreeParameter defaults to LINEAR transform."""
    fp = FreeParameter(key="species.linf.sp0", lower_bound=10.0, upper_bound=100.0)
    assert fp.transform == Transform.LINEAR


def test_invalid_bounds_raises() -> None:
    """lower_bound >= upper_bound should raise ValueError."""
    with pytest.raises(ValueError, match="lower_bound.*must be less than.*upper_bound"):
        FreeParameter(key="species.linf.sp0", lower_bound=100.0, upper_bound=10.0)


def test_equal_bounds_raises() -> None:
    """lower_bound == upper_bound should raise ValueError."""
    with pytest.raises(ValueError, match="lower_bound.*must be less than.*upper_bound"):
        FreeParameter(key="species.linf.sp0", lower_bound=50.0, upper_bound=50.0)


def test_valid_log_transform() -> None:
    """Log transform with valid positive bounds should work."""
    fp = FreeParameter(
        key="species.linf.sp0",
        lower_bound=0.01,
        upper_bound=10.0,
        transform=Transform.LOG,
    )
    assert fp.transform == Transform.LOG
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_types.py -v`
Expected: FAIL — `ImportError: cannot import name 'Transform'`

- [ ] **Step 3: Implement Transform enum and FreeParameter validation**

In `osmose/calibration/problem.py`:

Add `import enum` after `import re` (line 6).

Replace the `FreeParameter` dataclass (lines 30-37):

```python
class Transform(enum.Enum):
    """Parameter space transform for calibration."""

    LINEAR = "linear"
    LOG = "log"


@dataclass
class FreeParameter:
    """A parameter to optimize during calibration."""

    key: str  # OSMOSE parameter key
    lower_bound: float
    upper_bound: float
    transform: Transform = Transform.LINEAR

    def __post_init__(self) -> None:
        if not isinstance(self.transform, Transform):
            raise TypeError(
                f"transform must be a Transform enum, got {type(self.transform).__name__}"
            )
        if self.lower_bound >= self.upper_bound:
            raise ValueError(
                f"lower_bound ({self.lower_bound}) must be less than "
                f"upper_bound ({self.upper_bound})"
            )
```

Update `_evaluate_candidate` (line 117) — change:
```python
            if fp.transform == "log":
```
to:
```python
            if fp.transform == Transform.LOG:
```

- [ ] **Step 4: Update `__init__.py` to export Transform**

In `osmose/calibration/__init__.py`, change line 9:
```python
from osmose.calibration.problem import FreeParameter, OsmoseCalibrationProblem, Transform
```

Add `"Transform"` to `__all__` (line 18):
```python
__all__ = [
    "biomass_rmse",
    "abundance_rmse",
    "diet_distance",
    "normalized_rmse",
    "FreeParameter",
    "Transform",
    "OsmoseCalibrationProblem",
    "SurrogateCalibrator",
    "SensitivityAnalyzer",
]
```

- [ ] **Step 5: Update existing tests that use string transforms**

In `tests/test_calibration_problem.py`:

Add import at top:
```python
from osmose.calibration.problem import FreeParameter, OsmoseCalibrationProblem, Transform
```

Line 11 — change `assert fp.transform == "linear"` to:
```python
    assert fp.transform == Transform.LINEAR
```

Line 137 — change `FreeParameter("species.k.sp0", -2, 0, transform="log")` to:
```python
    params = [FreeParameter("species.k.sp0", -2, 0, transform=Transform.LOG)]
```

In `tests/test_study_workflows.py`:

Line 27 — add `Transform` to the existing import (do NOT add a separate import line):
```python
from osmose.calibration.problem import FreeParameter, OsmoseCalibrationProblem, Transform
```

Line 434 — change `transform="log"` to:
```python
        fp = FreeParameter(key=key, lower_bound=-2, upper_bound=0, transform=Transform.LOG)
```

- [ ] **Step 6: Run all tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_types.py tests/test_calibration_problem.py tests/test_study_workflows.py -v`
Expected: All passed

- [ ] **Step 7: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1734+ passed

- [ ] **Step 8: Commit**

```bash
git add tests/test_calibration_types.py tests/test_calibration_problem.py tests/test_study_workflows.py osmose/calibration/problem.py osmose/calibration/__init__.py
git commit -m "feat: FreeParameter uses Transform enum + bounds validation"
```

---

### Task 4: Path traversal guard in `_resolve_file`

**Files:**
- Modify: `osmose/engine/config.py:1-16` (add logging import)
- Modify: `osmose/engine/config.py:95-106` (`_resolve_file`)
- Test: `tests/test_resolve_file_security.py`

**Context:** `osmose/engine/config.py` does NOT have a logger — it has no `from osmose.logging import setup_logging` or `_log` variable. The `warnings` module IS already imported (line 9). We need to add logging.

- [ ] **Step 1: Write failing tests**

Create `tests/test_resolve_file_security.py`:

```python
"""Tests for _resolve_file path traversal guard."""

from pathlib import Path

from osmose.engine.config import _resolve_file, _set_config_dir


def test_rejects_parent_traversal(tmp_path: Path) -> None:
    """File keys with '..' should be rejected."""
    _set_config_dir(str(tmp_path))
    # Create a nested directory and a file that would be resolved via ..
    inner = tmp_path / "subdir"
    inner.mkdir()
    secret = tmp_path / "secret.csv"
    secret.write_text("data")
    result = _resolve_file("subdir/../secret.csv")
    # Should be rejected because path contains '..'
    assert result is None


def test_rejects_bare_parent_traversal(tmp_path: Path) -> None:
    """Bare '../file' should be rejected."""
    _set_config_dir(str(tmp_path / "subdir"))
    (tmp_path / "subdir").mkdir(exist_ok=True)
    secret = tmp_path / "secret.csv"
    secret.write_text("data")
    result = _resolve_file("../secret.csv")
    assert result is None


def test_rejects_absolute_path_outside_config_dir(tmp_path: Path) -> None:
    """Absolute paths not under any search dir should be rejected."""
    _set_config_dir(str(tmp_path))
    result = _resolve_file("/etc/hosts")
    assert result is None


def test_allows_valid_relative_path(tmp_path: Path) -> None:
    """Valid relative paths within search dirs should resolve normally."""
    _set_config_dir(str(tmp_path))
    data_file = tmp_path / "grid.csv"
    data_file.write_text("1;2;3")
    result = _resolve_file("grid.csv")
    assert result is not None
    assert result.name == "grid.csv"


def test_allows_subdirectory_path(tmp_path: Path) -> None:
    """Paths in subdirectories within config dir should work."""
    _set_config_dir(str(tmp_path))
    subdir = tmp_path / "maps"
    subdir.mkdir()
    data_file = subdir / "movement.csv"
    data_file.write_text("1;2;3")
    result = _resolve_file("maps/movement.csv")
    assert result is not None
    assert result.name == "movement.csv"


def test_empty_key_returns_none() -> None:
    """Empty string should return None (not raise)."""
    result = _resolve_file("")
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_resolve_file_security.py -v`
Expected: `test_rejects_parent_traversal` and `test_rejects_bare_parent_traversal` FAIL (currently resolves `../` paths), `test_rejects_absolute_path_outside_config_dir` FAIL (currently accepts `/etc/hosts` if it exists)

- [ ] **Step 3: Add logging and path containment check**

In `osmose/engine/config.py`, add after the existing imports (after line 15, `from osmose.engine.accessibility import AccessibilityMatrix`):

```python
from osmose.logging import setup_logging

_log = setup_logging("osmose.engine.config")
```

Replace `_resolve_file` (lines 95-106):

```python
def _resolve_file(file_key: str) -> Path | None:
    """Resolve a relative file path against multiple search directories.

    Rejects paths containing '..' segments and absolute paths not under
    a known search directory, to prevent path traversal.
    """
    if not file_key:
        return None
    if ".." in Path(file_key).parts:
        _log.warning("Rejecting file key with '..' traversal: %s", file_key)
        return None
    p = Path(file_key)
    if p.is_absolute():
        for base in _search_dirs():
            try:
                if p.is_relative_to(base.resolve()) and p.exists():
                    return p
            except (ValueError, OSError):
                continue
        _log.warning("Rejecting absolute path not under any search dir: %s", file_key)
        return None
    for base in _search_dirs():
        path = base / file_key
        if path.exists():
            return path
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_resolve_file_security.py -v`
Expected: 6 passed

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1734+ passed (existing configs use valid relative paths, no `..` segments)

- [ ] **Step 6: Commit**

```bash
git add tests/test_resolve_file_security.py osmose/engine/config.py
git commit -m "fix: add path traversal guard to _resolve_file"
```

---

### Task 5: EngineConfig `__post_init__` validation

**Files:**
- Modify: `osmose/engine/config.py:675-677` (add `__post_init__` before `from_dict`)
- Test: `tests/test_engine_config_validation.py`

**Context:** `EngineConfig` is a `@dataclass` with 70+ fields. The `from_dict()` classmethod is the canonical construction path — it builds all arrays correctly. Direct construction via `EngineConfig(...)` is only used in tests. The `__post_init__` must not break `from_dict()`, which always produces consistent arrays. Existing tests in `tests/test_engine_config.py` use `EngineConfig.from_dict(minimal_config)` — these must still pass.

- [ ] **Step 1: Write failing tests**

Create `tests/test_engine_config_validation.py`:

```python
"""Tests for EngineConfig __post_init__ validation."""

import numpy as np
import pytest

from osmose.engine.config import EngineConfig


def _minimal_config(n_species: int = 2, n_background: int = 0, **overrides) -> dict:
    """Build a minimal kwargs dict for EngineConfig with valid defaults.

    This constructs EngineConfig directly (not via from_dict) to test
    __post_init__ validation in isolation.
    """
    n_total = n_species + n_background
    defaults = dict(
        n_species=n_species,
        n_dt_per_year=24,
        n_year=1,
        n_steps=24,
        n_schools=np.array([100] * n_total, dtype=np.int32),
        species_names=[f"sp{i}" for i in range(n_species)],
        n_background=n_background,
        background_file_indices=[],
        all_species_names=[f"sp{i}" for i in range(n_total)],
        linf=np.array([30.0] * n_total),
        k=np.array([0.3] * n_total),
        t0=np.array([-0.5] * n_total),
        egg_size=np.array([0.1] * n_total),
        condition_factor=np.array([0.006] * n_total),
        allometric_power=np.array([3.0] * n_total),
        vb_threshold_age=np.array([0.0] * n_total),
        lifespan_dt=np.array([120] * n_total, dtype=np.int32),
        mortality_subdt=10,
        ingestion_rate=np.array([3.5] * n_total),
        critical_success_rate=np.array([0.57] * n_total),
        delta_lmax_factor=np.array([2.0] * n_total),
        additional_mortality_rate=np.array([0.0] * n_total),
        additional_mortality_by_dt=None,
        additional_mortality_spatial=None,
        sex_ratio=np.array([0.5] * n_total),
        relative_fecundity=np.array([500.0] * n_total),
        maturity_size=np.array([15.0] * n_total),
        seeding_biomass=np.array([1000.0] * n_total),
        seeding_max_step=np.array([120] * n_total, dtype=np.int32),
        larva_mortality_rate=np.array([0.0] * n_total),
        size_ratio_min=np.zeros((n_total, 1)),
        size_ratio_max=np.ones((n_total, 1)),
        feeding_stage_thresholds=[[] for _ in range(n_total)],
        feeding_stage_metric=["size"] * n_total,
        n_feeding_stages=np.array([1] * n_total, dtype=np.int32),
        starvation_rate_max=np.array([3.0] * n_total),
        fishing_enabled=False,
        fishing_rate=np.array([0.0] * n_total),
        fishing_selectivity_l50=np.array([0.0] * n_total),
        fishing_selectivity_a50=np.full(n_total, np.nan),
        fishing_selectivity_type=np.zeros(n_total, dtype=np.int32),
        fishing_selectivity_slope=np.ones(n_total),
        fishing_seasonality=None,
        fishing_rate_by_year=None,
        mpa_zones=None,
        fishing_discard_rate=None,
        accessibility_matrix=None,
        stage_accessibility=None,
        spawning_season=None,
        movement_method=["maps"] * n_total,
        random_walk_range=np.array([3] * n_total, dtype=np.int32),
        out_mortality_rate=np.array([0.0] * n_total),
        maturity_age_dt=np.zeros(n_total, dtype=np.int32),
        lmax=np.array([30.0] * n_total),
        fishing_spatial_maps=[None] * n_total,
        egg_weight_override=None,
        output_cutoff_age=None,
        output_record_frequency=1,
        diet_output_enabled=False,
        output_step0_include=False,
        movement_seed_fixed=False,
        mortality_seed_fixed=False,
        random_distribution_ncell=None,
        growth_class=["VB"] * n_total,
        raw_config={},
    )
    defaults.update(overrides)
    return defaults


def test_valid_config_passes() -> None:
    """A minimal valid config should construct without error."""
    cfg = _minimal_config()
    ec = EngineConfig(**cfg)
    assert ec.n_species == 2


def test_mismatched_array_length_raises() -> None:
    """Per-species arrays with wrong length should raise ValueError."""
    cfg = _minimal_config(n_species=2)
    cfg["linf"] = np.array([30.0, 30.0, 30.0])  # length 3 != n_total=2
    with pytest.raises(ValueError, match="linf.*length 3.*expected 2"):
        EngineConfig(**cfg)


def test_zero_linf_raises() -> None:
    """Zero linf for a focal species should raise ValueError."""
    cfg = _minimal_config(n_species=2)
    cfg["linf"] = np.array([0.0, 30.0])
    with pytest.raises(ValueError, match="linf.*positive"):
        EngineConfig(**cfg)


def test_negative_k_raises() -> None:
    """Negative k should raise ValueError."""
    cfg = _minimal_config(n_species=2)
    cfg["k"] = np.array([-0.3, 0.3])
    with pytest.raises(ValueError, match="k.*positive"):
        EngineConfig(**cfg)


def test_nsteps_consistency_raises() -> None:
    """n_steps != n_dt_per_year * n_year should raise ValueError."""
    cfg = _minimal_config()
    cfg["n_steps"] = 999
    with pytest.raises(ValueError, match="n_steps"):
        EngineConfig(**cfg)


def test_background_species_zero_linf_allowed() -> None:
    """Background species with zero linf should NOT raise (they don't grow)."""
    cfg = _minimal_config(n_species=2, n_background=1)
    # Background species (index 2) has zero linf — this is expected
    cfg["linf"] = np.array([30.0, 30.0, 0.0])
    cfg["k"] = np.array([0.3, 0.3, 0.0])
    ec = EngineConfig(**cfg)
    assert ec.n_background == 1


def test_from_dict_still_works() -> None:
    """Existing from_dict path should still produce valid configs."""
    minimal = {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "20",
        "species.name.sp0": "Anchovy",
        "species.linf.sp0": "15.0",
        "species.k.sp0": "0.4",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
    }
    cfg = EngineConfig.from_dict(minimal)
    assert cfg.n_species == 1
    assert cfg.linf[0] == 15.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_config_validation.py -v`
Expected: `test_valid_config_passes` PASS (no validation yet = no error), but `test_mismatched_array_length_raises`, `test_zero_linf_raises`, `test_negative_k_raises`, `test_nsteps_consistency_raises` all FAIL (no `__post_init__` = no `ValueError` raised)

- [ ] **Step 3: Add `__post_init__` to EngineConfig**

In `osmose/engine/config.py`, add `__post_init__` method inside the EngineConfig class, after the last field with a default value (`output_bioen_sizeinf: bool = False` at line 675) and before the `@classmethod` `from_dict` (line 677):

```python
    def __post_init__(self) -> None:
        """Validate invariants after construction."""
        n_total = self.n_species + self.n_background

        # Check n_steps consistency
        expected_steps = self.n_dt_per_year * self.n_year
        if self.n_steps != expected_steps:
            raise ValueError(
                f"n_steps ({self.n_steps}) != n_dt_per_year ({self.n_dt_per_year}) "
                f"* n_year ({self.n_year}) = {expected_steps}"
            )

        # Check per-species array lengths
        per_species_arrays = {
            "linf": self.linf,
            "k": self.k,
            "t0": self.t0,
            "egg_size": self.egg_size,
            "condition_factor": self.condition_factor,
            "allometric_power": self.allometric_power,
            "lifespan_dt": self.lifespan_dt,
            "ingestion_rate": self.ingestion_rate,
            "critical_success_rate": self.critical_success_rate,
            "additional_mortality_rate": self.additional_mortality_rate,
            "starvation_rate_max": self.starvation_rate_max,
        }
        for name, arr in per_species_arrays.items():
            if hasattr(arr, "__len__") and len(arr) != n_total:
                raise ValueError(
                    f"{name} has length {len(arr)}, expected {n_total} "
                    f"(n_species={self.n_species} + n_background={self.n_background})"
                )

        # Check biological positivity constraints (focal species only)
        for name, arr in [("linf", self.linf), ("k", self.k)]:
            if hasattr(arr, "__len__"):
                for i in range(self.n_species):
                    if arr[i] <= 0:
                        raise ValueError(
                            f"{name}[{i}] = {arr[i]}, must be positive for "
                            f"focal species '{self.species_names[i]}'"
                        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_config_validation.py -v`
Expected: 7 passed

- [ ] **Step 5: Run full test suite (including existing engine config tests)**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1734+ passed — `from_dict()` always produces consistent arrays, so `__post_init__` will not raise for any existing test

- [ ] **Step 6: Commit**

```bash
git add tests/test_engine_config_validation.py osmose/engine/config.py
git commit -m "feat: add EngineConfig __post_init__ validation for array lengths and biological constraints"
```

---

### Task 6: Final verification

**Files:** None (read-only verification)

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`
Expected: 1734+ passed + new tests (6 + 3 + 5 + 6 + 7 = 27 new tests), 0 failures

- [ ] **Step 2: Run linting**

Run: `.venv/bin/ruff check osmose/ tests/`
Expected: 0 errors

- [ ] **Step 3: Run type checking on modified files**

Run: `.venv/bin/pyright osmose/results.py osmose/engine/config.py osmose/calibration/problem.py osmose/calibration/__init__.py osmose/engine/processes/predation.py osmose/engine/processes/mortality.py osmose/engine/processes/movement.py`
Expected: 0 errors

- [ ] **Step 4: Verify git status is clean**

Run: `git -C /home/razinka/osmose/osmose-python status`
Expected: All changes committed, working tree clean
