# Python Engine Phase 1: Foundation — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundation layer for the Python OSMOSE engine — Engine protocol, SchoolState, Grid, EngineConfig, ResourceState, and simulation loop skeleton with stub processes.

**Architecture:** Structure-of-Arrays (SoA) design where all school data lives in flat NumPy arrays. An `Engine` protocol provides a common interface for Java and Python engines. The simulation loop follows Java's `SimulationStep.step()` ordering with stub process functions.

**Tech Stack:** Python 3.12+, NumPy, xarray, netCDF4. Tests with pytest.

**Spec:** `docs/superpowers/specs/2026-03-18-python-engine-design.md`

---

## File Structure

```
osmose/engine/                  # NEW package — Python simulation engine
    __init__.py                 # Engine protocol + PythonEngine + JavaEngine wrapper
    state.py                    # SchoolState dataclass (SoA)
    config.py                   # EngineConfig (typed params extracted from flat config dict)
    grid.py                     # Grid class (NetCDF loading, cell count, land mask)
    resources.py                # ResourceState placeholder (LTL forcing)
    simulate.py                 # Main simulation loop + stub process functions
osmose/runner.py                # MODIFY — re-export RunResult for engine protocol
tests/test_engine_state.py      # NEW — SchoolState tests
tests/test_engine_config.py     # NEW — EngineConfig tests
tests/test_engine_grid.py       # NEW — Engine Grid tests
tests/test_engine_simulate.py   # NEW — Simulation loop skeleton tests
```

---

### Task 1: Create the `osmose/engine/` package with Engine protocol

**Files:**
- Create: `osmose/engine/__init__.py`
- Reference: `osmose/runner.py:40-47` (existing `RunResult`)

- [ ] **Step 1: Write failing test for Engine protocol**

Create `tests/test_engine_state.py`:

```python
"""Tests for the engine protocol and state foundation."""

from pathlib import Path

import pytest

from osmose.engine import Engine, JavaEngine, PythonEngine
from osmose.runner import RunResult


def test_python_engine_satisfies_protocol():
    """PythonEngine must have run() and run_ensemble() methods."""
    engine = PythonEngine()
    assert hasattr(engine, "run")
    assert hasattr(engine, "run_ensemble")


def test_java_engine_satisfies_protocol():
    """JavaEngine must have run() and run_ensemble() methods."""
    engine = JavaEngine(jar_path=Path("/fake.jar"))
    assert hasattr(engine, "run")
    assert hasattr(engine, "run_ensemble")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_state.py::test_python_engine_satisfies_protocol -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'osmose.engine'`

- [ ] **Step 3: Create `osmose/engine/__init__.py` with Engine protocol**

```python
"""Python OSMOSE simulation engine.

Provides a common Engine protocol for both Java (subprocess) and
Python (in-process vectorized) backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from osmose.runner import RunResult


@runtime_checkable
class Engine(Protocol):
    """Common interface for Java and Python OSMOSE engines."""

    def run(
        self, config: dict[str, str], output_dir: Path, seed: int = 0
    ) -> RunResult: ...

    def run_ensemble(
        self,
        config: dict[str, str],
        output_dir: Path,
        n: int,
        base_seed: int = 0,
    ) -> list[RunResult]: ...


class PythonEngine:
    """Vectorized in-process OSMOSE simulation engine."""

    def __init__(self, backend: str = "numpy") -> None:
        self.backend = backend

    def run(
        self, config: dict[str, str], output_dir: Path, seed: int = 0
    ) -> RunResult:
        raise NotImplementedError("Phase 1 stub — simulation not yet implemented")

    def run_ensemble(
        self,
        config: dict[str, str],
        output_dir: Path,
        n: int,
        base_seed: int = 0,
    ) -> list[RunResult]:
        return [self.run(config, output_dir, seed=base_seed + i) for i in range(n)]


class JavaEngine:
    """Wrapper around existing OsmoseRunner for Engine protocol compatibility."""

    def __init__(self, jar_path: Path, java_cmd: str = "java") -> None:
        self.jar_path = jar_path
        self.java_cmd = java_cmd

    def run(
        self, config: dict[str, str], output_dir: Path, seed: int = 0
    ) -> RunResult:
        raise NotImplementedError("JavaEngine.run() requires config file path — use OsmoseRunner directly for now")

    def run_ensemble(
        self,
        config: dict[str, str],
        output_dir: Path,
        n: int,
        base_seed: int = 0,
    ) -> list[RunResult]:
        return [self.run(config, output_dir, seed=base_seed + i) for i in range(n)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_state.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/__init__.py tests/test_engine_state.py
git commit -m "feat(engine): add Engine protocol with PythonEngine and JavaEngine stubs"
```

---

### Task 2: SchoolState dataclass

**Files:**
- Create: `osmose/engine/state.py`
- Modify: `tests/test_engine_state.py` (append tests)

- [ ] **Step 1: Write failing tests for SchoolState**

Append to `tests/test_engine_state.py`:

```python
import numpy as np

from osmose.engine.state import MortalityCause, SchoolState


class TestSchoolState:
    """Tests for SchoolState creation, replace, append, and compact."""

    def _make_state(self, n: int = 5) -> SchoolState:
        """Helper: create a minimal SchoolState with n schools."""
        return SchoolState.create(
            n_schools=n,
            species_id=np.arange(n, dtype=np.int32) % 3,
        )

    def test_create_default(self):
        state = self._make_state(5)
        assert len(state) == 5
        assert state.species_id.shape == (5,)
        assert state.abundance.shape == (5,)
        assert state.n_dead.shape == (5, len(MortalityCause))

    def test_create_sets_species_id(self):
        state = self._make_state(3)
        np.testing.assert_array_equal(state.species_id, [0, 1, 2])

    def test_replace_returns_new_state(self):
        state = self._make_state(3)
        new_abundance = np.array([100.0, 200.0, 300.0])
        new_state = state.replace(abundance=new_abundance)
        np.testing.assert_array_equal(new_state.abundance, [100.0, 200.0, 300.0])
        # Original unchanged
        np.testing.assert_array_equal(state.abundance, np.zeros(3))

    def test_append_adds_schools(self):
        state = self._make_state(3)
        extra = self._make_state(2)
        merged = state.append(extra)
        assert len(merged) == 5

    def test_compact_removes_dead(self):
        state = self._make_state(5)
        state = state.replace(abundance=np.array([100.0, 0.0, 50.0, 0.0, 25.0]))
        compacted = state.compact()
        assert len(compacted) == 3
        np.testing.assert_array_equal(compacted.abundance, [100.0, 50.0, 25.0])

    def test_mortality_cause_enum(self):
        assert MortalityCause.PREDATION.value == 0
        assert MortalityCause.AGING.value == 7
        assert len(MortalityCause) == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_state.py::TestSchoolState -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'osmose.engine.state'`

- [ ] **Step 3: Implement SchoolState in `osmose/engine/state.py`**

```python
"""SchoolState: Structure-of-Arrays representation of all fish schools.

All school data is stored in flat NumPy arrays for vectorized operations.
This replaces Java's per-object School instances with cache-friendly
columnar storage.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class MortalityCause(IntEnum):
    """Mortality cause indices for the n_dead tracking array."""

    PREDATION = 0
    STARVATION = 1
    ADDITIONAL = 2
    FISHING = 3
    OUT = 4
    FORAGING = 5
    DISCARDS = 6
    AGING = 7


@dataclass
class SchoolState:
    """Structure-of-Arrays state for all fish schools.

    Every field is a 1D NumPy array of length n_schools, except n_dead
    which is (n_schools, N_MORTALITY_CAUSES).
    """

    # Identity
    species_id: NDArray[np.int32]
    is_background: NDArray[np.bool_]

    # Demographics
    abundance: NDArray[np.float64]
    biomass: NDArray[np.float64]
    length: NDArray[np.float64]
    length_start: NDArray[np.float64]
    weight: NDArray[np.float64]
    age_dt: NDArray[np.int32]
    trophic_level: NDArray[np.float64]

    # Spatial
    cell_x: NDArray[np.int32]
    cell_y: NDArray[np.int32]
    is_out: NDArray[np.bool_]

    # Feeding / predation
    pred_success_rate: NDArray[np.float64]
    preyed_biomass: NDArray[np.float64]
    feeding_stage: NDArray[np.int32]

    # Reproduction
    gonad_weight: NDArray[np.float64]

    # Mortality tracking
    starvation_rate: NDArray[np.float64]
    n_dead: NDArray[np.float64]  # shape (n_schools, len(MortalityCause))

    # Egg state
    is_egg: NDArray[np.bool_]
    first_feeding_age_dt: NDArray[np.int32]

    def __len__(self) -> int:
        return len(self.species_id)

    @classmethod
    def create(
        cls,
        n_schools: int,
        species_id: NDArray[np.int32] | None = None,
    ) -> SchoolState:
        """Create a SchoolState with all arrays zeroed.

        Args:
            n_schools: Number of schools to allocate.
            species_id: Species index per school. Defaults to all zeros.
        """
        n = n_schools
        n_causes = len(MortalityCause)
        return cls(
            species_id=species_id if species_id is not None else np.zeros(n, dtype=np.int32),
            is_background=np.zeros(n, dtype=np.bool_),
            abundance=np.zeros(n, dtype=np.float64),
            biomass=np.zeros(n, dtype=np.float64),
            length=np.zeros(n, dtype=np.float64),
            length_start=np.zeros(n, dtype=np.float64),
            weight=np.zeros(n, dtype=np.float64),
            age_dt=np.zeros(n, dtype=np.int32),
            trophic_level=np.zeros(n, dtype=np.float64),
            cell_x=np.zeros(n, dtype=np.int32),
            cell_y=np.zeros(n, dtype=np.int32),
            is_out=np.zeros(n, dtype=np.bool_),
            pred_success_rate=np.zeros(n, dtype=np.float64),
            preyed_biomass=np.zeros(n, dtype=np.float64),
            feeding_stage=np.zeros(n, dtype=np.int32),
            gonad_weight=np.zeros(n, dtype=np.float64),
            starvation_rate=np.zeros(n, dtype=np.float64),
            n_dead=np.zeros((n, n_causes), dtype=np.float64),
            is_egg=np.zeros(n, dtype=np.bool_),
            first_feeding_age_dt=np.zeros(n, dtype=np.int32),
        )

    def replace(self, **kwargs: NDArray) -> SchoolState:
        """Return a new SchoolState with specified fields replaced.

        Unspecified fields are shallow-copied from self.
        """
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values.update(kwargs)
        return SchoolState(**values)

    def append(self, other: SchoolState) -> SchoolState:
        """Concatenate another SchoolState onto this one."""
        merged = {}
        for f in fields(self):
            a = getattr(self, f.name)
            b = getattr(other, f.name)
            merged[f.name] = np.concatenate([a, b], axis=0)
        return SchoolState(**merged)

    def compact(self) -> SchoolState:
        """Remove dead schools (abundance <= 0)."""
        alive = self.abundance > 0
        compacted = {}
        for f in fields(self):
            arr = getattr(self, f.name)
            compacted[f.name] = arr[alive] if arr.ndim == 1 else arr[alive, :]
        return SchoolState(**compacted)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_state.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/state.py tests/test_engine_state.py
git commit -m "feat(engine): add SchoolState SoA dataclass with create/replace/append/compact"
```

---

### Task 3: EngineConfig

**Files:**
- Create: `osmose/engine/config.py`
- Create: `tests/test_engine_config.py`

- [ ] **Step 1: Write failing tests for EngineConfig**

Create `tests/test_engine_config.py`:

```python
"""Tests for EngineConfig — typed parameter extraction from flat config dicts."""

import pytest

from osmose.engine.config import EngineConfig


@pytest.fixture
def minimal_config() -> dict[str, str]:
    """Minimal config dict for 2 species, sufficient for EngineConfig extraction."""
    return {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "10",
        "simulation.nspecies": "2",
        "simulation.nschool.sp0": "20",
        "simulation.nschool.sp1": "15",
        "species.name.sp0": "Anchovy",
        "species.name.sp1": "Sardine",
        "species.linf.sp0": "15.0",
        "species.linf.sp1": "25.0",
        "species.k.sp0": "0.4",
        "species.k.sp1": "0.3",
        "species.t0.sp0": "-0.1",
        "species.t0.sp1": "-0.2",
        "species.egg.size.sp0": "0.1",
        "species.egg.size.sp1": "0.15",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.condition.factor.sp1": "0.008",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.length2weight.allometric.power.sp1": "3.1",
        "species.lifespan.sp0": "3",
        "species.lifespan.sp1": "5",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "species.vonbertalanffy.threshold.age.sp1": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.ingestion.rate.max.sp1": "3.0",
        "predation.efficiency.critical.sp0": "0.57",
        "predation.efficiency.critical.sp1": "0.57",
    }


class TestEngineConfig:
    def test_from_dict_basic(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.n_species == 2
        assert cfg.n_dt_per_year == 24
        assert cfg.n_year == 10
        assert cfg.n_steps == 240

    def test_species_names(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.species_names == ["Anchovy", "Sardine"]

    def test_growth_params_arrays(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.linf[0] == pytest.approx(15.0)
        assert cfg.linf[1] == pytest.approx(25.0)
        assert cfg.k[0] == pytest.approx(0.4)
        assert cfg.t0[1] == pytest.approx(-0.2)

    def test_lifespan_in_dt(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        # lifespan_dt = lifespan_years * n_dt_per_year
        assert cfg.lifespan_dt[0] == 3 * 24
        assert cfg.lifespan_dt[1] == 5 * 24

    def test_mortality_subdt(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.mortality_subdt == 10

    def test_missing_required_key_raises(self):
        with pytest.raises(KeyError):
            EngineConfig.from_dict({})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_config.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `osmose/engine/config.py`**

```python
"""EngineConfig: typed parameter extraction from flat OSMOSE config dicts.

Converts the flat string key-value config (as read by OsmoseConfigReader)
into typed NumPy arrays indexed by species, ready for vectorized computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _get(cfg: dict[str, str], key: str) -> str:
    """Get a config value, raising KeyError with a clear message."""
    val = cfg.get(key)
    if val is None:
        raise KeyError(f"Required OSMOSE config key missing: {key!r}")
    return val


def _species_float(cfg: dict[str, str], pattern: str, n: int) -> NDArray[np.float64]:
    """Extract a per-species float array from config.

    Args:
        cfg: Flat config dict.
        pattern: Key pattern with {i} placeholder, e.g. "species.linf.sp{i}".
        n: Number of species.
    """
    return np.array([float(_get(cfg, pattern.format(i=i))) for i in range(n)])


def _species_int(cfg: dict[str, str], pattern: str, n: int) -> NDArray[np.int32]:
    """Extract a per-species int array from config."""
    return np.array([int(_get(cfg, pattern.format(i=i))) for i in range(n)], dtype=np.int32)


def _species_str(cfg: dict[str, str], pattern: str, n: int) -> list[str]:
    """Extract a per-species string list from config."""
    return [_get(cfg, pattern.format(i=i)) for i in range(n)]


@dataclass
class EngineConfig:
    """Typed engine configuration extracted from a flat OSMOSE config dict.

    All per-species parameters are stored as NumPy arrays indexed by species ID.
    """

    # Simulation dimensions
    n_species: int
    n_dt_per_year: int
    n_year: int
    n_steps: int
    n_schools: NDArray[np.int32]       # per species
    species_names: list[str]

    # Growth (Von Bertalanffy)
    linf: NDArray[np.float64]          # L_inf per species
    k: NDArray[np.float64]             # growth rate K per species
    t0: NDArray[np.float64]            # t0 per species
    egg_size: NDArray[np.float64]      # L_egg per species
    condition_factor: NDArray[np.float64]   # c in W = c * L^b
    allometric_power: NDArray[np.float64]   # b in W = c * L^b
    vb_threshold_age: NDArray[np.float64]   # age below which linear growth applies
    lifespan_dt: NDArray[np.int32]          # lifespan in time steps

    # Mortality
    mortality_subdt: int

    # Predation
    ingestion_rate: NDArray[np.float64]     # max ingestion rate per species
    critical_success_rate: NDArray[np.float64]  # predation efficiency critical threshold

    @classmethod
    def from_dict(cls, cfg: dict[str, str]) -> EngineConfig:
        """Extract typed engine parameters from a flat OSMOSE config dict.

        Args:
            cfg: Flat config dict as returned by OsmoseConfigReader.read().
        """
        n_sp = int(_get(cfg, "simulation.nspecies"))
        n_dt = int(_get(cfg, "simulation.time.ndtperyear"))
        n_yr = int(_get(cfg, "simulation.time.nyear"))

        lifespan_years = _species_float(cfg, "species.lifespan.sp{i}", n_sp)

        return cls(
            n_species=n_sp,
            n_dt_per_year=n_dt,
            n_year=n_yr,
            n_steps=n_dt * n_yr,
            n_schools=_species_int(cfg, "simulation.nschool.sp{i}", n_sp),
            species_names=_species_str(cfg, "species.name.sp{i}", n_sp),
            linf=_species_float(cfg, "species.linf.sp{i}", n_sp),
            k=_species_float(cfg, "species.k.sp{i}", n_sp),
            t0=_species_float(cfg, "species.t0.sp{i}", n_sp),
            egg_size=_species_float(cfg, "species.egg.size.sp{i}", n_sp),
            condition_factor=_species_float(
                cfg, "species.length2weight.condition.factor.sp{i}", n_sp
            ),
            allometric_power=_species_float(
                cfg, "species.length2weight.allometric.power.sp{i}", n_sp
            ),
            vb_threshold_age=_species_float(
                cfg, "species.vonbertalanffy.threshold.age.sp{i}", n_sp
            ),
            lifespan_dt=(lifespan_years * n_dt).astype(np.int32),
            mortality_subdt=int(cfg.get("mortality.subdt", "10")),
            ingestion_rate=_species_float(
                cfg, "predation.ingestion.rate.max.sp{i}", n_sp
            ),
            critical_success_rate=_species_float(
                cfg, "predation.efficiency.critical.sp{i}", n_sp
            ),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_config.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/config.py tests/test_engine_config.py
git commit -m "feat(engine): add EngineConfig for typed parameter extraction from flat config"
```

---

### Task 4: Engine Grid class

**Files:**
- Create: `osmose/engine/grid.py`
- Create: `tests/test_engine_grid.py`
- Reference: `osmose/grid.py` (existing grid utilities — different purpose, no conflict)

- [ ] **Step 1: Write failing tests for engine Grid**

Create `tests/test_engine_grid.py`:

```python
"""Tests for the engine Grid class — spatial grid for simulation."""

import numpy as np
import pytest
import xarray as xr

from osmose.engine.grid import Grid


@pytest.fixture
def simple_grid_ds(tmp_path):
    """Create a simple 4x5 NetCDF grid file and return its path."""
    ny, nx = 4, 5
    lat = np.linspace(43.0, 48.0, ny)
    lon = np.linspace(-5.0, 0.0, nx)
    mask = np.ones((ny, nx), dtype=np.float32)
    mask[0, 0] = -1  # one land cell
    ds = xr.Dataset(
        {"mask": (["latitude", "longitude"], mask)},
        coords={"latitude": lat, "longitude": lon},
    )
    path = tmp_path / "grid.nc"
    ds.to_netcdf(path)
    return path


class TestGrid:
    def test_from_netcdf(self, simple_grid_ds):
        grid = Grid.from_netcdf(simple_grid_ds)
        assert grid.ny == 4
        assert grid.nx == 5
        assert grid.n_cells == 20

    def test_ocean_mask(self, simple_grid_ds):
        grid = Grid.from_netcdf(simple_grid_ds)
        assert grid.ocean_mask.shape == (4, 5)
        assert grid.ocean_mask[0, 0] is np.bool_(False)  # land
        assert grid.ocean_mask[1, 1] is np.bool_(True)   # ocean

    def test_n_ocean_cells(self, simple_grid_ds):
        grid = Grid.from_netcdf(simple_grid_ds)
        assert grid.n_ocean_cells == 19  # 20 - 1 land

    def test_from_dimensions(self):
        grid = Grid.from_dimensions(ny=10, nx=8)
        assert grid.ny == 10
        assert grid.nx == 8
        assert grid.n_cells == 80
        assert grid.n_ocean_cells == 80  # all ocean by default

    def test_cell_to_coords(self, simple_grid_ds):
        grid = Grid.from_netcdf(simple_grid_ds)
        y, x = grid.cell_to_yx(0)
        assert y == 0 and x == 0
        y, x = grid.cell_to_yx(6)
        assert y == 1 and x == 1

    def test_yx_to_cell(self, simple_grid_ds):
        grid = Grid.from_netcdf(simple_grid_ds)
        assert grid.yx_to_cell(0, 0) == 0
        assert grid.yx_to_cell(1, 1) == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_grid.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `osmose/engine/grid.py`**

```python
"""Grid: spatial grid representation for the engine simulation.

Loads grid topology from NetCDF or creates simple rectangular grids.
Provides cell indexing and ocean/land masking.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray


class Grid:
    """2D spatial grid for OSMOSE simulation.

    Cells are indexed in row-major order: cell_id = y * nx + x.
    """

    def __init__(
        self,
        ny: int,
        nx: int,
        ocean_mask: NDArray[np.bool_],
        lat: NDArray[np.float64] | None = None,
        lon: NDArray[np.float64] | None = None,
    ) -> None:
        self.ny = ny
        self.nx = nx
        self.ocean_mask = ocean_mask
        self.lat = lat
        self.lon = lon

    @property
    def n_cells(self) -> int:
        return self.ny * self.nx

    @property
    def n_ocean_cells(self) -> int:
        return int(self.ocean_mask.sum())

    def cell_to_yx(self, cell_id: int) -> tuple[int, int]:
        """Convert flat cell ID to (y, x) grid coordinates."""
        return divmod(cell_id, self.nx)

    def yx_to_cell(self, y: int, x: int) -> int:
        """Convert (y, x) grid coordinates to flat cell ID."""
        return y * self.nx + x

    @classmethod
    def from_netcdf(
        cls,
        path: Path,
        mask_var: str = "mask",
        lat_dim: str = "latitude",
        lon_dim: str = "longitude",
    ) -> Grid:
        """Load grid from a NetCDF file.

        Args:
            path: Path to the NetCDF grid file.
            mask_var: Variable name for the ocean/land mask.
                Positive values = ocean, non-positive = land.
            lat_dim: Name of the latitude dimension.
            lon_dim: Name of the longitude dimension.
        """
        ds = xr.open_dataset(path)
        mask_data = ds[mask_var].values
        lat = ds[lat_dim].values.astype(np.float64)
        lon = ds[lon_dim].values.astype(np.float64)
        ny, nx = mask_data.shape
        ocean_mask = mask_data > 0
        ds.close()
        return cls(ny=ny, nx=nx, ocean_mask=ocean_mask, lat=lat, lon=lon)

    @classmethod
    def from_dimensions(cls, ny: int, nx: int) -> Grid:
        """Create a simple rectangular grid with all ocean cells."""
        ocean_mask = np.ones((ny, nx), dtype=np.bool_)
        return cls(ny=ny, nx=nx, ocean_mask=ocean_mask)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_grid.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/grid.py tests/test_engine_grid.py
git commit -m "feat(engine): add Grid class with NetCDF loading and cell indexing"
```

---

### Task 5: ResourceState placeholder

**Files:**
- Create: `osmose/engine/resources.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_engine_grid.py`:

```python
from osmose.engine.resources import ResourceState
from osmose.engine.grid import Grid


class TestResourceState:
    def test_create_placeholder(self):
        grid = Grid.from_dimensions(ny=4, nx=5)
        rs = ResourceState(grid=grid)
        assert rs.grid is grid

    def test_update_is_noop(self):
        grid = Grid.from_dimensions(ny=4, nx=5)
        rs = ResourceState(grid=grid)
        rs.update(step=0)  # should not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_grid.py::TestResourceState -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `osmose/engine/resources.py`**

```python
"""ResourceState: low trophic level (LTL) resource forcing.

Phase 1 stub — provides the interface that the simulation loop expects.
Actual NetCDF forcing loading will be implemented in Phase 4+.
"""

from __future__ import annotations

from osmose.engine.grid import Grid


class ResourceState:
    """Container for LTL resource biomass per grid cell.

    In the full implementation, this loads spatiotemporal biomass fields
    from NetCDF forcing files. Phase 1 is a no-op placeholder.
    """

    def __init__(self, grid: Grid) -> None:
        self.grid = grid

    def update(self, step: int) -> None:
        """Load resource biomass for the given timestep.

        Phase 1 stub — does nothing.
        """
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_grid.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/resources.py tests/test_engine_grid.py
git commit -m "feat(engine): add ResourceState placeholder for LTL forcing"
```

---

### Task 6: Simulation loop skeleton with stub processes

**Files:**
- Create: `osmose/engine/simulate.py`
- Create: `osmose/engine/processes/__init__.py`
- Create: `tests/test_engine_simulate.py`

- [ ] **Step 1: Write failing tests for simulate loop**

Create `tests/test_engine_simulate.py`:

```python
"""Tests for the simulation loop skeleton."""

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import StepOutput, simulate


@pytest.fixture
def minimal_config() -> dict[str, str]:
    """Minimal 1-species config for simulation tests."""
    return {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "10",
        "species.name.sp0": "TestFish",
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
    }


class TestSimulate:
    def test_simulate_returns_outputs(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert len(outputs) == 12  # n_steps = 12 * 1

    def test_step_output_has_biomass(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert isinstance(outputs[0], StepOutput)
        assert outputs[0].step == 0
        assert outputs[0].biomass.shape == (1,)  # 1 species

    def test_simulate_correct_step_count(self, minimal_config):
        minimal_config["simulation.time.nyear"] = "2"
        cfg = EngineConfig.from_dict(minimal_config)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert len(outputs) == 24  # 12 * 2
        assert outputs[-1].step == 23
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_simulate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create `osmose/engine/processes/__init__.py`**

```python
"""OSMOSE simulation process functions.

Each process is a pure function: (state, config, ...) -> state.
Stub implementations for Phase 1; real logic added in later phases.
"""
```

- [ ] **Step 4: Implement `osmose/engine/simulate.py`**

```python
"""Main simulation loop for the Python OSMOSE engine.

Follows Java's SimulationStep.step() ordering:
  incoming_flux -> reset -> resources.update -> movement ->
  mortality (interleaved) -> growth -> aging_mortality ->
  reproduction -> collect_outputs -> compact
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.resources import ResourceState
from osmose.engine.state import SchoolState


@dataclass
class StepOutput:
    """Aggregated output for a single simulation timestep."""

    step: int
    biomass: NDArray[np.float64]      # per species
    abundance: NDArray[np.float64]    # per species


# ---------------------------------------------------------------------------
# Stub process functions (replaced in Phase 2-7)
# ---------------------------------------------------------------------------


def _incoming_flux(
    state: SchoolState, config: EngineConfig, step: int, rng: np.random.Generator
) -> SchoolState:
    """Phase 1 stub: incoming flux (migration injection)."""
    return state


def _reset_step_variables(state: SchoolState) -> SchoolState:
    """Reset per-step tracking variables at the start of each timestep."""
    return state.replace(
        n_dead=np.zeros_like(state.n_dead),
        pred_success_rate=np.zeros(len(state), dtype=np.float64),
        preyed_biomass=np.zeros(len(state), dtype=np.float64),
        length_start=state.length.copy(),
    )


def _movement(
    state: SchoolState,
    grid: Grid,
    config: EngineConfig,
    step: int,
    rng: np.random.Generator,
) -> SchoolState:
    """Phase 1 stub: spatial movement."""
    return state


def _mortality(
    state: SchoolState,
    resources: ResourceState,
    config: EngineConfig,
    rng: np.random.Generator,
) -> SchoolState:
    """Phase 1 stub: interleaved mortality (predation, fishing, starvation, additional)."""
    return state


def _growth(
    state: SchoolState, config: EngineConfig, rng: np.random.Generator
) -> SchoolState:
    """Phase 1 stub: growth (Von Bertalanffy / Gompertz)."""
    return state


def _aging_mortality(state: SchoolState, config: EngineConfig) -> SchoolState:
    """Phase 1 stub: kill schools exceeding species lifespan."""
    return state


def _reproduction(
    state: SchoolState, config: EngineConfig, step: int, rng: np.random.Generator
) -> SchoolState:
    """Phase 1 stub: egg production + age increment."""
    return state


# ---------------------------------------------------------------------------
# Output collection
# ---------------------------------------------------------------------------


def _collect_outputs(state: SchoolState, config: EngineConfig, step: int) -> StepOutput:
    """Aggregate per-species biomass and abundance from current state."""
    biomass = np.zeros(config.n_species, dtype=np.float64)
    abundance = np.zeros(config.n_species, dtype=np.float64)
    if len(state) > 0:
        np.add.at(biomass, state.species_id, state.abundance * state.weight)
        np.add.at(abundance, state.species_id, state.abundance)
    return StepOutput(step=step, biomass=biomass, abundance=abundance)


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------


def initialize(config: EngineConfig, grid: Grid, rng: np.random.Generator) -> SchoolState:
    """Create the initial population of schools.

    Phase 1: creates empty state. Phase 3 will add seeding/population init.
    """
    total_schools = int(config.n_schools.sum())
    species_ids = np.repeat(np.arange(config.n_species, dtype=np.int32), config.n_schools)
    return SchoolState.create(n_schools=total_schools, species_id=species_ids)


def simulate(
    config: EngineConfig,
    grid: Grid,
    rng: np.random.Generator,
) -> list[StepOutput]:
    """Run the OSMOSE simulation loop.

    Process ordering matches Java's SimulationStep.step().
    """
    state = initialize(config, grid, rng)
    resources = ResourceState(grid=grid)
    outputs: list[StepOutput] = []

    for step in range(config.n_steps):
        state = _incoming_flux(state, config, step, rng)
        state = _reset_step_variables(state)
        resources.update(step)
        state = _movement(state, grid, config, step, rng)
        state = _mortality(state, resources, config, rng)
        state = _growth(state, config, rng)
        state = _aging_mortality(state, config)
        state = _reproduction(state, config, step, rng)
        outputs.append(_collect_outputs(state, config, step))
        state = state.compact()

    return outputs
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_simulate.py -v`
Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/simulate.py osmose/engine/processes/__init__.py tests/test_engine_simulate.py
git commit -m "feat(engine): add simulation loop skeleton with stub processes and output collection"
```

---

### Task 7: Wire PythonEngine to simulation loop

**Files:**
- Modify: `osmose/engine/__init__.py`
- Modify: `tests/test_engine_state.py` (add integration test)

- [ ] **Step 1: Write failing integration test**

Append to `tests/test_engine_state.py`:

```python
from osmose.engine import PythonEngine


class TestPythonEngineIntegration:
    def test_run_raises_not_implemented_without_config(self):
        """PythonEngine.run() should attempt simulation but fail on bad config."""
        engine = PythonEngine()
        with pytest.raises(KeyError):
            engine.run(config={}, output_dir=Path("/tmp/test"), seed=42)

    def test_run_with_minimal_config(self, tmp_path):
        """PythonEngine.run() should complete with a minimal config."""
        config = {
            "simulation.time.ndtperyear": "12",
            "simulation.time.nyear": "1",
            "simulation.nspecies": "1",
            "simulation.nschool.sp0": "5",
            "species.name.sp0": "TestFish",
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
        }
        engine = PythonEngine()
        result = engine.run(config=config, output_dir=tmp_path, seed=42)
        assert result.returncode == 0
        assert result.output_dir == tmp_path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_state.py::TestPythonEngineIntegration -v`
Expected: FAIL (PythonEngine.run still raises NotImplementedError)

- [ ] **Step 3: Update PythonEngine.run() in `osmose/engine/__init__.py`**

Replace the `PythonEngine.run()` method:

```python
    def run(
        self, config: dict[str, str], output_dir: Path, seed: int = 0
    ) -> RunResult:
        from osmose.engine.config import EngineConfig
        from osmose.engine.grid import Grid
        from osmose.engine.simulate import simulate

        engine_config = EngineConfig.from_dict(config)
        # Phase 1: use simple grid; Phase 4+ will load from config
        nx = int(config.get("grid.ncol", "10"))
        ny = int(config.get("grid.nrow", "10"))
        grid = Grid.from_dimensions(ny=ny, nx=nx)
        rng = np.random.default_rng(seed)

        simulate(engine_config, grid, rng)

        return RunResult(
            returncode=0,
            output_dir=output_dir,
            stdout="",
            stderr="",
        )
```

Add `import numpy as np` at the top of `osmose/engine/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_state.py -v`
Expected: All PASSED

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `.venv/bin/python -m pytest -q`
Expected: All existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/__init__.py tests/test_engine_state.py
git commit -m "feat(engine): wire PythonEngine.run() to simulation loop"
```

---

### Task 8: Lint and final verification

**Files:** None new — verification only.

- [ ] **Step 1: Run linter**

Run: `.venv/bin/ruff check osmose/engine/ tests/test_engine_*.py`
Expected: No errors. If any, fix them.

- [ ] **Step 2: Run formatter**

Run: `.venv/bin/ruff format osmose/engine/ tests/test_engine_*.py`

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: All tests pass (existing 1151 + ~18 new engine tests)

- [ ] **Step 4: Commit any lint/format fixes**

```bash
git add -u
git commit -m "style: lint and format engine Phase 1 code"
```

- [ ] **Step 5: Tag Phase 1 milestone**

```bash
git tag -a engine-phase1 -m "Python engine Phase 1: foundation (protocol, state, grid, config, loop skeleton)"
```
