# SP-2: Additional Mortality Variants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete additional mortality variants to match all 5 Java OSMOSE 4.3.3 `mortality/additional/` classes, including time-varying larval rates, by-class rates, and spatial factor application.

**Architecture:** The Python engine already has constant rates, time-varying by-dt rates (via raw arrays), and spatial factor loading. This plan adds 3 missing pieces: (1) by-dt-by-class variant via SP-3 ByClassTimeSeries, (2) time-varying larval rates, (3) spatial factor application in the mortality function.

**Tech Stack:** Python 3.12+, NumPy, SP-3 timeseries framework (`osmose/engine/timeseries.py`).

**Spec:** `docs/superpowers/specs/2026-04-16-java-parity-full-design.md` (SP-2 section)

**Java reference:** `/home/razinka/osmose/osmose-master/java/src/main/java/fr/ird/osmose/process/mortality/additional/`

**What already exists:**
- `osmose/engine/processes/natural.py` — `additional_mortality()` with constant + by-dt rates, `larva_mortality()` with constant rate
- `osmose/engine/config.py` — `additional_mortality_rate`, `additional_mortality_by_dt` (from `bytdt.file`), `additional_mortality_spatial` (loaded but not applied), `larva_mortality_rate`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `osmose/engine/processes/natural.py` | **modify** | Add by-class rates, time-varying larval rates, spatial factor |
| `osmose/engine/config.py` | **modify** | Add by-class and larval by-dt config loading, `byDt` key support |
| `tests/test_engine_additional_mortality.py` | **create** | Tests for new variants and spatial factor |

---

### Task 1: ByDtByClass Additional Mortality

Rate varies per (dt, age/size class) using `ByClassTimeSeries`. Java: `ByDtByClassAdditionalMortality`.

**Files:**
- Modify: `osmose/engine/processes/natural.py`
- Modify: `osmose/engine/config.py`
- Create: `tests/test_engine_additional_mortality.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_engine_additional_mortality.py`:

```python
"""Tests for additional mortality variants (SP-2)."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.processes.natural import additional_mortality
from osmose.engine.state import MortalityCause, SchoolState
from osmose.engine.timeseries import ByClassTimeSeries


def _write_csv(path: Path, header: list[str], rows: list[list[str]], sep: str = ";") -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=sep)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _make_config(**overrides) -> MagicMock:
    """Create minimal EngineConfig mock for additional mortality tests."""
    config = MagicMock(spec=EngineConfig)
    config.n_species = 1
    config.n_dt_per_year = 24
    config.additional_mortality_rate = np.array([0.1])
    config.additional_mortality_by_dt = None
    config.additional_mortality_by_dt_by_class = None
    config.additional_mortality_spatial = None
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def _make_state(n: int = 1, sp: int = 0, abundance: float = 1000.0,
                age_dt: int = 48, length: float = 20.0) -> SchoolState:
    state = SchoolState.create(n_schools=n, species_id=np.full(n, sp, dtype=np.int32))
    return state.replace(
        abundance=np.full(n, abundance),
        weight=np.full(n, 0.01),
        length=np.full(n, length),
        age_dt=np.full(n, age_dt, dtype=np.int32),
        cell_x=np.zeros(n, dtype=np.int32),
        cell_y=np.zeros(n, dtype=np.int32),
    )


class TestByDtByClassAdditionalMortality:
    """Rate varies per (dt, age/size class) from ByClassTimeSeries."""

    def test_young_gets_low_rate(self, tmp_path: Path) -> None:
        """Young school (age class 0) gets rate from first column.

        CSV thresholds are in dt units (already converted by config loader
        for byAge: years * ndt_per_year). For testing, we use dt directly.
        """
        csv_file = tmp_path / "mort.csv"
        # Thresholds already in dt (as if config loader converted from years)
        _write_csv(csv_file, ["step", "0", "48"], [
            ["0", "0.1", "0.5"],
        ])
        ts = ByClassTimeSeries.from_csv(csv_file, ndt_per_year=1, ndt_simu=1)
        config = _make_config(additional_mortality_by_dt_by_class=[ts])

        state = _make_state(age_dt=10)  # age_dt=10 → class 0 (< 48)
        result = additional_mortality(state, config, n_subdt=1, step=0)
        dead = result.n_dead[0, MortalityCause.ADDITIONAL]
        assert dead > 0
        assert dead < 10

    def test_old_gets_high_rate(self, tmp_path: Path) -> None:
        """Old school (age class 1) gets rate from second column."""
        csv_file = tmp_path / "mort.csv"
        _write_csv(csv_file, ["step", "0", "48"], [
            ["0", "0.1", "2.0"],
        ])
        ts = ByClassTimeSeries.from_csv(csv_file, ndt_per_year=1, ndt_simu=1)
        config = _make_config(additional_mortality_by_dt_by_class=[ts])

        state = _make_state(age_dt=100)  # age_dt=100 → class 1 (≥ 48)
        result = additional_mortality(state, config, n_subdt=1, step=0)
        dead = result.n_dead[0, MortalityCause.ADDITIONAL]
        assert dead > 50

    def test_below_first_threshold_gets_zero(self, tmp_path: Path) -> None:
        """Age below first threshold → rate 0 (Java: return 0)."""
        csv_file = tmp_path / "mort.csv"
        _write_csv(csv_file, ["step", "24", "48"], [
            ["0", "0.5", "1.0"],
        ])
        ts = ByClassTimeSeries.from_csv(csv_file, ndt_per_year=1, ndt_simu=1)
        config = _make_config(additional_mortality_by_dt_by_class=[ts])

        state = _make_state(age_dt=10)  # below first threshold (24)
        result = additional_mortality(state, config, n_subdt=1, step=0)
        dead = result.n_dead[0, MortalityCause.ADDITIONAL]
        assert dead == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_additional_mortality.py::TestByDtByClassAdditionalMortality -v`
Expected: FAIL — `config has no attribute 'additional_mortality_by_dt_by_class'`

- [ ] **Step 3: Add config field and loading**

In `osmose/engine/config.py`, add field to EngineConfig (after `additional_mortality_by_dt`):

```python
    # Additional mortality by dt and age/size class: per-species ByClassTimeSeries
    additional_mortality_by_dt_by_class: list[ByClassTimeSeries | None] | None
```

Add loading function:

```python
def _load_additional_mortality_by_dt_by_class(
    cfg: dict[str, str], n_species: int, n_dt_per_year: int, n_dt_simu: int
) -> list[ByClassTimeSeries | None] | None:
    """Load by-dt-by-class additional mortality from CSV.

    For byAge: Java converts class thresholds from years to time steps
    (threshold * nStepYear). For bySize: thresholds are in cm, used as-is.
    """
    from osmose.engine.timeseries import ByClassTimeSeries

    result: list[ByClassTimeSeries | None] = [None] * n_species
    found = False
    for i in range(n_species):
        is_by_age = False
        for variant in ["byDt.byAge", "byDt.bySize"]:
            key = f"mortality.additional.rate.{variant}.file.sp{i}"
            if key in cfg:
                path = _require_file(cfg[key], _cfg_dir(cfg), key)
                ts = ByClassTimeSeries.from_csv(path, n_dt_per_year, n_dt_simu)
                # Java converts age thresholds from years to time steps
                if "byAge" in variant:
                    ts.classes = np.round(ts.classes * n_dt_per_year).astype(np.float64)
                result[i] = ts
                found = True
                break
    return result if found else None
```

- [ ] **Step 4: Add by-class branch to `additional_mortality()`**

In `osmose/engine/processes/natural.py`, add after the existing `additional_mortality_by_dt` block (line ~41):

```python
    # Override with by-dt-by-class rates (ByClassTimeSeries)
    if config.additional_mortality_by_dt_by_class is not None:
        for sp_i in range(config.n_species):
            ts = config.additional_mortality_by_dt_by_class[sp_i]
            if ts is None:
                continue
            sp_mask = sp == sp_i
            if not sp_mask.any():
                continue
            ages_dt = state.age_dt[sp_mask].astype(float)
            step_idx = min(step, len(ts.values) - 1)
            for j in range(len(ages_dt)):
                class_idx = ts.class_of(ages_dt[j])
                school_idx = np.where(sp_mask)[0][j]
                if class_idx >= 0:
                    m_rate[school_idx] = ts.get_by_class(step_idx, class_idx)
                else:
                    m_rate[school_idx] = 0.0  # below first threshold
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_additional_mortality.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Lint and commit**

```bash
git add osmose/engine/processes/natural.py osmose/engine/config.py tests/test_engine_additional_mortality.py
git commit -m "feat(engine): add by-dt-by-class additional mortality variant"
```

---

### Task 2: Time-Varying Larval Mortality

Java `ByDtLarvaMortality` — larval rate varies per time step from CSV. Config key: `mortality.additional.larva.rate.bytDt.file.sp{i}` (Java typo) or `mortality.additional.larva.rate.byDt.file.sp{i}` (corrected).

**Files:**
- Modify: `osmose/engine/processes/natural.py`
- Modify: `osmose/engine/config.py`
- Modify: `tests/test_engine_additional_mortality.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_engine_additional_mortality.py`:

```python
from osmose.engine.processes.natural import larva_mortality


class TestByDtLarvaMortality:
    """Larval rate varies per time step."""

    def test_time_varying_larva_rate(self, tmp_path: Path) -> None:
        """Larval rate changes per step from CSV."""
        csv_file = tmp_path / "larva.csv"
        # Step 0: rate 0.5, Step 1: rate 2.0
        _write_csv(csv_file, ["step", "value"], [["0", "0.5"], ["1", "2.0"]])

        from osmose.engine.timeseries import SingleTimeSeries
        ts = SingleTimeSeries.from_csv(csv_file, ndt_per_year=2, ndt_simu=2)
        config = _make_config(
            larva_mortality_rate=np.array([0.1]),  # base rate (overridden)
            larva_mortality_by_dt=[ts],
        )

        # Create egg state
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([10000.0]),
            weight=np.array([0.0001]),
            length=np.array([0.1]),
            age_dt=np.array([0], dtype=np.int32),
            cell_x=np.zeros(1, dtype=np.int32),
            cell_y=np.zeros(1, dtype=np.int32),
        )

        # Step 0: rate 0.5
        result0 = larva_mortality(state, config, step=0)
        dead0 = result0.n_dead[0, MortalityCause.ADDITIONAL]

        # Step 1: rate 2.0 → higher mortality
        result1 = larva_mortality(state, config, step=1)
        dead1 = result1.n_dead[0, MortalityCause.ADDITIONAL]

        assert dead1 > dead0

    def test_bytdt_typo_key_detected(self) -> None:
        """Config detection supports both bytDt (Java typo) and byDt."""
        from osmose.engine.config import _detect_larva_by_dt_key

        cfg1 = {"mortality.additional.larva.rate.bytDt.file.sp0": "/path.csv"}
        assert _detect_larva_by_dt_key(cfg1, 0) == "/path.csv"

        cfg2 = {"mortality.additional.larva.rate.byDt.file.sp0": "/other.csv"}
        assert _detect_larva_by_dt_key(cfg2, 0) == "/other.csv"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_additional_mortality.py::TestByDtLarvaMortality -v`
Expected: FAIL

- [ ] **Step 3: Add larva by-dt config field and loading**

In `osmose/engine/config.py`, add field:

```python
    larva_mortality_by_dt: list[SingleTimeSeries | None] | None
```

Add detection helper:

```python
def _detect_larva_by_dt_key(cfg: dict[str, str], species_idx: int) -> str | None:
    """Detect larval by-dt file, supporting both bytDt (Java typo) and byDt."""
    for variant in ["bytDt", "byDt"]:
        key = f"mortality.additional.larva.rate.{variant}.file.sp{species_idx}"
        if key in cfg:
            return cfg[key]
    return None
```

Add loading:

```python
def _load_larva_mortality_by_dt(
    cfg: dict[str, str], n_species: int, n_dt_per_year: int, n_dt_simu: int
) -> list[SingleTimeSeries | None] | None:
    from osmose.engine.timeseries import SingleTimeSeries

    result: list[SingleTimeSeries | None] = [None] * n_species
    found = False
    for i in range(n_species):
        path_str = _detect_larva_by_dt_key(cfg, i)
        if path_str:
            path = _require_file(path_str, _cfg_dir(cfg), f"larva.rate.byDt.sp{i}")
            result[i] = SingleTimeSeries.from_csv(path, n_dt_per_year, n_dt_simu)
            found = True
    return result if found else None
```

- [ ] **Step 4: Add step parameter and by-dt branch to `larva_mortality()`**

In `osmose/engine/processes/natural.py`, update `larva_mortality` signature and add by-dt logic:

```python
def larva_mortality(state: SchoolState, config: EngineConfig, step: int = 0) -> SchoolState:
    """Apply additional mortality to eggs/larvae.

    Supports constant (Annual) and time-varying (ByDt) rates.
    """
    if len(state) == 0:
        return state

    eggs = state.is_egg
    if not eggs.any():
        return state

    sp = state.species_id
    m_rate = config.larva_mortality_rate[sp].copy()

    # Override with time-varying BY_DT larval rates
    if config.larva_mortality_by_dt is not None:
        for i in range(len(state)):
            if not eggs[i]:
                continue
            sp_i = sp[i]
            if sp_i < len(config.larva_mortality_by_dt) and config.larva_mortality_by_dt[sp_i] is not None:
                ts = config.larva_mortality_by_dt[sp_i]
                m_rate[i] = ts.get(step)

    d = m_rate
    mortality_fraction = 1 - np.exp(-d)

    n_dead = np.zeros_like(state.abundance)
    n_dead[eggs] = state.abundance[eggs] * mortality_fraction[eggs]

    new_abundance = state.abundance - n_dead
    new_biomass = new_abundance * state.weight
    new_n_dead = state.n_dead.copy()
    new_n_dead[:, MortalityCause.ADDITIONAL] += n_dead

    return state.replace(abundance=new_abundance, biomass=new_biomass, n_dead=new_n_dead)
```

**Important:** All callers of `larva_mortality()` must be updated to pass `step=step`. Check `osmose/engine/simulate.py` for the call site.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_additional_mortality.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Lint and commit**

```bash
git add osmose/engine/processes/natural.py osmose/engine/config.py tests/test_engine_additional_mortality.py
git commit -m "feat(engine): add time-varying larval mortality (ByDtLarvaMortality)"
```

---

### Task 3: Spatial Factor Application

The spatial factor `additional_mortality_spatial` is already loaded in config but not applied in `additional_mortality()`. Add application matching Java's `AdditionalMortality` behavior.

**Files:**
- Modify: `osmose/engine/processes/natural.py`
- Modify: `tests/test_engine_additional_mortality.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_engine_additional_mortality.py`:

```python
class TestSpatialAdditionalMortality:
    """Spatial factor multiplies additional mortality per cell."""

    def test_spatial_factor_applied(self) -> None:
        """Schools in high-mortality cells get more deaths."""
        # Spatial map: cell (0,0)=2.0 (double rate), cell (0,1)=0.0 (no mortality)
        spatial_map = np.array([[2.0, 0.0]])  # shape (1, 2)
        config = _make_config(
            additional_mortality_rate=np.array([1.0]),  # base rate
            additional_mortality_spatial=[spatial_map],
        )

        # Two schools: one in cell (0,0), one in cell (0,1)
        state = SchoolState.create(n_schools=2, species_id=np.array([0, 0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0, 1000.0]),
            weight=np.array([0.01, 0.01]),
            length=np.array([20.0, 20.0]),
            age_dt=np.array([48, 48], dtype=np.int32),
            cell_x=np.array([0, 1], dtype=np.int32),
            cell_y=np.array([0, 0], dtype=np.int32),
        )

        result = additional_mortality(state, config, n_subdt=1, step=0)
        dead_high = result.n_dead[0, MortalityCause.ADDITIONAL]
        dead_zero = result.n_dead[1, MortalityCause.ADDITIONAL]

        assert dead_high > 0
        assert dead_zero == 0.0  # factor = 0 → no mortality

    def test_no_spatial_means_uniform(self) -> None:
        """Without spatial factor, mortality is uniform across cells."""
        config = _make_config(
            additional_mortality_rate=np.array([0.5]),
            additional_mortality_spatial=None,
        )
        state = _make_state(n=2, abundance=1000.0, age_dt=48)
        state = state.replace(
            cell_x=np.array([0, 5], dtype=np.int32),
            cell_y=np.array([0, 5], dtype=np.int32),
        )
        result = additional_mortality(state, config, n_subdt=1, step=0)
        dead0 = result.n_dead[0, MortalityCause.ADDITIONAL]
        dead1 = result.n_dead[1, MortalityCause.ADDITIONAL]
        assert dead0 == pytest.approx(dead1, rel=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_additional_mortality.py::TestSpatialAdditionalMortality -v`
Expected: FAIL — spatial factor not applied (both schools get same mortality)

- [ ] **Step 3: Apply spatial factor in `additional_mortality()`**

In `osmose/engine/processes/natural.py`, add spatial factor before computing `n_dead` (after line ~46):

```python
    # Apply spatial factor (per-cell multiplier)
    spatial_factor = np.ones(len(state), dtype=np.float64)
    if config.additional_mortality_spatial is not None:
        cy = state.cell_y.astype(np.intp)
        cx = state.cell_x.astype(np.intp)
        for sp_i in range(config.n_species):
            sp_map = config.additional_mortality_spatial[sp_i]
            if sp_map is None:
                continue
            sp_mask = sp == sp_i
            if not sp_mask.any():
                continue
            sy, sx = cy[sp_mask], cx[sp_mask]
            valid = (sy >= 0) & (sy < sp_map.shape[0]) & (sx >= 0) & (sx < sp_map.shape[1])
            vals = np.zeros(sp_mask.sum(), dtype=np.float64)
            vals[valid] = sp_map[sy[valid], sx[valid]]
            spatial_factor[sp_mask] = vals

    n_dead = state.abundance * mortality_fraction * spatial_factor
```

Replace the existing `n_dead = state.abundance * mortality_fraction` with this.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_additional_mortality.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All tests pass (check that existing additional mortality tests aren't broken)

- [ ] **Step 6: Lint and commit**

```bash
git add osmose/engine/processes/natural.py tests/test_engine_additional_mortality.py
git commit -m "feat(engine): apply spatial factor to additional mortality"
```
