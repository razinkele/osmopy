# SP-1: Fishing System Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the fishing system to match all Java OSMOSE 4.3.3 fishing mortality variants and selectivity types.

**Architecture:** The Python engine already has substantial fishing infrastructure: rate-by-season with seasonality, rate-by-year, sigmoid + knife-edge selectivity, spatial maps, MPA, and discards. This plan adds the 4 missing pieces: (1) Gaussian and log-normal selectivity types, (2) rate-by-dt-by-class variant using SP-3 ByClassTimeSeries, (3) catch-based proportional allocation variants, (4) config dispatch to auto-detect variant.

**Tech Stack:** Python 3.12+, NumPy, SP-3 timeseries framework (`osmose/engine/timeseries.py`).

**Spec:** `docs/superpowers/specs/2026-04-16-java-parity-full-design.md` (SP-1 section)

**Java reference:** `/home/razinka/osmose/osmose-master/java/src/main/java/fr/ird/osmose/process/mortality/fishing/` and `fishery/`

**What already exists:**
- `osmose/engine/processes/fishing.py` — rate-by-season, rate-by-year, seasonality, spatial, MPA, discards
- `osmose/engine/processes/selectivity.py` — knife-edge and sigmoid
- `osmose/engine/config.py` — fishing_rate, fishing_rate_by_year, fishing_seasonality, selectivity fields, MPA, spatial maps

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `osmose/engine/processes/selectivity.py` | **modify** | Add Gaussian and log-normal selectivity functions |
| `osmose/engine/processes/fishing.py` | **modify** | Add rate-by-dt-by-class and catch-based variants |
| `osmose/engine/config.py` | **modify** | Parse new fishing config keys, add scenario dispatch |
| `tests/test_engine_fishing_variants.py` | **create** | Tests for new variants and selectivity types |

---

### Task 1: Gaussian and Log-Normal Selectivity

Java has 4 selectivity types (0=knife-edge, 1=sigmoid, 2=Gaussian, 3=log-normal). Python has 0 and 1. Add types 2 and 3.

**Files:**
- Modify: `osmose/engine/processes/selectivity.py`
- Create: `tests/test_engine_fishing_variants.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_engine_fishing_variants.py`:

```python
"""Tests for fishing system variants and selectivity types."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine.processes.selectivity import (
    knife_edge,
    sigmoid,
    gaussian,
    log_normal,
)


class TestGaussianSelectivity:
    """Type 2: Normal distribution, normalized by peak."""

    def test_peak_at_l50(self) -> None:
        """Maximum selectivity (1.0) at L50."""
        length = np.array([20.0])
        sel = gaussian(length, l50=20.0, l75=25.0)
        assert sel[0] == pytest.approx(1.0)

    def test_symmetric_bell(self) -> None:
        """Symmetric around L50."""
        l50, l75 = 20.0, 25.0
        lengths = np.array([15.0, 25.0])  # equidistant from l50
        sel = gaussian(lengths, l50=l50, l75=l75)
        assert sel[0] == pytest.approx(sel[1], rel=1e-6)

    def test_decreases_from_peak(self) -> None:
        """Selectivity decreases away from L50."""
        lengths = np.array([10.0, 20.0, 30.0])
        sel = gaussian(lengths, l50=20.0, l75=25.0)
        assert sel[1] > sel[0]
        assert sel[1] > sel[2]

    def test_at_l75_approx_75pct(self) -> None:
        """At L75, selectivity should be approximately 0.75 of peak.

        Java: sd = (L75 - L50) / qnorm(0.75), so at L75 the normal PDF
        ratio = density(L75)/density(L50) ≈ exp(-0.5 * 0.6745²) ≈ 0.7978.
        """
        sel = gaussian(np.array([25.0]), l50=20.0, l75=25.0)
        assert sel[0] == pytest.approx(0.7978, rel=0.01)


class TestLogNormalSelectivity:
    """Type 3: Log-normal distribution, normalized by mode."""

    def test_peak_at_mode(self) -> None:
        """Maximum selectivity at the mode = exp(mean - sd²)."""
        l50, l75 = 20.0, 30.0
        mean = np.log(l50)
        q75 = 0.674489750196082
        sd = np.log(l75 / l50) / q75
        mode = np.exp(mean - sd**2)
        sel = log_normal(np.array([mode]), l50=l50, l75=l75)
        assert sel[0] == pytest.approx(1.0, rel=1e-4)

    def test_asymmetric(self) -> None:
        """Log-normal is right-skewed — more selectivity above mode than below."""
        l50, l75 = 20.0, 30.0
        lengths = np.array([5.0, 10.0, 20.0, 30.0, 50.0])
        sel = log_normal(lengths, l50=l50, l75=l75)
        # Should be highest near mode, asymmetric
        peak_idx = np.argmax(sel)
        assert sel[peak_idx] > sel[0]
        assert sel[peak_idx] > sel[-1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_fishing_variants.py::TestGaussianSelectivity tests/test_engine_fishing_variants.py::TestLogNormalSelectivity -v`
Expected: FAIL — `cannot import name 'gaussian'`

- [ ] **Step 3: Implement Gaussian and log-normal selectivity**

Add to `osmose/engine/processes/selectivity.py`:

```python
from scipy.stats import norm, lognorm


# qnorm(0.75) — the 75th percentile of the standard normal
_Q75 = 0.674489750196082


def gaussian(
    length: NDArray[np.float64], l50: float, l75: float, tiny: float = 1e-8
) -> NDArray[np.float64]:
    """Gaussian (normal) selectivity — type 2 in Java.

    Peak at L50, normalized so selectivity(L50) = 1.0.
    sd = (L75 - L50) / qnorm(0.75).
    Matches Java FisherySelectivity.getGaussianSelectivity().
    """
    sd = (l75 - l50) / _Q75
    peak_density = norm.pdf(l50, loc=l50, scale=sd)
    sel = norm.pdf(length, loc=l50, scale=sd) / peak_density
    sel[sel < tiny] = 0.0
    return sel


def log_normal(
    length: NDArray[np.float64], l50: float, l75: float, tiny: float = 1e-8
) -> NDArray[np.float64]:
    """Log-normal selectivity — type 3 in Java.

    Normalized by mode density. Parameters:
    mean = log(L50), sd = log(L75/L50) / qnorm(0.75).
    mode = exp(mean - sd²).
    Matches Java FisherySelectivity.getLogNormalSelectivity().
    """
    mean = np.log(l50)
    sd = np.log(l75 / l50) / _Q75
    mode = np.exp(mean - sd**2)
    # scipy lognorm: s=sd, scale=exp(mean)
    scale = np.exp(mean)
    mode_density = lognorm.pdf(mode, s=sd, scale=scale)
    sel = lognorm.pdf(length, s=sd, scale=scale) / mode_density
    sel[sel < tiny] = 0.0
    return sel
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_fishing_variants.py::TestGaussianSelectivity tests/test_engine_fishing_variants.py::TestLogNormalSelectivity -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Lint and commit**

```bash
git add osmose/engine/processes/selectivity.py tests/test_engine_fishing_variants.py
git commit -m "feat(engine): add Gaussian and log-normal fishing selectivity types"
```

---

### Task 2: Rate-by-Dt-by-Class Fishing Variant

Adds `F(dt, age/size class)` from a ByClassTimeSeries CSV. This is `RateByDtByClassFishingMortality` in Java.

**Files:**
- Modify: `osmose/engine/processes/fishing.py`
- Modify: `osmose/engine/config.py`
- Modify: `tests/test_engine_fishing_variants.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_engine_fishing_variants.py`:

```python
from osmose.engine.processes.fishing import fishing_mortality
from osmose.engine.config import EngineConfig
from osmose.engine.state import SchoolState, MortalityCause
from osmose.engine.timeseries import ByClassTimeSeries


class TestRateByDtByClass:
    """RateByDtByClassFishingMortality — rate varies per (dt, age/size class)."""

    def _make_config(self, by_class_ts: list[ByClassTimeSeries | None]) -> EngineConfig:
        """Create a minimal EngineConfig with fishing_rate_by_dt_by_class."""
        # Minimal config with 1 species, fishing enabled
        from tests.helpers import _make_school
        from unittest.mock import MagicMock

        config = MagicMock(spec=EngineConfig)
        config.fishing_enabled = True
        config.n_species = 1
        config.n_dt_per_year = 24
        config.fishing_rate = np.array([0.0])  # overridden by by_class
        config.fishing_rate_by_year = None
        config.fishing_seasonality = None
        config.fishing_rate_by_dt_by_class = by_class_ts
        config.fishing_catches = None
        config.fishing_selectivity_type = np.array([0], dtype=np.int32)
        config.fishing_selectivity_a50 = np.array([0.0])
        config.fishing_selectivity_l50 = np.array([0.0])
        config.fishing_selectivity_slope = np.array([0.0])
        config.fishing_selectivity_l75 = np.array([0.0])
        config.fishing_spatial_maps = [None]
        config.mpa_zones = None
        config.fishing_discard_rate = None
        return config

    def test_rate_from_age_class(self) -> None:
        """Rate looked up by simulation step and school age class."""
        import csv
        import tempfile
        from pathlib import Path

        # Create a ByClassTimeSeries with 2 age classes (0 and 2 years)
        # Step 0: class 0 = 0.1, class 1 = 0.5
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["step", "0", "48"])  # age thresholds in dt (0 and 48=2yr*24dt)
            writer.writerow(["0", "0.1", "0.5"])
            path = Path(f.name)

        ts = ByClassTimeSeries.from_csv(path, ndt_per_year=1, ndt_simu=1)
        config = self._make_config([ts])

        # Young school (age_dt=10 < 48) → class 0 → rate 0.1
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            weight=np.array([0.01]),
            length=np.array([10.0]),
            age_dt=np.array([10], dtype=np.int32),
            cell_x=np.array([0], dtype=np.int32),
            cell_y=np.array([0], dtype=np.int32),
        )

        result = fishing_mortality(state, config, n_subdt=1, step=0)
        # With F=0.1, mortality_fraction = 1 - exp(-0.1) ≈ 0.0952
        dead = result.n_dead[0, MortalityCause.FISHING]
        assert dead > 0
        assert dead < 200  # Roughly 95 dead from 1000

        path.unlink()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_fishing_variants.py::TestRateByDtByClass -v`
Expected: FAIL — `config has no attribute 'fishing_rate_by_dt_by_class'`

- [ ] **Step 3: Add `fishing_rate_by_dt_by_class` to EngineConfig**

In `osmose/engine/config.py`, add field after `fishing_rate_by_year` (around line 1055):

```python
    # Fishing rate by dt and age/size class: per-species ByClassTimeSeries, or None
    fishing_rate_by_dt_by_class: list[ByClassTimeSeries | None] | None
```

In `_build_engine_config()` (the function that constructs EngineConfig), add loading logic:

```python
    # Load fishing rate by dt by class
    fishing_rate_by_dt_by_class: list[ByClassTimeSeries | None] = []
    for i in range(n_species):
        loaded = False
        for variant in ["byDt.byAge", "byDt.bySize"]:
            key = f"mortality.fishing.rate.{variant}.file.sp{i}"
            if key in cfg:
                from osmose.engine.timeseries import ByClassTimeSeries
                ts = ByClassTimeSeries.from_csv(
                    Path(cfg[key]), n_dt_per_year, n_dt_per_year * n_years
                )
                fishing_rate_by_dt_by_class.append(ts)
                loaded = True
                break
        if not loaded:
            fishing_rate_by_dt_by_class.append(None)
```

- [ ] **Step 4: Add rate-by-dt-by-class branch to `fishing_mortality()`**

In `osmose/engine/processes/fishing.py`, after the rate-by-year block (line ~45), add:

```python
    # Rate by dt by class — overrides base rate with per-class rate for this step
    if config.fishing_rate_by_dt_by_class is not None:
        step_idx = min(step, config.n_dt_per_year * 100)  # safety bound
        for sp_i in range(config.n_species):
            ts = config.fishing_rate_by_dt_by_class[sp_i]
            if ts is None:
                continue
            sp_mask = sp == sp_i
            if not sp_mask.any():
                continue
            # Look up rate by age class for each school
            ages_dt = state.age_dt[sp_mask].astype(float)
            for j in range(len(ages_dt)):
                class_idx = ts.class_of(ages_dt[j])
                if class_idx >= 0:
                    school_idx = np.where(sp_mask)[0][j]
                    f_rate[school_idx] = ts.get_by_class(step_idx, class_idx)
                else:
                    school_idx = np.where(sp_mask)[0][j]
                    f_rate[school_idx] = 0.0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_fishing_variants.py -v`
Expected: All tests PASS

- [ ] **Step 6: Lint and commit**

```bash
git add osmose/engine/processes/fishing.py osmose/engine/config.py tests/test_engine_fishing_variants.py
git commit -m "feat(engine): add rate-by-dt-by-class fishing mortality variant"
```

---

### Task 3: Catch-Based Fishing Variants

Java's catch-based variants use proportional allocation: `catch = (school_biomass / fishable_biomass) × target_catches × season_weight`. Three variants: CatchesBySeason (constant annual), CatchesByYearBySeason (year-varying), CatchesByDtByClass (per dt per class).

**Files:**
- Modify: `osmose/engine/processes/fishing.py`
- Modify: `osmose/engine/config.py`
- Modify: `tests/test_engine_fishing_variants.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_engine_fishing_variants.py`:

```python
class TestCatchBasedFishing:
    """Catch-based proportional allocation."""

    def test_proportional_allocation(self) -> None:
        """Catch distributed proportionally to school biomass."""
        from unittest.mock import MagicMock

        config = MagicMock(spec=EngineConfig)
        config.fishing_enabled = True
        config.n_species = 1
        config.n_dt_per_year = 12
        config.fishing_rate = np.array([0.0])
        config.fishing_rate_by_year = None
        config.fishing_seasonality = None
        config.fishing_rate_by_dt_by_class = None
        # Catch-based: 100 tonnes annual catch, uniform season
        config.fishing_catches = np.array([100.0])
        config.fishing_catches_by_year = None
        config.fishing_catches_season = None
        config.fishing_selectivity_type = np.array([-1], dtype=np.int32)
        config.fishing_selectivity_a50 = np.array([np.nan])
        config.fishing_selectivity_l50 = np.array([0.0])
        config.fishing_selectivity_slope = np.array([0.0])
        config.fishing_selectivity_l75 = np.array([0.0])
        config.fishing_spatial_maps = [None]
        config.mpa_zones = None
        config.fishing_discard_rate = None

        # Two schools of same species, different biomass
        state = SchoolState.create(n_schools=2, species_id=np.array([0, 0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0, 3000.0]),
            weight=np.array([0.001, 0.001]),  # biomass: 1.0, 3.0 tonnes
            length=np.array([10.0, 10.0]),
            age_dt=np.array([24, 24], dtype=np.int32),
            cell_x=np.array([0, 0], dtype=np.int32),
            cell_y=np.array([0, 0], dtype=np.int32),
        )

        result = fishing_mortality(state, config, n_subdt=1, step=0)

        # Total fishable biomass = 4.0 tonnes
        # Season weight = 1/12 (uniform)
        # School 0 catch = (1/4) * 100 * (1/12) ≈ 2.08 tonnes
        # School 1 catch = (3/4) * 100 * (1/12) ≈ 6.25 tonnes
        dead_0 = result.n_dead[0, MortalityCause.FISHING]
        dead_1 = result.n_dead[1, MortalityCause.FISHING]
        assert dead_1 / dead_0 == pytest.approx(3.0, rel=0.1)  # 3:1 ratio

    def test_zero_fishable_biomass(self) -> None:
        """Zero fishable biomass → no catch."""
        from unittest.mock import MagicMock

        config = MagicMock(spec=EngineConfig)
        config.fishing_enabled = True
        config.n_species = 1
        config.n_dt_per_year = 12
        config.fishing_rate = np.array([0.0])
        config.fishing_rate_by_year = None
        config.fishing_seasonality = None
        config.fishing_rate_by_dt_by_class = None
        config.fishing_catches = np.array([100.0])
        config.fishing_catches_by_year = None
        config.fishing_catches_season = None
        config.fishing_selectivity_type = np.array([0], dtype=np.int32)
        config.fishing_selectivity_a50 = np.array([100.0])  # no school old enough
        config.fishing_selectivity_l50 = np.array([0.0])
        config.fishing_selectivity_slope = np.array([0.0])
        config.fishing_selectivity_l75 = np.array([0.0])
        config.fishing_spatial_maps = [None]
        config.mpa_zones = None
        config.fishing_discard_rate = None

        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            weight=np.array([0.001]),
            length=np.array([5.0]),
            age_dt=np.array([1], dtype=np.int32),  # too young for a50=100
            cell_x=np.array([0], dtype=np.int32),
            cell_y=np.array([0], dtype=np.int32),
        )

        result = fishing_mortality(state, config, n_subdt=1, step=0)
        assert result.n_dead[0, MortalityCause.FISHING] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_fishing_variants.py::TestCatchBasedFishing -v`
Expected: FAIL — `config has no attribute 'fishing_catches'`

- [ ] **Step 3: Add catch-based fields to EngineConfig**

In `osmose/engine/config.py`, add after the fishing_rate_by_dt_by_class field:

```python
    # Catch-based fishing: annual target catches (tonnes) per species, or None
    fishing_catches: NDArray[np.float64] | None
    # Catch by year: per-species ByYearTimeSeries, or None
    fishing_catches_by_year: list[ByYearTimeSeries | None] | None
    # Catch seasonality: same as fishing_seasonality
    fishing_catches_season: NDArray[np.float64] | None
```

In config loading, detect catch-based scenario:
```python
    # Detect catch-based scenarios
    fishing_catches = None
    fishing_catches_by_year = None
    for i in range(n_species):
        key = f"mortality.fishing.catches.sp{i}"
        if key in cfg:
            if fishing_catches is None:
                fishing_catches = np.zeros(n_species)
            fishing_catches[i] = float(cfg[key])
        year_key = f"mortality.fishing.catches.byYear.file.sp{i}"
        if year_key in cfg:
            if fishing_catches_by_year is None:
                fishing_catches_by_year = [None] * n_species
            from osmose.engine.timeseries import ByYearTimeSeries
            fishing_catches_by_year[i] = ByYearTimeSeries.from_csv(
                Path(cfg[year_key]), n_years
            )
```

- [ ] **Step 4: Add catch-based branch to `fishing_mortality()`**

In `osmose/engine/processes/fishing.py`, add a catch-based path. The key logic: if `config.fishing_catches` is not None, use proportional allocation instead of rate-based mortality:

```python
    # Catch-based fishing — proportional allocation (Java CatchesBySeason* variants)
    if config.fishing_catches is not None:
        step_in_year = step % config.n_dt_per_year
        year = step // config.n_dt_per_year

        for sp_i in range(config.n_species):
            sp_mask = sp == sp_i

            # Determine annual catch target
            annual_catch = config.fishing_catches[sp_i]
            if config.fishing_catches_by_year is not None:
                ts = config.fishing_catches_by_year[sp_i]
                if ts is not None and year < len(ts.values):
                    annual_catch = ts.get(year)

            if annual_catch <= 0:
                continue

            # Season weight (default uniform)
            if config.fishing_catches_season is not None:
                season_w = config.fishing_catches_season[sp_i, step_in_year]
            else:
                season_w = 1.0 / config.n_dt_per_year

            # Compute fishable biomass (selectivity-weighted)
            fishable = (state.abundance * state.weight * selectivity)[sp_mask]
            total_fishable = fishable.sum()

            if total_fishable <= 0:
                continue

            # Proportional allocation per school
            catch_this_step = annual_catch * season_w / n_subdt
            school_catch = (fishable / total_fishable) * catch_this_step

            # Convert biomass catch to abundance
            school_weights = state.weight[sp_mask]
            n_dead_catch = np.where(
                school_weights > 0, school_catch / school_weights, 0.0
            )
            # Cap at available abundance
            n_dead_catch = np.minimum(n_dead_catch, state.abundance[sp_mask])

            n_dead_total[sp_mask] = n_dead_catch

        # Skip rate-based path
        n_dead_total[state.is_background] = 0.0
        n_dead_total[state.age_dt < state.first_feeding_age_dt] = 0.0

        new_n_dead = state.n_dead.copy()
        if config.fishing_discard_rate is not None:
            discard_rate = config.fishing_discard_rate[sp]
            n_discarded = n_dead_total * discard_rate
            n_landed = n_dead_total - n_discarded
            new_n_dead[:, MortalityCause.FISHING] += n_landed
            new_n_dead[:, MortalityCause.DISCARDS] += n_discarded
        else:
            new_n_dead[:, MortalityCause.FISHING] += n_dead_total

        new_abundance = state.abundance - n_dead_total
        new_biomass = new_abundance * state.weight
        return state.replace(abundance=new_abundance, biomass=new_biomass, n_dead=new_n_dead)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_fishing_variants.py -v`
Expected: All tests PASS

- [ ] **Step 6: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All existing tests still pass

- [ ] **Step 7: Lint and commit**

```bash
git add osmose/engine/processes/fishing.py osmose/engine/config.py tests/test_engine_fishing_variants.py
git commit -m "feat(engine): add catch-based proportional allocation fishing variants"
```

---

### Task 4: Selectivity Type Integration in fishing_mortality()

Wire the new selectivity types (2=Gaussian, 3=log-normal) into the existing `fishing_mortality()` function and add L75 config field.

**Files:**
- Modify: `osmose/engine/processes/fishing.py`
- Modify: `osmose/engine/config.py`
- Modify: `tests/test_engine_fishing_variants.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_engine_fishing_variants.py`:

```python
class TestSelectivityIntegration:
    """Selectivity types 2 and 3 wired into fishing_mortality."""

    def test_gaussian_selectivity_in_fishing(self) -> None:
        """Type 2 (Gaussian) reduces fishing on small/large fish."""
        from unittest.mock import MagicMock

        config = MagicMock(spec=EngineConfig)
        config.fishing_enabled = True
        config.n_species = 1
        config.n_dt_per_year = 24
        config.fishing_rate = np.array([1.0])  # high rate
        config.fishing_rate_by_year = None
        config.fishing_seasonality = None
        config.fishing_rate_by_dt_by_class = None
        config.fishing_catches = None
        config.fishing_selectivity_type = np.array([2], dtype=np.int32)  # Gaussian
        config.fishing_selectivity_a50 = np.array([np.nan])
        config.fishing_selectivity_l50 = np.array([20.0])
        config.fishing_selectivity_l75 = np.array([25.0])
        config.fishing_selectivity_slope = np.array([0.0])
        config.fishing_spatial_maps = [None]
        config.mpa_zones = None
        config.fishing_discard_rate = None

        # Two schools: one at L50 (peak), one far from it
        state = SchoolState.create(n_schools=2, species_id=np.array([0, 0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0, 1000.0]),
            weight=np.array([0.01, 0.01]),
            length=np.array([20.0, 5.0]),  # at peak vs far below
            age_dt=np.array([48, 48], dtype=np.int32),
            cell_x=np.array([0, 0], dtype=np.int32),
            cell_y=np.array([0, 0], dtype=np.int32),
        )

        result = fishing_mortality(state, config, n_subdt=1, step=0)
        dead_at_peak = result.n_dead[0, MortalityCause.FISHING]
        dead_far = result.n_dead[1, MortalityCause.FISHING]
        assert dead_at_peak > dead_far  # More fishing at L50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_fishing_variants.py::TestSelectivityIntegration -v`
Expected: FAIL — `config has no attribute 'fishing_selectivity_l75'`

- [ ] **Step 3: Add L75 field and wire selectivity types 2-3**

In `osmose/engine/config.py`, add after `fishing_selectivity_slope`:

```python
    fishing_selectivity_l75: NDArray[np.float64]  # length at 75% selectivity (for Gaussian/log-normal)
```

In `osmose/engine/processes/fishing.py`, replace the existing selectivity block with an expanded version that handles all 4 types:

```python
    from osmose.engine.processes.selectivity import knife_edge, sigmoid, gaussian, log_normal

    sel_type = config.fishing_selectivity_type[sp]
    l50 = config.fishing_selectivity_l50[sp]
    l75 = config.fishing_selectivity_l75[sp]
    a50 = config.fishing_selectivity_a50[sp]

    selectivity = np.ones(len(state), dtype=np.float64)
    for sp_i in range(config.n_species):
        sp_mask = sp == sp_i
        if not sp_mask.any():
            continue
        t = int(sel_type[sp_mask][0]) if sp_mask.any() else -1
        lengths = state.length[sp_mask]

        if t == 0:
            # Age-based knife-edge
            age_years = state.age_dt[sp_mask].astype(np.float64) / config.n_dt_per_year
            selectivity[sp_mask] = np.where(age_years >= a50[sp_mask][0], 1.0, 0.0)
        elif t == 1:
            # Sigmoid
            selectivity[sp_mask] = sigmoid(lengths, l50[sp_mask][0], config.fishing_selectivity_slope[sp_mask][0])
        elif t == 2:
            # Gaussian
            selectivity[sp_mask] = gaussian(lengths, l50[sp_mask][0], l75[sp_mask][0])
        elif t == 3:
            # Log-normal
            selectivity[sp_mask] = log_normal(lengths, l50[sp_mask][0], l75[sp_mask][0])
        else:
            # Default: length knife-edge (if l50 > 0)
            l50_val = l50[sp_mask][0]
            selectivity[sp_mask] = knife_edge(lengths, l50_val) if l50_val > 0 else 1.0
```

- [ ] **Step 4: Parse L75 from config**

In the config loading section, add L75 parsing alongside existing L50:

```python
    focal_fishing_selectivity_l75 = _species_float_optional(
        cfg, n, "fisheries.selectivity.l75.fsh{i}", 0.0
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_fishing_variants.py -v`
Expected: All tests PASS

- [ ] **Step 6: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All existing tests pass (existing selectivity tests still work)

- [ ] **Step 7: Lint and commit**

```bash
git add osmose/engine/processes/fishing.py osmose/engine/processes/selectivity.py osmose/engine/config.py tests/test_engine_fishing_variants.py
git commit -m "feat(engine): wire Gaussian/log-normal selectivity into fishing mortality"
```

---

### Task 5: Config Scenario Dispatch

Java auto-detects the fishing scenario per species from config keys. Python should do the same — detect which variant is configured and set fields accordingly.

**Files:**
- Modify: `osmose/engine/config.py`
- Modify: `tests/test_engine_fishing_variants.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_engine_fishing_variants.py`:

```python
class TestFishingScenarioDispatch:
    """Config auto-detects fishing scenario from keys."""

    def test_rate_annual_detected(self) -> None:
        """mortality.fishing.rate.sp0 → rate-based."""
        from osmose.engine.config import detect_fishing_scenario

        config = {"mortality.fishing.rate.sp0": "0.3"}
        scenario = detect_fishing_scenario(config, 0)
        assert scenario == "rate_annual"

    def test_rate_by_year_detected(self) -> None:
        config = {"mortality.fishing.rate.byYear.file.sp0": "/path/to/file.csv"}
        scenario = detect_fishing_scenario(config, 0)
        assert scenario == "rate_by_year"

    def test_catches_annual_detected(self) -> None:
        config = {"mortality.fishing.catches.sp0": "1000"}
        scenario = detect_fishing_scenario(config, 0)
        assert scenario == "catches_annual"

    def test_rate_by_dt_by_age_detected(self) -> None:
        config = {"mortality.fishing.rate.byDt.byAge.file.sp0": "/path.csv"}
        scenario = detect_fishing_scenario(config, 0)
        assert scenario == "rate_by_dt_by_class"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_fishing_variants.py::TestFishingScenarioDispatch -v`
Expected: FAIL — `cannot import name 'detect_fishing_scenario'`

- [ ] **Step 3: Implement scenario detection**

Add to `osmose/engine/config.py`:

```python
# Fishing scenario config key patterns (matches Java FishingMortality.Scenario enum)
_FISHING_SCENARIOS = [
    ("rate_annual", "mortality.fishing.rate.sp"),
    ("rate_by_year", "mortality.fishing.rate.byYear.file.sp"),
    ("rate_by_dt_by_class", "mortality.fishing.rate.byDt.byAge.file.sp"),
    ("rate_by_dt_by_class", "mortality.fishing.rate.byDt.bySize.file.sp"),
    ("catches_annual", "mortality.fishing.catches.sp"),
    ("catches_by_year", "mortality.fishing.catches.byYear.file.sp"),
    ("catches_by_dt_by_class", "mortality.fishing.catches.byDt.byAge.file.sp"),
    ("catches_by_dt_by_class", "mortality.fishing.catches.byDt.bySize.file.sp"),
]


def detect_fishing_scenario(config: dict[str, str], species_idx: int) -> str | None:
    """Detect fishing scenario for a species from config keys.

    Matches Java FishingMortality.findScenario(). Returns scenario name or None.
    """
    for scenario_name, key_prefix in _FISHING_SCENARIOS:
        if f"{key_prefix}{species_idx}" in config:
            return scenario_name
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_fishing_variants.py::TestFishingScenarioDispatch -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All tests pass

- [ ] **Step 6: Lint and commit**

```bash
git add osmose/engine/config.py tests/test_engine_fishing_variants.py
git commit -m "feat(engine): add fishing scenario auto-detection from config keys"
```
