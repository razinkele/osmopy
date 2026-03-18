# Python Engine Phase 2: Growth + Natural Mortality + Aging — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Von Bertalanffy growth (with young-of-year linear phase, predation-gated growth factor, L_max cap), weight-length allometric conversion, natural (additional) mortality, and aging mortality. This is the first phase that adds real simulation dynamics.

**Architecture:** Process functions in `osmose/engine/processes/` replace the Phase 1 stubs in `simulate.py`. Each function takes `(state, config, ...) -> state`. The simulation loop in `simulate.py` calls these instead of the stubs.

**Tech Stack:** Python 3.12+, NumPy (vectorized array operations). Tests with pytest.

**Spec:** `docs/superpowers/specs/2026-03-18-python-engine-design.md` (Growth Algorithm section, lines 327-357; Aging Mortality, lines 201-215)

---

## File Structure

```
osmose/engine/processes/
    growth.py               # NEW — Von Bertalanffy + Gompertz growth functions
    natural.py              # NEW — additional mortality, aging mortality
osmose/engine/config.py     # MODIFY — add delta_lmax_factor, additional mortality rate
osmose/engine/simulate.py   # MODIFY — replace _growth and _aging_mortality stubs
tests/
    test_engine_growth.py   # NEW — analytical growth verification (Tier 1)
    test_engine_mortality.py # NEW — natural + aging mortality tests (Tier 1)
```

---

### Task 1: Add missing config parameters to EngineConfig

**Files:**
- Modify: `osmose/engine/config.py`
- Modify: `tests/test_engine_config.py`

Phase 2 needs parameters not yet extracted: `delta_lmax_factor`, `additional_mortality_rate`. Also need `lmax` (maximum length) which may or may not be in config.

- [ ] **Step 1: Write failing tests for new config fields**

Append to `tests/test_engine_config.py`:

```python
    def test_delta_lmax_factor(self, minimal_config):
        minimal_config["species.delta.lmax.factor.sp0"] = "2.0"
        minimal_config["species.delta.lmax.factor.sp1"] = "1.8"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.delta_lmax_factor[0] == pytest.approx(2.0)
        assert cfg.delta_lmax_factor[1] == pytest.approx(1.8)

    def test_delta_lmax_factor_default(self, minimal_config):
        """delta_lmax_factor defaults to 2.0 when not specified."""
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.delta_lmax_factor[0] == pytest.approx(2.0)
        assert cfg.delta_lmax_factor[1] == pytest.approx(2.0)

    def test_additional_mortality_rate(self, minimal_config):
        minimal_config["mortality.additional.rate.sp0"] = "0.2"
        minimal_config["mortality.additional.rate.sp1"] = "0.15"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.additional_mortality_rate[0] == pytest.approx(0.2)
        assert cfg.additional_mortality_rate[1] == pytest.approx(0.15)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_config.py -v -k "delta_lmax or additional_mortality"`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Add new fields to EngineConfig**

In `osmose/engine/config.py`, add fields to the dataclass and `from_dict()`:

New dataclass fields (after `critical_success_rate`):
```python
    # Growth
    delta_lmax_factor: NDArray[np.float64]      # max growth scaling factor (default 2.0)

    # Natural mortality
    additional_mortality_rate: NDArray[np.float64]  # annual additional mortality rate per species
```

New extraction in `from_dict()` — add a helper for optional species parameters:

```python
def _species_float_optional(
    cfg: dict[str, str], pattern: str, n: int, default: float
) -> NDArray[np.float64]:
    """Extract a per-species float array, using default if key is missing."""
    return np.array(
        [float(cfg.get(pattern.format(i=i), str(default))) for i in range(n)]
    )
```

Then in `from_dict()` return statement, add:
```python
            delta_lmax_factor=_species_float_optional(
                cfg, "species.delta.lmax.factor.sp{i}", n_sp, default=2.0
            ),
            additional_mortality_rate=_species_float_optional(
                cfg, "mortality.additional.rate.sp{i}", n_sp, default=0.0
            ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_config.py -v`
Expected: All PASSED (including 3 new)

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/config.py tests/test_engine_config.py
git commit -m "feat(engine): add delta_lmax_factor and additional_mortality_rate to EngineConfig"
```

---

### Task 2: Von Bertalanffy expected length function

**Files:**
- Create: `osmose/engine/processes/growth.py`
- Create: `tests/test_engine_growth.py`

- [ ] **Step 1: Write failing tests for expected length**

Create `tests/test_engine_growth.py`:

```python
"""Tests for growth process functions — Tier 1 analytical verification."""

import numpy as np
import pytest

from osmose.engine.processes.growth import expected_length_vb


class TestExpectedLengthVB:
    """Verify Von Bertalanffy expected length against known formula."""

    def test_age_zero_returns_egg_size(self):
        """At age 0, expected length = L_egg."""
        result = expected_length_vb(
            age_dt=np.array([0]),
            linf=np.array([30.0]),
            k=np.array([0.3]),
            t0=np.array([-0.1]),
            egg_size=np.array([0.1]),
            vb_threshold_age=np.array([1.0]),
            n_dt_per_year=24,
        )
        np.testing.assert_allclose(result, [0.1], atol=1e-10)

    def test_young_of_year_linear(self):
        """Below threshold age, growth is linear interpolation."""
        # At half the threshold age, should be halfway between egg and threshold length
        linf, k, t0, a_thres = 30.0, 0.3, -0.1, 1.0
        l_thres = linf * (1 - np.exp(-k * (a_thres - t0)))
        l_egg = 0.1
        n_dt = 24
        half_age_dt = int(a_thres * n_dt / 2)  # 12 dt = 0.5 years

        result = expected_length_vb(
            age_dt=np.array([half_age_dt]),
            linf=np.array([linf]),
            k=np.array([k]),
            t0=np.array([t0]),
            egg_size=np.array([l_egg]),
            vb_threshold_age=np.array([a_thres]),
            n_dt_per_year=n_dt,
        )
        expected = l_egg + (l_thres - l_egg) * 0.5
        np.testing.assert_allclose(result, [expected], atol=1e-10)

    def test_above_threshold_vb_formula(self):
        """Above threshold age, use standard VB formula."""
        linf, k, t0 = 30.0, 0.3, -0.1
        n_dt = 24
        age_years = 3.0
        age_dt = int(age_years * n_dt)

        result = expected_length_vb(
            age_dt=np.array([age_dt]),
            linf=np.array([linf]),
            k=np.array([k]),
            t0=np.array([t0]),
            egg_size=np.array([0.1]),
            vb_threshold_age=np.array([1.0]),
            n_dt_per_year=n_dt,
        )
        expected = linf * (1 - np.exp(-k * (age_years - t0)))
        np.testing.assert_allclose(result, [expected], atol=1e-10)

    def test_vectorized_multiple_species(self):
        """Multiple schools with different species params in one call."""
        n_dt = 24
        result = expected_length_vb(
            age_dt=np.array([48, 48]),       # both 2 years old
            linf=np.array([30.0, 50.0]),
            k=np.array([0.3, 0.2]),
            t0=np.array([-0.1, -0.2]),
            egg_size=np.array([0.1, 0.2]),
            vb_threshold_age=np.array([1.0, 1.0]),
            n_dt_per_year=n_dt,
        )
        age = 2.0
        e0 = 30.0 * (1 - np.exp(-0.3 * (age - (-0.1))))
        e1 = 50.0 * (1 - np.exp(-0.2 * (age - (-0.2))))
        np.testing.assert_allclose(result, [e0, e1], atol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_growth.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `expected_length_vb` in `osmose/engine/processes/growth.py`**

```python
"""Growth process functions for the OSMOSE Python engine.

Von Bertalanffy and Gompertz growth, with predation-success gating.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def expected_length_vb(
    age_dt: NDArray[np.int32],
    linf: NDArray[np.float64],
    k: NDArray[np.float64],
    t0: NDArray[np.float64],
    egg_size: NDArray[np.float64],
    vb_threshold_age: NDArray[np.float64],
    n_dt_per_year: int,
) -> NDArray[np.float64]:
    """Compute Von Bertalanffy expected length at a given age.

    Three phases:
      age == 0:             L_egg
      0 < age < a_thres:    linear interpolation from L_egg to L_thres
      age >= a_thres:        L_inf * (1 - exp(-K * (age - t0)))
    """
    age_years = age_dt.astype(np.float64) / n_dt_per_year
    threshold_years = vb_threshold_age

    # Standard VB formula (used for age >= threshold AND for computing L_thres)
    l_vb = linf * (1 - np.exp(-k * (age_years - t0)))
    l_thres = linf * (1 - np.exp(-k * (threshold_years - t0)))

    # Linear phase for young-of-year
    frac = np.where(threshold_years > 0, age_years / threshold_years, 1.0)
    l_linear = egg_size + (l_thres - egg_size) * frac

    # Select phase
    result = np.where(age_dt == 0, egg_size, np.where(age_years < threshold_years, l_linear, l_vb))
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_growth.py -v`
Expected: All 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/growth.py tests/test_engine_growth.py
git commit -m "feat(engine): add Von Bertalanffy expected length function"
```

---

### Task 3: Growth gating and full growth process function

**Files:**
- Modify: `osmose/engine/processes/growth.py`
- Modify: `tests/test_engine_growth.py`

- [ ] **Step 1: Write failing tests for growth gating**

Append to `tests/test_engine_growth.py`:

```python
from osmose.engine.processes.growth import growth
from osmose.engine.state import SchoolState
from osmose.engine.config import EngineConfig


def _make_growth_config() -> dict[str, str]:
    """Config dict for growth tests: 1 species, VB params."""
    return {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "1",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "5",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
    }


class TestGrowthGating:
    """Verify growth is gated by predation success rate."""

    def test_no_growth_below_critical(self):
        """When pred_success_rate < critical, growth = 0."""
        cfg = EngineConfig.from_dict(_make_growth_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            length=np.array([10.0]),
            weight=np.array([6.0]),
            age_dt=np.array([48], dtype=np.int32),  # 2 years
            abundance=np.array([100.0]),
            biomass=np.array([600.0]),
            pred_success_rate=np.array([0.3]),  # below 0.57 critical
        )
        rng = np.random.default_rng(42)
        new_state = growth(state, cfg, rng)
        # Length should not change
        np.testing.assert_allclose(new_state.length, [10.0], atol=1e-10)

    def test_max_growth_at_full_success(self):
        """When pred_success_rate = 1.0, growth = max_delta."""
        cfg = EngineConfig.from_dict(_make_growth_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            length=np.array([10.0]),
            weight=np.array([6.0]),
            age_dt=np.array([48], dtype=np.int32),  # 2 years
            abundance=np.array([100.0]),
            biomass=np.array([600.0]),
            pred_success_rate=np.array([1.0]),
        )
        rng = np.random.default_rng(42)
        new_state = growth(state, cfg, rng)
        # Should have grown
        assert new_state.length[0] > 10.0

    def test_egg_always_grows(self):
        """Eggs (age_dt == 0) get linear growth regardless of pred success."""
        cfg = EngineConfig.from_dict(_make_growth_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            length=np.array([0.1]),  # egg size
            weight=np.array([0.000006]),
            age_dt=np.array([0], dtype=np.int32),
            abundance=np.array([1000.0]),
            biomass=np.array([0.006]),
            pred_success_rate=np.array([0.0]),  # zero success
        )
        rng = np.random.default_rng(42)
        new_state = growth(state, cfg, rng)
        assert new_state.length[0] > 0.1  # grew despite zero success

    def test_weight_updated_after_growth(self):
        """Weight must be recalculated from new length via W = c * L^b."""
        cfg = EngineConfig.from_dict(_make_growth_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            length=np.array([10.0]),
            weight=np.array([6.0]),
            age_dt=np.array([48], dtype=np.int32),
            abundance=np.array([100.0]),
            biomass=np.array([600.0]),
            pred_success_rate=np.array([1.0]),
        )
        rng = np.random.default_rng(42)
        new_state = growth(state, cfg, rng)
        c = 0.006
        b = 3.0
        expected_weight = c * new_state.length[0] ** b
        np.testing.assert_allclose(new_state.weight[0], expected_weight, rtol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_growth.py::TestGrowthGating -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `growth()` function**

Add to `osmose/engine/processes/growth.py`:

```python
from osmose.engine.config import EngineConfig
from osmose.engine.state import SchoolState


def growth(state: SchoolState, config: EngineConfig, rng: np.random.Generator) -> SchoolState:
    """Apply Von Bertalanffy growth gated by predation success.

    Special cases:
    - Eggs (age_dt == 0): always get mean delta_L (bypass gating)
    - Out-of-domain schools: always get mean delta_L
    """
    if len(state) == 0:
        return state

    sp = state.species_id
    n_dt = config.n_dt_per_year

    # Expected length at current and next age
    l_current = expected_length_vb(
        state.age_dt, config.linf[sp], config.k[sp], config.t0[sp],
        config.egg_size[sp], config.vb_threshold_age[sp], n_dt,
    )
    l_next = expected_length_vb(
        state.age_dt + 1, config.linf[sp], config.k[sp], config.t0[sp],
        config.egg_size[sp], config.vb_threshold_age[sp], n_dt,
    )
    delta_l = l_next - l_current  # mean length increment

    # Growth factor gated by predation success
    csr = config.critical_success_rate[sp]
    sr = state.pred_success_rate
    max_delta = config.delta_lmax_factor[sp] * delta_l

    # Gated growth: 0 below critical, linear scaling above
    growth_factor = np.where(
        sr >= csr,
        max_delta * (sr - csr) / np.where(csr < 1.0, 1.0 - csr, 1.0),
        0.0,
    )

    # Special cases: eggs and out-of-domain always get mean delta_L
    bypass = (state.age_dt == 0) | state.is_out
    growth_factor = np.where(bypass, delta_l, growth_factor)

    # Apply growth, cap at L_inf (as proxy for L_max)
    new_length = np.minimum(state.length + growth_factor, config.linf[sp])

    # Update weight from new length: W = c * L^b
    new_weight = config.condition_factor[sp] * new_length ** config.allometric_power[sp]

    # Update biomass
    new_biomass = state.abundance * new_weight

    return state.replace(length=new_length, weight=new_weight, biomass=new_biomass)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_growth.py -v`
Expected: All 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/growth.py tests/test_engine_growth.py
git commit -m "feat(engine): add growth function with predation-success gating"
```

---

### Task 4: Natural (additional) mortality

**Files:**
- Create: `osmose/engine/processes/natural.py`
- Create: `tests/test_engine_mortality.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_engine_mortality.py`:

```python
"""Tests for natural and aging mortality — Tier 1 analytical verification."""

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.processes.natural import additional_mortality, aging_mortality
from osmose.engine.state import MortalityCause, SchoolState


def _make_mortality_config() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "1",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "30.0",
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
        "mortality.additional.rate.sp0": "0.2",
    }


class TestAdditionalMortality:
    def test_mortality_reduces_abundance(self):
        """N_dead = N * (1 - exp(-M / n_subdt))."""
        cfg = EngineConfig.from_dict(_make_mortality_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            biomass=np.array([6000.0]),
            weight=np.array([6.0]),
        )
        n_subdt = 10
        new_state = additional_mortality(state, cfg, n_subdt)
        m = 0.2  # annual rate
        expected_dead = 1000.0 * (1 - np.exp(-m / n_subdt))
        actual_dead = new_state.n_dead[0, MortalityCause.ADDITIONAL]
        np.testing.assert_allclose(actual_dead, expected_dead, rtol=1e-10)
        np.testing.assert_allclose(
            new_state.abundance[0], 1000.0 - expected_dead, rtol=1e-10
        )

    def test_zero_rate_no_mortality(self):
        cfg_dict = _make_mortality_config()
        cfg_dict["mortality.additional.rate.sp0"] = "0.0"
        cfg = EngineConfig.from_dict(cfg_dict)
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(abundance=np.array([1000.0]))
        new_state = additional_mortality(state, cfg, n_subdt=10)
        np.testing.assert_allclose(new_state.abundance[0], 1000.0)


class TestAgingMortality:
    def test_kills_old_schools(self):
        """Schools at lifespan - 1 should be killed (pre-increment check)."""
        cfg = EngineConfig.from_dict(_make_mortality_config())
        # lifespan = 3 years * 24 dt = 72 dt. Aging kills at age_dt >= 71.
        state = SchoolState.create(n_schools=3, species_id=np.zeros(3, dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0, 100.0, 100.0]),
            age_dt=np.array([70, 71, 72], dtype=np.int32),
        )
        new_state = aging_mortality(state, cfg)
        np.testing.assert_allclose(new_state.abundance, [100.0, 0.0, 0.0])
        assert new_state.n_dead[1, MortalityCause.AGING] == 100.0
        assert new_state.n_dead[2, MortalityCause.AGING] == 100.0

    def test_young_schools_survive(self):
        cfg = EngineConfig.from_dict(_make_mortality_config())
        state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
        state = state.replace(
            abundance=np.array([500.0]),
            age_dt=np.array([24], dtype=np.int32),  # 1 year, well below 3
        )
        new_state = aging_mortality(state, cfg)
        np.testing.assert_allclose(new_state.abundance, [500.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_mortality.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `osmose/engine/processes/natural.py`**

```python
"""Natural mortality processes: additional mortality and aging.

Additional mortality is a fixed annual rate applied per sub-timestep.
Aging mortality kills schools that have reached their species' lifespan.
"""

from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.state import MortalityCause, SchoolState


def additional_mortality(
    state: SchoolState, config: EngineConfig, n_subdt: int
) -> SchoolState:
    """Apply additional (background) mortality.

    N_dead = N * (1 - exp(-M_annual / n_subdt))
    """
    if len(state) == 0:
        return state

    sp = state.species_id
    m_rate = config.additional_mortality_rate[sp]
    mortality_fraction = 1 - np.exp(-m_rate / n_subdt)
    n_dead = state.abundance * mortality_fraction

    new_abundance = state.abundance - n_dead
    new_biomass = new_abundance * state.weight
    new_n_dead = state.n_dead.copy()
    new_n_dead[:, MortalityCause.ADDITIONAL] += n_dead

    return state.replace(abundance=new_abundance, biomass=new_biomass, n_dead=new_n_dead)


def aging_mortality(state: SchoolState, config: EngineConfig) -> SchoolState:
    """Kill schools that have reached their species' lifespan.

    Uses age_dt >= lifespan_dt - 1 because aging runs BEFORE
    reproduction (where age_dt is incremented).
    """
    if len(state) == 0:
        return state

    lifespan_dt = config.lifespan_dt[state.species_id]
    expired = state.age_dt >= lifespan_dt - 1

    if not expired.any():
        return state

    new_n_dead = state.n_dead.copy()
    new_n_dead[expired, MortalityCause.AGING] += state.abundance[expired]

    new_abundance = state.abundance.copy()
    new_abundance[expired] = 0.0
    new_biomass = new_abundance * state.weight

    return state.replace(abundance=new_abundance, biomass=new_biomass, n_dead=new_n_dead)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_mortality.py -v`
Expected: All 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/natural.py tests/test_engine_mortality.py
git commit -m "feat(engine): add additional mortality and aging mortality processes"
```

---

### Task 5: Wire growth and mortality into simulation loop

**Files:**
- Modify: `osmose/engine/simulate.py`

- [ ] **Step 1: Write failing integration test**

Append to `tests/test_engine_simulate.py`:

```python
    def test_aging_mortality_kills_old_schools(self, minimal_config):
        """Schools should die when they reach lifespan."""
        # 1 species, lifespan=3yr, 24 dt/yr => 72 dt lifespan
        # Run for 4 years (96 steps) — schools initialized at age 0 should die
        minimal_config["simulation.time.nyear"] = "4"
        minimal_config["mortality.additional.rate.sp0"] = "0.0"
        cfg = EngineConfig.from_dict(minimal_config)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        # After 4 years, all schools should have been killed by aging
        assert outputs[-1].abundance[0] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_simulate.py::TestSimulate::test_aging_mortality_kills_old_schools -v`
Expected: FAIL (stubs don't kill schools)

- [ ] **Step 3: Replace stubs in `simulate.py`**

Replace `_growth` and `_aging_mortality` stubs:

```python
def _growth(state: SchoolState, config: EngineConfig, rng: np.random.Generator) -> SchoolState:
    """Apply Von Bertalanffy growth gated by predation success."""
    from osmose.engine.processes.growth import growth
    return growth(state, config, rng)


def _aging_mortality(state: SchoolState, config: EngineConfig) -> SchoolState:
    """Kill schools exceeding species lifespan."""
    from osmose.engine.processes.natural import aging_mortality
    return aging_mortality(state, config)
```

Also add `_mortality` to delegate to additional mortality (partial — other sources in Phase 5-6):

```python
def _mortality(
    state: SchoolState,
    resources: ResourceState,
    config: EngineConfig,
    rng: np.random.Generator,
) -> SchoolState:
    """Apply mortality sources. Phase 2: only additional mortality."""
    from osmose.engine.processes.natural import additional_mortality
    for _sub in range(config.mortality_subdt):
        state = additional_mortality(state, config, config.mortality_subdt)
    return state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_simulate.py -v`
Expected: All PASSED

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/simulate.py tests/test_engine_simulate.py
git commit -m "feat(engine): wire growth and mortality processes into simulation loop"
```

---

### Task 6: Lint, format, full verification

- [ ] **Step 1: Run linter**

Run: `.venv/bin/ruff check osmose/engine/ tests/test_engine_*.py`
Expected: No errors

- [ ] **Step 2: Run formatter**

Run: `.venv/bin/ruff format osmose/engine/ tests/test_engine_*.py`

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: All tests pass

- [ ] **Step 4: Commit any lint/format fixes**

```bash
git add -u
git commit -m "style: lint and format engine Phase 2 code"
```

- [ ] **Step 5: Tag Phase 2 milestone**

```bash
git tag -a engine-phase2 -m "Python engine Phase 2: Von Bertalanffy growth, additional mortality, aging mortality"
```
