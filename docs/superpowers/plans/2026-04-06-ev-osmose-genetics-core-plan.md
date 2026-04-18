# Ev-OSMOSE Genetics Phase 2 (Core) Implementation Plan

> **STATUS-COMPLETE (2026-04-19):** All 5 deliverables shipped. `TraitRegistry` is name-agnostic (supports all 4 Java traits: imax, gsi, m0, m1 — verified in `tests/test_genetics_bioen_integration.py:11-73`). Bioenergetics integration wires `trait_overrides` into `_bioen_step` (targets `bioen_i_max` at `simulate.py:283`; `bioen_r` at `:383`; `bioen_m0`/`bioen_m1` at `:384-385`) and `_bioen_reproduction` (`:522-534`, commit `1b3384a`). Neutral loci on `GeneticState.neutral_alleles` + inheritance transmission (commit `d6b6295`). Seeding phase via `genetics_transmission_year` gate (commit `579bf66`). Eight genetics test files (expression, inheritance, neutral, seeding, statistics, trait, integration, bioen_integration) all pass at HEAD post-v0.9.0.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Phase 1 genetics MVP from 1 trait (imax) to all 4 Java traits (imax, gsi, m0, m1) with full bioenergetics integration, neutral loci for drift tracking, and a seeding phase for genetic initialization.

**Architecture:** The existing `TraitRegistry.from_config()` already discovers any trait from config keys. Phase 2 adds: (1) `trait_overrides` consumption in `_bioen_step` and `_bioen_reproduction` — replacing per-species scalars with per-school arrays from phenotypes; (2) neutral loci as a separate integer array on `GeneticState`; (3) seeding vs inheritance switch based on simulation year. All bioen functions already broadcast correctly with array parameters.

**Tech Stack:** Python 3.12+, NumPy, pytest, ruff

**Spec:** `docs/superpowers/specs/2026-04-05-ev-osmose-economic-design.md` (Phase 2: Core, lines 526-534)

**Depends on:** Phase 1 MVP (merged to master)

---

## File Structure

### Modified files

| File | Changes |
|------|---------|
| `osmose/engine/simulate.py` | Pass `trait_overrides` to `_bioen_step` and `_bioen_reproduction`; add seeding year logic to inheritance block |
| `osmose/engine/genetics/genotype.py` | Add `neutral_alleles` field to `GeneticState`; update compact/append/create |
| `osmose/engine/genetics/inheritance.py` | Add `seeding` flag to `create_offspring_genotypes`; handle neutral loci inheritance |
| `osmose/engine/config.py` | Parse `population.genotype.transmission.year.start`, `evolution.neutral.nlocus`, `evolution.neutral.nval` |

### New files

| File | Responsibility |
|------|---------------|
| `tests/test_genetics_bioen_integration.py` | Trait override consumed by bioen, 4-trait simulation |
| `tests/test_genetics_neutral.py` | Neutral loci inheritance, compact sync |
| `tests/test_genetics_seeding.py` | Seeding phase vs inheritance switching |
| `tests/test_genetics_statistics.py` | Hardy-Weinberg, allele drift, trait variance convergence |

---

## Task 1: Wire trait_overrides into _bioen_step

**Files:**
- Modify: `osmose/engine/simulate.py` (~lines 152-324, 960-970)
- Test: `tests/test_genetics_bioen_integration.py`

### Context

`_bioen_step` currently reads `config.bioen_i_max[sp]`, `config.bioen_r[sp]`, `config.bioen_m0[sp]`, `config.bioen_m1[sp]` as per-species scalars inside a `for sp, mask in sp_masks:` loop. When genetics is active, `trait_overrides` has per-school arrays keyed by the target_param name (e.g., `"bioen_i_max"`). We pass `trait_overrides` to `_bioen_step` and use the per-school values when available. The downstream functions (`bioen_ingestion_cap`, `compute_energy_budget`) already handle array parameters via numpy broadcasting.

- [ ] **Step 1: Write test for trait overrides consumed by bioenergetics**

```python
# tests/test_genetics_bioen_integration.py
"""Tests for genetics trait override integration with bioenergetics."""

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import simulate


def _bioen_genetics_config() -> dict[str, str]:
    """Config with bioenergetics AND genetics enabled, 4 evolving traits."""
    return {
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
        # Bioenergetics
        "simulation.bioen.enabled": "true",
        "species.bioen.beta.sp0": "0.8",
        "species.bioen.assimilation.sp0": "0.6",
        "species.bioen.c.m.sp0": "5.258",
        "species.bioen.eta.sp0": "1.0",
        "species.bioen.r.sp0": "0.5",
        "species.bioen.m0.sp0": "10.0",
        "species.bioen.m1.sp0": "0.5",
        "species.bioen.e.mobi.sp0": "0.45",
        "species.bioen.e.d.sp0": "2.46",
        "species.bioen.tp.sp0": "14.0",
        "species.bioen.e.maint.sp0": "0.45",
        "species.bioen.i.max.sp0": "3.5",
        "species.bioen.theta.sp0": "1.0",
        "species.bioen.c.rate.sp0": "0.0",
        "species.bioen.k.for.sp0": "0.0",
        # Genetics — all 4 traits
        "simulation.genetic.enabled": "true",
        "evolution.trait.imax.mean.sp0": "3.5",
        "evolution.trait.imax.var.sp0": "0.1",
        "evolution.trait.imax.envvar.sp0": "0.0",
        "evolution.trait.imax.nlocus.sp0": "5",
        "evolution.trait.imax.nval.sp0": "20",
        "evolution.trait.imax.target": "bioen_i_max",
        "evolution.trait.gsi.mean.sp0": "0.5",
        "evolution.trait.gsi.var.sp0": "0.01",
        "evolution.trait.gsi.envvar.sp0": "0.0",
        "evolution.trait.gsi.nlocus.sp0": "5",
        "evolution.trait.gsi.nval.sp0": "20",
        "evolution.trait.gsi.target": "bioen_r",
        "evolution.trait.m0.mean.sp0": "10.0",
        "evolution.trait.m0.var.sp0": "1.0",
        "evolution.trait.m0.envvar.sp0": "0.0",
        "evolution.trait.m0.nlocus.sp0": "5",
        "evolution.trait.m0.nval.sp0": "20",
        "evolution.trait.m0.target": "bioen_m0",
        "evolution.trait.m1.mean.sp0": "0.5",
        "evolution.trait.m1.var.sp0": "0.01",
        "evolution.trait.m1.envvar.sp0": "0.0",
        "evolution.trait.m1.nlocus.sp0": "5",
        "evolution.trait.m1.nval.sp0": "20",
        "evolution.trait.m1.target": "bioen_m1",
    }


class TestBioenGeneticsIntegration:
    def test_simulation_completes_with_4_traits_and_bioen(self):
        """Full simulation with bioenergetics + all 4 evolving traits should complete."""
        cfg = EngineConfig.from_dict(_bioen_genetics_config())
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert len(outputs) == 12

    def test_trait_overrides_affect_growth(self):
        """With genetic variance in imax, different RNG seeds should produce different biomass."""
        cfg_dict = _bioen_genetics_config()
        cfg = EngineConfig.from_dict(cfg_dict)
        grid = Grid.from_dimensions(ny=3, nx=3)

        outputs_a = simulate(cfg, grid, np.random.default_rng(42))
        outputs_b = simulate(cfg, grid, np.random.default_rng(99))

        # Different seeds → different initial genotypes → different growth trajectories
        # (With zero variance this wouldn't hold)
        biomass_a = outputs_a[-1].biomass[0]
        biomass_b = outputs_b[-1].biomass[0]
        # They should differ (not exactly equal) since genotypes differ
        assert biomass_a != pytest.approx(biomass_b, abs=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_genetics_bioen_integration.py -v`
Expected: FAIL — either TypeError (trait_overrides not passed) or simulation error

- [ ] **Step 3: Add trait_overrides parameter to _bioen_step**

In `osmose/engine/simulate.py`, modify `_bioen_step` signature (line ~152):

```python
def _bioen_step(
    state: SchoolState,
    config: EngineConfig,
    temp_data: PhysicalData | None,
    step: int,
    o2_data: PhysicalData | None = None,
    trait_overrides: dict[str, NDArray[np.float64]] | None = None,
) -> SchoolState:
```

Then in the per-species loop for ingestion cap (line ~210-222), add a helper to resolve per-school values:

```python
    # Helper: resolve parameter as per-school array or species scalar
    def _resolve(param_name: str, sp: int, mask: NDArray[np.bool_]) -> float | NDArray[np.float64]:
        if trait_overrides and param_name in trait_overrides:
            return trait_overrides[param_name][mask]
        return float(getattr(config, param_name)[sp])
```

Replace the hardcoded config reads in the 3 locations:

**Ingestion cap** (line ~213):
```python
        cap = bioen_ingestion_cap(
            weight=state.weight[mask],
            i_max=_resolve("bioen_i_max", sp, mask),
            beta=float(config.bioen_beta[sp]),
            ...
        )
```

**Energy budget** (lines ~310-312):
```python
        dw_sp, dg_sp, en_sp, eg_sp, em_sp, rho_sp = compute_energy_budget(
            ...
            r=_resolve("bioen_r", sp, mask),
            m0=_resolve("bioen_m0", sp, mask),
            m1=_resolve("bioen_m1", sp, mask),
            ...
        )
```

- [ ] **Step 4: Pass trait_overrides through the call chain**

In the main loop (line ~960-970), change the `_bioen_step` call:

```python
        if config.bioen_enabled:
            state = _bioen_step(
                state, config, temp_data, step, o2_data=o2_data,
                trait_overrides=trait_overrides if trait_overrides else None,
            )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_genetics_bioen_integration.py -v`
Expected: 2 PASSED

- [ ] **Step 6: Run existing tests**

Run: `.venv/bin/python -m pytest tests/test_engine_simulate.py tests/test_genetics_integration.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/simulate.py tests/test_genetics_bioen_integration.py
git commit -m "feat(genetics): wire trait overrides into bioenergetics step"
```

---

## Task 2: Wire trait_overrides into _bioen_reproduction

**Files:**
- Modify: `osmose/engine/simulate.py` (~lines 416-495, 975-980)
- Test: `tests/test_genetics_bioen_integration.py` (append)

### Context

`_bioen_reproduction` uses `config.bioen_m0[sp]` and `config.bioen_m1[sp]` to determine maturity. With genetics, per-school m0/m1 come from `trait_overrides`.

- [ ] **Step 1: Write test for trait override in reproduction**

Append to `tests/test_genetics_bioen_integration.py`:

```python
class TestBioenReproductionOverride:
    def test_different_m0_affects_maturity(self):
        """Per-school m0 from genetics should change which schools mature."""
        from osmose.engine.processes.bioen_reproduction import bioen_egg_production

        length = np.array([12.0, 12.0, 12.0])
        age_dt = np.array([24, 24, 24], dtype=np.int32)
        gonad = np.array([0.01, 0.01, 0.01])

        # Scalar m0=10 → all mature (12 > 10 + 0.5*1 = 10.5)
        eggs_scalar = bioen_egg_production(gonad, length, age_dt, m0=10.0, m1=0.5,
                                           egg_weight=1e-6, n_dt_per_year=24)
        assert (eggs_scalar > 0).all()

        # Array m0 → school 2 has high m0, won't mature
        m0_arr = np.array([10.0, 10.0, 20.0])
        eggs_arr = bioen_egg_production(gonad, length, age_dt, m0=m0_arr, m1=0.5,
                                        egg_weight=1e-6, n_dt_per_year=24)
        assert eggs_arr[0] > 0
        assert eggs_arr[2] == 0.0  # 12 < 20 + 0.5*1 = 20.5, not mature
```

- [ ] **Step 2: Run test to verify it passes (bioen functions already broadcast)**

Run: `.venv/bin/python -m pytest tests/test_genetics_bioen_integration.py::TestBioenReproductionOverride -v`
Expected: PASS (numpy broadcasting handles arrays in `m0 + m1 * age_years`)

- [ ] **Step 3: Modify _bioen_reproduction to accept and use trait_overrides**

In `osmose/engine/simulate.py`, modify `_bioen_reproduction` signature:

```python
def _bioen_reproduction(
    state: SchoolState,
    config: EngineConfig,
    step: int,
    rng: np.random.Generator,
    grid_ny: int = 10,
    grid_nx: int = 10,
    trait_overrides: dict[str, NDArray[np.float64]] | None = None,
) -> SchoolState:
```

Inside the per-species loop (lines ~451-452), replace:

```python
        # Before:
        m0=float(config.bioen_m0[sp]),
        m1=float(config.bioen_m1[sp]),

        # After:
        m0=(trait_overrides["bioen_m0"][mask] if trait_overrides and "bioen_m0" in trait_overrides
            else float(config.bioen_m0[sp])),
        m1=(trait_overrides["bioen_m1"][mask] if trait_overrides and "bioen_m1" in trait_overrides
            else float(config.bioen_m1[sp])),
```

And pass `trait_overrides` in the main loop call:

```python
        if config.bioen_enabled:
            state = _bioen_reproduction(
                state, config, step, rng, grid_ny=grid.ny, grid_nx=grid.nx,
                trait_overrides=trait_overrides if trait_overrides else None,
            )
```

- [ ] **Step 4: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_genetics_bioen_integration.py tests/test_genetics_integration.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/simulate.py tests/test_genetics_bioen_integration.py
git commit -m "feat(genetics): wire trait overrides into bioenergetics reproduction"
```

---

## Task 3: Add neutral loci to GeneticState

**Files:**
- Modify: `osmose/engine/genetics/genotype.py`
- Modify: `osmose/engine/genetics/inheritance.py`
- Modify: `osmose/engine/config.py`
- Test: `tests/test_genetics_neutral.py`

### Context

Neutral loci track genetic drift without affecting phenotype. They're stored as integer alleles (indices into a fixed pool of values) in a `neutral_alleles` array on `GeneticState`, shape `(n_schools, n_neutral_loci, 2)`. Config keys: `evolution.neutral.nlocus`, `evolution.neutral.nval`. Neutral loci inherit alongside trait alleles (same meiotic segregation). They're `None` when neutral loci are disabled.

- [ ] **Step 1: Write tests for neutral loci**

```python
# tests/test_genetics_neutral.py
"""Tests for neutral loci: inheritance, compact sync, no phenotype effect."""

import numpy as np

from osmose.engine.genetics.genotype import GeneticState, compact_genetic_state, create_initial_genotypes
from osmose.engine.genetics.inheritance import create_offspring_genotypes
from osmose.engine.genetics.trait import Trait, TraitRegistry


def _make_registry_with_neutral(n_neutral: int = 10, n_neutral_val: int = 50) -> TraitRegistry:
    cfg = {
        "evolution.trait.imax.mean.sp0": "3.5",
        "evolution.trait.imax.var.sp0": "0.1",
        "evolution.trait.imax.envvar.sp0": "0.0",
        "evolution.trait.imax.nlocus.sp0": "3",
        "evolution.trait.imax.nval.sp0": "20",
        "evolution.trait.imax.target": "ingestion_rate",
    }
    return TraitRegistry.from_config(cfg, n_species=1)


class TestNeutralLoci:
    def test_create_with_neutral(self):
        """create_initial_genotypes with n_neutral > 0 should populate neutral_alleles."""
        registry = _make_registry_with_neutral()
        species_id = np.array([0, 0, 0], dtype=np.int32)
        rng = np.random.default_rng(42)
        gs = create_initial_genotypes(registry, species_id, rng, n_neutral=10, n_neutral_val=50)
        assert gs.neutral_alleles is not None
        assert gs.neutral_alleles.shape == (3, 10, 2)
        assert gs.neutral_alleles.dtype == np.int32
        # All values should be in [0, 50)
        assert gs.neutral_alleles.min() >= 0
        assert gs.neutral_alleles.max() < 50

    def test_create_without_neutral(self):
        """create_initial_genotypes with n_neutral=0 should have neutral_alleles=None."""
        registry = _make_registry_with_neutral()
        species_id = np.array([0, 0], dtype=np.int32)
        rng = np.random.default_rng(42)
        gs = create_initial_genotypes(registry, species_id, rng)
        assert gs.neutral_alleles is None

    def test_compact_neutral(self):
        registry = _make_registry_with_neutral()
        rng = np.random.default_rng(42)
        gs = create_initial_genotypes(registry, np.array([0, 0, 0], dtype=np.int32), rng,
                                      n_neutral=5, n_neutral_val=20)
        alive = np.array([True, False, True])
        compacted = compact_genetic_state(gs, alive)
        assert compacted.neutral_alleles.shape == (2, 5, 2)

    def test_append_neutral(self):
        registry = _make_registry_with_neutral()
        rng = np.random.default_rng(42)
        gs1 = create_initial_genotypes(registry, np.array([0, 0], dtype=np.int32), rng,
                                       n_neutral=5, n_neutral_val=20)
        gs2 = create_initial_genotypes(registry, np.array([0], dtype=np.int32), rng,
                                       n_neutral=5, n_neutral_val=20)
        merged = gs1.append(gs2)
        assert merged.neutral_alleles.shape == (3, 5, 2)

    def test_offspring_inherit_neutral(self):
        """Offspring should inherit neutral alleles from parents via meiotic segregation."""
        registry = _make_registry_with_neutral()
        rng = np.random.default_rng(42)
        parent_gs = create_initial_genotypes(registry, np.array([0, 0], dtype=np.int32), rng,
                                             n_neutral=5, n_neutral_val=20)
        offspring_gs = create_offspring_genotypes(
            parent_gs=parent_gs,
            gonad_weight=np.array([50.0, 50.0]),
            species_id=np.array([0, 0], dtype=np.int32),
            offspring_species=0,
            n_offspring=10,
            rng=rng,
        )
        assert offspring_gs.neutral_alleles is not None
        assert offspring_gs.neutral_alleles.shape == (10, 5, 2)
        # All offspring alleles should come from parent pool
        parent_vals = set(parent_gs.neutral_alleles.flatten())
        for val in offspring_gs.neutral_alleles.flatten():
            assert val in parent_vals
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_genetics_neutral.py -v`
Expected: FAIL — `create_initial_genotypes() got an unexpected keyword argument 'n_neutral'`

- [ ] **Step 3: Add neutral_alleles to GeneticState**

In `osmose/engine/genetics/genotype.py`, modify `GeneticState`:

```python
@dataclass
class GeneticState:
    """Per-school genetic data, parallel to SchoolState."""

    alleles: dict[str, NDArray[np.float64]]
    env_noise: dict[str, NDArray[np.float64]]
    registry: TraitRegistry
    neutral_alleles: NDArray[np.int32] | None = None

    def append(self, other: GeneticState) -> GeneticState:
        new_alleles = {}
        new_noise = {}
        for name in self.alleles:
            new_alleles[name] = np.concatenate([self.alleles[name], other.alleles[name]], axis=0)
            new_noise[name] = np.concatenate([self.env_noise[name], other.env_noise[name]], axis=0)
        neutral = None
        if self.neutral_alleles is not None and other.neutral_alleles is not None:
            neutral = np.concatenate([self.neutral_alleles, other.neutral_alleles], axis=0)
        return GeneticState(alleles=new_alleles, env_noise=new_noise, registry=self.registry,
                            neutral_alleles=neutral)
```

Update `compact_genetic_state`:

```python
def compact_genetic_state(gs: GeneticState, alive_mask: NDArray[np.bool_]) -> GeneticState:
    new_alleles = {name: arr[alive_mask] for name, arr in gs.alleles.items()}
    new_noise = {name: arr[alive_mask] for name, arr in gs.env_noise.items()}
    neutral = gs.neutral_alleles[alive_mask] if gs.neutral_alleles is not None else None
    return GeneticState(alleles=new_alleles, env_noise=new_noise, registry=gs.registry,
                        neutral_alleles=neutral)
```

Update `create_initial_genotypes` to accept `n_neutral` and `n_neutral_val`:

```python
def create_initial_genotypes(
    registry: TraitRegistry,
    species_id: NDArray[np.int32],
    rng: np.random.Generator,
    n_neutral: int = 0,
    n_neutral_val: int = 50,
) -> GeneticState:
    n_schools = len(species_id)
    alleles: dict[str, NDArray[np.float64]] = {}
    env_noise: dict[str, NDArray[np.float64]] = {}

    for name, trait in registry.traits.items():
        max_loci = int(trait.n_loci.max())
        arr = np.zeros((n_schools, max_loci, 2), dtype=np.float64)
        noise = np.zeros(n_schools, dtype=np.float64)

        for i in range(n_schools):
            sp = species_id[i]
            n_loc = int(trait.n_loci[sp])
            pool = trait.allele_pool[sp]
            for loc in range(n_loc):
                arr[i, loc, :] = rng.choice(pool[loc], size=2, replace=True)
            if trait.env_var[sp] > 0:
                noise[i] = rng.normal(0.0, np.sqrt(trait.env_var[sp]))

        alleles[name] = arr
        env_noise[name] = noise

    # Neutral loci: random integer alleles
    neutral = None
    if n_neutral > 0:
        neutral = rng.integers(0, n_neutral_val, size=(n_schools, n_neutral, 2), dtype=np.int32)

    return GeneticState(alleles=alleles, env_noise=env_noise, registry=registry,
                        neutral_alleles=neutral)
```

- [ ] **Step 4: Add neutral inheritance to create_offspring_genotypes**

In `osmose/engine/genetics/inheritance.py`, modify `create_offspring_genotypes` to handle neutral loci:

After the trait allele loop, add:

```python
    # Neutral loci inheritance
    neutral = None
    if parent_gs.neutral_alleles is not None:
        n_neutral = parent_gs.neutral_alleles.shape[1]
        off_neutral = np.zeros((n_offspring, n_neutral, 2), dtype=np.int32)
        if len(sp_indices) > 0:
            sp_neutral = parent_gs.neutral_alleles[sp_indices]
            for j in range(n_offspring):
                pa, pb = select_parents(gonad_weight[sp_indices], rng)
                for loc in range(n_neutral):
                    off_neutral[j, loc, 0] = sp_neutral[pa, loc, rng.integers(2)]
                    off_neutral[j, loc, 1] = sp_neutral[pb, loc, rng.integers(2)]
        neutral = off_neutral

    return GeneticState(alleles=alleles, env_noise=env_noise, registry=parent_gs.registry,
                        neutral_alleles=neutral)
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_genetics_neutral.py tests/test_genetics_trait.py -v`
Expected: All PASS (existing tests still pass since neutral defaults to None)

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/genetics/genotype.py osmose/engine/genetics/inheritance.py tests/test_genetics_neutral.py
git commit -m "feat(genetics): add neutral loci with inheritance and compact sync"
```

---

## Task 4: Seeding phase (random alleles before transmission year)

**Files:**
- Modify: `osmose/engine/genetics/inheritance.py`
- Modify: `osmose/engine/config.py`
- Modify: `osmose/engine/simulate.py`
- Test: `tests/test_genetics_seeding.py`

### Context

Before `population.genotype.transmission.year.start`, offspring get random alleles from the allele pool instead of inheriting from parents. This allows the genetic system to equilibrate before selection pressure acts. The seeding flag is passed from `simulate.py` based on the current year.

- [ ] **Step 1: Write tests for seeding phase**

```python
# tests/test_genetics_seeding.py
"""Tests for seeding phase: random alleles before transmission year."""

import numpy as np

from osmose.engine.genetics.genotype import GeneticState, create_initial_genotypes
from osmose.engine.genetics.inheritance import create_offspring_genotypes
from osmose.engine.genetics.trait import Trait, TraitRegistry


def _make_registry() -> TraitRegistry:
    registry = TraitRegistry()
    registry.register(Trait(
        name="imax",
        species_mean=np.array([0.0]),
        species_var=np.array([1.0]),
        env_var=np.array([0.0]),
        n_loci=np.array([3], dtype=np.int32),
        n_alleles=np.array([20], dtype=np.int32),
        target_param="ingestion_rate",
        allele_pool=[[np.arange(20, dtype=np.float64) for _ in range(3)]],
    ))
    return registry


class TestSeedingPhase:
    def test_seeding_uses_pool_not_parents(self):
        """In seeding mode, offspring alleles come from pool, not parents."""
        registry = _make_registry()
        # Parents with alleles all == 99 (not in pool)
        parent_alleles = np.full((2, 3, 2), 99.0)
        parent_gs = GeneticState(
            alleles={"imax": parent_alleles},
            env_noise={"imax": np.zeros(2)},
            registry=registry,
        )

        rng = np.random.default_rng(42)
        offspring = create_offspring_genotypes(
            parent_gs=parent_gs,
            gonad_weight=np.array([50.0, 50.0]),
            species_id=np.array([0, 0], dtype=np.int32),
            offspring_species=0,
            n_offspring=10,
            rng=rng,
            seeding=True,  # <-- seeding mode
        )
        # None of the offspring alleles should be 99 (from parents)
        # They should come from the pool (0-19)
        assert not np.any(offspring.alleles["imax"] == 99.0)
        assert offspring.alleles["imax"].max() < 20.0

    def test_non_seeding_inherits_from_parents(self):
        """Without seeding, offspring alleles come from parents."""
        registry = _make_registry()
        parent_alleles = np.full((2, 3, 2), 99.0)
        parent_gs = GeneticState(
            alleles={"imax": parent_alleles},
            env_noise={"imax": np.zeros(2)},
            registry=registry,
        )

        rng = np.random.default_rng(42)
        offspring = create_offspring_genotypes(
            parent_gs=parent_gs,
            gonad_weight=np.array([50.0, 50.0]),
            species_id=np.array([0, 0], dtype=np.int32),
            offspring_species=0,
            n_offspring=10,
            rng=rng,
            seeding=False,
        )
        # All alleles should be 99 (from parents)
        assert np.all(offspring.alleles["imax"] == 99.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_genetics_seeding.py -v`
Expected: FAIL — `create_offspring_genotypes() got an unexpected keyword argument 'seeding'`

- [ ] **Step 3: Add seeding parameter to create_offspring_genotypes**

In `osmose/engine/genetics/inheritance.py`, add `seeding: bool = False` parameter:

```python
def create_offspring_genotypes(
    parent_gs: GeneticState,
    gonad_weight: NDArray[np.float64],
    species_id: NDArray[np.int32],
    offspring_species: int,
    n_offspring: int,
    rng: np.random.Generator,
    seeding: bool = False,
) -> GeneticState:
```

In the trait allele loop, when `seeding=True`, draw from pool instead of inheriting:

```python
        if seeding:
            # Seeding phase: random alleles from pool
            pool = parent_gs.registry.traits[name].allele_pool[offspring_species]
            for j in range(n_offspring):
                for loc in range(n_loc):
                    off_alleles[j, loc, :] = rng.choice(pool[loc], size=2, replace=True)
        elif len(sp_indices) > 0:
            # Inheritance phase: gametes from parents
            sp_gonad = gonad_weight[sp_indices]
            sp_alleles = parent_gs.alleles[name][sp_indices]
            for j in range(n_offspring):
                pa, pb = select_parents(sp_gonad, rng)
                off_alleles[j, :n_loc, 0] = form_gamete(sp_alleles[pa, :n_loc, :], rng)
                off_alleles[j, :n_loc, 1] = form_gamete(sp_alleles[pb, :n_loc, :], rng)
```

- [ ] **Step 4: Add config parsing for transmission year start**

In `osmose/engine/config.py`, add after `genetics_enabled`:

```python
    # Seeding phase: random genotypes before this year, inheritance after
    genetics_transmission_year: int = 0
```

In `from_dict()`:

```python
        genetics_transmission_year = int(
            cfg.get("population.genotype.transmission.year.start", "0")
        )
```

- [ ] **Step 5: Wire seeding flag in simulate.py**

In the inheritance block (around line 983), compute the seeding flag:

```python
        if ctx.genetic_state is not None:
            from osmose.engine.genetics import create_offspring_genotypes

            current_year = step // config.n_dt_per_year
            seeding = current_year < config.genetics_transmission_year

            n_new = len(state) - n_before_repro
            if n_new > 0:
                new_ids = state.species_id[-n_new:]
                offspring_parts: list = []
                for sp in np.unique(new_ids):
                    sp_mask = new_ids == sp
                    n_off = int(sp_mask.sum())
                    offspring_parts.append(create_offspring_genotypes(
                        parent_gs=ctx.genetic_state,
                        gonad_weight=state.gonad_weight[:len(state) - n_new],
                        species_id=state.species_id[:len(state) - n_new],
                        offspring_species=int(sp),
                        n_offspring=n_off,
                        rng=rng,
                        seeding=seeding,
                    ))
                for part in offspring_parts:
                    ctx.genetic_state = ctx.genetic_state.append(part)
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/test_genetics_seeding.py tests/test_genetics_inheritance.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/genetics/inheritance.py osmose/engine/config.py osmose/engine/simulate.py tests/test_genetics_seeding.py
git commit -m "feat(genetics): add seeding phase for random alleles before transmission year"
```

---

## Task 5: Parse neutral loci config and wire into simulate.py

**Files:**
- Modify: `osmose/engine/config.py`
- Modify: `osmose/engine/simulate.py`
- No new test file (covered by Task 3 unit tests + existing integration tests)

### Context

Wire the neutral loci config (`evolution.neutral.nlocus`, `evolution.neutral.nval`) into the genetics initialization in `simulate()`.

- [ ] **Step 1: Add config fields**

In `osmose/engine/config.py`, add after `genetics_transmission_year`:

```python
    # Neutral loci (0 = disabled)
    genetics_n_neutral: int = 0
    genetics_n_neutral_val: int = 50
```

In `from_dict()`:

```python
        genetics_n_neutral = int(cfg.get("evolution.neutral.nlocus", "0"))
        genetics_n_neutral_val = int(cfg.get("evolution.neutral.nval", "50"))
```

- [ ] **Step 2: Pass neutral params to create_initial_genotypes in simulate()**

In the genetics initialization block:

```python
    if config.genetics_enabled:
        from osmose.engine.genetics import TraitRegistry, create_initial_genotypes

        trait_registry = TraitRegistry.from_config(config.raw_config, config.n_species)
        ctx.genetic_state = create_initial_genotypes(
            trait_registry, state.species_id, rng,
            n_neutral=config.genetics_n_neutral,
            n_neutral_val=config.genetics_n_neutral_val,
        )
```

- [ ] **Step 3: Run existing tests**

Run: `.venv/bin/python -m pytest tests/test_genetics_integration.py tests/test_genetics_neutral.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add osmose/engine/config.py osmose/engine/simulate.py
git commit -m "feat(config): add neutral loci and seeding phase config parsing"
```

---

## Task 6: Statistical validation tests

**Files:**
- Create: `tests/test_genetics_statistics.py`

### Context

Statistical tests verify genetic dynamics over many generations: allele frequencies converge correctly, Hardy-Weinberg equilibrium holds in random mating, and trait variance is maintained.

- [ ] **Step 1: Write statistical tests**

```python
# tests/test_genetics_statistics.py
"""Statistical tests for genetic dynamics over multiple generations."""

import numpy as np
import pytest

from osmose.engine.genetics.expression import express_traits
from osmose.engine.genetics.genotype import GeneticState, create_initial_genotypes
from osmose.engine.genetics.inheritance import create_offspring_genotypes
from osmose.engine.genetics.trait import Trait, TraitRegistry


def _make_simple_registry(n_loci: int = 5, n_alleles: int = 100) -> TraitRegistry:
    """Registry with one trait, zero env noise, known variance."""
    cfg = {
        "evolution.trait.imax.mean.sp0": "0.0",
        "evolution.trait.imax.var.sp0": "1.0",
        "evolution.trait.imax.envvar.sp0": "0.0",
        f"evolution.trait.imax.nlocus.sp0": str(n_loci),
        f"evolution.trait.imax.nval.sp0": str(n_alleles),
        "evolution.trait.imax.target": "ingestion_rate",
    }
    return TraitRegistry.from_config(cfg, n_species=1)


class TestGeneticVariance:
    def test_initial_phenotype_variance_matches_spec(self):
        """Initial phenotypic variance should approximate species_var (with env_var=0)."""
        registry = _make_simple_registry(n_loci=20, n_alleles=500)
        species_id = np.zeros(1000, dtype=np.int32)
        rng = np.random.default_rng(42)
        gs = create_initial_genotypes(registry, species_id, rng)
        phenotypes = express_traits(gs, species_id)
        # phenotype = mean(0) + sum(alleles) + noise(0) → variance should ≈ 1.0
        observed_var = np.var(phenotypes["imax"])
        assert observed_var == pytest.approx(1.0, rel=0.3)

    def test_variance_preserved_across_generations(self):
        """Trait variance should not collapse over 10 generations of random mating."""
        registry = _make_simple_registry(n_loci=10, n_alleles=100)
        n_schools = 300
        species_id = np.zeros(n_schools, dtype=np.int32)
        rng = np.random.default_rng(42)
        gs = create_initial_genotypes(registry, species_id, rng)

        initial_var = np.var(express_traits(gs, species_id)["imax"])

        for _gen in range(10):
            gonad = np.ones(n_schools)
            gs = create_offspring_genotypes(
                parent_gs=gs, gonad_weight=gonad,
                species_id=np.zeros(n_schools, dtype=np.int32),
                offspring_species=0, n_offspring=n_schools, rng=rng,
            )

        final_var = np.var(express_traits(gs, np.zeros(n_schools, dtype=np.int32))["imax"])
        # Variance shouldn't collapse (expect some drift but not to zero)
        assert final_var > initial_var * 0.3

    def test_four_traits_independent(self):
        """Four traits with different variances should express independently."""
        cfg = {
            "evolution.trait.imax.mean.sp0": "0.0",
            "evolution.trait.imax.var.sp0": "1.0",
            "evolution.trait.imax.envvar.sp0": "0.0",
            "evolution.trait.imax.nlocus.sp0": "10",
            "evolution.trait.imax.nval.sp0": "100",
            "evolution.trait.imax.target": "bioen_i_max",
            "evolution.trait.gsi.mean.sp0": "0.5",
            "evolution.trait.gsi.var.sp0": "0.01",
            "evolution.trait.gsi.envvar.sp0": "0.0",
            "evolution.trait.gsi.nlocus.sp0": "10",
            "evolution.trait.gsi.nval.sp0": "100",
            "evolution.trait.gsi.target": "bioen_r",
            "evolution.trait.m0.mean.sp0": "10.0",
            "evolution.trait.m0.var.sp0": "2.0",
            "evolution.trait.m0.envvar.sp0": "0.0",
            "evolution.trait.m0.nlocus.sp0": "10",
            "evolution.trait.m0.nval.sp0": "100",
            "evolution.trait.m0.target": "bioen_m0",
            "evolution.trait.m1.mean.sp0": "0.5",
            "evolution.trait.m1.var.sp0": "0.005",
            "evolution.trait.m1.envvar.sp0": "0.0",
            "evolution.trait.m1.nlocus.sp0": "10",
            "evolution.trait.m1.nval.sp0": "100",
            "evolution.trait.m1.target": "bioen_m1",
        }
        registry = TraitRegistry.from_config(cfg, n_species=1)
        assert len(registry.traits) == 4
        species_id = np.zeros(500, dtype=np.int32)
        rng = np.random.default_rng(42)
        gs = create_initial_genotypes(registry, species_id, rng)
        phenotypes = express_traits(gs, species_id)
        # Each trait should have variance proportional to its species_var
        assert np.var(phenotypes["imax"]) > np.var(phenotypes["gsi"])
        assert np.var(phenotypes["m0"]) > np.var(phenotypes["m1"])


class TestInheritanceStatistics:
    def test_allele_frequencies_stable_under_drift(self):
        """Over many generations of random mating (no selection), mean phenotype stays near 0."""
        registry = _make_simple_registry(n_loci=5, n_alleles=50)
        n_schools = 200
        species_id = np.zeros(n_schools, dtype=np.int32)
        rng = np.random.default_rng(42)
        gs = create_initial_genotypes(registry, species_id, rng)

        # Simulate 20 generations of random mating
        for _gen in range(20):
            gonad = np.ones(len(gs.alleles["imax"]))
            new_gs = create_offspring_genotypes(
                parent_gs=gs,
                gonad_weight=gonad,
                species_id=np.zeros(len(gonad), dtype=np.int32),
                offspring_species=0,
                n_offspring=n_schools,
                rng=rng,
            )
            gs = new_gs

        phenotypes = express_traits(gs, np.zeros(n_schools, dtype=np.int32))
        mean_pheno = np.mean(phenotypes["imax"])
        # Mean should still be near 0 (no directional selection)
        assert abs(mean_pheno) < 1.0  # generous bound for drift

    def test_neutral_allele_diversity_maintained(self):
        """Neutral loci should maintain diversity over generations (no purging)."""
        registry = _make_simple_registry()
        n_schools = 100
        species_id = np.zeros(n_schools, dtype=np.int32)
        rng = np.random.default_rng(42)
        gs = create_initial_genotypes(registry, species_id, rng, n_neutral=5, n_neutral_val=20)

        initial_unique = len(np.unique(gs.neutral_alleles))

        for _gen in range(10):
            gonad = np.ones(n_schools)
            new_gs = create_offspring_genotypes(
                parent_gs=gs,
                gonad_weight=gonad,
                species_id=np.zeros(n_schools, dtype=np.int32),
                offspring_species=0,
                n_offspring=n_schools,
                rng=rng,
            )
            gs = new_gs

        final_unique = len(np.unique(gs.neutral_alleles))
        # With 100 schools and 20 possible values per locus, diversity should persist
        # (some loss expected from drift, but shouldn't collapse completely)
        assert final_unique > initial_unique * 0.3
```

- [ ] **Step 2: Run statistical tests**

Run: `.venv/bin/python -m pytest tests/test_genetics_statistics.py -v`
Expected: 5 PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/test_genetics_statistics.py
git commit -m "test(genetics): add statistical validation tests for drift and variance"
```

---

## Task 7: Lint, full test suite, final verification

**Files:** None — verification only

- [ ] **Step 1: Run ruff lint**

Run: `.venv/bin/ruff check osmose/engine/genetics/ tests/test_genetics_*.py`

- [ ] **Step 2: Run ruff format**

Run: `.venv/bin/ruff format osmose/engine/genetics/ tests/test_genetics_*.py`

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All tests pass including new ones.

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -u
git commit -m "style: ruff format genetics Phase 2 files"
```

---

## Summary

| Task | What | New Tests |
|------|------|-----------|
| 1 | Wire trait_overrides into _bioen_step (imax, r, m0, m1) | 2 |
| 2 | Wire trait_overrides into _bioen_reproduction (m0, m1) | 1 |
| 3 | Neutral loci on GeneticState + inheritance | 5 |
| 4 | Seeding phase (random alleles before transmission year) | 2 |
| 5 | Config parsing for neutral loci + wiring | 0 |
| 6 | Statistical validation (drift, variance, neutral diversity, 4-trait independence) | 5 |
| 7 | Lint, format, full verification | 0 |
| **Total** | | **~15 tests** |

**Note:** The spec mentions "Validation against Java Ev-OSMOSE (North Sea config from Zenodo)." This requires the North Sea config dataset from Zenodo (https://zenodo.org/records/7636112). Java parity validation will be a separate task once the data is downloaded and integrated into the test fixtures. The statistical tests above validate genetic dynamics correctness independently of Java comparison.
