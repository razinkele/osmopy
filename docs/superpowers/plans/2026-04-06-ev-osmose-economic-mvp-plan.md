# Ev-OSMOSE + Economic Fleet Dynamics — Phase 1 MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two optional modules — Ev-OSMOSE diploid genetics (1 evolving trait: Imax) and DSVM fleet dynamics (single fleet, revenue-only logit) — toggleable via config, with zero overhead when disabled, and all existing tests passing.

**Architecture:** Both modules store mutable state on `SimulationContext` as optional fields (`genetic_state: GeneticState | None`, `fleet_state: FleetState | None`). Process functions follow the existing pattern `(state, config, ...) -> state`. Genetics hooks into reproduction (inheritance) and pre-growth (trait expression). Economics hooks before mortality (fleet decision) and after mortality (memory update). The simulation loop in `simulate.py` gains `if ctx.genetic_state is not None` / `if ctx.fleet_state is not None` guards at each insertion point.

**Tech Stack:** Python 3.12+, NumPy, pytest, ruff

**Spec:** `docs/superpowers/specs/2026-04-05-ev-osmose-economic-design.md`

---

## File Structure

### New files (genetics)

| File | Responsibility |
|------|---------------|
| `osmose/engine/genetics/__init__.py` | Package exports: `GeneticState`, `TraitRegistry`, `Trait` |
| `osmose/engine/genetics/trait.py` | `Trait` dataclass, `TraitRegistry` (config parsing, allele pool init) |
| `osmose/engine/genetics/genotype.py` | `GeneticState` dataclass, `compact_genetic_state`, `create_initial_genotypes` |
| `osmose/engine/genetics/inheritance.py` | `select_parents`, `form_gamete`, `create_offspring_genotypes` |
| `osmose/engine/genetics/expression.py` | `express_traits`, `apply_trait_overrides` |
| `tests/test_genetics_trait.py` | TraitRegistry parsing, allele pool variance |
| `tests/test_genetics_inheritance.py` | Gamete formation, parent selection, offspring genotype validity |
| `tests/test_genetics_expression.py` | Trait expression, phenotype override |

### New files (economics)

| File | Responsibility |
|------|---------------|
| `osmose/engine/economics/__init__.py` | Package exports: `FleetConfig`, `FleetState` |
| `osmose/engine/economics/fleet.py` | `FleetConfig`, `FleetState` dataclasses, config parsing, `create_fleet_state` |
| `osmose/engine/economics/choice.py` | `fleet_decision` (DSVM logit), `aggregate_effort` |
| `tests/test_economics_fleet.py` | Fleet config parsing, state initialization |
| `tests/test_economics_choice.py` | Logit probabilities, effort aggregation, port choice |

### Modified files

| File | Changes |
|------|---------|
| `osmose/engine/simulate.py` | Add `genetic_state`/`fleet_state` to `SimulationContext`; insert genetics/economics calls in loop |
| `osmose/engine/config.py` | Add `genetics_enabled`, `economics_enabled` flags + `raw_config` access for submodule parsing |
| `osmose/engine/processes/mortality.py` | Add `fleet_state` param to `_precompute_effective_rates`; scale fishing by effort for targeted species |
| `osmose/engine/__init__.py` | No changes needed (PythonEngine passes `config` dict which already contains raw keys) |

---

## Task 1: Trait Dataclass and TraitRegistry

**Files:**
- Create: `osmose/engine/genetics/__init__.py`
- Create: `osmose/engine/genetics/trait.py`
- Test: `tests/test_genetics_trait.py`

### Context

The `Trait` dataclass defines one evolving trait. `TraitRegistry` parses traits from OSMOSE config keys like `evolution.trait.imax.mean.sp0`. For MVP, only `imax` (max ingestion rate) is supported.

The allele pool for each trait/species/locus is drawn from `N(0, σ²_A / (2 × n_loci))` so that when summed across all loci (2 alleles each), total genotypic variance equals the prescribed `species_var`.

- [ ] **Step 1: Write test for Trait creation and TraitRegistry config parsing**

```python
# tests/test_genetics_trait.py
"""Tests for Trait and TraitRegistry."""

import numpy as np
import pytest

from osmose.engine.genetics.trait import Trait, TraitRegistry


class TestTrait:
    def test_trait_frozen(self):
        t = Trait(
            name="imax",
            species_mean=np.array([3.5]),
            species_var=np.array([0.1]),
            env_var=np.array([0.05]),
            n_loci=np.array([10], dtype=np.int32),
            n_alleles=np.array([20], dtype=np.int32),
            target_param="ingestion_rate",
        )
        assert t.name == "imax"
        assert t.target_param == "ingestion_rate"
        with pytest.raises(AttributeError):
            t.name = "other"


class TestTraitRegistry:
    def test_from_config_single_trait(self):
        cfg = {
            "simulation.genetic.enabled": "true",
            "simulation.nspecies": "2",
            "evolution.trait.imax.mean.sp0": "3.5",
            "evolution.trait.imax.mean.sp1": "4.0",
            "evolution.trait.imax.var.sp0": "0.1",
            "evolution.trait.imax.var.sp1": "0.2",
            "evolution.trait.imax.envvar.sp0": "0.05",
            "evolution.trait.imax.envvar.sp1": "0.08",
            "evolution.trait.imax.nlocus.sp0": "10",
            "evolution.trait.imax.nlocus.sp1": "10",
            "evolution.trait.imax.nval.sp0": "20",
            "evolution.trait.imax.nval.sp1": "20",
            "evolution.trait.imax.target": "ingestion_rate",
        }
        registry = TraitRegistry.from_config(cfg, n_species=2)
        assert "imax" in registry.traits
        trait = registry.traits["imax"]
        assert trait.species_mean[0] == pytest.approx(3.5)
        assert trait.species_mean[1] == pytest.approx(4.0)
        assert trait.n_loci[0] == 10

    def test_allele_pool_variance(self):
        """Allele pool values should produce correct total genotypic variance."""
        cfg = {
            "evolution.trait.imax.mean.sp0": "0.0",
            "evolution.trait.imax.var.sp0": "1.0",
            "evolution.trait.imax.envvar.sp0": "0.0",
            "evolution.trait.imax.nlocus.sp0": "50",
            "evolution.trait.imax.nval.sp0": "1000",
            "evolution.trait.imax.target": "ingestion_rate",
        }
        registry = TraitRegistry.from_config(cfg, n_species=1)
        trait = registry.traits["imax"]
        # Each locus contributes var = σ²_A / (2 * n_loci)
        # With 1000 alleles per locus and 50 loci, 2 alleles per locus:
        # Total variance ≈ 2 * n_loci * per_locus_var = species_var
        pool = trait.allele_pool[0]  # species 0
        assert len(pool) == 50  # n_loci pools
        per_locus_var = np.var(pool[0])
        expected_per_locus = 1.0 / (2 * 50)
        assert per_locus_var == pytest.approx(expected_per_locus, rel=0.3)

    def test_empty_when_no_traits(self):
        registry = TraitRegistry.from_config({}, n_species=1)
        assert len(registry.traits) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_genetics_trait.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.engine.genetics'`

- [ ] **Step 3: Create package and implement Trait + TraitRegistry**

```python
# osmose/engine/genetics/__init__.py
"""Ev-OSMOSE eco-evolutionary genetics module."""

from osmose.engine.genetics.trait import Trait, TraitRegistry

__all__ = ["Trait", "TraitRegistry"]
```

```python
# osmose/engine/genetics/trait.py
"""Trait definitions and registry for evolving parameters."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Trait:
    """One evolving trait with per-species genetic parameters.

    Attributes:
        name: Trait identifier (e.g., "imax").
        species_mean: Initial genotypic mean per species, shape (n_species,).
        species_var: Additive genetic variance per species, shape (n_species,).
        env_var: Environmental noise variance per species, shape (n_species,).
        n_loci: Number of loci per species, shape (n_species,).
        n_alleles: Number of possible allelic values per locus per species, shape (n_species,).
        target_param: EngineConfig field this trait overrides.
        allele_pool: Pre-drawn allelic values — allele_pool[species][locus] is a 1-D array
            of n_alleles possible values. Populated by TraitRegistry.
    """

    name: str
    species_mean: NDArray[np.float64]
    species_var: NDArray[np.float64]
    env_var: NDArray[np.float64]
    n_loci: NDArray[np.int32]
    n_alleles: NDArray[np.int32]
    target_param: str
    allele_pool: list[list[NDArray[np.float64]]] = field(default_factory=list)


class TraitRegistry:
    """Collection of evolving traits, parsed from OSMOSE config."""

    def __init__(self) -> None:
        self.traits: dict[str, Trait] = {}

    def register(self, trait: Trait) -> None:
        self.traits[trait.name] = trait

    @classmethod
    def from_config(cls, cfg: dict[str, str], n_species: int) -> TraitRegistry:
        """Parse all evolution.trait.<name>.* keys from config."""
        registry = cls()

        # Discover trait names from keys matching evolution.trait.<name>.target
        trait_names: set[str] = set()
        for key in cfg:
            m = re.match(r"evolution\.trait\.(\w+)\.target", key)
            if m:
                trait_names.add(m.group(1))

        for name in sorted(trait_names):
            prefix = f"evolution.trait.{name}"
            target = cfg[f"{prefix}.target"]

            means = np.array(
                [float(cfg.get(f"{prefix}.mean.sp{i}", "0.0")) for i in range(n_species)]
            )
            variances = np.array(
                [float(cfg.get(f"{prefix}.var.sp{i}", "0.0")) for i in range(n_species)]
            )
            env_vars = np.array(
                [float(cfg.get(f"{prefix}.envvar.sp{i}", "0.0")) for i in range(n_species)]
            )
            n_loci = np.array(
                [int(cfg.get(f"{prefix}.nlocus.sp{i}", "10")) for i in range(n_species)],
                dtype=np.int32,
            )
            n_alleles = np.array(
                [int(cfg.get(f"{prefix}.nval.sp{i}", "20")) for i in range(n_species)],
                dtype=np.int32,
            )

            # Build allele pools: for each species, each locus, draw n_alleles values
            # from N(0, sigma²_A / (2 * n_loci)) so total genotypic var ≈ species_var
            rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
            allele_pool: list[list[NDArray[np.float64]]] = []
            for sp in range(n_species):
                sp_pools: list[NDArray[np.float64]] = []
                if variances[sp] > 0 and n_loci[sp] > 0:
                    per_locus_sd = np.sqrt(variances[sp] / (2 * n_loci[sp]))
                    for _loc in range(n_loci[sp]):
                        values = rng.normal(0.0, per_locus_sd, size=int(n_alleles[sp]))
                        sp_pools.append(values)
                allele_pool.append(sp_pools)

            trait = Trait(
                name=name,
                species_mean=means,
                species_var=variances,
                env_var=env_vars,
                n_loci=n_loci,
                n_alleles=n_alleles,
                target_param=target,
                allele_pool=allele_pool,
            )
            registry.register(trait)

        return registry
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_genetics_trait.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/genetics/__init__.py osmose/engine/genetics/trait.py tests/test_genetics_trait.py
git commit -m "feat(genetics): add Trait dataclass and TraitRegistry with config parsing"
```

---

## Task 2: GeneticState and Compact Sync

**Files:**
- Create: `osmose/engine/genetics/genotype.py`
- Modify: `osmose/engine/genetics/__init__.py`
- Test: `tests/test_genetics_trait.py` (append to existing)

### Context

`GeneticState` is a parallel data structure to `SchoolState` — indexed by school position. It holds per-school allele arrays and environmental noise for each trait. It must stay in sync through `compact()` (dead school removal) and `append()` (reproduction). The `compact_genetic_state` function takes an alive mask and slices all arrays. `create_initial_genotypes` assigns random alleles from the trait's allele pool to each school.

- [ ] **Step 1: Write tests for GeneticState creation, compact, and append**

Append to `tests/test_genetics_trait.py`:

```python
from osmose.engine.genetics.genotype import (
    GeneticState,
    compact_genetic_state,
    create_initial_genotypes,
)


class TestGeneticState:
    def _make_registry(self) -> TraitRegistry:
        cfg = {
            "evolution.trait.imax.mean.sp0": "3.5",
            "evolution.trait.imax.var.sp0": "0.1",
            "evolution.trait.imax.envvar.sp0": "0.05",
            "evolution.trait.imax.nlocus.sp0": "5",
            "evolution.trait.imax.nval.sp0": "20",
            "evolution.trait.imax.target": "ingestion_rate",
        }
        return TraitRegistry.from_config(cfg, n_species=1)

    def test_create_initial_genotypes(self):
        registry = self._make_registry()
        species_id = np.array([0, 0, 0], dtype=np.int32)
        rng = np.random.default_rng(42)
        gs = create_initial_genotypes(registry, species_id, rng)
        assert "imax" in gs.alleles
        assert gs.alleles["imax"].shape == (3, 5, 2)  # (n_schools, n_loci, 2 alleles)
        assert "imax" in gs.env_noise
        assert gs.env_noise["imax"].shape == (3,)

    def test_compact_removes_dead(self):
        registry = self._make_registry()
        species_id = np.array([0, 0, 0], dtype=np.int32)
        rng = np.random.default_rng(42)
        gs = create_initial_genotypes(registry, species_id, rng)
        alive = np.array([True, False, True])
        compacted = compact_genetic_state(gs, alive)
        assert compacted.alleles["imax"].shape[0] == 2
        assert compacted.env_noise["imax"].shape[0] == 2

    def test_append_concatenates(self):
        registry = self._make_registry()
        rng = np.random.default_rng(42)
        gs1 = create_initial_genotypes(registry, np.array([0, 0], dtype=np.int32), rng)
        gs2 = create_initial_genotypes(registry, np.array([0], dtype=np.int32), rng)
        merged = gs1.append(gs2)
        assert merged.alleles["imax"].shape[0] == 3
        assert merged.env_noise["imax"].shape[0] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_genetics_trait.py::TestGeneticState -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.engine.genetics.genotype'`

- [ ] **Step 3: Implement GeneticState**

```python
# osmose/engine/genetics/genotype.py
"""GeneticState: parallel allele/noise arrays for all schools."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from osmose.engine.genetics.trait import TraitRegistry


@dataclass
class GeneticState:
    """Per-school genetic data, parallel to SchoolState.

    Attributes:
        alleles: Per trait, shape (n_schools, max_loci, 2) — diploid allele values.
        env_noise: Per trait, shape (n_schools,) — environmental noise fixed at birth.
        registry: Reference to the TraitRegistry for trait metadata.
    """

    alleles: dict[str, NDArray[np.float64]]
    env_noise: dict[str, NDArray[np.float64]]
    registry: TraitRegistry

    def append(self, other: GeneticState) -> GeneticState:
        """Concatenate another GeneticState (for reproduction append)."""
        new_alleles = {}
        new_noise = {}
        for name in self.alleles:
            new_alleles[name] = np.concatenate(
                [self.alleles[name], other.alleles[name]], axis=0
            )
            new_noise[name] = np.concatenate(
                [self.env_noise[name], other.env_noise[name]], axis=0
            )
        return GeneticState(alleles=new_alleles, env_noise=new_noise, registry=self.registry)


def compact_genetic_state(gs: GeneticState, alive_mask: NDArray[np.bool_]) -> GeneticState:
    """Remove dead schools from genetic state using the same mask as SchoolState.compact()."""
    new_alleles = {name: arr[alive_mask] for name, arr in gs.alleles.items()}
    new_noise = {name: arr[alive_mask] for name, arr in gs.env_noise.items()}
    return GeneticState(alleles=new_alleles, env_noise=new_noise, registry=gs.registry)


def create_initial_genotypes(
    registry: TraitRegistry,
    species_id: NDArray[np.int32],
    rng: np.random.Generator,
) -> GeneticState:
    """Assign random alleles from each trait's allele pool to each school.

    For each school, for each locus, two alleles are drawn uniformly from
    the pre-computed allele pool for that species/locus.
    """
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
                # Draw two alleles from the pool for this locus
                arr[i, loc, :] = rng.choice(pool[loc], size=2, replace=True)
            # Environmental noise drawn at birth
            if trait.env_var[sp] > 0:
                noise[i] = rng.normal(0.0, np.sqrt(trait.env_var[sp]))

        alleles[name] = arr
        env_noise[name] = noise

    return GeneticState(alleles=alleles, env_noise=env_noise, registry=registry)
```

Update `osmose/engine/genetics/__init__.py`:

```python
# osmose/engine/genetics/__init__.py
"""Ev-OSMOSE eco-evolutionary genetics module."""

from osmose.engine.genetics.genotype import GeneticState, compact_genetic_state, create_initial_genotypes
from osmose.engine.genetics.trait import Trait, TraitRegistry

__all__ = [
    "GeneticState",
    "Trait",
    "TraitRegistry",
    "compact_genetic_state",
    "create_initial_genotypes",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_genetics_trait.py -v`
Expected: 7 PASSED (4 from Task 1 + 3 new)

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/genetics/genotype.py osmose/engine/genetics/__init__.py tests/test_genetics_trait.py
git commit -m "feat(genetics): add GeneticState with compact/append sync and initial genotype creation"
```

---

## Task 3: Trait Expression and Phenotype Override

**Files:**
- Create: `osmose/engine/genetics/expression.py`
- Modify: `osmose/engine/genetics/__init__.py`
- Test: `tests/test_genetics_expression.py`

### Context

`express_traits()` computes phenotypic values: `phenotype = species_mean[sp] + sum(alleles) + env_noise`. `apply_trait_overrides()` populates a standalone `overrides` dict mapping EngineConfig field names to per-school phenotypic arrays.

The override mechanism uses a plain `dict[str, NDArray]` passed through the call chain — not stored on EngineConfig. This avoids modifying the frozen config dataclass. In Phase 2, growth/bioen functions will accept this dict and use per-school values instead of species-level constants.

- [ ] **Step 1: Write tests for trait expression and override**

```python
# tests/test_genetics_expression.py
"""Tests for genotype-to-phenotype expression and config override."""

import numpy as np
import pytest

from osmose.engine.genetics.expression import apply_trait_overrides, express_traits
from osmose.engine.genetics.genotype import GeneticState
from osmose.engine.genetics.trait import Trait, TraitRegistry


class TestExpressTraits:
    def test_phenotype_is_mean_plus_alleles_plus_noise(self):
        """Phenotype = species_mean + sum(alleles) + env_noise."""
        registry = TraitRegistry()
        trait = Trait(
            name="imax",
            species_mean=np.array([3.5]),
            species_var=np.array([0.1]),
            env_var=np.array([0.0]),
            n_loci=np.array([2], dtype=np.int32),
            n_alleles=np.array([10], dtype=np.int32),
            target_param="ingestion_rate",
        )
        registry.register(trait)

        # 1 school, species 0, 2 loci, 2 alleles each
        alleles = np.array([[[0.1, 0.2], [0.3, -0.1]]])  # shape (1, 2, 2)
        noise = np.array([0.05])
        gs = GeneticState(
            alleles={"imax": alleles},
            env_noise={"imax": noise},
            registry=registry,
        )
        species_id = np.array([0], dtype=np.int32)

        phenotypes = express_traits(gs, species_id)
        # 3.5 + (0.1 + 0.2 + 0.3 + (-0.1)) + 0.05 = 4.05
        assert phenotypes["imax"][0] == pytest.approx(4.05)

    def test_multi_school_expression(self):
        registry = TraitRegistry()
        trait = Trait(
            name="imax",
            species_mean=np.array([0.0, 10.0]),
            species_var=np.array([1.0, 1.0]),
            env_var=np.array([0.0, 0.0]),
            n_loci=np.array([1, 1], dtype=np.int32),
            n_alleles=np.array([10, 10], dtype=np.int32),
            target_param="ingestion_rate",
        )
        registry.register(trait)

        # 2 schools: sp0 and sp1
        alleles = np.array([[[0.5, 0.5]], [[-1.0, 1.0]]])  # shape (2, 1, 2)
        noise = np.zeros(2)
        gs = GeneticState(alleles={"imax": alleles}, env_noise={"imax": noise}, registry=registry)
        species_id = np.array([0, 1], dtype=np.int32)

        phenotypes = express_traits(gs, species_id)
        assert phenotypes["imax"][0] == pytest.approx(1.0)   # 0 + 0.5 + 0.5
        assert phenotypes["imax"][1] == pytest.approx(10.0)  # 10 + (-1) + 1


class TestApplyTraitOverrides:
    def test_override_stored_in_dict(self):
        """apply_trait_overrides stores per-school values in the overrides dict."""
        registry = TraitRegistry()
        trait = Trait(
            name="imax",
            species_mean=np.array([3.5]),
            species_var=np.array([0.1]),
            env_var=np.array([0.0]),
            n_loci=np.array([2], dtype=np.int32),
            n_alleles=np.array([10], dtype=np.int32),
            target_param="ingestion_rate",
        )
        registry.register(trait)

        phenotypes = {"imax": np.array([4.0, 3.8])}
        overrides: dict[str, np.ndarray] = {}
        apply_trait_overrides(overrides, phenotypes, registry)
        assert "ingestion_rate" in overrides
        assert overrides["ingestion_rate"][0] == pytest.approx(4.0)
        assert overrides["ingestion_rate"][1] == pytest.approx(3.8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_genetics_expression.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.engine.genetics.expression'`

- [ ] **Step 3: Implement expression.py**

```python
# osmose/engine/genetics/expression.py
"""Genotype-to-phenotype expression and config override application."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.genetics.genotype import GeneticState
from osmose.engine.genetics.trait import TraitRegistry


def express_traits(
    gs: GeneticState,
    species_id: NDArray[np.int32],
) -> dict[str, NDArray[np.float64]]:
    """Compute phenotypic values for all traits, all schools.

    phenotype[i] = species_mean[sp_i] + sum(alleles[i, :, :]) + env_noise[i]
    """
    phenotypes: dict[str, NDArray[np.float64]] = {}
    for name, trait in gs.registry.traits.items():
        sp = species_id
        g = trait.species_mean[sp] + gs.alleles[name].sum(axis=(1, 2))
        phenotypes[name] = g + gs.env_noise[name]
    return phenotypes


def apply_trait_overrides(
    overrides: dict[str, NDArray[np.float64]],
    phenotypes: dict[str, NDArray[np.float64]],
    registry: TraitRegistry,
) -> None:
    """Store per-school phenotypic values keyed by target parameter name.

    Args:
        overrides: Mutable dict to populate. Keyed by EngineConfig field name
            (e.g., "ingestion_rate"), values are per-school arrays.
        phenotypes: Output of express_traits().
        registry: TraitRegistry for target_param lookup.
    """
    for trait_name, values in phenotypes.items():
        target = registry.traits[trait_name].target_param
        overrides[target] = values
```

Update `osmose/engine/genetics/__init__.py` to export the new functions:

```python
# osmose/engine/genetics/__init__.py
"""Ev-OSMOSE eco-evolutionary genetics module."""

from osmose.engine.genetics.expression import apply_trait_overrides, express_traits
from osmose.engine.genetics.genotype import (
    GeneticState,
    compact_genetic_state,
    create_initial_genotypes,
)
from osmose.engine.genetics.trait import Trait, TraitRegistry

__all__ = [
    "GeneticState",
    "Trait",
    "TraitRegistry",
    "apply_trait_overrides",
    "compact_genetic_state",
    "create_initial_genotypes",
    "express_traits",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_genetics_expression.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/genetics/expression.py osmose/engine/genetics/__init__.py tests/test_genetics_expression.py
git commit -m "feat(genetics): add trait expression and phenotype override mechanism"
```

---

## Task 4: Gametic Inheritance

**Files:**
- Create: `osmose/engine/genetics/inheritance.py`
- Modify: `osmose/engine/genetics/__init__.py`
- Test: `tests/test_genetics_inheritance.py`

### Context

At reproduction, two parents are selected with probability proportional to gonad weight (fecundity proxy). Each parent contributes one gamete per locus (random pick of one allele = free recombination). The offspring gets allele 0 from parent A's gamete and allele 1 from parent B's gamete. Environmental noise is drawn fresh at birth.

For MVP, there is no seeding phase — all offspring inherit from parents.

- [ ] **Step 1: Write tests for inheritance functions**

```python
# tests/test_genetics_inheritance.py
"""Tests for gametic inheritance: parent selection, gamete formation, offspring creation."""

import numpy as np

from osmose.engine.genetics.genotype import GeneticState
from osmose.engine.genetics.inheritance import (
    create_offspring_genotypes,
    form_gamete,
    select_parents,
)
from osmose.engine.genetics.trait import Trait, TraitRegistry


class TestSelectParents:
    def test_fecundity_weighted(self):
        """Parents should be drawn proportional to gonad_weight."""
        gonad_weight = np.array([0.0, 0.0, 100.0])  # only school 2 has eggs
        rng = np.random.default_rng(42)
        counts = np.zeros(3, dtype=int)
        for _ in range(200):
            a, b = select_parents(gonad_weight, rng)
            counts[a] += 1
            counts[b] += 1
        # All selections should be school 2
        assert counts[0] == 0
        assert counts[1] == 0
        assert counts[2] == 400


class TestFormGamete:
    def test_picks_one_allele_per_locus(self):
        """Gamete should contain exactly one allele from each locus."""
        parent_alleles = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3 loci, 2 alleles)
        rng = np.random.default_rng(42)
        gamete = form_gamete(parent_alleles, rng)
        assert gamete.shape == (3,)
        for i in range(3):
            assert gamete[i] in [parent_alleles[i, 0], parent_alleles[i, 1]]


class TestCreateOffspringGenotypes:
    def _make_registry(self) -> TraitRegistry:
        registry = TraitRegistry()
        registry.register(Trait(
            name="imax",
            species_mean=np.array([3.5]),
            species_var=np.array([0.1]),
            env_var=np.array([0.05]),
            n_loci=np.array([3], dtype=np.int32),
            n_alleles=np.array([20], dtype=np.int32),
            target_param="ingestion_rate",
        ))
        return registry

    def test_offspring_has_correct_shape(self):
        registry = self._make_registry()
        # 2 parent schools, species 0
        parent_alleles = np.zeros((2, 3, 2))
        parent_alleles[0] = [[1, 2], [3, 4], [5, 6]]
        parent_alleles[1] = [[7, 8], [9, 10], [11, 12]]
        parent_noise = np.array([0.1, 0.2])
        parent_gs = GeneticState(
            alleles={"imax": parent_alleles},
            env_noise={"imax": parent_noise},
            registry=registry,
        )

        gonad_weight = np.array([50.0, 50.0])
        rng = np.random.default_rng(42)

        offspring_gs = create_offspring_genotypes(
            parent_gs=parent_gs,
            gonad_weight=gonad_weight,
            species_id=np.array([0, 0], dtype=np.int32),
            offspring_species=0,
            n_offspring=5,
            rng=rng,
        )
        assert offspring_gs.alleles["imax"].shape == (5, 3, 2)
        assert offspring_gs.env_noise["imax"].shape == (5,)

    def test_offspring_alleles_come_from_parents(self):
        """Each offspring allele at each locus must be from one of the two parents' alleles."""
        registry = self._make_registry()
        parent_alleles = np.zeros((2, 3, 2))
        parent_alleles[0] = [[1, 2], [3, 4], [5, 6]]
        parent_alleles[1] = [[7, 8], [9, 10], [11, 12]]
        parent_gs = GeneticState(
            alleles={"imax": parent_alleles},
            env_noise={"imax": np.zeros(2)},
            registry=registry,
        )

        rng = np.random.default_rng(99)
        offspring_gs = create_offspring_genotypes(
            parent_gs=parent_gs,
            gonad_weight=np.array([50.0, 50.0]),
            species_id=np.array([0, 0], dtype=np.int32),
            offspring_species=0,
            n_offspring=10,
            rng=rng,
        )
        # Every allele value in offspring must come from one of the two parents
        parent_vals = set(parent_alleles.flatten())
        for val in offspring_gs.alleles["imax"].flatten():
            assert val in parent_vals
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_genetics_inheritance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.engine.genetics.inheritance'`

- [ ] **Step 3: Implement inheritance.py**

```python
# osmose/engine/genetics/inheritance.py
"""Gametic inheritance: parent selection, meiotic segregation, offspring genotype creation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.genetics.genotype import GeneticState


def select_parents(
    gonad_weight: NDArray[np.float64],
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Select two parents with probability proportional to gonad weight (fecundity proxy)."""
    weights = gonad_weight / gonad_weight.sum()
    parent_a, parent_b = rng.choice(len(gonad_weight), size=2, replace=True, p=weights)
    return int(parent_a), int(parent_b)


def form_gamete(
    parent_alleles: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Meiotic segregation: pick one allele per locus (free recombination).

    Args:
        parent_alleles: Shape (n_loci, 2) — diploid alleles for one parent, one trait.
        rng: Random generator.

    Returns:
        Gamete array of shape (n_loci,).
    """
    n_loci = parent_alleles.shape[0]
    picks = rng.integers(0, 2, size=n_loci)
    return parent_alleles[np.arange(n_loci), picks]


def create_offspring_genotypes(
    parent_gs: GeneticState,
    gonad_weight: NDArray[np.float64],
    species_id: NDArray[np.int32],
    offspring_species: int,
    n_offspring: int,
    rng: np.random.Generator,
) -> GeneticState:
    """Create genotypes for n_offspring new schools of one species.

    For each offspring:
    1. Select two parents (fecundity-weighted).
    2. Form gamete from each parent (meiotic segregation).
    3. Combine gametes into diploid offspring genotype.
    4. Draw environmental noise from N(0, sqrt(env_var)).

    Args:
        parent_gs: Genetic state of existing (parent) schools.
        gonad_weight: Gonad weight of parent schools (fecundity proxy).
        species_id: Species ID of parent schools.
        offspring_species: Species index for the offspring.
        n_offspring: Number of offspring genotypes to create.
        rng: Random generator.

    Returns:
        GeneticState for the offspring schools.
    """
    # Filter to conspecific parents with positive gonad weight
    sp_mask = (species_id == offspring_species) & (gonad_weight > 0)
    sp_indices = np.where(sp_mask)[0]

    alleles: dict[str, NDArray[np.float64]] = {}
    env_noise: dict[str, NDArray[np.float64]] = {}

    for name, trait in parent_gs.registry.traits.items():
        max_loci = parent_gs.alleles[name].shape[1]
        n_loc = int(trait.n_loci[offspring_species])
        off_alleles = np.zeros((n_offspring, max_loci, 2), dtype=np.float64)

        if len(sp_indices) > 0:
            sp_gonad = gonad_weight[sp_indices]
            sp_alleles = parent_gs.alleles[name][sp_indices]

            for j in range(n_offspring):
                # Select parents from conspecific pool
                pa, pb = select_parents(sp_gonad, rng)
                off_alleles[j, :n_loc, 0] = form_gamete(sp_alleles[pa, :n_loc, :], rng)
                off_alleles[j, :n_loc, 1] = form_gamete(sp_alleles[pb, :n_loc, :], rng)

        # Environmental noise drawn at birth
        ev = float(trait.env_var[offspring_species])
        if ev > 0:
            noise = rng.normal(0.0, np.sqrt(ev), size=n_offspring)
        else:
            noise = np.zeros(n_offspring)

        alleles[name] = off_alleles
        env_noise[name] = noise

    return GeneticState(alleles=alleles, env_noise=env_noise, registry=parent_gs.registry)
```

Update `osmose/engine/genetics/__init__.py`:

```python
# osmose/engine/genetics/__init__.py
"""Ev-OSMOSE eco-evolutionary genetics module."""

from osmose.engine.genetics.expression import apply_trait_overrides, express_traits
from osmose.engine.genetics.genotype import (
    GeneticState,
    compact_genetic_state,
    create_initial_genotypes,
)
from osmose.engine.genetics.inheritance import (
    create_offspring_genotypes,
    form_gamete,
    select_parents,
)
from osmose.engine.genetics.trait import Trait, TraitRegistry

__all__ = [
    "GeneticState",
    "Trait",
    "TraitRegistry",
    "apply_trait_overrides",
    "compact_genetic_state",
    "create_initial_genotypes",
    "create_offspring_genotypes",
    "express_traits",
    "form_gamete",
    "select_parents",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_genetics_inheritance.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/genetics/inheritance.py osmose/engine/genetics/__init__.py tests/test_genetics_inheritance.py
git commit -m "feat(genetics): add gametic inheritance with fecundity-weighted parent selection"
```

---

## Task 5: FleetConfig, FleetState, and Config Parsing

**Files:**
- Create: `osmose/engine/economics/__init__.py`
- Create: `osmose/engine/economics/fleet.py`
- Test: `tests/test_economics_fleet.py`

### Context

`FleetConfig` is a frozen dataclass parsed from OSMOSE config keys like `economic.fleet.name.fsh0`. `FleetState` holds mutable per-vessel arrays (position, days used, revenue) plus the effort map. For MVP, we support a single fleet with basic config. `create_fleet_state` initializes all vessels at their fleet's home port.

- [ ] **Step 1: Write tests for fleet config parsing and state creation**

```python
# tests/test_economics_fleet.py
"""Tests for FleetConfig, FleetState, and config parsing."""

import numpy as np
import pytest

from osmose.engine.economics.fleet import FleetConfig, FleetState, create_fleet_state, parse_fleets


class TestParseFleets:
    def test_single_fleet(self):
        cfg = {
            "simulation.economic.enabled": "true",
            "economic.fleet.number": "1",
            "economic.fleet.name.fsh0": "Trawlers",
            "economic.fleet.nvessels.fsh0": "10",
            "economic.fleet.homeport.y.fsh0": "2",
            "economic.fleet.homeport.x.fsh0": "3",
            "economic.fleet.gear.fsh0": "bottom_trawl",
            "economic.fleet.max.days.fsh0": "200",
            "economic.fleet.fuel.cost.fsh0": "500.0",
            "economic.fleet.operating.cost.fsh0": "1000.0",
            "economic.fleet.target.species.fsh0": "0,1",
            "economic.fleet.price.sp0.fsh0": "2500.0",
            "economic.fleet.price.sp1.fsh0": "1800.0",
            "economic.fleet.stock.elasticity.sp0.fsh0": "0.5",
            "economic.fleet.stock.elasticity.sp1.fsh0": "0.3",
        }
        fleets = parse_fleets(cfg, n_species=2)
        assert len(fleets) == 1
        f = fleets[0]
        assert f.name == "Trawlers"
        assert f.n_vessels == 10
        assert f.home_port_y == 2
        assert f.home_port_x == 3
        assert f.target_species == [0, 1]
        assert f.price_per_tonne[0] == pytest.approx(2500.0)
        assert f.stock_elasticity[1] == pytest.approx(0.3)

    def test_empty_when_disabled(self):
        fleets = parse_fleets({}, n_species=2)
        assert len(fleets) == 0


class TestCreateFleetState:
    def test_vessels_start_at_home_port(self):
        fleet = FleetConfig(
            name="Trawlers",
            n_vessels=5,
            home_port_y=2,
            home_port_x=3,
            gear_type="bottom_trawl",
            max_days_at_sea=200,
            fuel_cost_per_cell=500.0,
            base_operating_cost=1000.0,
            stock_elasticity=np.array([0.5]),
            target_species=[0],
            price_per_tonne=np.array([2500.0]),
        )
        state = create_fleet_state(
            fleets=[fleet],
            grid_ny=5,
            grid_nx=5,
            rationality=1.0,
            memory_decay=0.7,
        )
        assert len(state.vessel_fleet) == 5
        assert np.all(state.vessel_cell_y == 2)
        assert np.all(state.vessel_cell_x == 3)
        assert np.all(state.vessel_days_used == 0)
        assert state.effort_map.shape == (1, 5, 5)
        assert state.rationality == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_economics_fleet.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.engine.economics'`

- [ ] **Step 3: Implement fleet.py**

```python
# osmose/engine/economics/__init__.py
"""DSVM fleet dynamics bioeconomic module."""

from osmose.engine.economics.fleet import FleetConfig, FleetState, create_fleet_state, parse_fleets

__all__ = ["FleetConfig", "FleetState", "create_fleet_state", "parse_fleets"]
```

```python
# osmose/engine/economics/fleet.py
"""Fleet configuration, state, and config parsing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FleetConfig:
    """Immutable per-fleet configuration."""

    name: str
    n_vessels: int
    home_port_y: int
    home_port_x: int
    gear_type: str
    max_days_at_sea: int
    fuel_cost_per_cell: float
    base_operating_cost: float
    stock_elasticity: NDArray[np.float64]   # shape (n_species,)
    target_species: list[int]
    price_per_tonne: NDArray[np.float64]    # shape (n_species,)


@dataclass
class FleetState:
    """Mutable per-simulation fleet state."""

    fleets: list[FleetConfig]

    # Per-vessel arrays (total_vessels = sum of fleet.n_vessels)
    vessel_fleet: NDArray[np.int32]         # fleet index per vessel
    vessel_cell_y: NDArray[np.int32]        # current y position
    vessel_cell_x: NDArray[np.int32]        # current x position
    vessel_days_used: NDArray[np.int32]     # days at sea this year
    vessel_revenue: NDArray[np.float64]     # cumulative revenue this year
    vessel_costs: NDArray[np.float64]       # cumulative costs this year

    # Effort map: shape (n_fleets, grid_ny, grid_nx)
    effort_map: NDArray[np.float64]

    # Catch memory: shape (n_fleets, grid_ny, grid_nx)
    catch_memory: NDArray[np.float64]
    memory_decay: float

    # Rationality parameter for logit choice
    rationality: float


def parse_fleets(cfg: dict[str, str], n_species: int) -> list[FleetConfig]:
    """Parse fleet definitions from OSMOSE config.

    Keys follow pattern: economic.fleet.<param>.fsh<i>
    """
    n_fleets = int(cfg.get("economic.fleet.number", "0"))
    if n_fleets == 0:
        return []

    fleets: list[FleetConfig] = []
    for fi in range(n_fleets):
        fid = f"fsh{fi}"
        prefix = "economic.fleet"

        name = cfg.get(f"{prefix}.name.{fid}", f"Fleet{fi}")
        n_vessels = int(cfg.get(f"{prefix}.nvessels.{fid}", "1"))
        home_y = int(cfg.get(f"{prefix}.homeport.y.{fid}", "0"))
        home_x = int(cfg.get(f"{prefix}.homeport.x.{fid}", "0"))
        gear = cfg.get(f"{prefix}.gear.{fid}", "generic")
        max_days = int(cfg.get(f"{prefix}.max.days.{fid}", "200"))
        fuel_cost = float(cfg.get(f"{prefix}.fuel.cost.{fid}", "0.0"))
        op_cost = float(cfg.get(f"{prefix}.operating.cost.{fid}", "0.0"))

        # Target species: comma-separated indices
        target_str = cfg.get(f"{prefix}.target.species.{fid}", "")
        target_species = [int(s.strip()) for s in target_str.split(",") if s.strip()]

        # Per-species price and elasticity
        price = np.array(
            [float(cfg.get(f"{prefix}.price.sp{sp}.{fid}", "0.0")) for sp in range(n_species)]
        )
        elasticity = np.array(
            [float(cfg.get(f"{prefix}.stock.elasticity.sp{sp}.{fid}", "0.0"))
             for sp in range(n_species)]
        )

        fleets.append(FleetConfig(
            name=name,
            n_vessels=n_vessels,
            home_port_y=home_y,
            home_port_x=home_x,
            gear_type=gear,
            max_days_at_sea=max_days,
            fuel_cost_per_cell=fuel_cost,
            base_operating_cost=op_cost,
            stock_elasticity=elasticity,
            target_species=target_species,
            price_per_tonne=price,
        ))

    return fleets


def create_fleet_state(
    fleets: list[FleetConfig],
    grid_ny: int,
    grid_nx: int,
    rationality: float = 1.0,
    memory_decay: float = 0.7,
) -> FleetState:
    """Initialize fleet state with all vessels at their home ports."""
    total_vessels = sum(f.n_vessels for f in fleets)
    n_fleets = len(fleets)

    vessel_fleet = np.empty(total_vessels, dtype=np.int32)
    vessel_cell_y = np.empty(total_vessels, dtype=np.int32)
    vessel_cell_x = np.empty(total_vessels, dtype=np.int32)

    offset = 0
    for fi, fleet in enumerate(fleets):
        end = offset + fleet.n_vessels
        vessel_fleet[offset:end] = fi
        vessel_cell_y[offset:end] = fleet.home_port_y
        vessel_cell_x[offset:end] = fleet.home_port_x
        offset = end

    return FleetState(
        fleets=fleets,
        vessel_fleet=vessel_fleet,
        vessel_cell_y=vessel_cell_y,
        vessel_cell_x=vessel_cell_x,
        vessel_days_used=np.zeros(total_vessels, dtype=np.int32),
        vessel_revenue=np.zeros(total_vessels, dtype=np.float64),
        vessel_costs=np.zeros(total_vessels, dtype=np.float64),
        effort_map=np.zeros((n_fleets, grid_ny, grid_nx), dtype=np.float64),
        catch_memory=np.zeros((n_fleets, grid_ny, grid_nx), dtype=np.float64),
        memory_decay=memory_decay,
        rationality=rationality,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_economics_fleet.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/economics/__init__.py osmose/engine/economics/fleet.py tests/test_economics_fleet.py
git commit -m "feat(economics): add FleetConfig, FleetState, and config parsing"
```

---

## Task 6: DSVM Discrete Choice (Revenue-Only Logit)

**Files:**
- Create: `osmose/engine/economics/choice.py`
- Modify: `osmose/engine/economics/__init__.py`
- Test: `tests/test_economics_choice.py`

### Context

The DSVM decision model computes expected revenue per cell for each fleet, then applies a multinomial logit to assign vessels to cells. For MVP: revenue-only (no travel costs, no memory blending). The choice set includes a port option with `V(port) = 0`.

The logit: `P(c) = exp(β × V(c)) / Σ exp(β × V(c'))` where β is the rationality parameter.

`fleet_decision` takes SchoolState (for biomass per cell) and FleetState, updates vessel positions and the effort map. `aggregate_effort` counts vessels per cell per fleet.

Biomass per cell is computed by summing `state.biomass` grouped by `(cell_y, cell_x)` for target species only.

- [ ] **Step 1: Write tests for logit choice and effort aggregation**

```python
# tests/test_economics_choice.py
"""Tests for DSVM discrete choice and effort aggregation."""

import numpy as np
import pytest

from osmose.engine.economics.choice import aggregate_effort, fleet_decision, logit_probabilities
from osmose.engine.economics.fleet import FleetConfig, FleetState, create_fleet_state


class TestLogitProbabilities:
    def test_uniform_when_beta_zero(self):
        """β=0 → uniform probability across all cells."""
        values = np.array([10.0, 20.0, 5.0, 0.0])  # 3 cells + port
        probs = logit_probabilities(values, beta=0.0)
        assert probs.shape == (4,)
        assert np.allclose(probs, 0.25)

    def test_deterministic_when_beta_large(self):
        """Large β → probability concentrated on highest-value cell."""
        values = np.array([10.0, 100.0, 5.0, 0.0])
        probs = logit_probabilities(values, beta=50.0)
        assert probs[1] > 0.99

    def test_probabilities_sum_to_one(self):
        values = np.array([1.0, 2.0, 3.0, 0.0])
        probs = logit_probabilities(values, beta=1.0)
        assert np.sum(probs) == pytest.approx(1.0)


class TestAggregateEffort:
    def test_counts_vessels_per_cell(self):
        vessel_fleet = np.array([0, 0, 0], dtype=np.int32)
        vessel_cell_y = np.array([0, 0, 1], dtype=np.int32)
        vessel_cell_x = np.array([1, 1, 0], dtype=np.int32)
        effort = aggregate_effort(vessel_fleet, vessel_cell_y, vessel_cell_x, n_fleets=1, ny=2, nx=2)
        assert effort.shape == (1, 2, 2)
        assert effort[0, 0, 1] == 2.0  # two vessels at (0,1)
        assert effort[0, 1, 0] == 1.0  # one vessel at (1,0)
        assert effort[0, 0, 0] == 0.0
        assert effort[0, 1, 1] == 0.0


class TestFleetDecision:
    def _make_fleet_and_state(self, n_vessels: int = 20) -> tuple[FleetConfig, FleetState]:
        fleet = FleetConfig(
            name="Trawlers",
            n_vessels=n_vessels,
            home_port_y=0,
            home_port_x=0,
            gear_type="bottom_trawl",
            max_days_at_sea=200,
            fuel_cost_per_cell=0.0,
            base_operating_cost=0.0,
            stock_elasticity=np.array([0.0]),
            target_species=[0],
            price_per_tonne=np.array([1000.0]),
        )
        state = create_fleet_state([fleet], grid_ny=3, grid_nx=3, rationality=1.0)
        return fleet, state

    def test_vessels_move_to_fish(self):
        """With high rationality, vessels should concentrate where biomass is highest."""
        fleet, fs = self._make_fleet_and_state(n_vessels=100)

        # Biomass map: all fish in cell (1, 1)
        biomass_by_cell = np.zeros((3, 3), dtype=np.float64)
        biomass_by_cell[1, 1] = 1000.0

        rng = np.random.default_rng(42)
        fs = fleet_decision(
            fleet_state=fs,
            biomass_by_cell_species=biomass_by_cell.reshape(1, 3, 3),  # (n_species, ny, nx)
            rng=rng,
        )
        # Most vessels should be at (1,1)
        at_target = np.sum((fs.vessel_cell_y == 1) & (fs.vessel_cell_x == 1))
        assert at_target > 50  # with rationality=1.0, most should go there

    def test_port_option_chosen_when_no_fish(self):
        """When no biomass anywhere, vessels should stay at port (home)."""
        fleet, fs = self._make_fleet_and_state(n_vessels=50)
        biomass_by_cell = np.zeros((1, 3, 3), dtype=np.float64)
        rng = np.random.default_rng(42)
        fs = fleet_decision(fleet_state=fs, biomass_by_cell_species=biomass_by_cell, rng=rng)
        # All values are 0, port V=0 too, so should be roughly uniform
        # But with no biomass, revenue=0 everywhere, port is equally attractive
        # Vessels stay at port or go randomly — check effort sums correctly
        assert fs.effort_map.sum() == pytest.approx(50.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_economics_choice.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.engine.economics.choice'`

- [ ] **Step 3: Implement choice.py**

```python
# osmose/engine/economics/choice.py
"""DSVM discrete choice model: multinomial logit for vessel location decisions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.economics.fleet import FleetState


def logit_probabilities(
    values: NDArray[np.float64],
    beta: float,
) -> NDArray[np.float64]:
    """Multinomial logit probabilities.

    P(i) = exp(β × V(i)) / Σ exp(β × V(j))

    Uses log-sum-exp trick for numerical stability.
    When β=0, returns uniform probabilities.
    """
    if beta == 0.0:
        n = len(values)
        return np.full(n, 1.0 / n)

    scaled = beta * values
    # Log-sum-exp trick
    max_v = scaled.max()
    exp_v = np.exp(scaled - max_v)
    return exp_v / exp_v.sum()


def aggregate_effort(
    vessel_fleet: NDArray[np.int32],
    vessel_cell_y: NDArray[np.int32],
    vessel_cell_x: NDArray[np.int32],
    n_fleets: int,
    ny: int,
    nx: int,
) -> NDArray[np.float64]:
    """Count vessels per cell per fleet → effort map (n_fleets, ny, nx)."""
    effort = np.zeros((n_fleets, ny, nx), dtype=np.float64)
    for i in range(len(vessel_fleet)):
        fi = vessel_fleet[i]
        cy = vessel_cell_y[i]
        cx = vessel_cell_x[i]
        if 0 <= cy < ny and 0 <= cx < nx:
            effort[fi, cy, cx] += 1.0
    return effort


def fleet_decision(
    fleet_state: FleetState,
    biomass_by_cell_species: NDArray[np.float64],
    rng: np.random.Generator,
) -> FleetState:
    """Execute DSVM decision for all vessels: compute expected revenue per cell, apply logit.

    MVP: Revenue-only (no travel costs, no memory blending).

    Args:
        fleet_state: Current fleet state (modified in place and returned).
        biomass_by_cell_species: Shape (n_species, grid_ny, grid_nx) — biomass per cell per species.
        rng: Random generator for stochastic choice.

    Returns:
        Updated FleetState with new vessel positions and effort map.
    """
    n_species, ny, nx = biomass_by_cell_species.shape
    n_cells = ny * nx

    for fi, fleet in enumerate(fleet_state.fleets):
        # Compute expected revenue per cell (sum over target species)
        revenue_map = np.zeros(n_cells, dtype=np.float64)
        for sp in fleet.target_species:
            if sp < n_species:
                # Revenue = biomass × price (MVP: catchability = 1, selectivity = 1)
                revenue_map += biomass_by_cell_species[sp].ravel() * fleet.price_per_tonne[sp]

        # Add port option: V(port) = 0
        values = np.append(revenue_map, 0.0)  # last element = port

        # Compute logit probabilities
        probs = logit_probabilities(values, fleet_state.rationality)

        # Assign each vessel in this fleet
        vessel_mask = fleet_state.vessel_fleet == fi
        vessel_indices = np.where(vessel_mask)[0]

        for vi in vessel_indices:
            choice = rng.choice(len(values), p=probs)
            if choice == n_cells:
                # Port — return to home port
                fleet_state.vessel_cell_y[vi] = fleet.home_port_y
                fleet_state.vessel_cell_x[vi] = fleet.home_port_x
            else:
                # Chosen cell
                fleet_state.vessel_cell_y[vi] = choice // nx
                fleet_state.vessel_cell_x[vi] = choice % nx

    # Update effort map
    fleet_state.effort_map = aggregate_effort(
        fleet_state.vessel_fleet,
        fleet_state.vessel_cell_y,
        fleet_state.vessel_cell_x,
        n_fleets=len(fleet_state.fleets),
        ny=ny,
        nx=nx,
    )

    return fleet_state
```

Update `osmose/engine/economics/__init__.py`:

```python
# osmose/engine/economics/__init__.py
"""DSVM fleet dynamics bioeconomic module."""

from osmose.engine.economics.choice import aggregate_effort, fleet_decision, logit_probabilities
from osmose.engine.economics.fleet import FleetConfig, FleetState, create_fleet_state, parse_fleets

__all__ = [
    "FleetConfig",
    "FleetState",
    "aggregate_effort",
    "create_fleet_state",
    "fleet_decision",
    "logit_probabilities",
    "parse_fleets",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_economics_choice.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/economics/choice.py osmose/engine/economics/__init__.py tests/test_economics_choice.py
git commit -m "feat(economics): add DSVM logit choice model and effort aggregation"
```

---

## Task 7: EngineConfig Extensions (genetics_enabled, economics_enabled)

**Files:**
- Modify: `osmose/engine/config.py`
- No separate test file — tested via integration in Task 9

### Context

Add two boolean flags to `EngineConfig`: `genetics_enabled` and `economics_enabled`. These are parsed from `simulation.genetic.enabled` and `simulation.economic.enabled`. The `raw_config` field already exists and provides the full config dict to submodules for their own parsing.

- [ ] **Step 1: Read current EngineConfig tail to find where to add new fields**

Read `osmose/engine/config.py` starting at the bioen fields (around line 630) to find the right location for new fields.

The bioen fields follow a pattern: `bioen_enabled: bool = False` as a default field. Follow the same pattern.

- [ ] **Step 2: Add genetics_enabled and economics_enabled to EngineConfig**

Add these fields after the bioen-related fields (after `bioen_k_for`, around line 654), before the output flags:

```python
    # Ev-OSMOSE genetics toggle
    genetics_enabled: bool = False

    # DSVM fleet dynamics toggle
    economics_enabled: bool = False
```

- [ ] **Step 3: Parse the new flags in from_dict()**

In `EngineConfig.from_dict()`, after the bioenergetics parsing block, add:

```python
        # Ev-OSMOSE genetics
        genetics_enabled = _enabled(cfg, "simulation.genetic.enabled")

        # DSVM fleet economics
        economics_enabled = _enabled(cfg, "simulation.economic.enabled")
```

And include them in the `EngineConfig(...)` constructor call:

```python
            genetics_enabled=genetics_enabled,
            economics_enabled=economics_enabled,
```

- [ ] **Step 4: Run existing tests to verify nothing breaks**

Run: `.venv/bin/python -m pytest tests/test_engine_simulate.py tests/test_engine_parity.py -v --timeout=60`
Expected: All existing tests PASS (new flags default to `False`)

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/config.py
git commit -m "feat(config): add genetics_enabled and economics_enabled flags to EngineConfig"
```

---

## Task 8: SimulationContext Extensions and Loop Integration

**Files:**
- Modify: `osmose/engine/simulate.py`
- No separate test file — tested via integration in Task 9

### Context

This is the core wiring task. `SimulationContext` gains `genetic_state` and `fleet_state` fields. The simulation loop (`simulate()`) gets new code at specific insertion points:

1. **Initialization** (before the loop): If genetics enabled, parse `TraitRegistry`, create `GeneticState`. If economics enabled, parse fleets, create `FleetState`.
2. **Pre-mortality** (line 899): If economics enabled, compute biomass per cell per species, run `fleet_decision` to update effort map. Effort→fishing wiring is in Task 8b.
3. **Pre-growth** (before line 909): If genetics enabled, express traits and apply overrides.
4. **Post-reproduction** (after line 917): If genetics enabled, create offspring genotypes and append to `GeneticState`.
5. **Compact** (line 928): If genetics enabled, compact genetic state with same alive mask.

This task wires both modules into the loop. The effort map's integration with fishing mortality is handled in the next task (Task 8b).

- [ ] **Step 1: Add fields to SimulationContext**

In `osmose/engine/simulate.py`, modify the `SimulationContext` dataclass (lines 25–36):

```python
@dataclass
class SimulationContext:
    """Per-simulation mutable state -- replaces module-level globals."""

    diet_tracking_enabled: bool = False
    diet_matrix: NDArray[np.float64] | None = None
    tl_weighted_sum: NDArray[np.float64] | None = None
    config_dir: str = ""
    # Ev-OSMOSE genetics (None when disabled)
    genetic_state: GeneticState | None = None
    # DSVM fleet dynamics (None when disabled)
    fleet_state: FleetState | None = None
```

Add the necessary imports at the top of the file (use `TYPE_CHECKING` to avoid circular imports):

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osmose.engine.economics.fleet import FleetState
    from osmose.engine.genetics.genotype import GeneticState
```

Note: Since the file already has `from __future__ import annotations`, the TYPE_CHECKING imports will work. But the actual runtime imports happen inside the conditional blocks in `simulate()`.

- [ ] **Step 2: Add genetics/economics initialization in simulate()**

In the `simulate()` function, after the existing initialization code (around line 840, after `ctx` is created) but before the main loop, add:

```python
    # -- Ev-OSMOSE genetics initialization --
    if config.genetics_enabled:
        from osmose.engine.genetics import TraitRegistry, create_initial_genotypes

        trait_registry = TraitRegistry.from_config(config.raw_config, config.n_species)
        ctx.genetic_state = create_initial_genotypes(
            trait_registry, state.species_id, rng
        )

    # -- DSVM fleet economics initialization --
    if config.economics_enabled:
        from osmose.engine.economics import create_fleet_state, parse_fleets

        fleets = parse_fleets(config.raw_config, config.n_species)
        if fleets:
            rationality = float(config.raw_config.get("simulation.economic.rationality", "1.0"))
            memory_decay = float(config.raw_config.get("simulation.economic.memory.decay", "0.7"))
            ctx.fleet_state = create_fleet_state(
                fleets, grid_ny=grid.ny, grid_nx=grid.nx,
                rationality=rationality, memory_decay=memory_decay,
            )
```

- [ ] **Step 3: Add economics fleet decision before mortality**

In the main loop, before the `_mortality` call (line 899), add:

```python
        # -- DSVM fleet decision (before mortality) --
        if ctx.fleet_state is not None:
            from osmose.engine.economics import fleet_decision

            # Compute biomass per cell per species
            n_sp = config.n_species
            biomass_by_cell = np.zeros((n_sp, grid.ny, grid.nx), dtype=np.float64)
            for i in range(len(state)):
                sp = state.species_id[i]
                if sp < n_sp:
                    cy, cx = state.cell_y[i], state.cell_x[i]
                    if 0 <= cy < grid.ny and 0 <= cx < grid.nx:
                        biomass_by_cell[sp, cy, cx] += state.biomass[i]

            ctx.fleet_state = fleet_decision(ctx.fleet_state, biomass_by_cell, rng)
```

- [ ] **Step 4: Add genetics trait expression before growth**

Before the growth/bioen block (line 909), add:

```python
        # -- Genetics trait expression (before growth/bioen) --
        trait_overrides: dict[str, NDArray[np.float64]] = {}
        if ctx.genetic_state is not None:
            from osmose.engine.genetics import apply_trait_overrides, express_traits

            phenotypes = express_traits(ctx.genetic_state, state.species_id)
            apply_trait_overrides(trait_overrides, phenotypes, ctx.genetic_state.registry)
```

Note: For MVP, `trait_overrides` is computed but not yet consumed by growth/bioen. The actual consumption by growth/bioen happens in Phase 2 when all 4 traits are integrated. The override dict is ready for use — this step validates that expression works end-to-end in the loop.

- [ ] **Step 5: Add genetics inheritance after reproduction**

After the reproduction block (after line 917), add:

```python
        # -- Genetics inheritance (after reproduction) --
        if ctx.genetic_state is not None:
            from osmose.engine.genetics import create_offspring_genotypes

            # Count how many new schools were added by reproduction
            n_new = len(state) - n_before_repro
            if n_new > 0:
                # Create offspring genotypes for each new egg school
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
                    ))
                # Append all offspring genotypes to genetic state
                for part in offspring_parts:
                    ctx.genetic_state = ctx.genetic_state.append(part)
```

Also, capture `n_before_repro = len(state)` just before the reproduction call:

```python
        n_before_repro = len(state)
        if config.bioen_enabled:
            state = _bioen_reproduction(state, config, step, rng, grid_ny=grid.ny, grid_nx=grid.nx)
        else:
            state = _reproduction(state, config, step, rng, grid_ny=grid.ny, grid_nx=grid.nx)
```

- [ ] **Step 6: Add genetics compact sync after state.compact()**

Replace the compact call (line 928) with:

```python
        # Compact dead schools — sync genetic state with same mask
        if ctx.genetic_state is not None:
            alive = state.abundance > 0
            from osmose.engine.genetics import compact_genetic_state

            ctx.genetic_state = compact_genetic_state(ctx.genetic_state, alive)
        state = state.compact()
```

- [ ] **Step 7: Run existing tests to verify nothing breaks**

Run: `.venv/bin/python -m pytest tests/test_engine_simulate.py -v`
Expected: All existing tests PASS (genetics/economics are disabled by default)

- [ ] **Step 8: Commit**

```bash
git add osmose/engine/simulate.py
git commit -m "feat(simulate): wire genetics and economics into simulation loop"
```

---

## Task 8b: Effort Map → Fishing Mortality Integration

**Files:**
- Modify: `osmose/engine/processes/mortality.py` (function `_precompute_effective_rates`)
- Modify: `osmose/engine/simulate.py` (pass `ctx` through to mortality — already done)
- Test: `tests/test_economics_choice.py` (append effort-fishing test)

### Context

The spec MVP requires: "Effort map feeds into fishing mortality via optional parameter." The effort map from `FleetState` must scale the prescribed fishing rate for species targeted by active fleets. This happens inside `_precompute_effective_rates` in `mortality.py`, which already computes `eff_fishing` per school.

The approach: `mortality()` already receives `ctx: SimulationContext | None`. When `ctx.fleet_state` is not None, for each school of a targeted species, multiply the prescribed fishing rate by the total vessel effort in that school's cell. Non-targeted species retain their prescribed rates unchanged.

This is a minimal integration — the effort acts as a spatial multiplier on existing rates. Phase 2 will replace this with the full effort-based mortality formula `F = Σ_fleets effort × catchability × selectivity`.

- [ ] **Step 1: Write test for effort-scaled fishing mortality**

Append to `tests/test_economics_choice.py`:

```python
from osmose.engine.config import EngineConfig
from osmose.engine.state import SchoolState


class TestEffortFishingIntegration:
    def test_effort_scales_fishing_mortality(self):
        """Fishing mortality should be higher where fleet effort is concentrated."""
        cfg_dict = {
            "simulation.time.ndtperyear": "12",
            "simulation.time.nyear": "1",
            "simulation.nspecies": "1",
            "simulation.nschool.sp0": "2",
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
            "mortality.fishing.rate.sp0": "0.5",
        }
        config = EngineConfig.from_dict(cfg_dict)

        # Two schools in different cells, same species, same abundance
        state = SchoolState.create(n_schools=2, species_id=np.array([0, 0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0, 1000.0]),
            biomass=np.array([100.0, 100.0]),
            length=np.array([15.0, 15.0]),
            weight=np.array([0.1, 0.1]),
            age_dt=np.array([24, 24], dtype=np.int32),
            cell_y=np.array([0, 1], dtype=np.int32),
            cell_x=np.array([0, 0], dtype=np.int32),
        )

        # Build effort map: 10 vessels in cell (0,0), 0 in cell (1,0)
        fleet = FleetConfig(
            name="Trawlers",
            n_vessels=10,
            home_port_y=0,
            home_port_x=0,
            gear_type="bottom_trawl",
            max_days_at_sea=200,
            fuel_cost_per_cell=0.0,
            base_operating_cost=0.0,
            stock_elasticity=np.array([0.0]),
            target_species=[0],
            price_per_tonne=np.array([1000.0]),
        )
        fs = create_fleet_state([fleet], grid_ny=2, grid_nx=1, rationality=1.0)
        # All vessels at (0,0)
        fs.effort_map[0, 0, 0] = 10.0
        fs.effort_map[0, 1, 0] = 0.0

        from osmose.engine.processes.mortality import _precompute_effective_rates

        # Without fleet state — both schools get same fishing rate
        _, _, eff_fishing_base, _ = _precompute_effective_rates(state, config, 10, 0)
        assert eff_fishing_base[0] == eff_fishing_base[1]  # same species, same rate
        assert eff_fishing_base[0] > 0  # fishing is enabled

        # With fleet state — school at (0,0) should have higher fishing, school at (1,0) should have zero
        _, _, eff_fishing_effort, _ = _precompute_effective_rates(
            state, config, 10, 0, fleet_state=fs
        )

        # School 0 (cell with 10 vessels) should have fishing > 0
        assert eff_fishing_effort[0] > 0
        # School 1 (cell with 0 vessels) should have zero fishing for targeted species
        assert eff_fishing_effort[1] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_economics_choice.py::TestEffortFishingIntegration -v`
Expected: FAIL — `_precompute_effective_rates() got an unexpected keyword argument 'fleet_state'`

- [ ] **Step 3: Add fleet_state parameter to _precompute_effective_rates**

In `osmose/engine/processes/mortality.py`, modify `_precompute_effective_rates` (line 533):

Change the signature from:
```python
def _precompute_effective_rates(work_state, config, n_subdt, step):
```
to:
```python
def _precompute_effective_rates(work_state, config, n_subdt, step, fleet_state=None):
```

Then, after the existing fishing rate computation (after line 644, before the return), add effort scaling:

```python
        # Scale fishing by fleet effort when economic module is active
        if fleet_state is not None:
            # Build per-school effort factor: sum of fleet vessels in the school's cell,
            # but only for species targeted by at least one fleet
            targeted_species: set[int] = set()
            for fleet_cfg in fleet_state.fleets:
                targeted_species.update(fleet_cfg.target_species)

            effort_factor = np.zeros(n, dtype=np.float64)
            for i in range(n):
                sp_id = work_state.species_id[i]
                if sp_id in targeted_species:
                    cy, cx = work_state.cell_y[i], work_state.cell_x[i]
                    ny, nx = fleet_state.effort_map.shape[1], fleet_state.effort_map.shape[2]
                    if 0 <= cy < ny and 0 <= cx < nx:
                        effort_factor[i] = fleet_state.effort_map[:, cy, cx].sum()

            # Replace prescribed fishing with effort-scaled fishing for targeted species
            for i in range(n):
                if work_state.species_id[i] in targeted_species:
                    eff_fishing[i] *= effort_factor[i]
```

- [ ] **Step 4: Pass fleet_state from mortality() to _precompute_effective_rates**

In `mortality.py`, there is exactly one call site for `_precompute_effective_rates` at line 1536:

```python
    eff_s, eff_a, eff_f, f_disc = _precompute_effective_rates(work_state, config, n_subdt, step)
```

Before this line, extract `fleet_state` from the context:

```python
    fleet_state = ctx.fleet_state if ctx is not None else None
```

Then update the call:

```python
    eff_s, eff_a, eff_f, f_disc = _precompute_effective_rates(
        work_state, config, n_subdt, step, fleet_state=fleet_state
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_economics_choice.py::TestEffortFishingIntegration -v`
Expected: PASS

- [ ] **Step 6: Run existing fishing tests to verify no regression**

Run: `.venv/bin/python -m pytest tests/test_engine_mortality.py tests/test_engine_simulate.py -v --timeout=60`
Expected: All existing tests PASS (fleet_state defaults to None, so no behavior change)

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/processes/mortality.py tests/test_economics_choice.py
git commit -m "feat(economics): integrate effort map into fishing mortality scaling"
```

---

## Task 9: Integration Tests — Both Modules Enabled

**Files:**
- Create: `tests/test_genetics_integration.py`
- Create: `tests/test_economics_integration.py`

### Context

Smoke tests that run a full simulation with each module enabled independently and both together. These verify:
1. No crashes when enabled.
2. Genetics state stays in sync with school state across compact/reproduction.
3. Fleet effort map is populated.
4. All 1766+ existing tests still pass with both disabled.

- [ ] **Step 1: Write genetics integration test**

```python
# tests/test_genetics_integration.py
"""Integration test: full simulation with Ev-OSMOSE genetics enabled."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import simulate


def _genetics_config() -> dict[str, str]:
    """Minimal config with genetics enabled and 1 evolving trait (imax)."""
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
        # Genetics
        "simulation.genetic.enabled": "true",
        "evolution.trait.imax.mean.sp0": "3.5",
        "evolution.trait.imax.var.sp0": "0.1",
        "evolution.trait.imax.envvar.sp0": "0.05",
        "evolution.trait.imax.nlocus.sp0": "5",
        "evolution.trait.imax.nval.sp0": "20",
        "evolution.trait.imax.target": "ingestion_rate",
    }


class TestGeneticsIntegration:
    def test_simulation_completes_with_genetics(self):
        """Full sim with genetics should complete without errors."""
        cfg = EngineConfig.from_dict(_genetics_config())
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert len(outputs) == 12

    def test_genetics_disabled_matches_baseline(self):
        """Disabling genetics should produce identical results to no-genetics config."""
        base_cfg = _genetics_config()
        base_cfg["simulation.genetic.enabled"] = "false"
        cfg_off = EngineConfig.from_dict(base_cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        outputs_off = simulate(cfg_off, grid, np.random.default_rng(42))

        # With genetics disabled, same RNG seed should give same biomass
        cfg_plain = _genetics_config()
        del cfg_plain["simulation.genetic.enabled"]
        del cfg_plain["evolution.trait.imax.mean.sp0"]
        del cfg_plain["evolution.trait.imax.var.sp0"]
        del cfg_plain["evolution.trait.imax.envvar.sp0"]
        del cfg_plain["evolution.trait.imax.nlocus.sp0"]
        del cfg_plain["evolution.trait.imax.nval.sp0"]
        del cfg_plain["evolution.trait.imax.target"]
        cfg_none = EngineConfig.from_dict(cfg_plain)
        outputs_none = simulate(cfg_none, grid, np.random.default_rng(42))

        for a, b in zip(outputs_off, outputs_none):
            np.testing.assert_array_almost_equal(a.biomass, b.biomass)
```

- [ ] **Step 2: Write economics integration test**

```python
# tests/test_economics_integration.py
"""Integration test: full simulation with DSVM fleet economics enabled."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import simulate


def _economics_config() -> dict[str, str]:
    """Minimal config with economics enabled and 1 fleet."""
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
        # Economics
        "simulation.economic.enabled": "true",
        "simulation.economic.rationality": "1.0",
        "simulation.economic.memory.decay": "0.7",
        "economic.fleet.number": "1",
        "economic.fleet.name.fsh0": "Trawlers",
        "economic.fleet.nvessels.fsh0": "10",
        "economic.fleet.homeport.y.fsh0": "1",
        "economic.fleet.homeport.x.fsh0": "1",
        "economic.fleet.gear.fsh0": "bottom_trawl",
        "economic.fleet.max.days.fsh0": "200",
        "economic.fleet.fuel.cost.fsh0": "0.0",
        "economic.fleet.operating.cost.fsh0": "0.0",
        "economic.fleet.target.species.fsh0": "0",
        "economic.fleet.price.sp0.fsh0": "1000.0",
        "economic.fleet.stock.elasticity.sp0.fsh0": "0.0",
    }


class TestEconomicsIntegration:
    def test_simulation_completes_with_economics(self):
        """Full sim with economics should complete without errors."""
        cfg = EngineConfig.from_dict(_economics_config())
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert len(outputs) == 12

    def test_both_modules_enabled(self):
        """Simulation with BOTH genetics and economics enabled should complete."""
        cfg_dict = _economics_config()
        # Add genetics config
        cfg_dict["simulation.genetic.enabled"] = "true"
        cfg_dict["evolution.trait.imax.mean.sp0"] = "3.5"
        cfg_dict["evolution.trait.imax.var.sp0"] = "0.1"
        cfg_dict["evolution.trait.imax.envvar.sp0"] = "0.05"
        cfg_dict["evolution.trait.imax.nlocus.sp0"] = "5"
        cfg_dict["evolution.trait.imax.nval.sp0"] = "20"
        cfg_dict["evolution.trait.imax.target"] = "ingestion_rate"

        cfg = EngineConfig.from_dict(cfg_dict)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert len(outputs) == 12
```

- [ ] **Step 3: Run integration tests**

Run: `.venv/bin/python -m pytest tests/test_genetics_integration.py tests/test_economics_integration.py -v`
Expected: 4 PASSED

- [ ] **Step 4: Run full test suite to verify backward compatibility**

Run: `.venv/bin/python -m pytest --timeout=120 -x -q`
Expected: 1766+ tests PASS, 0 failures

- [ ] **Step 5: Commit**

```bash
git add tests/test_genetics_integration.py tests/test_economics_integration.py
git commit -m "test: add integration tests for genetics and economics modules"
```

---

## Task 10: Lint, Final Verification, and Tag

**Files:**
- No new files — verification only

- [ ] **Step 1: Run ruff lint**

Run: `.venv/bin/ruff check osmose/engine/genetics/ osmose/engine/economics/ tests/test_genetics_*.py tests/test_economics_*.py`
Expected: No errors. Fix any that appear.

- [ ] **Step 2: Run ruff format**

Run: `.venv/bin/ruff format osmose/engine/genetics/ osmose/engine/economics/ tests/test_genetics_*.py tests/test_economics_*.py`
Expected: Files formatted (or already formatted).

- [ ] **Step 3: Run full test suite one final time**

Run: `.venv/bin/python -m pytest --timeout=120 -q`
Expected: All tests pass including the ~10 new ones.

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -u
git commit -m "style: ruff format genetics and economics modules"
```

- [ ] **Step 5: Verify file structure matches spec**

Run: `find osmose/engine/genetics osmose/engine/economics -type f -name '*.py' | sort`
Expected:
```
osmose/engine/economics/__init__.py
osmose/engine/economics/choice.py
osmose/engine/economics/fleet.py
osmose/engine/genetics/__init__.py
osmose/engine/genetics/expression.py
osmose/engine/genetics/genotype.py
osmose/engine/genetics/inheritance.py
osmose/engine/genetics/trait.py
```

---

## Summary

| Task | Module | What | New Tests |
|------|--------|------|-----------|
| 1 | Genetics | Trait + TraitRegistry + config parsing | 4 |
| 2 | Genetics | GeneticState + compact/append sync | 3 |
| 3 | Genetics | Trait expression + phenotype override | 3 |
| 4 | Genetics | Gametic inheritance | 4 |
| 5 | Economics | FleetConfig + FleetState + config parsing | 3 |
| 6 | Economics | DSVM logit choice + effort aggregation | 6 |
| 7 | Config | genetics_enabled + economics_enabled flags | 0 (existing tests verify) |
| 8 | Simulate | SimulationContext + loop wiring | 0 (integration tests verify) |
| 8b | Economics | Effort map → fishing mortality scaling | 1 |
| 9 | Integration | Both modules smoke tests | 4 |
| 10 | QA | Lint, format, final verification | 0 |
| **Total** | | | **28 tests** |

**Phase 2 and 3** will be planned separately after Phase 1 MVP is complete and validated.
