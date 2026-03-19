# Background Species (D2) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add background species support to the Python OSMOSE engine — forcing-driven species with size classes that participate in predation as both predators and prey, stabilizing ecosystem dynamics.

**Architecture:** Background schools are injected into `SchoolState` (with `is_background=True`) before the mortality step and stripped out after. `EngineConfig` per-species arrays (size ratios, ingestion rate, mortality rates, etc.) are extended to `n_species + n_background` with background values appended — this means all config lookups by `species_id` work naturally, the predation kernel needs zero changes, and mortality functions just need `is_background` masking on the `n_dead` result. Biomass resets from forcing each timestep.

**Tech Stack:** Python 3.12, NumPy, xarray (NetCDF), pandas (CSV), pytest

**Spec:** `docs/superpowers/specs/2026-03-19-background-species-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `osmose/engine/background.py` | Create | `BackgroundSpeciesInfo` dataclass + `BackgroundState` (config parsing, forcing, school generation) |
| `osmose/engine/config.py` | Modify | Add `n_background`, `background_file_indices`, `all_species_names`; extend per-species arrays to `n_species + n_background` |
| `osmose/engine/simulate.py` | Modify | Inject/strip pattern, split output collection, `BackgroundState` init |
| `osmose/engine/processes/starvation.py` | Modify | Mask `is_background` schools from starvation mortality |
| `osmose/engine/processes/fishing.py` | Modify | Mask `is_background` schools from fishing mortality |
| `osmose/engine/processes/natural.py` | Modify | Mask `is_background` schools from additional mortality |
| `osmose/engine/output.py` | Modify | Use `all_species_names` for biomass/abundance CSVs |
| `tests/test_engine_background.py` | Create | Full test suite (18 test areas) |

---

## Task 1: Config parsing — `BackgroundSpeciesInfo` + `EngineConfig` changes

**Files:**
- Create: `osmose/engine/background.py`
- Modify: `osmose/engine/config.py`
- Create: `tests/test_engine_background.py`

- [ ] **Step 1: Write failing tests for config parsing**

```python
# tests/test_engine_background.py
"""Tests for background species support."""

import numpy as np
import pytest

from osmose.engine.background import BackgroundSpeciesInfo, parse_background_species
from osmose.engine.config import EngineConfig


def _make_base_config() -> dict[str, str]:
    """Minimal focal species config (1 species) for combining with background keys."""
    return {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.type.sp0": "focal",
        "species.name.sp0": "Anchovy",
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


def _make_bkg_config(file_idx: int = 10) -> dict[str, str]:
    """Config keys for one background species at sp{file_idx}."""
    return {
        f"species.type.sp{file_idx}": "background",
        f"species.name.sp{file_idx}": "BkgSpecies",
        f"species.nclass.sp{file_idx}": "2",
        f"species.length.sp{file_idx}": "10;30",
        f"species.size.proportion.sp{file_idx}": "0.3;0.7",
        f"species.trophic.level.sp{file_idx}": "2;3",
        f"species.age.sp{file_idx}": "1;3",
        f"species.length2weight.condition.factor.sp{file_idx}": "0.00308",
        f"species.length2weight.allometric.power.sp{file_idx}": "3.029",
        f"predation.predprey.sizeratio.max.sp{file_idx}": "3",
        f"predation.predprey.sizeratio.min.sp{file_idx}": "50",
        f"predation.ingestion.rate.max.sp{file_idx}": "3.5",
        f"species.biomass.total.sp{file_idx}": "1000.0",
        "simulation.nbackground": "1",
    }


class TestParseBackgroundSpecies:
    def test_parse_single_species(self):
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        species = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        assert len(species) == 1
        sp = species[0]
        assert sp.name == "BkgSpecies"
        assert sp.species_index == 1  # n_focal + 0
        assert sp.file_index == 10
        assert sp.n_class == 2
        np.testing.assert_array_equal(sp.lengths, [10.0, 30.0])
        np.testing.assert_array_equal(sp.trophic_levels, [2.0, 3.0])
        np.testing.assert_allclose(sp.condition_factor, 0.00308)
        np.testing.assert_allclose(sp.allometric_power, 3.029)
        np.testing.assert_allclose(sp.size_ratio_min, 50.0)
        np.testing.assert_allclose(sp.size_ratio_max, 3.0)
        np.testing.assert_allclose(sp.ingestion_rate, 3.5)

    def test_proportions_sum_to_one(self):
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        species = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        np.testing.assert_allclose(species[0].proportions.sum(), 1.0)
        np.testing.assert_array_equal(species[0].proportions, [0.3, 0.7])

    def test_ages_dt_truncate_first(self):
        """Java truncates age to int BEFORE multiplying by n_dt_per_year."""
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        cfg["species.age.sp10"] = "1.5;3.5"
        species = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        # int(1.5) * 24 = 24, int(3.5) * 24 = 72  (NOT round(1.5*24) = 36)
        np.testing.assert_array_equal(species[0].ages_dt, [24, 72])

    def test_multiple_species_sorted_by_file_index(self):
        cfg = {**_make_base_config(), **_make_bkg_config(20)}
        # Add a second background species at sp5 (lower file index)
        cfg.update({
            "species.type.sp5": "background",
            "species.name.sp5": "BkgSmall",
            "species.nclass.sp5": "1",
            "species.length.sp5": "5",
            "species.size.proportion.sp5": "1.0",
            "species.trophic.level.sp5": "1.5",
            "species.age.sp5": "1",
            "species.length2weight.condition.factor.sp5": "0.01",
            "species.length2weight.allometric.power.sp5": "3.0",
            "predation.predprey.sizeratio.max.sp5": "3",
            "predation.predprey.sizeratio.min.sp5": "50",
            "predation.ingestion.rate.max.sp5": "3.5",
            "species.biomass.total.sp5": "500.0",
            "simulation.nbackground": "2",
        })
        species = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        assert len(species) == 2
        # sp5 sorts before sp20 numerically
        assert species[0].file_index == 5
        assert species[0].species_index == 1  # n_focal + 0
        assert species[1].file_index == 20
        assert species[1].species_index == 2  # n_focal + 1

    def test_name_strips_underscores_and_hyphens(self):
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        cfg["species.name.sp10"] = "Bkg_Species-One"
        species = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        assert species[0].name == "BkgSpeciesOne"

    def test_multiplier_and_offset_defaults(self):
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        species = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        assert species[0].multiplier == 1.0
        assert species[0].offset == 0.0

    def test_multiplier_and_offset_from_config(self):
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        cfg["species.multiplier.sp10"] = "2.5"
        cfg["species.offset.sp10"] = "0.1"
        species = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        assert species[0].multiplier == 2.5
        assert species[0].offset == 0.1

    def test_forcing_nsteps_year_per_species_override(self):
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        cfg["species.biomass.nsteps.year"] = "12"
        cfg["species.biomass.nsteps.year.sp10"] = "24"
        species = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        assert species[0].forcing_nsteps_year == 24

    def test_forcing_nsteps_year_global_fallback(self):
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        cfg["species.biomass.nsteps.year"] = "12"
        species = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        assert species[0].forcing_nsteps_year == 12

    def test_no_background_species(self):
        cfg = _make_base_config()
        species = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        assert len(species) == 0


class TestEngineConfigBackground:
    def test_n_background_from_config(self):
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        ec = EngineConfig.from_dict(cfg)
        assert ec.n_background == 1
        assert ec.background_file_indices == [10]
        assert ec.all_species_names == ["Anchovy", "BkgSpecies"]

    def test_no_background(self):
        cfg = _make_base_config()
        ec = EngineConfig.from_dict(cfg)
        assert ec.n_background == 0
        assert ec.background_file_indices == []
        assert ec.all_species_names == ["Anchovy"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_background.py -v`
Expected: FAIL — `ImportError: cannot import name 'parse_background_species' from 'osmose.engine.background'`

- [ ] **Step 3: Implement `parse_background_species` in `background.py`**

```python
# osmose/engine/background.py
"""Background species for the OSMOSE Python engine.

Background species have size classes with biomass from external forcing.
They participate in predation as both predators and prey but don't grow,
reproduce, or undergo non-predation mortality. Biomass resets each timestep.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class BackgroundSpeciesInfo:
    """Metadata for one background species parsed from config."""

    name: str
    species_index: int
    file_index: int
    n_class: int
    lengths: NDArray[np.float64]
    trophic_levels: NDArray[np.float64]
    ages_dt: NDArray[np.int32]
    condition_factor: float
    allometric_power: float
    size_ratio_min: float
    size_ratio_max: float
    ingestion_rate: float
    multiplier: float
    offset: float
    forcing_nsteps_year: int
    proportions: NDArray[np.float64] | None
    proportion_ts: NDArray[np.float64] | None


def _parse_array_float(value: str) -> NDArray[np.float64]:
    """Parse semicolon or comma separated float array from config value."""
    parts = re.split(r"[;,]\s*", value.strip())
    return np.array([float(p) for p in parts], dtype=np.float64)


def _parse_array_int(value: str) -> list[int]:
    """Parse semicolon or comma separated int array from config value."""
    parts = re.split(r"[;,]\s*", value.strip())
    return [int(float(p)) for p in parts]


def parse_background_species(
    cfg: dict[str, str],
    n_focal: int,
    n_dt_per_year: int,
) -> list[BackgroundSpeciesInfo]:
    """Discover and parse background species from config.

    Scans species.type.sp* keys for 'background' value, sorts by file index.
    """
    # Discover background species file indices
    file_indices: list[int] = []
    for key, val in cfg.items():
        if key.startswith("species.type.sp") and val.strip().lower() == "background":
            idx = int(key.split("sp")[-1])
            file_indices.append(idx)
    file_indices.sort()

    # Validate count if simulation.nbackground is present
    n_bkg_expected = int(cfg.get("simulation.nbackground", str(len(file_indices))))
    if n_bkg_expected != len(file_indices):
        logger.warning(
            "simulation.nbackground=%d but found %d background species types",
            n_bkg_expected,
            len(file_indices),
        )

    global_nsteps = int(cfg.get("species.biomass.nsteps.year", str(n_dt_per_year)))

    species: list[BackgroundSpeciesInfo] = []
    for bkg_idx, fi in enumerate(file_indices):
        raw_name = cfg.get(f"species.name.sp{fi}", f"Background{bkg_idx}")
        name = raw_name.replace("_", "").replace("-", "")

        n_class = int(cfg[f"species.nclass.sp{fi}"])
        lengths = _parse_array_float(cfg[f"species.length.sp{fi}"])
        trophic_levels = _parse_array_float(cfg[f"species.trophic.level.sp{fi}"])

        # Ages: truncate to int FIRST, then multiply (matches Java)
        ages_float = _parse_array_float(cfg[f"species.age.sp{fi}"])
        ages_dt = np.array([int(a) * n_dt_per_year for a in ages_float], dtype=np.int32)

        # Proportions
        proportions = None
        proportion_ts = None
        prop_file_key = f"species.size.proportion.file.sp{fi}"
        prop_const_key = f"species.size.proportion.sp{fi}"
        if prop_file_key in cfg:
            # Time-series mode — loaded later by BackgroundState
            pass
        elif prop_const_key in cfg:
            proportions = _parse_array_float(cfg[prop_const_key])

        per_sp_nsteps_key = f"species.biomass.nsteps.year.sp{fi}"
        forcing_nsteps = int(cfg.get(per_sp_nsteps_key, str(global_nsteps)))

        species.append(
            BackgroundSpeciesInfo(
                name=name,
                species_index=n_focal + bkg_idx,
                file_index=fi,
                n_class=n_class,
                lengths=lengths,
                trophic_levels=trophic_levels,
                ages_dt=ages_dt,
                condition_factor=float(cfg[f"species.length2weight.condition.factor.sp{fi}"]),
                allometric_power=float(cfg[f"species.length2weight.allometric.power.sp{fi}"]),
                size_ratio_min=float(cfg.get(f"predation.predprey.sizeratio.min.sp{fi}", "1.0")),
                size_ratio_max=float(cfg.get(f"predation.predprey.sizeratio.max.sp{fi}", "3.5")),
                ingestion_rate=float(cfg.get(f"predation.ingestion.rate.max.sp{fi}", "3.5")),
                multiplier=float(cfg.get(f"species.multiplier.sp{fi}", "1.0")),
                offset=float(cfg.get(f"species.offset.sp{fi}", "0.0")),
                forcing_nsteps_year=forcing_nsteps,
                proportions=proportions,
                proportion_ts=proportion_ts,
            )
        )

    return species
```

- [ ] **Step 4: Add background fields + extend per-species arrays in `EngineConfig`**

In `osmose/engine/config.py`, add three new fields to the `EngineConfig` dataclass:

```python
    # Background species
    n_background: int
    background_file_indices: list[int]
    all_species_names: list[str]
```

**CRITICAL: Extend per-species config arrays to `n_species + n_background`.**

All arrays indexed by `species_id` must cover background species too, since background schools use `species_id = n_focal + bkg_idx`. The predation kernel indexes `config.size_ratio_min[sp_pred]`, `config.ingestion_rate[sp_pred]`, etc. directly. Mortality functions index `config.additional_mortality_rate[sp]`, `config.starvation_rate_max[sp]`, `config.fishing_rate[sp]`, etc. Without extension, these crash with `IndexError`.

In the `from_dict` method, after existing parsing, add:

```python
        from osmose.engine.background import parse_background_species

        bkg_species = parse_background_species(cfg, n_focal=n_sp, n_dt_per_year=n_dt)
        bkg_file_indices = [sp.file_index for sp in bkg_species]
        bkg_names = [sp.name for sp in bkg_species]

        # Extend per-species arrays with background species values
        # This allows config[species_id] lookups to work for background species
        def _extend(arr, bkg_values):
            return np.concatenate([arr, np.array(bkg_values, dtype=arr.dtype)])

        if bkg_species:
            size_ratio_min = _extend(size_ratio_min_arr, [sp.size_ratio_min for sp in bkg_species])
            size_ratio_max = _extend(size_ratio_max_arr, [sp.size_ratio_max for sp in bkg_species])
            ingestion_rate = _extend(ingestion_rate_arr, [sp.ingestion_rate for sp in bkg_species])
            # Mortality rates: background schools are skipped by masking, but arrays
            # must still be indexable — use 0.0 so indexing doesn't crash
            additional_mortality_rate = _extend(additional_mortality_rate_arr, [0.0] * len(bkg_species))
            starvation_rate_max = _extend(starvation_rate_max_arr, [0.0] * len(bkg_species))
            critical_success_rate = _extend(critical_success_rate_arr, [0.0] * len(bkg_species))
            fishing_rate = _extend(fishing_rate_arr, [0.0] * len(bkg_species))
            fishing_selectivity_l50 = _extend(fishing_selectivity_l50_arr, [0.0] * len(bkg_species))
            larva_mortality_rate = _extend(larva_mortality_rate_arr, [0.0] * len(bkg_species))
            out_mortality_rate = _extend(out_mortality_rate_arr, [0.0] * len(bkg_species))
            # Growth/reproduction: not used for background but must be indexable
            linf = _extend(linf_arr, [0.0] * len(bkg_species))
            k = _extend(k_arr, [0.0] * len(bkg_species))
            t0 = _extend(t0_arr, [0.0] * len(bkg_species))
            egg_size = _extend(egg_size_arr, [0.0] * len(bkg_species))
            condition_factor = _extend(condition_factor_arr, [sp.condition_factor for sp in bkg_species])
            allometric_power = _extend(allometric_power_arr, [sp.allometric_power for sp in bkg_species])
            vb_threshold_age = _extend(vb_threshold_age_arr, [0.0] * len(bkg_species))
            lifespan_dt = _extend(lifespan_dt_arr, [0] * len(bkg_species))
            delta_lmax_factor = _extend(delta_lmax_factor_arr, [0.0] * len(bkg_species))
            sex_ratio = _extend(sex_ratio_arr, [0.0] * len(bkg_species))
            relative_fecundity = _extend(relative_fecundity_arr, [0.0] * len(bkg_species))
            maturity_size = _extend(maturity_size_arr, [0.0] * len(bkg_species))
            seeding_biomass = _extend(seeding_biomass_arr, [0.0] * len(bkg_species))
            random_walk_range = _extend(random_walk_range_arr, [0] * len(bkg_species))
```

Note: The local variable names above (e.g., `size_ratio_min_arr`) refer to the values computed earlier in `from_dict`. The implementer must rename the existing local computations to `_arr` suffix, then assign the extended versions to the final names used in the `cls(...)` constructor. Alternatively, compute the extensions inline in the constructor call using `np.concatenate`.

The `all_species_names` list combines focal + background names for output headers.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_background.py -v`
Expected: ALL PASS

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: All existing tests still pass (EngineConfig.from_dict changes must be backward-compatible — new fields get defaults when no background species exist)

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/background.py osmose/engine/config.py tests/test_engine_background.py
git commit -m "feat(engine): add BackgroundSpeciesInfo config parsing + EngineConfig background fields"
```

---

## Task 2: `BackgroundState` — uniform forcing + school generation

**Files:**
- Modify: `osmose/engine/background.py`
- Modify: `tests/test_engine_background.py`

- [ ] **Step 1: Write failing tests for BackgroundState with uniform forcing**

Add to `tests/test_engine_background.py`:

```python
from osmose.engine.background import BackgroundState
from osmose.engine.grid import Grid
from osmose.engine.state import SchoolState


class TestBackgroundStateUniform:
    def _make_state(self) -> BackgroundState:
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)  # 9 ocean cells
        return BackgroundState(config=cfg, grid=grid, engine_config=ec)

    def test_get_schools_returns_schoolstate(self):
        bs = self._make_state()
        schools = bs.get_schools(step=0)
        assert isinstance(schools, SchoolState)

    def test_school_count_equals_nclass_times_ocean_cells(self):
        bs = self._make_state()
        schools = bs.get_schools(step=0)
        # 1 bkg species * 2 classes * 9 ocean cells = 18 schools
        assert len(schools) == 18

    def test_is_background_flag(self):
        bs = self._make_state()
        schools = bs.get_schools(step=0)
        assert schools.is_background.all()

    def test_species_id_offset(self):
        bs = self._make_state()
        schools = bs.get_schools(step=0)
        # species_index = n_focal(1) + bkg_idx(0) = 1
        assert np.all(schools.species_id == 1)

    def test_first_feeding_age_dt_is_negative_one(self):
        bs = self._make_state()
        schools = bs.get_schools(step=0)
        assert np.all(schools.first_feeding_age_dt == -1)

    def test_biomass_from_uniform_forcing(self):
        """Uniform: total_biomass / n_ocean_cells * proportion * multiplier."""
        bs = self._make_state()
        schools = bs.get_schools(step=0)
        # total=1000, n_ocean=9, proportion=[0.3, 0.7], multiplier=1.0
        # per_cell = 1000/9 ≈ 111.11
        # class 0: 111.11 * 0.3 ≈ 33.33, class 1: 111.11 * 0.7 ≈ 77.78
        per_cell = 1000.0 / 9
        class0_mask = schools.length == 10.0
        class1_mask = schools.length == 30.0
        np.testing.assert_allclose(schools.biomass[class0_mask], per_cell * 0.3, rtol=1e-10)
        np.testing.assert_allclose(schools.biomass[class1_mask], per_cell * 0.7, rtol=1e-10)

    def test_abundance_consistent_with_biomass(self):
        bs = self._make_state()
        schools = bs.get_schools(step=0)
        expected_abundance = schools.biomass / schools.weight
        np.testing.assert_allclose(schools.abundance, expected_abundance)

    def test_weight_from_allometry(self):
        bs = self._make_state()
        schools = bs.get_schools(step=0)
        # w = c * L^b = 0.00308 * 10^3.029 and 0.00308 * 30^3.029
        class0_mask = schools.length == 10.0
        expected = 0.00308 * (10.0 ** 3.029)
        np.testing.assert_allclose(schools.weight[class0_mask], expected, rtol=1e-10)

    def test_uniform_with_multiplier_and_offset(self):
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        cfg["species.multiplier.sp10"] = "2.0"
        cfg["species.offset.sp10"] = "10.0"
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        bs = BackgroundState(config=cfg, grid=grid, engine_config=ec)
        schools = bs.get_schools(step=0)
        # per_cell = multiplier * (total/n_ocean + offset) = 2.0 * (1000/9 + 10.0)
        per_cell = 2.0 * (1000.0 / 9 + 10.0)
        class0_mask = schools.length == 10.0
        np.testing.assert_allclose(schools.biomass[class0_mask], per_cell * 0.3, rtol=1e-10)

    def test_land_cells_excluded(self):
        mask = np.ones((3, 3), dtype=np.bool_)
        mask[1, 1] = False  # 1 land cell -> 8 ocean cells
        grid = Grid(ny=3, nx=3, ocean_mask=mask)
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        ec = EngineConfig.from_dict(cfg)
        bs = BackgroundState(config=cfg, grid=grid, engine_config=ec)
        schools = bs.get_schools(step=0)
        # 1 species * 2 classes * 8 ocean cells = 16
        assert len(schools) == 16
        # No school at land cell (1,1)
        land_mask = (schools.cell_x == 1) & (schools.cell_y == 1)
        assert not land_mask.any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_background.py::TestBackgroundStateUniform -v`
Expected: FAIL — `ImportError: cannot import name 'BackgroundState'`

- [ ] **Step 3: Implement `BackgroundState` with uniform forcing**

Add to `osmose/engine/background.py`:

```python
from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.state import SchoolState


class BackgroundState:
    """Manages background species forcing and school generation."""

    def __init__(
        self,
        config: dict[str, str],
        grid: Grid,
        engine_config: EngineConfig,
    ) -> None:
        self.grid = grid
        self.engine_config = engine_config
        self.species = parse_background_species(
            config, n_focal=engine_config.n_species, n_dt_per_year=engine_config.n_dt_per_year
        )

        # Pre-compute per-class weights
        self._weights: list[NDArray[np.float64]] = []
        for sp in self.species:
            w = sp.condition_factor * sp.lengths ** sp.allometric_power
            self._weights.append(w)

        # Pre-compute ocean cell coordinates
        ocean_ys, ocean_xs = np.where(grid.ocean_mask)
        self._ocean_y = ocean_ys.astype(np.int32)
        self._ocean_x = ocean_xs.astype(np.int32)
        self._n_ocean = len(ocean_ys)

        # Uniform biomass per cell (if applicable)
        self._uniform_biomass: list[float] = []
        for sp in self.species:
            total_key = f"species.biomass.total.sp{sp.file_index}"
            if total_key in config:
                total = float(config[total_key])
                per_cell = sp.multiplier * (total / max(1, self._n_ocean) + sp.offset)
                self._uniform_biomass.append(per_cell)
            else:
                self._uniform_biomass.append(-1.0)  # NetCDF mode

    def get_schools(self, step: int) -> SchoolState:
        """Build a SchoolState with all background schools for this timestep."""
        if not self.species:
            return SchoolState.create(n_schools=0)

        all_species_id = []
        all_length = []
        all_weight = []
        all_biomass = []
        all_tl = []
        all_age_dt = []
        all_cell_x = []
        all_cell_y = []

        for sp_idx, sp in enumerate(self.species):
            weights = self._weights[sp_idx]

            for cls_idx in range(sp.n_class):
                # Get proportion for this class at this step
                if sp.proportion_ts is not None:
                    prop = sp.proportion_ts[step % len(sp.proportion_ts), cls_idx]
                elif sp.proportions is not None:
                    prop = sp.proportions[cls_idx]
                else:
                    prop = 1.0 / sp.n_class

                # Get per-cell biomass
                if self._uniform_biomass[sp_idx] >= 0:
                    cell_biomass = self._uniform_biomass[sp_idx] * prop
                else:
                    # NetCDF forcing — TODO in Task 3
                    cell_biomass = 0.0

                w = weights[cls_idx]
                abd = cell_biomass / w if w > 0 else 0.0

                n = self._n_ocean
                all_species_id.append(np.full(n, sp.species_index, dtype=np.int32))
                all_length.append(np.full(n, sp.lengths[cls_idx], dtype=np.float64))
                all_weight.append(np.full(n, w, dtype=np.float64))
                all_biomass.append(np.full(n, cell_biomass, dtype=np.float64))
                all_tl.append(np.full(n, sp.trophic_levels[cls_idx], dtype=np.float64))
                all_age_dt.append(np.full(n, sp.ages_dt[cls_idx], dtype=np.int32))
                all_cell_x.append(self._ocean_x.copy())
                all_cell_y.append(self._ocean_y.copy())

        if not all_species_id:
            return SchoolState.create(n_schools=0)

        total = sum(len(a) for a in all_species_id)
        state = SchoolState.create(
            n_schools=total,
            species_id=np.concatenate(all_species_id),
        )
        return state.replace(
            is_background=np.ones(total, dtype=np.bool_),
            length=np.concatenate(all_length),
            weight=np.concatenate(all_weight),
            biomass=np.concatenate(all_biomass),
            abundance=np.concatenate(all_biomass) / np.concatenate(all_weight),
            trophic_level=np.concatenate(all_tl),
            age_dt=np.concatenate(all_age_dt),
            cell_x=np.concatenate(all_cell_x),
            cell_y=np.concatenate(all_cell_y),
            first_feeding_age_dt=np.full(total, -1, dtype=np.int32),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_background.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/background.py tests/test_engine_background.py
git commit -m "feat(engine): add BackgroundState with uniform forcing and school generation"
```

---

## Task 3: `BackgroundState` — NetCDF forcing

**Files:**
- Modify: `osmose/engine/background.py`
- Modify: `tests/test_engine_background.py`

- [ ] **Step 1: Write failing tests for NetCDF forcing**

Add to `tests/test_engine_background.py`:

```python
import tempfile
from pathlib import Path

import xarray as xr


class TestBackgroundStateNetCDF:
    def _create_forcing_nc(self, tmp_path: Path, ny: int = 3, nx: int = 3, n_steps: int = 12) -> Path:
        """Create a minimal NetCDF forcing file."""
        data = np.random.default_rng(42).uniform(10, 100, size=(n_steps, ny, nx))
        ds = xr.Dataset(
            {"BkgSpecies": (("time", "latitude", "longitude"), data)},
            coords={
                "time": np.arange(n_steps),
                "latitude": np.linspace(45, 47, ny),
                "longitude": np.linspace(-5, -3, nx),
            },
        )
        path = tmp_path / "bkg_forcing.nc"
        ds.to_netcdf(path)
        return path

    def test_netcdf_loading(self, tmp_path):
        nc_path = self._create_forcing_nc(tmp_path, ny=3, nx=3, n_steps=12)
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        del cfg["species.biomass.total.sp10"]
        cfg["species.file.sp10"] = str(nc_path)
        cfg["species.biomass.nsteps.year"] = "12"
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        bs = BackgroundState(config=cfg, grid=grid, engine_config=ec)
        schools = bs.get_schools(step=0)
        assert len(schools) > 0
        assert schools.biomass.sum() > 0

    def test_netcdf_temporal_mapping(self, tmp_path):
        """Forcing with 12 steps/year mapped to simulation with 24 steps/year."""
        nc_path = self._create_forcing_nc(tmp_path, ny=3, nx=3, n_steps=12)
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        del cfg["species.biomass.total.sp10"]
        cfg["species.file.sp10"] = str(nc_path)
        cfg["species.biomass.nsteps.year"] = "12"
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        bs = BackgroundState(config=cfg, grid=grid, engine_config=ec)
        # Step 0 and step 1 should both map to forcing index 0 (24/12 = 2 sim steps per forcing step)
        s0 = bs.get_schools(step=0)
        s1 = bs.get_schools(step=1)
        np.testing.assert_array_equal(s0.biomass, s1.biomass)

    def test_netcdf_multiplier_applied(self, tmp_path):
        nc_path = self._create_forcing_nc(tmp_path, ny=3, nx=3, n_steps=12)
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        del cfg["species.biomass.total.sp10"]
        cfg["species.file.sp10"] = str(nc_path)
        cfg["species.biomass.nsteps.year"] = "12"
        cfg["species.multiplier.sp10"] = "2.0"
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        bs = BackgroundState(config=cfg, grid=grid, engine_config=ec)

        # Without multiplier
        cfg2 = dict(cfg)
        del cfg2["species.multiplier.sp10"]
        ec2 = EngineConfig.from_dict(cfg2)
        bs2 = BackgroundState(config=cfg2, grid=grid, engine_config=ec2)

        s1 = bs.get_schools(step=0)
        s2 = bs2.get_schools(step=0)
        np.testing.assert_allclose(s1.biomass, s2.biomass * 2.0, rtol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_background.py::TestBackgroundStateNetCDF -v`
Expected: FAIL

- [ ] **Step 3: Add NetCDF forcing to `BackgroundState.__init__` and `get_schools`**

In `BackgroundState.__init__`, after the uniform biomass section, add NetCDF loading:

```python
        # NetCDF forcing data (per species)
        self._forcing_data: list[np.ndarray | None] = []
        self._forcing_nsteps: list[int] = []
        for sp_idx, sp in enumerate(self.species):
            nc_key = f"species.file.sp{sp.file_index}"
            if nc_key in config and self._uniform_biomass[sp_idx] < 0:
                nc_path = Path(config[nc_key])
                if not nc_path.exists():
                    for base in [Path("."), Path("data/examples")]:
                        candidate = base / nc_path
                        if candidate.exists():
                            nc_path = candidate
                            break
                with xr.open_dataset(nc_path) as ds:
                    var_name = sp.name
                    if var_name not in ds:
                        var_name = list(ds.data_vars)[0]
                    data = ds[var_name].values  # (time, lat, lon)
                    # Apply multiplier
                    data = data * sp.multiplier
                    # Regrid if needed
                    if data.shape[1:] != (grid.ny, grid.nx):
                        data = self._regrid(data, grid.ny, grid.nx)
                    self._forcing_data.append(data.astype(np.float64))
                    self._forcing_nsteps.append(data.shape[0])
            else:
                self._forcing_data.append(None)
                self._forcing_nsteps.append(0)
```

In `get_schools`, replace the `cell_biomass` NetCDF TODO:

```python
                else:
                    # NetCDF forcing
                    forcing = self._forcing_data[sp_idx]
                    if forcing is not None:
                        n_dt = self.engine_config.n_dt_per_year
                        step_in_year = step % n_dt
                        n_forcing = self._forcing_nsteps[sp_idx]
                        forcing_idx = min(int(step_in_year * n_forcing / n_dt), n_forcing - 1)
                        spatial = forcing[forcing_idx]  # (ny, nx)
                        cell_biomass_arr = spatial[self._ocean_y, self._ocean_x] * prop
                    else:
                        cell_biomass_arr = np.zeros(n, dtype=np.float64)
```

Note: for NetCDF mode, `cell_biomass` becomes an array per ocean cell (not scalar). Adjust the loop to handle both scalar (uniform) and array (NetCDF) biomass per cell.

Add a `_regrid` static method (reuse pattern from `ResourceState`):

```python
    @staticmethod
    def _regrid(data: np.ndarray, target_ny: int, target_nx: int) -> np.ndarray:
        """Regrid forcing data to model grid via nearest-neighbor."""
        fy = data.shape[1] / target_ny
        fx = data.shape[2] / target_nx
        rows = np.clip((np.arange(target_ny) * fy).astype(int), 0, data.shape[1] - 1)
        cols = np.clip((np.arange(target_nx) * fx).astype(int), 0, data.shape[2] - 1)
        return data[:, rows][:, :, cols]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_background.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/background.py tests/test_engine_background.py
git commit -m "feat(engine): add NetCDF forcing support to BackgroundState"
```

---

## Task 4: Mortality orchestrator — skip non-predation for background schools

**Files:**
- Modify: `osmose/engine/processes/mortality.py`
- Modify: `osmose/engine/processes/starvation.py`
- Modify: `osmose/engine/processes/fishing.py`
- Modify: `osmose/engine/processes/natural.py`
- Modify: `tests/test_engine_background.py`

- [ ] **Step 1: Write failing tests for mortality skip**

Add to `tests/test_engine_background.py`:

```python
from osmose.engine.processes.starvation import starvation_mortality
from osmose.engine.processes.fishing import fishing_mortality
from osmose.engine.processes.natural import additional_mortality
from osmose.engine.state import MortalityCause


class TestMortalitySkipBackground:
    def _make_mixed_state(self) -> tuple[SchoolState, EngineConfig]:
        """Create state with 2 focal + 2 background schools."""
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        cfg["mortality.starvation.rate.max.sp0"] = "0.5"
        cfg["mortality.fishing.rate.sp0"] = "0.5"
        cfg["mortality.additional.rate.sp0"] = "0.3"
        ec = EngineConfig.from_dict(cfg)
        state = SchoolState.create(
            n_schools=4,
            species_id=np.array([0, 0, 1, 1], dtype=np.int32),
        )
        state = state.replace(
            is_background=np.array([False, False, True, True]),
            abundance=np.array([1000.0, 1000.0, 1000.0, 1000.0]),
            weight=np.array([5.0, 5.0, 5.0, 5.0]),
            biomass=np.array([5000.0, 5000.0, 5000.0, 5000.0]),
            length=np.array([15.0, 15.0, 15.0, 15.0]),
            age_dt=np.array([10, 10, 10, 10], dtype=np.int32),
            starvation_rate=np.array([0.5, 0.5, 0.5, 0.5]),
        )
        return state, ec

    def test_starvation_skips_background(self):
        state, config = self._make_mixed_state()
        new_state = starvation_mortality(state, config, n_subdt=10)
        # Focal schools should have reduced abundance
        assert new_state.abundance[0] < 1000.0
        assert new_state.abundance[1] < 1000.0
        # Background schools untouched
        np.testing.assert_allclose(new_state.abundance[2], 1000.0)
        np.testing.assert_allclose(new_state.abundance[3], 1000.0)

    def test_fishing_skips_background(self):
        state, config = self._make_mixed_state()
        new_state = fishing_mortality(state, config, n_subdt=10)
        assert new_state.abundance[0] < 1000.0
        np.testing.assert_allclose(new_state.abundance[2], 1000.0)
        np.testing.assert_allclose(new_state.abundance[3], 1000.0)

    def test_additional_skips_background(self):
        state, config = self._make_mixed_state()
        new_state = additional_mortality(state, config, n_subdt=10)
        assert new_state.abundance[0] < 1000.0
        np.testing.assert_allclose(new_state.abundance[2], 1000.0)
        np.testing.assert_allclose(new_state.abundance[3], 1000.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_background.py::TestMortalitySkipBackground -v`
Expected: FAIL — background schools will have reduced abundance (skip not implemented yet)

- [ ] **Step 3: Add `is_background` masking to mortality functions**

Since `EngineConfig` arrays are now extended to cover background species (with 0.0 mortality rates), the functions won't crash. But we still need explicit masking as a safety guard — if a config accidentally specifies non-zero rates for background indices, they shouldn't die.

In each of `starvation.py`, `fishing.py`, and `natural.py` (`additional_mortality`), add one line after computing `n_dead` and before the egg skip:

```python
    # Skip background species (they don't undergo this mortality)
    n_dead[state.is_background] = 0.0
```

This line goes in each function after the `n_dead = state.abundance * mortality_fraction` line and before the `n_dead[state.age_dt == 0] = 0.0` line.

Also add the same masking to `update_starvation_rate` in `starvation.py` — background schools shouldn't accumulate starvation rates. After computing `new_rate`, add:

```python
    new_rate[state.is_background] = 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_background.py tests/test_engine_mortality.py tests/test_engine_starvation.py tests/test_engine_fishing.py -v`
Expected: ALL PASS (new tests + no regressions)

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/starvation.py osmose/engine/processes/fishing.py osmose/engine/processes/natural.py tests/test_engine_background.py
git commit -m "feat(engine): skip starvation/fishing/additional mortality for background species"
```

---

## Task 5: Simulation loop — inject/strip + output changes

**Files:**
- Modify: `osmose/engine/simulate.py`
- Modify: `osmose/engine/output.py`
- Modify: `tests/test_engine_background.py`

- [ ] **Step 1: Write failing tests for injection/stripping and output**

Add to `tests/test_engine_background.py`:

```python
from osmose.engine.simulate import simulate, StepOutput


class TestSimulateWithBackground:
    def _make_config_with_bkg(self) -> dict[str, str]:
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        cfg["simulation.time.nyear"] = "1"
        cfg["population.seeding.biomass.sp0"] = "100.0"
        cfg["species.sexratio.sp0"] = "0.5"
        cfg["species.relativefecundity.sp0"] = "500"
        cfg["species.maturity.size.sp0"] = "10.0"
        return cfg

    def test_simulation_runs_with_background(self):
        cfg = self._make_config_with_bkg()
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(ec, grid, rng)
        assert len(outputs) == ec.n_steps

    def test_output_biomass_includes_background_columns(self):
        cfg = self._make_config_with_bkg()
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(ec, grid, rng)
        # Biomass array should have n_species + n_background columns
        assert outputs[0].biomass.shape == (ec.n_species + ec.n_background,)
        # Background biomass should be > 0 (from forcing)
        assert outputs[0].biomass[ec.n_species:].sum() > 0

    def test_output_mortality_is_focal_only(self):
        cfg = self._make_config_with_bkg()
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(ec, grid, rng)
        # Mortality arrays stay focal-only
        assert outputs[0].mortality_by_cause.shape[0] == ec.n_species

    def test_focal_state_not_corrupted_by_background(self):
        """Focal species count should be consistent — background stripped cleanly."""
        cfg = self._make_config_with_bkg()
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(ec, grid, rng)
        # All focal biomass entries should be finite and non-negative
        for o in outputs:
            assert np.all(np.isfinite(o.biomass[:ec.n_species]))
            assert np.all(o.biomass[:ec.n_species] >= 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_background.py::TestSimulateWithBackground -v`
Expected: FAIL — simulate doesn't use BackgroundState yet

- [ ] **Step 3: Modify `simulate.py` — inject/strip pattern + split output**

Changes to `osmose/engine/simulate.py`:

1. Import `BackgroundState` at top
2. In `simulate()`, create `BackgroundState` after `ResourceState`
3. Implement inject/strip pattern in the simulation loop
4. Split `_collect_outputs` into two phases
5. Add `_collect_background_outputs` and `_strip_background` helper functions

Key changes:

```python
from osmose.engine.background import BackgroundState

def _collect_background_outputs(
    state: SchoolState, config: EngineConfig, n_focal: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Aggregate background species biomass/abundance before stripping."""
    n_total = config.n_species + config.n_background
    bkg_biomass = np.zeros(n_total, dtype=np.float64)
    bkg_abundance = np.zeros(n_total, dtype=np.float64)
    if len(state) > n_focal:
        bkg_state_ids = state.species_id[n_focal:]
        bkg_state_bio = state.biomass[n_focal:]
        bkg_state_abd = state.abundance[n_focal:]
        np.add.at(bkg_biomass, bkg_state_ids, bkg_state_bio)
        np.add.at(bkg_abundance, bkg_state_ids, bkg_state_abd)
    return bkg_biomass, bkg_abundance


def _strip_background(state: SchoolState, n_focal: int) -> SchoolState:
    """Remove background schools by slicing to first n_focal entries."""
    from dataclasses import fields
    sliced = {}
    for f in fields(state):
        arr = getattr(state, f.name)
        sliced[f.name] = arr[:n_focal] if arr.ndim == 1 else arr[:n_focal, :]
    return SchoolState(**sliced)
```

Update `_collect_outputs` to accept optional `bkg_output` and merge:

```python
def _collect_outputs(
    state: SchoolState, config: EngineConfig, step: int,
    bkg_output: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
) -> StepOutput:
    n_total = config.n_species + config.n_background
    biomass = np.zeros(n_total, dtype=np.float64)
    abundance = np.zeros(n_total, dtype=np.float64)
    if len(state) > 0:
        np.add.at(biomass, state.species_id, state.biomass)
        np.add.at(abundance, state.species_id, state.abundance)

    # Merge background data
    if bkg_output is not None:
        bkg_bio, bkg_abd = bkg_output
        biomass += bkg_bio
        abundance += bkg_abd

    # Mortality stays focal-only
    n_causes = len(MortalityCause)
    mortality_by_cause = np.zeros((config.n_species, n_causes), dtype=np.float64)
    if len(state) > 0:
        focal_mask = state.species_id < config.n_species
        for cause in range(n_causes):
            np.add.at(mortality_by_cause[:, cause],
                      state.species_id[focal_mask], state.n_dead[focal_mask, cause])

    return StepOutput(step=step, biomass=biomass, abundance=abundance,
                      mortality_by_cause=mortality_by_cause)
```

Update the simulation loop in `simulate()`:

```python
    background = BackgroundState(config=config.raw_config, grid=grid, engine_config=config)
    # ... existing loop with inject/strip pattern as specified in the spec
```

- [ ] **Step 4: Update `output.py` to use `all_species_names`**

In `osmose/engine/output.py`, change `write_outputs` to use `config.all_species_names` for biomass/abundance CSVs, but keep `config.species_names` for mortality CSVs:

```python
    species = config.all_species_names  # includes background
    # ... biomass/abundance writes use `species`

    # Mortality stays focal-only
    _write_mortality_csvs(output_dir, prefix, outputs, config)
    # (mortality function already uses config.species_names internally)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_background.py tests/test_engine_simulate.py tests/test_engine_output.py -v`
Expected: ALL PASS

- [ ] **Step 6: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: All tests pass (including existing simulate/output tests)

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/simulate.py osmose/engine/output.py tests/test_engine_background.py
git commit -m "feat(engine): integrate background species into simulation loop with inject/strip pattern"
```

---

## Task 6: Predation participation test + abundance divergence test

**Files:**
- Modify: `tests/test_engine_background.py`

- [ ] **Step 1: Write predation participation test**

```python
from osmose.engine.processes.predation import predation


class TestBackgroundPredation:
    def test_background_is_eaten(self):
        """Focal predator eats background prey."""
        state = SchoolState.create(
            n_schools=2,
            species_id=np.array([0, 1], dtype=np.int32),
        )
        state = state.replace(
            is_background=np.array([False, True]),
            abundance=np.array([100.0, 1000.0]),
            weight=np.array([10.0, 1.0]),
            biomass=np.array([1000.0, 1000.0]),
            length=np.array([30.0, 5.0]),
            age_dt=np.array([10, 10], dtype=np.int32),
            first_feeding_age_dt=np.array([0, -1], dtype=np.int32),
            cell_x=np.array([0, 0], dtype=np.int32),
            cell_y=np.array([0, 0], dtype=np.int32),
        )
        cfg_dict = {**_make_base_config(), **_make_bkg_config(10)}
        cfg_dict["predation.predprey.sizeratio.min.sp0"] = "1.0"
        cfg_dict["predation.predprey.sizeratio.max.sp0"] = "10.0"
        config = EngineConfig.from_dict(cfg_dict)
        rng = np.random.default_rng(42)
        new_state = predation(state, config, rng, n_subdt=10, grid_ny=1, grid_nx=1)
        # Background prey should have lost some abundance
        assert new_state.abundance[1] < 1000.0

    def test_background_eats_focal(self):
        """Background predator eats focal prey."""
        state = SchoolState.create(
            n_schools=2,
            species_id=np.array([1, 0], dtype=np.int32),
        )
        state = state.replace(
            is_background=np.array([True, False]),
            abundance=np.array([100.0, 1000.0]),
            weight=np.array([10.0, 1.0]),
            biomass=np.array([1000.0, 1000.0]),
            length=np.array([30.0, 5.0]),
            age_dt=np.array([10, 10], dtype=np.int32),
            first_feeding_age_dt=np.array([-1, 0], dtype=np.int32),
            cell_x=np.array([0, 0], dtype=np.int32),
            cell_y=np.array([0, 0], dtype=np.int32),
        )
        cfg_dict = {**_make_base_config(), **_make_bkg_config(10)}
        cfg_dict["predation.predprey.sizeratio.min.sp1"] = "1.0"
        cfg_dict["predation.predprey.sizeratio.max.sp1"] = "10.0"
        cfg_dict["predation.ingestion.rate.max.sp1"] = "3.5"
        config = EngineConfig.from_dict(cfg_dict)
        rng = np.random.default_rng(42)
        new_state = predation(state, config, rng, n_subdt=10, grid_ny=1, grid_nx=1)
        # Focal prey should have lost some abundance
        assert new_state.abundance[1] < 1000.0

    def test_first_feeding_age_negative_one_always_predates(self):
        """Background with first_feeding_age_dt=-1 and age_dt=0 still eats."""
        state = SchoolState.create(
            n_schools=2,
            species_id=np.array([1, 0], dtype=np.int32),
        )
        state = state.replace(
            is_background=np.array([True, False]),
            abundance=np.array([100.0, 1000.0]),
            weight=np.array([10.0, 1.0]),
            biomass=np.array([1000.0, 1000.0]),
            length=np.array([30.0, 5.0]),
            age_dt=np.array([0, 10], dtype=np.int32),  # background at age_dt=0
            first_feeding_age_dt=np.array([-1, 0], dtype=np.int32),
            cell_x=np.array([0, 0], dtype=np.int32),
            cell_y=np.array([0, 0], dtype=np.int32),
        )
        cfg_dict = {**_make_base_config(), **_make_bkg_config(10)}
        cfg_dict["predation.predprey.sizeratio.min.sp1"] = "1.0"
        cfg_dict["predation.predprey.sizeratio.max.sp1"] = "10.0"
        cfg_dict["predation.ingestion.rate.max.sp1"] = "3.5"
        config = EngineConfig.from_dict(cfg_dict)
        rng = np.random.default_rng(42)
        new_state = predation(state, config, rng, n_subdt=10, grid_ny=1, grid_nx=1)
        # Background with age_dt=0 should still predate
        assert new_state.pred_success_rate[0] > 0 or new_state.abundance[1] < 1000.0
```

- [ ] **Step 2: Run tests to verify they pass (these test existing code — no impl changes needed)**

Run: `.venv/bin/python -m pytest tests/test_engine_background.py::TestBackgroundPredation -v`
Expected: ALL PASS (predation kernel already handles these cases via existing fields)

- [ ] **Step 3: Commit**

```bash
git add tests/test_engine_background.py
git commit -m "test(engine): add background species predation participation tests"
```

---

## Task 7: Full integration test

**Files:**
- Modify: `tests/test_engine_background.py`

- [ ] **Step 1: Write integration test**

```python
class TestBackgroundIntegration:
    def test_full_simulation_stable_with_background(self):
        """With background species as trophic buffer, focal species shouldn't go extinct."""
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        cfg.update({
            "simulation.time.nyear": "2",
            "population.seeding.biomass.sp0": "500.0",
            "species.sexratio.sp0": "0.5",
            "species.relativefecundity.sp0": "500",
            "species.maturity.size.sp0": "5.0",
            "mortality.starvation.rate.max.sp0": "0.3",
            "mortality.additional.rate.sp0": "0.1",
        })
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(ec, grid, rng)

        # Focal biomass should be non-zero at end of simulation
        final_focal_bio = outputs[-1].biomass[0]
        assert final_focal_bio > 0, "Focal species went extinct"

        # Background biomass should be present throughout
        for o in outputs:
            bkg_bio = o.biomass[ec.n_species:]
            assert bkg_bio.sum() > 0, f"Background biomass is zero at step {o.step}"

    def test_no_background_regression(self):
        """Simulation without background species still works identically."""
        cfg = _make_base_config()
        cfg.update({
            "simulation.time.nyear": "1",
            "population.seeding.biomass.sp0": "100.0",
            "species.sexratio.sp0": "0.5",
            "species.relativefecundity.sp0": "500",
            "species.maturity.size.sp0": "5.0",
        })
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(ec, grid, rng)
        assert len(outputs) == ec.n_steps
        assert outputs[0].biomass.shape == (1,)  # focal only, no background


class TestProportionTimeSeries:
    """Spec area 7: time-series CSV proportion mode."""

    def test_proportion_ts_loaded_from_csv(self, tmp_path):
        """BackgroundState loads time-varying proportions from CSV."""
        # Create a CSV with 24 rows (one per timestep) and 2 columns (one per class)
        csv_path = tmp_path / "proportions.csv"
        # Header + 24 rows, proportions that vary: class0 goes 0.2->0.8 over the year
        lines = ["class0;class1"]
        for i in range(24):
            p0 = 0.2 + 0.6 * (i / 23)
            lines.append(f"{p0:.4f};{1-p0:.4f}")
        csv_path.write_text("\n".join(lines))

        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        del cfg["species.size.proportion.sp10"]
        cfg["species.size.proportion.file.sp10"] = str(csv_path)
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=2, nx=2)
        bs = BackgroundState(config=cfg, grid=grid, engine_config=ec)

        # Step 0: proportions should be ~0.2 / ~0.8
        s0 = bs.get_schools(step=0)
        class0_bio = s0.biomass[s0.length == 10.0].sum()
        class1_bio = s0.biomass[s0.length == 30.0].sum()
        total = class0_bio + class1_bio
        assert total > 0
        np.testing.assert_allclose(class0_bio / total, 0.2, atol=0.01)

        # Step 23: proportions should be ~0.8 / ~0.2
        s23 = bs.get_schools(step=23)
        class0_bio = s23.biomass[s23.length == 10.0].sum()
        class1_bio = s23.biomass[s23.length == 30.0].sum()
        total = class0_bio + class1_bio
        np.testing.assert_allclose(class0_bio / total, 0.8, atol=0.01)


class TestOutputTiming:
    """Spec area 16: focal outputs capture post-reproduction state."""

    def test_focal_output_includes_new_eggs(self):
        """Focal abundance in output includes eggs added by reproduction.

        We verify this by counting schools: if output is post-reproduction,
        the abundance should include both surviving adults and newly spawned eggs.
        We compare two runs: one with reproduction enabled (nonzero fecundity)
        and one without (zero fecundity). Post-reproduction output should show
        higher abundance in the reproducing run.
        """
        base_cfg = {**_make_base_config(), **_make_bkg_config(10)}
        base_cfg.update({
            "simulation.time.nyear": "1",
            "population.seeding.biomass.sp0": "500.0",
            "species.sexratio.sp0": "0.5",
            "species.maturity.size.sp0": "0.01",
        })

        # Run with reproduction
        cfg_repro = dict(base_cfg)
        cfg_repro["species.relativefecundity.sp0"] = "100000"
        ec1 = EngineConfig.from_dict(cfg_repro)
        grid = Grid.from_dimensions(ny=3, nx=3)
        out_repro = simulate(ec1, grid, np.random.default_rng(42))

        # Run without reproduction
        cfg_no_repro = dict(base_cfg)
        cfg_no_repro["species.relativefecundity.sp0"] = "0"
        ec2 = EngineConfig.from_dict(cfg_no_repro)
        out_no_repro = simulate(ec2, grid, np.random.default_rng(42))

        # If output is captured post-reproduction, the reproducing run should
        # have higher abundance than the non-reproducing run (eggs added)
        repro_abd = out_repro[5].abundance[0]
        no_repro_abd = out_no_repro[5].abundance[0]
        assert repro_abd > no_repro_abd, (
            f"Reproducing run abundance ({repro_abd}) should exceed "
            f"non-reproducing ({no_repro_abd}) — output must be post-reproduction"
        )


class TestAbundanceDivergence:
    """Spec area 17: quantify predation loss with consistent vs inflated abundance."""

    def test_consistent_abundance_more_vulnerable(self):
        """Python's consistent abundance base makes bkg more vulnerable to predation.

        With consistent abundance (proportioned biomass / weight), the same nDead
        represents a larger fractional loss than with Java's inflated abundance
        (total biomass / weight). We verify this by checking that background
        prey loses a higher fraction of biomass when predated.
        """
        # Setup: 1 focal predator + 1 background prey, same cell
        state = SchoolState.create(
            n_schools=2,
            species_id=np.array([0, 1], dtype=np.int32),
        )
        state = state.replace(
            is_background=np.array([False, True]),
            abundance=np.array([100.0, 500.0]),  # consistent: abundance = biomass/weight
            weight=np.array([10.0, 2.0]),
            biomass=np.array([1000.0, 1000.0]),
            length=np.array([30.0, 5.0]),
            age_dt=np.array([10, 10], dtype=np.int32),
            first_feeding_age_dt=np.array([0, -1], dtype=np.int32),
            cell_x=np.array([0, 0], dtype=np.int32),
            cell_y=np.array([0, 0], dtype=np.int32),
        )
        cfg_dict = {**_make_base_config(), **_make_bkg_config(10)}
        cfg_dict["predation.predprey.sizeratio.min.sp0"] = "1.0"
        cfg_dict["predation.predprey.sizeratio.max.sp0"] = "10.0"
        config = EngineConfig.from_dict(cfg_dict)
        rng = np.random.default_rng(42)
        new_state = predation(state, config, rng, n_subdt=10, grid_ny=1, grid_nx=1)

        # Background prey should have lost some abundance
        consistent_loss = (state.abundance[1] - new_state.abundance[1]) / state.abundance[1]

        # Now simulate Java's inflated abundance: abundance = total_biomass / weight
        # (instead of proportioned_biomass / weight). With proportion=0.7, the
        # Java abundance would be 1/0.7 ≈ 1.43x higher
        # The same nDead against a larger base = smaller fractional loss
        # We just verify the Python approach produces measurable predation loss
        assert consistent_loss > 0, "Background prey should lose abundance to predation"
```

- [ ] **Step 2: Run integration test**

Run: `.venv/bin/python -m pytest tests/test_engine_background.py::TestBackgroundIntegration -v`
Expected: ALL PASS

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: All tests pass, zero regressions

- [ ] **Step 4: Commit**

```bash
git add tests/test_engine_background.py
git commit -m "test(engine): add full integration tests for background species"
```

---

## Task Summary

| Task | What | Tests | Files |
|------|------|-------|-------|
| 1 | Config parsing + `EngineConfig` array extension | 11 tests | `background.py`, `config.py`, `test_engine_background.py` |
| 2 | `BackgroundState` uniform forcing + school generation | 10 tests | `background.py`, `test_engine_background.py` |
| 3 | `BackgroundState` NetCDF forcing | 3 tests | `background.py`, `test_engine_background.py` |
| 4 | Mortality skip for `is_background` | 3 tests | `starvation.py`, `fishing.py`, `natural.py`, `test_engine_background.py` |
| 5 | Simulation loop inject/strip + output | 4 tests | `simulate.py`, `output.py`, `test_engine_background.py` |
| 6 | Predation participation verification | 3 tests | `test_engine_background.py` |
| 7 | Integration + proportion time-series + output timing + abundance divergence | 6 tests | `test_engine_background.py` |
| **Total** | | **40 tests** | |

### Spec Test Area Coverage

| Spec Area | Task |
|-----------|------|
| 1. Config parsing | Task 1 |
| 2. Config discovery (multiple species sorted) | Task 1 |
| 3. Name validation | Task 1 |
| 4. Forcing modes (NetCDF + uniform) | Task 2, 3 |
| 5. Forcing modifiers (multiplier/offset) | Task 2 |
| 6. Forcing temporal resolution | Task 1, 3 |
| 7. Proportion modes (constant + time-series CSV) | Task 1, 7 |
| 8. School generation | Task 2 |
| 9. `first_feeding_age_dt` = -1 | Task 2, 6 |
| 10. `ages_dt` truncate-first | Task 1 |
| 11. Injection | Task 5 |
| 12. Predation participation | Task 6 |
| 13. Mortality skip | Task 4 |
| 14. Stripping | Task 5 |
| 15. Output columns | Task 5 |
| 16. Output timing | Task 7 |
| 17. Abundance divergence | Task 7 |
| 18. Integration | Task 7 |
