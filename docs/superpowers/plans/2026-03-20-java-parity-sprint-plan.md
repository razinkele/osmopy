# Java Parity Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Java feature parity for the Python OSMOSE engine — growth dispatch, size/age distribution outputs, per-species RNG, and bioenergetic (Ev-OSMOSE) module.

**Architecture:** Four phases in dependency order. Phases 1-3 are independent (parallelizable). Phase 4 (bioenergetics) depends on all three. Each phase extends the existing Structure-of-Arrays engine with new process modules and config parsing, following the established TDD pattern.

**Tech Stack:** Python 3.12, NumPy, Numba (optional JIT), pandas (CSV output), xarray/netCDF4 (forcing data), pytest

**Spec:** `docs/superpowers/specs/2026-03-20-java-parity-sprint-design.md`

---

## File Structure

### Phase 1 — Code Quality
| File | Action | Responsibility |
|------|--------|----------------|
| `osmose/engine/processes/predation.py` | Modify | Add diet matrix to Numba path, simplify path selection |
| `osmose/engine/processes/growth.py` | Modify | Add growth dispatch, wire Gompertz |
| `osmose/engine/config.py` | Modify | Parse growth classname + Gompertz params |
| `osmose/schema/species.py` | Modify | Fix growth enum, add Gompertz fields |
| `tests/test_engine_predation_helpers.py` | Create | Tests for predation reconciliation |
| `tests/test_engine_growth_dispatch.py` | Create | Tests for growth dispatch + Gompertz |

### Phase 2 — Distribution Outputs
| File | Action | Responsibility |
|------|--------|----------------|
| `osmose/engine/output.py` | Modify | Add distribution binning + CSV writers |
| `osmose/engine/simulate.py` | Modify | Extend StepOutput, wire binning in collect |
| `osmose/engine/config.py` | Modify | Parse distribution config keys |
| `tests/test_engine_distribution_output.py` | Create | Tests for age/size binning + CSV format |

### Phase 3 — Per-Species RNG
| File | Action | Responsibility |
|------|--------|----------------|
| `osmose/engine/rng.py` | Create | `build_rng()` factory |
| `osmose/engine/__init__.py` | Modify | Use `build_rng()` in PythonEngine |
| `osmose/engine/simulate.py` | Modify | Accept and pass species-specific rngs |
| `osmose/engine/processes/predation.py` | Modify | Accept rng list |
| `osmose/engine/movement_maps.py` | Modify | Accept rng list |
| `tests/test_engine_rng.py` | Create | Tests for RNG factory + reproducibility |

### Phase 4 — Bioenergetics
| File | Action | Responsibility |
|------|--------|----------------|
| `osmose/engine/processes/temp_function.py` | Create | Johnson phi_T + Arrhenius |
| `osmose/engine/processes/oxygen_function.py` | Create | f_O2 dose-response |
| `osmose/engine/physical_data.py` | Create | NetCDF/constant loader for temp + O2 |
| `osmose/engine/processes/energy_budget.py` | Create | Core E_gross → E_net → dw/dg pipeline |
| `osmose/engine/processes/bioen_predation.py` | Create | Allometric I_max ingestion cap |
| `osmose/engine/processes/bioen_starvation.py` | Create | Energy-deficit starvation + gonad buffer |
| `osmose/engine/processes/bioen_reproduction.py` | Create | Gonad-weight egg production |
| `osmose/engine/state.py` | Modify | Add bioen fields to SchoolState |
| `osmose/engine/config.py` | Modify | Parse all bioen config keys |
| `osmose/engine/simulate.py` | Modify | Bioen branch in simulation loop |
| `osmose/engine/output.py` | Modify | Bioen-specific output writers |
| `osmose/schema/bioenergetics.py` | Modify | Expand with all missing fields |
| `osmose/schema/output.py` | Modify | Add bioen.rho + bioen.sizeInf flags |
| `tests/test_engine_temp_function.py` | Create | Tests for phi_T + Arrhenius |
| `tests/test_engine_oxygen_function.py` | Create | Tests for f_O2 |
| `tests/test_engine_physical_data.py` | Create | Tests for NetCDF/constant loader |
| `tests/test_engine_energy_budget.py` | Create | Tests for energy pipeline |
| `tests/test_engine_bioen_predation.py` | Create | Tests for allometric ingestion |
| `tests/test_engine_bioen_starvation.py` | Create | Tests for gonad-buffer starvation |
| `tests/test_engine_bioen_reproduction.py` | Create | Tests for gonad-weight reproduction |
| `tests/test_engine_bioen_integration.py` | Create | End-to-end bioen simulation test |

---

## Phase 1: Code Quality — Predation Reconciliation & Growth Dispatch

### Task 1: Add diet tracking to Numba predation path

**Files:**
- Modify: `osmose/engine/processes/predation.py:59-178` (Numba function)
- Modify: `osmose/engine/processes/predation.py:509-541` (path selection)
- Test: `tests/test_engine_predation_helpers.py`

- [ ] **Step 1: Write failing test — Numba path produces diet matrix**

```python
# tests/test_engine_predation_helpers.py
import numpy as np
import pytest
from osmose.engine.processes.predation import (
    enable_diet_tracking,
    disable_diet_tracking,
    get_diet_matrix,
    predation,
)
from osmose.engine.state import SchoolState
from tests.test_engine_predation import _make_predation_config
from osmose.engine.config import EngineConfig


class TestNumbaPathDietTracking:
    """Verify that the Numba path accumulates diet when tracking is enabled."""

    def test_numba_path_records_diet(self):
        """When diet tracking is enabled, Numba path should fill the diet matrix."""
        cfg = EngineConfig.from_dict(_make_predation_config())
        state = SchoolState.create(n_schools=2, species_id=np.array([0, 1], dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0, 500.0]),
            length=np.array([30.0, 10.0]),
            weight=np.array([0.5, 0.05]),
            biomass=np.array([50.0, 25.0]),
            age_dt=np.array([5, 5], dtype=np.int32),
            first_feeding_age_dt=np.array([1, 1], dtype=np.int32),
            cell_x=np.array([0, 0], dtype=np.int32),
            cell_y=np.array([0, 0], dtype=np.int32),
            feeding_stage=np.array([0, 0], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        enable_diet_tracking(n_schools=2, n_species=2)
        try:
            result = predation(state, cfg, rng, n_subdt=10, grid_ny=1, grid_nx=1)
            diet = get_diet_matrix()
            assert diet is not None
            # Predator (sp0) should have eaten some of prey (sp1)
            assert diet[0, 1] > 0, "Numba path should record diet"
        finally:
            disable_diet_tracking()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_predation_helpers.py::TestNumbaPathDietTracking::test_numba_path_records_diet -v`
Expected: FAIL — Numba path currently skips diet tracking

- [ ] **Step 3: Modify Numba function to accept and fill diet matrix**

In `osmose/engine/processes/predation.py`, add a `diet_matrix` parameter to `_predation_in_cell_numba()`:

```python
# Add parameter after use_stage_access:
    diet_matrix: NDArray[np.float64],   # shape (n_schools, n_species) or (1,1) dummy
    diet_enabled: bool,
```

Inside the inner prey loop (after `eaten_from_prey` is computed, around line 169-173), add:

```python
                if diet_enabled and eaten_from_prey > 0:
                    prey_sp = species_id[q_idx]
                    if p_idx < diet_matrix.shape[0] and prey_sp < diet_matrix.shape[1]:
                        diet_matrix[p_idx, prey_sp] += eaten_from_prey
```

- [ ] **Step 4: Update path selection to always use Numba when available**

In the `predation()` function (around line 517), change the condition:

```python
        # OLD: if _HAS_NUMBA and not _diet_tracking_enabled:
        # NEW: Numba availability is the only selector
        if _HAS_NUMBA:
            pred_order = rng.permutation(len(cell_indices)).astype(np.int32)
            diet_mat = _diet_matrix if _diet_tracking_enabled and _diet_matrix is not None else _DUMMY_DIET
            _predation_in_cell_numba(
                cell_indices, pred_order,
                work_state.abundance, work_state.length, work_state.weight,
                work_state.age_dt, work_state.first_feeding_age_dt,
                work_state.species_id, work_state.pred_success_rate,
                work_state.preyed_biomass,
                config.size_ratio_min, config.size_ratio_max,
                config.ingestion_rate, access_matrix, has_access,
                n_subdt, config.n_dt_per_year,
                work_state.feeding_stage, prey_access_idx, pred_access_idx,
                use_stage_access,
                diet_mat, _diet_tracking_enabled,
            )
        else:
            _predation_in_cell_python(...)
```

Add a module-level dummy for the diet matrix:

```python
_DUMMY_DIET = np.zeros((1, 1), dtype=np.float64)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_predation_helpers.py -v`
Expected: PASS

- [ ] **Step 6: Run full predation test suite for regression**

Run: `.venv/bin/python -m pytest tests/test_engine_predation.py tests/test_engine_diet.py -v`
Expected: All existing tests PASS

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/processes/predation.py tests/test_engine_predation_helpers.py
git commit -m "feat(engine): add diet tracking to Numba predation path"
```

---

### Task 2: Fix growth schema enum and add Gompertz config fields

**Files:**
- Modify: `osmose/schema/species.py:81-94`
- Modify: `osmose/engine/config.py`
- Test: `tests/test_engine_growth_dispatch.py`

- [ ] **Step 1: Write failing test — EngineConfig parses growth classname**

```python
# tests/test_engine_growth_dispatch.py
import numpy as np
import pytest
from osmose.engine.config import EngineConfig


def _make_growth_config(classname: str = "fr.ird.osmose.process.growth.VonBertalanffyGrowth") -> dict[str, str]:
    """Minimal config with growth classname set."""
    return {
        "simulation.nspecies": "2",
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nschool": "10",
        "species.name.sp0": "Anchovy",
        "species.name.sp1": "Sardine",
        "species.linf.sp0": "16.83",
        "species.linf.sp1": "22.0",
        "species.k.sp0": "0.589",
        "species.k.sp1": "0.4",
        "species.t0.sp0": "-0.253",
        "species.t0.sp1": "-0.3",
        "species.egg.size.sp0": "0.1",
        "species.egg.size.sp1": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.condition.factor.sp1": "0.006",
        "species.length2weight.allometric.power.sp0": "3.08",
        "species.length2weight.allometric.power.sp1": "3.0",
        "species.vonbertalanffy.threshold.age.sp0": "0.0",
        "species.vonbertalanffy.threshold.age.sp1": "0.0",
        "species.lifespan.sp0": "3",
        "species.lifespan.sp1": "5",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.ingestion.rate.max.sp1": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "predation.efficiency.critical.sp1": "0.57",
        "species.growth.delta.lmax.factor.sp0": "2.0",
        "species.growth.delta.lmax.factor.sp1": "2.0",
        "mortality.natural.rate.sp0": "0.0",
        "mortality.natural.rate.sp1": "0.0",
        "species.sexratio.sp0": "0.5",
        "species.sexratio.sp1": "0.5",
        "species.relativefecundity.sp0": "1000",
        "species.relativefecundity.sp1": "1000",
        "species.maturity.size.sp0": "12.0",
        "species.maturity.size.sp1": "15.0",
        "species.maturity.age.sp0": "1",
        "species.maturity.age.sp1": "2",
        "species.lmax.sp0": "20.0",
        "species.lmax.sp1": "25.0",
        "predation.predprey.sizeratio.min.sp0": "0;0",
        "predation.predprey.sizeratio.max.sp0": "0;0",
        "predation.predprey.sizeratio.min.sp1": "0;0",
        "predation.predprey.sizeratio.max.sp1": "0;0",
        "species.starvation.rate.max.sp0": "3.0",
        "species.starvation.rate.max.sp1": "3.0",
        "species.egg.weight.sp0": "0.00054",
        "species.egg.weight.sp1": "0.00054",
        "population.seeding.biomass.sp0": "1000",
        "population.seeding.biomass.sp1": "1000",
        "mortality.natural.larva.rate.sp0": "0.0",
        "mortality.natural.larva.rate.sp1": "0.0",
        "growth.java.classname.sp0": classname,
        "growth.java.classname.sp1": classname,
    }


class TestGrowthClassnameParsing:
    def test_vb_classname_parsed(self):
        cfg = EngineConfig.from_dict(_make_growth_config(
            "fr.ird.osmose.process.growth.VonBertalanffyGrowth"
        ))
        assert cfg.growth_class[0] == "VB"
        assert cfg.growth_class[1] == "VB"

    def test_gompertz_classname_parsed(self):
        cfg = EngineConfig.from_dict(_make_growth_config(
            "fr.ird.osmose.process.growth.GompertzGrowth"
        ))
        assert cfg.growth_class[0] == "GOMPERTZ"

    def test_unknown_classname_defaults_to_vb(self):
        cfg = EngineConfig.from_dict(_make_growth_config(
            "fr.ird.osmose.growth.VonBertalanffy"  # old format
        ))
        assert cfg.growth_class[0] == "VB"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_growth_dispatch.py::TestGrowthClassnameParsing -v`
Expected: FAIL — `growth_class` attribute doesn't exist on EngineConfig

- [ ] **Step 3: Fix schema enum values**

In `osmose/schema/species.py`, update the growth classname field (lines 82-94):

```python
    OsmoseField(
        key_pattern="growth.java.classname.sp{idx}",
        param_type=ParamType.ENUM,
        default="fr.ird.osmose.process.growth.VonBertalanffyGrowth",
        choices=[
            "fr.ird.osmose.process.growth.VonBertalanffyGrowth",
            "fr.ird.osmose.process.growth.GompertzGrowth",
        ],
        description="Java class implementing the growth model",
        category="growth",
        indexed=True,
        advanced=True,
    ),
```

- [ ] **Step 4: Add Gompertz schema fields**

In `osmose/schema/species.py`, add after the growth classname field:

```python
    OsmoseField(
        key_pattern="growth.exponential.ke.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Exponential growth rate (early phase)",
        category="growth",
        indexed=True,
        required=False,
    ),
    OsmoseField(
        key_pattern="growth.exponential.thr.age.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Age switching to exponential phase (years)",
        category="growth",
        indexed=True,
        required=False,
    ),
    OsmoseField(
        key_pattern="growth.exponential.lstart.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Starting length for exponential phase (cm)",
        category="growth",
        indexed=True,
        required=False,
    ),
    OsmoseField(
        key_pattern="growth.gompertz.thr.age.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Age switching to Gompertz phase (years)",
        category="growth",
        indexed=True,
        required=False,
    ),
    OsmoseField(
        key_pattern="growth.gompertz.kg.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Gompertz growth rate",
        category="growth",
        indexed=True,
        required=False,
    ),
    OsmoseField(
        key_pattern="growth.gompertz.tg.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Gompertz inflection age (years)",
        category="growth",
        indexed=True,
        required=False,
    ),
    OsmoseField(
        key_pattern="growth.gompertz.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Gompertz asymptotic length (cm)",
        category="growth",
        indexed=True,
        required=False,
    ),
```

- [ ] **Step 5: Add growth_class parsing to EngineConfig**

In `osmose/engine/config.py`, add to the `EngineConfig` dataclass:

```python
    growth_class: list[str]  # "VB" or "GOMPERTZ" per species
```

In `from_dict()`, before the `return cls(...)`:

```python
        # Growth class dispatch
        _GROWTH_MAP = {
            "fr.ird.osmose.process.growth.VonBertalanffyGrowth": "VB",
            "fr.ird.osmose.process.growth.GompertzGrowth": "GOMPERTZ",
            # Legacy names (backward compat)
            "fr.ird.osmose.growth.VonBertalanffy": "VB",
            "fr.ird.osmose.growth.Gompertz": "GOMPERTZ",
        }
        growth_class = []
        for i in range(n_sp):
            raw = cfg.get(f"growth.java.classname.sp{i}",
                          "fr.ird.osmose.process.growth.VonBertalanffyGrowth")
            growth_class.append(_GROWTH_MAP.get(raw, "VB"))
```

Add `growth_class=growth_class` to the `return cls(...)` call.

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_growth_dispatch.py::TestGrowthClassnameParsing -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add osmose/schema/species.py osmose/engine/config.py tests/test_engine_growth_dispatch.py
git commit -m "feat(engine): parse growth classname, add Gompertz schema fields"
```

---

### Task 3: Wire Gompertz growth dispatch

**Files:**
- Modify: `osmose/engine/processes/growth.py:48-106` (growth function)
- Modify: `osmose/engine/config.py` (parse Gompertz params)
- Test: `tests/test_engine_growth_dispatch.py`

- [ ] **Step 1: Write failing test — Gompertz growth produces different lengths than VB**

```python
# Append to tests/test_engine_growth_dispatch.py

class TestGompertzGrowth:
    def _gompertz_config(self) -> dict[str, str]:
        cfg = _make_growth_config("fr.ird.osmose.process.growth.GompertzGrowth")
        # Add Gompertz-specific params for sp0 and sp1
        for sp in range(2):
            cfg[f"growth.exponential.ke.sp{sp}"] = "0.5"
            cfg[f"growth.exponential.thr.age.sp{sp}"] = "0.1"
            cfg[f"growth.exponential.lstart.sp{sp}"] = "0.2"
            cfg[f"growth.gompertz.thr.age.sp{sp}"] = "0.5"
            cfg[f"growth.gompertz.kg.sp{sp}"] = "0.8"
            cfg[f"growth.gompertz.tg.sp{sp}"] = "1.0"
            cfg[f"growth.gompertz.linf.sp{sp}"] = "18.0"
        return cfg

    def test_gompertz_growth_applied(self):
        """Gompertz dispatch should use expected_length_gompertz, not VB."""
        from osmose.engine.processes.growth import growth
        cfg = EngineConfig.from_dict(self._gompertz_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0]),
            length=np.array([5.0]),
            weight=np.array([0.001]),
            biomass=np.array([0.1]),
            age_dt=np.array([12], dtype=np.int32),  # ~0.5 years at 24 dt/yr
            pred_success_rate=np.array([1.0]),
            is_out=np.array([False]),
        )
        rng = np.random.default_rng(42)
        result = growth(state, cfg, rng)
        # Length should have changed (not zero growth)
        assert result.length[0] > state.length[0]
        # Should NOT match VB result — different formula
        from osmose.engine.processes.growth import expected_length_vb
        l_vb = expected_length_vb(
            state.age_dt + 1, cfg.linf[0:1], cfg.k[0:1], cfg.t0[0:1],
            cfg.egg_size[0:1], cfg.vb_threshold_age[0:1], cfg.n_dt_per_year,
        )
        # Gompertz and VB should give different results
        assert not np.isclose(result.length[0], l_vb[0], rtol=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_growth_dispatch.py::TestGompertzGrowth -v`
Expected: FAIL — growth() doesn't dispatch on growth_class

- [ ] **Step 3: Parse Gompertz params in EngineConfig**

In `osmose/engine/config.py`, add to the dataclass:

```python
    # Gompertz params (None for VB species)
    gompertz_ke: NDArray[np.float64] | None = None
    gompertz_thr_age_exp_dt: NDArray[np.int32] | None = None
    gompertz_lstart: NDArray[np.float64] | None = None
    gompertz_thr_age_gom_dt: NDArray[np.int32] | None = None
    gompertz_kg: NDArray[np.float64] | None = None
    gompertz_tg: NDArray[np.float64] | None = None
    gompertz_linf: NDArray[np.float64] | None = None
```

In `from_dict()`, after growth_class parsing:

```python
        # Gompertz params (only if any species uses Gompertz)
        gompertz_ke = gompertz_lstart = gompertz_kg = gompertz_tg = gompertz_linf = None
        gompertz_thr_age_exp_dt = gompertz_thr_age_gom_dt = None
        if "GOMPERTZ" in growth_class:
            gompertz_ke = _species_float_optional(cfg, "growth.exponential.ke.sp{i}", n_sp, 0.0)
            gompertz_lstart = _species_float_optional(cfg, "growth.exponential.lstart.sp{i}", n_sp, 0.1)
            gompertz_kg = _species_float_optional(cfg, "growth.gompertz.kg.sp{i}", n_sp, 0.0)
            gompertz_tg = _species_float_optional(cfg, "growth.gompertz.tg.sp{i}", n_sp, 0.0)
            gompertz_linf = _species_float_optional(cfg, "growth.gompertz.linf.sp{i}", n_sp, 0.0)
            # Convert year thresholds to dt
            exp_yrs = _species_float_optional(cfg, "growth.exponential.thr.age.sp{i}", n_sp, 0.0)
            gom_yrs = _species_float_optional(cfg, "growth.gompertz.thr.age.sp{i}", n_sp, 0.0)
            gompertz_thr_age_exp_dt = (exp_yrs * n_dt).astype(np.int32)
            gompertz_thr_age_gom_dt = (gom_yrs * n_dt).astype(np.int32)
```

- [ ] **Step 4: Add growth dispatch to `growth()` function**

In `osmose/engine/processes/growth.py`, modify `growth()`:

```python
def growth(state: SchoolState, config: EngineConfig, rng: np.random.Generator) -> SchoolState:
    """Apply growth gated by predation success, dispatching by growth class."""
    if len(state) == 0:
        return state

    sp = state.species_id
    n_dt = config.n_dt_per_year

    # Compute expected length at current and next age per species growth class
    l_current = _expected_length(state.age_dt, sp, config, n_dt)
    l_next = _expected_length(state.age_dt + 1, sp, config, n_dt)
    delta_l = l_next - l_current

    # ... rest of gating logic unchanged ...
```

Add the dispatch helper:

```python
def _expected_length(
    age_dt: NDArray[np.int32],
    sp: NDArray[np.int32],
    config: EngineConfig,
    n_dt: int,
) -> NDArray[np.float64]:
    """Dispatch growth model per species."""
    result = np.zeros(len(age_dt), dtype=np.float64)

    # VB species
    vb_mask = np.array([config.growth_class[s] == "VB" for s in sp])
    if vb_mask.any():
        result[vb_mask] = expected_length_vb(
            age_dt[vb_mask], config.linf[sp[vb_mask]], config.k[sp[vb_mask]],
            config.t0[sp[vb_mask]], config.egg_size[sp[vb_mask]],
            config.vb_threshold_age[sp[vb_mask]], n_dt,
        )

    # Gompertz species
    gom_mask = np.array([config.growth_class[s] == "GOMPERTZ" for s in sp])
    if gom_mask.any() and config.gompertz_ke is not None:
        result[gom_mask] = expected_length_gompertz(
            age_dt[gom_mask],
            config.gompertz_linf[sp[gom_mask]],
            config.gompertz_kg[sp[gom_mask]],
            config.gompertz_tg[sp[gom_mask]],
            config.gompertz_ke[sp[gom_mask]],
            config.gompertz_thr_age_exp_dt[sp[gom_mask]],
            config.gompertz_thr_age_gom_dt[sp[gom_mask]],
            config.gompertz_lstart[sp[gom_mask]],
            n_dt,
        )

    return result
```

Update `expected_length_gompertz()` signature: replace `egg_size` param with `lstart`:

```python
def expected_length_gompertz(
    age_dt, linf, k_gom, t_gom, k_exp, a_exp_dt, a_gom_dt,
    lstart,  # was: egg_size
    n_dt_per_year,
) -> NDArray[np.float64]:
```

And update line 137: `l_exp = lstart * np.exp(k_exp * age_years)` and line 138: `l_exp_at_boundary = lstart * np.exp(k_exp * a_exp_years)`.

- [ ] **Step 5: Run tests to verify**

Run: `.venv/bin/python -m pytest tests/test_engine_growth_dispatch.py tests/test_engine_growth.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/processes/growth.py osmose/engine/config.py tests/test_engine_growth_dispatch.py
git commit -m "feat(engine): wire Gompertz growth dispatch with config-driven class selection"
```

---

## Phase 2: Size/Age Distribution Outputs

### Task 4: Extend StepOutput and add binning logic

**Files:**
- Modify: `osmose/engine/simulate.py:24-33` (StepOutput dataclass)
- Modify: `osmose/engine/simulate.py:159-210` (_collect_outputs)
- Modify: `osmose/engine/config.py`
- Test: `tests/test_engine_distribution_output.py`

- [ ] **Step 1: Write failing test — biomass binned by age**

```python
# tests/test_engine_distribution_output.py
import numpy as np
import pytest
from osmose.engine.state import SchoolState
from osmose.engine.simulate import _collect_outputs, StepOutput


class TestAgeBinning:
    def test_biomass_by_age_bins(self):
        """Schools at different ages should land in correct age bins."""
        from tests.test_engine_growth_dispatch import _make_growth_config
        from osmose.engine.config import EngineConfig

        cfg_dict = _make_growth_config()
        cfg_dict["output.biomass.byage.enabled"] = "true"
        cfg = EngineConfig.from_dict(cfg_dict)

        state = SchoolState.create(n_schools=3, species_id=np.array([0, 0, 1], dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0, 200.0, 50.0]),
            biomass=np.array([10.0, 20.0, 5.0]),
            age_dt=np.array([0, 24, 48], dtype=np.int32),  # 0yr, 1yr, 2yr at 24 dt/yr
        )

        out = _collect_outputs(state, cfg, step=0)
        assert out.biomass_by_age is not None
        # sp0 has schools at age 0 and 1 (years)
        assert out.biomass_by_age[0][0] == pytest.approx(10.0)  # age 0
        assert out.biomass_by_age[0][1] == pytest.approx(20.0)  # age 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_distribution_output.py::TestAgeBinning -v`
Expected: FAIL — `biomass_by_age` attribute doesn't exist on StepOutput

- [ ] **Step 3: Extend StepOutput dataclass**

In `osmose/engine/simulate.py`, update StepOutput:

```python
@dataclass
class StepOutput:
    """Aggregated output for a single simulation timestep."""
    step: int
    biomass: NDArray[np.float64]
    abundance: NDArray[np.float64]
    mortality_by_cause: NDArray[np.float64]
    yield_by_species: NDArray[np.float64] | None = None
    biomass_by_age: dict[int, NDArray[np.float64]] | None = None
    abundance_by_age: dict[int, NDArray[np.float64]] | None = None
    biomass_by_size: dict[int, NDArray[np.float64]] | None = None
    abundance_by_size: dict[int, NDArray[np.float64]] | None = None
```

- [ ] **Step 4: Parse distribution config keys in EngineConfig**

In `osmose/engine/config.py`, add to dataclass:

```python
    output_biomass_byage: bool = False
    output_biomass_bysize: bool = False
    output_abundance_byage: bool = False
    output_abundance_bysize: bool = False
    output_size_min: float = 0.0
    output_size_max: float = 205.0
    output_size_incr: float = 10.0
```

In `from_dict()`:

```python
        output_biomass_byage = cfg.get("output.biomass.byage.enabled", "false").lower() == "true"
        output_biomass_bysize = cfg.get("output.biomass.bysize.enabled", "false").lower() == "true"
        output_abundance_byage = cfg.get("output.abundance.byage.enabled", "false").lower() == "true"
        output_abundance_bysize = cfg.get("output.abundance.bysize.enabled", "false").lower() == "true"
        output_size_min = float(cfg.get("output.distrib.bysize.min", "0"))
        output_size_max = float(cfg.get("output.distrib.bysize.max", "205"))
        output_size_incr = float(cfg.get("output.distrib.bysize.incr", "10"))
```

- [ ] **Step 5: Add binning logic to `_collect_outputs()`**

In `osmose/engine/simulate.py`, at the end of `_collect_outputs()` before `return`:

```python
    # Distribution outputs
    biomass_by_age = abundance_by_age = biomass_by_size = abundance_by_size = None

    if config.output_biomass_byage or config.output_abundance_byage:
        biomass_by_age_dict: dict[int, NDArray[np.float64]] = {}
        abundance_by_age_dict: dict[int, NDArray[np.float64]] = {}
        for sp_idx in range(config.n_species):
            max_age_yr = int(config.lifespan_dt[sp_idx] / config.n_dt_per_year) + 1
            bio_bins = np.zeros(max_age_yr, dtype=np.float64)
            abd_bins = np.zeros(max_age_yr, dtype=np.float64)
            sp_mask = (state.species_id == sp_idx)
            if sp_mask.any():
                age_yr = (state.age_dt[sp_mask] // config.n_dt_per_year).astype(np.int32)
                age_yr = np.clip(age_yr, 0, max_age_yr - 1)
                np.add.at(bio_bins, age_yr, state.biomass[sp_mask])
                np.add.at(abd_bins, age_yr, state.abundance[sp_mask])
            biomass_by_age_dict[sp_idx] = bio_bins
            abundance_by_age_dict[sp_idx] = abd_bins
        if config.output_biomass_byage:
            biomass_by_age = biomass_by_age_dict
        if config.output_abundance_byage:
            abundance_by_age = abundance_by_age_dict

    if config.output_biomass_bysize or config.output_abundance_bysize:
        edges = np.arange(config.output_size_min, config.output_size_max + config.output_size_incr, config.output_size_incr)
        n_size_bins = len(edges) - 1
        biomass_by_size_dict: dict[int, NDArray[np.float64]] = {}
        abundance_by_size_dict: dict[int, NDArray[np.float64]] = {}
        for sp_idx in range(config.n_species):
            bio_bins = np.zeros(n_size_bins, dtype=np.float64)
            abd_bins = np.zeros(n_size_bins, dtype=np.float64)
            sp_mask = (state.species_id == sp_idx)
            if sp_mask.any():
                bin_idx = np.searchsorted(edges, state.length[sp_mask], side="right") - 1
                bin_idx = np.clip(bin_idx, 0, n_size_bins - 1)
                np.add.at(bio_bins, bin_idx, state.biomass[sp_mask])
                np.add.at(abd_bins, bin_idx, state.abundance[sp_mask])
            biomass_by_size_dict[sp_idx] = bio_bins
            abundance_by_size_dict[sp_idx] = abd_bins
        if config.output_biomass_bysize:
            biomass_by_size = biomass_by_size_dict
        if config.output_abundance_bysize:
            abundance_by_size = abundance_by_size_dict

    return StepOutput(
        step=step,
        biomass=biomass,
        abundance=abundance,
        mortality_by_cause=mortality_by_cause,
        yield_by_species=yield_by_species,
        biomass_by_age=biomass_by_age,
        abundance_by_age=abundance_by_age,
        biomass_by_size=biomass_by_size,
        abundance_by_size=abundance_by_size,
    )
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_distribution_output.py tests/test_engine_simulate.py tests/test_engine_output.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/simulate.py osmose/engine/config.py tests/test_engine_distribution_output.py
git commit -m "feat(engine): add size/age distribution binning to StepOutput"
```

---

### Task 5: Write distribution CSV output files

**Files:**
- Modify: `osmose/engine/output.py`
- Test: `tests/test_engine_distribution_output.py`

- [ ] **Step 1: Write failing test — distribution CSV file is created**

```python
# Append to tests/test_engine_distribution_output.py
from pathlib import Path

class TestDistributionCSVOutput:
    def test_biomass_by_age_csv_written(self, tmp_path):
        """write_outputs should create biomassByAge CSVs when enabled."""
        from osmose.engine.output import write_outputs
        from tests.test_engine_growth_dispatch import _make_growth_config
        from osmose.engine.config import EngineConfig

        cfg_dict = _make_growth_config()
        cfg_dict["output.biomass.byage.enabled"] = "true"
        cfg = EngineConfig.from_dict(cfg_dict)

        # Create a mock StepOutput with age distribution data
        out = StepOutput(
            step=0,
            biomass=np.array([30.0, 5.0]),
            abundance=np.array([300.0, 50.0]),
            mortality_by_cause=np.zeros((2, 8)),
            biomass_by_age={0: np.array([10.0, 20.0, 0.0]), 1: np.array([5.0, 0.0, 0.0, 0.0, 0.0])},
        )

        write_outputs([out], tmp_path, cfg)
        csv_path = tmp_path / "osmose_biomassByAge_Anchovy_Simu0.csv"
        assert csv_path.exists(), f"Expected {csv_path}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_distribution_output.py::TestDistributionCSVOutput -v`
Expected: FAIL — CSV file not created

- [ ] **Step 3: Add distribution CSV writers to output.py**

In `osmose/engine/output.py`, add after `_write_yield_csv` call in `write_outputs()`:

```python
    # Write distribution CSVs
    _write_distribution_csvs(output_dir, prefix, outputs, config)
```

Add the function:

```python
def _write_distribution_csvs(
    output_dir: Path,
    prefix: str,
    outputs: list[StepOutput],
    config: EngineConfig,
) -> None:
    """Write per-species age/size distribution CSVs matching Java format."""
    times = np.array([o.step / config.n_dt_per_year for o in outputs])

    for label, attr_name in [
        ("biomassByAge", "biomass_by_age"),
        ("abundanceByAge", "abundance_by_age"),
        ("biomassBySize", "biomass_by_size"),
        ("abundanceBySize", "abundance_by_size"),
    ]:
        first_out = next((o for o in outputs if getattr(o, attr_name) is not None), None)
        if first_out is None:
            continue

        dist_data = getattr(first_out, attr_name)
        for sp_idx, sp_name in enumerate(config.species_names):
            if sp_idx not in dist_data:
                continue
            n_bins = len(dist_data[sp_idx])
            data_matrix = np.zeros((len(outputs), n_bins), dtype=np.float64)
            for t_idx, o in enumerate(outputs):
                d = getattr(o, attr_name)
                if d is not None and sp_idx in d:
                    data_matrix[t_idx, :len(d[sp_idx])] = d[sp_idx]

            # Column headers: age years or size bin edges
            if "Age" in label:
                columns = [str(i) for i in range(n_bins)]
            else:
                edges = np.arange(
                    config.output_size_min,
                    config.output_size_min + n_bins * config.output_size_incr,
                    config.output_size_incr,
                )
                columns = [f"{e:.1f}" for e in edges]

            df = pd.DataFrame(data_matrix, columns=columns)
            df.insert(0, "Time", times)
            path = output_dir / f"{prefix}_{label}_{sp_name}_Simu0.csv"
            df.to_csv(path, index=False)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_distribution_output.py tests/test_engine_output.py -v`
Expected: All PASS

- [ ] **Step 5: Update `_average_step_outputs` to handle distribution fields**

In `osmose/engine/simulate.py`, update `_average_step_outputs()` to pass through distribution data (averaging dicts):

```python
    # Pass through distribution data from last accumulated step (no averaging needed for distributions)
    biomass_by_age = accumulated[-1].biomass_by_age
    abundance_by_age = accumulated[-1].abundance_by_age
    biomass_by_size = accumulated[-1].biomass_by_size
    abundance_by_size = accumulated[-1].abundance_by_size

    return StepOutput(
        step=record_step,
        biomass=biomass,
        abundance=abundance,
        mortality_by_cause=mortality,
        yield_by_species=yield_sum,
        biomass_by_age=biomass_by_age,
        abundance_by_age=abundance_by_age,
        biomass_by_size=biomass_by_size,
        abundance_by_size=abundance_by_size,
    )
```

- [ ] **Step 6: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/output.py osmose/engine/simulate.py tests/test_engine_distribution_output.py
git commit -m "feat(engine): write size/age distribution CSV outputs matching Java format"
```

---

## Phase 3: Per-Species Deterministic RNG

### Task 6: Create RNG factory and wire into simulation

**Files:**
- Create: `osmose/engine/rng.py`
- Modify: `osmose/engine/__init__.py:68`
- Modify: `osmose/engine/simulate.py`
- Test: `tests/test_engine_rng.py`

- [ ] **Step 1: Write failing test — `build_rng` factory**

```python
# tests/test_engine_rng.py
import numpy as np
import pytest
from osmose.engine.rng import build_rng


class TestBuildRng:
    def test_shared_rng_when_not_fixed(self):
        """All species get the same Generator instance when fixed=False."""
        rngs = build_rng(seed=42, n_species=3, fixed=False)
        assert len(rngs) == 3
        assert rngs[0] is rngs[1]
        assert rngs[1] is rngs[2]

    def test_independent_rng_when_fixed(self):
        """Each species gets a distinct Generator when fixed=True."""
        rngs = build_rng(seed=42, n_species=3, fixed=True)
        assert len(rngs) == 3
        assert rngs[0] is not rngs[1]
        # Different generators should produce different sequences
        a = rngs[0].random()
        b = rngs[1].random()
        assert a != b

    def test_fixed_rng_reproducible(self):
        """Same seed produces same per-species sequences."""
        rngs1 = build_rng(seed=42, n_species=3, fixed=True)
        rngs2 = build_rng(seed=42, n_species=3, fixed=True)
        for i in range(3):
            assert rngs1[i].random() == rngs2[i].random()

    def test_adding_species_doesnt_change_existing(self):
        """Adding a 4th species shouldn't change sequences for species 0-2."""
        rngs3 = build_rng(seed=42, n_species=3, fixed=True)
        rngs4 = build_rng(seed=42, n_species=4, fixed=True)
        for i in range(3):
            assert rngs3[i].random() == rngs4[i].random()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_rng.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement `build_rng`**

```python
# osmose/engine/rng.py
"""Per-species deterministic RNG factory."""

from __future__ import annotations

import numpy as np


def build_rng(seed: int, n_species: int, fixed: bool) -> list[np.random.Generator]:
    """Create RNG instances for each species.

    When fixed=False: all species share a single Generator (current behavior).
    When fixed=True: each species gets an independent Generator seeded from
    a SeedSequence child, ensuring that adding/removing species doesn't change
    existing species' random sequences.
    """
    if not fixed:
        shared = np.random.default_rng(seed)
        return [shared] * n_species
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(n_species)
    return [np.random.default_rng(child) for child in children]
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_rng.py -v`
Expected: All PASS

- [ ] **Step 5: Wire into PythonEngine and simulate**

In `osmose/engine/__init__.py`, replace `rng = np.random.default_rng(seed)` (line 68):

```python
        from osmose.engine.rng import build_rng
        rng = np.random.default_rng(seed)
        movement_rngs = build_rng(seed, engine_config.n_species, engine_config.movement_seed_fixed)
        mortality_rngs = build_rng(seed + 1, engine_config.n_species, engine_config.mortality_seed_fixed)

        outputs = simulate(engine_config, grid, rng, movement_rngs=movement_rngs, mortality_rngs=mortality_rngs)
```

Add to `EngineConfig` dataclass (already parsed as `movement_seed_fixed` and `mortality_seed_fixed` — rename if needed):

```python
    movement_seed_fixed: bool = False
    mortality_seed_fixed: bool = False
```

Update `simulate()` signature:

```python
def simulate(
    config: EngineConfig,
    grid: Grid,
    rng: np.random.Generator,
    movement_rngs: list[np.random.Generator] | None = None,
    mortality_rngs: list[np.random.Generator] | None = None,
) -> list[StepOutput]:
```

Default fallback when rngs are None (backward compat):

```python
    if movement_rngs is None:
        movement_rngs = [rng] * config.n_species
    if mortality_rngs is None:
        mortality_rngs = [rng] * config.n_species
```

Pass species-specific rngs to movement and mortality calls. For movement, pass `movement_rngs`. For mortality/predation, pass `mortality_rngs`.

- [ ] **Step 6: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: All PASS (backward compat: default None produces same behavior)

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/rng.py osmose/engine/__init__.py osmose/engine/simulate.py tests/test_engine_rng.py
git commit -m "feat(engine): add per-species deterministic RNG via SeedSequence"
```

---

## Phase 4: Bioenergetic (Ev-OSMOSE) Module

### Task 7: Temperature and oxygen functions

**Files:**
- Create: `osmose/engine/processes/temp_function.py`
- Create: `osmose/engine/processes/oxygen_function.py`
- Test: `tests/test_engine_temp_function.py`
- Test: `tests/test_engine_oxygen_function.py`

- [ ] **Step 1: Write failing test — phi_T at peak temperature equals 1.0**

```python
# tests/test_engine_temp_function.py
import numpy as np
import pytest


class TestPhiT:
    def test_phi_t_at_peak_is_one(self):
        """phi_T(T_P) should equal 1.0 by normalization."""
        from osmose.engine.processes.temp_function import phi_t
        # Typical values for a marine fish
        e_m = 0.6   # eV
        e_d = 3.0   # eV
        t_p = 15.0  # °C
        result = phi_t(np.array([t_p]), e_m, e_d, t_p)
        np.testing.assert_allclose(result, 1.0, rtol=1e-10)

    def test_phi_t_decreases_away_from_peak(self):
        """phi_T should be < 1 at temperatures away from T_P."""
        from osmose.engine.processes.temp_function import phi_t
        e_m, e_d, t_p = 0.6, 3.0, 15.0
        temps = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        result = phi_t(temps, e_m, e_d, t_p)
        assert result[2] == pytest.approx(1.0, rel=1e-10)
        assert result[0] < 1.0
        assert result[4] < 1.0

    def test_phi_t_vectorized(self):
        """phi_T should handle arrays."""
        from osmose.engine.processes.temp_function import phi_t
        temps = np.linspace(5, 25, 100)
        result = phi_t(temps, 0.6, 3.0, 15.0)
        assert result.shape == (100,)
        assert np.all(result > 0)
        assert np.all(result <= 1.0)


class TestArrhenius:
    def test_arrhenius_increases_with_temp(self):
        """Arrhenius function should increase with temperature."""
        from osmose.engine.processes.temp_function import arrhenius
        e_m = 0.6
        temps = np.array([5.0, 15.0, 25.0])
        result = arrhenius(temps, e_m)
        assert result[1] > result[0]
        assert result[2] > result[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_temp_function.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement temp_function.py**

```python
# osmose/engine/processes/temp_function.py
"""Johnson thermal performance curve and Arrhenius function.

Implements the bioenergetic temperature response from Java TempFunction class.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Boltzmann constant in eV/K
K_B = 8.62e-5


def phi_t(
    temp_c: NDArray[np.float64],
    e_m: float,
    e_d: float,
    t_p: float,
) -> NDArray[np.float64]:
    """Johnson thermal performance curve.

    Normalized so phi_T(T_P) = 1.0.

    Args:
        temp_c: Temperature in Celsius.
        e_m: Increasing activation energy (eV).
        e_d: Declining activation energy (eV).
        t_p: Peak temperature (Celsius).

    Returns:
        Temperature response factor [0, 1].
    """
    t_k = temp_c + 273.15
    t_p_k = t_p + 273.15

    def _raw(t: NDArray[np.float64] | float) -> NDArray[np.float64]:
        num = np.exp(-e_m / (K_B * t))
        ratio = e_m / (e_d - e_m)
        denom = 1.0 + ratio * np.exp(e_d / K_B * (1.0 / t_p_k - 1.0 / t))
        return num / denom

    raw_vals = _raw(t_k)
    raw_at_peak = _raw(np.asarray(t_p_k))
    return raw_vals / raw_at_peak


def arrhenius(
    temp_c: NDArray[np.float64],
    e_m: float,
) -> NDArray[np.float64]:
    """Arrhenius function for maintenance metabolic rate.

    Args:
        temp_c: Temperature in Celsius.
        e_m: Activation energy for maintenance (eV).

    Returns:
        Arrhenius scaling factor (unnormalized).
    """
    t_k = temp_c + 273.15
    return np.exp(-e_m / (K_B * t_k))
```

- [ ] **Step 4: Run temp tests**

Run: `.venv/bin/python -m pytest tests/test_engine_temp_function.py -v`
Expected: All PASS

- [ ] **Step 5: Write failing test — f_O2**

```python
# tests/test_engine_oxygen_function.py
import numpy as np
import pytest


class TestFO2:
    def test_fo2_at_zero_oxygen(self):
        """f_O2(0) should be 0."""
        from osmose.engine.processes.oxygen_function import f_o2
        result = f_o2(np.array([0.0]), c1=1.0, c2=5.0)
        assert result[0] == 0.0

    def test_fo2_saturates_at_high_oxygen(self):
        """f_O2 should approach C1 at very high O2."""
        from osmose.engine.processes.oxygen_function import f_o2
        result = f_o2(np.array([1e6]), c1=1.0, c2=5.0)
        np.testing.assert_allclose(result, 1.0, rtol=1e-4)

    def test_fo2_half_saturation(self):
        """f_O2(C2) = C1 * 0.5."""
        from osmose.engine.processes.oxygen_function import f_o2
        result = f_o2(np.array([5.0]), c1=1.0, c2=5.0)
        np.testing.assert_allclose(result, 0.5, rtol=1e-10)
```

- [ ] **Step 6: Implement oxygen_function.py**

```python
# osmose/engine/processes/oxygen_function.py
"""Dissolved oxygen dose-response function."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def f_o2(
    o2: NDArray[np.float64],
    c1: float,
    c2: float,
) -> NDArray[np.float64]:
    """Oxygen dose-response: f_O2 = C1 * O2 / (O2 + C2)."""
    return c1 * o2 / (o2 + c2)
```

- [ ] **Step 7: Run all temp+oxygen tests**

Run: `.venv/bin/python -m pytest tests/test_engine_temp_function.py tests/test_engine_oxygen_function.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add osmose/engine/processes/temp_function.py osmose/engine/processes/oxygen_function.py tests/test_engine_temp_function.py tests/test_engine_oxygen_function.py
git commit -m "feat(engine): add Johnson thermal performance curve and O2 dose-response"
```

---

### Task 8: Physical data loader (temperature/oxygen forcing)

**Files:**
- Create: `osmose/engine/physical_data.py`
- Test: `tests/test_engine_physical_data.py`

- [ ] **Step 1: Write failing test — constant temperature**

```python
# tests/test_engine_physical_data.py
import numpy as np
import pytest


class TestPhysicalDataConstant:
    def test_constant_value_all_cells(self):
        """Constant mode: same value everywhere, all timesteps."""
        from osmose.engine.physical_data import PhysicalData
        pd = PhysicalData.from_constant(value=15.0, factor=1.0, offset=0.0)
        result = pd.get_value(step=0, cell_y=0, cell_x=0)
        assert result == pytest.approx(15.0)

    def test_constant_with_factor_offset(self):
        """factor * (value + offset) conversion."""
        from osmose.engine.physical_data import PhysicalData
        pd = PhysicalData.from_constant(value=288.15, factor=1.0, offset=-273.15)
        result = pd.get_value(step=0, cell_y=0, cell_x=0)
        assert result == pytest.approx(15.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_physical_data.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement PhysicalData**

```python
# osmose/engine/physical_data.py
"""Generic NetCDF/constant physical data loader for temperature and oxygen forcing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class PhysicalData:
    """Physical forcing data (temperature or oxygen).

    Two modes:
    - Constant: single value applied everywhere.
    - NetCDF: 3D array (time, y, x) with periodic cycling.
    """

    def __init__(
        self,
        data: NDArray[np.float64] | None,
        constant: float | None,
        factor: float,
        offset: float,
        nsteps_year: int,
    ) -> None:
        self._data = data       # (n_time, ny, nx) or None
        self._constant = constant
        self._factor = factor
        self._offset = offset
        self._nsteps_year = nsteps_year

    @classmethod
    def from_constant(cls, value: float, factor: float = 1.0, offset: float = 0.0) -> PhysicalData:
        return cls(data=None, constant=factor * (value + offset), factor=factor, offset=offset, nsteps_year=1)

    @classmethod
    def from_netcdf(
        cls,
        path: Path,
        varname: str = "temp",
        nsteps_year: int = 12,
        factor: float = 1.0,
        offset: float = 0.0,
    ) -> PhysicalData:
        import xarray as xr
        ds = xr.open_dataset(path)
        raw = ds[varname].values  # (time, y, x) or similar
        if raw.ndim == 2:
            raw = raw[np.newaxis, :, :]  # add time dim
        data = factor * (raw.astype(np.float64) + offset)
        return cls(data=data, constant=None, factor=factor, offset=offset, nsteps_year=nsteps_year)

    def get_value(self, step: int, cell_y: int, cell_x: int) -> float:
        if self._constant is not None:
            return self._constant
        assert self._data is not None
        t_idx = step % self._data.shape[0]
        return float(self._data[t_idx, cell_y, cell_x])

    def get_grid(self, step: int) -> NDArray[np.float64]:
        """Return full (ny, nx) grid for a timestep."""
        if self._constant is not None:
            raise ValueError("Constant PhysicalData has no spatial grid")
        assert self._data is not None
        t_idx = step % self._data.shape[0]
        return self._data[t_idx]
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_physical_data.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/physical_data.py tests/test_engine_physical_data.py
git commit -m "feat(engine): add PhysicalData loader for temperature/oxygen forcing"
```

---

### Task 9: Energy budget process

**Files:**
- Create: `osmose/engine/processes/energy_budget.py`
- Modify: `osmose/engine/state.py` (add bioen fields)
- Test: `tests/test_engine_energy_budget.py`

- [ ] **Step 1: Add bioen fields to SchoolState**

In `osmose/engine/state.py`, add after `gonad_weight`:

```python
    # Bioenergetics (active only when simulation.bioen.enabled)
    e_net_avg: NDArray[np.float64]     # running average mass-specific E_net
    e_gross: NDArray[np.float64]       # current step gross energy
    e_maint: NDArray[np.float64]       # current step maintenance
    e_net: NDArray[np.float64]         # current step net energy
    rho: NDArray[np.float64]           # allocation fraction
```

Update `SchoolState.create()` to initialize these:

```python
            e_net_avg=np.zeros(n, dtype=np.float64),
            e_gross=np.zeros(n, dtype=np.float64),
            e_maint=np.zeros(n, dtype=np.float64),
            e_net=np.zeros(n, dtype=np.float64),
            rho=np.zeros(n, dtype=np.float64),
```

- [ ] **Step 2: Write failing test — energy budget computation**

```python
# tests/test_engine_energy_budget.py
import numpy as np
import pytest


class TestEnergyBudget:
    def test_positive_enet_increases_weight(self):
        """When E_net > 0, somatic weight should increase."""
        from osmose.engine.processes.energy_budget import compute_energy_budget
        # School that just ate well
        ingestion = np.array([0.5])      # tonnes eaten
        weight = np.array([0.001])       # individual weight in tonnes
        gonad_weight = np.array([0.0])
        age_dt = np.array([24], dtype=np.int32)  # 1 year old
        length = np.array([10.0])
        assimilation = 0.7
        c_m = 0.01
        beta = 0.75
        eta = 1.5
        r = 0.5
        m0 = 5.0
        m1 = 2.0
        phi_t_val = np.array([1.0])
        f_o2_val = np.array([1.0])
        n_dt_per_year = 24

        dw, dg, e_net = compute_energy_budget(
            ingestion=ingestion,
            weight=weight,
            gonad_weight=gonad_weight,
            age_dt=age_dt,
            length=length,
            assimilation=assimilation,
            c_m=c_m,
            beta=beta,
            eta=eta,
            r=r,
            m0=m0,
            m1=m1,
            phi_t=phi_t_val,
            f_o2=f_o2_val,
            n_dt_per_year=n_dt_per_year,
            e_net_avg=np.array([0.01]),
        )
        assert dw[0] > 0, "Positive E_net should produce weight gain"

    def test_zero_ingestion_means_maintenance_only(self):
        """No food: E_gross=0, E_net<0, no growth."""
        from osmose.engine.processes.energy_budget import compute_energy_budget
        dw, dg, e_net = compute_energy_budget(
            ingestion=np.array([0.0]),
            weight=np.array([0.001]),
            gonad_weight=np.array([0.0]),
            age_dt=np.array([24], dtype=np.int32),
            length=np.array([10.0]),
            assimilation=0.7, c_m=0.01, beta=0.75,
            eta=1.5, r=0.5, m0=5.0, m1=2.0,
            phi_t=np.array([1.0]), f_o2=np.array([1.0]),
            n_dt_per_year=24,
            e_net_avg=np.array([0.01]),
        )
        assert e_net[0] < 0
        assert dw[0] == 0.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_energy_budget.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 4: Implement energy_budget.py**

```python
# osmose/engine/processes/energy_budget.py
"""Core energy budget for the bioenergetic (Ev-OSMOSE) module.

Pipeline: E_gross -> E_maint -> E_net -> rho -> dw, dg
Matches Java EnergyBudget.run().
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.processes.temp_function import arrhenius


def compute_energy_budget(
    ingestion: NDArray[np.float64],
    weight: NDArray[np.float64],
    gonad_weight: NDArray[np.float64],
    age_dt: NDArray[np.int32],
    length: NDArray[np.float64],
    assimilation: float,
    c_m: float,
    beta: float,
    eta: float,
    r: float,
    m0: float,
    m1: float,
    phi_t: NDArray[np.float64],
    f_o2: NDArray[np.float64],
    n_dt_per_year: int,
    e_net_avg: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute energy budget for a set of schools.

    Returns:
        (dw, dg, e_net) — somatic weight change, gonadic weight change, net energy.
    """
    n = len(ingestion)

    # E_gross = ingestion * assimilation * phi_T * f_O2
    e_gross = ingestion * assimilation * phi_t * f_o2

    # E_maint = C_m * w^beta * arrhenius(T) / n_dt_per_year
    # Note: arrhenius already applied via phi_t in the thermal term.
    # Java: maintenance uses a SEPARATE Arrhenius with e_maint parameter.
    # For now, assume phi_t already encodes thermal response for maintenance.
    # Weight must be in grams for the allometric formula, then convert back.
    w_grams = weight * 1e6  # tonnes -> grams
    e_maint = c_m * np.power(w_grams, beta) / n_dt_per_year

    # E_net
    e_net = e_gross - e_maint

    # Rho: maturity allocation
    age_years = age_dt.astype(np.float64) / n_dt_per_year
    l_mature = m0 + m1 * age_years
    is_mature = length >= l_mature

    rho = np.zeros(n, dtype=np.float64)
    safe_enet = np.where(e_net_avg > 0, e_net_avg, 1.0)
    rho_val = r / (eta * safe_enet) * np.power(w_grams, 1.0 - beta)
    rho = np.where(is_mature & (e_net_avg > 0), np.clip(rho_val, 0.0, 1.0), 0.0)

    # dw = (1 - rho) * E_net if E_net > 0 else 0
    positive_enet = np.maximum(e_net, 0.0)
    dw = (1.0 - rho) * positive_enet
    dg = rho * positive_enet

    # Convert back to tonnes
    dw_tonnes = dw * 1e-6
    dg_tonnes = dg * 1e-6

    return dw_tonnes, dg_tonnes, e_net


def update_e_net_avg(
    e_net_avg: NDArray[np.float64],
    e_net: NDArray[np.float64],
    weight: NDArray[np.float64],
    age_dt: NDArray[np.int32],
    first_feeding_age_dt: NDArray[np.int32],
    n_dt_per_year: int,
) -> NDArray[np.float64]:
    """Update running average of mass-specific E_net (4 regimes)."""
    result = e_net_avg.copy()
    w_grams = weight * 1e6
    e_net_specific = np.where(w_grams > 0, e_net / w_grams, 0.0)

    # Steps since first feeding
    steps_feeding = age_dt - first_feeding_age_dt
    n = np.maximum(steps_feeding, 1).astype(np.float64)

    # Before first feeding: stays 0
    feeding = age_dt >= first_feeding_age_dt

    # Incremental average for feeding schools
    result = np.where(
        feeding,
        (result * (n - 1) + e_net_specific) / n,
        0.0,
    )

    return result
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_energy_budget.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/processes/energy_budget.py osmose/engine/state.py tests/test_engine_energy_budget.py
git commit -m "feat(engine): add energy budget process + bioen SchoolState fields"
```

---

### Task 10: Bioen predation (allometric ingestion cap)

**Files:**
- Create: `osmose/engine/processes/bioen_predation.py`
- Test: `tests/test_engine_bioen_predation.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_engine_bioen_predation.py
import numpy as np
import pytest


class TestBioenIngestionCap:
    def test_adult_ingestion_cap(self):
        """Adult ingestion capped at I_max * w^beta / n_dt_per_year."""
        from osmose.engine.processes.bioen_predation import bioen_ingestion_cap
        weight = np.array([0.001])  # tonnes
        i_max = 10.0
        beta = 0.75
        n_dt = 24
        n_subdt = 10
        is_larvae = np.array([False])
        cap = bioen_ingestion_cap(weight, i_max, beta, n_dt, n_subdt, is_larvae, theta=1.0, c_rate=0.0)
        w_g = 0.001 * 1e6
        expected = i_max * w_g ** beta / (n_dt * n_subdt)
        np.testing.assert_allclose(cap, expected, rtol=1e-10)

    def test_larvae_get_additive_correction(self):
        """Larvae: (I_max + (theta-1)*c_rate) * w^beta / (n_dt * subdt)."""
        from osmose.engine.processes.bioen_predation import bioen_ingestion_cap
        weight = np.array([0.0001])
        i_max = 10.0
        theta = 3.0
        c_rate = 2.0
        cap = bioen_ingestion_cap(
            weight, i_max, 0.75, 24, 10,
            is_larvae=np.array([True]), theta=theta, c_rate=c_rate,
        )
        w_g = 0.0001 * 1e6
        expected = (i_max + (theta - 1) * c_rate) * w_g ** 0.75 / (24 * 10)
        np.testing.assert_allclose(cap, expected, rtol=1e-10)
```

- [ ] **Step 2: Implement bioen_predation.py**

```python
# osmose/engine/processes/bioen_predation.py
"""Bioenergetic predation — allometric ingestion cap."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bioen_ingestion_cap(
    weight: NDArray[np.float64],
    i_max: float,
    beta: float,
    n_dt_per_year: int,
    n_subdt: int,
    is_larvae: NDArray[np.bool_],
    theta: float = 1.0,
    c_rate: float = 0.0,
) -> NDArray[np.float64]:
    """Compute max ingestion per sub-timestep for bioen mode.

    Adults: I_max * w^beta / (n_dt * subdt)
    Larvae: (I_max + (theta-1)*c_rate) * w^beta / (n_dt * subdt)
    """
    w_grams = weight * 1e6
    i_eff = np.where(
        is_larvae,
        i_max + (theta - 1.0) * c_rate,
        i_max,
    )
    return i_eff * np.power(w_grams, beta) / (n_dt_per_year * n_subdt)
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_predation.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add osmose/engine/processes/bioen_predation.py tests/test_engine_bioen_predation.py
git commit -m "feat(engine): add bioenergetic allometric ingestion cap"
```

---

### Task 11: Bioen starvation (gonad-buffer energy deficit)

**Files:**
- Create: `osmose/engine/processes/bioen_starvation.py`
- Test: `tests/test_engine_bioen_starvation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_engine_bioen_starvation.py
import numpy as np
import pytest


class TestBioenStarvation:
    def test_positive_enet_no_starvation(self):
        """No deaths when E_net >= 0."""
        from osmose.engine.processes.bioen_starvation import bioen_starvation
        n_dead, new_gonad = bioen_starvation(
            e_net=np.array([0.1]),
            gonad_weight=np.array([0.001]),
            weight=np.array([0.01]),
            eta=1.5,
            n_subdt=10,
        )
        assert n_dead[0] == 0.0
        assert new_gonad[0] == pytest.approx(0.001)

    def test_gonad_absorbs_deficit(self):
        """When gonad is sufficient, no deaths but gonad decreases."""
        from osmose.engine.processes.bioen_starvation import bioen_starvation
        n_dead, new_gonad = bioen_starvation(
            e_net=np.array([-0.001]),
            gonad_weight=np.array([0.01]),   # plenty of gonad
            weight=np.array([0.01]),
            eta=1.5,
            n_subdt=1,
        )
        assert n_dead[0] == 0.0
        assert new_gonad[0] < 0.01

    def test_gonad_insufficient_causes_death(self):
        """When gonad can't cover deficit, fish die."""
        from osmose.engine.processes.bioen_starvation import bioen_starvation
        n_dead, new_gonad = bioen_starvation(
            e_net=np.array([-10.0]),       # large deficit
            gonad_weight=np.array([0.0]),  # no gonad buffer
            weight=np.array([0.01]),
            eta=1.5,
            n_subdt=1,
        )
        assert n_dead[0] > 0
        assert new_gonad[0] == 0.0
```

- [ ] **Step 2: Implement bioen_starvation.py**

```python
# osmose/engine/processes/bioen_starvation.py
"""Energy-deficit starvation with gonad buffer.

Matches Java BioenStarvationMortality: per-sub-timestep processing,
gonad flushed before computing death toll.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bioen_starvation(
    e_net: NDArray[np.float64],
    gonad_weight: NDArray[np.float64],
    weight: NDArray[np.float64],
    eta: float,
    n_subdt: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute starvation deaths and gonad depletion.

    Args:
        e_net: Net energy for the full timestep.
        gonad_weight: Current gonad weight (tonnes).
        weight: Individual somatic weight (tonnes).
        eta: Energy density ratio soma/gonad.
        n_subdt: Number of sub-timesteps.

    Returns:
        (n_dead, new_gonad_weight)
    """
    n_dead = np.zeros_like(e_net)
    new_gonad = gonad_weight.copy()

    for _ in range(n_subdt):
        e_sub = e_net / n_subdt
        deficit = np.maximum(-e_sub, 0.0)

        # Where gonad is sufficient
        sufficient = new_gonad >= eta * deficit
        new_gonad = np.where(sufficient, new_gonad - eta * deficit, new_gonad)

        # Where gonad is insufficient: flush gonad, compute deaths
        insufficient = (~sufficient) & (deficit > 0)
        gonad_buffered = np.where(insufficient, new_gonad / eta, 0.0)
        remaining_deficit = np.where(insufficient, deficit - gonad_buffered, 0.0)
        safe_weight = np.where(weight > 0, weight, 1.0)
        n_dead += np.where(insufficient, remaining_deficit / safe_weight, 0.0)
        new_gonad = np.where(insufficient, 0.0, new_gonad)

    return n_dead, new_gonad
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_starvation.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add osmose/engine/processes/bioen_starvation.py tests/test_engine_bioen_starvation.py
git commit -m "feat(engine): add bioen starvation with gonad-buffer deficit"
```

---

### Task 12: Bioen reproduction (gonad-weight eggs)

**Files:**
- Create: `osmose/engine/processes/bioen_reproduction.py`
- Test: `tests/test_engine_bioen_reproduction.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_engine_bioen_reproduction.py
import numpy as np
import pytest


class TestBioenReproduction:
    def test_immature_no_eggs(self):
        """Immature fish produce no eggs."""
        from osmose.engine.processes.bioen_reproduction import bioen_egg_production
        eggs = bioen_egg_production(
            gonad_weight=np.array([0.01]),
            length=np.array([5.0]),
            age_dt=np.array([1], dtype=np.int32),
            m0=10.0, m1=2.0,
            egg_weight=0.001,
            n_dt_per_year=24,
        )
        assert eggs[0] == 0.0

    def test_mature_with_gonad_produces_eggs(self):
        """Mature fish with gonad weight produce eggs."""
        from osmose.engine.processes.bioen_reproduction import bioen_egg_production
        eggs = bioen_egg_production(
            gonad_weight=np.array([0.5]),  # 0.5 tonnes gonad
            length=np.array([20.0]),
            age_dt=np.array([48], dtype=np.int32),  # 2 years
            m0=5.0, m1=2.0,  # L_mature = 5 + 2*2 = 9 cm (< 20, so mature)
            egg_weight=0.001,
            n_dt_per_year=24,
        )
        assert eggs[0] > 0
```

- [ ] **Step 2: Implement bioen_reproduction.py**

```python
# osmose/engine/processes/bioen_reproduction.py
"""Gonad-weight-based reproduction for bioenergetic mode."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bioen_egg_production(
    gonad_weight: NDArray[np.float64],
    length: NDArray[np.float64],
    age_dt: NDArray[np.int32],
    m0: float,
    m1: float,
    egg_weight: float,
    n_dt_per_year: int,
) -> NDArray[np.float64]:
    """Compute number of eggs from gonad weight.

    Maturity by LMRN: L_mature = m0 + m1 * age_years.
    Eggs = gonad_weight / egg_weight for mature fish.
    """
    age_years = age_dt.astype(np.float64) / n_dt_per_year
    l_mature = m0 + m1 * age_years
    is_mature = length >= l_mature
    safe_egg_weight = max(egg_weight, 1e-20)
    eggs = np.where(is_mature & (gonad_weight > 0), gonad_weight / safe_egg_weight, 0.0)
    return eggs
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_reproduction.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add osmose/engine/processes/bioen_reproduction.py tests/test_engine_bioen_reproduction.py
git commit -m "feat(engine): add gonad-weight egg production for bioen mode"
```

---

### Task 13: Bioen config parsing and schema expansion

**Files:**
- Modify: `osmose/schema/bioenergetics.py`
- Modify: `osmose/schema/output.py`
- Modify: `osmose/engine/config.py`
- Test: `tests/test_engine_bioen_integration.py`

- [ ] **Step 1: Expand bioenergetics schema**

In `osmose/schema/bioenergetics.py`, add all missing fields (see spec lines 333-355 for the full list):

Add fields for: `simulation.bioen.enabled`, `simulation.bioen.phit.enabled`, `simulation.bioen.fo2.enabled`, `species.zlayer.sp{idx}`, `species.bioen.maturity.eta.sp{idx}`, `species.bioen.maturity.r.sp{idx}`, `species.bioen.maturity.m0.sp{idx}`, `species.bioen.maturity.m1.sp{idx}`, `species.bioen.mobilized.e.mobi.sp{idx}`, `species.bioen.mobilized.e.D.sp{idx}`, `species.bioen.mobilized.Tp.sp{idx}`, `species.bioen.maint.e.maint.sp{idx}`, `species.oxygen.c1.sp{idx}`, `species.oxygen.c2.sp{idx}`, `predation.ingestion.rate.max.bioen.sp{idx}`, `predation.coef.ingestion.rate.max.larvae.bioen.sp{idx}`, `predation.c.bioen.sp{idx}`, `species.bioen.forage.k_for.sp{idx}`, plus oxygen forcing fields mirroring temperature.

- [ ] **Step 2: Add bioen output flags to schema**

In `osmose/schema/output.py`, add to `_OUTPUT_ENABLE_FLAGS`:

```python
    "output.bioen.rho.enabled",
    "output.bioen.sizeInf.enabled",
```

- [ ] **Step 3: Parse bioen config in EngineConfig**

Add to `EngineConfig` dataclass:

```python
    bioen_enabled: bool = False
    bioen_phit_enabled: bool = True
    bioen_fo2_enabled: bool = True
    # Per-species bioen params (all None when bioen disabled)
    bioen_assimilation: NDArray[np.float64] | None = None
    bioen_c_m: NDArray[np.float64] | None = None
    bioen_beta: NDArray[np.float64] | None = None
    bioen_eta: NDArray[np.float64] | None = None
    bioen_r: NDArray[np.float64] | None = None
    bioen_m0: NDArray[np.float64] | None = None
    bioen_m1: NDArray[np.float64] | None = None
    bioen_e_mobi: NDArray[np.float64] | None = None
    bioen_e_d: NDArray[np.float64] | None = None
    bioen_tp: NDArray[np.float64] | None = None
    bioen_e_maint: NDArray[np.float64] | None = None
    bioen_o2_c1: NDArray[np.float64] | None = None
    bioen_o2_c2: NDArray[np.float64] | None = None
    bioen_i_max: NDArray[np.float64] | None = None
    bioen_larvae_theta: NDArray[np.float64] | None = None
    bioen_c_rate: NDArray[np.float64] | None = None
    bioen_k_for: NDArray[np.float64] | None = None
    bioen_zlayer: NDArray[np.int32] | None = None
```

In `from_dict()`, add conditional parsing:

```python
        bioen_enabled = cfg.get("simulation.bioen.enabled", "false").lower() == "true"
        bioen_params = {}
        if bioen_enabled:
            bioen_params = {
                "bioen_phit_enabled": cfg.get("simulation.bioen.phit.enabled", "true").lower() == "true",
                "bioen_fo2_enabled": cfg.get("simulation.bioen.fo2.enabled", "true").lower() == "true",
                "bioen_assimilation": _species_float(cfg, "species.bioen.assimilation.sp{i}", n_sp),
                "bioen_c_m": _species_float(cfg, "species.bioen.maint.energy.c_m.sp{i}", n_sp),
                "bioen_beta": _species_float(cfg, "species.beta.sp{i}", n_sp),
                "bioen_eta": _species_float(cfg, "species.bioen.maturity.eta.sp{i}", n_sp),
                "bioen_r": _species_float(cfg, "species.bioen.maturity.r.sp{i}", n_sp),
                "bioen_m0": _species_float(cfg, "species.bioen.maturity.m0.sp{i}", n_sp),
                "bioen_m1": _species_float(cfg, "species.bioen.maturity.m1.sp{i}", n_sp),
                "bioen_e_mobi": _species_float(cfg, "species.bioen.mobilized.e.mobi.sp{i}", n_sp),
                "bioen_e_d": _species_float(cfg, "species.bioen.mobilized.e.D.sp{i}", n_sp),
                "bioen_tp": _species_float(cfg, "species.bioen.mobilized.Tp.sp{i}", n_sp),
                "bioen_e_maint": _species_float(cfg, "species.bioen.maint.e.maint.sp{i}", n_sp),
                "bioen_o2_c1": _species_float_optional(cfg, "species.oxygen.c1.sp{i}", n_sp, 1.0),
                "bioen_o2_c2": _species_float_optional(cfg, "species.oxygen.c2.sp{i}", n_sp, 5.0),
                "bioen_i_max": _species_float(cfg, "predation.ingestion.rate.max.bioen.sp{i}", n_sp),
                "bioen_larvae_theta": _species_float_optional(cfg, "predation.coef.ingestion.rate.max.larvae.bioen.sp{i}", n_sp, 1.0),
                "bioen_c_rate": _species_float_optional(cfg, "predation.c.bioen.sp{i}", n_sp, 0.0),
                "bioen_k_for": _species_float_optional(cfg, "species.bioen.forage.k_for.sp{i}", n_sp, 0.0),
                "bioen_zlayer": _species_int_optional(cfg, "species.zlayer.sp{i}", n_sp, 0),
            }
```

- [ ] **Step 4: Write failing test — bioen config parsed**

```python
# tests/test_engine_bioen_integration.py
import numpy as np
import pytest
from osmose.engine.config import EngineConfig


def _make_bioen_config() -> dict[str, str]:
    """Minimal 1-species config with bioenergetics enabled."""
    return {
        "simulation.nspecies": "1",
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nschool": "10",
        "simulation.bioen.enabled": "true",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "20.0",
        "species.k.sp0": "0.5",
        "species.t0.sp0": "-0.2",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.vonbertalanffy.threshold.age.sp0": "0.0",
        "species.lifespan.sp0": "5",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "species.growth.delta.lmax.factor.sp0": "2.0",
        "mortality.natural.rate.sp0": "0.0",
        "species.sexratio.sp0": "0.5",
        "species.relativefecundity.sp0": "1000",
        "species.maturity.size.sp0": "12.0",
        "species.maturity.age.sp0": "1",
        "species.lmax.sp0": "25.0",
        "predation.predprey.sizeratio.min.sp0": "0",
        "predation.predprey.sizeratio.max.sp0": "0",
        "species.starvation.rate.max.sp0": "3.0",
        "species.egg.weight.sp0": "0.00054",
        "population.seeding.biomass.sp0": "1000",
        "mortality.natural.larva.rate.sp0": "0.0",
        # Bioen params
        "species.bioen.assimilation.sp0": "0.7",
        "species.bioen.maint.energy.c_m.sp0": "0.01",
        "species.beta.sp0": "0.75",
        "species.bioen.maturity.eta.sp0": "1.5",
        "species.bioen.maturity.r.sp0": "0.5",
        "species.bioen.maturity.m0.sp0": "5.0",
        "species.bioen.maturity.m1.sp0": "2.0",
        "species.bioen.mobilized.e.mobi.sp0": "0.6",
        "species.bioen.mobilized.e.D.sp0": "3.0",
        "species.bioen.mobilized.Tp.sp0": "15.0",
        "species.bioen.maint.e.maint.sp0": "0.4",
        "predation.ingestion.rate.max.bioen.sp0": "10.0",
    }


class TestBioenConfigParsing:
    def test_bioen_enabled_parsed(self):
        cfg = EngineConfig.from_dict(_make_bioen_config())
        assert cfg.bioen_enabled is True
        assert cfg.bioen_assimilation is not None
        np.testing.assert_allclose(cfg.bioen_assimilation, [0.7])
        np.testing.assert_allclose(cfg.bioen_c_m, [0.01])
        np.testing.assert_allclose(cfg.bioen_beta, [0.75])
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_integration.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add osmose/schema/bioenergetics.py osmose/schema/output.py osmose/engine/config.py tests/test_engine_bioen_integration.py
git commit -m "feat(engine): parse all bioenergetic config keys + expand schema"
```

---

### Task 14: Wire bioen into simulation loop

**Files:**
- Modify: `osmose/engine/simulate.py`
- Test: `tests/test_engine_bioen_integration.py`

- [ ] **Step 1: Write failing test — bioen simulation runs without error**

```python
# Append to tests/test_engine_bioen_integration.py

class TestBioenSimulation:
    def test_bioen_simulation_runs(self):
        """A bioen config should run the simulation without error."""
        from osmose.engine.simulate import simulate
        from osmose.engine.grid import Grid

        cfg_dict = _make_bioen_config()
        cfg_dict["temperature.value"] = "15.0"
        cfg = EngineConfig.from_dict(cfg_dict)
        grid = Grid.from_dimensions(ny=2, nx=2)
        rng = np.random.default_rng(42)

        outputs = simulate(cfg, grid, rng)
        assert len(outputs) > 0
        # Biomass should exist for the single species
        assert outputs[-1].biomass[0] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_integration.py::TestBioenSimulation -v`
Expected: FAIL — simulate doesn't handle bioen mode

- [ ] **Step 3: Add bioen branch to simulation loop**

In `osmose/engine/simulate.py`, add imports at top:

```python
from osmose.engine.physical_data import PhysicalData
```

In `simulate()`, after building resources/background/flux but before the main loop, add bioen setup:

```python
    # Bioenergetics setup
    temp_data = o2_data = None
    if config.bioen_enabled:
        temp_val = config.raw_config.get("temperature.value", "")
        if temp_val:
            temp_data = PhysicalData.from_constant(float(temp_val))
        # O2 similarly (skip for now if not configured)
```

In the main loop, replace the `_growth` call with conditional:

```python
        if config.bioen_enabled:
            state = _bioen_step(state, config, temp_data, o2_data)
        else:
            state = _growth(state, config, rng)
```

Implement `_bioen_step`:

```python
def _bioen_step(
    state: SchoolState,
    config: EngineConfig,
    temp_data: PhysicalData | None,
    o2_data: PhysicalData | None,
) -> SchoolState:
    """Apply bioenergetic processes: energy budget, starvation, weight update."""
    from osmose.engine.processes.energy_budget import compute_energy_budget, update_e_net_avg
    from osmose.engine.processes.bioen_starvation import bioen_starvation
    from osmose.engine.processes.temp_function import phi_t, arrhenius
    from osmose.engine.processes.oxygen_function import f_o2

    if len(state) == 0:
        return state

    sp = state.species_id
    n = len(state)

    # Temperature response per school
    phi_vals = np.ones(n, dtype=np.float64)
    if config.bioen_phit_enabled and temp_data is not None and config.bioen_e_mobi is not None:
        temp = temp_data._constant if temp_data._constant is not None else 15.0
        temp_arr = np.full(n, temp)
        for s in range(config.n_species):
            mask = sp == s
            if mask.any():
                phi_vals[mask] = phi_t(temp_arr[mask], config.bioen_e_mobi[s], config.bioen_e_d[s], config.bioen_tp[s])

    # O2 response
    o2_vals = np.ones(n, dtype=np.float64)

    # Energy budget per species
    dw = np.zeros(n, dtype=np.float64)
    dg = np.zeros(n, dtype=np.float64)
    e_net = np.zeros(n, dtype=np.float64)

    for s in range(config.n_species):
        mask = sp == s
        if not mask.any():
            continue
        dw_s, dg_s, e_net_s = compute_energy_budget(
            ingestion=state.preyed_biomass[mask],
            weight=state.weight[mask],
            gonad_weight=state.gonad_weight[mask],
            age_dt=state.age_dt[mask],
            length=state.length[mask],
            assimilation=config.bioen_assimilation[s],
            c_m=config.bioen_c_m[s],
            beta=config.bioen_beta[s],
            eta=config.bioen_eta[s],
            r=config.bioen_r[s],
            m0=config.bioen_m0[s],
            m1=config.bioen_m1[s],
            phi_t=phi_vals[mask],
            f_o2=o2_vals[mask],
            n_dt_per_year=config.n_dt_per_year,
            e_net_avg=state.e_net_avg[mask],
        )
        dw[mask] = dw_s
        dg[mask] = dg_s
        e_net[mask] = e_net_s

    # Starvation
    n_dead_starv = np.zeros(n, dtype=np.float64)
    new_gonad = state.gonad_weight.copy()
    for s in range(config.n_species):
        mask = sp == s
        if not mask.any():
            continue
        nd, ng = bioen_starvation(e_net[mask], state.gonad_weight[mask], state.weight[mask], config.bioen_eta[s], config.mortality_subdt)
        n_dead_starv[mask] = nd
        new_gonad[mask] = ng

    # Update state
    new_weight = np.maximum(state.weight + dw, 1e-20)
    new_gonad = np.maximum(new_gonad + dg, 0.0)
    new_length = (new_weight * 1e6 / config.condition_factor[sp]) ** (1.0 / config.allometric_power[sp])
    new_biomass = state.abundance * new_weight
    new_e_net_avg = update_e_net_avg(state.e_net_avg, e_net, state.weight, state.age_dt, state.first_feeding_age_dt, config.n_dt_per_year)

    # Apply starvation deaths
    new_abundance = np.maximum(state.abundance - n_dead_starv, 0.0)

    return state.replace(
        weight=new_weight,
        length=new_length,
        biomass=new_biomass,
        gonad_weight=new_gonad,
        abundance=new_abundance,
        e_net_avg=new_e_net_avg,
        e_gross=np.zeros(n, dtype=np.float64),  # TODO: fill from compute
        e_maint=np.zeros(n, dtype=np.float64),
        e_net=e_net,
        rho=np.zeros(n, dtype=np.float64),
    )
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_integration.py tests/test_engine_simulate.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: All 1553+ tests PASS

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/simulate.py tests/test_engine_bioen_integration.py
git commit -m "feat(engine): wire bioenergetic processes into simulation loop"
```

---

### Task 15: Bioen output writers

**Files:**
- Modify: `osmose/engine/output.py`
- Test: `tests/test_engine_bioen_integration.py`

- [ ] **Step 1: Add bioen CSV output stubs**

In `osmose/engine/output.py`, add after distribution CSV writing:

```python
    # Bioen outputs (when enabled)
    if config.bioen_enabled:
        _write_bioen_csvs(output_dir, prefix, outputs, config)
```

Implement `_write_bioen_csvs` to write per-species CSVs for ingestion, maintenance, E_net, rho, and sizeInf — reading from StepOutput's bioen fields. Follow the same pattern as `_write_mortality_csvs`.

- [ ] **Step 2: Write test**

```python
# Append to tests/test_engine_bioen_integration.py

class TestBioenOutput:
    def test_bioen_csv_written(self, tmp_path):
        """Bioen output CSVs should be created."""
        from osmose.engine.output import write_outputs
        cfg_dict = _make_bioen_config()
        cfg_dict["temperature.value"] = "15.0"
        cfg_dict["output.bioen.enet.enabled"] = "true"
        cfg = EngineConfig.from_dict(cfg_dict)

        from osmose.engine.simulate import simulate, StepOutput
        from osmose.engine.grid import Grid
        grid = Grid.from_dimensions(ny=2, nx=2)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        write_outputs(outputs, tmp_path, cfg)
        # Check Bioen directory exists
        bioen_dir = tmp_path / "Bioen"
        assert bioen_dir.exists() or not cfg_dict.get("output.bioen.enet.enabled")
```

- [ ] **Step 3: Run tests and commit**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_integration.py -v`

```bash
git add osmose/engine/output.py tests/test_engine_bioen_integration.py
git commit -m "feat(engine): add bioenergetic output CSV writers"
```

---

### Task 16: Final regression + tag

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`
Expected: All tests PASS (1553 existing + ~105 new)

- [ ] **Step 2: Run lint**

Run: `.venv/bin/ruff check osmose/ tests/`
Expected: Clean

- [ ] **Step 3: Commit any cleanup**

- [ ] **Step 4: Tag release**

```bash
git tag -a engine-phase8 -m "Phase 8: Java parity sprint — growth dispatch, distributions, RNG, bioenergetics"
```

---

## Errata — Review Corrections

The following corrections MUST be applied during implementation. They address issues found during plan review.

### E1: Task 1 — Extract shared predation helpers (spec requirement)

The spec requires extracting `compute_size_overlap()`, `compute_appetite()`, and `apply_biomass_transfer()` as shared functions. Task 1 as written only adds diet tracking. During implementation, also:

1. Extract the three helpers from both paths
2. Create `@njit` variants for the Numba path
3. Rewire both paths to call the shared helpers
4. Add unit tests for each extracted helper
5. Add a test that runs the same scenario with Numba on/off and asserts identical results

### E2: Task 3 — Add bioen bypass test

Add this test to `tests/test_engine_growth_dispatch.py`:

```python
class TestBioenBypass:
    def test_bioen_enabled_skips_growth_dispatch(self):
        """When bioen is enabled, growth() should be a no-op or skipped."""
        # Verify that simulate() calls _bioen_step instead of _growth
        # when config.bioen_enabled is True
```

### E3: Task 4 — Fix cross-phase test dependency

The `_make_growth_config` helper imported from `test_engine_growth_dispatch.py` creates a dependency between Phase 2 and Phase 1. During implementation, either:
- Extract shared config helpers into `tests/conftest.py`, OR
- Duplicate the helper in the Phase 2 test file

### E4: Task 4 — Add size binning tests and edge cases

Add to `tests/test_engine_distribution_output.py`:

```python
class TestSizeBinning:
    def test_biomass_by_size_bins(self):
        """Schools at different lengths should land in correct size bins."""
        # Create schools with lengths 5.0, 15.0, 25.0
        # Assert they land in bins [0-10), [10-20), [20-30) respectively

    def test_size_at_bin_boundary(self):
        """Length exactly at bin edge goes to the right bin."""

class TestDistributionEdgeCases:
    def test_empty_state_produces_zero_bins(self):
        """Empty SchoolState produces all-zero distribution bins."""

    def test_single_school_single_bin(self):
        """Single school produces one non-zero bin entry."""

    def test_all_same_age_single_bin(self):
        """All schools at same age -> all biomass in one age bin."""
```

Also assert species 1 in the age binning test: `out.biomass_by_age[1][2] == pytest.approx(5.0)`

### E5: Task 5 — Verify CSV column headers

Add to the distribution CSV test:
```python
    # Read CSV and verify headers
    df = pd.read_csv(csv_path)
    assert "Time" in df.columns
    assert "0" in df.columns  # age 0 bin
```

### E6: Task 6 — Parse RNG flags from config

The current `config.py` already parses `movement_seed_fixed` and `mortality_seed_fixed` at lines 933-936 but they are stored as local variables, not dataclass fields. During implementation, ensure these are wired into the `EngineConfig` dataclass properly:

```python
    movement_seed_fixed: bool = False
    mortality_seed_fixed: bool = False
```

And in `from_dict()` return, add: `movement_seed_fixed=movement_seed_fixed, mortality_seed_fixed=mortality_seed_fixed`

### E7: Task 9 — Fix `compute_energy_budget` to include Arrhenius maintenance

**Critical.** The plan's implementation omits the `arrhenius(T)` term from `E_maint`. Fix:

1. Add parameters: `temp_c: NDArray[np.float64]` and `e_maint_energy: float`
2. Compute: `e_maint = c_m * np.power(w_grams, beta) * arrhenius(temp_c, e_maint_energy) / n_dt_per_year`
3. Update all callers and tests to pass temperature and maintenance Arrhenius energy

Also: the rho formula (`r / (eta * E_net_avg) * w^(1-beta)`) must be verified against Java `EnergyBudget.computeRho()` during implementation. Add a comment citing the Java source.

### E8: Task 13 — Add `raw_config` or parse temperature into EngineConfig

Task 14's `_bioen_step` accesses `config.raw_config` which doesn't exist. Fix by adding to EngineConfig:

```python
    raw_config: dict[str, str] | None = None  # reference to original config dict
```

And in `from_dict()`, add `raw_config=cfg` to the return. This is already needed for `ResourceState` and `BackgroundState` initialization.

Alternatively, parse `temperature.value` and `temperature.filename` directly into EngineConfig fields during bioen config parsing in Task 13.

### E9: Task 14 — Wire bioen reproduction and ingestion cap

**Critical.** `_bioen_step` must:

1. **Apply ingestion cap** before energy budget:
```python
    from osmose.engine.processes.bioen_predation import bioen_ingestion_cap
    is_larvae = state.age_dt < state.first_feeding_age_dt
    for s in range(config.n_species):
        mask = sp == s
        if mask.any():
            cap = bioen_ingestion_cap(
                state.weight[mask], config.bioen_i_max[s], config.bioen_beta[s],
                config.n_dt_per_year, config.mortality_subdt,
                is_larvae[mask], config.bioen_larvae_theta[s], config.bioen_c_rate[s],
            )
            capped_ingestion = np.minimum(state.preyed_biomass[mask], cap)
            # Use capped_ingestion in energy budget call
```

2. **Call bioen reproduction** after starvation:
```python
    from osmose.engine.processes.bioen_reproduction import bioen_egg_production
    # After starvation, compute eggs from gonad weight for mature fish
    # Reset gonad_weight after spawning
```

### E10: Task 14 — Fix PhysicalData private attribute access

Replace `temp_data._constant` with `temp_data.get_value(step, 0, 0)` or add a public `is_constant` property to `PhysicalData`.
