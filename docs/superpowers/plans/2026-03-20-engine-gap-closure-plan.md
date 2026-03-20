# Engine Gap Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close 4 remaining wiring gaps connecting already-implemented bioen + RNG modules into the simulation loop.

**Architecture:** All formula modules exist and are unit tested. These tasks wire them into `simulate.py` and `output.py`. Gaps 1-3 are independent. Gap 4 depends on 1+2.

**Tech Stack:** Python 3.12, NumPy, pytest

**Spec:** `docs/superpowers/specs/2026-03-20-engine-gap-closure-design.md`

---

## Prerequisite: Fix StepOutput duplicate fields

Before starting tasks, fix the merge artifact in `osmose/engine/simulate.py` lines 34-43. The fields `biomass_by_age`, `abundance_by_age`, `biomass_by_size`, `abundance_by_size` are defined twice. Remove the duplicate block (lines 39-43):

```python
    # REMOVE these duplicate lines:
    # Per-species age/size distribution dicts (sp_idx -> 1-D array), or None if disabled
    biomass_by_age: dict[int, NDArray[np.float64]] | None = None
    abundance_by_age: dict[int, NDArray[np.float64]] | None = None
    biomass_by_size: dict[int, NDArray[np.float64]] | None = None
    abundance_by_size: dict[int, NDArray[np.float64]] | None = None
```

Run: `.venv/bin/python -m pytest tests/ -x -q` to verify no regression.
Commit: `fix: remove duplicate StepOutput fields from merge artifact`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `osmose/engine/simulate.py` | Modify | Add `_bioen_reproduction()`, update `_bioen_step` for O2, thread RNG, add bioen StepOutput fields |
| `osmose/engine/processes/energy_budget.py` | Modify | Extend return to 6 values |
| `osmose/engine/processes/movement.py` | Modify | Accept `species_rngs` |
| `osmose/engine/processes/mortality.py` | Modify | Accept `species_rngs`, pass to predation |
| `osmose/engine/processes/predation.py` | Modify | Accept `species_rngs`, use for cell shuffle |
| `osmose/engine/config.py` | Modify | Add bioen output flag fields |
| `osmose/engine/output.py` | Modify | Write 4 additional bioen CSV types |
| `tests/test_engine_bioen_reproduction_wiring.py` | Create | Tests for bioen reproduction in sim loop |
| `tests/test_engine_o2_wiring.py` | Create | Tests for O2 forcing |
| `tests/test_engine_rng_consumers.py` | Create | Tests for per-species RNG threading |
| `tests/test_engine_bioen_outputs_complete.py` | Create | Tests for all 5 bioen outputs |

---

### Task 1: Wire bioen reproduction into simulation loop

**Files:**
- Modify: `osmose/engine/simulate.py:696-701`
- Test: `tests/test_engine_bioen_reproduction_wiring.py`

- [ ] **Step 1: Write failing test — bioen reproduction resets gonad weight**

```python
# tests/test_engine_bioen_reproduction_wiring.py
import numpy as np
import pytest
from osmose.engine.state import SchoolState


class TestBioenReproductionWiring:
    def test_gonad_weight_resets_after_spawning(self):
        """Mature fish with gonad weight should have gonad reset to 0 after reproduction."""
        from osmose.engine.simulate import _bioen_reproduction
        # This import will fail until the function exists
        assert callable(_bioen_reproduction)

    def test_egg_school_created(self):
        """Bioen reproduction should create egg schools from gonad weight."""
        from osmose.engine.simulate import _bioen_reproduction
        from tests.test_engine_bioen_integration import _make_bioen_config
        from osmose.engine.config import EngineConfig

        cfg = EngineConfig.from_dict({**_make_bioen_config(), "temperature.value": "15.0"})
        # Create a mature fish with gonad weight
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0]),
            length=np.array([20.0]),  # above LMRN threshold
            weight=np.array([0.01]),
            biomass=np.array([1.0]),
            age_dt=np.array([48], dtype=np.int32),
            gonad_weight=np.array([0.5]),  # significant gonad
            cell_x=np.array([0], dtype=np.int32),
            cell_y=np.array([0], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        result = _bioen_reproduction(state, cfg, step=0, rng=rng, grid_ny=2, grid_nx=2)
        # Should have original school + new egg school
        assert len(result) > 1
        # Original school's gonad should be reset
        assert result.gonad_weight[0] == 0.0

    def test_immature_no_eggs(self):
        """Immature fish should not produce eggs."""
        from osmose.engine.simulate import _bioen_reproduction
        from tests.test_engine_bioen_integration import _make_bioen_config
        from osmose.engine.config import EngineConfig

        cfg = EngineConfig.from_dict({**_make_bioen_config(), "temperature.value": "15.0"})
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0]),
            length=np.array([2.0]),  # below LMRN threshold
            weight=np.array([0.001]),
            biomass=np.array([0.1]),
            age_dt=np.array([1], dtype=np.int32),
            gonad_weight=np.array([0.1]),
            cell_x=np.array([0], dtype=np.int32),
            cell_y=np.array([0], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        result = _bioen_reproduction(state, cfg, step=0, rng=rng, grid_ny=2, grid_nx=2)
        # No egg school created, only age increment
        assert len(result) == 1
        assert result.gonad_weight[0] == 0.1  # unchanged

    def test_standard_reproduction_unchanged(self):
        """Non-bioen config should still use standard reproduction."""
        from osmose.engine.simulate import simulate
        from osmose.engine.grid import Grid
        from osmose.engine.config import EngineConfig
        from tests.test_engine_bioen_integration import _make_bioen_config

        cfg_dict = _make_bioen_config()
        cfg_dict["simulation.bioen.enabled"] = "false"
        cfg_dict["temperature.value"] = "15.0"
        cfg = EngineConfig.from_dict(cfg_dict)
        grid = Grid.from_dimensions(ny=2, nx=2)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert len(outputs) > 0  # no error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_reproduction_wiring.py -v`
Expected: FAIL — `_bioen_reproduction` doesn't exist

- [ ] **Step 3: Implement `_bioen_reproduction()` in simulate.py**

Add the function from the spec pseudocode (spec lines 22-93) before `_aging_mortality()`. Then update the main loop (line ~701):

```python
        if config.bioen_enabled:
            state = _bioen_reproduction(state, config, step, rng, grid_ny=grid.ny, grid_nx=grid.nx)
        else:
            state = _reproduction(state, config, step, rng, grid_ny=grid.ny, grid_nx=grid.nx)
```

Key implementation details:
- Import `bioen_egg_production` from `osmose.engine.processes.bioen_reproduction`
- Handle NaN `egg_weight_override` with allometric fallback
- Create egg schools via `SchoolState.create()` + `.replace()`
- Reset gonad weight for spawning schools
- Age increment for existing schools only (not new eggs)

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_reproduction_wiring.py tests/test_engine_bioen_integration.py -v`
Expected: All PASS

- [ ] **Step 5: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/simulate.py tests/test_engine_bioen_reproduction_wiring.py
git commit -m "feat(engine): wire bioen reproduction — gonad-weight egg schools"
```

---

### Task 2: Wire O2 forcing into `_bioen_step`

**Files:**
- Modify: `osmose/engine/simulate.py:117-121` (`_bioen_step` signature)
- Modify: `osmose/engine/simulate.py:638-643` (simulate setup)
- Modify: `osmose/engine/simulate.py:696-697` (call site)
- Test: `tests/test_engine_o2_wiring.py`

- [ ] **Step 1: Write failing test — O2 forcing affects E_gross**

```python
# tests/test_engine_o2_wiring.py
import numpy as np
import pytest
from osmose.engine.simulate import simulate
from osmose.engine.grid import Grid
from osmose.engine.config import EngineConfig
from tests.test_engine_bioen_integration import _make_bioen_config


class TestO2Wiring:
    def test_o2_reduces_egross(self):
        """With low O2, E_gross should be lower than without O2 forcing."""
        cfg_base = {**_make_bioen_config(), "temperature.value": "15.0"}

        # Run without O2 (default: f_o2 = 1.0)
        cfg1 = EngineConfig.from_dict(cfg_base)
        grid = Grid.from_dimensions(ny=2, nx=2)
        out1 = simulate(cfg1, grid, np.random.default_rng(42))

        # Run with low O2 (f_o2 < 1.0)
        cfg_o2 = {**cfg_base, "oxygen.value": "2.0"}  # low O2
        cfg2 = EngineConfig.from_dict(cfg_o2)
        out2 = simulate(cfg2, grid, np.random.default_rng(42))

        # With low O2, biomass should be lower (less energy from food)
        # This is a directional test, not exact values
        assert out2[-1].biomass[0] <= out1[-1].biomass[0] + 1e-6

    def test_no_o2_defaults_to_one(self):
        """Without oxygen config, f_o2 should be 1.0 (no effect)."""
        cfg = EngineConfig.from_dict({**_make_bioen_config(), "temperature.value": "15.0"})
        grid = Grid.from_dimensions(ny=2, nx=2)
        outputs = simulate(cfg, grid, np.random.default_rng(42))
        assert len(outputs) > 0  # runs without error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_o2_wiring.py -v`
Expected: FAIL or no effect (O2 hardcoded to 1.0)

- [ ] **Step 3: Implement O2 loading and wiring**

In `simulate()`, after temperature loading (~line 643):
```python
        o2_data = None
        if config.bioen_enabled:
            o2_val = config.raw_config.get("oxygen.value", "")
            if o2_val:
                o2_data = PhysicalData.from_constant(float(o2_val))
```

Update `_bioen_step` signature to accept `o2_data=None`.

In `_bioen_step`, replace the hardcoded `f_o2_arr = np.ones(...)` with:
```python
    if config.bioen_fo2_enabled and o2_data is not None:
        from osmose.engine.processes.oxygen_function import f_o2
        f_o2_arr = np.ones(len(state), dtype=np.float64)
        if o2_data.is_constant:
            o2_scalar = o2_data.get_value(step, 0, 0)
            for sp in range(config.n_species):
                mask = state.species_id == sp
                if mask.any():
                    o2_vals = np.full(mask.sum(), o2_scalar)
                    f_o2_arr[mask] = f_o2(o2_vals, float(config.bioen_o2_c1[sp]), float(config.bioen_o2_c2[sp]))
    else:
        f_o2_arr = np.ones(len(state), dtype=np.float64)
```

Update call site: `state = _bioen_step(state, config, temp_data, step, o2_data=o2_data)`

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_o2_wiring.py tests/test_engine_bioen_integration.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/simulate.py tests/test_engine_o2_wiring.py
git commit -m "feat(engine): wire O2 forcing into bioenergetic step"
```

---

### Task 3: Wire per-species RNG into consumers

**Files:**
- Modify: `osmose/engine/simulate.py:73-80` (`_movement` wrapper)
- Modify: `osmose/engine/simulate.py:86-97` (`_mortality` wrapper)
- Modify: `osmose/engine/processes/movement.py:177`
- Modify: `osmose/engine/processes/mortality.py:528`
- Modify: `osmose/engine/processes/predation.py:489`
- Test: `tests/test_engine_rng_consumers.py`

- [ ] **Step 1: Write failing test — per-species RNG independence**

```python
# tests/test_engine_rng_consumers.py
import numpy as np
import pytest
from osmose.engine.rng import build_rng


class TestRngConsumerWiring:
    def test_fixed_false_backward_compat(self):
        """With fixed=False, behavior should be identical to global rng."""
        rngs = build_rng(seed=42, n_species=3, fixed=False)
        # All should be the same object
        assert rngs[0] is rngs[1]

    def test_movement_accepts_species_rngs(self):
        """movement() should accept optional species_rngs parameter."""
        from osmose.engine.processes.movement import movement
        import inspect
        sig = inspect.signature(movement)
        assert "species_rngs" in sig.parameters

    def test_mortality_accepts_species_rngs(self):
        """mortality() should accept optional species_rngs parameter."""
        from osmose.engine.processes.mortality import mortality
        import inspect
        sig = inspect.signature(mortality)
        assert "species_rngs" in sig.parameters

    def test_predation_accepts_species_rngs(self):
        """predation() should accept optional species_rngs parameter."""
        from osmose.engine.processes.predation import predation
        import inspect
        sig = inspect.signature(predation)
        assert "species_rngs" in sig.parameters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_rng_consumers.py -v`
Expected: FAIL — `species_rngs` not in signatures

- [ ] **Step 3: Add `species_rngs` parameter to movement**

In `osmose/engine/processes/movement.py`, add to `movement()` signature:
```python
def movement(
    state, grid, config, step, rng,
    map_sets=None, random_patches=None,
    species_rngs: list[np.random.Generator] | None = None,  # NEW
) -> SchoolState:
```

Inside the function, where random walk is applied per species, use `species_rngs[sp]` when available:
```python
    sp_rng = species_rngs[sp] if species_rngs is not None else rng
```

- [ ] **Step 4: Add `species_rngs` parameter to mortality and predation**

In `osmose/engine/processes/mortality.py`, add to `mortality()` signature:
```python
def mortality(
    state, resources, config, rng, grid, step=0,
    species_rngs: list[np.random.Generator] | None = None,  # NEW
) -> SchoolState:
```

Pass through to `predation()`:
```python
    state = predation(state, config, rng, ..., species_rngs=species_rngs)
```

In `osmose/engine/processes/predation.py`, add to `predation()` signature:
```python
def predation(
    state, config, rng, n_subdt, grid_ny, grid_nx,
    resources=None,
    species_rngs: list[np.random.Generator] | None = None,  # NEW
) -> SchoolState:
```

For cell-level shuffle, use first predator's rng when available:
```python
    if species_rngs is not None:
        first_pred_sp = state.species_id[cell_indices[0]]
        cell_rng = species_rngs[first_pred_sp]
    else:
        cell_rng = rng
    pred_order = cell_rng.permutation(len(cell_indices)).astype(np.int32)
```

- [ ] **Step 5: Update simulate.py wrappers and call sites**

Update `_movement()` and `_mortality()` wrappers to accept and pass `species_rngs`. Update call sites in main loop:
```python
    state = _movement(state, grid, config, step, rng,
                      map_sets=map_sets, random_patches=random_patches,
                      species_rngs=movement_rngs)
    state = _mortality(state, resources, config, rng, grid, step=step,
                       species_rngs=mortality_rngs)
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_rng_consumers.py tests/test_engine_predation.py tests/test_engine_movement.py tests/test_engine_simulate.py -v`
Expected: All PASS

- [ ] **Step 7: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: All PASS (backward compat: None defaults to global rng)

- [ ] **Step 8: Commit**

```bash
git add osmose/engine/simulate.py osmose/engine/processes/movement.py osmose/engine/processes/mortality.py osmose/engine/processes/predation.py tests/test_engine_rng_consumers.py
git commit -m "feat(engine): wire per-species RNG into movement and predation consumers"
```

---

### Task 4: Complete bioen outputs (4 missing)

**Files:**
- Modify: `osmose/engine/processes/energy_budget.py:115`
- Modify: `osmose/engine/simulate.py:237-339` (`_bioen_step` + `_collect_outputs` + StepOutput)
- Modify: `osmose/engine/config.py`
- Modify: `osmose/engine/output.py`
- Test: `tests/test_engine_bioen_outputs_complete.py`

- [ ] **Step 1: Extend `compute_energy_budget()` return**

In `osmose/engine/processes/energy_budget.py`, change line 115:

```python
    # OLD: return dw_tonnes, dg_tonnes, e_net
    # NEW:
    return dw_tonnes, dg_tonnes, e_net, e_gross, e_maint, rho
```

Where `e_gross` is at line ~86, `e_maint` at ~90, `rho` at ~104. These are already local variables — just add them to the return.

- [ ] **Step 2: Update `_bioen_step()` to capture and store all 6 values**

In `osmose/engine/simulate.py`, update the destructured call (~line 241):

```python
    # OLD: dw_sp, dg_sp, en_sp = compute_energy_budget(...)
    # NEW:
    dw_sp, dg_sp, en_sp, eg_sp, em_sp, rho_sp = compute_energy_budget(...)
```

Add arrays for the new values at the top of `_bioen_step`:
```python
    e_gross_arr = np.zeros(len(state), dtype=np.float64)
    e_maint_arr = np.zeros(len(state), dtype=np.float64)
    rho_arr = np.zeros(len(state), dtype=np.float64)
```

Store per-species results:
```python
    e_gross_arr[mask] = eg_sp
    e_maint_arr[mask] = em_sp
    rho_arr[mask] = rho_sp
```

Update the `state.replace()` at the end (~line 330) to include:
```python
    e_gross=e_gross_arr,
    e_maint=e_maint_arr,
    rho=rho_arr,
```

- [ ] **Step 3: Add StepOutput fields and aggregate in `_collect_outputs()`**

Add to StepOutput dataclass:
```python
    bioen_ingestion_by_species: NDArray[np.float64] | None = None
    bioen_maint_by_species: NDArray[np.float64] | None = None
    bioen_rho_by_species: NDArray[np.float64] | None = None
    bioen_size_inf_by_species: NDArray[np.float64] | None = None
```

In `_collect_outputs()`, after the existing `bioen_e_net_by_species` aggregation:
```python
    bioen_ingestion = bioen_maint = bioen_rho = bioen_sizeinf = None
    if config.bioen_enabled and len(state) > 0:
        bioen_ingestion = np.zeros(config.n_species, dtype=np.float64)
        bioen_maint = np.zeros(config.n_species, dtype=np.float64)
        bioen_rho = np.zeros(config.n_species, dtype=np.float64)
        bioen_sizeinf = np.zeros(config.n_species, dtype=np.float64)
        counts = np.zeros(config.n_species, dtype=np.float64)
        focal = state.species_id < config.n_species
        np.add.at(bioen_ingestion, state.species_id[focal], state.e_gross[focal])
        np.add.at(bioen_maint, state.species_id[focal], state.e_maint[focal])
        np.add.at(bioen_rho, state.species_id[focal], state.rho[focal])
        np.add.at(counts, state.species_id[focal], 1)
        safe = np.where(counts > 0, counts, 1)
        bioen_ingestion /= safe
        bioen_maint /= safe
        bioen_rho /= safe
        # sizeInf: max length per species
        for sp in range(config.n_species):
            sp_mask = (state.species_id == sp) & focal
            if sp_mask.any():
                bioen_sizeinf[sp] = state.length[sp_mask].max()
```

Add to the returned StepOutput.

- [ ] **Step 4: Add bioen output config flags**

In `osmose/engine/config.py`, add to EngineConfig dataclass (after existing bioen fields):
```python
    output_bioen_ingest: bool = False
    output_bioen_maint: bool = False
    output_bioen_rho: bool = False
    output_bioen_sizeinf: bool = False
```

In `from_dict()`, parse:
```python
    output_bioen_ingest = cfg.get("output.bioen.ingest.enabled", "false").lower() == "true"
    output_bioen_maint = cfg.get("output.bioen.maint.enabled", "false").lower() == "true"
    output_bioen_rho = cfg.get("output.bioen.rho.enabled", "false").lower() == "true"
    output_bioen_sizeinf = cfg.get("output.bioen.sizeInf.enabled", "false").lower() == "true"
```

- [ ] **Step 5: Extend `_write_bioen_csvs()` to write all 5 outputs**

In `osmose/engine/output.py`, update `_write_bioen_csvs()`:
```python
def _write_bioen_csvs(output_dir, prefix, outputs, config):
    bioen_dir = output_dir / "Bioen"
    bioen_dir.mkdir(exist_ok=True)
    times = np.array([o.step / config.n_dt_per_year for o in outputs])

    bioen_outputs = [
        ("bioen_e_net_by_species", "meanEnet", True),  # always write when bioen enabled
        ("bioen_ingestion_by_species", "ingestion", config.output_bioen_ingest),
        ("bioen_maint_by_species", "maintenance", config.output_bioen_maint),
        ("bioen_rho_by_species", "rho", config.output_bioen_rho),
        ("bioen_size_inf_by_species", "sizeInf", config.output_bioen_sizeinf),
    ]

    for attr, label, enabled in bioen_outputs:
        if not enabled:
            continue
        data_list = [getattr(o, attr) for o in outputs]
        if not any(d is not None for d in data_list):
            continue
        data = np.array([d if d is not None else np.zeros(config.n_species) for d in data_list])
        for sp_idx, sp_name in enumerate(config.species_names):
            df = pd.DataFrame({"Time": times, label: data[:, sp_idx]})
            df.to_csv(bioen_dir / f"{prefix}_{label}_{sp_name}_Simu0.csv", index=False)
```

- [ ] **Step 6: Write tests**

```python
# tests/test_engine_bioen_outputs_complete.py
import numpy as np
import pytest
from pathlib import Path
from osmose.engine.config import EngineConfig
from osmose.engine.simulate import simulate
from osmose.engine.output import write_outputs
from osmose.engine.grid import Grid
from tests.test_engine_bioen_integration import _make_bioen_config


class TestBioenOutputsComplete:
    def _run_bioen(self, tmp_path, extra_config=None):
        cfg_dict = {**_make_bioen_config(), "temperature.value": "15.0"}
        if extra_config:
            cfg_dict.update(extra_config)
        cfg = EngineConfig.from_dict(cfg_dict)
        grid = Grid.from_dimensions(ny=2, nx=2)
        outputs = simulate(cfg, grid, np.random.default_rng(42))
        write_outputs(outputs, tmp_path, cfg)
        return outputs

    def test_all_bioen_csvs_created(self, tmp_path):
        """All 5 bioen output CSVs should be created when flags are enabled."""
        self._run_bioen(tmp_path, {
            "output.bioen.ingest.enabled": "true",
            "output.bioen.maint.enabled": "true",
            "output.bioen.rho.enabled": "true",
            "output.bioen.sizeInf.enabled": "true",
        })
        bioen_dir = tmp_path / "Bioen"
        assert bioen_dir.exists()
        assert (bioen_dir / "osmose_meanEnet_TestFish_Simu0.csv").exists()
        assert (bioen_dir / "osmose_ingestion_TestFish_Simu0.csv").exists()
        assert (bioen_dir / "osmose_maintenance_TestFish_Simu0.csv").exists()
        assert (bioen_dir / "osmose_rho_TestFish_Simu0.csv").exists()
        assert (bioen_dir / "osmose_sizeInf_TestFish_Simu0.csv").exists()

    def test_disabled_flags_no_csv(self, tmp_path):
        """Disabled flags should not produce CSVs."""
        self._run_bioen(tmp_path)  # no extra flags
        bioen_dir = tmp_path / "Bioen"
        # meanEnet is always written; others should not be
        if bioen_dir.exists():
            assert not (bioen_dir / "osmose_ingestion_TestFish_Simu0.csv").exists()

    def test_no_bioen_no_dir(self, tmp_path):
        """Non-bioen config should not create Bioen/ directory."""
        cfg_dict = {**_make_bioen_config(), "simulation.bioen.enabled": "false", "temperature.value": "15.0"}
        cfg = EngineConfig.from_dict(cfg_dict)
        grid = Grid.from_dimensions(ny=2, nx=2)
        outputs = simulate(cfg, grid, np.random.default_rng(42))
        write_outputs(outputs, tmp_path, cfg)
        assert not (tmp_path / "Bioen").exists()
```

- [ ] **Step 7: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_outputs_complete.py tests/test_engine_energy_budget.py tests/test_engine_bioen_integration.py -v`
Expected: All PASS

- [ ] **Step 8: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add osmose/engine/processes/energy_budget.py osmose/engine/simulate.py osmose/engine/config.py osmose/engine/output.py tests/test_engine_bioen_outputs_complete.py
git commit -m "feat(engine): complete all 5 bioen output CSVs (ingestion, maintenance, rho, sizeInf)"
```

---

### Task 5: Final regression + spatial TODO

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Run lint**

Run: `.venv/bin/ruff check osmose/ tests/`
Expected: Clean

- [ ] **Step 3: Add spatial bioen TODO**

In `osmose/engine/output.py`, after `_write_bioen_csvs`, add:
```python
# TODO: Spatial bioen outputs — Java has SpatialEnetOutput, SpatialEnetOutputjuv,
# SpatialEnetOutputlarvae, SpatialdGOutput. Requires per-cell aggregation framework
# (also needed for output.spatial.biomass.enabled etc.). Deferred.
```

- [ ] **Step 4: Commit and tag**

```bash
git add -A
git commit -m "chore: add spatial bioen TODO + final cleanup"
git tag -a engine-phase9 -m "Phase 9: Engine gap closure — bioen reproduction, O2 forcing, per-species RNG, complete outputs"
```

---

## Errata — Review Corrections

Apply these corrections during implementation.

### E1: Replace `_make_bioen_config` with actual fixture pattern

The plan references `from tests.test_engine_bioen_integration import _make_bioen_config` which does NOT exist. The actual test file uses a **pytest fixture** `bioen_config` (defined at `tests/test_engine_bioen_integration.py:47`).

For new test files that need a bioen config dict outside of pytest fixtures, either:
- (a) Extract a standalone `_make_bioen_config()` function from the fixture in `test_engine_bioen_integration.py` (convert the fixture body to a plain function, keep the fixture as a wrapper), OR
- (b) Build the config dict inline in each test, using the same keys as the `bioen_config` fixture

Option (a) is preferred — extract a function, make the fixture call it.

### E2: Update `_average_step_outputs()` for 4 new bioen fields

**Critical.** Task 4 adds `bioen_ingestion_by_species`, `bioen_maint_by_species`, `bioen_rho_by_species`, `bioen_size_inf_by_species` to StepOutput, but `_average_step_outputs()` in `simulate.py` (~line 582) does NOT average them. Without this fix, the new fields will be `None` in averaged outputs and CSVs will contain zeros.

Add to `_average_step_outputs()`, after the existing `bioen_e_net_by_species` handling:

```python
    # Average new bioen fields
    for field in ["bioen_ingestion_by_species", "bioen_maint_by_species",
                  "bioen_rho_by_species", "bioen_size_inf_by_species"]:
        vals = [getattr(o, field) for o in accumulated if getattr(o, field) is not None]
        if vals:
            setattr_kwargs[field] = np.mean(vals, axis=0)
```

Or use the same pass-through-last pattern already used for distribution fields.

### E3: Add full-suite run step to Task 2

Task 2 (O2 wiring) is missing the "Run full suite" step before the commit. Add:

```
- [ ] **Step 5: Run full suite**
Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: All PASS
```

And renumber Step 5 (Commit) to Step 6.

### E4: Clarify `e_gross` as "ingestion" output

In Task 4 Step 3, add a comment that `bioen_ingestion_by_species` aggregates `state.e_gross` (= `ingestion * assimilation * phi_T * f_O2`), which is the assimilated gross energy — matching Java's `Bioen/ingestion` output convention.

### E5: Clarify `sizeInf` semantics

`sizeInf` = max observed length per species per timestep (runtime measurement), NOT `config.linf` (theoretical asymptote). This matches Java's `Bioen/sizeInf` output which tracks the largest individual.
