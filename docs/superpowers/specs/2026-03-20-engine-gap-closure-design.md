# Engine Gap Closure — Design Spec

**Date:** 2026-03-20
**Goal:** Close 5 remaining wiring gaps to complete bioenergetic + RNG infrastructure
**Approach:** Small focused tasks connecting already-implemented modules

## Motivation

Phase 8 (Java parity sprint) implemented all formula modules but left 5 wiring gaps: bioen reproduction not connected to simulation loop, O2 forcing hardcoded to 1.0, per-species RNG accepted but unused, 4 of 5 bioen outputs missing, and no spatial bioen outputs. All underlying functions exist and are unit tested — these tasks connect them.

## Gap 1: Wire Bioen Reproduction

### Problem

`bioen_egg_production()` exists in `osmose/engine/processes/bioen_reproduction.py` but is never called. The simulation loop at `simulate.py` line ~701 calls standard `_reproduction()` unconditionally, which uses SSB-based egg production. In bioen mode, eggs should come from accumulated gonad weight via LMRN maturity.

### Design

Add `_bioen_reproduction()` in `simulate.py` that **directly creates egg schools** from gonad weight, bypassing the SSB-based reproduction entirely:

```python
def _bioen_reproduction(state, config, step, rng, grid_ny, grid_nx):
    """Bioen reproduction: create egg schools from gonad weight (replaces SSB method)."""
    from osmose.engine.processes.bioen_reproduction import bioen_egg_production

    new_egg_schools = []
    gonad = state.gonad_weight.copy()

    for sp in range(config.n_species):
        mask = state.species_id == sp
        if not mask.any():
            continue

        # Get egg weight (handle NaN from missing config)
        ew = np.nan
        if config.egg_weight_override is not None:
            ew = config.egg_weight_override[sp]
        if np.isnan(ew):
            # Fallback: allometric weight at egg size
            ew = config.condition_factor[sp] * config.egg_size[sp] ** config.allometric_power[sp] * 1e-6

        eggs = bioen_egg_production(
            gonad_weight=state.gonad_weight[mask],
            length=state.length[mask],
            age_dt=state.age_dt[mask],
            m0=float(config.bioen_m0[sp]),
            m1=float(config.bioen_m1[sp]),
            egg_weight=float(ew),
            n_dt_per_year=config.n_dt_per_year,
        )

        total_eggs = eggs.sum()
        if total_eggs <= 0:
            continue

        # Reset gonad weight for schools that spawned
        spawned = eggs > 0
        indices = np.where(mask)[0]
        gonad[indices[spawned]] = 0.0

        # Create a single egg school per species (matching Java convention)
        # Place in random cell occupied by parent schools
        parent_cells_y = state.cell_y[mask][spawned]
        parent_cells_x = state.cell_x[mask][spawned]
        if len(parent_cells_y) > 0:
            idx = rng.integers(len(parent_cells_y))
            egg_school = SchoolState.create(n_schools=1, species_id=np.array([sp], dtype=np.int32))
            egg_school = egg_school.replace(
                abundance=np.array([total_eggs]),
                weight=np.array([float(ew)]),
                biomass=np.array([total_eggs * float(ew)]),
                length=np.array([config.egg_size[sp]]),
                cell_x=np.array([parent_cells_x[idx]], dtype=np.int32),
                cell_y=np.array([parent_cells_y[idx]], dtype=np.int32),
                is_egg=np.array([True]),
                first_feeding_age_dt=np.array([1], dtype=np.int32),
            )
            new_egg_schools.append(egg_school)

    state = state.replace(gonad_weight=gonad)

    # Append egg schools
    for egg_school in new_egg_schools:
        state = state.append(egg_school)

    # Age increment for existing schools only (NOT new eggs — they start at age 0)
    # Java's BioenReproductionProcess increments age separately from egg creation
    n_existing = len(state) - sum(len(e) for e in new_egg_schools)
    new_age = state.age_dt.copy()
    new_age[:n_existing] += 1
    state = state.replace(age_dt=new_age)

    return state
```

**Key difference from the earlier flawed approach:** This does NOT delegate to `_reproduction()` (which would compute SSB-based eggs on top). Instead it directly creates egg schools from gonad weight, then increments age for existing schools only. New egg schools retain `age_dt=0`.

**Note on `egg_weight_override` units:** Config values from `species.egg.weight.sp{i}` are already converted to tonnes (`* 1e-6`) during parsing in `config.py` line ~942. The fallback formula (`condition_factor * egg_size^b * 1e-6`) also produces tonnes. Both paths are consistent.

In the main loop, replace:
```python
    # OLD:
    state = _reproduction(state, config, step, rng, ...)
    # NEW:
    if config.bioen_enabled:
        state = _bioen_reproduction(state, config, step, rng, ...)
    else:
        state = _reproduction(state, config, step, rng, ...)
```

### Files Changed

| File | Change |
|------|--------|
| `osmose/engine/simulate.py` | Add `_bioen_reproduction()`, conditional in main loop |

### Tests

- Test bioen reproduction resets gonad weight after spawning
- Test standard reproduction still works when bioen disabled
- Test immature fish don't produce eggs in bioen mode

---

## Gap 2: Wire O2 Forcing

### Problem

In `_bioen_step()`, `f_o2_arr` is always `np.ones()` (line 216). The oxygen function `f_o2()`, schema fields (`oxygen.filename`, `oxygen.value`, etc.), and config parsing (`bioen_o2_c1`, `bioen_o2_c2`) all exist but aren't connected.

### Design

Mirror the temperature loading pattern:

**In `simulate()`**, after temperature loading:
```python
    o2_data = None
    if config.bioen_enabled:
        o2_val = config.raw_config.get("oxygen.value", "")
        if o2_val:
            o2_data = PhysicalData.from_constant(float(o2_val))
        # Note: NetCDF loading for O2 (and temperature) is deferred — only constant mode
        # is currently supported. The PhysicalData.from_netcdf() path exists but neither
        # temperature nor O2 wires it from simulate(). Add when spatial forcing is needed.
```

**In `_bioen_step()`**, pass `o2_data` and replace hardcoded 1.0:
```python
    if config.bioen_fo2_enabled and o2_data is not None:
        from osmose.engine.processes.oxygen_function import f_o2
        for sp in range(config.n_species):
            mask = state.species_id == sp
            if mask.any():
                o2_vals = ... # get O2 from PhysicalData per school
                f_o2_arr[mask] = f_o2(o2_vals, config.bioen_o2_c1[sp], config.bioen_o2_c2[sp])
    else:
        f_o2_arr = np.ones(len(state))
```

Update `_bioen_step` signature to accept `o2_data`:
```python
def _bioen_step(state, config, temp_data, step, o2_data=None):
```

Update call site in main loop:
```python
    state = _bioen_step(state, config, temp_data, step, o2_data=o2_data)
```

### Files Changed

| File | Change |
|------|--------|
| `osmose/engine/simulate.py` | Load O2 PhysicalData, update `_bioen_step` signature + call site, compute f_o2 |

### Tests

- Test f_o2 applied when oxygen.value is set
- Test f_o2 remains 1.0 when no oxygen config
- Test higher O2 -> higher f_o2 -> higher E_gross

---

## Gap 3: Wire Per-Species RNG into Consumers

### Problem

`movement_rngs` and `mortality_rngs` are created in `PythonEngine.run()` and accepted by `simulate()` but never passed to actual consumers. Movement and mortality still use the global `rng`.

### Design

**Movement:** The `movement()` function in `processes/movement.py` takes a single `rng`. Change to accept an optional `species_rngs: list[Generator] | None`. When provided and a species has `movement.randomseed.fixed`, use `species_rngs[sp]` for that species' random walk/map movement. When None, use the global rng (backward compat).

**Predation/Mortality:** The `mortality()` function in `processes/mortality.py` takes a single `rng` which it passes to `predation()`. For cell-level predation ordering (multi-species shuffle), seed from `mortality_rngs[first_predator_sp]` per the Java convention.

**Threading through simulate():**
```python
    state = _movement(state, grid, config, step, rng,
                      map_sets=map_sets, random_patches=random_patches,
                      species_rngs=movement_rngs)
    state = _mortality(state, resources, config, rng, grid, step=step,
                       species_rngs=mortality_rngs)
```

### Files Changed

| File | Change |
|------|--------|
| `osmose/engine/simulate.py` | Update `_movement()` and `_mortality()` wrappers + main loop calls to accept and pass `species_rngs` |
| `osmose/engine/processes/movement.py` | Accept `species_rngs`, use per-species for random walk |
| `osmose/engine/processes/mortality.py` | Accept `species_rngs`, pass to predation |
| `osmose/engine/processes/predation.py` | Accept `species_rngs`, seed cell shuffle from first predator |

### Tests

- Test that `fixed=False` produces identical results to global rng (backward compat)
- Test that `fixed=True` with 3 species, adding a 4th doesn't change species 0-2 trajectories
- Test predation ordering uses first-predator-species rng for cell shuffle

---

## Gap 4: Complete Bioen Outputs (4 missing)

### Problem

Only `meanEnet` is written to `Bioen/` directory. Missing: `ingestion`, `maintenance`, `rho`, `sizeInf`.

### Design

**Extend `compute_energy_budget()` return:** Currently returns `(dw, dg, e_net)`. Change to return `(dw, dg, e_net, e_gross, e_maint, rho)`. These values are already computed internally — just need to be added to the return statement. Update the destructured call in `_bioen_step()` (currently `dw_sp, dg_sp, en_sp = compute_energy_budget(...)`) to capture all 6 values.

**Store in SchoolState:** The fields `e_gross`, `e_maint`, `rho` already exist on SchoolState (added in Task 9) but are NOT currently populated by `_bioen_step()` — update `state.replace()` at the end of `_bioen_step()` to include them.

**Add StepOutput fields:**
```python
    bioen_ingestion_by_species: NDArray[np.float64] | None = None
    bioen_maint_by_species: NDArray[np.float64] | None = None
    bioen_rho_by_species: NDArray[np.float64] | None = None
    bioen_size_inf_by_species: NDArray[np.float64] | None = None
```

**Aggregate in `_collect_outputs()`:** Mean per species for ingestion/maintenance/rho. Max length per species for sizeInf.

**Write in `_write_bioen_csvs()`:** Same pattern as meanEnet — per-species CSV in `Bioen/` directory, gated by `output.bioen.{ingest,maint,rho,sizeInf}.enabled` config flags.

### Files Changed

| File | Change |
|------|--------|
| `osmose/engine/processes/energy_budget.py` | Return e_gross, e_maint, rho in tuple |
| `osmose/engine/simulate.py` | Store bioen fields in state, aggregate to StepOutput |
| `osmose/engine/config.py` | Add `output_bioen_ingest`, `output_bioen_maint`, `output_bioen_rho`, `output_bioen_sizeinf` bool fields + parsing |
| `osmose/engine/output.py` | Write 4 additional bioen CSV types, gated by config flags |

### Tests

- Test each output CSV is created when its flag is enabled
- Test CSV values match expected aggregation (mean for rates, max for sizeInf)
- Test outputs are None/absent when bioen disabled

---

## Gap 5: Spatial Bioen Outputs (Stretch Goal)

### Problem

Java has `SpatialEnetOutput`, `SpatialEnetOutputjuv`, `SpatialEnetOutputlarvae`, `SpatialdGOutput` — 4 spatial output classes. Python has none.

### Design

**Defer to future.** This requires per-cell aggregation infrastructure that doesn't exist yet (the standard spatial outputs like `output.spatial.biomass.enabled` are also unimplemented). Adding spatial bioen outputs without the general spatial output framework would be premature.

**Mark as explicit TODO** in output.py with a comment referencing the Java class names.

---

## Phase Dependencies

```
Gap 1 (bioen reproduction) ─── independent
Gap 2 (O2 forcing)         ─── independent
Gap 3 (per-species RNG)    ─── independent
Gap 4 (bioen outputs)      ─── depends on Gap 1 + 2 (for correct aggregation)
Gap 5 (spatial outputs)    ─── deferred
```

Gaps 1-3 are independent and can be parallelized. Gap 4 should run after 1+2.

## Success Criteria

- `bioen_egg_production()` called in bioen mode; gonad weight resets after spawning
- O2 forcing applied when `oxygen.value` or `oxygen.filename` configured
- Per-species RNG produces independent sequences when `*.randomseed.fixed=true`
- All 5 bioen output CSVs written when their flags are enabled
- All existing 1657 tests still pass (verified count as of engine-phase8 tag)
- ~25 new tests for the 4 gaps
