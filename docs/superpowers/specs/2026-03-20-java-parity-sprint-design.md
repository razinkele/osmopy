# Java Parity Sprint — Design Spec

**Date:** 2026-03-20
**Goal:** Complete Java feature parity for the Python OSMOSE engine
**Approach:** Focused sprint, 4 dependency-ordered phases

## Motivation

The Python engine (phases 1-7) covers ~95% of Java features with 1553 tests collected and Bay of Biscay 8/8 PASS validation. Four gaps remain before the Python engine can fully replace Java for all configuration types:

1. Code quality debt in predation and growth
2. Missing size/age distribution outputs
3. Stubbed per-species deterministic RNG
4. Unimplemented bioenergetic (Ev-OSMOSE) module

## Phase 1: Code Quality — Reconcile Predation & Wire Growth Dispatch

### Problem

**Dual predation paths:** `_predation_in_cell_numba()` and `_predation_in_cell_python()` in `osmose/engine/processes/predation.py` duplicate ~150 lines of core logic. The Numba path lacks diet tracking; the Python path has it. Selection logic (line ~517) branches on `_HAS_NUMBA and not _diet_tracking_enabled`. If one path gets a bug fix, the other can silently diverge.

**Unwired growth dispatch:** Schema declares `growth.java.classname.sp{idx}` with Java classnames. The actual Java classes are `fr.ird.osmose.process.growth.VonBertalanffyGrowth` and `fr.ird.osmose.process.growth.GompertzGrowth` (note: `process.growth` package, `Growth` suffix). There is no `LinearGrowth` Java class — the "Linear" in the current schema is incorrect. Only VonBertalanffy is integrated. `expected_length_gompertz()` exists in `growth.py` (lines 109-158) but is never called. Config never reads the classname key.

### Design

**Predation — extract shared helpers (minimal reconciliation):**

- Extract into shared functions:
  - `compute_size_overlap(pred_length, prey_length, ratio_min, ratio_max) -> bool`
  - `compute_appetite(weight, ingestion_rate, n_subdt) -> float`
  - `apply_biomass_transfer(pred_biomass, prey_biomass, prey_abundance, appetite, access_coeff) -> (eaten, new_prey_abundance, new_prey_biomass)`
- Both Numba and Python paths call these helpers
- Numba helpers get `@njit` decorators; Python helpers are plain functions with identical signatures
- Diet tracking: pass a pre-allocated diet matrix into the Numba path (Numba can mutate passed-in arrays). Both paths accumulate diet in-place. This is simpler than returning per-interaction amounts.
- Diet tracking flag no longer determines which path runs; Numba availability is the only selector

**Growth — config-driven dispatch:**

- Parse `growth.java.classname.sp{idx}` in `EngineConfig.from_dict()`
- Store as `config.growth_class[sp]` enum: `VB | GOMPERTZ` (mapped from `fr.ird.osmose.process.growth.{name}Growth`)
- Fix schema enum values to match Java: `fr.ird.osmose.process.growth.VonBertalanffyGrowth`, `fr.ird.osmose.process.growth.GompertzGrowth` (remove `Linear` — no Java counterpart)
- In `growth()`, dispatch per species:
  - `VB`: existing `expected_length_vb()` (no change)
  - `GOMPERTZ`: wire existing `expected_length_gompertz()`, parse additional params
- When `simulation.bioen.enabled = true`: growth dispatch is bypassed entirely; `EnergyBudget` handles weight/length updates

**Gompertz config keys to add** (from Java `GompertzGrowth.java`):

| Key | Type | Description |
|-----|------|-------------|
| `growth.exponential.ke.sp{idx}` | FLOAT | Exponential growth rate (early phase) |
| `growth.exponential.thr.age.sp{idx}` | FLOAT | Age switching to exponential (years) |
| `growth.exponential.lstart.sp{idx}` | FLOAT | Starting length for exponential phase |
| `growth.gompertz.thr.age.sp{idx}` | FLOAT | Age switching to Gompertz (years) |
| `growth.gompertz.kg.sp{idx}` | FLOAT | Gompertz growth rate |
| `growth.gompertz.tg.sp{idx}` | FLOAT | Gompertz inflection age |
| `growth.gompertz.linf.sp{idx}` | FLOAT | Gompertz asymptotic length |

### Files Changed

| File | Change |
|------|--------|
| `osmose/engine/processes/predation.py` | Extract helpers, simplify path selection |
| `osmose/engine/processes/growth.py` | Add dispatch, wire Gompertz params |
| `osmose/engine/config.py` | Parse `growth.java.classname`, Gompertz params |
| `osmose/schema/species.py` | Fix growth classname enum, add Gompertz config fields |

### Tests

- Unit tests for each extracted predation helper
- Test that Numba and Python paths produce identical results for same inputs
- Test diet tracking works with both paths
- Test growth dispatch for each class (VB, Gompertz, Linear)
- Test bioen bypass skips growth dispatch

**Estimated:** ~300 LOC changed, ~15 new tests

---

## Phase 2: Size/Age Distribution Outputs

### Problem

`StepOutput` stores only species-level aggregates (1D arrays). Java produces per-species CSV files binned by age class or size class: `biomassByAge`, `abundanceByAge`, `biomassBySize`, `abundanceBySize`. These are essential for calibration workflows and publication-quality analysis. The schema already defines `output.distrib.bysize.{min,max,incr}` but nothing reads them.

### Design

**Extend StepOutput:**

```python
@dataclass
class StepOutput:
    step: int
    biomass: NDArray[np.float64]                          # (n_species,)
    abundance: NDArray[np.float64]                        # (n_species,)
    mortality_by_cause: NDArray[np.float64]                # (n_species, n_causes)
    yield_by_species: NDArray[np.float64] | None
    # New:
    biomass_by_age: dict[int, NDArray[np.float64]] | None   # sp -> (n_age_bins,)
    abundance_by_age: dict[int, NDArray[np.float64]] | None
    biomass_by_size: dict[int, NDArray[np.float64]] | None  # sp -> (n_size_bins,)
    abundance_by_size: dict[int, NDArray[np.float64]] | None
```

**Binning logic in `_collect_outputs()`:**

- Age bins: integer age classes `0, 1, ..., lifespan` (one bin per age in dt units, aggregated to years for output)
- Size bins: from config `output.distrib.bysize.{min, max, incr}` — e.g., 0-205 cm in 10 cm steps
- For each species, iterate schools, accumulate biomass/abundance into appropriate bin
- Only compute when corresponding `output.{biomass,abundance}.by{age,size}.enabled` is true

**Output format (matching Java):**

- Per-species CSV: `{prefix}_biomassByAge_{species_name}_Simu0.csv`
- Rows = timesteps, columns = bin edges
- Header row with bin labels

**Config keys to parse:**

Size bin parameters (existing in schema):
| Key | Type | Default |
|-----|------|---------|
| `output.distrib.bysize.min` | FLOAT | 0 |
| `output.distrib.bysize.max` | FLOAT | 205 |
| `output.distrib.bysize.incr` | FLOAT | 10 |

Enable flags (already in `osmose/schema/output.py` — wire to engine config):
| Key | Type | Default |
|-----|------|---------|
| `output.biomass.byage.enabled` | BOOL | false |
| `output.biomass.bysize.enabled` | BOOL | false |
| `output.abundance.byage.enabled` | BOOL | false |
| `output.abundance.bysize.enabled` | BOOL | false |

### Files Changed

| File | Change |
|------|--------|
| `osmose/engine/output.py` | Extend StepOutput, add binning, add CSV writers |
| `osmose/engine/simulate.py` | Pass distribution data through collect step |
| `osmose/engine/config.py` | Parse distribution config keys |
| `osmose/schema/output.py` | Verify/add missing schema fields |

### Tests

- Unit test binning logic with known school distributions
- Test edge cases: empty bins, single school, all schools same age/size
- Test CSV output format matches Java column headers
- Test disabled flags produce no output files

**Estimated:** ~200 LOC new, ~20 tests

---

## Phase 3: Per-Species Deterministic RNG

### Problem

Two config flags are parsed but never consulted:
- `movement.randomseed.fixed` (in `movement.py` schema, line 88)
- `stochastic.mortality.randomseed.fixed` (in `simulation.py` schema, line 137)

A single global `rng = np.random.default_rng(seed)` is used everywhere. Java creates per-species RNG instances from fixed seeds when these flags are true, ensuring that adding/removing a species doesn't change the random sequence for other species.

### Design

**RNG factory in engine init:**

```python
def build_rng(seed: int, n_species: int, fixed: bool) -> list[np.random.Generator]:
    if not fixed:
        # All species share one RNG (current behavior)
        shared = np.random.default_rng(seed)
        return [shared] * n_species
    # Per-species deterministic seeds
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_species)
    return [np.random.default_rng(s) for s in child_seeds]
```

**Two RNG sets:**
- `movement_rngs: list[Generator]` — controlled by `movement.randomseed.fixed`
- `mortality_rngs: list[Generator]` — controlled by `stochastic.mortality.randomseed.fixed`

**Update consumers:**

| Consumer | Current | New |
|----------|---------|-----|
| Predation ordering | `rng.permutation()` over all schools in cell | Seed cell-level shuffle from `mortality_rngs[first_predator_sp]` — cell-level operation can't use per-species RNG directly |
| Movement (map-based) | `rng.choice()` | `movement_rngs[sp].choice()` per species |
| Reproduction | `rng` (minimal use) | `mortality_rngs[sp]` |
| Starvation | Global rng | `mortality_rngs[sp]` |

**Note on predation:** Cell-level predation ordering shuffles all schools in a cell (multi-species). Java handles this by seeding from the first predator species. We follow the same convention — the per-species guarantee applies to movement and mortality, not to inter-species predation order.

**Backward compatibility:** When both flags are false (default), behavior is identical to current — all species share one Generator instance.

### Files Changed

| File | Change |
|------|--------|
| `osmose/engine/__init__.py` | Replace single rng with `build_rng()` |
| `osmose/engine/simulate.py` | Pass species-specific rng to process functions |
| `osmose/engine/processes/predation.py` | Accept per-species rng |
| `osmose/engine/movement_maps.py` | Accept per-species rng |
| `osmose/engine/config.py` | Read the two fixed-seed flags |

### Tests

- Test that `fixed=false` produces same results as current global rng
- Test that `fixed=true` gives reproducible per-species sequences
- Test that adding a species doesn't change sequences for existing species
- Test both flags independently

**Estimated:** ~100 LOC changed, ~10 tests

---

## Phase 4: Bioenergetic (Ev-OSMOSE) Module

### Problem

The bioenergetic module replaces standard growth/starvation/reproduction with energy-budget-driven processes. It's required for Ev-OSMOSE configurations. Schema defines 9 keys; Java has 8 classes with ~25 per-species config keys plus temperature/oxygen forcing. No public example configs exist, so validation is formula-by-formula against Java source.

### Design

**New engine modules:**

#### `osmose/engine/processes/energy_budget.py`
Core energy allocation pipeline, called per school per timestep:

```
E_gross = ingestion * assimilation * phi_T(T) * f_O2(O2)
E_maint = C_m * w^beta * arrhenius(T) / n_dt_per_year
E_net   = E_gross - E_maint
rho     = maturity_allocation(age, length, weight, m0, m1, r, eta, E_net_avg)
dw      = (1 - rho) * E_net   if E_net > 0 else 0
dg      = rho * E_net          if E_net > 0 else 0
```

State updates: `weight += dw`, `gonad_weight += dg`, `length` derived from new weight via allometric relationship.

#### `osmose/engine/processes/temp_function.py`
Johnson thermal performance curve:

```
phi_T(T) = Phi * exp(-e_M / (k_B * T_K)) / (1 + (e_M/(e_D - e_M)) * exp(e_D/k_B * (1/T_P - 1/T_K)))
```

Where `k_B = 8.62e-5` eV/K, `T_K = T + 273.15`, `Phi` normalizes so `phi_T(T_P) = 1`.

Arrhenius for maintenance: `arrhenius(T) = exp(-e_m / (k_B * T_K))`

#### `osmose/engine/processes/oxygen_function.py`
Dose-response: `f_O2(O2) = C1 * O2 / (O2 + C2)`

#### `osmose/engine/physical_data.py`
Generic NetCDF/constant loader for temperature and oxygen forcing:
- Constant mode: `temperature.value` overrides file
- NetCDF mode: 3D array indexed by `(time_step, cell_y, cell_x)`, updated per simulation step
- Config: `{var}.filename`, `{var}.varname`, `{var}.nsteps.year`, `{var}.factor`, `{var}.offset`
- Species access via `species.zlayer.sp{N}` (depth layer index)

#### `osmose/engine/processes/bioen_predation.py`
Overrides standard predation with allometric ingestion:

```
# For adults:
I_max_effective = I_max * w^beta / n_dt_per_year

# For larvae (age < threshold):
I_max_effective = (I_max + (theta - 1) * c_rate) * w^beta / n_dt_per_year

ingestion = min(predated_biomass, I_max_effective)
```

Where `theta = predation.coef.ingestion.rate.max.larvae.bioen` and `c_rate = predation.c.bioen` (additive correction, not simple multiplier).

#### `osmose/engine/processes/bioen_starvation.py`
Energy-deficit starvation with gonad buffer, processed per sub-timestep (matching Java `BioenStarvationMortality`):

```
for each sub-timestep:
    e_net_subdt = E_net / n_subdt
    if e_net_subdt < 0:
        deficit = abs(e_net_subdt)
        if gonad_weight >= eta * deficit:
            # Gonad absorbs the full deficit
            gonad_weight -= eta * deficit
        else:
            # Gonad insufficient — fish die proportionally
            remaining = deficit - gonad_weight / eta
            n_dead += remaining / weight
            gonad_weight = 0
```

Note: `eta` converts somatic energy deficit to gonadic weight cost (gonad pays `eta * deficit`). This matches Java where `gonad_weight >= eta * eNetSubDt` is the sufficiency check.

#### `osmose/engine/processes/bioen_reproduction.py`
Gonad-weight egg production (replaces SSB-based reproduction):
- Eggs produced proportional to accumulated `gonad_weight`
- Gonad weight reset after spawning
- Maturity determined by LMRN: `L_mature = m0 + m1 * age`

### SchoolState Extensions

Add to `SchoolState`:

| Field | Type | Purpose |
|-------|------|---------|
| `gonad_weight` | `NDArray[np.float64]` | Already exists in SchoolState (initialized to 0); populate with meaningful values during bioen runs |
| `e_net_avg` | `NDArray[np.float64]` | Running lifetime average of mass-specific E_net (4 regimes — see below) |
| `e_gross` | `NDArray[np.float64]` | Current step gross energy (for output) |
| `e_maint` | `NDArray[np.float64]` | Current step maintenance (for output) |
| `e_net` | `NDArray[np.float64]` | Current step net energy (for output) |
| `rho` | `NDArray[np.float64]` | Current allocation fraction (for output) |

**`e_net_avg` computation regimes** (from Java `EnergyBudget.computeEnetFaced()`):
1. Before first-feeding age: `e_net_avg = 0`
2. At first-feeding step: special init with larvae factor
3. Larvae phase (age < threshold): incremental average WITH larvae correction factor
4. Adult phase: incremental average WITHOUT larvae factor (`e_net_avg = (e_net_avg * (n-1) + e_net_mass_specific) / n`)

### Config Keys (full list)

**Global flags:**
- `simulation.bioen.enabled` (BOOL)
- `simulation.bioen.phit.enabled` (BOOL, default true)
- `simulation.bioen.fo2.enabled` (BOOL, default true)

**Per-species (all `*.sp{idx}`):**
- `species.beta.sp{N}` — allometric exponent
- `species.zlayer.sp{N}` — depth layer index
- `species.bioen.assimilation.sp{N}` — assimilation efficiency
- `species.bioen.maint.energy.c_m.sp{N}` — maintenance rate
- `species.bioen.maturity.eta.sp{N}` — energy density ratio
- `species.bioen.maturity.r.sp{N}` — reproductive allocation
- `species.bioen.maturity.m0.sp{N}` — LMRN intercept
- `species.bioen.maturity.m1.sp{N}` — LMRN slope
- `species.bioen.mobilized.e.mobi.sp{N}` — activation energy
- `species.bioen.mobilized.e.D.sp{N}` — deactivation energy
- `species.bioen.mobilized.Tp.sp{N}` — peak temperature
- `species.bioen.maint.e.maint.sp{N}` — maintenance Arrhenius energy
- `species.oxygen.c1.sp{N}` — O2 asymptote
- `species.oxygen.c2.sp{N}` — O2 half-saturation
- `predation.ingestion.rate.max.bioen.sp{N}` — I_max
- `predation.coef.ingestion.rate.max.larvae.bioen.sp{N}` — larvae multiplier
- `predation.c.bioen.sp{N}` — larvae correction coefficient
- `species.bioen.forage.k_for.sp{N}` — foraging mortality

**Physical forcing:**
- `temperature.{filename,varname,nsteps.year,factor,offset,value}`
- `oxygen.{filename,varname,nsteps.year,factor,offset,value}`

### Simulation Loop Integration

In `simulate.py`, when `config.bioen_enabled`:
1. Load temperature/oxygen grids at start
2. Each timestep: update physical data → bioen predation → energy budget → bioen starvation → bioen reproduction
3. Skip standard growth, starvation, reproduction processes
4. Standard movement, natural mortality, fishing still run

### Bioen-Specific Outputs

| Config Key | Output File |
|------------|-------------|
| `output.bioen.ingest.enabled` | `Bioen/ingestion_{species}_Simu0.csv` |
| `output.bioen.maint.enabled` | `Bioen/maintenance_{species}_Simu0.csv` |
| `output.bioen.enet.enabled` | `Bioen/meanEnet_{species}_Simu0.csv` |
| `output.bioen.rho.enabled` | `Bioen/rho_{species}_Simu0.csv` |
| `output.bioen.sizeInf.enabled` | `Bioen/sizeInf_{species}_Simu0.csv` |

**Note:** `output.bioen.rho.enabled` and `output.bioen.sizeInf.enabled` are not yet in `osmose/schema/output.py` — add them alongside the existing `ingest/maint/enet` flags.

Distribution variants (`by age/size`) use Phase 2 infrastructure.

### Validation Strategy

No integration benchmark available. Validate formula-by-formula:
- `phi_T(T)` at known temperatures against hand-computed values from Java equation
- `arrhenius(T)` same approach
- `f_O2(O2)` same approach
- `E_gross`, `E_maint`, `E_net` pipeline with synthetic inputs
- Starvation: gonad depletion arithmetic
- Reproduction: egg count from gonad weight
- Rho allocation: test immature (rho=0) vs mature cases

### Out of Scope (Ev-OSMOSE Genetics)

The genetic evolution layer (`Genotype`, `Trait`, `Locus`, weighted parent selection) is **not included** in this sprint. It's a separate system layered on top of bioenergetics — deferring it doesn't block any bioen functionality. Can be added later when genetic configs are available for testing.

### Files Created/Changed

| File | Status |
|------|--------|
| `osmose/engine/processes/energy_budget.py` | New |
| `osmose/engine/processes/temp_function.py` | New |
| `osmose/engine/processes/oxygen_function.py` | New |
| `osmose/engine/physical_data.py` | New |
| `osmose/engine/processes/bioen_predation.py` | New |
| `osmose/engine/processes/bioen_starvation.py` | New |
| `osmose/engine/processes/bioen_reproduction.py` | New |
| `osmose/engine/state.py` | Extend SchoolState |
| `osmose/engine/config.py` | Parse all bioen config keys |
| `osmose/engine/simulate.py` | Bioen branch in simulation loop |
| `osmose/engine/output.py` | Bioen output writers |
| `osmose/schema/bioenergetics.py` | Expand with missing fields |

**Estimated:** ~800 LOC new, ~60 tests

---

## Phase Dependencies

```
Phase 1 (code quality) ──┐
Phase 2 (outputs)     ───┼──> Phase 4 (bioenergetics)
Phase 3 (RNG)         ───┘
```

Phases 1-3 are independent and can be parallelized. Phase 4 depends on all three.

## Success Criteria

- All existing 1553+ tests still pass
- ~105 new tests for the four phases
- Growth dispatch works for VB, Gompertz, Linear configs
- Size/age distribution CSVs match Java column format
- Per-species RNG produces reproducible, independent sequences
- All bioenergetic equations match Java source formulas (unit tested)
- `simulation.bioen.enabled = true` runs without error on a synthetic config
