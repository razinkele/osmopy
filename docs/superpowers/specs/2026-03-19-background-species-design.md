# D2: Background Species — Design Spec

> **Date:** 2026-03-19
> **Feature:** Background species support for the Python OSMOSE engine
> **Approach:** Inject into SchoolState with `is_background` flag (Approach B)

## Overview

Background species are an intermediate tier between focal (full IBM) and resource (LTL plankton) species. They have multiple size classes with biomass from external forcing, participate in predation as both predators and prey, but don't grow, reproduce, or undergo non-predation mortality. Their biomass resets from forcing each timestep.

They serve as the critical **trophic buffer** — species you don't model fully but whose biomass presence stabilizes predation dynamics. Without them, focal species lack a stable prey base, causing ecosystem collapse in validation.

## OSMOSE Species Hierarchy

```
Focal species    → full IBM: growth, predation, reproduction, mortality, movement
Background species → size classes, forcing biomass, predation only (eat and are eaten)
Resource species  → single size range, forcing only, prey only (eaten, don't eat)
```

## Config Keys

From the eec_full example (background species uses sp55):

```
simulation.nbackground = 1                          # validation check
species.type.sp55 = background                      # type discovery
species.name.sp55 = backgroundSpecies
species.nclass.sp55 = 2                             # number of size classes
species.length.sp55 = 10;30                         # length per class (cm)
species.size.proportion.sp55 = 0.3;0.7              # constant proportions
# OR: species.size.proportion.file.sp55 = path.csv  # time-series proportions
species.trophic.level.sp55 = 2, 3                   # TL per class
species.age.sp55 = 1;3                              # age per class (years)
species.length2weight.condition.factor.sp55 = 0.00308
species.length2weight.allometric.power.sp55 = 3.029
predation.predprey.sizeratio.max.sp55 = 3
predation.predprey.sizeratio.min.sp55 = 50
predation.ingestion.rate.max.sp55 = 3.5
species.file.sp55 = biomass.nc                      # NetCDF spatial forcing
# OR: species.biomass.total.sp55 = 1000.0           # uniform total biomass
```

Species type discovery: scan all `species.type.sp*` keys for `"background"` value and extract the `sp{N}` file index. Sort discovered file indices numerically — this determines the internal background index (0, 1, 2...) which maps to the global species ID. The `simulation.nbackground` count is a validation check (Java calls `error()` on mismatch — behavior depends on error handler configuration). Python should log a warning and continue.

### Proportion Modes

1. **Constant:** `species.size.proportion.sp{N}` — semicolon-separated floats summing to 1.0
2. **Time-series:** `species.size.proportion.file.sp{N}` — CSV with columns per class, rows per timestep

### Forcing Modes

1. **NetCDF spatial:** `species.file.sp{N}` — spatial biomass per cell per timestep (primary mode)
2. **Uniform:** `species.biomass.total.sp{N}` — total biomass divided equally among ocean cells

### Forcing Modifiers

From `ResourceForcing.java`:

- **`species.multiplier.sp{N}`** (optional, default 1.0) — scales raw biomass from forcing
- **`species.offset.sp{N}`** (optional, default 0.0) — additive offset before multiplier
- Formula (uniform mode): `per_cell_biomass = multiplier * (total / n_ocean_cells + offset)`
- Formula (NetCDF mode): multiplier is passed to `ForcingFile` constructor; offset is read but NOT applied to NetCDF data in Java (this is a Java bug — offset only affects uniform mode)
- **`species.biomass.nsteps.year.sp{N}`** (optional) — number of time steps per year in the forcing file. Falls back to global `species.biomass.nsteps.year`. Controls temporal interpolation when forcing resolution differs from simulation resolution (e.g., monthly forcing at 24 steps/year simulation).
- **`species.file.caching.sp{N}`** (optional, default `ALL`) — NetCDF caching mode. Python implementation will always cache all data in memory (equivalent to Java's `ALL` mode).

### Name Validation

Java strips underscores and hyphens from background species names (`name.replaceAll("_", "").replaceAll("-", "")`) and requires the result to be alphanumeric only (`[a-zA-Z0-9]`). Python should apply the same stripping for output file compatibility.

## Species Indexing

Matches Java's indexing convention:

```
[0, n_focal)                              → focal species
[n_focal, n_focal + n_bkg)                → background species
[n_focal + n_bkg, n_focal + n_bkg + n_rsc) → resource species
```

The accessibility matrix must be sized to cover all three tiers. Background species use their global index (`n_focal + bkg_internal_index`) for accessibility lookups.

## Data Model

### `BackgroundSpeciesInfo` dataclass

Per-species metadata parsed from config:

```python
@dataclass
class BackgroundSpeciesInfo:
    name: str
    species_index: int            # global index: n_focal + bkg_idx
    file_index: int               # sp{N} index in config keys
    n_class: int
    lengths: NDArray[np.float64]  # (n_class,) cm
    trophic_levels: NDArray[np.float64]  # (n_class,)
    ages_dt: NDArray[np.int32]    # (n_class,) in timesteps; uses Java's truncate-first: int(age_years) * n_dt_per_year
    condition_factor: float       # c in W = c * L^b
    allometric_power: float       # b in W = c * L^b
    size_ratio_min: float         # predation min ratio
    size_ratio_max: float         # predation max ratio
    ingestion_rate: float         # max predation rate
    multiplier: float             # biomass scaling (default 1.0)
    offset: float                 # biomass additive offset (default 0.0, uniform mode only)
    forcing_nsteps_year: int      # temporal steps per year in NetCDF forcing
    proportions: NDArray[np.float64] | None     # (n_class,) constant mode
    proportion_ts: NDArray[np.float64] | None   # (n_steps, n_class) time-series
```

### `BackgroundState` class

Manages forcing data and school generation:

```python
class BackgroundState:
    def __init__(self, config: dict[str, str], grid: Grid, engine_config: EngineConfig):
        # Parse background species from config
        # Load NetCDF forcing or uniform biomass
        # Pre-compute weights from allometry: w = c * length^b

    def get_schools(self, step: int) -> SchoolState:
        # Returns a SchoolState with all background schools for this timestep
        # One school per species per class per ocean cell
        # Biomass = forcing_biomass[cell] * proportion[class, step]
        # Weight = c * length^b (fixed per class)
        # Abundance = biomass / weight
        # is_background = True for all
        # species_id = n_focal + bkg_internal_index
        # first_feeding_age_dt = -1 (Java convention: always eligible to predate)
        # age_dt = int(age_years) * n_dt_per_year (truncate-first, matching Java)
        #
        # NOTE: Allocates new arrays each call (Java reuses objects via bkg.init()).
        # This is a known trade-off: cleaner code, small perf cost (~400 rows/step).
```

**File:** `osmose/engine/background.py`

## Integration: Inject/Strip Pattern

Background schools are temporarily injected into `SchoolState` for the mortality step, then stripped out. This leverages the existing `is_background` field and requires zero changes to the predation kernel.

### Simulation Loop Changes (`simulate.py`)

```python
for step in range(config.n_steps):
    state = _incoming_flux(state, config, step, rng)
    state = _reset_step_variables(state)
    resources.update(step)
    background.update(step)  # reload forcing for this timestep
    state = _movement(state, grid, config, step, rng)

    # Inject background schools before mortality
    bkg_schools = background.get_schools(step)
    n_focal = len(state)
    if len(bkg_schools) > 0:
        state = state.append(bkg_schools)

    state = _mortality(state, resources, config, rng, grid)

    # Collect background biomass/abundance BEFORE stripping
    bkg_output = _collect_background_outputs(state, config, n_focal)

    # Strip background schools (they were appended at the end)
    state = _strip_background(state, n_focal)

    state = _growth(state, config, rng)
    state = _aging_mortality(state, config)
    state = _reproduction(state, config, step, rng)

    # Collect focal outputs at the same point as before (after reproduction)
    outputs.append(_collect_outputs(state, config, step, bkg_output))
    state = state.compact()
```

`_strip_background(state, n_focal)` slices all arrays to `[:n_focal]`.

### Mortality Orchestrator Changes (`processes/mortality.py`)

The interleaved mortality loop already processes schools by index. Changes needed:

- **Starvation:** `if is_background[idx]: skip`
- **Additional mortality:** `if is_background[idx]: skip`
- **Fishing (Osmose 3 old-style):** `if is_background[idx]: skip`
- **Predation:** No change — background schools participate normally as both predator and prey

This matches Java's behavior (MortalityProcess.java lines 606, 629, 684).

**Note on Osmose 4 fisheries:** In Java's v4 fishery path (lines 648-679), background species CAN be fished — `listPred.get(seqFish[i])` draws from the combined focal+background list. The current Python engine only implements Osmose 3 fishing, so we skip fishing for background schools. If/when v4 fisheries are added, this skip must be conditional.

### Predation Kernel

**Zero changes needed.** Background schools have valid `species_id`, `length`, `weight`, `abundance`, `age_dt`, and `first_feeding_age_dt` (set to `-1`) fields. The Numba kernel and Python fallback treat them identically to focal schools.

**`first_feeding_age_dt` = -1:** Java's `BackgroundSchool.getFirstFeedingAgeDt()` returns `-1`. The predation gate checks `age_dt < first_feeding_age_dt` — since `age_dt >= 0` and `first_feeding_age_dt = -1`, this is always false, so background schools always participate as predators. This must NOT be set to `0` (the `SchoolState.create()` default), or background schools with `age_dt=0` would be gated out.

The accessibility matrix is looked up by `species_id`, which uses the global index (`n_focal + bkg_idx`). As long as the matrix is sized correctly, this works automatically.

## EngineConfig Changes

New fields:

- `n_background: int` — number of background species
- `background_file_indices: list[int]` — raw `sp{N}` file indices
- `all_species_names: list[str]` — `focal_names + background_names` (for output headers)

Discovery: scan config for `species.type.sp*` keys with value `"background"`. Extract file index from key suffix.

## Output Changes

### Biomass/abundance output

Arrays sized `n_focal + n_background` (was `n_focal`). Background species appear as additional columns in CSV. The existing `np.add.at(biomass, state.species_id, state.biomass)` logic works unchanged since background schools have species IDs in range `[n_focal, n_focal + n_bkg)`.

### Mortality rate output

Remains **focal-only** (`n_focal` columns). Background species are excluded from mortality rate CSVs. This matches Java's `MortalitySpeciesOutput` which only iterates focal schools.

### Diet output (future)

Java's `DietOutput` sizes the predator dimension to `n_focal + n_bkg` and includes background species names in headers. However, `DietOutput.update()` only iterates focal schools — background predator columns are always zero. If/when diet output is added to the Python engine, it should include background species columns (with zeros) for compatibility. Background schools' `preys` tracking (populated during `predator.preyedUpon()`) is not read back in Java output — it exists for internal bookkeeping only.

### `StepOutput` changes

```python
@dataclass
class StepOutput:
    step: int
    biomass: NDArray[np.float64]          # (n_focal + n_bkg,)
    abundance: NDArray[np.float64]        # (n_focal + n_bkg,)
    mortality_by_cause: NDArray[np.float64]  # (n_focal, n_causes) — focal only
```

### Timing

Output collection is split into two phases to avoid changing focal species output behavior:

1. **`_collect_background_outputs(state, config, n_focal)`** — called BEFORE stripping, aggregates background species biomass/abundance from the appended schools (indices `>= n_focal`).
2. **`_collect_outputs(state, config, step, bkg_output)`** — called AFTER reproduction (same position as before), aggregates focal species and merges in the pre-collected background data.

This preserves the existing behavior where focal outputs capture post-reproduction state, while background outputs capture post-predation state (which is the only meaningful snapshot for forcing-driven species).

## Java Parity Notes

### `setBiomass` abundance base divergence (DELIBERATE)

Java's `BackgroundSchool.setBiomass(biomass, step)` has a biomass/abundance inconsistency:
```java
this.biomass = this.instantaneousBiomass = biomass * this.getProportion(step);  // proportioned
this.abundance = this.instantaneousAbundance = this.biom2abd(biomass);           // NOT proportioned (total cell biomass / weight)
```

This means:
- `instantaneousBiomass` starts at `total * proportion` (correct for predation biomass checks)
- `instantaneousAbundance` starts at `total / weight` (inflated by `1/proportion`)
- After `incrementNdead`, `updateBiomAndAbd()` recomputes: `instantaneousAbundance = abundance - sum(nDead)`, then `instantaneousBiomass = instantaneousAbundance * weight`

The effect: background schools have a larger initial abundance base, so the same `nDead` represents a smaller fractional loss. This makes background schools more resilient to predation than if abundance were derived from proportioned biomass.

**Python decision:** Use **consistent** values (`abundance = proportioned_biomass / weight`). This is a deliberate divergence from Java that will make background schools slightly more vulnerable to predation pressure. A quantitative test must compare predation loss between the two approaches to measure the magnitude.

**Rationale:** The Java behavior appears to be a bug (biomass and abundance should be consistent). Replicating a bug for parity alone would be wrong — we should match the *intent*, not the *accident*. If validation later shows this causes ecosystem divergence, we can add a `java_compat` flag to replicate the Java behavior.

### ResourceForcing shared system

Java uses a shared `ResourceForcing` array indexed `[0, n_bkg)` for background species and `[n_bkg, n_bkg + n_rsc)` for resource species. Both load from `species.file.sp{N}`.

In Python, `BackgroundState` handles its own forcing loading (similar to how `ResourceState` handles resource forcing). This avoids coupling the two systems while using the same NetCDF loading patterns.

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `osmose/engine/background.py` | Create | `BackgroundSpeciesInfo`, `BackgroundState` |
| `osmose/engine/config.py` | Modify | Add `n_background`, `background_file_indices`, `all_species_names` |
| `osmose/engine/simulate.py` | Modify | Inject/strip pattern, output timing |
| `osmose/engine/processes/mortality.py` | Modify | Skip non-predation for `is_background` |
| `osmose/engine/output.py` | Modify | Resize arrays for `n_focal + n_bkg` |
| `tests/test_engine_background.py` | Create | Full test suite |

## Testing Strategy (TDD)

Tests written before implementation, covering:

1. **Config parsing:** Parse background species from config dict, validate proportions sum to 1.0
2. **Config discovery:** Multiple background species sorted by file index numerically
3. **Name validation:** Underscore/hyphen stripping, alphanumeric-only check
4. **Forcing modes:** NetCDF spatial loading, uniform biomass fallback
5. **Forcing modifiers:** `species.multiplier.sp{N}` and `species.offset.sp{N}` applied correctly; offset only affects uniform mode (matching Java bug)
6. **Forcing temporal resolution:** `species.biomass.nsteps.year.sp{N}` per-species override with global fallback
7. **Proportion modes:** Constant array, time-series CSV
8. **School generation:** Correct biomass/abundance/weight per class per cell
9. **`first_feeding_age_dt`:** Must be -1, not 0; background schools with age_dt=0 still predate
10. **`ages_dt` conversion:** Verify truncate-first matches Java: `int(1.5) * 24 = 24`, not `round(1.5 * 24) = 36`
11. **Injection:** Background schools appended with correct `is_background=True` and species IDs
12. **Predation participation:** Background schools are eaten and eat focal species
13. **Mortality skip:** Starvation/additional/fishing skip background schools
14. **Stripping:** Background schools cleanly removed after mortality; focal state unaffected
15. **Output:** Biomass CSV has `n_focal + n_bkg` columns, mortality CSV has `n_focal` columns
16. **Output timing:** Focal outputs still capture post-reproduction state
17. **Abundance divergence test:** Quantify predation loss difference between consistent (Python) vs inflated (Java) abundance base
18. **Integration:** Full simulation with background species produces stable ecosystem
