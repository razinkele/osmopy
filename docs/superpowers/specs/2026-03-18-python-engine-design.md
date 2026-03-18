# Python OSMOSE Engine — Design Specification

**Date:** 2026-03-18
**Status:** Approved (rev 2 — post spec review)
**Goal:** Reimplement the OSMOSE Java simulation engine in Python using vectorized NumPy/JAX operations for performance and extensibility, coexisting alongside the Java engine.

## Motivation

The OSMOSE Java engine (v4.3.3, 350+ classes) is mature but presents barriers:

- **Performance ceiling:** No GPU path; calibration requires thousands of subprocess invocations
- **Extensibility:** Adding/modifying processes requires Java development, recompilation, and JAR rebuilding
- **Deployment:** JVM dependency complicates pure-Python deployments
- **Integration:** Subprocess boundary prevents tight coupling with Python ML/optimization ecosystem

A vectorized Python engine addresses all four while preserving full compatibility with the existing config system, output format, UI, and calibration pipeline.

## Design Priorities

1. **Performance (primary):** 10-100x faster than Java via NumPy vectorization; 100-1000x with JAX/GPU
2. **Extensibility (primary):** Custom processes are plain functions, not class hierarchies
3. **Java elimination:** No JVM dependency for Python-only users
4. **Ecosystem integration:** In-process engine enables direct array access from calibration/ML

## Architecture

### Engine Protocol

Common interface for Java and Python engines:

```python
class Engine(Protocol):
    def run(self, config: dict[str, str], output_dir: Path, seed: int = 0) -> RunResult: ...
    def run_ensemble(self, config: dict[str, str], output_dir: Path, n: int, base_seed: int = 0) -> list[RunResult]: ...
```

- `JavaEngine` wraps existing `OsmoseRunner` (subprocess to JAR)
- `PythonEngine` runs vectorized simulation in-process
- UI exposes engine selector dropdown
- `OsmoseCalibrationProblem` accepts either engine via the protocol
- **Seed management:** `run_ensemble` assigns seed `base_seed + i` to replicate `i` for reproducibility

### SchoolState (Structure-of-Arrays)

All school data in flat NumPy arrays instead of Java's per-object `School` instances:

```python
@dataclass
class SchoolState:
    # Identity
    species_id: NDArray[np.int32]           # (n_schools,)
    is_background: NDArray[np.bool_]        # True for background species schools

    # Demographics
    abundance: NDArray[np.float64]
    biomass: NDArray[np.float64]            # tracked separately from abundance * weight
    length: NDArray[np.float64]
    length_start: NDArray[np.float64]       # length at start of time step (for growth calc)
    weight: NDArray[np.float64]
    age_dt: NDArray[np.int32]               # age in time steps (integer)
    trophic_level: NDArray[np.float64]

    # Spatial
    cell_x: NDArray[np.int32]
    cell_y: NDArray[np.int32]
    is_out: NDArray[np.bool_]               # out-of-domain flag

    # Feeding / predation
    pred_success_rate: NDArray[np.float64]
    preyed_biomass: NDArray[np.float64]     # total biomass preyed this step
    feeding_stage: NDArray[np.int32]        # current feeding stage index

    # Reproduction
    gonad_weight: NDArray[np.float64]

    # Mortality tracking
    starvation_rate: NDArray[np.float64]    # lagged from PREVIOUS step
    n_dead: NDArray[np.float64]             # (n_schools, N_MORTALITY_CAUSES)

    # Egg state
    is_egg: NDArray[np.bool_]               # True for age_dt < first_feeding_age_dt (NOT just age 0)
    first_feeding_age_dt: NDArray[np.int32] # age (in dt) when feeding begins
```

**Mortality causes** tracked as enum indices: PREDATION, STARVATION, ADDITIONAL, FISHING, OUT, FORAGING, DISCARDS, AGING.

Schools are born (appended), die (masked), and periodically compacted.

### Process Functions

Each OSMOSE process is a pure function:

```python
def growth(state: SchoolState, config: EngineConfig, rng: Generator) -> SchoolState: ...
def mortality(state: SchoolState, resources: ResourceState, config: EngineConfig, rng: Generator) -> SchoolState: ...
def movement(state: SchoolState, grid: Grid, config: EngineConfig, rng: Generator) -> SchoolState: ...
def reproduction(state: SchoolState, config: EngineConfig, rng: Generator) -> SchoolState: ...
```

Extensibility: adding a process = writing one function with the same signature. No framework, no class hierarchy.

### Simulation Loop

Matches Java's `SimulationStep.step()` ordering:

```python
def simulate(config, grid, n_steps, rng) -> list[StepOutput]:
    state = initialize(config, grid, rng)
    resources = ResourceState(config, grid)
    outputs = []
    for step in range(n_steps):
        state = incoming_flux(state, config, step, rng)
        state = reset_step_variables(state)              # reset nDead, predSuccessRate, preyedBiomass
        resources.update(step)                            # load LTL biomass for current timestep
        state = movement(state, grid, config, step, rng)
        state = mortality(state, resources, config, rng)  # ALL sources interleaved (see below)
        state = growth(state, config, rng)
        state = aging_mortality(state, config)            # kill schools exceeding species lifespan
        state = reproduction(state, config, step, rng)    # also increments age_dt for ALL schools
        outputs.append(collect_outputs(state, config))
        state = compact(state)                            # remove dead schools, merge new
    return outputs
```

### Reset Step Variables

At the start of each timestep (matching Java's `School.init()`):

```python
def reset_step_variables(state: SchoolState) -> SchoolState:
    return state.replace(
        abundance=state.abundance.copy(),       # snapshot for instantaneous tracking
        n_dead=np.zeros_like(state.n_dead),     # clear mortality counters
        pred_success_rate=np.zeros(len(state)),  # clear predation success
        preyed_biomass=np.zeros(len(state)),     # clear diet tracking
        length_start=state.length.copy(),        # snapshot for growth calculation
    )
```

## Mortality System

**Critical design point:** In Java, all mortality sources (predation, fishing, starvation, additional) are NOT separate sequential processes. They are interleaved within a single stochastic double-loop: the outer loop iterates over "school slots" (one per school), and for each slot, the four mortality causes are reshuffled and applied — but each cause targets a **different** school from its own pre-shuffled ordering. This ensures maximum stochasticity. The `mortality()` function orchestrates all sources:

```python
def mortality(state: SchoolState, resources: ResourceState, config: EngineConfig, rng: Generator) -> SchoolState:
    n_subdt = config.mortality_subdt

    # Pre-pass: larva mortality on eggs (before main loop)
    state = larva_mortality(state, config, n_subdt)
    state = retain_eggs(state)  # exclude eggs from predation initially

    n_schools = len(state)
    for sub in range(n_subdt):
        # Release fraction of eggs into the prey pool
        state = release_eggs(state, sub, n_subdt)

        # Each mortality cause gets its own independent random school ordering
        seq_pred = rng.permutation(n_schools)
        seq_fish = rng.permutation(n_schools)
        seq_starv = rng.permutation(n_schools)
        seq_nat = rng.permutation(n_schools)

        # Process one school-slot at a time; causes reshuffled per slot
        for i in range(n_schools):
            causes = rng.permutation(['predation', 'fishing', 'starvation', 'additional'])
            for cause in causes:
                school_idx = {'predation': seq_pred, 'fishing': seq_fish,
                              'starvation': seq_starv, 'additional': seq_nat}[cause][i]
                state = apply_mortality_to_school(state, resources, school_idx, cause, config, rng, n_subdt)

    # Out-of-domain mortality (after main loop)
    state = out_mortality(state, config)

    # Compute new starvation rate from THIS step's predation success
    # (will be APPLIED in the NEXT step — lagged variable)
    state = update_starvation_rate(state, config)

    return state
```

### Egg Handling

Eggs have special treatment in the mortality system:

- **Retain:** At the start of mortality, eggs (`age_dt < first_feeding_age_dt`) are excluded from the prey pool
- **Larva mortality:** Eggs (`age_dt < first_feeding_age_dt`) receive separate additional mortality (`mortality.additional.larva.rate`) before the main stochastic loop; they are also excluded from regular additional mortality in the stochastic loop
- **Progressive release:** Each sub-timestep, `1/n_subdt` of surviving eggs are "released" into the prey pool, becoming eligible to be preyed upon
- **No feeding:** Schools with `age_dt < first_feeding_age_dt` cannot predate (skip predation as predators)

### Starvation (Lagged Variable)

Starvation mortality applied in the current step uses the starvation rate computed at the END of the previous step:

```
M_starv(t) = M_max * (1 - S_R(t-1) / C_SR)    if S_R(t-1) <= C_SR
M_starv(t) = 0                                   if S_R(t-1) > C_SR
```

The `starvation_rate` field on SchoolState stores this lagged value. At the end of each step's mortality phase, the new starvation rate is computed from the current step's `pred_success_rate` and stored for use in the next step.

### Aging Mortality

Schools exceeding their species' lifespan are killed entirely. Runs between growth and reproduction:

```python
def aging_mortality(state: SchoolState, config: EngineConfig) -> SchoolState:
    lifespan_dt = config.lifespan_dt[state.species_id]  # lifespan in time steps
    # Java uses ageDt > lifespanDt - 2 (i.e., >= lifespanDt - 1) because aging
    # runs BEFORE reproduction where age_dt is incremented. The school would
    # reach/exceed lifespan after the upcoming age increment.
    expired = state.age_dt >= lifespan_dt - 1
    state.n_dead[expired, AGING] += state.abundance[expired]
    state.abundance[expired] = 0
    return state
```

### Out-of-Domain Mortality

Schools flagged as `is_out` (outside simulated area after movement) receive separate mortality and are excluded from all other processes:

```
N_out = N * (1 - exp(-M_out / n_subdt))
```

Parameter: `mortality.out.rate.sp#`

## Vectorized Predation Algorithm

The performance-critical centerpiece of OSMOSE.

### Cell-Based Grouping

```python
cell_ids = state.cell_x * ny + state.cell_y
order = np.argsort(cell_ids)
boundaries = np.searchsorted(cell_ids[order], range(n_cells + 1))
```

O(1) lookup for all schools in any cell. **Vectorization is across cells** (which are independent), not across predators within a cell.

### Within-Cell Sequential Predation

**Critical:** Java processes predators one at a time in random order, with asynchronous biomass updates — prey biomass is decremented immediately after each predator eats. This means early predators reduce prey available to later predators within the same sub-timestep.

The vectorization opportunity is parallelizing across independent cells, NOT batching all predator-prey interactions in a cell simultaneously.

```python
def predation_in_cell(local_state, resources_in_cell, config, rng, n_subdt):
    # Predator processing order is randomized
    predator_order = rng.permutation(n_local)

    for p_idx in predator_order:
        # Skip eggs and non-feeding schools
        if local_state.age_dt[p_idx] < local_state.first_feeding_age_dt[p_idx]:
            continue
        # Skip self-predation (explicit exclusion)
        prey_mask = np.arange(n_local) != p_idx

        pred_len = local_state.length[p_idx]
        prey_lens = local_state.length[prey_mask]
        prey_bio = local_state.biomass[prey_mask]

        # Size-ratio eligibility (indexed by species AND feeding stage)
        sp_pred = local_state.species_id[p_idx]
        stage_pred = local_state.feeding_stage[p_idx]
        r_min = config.size_ratio_min[sp_pred, stage_pred]
        r_max = config.size_ratio_max[sp_pred, stage_pred]
        ratios = pred_len / prey_lens
        eligible = (ratios > r_max) & (ratios <= r_min)

        # Accessibility coefficients
        sp_prey = local_state.species_id[prey_mask]
        access = config.accessibility[sp_pred, stage_pred, sp_prey]

        # Resource species: partial accessibility based on size overlap
        resource_mask = local_state.is_background[prey_mask] | is_resource[prey_mask]
        # ... percentResource computation for resource prey ...

        # Available biomass
        available = (eligible * access * prey_bio).sum()
        if available <= 0:
            continue  # no prey — predation success stays 0

        # Ingestion-limited consumption
        max_eatable = local_state.biomass[p_idx] * config.ingestion_rate[sp_pred] / n_subdt
        eaten_total = min(available, max_eatable)

        # Proportional allocation to each prey
        prey_share = eligible * access * prey_bio / available
        eaten_per_prey = eaten_total * prey_share

        # Update prey biomass IMMEDIATELY (asynchronous update)
        local_state.biomass[prey_mask] -= eaten_per_prey
        local_state.abundance[prey_mask] -= eaten_per_prey / local_state.weight[prey_mask]

        # Update predator tracking
        local_state.pred_success_rate[p_idx] += eaten_total / max_eatable
        local_state.preyed_biomass[p_idx] += eaten_total
```

**Parallelism strategy:** Cells are independent — process all occupied cells in parallel via thread pool or vectorized cell dispatch. The per-predator loop within a cell is sequential (typically 10-50 schools per cell).

### Resource Species as Prey

Resource species (LTL plankton/benthos) participate in predation as prey with special handling:

- They have a **size range** (`[L_min, L_max]`), not a single length
- **Partial accessibility** via `percentResource`: the fraction of the resource's size range that falls within the predator's prey window:

```
percentResource = (min(L_max_resource, pred_len/r_max) - max(L_min_resource, pred_len/r_min))
                  / (L_max_resource - L_min_resource)
```

- Resource biomass per cell is updated from NetCDF forcing each timestep via `ResourceState.update(step)`
- Predation reduces resource biomass in-cell; it regenerates from forcing at the next step

### Background Species

Background species participate as **both predators and prey**:

- They are included in the predator processing loop alongside focal species
- They can predate on focal species, background species, and resources
- Their life cycle is not simulated — biomass is provided as forcing input
- They are represented as schools in `SchoolState` with `is_background=True`

## Growth Algorithm

Von Bertalanffy with young-of-year linear phase, gated by predation success:

```
delta_L(a) = L_expected(a+1) - L_expected(a)         mean length increment from growth function
max_delta = delta_lmax_factor * delta_L(a)            maximum possible growth (default factor=2.0)
min_delta = 0                                          minimum growth when barely above threshold

G(s,a) = min_delta + (max_delta - min_delta) * (S_R - C_SR) / (1 - C_SR)    if S_R >= C_SR
G(s,a) = 0                                                                    if S_R < C_SR

new_length = min(length + G(s,a), L_max)              capped at species maximum length
```

**Special cases:**
- Eggs (`age_dt == 0`) always get linear growth (`delta_L`), bypassing predation success gating
- Unlocated/out-of-domain schools always get linear growth
- `delta_lmax_factor` parameter: `species.delta.lmax.factor.sp#` (default 2.0)

```
L_expected(a) = L_egg                                                    if a = 0
L_expected(a) = L_egg + (L_thres - L_egg) * a/a_thres                   if 0 < a < a_thres
L_expected(a) = L_inf * (1 - exp(-K * (a - t0)))                         if a >= a_thres
```

Where `L_thres = L_inf * (1 - exp(-K * (a_thres - t0)))`.

All species processed in one vectorized pass via species-indexed parameter arrays.

Gompertz growth supported as alternative (four-phase: egg, exponential, linear transition, Gompertz curve).

## Reproduction

At the end of each step, reproduction runs AND age is incremented for ALL schools:

```
N_eggs = sex_ratio * relative_fecundity * SSB * season_factor

SSB = sum(abundance * weight) for mature schools per species
mature = (age >= age_maturity) AND (length >= size_maturity)
```

Per-species SSB aggregated via `np.add.at`. New schools created and appended to state.

After egg creation, **all schools have their age incremented** by one time step (`age_dt += 1`). This is a side effect of reproduction in Java that must be preserved.

## Movement

**Map-based (default):** NetCDF probability maps per species/age/season. Vectorized weighted random sampling for all schools of a species simultaneously.

**Random walk:** Cell offset by random integer within configured range. Pure array addition.

Schools that land outside the grid after movement are flagged `is_out=True` and receive out-of-domain mortality instead of normal processes.

## Output System

The Python engine writes CSV files matching Java's naming convention:

- `osm_biomass_<Species>.csv`
- `osm_abundance_<Species>.csv`
- `osm_yield_<Species>.csv`
- `osm_dietMatrix.csv`
- `osm_mortalityRate_<Species>.csv` (mortality by cause — PREDATION, STARVATION, FISHING, ADDITIONAL, OUT, AGING)
- etc.

This means the existing `OsmoseResults` reader, UI charts, calibration objectives, and ensemble statistics work unchanged with either engine.

## Integration Points

| Component | Change Required |
|-----------|----------------|
| `runner.py` | Add `PythonEngine` alongside `OsmoseRunner` |
| `calibration/problem.py` | Accept `Engine` protocol instead of hardcoded Java |
| `ui/pages/run_page.py` | Engine selector dropdown |
| `results.py` | None — reads same output format |
| `scenarios.py` | None — stores config dicts |
| `ensemble.py` | None — aggregates output files |

## NumPy-to-JAX Backend Path

The `xp` pattern keeps the engine backend-agnostic:

```python
def growth(state, config, rng, xp=np):
    l_expected = config.linf[state.species_id] * (1 - xp.exp(-config.k[state.species_id] * ...))
    ...
```

Configuration: `engine.backend = numpy` (default) or `engine.backend = jax` (GPU).

### JAX Dynamic Array Challenge

JAX requires static array shapes for JIT compilation, but OSMOSE has dynamic school counts (reproduction creates, death removes). Strategy:

- **Fixed-capacity buffer:** Allocate `SchoolState` arrays with `max_schools` capacity (configurable, e.g., 2x expected peak). Active schools tracked via `alive` boolean mask.
- **Dead schools:** Set `alive[i] = False` instead of removing. Compaction (defragmenting the buffer) runs periodically outside JIT.
- **New schools:** Append into dead slots. If buffer full, trigger compaction + resize outside JIT.
- **Per-cell padding:** For `jax.vmap` over cells, pad each cell's school list to `max_schools_per_cell`. Inactive slots masked out of computations.
- **Performance implication:** Over-padding wastes compute on masked slots. Profile to tune `max_schools` and `max_schools_per_cell` for each study.

JAX specifics:
- Cell grouping: `jax.numpy.argsort`
- Per-cell predation: padded batches via `jax.vmap` with masking
- Sub-timestep loop: `jax.lax.fori_loop` for JIT compilation
- Growth/reproduction: standard `jax.numpy` operations on masked arrays

## Validation Strategy

### Tier 1: Unit-Level Analytical Verification

Each process function tested against known mathematical solutions:

| Test | Method | Tolerance |
|------|--------|-----------|
| Von Bertalanffy growth | Verify L(age) against formula | 1e-10 |
| Growth gating | Verify G=0 when S_R < C_SR, G=max_delta when S_R=1 | 1e-10 |
| Weight-length conversion | Verify W = c * L^b | 1e-10 |
| Predation eligibility | Hand-calculated 3-school scenario, self-exclusion | Exact |
| Predation asynchronous | 2 predators eating same prey: order matters | Exact |
| Mortality decay | N * (1 - exp(-F)) | 1e-10 |
| Starvation lag | Rate from step t-1 applied at step t | Exact |
| Larva mortality | Eggs receive separate rate before main loop | 1e-10 |
| Reproduction egg count | SSB * fecundity * season | 1e-10 |
| Age increment | All schools aged after reproduction | Exact |
| Aging mortality | Schools at lifespan killed | Exact |
| Movement distribution | Convergence to map probabilities | Chi-squared p > 0.05 |
| Resource predation | percentResource size overlap | 1e-10 |

### Tier 1.5: Deterministic Process Isolation

Test individual processes with fixed configurations and known seeds, comparing Python vs Java output for the exact same scenario (not ensemble — single deterministic run with controlled RNG):

| Test | Method |
|------|--------|
| Growth only | Disable predation/fishing, verify length trajectory matches Java |
| Mortality only | Fixed schools, verify survivor counts per cause match Java |
| Movement only | Fixed map, verify spatial distribution matches Java |

### Tier 2: Single-Species Benchmarks

1 focal species + 1 resource, 50-year run, 100 replicates per engine.

| Metric | Tolerance |
|--------|-----------|
| Mean biomass trajectory | KS test p > 0.05 per decade |
| Mean size-at-age | Within 5% of Java ensemble mean |
| Recruitment variability | CV within 20% of Java CV |
| Biomass equilibrium | Final-decade mean within 10% |
| Mortality-by-cause proportions | Within 5% of Java per cause |

### Tier 3: Full Ecosystem Validation

2-3 example studies, full species complement, 100-year runs, 50+ replicates.

| Metric | Tolerance |
|--------|-----------|
| Species biomass ranking | Same rank order in final decade |
| Diet matrix distance | Frobenius norm / n_species² < 0.1 |
| Size spectrum slope | Within 0.2 of Java (log-log) |
| Shannon diversity | Within 10% of Java mean |
| Mean trophic level | Within 0.3 per species |
| Extinction events | Same species persist/go extinct |
| Mortality-by-cause proportions | Within 10% per species per cause |

### Automated Validation Pipeline

`scripts/validate_engines.py` runs both engines on example studies and produces HTML comparison reports. Integrated as CI gate.

## Implementation Roadmap

### Phase 1: Foundation
- `Engine` protocol, `PythonEngine` skeleton
- `SchoolState` dataclass with full field set (all fields above)
- `Grid` class (NetCDF loading, cell adjacency, land mask)
- `EngineConfig` from flat config dict (reuses existing schema)
- `ResourceState` class (LTL forcing loader)
- `simulate()` loop skeleton with correct process ordering and stub functions
- `reset_step_variables()` implementation
- **Validation:** Tier 1 — grid, state creation/compaction, config parsing

### Phase 2: Growth + Natural Mortality + Aging
- Von Bertalanffy growth (with linear young-of-year phase, delta_lmax_factor, L_max cap)
- Gompertz growth (alternative)
- Weight-length allometric conversion
- Natural (additional) mortality
- Aging mortality (lifespan enforcement)
- **Validation:** Tier 1 analytical + Tier 1.5 growth-only comparison with Java

### Phase 3: Reproduction + Initialization
- Egg production from SSB with spawning seasonality
- School creation, state compaction
- Egg state tracking (is_egg, first_feeding_age_dt)
- Larva mortality (separate rate for eggs)
- Age increment for all schools
- Population seeding (spin-up mode)
- **Validation:** Tier 1 + Tier 2 partial — 1 species equilibrium without predation

### Phase 4: Movement
- Map-based distribution (NetCDF probability maps, weighted sampling)
- Random walk
- Age/season-based map switching
- Out-of-domain flagging
- Out-of-domain mortality
- Incoming flux process
- **Validation:** Tier 1 + Tier 1.5 movement-only + Tier 2 partial spatial distribution

### Phase 5: Predation (critical phase)
- Cell-based school grouping (argsort + searchsorted)
- Sequential within-cell predation with randomized predator order
- Size-ratio eligibility indexed by species AND feeding stage
- Accessibility matrix loading
- Ingestion rate, proportional prey allocation
- Asynchronous prey biomass update (immediate decrement)
- Self-predation exclusion
- Egg retain/release mechanism
- Resource species as prey (percentResource size overlap)
- Predation success rate tracking
- Trophic level computation from diet
- Growth gating by predation success (connect to Phase 2 growth)
- Mortality sub-timestep loop with randomized source ordering
- **Validation:** Tier 1 (including asynchronous predation test) + Tier 2 full — single species + resource

### Phase 6: Fishing + Starvation
- Fishing mortality (by-rate and by-catches modes)
- Size/age selectivity (knife-edge, sigmoid, Gaussian, log-normal)
- Starvation mortality (lagged: uses previous step's predation success rate)
- Fishery seasonality and spatial effort maps
- **Note:** OSMOSE 4 fishery system (FishingGear, gear-level catchability, per-gear discards, economic module) is deferred to a future phase
- **Validation:** Tier 1.5 + Tier 2 complete — all mortality sources active, mortality-by-cause comparison

### Phase 7: Multi-Species + Full Validation
- Multi-species predation interactions
- Background species as both predators AND prey
- Resource species (LTL forcing from NetCDF, per-cell biomass)
- Output writer (CSV files matching Java naming, including mortality-by-cause)
- Automated validation pipeline (`scripts/validate_engines.py`)
- **Validation:** Tier 3 — full ecosystem runs against example studies

### Phase 8: Performance + JAX Backend
- Profile and optimize bottlenecks (predation will dominate)
- Fixed-capacity buffer with alive mask for JAX JIT compatibility
- JAX backend via `xp` pattern
- `jax.vmap` padded batched cell predation for GPU
- `jax.lax.fori_loop` for JIT sub-timesteps
- Benchmarks: Java vs NumPy vs JAX across study sizes
- **Validation:** All tiers re-run with JAX backend, same tolerances

## File Structure

New files under `osmose/engine/`:

```
osmose/engine/
    __init__.py          # Engine protocol, PythonEngine, JavaEngine
    state.py             # SchoolState dataclass
    config.py            # EngineConfig (typed params from flat dict)
    grid.py              # Grid class (NetCDF, adjacency)
    resources.py         # ResourceState (LTL forcing)
    simulate.py          # Main simulation loop
    processes/
        __init__.py
        growth.py        # Von Bertalanffy, Gompertz
        predation.py     # Sequential within-cell, parallel across cells
        fishing.py       # Fishing mortality + selectivity
        starvation.py    # Starvation mortality (lagged)
        mortality.py     # Orchestrator: sub-timestep interleaving of all sources
        natural.py       # Additional + aging + larva + out-of-domain mortality
        movement.py      # Map-based + random walk
        reproduction.py  # Egg production + school creation + age increment
    output.py            # StepOutput collection + CSV writer (matching Java format)
    backend.py           # NumPy/JAX backend abstraction
tests/
    test_engine_state.py
    test_engine_grid.py
    test_engine_growth.py
    test_engine_predation.py
    test_engine_fishing.py
    test_engine_mortality.py
    test_engine_movement.py
    test_engine_reproduction.py
    test_engine_integration.py
    test_engine_validation.py   # Tier 1.5, 2, 3 Java comparison
scripts/
    validate_engines.py          # Automated Java vs Python comparison
```

## Non-Goals

- **Bit-exact Java reproduction:** We target behavioral equivalence, not identical RNG sequences
- **Bioen-OSMOSE / Ev-OSMOSE:** Extensions are future work after core engine validation
- **OSMOSE 4 fishery system:** Gear-level fisheries, economic module deferred to post-Phase 8
- **Replacing Java for existing users:** Both engines coexist; Java remains the reference implementation
- **Real-time simulation:** Target is batch throughput, not interactive stepping
