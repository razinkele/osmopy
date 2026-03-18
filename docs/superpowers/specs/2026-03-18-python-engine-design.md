# Python OSMOSE Engine — Design Specification

**Date:** 2026-03-18
**Status:** Approved
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
    def run_ensemble(self, config: dict[str, str], output_dir: Path, n: int) -> list[RunResult]: ...
```

- `JavaEngine` wraps existing `OsmoseRunner` (subprocess to JAR)
- `PythonEngine` runs vectorized simulation in-process
- UI exposes engine selector dropdown
- `OsmoseCalibrationProblem` accepts either engine via the protocol

### SchoolState (Structure-of-Arrays)

All school data in flat NumPy arrays instead of Java's per-object `School` instances:

```python
@dataclass
class SchoolState:
    species_id: NDArray[np.int32]       # (n_schools,)
    abundance: NDArray[np.float64]
    length: NDArray[np.float64]
    weight: NDArray[np.float64]
    age: NDArray[np.float64]
    trophic_level: NDArray[np.float64]
    cell_x: NDArray[np.int32]
    cell_y: NDArray[np.int32]
    pred_success_rate: NDArray[np.float64]
    gonad_weight: NDArray[np.float64]
```

Schools are born (appended), die (masked), and periodically compacted.

### Process Functions

Each OSMOSE process is a pure function:

```python
def growth(state: SchoolState, config: EngineConfig, rng: Generator) -> SchoolState: ...
def predation(state: SchoolState, config: EngineConfig, rng: Generator) -> SchoolState: ...
def fishing(state: SchoolState, config: EngineConfig, rng: Generator) -> SchoolState: ...
def movement(state: SchoolState, grid: Grid, config: EngineConfig, rng: Generator) -> SchoolState: ...
def reproduction(state: SchoolState, config: EngineConfig, rng: Generator) -> SchoolState: ...
```

Extensibility: adding a process = writing one function with the same signature. No framework, no class hierarchy.

### Simulation Loop

```python
def simulate(config, grid, n_steps, rng) -> list[StepOutput]:
    state = initialize(config, grid, rng)
    for step in range(n_steps):
        state = movement(state, grid, config, rng)
        state = predation(state, config, rng)
        state = starvation(state, config, rng)
        state = fishing(state, config, rng)
        state = natural_mortality(state, config, rng)
        state = growth(state, config, rng)
        state = reproduction(state, config, rng)
        outputs.append(collect_outputs(state, config))
        state = compact(state)
    return outputs
```

## Vectorized Predation Algorithm

The performance-critical centerpiece of OSMOSE.

### Cell-Based Grouping

```python
cell_ids = state.cell_x * ny + state.cell_y
order = np.argsort(cell_ids)
boundaries = np.searchsorted(cell_ids[order], range(n_cells + 1))
```

O(1) lookup for all schools in any cell, no Python loops.

### Within-Cell Interaction Matrix

```python
# For n_local schools in a cell
ratio = pred_lengths[:, None] / prey_lengths[None, :]        # (n_local, n_local)
eligible = (ratio > r_min[sp_pred, sp_prey]) & (ratio <= r_max[sp_pred, sp_prey])
access = accessibility_matrix[sp_pred[:, None], sp_prey[None, :]]

available = (eligible * access * prey_biomass[None, :]).sum(axis=1)
max_eatable = pred_biomass * ingestion_rate[sp_pred] / n_subdt
eaten_total = np.minimum(available, max_eatable)

prey_share = eligible * access * prey_biomass[None, :] / available[:, None]
eaten_matrix = eaten_total[:, None] * prey_share
```

### Mortality Sub-Timesteps

```python
for sub in range(n_subdt):
    mortality_order = rng.permutation(['predation', 'fishing', 'starvation', 'natural'])
    for source in mortality_order:
        state = MORTALITY_FNS[source](state, config, rng, n_subdt)
```

Preserves Java's stochastic ordering of mortality sources.

## Growth Algorithm

Von Bertalanffy with young-of-year linear phase, gated by predation success:

```
G(s,a) = lambda * delta_L(a) * (S_R - C_SR) / (1 - C_SR)    if S_R >= C_SR
G(s,a) = 0                                                     if S_R < C_SR

L(a) = L_egg + (L_thres - L_egg) * a/a_thres                  if a < a_thres
L(a) = L_inf * (1 - exp(-K * (a - t0)))                        if a >= a_thres
```

All species processed in one vectorized pass via species-indexed parameter arrays.

Gompertz growth supported as alternative (four-phase: egg, exponential, linear transition, Gompertz curve).

## Reproduction

```
N_eggs = sex_ratio * relative_fecundity * SSB * season_factor

SSB = sum(abundance * weight) for mature schools per species
mature = (age >= age_maturity) AND (length >= size_maturity)
```

Per-species SSB aggregated via `np.add.at`. New schools created and appended to state.

## Movement

**Map-based (default):** NetCDF probability maps per species/age/season. Vectorized weighted random sampling for all schools of a species simultaneously.

**Random walk:** Cell offset by random integer within configured range. Pure array addition.

## Output System

The Python engine writes CSV files matching Java's naming convention:

- `osm_biomass_<Species>.csv`
- `osm_abundance_<Species>.csv`
- `osm_yield_<Species>.csv`
- `osm_dietMatrix.csv`
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
def predation(state, config, rng, xp=np):
    ratio = xp.outer(state.length, 1.0 / state.length)
    ...
```

Configuration: `engine.backend = numpy` (default) or `engine.backend = jax` (GPU).

JAX specifics:
- Cell grouping: `jax.numpy.argsort`
- Per-cell predation: padded batches via `jax.vmap`
- Sub-timestep loop: `jax.lax.fori_loop` for JIT compilation

## Validation Strategy

### Tier 1: Unit-Level Analytical Verification

Each process function tested against known mathematical solutions:

| Test | Method | Tolerance |
|------|--------|-----------|
| Von Bertalanffy growth | Verify L(age) against formula | 1e-10 |
| Weight-length conversion | Verify W = c * L^b | 1e-10 |
| Predation eligibility | Hand-calculated 3-school scenario | Exact |
| Mortality decay | N * (1 - exp(-F)) | 1e-10 |
| Reproduction egg count | SSB * fecundity * season | 1e-10 |
| Movement distribution | Convergence to map probabilities | Chi-squared p > 0.05 |

### Tier 2: Single-Species Benchmarks

1 focal species + 1 resource, 50-year run, 100 replicates per engine.

| Metric | Tolerance |
|--------|-----------|
| Mean biomass trajectory | KS test p > 0.05 per decade |
| Mean size-at-age | Within 5% of Java ensemble mean |
| Recruitment variability | CV within 20% of Java CV |
| Biomass equilibrium | Final-decade mean within 10% |

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

### Automated Validation Pipeline

`scripts/validate_engines.py` runs both engines on example studies and produces HTML comparison reports. Integrated as CI gate.

## Implementation Roadmap

### Phase 1: Foundation
- `Engine` protocol, `PythonEngine` skeleton
- `SchoolState` dataclass with NumPy arrays
- `Grid` class (NetCDF loading, cell adjacency, land mask)
- `EngineConfig` from flat config dict (reuses existing schema)
- `simulate()` loop skeleton with stub process functions
- **Validation:** Tier 1 — grid, state, config

### Phase 2: Growth + Natural Mortality
- Von Bertalanffy growth (with linear young-of-year phase)
- Gompertz growth (alternative)
- Weight-length allometric conversion
- Natural (additional) mortality
- **Validation:** Tier 1 analytical + Java decay curve comparison

### Phase 3: Reproduction + Initialization
- Egg production from SSB with spawning seasonality
- School creation, state compaction
- Population seeding (spin-up mode)
- **Validation:** Tier 2 partial — 1 species equilibrium without predation

### Phase 4: Movement
- Map-based distribution (NetCDF probability maps, weighted sampling)
- Random walk
- Age/season-based map switching
- **Validation:** Tier 2 partial — spatial distribution vs Java

### Phase 5: Predation (critical phase)
- Cell-based grouping (argsort + searchsorted)
- Size-ratio eligibility matrix
- Accessibility matrix, ingestion rate, proportional prey allocation
- Predation success rate, trophic level computation
- Growth gating by predation success
- Mortality sub-timestep loop with randomized ordering
- **Validation:** Tier 2 full — single species + resource, biomass + diet + size-at-age

### Phase 6: Fishing + Starvation
- Fishing mortality (by-rate and by-catches)
- Size/age selectivity (knife-edge, sigmoid, Gaussian, log-normal)
- Starvation mortality (gated by previous step predation success)
- Fishery seasonality and spatial effort maps
- **Validation:** Tier 2 complete — all mortality sources active

### Phase 7: Multi-Species + Full Validation
- Multi-species predation interactions
- Background species (biomass forcing)
- Resource species (LTL forcing from NetCDF)
- Incoming flux (migration)
- Output writer (CSV matching Java naming)
- Automated validation pipeline
- **Validation:** Tier 3 — full ecosystem runs against example studies

### Phase 8: Performance + JAX Backend
- Profile and optimize bottlenecks
- JAX backend via `xp` pattern
- `jax.vmap` batched predation for GPU
- `jax.lax.fori_loop` for JIT sub-timesteps
- Benchmarks: Java vs NumPy vs JAX
- **Validation:** All tiers re-run with JAX backend

## File Structure

New files under `osmose/engine/`:

```
osmose/engine/
    __init__.py          # Engine protocol, PythonEngine, JavaEngine
    state.py             # SchoolState dataclass
    config.py            # EngineConfig (typed params from flat dict)
    grid.py              # Grid class (NetCDF, adjacency)
    simulate.py          # Main simulation loop
    processes/
        __init__.py
        growth.py        # Von Bertalanffy, Gompertz
        predation.py     # Vectorized size-based predation
        fishing.py       # Fishing mortality + selectivity
        starvation.py    # Starvation mortality
        mortality.py     # Natural mortality + sub-timestep orchestration
        movement.py      # Map-based + random walk
        reproduction.py  # Egg production + school creation
    output.py            # StepOutput collection + CSV writer
    backend.py           # NumPy/JAX backend abstraction
tests/
    test_engine_state.py
    test_engine_grid.py
    test_engine_growth.py
    test_engine_predation.py
    test_engine_fishing.py
    test_engine_movement.py
    test_engine_reproduction.py
    test_engine_integration.py
    test_engine_validation.py   # Tier 2-3 Java comparison
scripts/
    validate_engines.py          # Automated Java vs Python comparison
```

## Non-Goals

- **Bit-exact Java reproduction:** We target behavioral equivalence, not identical RNG sequences
- **Bioen-OSMOSE / Ev-OSMOSE:** Extensions are future work after core engine validation
- **Replacing Java for existing users:** Both engines coexist; Java remains the reference implementation
- **Real-time simulation:** Target is batch throughput, not interactive stepping
