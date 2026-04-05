# Ev-OSMOSE + Bioeconomic Fleet Dynamics Design Specification

> **Status:** Approved design, awaiting implementation plan
> **Date:** 2026-04-05
> **Modules:** Ev-OSMOSE (eco-evolutionary genetics), DSVM Fleet Dynamics (bioeconomic)

## Goal

Extend osmopy with two major modules:

1. **Ev-OSMOSE** — eco-evolutionary genetics with diploid multi-locus inheritance and extensible evolving traits, matching and extending the Java Ev-OSMOSE implementation.
2. **DSVM Fleet Dynamics** — spatially-explicit bioeconomic model where individual vessels make discrete choices about where to fish based on expected profit, following the Bourdaud et al. (2025) approach.

Both modules are optional (toggled via config), backward-compatible, and designed for research, teaching, and management decision support.

## References

- Morell et al. (2023) "Ev-OSMOSE: An eco-genetic marine ecosystem model" — bioRxiv 10.1101/2023.02.08.527669
- Morell et al. (2023) "Bioen-OSMOSE" — Progress in Oceanography
- Bourdaud et al. (2025) "Thirty-year impact of a landing obligation on coupled dynamics ecosystem-fishers" — Canadian J. Fisheries and Aquatic Sciences
- Java source: https://github.com/osmose-model/osmose (`process/genet/`, `eco/`)
- Ev-OSMOSE North Sea config: https://zenodo.org/records/7636112

## Architecture Overview

### Integration into Simulation Loop

```
Existing:
  movement → mortality → growth/bioen → reproduction → collect_outputs → compact

With Ev-OSMOSE + Economic:
  fleet_decision (NEW) → movement → mortality(with fleet effort) → growth/bioen(with trait overrides) →
  reproduction(with inheritance) → evolution_bookkeeping (NEW) → collect_outputs → compact
```

### SimulationContext Extensions

`SimulationContext` gains two optional fields:

```python
@dataclass
class SimulationContext:
    # ... existing fields ...
    genetic_state: GeneticState | None = None
    fleet_state: FleetState | None = None
```

Both are `None` when their module is disabled. All existing code paths check `if ctx.genetic_state is not None` before accessing genetics, and likewise for fleet state.

### New File Structure

```
osmose/engine/genetics/          # Ev-OSMOSE
  __init__.py
  trait.py                       # Trait, TraitRegistry, config parsing
  genotype.py                    # GeneticState (allele arrays, env noise, neutral loci)
  inheritance.py                 # gamete formation, parental selection, offspring genotype
  expression.py                  # genotype→phenotype, trait override application

osmose/engine/economics/         # Fleet dynamics
  __init__.py
  fleet.py                       # FleetConfig, FleetState, config parsing
  choice.py                      # DSVM discrete choice (logit model)
  costs.py                       # harvesting costs, fuel costs, stock-effort relationship
  policy.py                      # quotas, MPAs, landing obligations, seasonal closures

tests/test_genetics_trait.py
tests/test_genetics_inheritance.py
tests/test_genetics_expression.py
tests/test_genetics_integration.py
tests/test_economics_fleet.py
tests/test_economics_choice.py
tests/test_economics_costs.py
tests/test_economics_policy.py
tests/test_economics_integration.py
```

### Enabling Modules

```properties
# Ev-OSMOSE
simulation.genetic.enabled ; true
population.genotype.transmission.year.start ; 5

# Economic
simulation.economic.enabled ; true
simulation.economic.rationality ; 1.0
simulation.economic.memory.decay ; 0.7
```

---

## Module 1: Ev-OSMOSE (Eco-Evolutionary Genetics)

### Overview

Adds diploid polygenic genetics to the individual-based model. Each school carries a genotype encoding evolving traits. Traits are expressed as phenotypes that override species-level config parameters, making growth, maturation, and reproduction emergent from both genetics and environment. Selection is implicit — schools that survive and produce more eggs contribute more alleles to the next generation.

### Data Model

#### Trait

Defines one evolving trait. The 4 Java default traits ship with the module; users register additional traits via config.

```python
@dataclass(frozen=True)
class Trait:
    name: str                          # e.g., "imax", "gsi", "m0", "m1"
    species_mean: NDArray[np.float64]  # shape (n_species,) — initial genotypic mean
    species_var: NDArray[np.float64]   # shape (n_species,) — initial additive genetic variance
    env_var: NDArray[np.float64]       # shape (n_species,) — environmental noise variance
    n_loci: NDArray[np.int32]          # shape (n_species,) — loci count per species
    n_alleles: NDArray[np.int32]       # shape (n_species,) — alleles per locus per species
    allele_pool: list[list[NDArray[np.float64]]]  # [species][locus] → array of possible allelic values
    target_param: str                  # EngineConfig field this overrides (e.g., "ingestion_rate")
```

#### TraitRegistry

```python
class TraitRegistry:
    traits: dict[str, Trait]           # name → Trait

    @classmethod
    def from_config(cls, cfg: dict, n_species: int) -> TraitRegistry:
        """Parse traits from config keys: evolution.trait.<name>.<param>.sp<i>"""

    def register(self, trait: Trait) -> None:
        """Add a user-defined trait at runtime."""
```

**Default traits** (registered when `simulation.genetic.enabled = true`):

| Name | `target_param` | Description |
|------|----------------|-------------|
| `imax` | `ingestion_rate` | Maximum mass-specific ingestion rate |
| `gsi` | `gsi_target` | Gonado-somatic index (gonad allocation fraction) |
| `m0` | `maturation_rn_intercept` | Maturation reaction norm intercept |
| `m1` | `maturation_rn_slope` | Maturation reaction norm slope |

**User-defined traits** follow the same config pattern:
```properties
evolution.trait.prey_size_ratio.mean.sp0 ; 0.5
evolution.trait.prey_size_ratio.var.sp0 ; 0.01
evolution.trait.prey_size_ratio.envvar.sp0 ; 0.005
evolution.trait.prey_size_ratio.nlocus.sp0 ; 5
evolution.trait.prey_size_ratio.nval.sp0 ; 20
evolution.trait.prey_size_ratio.target ; size_ratio_max
```

#### GeneticState

Parallel to SchoolState, indexed by school position:

```python
@dataclass
class GeneticState:
    # Per trait: allele arrays — shape (n_schools, max_loci, 2) per trait
    alleles: dict[str, NDArray[np.float64]]

    # Per trait: environmental noise (fixed at birth) — shape (n_schools,) per trait
    env_noise: dict[str, NDArray[np.float64]]

    # Neutral loci (optional) — shape (n_schools, n_neutral_loci, 2)
    neutral_alleles: NDArray[np.int32] | None

    # Reference to trait registry
    registry: TraitRegistry
```

### Allele Pool Initialization

For each trait Z, species s, locus l, the `n_alleles[s]` allelic values are drawn from:

```
A_{Z,l,k} ~ N(0, sigma²_{A,Z}(s) / (2 × n_loci[s]))
```

This ensures that when summed across all loci (2 alleles each), the total genotypic variance equals the prescribed `species_var[s]`.

### Trait Expression (Genotype → Phenotype)

Computed once per timestep before bioenergetics:

```python
def express_traits(genetic_state: GeneticState, state: SchoolState) -> dict[str, NDArray[np.float64]]:
    """Compute phenotypic values for all traits, all schools."""
    phenotypes = {}
    for name, trait in genetic_state.registry.traits.items():
        sp = state.species_id
        # Genotypic value: mean + sum of all alleles
        g = trait.species_mean[sp] + genetic_state.alleles[name].sum(axis=(1, 2))
        # Phenotype = genotype + environmental noise
        phenotypes[name] = g + genetic_state.env_noise[name]
    return phenotypes
```

These phenotypic values then override the corresponding EngineConfig arrays. For example, school `i`'s `ingestion_rate` comes from its Imax phenotype instead of the species-level constant.

The override is applied via a helper that patches config arrays in-place for the current timestep:

```python
def apply_trait_overrides(config: EngineConfig, phenotypes: dict[str, NDArray], state: SchoolState) -> None:
    """Replace species-level config values with per-school phenotypic values."""
    for trait_name, values in phenotypes.items():
        target = genetic_state.registry.traits[trait_name].target_param
        # e.g., config.ingestion_rate becomes per-school instead of per-species
        setattr(config, f"_school_{target}", values)
```

### Gametic Inheritance

At reproduction (when `bioen_egg_production` creates new schools):

**1. Parental selection** — Two parents drawn with probability proportional to egg count:

```python
def select_parents(state: SchoolState, rng: np.random.Generator) -> tuple[int, int]:
    """Fecundity-weighted random draft of two parents."""
    eggs = state.gonad_weight  # proxy for fecundity
    weights = eggs / eggs.sum()
    parent_a, parent_b = rng.choice(len(state), size=2, replace=True, p=weights)
    return parent_a, parent_b
```

**2. Gamete formation** — For each locus, randomly pick one of the two alleles (free recombination):

```python
def form_gamete(parent_alleles: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
    """Meiotic segregation: pick one allele per locus. Shape: (n_loci,)"""
    n_loci = parent_alleles.shape[0]
    picks = rng.integers(0, 2, size=n_loci)
    return parent_alleles[np.arange(n_loci), picks]
```

**3. Fertilization** — Offspring gets one gamete from each parent:

```python
offspring_alleles[:, 0] = form_gamete(parent_a_alleles, rng)
offspring_alleles[:, 1] = form_gamete(parent_b_alleles, rng)
```

**4. Environmental noise** — Drawn at birth, fixed for lifetime:

```python
env_noise[trait] = rng.normal(0, sqrt(trait.env_var[species]), size=n_offspring)
```

**5. Seeding phase** — Before `genotype.transmission.year.start`, offspring get random alleles sampled from the initial allele pool instead of inheriting from parents.

### Sync with SchoolState

`GeneticState` must stay aligned with SchoolState through all operations:

- **append** (reproduction): new rows added to allele and env_noise arrays
- **compact** (dead school removal): same mask applied to genetic arrays
- **No replace needed**: genetic state is separate from SchoolState fields

Sync is managed via hooks in `simulate.py`:

```python
# In _compact():
if ctx.genetic_state is not None:
    ctx.genetic_state = compact_genetic_state(ctx.genetic_state, alive_mask)

# In _reproduction():
if ctx.genetic_state is not None:
    ctx.genetic_state = append_offspring_genotypes(ctx.genetic_state, offspring_genotypes)
```

### Config Keys

```properties
# Master toggle
simulation.genetic.enabled ; true

# Seeding phase: random genotypes before this year, inheritance after
population.genotype.transmission.year.start ; 5

# Neutral loci (optional)
evolution.neutral.nlocus ; 10
evolution.neutral.nval ; 50

# Per-trait, per-species (example for imax on species 0):
evolution.trait.imax.mean.sp0 ; 3.5
evolution.trait.imax.var.sp0 ; 0.1
evolution.trait.imax.envvar.sp0 ; 0.05
evolution.trait.imax.nlocus.sp0 ; 10
evolution.trait.imax.nval.sp0 ; 20
```

---

## Module 2: DSVM Fleet Dynamics

### Overview

A spatially-explicit bioeconomic model where individual fishing vessels make discrete choices about where to fish each timestep. Vessels observe fish abundance, estimate expected profit per grid cell, and choose locations via a multinomial logit model. Aggregated fleet effort feeds back into OSMOSE fishing mortality, creating a two-way ecological-economic coupling.

### Data Model

#### FleetConfig

Per-fleet configuration, parsed from OSMOSE config:

```python
@dataclass(frozen=True)
class FleetConfig:
    name: str
    n_vessels: int
    home_port_y: int                          # grid cell of home port
    home_port_x: int
    gear_type: str                            # maps to selectivity function
    max_days_at_sea: int                      # annual limit
    fuel_cost_per_cell: float                 # travel cost per grid cell distance
    base_operating_cost: float                # fixed daily cost at sea
    stock_elasticity: NDArray[np.float64]     # shape (n_species,)
    target_species: list[int]                 # species indices this fleet targets
    price_per_tonne: NDArray[np.float64]      # shape (n_species,) — ex-vessel price
    selectivity_params: dict                  # gear-specific selectivity parameters
```

#### FleetState

Mutable per-simulation state, lives in `SimulationContext.fleet_state`:

```python
@dataclass
class FleetState:
    fleets: list[FleetConfig]

    # Per-vessel arrays (total_vessels = sum of fleet.n_vessels)
    vessel_fleet: NDArray[np.int32]           # shape (total_vessels,) — fleet index
    vessel_cell_y: NDArray[np.int32]          # shape (total_vessels,) — current location
    vessel_cell_x: NDArray[np.int32]
    vessel_days_used: NDArray[np.int32]       # days at sea this year
    vessel_revenue: NDArray[np.float64]       # cumulative revenue this year
    vessel_costs: NDArray[np.float64]         # cumulative costs this year

    # Effort map: shape (n_fleets, grid_ny, grid_nx)
    effort_map: NDArray[np.float64]

    # Catch memory: shape (n_fleets, grid_ny, grid_nx) — running avg of past catches
    catch_memory: NDArray[np.float64]
    memory_decay: float                       # exponential decay weight (e.g., 0.7)

    # Rationality parameter for logit choice
    rationality: float                        # β in the logit model
```

### DSVM Decision Model

Executed once per timestep, before mortality. Implemented in `choice.py`.

#### Step 1: Expected Revenue Per Cell

For each fleet `f`, for each candidate cell `c`:

```
accessible_biomass(c, sp) = Σ_schools_in_c biomass(school) × selectivity(school, gear_f)

expected_catch(c, sp) = catchability(f, sp) × accessible_biomass(c, sp)
    where catchability(f, sp) = base_rate × (biomass(c, sp) / ref_biomass(sp))^stock_elasticity(sp)

expected_revenue(c) = Σ_sp expected_catch(c, sp) × price(sp)
```

Vessels use a weighted blend of current biomass and catch memory:

```
biomass_estimate(c) = (1 - memory_decay) × observed(c) + memory_decay × catch_memory(c)
```

#### Step 2: Costs Per Cell

```
travel_distance(c) = |current_y - c_y| + |current_x - c_x|    (Manhattan distance)
travel_cost(c) = fuel_cost_per_cell × travel_distance(c)
total_cost(c) = travel_cost(c) + base_operating_cost
```

#### Step 3: Discrete Choice (Multinomial Logit)

Expected profit:
```
V(c) = expected_revenue(c) - total_cost(c)
```

Probability of choosing cell `c`:
```
P(c) = exp(β × V(c)) / Σ_c' exp(β × V(c'))
```

Where `β` (rationality) controls exploration vs. exploitation:
- β → ∞: vessels always pick highest-profit cell (deterministic)
- β → 0: uniform random choice (maximum exploration)
- β = 1.0: default, moderate rationality

The choice set includes a **port option** with `V(port) = 0` (opportunity cost baseline). Vessels at their days-at-sea limit are forced to port.

#### Step 4: Aggregate to Effort Map

After all vessels choose:
```
effort_map[fleet, cell_y, cell_x] = count of fleet's vessels in that cell
```

#### Step 5: Update Memory

After fishing outcomes are observed:
```
catch_memory(c) = memory_decay × catch_memory(c) + (1 - memory_decay) × realized_catch(c)
```

### Integration with Fishing Mortality

The existing `fishing_mortality()` function gains an optional `effort_map` parameter:

```python
def fishing_mortality(state, config, step, effort_map=None, ...):
    if effort_map is not None:
        # Compute fishing rate from fleet effort
        F = compute_effort_based_mortality(state, config, effort_map)
    else:
        # Existing behavior: prescribed rates from config
        F = compute_prescribed_mortality(state, config, step)
```

The effort-based mortality calculation:

```
F(school) = Σ_fleets effort_map[fleet, cell_y, cell_x] × catchability(fleet, sp) × selectivity(fleet, school)
```

This replaces (not adds to) the prescribed fishing rate for species targeted by active fleets. Non-targeted species retain their prescribed rates.

### Policy Levers

Implemented in `policy.py`. Each policy modifies the vessel choice set or fishing outcomes:

| Policy | Config key | Effect |
|--------|-----------|--------|
| **MPA closure** | `economic.mpa.cells.<id>` | Cells removed from choice set (V = -∞) |
| **TAC quota** | `economic.quota.<sp>` | When cumulative catch reaches quota, expected revenue for that species set to 0 |
| **Landing obligation** | `economic.landing.obligation.<sp>` | Discard rate forced to 0; all catch counts against quota |
| **Seasonal closure** | `economic.seasonal.closure.<fleet>` | Fleet forced to port during closure timesteps |
| **Days-at-sea limit** | `economic.max.days.<fleet>` | Vessel forced to port when limit reached |
| **Gear restriction** | `economic.gear.restriction.<fleet>.<cell>` | Specific cells removed from fleet's choice set |

### Annual Reset

At the start of each simulation year:
- `vessel_days_used` reset to 0
- `vessel_revenue` and `vessel_costs` reset to 0
- Quotas reset to annual TAC values
- Seasonal closures re-evaluated

### Config Keys

```properties
# Master toggle
simulation.economic.enabled ; true

# Global parameters
simulation.economic.rationality ; 1.0
simulation.economic.memory.decay ; 0.7

# Fleet definitions
economic.fleet.number ; 2

economic.fleet.name.fsh0 ; Trawlers
economic.fleet.nvessels.fsh0 ; 50
economic.fleet.homeport.y.fsh0 ; 5
economic.fleet.homeport.x.fsh0 ; 3
economic.fleet.gear.fsh0 ; bottom_trawl
economic.fleet.max.days.fsh0 ; 200
economic.fleet.fuel.cost.fsh0 ; 500.0
economic.fleet.operating.cost.fsh0 ; 1000.0
economic.fleet.target.species.fsh0 ; 0,1,3,5
economic.fleet.price.sp0.fsh0 ; 2500.0
economic.fleet.price.sp1.fsh0 ; 1800.0
economic.fleet.stock.elasticity.sp0.fsh0 ; 0.5

# Policies
economic.quota.sp0 ; 50000
economic.landing.obligation.sp0 ; true
economic.mpa.cells.0 ; 3,4;3,5;4,4;4,5
economic.seasonal.closure.fsh0 ; 0-5
```

### Output Files

| File | Content | Shape |
|------|---------|-------|
| `econ_effort_<fleet>.csv` | Vessels per cell per timestep | (timesteps, ny × nx) |
| `econ_catch_<fleet>.csv` | Realized catch per species per timestep | (timesteps, n_species) |
| `econ_revenue_<fleet>.csv` | Revenue per species per timestep | (timesteps, n_species) |
| `econ_costs_<fleet>.csv` | Travel + operating costs per timestep | (timesteps, 2) |
| `econ_profit_summary.csv` | Per-fleet annual profit | (years, n_fleets) |

---

## Phasing

### Phase 1: MVP

**Ev-OSMOSE MVP:**
- TraitRegistry with 1 trait (Imax)
- GeneticState with diploid alleles and trait expression
- Gametic inheritance at bioen reproduction (fecundity-weighted parents, meiotic segregation)
- Sync with SchoolState (append/compact hooks in simulate.py)
- No neutral loci, no seeding phase delay
- Config parsing for `evolution.trait.imax.*`
- ~5 tests: trait expression correctness, inheritance produces valid offspring, compact sync, phenotype override modifies bioen behavior

**Economic MVP:**
- FleetConfig and FleetState parsed from config
- Revenue-only logit choice (no costs, no memory)
- Effort map feeds into fishing mortality via optional parameter
- Single fleet, no policies
- ~5 tests: effort allocation sums correctly, logit probabilities valid, fishing mortality responds to effort, port choice works

**Deliverables:** Both modules toggleable via config. All 1864+ existing tests pass with both disabled. Smoke tests with each enabled independently and both together.

### Phase 2: Core

**Ev-OSMOSE Core:**
- All 4 Java traits (Imax, gsi, m0, m1) with full bioen integration
- Neutral loci for genetic drift tracking
- Seeding phase (random genotypes before `transmission.year.start`)
- Proper fecundity-weighted multinomial sampling
- Validation against Java Ev-OSMOSE (North Sea config from Zenodo)
- ~20 tests: per-trait expression, inheritance statistics over many generations, allele frequency drift, Hardy-Weinberg equilibrium check, neutral vs selected loci divergence

**Economic Core:**
- Full cost model: fuel (Manhattan distance), operating costs, stock-dependent catchability
- Vessel memory (exponential moving average of past catches)
- Multiple fleets with different gear types, home ports, target species
- Days-at-sea tracking with forced port return
- Revenue, cost, profit tracking and CSV output
- ~20 tests: cost calculation correctness, memory update dynamics, multi-fleet non-interference, days-at-sea enforcement, output file format, profit = revenue - costs

### Phase 3: Extended

**Ev-OSMOSE Extended:**
- User-defined evolving traits via config (any per-species float config param → evolving)
- Optional mutation operator (allelic mutation at configurable rate per locus)
- Genetic diversity output: heterozygosity, allele frequency spectra, per-trait variance over time
- UI tab: trait distribution histograms, allele frequency time series, heritability estimates

**Economic Extended:**
- Policy levers: MPA closures, TAC quotas, landing obligations, seasonal closures, gear restrictions
- Fleet entry/exit dynamics (vessels enter when expected profit > threshold, exit otherwise)
- Endogenous price dynamics (prices respond to aggregate landings via demand elasticity)
- UI tab: fleet spatial heatmap, profit time series, policy scenario comparison panel
- Integration with calibration module (optimize policy parameters)
- Sensitivity analysis: which economic parameters drive fleet behavior most?

---

## Estimated Scope

| Phase | Ev-OSMOSE | Economic | Combined |
|-------|-----------|----------|----------|
| MVP | ~800 LOC, 5 tests | ~600 LOC, 5 tests | ~1400 LOC, 10 tests |
| Core | ~1500 LOC, 25 tests | ~1200 LOC, 25 tests | ~2700 LOC, 50 tests |
| Extended | ~800 LOC, 15 tests | ~1000 LOC, 20 tests | ~1800 LOC, 35 tests |
| **Total** | **~3100 LOC, 45 tests** | **~2800 LOC, 50 tests** | **~5900 LOC, 95 tests** |

## Testing Strategy

- **Unit tests:** Each function in isolation (trait expression, gamete formation, logit choice, cost computation)
- **Statistical tests:** Inheritance over many generations converges to expected allele frequencies; logit choice frequencies match analytical probabilities
- **Integration tests:** Full simulation with genetics enabled produces different trait distributions than without; fleet effort responds to fish abundance changes
- **Parity tests (Phase 2):** Ev-OSMOSE output matches Java Ev-OSMOSE for North Sea config within statistical tolerance
- **Policy tests (Phase 3):** MPA closure zeroes effort in closed cells; quota enforcement stops fishing at limit

## Backward Compatibility

Both modules are strictly opt-in:
- `simulation.genetic.enabled` defaults to `false`
- `simulation.economic.enabled` defaults to `false`
- When disabled, zero performance overhead (no code paths touched)
- All existing config files work unchanged
- All existing tests pass unchanged
