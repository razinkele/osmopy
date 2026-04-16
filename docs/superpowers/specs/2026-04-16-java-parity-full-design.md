# Full Java OSMOSE 4.3.3 Parity — Design Spec

> Systematic implementation of all missing Java OSMOSE 4.3.3 features in the Python engine. Decomposed into 5 sub-projects executed in dependency order.

## Reference

- **Java source:** `/home/razinka/osmose/osmose-master/java/src/main/java/fr/ird/osmose/`
- **Java JAR:** `osmose-java/osmose_4.3.3-jar-with-dependencies.jar`
- **Parity tests:** `tests/test_engine_parity.py`, `tests/test_engine_java_comparison.py`
- **Validation script:** `scripts/validate_engines.py`

## Current Parity Status

- Core ecology: 95% (14/14 EEC, 8/8 BoB validated)
- Fishing system: 14% (1/7 mortality variants, 1/4 selectivity types)
- Additional mortality: 40% (2/5 variants)
- Output system: 6% (~5/88 output classes)
- Bioen activation: stubs exist, not wired
- Time-series loading: 29% (2/7 variant types)

## Execution Order

```
SP-3 (TimeSeries) → SP-1 (Fishing) → SP-2 (Mortality) → SP-5 (Bioen) → SP-4 (Outputs)
```

SP-3 first: provides the time-varying parameter loading infrastructure needed by SP-1 and SP-2. SP-4 last: benefits from all processes being complete.

---

## SP-3: Time-Series Framework

**Problem:** Java has 7 `TimeSeries` classes in `util/timeseries/` for loading parameter values that vary over time. Python hardcodes values or reads single-year CSVs.

**Java classes (7):**
- `SingleTimeSeries` — reads CSV (header + data, column[1]); cycles if shorter than simulation length. NOT a scalar wrapper — handles full time-series with cycling.
- `SeasonTimeSeries` — reads from **config array key** (not CSV file); if array length = ndt_per_year, cycles annually over simulation.
- `ByYearTimeSeries` — reads CSV with **one value per year** (not year×season); cycles if fewer years than simulation.
- `BySpeciesTimeSeries` — per-species values from config array.
- `ByClassTimeSeries` — reads CSV: header row = class thresholds, data rows = per-class values per dt; cycles if shorter.
- `ByRegimeTimeSeries` — reads from **config array keys** (not CSV files); takes shift years + values from config; converts shift years to time steps internally.
- `GenericTimeSeries` — reads CSV (header + data, column[1]); NO cycling, raw values only.

**Key shared behavior:** `SingleTimeSeries`, `ByClassTimeSeries`, and `ByYearTimeSeries` all **cycle** shorter series to fill the simulation length. Python must replicate this cycling behavior.

**Design:**

Create `osmose/engine/timeseries.py`:

```python
class TimeSeries(Protocol):
    def get(self, step: int) -> float: ...

class SingleTimeSeries:
    """CSV time-series (header + data, column[1]). Cycles if shorter than ndt_simu."""

class SeasonTimeSeries:
    """Config array repeated annually. If len = ndt_per_year, cycles over simulation."""

class ByYearTimeSeries:
    """CSV with one value per year (column[1]). Cycles if fewer years than simulation."""

class ByClassTimeSeries:
    """CSV: header = class thresholds, rows = per-class values per dt. Cycles if shorter."""
    def get_by_class(self, step: int, class_idx: int) -> float: ...

class ByRegimeTimeSeries:
    """Config arrays: shift years + values. Regime changes at year boundaries."""

class GenericTimeSeries:
    """CSV time-series (header + data, column[1]). No cycling — raw values only."""
```

Factory function:
```python
def load_timeseries(config: dict, key_prefix: str, species_idx: int, ndt: int) -> TimeSeries:
    """Auto-detect timeseries type from config keys and load appropriate class."""
```

**Note:** Java has NO centralized factory — each process class instantiates TimeSeries directly in its `init()`/`readParameters()` method. The Python `load_timeseries()` factory is a design improvement over Java's scattered approach, consolidating the detection logic in one place.

Detection logic (derived from reading each Java process's `readParameters()`):
- If value is numeric string → `SingleTimeSeries`
- If `.byDt.file` key exists → `GenericTimeSeries` or `ByClassTimeSeries` (if has class columns)
- If `.season.file` or `.season.distrib.file` key exists → `SeasonTimeSeries`
- If `.byYear.file` key exists → `ByYearTimeSeries`
- If `.byRegime.file` + `.byRegime.shift` keys exist → `ByRegimeTimeSeries` (value changes at year boundaries specified by shift array)

**Files:**
- Create: `osmose/engine/timeseries.py`
- Test: `tests/test_engine_timeseries.py`

**Tests:** 7 tests (one per type) + factory dispatch test + edge cases (missing file, wrong shape).

---

## SP-1: Fishing System

**Problem:** Python has only `RateBySeasonFishingMortality` (constant annual rate). Java has 6 more fishing mortality variants plus a full fishery infrastructure (selectivity types, seasonality, spatial maps, gear, MPA).

### 1A. Fishing Mortality Variants

**Java classes (7 in `mortality/fishing/`):**

| Class | Config key pattern | Formula |
|-------|-------------------|---------|
| `RateBySeasonFishingMortality` | `mortality.fishing.rate.sp{i}` (scalar) | F = annualF × season[step_in_year]. Default season = 1/ndt (uniform). Optional seasonality file: `mortality.fishing.season.distrib.file.sp{i}` |
| `RateByYearBySeasonFishingMortality` | `mortality.fishing.rate.byYear.file.sp{i}` | F = annualF[year] × season[step_in_year]. Year-varying rate from ByYearTimeSeries + optional seasonality file |
| `RateByDtByClassFishingMortality` | `mortality.fishing.rate.byDt.byAge.file.sp{i}` OR `.bySize.file.sp{i}` | F = rate(dt, age/size class) from CSV |
| `CatchesBySeasonFishingMortality` | `mortality.fishing.catches.sp{i}` (scalar) | Proportional allocation: catch = (school_biomass / fishable_biomass) × annual_catches × season_weight |
| `CatchesByYearBySeasonFishingMortality` | `mortality.fishing.catches.byYear.file.sp{i}` | Same proportional allocation with year-varying target |
| `CatchesByDtByClassFishingMortality` | `mortality.fishing.catches.byDt.byAge.file.sp{i}` OR `.bySize.file.sp{i}` | Same proportional allocation per class |

Rate-based variants use SP-3 `TimeSeries` to load values. Catch-based variants use **proportional allocation** (NOT bisection): each school's catch = `(school_biomass / total_fishable_biomass) × target_catches × seasonal_weight`. This is from Java `CatchesBySeasonFishingMortality.getCatches()`. The `RateByDtByClassFishingMortality` and `CatchesByDtByClassFishingMortality` each support BOTH `byAge` and `bySize` config key variants.

**Files:**
- Modify: `osmose/engine/processes/fishing.py` — add 5 new rate/catch variants, dispatch by config
- Modify: `osmose/engine/config.py` — parse fishing config keys, detect variant type

### 1B. Fishery Infrastructure

**Java classes (6 in `mortality/fishery/` + 1 in `util/`):**

| Class | Purpose |
|-------|---------|
| `FisheryBase` | Per-fishery configuration: name, target species, period, selectivity, maps, seasonality |
| `FisherySelectivity` | 4 selectivity functions: knife-edge (exists), sigmoid (L50/L75 logistic), Gaussian, log-normal |
| `FisherySeasonality` | Monthly effort multiplier loaded via TimeSeries |
| `FisheryMapSet` | Spatial distribution maps per fishery (same MapSet pattern as movement) |
| `FisheryPeriod` | Start/end year for fishery activity |
| `FishingGear` (in `mortality/`, not `mortality/fishery/`) | Maps gear type → selectivity parameters |
| `MPA` (util) | Spatial closure grid with time windows — cells where fishing = 0 |

**Files:**
- Create: `osmose/engine/fishery.py` — `Fishery` dataclass (replaces FisheryBase, FisheryPeriod, FishingGear)
- Modify: `osmose/engine/processes/selectivity.py` — add sigmoid, Gaussian, log-normal selectivity
- Modify: `osmose/engine/config.py` — parse fishery config keys (`fisheries.name.fsh{i}`, etc.)
- Modify: `osmose/engine/processes/mortality.py` — apply fishery selectivity + seasonality + spatial maps + MPA in fishing mortality step

### 1C. Spatial Fishing Distribution

Config key: `mortality.fishing.spatial.distrib.file.sp{i}` — per-species spatial factor grid (GridMap). Multiplies the fishing rate per cell. Java normalizes so the mean over fished cells = 1.0. Separate from fishery-level FisheryMapSet.

### 1D. MPA (Marine Protected Areas)

Config keys: `mpa.file.mpa{i}`, `mpa.start.year.mpa{i}`, `mpa.end.year.mpa{i}`

MPA supports **percentage** coverage per cell (`getPercentageMPA(cell)`), not just binary 0/1. MPA factor = `1 - percentageMPA`. When MPA closes cells, Java redistributes fishing effort to non-MPA cells to keep total effort constant (normalization within `setMPA()`).

**Tests:**
- Rate variants: 6 tests (one per type, including seasonality) with synthetic configs
- Catch-based allocation: 3 tests (proportional allocation, zero fishable biomass, non-fishable school)
- Selectivity: 4 tests (one per type, verify shape against Java formulas)
- Spatial distribution: 1 test (verify normalization)
- MPA: 2 tests (percentage coverage, effort redistribution)
- Fishery integration: 2 tests (seasonality × selectivity, spatial × MPA)
- Config dispatch: 1 test (auto-detect variant from config keys)

---

## SP-2: Additional Mortality Variants

**Problem:** Python has constant annual rates only. Java has 3 more time-varying variants.

**Java classes (5 in `mortality/additional/`):**

| Class | Config detection | Python plan |
|-------|-----------------|-------------|
| `AnnualAdditionalMortality` | `mortality.additional.rate.sp{i}` (scalar) | **EXISTS** |
| `AnnualLarvaMortality` | `mortality.additional.larva.rate.sp{i}` (scalar) | **EXISTS** |
| `ByDtAdditionalMortality` | `mortality.additional.rate.bytDt.file.sp{i}` (**note Java typo: `bytDt`**) | Load via SP-3 `SingleTimeSeries` (has cycling behavior) |
| `ByDtLarvaMortality` | `mortality.additional.larva.rate.bytDt.file.sp{i}` (**same typo**) | Load via SP-3 `SingleTimeSeries` (has cycling behavior) |
| `ByDtByClassAdditionalMortality` | `mortality.additional.rate.byDt.byAge.file.sp{i}` OR `.bySize.file.sp{i}` (correct `byDt` here) | Load via SP-3 `ByClassTimeSeries` (supports both age and size classes) |

**IMPORTANT:** Java has a typo in config keys — `bytDt` instead of `byDt` for the ByDt and ByDtLarva variants (confirmed in `AdditionalMortality.Scenario` enum and the class `init()` methods). The `ByDtByClass` variants use the correct `byDt`. Python must accept BOTH `bytDt` (backward compat with Java configs) and `byDt` (corrected form) during config key detection.

**Also:** Java loads `ByDtAdditionalMortality` and `ByDtLarvaMortality` via `SingleTimeSeries` (which has cycling), NOT `GenericTimeSeries` (which doesn't cycle). Python must use the cycling variant.

**Spatial additional mortality:** Java also loads `mortality.additional.spatial.distrib.file.sp{i}` — a per-species GridMap spatial factor that multiplies additional mortality rates per cell (lines 127-136 of `AdditionalMortality.java`).

**Design:**

Modify `osmose/engine/processes/natural.py`:
- At init, detect variant type from config keys (check both `bytDt` and `byDt`)
- If `.bytDt.file` or `.byDt.file` key exists → load SingleTimeSeries (with cycling), index by step
- If `.byDt.byAge.file` key exists → load ByClassTimeSeries, index by step and age class
- Otherwise → use existing constant rate
- Load spatial factor grid if `mortality.additional.spatial.distrib.file.sp{i}` exists

Modify `osmose/engine/config.py`:
- Parse additional mortality file keys (both typo and corrected forms)
- Parse spatial distribution file key
- Store file paths in EngineConfig

**Files:**
- Modify: `osmose/engine/processes/natural.py`
- Modify: `osmose/engine/config.py`
- Test: `tests/test_engine_additional_mortality.py`

**Tests:** 5 tests (ByDt with cycling, ByDt varying, ByDtByClass by age, ByDtByClass by size, spatial factor application).

---

## SP-5: Bioen Process Activation

**Problem:** Three bioen mortality processes have stub code but aren't called in the simulation loop.

### ForagingMortality

**Java:** `process/mortality/ForagingMortality.java` (lines 83-105)

Two modes depending on whether genetics is enabled:

**Without genetics (standard bioen):**
```
F_foraging = k_for / ndt_per_year
```
Simple constant rate. Config key: `species.bioen.forage.k_for.sp{i}`.

**With genetics (Ev-OSMOSE):**
```
F_foraging = k1_for * exp(k2_for * (imax_trait - I_max)) / ndt_per_year
```
Exponential penalty: schools whose evolved `imax` trait deviates from the species baseline `I_max` suffer higher foraging mortality. Config keys:
- `species.bioen.forage.k1_for.sp{i}` — base rate coefficient
- `species.bioen.forage.k2_for.sp{i}` — exponential scaling
- `predation.ingestion.rate.max.bioen.sp{i}` — reference I_max

In both modes: if result < 0, clamp to 0.

### BioenStarvationMortality

**Java:** `process/bioen/BioenStarvationMortality.java` (lines 76-116)

Called via `computeStarvation(school, subdt)` (NOT via `getRate()`). Formula:

```python
if E_net >= 0:
    return 0  # maintenance paid, no starvation

e_net_sub = abs(E_net) / subdt  # per-substep deficit

if gonad_weight >= eta * e_net_sub:
    # Enough gonadic energy — pay with gonad, no deaths
    gonad_weight -= e_net_sub * eta
    E_net += e_net_sub
    return 0
else:
    # Not enough gonad — flush gonad, starvation occurs
    E_net += gonad_weight * eta
    death_toll = e_net_sub - gonad_weight / eta
    gonad_weight = 0
    return death_toll / school_weight  # fraction of school that dies
```

Config key: `species.bioen.maturity.eta.sp{i}` (gonadic energy conversion factor).

### BioenPredationMortality

**Java:** `process/bioen/BioenPredationMortality.java`

Extends `PredationMortality`. Overrides two methods:

1. **`getMaxPredationRate(predator)`** — bioen-specific ingestion rate:
   ```
   imax = trait("imax") if genetic else predationRateBioen[sp]
   larvae_factor = larvaePredationRateBioen[sp] if age < threshold else 1
   rate = (imax + (larvae_factor - 1) * c_rateBioen[sp]) / ndt_per_year
   ```

2. **`computePredation()`** — same proportional prey distribution as standard, but uses bioen rate and weight in **grams** (`weight * 1e6`) with `betaBioen` exponent:
   ```
   maxBiomassToPredate = rate * (weight_g ^ betaBioen) / subdt * abundance * 1e-6
   ```

Config keys:
- `predation.ingestion.rate.max.bioen.sp{i}` — base Imax
- `predation.coef.ingestion.rate.max.larvae.bioen.sp{i}` — larvae multiplier
- `predation.c.bioen.sp{i}` — c-rate for larvae formula

**Note:** Java initializes `OxygenFunction` in constructor but it's not directly called in `computePredation()` or `getMaxPredationRate()` — it may be used elsewhere in the bioen pipeline or be a dead reference. Implement the constructor initialization but verify usage.

**Files:**
- Create: `osmose/engine/processes/foraging_mortality.py` — both modes (constant + genetic exponential)
- Modify: `osmose/engine/processes/mortality.py` — add FORAGING cause to interleaved loop when `bioen_enabled`
- Modify: `osmose/engine/processes/bioen_predation.py` — implement oxygen-constrained ingestion, wire into loop
- Modify: `osmose/engine/processes/bioen_starvation.py` — implement gonad-depletion starvation with eta, wire into loop
- Modify: `osmose/engine/config.py` — parse foraging/bioen mortality config keys
- Test: `tests/test_engine_bioen_activation.py`

**Tests:** 6 tests (foraging constant mode, foraging genetic mode, bioen starvation with gonad compensation, bioen starvation without gonad, bioen predation with O2, all three together in simulation).

---

## SP-4: Output System

**Problem:** Java has 88 output classes. Python writes ~5 output types. Users need distributions, yields, spatial maps, and NetCDF for analysis.

### Architecture

**Distribution infrastructure** (`output_distributions.py`):

Java has 4 distribution types (from `DistributionType` enum): SIZE, AGE, WEIGHT, TL.

```python
class OutputDistribution:
    """Bin schools by age, size, weight, or TL class for distribution outputs."""
    def bin_by_age(self, schools, n_age_classes) -> np.ndarray: ...
    def bin_by_size(self, schools, size_breaks) -> np.ndarray: ...
    def bin_by_weight(self, schools, weight_breaks) -> np.ndarray: ...
    def bin_by_tl(self, schools, tl_breaks) -> np.ndarray: ...
```

Config keys: `output.distribution.bySize.min/max/step`, `output.distribution.byAge.max`, `output.distribution.byWeight.min/max/step`, `output.distribution.byTL.min/max/step`

**Recording frequency** (`output.py`):
```python
# Java: output.recordfrequency.ndt — average over N steps before writing
# Default: 1 (write every step). Common: 24 (write yearly averages).
```

### Tier 1: Core Outputs

| Java Class | Output file | What |
|-----------|-------------|------|
| `BiomassDistribOutput` | `biomass_byAge.csv`, `biomass_bySize.csv` | Biomass binned by age/size class |
| `AbundanceDistribOutput` | `abundance_byAge.csv`, `abundance_bySize.csv` | Abundance binned by age/size class |
| `YieldOutput` | `yield.csv` | Total fishing yield per species |
| `YieldNOutput` | `yieldN.csv` | Fishing yield in numbers |
| `FisheryOutput` | `yield_byFishery.csv` | Yield split by fishery |
| `MeanSizeOutput` | `meanSize.csv` | Mean length per species |
| `MeanSizeByAge` | `meanSize_byAge.csv` | Mean length per age class |
| `MeanTrophicLevelOutput` | `meanTL.csv` | Mean trophic level per species |
| `DietOutput` | `diet.csv` | Diet matrix (predator × prey) |
| `AbundanceOutput_age1` | `abundance_age1.csv` | Age 1+ abundance |
| `OutputManager` | (controls frequency) | `output.recordfrequency.ndt` config key |

### Tier 2: Distribution & Pressure Outputs

| Java Class | Output file |
|-----------|-------------|
| `YieldDistribOutput` | `yield_bySize.csv`, `yield_byAge.csv` |
| `YieldNDistribOutput` | `yieldN_bySize.csv`, `yieldN_byAge.csv` |
| `MortalitySpeciesOutput` | `mortalityRate_byAge.csv` |
| `DietDistribOutput` | `diet_bySize.csv`, `diet_byAge.csv` |
| `PredatorPressureOutput` | `predatorPressure.csv` |
| `PredatorPressureDistribOutput` | `predatorPressure_bySize.csv` |
| `MeanTrophicLevelCatchOutput` | `meanTLCatch.csv` |
| `MeanSizeCatchOutput` | `meanSizeCatch.csv` |
| `BiomassDietStageOutput` | `biomassDietStage.csv` |

### Tier 3: Spatial Outputs (14 classes)

All follow the same pattern: aggregate per-cell per-species, write 2D grid CSV per timestep.

| Java Class | What |
|-----------|------|
| `SpatialBiomassOutput` | Biomass per cell |
| `SpatialAbundanceOutput` | Abundance per cell |
| `SpatialYieldOutput` / `SpatialYieldNOutput` | Yield per cell |
| `SpatialMortaPredOutput` / `SpatialMortaStarvOutput` | Mortality per cell |
| `SpatialSizeOutput` / `SpatialSizeSpeciesOutput` | Mean size per cell |
| `SpatialTLOutput` | Mean TL per cell |
| `SpatialEggOutput` | Egg production per cell |
| `SpatialEnetOutput` / juveniles / larvae variants | Bioen Enet per cell |
| `SpatialdGOutput` | Gonad growth per cell |

**Infrastructure:** `osmose/engine/output_spatial.py` — generic per-cell aggregator that takes a school-level metric function and produces a 2D grid.

### Tier 4: NetCDF Format (22 classes)

All are NetCDF mirrors of Tier 1-3 CSV outputs using xarray (already a dependency).

**Infrastructure:** `osmose/engine/output_netcdf.py` — wrapper that converts CSV-style data to xarray Dataset and writes `.nc`.

**Important:** There is NO single `output.format` toggle. CSV and NetCDF outputs are **independently enabled** via separate config keys. For example:
- `output.biomass.enabled = true` → CSV biomass output
- `output.biomass.netcdf.enabled = true` → NetCDF biomass output
- Both can be enabled simultaneously

The full config key pattern is: `output.<metric>[.<distribution>][.netcdf].enabled`

### Tier 5: Specialized & Economic Outputs

| Java Class | What | Priority |
|-----------|------|----------|
| `NSchoolOutput` / `NSchoolDistribOutput` | Number of schools alive | LOW |
| `NDeadSchoolOutput` / `NDeadSchoolDistribOutput` | Dead school counts | LOW |
| `NewSchoolOutput` | Newly created schools | LOW |
| `MeanGenotypeOutput` | Mean genotype per species | LOW |
| `VariableTraitOutput` | Trait variance per species | LOW |
| `AgeAtDeathOutput` | Age distribution at death | LOW |
| `DiscardOutput` | Discard biomass (needs SP-1 discards) | LOW |
| `Surveys` | Survey-style sampling | LOW |
| `SchoolSetSnapshot` / `ModularSchoolSetSnapshot` | Full school state dump | LOW |
| `FishingAccessBiomassOutput` | Biomass accessible to each fishery | LOW |
| `FishingHarvestedBiomassDistribOutput` | Harvested biomass by size/age | LOW |
| `FishingPriceAccessBiomassOutput` | Price-weighted accessible biomass | LOW |
| `ResourceOutput` | Resource/LTL biomass time series | LOW |
| `MortalityOutput` | Aggregate mortality rates | MEDIUM |

**Bioen outputs** (`BioenMaintOutput`, `BioenMeanEnergyNet`, `BioenRhoOutput`, `BioenSizeInfOutput`) are already implemented in Python `output.py` — no work needed.

### Output Control Config Keys

Java `OutputManager` uses per-output toggle keys (from `OutputManager.init()`):
```
output.abundance.netcdf.enabled = true/false
output.biomass.netcdf.enabled = true/false
output.yield.biomass.netcdf.enabled = true/false
output.biomass.bysize.netcdf.enabled = true/false
output.biomass.byage.netcdf.enabled = true/false
output.biomass.bytl.netcdf.enabled = true/false
output.diet.composition.netcdf.enabled = true/false
output.recordfrequency.ndt = 24  (averaging period)
```

Python should support both `output.recordfrequency.ndt` and per-output `output.*.enabled` toggles.

**Files:**
- Create: `osmose/engine/output_distributions.py` — binning infrastructure
- Create: `osmose/engine/output_spatial.py` — per-cell aggregation
- Create: `osmose/engine/output_netcdf.py` — NetCDF writer
- Modify: `osmose/engine/output.py` — add Tier 1-5 writers, recording frequency, output manager with per-output toggles
- Modify: `osmose/engine/config.py` — parse output config keys
- Modify: `osmose/engine/simulate.py` — collect additional metrics (yield, diet, TL, predator pressure) during simulation for output
- Test: `tests/test_engine_outputs.py`

**Tests:** Per-tier verification against Java reference outputs from Bay of Biscay config.

---

## Parity Validation Strategy

After each sub-project, run the existing parity validation:

```bash
# Python-only baseline regression
.venv/bin/python -m pytest tests/test_engine_parity.py -v

# Python vs Java comparison (requires Java installed)
.venv/bin/python scripts/validate_engines.py --years 5 --seed 42
```

After SP-4 (outputs), add output-level parity tests:
- Run both Java and Python on Bay of Biscay for 5 years
- Compare output CSV files column-by-column within tolerance

---

## Estimated Effort

| Sub-project | New files | Modified files | Tests | Est. hours |
|-------------|-----------|----------------|-------|------------|
| SP-3: TimeSeries | 1 | 0 | ~10 | 4 |
| SP-1: Fishing | 2 | 4 | ~16 | 8 |
| SP-2: Mortality | 0 | 2 | ~3 | 3 |
| SP-5: Bioen | 1 | 4 | ~6 | 3 |
| SP-4: Outputs | 3 | 3 | ~20 | 15 |
| **Total** | **7** | **13** | **~55** | **~33** |
