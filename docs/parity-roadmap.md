# OSMOSE Python Engine — Java Parity Roadmap

> **Date:** 2026-03-22 (updated)
> **Current status:** Bay of Biscay **8/8 PASS**, EEC **14/14 PASS** within 1 OoM
> **Goal:** Close all functional gaps between Python and Java OSMOSE engines
>
> **Milestone achieved:** Full parity on both validation configs. The critical
> predation architecture fix (unified school+resource proportional distribution)
> closed the remaining 2 species gaps (redMullet -2.0 OoM → +0.32, sardine -1.4 OoM → +0.05).

---

## Phase 1: EEC Parity Fixes — COMPLETE ✓

All Phase 1 items implemented. EEC now at 14/14 within 1 OoM.

**Critical fix (2026-03-22):** Unified predation architecture — `_apply_predation_for_school` now
scans both school prey AND resource prey into a single accessible-biomass pool and distributes
eating proportionally across all prey types, matching Java's `computePredation()`. Previously,
the two-phase approach (schools first, resources second) caused excess predation on small
forage fish and a remaining-appetite miscalculation that allowed ~45% over-eating per sub-timestep.

### 1.1 Maturity age check
**Impact:** HIGH — SSB includes immature schools → excess egg production
**Gap:** Python checks `length >= maturity_size` only. Java also checks `age_dt >= maturity_age_dt`.
**Config key:** `species.maturity.age.sp{i}`
**Fix:** In `reproduction.py`, add age threshold to maturity mask. In `config.py`, parse `species.maturity.age.sp{i}` into new `maturity_age_dt` field.
**Files:** `config.py`, `reproduction.py`
**Tests:** 2 new tests
**Effort:** 30 min

### 1.2 Spatial fishing distribution
**Impact:** HIGH — fishing applied uniformly instead of concentrated in fishable cells
**Gap:** Java reads `mortality.fishing.spatial.distrib.file.sp{i}` (a CSV grid map of fishing intensity per cell). Python applies uniform fishing rate.
**Config key:** `mortality.fishing.spatial.distrib.file.sp{i}`
**Fix:** Load fishing spatial CSV, multiply fishing rate by cell value in `_apply_fishing_for_school`. For v4 fisheries, load `fisheries.movement.file.map{N}.fsh{i}`.
**Files:** `config.py`, `mortality.py`, `fishing.py`
**Tests:** 3 new tests
**Effort:** 45 min

### 1.3 `species.lmax.sp{i}` — maximum length cap
**Impact:** MEDIUM-HIGH — schools grow past the correct maximum length
**Gap:** Python caps growth at `L_inf`. Java caps at `lmax` which can differ from `L_inf`.
**Config key:** `species.lmax.sp{i}`
**Fix:** Parse into `EngineConfig.lmax` field. Use in `growth.py` instead of `linf` for the cap.
**Files:** `config.py`, `growth.py`
**Tests:** 1 new test
**Effort:** 15 min

### 1.4 Resource biomass multiplier and offset
**Impact:** MEDIUM — resource biomass scaling is wrong for configs that use these
**Gap:** Java applies `multiplier * (biomass + offset)` to resource forcing. Python ignores these.
**Config keys:** `species.multiplier.sp{i}`, `species.offset.sp{i}`
**Fix:** Already parsed in `BackgroundState` for background species. Apply same logic in `ResourceState._load_config_species_type()`.
**Files:** `resources.py`
**Tests:** 2 new tests
**Effort:** 20 min

### 1.5 Time-varying resource accessibility
**Impact:** MEDIUM — resource accessibility is constant instead of time-varying
**Gap:** Java reads `species.accessibility2fish.file.sp{i}` (CSV time series). Python uses a constant scalar.
**Config key:** `species.accessibility2fish.file.sp{i}`
**Fix:** In `ResourceState`, load the accessibility CSV and index by timestep in `update()`.
**Files:** `resources.py`
**Tests:** 2 new tests
**Effort:** 30 min

### 1.6 Resource accessibility cap at 0.99
**Impact:** LOW — prevents numerical instability with accessibility = 1.0
**Gap:** Java caps at 0.99. Python applies raw value.
**Fix:** `min(accessibility, 0.99)` in `ResourceState._load_config`.
**Files:** `resources.py`
**Tests:** 1 new test
**Effort:** 5 min

### 1.7 Trophic level computation
**Impact:** MEDIUM — affects diet output and trophic-level-based outputs
**Gap:** Python sets `trophic_level = 2.0` placeholder. Java computes weighted average from prey TLs.
**Fix:** In the per-school predation function, accumulate prey TL * biomass. After predation, compute `TL = 1 + sum(prey_TL * prey_biomass) / total_preyed_biomass`.
**Files:** `mortality.py` (per-school predation)
**Tests:** 2 new tests
**Effort:** 30 min

---

## Phase 2: Fishing System Completion (MEDIUM impact)

The v4 fisheries system is partially implemented. These additions make it complete for EEC and similar configs.

### 2.1 Fishery seasonality
**Impact:** MEDIUM — fishing rate is flat instead of seasonal
**Gap:** `fisheries.seasonality.file.fsh{i}` provides intra-annual seasonality multiplier.
**Config key:** `fisheries.seasonality.file.fsh{i}`
**Fix:** Load seasonality CSV, normalize to sum=1 per year, multiply fishing rate by `seasonality[step_in_year]`.
**Files:** `config.py`, `fishing.py` or `mortality.py`
**Tests:** 2 new tests
**Effort:** 30 min

### 2.2 Fishery spatial maps
**Impact:** MEDIUM — fisheries apply to all cells instead of their fishing grounds
**Gap:** `fisheries.movement.file.map{N}.fsh{i}` defines spatial extent per fishery.
**Config key:** `fisheries.movement.file.map{N}.fsh{i}`
**Fix:** Load fishery spatial maps, multiply rate by cell value.
**Files:** `config.py`, `mortality.py`
**Tests:** 2 new tests
**Effort:** 45 min

### 2.3 v3 fishing scenarios (beyond RATE_ANNUAL)
**Impact:** LOW (EEC uses v4) — needed for older configs
**Gap:** 7 of 8 v3 scenarios absent (by-year, by-dt-by-age, by-dt-by-size, catches).
**Config keys:** `mortality.fishing.rate.byYear.file.sp{i}`, etc.
**Fix:** Load CSV time series for each scenario type.
**Files:** `fishing.py`, `config.py`
**Tests:** 4 new tests
**Effort:** 1.5 hours

### 2.4 Fishing selectivity types (beyond knife-edge)
**Impact:** LOW-MEDIUM — sigmoidal selectivity for v4 fisheries
**Gap:** Java supports type 0 (knife-edge age), type 1 (sigmoidal size), type 2 (knife-edge size).
**Config keys:** `fisheries.selectivity.type.fsh{i}`, `fisheries.selectivity.slope.fsh{i}`
**Fix:** Add sigmoid selectivity function.
**Files:** `fishing.py` or `mortality.py`
**Tests:** 2 new tests
**Effort:** 20 min

### 2.5 MPA (Marine Protected Areas)
**Impact:** LOW — only configs with MPAs
**Gap:** Spatial fishing reduction in protected cells.
**Config keys:** `mpa.file.mpa{i}`, `mpa.start.step.mpa{i}`, `mpa.end.step.mpa{i}`, `mpa.percentage.mpa{i}`
**Fix:** Load MPA grid, apply percentage reduction to fishing rate in MPA cells.
**Files:** `config.py`, `mortality.py`
**Tests:** 3 new tests
**Effort:** 45 min

### 2.6 Fishery discards
**Impact:** LOW — discard tracking for output/economy
**Gap:** `fisheries.discards.file` matrix defines discard rates per species per fishery.
**Config key:** `fisheries.discards.file`
**Fix:** Load discard matrix, split fishing mortality into FISHING + DISCARDS causes.
**Files:** `mortality.py`, `config.py`
**Tests:** 2 new tests
**Effort:** 30 min

---

## Phase 3: Reproduction & Mortality Refinement (MEDIUM impact)

### 3.1 Spawning season normalization
**Impact:** MEDIUM — season values may not sum to 1 without normalization
**Gap:** `reproduction.normalisation.enabled` flag triggers per-year normalization.
**Config key:** `reproduction.normalisation.enabled`
**Fix:** In reproduction, normalize season array per year if flag is set.
**Files:** `reproduction.py`, `config.py`
**Tests:** 2 new tests
**Effort:** 20 min

### 3.2 Multi-year spawning season time series
**Impact:** MEDIUM — interannual variability lost
**Gap:** Java handles season CSV with > n_dt_per_year rows as interannual time series.
**Fix:** In `_load_spawning_seasons`, detect multi-year CSVs and index by absolute step.
**Files:** `config.py`
**Tests:** 1 new test
**Effort:** 20 min

### 3.3 Time-varying additional mortality
**Impact:** MEDIUM — constant rate instead of time/age/size varying
**Gap:** 3 scenarios: BY_DT, BY_DT_BY_AGE, BY_DT_BY_SIZE.
**Config keys:** `mortality.additional.rate.bytDt.file.sp{i}`, etc.
**Fix:** Load CSV, index by timestep and age/size class.
**Files:** `natural.py`, `config.py`
**Tests:** 3 new tests
**Effort:** 45 min

### 3.4 Spatial additional mortality
**Impact:** MEDIUM — uniform rate instead of spatially varying
**Gap:** `mortality.additional.spatial.distrib.file.sp{i}` grid map.
**Config key:** `mortality.additional.spatial.distrib.file.sp{i}`
**Fix:** Load spatial map, multiply rate by cell value.
**Files:** `natural.py`, `config.py`, `mortality.py`
**Tests:** 2 new tests
**Effort:** 30 min

### 3.5 Egg placement timing
**Impact:** LOW-MEDIUM — eggs placed immediately vs deferred to next step movement
**Gap:** Python places eggs at random cells immediately. Java creates unlocated eggs that get placed by movement on the next step.
**Fix:** Create eggs with `cell_x = cell_y = -1` (unlocated). Movement will place them.
**Files:** `reproduction.py`
**Tests:** 1 new test
**Effort:** 10 min

### 3.6 nEgg < nSchool edge case
**Impact:** LOW — Python creates many tiny schools, Java creates one
**Fix:** `if n_eggs < n_new: n_new = 1`
**Files:** `reproduction.py`
**Tests:** 1 new test
**Effort:** 5 min

### 3.7 `population.seeding.year.max` config key
**Impact:** LOW — Python uses max lifespan as default (matches Java default)
**Config key:** `population.seeding.year.max`
**Fix:** Parse and use in seeding condition.
**Files:** `config.py`, `reproduction.py`
**Tests:** 1 new test
**Effort:** 10 min

---

## Phase 4: Movement Refinement (MEDIUM impact)

### 4.1 Random distribution patch constraint
**Impact:** MEDIUM — species with `ncell` config spread across full grid instead of connected patch
**Gap:** Java's `RandomDistribution.createRandomMap()` builds a BFS-connected patch.
**Config key:** `movement.distribution.ncell.sp{i}`
**Fix:** Implement BFS patch creation for `random` movement method.
**Files:** `movement.py`
**Tests:** 3 new tests
**Effort:** 45 min

### 4.2 Deterministic random seeds
**Impact:** LOW — affects reproducibility only
**Gap:** `movement.randomseed.fixed` and `stochastic.mortality.randomseed.fixed`
**Config keys:** `movement.randomseed.fixed`, `stochastic.mortality.randomseed.fixed`
**Fix:** Create per-species RNG instances with deterministic seeds when enabled.
**Files:** `movement.py`, `mortality.py`
**Tests:** 2 new tests
**Effort:** 30 min

---

## Phase 5: Output System (LOW simulation impact, HIGH usability) — STATUS-COMPLETE (2026-04-19)

All seven items shipped. 5.1 / 5.2 / 5.3 / 5.7 were already in the Python engine before SP-4; the SP-4 front (commits through 2026-04-19) closed 5.5 (diet Java-parity), 5.6 (NetCDF distributions + mortality), and 5.4 (spatial outputs: biomass / abundance / yield-biomass). Remaining Java-side spatial variants (TL, size, mortality, egg, bioen-spatial) and Ev-OSMOSE output families are deferred to Phase 6 — they don't block full standard-OSMOSE parity.

### 5.1 Output recording frequency
**Impact:** LOW simulation, HIGH usability
**Gap:** Python records every step. Java records every `output.recordfrequency.ndt` steps with averaging.
**Config key:** `output.recordfrequency.ndt`
**Fix:** Accumulate outputs, average over recording period, write at frequency.
**Files:** `simulate.py`, `output.py`
**Tests:** 2 new tests
**Effort:** 30 min

### 5.2 Yield/catches output
**Impact:** LOW simulation, HIGH usability
**Gap:** No fishing yield CSVs.
**Fix:** Track fished biomass per species per timestep, write yield CSVs.
**Files:** `output.py`, `mortality.py`
**Tests:** 2 new tests
**Effort:** 30 min

### 5.3 Size/age distribution outputs
**Impact:** LOW simulation, MEDIUM usability
**Gap:** No biomass/abundance by size or age class.
**Fix:** Bin schools by size/age, aggregate, write distribution CSVs.
**Files:** `output.py`
**Tests:** 2 new tests
**Effort:** 45 min

### 5.4 Spatial outputs
**Impact:** LOW simulation, MEDIUM usability
**Gap:** No spatial biomass/abundance maps.
**Fix:** Aggregate by cell, write per-cell CSV/NetCDF.
**Files:** `output.py`
**Tests:** 2 new tests
**Effort:** 45 min

### 5.5 True diet matrix output
**Impact:** LOW simulation, MEDIUM usability
**Gap:** Diet tracking exists but isn't written per timestep in simulate loop.
**Fix:** Enable diet tracking when `output.diet.composition.enabled=true`, write CSV per recording period.
**Files:** `simulate.py`, `output.py`
**Tests:** 1 new test
**Effort:** 20 min

### 5.6 NetCDF output format
**Impact:** LOW simulation, MEDIUM usability
**Gap:** No NetCDF output at all.
**Fix:** Write biomass/abundance/mortality/yield as NetCDF using xarray.
**Files:** `output.py`
**Tests:** 2 new tests
**Effort:** 1 hour

### 5.7 Initial state output (`output.step0.include`)
**Impact:** LOW
**Config key:** `output.step0.include`
**Fix:** Record step -1 output before main loop if enabled.
**Files:** `simulate.py`
**Tests:** 1 new test
**Effort:** 10 min

---

## Phase 6: Bioenergetic Module (Ev-OSMOSE) — STATUS-COMPLETE (2026-04-19)

All five items shipped as part of the Ev-OSMOSE + DSVM fleet dynamics track (merge `5b0e0aa` Phase 1 MVP and the subsequent Phase 2 Genetics / Economics Core commits). Gated by `config.bioen_enabled`. 80/80 tests pass across `tests/test_engine_bioen*.py` and `tests/test_bioen_orchestration.py`.

### 6.1 Energy budget growth — SHIPPED
Bioenergetic growth/maintenance path wired into `_bioen_step` at `osmose/engine/simulate.py`; trait-overridable ingestion cap (`bioen_i_max`) via genetics. Alternative to standard `GrowthProcess`, selected by `bioen_enabled`.

### 6.2 Bioenergetic reproduction — SHIPPED
`osmose/engine/processes/bioen_reproduction.py`; `_bioen_reproduction` in simulate accepts `trait_overrides` (`bioen_r` for GSI, `bioen_m0`/`bioen_m1` for maturity). Replaces standard reproduction when `bioen_enabled`.

### 6.3 Bioenergetic predation — SHIPPED
`osmose/engine/processes/bioen_predation.py`; modifies predation rate via bioenergetic ingestion cap. Active under `bioen_enabled`.

### 6.4 Bioenergetic starvation — SHIPPED
`osmose/engine/processes/bioen_starvation.py` with internal subloop matching Java `BioenStarvationMortality` (gonad depletion). Dispatched from `mortality.py:98` when `bioen_enabled=True`.

### 6.5 Foraging mortality — SHIPPED
`osmose/engine/processes/foraging_mortality.py` matches Java `ForagingMortality.getRate()` with constant and genetic modes. `MortalityCause.FORAGING` (enum index 5) documented in the SP-4 `cause_descriptions` attr.

**Priority fulfilled:** Ev-OSMOSE configurations now runnable on the Python engine.

---

## Phase 7: Code Quality & Architecture (ongoing)

### 7.1 Reconcile dual predation paths
**Gap:** `predation.py` (Numba) and `mortality.py` (per-cell Python) are two independent predation implementations
**Fix:** Remove the standalone batch predation path; always use per-cell
**Effort:** 2 hours

### 7.2 Pluggable growth classes
**Gap:** `growth.java.classname.sp{i}` not supported
**Fix:** Add growth class dispatch (VB, Gompertz, custom)
**Effort:** 1 hour

### 7.3 Config validation
**Gap:** Many config keys silently ignored when misspelled or missing
**Fix:** Add validation warnings for unknown keys, required key checks
**Effort:** 2 hours

---

## Estimated Timeline

| Phase | Items | Estimated Effort | Expected Impact |
|-------|-------|-----------------|-----------------|
| **Phase 1** | 7 fixes + predation architecture | DONE ✓ | EEC 14/14 PASS |
| **Phase 2** | 6 features | 4 hours | Complete fishing system |
| **Phase 3** | 7 fixes | 2.5 hours | Reproduction/mortality refinement |
| **Phase 4** | 2 features | 1.5 hours | Movement refinement |
| **Phase 5** | 7 features | 4 hours | Full output system |
| **Phase 6** | 5 features | 30 hours | Ev-OSMOSE support |
| **Phase 7** | 3 tasks | 5 hours | Code quality |
| **Total (excl. Phase 6)** | | **~20 hours** | **Full standard OSMOSE parity** |

---

## Priority Order for Maximum Parity Improvement

1. **1.1 Maturity age check** — likely biggest single EEC improvement
2. **1.2 Spatial fishing distribution** — fishing applied to wrong cells
3. **1.3 `species.lmax` cap** — growth overshoots
4. **3.5 Egg placement timing** — 5-minute fix with spatial impact
5. **1.4 Resource multiplier/offset** — biomass scaling
6. **1.5 Time-varying resource accessibility** — seasonal prey availability
7. **2.1 Fishery seasonality** — temporal fishing pattern
8. **1.7 Trophic level computation** — needed for diet outputs
9. **2.2 Fishery spatial maps** — fishing grounds
10. **3.1 Spawning season normalization** — reproduction timing
