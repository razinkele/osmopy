# OSMOSE Python Engine — Java Parity Roadmap

> **Date:** 2026-04-19 (STATUS-COMPLETE across all phases)
> **Current status:** Bay of Biscay **8/8 PASS**, EEC **14/14 PASS** within 1 OoM.
> **Goal:** Close all functional gaps between Python and Java OSMOSE engines — **reached**.
>
> **Parity complete.** All seven phases are SHIPPED. 2510 tests passing on master; Ev-OSMOSE bioenergetic + genetics + economics also complete. Python engine is faster than Java on every benchmarked configuration. The detailed phase-by-phase ledger below is retained for provenance; no roadmap-scoped work is open.
>
> **Milestone history:** The critical Phase 1 predation architecture fix (unified school+resource proportional distribution) closed the EEC parity — redMullet went from −2.0 OoM to +0.32, sardine from −1.4 to +0.05. Phases 2–4 landed incrementally during the 5/6/7 tracks and were retro-verified as SHIPPED during the 2026-04-19 audit. Phase 5 (SP-4 output system) shipped in v0.9.0; Phase 6 (Ev-OSMOSE) in v0.9.1; Phase 7 closed across v0.9.2 (7.3) and v0.9.3 (7.1).

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

## Phase 2: Fishing System Completion — STATUS-COMPLETE (2026-04-19)

All six items shipped incrementally during the Phase 5/6/7 tracks. The v4 fisheries system is complete; `osmose/engine/processes/fishing.py:4` summarizes: "Supports seasonality, time-varying rates, sigmoid selectivity, MPA, and discards." 270 tests pass across `tests/test_engine_selectivity.py`, `tests/test_engine_timeseries.py`, `tests/test_phase2_3_fixes.py`, `tests/test_overlay_display.py` (MPA coverage), and the `fish`/`mpa`/`discard`/`selectivity` test families.

### 2.1 Fishery seasonality — SHIPPED
`EngineConfig.fishing_seasonality: NDArray[np.float64] | None` and `fishing_catches_season` loaded via `_load_fishing_seasonality` in `config.py` (handles v3 `sp{i}` files, v4 `fsh{i}` files, and inline `fisheries.seasonality.fsh{i}` values). Applied at `fishing.py:66-70` as `season_weight = config.fishing_seasonality[sp, step_in_year]`.

### 2.2 Fishery spatial maps — SHIPPED
`EngineConfig.fishing_spatial_maps: list` parsed at `config.py:1417-1436` from `fisheries.movement.file.map0` (shared map) with per-species overrides. Applied in fishing mortality to scale rate per cell.

### 2.3 v3 fishing scenarios — SHIPPED
All four beyond-RATE_ANNUAL variants parsed at `config.py:1035-1041` and stored on `EngineConfig`: `fishing_rate_by_year`, `fishing_rate_by_dt_by_class`, `fishing_catches_by_year`, `fishing_catches_by_dt_by_class`. Runtime at `_catch_based_fishing(fishing.py:177)` for catch-based scenarios.

### 2.4 Fishing selectivity types (beyond knife-edge) — SHIPPED
`fishing.py:77-118` handles five types: 0 (knife-edge age), 1 (sigmoid size via l50/l75 OR l50/slope), 2 (Gaussian), 3 (log-normal), and -1 (default 1.0). Tests in `tests/test_engine_selectivity.py` cover the sigmoid, Gaussian, log-normal, and knife-edge families.

### 2.5 MPA (Marine Protected Areas) — SHIPPED
`EngineConfig.mpa_zones: list[MPAZone] | None` with per-zone grid, start/end step, and percentage. Applied at `fishing.py:143-152` as `mpa_factor[in_mpa] *= 1.0 - mpa.percentage`. UI MPA overlay shipped too — see `tests/test_overlay_display.py::test_mpa_*`.

### 2.6 Fishery discards — SHIPPED
`EngineConfig.fishing_discard_rate: NDArray[np.float64] | None` loaded from `fisheries.discards.file` at `config.py:853`. Applied at `fishing.py:167` and `fishing.py:244` with `new_n_dead[:, MortalityCause.DISCARDS] += n_discarded`. `MortalityCause.DISCARDS` bucket added to the mortality enum.

---

## Phase 3: Reproduction & Mortality Refinement — STATUS-COMPLETE (2026-04-19)

All seven items shipped. Coverage under `tests/test_engine_phase3.py` (10 tests).

### 3.1 Spawning season normalization — SHIPPED
`reproduction.normalisation.enabled` flag parsed at `config.py:883`. `_load_spawning_seasons` (at `config.py:874`) normalizes each `n_dt_per_year`-sized chunk to sum=1 per year when enabled. Tests: `test_normalized_season_sums_to_one`, `test_unnormalized_season_preserves_raw_values`.

### 3.2 Multi-year spawning season time series — SHIPPED
`_load_spawning_seasons` (at `config.py:874-920`) detects CSV files with `len(values) >= n_dt_per_year` and stores them as `(n_species, n_dt_per_year * n_years)` arrays; `reproduction.py:46-48` indexes by `season_idx = step % n_cols` — correctly handles single- or multi-year shapes.

### 3.3 Time-varying additional mortality — SHIPPED
`EngineConfig.additional_mortality_by_dt: list[NDArray[np.float64] | None] | None` and `additional_mortality_by_dt_by_class: list` carry BY_DT, BY_DT_BY_AGE, and BY_DT_BY_SIZE variants. Parsed at `config.py:989` (`for variant in ["byDt.byAge", "byDt.bySize"]`).

### 3.4 Spatial additional mortality — SHIPPED
`EngineConfig.additional_mortality_spatial: list[NDArray[np.float64] | None] | None` parsed at `config.py:963-968` from `mortality.additional.spatial.distrib.file.sp{i}`.

### 3.5 Egg placement timing — SHIPPED
`reproduction.py:92-95` creates eggs with `cell_x = cell_y = -1` (unlocated); `movement.py:238-246` places them on the next step via `random_patches` fallback when `new_cx[i] < 0`.

### 3.6 nEgg < nSchool edge case — SHIPPED
`reproduction.py:73-74` handles this exactly: `if n_eggs[sp] < n_new: n_new = 1`. One school of small eggs instead of many tiny schools.

### 3.7 `population.seeding.year.max` config key — SHIPPED
Parsed at `config.py:480` into `EngineConfig.seeding_max_step: NDArray[np.int32]`. Tested by `test_seeding_stops_at_configured_year`.

---

## Phase 4: Movement Refinement — STATUS-COMPLETE (2026-04-19)

Both items shipped. Coverage under `tests/test_engine_phase4.py` (4 tests) and `tests/test_engine_rng_consumers.py`.

### 4.1 Random distribution patch constraint — SHIPPED
`movement.py:106-152` implements `build_random_patches` — BFS from a random seed cell, collects `ncell` connected ocean cells per species. `EngineConfig.random_distribution_ncell: NDArray[np.int32] | None` at `config.py:1196-1197`. Applied per-species at `movement.py:238-246`.

### 4.2 Deterministic random seeds — SHIPPED
`movement.randomseed.fixed` → `EngineConfig.movement_seed_fixed: bool`; `stochastic.mortality.randomseed.fixed` → `mortality_seed_fixed: bool` (config.py:1629-1631). Per-species `species_rngs` wired through `movement.py:204`, `mortality.py:1672`, `predation.py:539`. Tested by `test_movement_seed_fixed_false_ignores_species_rngs` and the broader `test_engine_rng_consumers.py` suite.

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

### 7.1 Reconcile dual predation paths — SHIPPED (2026-04-19)
**STATUS:** SHIPPED. Branch `feature/phase71-predation-reconciliation` migrated all 6 test files (test_engine_predation_helpers, test_engine_diet, test_engine_predation, test_engine_background, test_engine_feeding_stages, test_engine_rng_consumers) to the new public `osmose.engine.processes.predation.predation_for_cell()` API and deleted the standalone batch `predation()` orchestrator. Commits `72f80a3..009c781` (Task 1: API + behavior-preserving delegation; Task 2 sub-commits 2a-2f: per-file test migrations plus the 2-school resource-test follow-up; Task 3: batch-function deletion + module docstring cleanup; Task 4: this roadmap entry). Spec at `docs/superpowers/specs/2026-04-19-phase71-predation-reconciliation-design.md`; plan at `docs/superpowers/plans/2026-04-19-phase71-predation-reconciliation-plan.md`. Final suite: 2510 passed / 15 skipped / 41 deselected (unchanged). Ruff clean. Production path (`mortality.mortality()`) untouched.

### 7.2 Pluggable growth classes — SHIPPED (pre-v0.9.0)
Java class-name → dispatch-token mapping at `osmose/engine/config.py:32-36` (`fr.ird.osmose.process.growth.VonBertalanffyGrowth` → `VB`; `GompertzGrowth` → `GOMPERTZ`; plus the short-form `fr.ird.osmose.growth.*` aliases). `growth_class: list[str]` field on `EngineConfig` parsed at `config.py:1631-1633` from `growth.java.classname.sp{i}`. Runtime dispatch at `osmose/engine/processes/growth.py:151` (`expected_length_for_class`). Gompertz-specific parameters parsed conditionally at `config.py:1639-1642`. No roadmap action — header retained for history.

### 7.3 Config validation
**Gap:** Reader accepts ~134 keys; `ParameterRegistry` knows ~42. A naive "warn on unknown key" hook in `EngineConfig.from_dict()` would fire on ~92 legitimate keys (fisheries.*, evolution.*, fishing.rate.*, etc.) that are reader-honored but not schema-registered, creating noise rather than signal.
**Fix (revised):** Two-layer approach. (a) Enumerate a reader-known key-pattern allowlist by introspecting the reader. (b) Union allowlist + schema-registered patterns = known-key set. (c) Warn only on keys outside the union. Optionally gate behind `validation.strict.enabled=true` config flag until confidence grows. **Design needed before implementation** — straightforward warn-on-unmatched would regress UX.
**Effort:** ~1 day including the allowlist introspection pass and a careful test matrix across EEC / Baltic / Bay-of-Biscay configs.

---

## Status Summary — ALL PHASES STATUS-COMPLETE

| Phase | Items | Status | Shipped |
|-------|-------|--------|---------|
| **Phase 1** | 7 fixes + predation architecture | DONE ✓ | pre-v0.9.0 |
| **Phase 2** | 6 fisheries features | DONE ✓ | incremental through v0.9.2 |
| **Phase 3** | 7 reproduction + mortality fixes | DONE ✓ | pre-v0.9.0 |
| **Phase 4** | 2 movement features | DONE ✓ | pre-v0.9.0 |
| **Phase 5** | 7 output features | DONE ✓ | v0.9.0 (SP-4) |
| **Phase 6** | 5 Ev-OSMOSE features | DONE ✓ | v0.9.1 |
| **Phase 7** | 3 code-quality tasks | DONE ✓ | 7.2 pre-v0.9.0, 7.3 v0.9.2, 7.1 v0.9.3 |

**Full standard OSMOSE parity reached.** Ev-OSMOSE (bioenergetic + genetics + economics) also complete.

Evidence: 2510 tests passing. EEC 14/14 parity, Bay of Biscay 8/8. Python faster than Java on every benchmarked config.

---

## Post-parity divergences

OSMOSE-Python preserves Java parity by default but adds the following opt-in
features that have no Java counterpart. These are documented here so any future
parity audit knows to expect a divergence when the corresponding config keys
are set.

### Beverton-Holt / Ricker stock-recruitment (2026-04-28)

- Config: `stock.recruitment.type.sp{i}` ∈ `{none, beverton_holt, ricker}`,
  `stock.recruitment.ssbhalf.sp{i}` (tonnes).
- Default: `type=none` for every species → byte-for-byte equivalent to the
  Java linear formula `n_eggs = sex_ratio · relative_fecundity · SSB · season · 1e6`.
- Rationale: Java OSMOSE has no SR; DE calibration of the linear regime can
  trade off adult vs larval mortality to defeat single-axis biomass bounds
  (verified 2026-04-27 with cod-floor experiment: forcing adult mortality up
  14× moved cod biomass only +8% because larval mortality dropped 24× to
  compensate). Density-dependent recruitment is the structural fix.
- Code: `osmose/engine/processes/reproduction.py:apply_stock_recruitment`

---

## What's next (post-parity)

The parity roadmap is exhausted against standard OSMOSE + Ev-OSMOSE. Future work falls outside this document. Possible directions (no commitment implied):
- **Performance:** further Numba tuning, parallel-simulation throughput, larger-grid memory optimizations.
- **Usability:** UI polish, calibration workflows, scenario comparison tooling.
- **Science extensions:** features not in upstream OSMOSE — new mortality causes, alternative growth models, richer genetics.
- **Test infrastructure:** property-based testing, mutation testing, CI performance gates.
- **Java-side contributions:** upstream patches matching `docs/osmose-master-java-fixes.patch`.

These are brainstorming seeds, not planned work. A concrete next direction needs its own spec.
