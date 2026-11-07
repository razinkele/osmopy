# Temporal and spatial dynamics — deep review against the scientific evidence (2026-11-07)

**Code reviewed:** `master` at `8a874fb4` (v1.8.1). **Production config:** `data/baltic` (24 steps per year).
**Nothing was changed.** This document reports; every fix it implies is a separate decision.

## How the review was done

Three independent reviews ran in parallel, each briefed with the same reference hierarchy and with the issues
already documented in CLAUDE.md and `docs/`:

1. **Temporal implementation** — process order, rate conversions, sub-stepping, ages and stages, growth,
   reproduction seasonality, forcing alignment and output windows.
2. **Spatial implementation** — grid and orientation of every spatial input, movement, cell-level
   interactions, fishing in space, physical forcing per cell.
3. **Baltic parameterization vs the literature** — the resolved inputs (loaded through the engine's own
   loaders), checked against papers retrieved through scite.

**Reference hierarchy for the engine:**
1. the OSMOSE ODD (`~/osmose-reference/osmose-master/doc/odd_des/`, `papers/Osmose-ODD-Description.pdf`);
2. **Java 4.3.3** (`~/osmose-reference/osmose-master/java/`), the version this engine was ported from and
   against which its parity claims (14/14 EEC, 8/8 Bay of Biscay) were measured;
3. Java 4.4.1, only for later changes not already listed in `docs/osmose_4.4_parity.md`.

**Verification.** Every finding below marked ✔ was re-checked by reading both sides at the cited lines,
and S1–S4 by loading inputs through the engine or running the reviewer's falsification test. Findings
marked ◦ rest on the reviewer's reading of cited code and were not independently re-run. No finding was
contradicted on re-check.

**Legend.** *Active* = in effect on the production Baltic config today. *Latent* = no shipped config
triggers it, but it fails silently when one does. *Other configs* = active outside the Baltic.

## Summary

- **The core machinery matches the reference.** These are faithful to Java 4.3.3 and the ODD:
  - the order of processes in each step;
  - how annual rates convert to per-step and per-sub-step rates;
  - the larval mortality pass and mortality sub-stepping, including the egg-retention fix;
  - the growth formula and its predation-success gate;
  - spawning-to-egg arithmetic;
  - map selection and map-movement rules;
  - same-cell predation;
  - the orientation of every Baltic spatial input.
- **Engine deviations active on the Baltic today:**
  - fish length is clipped at L∞ by default, where Java has no cap (T1);
  - the emergency seeding window is each species' own lifespan instead of the longest one (T2);
  - schools of fewer than one fish stay alive (T3);
  - smaller timing and accounting differences (T4–T7).
- **One high-severity engine bug outside the Baltic.** With the "random" distribution method and no
  `ncell`, every new cohort is placed in the grid's corner cell (S1, confirmed by test). That covers the
  Bay of Biscay parity benchmark. Two bundled grids also load with no ocean at all (S2).
- **Two production-input gaps:** 65 of 616 Baltic ocean cells get no plankton or benthos forcing (S3),
  and the Java cross-check arms place seals and cormorants completely differently from Python (S4).
- **The Baltic parameterization departs from the evidence in nine places.** The largest:
  - all adult herring are packed into 10 cells for three months a year to carry a 25% autumn spawning
    share the literature describes as depressed (C1);
  - eastern cod are spread evenly over SD 25–29, but the literature has them concentrated in SD 25 (C2);
  - western cod still uses the combined pre-split spawning season (C3).

  These are configuration choices, not engine defects, and none was acted on.
- **Two documented defects are still present:** GreySeal's full (1.0) accessibility in the Python
  production matrix, and the Bornholm cod-spawning map.
- **Roughly 20 statements in project docs are wrong or stale** (section 7).

---

## 1. Engine deviations — temporal

| # | Deviation | Status | Severity | ✔/◦ |
|---|---|---|---|---|
| T1 | Length clipped at L∞ by default; Java has no cap | Active | Medium | ✔ |
| T2 | Seeding window = each species' lifespan, not the longest | Active | Medium | ✔ |
| T3 | No one-fish floor: sub-1 residue schools stay alive | Active | Medium | ✔ |
| T4 | First-feeding larvae get the maximum starvation rate | Active | Low (Baltic) | ✔ |
| T5 | Same-step newborn eggs appear in outputs | Active (distribution outputs) | Low–medium | ✔ |
| T6 | Fish dying of old age leave SSB one step early | Active | Low | ✔ |
| T7 | Annual mean size and mean TL averaged differently | Active | Low | ✔ |
| T8 | Multi-year LTL and background forcing folded into one year | Latent | Medium | ✔ |
| T9 | By-time-step larval and by-age/size additional mortality never load | Latent | Medium | ✔ |
| T10 | Java mortality/fishing time modifiers ignored with no warning | Latent | Low–medium | ✔ |
| T11 | Unequal-length season files padded, not repeated | Latent | Low | ✔ |
| T12 | Maturity = size AND age; Java uses one or the other | Latent | Low | ✔ |
| T13 | Minor rounding and edge cases | Latent | Negligible | partly |

**T1 — Length is clipped at L∞ by default.**
- **Python:** `species.lmax` defaults to L∞ (`osmose/engine/config.py:485-488`), and growth clips to it
  with `np.minimum(state.length + growth_factor, config.lmax[sp])` (`processes/growth.py:82`).
  `tests/test_engine_eec_compat.py:514-517` pins that default.
- **Java 4.3.3:** defaults `lmax` to `Float.POSITIVE_INFINITY` (`process/GrowthProcess.java:118-121`) and
  uses it only as a gate, `if (school.getLength() < lmax[i]) grow(...)` (`:140`). Java 4.4.1 comments
  even the gate out (`GrowthProcess.java:177`).
- **ODD** (`submodel/growth.rst`): growth is G = λ·ΔL(a)·(S_R − C)/(1 − C), where λ (default 2) "controls
  the maximum length at a given age". The cap is per step; there is no absolute cap.
- **Baltic:** `species.lmax` is absent, so the clip is live for all 9 species. Well-fed schools (S_R above
  about 0.785) lose the top of their size range. That feeds into weight, SSB, fecundity and predator size
  windows. The 2026-09-27 EEC cross-check bounds the aggregate effect (mean size within 1.2×), but the
  large-size tail was never compared.
- **Falsify:** count schools with `length == lmax` per step in a Baltic run; zero means the clip is inert.

**T2 — The default seeding window is each species' own lifespan.**
- **Python:** with `population.seeding.year.max` absent, it uses `lifespan_years × n_dt` per species
  (`config.py:541-542`); used at `processes/reproduction.py:320` and on the bioen path.
- **Java and the ODD:** Java 4.3.3 takes `Math.max` over all species' lifespans and logs "the lifespan of
  the longest-lived species" (`process/ReproductionProcess.java:145-151`); 4.4.1 does the same
  (`:247-258`). ODD `init.rst:39` agrees.
- **Baltic:** the key is absent. Python stops seeding stickleback at step 96, smelt 168, sprat 192,
  herring 288, and five species at 360; Java uses 480 for all nine.
- **Impact:** a stock whose SSB returns to exactly zero between its own lifespan and 20 years is re-seeded
  by Java but not by Python. This matters for collapse and rescue experiments, not for persisting stocks.

**T3 — Schools of fewer than one fish stay alive.**
- **Java 4.3.3** floors instantaneous abundance below one fish to zero (`School.java:583-586`), and
  `isAlive()` reads the floored value (`:461`); 4.4.1 is the same.
- **Python** keeps any school with `abundance > 0` (`state.py:240`) and only clamps at zero
  (`processes/mortality.py:2767`). No sub-one floor exists anywhere in `osmose/engine/`.
- **Impact:**
  - residues of mature fish keep a stock's SSB above zero, which blocks the `ssb == 0` seeding rescue
    Java would fire (this compounds T2);
  - residues also produce the stage-rate outliers and `inf` values documented in CLAUDE.md;
  - CLAUDE.md's "alive while `abundance > 0`, on both engines" is wrong for Java.

**T4 — First-feeding larvae get the maximum starvation rate.**
- **Python** computes the next step's starvation rate for every school, with no age gate
  (`processes/starvation.py:58-67`, called at `mortality.py:2790`). An egg (predation success 0) therefore
  stores M_max, which applies at the next step, the school's first feeding step.
- **Java** gates starvation on `ageDt >= firstFeedingAgeDt` (`Species.java:229-231`), so an egg stores 0
  (`StarvationMortality.java:75-79`, computed before ageing at `MortalityProcess.java:390-394`).
- **Baltic:** M_max is 0.3/yr, so 1.24% of every cohort is lost that Java keeps. The loss grows with M_max.

**T5 — Same-step newborn eggs appear in outputs.**
- **Python** merges new egg schools inside `reproduction()` (`reproduction.py:353`), before
  `_collect_outputs` (`simulate.py:2150`, then `:2202`).
- **Java** adds them to a separate reproduction set and merges it after `indicators.update`
  (`SimulationStep.java:203-213`).
- **Baltic:** the 0.5-yr output cutoff hides them in biomass and abundance, but the by-size and by-age
  distributions gain each step's new eggs in their first bin. Part of CLAUDE.md's "4× by-age vs
  `abundance()` gap" is therefore a term Java does not have.

**T6 — Fish dying of old age leave SSB one step early.**
- **Python** zeroes expired schools in ageing mortality, before reproduction (`processes/natural.py:220-230`),
  so they miss their final SSB contribution.
- **Java** books ageing deaths but adds them back into instantaneous abundance (`School.java:583`), so the
  same fish still count in that step's SSB.
- The bias is small and systematic.

**T7 — Annual mean size and mean TL are averaged differently.**
- **Python** averages the per-step means (`simulate.py:1697-1698`).
- **Java** divides window sums (`output/WeightedSpeciesOutput.java`).
- The two differ whenever abundance changes within the year. Outputs only.

**T8 — Multi-year LTL and background forcing is folded into one year.**
- **Python:** `ResourceState` replaces the declared `species.biomass.nsteps.year` with the file's frame
  count (`resources.py:246`) and maps step → frame within one year (`:281-283`). A 15-year file is read as
  frames 0, 15, 30… every year.
- **Java** reads `(iStepSimu / (ndt / ncPerYear)) % timeLength` (`util/io/ForcingFile.java:278`).
- **Status:** every shipped file has 24 frames. Temperature and oxygen loaders reject a mismatched frame
  count; the LTL and background loaders don't.

**T9 — By-time-step larval and by-age/size additional mortality can never load.**
- The lookups use mixed-case keys (`config.py:1602`, `byDt.byAge`/`byDt.bySize`; `:1618`, `bytDt`/`byDt`).
  The reader lowercases every key (`config/reader.py:174`), so they never match.
- **Consequence:** a Java-valid config that uses the by-time-step larval series loses all larval
  mortality.
- **Status:** the 2026-08 case fix covered only the fishing keys (`:1652-1657`). No shipped config uses
  these keys.

**T10 — Java's time modifiers are silently ignored.** These keys are never read:
- `mortality.additional(.larva).rate.seasonality/.multiplier/.shift`;
- `mortality.fishing.season.distrib.file.sp#`;
- `fisheries.rate.base.shift/.log`;
- time-varying accessibility and catchability files.

`SeasonTimeSeries` exists but is unused. In the default validation mode, unknown keys produce one
info-level count and nothing else (`config_validation.py:572-575`).

**T11 — Unequal-length season files.**
- A season file shorter than one year is silently ignored, and the species gets a uniform season
  (`config.py:1074`).
- Shorter series are padded with 1/ndt instead of repeated (`:1129-1131`).
- Java rejects malformed series and repeats each one independently (`SingleTimeSeries.java`).
- All Baltic files have 24 rows.

**T12 — Maturity rule.**
- **Python** requires size AND age (`reproduction.py:297-301`).
- **Java 4.3.3** uses size if `species.maturity.size` is set, otherwise age (`Species.java:164-172`, `:344`).
- **Status:** both Python thresholds default to 0, so the two diverge only when both keys are set, or for
  age-only configs whose age × ndt is fractional. Every shipped config sets size only.

**T13 — Minor (◦ except a).**
- (a) Lifespan is truncated to steps where Java rounds; inert on Baltic, where all lifespans are whole years.
- (b) Python's banker's rounding vs Java's `Math.round` for map ages.
- (c) No egg-size floor on the threshold length.
- (d) An explicitly set `lmax` clips in Python where Java stops growth.
- (e) Python renormalises v4 fishing seasonality.
- (f) The larval pass also kills out-of-domain eggs; see S12.

## 2. Engine deviations — spatial

| # | Deviation | Status | Severity | ✔/◦ |
|---|---|---|---|---|
| S1 | "random" method without `ncell`: new cohorts placed in the corner cell | **Other configs** (incl. the Bay of Biscay parity benchmark) | **High** | ✔ test |
| S2 | `data/minimal` and `data/eec` grids load with zero ocean cells | Other configs (demos and fixtures) | Medium–high | ✔ loaded |
| S5 | Unresolved LTL or O₂ file silently drops the forcing; only the first resource file is read | Latent | Medium | ✔ partly |
| S6 | MPAs: no effort redistribution, end year exclusive, fraction ignored | Latent | Medium–low | ◦ |
| S8 | Only `fisheries.movement.file.map0` is read and applied to every species | Latent (Baltic uses one all-1 map) | Low | ◦ |
| S9 | No shape, row-count or time checks on spatial inputs; resolver falls back to other configs' files | Latent | Low | ✔ resolver |
| S10 | Incoming-flux schools dropped on a random ocean cell | Latent | Low | ◦ |
| S11 | Uncovered map slots only warn; movement-method value is case-sensitive | Latent | Low | ◦ |
| S12 | Larval mortality also kills out-of-domain eggs | Latent | Low | ◦ |

**S1 — "random" distribution: every new cohort starts in the corner.**
- **Python:**
  - Eggs are created unlocated at (−1, −1) (`processes/reproduction.py:237-241`).
  - Unlocated schools are placed only by the `ncell` patch block (`processes/movement.py:284`), which is
    empty when `ncell` is absent (`:163`, `:181`).
  - They therefore go straight into `random_walk`, where `clip(−1 + dx, 0, nx−1)` can only land in columns
    0…r−1 (`:224-235`). If that corner is land, the school stays unlocated for good.
- **ODD and Java:** the ODD (`submodel/movement.rst:40,52-53`) and Java 4.3.3
  (`RandomDistribution.java:97,111-112`) place unlocated schools uniformly over the whole ocean domain.
- **Test** (`data/examples`, Bay of Biscay; 5 seeds × 200 unlocated schools per species, through the real
  `movement()`):
  - all 8 species use the random method and `ncell` is unset;
  - range-1 species land **200 of 200 in cell (0, 0)** on every seed;
  - range-2 species land only in the 2×2 corner, and range-3 only in the 3×3 corner.
- **Located schools walk differently too.** The displacement is drawn from ±(the largest range of any
  species), then clipped to the school's own range. For a range-1 species in Bay of Biscay (largest range
  3) a zero step has probability 1/7 per axis instead of 1/3, so the school stays in its cell with
  probability 1/49 against about 1/9 in Java. Schools pile up on grid edges, and a draw onto land means
  staying put rather than choosing among sea cells.
- **Affected configs:** `data/examples` (the 8/8 Bay of Biscay parity benchmark), `data/examples_bioen`
  (C3 Gate B), `data/examples_433_orig`, `data/eec`, `data/minimal`. Baltic and EEC-full use maps and are
  unaffected.
- **What it does to the parity claim:** the Bay of Biscay "8/8 within one order of magnitude" was measured
  with every cohort spawning in one corner cell. The tolerance is wide enough to pass anyway; nothing
  tighter has been tested on that config. It may also bear on the unexplained cross-engine mean-size gap
  in `docs/diagnostics/c3_gate_b_cross_engine.md` (hypothesis only).
- **Test coverage:** none. `tests/test_engine_phase4.py` exercises only the `ncell`-set path, and
  `docs/parity-roadmap.md` lists §4.1 as "SHIPPED".

**S2 — Two bundled grids have no ocean.**
- `data/minimal` and `data/eec` store ocean as 0 and land as −99. The engine (like Java,
  `NcGrid.java:177`) treats `mask > 0` as ocean, so both load with **0 ocean cells**
  (minimal 0/100, EEC 0/180; engine-loaded).
- Combined with S1, their eggs never get a cell, and no cell-level process (predation, starvation,
  fishing, adult additional mortality) ever runs. Only the larval pass and ageing do.
- This, not the size windows, is why `data/minimal` "never predates". CLAUDE.md gives the other cause;
  `tests/test_jit_determinism.py` already states this one.
- The UI demo list offers both configs as runnable.

**S5 — Missing forcing files fail silently.**
- An unresolved LTL file leaves `_forcing_data` as `None`, and every resource then gets zero K with no
  error (`resources.py:243-246`, `:278`).
- Only the first resource's file is ever opened (`:214-220`).
- An unresolved oxygen file silently disables the O₂→benthos coupling, even with
  `ltl.oxygen.benthos.enabled=true`.
- Java stops with an error (`util/io/ForcingFile.java:173-177`).
- All Baltic files resolve today.

**S6, S8, S10, S11, S12 (◦)** are latent; details are in the reviewer's report.
- **S6:** Python's MPAs don't redistribute effort, treat the end year as exclusive and use a single
  scalar fraction. They also apply in v4 fisheries mode, which Java ignores (`MPA.java`,
  `FishingMortality.setMPA`).
- **S8:** fleet binding, per-map steps/years and normalisation (`FisheryMapSet.java`) are absent. The
  #123 validator lets `map1..N` pass as read.
- **S10:** Java leaves incoming-flux schools unlocated for map placement.
- **S11:** Java aborts on an uncovered map slot or an unknown method. Python warns or silently places
  nothing.
- **S12:** Java's egg pass covers ocean cells only.

**S9 — Resolver fallback.** `path_resolution.py:62-63` searches the config directory, then `.`, then
`data/examples`, then `sorted(data/*/)`. Because '-' sorts before '/', `data/baltic-fine/` is searched
before `data/baltic/`. A file missing from a staged Baltic directory would therefore be taken from the
fine-grid config and regridded without a word. Spatial inputs also get no shape or row-count checks:
a movement CSV with too few lines leaves uninitialised rows.

## 3. Production inputs

**S3 — 65 of the 616 Baltic ocean cells have no plankton or benthos.** ✔
- `baltic_ltl_biomass.nc` (24, 40, 50) has the same north-first orientation as the grid.
- 65 ocean cells (10.6%) are exactly 0 in all six groups and all 24 frames. All are coastal-edge cells.
- 3 land cells carry forcing that is lost.
- **Overlap with habitat** (reviewer): 10–19% of the cells in most species' maps, e.g. herring adult
  65/593 and perch juvenile 9/47.
- **Origin:** the mask rebuild in `d61ea1c1` (912 → 616 cells) added these coastal cells, but the LTL file
  was not regenerated. `docs/claude-memory/project_baltic_grid_orientation.md:29,41` predates that and now
  calls them intentional or negligible.
- **Effect:** both engines read the same file, and the current calibration was fitted with the gap in place.

**S4 — The Java cross-check arm places seals and cormorants completely differently from Python.** ✔
- **Python** (like Java 4.3.3) puts one school per size class in **every** ocean cell, carrying that
  cell's own forcing (`background.py:427-443, 485-487`). The forcing field is spatially structured: seals
  range 0.7–18 t per cell.
- **Java 4.4.1** creates `simulation.nschool` schools per class and places them with a map.
- **Our staging** (`osmose/java_background_staging.py`) sets `nschool = 10`, writes the domain-total
  series and gives a uniform all-sea map.
- **Result:** the Java arm has 10 seal schools of 292.5 t and 10 of 157.5 t (4,500 t × 0.65 or 0.35 ÷ 10),
  wandering uniformly, so the forcing's spatial pattern is discarded. This matches the 292.5 t chunks in
  `docs/baltic_java_crosscheck_2026-11-06.md` §3.
- **Docs:** `docs/osmose_4.4_parity.md` doesn't record this.
- **Impact:** every Python-vs-Java comparison of seal or cormorant predation sets a diffuse per-cell
  predator against 20 concentrated schools per species.

**S7 — Herring and sprat out-of-domain mortality never applies.** ✔
- `baltic_param-out-mortality.csv` sets 0.05 and 0.08 per year.
- The movement config has no null maps, and the reviewer's probe found no uncovered slots, so `is_out` is
  never set and the rates have exactly zero effect.
- The docs say they are active and calibrated (`docs/baltic-fish-lifecycle.md:253,328`,
  `docs/baltic_example.md:252`). No calibration script references the keys.

## 4. Baltic parameterization vs the literature (configuration, not engine)

These compare the resolved Baltic inputs with published evidence. They are calibration and design
decisions for the maintainer; **none was acted on**.

**Provenance.**
- I re-checked the model side of C1–C3 through the engine loader.
- I re-retrieved two quotes myself through scite: Eero et al. 2012 (full text) and Bekkevold et al. 2016
  (abstract).
- Every other quote rests on the reviewer's retrieval; the reviewer checked editorial notices on every DOI
  and found none.

| # | Finding | Severity | Status |
|---|---|---|---|
| C1 | Herring autumn spawning over-represented, and all adults packed into 10 cells | High | Inconsistent |
| C2 | Eastern cod spread evenly over SD 25–29, not concentrated in SD 25 | High | Inconsistent |
| C3 | Western cod uses the combined pre-split spawning season | Medium–high | Inconsistent |
| C4 | No eastern cod in SD 24 (the mixing zone); a third of western cod cells in the Kattegat/Skagerrak | Medium | Inconsistent |
| C5 | Flounder modelled as one southern deep-basin spawner | Medium | Inconsistent (space), partial (timing) |
| C6 | Perch absent south of 57 °N; pikeperch absent from the Archipelago Sea | Medium | Inconsistent |
| C7 | Sticklebacks spawn up to three times; the evidence says once | Low–medium | Inconsistent |
| C8 | Cormorants present year-round everywhere | Low–medium | Inconsistent |
| C9 | Eastern cod maturity mixes a current L50 with pre-collapse growth | Low–medium | Partial |
| C10 | O₂→benthos half-saturation at 1.92 mg/L | Low | **Consistent**, with resolution caveats |
| C11 | Fishing uniform in space and season | Low | Partial; defensible because annual F is calibrated |
| C12 | One-step (15-day) egg stage | Low | Partial: OSMOSE's design (Python = Java), not an osmopy deviation |
| C13 | Phytoplankton split uniform in space; the "dinoflagellate" summer peak is wrong | Low | Inconsistent |

**C1 — Herring autumn spawning.**
- **Model:**
  - 25% of herring eggs are released in steps 16–21 (September–November).
  - For those six steps, **every adult herring (ages 1–13) is placed on the 10-cell autumn map**
    (`baltic_param-movement.csv:88-93`, map 25; 5 cells in the Gulf of Riga, 5 around Rügen).
  - The model has one herring population, so the spring-spawning majority is moved there too.
- **Evidence:**
  - Bekkevold et al. (2016): *"autumn spawners have faced strong depression for decades"*
    (abstract, re-retrieved).
  - MacKenzie & Ojaveer (2018) and Ojaveer et al. (2015) describe the Gulf of Riga autumn stock as
    collapsed since the 1960s–70s.
  - Eero et al. (2012): the largest herring and sprat concentrations are in SD 28–32.
- **Unsourced:** the 20–30% autumn share claimed in the config comment was not found in any retrieved
  source (section 6).
- **Why it matters:** herring is the dominant fish biomass, and the packing sets its predation and
  overlap with percids and stickleback for a quarter of the year.

**C2 — Eastern cod distribution.**
- **Model:** 15% of adult-map cells and 5% of spawning-map cells fall in SD 25 (engine-loaded, SD 25 box
  54.5–56.2 °N × 14–18 °E). Occupancy within a map is approximately uniform.
- **Evidence:** Eero et al. (2012), full text re-retrieved: *"Thus, most of the adult cod is currently
  concentrated in SD 25"*, and *"SD 25 has been basically the only area supporting successful reproduction
  of the eastern Baltic cod since the mid-1980s."* Bartolino et al. (2017) and Orio et al. (2019) report
  the same contraction.
- **Why it matters:** this is the stock the RV gate controls. Its prey overlap and hypoxia exposure follow
  this footprint.

**C3 — Western cod spawning season.**
- **Model:** cod_west spawns March–August with its peak in late May–June
  (`reproduction-seasonality-sp0.csv`, header "Cod"). The config comment reads *"Combined Mar-Aug covers
  both stocks"*.
- **Evidence:** Bleil et al. (2009) and Helmerson et al. (2023) put western Baltic spawning in
  February–April and eastern in June–August. Larvae therefore appear 2–3 months late relative to the
  plankton cycle.

**C4–C13.**
- **C4:** eastern cod have no cells west of 15.4 °E, although SD 24 is the documented mixing zone
  (Mion et al., 2022).
- **C5:** flounder's single southern spawning map moves northern adults south every February and never
  places a recruit north of 58.65 °N. That contradicts the demersal *P. solemdali* spawning in northern
  coastal waters (Momigliano et al., 2017, 2019).
- **C6:** the percid maps leave out documented perch coasts south of 57 °N and pikeperch in the
  Archipelago Sea (Olsson, 2019; Kokkonen et al., 2019).
- **C7:** stickleback live 4 years and mature at about 0.9 years, against mostly one spawning season at
  age 2+ (Candolin & Voigt, 2020).
- **C8:** cormorants are constant everywhere, against seasonal migration out of the Baltic
  (Tverin et al., 2021).
- **C9:** eastern cod L50 = 22 cm with pre-collapse growth implies maturity at 1.3 years (Svedäng et al., 2024).
- **C10:** the Hill c50 of 1.92 mg/L sits on the mean LC50 of about 2 mg/L (Vaquer-Sunyer & Duarte, 2008).
- **C11:** fishing is spatially and seasonally uniform.
- **C12:** the egg stage lasts one step (15 days). Cod eggs take 10–16 days to hatch (Geffen et al., 2006);
  sprat match.
- **C13:** the phytoplankton split ignores the regional dinoflagellate-dominated spring blooms
  (Klais et al., 2011).

**Consistent with the evidence:**
- spawning timing of eastern cod, herring (spring), sprat, smelt and stickleback, and perch (partially:
  April is omitted);
- the RV thresholds (≥ 11 PSU, ≥ 2 ml/L O₂; Köster et al., 2005) and their restriction to eastern cod;
- the adult salinity ramp (3–6 PSU); its config comment's rationale cites an egg threshold;
- the seasonal and spatial pattern of bottom O₂.

## 5. Previously documented issues that are still present

| Issue | Documented in | State at `8a874fb4` |
|---|---|---|
| GreySeal resolves to −1, so it gets **full (1.0) accessibility** to every focal species in the Python production matrix (20× the 0.05 cap) | CLAUDE.md (2026-09-13), accessibility memory | **Unfixed.** `data/baltic/predation-accessibility.csv` still has no GreySeal column and no GreySeal or Cormorant prey row; last changed 2026-07-28 (`0fc6c49e`). Java arms meanwhile get 0.3–0.4 from `BG_ACCESS`, so the seal accessibility coefficient differs 2.5–3.3× between the engines. Realised predation also depends on ingestion caps and on placement (S4). |
| Bornholm Deep assigned to cod_west; cod_east cannot spawn there | `docs/baltic_cod_bornholm_spawning_defect_2026-08-10.md` | **Unfixed.** The cod maps have not changed since `c278af3c` (2026-07-25). `cod_east_spawning` has 0 cells in the Bornholm band; `cod_west_spawning` has 12. |

## 6. Checked and consistent

- **Step order:** incoming flux → reset → LTL update → movement → mortality → growth or bioen → ageing →
  reproduction → outputs → removal and merge (`simulate.py:2011-2228` vs `SimulationStep.java:155-215`).
- **Rate conversions:**
  - additional and starvation mortality ÷(ndt·subdt);
  - maximum ingestion per step;
  - v4 fishing base × seasonality ÷ subdt;
  - out-of-domain mortality ÷ ndt once per step;
  - the larval pass at the full per-step rate, once per cohort;
  - the per-year → per-step larval conversion only for 4.4+ sources.
- **Sub-stepping:**
  - per-cell school and cause shuffles;
  - eggs released 1/subdt per sub-step (the 2026-06-22 "egg-retention no-op" is fixed);
  - predation success is the mean over sub-steps.
- **Ages and stages:** egg = first step; ageing at `age_dt >= lifespan − 1`; newborns stay at age 0.
- **Growth:** increment from the VB curve, gated on predation success; eggs and out-of-domain schools get
  the mean increment, as in Java.
- **Reproduction:** egg formula, default 1/ndt season, normalisation flag, one school when eggs are fewer
  than schools.
- **Yield:** post-growth weight in **both** engines, so the 2026-06-22 deep review's yield item is not a
  deviation from Java.
- **Grid and orientation:**
  - the land rule `mask > 0` matches Java;
  - CSV maps are flipped like Java's `GridMap`, and every Baltic NetCDF is north-first like the grid;
  - 0 positive map cells on land; 0 uncovered (age, step) slots.
- **Map movement:** age/step/year map selection, the same-map test, rejection placement and the
  random-walk accessible set all match `MapDistribution`/`MapSet`.
- **Cells:** same-cell predation, LTL tonnes per cell, background per-cell placement (as in 4.3.3),
  physical forcing per cell.

## 7. Documentation that is wrong or stale

| Where | Statement | Correction |
|---|---|---|
| CLAUDE.md:128 | residue schools "alive while `abundance > 0`, on both engines" | Java floors under one fish to zero (`School.java:583-586`) (T3) |
| CLAUDE.md:136 | `data/minimal` never predates because of `nresource = 0` and non-overlapping size windows | Its grid has zero ocean cells, so no cell process runs (S2) |
| CLAUDE.md:122 | the by-age vs `abundance()` gap "is the cutoff" | Partly also same-step eggs, which Java excludes (T5) |
| `docs/parity-roadmap.md` | §1.1 (Java also checks age), §1.3 (Java caps at lmax), §3.2 (season files handled), §3.3 (by-time-step mortality SHIPPED), §4.1 (random distribution SHIPPED), §2.5 (MPA SHIPPED), :88/:94 (v4 fisheries complete) | T12, T1, T11, T9, S1, S6, S8 |
| `docs/baltic_java_crossengine_fidelity_2026-07-30.md:737-755` | Java's seeding default "MATCHES Python" (per-species) | Java uses the longest lifespan; the bytecode reading skipped the `Math.max` loop (T2) |
| `osmose/engine/config.py:533-535` (comment) | "Python's np.full broadcast is correct parity" | True only when the key is set; the default branch is per-species (T2) |
| `docs/osmose_4.4_parity.md` | background differences listed as multiplier and units only | Java arms also place seals and cormorants as 20 wandering schools (S4) |
| `docs/baltic-fish-lifecycle.md:253,328`; `docs/baltic_example.md:252` | herring and sprat out-of-domain mortality active and calibrated | Inert in production (S7) |
| `docs/claude-memory/project_baltic_grid_orientation.md:29,41` | LTL-less mask cells intentional or negligible | 65 ocean cells after the rebuild (S3) |
| `data/baltic/baltic_param-movement.csv` comments | "Smelt: Steps 4-11 = Feb-May"; Bleil 2009 cited for Bornholm "May-Aug" | Steps 4–11 are March–June; Bleil gives June–September with a June–August peak |
| Salinity-gate config comment | justifies the adult 3–6 PSU ramp with the ≥ 11 PSU egg threshold | Separate quantities (C, consistent section) |
| `docs/baltic-fish-lifecycle-scientific-review.md` | doi:10.1093/icesjms/fsv103 as "Rajasilta et al. 2015"; doi:10.1111/jfb.15811 as "Polte et al. 2023"; doi:10.1016/j.fishres.2012.08.002 as "Haslob 2012" | Ojaveer et al. (2015); Ory et al. (2024); Haslob et al. (2013) |
| Recruitment and percid literature reviews | doi:10.1007/s10750-019-04008-z as "Olin et al. 2019"; doi:10.1371/journal.pone.0286247 as "Orio et al. 2023"; doi:10.1007/BF01986342 as "Wieland et al. 1994"; doi:10.1093/icesjms/fsaf197 mis-titled; doi:10.1139/cjfas-2015-0033 as "Salmi et al. 2015" | Kokkonen et al. (2019); Eero et al. (2023); Nissling (1991); "Recruitment mechanisms of Atlantic cod … extreme low-salinity environment"; Heikinheimo & Rusanen (2016) |
| `docs/baltic_percid_low_salinity_refuge_literature_review_2026-07-04.md` | a Bergström et al. (2025) quote ("no evidence that extremely low salinity provides predator refuge") | Not found in the retrieved abstract; unverified |

The citation corrections come from the literature reviewer's scite retrievals; I did not re-retrieve them.

## 8. Not verified, and limits

- **Magnitudes.** No simulation was run, so none of the effects is quantified. T1, T2, T3, S1, S3 and C1–C2
  most need a measured A/B before any fix.
- **Java behaviour** comes from reading source; the jars were not run, apart from the earlier S4 data.
- **Unsourced literature claims:**
  - the 20–30% herring autumn share;
  - the benthos seasonal amplitude;
  - pikeperch spawning timing;
  - the current sprat spawning distribution;
  - cod egg development at Baltic temperatures.
- **Not reviewed:** bioen timing internals, Gompertz growth, economics, and stage-structured
  accessibility.

## References

Only papers retrieved through scite during this review. **Bold** entries were re-retrieved by me; the rest
rest on the literature reviewer's retrieval (all checked for editorial notices; none found).

- **Bekkevold, D., Gross, R., Arula, T., Helyar, S., & Ojaveer, H. (2016).** Outlier loci detect intraspecific biodiversity amongst spring and autumn spawning herring across local scales. *PLOS ONE, 11*(4), e0148499. https://doi.org/10.1371/journal.pone.0148499
- **Eero, M., Vinther, M., Haslob, H., Huwer, B., Casini, M., Storr-Paulsen, M., & Köster, F. (2012).** Spatial management of marine resources can enhance the recovery of predators and avoid local depletion of forage fish. *Conservation Letters, 5*(6), 486–492. https://doi.org/10.1111/j.1755-263x.2012.00266.x
- Bartolino, V., et al. (2017). Spatio-temporal dynamics of a fish predator: Density-dependent and hydrographic effects on Baltic Sea cod population. *PLOS ONE, 12*(2), e0172004. https://doi.org/10.1371/journal.pone.0172004
- Bleil, M., Oeberst, R., & Urrutia, P. (2009). Seasonal maturity development of Baltic cod in different spawning areas. *Journal of Applied Ichthyology, 25*(1), 10–17. https://doi.org/10.1111/j.1439-0426.2008.01172.x
- Candolin, U., & Voigt, H. R. (2020). Population growth correlates with increased fecundity in three-spined stickleback populations in a human-disturbed environment. *Aquatic Sciences, 82*(2). https://doi.org/10.1007/s00027-020-0695-3
- Geffen, A. J., Fox, C. J., & Nash, R. D. (2006). Temperature-dependent development rates of cod *Gadus morhua* eggs. *Journal of Fish Biology, 69*(4), 1060–1080. https://doi.org/10.1111/j.1095-8649.2006.01181.x
- Helmerson, C., Weist, P., Brieuc, M. S. O., et al. (2023). Evidence of hybridization between genetically distinct Baltic cod stocks during peak population abundance(s). *Evolutionary Applications, 16*(7), 1359–1376. https://doi.org/10.1111/eva.13575
- Klais, R., Tamminen, T., Kremp, A., Spilling, K., & Olli, K. (2011). Decadal-scale changes of dinoflagellates and diatoms in the anomalous Baltic Sea spring bloom. *PLOS ONE, 6*(6), e21567. https://doi.org/10.1371/journal.pone.0021567
- Kokkonen, E., Heikinheimo, O., Pekcan-Hekim, Z., & Vainikka, A. (2019). Effects of water temperature and pikeperch (*Sander lucioperca*) abundance on the stock–recruitment relationship of Eurasian perch (*Perca fluviatilis*) in the northern Baltic Sea. *Hydrobiologia, 841*(1), 79–94. https://doi.org/10.1007/s10750-019-04008-z
- Köster, F., Möllmann, C., Hinrichsen, H., et al. (2005). Baltic cod recruitment – the impact of climate variability on key processes. *ICES Journal of Marine Science, 62*(7), 1408–1425. https://doi.org/10.1016/j.icesjms.2005.05.004
- MacKenzie, B. R., & Ojaveer, H. (2018). Evidence from the past: exploitation as cause of commercial extinction of autumn-spawning herring in the Gulf of Riga, Baltic Sea. *ICES Journal of Marine Science, 75*(7), 2476–2487. https://doi.org/10.1093/icesjms/fsy028
- Mion, M. M., Griffiths, C. A., Bartolino, V., et al. (2022). New perspectives on Eastern Baltic cod movement patterns from historical and contemporary tagging data. *Marine Ecology Progress Series, 689*, 109–126. https://doi.org/10.3354/meps14047
- Momigliano, P., Jokinen, H., Calboli, F. C. F., Aro, E., & Merilä, J. (2019). Cryptic temporal changes in stock composition explain the decline of a flounder (*Platichthys* spp.) assemblage. *Evolutionary Applications, 12*(3), 549–559. https://doi.org/10.1111/eva.12738
- Momigliano, P., Jokinen, H., Florin, A., Norkko, A. M., & Merilä, J. (2017). Extraordinarily rapid speciation in a marine fish. *Proceedings of the National Academy of Sciences, 114*(23), 6074–6079. https://doi.org/10.1073/pnas.1615109114
- Ojaveer, H., Tomkiewicz, J., Arula, T., & Klais, R. (2015). Female ovarian abnormalities and reproductive failure of autumn-spawning herring (*Clupea harengus membras*) in the Baltic Sea. *ICES Journal of Marine Science, 72*(8), 2332–2340. https://doi.org/10.1093/icesjms/fsv103
- Olsson, J. (2019). Past and current trends of coastal predatory fish in the Baltic Sea with a focus on perch, pike, and pikeperch. *Fishes, 4*(1), 7. https://doi.org/10.3390/fishes4010007
- Orio, A., Bergström, U., Florin, A., Lehmann, A., Šics, I., & Casini, M. (2019). Spatial contraction of demersal fish populations in a large marine ecosystem. *Journal of Biogeography, 46*(3), 633–645. https://doi.org/10.1111/jbi.13510
- Svedäng, H., Hornborg, S., & Grimvall, A. B. (2024). Centurial variation in size at maturity of Eastern Baltic cod (*Gadus morhua*) mirrors conditions for growth. *Ecology and Evolution, 14*(10). https://doi.org/10.1002/ece3.70382
- Tverin, M., Granroth, J., Tang, P., Pihlström, H., & Käkelä, R. (2021). Adipose tissue fatty acids suggest spatial and temporal dietary differences in great cormorants of the Baltic Sea area. *Journal of the Marine Biological Association of the United Kingdom, 101*(1), 199–213. https://doi.org/10.1017/s0025315420001320
- Vaquer-Sunyer, R., & Duarte, C. M. (2008). Thresholds of hypoxia for marine biodiversity. *Proceedings of the National Academy of Sciences, 105*(40), 15452–15457. https://doi.org/10.1073/pnas.0803833105

Full reviewer reports, probe scripts and verification worksheets are in the session scratchpad
(`review_temporal/`, `review_spatial/`, `review_science/`).
