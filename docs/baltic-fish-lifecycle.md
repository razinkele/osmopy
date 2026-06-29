# The Fish Life Cycle in the Baltic OSMOSE Example

> How the Baltic Sea configuration describes the full life cycle of its eight focal fish
> species — egg → larva → juvenile → adult — and how the OSMOSE engine realizes each phase.
> Sources: `data/baltic/baltic_param-*.csv` (the configuration) and `osmose/engine/` (the
> mechanics). Config version `osmose.version = 4.3.3`.

---

## 1. What the Baltic example is

The Baltic configuration is an individual-based, spatially-explicit, multi-species model of the
central/eastern Baltic Sea. It tracks **8 focal fish species** as super-individuals ("schools")
on a 2-D grid, feeding on each other and on **6 lower-trophic-level (LTL) resource groups**
(Diatoms, Dinoflagellates, Micro-/Meso-/Macro-zooplankton, Benthos).

| idx | Species | Role |
|-----|---------|------|
| sp0 | Cod (*Gadus morhua*) | apex piscivore |
| sp1 | Herring (*Clupea harengus membras*) | pelagic planktivore (spring + autumn spawner) |
| sp2 | Sprat (*Sprattus sprattus*) | pelagic planktivore (key forage fish) |
| sp3 | Flounder (*Platichthys flesus*) | demersal benthivore |
| sp4 | Perch (*Perca fluviatilis*) | coastal meso-predator |
| sp5 | Pike-perch (*Sander lucioperca*) | coastal piscivore |
| sp6 | Smelt (*Osmerus eperlanus*) | semi-pelagic forage fish |
| sp7 | Stickleback (*Gasterosteus aculeatus*) | small coastal planktivore |

**Time resolution:** `simulation.time.ndtPerYear = 24` (each timestep ≈ half a month),
`simulation.time.nyear = 50` → 1 200 timesteps. Each species is represented by 30–60 schools
(`simulation.nschool.spN`). Mortality is resolved on `mortality.subdt = 10` sub-steps per timestep.

### The four life stages, and how the model represents them

OSMOSE has **no explicit "stage" label** on a school. A school is a record in a
Structure-of-Arrays (`osmose/engine/state.py:30-89`) holding `age_dt`, `length`, `weight`,
`abundance`, `is_egg`, `feeding_stage`, position, etc. The four biological stages are *derived*
from those numbers:

| Stage | Condition in the engine | Effect |
|-------|-------------------------|--------|
| **Egg** | `is_egg = True` ⇔ `age_dt < first_feeding_age_dt` (`reproduction.py:172-192`) | Unlocated (cell −1), withheld from the prey pool, exempt from predation/starvation/fishing |
| **Larva** | feeding, but young: `age_dt ≥ first_feeding_age_dt` and below the growth-threshold age (0.5 yr) | Eats zooplankton, very high mortality, linear-segment growth |
| **Juvenile** | grown past the larval phase but `length < maturity_size` | Full feeding/predation/growth; immature |
| **Adult** | mature ⇔ `length ≥ maturity_size AND age_dt ≥ maturity_age_dt` (`reproduction.py:102-106`) | Contributes to spawning-stock biomass (SSB); reproduces |

In the Baltic config **maturity is size-only** — `species.maturity.size.spN` is set, but
`species.maturity.age.spN` is *absent* (defaults to 0), so the age clause is always satisfied and
a fish matures the instant it reaches its maturity length. There is also **no explicit
first-feeding-age parameter**; eggs are created with `first_feeding_age_dt = 1`, so an egg cohort
spends exactly **one timestep** as an egg before becoming a feeding larva.

### What happens to a school each timestep

The per-timestep loop (`osmose/engine/simulate.py:1502-1707`) processes, in order:

1. **Incoming flux** — inject migrating schools.
2. **Reset** per-step accumulators; **update LTL resources** for the step.
3. **Movement** — place each school by its species/age/season map (this is where freshly-spawned
   eggs, created at cell −1, get located).
4. **Mortality** — the interleaved sub-timestep loop: an egg larval-mortality pre-pass, then
   predation + starvation + additional + fishing.
5. **Growth** — increment length (gated von Bertalanffy) and recompute weight/biomass.
6. **Aging mortality** — remove schools older than the species lifespan.
7. **Reproduction** — compute SSB, apply the stock-recruitment relationship, create new egg
   schools, and **increment every school's age**.
8. **Collect outputs.**

So a single calendar half-month moves every fish: it relocates, runs the gauntlet of predation
and fishing, grows a little, ages, and (if mature and in season) spawns.

---

## 2. Stage 1 — Egg and larva

### Egg creation

Eggs are produced in `reproduction()` (`osmose/engine/processes/reproduction.py:84-216`) from the
**spawning-stock biomass (SSB)** of each species:

```
n_eggs  =  sex_ratio · relative_fecundity · SSB · season_factor · 10^6
```

(`reproduction.py:138-145`; the 10⁶ converts SSB from tonnes to grams, because fecundity is
*eggs per gram of mature female*). The terms come straight from the config:

- **`species.sexratio.spN` = 0.5** for all species (half the SSB is female).
- **`species.relativefecundity.spN`** (eggs · g⁻¹): Sprat 1200, Smelt 1000, Herring 600,
  Stickleback 600, Cod 500, Flounder 400, Perch 350, Pike-perch 300. Small forage fish are
  vastly more fecund per gram than the large piscivores.
- **`season_factor`** — the fraction of the year's eggs released this timestep (see §5).

The resulting eggs are split across `simulation.nschool.spN` new schools, each created with
`length = species.egg.size.spN`, `is_egg = True`, `age_dt = 0`, `first_feeding_age_dt = 1`, and
**`cell_x = cell_y = −1`** (unlocated; the next movement step places them) — `reproduction.py:154-187`.

**Egg sizes** (`species.egg.size.spN`, cm) and weights (`species.egg.weight.spN`, g):

| | Cod | Herring | Sprat | Flounder | Perch | Pike-perch | Smelt | Stickleback |
|--|----|----|----|----|----|----|----|----|
| egg size (cm) | 0.15 | 0.12 | 0.10 | 0.13 | 0.20 | 0.15 | 0.09 | 0.15 |
| egg weight (g) | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | **0.0005** | 0.001 |

Smelt has the lightest, smallest egg (a relic of the 2026-04 "whitefish→smelt" species swap).

### The first-feeding transition

After the age increment inside reproduction, `is_egg` is recomputed as
`age_dt < first_feeding_age_dt` (`reproduction.py:191-192`). With `first_feeding_age_dt = 1`, a
cohort is an egg for one timestep, then a **feeding larva**. While `is_egg` is true the cohort is
held out of the prey pool (`egg_retained`) and skips predation, starvation, additional and fishing
mortality entirely (the per-school mortality functions early-return for eggs).

### Larval mortality — the recruitment bottleneck

The single most important number for each species' dynamics is its **larval additional mortality**
(`mortality.additional.larva.rate.spN`, yr⁻¹), applied once to each egg cohort in
`larva_mortality()` (`natural.py:103-150`). In the base config:

| | Cod | Herring | Sprat | Flounder | Perch | Pike-perch | Smelt | Stickleback |
|--|----|----|----|----|----|----|----|----|
| larval M (yr⁻¹) | 15.0 | 8.0 | 9.0 | 12.0 | 13.0 | 15.0 | 13.5 | 3.5 |

These are **2–3 orders of magnitude higher** than adult background mortality. The config comments
call this "the primary recruitment-control lever in OSMOSE calibration" — it is not directly
measurable and is the main parameter tuned to make recruitment realistic. (The canonical calibrated
run rewrites these substantially — see §8.)

### Where eggs and larvae live

Eggs are created unlocated and inherit a position from the species' youngest/juvenile movement map
on the next step. Larvae and young juveniles use **broad coastal/nursery maps active all year**
(see §6). Because predation prey-size windows scale with predator size (§4), a larva can physically
only reach zooplankton-sized prey regardless of the accessibility matrix.

---

## 3. Stage 2 — Juvenile

A larva becomes a juvenile as it grows past the larval phase (growth-threshold age 0.5 yr) but
before reaching maturity length. Juveniles are fully exposed to the model's dynamics:

- **They feed** on whatever falls inside their predator/prey size window and the accessibility
  matrix allows — for small juveniles that is overwhelmingly zooplankton (the Mesozooplankton
  resource is accessible to every fish at 0.3–0.8).
- **They are prey.** Small fish sit inside the size-ratio windows of cod, perch and pike-perch, and
  the accessibility matrix routes them as forage: sprat→cod 0.4, herring→cod 0.4, smelt→cod 0.6,
  smelt→pike-perch 0.6, smelt→perch 0.5. Juvenile mortality is **predation-dominated**.
- **Fishing knife-edges in.** Fishing selectivity is age-based (`fisheries.selectivity.type = 0`,
  knife-edge by age `a50`): pelagic herring and sprat are caught from **age 1** (i.e. as late
  juveniles), while demersal/coastal species are unfished until age 2–3. So for the pelagics the
  juvenile stage already carries fishing mortality.
- **Movement.** Juveniles use coastal/nursery maps (e.g. `cod_juvenile` for ages 0–1,
  `flounder_juvenile`/`perch_juvenile`/`pikeperch_juvenile` on coastal grounds for ages 0–2).

### Diet and the ontogenetic shift

There is **no per-species diet-stage threshold** in this config
(`predation.predPrey.stage.threshold.spN = null` for all species). Instead the ontogenetic diet
shift is produced entirely by the **fixed size-ratio window** combined with the predator's own
growth. Each predator can take prey between `1/sizeRatio.max` and `1/sizeRatio.min` of its own
length (`predation.predPrey.sizeRatio.{max,min}.spN`):

| | Cod | Herring | Sprat | Flounder | Perch | Pike-perch | Smelt | Stickleback |
|--|----|----|----|----|----|----|----|----|
| sizeRatio.max (smallest prey) | 50 | 500 | 500 | 100 | 50 | 30 | 500 | 1000 |
| sizeRatio.min (largest prey) | 3.5 | 5.0 | 5.0 | 5.0 | 3.0 | 2.5 | 5.0 | 10.0 |
| max ingestion (g·g⁻¹·yr⁻¹) | 3.5 | 6.0 | 7.0 | 3.0 | 3.5 | 3.5 | 4.0 | 5.0 |

Because the window is fixed in *relative* terms, the *absolute* prey size slides up as the fish
grows. Two niches emerge:

- **Piscivores** (Cod, Perch, Pike-perch) have a low `sizeRatio.min` (2.5–3.5), so adults can eat
  fish up to ~⅓ of their own length. Pike-perch (min 2.5) is the most extreme — eating prey up to
  40 % of its length.
- **Planktivores** (Herring, Sprat, Smelt, Stickleback) have a very high `sizeRatio.max`
  (500–1000) and a high `sizeRatio.min` (5–10): they keep eating tiny plankton even when large and
  never take big fish. Stickleback (1000/10) is locked into the smallest-prey niche for life.

Whether a school actually finds enough food is governed by its **predation efficiency** (ingested /
maximum ration). Below the **critical efficiency `predation.efficiency.critical.spN = 0.57`** (the
OSMOSE/Shin & Cury default, identical for all 8 species) the school starves (§7) and stops growing
(§5).

---

## 4. Stage 3 — Adult: maturity and reproduction

### Maturity

A school becomes adult when `length ≥ species.maturity.size.spN` (the age clause is moot here).
Maturity lengths (cm):

| | Cod | Herring | Sprat | Flounder | Perch | Pike-perch | Smelt | Stickleback |
|--|----|----|----|----|----|----|----|----|
| maturity L (cm) | 38.0 | 18.0 | 9.0 | 22.0 | 18.0 | 40.0 | 10.0 | 4.5 |

Mature schools are summed into **SSB** each reproduction step
(`SSB = Σ abundance·weight` over mature schools, `reproduction.py:108-111`). On a fresh run, if a
species' SSB is still zero, it is bootstrapped from `population.seeding.biomass.spN` (Cod 150 kt,
Herring 800 kt, Sprat 600 kt, Flounder 80 kt, Perch 30 kt, Pike-perch 15 kt, Smelt 20 kt,
Stickleback 100 kt) during the seeding window.

### Spawning seasons

The fraction of annual eggs released per timestep comes from per-species seasonality files
(`reproduction.season.file.spN → reproduction/reproduction-seasonality-spN.csv`, 24 rows summing to
≈1). These encode the real Baltic phenology:

| Species | Spawning window | Peak | Shape |
|---------|-----------------|------|-------|
| **Cod** | mid-Mar → mid-Aug | late May–early Jun | broad ~5-month bell (deep-basin summer spawning) |
| **Herring** | spring mid-Mar→mid-Jun **and** autumn Sep→mid-Nov | spring late-Apr–May (autumn secondary) | **bimodal** — spring + autumn cohorts in one curve |
| **Sprat** | Apr → mid-Jul | June | sharp single late-spring peak |
| **Flounder** | Feb → Jun | Apr–early May | earliest spawner; late-winter/spring |
| **Perch** | May → early Jul | May–Jun | very narrow (4 timesteps) |
| **Pike-perch** | Apr → mid-Jul | May–Jun | narrow spring/early-summer |
| **Smelt** | mid-Mar → mid-Jun | April | skewed-early single peak |
| **Stickleback** | May → mid-Jul | June | narrow late-spring/summer |

This is exactly what the live-movement "Egg/larva" filter visualizes — eggs appearing on each
species' spawning grounds during its window.

### Recruitment (stock-recruitment relationship)

Egg production is multiplied by a density-dependent factor in `apply_stock_recruitment()`
(`reproduction.py:15-81`), selected by `stock.recruitment.type.spN`. The **base config** uses:

- **Beverton-Holt** — `recruits = linear / (1 + SSB/ssb_half)` — for **Cod, Flounder, Perch,
  Pike-perch** (`ssb_half` = 120 kt cod, 50 kt flounder, 10 kt perch/pike-perch). Cod's half-
  saturation is pinned to the ICES Bpa for the eastern stock.
- **Density-independent** (no SR limitation; constant per-gram fecundity) for **Herring, Sprat,
  Smelt, Stickleback** — the forage species.

(The canonical *calibrated* run upgrades all eight species to a **Shepherd** relationship — §8.)

### Adult mortality

Adults face three ongoing hazards beyond predation (from which large fish increasingly escape as
they outgrow predators' size windows):

- **Background/additional mortality** `mortality.additional.rate.spN` (yr⁻¹): Cod **0.20**,
  Pike-perch 0.06, Sprat/Flounder/Perch 0.05, Smelt 0.02, Herring & Stickleback 0.0. Cod's elevated
  0.20 explicitly represents **grey-seal predation** (raised from 0.05); perch/pike-perch carry a
  small cormorant allowance.
- **Fishing** `fisheries.rate.base.fshN` (knife-edge by age): Sprat **0.32**, Herring 0.15, Cod
  0.08, Flounder 0.04, Perch = Pike-perch 0.03, Smelt 0.02, Stickleback 0.01. The pelagic forage
  fishery dominates fishing pressure.
- **Out-of-domain mortality** `mortality.out.rate.spN`: only the two pelagic migrants carry it —
  Sprat 0.08, Herring 0.05 (losses when schools leave the modeled area). All others 0.

### Aging

A school is removed when it exceeds `species.lifespan.spN` (AGING cause): Cod 20 yr, Flounder/Perch/
Pike-perch 15, Herring 12, Sprat 8, Smelt 7, Stickleback 4.

---

## 5. Growth — how a fish gets bigger

The Baltic config uses **classic deterministic von Bertalanffy growth modulated by feeding
success** (it has *no* bioenergetics — there are no `bioen.*` keys). Growth lives in
`osmose/engine/processes/growth.py`:

- **Expected length at age** is a 3-phase curve (`expected_length_vb`, `growth.py:15-45`):
  `age 0` → `egg_size`; `0 < age < threshold (0.5 yr)` → linear from egg size up to the length at
  the threshold age; `age ≥ threshold` → `L∞·(1 − exp(−K·(age − t0)))`.
- **Per-step increment** `Δl = L(age+1) − L(age)`, then **gated by feeding success**
  (`growth.py:67-79`): a school feeding below its critical success rate gets **zero growth**; above
  it, growth scales up to a maximum. Eggs and out-of-domain schools always get the mean increment.
  So a starving juvenile both dies faster (§7) *and* stunts.
- **Weight** follows length-weight allometry `W = a·L^b` (`species.length2weight.condition.factor`
  = *a*, `species.length2weight.allometric.power` = *b*; `growth.py:84-85`), and
  `biomass = abundance · weight`.

Growth parameters (`species.lInf` cm, `species.K` yr⁻¹, `species.t0` yr; growth-threshold age =
0.5 yr for all species):

| | Cod | Herring | Sprat | Flounder | Perch | Pike-perch | Smelt | Stickleback |
|--|----|----|----|----|----|----|----|----|
| L∞ (cm) | 110.0 | 27.0 | 16.0 | 42.0 | 45.0 | 90.0 | 25.0 | 8.0 |
| K (yr⁻¹) | 0.15 | 0.35 | 0.45 | 0.20 | 0.15 | 0.18 | 0.35 | 0.80 |
| t0 (yr) | −0.20 | −0.80 | −0.50 | −0.40 | −0.50 | −0.30 | −0.30 | −0.10 |
| a | 0.00870 | 0.00560 | 0.00490 | 0.00890 | 0.01070 | 0.00620 | 0.00500 | 0.00800 |
| b | 3.050 | 3.130 | 3.120 | 3.080 | 3.100 | 3.090 | 3.050 | 3.050 |

The classic life-history spectrum is visible: stickleback is fast-living (high K, tiny L∞, short
life), cod is slow-growing and long-lived to a very large size.

---

## 6. The spatial life cycle — movement maps per stage

All species use map-based movement (`movement.distribution.method.spN = maps`) with a small
random-walk range (1–3 cells). Each map binds a species + an **age band** (`initialAge`/`lastAge`,
years) + a set of **timesteps** to a spatial probability grid, producing a life cycle that moves
through space as well as time:

- **Juveniles** use broad **coastal/nursery** maps active all year (e.g. `cod_juvenile` ages 0–1).
- **Adults** use **offshore/feeding** maps during the non-spawning steps.
- **Spawning-stage adults** are redirected to **spawning-ground** maps during the species' spawning
  window — e.g. `cod_spawning` (ages 4–21, steps 4–15 = Mar–Aug → the Bornholm/Gdańsk deep basins),
  `herring_spawning` (spring, coastal) plus a separate `herring_spawning_autumn` (steps 16–21),
  `sprat_spawning` (deep), `flounder_spawning` (deep south), and coastal/estuarine spawning maps for
  perch, pike-perch, smelt and stickleback.

So the model reproduces the Baltic pattern of coastal nurseries, offshore adult feeding, and
species-specific spawning grounds visited seasonally. Eggs, created unlocated, are dropped onto the
youngest map at the next movement step — which is why the live-movement "Egg/larva" view lights up
the spawning grounds during each species' season.

---

## 7. Mortality across the whole life cycle

The configuration encodes a strongly **stage-structured** mortality regime — different killers
dominate at different ages:

| Stage | Dominant mortality |
|-------|--------------------|
| **Egg** | exempt from predation/fishing; killed only by the larval additional rate when it transitions |
| **Larva** | `mortality.additional.larva.rate` (3.5–15 yr⁻¹) — orders of magnitude above everything else; the recruitment bottleneck |
| **Juvenile / small fish** | **predation** — small individuals sit inside cod/perch/pike-perch size windows; also subject to fishing once past `a50` (pelagics from age 1) |
| **Adult** | low background M (0–0.20, cod highest for seals) + fishing F (0.01–0.32, sprat highest) + a growing size-refuge from predation; pelagics also lose to out-of-domain mortality |
| **All ages** | **starvation** whenever feeding efficiency < 0.57, up to 0.3 yr⁻¹ (uniform across species); and **aging** at the lifespan |

Mechanically, mortality is resolved in `mortality()` (`processes/mortality.py`) over
`mortality.subdt = 10` sub-steps per timestep, with the four causes (predation, starvation,
additional, fishing) applied in a **shuffled order per school** as instantaneous-rate hazards
`n_dead = abundance·(1 − exp(−rate/(n_dt·n_subdt)))`. Predation uses a size-ratio + accessibility +
functional-response kernel. In production this runs in compiled Numba kernels
(`_apply_predation_numba`, `_mortality_all_cells_numba`).

### Starvation in detail

Starvation (`mortality.starvation.rate.max.spN = 0.3` yr⁻¹ for all species) switches on when a
school's realized predation efficiency drops below the critical 0.57: it cannot meet maintenance,
so it incurs mortality scaling up to 0.3 yr⁻¹ as feeding approaches zero — and simultaneously stops
growing (§5). This couples the food web to growth and survival: a species whose prey collapses both
starves and stunts.

---

## 8. The calibrated run (phase-13 Shepherd)

The base `baltic_param-*.csv` files above are the **structural description** of the life cycle.
The *canonical* Baltic run, however, applies the **phase-13 calibration**
(`data/baltic/calibration_results/phase13_results.json`, documented in
`docs/baltic_shepherd_calibration_2026-05-30.md`), a 40-year fit that **overrides** several
lifecycle families at runtime. The two differences that matter most for the life cycle:

1. **Recruitment is upgraded to a Shepherd relationship for all 8 species**
   (`recruits = linear / (1 + (SSB/ssb_half)^β)`), replacing the base Beverton-Holt/density-
   independent mix. Calibrated half-saturations `stock.recruitment.ssbhalf` range from ~6 kt
   (pike-perch, flounder) to ~230 kt (stickleback), with Shepherd exponents
   `stock.recruitment.shape` (β) from 0.50 (pike-perch) to 2.56 (smelt). Cod's `ssb_half` stays
   pinned at 120 kt (ICES Bpa).
2. **Larval and adult additional mortality are re-fitted.** The calibrated
   `mortality.additional.larva.rate` (e.g. Cod 0.50, Herring 4.20, Flounder 9.65, Pike-perch 3.00)
   and `mortality.additional.rate` differ substantially from the base-config values in §2/§4 — the
   calibration is what reconciles the structural rates with observed stock sizes.

A separate later calibration (phase-14, `phase14_results.json`) tuned only the **predation
functional-response half-saturation** for a few species; per project notes it was a diagnostic and
is **not** part of the canonical lifecycle calibration. Use **phase-13 (Shepherd)** as the
reference parameterization.

---

## 9. Caveats and modelling notes

- **Cod parameters are historical (pre-2015).** The `L∞ = 110 cm` / maturity `38 cm` values are the
  stable 1930s–1990s eastern-Baltic figures. The config headers warn (Svedäng et al. 2024) that the
  stock has since collapsed in growth — effective L∞ now ~60–80 cm and maturity length ~halved.
  Anyone modelling the post-2015 period must revise sp0.
- **Maturity is size-only.** No `species.maturity.age` is set, so a fish matures purely on reaching
  its maturity length.
- **No ontogenetic diet thresholds.** `predation.predPrey.stage.threshold = null` for every species;
  the diet shift comes only from the fixed size-ratio window scaling with predator growth, plus the
  age-structured accessibility matrix.
- **No bioenergetics.** Growth is classic feeding-gated vBGF, not a `bioen` energy budget.
- **Base config vs. calibration.** The CSV files describe the *structure*; the phase-13 Shepherd
  JSON supplies the *canonical values* for recruitment and additional mortality. A run that does not
  load phase-13 will use the (uncalibrated) base rates and behave differently.
- **Baltic is Python-engine only.** It declares background/LTL resource groups (`nbackground > 0`),
  which the bundled Java engine cannot load; run it on the Python engine.

---

### Quick reference — per-species life-history summary (base config)

| | Cod | Herring | Sprat | Flounder | Perch | Pike-perch | Smelt | Stickleback |
|--|----|----|----|----|----|----|----|----|
| L∞ (cm) | 110 | 27 | 16 | 42 | 45 | 90 | 25 | 8 |
| K (yr⁻¹) | 0.15 | 0.35 | 0.45 | 0.20 | 0.15 | 0.18 | 0.35 | 0.80 |
| maturity L (cm) | 38 | 18 | 9 | 22 | 18 | 40 | 10 | 4.5 |
| lifespan (yr) | 20 | 12 | 8 | 15 | 15 | 15 | 7 | 4 |
| egg size (cm) | 0.15 | 0.12 | 0.10 | 0.13 | 0.20 | 0.15 | 0.09 | 0.15 |
| rel. fecundity (eggs/g) | 500 | 600 | 1200 | 400 | 350 | 300 | 1000 | 600 |
| spawning peak | May–Jun | Apr–May (+Oct) | Jun | Apr–May | May–Jun | May–Jun | Apr | Jun |
| recruitment (base) | Bev-Holt | density-indep. | density-indep. | Bev-Holt | Bev-Holt | Bev-Holt | density-indep. | density-indep. |
| larval M (yr⁻¹, base) | 15.0 | 8.0 | 9.0 | 12.0 | 13.0 | 15.0 | 13.5 | 3.5 |
| background M (yr⁻¹) | 0.20 | 0.0 | 0.05 | 0.05 | 0.05 | 0.06 | 0.02 | 0.0 |
| fishing F (yr⁻¹) | 0.08 | 0.15 | 0.32 | 0.04 | 0.03 | 0.03 | 0.02 | 0.01 |
| trophic role | apex piscivore | planktivore | planktivore | benthivore | meso-predator | piscivore | forage | planktivore |
