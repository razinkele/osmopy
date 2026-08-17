# Why does stickleback swing ±20% when herring spawns two weeks earlier? — partial answer

**Date:** 2026-08-12
**Follows:** `docs/baltic_herring_phenology_a0_findings_2026-08-12.md`, where shifting herring's
spawning window moved stickleback −20.6% / +20.7% — inversely to herring and twice its magnitude,
while sprat stayed flat (±0.4%).
**Verdict: one hypothesis refuted, the other insufficient. The mechanism is NOT resolved.**

## Pre-registered discrimination

Fixed before results, because biomass anti-correlation is predicted by both hypotheses and
therefore cannot separate them:

* **H1 — food competition.** Herring and stickleback are both zooplanktivores ⇒ stickleback
  **starvation** should move, and/or its realised **diet composition** should shift.
* **H2 — predation-mediated.** ⇒ stickleback **predation mortality** should move.
* **H3 — both / neither**, reported as such rather than forced into one story.

Lead hypothesis going in was H2, because the herring experiment found predation doing all the work
and starvation none.

## Measurement (final-decade mean rates, 30 yr, seed 42, per arm)

| stickleback cause | shift −2 | shift 0 | shift +2 | Δ(−2) | Δ(+2) |
|---|---|---|---|---|---|
| Starvation / Adult | 0.0295 | 0.0299 | 0.0291 | −1.5% | −3.0% |
| Starvation / Juvenile | 0.111 | 0.106 | 0.102 | +4.6% | −3.7% |
| Predation / Adult | 0.487 | 0.406 | 0.461 | **+19.9%** | **+13.6%** |
| Predation / Juvenile | 15.11 | 13.46 | 15.05 | **+12.3%** | **+11.8%** |
| Predation / Eggs | 1.233 | 1.088 | 1.049 | +13.4% | −3.6% |
| Additional / Juvenile | 0.805 | 0.776 | 0.829 | +3.8% | +6.8% |

**Stickleback biomass: −20.6% (shift −2), +20.7% (shift +2).**

## What this establishes

1. **H1 is refuted.** Stickleback starvation moves by at most 4.6%, non-monotonically, and in the
   *wrong* direction for competition: the arm where stickleback loses 20% of its biomass shows
   *lower* adult starvation, not higher. Food competition with herring is not the driver.
2. **H2 is implicated but insufficient.** Predation is the only cause that moves substantially —
   but it moves **the same way in both arms** (+12–20%), while biomass moves in **opposite**
   directions. A single mortality term that rises in both arms cannot produce a −20%/+20% split.
3. **Therefore the swing must be substantially driven by the production side**, not by mortality
   alone. Stickleback biomass rises 20.7% in the arm where it is *also* being eaten 12–14% harder,
   which requires more recruits entering.

That is as far as this measurement goes. Claiming a mechanism beyond it would be the error this
series of experiments exists to avoid.

## A structural observation worth keeping

The unshifted (calibrated) arm sits at a **local minimum of predation on stickleback** for every
stage — adults 0.406 vs 0.487/0.461, juveniles 13.46 vs 15.11/15.05. The same was true of herring
egg predation in the previous experiment. The configuration was calibrated with this spawning
window, so both species' predation exposure is at a local optimum for it; any perturbation in
either direction increases predation. That is a property of the calibration, not of Baltic ecology,
and it is worth remembering whenever a timing perturbation "makes things worse" in this model.

Note also the sheer scale of stickleback juvenile predation — **~13–15 per year**, against
herring's 0.67. Stickleback is being cropped extremely hard in this configuration, so its
equilibrium is highly sensitive to small changes in either predation or recruitment.

## UPDATE 2026-08-13: diet extraction fixed — H1 refuted on BOTH limbs

The earlier `StopIteration` was a defect in my script, not the engine: `diet_matrix()` returns a
**wide** frame with `{predator}_{prey}` column names, not a stacked frame with a species column,
and species names containing underscores (`cod_west`) require longest-prefix parsing. Fixed and
re-run over the same three arms (30 yr, seed 42, final-decade means):

| stickleback prey | shift −2 | shift 0 | shift +2 | Δ(−2) | Δ(+2) |
|---|---|---|---|---|---|
| Mesozooplankton | 39.76 | 40.25 | 39.90 | −1.2% | −0.9% |
| Diatoms | 28.78 | 28.63 | 28.87 | +0.5% | +0.8% |
| Microzooplankton | 28.08 | 27.68 | 27.82 | +1.4% | +0.5% |
| Macrozooplankton | 2.82 | 2.88 | 2.85 | −2.0% | −1.2% |
| Benthos | 0.53 | 0.52 | 0.53 | +0.6% | +2.2% |

**Herring/stickleback dietary overlap (Schoener index): 0.4907 / 0.4967 / 0.4985** — constant to
within 1.6% across arms in which stickleback biomass swings ±20.7%.

Herring's diet is likewise stable in its dominant prey (Mesozooplankton 62.4/62.8/62.6, ±0.6%);
its only notable moves are in trace items (Diatoms +11.9%, Microzooplankton +7.3% in the later
arm, from bases of 2.6% and 3.4% of intake).

**So *compositional* competition is refuted** — neither species' realised prey proportions change
materially, and their overlap does not move. (This was originally written as "H1 refuted on both
limbs"; see the RETRACTION below — proportions are scale-free and cannot test *quantitative*
competition.) Note also that **neither eats the other**: no fish
prey appears in either diet, so the interaction is not predatory between them either.

## RETRACTION 2026-08-13: the equilibrium-shift hypothesis was wrong, and "refuted on both limbs" over-claimed

I published an "alternative equilibria" hypothesis earlier today and am withdrawing it within the
hour. Two errors, both mine, both in the reasoning rather than the data.

### Error 1 — I mischaracterised the response shape

I wrote that the swing was "symmetric-in-magnitude, opposite-in-sign" and that "no single trophic
pathway produces that". Read the actual numbers:

| | shift −2 | shift 0 | shift +2 | step | step |
|---|---|---|---|---|---|
| stickleback (t) | 64,357 | 81,025 | 97,820 | **+16,668** | **+16,795** |

That is **monotonic and very nearly perfectly linear** — the two steps agree to 0.8%. A linear
dose–response is the signature of an *ordinary* mechanism. Bistability would show discrete jumps,
hysteresis, or seed-dependent bimodality, and shows none of them here. Describing a straight line
as an anomaly requiring alternative equilibria was a misreading, and the "cheap decisive test"
I proposed off the back of it (per-seed spread across the 5-seed A0 run) would have spent ~40 min
of compute rejecting a hypothesis I could have reasoned away for free.

Herring, for contrast, moves 2,690,044 / 2,547,746 / 2,292,245 — monotonic but **convex**
(−142,298 then −255,501), so the two species are not simply mirror images.

### Error 2 — my H1 test could not detect the competition mechanism that matters

Schoener overlap and diet-percentage composition are **scale-free**. If herring biomass falls, herring
removes less zooplankton in total, leaving more per stickleback; stickleback then eats **more in
absolute terms, in identical proportions**. Composition unchanged, overlap unchanged — and
competition is exactly what happened.

So the diet run refuted **compositional** competition (a niche-shift mechanism). It said nothing
about **quantitative** competition, which is the standard one. "H1 refuted on both limbs" claimed
more than the metrics can carry, and is withdrawn.

I also argued past evidence already sitting in my own table. Stickleback **juvenile** starvation is
0.111 / 0.1061 / 0.1022 — **+4.6% when it loses 20%, −3.7% when it gains 20%**: monotonic, and in
precisely the direction competition predicts. I called it "non-monotonic" only by averaging it
against the *adult* limb, a different life stage under a different food-limitation regime.

### What survives, and the corrected hypothesis

My argument that starvation *mortality* is too small to carry a 20% swing still holds: 0.106
against juvenile predation of 13.46 is under 1% of juvenile mortality. But that argues the pathway
is **growth and fecundity**, not that competition is absent — more food → faster growth → earlier
maturity and higher per-capita fecundity → more recruits.

That means the "production side" I inferred and the competition hypothesis I "refuted" are **one
explanation, not two competing ones**. I set them against each other and scored a refutation
against my own leading candidate.

**Leading hypothesis: quantitative food competition acting through growth and fecundity.**

### Pre-registered reading for the next run (fixed before results)

* **Supported** — total zooplankton biomass rises monotonically across shift −2 → shift +2,
  tracking herring's decline, **and** stickleback per-capita intake rises with it.
* **Refuted** — zooplankton is materially flat while stickleback biomass swings ±20%.

Measurements, all from the same three 30-yr runs: (1) total zooplankton biomass by group per arm;
(2) stickleback per-capita intake / mean weight-at-age; (3) stickleback recruits and eggs.

Caution carried forward: a linear response is consistent with an ordinary mechanism but does not
identify *which* one. "Linear ⇒ competition" must not become the next claim I retract — the
zooplankton and intake measurements are what would earn it.

## What failed, and what to measure next

~~The diet output did not parse…~~ **Resolved 2026-08-13 — see the UPDATE above. Compositional competition is refuted; quantitative competition is untested and now leading.**

Next measurement, in order of value:

1. **Fix the diet extraction** and re-run the same three arms — closes the H1 limb properly and
   shows whether the herring–stickleback overlap changes at all.
2. **Stickleback recruitment / egg production per arm**, to test the production-side inference in
   point 3 above. If recruits track the biomass swing, the mechanism is recruitment-mediated and
   the question becomes what changes stickleback's SSB or its stock–recruitment position.
3. **Who eats stickleback, and does that predator change?** `predatorPressure` by predator across
   the arms would say whether the +12–20% predation is coming from herring itself, from cod, or
   from redistribution.

## Status

No production change; this is diagnostic work on the existing configuration. All three arms
certify (`docs/baltic_herring_phenology_a0_2026-08-12.md`).

---

# Run 3 (2026-08-13): quantitative competition REFUTED on the growth limb

Pre-registered in `1e3c566` before running. 3 arms × 30 yr × seed 42, final-decade means.

## Verdict against the pre-registered rule: REFUTED

The rule required **both** limbs. One is refuted outright; the other turns out not to be
measurable with existing outputs, so the rule as written cannot be satisfied. Reporting it as
refuted, not "partially tested".

### Limb 1 — per-capita growth: refuted by decomposition, not inference

| | shift −2 | shift 0 | shift +2 | Δ(−2) | Δ(+2) |
|---|---|---|---|---|---|
| stickleback **biomass** | 66,439 | 85,544 | 97,930 | −22.33% | +14.48% |
| stickleback **abundance** | 1.796e10 | 2.307e10 | 2.634e10 | −22.15% | +14.18% |
| stickleback **mean weight** (B/N) | 3.699e−6 | 3.708e−6 | 3.718e−6 | −0.23% | +0.26% |
| stickleback **mean size** L | 7.5066 | 7.5137 | 7.5187 | −0.09% | +0.07% |

Abundance tracks biomass to within 0.3 percentage points, while per-capita weight moves 0.5%
across the whole range. **The biomass swing is entirely an abundance change.** Individual
stickleback are not growing better in the arm where the stock gains 20% — there are simply more
of them. The growth → maturity → fecundity pathway is dead.

### Limb 2 — zooplankton standing stock: NOT MEASURABLE (tooling gap)

`biomass()` returns only focal species plus the two forced predators
(`GreySeal`, `Cormorant`); resource groups are absent, and `osmose/schema/output.py` defines no
resource-biomass output key at all. Measuring LTL standing stock requires instrumenting
`ResourceState` directly. **Recording this as a tooling gap** — it is a reasonable thing to want
and the schema cannot express it today.

## New finding: herring and stickleback respond through ORTHOGONAL channels

This is the strongest constraint the run produced, and it is not about competition.

| | abundance Δ(−2)/Δ(+2) | per-capita weight Δ(−2)/Δ(+2) | mean size L |
|---|---|---|---|
| **herring** | +4.06% / +0.69% (flat, non-monotonic) | **+3.96% / −8.40%** | 13.396 / 13.303 / 13.037 (mono ↓) |
| **stickleback** | **−22.15% / +14.18%** (mono ↑) | −0.23% / +0.26% (flat) | flat |

Herring's biomass loss is its **individuals getting smaller** — abundance barely moves while mean
weight falls 8.4% and mean length declines monotonically. Stickleback's is **purely numerical**.

Same perturbation, two species, orthogonal response channels: it hits herring's *growth* and
stickleback's *recruitment*, by different routes. Herring's channel is at least consistent with an
ordinary match–mismatch reading (spawning later ⇒ larvae meet a worse feeding window ⇒ poorer
growth). Stickleback's is not yet explained by anything measured.

## A candidate I checked and discarded before publishing it

Egg mortality looked compelling: stickleback egg predation falls monotonically across the arms
(1.233 / 1.088 / 1.049), and treating total egg M as an exposure integral gave **−22.04% predicted
vs −22.15% observed** on the shift−2 limb — a 0.11 pp match.

**It is arithmetic coincidence.** `mortality_rates_from_counts`
(`osmose/engine/simulate.py:880`) documents that reported rates are **sums of per-step rates over
the saving interval**, while an egg exists for a **single step** (`first_feeding_age_dt=1`). So
`exp(−annual_sum)` is the wrong integral. Per-egg exposure is one step's rate:

| n steps carrying eggs | survival ratio Δ(−2) | Δ(+2) |
|---|---|---|
| 10 | −2.46% | +0.67% |
| 14 (spring 8 + autumn 6) | −1.76% | +0.48% |
| 24 | −1.03% | +0.28% |

One to two percent, not twenty, under **every** plausible window width — and the window is
translated rather than widened across arms, so the step count is identical in all three. The
shift+2 limb also missed by 2× (+6.93% predicted vs +14.18%), which should have been the tell:
a mechanism matching one limb to 0.11 pp and missing the other by half explains half the pattern
plus a coincidence.

## Status: the mechanism is UNRESOLVED

Refuted so far: compositional competition; growth-mediated quantitative competition; starvation
mortality (<1% of juvenile mortality); egg survival; bistability. Predation moves up in *both*
arms, so it tracks the calibration optimum rather than the trend.

What survives is the bare fact plus one constraint: herring spawning timing changes stickleback
**recruitment numbers** monotonically, through a route that is not diet, not growth, and not egg
survival. I am not going to name a sixth mechanism without measuring one.

Next measurement, and the only one I would trust: **stickleback egg production (numbers spawned)
and SSB per arm**, which separates "more eggs laid" from "more eggs surviving" — the latter is now
excluded. If egg production tracks the swing, the question becomes what drives stickleback SSB,
and the answer is likely a feedback loop rather than a herring-driven forcing.

## Caveat on the linearity claim

This run is 30 yr × seed 42 and gives −22.33% / +14.48% — visibly **asymmetric**. The A0 numbers
(50 yr × 5 seeds) gave −20.6% / +20.7%, whose steps agree to 0.8%. Monotonicity holds in both, so
the bistability retraction stands, but **the near-perfect linearity is a 5-seed 50-yr property**
and nothing further should be built on the "steps agree to 0.8%" framing.

---

# Run 4 (2026-08-13): egg production and SSB per arm — PRODUCTION REFUTED

Pre-registered in the script header before running. 3 arms × 30 yr × seed 42, final-decade
mean annual values. Egg counts captured by wrapping `reproduction()` and summing the schools it
appends — the **final** count after every gate (RV / ceiling / thermal / depensation), which
avoids having to be right about which gates bind on sp7.

## Verdict: the change does NOT enter at egg production

| | shift −2 | shift 0 | shift +2 | Δ(−2) | Δ(+2) | mono |
|---|---|---|---|---|---|---|
| **eggs** stickleback | 9.109e11 | 8.448e11 | 8.058e11 | **+7.82%** | **−4.62%** | ↓ |
| **SSB** stickleback | 68,395 | 87,956 | 100,700 | −22.24% | +14.48% | ↑ |
| **N** stickleback | 1.796e10 | 2.307e10 | 2.634e10 | −22.15% | +14.18% | ↑ |
| **N/eggs** stickleback | 0.01972 | 0.02731 | 0.03269 | −27.80% | +19.71% | ↑ |

**Egg production is anti-correlated with abundance.** The arm with the *most* eggs (shift−2,
9.11e11) has the *fewest* stickleback (1.80e10); the arm with the *fewest* eggs has the most fish.
This is stronger than the "eggs flat" case I pre-registered — indifference would have been
consistent with production being irrelevant, whereas anti-correlation rules it out outright.

So the swing enters **downstream of egg production**.

## Why eggs run backwards: over-compensation, and it is internal

`ssb_half` for sp7 is 7,245 while SSB sits at 68,395–100,700 — **9.4× to 13.9× above
half-saturation**, deep in the declining limb of a Shepherd curve
(`eggs = c·SSB / (1 + (SSB/h)^β)`, `reproduction.py:76-78`). For SSB ≫ h this is
≈ `c·h^β·SSB^(1−β)`, so **β > 1 makes eggs fall as SSB rises**. The two arm-intervals imply
β = **1.299** and **1.349** independently — consistent, and over-compensatory.

`eggs/SSB` moves +38.66% / −16.69%. Per the pre-registration this is **explicitly not evidence of
an external driver** — it is what a saturating/over-compensating curve does when SSB moves, and
reading a mechanism into it would repeat the error that killed the egg-survival candidate.

## Herring contrast sharpens it

| | Δ(−2) | Δ(+2) |
|---|---|---|
| herring **N/eggs** | +1.41% | +0.66% (flat) |
| stickleback **N/eggs** | −27.80% | +19.71% |

Herring's egg-to-standing-stock conversion does not move; stickleback's does. Same perturbation,
and only one species' conversion responds — consistent with the orthogonal-channels finding in
Run 3 (herring responds through *weight*, stickleback through *numbers*).

## The tension this leaves — stated, not smoothed over

`N/eggs` rises 19.71% in shift+2, yet **juvenile predation is *higher* in that arm than at base**
(15.11 / 13.46 / 15.05). Higher predation with higher egg-to-stock conversion is contradictory on
its face. Two reasons neither number pins the stage:

1. **`N/eggs` is not a survival rate.** `N` is standing abundance across all ages; eggs is annual
   production. The ratio carries time units and moves with age structure and longevity, not only
   with per-capita survival. More recruits shifting the age distribution would move it with no
   survival rate changing at all.
2. **The juvenile rates are annual SUMS of per-step rates** — the same convention that killed the
   egg candidate. Read as a cohort integral, `exp(−13.46)` ≈ 1.4e−6 and *both* arms would sit ~80%
   below base, which is absurd against a −22%/+14% biomass swing. So that number is not
   interpretable as stage survival without knowing the juvenile stage duration in steps.

**Honest scope: the change enters downstream of egg production; which post-egg stage is not
pinned, and the two available measurements disagree in sign while neither is a clean survival
estimate.**

What would resolve it: an age-resolved abundance series
(`output.abundance.byage.enabled`) separating "more recruits" from "shifted age structure" — and it
needs the same units scrutiny applied before anything is read off it.

## Running tally

Refuted to date: compositional competition · growth-mediated quantitative competition · starvation
mortality · egg survival · bistability · **egg production**. Six refutations, no positive
identification. That is a legitimate state to stop at, and preferable to naming a seventh
mechanism the data has not earned.

---

# Run 5 (2026-08-13): age-resolved abundance — POSITIVE IDENTIFICATION, and a convention that reframes the series

3 arms × 30 yr × seed 42. Units scrutiny done before reading anything off it: age bins are
**years** (`age_dt // n_dt_per_year`, `simulate.py:1033-1045`), and the array is an instantaneous
standing-stock snapshot, not an integral.

## The finding, from two raw numbers that need no decomposition

| | shift −2 | shift 0 | shift +2 | Δ(−2) | Δ(+2) |
|---|---|---|---|---|---|
| **age-0 bin** — ALL fish [0,1) yr | 8.399e10 | 8.561e10 | 8.801e10 | **−1.89%** | **+2.81%** |
| **`abundance()`** — fish ≥0.5 yr | 1.796e10 | 2.307e10 | 2.634e10 | **−22.15%** | **+14.18%** |

**The swing is concentrated in survival through early life, not in egg production.** Both numbers
are raw output — no ratios, no assumptions.

> **Amended 2026-08-13 after Run 6.** This originally read "the population produces essentially the
> same number of young in every arm". That over-claimed: the young-of-year pool's own response is
> **sampling-dependent** — −1.9%/+2.8% snapshot-averaged here, −8.4%/+9.7% step-summed in Run 6 (see
> Run 6's sampling note). The `abundance()`-vs-age-0-bin comparison stands as two raw numbers; the
> word "flat" does not.

Age composition of the ≥0.5 yr pool is stable, so past the cutoff everything scales uniformly:

| age | shift −2 | shift 0 | shift +2 |
|---|---|---|---|
| [0.5,1) | 47.51% | 47.70% | 47.75% |
| 1 | 38.45% | 39.82% | 38.88% |
| 2 | 11.17% | 10.22% | 10.67% |
| 3 | 2.87% | 2.27% | 2.69% |

Per the pre-registration this is the **recruitment-side** branch — but located more precisely than
that branch anticipated: not egg production (Run 4 refuted it, eggs move *opposite*), and not the
number of young produced (flat here). It is **how many survive to the cutoff age**.

## The convention that made this series confusing — and that reframes every biomass number in it

`abundance()` and `biomass()` apply `output.cutoff.age` (Java convention: exclude young-of-year;
**0.5 yr for every Baltic species**, `_collect_biomass_abundance`, `simulate.py:836-855`).
`abundance_by_age()` does **not** — it bins every school, eggs included. That is the entire reason
the by-age total (9.77e10) and `abundance()` (2.31e10) disagree, and I could not have interpreted
either without checking.

**Consequence for the whole series:** every biomass and abundance figure here — the A0 report, the
certification tables — counts only fish **older than six months**. "stickleback biomass +20.7%"
means "+20.7% of fish ≥0.5 yr". That is intended and correct, but it is nowhere stated in the docs,
and it is exactly the quantity this investigation has been chasing.

Worth flagging for whoever touches fixtures next: the cutoff applies to every absolute-biomass
assertion in the repo. Nothing is wrong today — `CLAUDE.md` already warns that fixtures asserting
absolute Baltic biomass need re-deriving when the config changes — but the convention is now
written down where the next person will look.

## What I am NOT claiming

The `[0,0.5)` pool contains egg schools. Annual egg production is ~8.45e11 spread over roughly 14
spawning steps with eggs living one step, so order 3.5–6e10 of the 7.46e10 pool may be eggs. Any
ratio through that pool is standing-stock over an **egg-contaminated** denominator:

* `[0.5,1) / [0,0.5)` = 0.1131 / 0.1475 / 0.1668 (−23.33% / +13.06%, monotonic). Report as an
  **index**, not as survival through the first half-year.
* An `egg → YOY` ratio has eggs in numerator *and* denominator. It is not interpretable and I am
  not reporting a number for it.

The oddity that keeps me honest here: eggs move +7.82%/−4.62% while the pool containing them moves
+1.14%/+1.11% — non-monotonic, matching neither. Something compensates and I cannot see what, which
is precisely why the ratio does not get called a survival rate.

## Two earlier conclusions that must be corrected

1. **The Run-3 exclusion of egg survival is UNSUPPORTED, not overturned.** It rested on annual-sum
   mortality rates, which this run shows are not cohort integrals: `exp(−13.46)` = 1.4e−6 against an
   observed ~0.15 through the first half-year — **five orders of magnitude out**. The rate-based
   argument simply does not work. The direct measurement does not reinstate egg survival either,
   because the age-0 bin being flat puts the action *after* the young-of-year pool exists. Two
   different reasons, and neither is "egg survival is refuted".
2. **The Run-4 tension was never a contradiction between measurements.** Juvenile predation rising
   in the arm where conversion rises looked contradictory only because I was treating an annual sum
   of per-step rates as a stage survival. It is not one. **Retiring the annual-sum mortality rates
   as evidence for cohort survival** anywhere in this series.

## Status

First positive identification after six refutations: **the swing is survival to the 0.5 yr cutoff,
with the number of young produced held flat.** What *drives* that survival is a new question and I
have not measured it — and the cutoff finding is significant enough that it deserves a decision
before more compute goes in.

---

# Run 6 (2026-08-13): mortality in [0,0.5) yr attributed by cause — PREDATION

Death **counts**, not rates: counts are additive and carry no exponentials, so summing them over
the final decade is safe in the way `exp(-annual_sum)` was not. Hook is the engine's own
`_collect_by_life_stage` (`simulate.py:1834`) — called every step after all mortality and *before*
reproduction increments ages and clears `is_egg`, which is the engine's convention for binning
deaths by the stage held at time of death (#142). Eggs split out of the youngest window, since they
live one step and would otherwise swamp the post-egg signal.

## Answer: predation, and it carries more than all of the change

Share of the **change** in YOY [0,0.5) yr deaths (the pre-registered discriminator — not share of
the total, since a cause can dominate the count and be irrelevant to the difference):

| cause | % of total deaths | Δ(−2) | Δ(+2) | **share of CHANGE** |
|---|---|---|---|---|
| **Predation** | 86.3% | +8.64% | −7.43% | **+112.3% / +116.6%** |
| Additional | 12.4% | −6.75% | +7.44% | −12.6% / −16.8% (offsetting) |
| Starvation | 1.2% | +2.00% | −0.94% | +0.4% / +0.2% |

Predation over-explains the change; `Additional` partially cancels it. Starvation is 0.2–0.4% of
the change — dead again, now on a counts basis rather than a rate one.

## Per-capita mortality: only the earliest stages move monotonically

Deaths per fish-step, same hook, same steps for numerator and denominator:

| stage | shift −2 | shift 0 | shift +2 | Δ(−2) | Δ(+2) | |
|---|---|---|---|---|---|---|
| **YOY predation** | 0.3133 | 0.2643 | 0.2229 | +18.56% | **−15.64%** | **monotonic ↓ (−29% range)** |
| egg predation | 0.4049 | 0.3525 | 0.3292 | +14.86% | −6.60% | monotonic ↓ |
| juv [0.5,1) total | 0.0519 | 0.0489 | 0.0503 | +6.17% | +2.84% | non-monotonic |
| adult total | 0.0589 | 0.0570 | 0.0601 | +3.38% | +5.35% | non-monotonic |

**Only the two earliest stages show a monotonic per-capita change, and both fall in the same
direction.** The older windows show the familiar U-shape — the calibration sits at a local
optimum, so any perturbation makes things slightly worse. Predation pressure on early life stages
falling by ~29% is the first quantity in this investigation that moves monotonically with the
perturbation *and* is large enough to carry a ±20% swing.

## This is the empirical vindication of retiring the rate outputs

Counts say YOY predation falls **monotonically** (0.3133 / 0.2643 / 0.2229). The rate output said
`Predation/Juvenil` **rose in both arms** (15.11 / 13.46 / 15.05). Same underlying quantity,
opposite shape. The retirement in Run 5 was not merely cautious — the rates were giving the wrong
answer.

## Sampling note, and a cross-run disagreement I am not hiding

Run 5 (snapshot) put the `[0,1)` yr pool at −1.89%/+2.81%; Run 6 (step-sum) puts it at
−8.86%/+8.08%. Same sign, different magnitude. Mechanical cause: `output.recordfrequency.ndt=24`
with window *averaging*, collected **after** reproduction (ages incremented, new eggs appended),
against this hook summing **before** it — so age-bin membership at the boundaries and egg inclusion
both differ.

Which to trust for what: **Run 6's sampling is the commensurate one for per-capita mortality**,
because deaths and abundance come from the same hook over the same steps. The per-capita numbers
above are sound regardless of how the cross-run gap resolves. Run 5's raw two-number comparison
also stands; only its "flat" wording is withdrawn (amended above).

## An observation, explicitly not a mechanism

Predation on stickleback early stages is **positively** correlated with herring biomass across arms
(herring 2.69e6 / 2.55e6 / 2.29e6 against YOY predation 0.3133 / 0.2643 / 0.2229). That is the
**opposite** of prey-switching, which would predict more herring drawing predators *away* from
stickleback. It is a real constraint on any candidate mechanism.

The predator is **unidentified** and I am not naming one. The direct test is the `*_stickleback`
columns of `diet_matrix()` with **no threshold filter** — my earlier diet run filtered at >0.05% of
intake, which would discard exactly the kind of predation that is negligible to a large predator's
diet while decisive for a small prey's recruitment. That attribution by predator is the next
measurement, and it is one run.

## Status

Attribution answered: **predation, on eggs and young-of-year, monotonic and large enough to carry
the swing.** Which predator is a separate question, unmeasured.

---

# Run 7 (2026-08-14): who eats stickleback — NULL, and the herring candidate is refuted

`dietMatrix` (per-predator %) and `predatorPressure` (absolute tonnes), both with **no threshold
filter**, final-decade means. The threshold in the earlier diet run (>0.05% of intake) was the
specific concern; removing it does not rescue the measurement, for a different reason given below.

## The herring candidate is refuted — both halves of its prediction fail

It predicted herring's pressure on stickleback falls monotonically **while** stickleback stays
negligible in herring's diet. The second half holds; the first does not.

| | shift −2 | shift 0 | shift +2 | |
|---|---|---|---|---|
| herring pressure (t/step) | 172.15 | 180.51 | 171.80 | **non-monotonic** (−4.63% / −4.83%) |
| herring's share of change | — | — | — | +4.0% / −2.2% (negligible) |
| stickleback as % of herring diet | 0.0338% | 0.0378% | 0.0379% | flat, and tiny |

Herring's pressure peaks at the *unshifted* arm and falls in **both** directions — the
calibration-optimum U-shape, not a driver. The idea was post-hoc, it was tested rather than
asserted, and it is dead.

## But the predator is still unidentified — the measurement is confounded

Everything in this output tracks stickleback abundance rather than explaining it:

| | Δ(−2) | Δ(+2) | |
|---|---|---|---|
| stickleback biomass | −22.33% | +14.48% | |
| **total pressure** | −12.03% | **+22.87%** | same direction as its own prey |
| pressure per tonne of prey | +13.26% | +7.33% | **non-monotonic** (U-shape) |

Three reasons this cannot identify a driver:

1. **Direction.** Total pressure rises where stickleback is *more* abundant. More prey ⇒ more
   tonnage eaten. That is predators responding to abundance, not causing it.
2. **Uniformity.** All eight predators move the same way (pikeperch, cod_east, perch, flounder,
   cod_west, sprat, smelt all monotonic ↑; herring alone U-shaped), and every predator's stickleback
   *diet share* also rises monotonically. If one predator were driving the swing, its pressure should
   move **differently** from the rest. A uniform shared response is the signature of a common cause —
   prey availability — not of a culprit.
3. **Wrong stage, quantitatively.** This is the limitation stated before the run, now confirmed:
   tonnage is dominated by adult stickleback, and the two stages move in **opposite** directions.

   | predation deaths (counts, Run 6) | Δ(−2) | Δ(+2) |
   |---|---|---|
   | YOY — carries the monotonic signal | +8.64% | **−7.43%** |
   | adult — carries the tonnage | −14.90% | **+31.31%** |

   `predatorPressure` is in tonnes, so it sees the adult channel and is blind to the young-of-year
   channel where Run 6 located the mechanism.

## What would actually answer it

Numbers eaten, resolved by predator **and** prey life stage. Neither shipped output provides it:
`predatorPressure` is biomass-pooled across stages and `dietMatrix` is a per-predator percentage of
the same pooled quantity. It needs the predation kernel instrumented to record prey counts by
predator × prey stage — the same technique used for Run 6's death counts, applied one level deeper.

## Running tally

Refuted: compositional competition · growth-mediated quantitative competition · starvation mortality
· egg survival · bistability · egg production · **herring as the early-stage predator**.

Established: the swing is carried by **predation on eggs and young-of-year**, monotonic and large
enough (per-capita YOY predation −29% across arms, Run 6). **Which predator delivers it is
unresolved**, and the shipped outputs cannot resolve it.

---

# Run 8 (2026-08-14): predation kernel instrumented — **HERRING IS THE PREDATOR**, and Run 7's refutation is retracted

Prey **counts** by predator × prey life stage, from the production predation path. 3 arms × 30 yr,
seed 42, final **1 year** recorded (Python path is ~65 s/step; the 3-year version was ~2.3 h).

Method notes that mattered:
* **`predation.py` is not production.** Its `predation_for_cell` is a test harness — the docstring
  says "Production code uses mortality.mortality() instead". The live path is `mortality.py`'s
  interleaved Tier-2 predation. Instrumenting the wrong module would have recorded **zeros**.
* **Exact attribution without reimplementation.** `_apply_predation_for_school` applies predation
  for ONE predator against all prey in a cell and records deaths in `n_dead`, so diffing
  `n_dead[:, PREDATION]` around one call attributes every death to that predator. Engine code runs
  unmodified between snapshots.
* **Background predators exist.** GreySeal and Cormorant are internal species 9–10. My first
  attempt crashed on them — and `predatorPressure` has **no columns for them at all**, so Run 7 was
  predator-incomplete as well as stage-blind. (In the event neither takes a measurable share.)

## Result: herring dominates stickleback early-stage mortality

**Stickleback EGGS — numbers eaten:**

| predator | shift −2 | shift 0 | shift +2 | % of total | mono | share of CHANGE |
|---|---|---|---|---|---|---|
| **herring** | 8.098e10 | 5.678e10 | 5.499e10 | **62.85%** | **↓** | **+87.0% / +89.3%** |
| sprat | 2.582e10 | 2.321e10 | 2.707e10 | 25.69% | — | +9.4% / −192.3% |
| smelt | 1.076e10 | 9.685e9 | 5.514e9 | 10.72% | ↓ | +3.8% / +208.0% |

**Stickleback YOY [0,0.5) yr — numbers eaten:**

| predator | shift −2 | shift 0 | shift +2 | % of total | mono | share of CHANGE |
|---|---|---|---|---|---|---|
| **herring** | 1.264e11 | 1.051e11 | 9.573e10 | **54.88%** | **↓** | **+315.3% / +56.8%** |
| sprat | 3.327e10 | 4.106e10 | 3.427e10 | 21.44% | — | −115.5% / +41.2% |
| smelt | 1.136e10 | 1.774e10 | 1.535e10 | 9.26% | — | −94.5% / +14.5% |
| pikeperch | 1.241e10 | 1.172e10 | 1.481e10 | 6.12% | — | +10.2% / −18.7% |
| flounder | 1.034e10 | 1.071e10 | 8.173e9 | 5.59% | — | −5.5% / +15.4% |

**Herring takes ~55–63% of stickleback's early-stage deaths by number, and is the only predator
whose take declines monotonically across the arms.** Shares above 100% are real: the other
predators move non-monotonically and partly offset, so herring's absolute change exceeds the net.

This is the **CULPRIT** branch of the pre-registration, not the diffuse one.

## The asymmetry — and why six earlier measurements missed it

Stickleback is **0.034–0.038% of herring's diet** (Run 7) while herring inflicts **55–63% of
stickleback's early-stage mortality**. A rounding error to the predator; the dominant term for the
prey. Herring biomass is ~30× stickleback's, so a negligible diet fraction of a very large stock is
an enormous absolute take from a small one.

That asymmetry is exactly why the diet-composition work (Runs 1–2) found nothing: the interaction
is **predation, not competition**, and it is invisible in herring's own diet percentages. The
constant Schoener overlap (0.4907/0.4967/0.4985) was never evidence against an interaction — it was
measuring the wrong interaction.

## RETRACTION: Run 7's refutation of the herring candidate was wrong

Run 7 concluded "the herring candidate is refuted — both halves of its prediction fail", because
herring's `predatorPressure` was non-monotonic (172.15 / 180.51 / 171.80 t/step). That verdict is
**withdrawn**. `predatorPressure` is in **tonnes**, and stickleback eggs and YOY weigh almost
nothing individually, so the output is dominated by adult stickleback and cannot see the channel
the mechanism runs through. Measured in **numbers at the stages that matter**, herring's take is
both dominant and monotonic.

The irony is on the record: I raised herring as a post-hoc idea after Run 6, correctly flagged it
as untested, then "refuted" it with an output incapable of testing it — and said so in the same
document that noted the output was stage-blind. **The limitation I documented should have blocked
the verdict I drew.**

## The causal chain, end to end

Herring spawns later → herring biomass falls (2.69e6 / 2.55e6 / 2.29e6 t, −10.0% at shift +2) →
herring's predation on stickleback eggs and YOY falls (−8.9% YOY, nearly proportional) →
more stickleback survive to the 0.5 yr cutoff (Run 5: age-0 pool roughly steady, survivors
+14.3%) → stickleback biomass rises +20.7%.

At shift −2 the response is **more** than proportional (herring biomass +5.6%, egg predation
+42.6%), so biomass alone does not explain that limb — herring's own size structure shifts too
(mean length 13.396 / 13.303 / 13.037, monotonic), which changes size-selective access to tiny
prey. Not measured here; flagged, not claimed.

## Caveats

* **1-year recording window**, not the final decade. All arms share seed 42 and identical
  trajectory structure, so the cross-arm contrast is controlled, but small predators carry more
  noise and absolute values are not comparable to earlier runs.
* **Backend switch mid-run** (Numba spin-up → Python recording) consumes RNG differently, so this
  window is not bit-identical to an all-Numba run. Identical treatment across arms.
* Consistency check passed: total YOY predation moves +3.5% / −8.6% here against Run 6's
  +8.64% / −7.43% over the final decade — same sign and shape from independent instrumentation.

## Final tally

Refuted: compositional competition · growth-mediated competition · starvation mortality · egg
survival · bistability · egg production. **Withdrawn: Run 7's refutation of herring.**

**Identified: herring predation on stickleback eggs and young-of-year.** Dominant (55–63% of
early-stage deaths), monotonic across arms, carrying the change, and asymmetric — 0.038% of the
predator's diet, decisive for the prey.

---

# Run 9 (2026-08-16): predation by predator SIZE CLASS — size-gate REFUTED, and the shift−2 excess is not mass-action

Same instrumentation as Run 8, additionally binning by predator length (1 cm), prey length (0.25 cm)
and predator/prey size ratio. 3 arms × 30 yr, seed 42, final 1 year.

## The size-gate hypothesis is refuted

Run 8 left the shift−2 limb unexplained (herring biomass +5.6%, egg predation +42.6%) and I proposed
herring's declining mean length as the cause: `sizeratio.min = 5.0` means a herring of length *L*
can only take prey shorter than *L/5*, and herring's mean length falls 13.396 → 13.037 cm. The test:
do the prey taken sit near that ceiling?

**They do not.**

| predator/prey size ratio | share of herring's take |
|---|---|
| 5–6 (at the ceiling) | 6.4% |
| 6–8 | 9.1% |
| 8–12 | 12.1% |
| 12–20 | 13.7% |
| **20–50** | **24.7%** |
| **50–150** | **31.8%** |
| ≥150 | 2.1% |

Only **15.5%** of the take sits below ratio 8; **58.6%** is at ratio ≥20, i.e. prey an order of
magnitude smaller than the ceiling allows. The prey lengths say the same: **72.9% of what herring
takes is ≤0.75 cm**, and only **4.5% exceeds 2.5 cm**, against a ceiling of 2.6–2.7 cm. The length
change moves that ceiling by **0.072 cm** — across a region containing almost no predation.

Per the pre-registration this is the **NOT SUPPORTED** branch. The conjecture is dead; it is the
fourth of this investigation to fall, and I am not substituting another.

## What the size-class attribution does show

Herring's predation on stickleback eggs is spread across **its entire size range, 0–28 cm**, with no
concentration — the largest 1 cm bin is 9.4% of the take, and bins from 10–18 cm each contribute
6–9%. Young-of-year predation is similarly broad.

**This is the management-relevant result, and it is a negative one:** because every herring size
class participates roughly in proportion to its abundance, no size-selective measure on herring
(mesh size, minimum landing size) would differentially relieve predation on stickleback. The
coupling runs through herring's *total* abundance, not through any particular cohort.

Predator ranking is unchanged from Run 8 — herring 1.62e11 early-stage prey, sprat 6.4e10, smelt
2.7e10, pikeperch 1.2e10, flounder 1.1e10.

## The residual: the coupling is NOT proportional to herring biomass

Normalising herring's egg take by the eggs actually produced (which differ across arms):

| | shift −2 | shift 0 | shift +2 |
|---|---|---|---|
| per-egg predation rate by herring | 0.0889 | 0.0672 | 0.0682 |
| change vs base | **+32.3%** | — | **+1.5%** |
| herring biomass change | +5.6% | — | −10.0% |
| ratio (mass-action would be 1.0×) | **5.8×** | — | **−0.2×** |

Simple mass action predicts the per-egg rate to track herring biomass one-for-one. It does not, in
**either** direction: at shift −2 the rate rises nearly six times faster than biomass, and at
shift +2 biomass falls 10% while the rate barely moves. The response is also **non-monotonic**,
unlike every other quantity in this investigation.

So herring's *abundance* is not the whole coupling. The obvious remaining candidate is **temporal
overlap** — moving herring's spawning window changes when herring are feeding relative to when
stickleback eggs are present — but that is **unmeasured**, and after four refuted conjectures it
gets named as a candidate and nothing more. Testing it means resolving predation on stickleback eggs
*within* the year, by timestep, against the two species' spawning curves.

## Caveats

* 1-year recording window (Python path ~55 s/step). The egg-predation figures reproduce Run 8's
  exactly because they come from the same instrumentation and window — this is **not** independent
  confirmation of them.
* Small size bins carry real noise; several show swings of >100% on <1% of the take. The conclusions
  above rest on the aggregate distribution, not on individual bins.

---

# Run 10 (2026-08-17): temporal overlap — REFUTED, structurally. Residual reported UNEXPLAINED.

Within-year resolution of herring's predation on stickleback eggs, 3 arms × 30 yr, seed 42, final
year (24 steps = one full seasonal cycle). Decomposition fixed before running:
`total eaten = Σ_t present[t]·rate[t]`, with the timing contribution isolated as
`I = Σ_t (present_t/Σpresent)·(rate_t/mean_rate)`.

## Refuted — and the reason is structural, not statistical

**Stickleback eggs exist only in steps 9–14 of 24, and that window is IDENTICAL in all three arms.**
Only herring's spawning file was changed, so stickleback's spawning curve never moves. Temporal
overlap could therefore only operate through how herring's feeding varies *within* those six fixed
steps — and it barely does:

| | shift −2 | shift 0 | shift +2 |
|---|---|---|---|
| overlap index `I` | 1.0625 | 1.0040 | 1.0604 |
| timing contribution | **+5.83%** | — | **+5.61%** |

The timing term is small **and nearly identical in both directions**, so it cannot discriminate
between arms whose per-egg rates move +37.7% and −6.1%. Note also the familiar shape: the
calibrated arm sits at a local *minimum* of overlap (1.0040 against ~1.06 both ways), the same
calibration-optimum artefact seen in Runs 3 and 6.

Per the pre-registration this is the **NOT SUPPORTED** branch. Temporal overlap is the **fifth**
conjecture of this investigation to fall.

## A defect in my own index, disclosed

`present[t]` was recorded in the step hook, which runs **after** mortality — so the ratio I computed
is `eaten / survivors`, not `eaten / initial`, and it inflates when predation is heavy. Correcting
approximately via `p = r/(1+r)`:

| | shift −2 | shift +2 |
|---|---|---|
| level term as computed | +30.12% | −11.06% |
| level term corrected | **+23.11%** | **−9.16%** |
| herring abundance | +3.14% | −3.05% |
| disproportion vs mass action | **7.4×** | **3.0×** |

The correction shrinks the effect but does not remove it: the per-egg predation probability still
moves 3–7× faster than herring abundance. The conclusion survives the defect, but the index as
published in the script was not the quantity I intended, and the corrected figures are the ones to
quote.

## Status of the residual: UNEXPLAINED

After ten measurement rounds the mechanism is identified and the *magnitude* of one limb is not.
Established: herring predation on stickleback eggs and young-of-year is the control (Runs 6, 8), it
is not gated by predator size (Run 9), and it is not a timing effect (this run). What remains
unexplained is why the per-egg predation **probability** moves 3–7× faster than herring abundance.

Candidates not tested: spatial co-location of herring and stickleback eggs, and a functional-response
/ prey-switching effect (herring taking proportionally more eggs when its zooplankton supply
changes). **I am not queuing a sixth conjecture**, as pre-registered. Anyone resuming this should
note that five plausible mechanisms have already been refuted here, and that the honest current
state is: *the direction and dominant predator are established; the gain is not.*

## Caveats

* 1-year window; per-step rates within the six-step egg window are visibly noisy (e.g. shift −2 at
  step 11 is 0.4427 against 0.2432 at base), so the timing index rests on six numbers per arm.
* Herring abundance here is summed standing abundance (+3.14%/−3.05%), a different quantity from the
  biomass figures used earlier (+5.6%/−10.0%); the disproportion holds against either.

---

# Run 11 (2026-08-17): spatial co-location — SUPPORTED on the shift −2 limb, unexplained on shift +2

Run 10 left this residual: the per-egg predation probability moved **+23.11% / −9.16%** while
stock-wide herring moved only ~±3%. Predation in OSMOSE is computed **per cell**, so an egg's
encounter rate depends on herring density in its *own* cell. Measured at `_mortality`, i.e. on state
**before** that step's predation (measuring after would deplete eggs exactly in the high-herring
cells and bias toward a false negative). Numba stays on, so this recorded **3 years**, not 1.

## A first pass I discarded

I initially weighted herring by **abundance**. That was wrong: 99.3% of herring's numbers are age-0
larvae which do ~1.9% of the egg predation (Run 9's size table), and shifting herring's spawning
moves *when* that cohort exists — so the index swung 6× for reasons unrelated to who eats eggs, and
put the highest exposure in the arm with the *lowest* predation. Ingestion capacity scales with
**biomass**. Rerun with biomass weighting; the abundance-weighted numbers are discarded, not
interpreted.

## Result

Exposure decomposes exactly as `Hbar = (mean herring per occupied cell) × C`, verified numerically.

| | Δ(−2) | Δ(+2) |
|---|---|---|
| **per-egg predation probability (target)** | **+23.11%** | **−9.16%** |
| `Hbar`, all herring (biomass) | **+27.68%** | −1.32% |
| `Hbar`, herring ≥8 cm | **+23.47%** | +5.85% |
| — of which local density | +17.31% | −10.56% |
| — of which co-location `C` | +8.84% | +10.34% |
| stock-wide herring biomass | +5.60% | −10.00% |

**The shift −2 limb is explained.** This was the anomalous one — a 7.4× disproportion against stock
abundance. Egg-weighted herring exposure rose **+27.68%** (all herring) or **+23.47%** (≥8 cm, the
sizes that do the eating) against a target of +23.11%. The ≥8 cm figure matches to **0.36
percentage points**. Both components contribute: herring is more *concentrated* per occupied cell
(+17.31%, well above the +5.60% stock growth) *and* better co-located with eggs (+8.84%).

**The shift +2 limb is not explained.** Exposure is roughly flat (−1.32%) against a −9.16% target, a
7.8 pp gap, and the ≥8 cm cross-check has the **wrong sign** (+5.85%). Whatever suppresses predation
in that arm is not spatial exposure.

Note `C` rises in **both** arms (+8.84% / +10.34%) — the calibrated arm again sits at a local
*minimum*, the same artefact as Runs 3, 6 and 10. So co-location is not a clean monotonic driver on
its own; on the shift −2 limb it acts together with local concentration.

The number of cells containing eggs is unchanged (37.8 / 38.3 / 38.2, ±1.3%), so this is not eggs
being spread differently — it is herring moving relative to a fixed egg distribution.

## Status

First conjecture in five to be **supported**, and only on one limb. Honest summary of the whole
chain: the control is herring predation on stickleback eggs and young-of-year (Runs 6, 8); it is not
size-gated (Run 9) and not a timing effect (Run 10); the shift −2 gain is **spatial** — herring
concentrates and co-locates with eggs (this run); the shift +2 limb remains unexplained.

Not pursuing the shift +2 gap further without direction. The remaining untested candidate is a
functional-response effect — herring taking proportionally more eggs as its zooplankton supply
shifts — which would plausibly act asymmetrically between arms.

## Caveats

* One seed (42), 3-year window. The 0.36 pp agreement on the ≥8 cm shift −2 figure is a single
  coincidence on one arm and should not be read as precision.
* `Hbar` uses biomass as a proxy for ingestion capacity; true capacity also carries the
  species ingestion-rate constant, which is common to all arms and so cancels in these ratios.
* The 8 cm threshold is a cross-check drawn from Run 9's size distribution, not a config boundary.
