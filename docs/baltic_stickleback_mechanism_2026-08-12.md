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
