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

**So H1 is refuted on both limbs — starvation and diet.** Neither species' realised feeding changes
materially, and their overlap does not move. Note also that **neither eats the other**: no fish
prey appears in either diet, so the interaction is not predatory between them either.

## Leading hypothesis after both refutations: an equilibrium shift, not a trophic response

What the three measurements jointly show for stickleback: diet composition unchanged, starvation
unchanged, predation **up in both arms** (+12–20%), biomass **down 20.6% in one arm and up 20.7%
in the other**. No single trophic pathway produces a symmetric-in-magnitude, opposite-in-sign
response with unchanged feeding.

That pattern is more consistent with the perturbation moving the system between **alternative
equilibria** than with a mechanistic dose–response — and this repo has prior history with Baltic
bistability (the cod_east bistability harness). Stickleback is the natural candidate: its juvenile
predation rate is ~13–15/yr, so its equilibrium is held by a very tight mortality–recruitment
balance that a small timing change could tip either way.

**Decisive next test (cheap):** the A0 run already used 5 seeds. Extract stickleback's
**per-seed** final-decade values rather than the midpoint. Bistability predicts a **wide,
possibly bimodal** spread across seeds within an arm; a smooth mechanistic response predicts a
tight one. That distinguishes the hypotheses without a new simulation.

## What failed, and what to measure next

~~The diet output did not parse…~~ **Resolved 2026-08-13 — see the UPDATE above. H1 is now refuted on both limbs.**

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
