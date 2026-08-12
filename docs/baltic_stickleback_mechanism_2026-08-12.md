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

## What failed, and what to measure next

The **diet output did not parse** (`diet_matrix()` returned a frame whose species column the script
could not locate — `StopIteration` on all three arms). That is the measurement that would have
tested competition directly, by showing whether stickleback's realised prey composition shifts when
herring larvae arrive earlier or later. Its absence is why H1 is refuted only on the starvation
limb, not on both.

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
