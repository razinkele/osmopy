# Baltic percid (perch / pikeperch) overshoot — diagnostic finding

**Date:** 2026-06-03
**Supersedes** the *inferred* "percids are an unfixable spatial limit" caveat in
`docs/baltic_shepherd_calibration_2026-05-30.md` and `docs/baltic_fr_calibration_2026-06-02.md`
with a *proven, quantified* understanding.
**Decision: do NOT build a fix now** — the proven cause is real but the payoff is two
weight-0.2 species; the knowledge below is the deliverable.

## Background

Across two shipped calibration features — Shepherd stock-recruitment (PR #50) and predator
functional response (PR-A/PR-B) — perch (sp4) and pikeperch (sp5) remained the dominant
residual overshoot (×80–205 over their ICES envelopes), and the strict ICES in-range count
never rose above 2/8. Both prior verdicts *inferred* this was a coarse-grid spatial limit
that recruitment/predation tuning could not fix. That inference was never tested — the
percids are weight 0.2 in the calibration objective, so the optimizer simply ignored them.

This diagnostic tests it directly.

## What the data shows

**Footprints are already confined (not over-broad).** Baltic grid = 616 ocean cells. Perch
adult occupies 62 cells (10.1%), pikeperch adult 27 (4.4%) — already as tight as the coarse
grid allows (the 2026-04-21 map-differentiation work confined them). So "concentrate
over-broad maps" is a **dead end** — the maps are not the lever.

**Seeded near target, but they explode at equilibrium.** Seeding biomass: perch 30 kt,
pikeperch 15 kt (targets 20 kt / 10 kt). Equilibrium: perch ×166, pikeperch ×127 (FR-off
phase-13 base, seed 42) — a ~100× growth *within* their confined cells.

## The β-probe (the decisive test)

Single 40-y sims on the phase-13 base, FR-off, seed 42. Cranked both percid Shepherd β to
the maximum (5.0 = strongest over-compensation) vs the calibrated baseline (perch β=1.6,
pikeperch β=0.5):

| species (weight) | baseline | β=5.0 (max) | |
|---|---|---|---|
| **perch** (0.2) | ×165.9, CV 0.03 | **×107.5, CV 0.42** | −35%, now oscillating |
| **pikeperch** (0.2) | ×126.7, CV 0.07 | **×23.0, CV 0.77** | −82%, wildly unstable |
| cod (1.0) | ×33.2 | ×37.5 | ↑ worse |
| herring (1.0) | ×2.34 | ×3.90 | ↑ worse |
| sprat (1.0) | ×5.00 | ×6.10 | ↑ worse |

## Proven conclusions

1. **The recruitment lever has real authority — the "unfixable" claim was false.** Pikeperch
   fell 82% and perch 35% under max β. DE leaving pikeperch at β=0.50 (under-compensation)
   was simply an unused lever (its weight-0.2 gave the optimizer no incentive).
2. **But recruitment cannot finish the job, and it destabilizes.** Even at *maximum* β, perch
   floors at ×107 (still 2× over the upper bound) and inter-annual CV explodes (0.42, 0.77) —
   classic over-compensation oscillation (paradox of enrichment). **Perch's residual floor is
   the signature of a carrying-capacity limit** (growth/resource in its coarse cells), not a
   recruitment limit. Pikeperch is more recruitment-responsive.
3. **Capping percids via recruitment damages the high-weight species.** Suppressing percid
   recruitment frees prey and pushes cod/herring/sprat *further* over — degrading the part of
   the calibration that actually matters for ICES credibility.

So the true picture (which neither prior doc captured): **pikeperch is substantially
recruitment-fixable; perch is carrying-capacity-limited; brute-forcing recruitment
oscillates the system and harms the high-weight stocks.**

## Disposition — not building a fix now

The clean fix for perch's carrying-capacity floor would be an **opt-in per-species spatial
carrying-capacity cap** (density-dependent local mortality scaled to habitat capacity) — a
general engine mechanism, gentler than β-cranking. It was considered and **deferred**: it is
medium engine effort plus a follow-on multi-hour calibration, its headline payoff is two
**weight-0.2** species, and it carries real risk of perturbing the sound high-weight fit. It
would also be the third recruitment/predation/capacity mechanism in a row chasing the same
low-weight species. Poor value-per-effort right now.

## Actionable takeaways (for whenever the Baltic is recalibrated)

1. **Free / raise pikeperch β.** It has been stuck at 0.50 (under-compensation); the probe
   shows a *moderate* increase (not max — max destabilizes at CV 0.77) would cut pikeperch
   substantially toward range. An easy partial win currently left on the table — but only
   worth taking inside a re-weighted run that also guards the high-weight species from the
   freed-prey side effect.
2. **Perch is carrying-capacity-limited.** A spatial carrying-capacity cap is the right
   mechanism *if and when* a **high-weight** species shows the same signature (confined
   habitat + recruitment-β floor + CV blow-up under hard capping). That is when the feature
   earns its keep — not for weight-0.2 perch alone.
3. **Don't chase the strict in-range count via the weighted objective.** The count weights a
   grid-under-resolved weight-0.2 pikeperch the same as a weight-1.0 cod; the weighted
   objective correctly tolerates the percid overshoot. A strict-count term would just force
   the destabilizing behavior above.

## Reproduce

```bash
python scripts/reconstruct_phase13_results.py
# baseline:
PYTHONPATH=. python scripts/evaluate_calibration_vs_ices.py \
    --params data/baltic/calibration_results/phase13_results.json --mode shepherd --years 40
# cranked: set stock.recruitment.shape.sp4 = sp5 = 5.0 in a copy, then:
PYTHONPATH=. python scripts/evaluate_calibration_vs_ices.py \
    --params <copy_with_beta5> --mode shepherd --years 40
```
