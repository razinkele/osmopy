# Phase-14 FR half-saturation identifiability: pre-stated criterion (issue #232), 2026-11-11

Written and committed **before** any run. The harness is `scripts/phase14_fr_identifiability.py`.

## Question

Phase 14 fixes the functional-response shape at type III for cod_west (sp0), pikeperch (sp5),
GreySeal (sp15) and Cormorant (sp16), and tunes each predator's half-saturation K in [0.5, 5.0].
Issue #232 claims GreySeal's K is barely identifiable because seals are food-saturated. That
claim was measured as type III vs type I, over 1 year, on GreySeal's top prey only. Phase 14
never makes that comparison. It moves K within type III, scores 40-year final-decade biomass,
and covers every target species. So the claim is re-tested here at the calibration's own scale.

## Design

- **Base:** the `--phase 14` base config exactly as `run_calibration` assembles it: production
  Baltic, plus the frozen phase-13 parameters, plus Shepherd recruitment on all 9 focal species,
  plus type III on the four FR predators.
- **Treatment:** one predator at a time. Its K is set to 0.5 (lower bound), then 5.0 (upper bound).
  The other three stay at the x0 value K = 1.0.
- **Predators:** GreySeal (sp15) and Cormorant (sp16) are the question.
- **Positive control:** cod_west (sp0). Its K moved realized predation far beyond noise in
  `fr_diagnostic_phase14.json`.
- **Runs:** 40 years (phase 14's `--years` default), seeds 42, 1, 7 and 11, paired by seed.
- **Instruments:**
  1. Primary: the phase-14 objective (`make_objective`, default weights), scored on each run.
  2. Secondary: final-decade mean biomass (log10) of the five assessed stocks.

## Criterion

Fixed before the runs:

1. For each predator, take the paired bound-to-bound effect: Δ = value(K = 5.0) − value(K = 0.5),
   averaged over the 4 seeds.
2. The noise is the pooled across-seed SD at a fixed K: sqrt((var(K=0.5) + var(K=5.0)) / 2).
3. **Identifiable** ⇔ |mean Δ| > 2 × pooled SD, applied to the primary instrument (the objective).
4. The per-stock version of the same test is reported alongside. It does not override the
   primary verdict.

## Validity gate

**Validity gate:** if the cod_west control does **not** pass on the objective, the instrument
cannot see a knob known to matter. The result is then **inconclusive** for GreySeal and
Cormorant, not "unidentifiable", and no parameter is dropped.

## Decision

| control | GreySeal | Cormorant | action |
|---|---|---|---|
| pass | not identifiable | not identifiable | drop sp15 and sp16 from `get_phase14_params` (option 1 of #232); close #232 |
| pass | not identifiable | identifiable | drop sp15 only; record Cormorant as identifiable |
| pass | identifiable | either | #232 is refuted for GreySeal. Keep it, and close the issue with these numbers. Cormorant follows its own row |
| fail | — | — | inconclusive. Change nothing, and record why |

## Scope note

Production `data/baltic` sets no `predation.functional.response.*` key, so no phase-14 K is live
in any shipped configuration. Whatever the verdict, this changes only the calibration script's
free set, not the model.
