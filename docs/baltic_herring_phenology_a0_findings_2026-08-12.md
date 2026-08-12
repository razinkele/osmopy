# A0: shifting herring's spawning window — a real signal, an unexpected mechanism, and a refuted hypothesis

**Date:** 2026-08-12
**Experiment:** translate herring's spring spawning window −2 / 0 / +2 steps (autumn component
fixed; every vector sums to 1.000000), 50 yr × 5 seeds via the identity-gated harness, plus a
30-yr mortality-by-cause run per arm.
**Status:** complete. Not adoption-relevant — all four gates PASS. This is mechanism knowledge.

## Pre-registration

The interpretation rule was fixed **before any result existed**, because a naive reading of this
experiment produces a false positive (see §2):

1. Deltas within the SSB-re-sampling confound (−3.90% earlier / +8.10% later) ⇒ **null for
   phenology**.
2. Deltas of opposite sign or materially larger ⇒ a real signal beyond the confound.
3. A **match–mismatch** reading additionally requires the **starvation share to move**; biomass
   alone cannot distinguish mechanisms.

## 1. Result: a strong signal, opposite in sign to the confound

| arm | confound predicts (annual eggs) | measured herring biomass |
|---|---|---|
| shift −2 (earlier) | −3.90% | **+5.6%** |
| shift 0 | — | — |
| shift +2 (later) | **+8.10%** | **−10.0%** |

The model converts an **+8.1% egg-production advantage into a 10% biomass loss**. By rule 2 this
is a genuine mechanism, not arithmetic.

## 2. Why the confound had to be measured first

Herring **SSB bottoms at step 4 — exactly where the spring window opens — and peaks at step 17**.
Translating the window therefore re-samples SSB and changes total egg production with no phenology
involved. Unchecked, that alone predicts *later spawning → more herring*, which is precisely the
pattern the withdrawn spec's direction test would have scored as **confirming** the hypothesis.

A first attempt at this control used *total* biomass and gave the wrong answer (−6.3% / −0.4%);
only **SSB** enters egg production. Correcting it reproduced an independent reviewer estimate
(+8.1% vs their +6.6%). The confound is asymmetric and its sign depends on where SSB peaks — it
cannot be guessed.

## 3. Mechanism: predation, not starvation — the hypothesis is refuted

Mortality-by-cause for herring, final-decade means, relative to the unshifted arm:

| cause | shift −2 | shift +2 |
|---|---|---|
| **Starvation / Eggs** | **0 (absolute) in all arms** | **0** |
| Starvation / Juvenile | −1.7% | +1.3% |
| Starvation / Adult | +5.1% | −4.6% |
| **Predation / Eggs** | **+26.2%** | **+51.1%** |
| Predation / Juvenile | +2.3% | −6.0% |
| **Predation / Adult** | +2.9% | **+9.5%** |
| Fishing / Adult | −1.7% | −0.6% |

**Egg starvation is identically zero in every arm.** That is structural, not marginal: eggs carry
`first_feeding_age_dt = 1`, so they cannot feed and therefore cannot starve
(`reproduction.py:242-244`). Juvenile starvation moves ±1.7% — flat. **Rule 3 is not met, so the
match–mismatch reading is refused.**

What responds is **predation**: egg predation rises 26% (earlier) and 51% (later), and adult
predation rises 9.5% in the later arm. Note the unshifted arm sits at a local *minimum* of egg
predation — unsurprising, since the configuration was calibrated with that window.

So the spawning window's effect runs through **when eggs and adults are exposed to predators**,
not through whether larvae find food. This is a timing-of-exposure result, not a
timing-of-feeding result.

## 4. The direction contradicts the motivating literature

Polte et al. 2021 (doi:10.3389/fmars.2021.589242) report warming-driven **earlier** spawning
*reducing* western Baltic herring larval production. This model says earlier spawning makes herring
**better off** (+5.6%), later spawning worse (−10.0%).

Two honest readings, and the data here cannot separate them:

* the model lacks the mechanism Polte documents (egg starvation is structurally impossible, and
  larval feeding barely responds), so it cannot reproduce the finding even in principle; or
* the model's predation-mediated timing effect is real but describes a different system — `sp1` is
  an **aggregate** across all Baltic herring management units, in which Polte's western spring
  spawners are a few percent.

Either way, **this configuration cannot be used to represent Polte's mechanism**, and a
temperature-driven phenology feature built on it would have encoded a predation-timing artefact
under a match–mismatch label.

## 5. The unexpected result is larger than the intended one

**Stickleback moves −20.6% / +20.7%, inversely to herring and roughly twice its magnitude**, while
sprat barely moves (±0.4%) and cod_east is nearly inert (−1.4% / +0.2%).

Herring's spawning window is therefore a strong lever on stickleback — the species this project has
repeatedly found hardest to explain. The inverse coupling with herring, together with the predation
signature above, points at a predator/prey-timing interaction rather than food competition, but
that is a hypothesis this experiment did not test. It is the most promising thread here.

## 6. What this changes

* The withdrawn herring-phenology spec (`…2026-08-11-baltic-herring-phenology-design.md`) is
  confirmed dead on mechanism grounds as well as the five implementation criticals: its A0 would
  have measured a predation effect and its A1 would have mislabelled it.
* **Engine fact worth keeping:** eggs cannot starve. Any future design invoking larval food
  limitation must act on juveniles, not eggs, or add an egg-stage feeding term.
* No production change. All arms certify; the shifted vectors and arm files are committed for
  reproducibility only.

## 7. Method note

Confound measured before the experiment, interpretation rule frozen before results, mechanism
condition specified in advance and then **allowed to refute the hypothesis it was written for**.
The biomass result alone would have read as a phenology finding of the wrong sign; the pre-declared
starvation condition is what turned it into a predation finding instead.
