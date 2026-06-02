# Baltic predator functional-response (FR) calibration — phase 14 results

**Date:** 2026-06-02
**Plan:** `docs/superpowers/plans/2026-06-01-predator-functional-response-plan.md` (Part B)
**Design:** `docs/superpowers/specs/2026-05-31-predator-functional-response-design.md`
**Engine capability:** shipped in PR-A (merged to master `e1f4173`).

## Question

Does adding an opt-in **type-III** Holling functional response (a low-density prey
*refuge*) to selected Baltic predators, with the half-saturation `K` calibrated, improve
the Baltic calibration against ICES biomass envelopes — and does it measurably change
realized predation at the process level?

FR predators (type-III, K calibrated in phase 14): **cod (sp0)**, **pikeperch (sp5)**,
**GreySeal (sp14→slot 8)**, **Cormorant (sp15→slot 9)**.

## ⚠️ Load-bearing caveat — the phase-13 base is an APPROXIMATE reconstruction

Phase 14 freezes a phase-13 Shepherd base and tunes only the 4 FR `K`'s on top. **The
original PR #50 phase-13 result JSON no longer exists** (`data/baltic/calibration_results/`
is gitignored; the transient run output was deleted). It was reconstructed
(`scripts/reconstruct_phase13_results.py`) from **phase-12 mortality/fishing + the 15 SR
params documented in `docs/baltic_shepherd_calibration_2026-05-30.md`**.

This reconstruction does **not** reproduce the documented phase-13 equilibrium: under it the
FR-off biomasses are 2–3× higher than the doc's phase-13 (e.g. cod ×33 vs documented ×10.9;
sprat ×5.0 vs ×2.45; pikeperch ×127 vs ×167). The phase-13 SR params (β, ssb_half) were fit
*jointly with* phase-13's own mortality/fishing; pairing them with phase-12's mortality/
fishing yields a different, more-overshooting equilibrium. **Consequence:** the phase-14
objective (below) is **not comparable** to the documented phase-13 objective of 2.133. The
only valid comparison here is **FR-on vs FR-off on this same (approximate) base.**

## Phase-14 calibration run

`--phase 14 --optimizer de --seeds 3 --years 40 --popsize-mult 10 --patience 20
--wall-clock-cap-h 12 --checkpoint-every 5`, `OSMOSE_DE_WORKERS=16`, eff_popsize 40.
720 evaluations, 6.43 h wall-clock. **Objective: 3.361 single-seed / 3.386 ± 0.0155
multi-seed (3 re-rank seeds).**

**Calibrated K** (type-III `g(r)=r²/(r²+K²)`; K→small ≈ type-I/no-refuge, K→large = strong
intake suppression; DE bound [0.5, 5.0]):

| predator | calibrated K | reading |
|---|---|---|
| cod (sp0) | **4.98** (near upper bound) | strong suppression — DE crushes the dominant high-weight overshooter |
| pikeperch (sp5) | **0.53** (near lower bound) | near-neutralized — FR on pikeperch doesn't help the objective |
| GreySeal (sp14) | 1.62 | moderate suppression |
| Cormorant (sp15) | 3.95 | strong suppression |

## FR-on vs FR-off — ICES envelopes (same base, seed 42, 40 y)

| species (weight) | ICES range (t) | FR-off ×target | FR-on ×target | effect |
|---|---|---|---|---|
| cod (1.0) | 60k–250k | ×33.2 OVER | **×17.5 OVER** | overshoot ~halved |
| herring (1.0) | 800k–3M | ×2.34 OVER | ×3.29 OVER | ↑ |
| sprat (1.0) | 800k–2.5M | ×5.00 OVER | ×7.31 OVER | ↑ |
| flounder (0.5) | 20k–100k | **×0.00 EXTINCT** | **×0.37 under** | recovered from extinction |
| perch (0.2) | 8k–50k | ×165.9 OVER | ×189.8 OVER | ↑ (worse) |
| pikeperch (0.2) | 4k–25k | ×126.7 OVER | ×205.0 OVER | ↑ (worse) |
| smelt (0.3) | 20k–120k | ×13.7 OVER | ×11.7 OVER | ↓ |
| stickleback (0.2) | 50k–500k | ×2.17 in-range | ×2.12 in-range | ~ |
| **IN ICES RANGE** | | **1 / 8** | **1 / 8** | unchanged |

**Objective disposition:** the phase-14 FR-on objective (3.386) is ≤ the FR-off objective on
this base *by construction* — DE could have neutralized FR (all K→0.5) but instead chose
cod K=4.98, so the FR-on optimum is at least as good as FR-off. The weighted gain is
cod-dominated (cod ×33→×17.5 at weight 1.0). So **the objective does not regress**; the
strict in-range **count is unchanged (1/8→1/8)** because the helped species do not cross
their thresholds (cod still ×17.5; flounder still under) and the worsened species are
low-weight percids.

## Process diagnostic — realized predation mortality (FR-on − FR-off, 3 seeds, 40 y, 10-y window)

Realized predation mortality of predator *p* on prey *q* = Σ(eaten of q by p over last 10 y)
/ (mean biomass of q) / yr, via `aggregate_diet_all_predators` at diet width 16 (background
predators included). `***` = |mean delta| exceeds 2× the across-seed std (noise band).
**Per-predator calibrated K** used (the header's single `K=2.769` is a cosmetic mean; the
runs used cod 4.98 / pikeperch 0.53 / seal 1.62 / cormorant 3.95 — confirmed by the
direction of each predator's effect tracking its own K).

**Headline beyond-noise effects:**
- **cod** (K=4.98, strong): realized predation **drops on all 8 prey** — sprat −0.518±0.019,
  stickleback −0.333±0.017, smelt −0.040, herring −0.016, pikeperch −0.016, flounder −0.009
  (all `***`). The direct, intended type-III refuge effect.
- **GreySeal → flounder −0.442±0.053** and **Cormorant → flounder −0.237±0.027** (both `***`)
  — the mechanism behind flounder's recovery from extinction in the envelope table.
- **pikeperch** (K=0.53, near-neutral): realized predation *increases* on several prey
  (cod +0.135, smelt +0.079) — partly its near-neutral K leaving intake unchanged while
  cod's suppression frees prey, partly a denominator effect (prey biomass shifts).
- **28 / 32 predator-prey pairs exceed the 2σ multi-seed noise band.**

The multi-seed std is tight (e.g. cod→sprat ±0.019 on a −0.518 delta), so the effects are
decisively beyond seed noise — not the structurally-guaranteed "some negative delta" the
spec warned against.

## Verdict

**Mechanism: validated, decisively.** Type-III FR produces large, beyond-multi-seed-noise
reductions in realized predation for the suppressed predators (cod on all prey; GreySeal and
Cormorant on flounder), and the effect propagates to biomass (cod overshoot halved; flounder
back from extinction). The engine feature behaves exactly as designed.

**Baltic calibration improvement: NOT claimed.** Two honest reasons:
1. **Approximate base.** The phase-13 base does not reproduce the documented phase-13
   equilibrium, so no claim anchored to the shipped Shepherd calibration (2.133, 2/8) is
   defensible from this run.
2. **Strict in-range count unchanged (1/8 → 1/8).** FR helps the high-weight species (cod,
   and indirectly flounder) but not enough to cross ICES thresholds, while the low-weight
   percids (perch, pikeperch) — the known grid-under-resolved species from PR #50 — worsen.
   This mirrors PR #50's finding that the residual percid overshoots are a
   spatial-resolution limitation, not something a predation refuge can fix.

**Disposition (per spec §5):** ships as **engine capability (PR-A, already merged) plus this
documented Baltic exploration.** PR-B contributes reusable calibration + diagnostic tooling
(`phase 14`, `--mode shepherd-fr`, `scripts/fr_process_diagnostic.py`) and this honest
result; it is **not** presented as a calibration improvement.

## Clean follow-up (to actually settle the science)

Re-run on an **exact phase-13 base** — either (a) regenerate `phase13_results.json` via a
fresh full phase-13 run (~4.5 h), or (b) commit phase-13's full 39-param result to a tracked
location next time so it isn't lost to the `.gitignore`. Then re-run phase 14 + this
diagnostic. Only then is "FR improves (or doesn't improve) the *validated* Baltic
calibration" an answerable question.

## Reproduce

```bash
python scripts/reconstruct_phase13_results.py
OSMOSE_DE_WORKERS=16 PYTHONPATH=. python scripts/calibrate_baltic.py --phase 14 \
    --optimizer de --seeds 3 --years 40 --popsize-mult 10 \
    --patience 20 --wall-clock-cap-h 12 --checkpoint-every 5
# merge frozen base + calibrated K, then evaluate FR-off vs FR-on + diagnostic
PYTHONPATH=. python scripts/evaluate_calibration_vs_ices.py \
    --params <merged> --mode shepherd --years 40       # FR-off
PYTHONPATH=. python scripts/evaluate_calibration_vs_ices.py \
    --params <merged> --mode shepherd-fr --years 40    # FR-on
PYTHONPATH=. python scripts/fr_process_diagnostic.py \
    --params <merged> --halfsat-from-params --seeds 3 --years 40 --window 10
```

Note: a phase-14 result JSON holds only the 4 free `K`'s; to evaluate it you must merge it
with the frozen phase-13 base (the 39 SR/mortality/fishing params), as done above.
