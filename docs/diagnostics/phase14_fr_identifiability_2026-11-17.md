# Phase-14 FR half-saturation identifiability, rerun on the production base (issue #232), 2026-11-17

**Verdict, by the pre-stated rule: INCONCLUSIVE.** The cod_west positive control fails its gate
again, so no verdict is drawn on GreySeal or Cormorant, and phase 14's free set stays at four.

The base is new. After #243 the maintainer chose to build phase 14 on the certified production
parameters rather than the frozen phase-13 set, which was not a viable ecosystem post-split. The
rule was committed before this rerun (`phase14_fr_identifiability_prestated_2026-11-11.md`, with
its 2026-11-17 addendum); only the base changed.

**What the runs show this time:** a viable ecosystem in which no predator's K, anywhere in
[0.5, 5.0], moves any assessed stock or the objective beyond seed noise. That includes cod_west,
the control.

Harness: `scripts/phase14_fr_identifiability.py` at `c2489800`. Raw output:
`phase14_fr_identifiability_2026-11-17.json`. Runs: 24 × 40 yr, 12 single-threaded workers, 713 s.

## Primary instrument: the phase-14 objective

| predator | mean Δ (K 5.0 − 0.5) | pooled seed SD | ratio | criterion |
|---|---:|---:|---:|---|
| **cod_west (control)** | +0.0004 | 0.0029 | **0.12** | **fails → gate fails** |
| GreySeal | −0.0004 | 0.0041 | 0.10 | fails |
| Cormorant | −0.0026 | 0.0029 | 0.91 | fails |

## The runs (final-decade mean biomass, kt; 4 seeds per row)

| arm | objective | cod_west | cod_east | herring | sprat | flounder |
|---|---|---|---|---|---|---|
| cod_west K 0.5 | 0.725–0.730 | 20.0–20.7 | 105.4–106.4 | 1479–1514 | 919–932 | 46.6–48.0 |
| cod_west K 5.0 | 0.725–0.733 | 20.1–21.1 | 104.7–106.3 | 1487–1509 | 914–924 | 47.9–49.9 |
| GreySeal K 0.5 | 0.723–0.730 | 20.2–21.1 | 104.5–107.4 | 1485–1526 | 906–924 | 46.4–48.5 |
| GreySeal K 5.0 | 0.722–0.733 | 20.2–20.5 | 104.6–107.2 | 1450–1529 | 910–931 | 47.4–48.6 |
| Cormorant K 0.5 | 0.727–0.734 | 20.4–20.8 | 104.6–106.1 | 1507–1540 | 915–931 | 46.7–48.0 |
| Cormorant K 5.0 | 0.723–0.730 | 20.4–20.7 | 105.1–107.2 | 1480–1504 | 915–925 | 47.1–48.6 |

On the old base (2026-11-11) cod_west sat at 1.1–1.7 Mt and cod_east and flounder were extinct.
Here every stock is alive and cod_west is inside its 4–25 kt envelope.

**Secondary instrument (per stock, |mean Δ log10 B| / pooled SD):**
- One ratio exceeds 2: Cormorant's K on herring, 2.01. The secondary instrument does not override
  the primary verdict.
- cod_west's own biomass under its own K reads 0.91. On the old base it read 43.
- The rest are in the JSON.

## Is the instrument blind, or is the knob inert?

The pre-stated gate cannot tell these apart; it exists so that a failed control is not read as
"unidentifiable". Three checks, all outside the rule, bear on it:
- **The knob reaches the engine.** `EngineConfig` built from this base carries `fr_shape = 3` and
  `fr_halfsat` = 0.5 or 5.0 in the runtime slots of cod_west (0), pikeperch (5), GreySeal (9) and
  Cormorant (10).
- **The knob does change the simulation.** Same seed, 10 yr: K = 0.5 and K = 5.0 on cod_west give
  different trajectories. This is not evidence of an effect. A single stochastic trajectory
  diverges after any perturbation; type I against type III diverges by a similar amount.
- **The instrument is precise.** The objective's seed SD is 0.003 at a value of 0.73, and the
  assessed stocks' final-decade means vary by about 1–3% across seeds. A systematic K effect of a
  few percent on any assessed stock would have shown.

So the reading is that **on the production model, the type-III half-saturation K in [0.5, 5.0]
has no detectable effect at the calibration's own scale (40 yr, final decade), for any of the three
predators tested.** This is an interpretation, not the rule's verdict.

**Not measured: the mechanism.** A plausible one is food saturation. The type-III intake factor is
g = min(1, r, r²/(r² + K²)), with r the available prey over the predator's maximum intake. When r
is well above 5 at most visits, both bounds give g ≈ 1. The discriminating number would be the
share of each predator's visits with r below about 5, where K = 5 would bite; it was not taken.
Compensation could also absorb an intake change before the final decade.

## What this means for phase 14 and #232

- **By the decision table (control fails):** change nothing. The free set stays at four.
- **Beyond the table, for the maintainer:** on the production base, phase 14's four free
  parameters move nothing a calibration scores. A phase-14 DE would fit seed noise. Retiring
  phase 14 (option 3 of #243) is now supported by measurement, not just by "no FR in production".
- **#232's premise** (GreySeal's K barely identifiable because seals are food-saturated) is
  consistent with these runs, but so is every predator's K being inert, cod_west's included. The
  rule draws no GreySeal-specific verdict.
