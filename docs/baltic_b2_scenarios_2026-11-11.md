# Baltic B2 re-run on v1.9.0: RCP × load scenarios, 2026-11-11

This re-runs `docs/baltic_b2_scenarios_2026-08-29.md` on **v1.9.0** (`95baf49b`) with the same
harness, arms, seeds and literature deltas. In between, the model's inputs changed:
- the corrected LTL forcing (#234, #236);
- herring's autumn spawning folded into spring;
- the larval re-fit (herring sp1 98.56 → 28.89 /yr, cod_west sp0 243.76 → 250.66 /yr);
- C2 (cod_east spawning moved to SD 25);
- the GreySeal accessibility column (#227).

**Verdict: all gates pass again, but herring's temperature response is less than half what was
published.**
- **Gates:** zero-arm identity holds (0 violations, 5 seeds); load-through, Hill ordering and the
  knob-factor instrument are true for every arm.
- **Herring:** −17% / −18% at RCP4.5 and −26.5% at RCP8.5. The August run gave −40% / −41% and
  −62% / −63%.
- **Demersal load contrast:** cod_east and flounder, BSAP vs REF, is essentially unchanged.

**The August herring figures should no longer be quoted** for the current model.

**Run:** 6 arms × seeds `[42, 123, 7, 999, 2024]` × 50 yr, in 4370 s (73 min; it was 1h52m).
Harness: `scripts/baltic_b2_scenario_ab.py`, run with its report redirected to
`docs/diagnostics/baltic_b2_scenario_report_2026-11-11.json`. The `labels` array, seeds and arms
are identical to the August report. Delta spec, citations and the six caveat labels: see the
August note. They are unchanged.

## Herring (sp1): final-decade mean vs baseline

| arm | ΔT | August 2026 | **v1.9.0** |
|---|---|---|---|
| `zero` | 0.0 °C | 0.0000% | **0.0000%** (bit-identical to baseline) |
| `rcp45_bsap` | +1.9 °C | −40.23% | **−17.20%** |
| `rcp45_ref` | +1.9 °C | −41.00% | **−17.93%** |
| `rcp85_bsap` | +2.9 °C | −62.81% | **−26.50%** |
| `rcp85_ref` | +2.9 °C | −62.27% | **−26.47%** |

The response still runs through temperature alone: BSAP and REF are near-identical within each
RCP. It is still monotone in ΔT. Its size fell 2.3× at both warming levels.

**Attribution is not measured.** The knob is the C1 reproduction thermal gate
(`reproduction.thermal.gate.*`, exponential): it scales herring egg production by
exp(β·(T − tref)). Of the changes since August, two act on that pathway directly:
- herring's larval mortality was re-fitted 3.4× lower;
- its autumn spawning share was folded into spring.

The others act on it indirectly:
- herring's baseline went from about 2.7 Mt to its 1.5 Mt target, through the re-fit on the
  corrected food field;
- the GreySeal column and C2 changed its predators.

Which of these halved the response was not tested. A one-change A/B on the RCP8.5 arm would settle
it.

## Demersal load contrast (BSAP vs REF within an RCP)

| RCP | cod_east, August → **now** | flounder, August → **now** | predicted ΔK spread, August → **now** | cod_east seed floor |
|---|---|---|---|---|
| RCP4.5 | +17.74% → **+19.24%** | +15.60% → **+15.19%** | 3.42 → **5.37** pp | ±1.9% |
| RCP8.5 | +17.03% → **+18.46%** | +12.67% → **+10.03%** | 2.85 → **4.36** pp | ±1.9% |

- **The measured contrasts held.** They are within 1.5 pp of August for cod_east, and within
  0.4–2.6 pp for flounder. All four remain 5–10× the cod_east seed-noise floor.
- **The predicted effective-K change rose 1.6×**: +5.37% / 0% / +3.16% / −1.20% for the four
  scenario arms, against +3.42% / 0% / +2.04% / −0.81% in August. No engine run feeds it. The
  builder weights the O₂ → Hill response by the `Benthos` field of
  `data/baltic/baltic_ltl_biomass.nc` (`load_benthos_k_weights`), and v1.9.0 regenerated that
  field. So the same O₂ deltas now land on a differently distributed benthos.
- **The ratio of measured contrast to predicted ΔK fell.** For cod_east it was 5.2× / 6.0×
  (RCP4.5 / RCP8.5) and is now 3.6× / 4.2×. The August note's reading of "strong nonlinear transmission" from benthos K
  to the demersals is weaker on the corrected field, but still well above 1×.
- **The ±1.9% floor is the August-era figure**, from the 2026-08-09 hypoxia A/B. This harness
  reports it as a constant and does not recompute it on the current config.

## What changes for anyone citing B2

- **Herring's warming decline is −17% to −27%** on the current model, not −40% to −63%.
  "Temperature-owned" still holds.
- **The demersal load signal stands**: about +19% cod_east and +10–15% flounder for the BSAP vs
  REF contrast. It is 5–10× noise.
- **The reference-period caveat applies unchanged.** These are raw end-century literature deltas
  on a present-day baseline.

The August note now carries a superseded line.
