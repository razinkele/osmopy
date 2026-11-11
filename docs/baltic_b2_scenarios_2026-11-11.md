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
Harness: `scripts/baltic_b2_scenario_ab.py`. The run set the module constant `REPORT_PATH`, which
defaults to `/tmp/b2_scenario_report.json`, to a private path, then called `run_b2()`. The JSON was
then copied to `docs/diagnostics/baltic_b2_scenario_report_2026-11-11.json`. Redirecting stdout
would not reproduce the report: stdout carries only status lines. The `labels` array, seeds and arms
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

Temperature is still the dominant driver. Within each RCP, BSAP and REF differ by at most 0.7 pp
(RCP4.5: −17.20% vs −17.93%, at different O₂ deltas). RCP8.5 has no zero-O₂ control, so a small O₂
contribution is not excluded. The response is still monotone in ΔT, and its size fell 2.3× at both
warming levels.

**Attribution (measured 2026-11-12, `docs/diagnostics/b2_herring_attribution_2026-11-12.md`).**
The knob is the C1 reproduction thermal gate (`reproduction.thermal.gate.*`, exponential): it
scales herring egg production by exp(β·(T − tref)). A pre-stated A/B reverted v1.9.0's data
changes (#236) to v1.8.8 at file level and re-ran this `rcp85_ref` pair. Its control reproduced
the −26.47% above to the last bit.

- **Reverting all of #236 restores −61.25%**, which is 97% of the August response. So nothing
  before v1.8.8 is needed to explain the halving.
- **No single change explains it.** Reverting herring's larval rate alone restores 27% of the
  gap; the LTL file alone, 6%; the cod pair alone, 2%. The autumn-spawning share alone moves it
  9% the *other* way.
- **Along one path, three steps carry it:**
  - herring's larval rate and the LTL file it was re-fitted onto, together, restore 66%;
  - the autumn share then adds 28%;
  - the cod pair adds the last 4%.
- **The response belongs to the jointly recalibrated state.** It is not an artefact of one input.
  Why that state is less sensitive was not measured.

## Demersal load contrast (BSAP vs REF within an RCP)

| RCP | cod_east, August → **now** | flounder, August → **now** | predicted ΔK spread, August → **now** | cod_east seed floor |
|---|---|---|---|---|
| RCP4.5 | +17.74% → **+19.24%** | +15.60% → **+15.19%** | 3.42 → **5.37** pp | ±1.9% |
| RCP8.5 | +17.03% → **+18.46%** | +12.67% → **+10.03%** | 2.85 → **4.36** pp | ±1.9% |

- **The measured contrasts held.** They are within 1.5 pp of August for cod_east, and within
  0.4–2.6 pp for flounder. All four are 5–10× the historical ±1.9% cod_east reference (see the
  last bullet). That reference is not a measured noise level for these contrasts on the current
  config, and not for flounder at all.
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
  Temperature is the dominant driver (see above).
- **The demersal load signal stands**: about +19% cod_east and +10–15% flounder for the BSAP vs
  REF contrast. It is 5–10× the historical cod_east seed reference.
- **Two caveats apply to that contrast.**
  - It is the O₂ → benthos-K pathway only. Plankton and nutrient pathways stay at baseline in
    every arm (the LTL-at-baseline label), so it is not a complete load-policy response.
  - cod_east's deltas are conditioned on its partly prescribed RV narrative trajectory (label 6
    of the August note), so its contrast is not as clean a measurement as flounder's.
- **The reference-period caveat applies unchanged.** These are raw end-century literature deltas
  on a present-day baseline.

The August note now carries a superseded line.
