# Baltic B2 re-run on v1.10.0: RCP × load scenarios, 2026-11-12

> **Superseded 2026-11-14** by `docs/baltic_b2_scenarios_2026-11-14.md` (v1.10.1, sprat matures by
> age). Herring's warming decline is −51% / −72% there, within 1.3 points of the figures below; the
> demersal load contrast holds.

This re-runs `docs/baltic_b2_scenarios_2026-11-11.md` (v1.9.0) on **v1.10.0** (`05b517f3`), whose
#250 recalibrated herring:
- maturity 14.5 cm;
- adult additional mortality 0.2 /yr;
- larval rate re-fitted.

Harness, arms, seeds, literature deltas and caveat labels are unchanged. The labels array is
identical to August's.

**Verdict: all gates pass, and herring is now far more warming-sensitive.** With a realistic
spawning stock, herring falls 50% at RCP4.5 (+1.9 °C) and 71% at RCP8.5 (+2.9 °C). That is more than
any earlier version: v1.9.0 gave −17% / −27%, and August −40% / −62%. The demersal load contrast
holds.

**Run:** 6 arms × 5 seeds × 50 yr, in 4,835 s. Report:
`docs/diagnostics/baltic_b2_scenario_report_2026-11-12.json`.

**Gates:**
- zero-arm identity: 0 violations;
- load-through: true for every arm;
- Hill ordering: true for every arm;
- knob factor: true for every arm.

## Herring (sp1): final-decade mean vs baseline

| arm | ΔT | August 2026 | v1.9.0 | **v1.10.0** |
|---|---|---|---|---|
| `rcp45_bsap` | +1.9 °C | −40.23% | −17.20% | **−50.56%** |
| `rcp45_ref` | +1.9 °C | −41.00% | −17.93% | **−50.43%** |
| `rcp85_bsap` | +2.9 °C | −62.81% | −26.50% | **−71.64%** |
| `rcp85_ref` | +2.9 °C | −62.27% | −26.47% | **−71.26%** |

- **Temperature is still the dominant driver.** Within each RCP, BSAP and REF differ by at most
  0.4 pp. The response is monotone in ΔT.
- **The knob's egg factor is f ≈ 0.38 at +1.9 °C and 0.228 at +2.9 °C,** so egg production is cut by
  62% and 77%. Herring biomass now falls by 50% and 71%: 81% and 93% of the egg cut, against 28% and
  34% on v1.9.0.
- **Why it is larger, consistent with but not measured here:** `herring_compensation_2026-11-12.md`
  found that v1.9.0 compensated for lost eggs through a density-elastic rise in spawning biomass per
  fish. Only 1.5% of its biomass was mature, a thin tail above 18 cm. v1.10.0's herring is 93%
  mature, so little immature stock is left to mature faster when density falls. On that reading,
  much of the compensation that damped v1.9.0's response is gone. This run does not test it.

## Demersal load contrast (BSAP vs REF within an RCP)

| RCP | cod_east: v1.9.0 → **v1.10.0** | flounder: v1.9.0 → **v1.10.0** | predicted ΔK spread |
|---|---|---|---|
| RCP4.5 | +19.24% → **+20.56%** | +15.19% → **+12.09%** | 5.37 pp |
| RCP8.5 | +18.46% → **+15.06%** | +10.03% → **+9.76%** | 4.36 pp |

- **The load signal holds:** cod_east +15–21%, flounder +10–12%.
- **The predicted ΔK is unchanged from v1.9.0,** as expected. It depends only on the O₂ and Benthos
  fields, which #250 did not touch.
- **The caveats from the v1.9.0 note still apply:**
  - the ±1.9% floor is a historical cod_east reference;
  - the contrast covers only the O₂ → benthos-K pathway;
  - cod_east's deltas are conditioned on its RV trajectory.

## What changes for anyone citing B2

- **Herring's warming decline on the current model is −50% (RCP4.5) and −71% (RCP8.5).** It
  belongs to the v1.10.0 herring, whose spawning stock matches the assessment data far better.
- **Neither earlier figure is current.** The v1.9.0 figures (−17% / −27%) belonged to a herring with
  a 1.5% spawning stock, and the August ones to the overstated food field.
- **The demersal load contrast is essentially unchanged.**
- **The reference-period caveat applies unchanged.** These are raw end-century literature deltas on
  a present-day baseline.

The v1.9.0 note now carries a superseded line.
