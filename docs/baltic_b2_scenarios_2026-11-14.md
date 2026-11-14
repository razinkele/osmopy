# Baltic B2 re-run on v1.10.1: RCP × load scenarios, 2026-11-14

This re-runs `docs/baltic_b2_scenarios_2026-11-12.md` (v1.10.0) on **v1.10.1** (`8a578f9e`), whose
#255 changed sprat's maturity: it now matures by age only, at 1 yr (the 9 cm maturity size was
removed), and SP1b's cod larval rates were re-solved. Harness, arms, seeds, literature deltas and
caveat labels are unchanged; the labels array and the seeds are identical to v1.10.0's.

**Verdict: all gates pass, and the B2 results are essentially unchanged.** Herring falls 51% at
RCP4.5 and 72% at RCP8.5, within 1.3 points of v1.10.0. The demersal load contrast holds and grows
slightly.

**Run:** 6 arms × 5 seeds × 50 yr, in 4,785 s. Report:
`docs/diagnostics/baltic_b2_scenario_report_2026-11-14.json`.

**Gates:**
- zero-arm identity: 0 violations;
- load-through, Hill ordering and knob factor: true for every arm;
- predicted ΔK: identical to v1.10.0 (it depends only on the O₂ and Benthos fields).

## Herring (sp1): final-decade mean vs baseline

| arm | ΔT | v1.10.0 | **v1.10.1** |
|---|---|---|---|
| `rcp45_bsap` | +1.9 °C | −50.56% | **−50.83%** |
| `rcp45_ref` | +1.9 °C | −50.43% | **−51.53%** |
| `rcp85_bsap` | +2.9 °C | −71.64% | **−71.95%** |
| `rcp85_ref` | +2.9 °C | −71.26% | **−72.58%** |

- **Temperature is still the dominant driver.** Within each RCP, BSAP and REF differ by at most
  0.7 points. The response is monotone in ΔT.
- **Sprat's maturity change barely reaches herring's warming response.** Every arm moved by 0.3–1.3
  points, all slightly more negative. The arms run the same seeds, but no attribution run separates
  this shift from seed-level noise; it is reported, not interpreted.

## Demersal load contrast (BSAP vs REF within an RCP)

| RCP | cod_east: v1.10.0 → **v1.10.1** | flounder: v1.10.0 → **v1.10.1** |
|---|---|---|
| RCP4.5 | +20.56% → **+21.39%** | +12.09% → **+15.19%** |
| RCP8.5 | +15.06% → **+16.80%** | +9.76% → **+12.32%** |

- **The load signal holds:** cod_east +17–21%, flounder +12–15%.
- **The caveats from the earlier notes still apply:**
  - the ±1.9% floor is a historical cod_east reference (unchanged at 0.019);
  - the contrast covers only the O₂ → benthos-K pathway;
  - cod_east's deltas are conditioned on its RV trajectory.

## What changes for anyone citing B2

- **Nothing material.** Herring's warming decline on v1.10.1 is −51% (RCP4.5) and −72% (RCP8.5),
  against −50% and −71% on v1.10.0. The demersal load contrast is unchanged in sign and similar in
  size.
- **The reference-period caveat applies unchanged.** These are raw end-century literature deltas on
  a present-day baseline.

The v1.10.0 note now carries a superseded line.
