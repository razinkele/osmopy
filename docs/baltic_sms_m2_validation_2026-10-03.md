# Baltic cod-predation mortality (M2) vs the WGSAM SMS key run — first comparison, 2026-10-03

**Issue #136.** First run of `scripts/validate_outputs_vs_ices.py --sms-m2` on the production
Baltic config (`data/baltic/baltic_all-parameters.csv`, 15 yr, fixed seeds, Python engine, output
prefix `osm`), trailing 5-yr model window against the SMS 2018–2022 window. **Report-only; nothing
is gated on this.** Method, provenance and caveats: `data/baltic/reference/ices_snapshots/README.md`
§ WGSAM SMS and the `sms_m2` block of `index.json`. Artifacts: `baltic_sms_m2_validation_2026-10-03.json`
(the validator's JSON output for this run) beside this note.

## WGSAM SMS cod-predation mortality (M2) — report-only, not gated

SMS M2 = annual instantaneous predation mortality BY COD on the prey stock, at age, weighted here across ages by biomass at age from the same key run (effectively ages 1+). Model = cod-ATTRIBUTED M2 on a biomass basis: annual tonnes of the prey eaten by the cod stocks (predatorPressure) over the prey's mean biomass (young-of-year excluded by output.cutoff.age). The two bases differ only through age structure. The model's TOTAL predation rate (all predators, per stage) is shown beside it so the non-cod residual is visible. SMS domain: ICES SD 25-32 excl. Gulf of Riga; the model grid also holds the western basin.

Model window: last 5 years of run · SMS window: 2018-2022

| prey | SMS stock | SMS M2 mean [min, max] | model cod M2 mean [min, max] | model/SMS | by cod stock | model total predation (Juvenil / Adult) | note |
|---|---|---:|---:|---:|---|---|---|
| herring | `her.27.25-2932` | 0.069 [0.060, 0.083] | 0.035 [0.032, 0.036] | 0.51 | cod_west 0.004, cod_east 0.031 | 0.612 / 0.567 | — |
| sprat | `spr.27.22-32` | 0.087 [0.079, 0.094] | 0.277 [0.252, 0.299] | 3.19 | cod_west 0.005, cod_east 0.272 | 3.126 / 0.603 | — |

## Reading it

- **Herring.** Cod-attributed M2 in the model is 0.035 yr⁻¹ against the SMS
  0.069 (ratio 0.51) — half, with cod_east carrying nearly all of it.
  The model's **total** predation on adult herring is 0.57 yr⁻¹,
  so cod is ~6% of what eats herring in the model.
- **Sprat.** Cod-attributed M2 is 0.277 yr⁻¹, **3.2×** the SMS
  0.087, again almost entirely cod_east. Total predation on juvenile sprat is
  3.13 yr⁻¹, an order of magnitude above anything SMS
  attributes to cod.
- **The residual is pikeperch, and the budget closes.** Summing EVERY focal predator's consumption of
  each prey over the window and dividing by prey biomass reproduces the `mortalityRate` adult predation
  rate: 0.585 vs 0.567 yr⁻¹ on herring (103%), 0.582 vs 0.603 on sprat (97%). The shares: herring —
  pikeperch 0.530, cod_east 0.031, perch 0.020, cod_west 0.004; sprat — pikeperch 0.298, cod_east 0.272,
  perch 0.007, cod_west 0.005. So the focal predators account for the whole adult predation term, and
  **pikeperch, at 1.35 Mt in this run (the known percid overshoot,
  `docs/baltic_percid_overshoot_conclusion_2026-07-05.md`), is the predator the SMS does not have.**
  The background predators — grey seal (no accessibility column, hence full access; CLAUDE.md gotcha)
  and cormorant — are absent from `predatorPressure` by construction, but with the focal budget
  already closed on the ADULT stage their share of adult herring/sprat predation is at most a few
  percent here. The juvenile sprat term (3.1 yr⁻¹) is not budgeted by this table and is where egg/YOY
  predation by clupeids, percids and the background species would sit.
- **What this does and does not say.** The SMS figure is cod-only on an SD 25–32 stock; the model's
  prey pools differ from ICES (SSB table above: herring 3.3× the ICES envelope, sprat 0.85×), so the cod
  rates are rates on differently sized prey pools. The biomass-basis vs numbers-at-age difference is
  small next to these gaps. The comparison does its job: the predation engine's cod term is within a
  factor 2–3 of SMS on both stocks, and what the validator surfaces is the percid predation the SMS
  food web lacks — a calibration finding already on record, now with a number against an external
  multispecies benchmark.

## Next

1. Re-run after any percid or seal-matrix recalibration — this comparison is the cheap external check
   for both (the seal column fix is a recalibration decision: one cell moved cod_west 15 → 75 cm).
2. If the JUVENILE term needs attributing, extend predatorPressure (or a sibling output) to background
   predators via `aggregate_diet_all_predators`, which already exists for diagnostics.
3. The SAG config validator is stale after the cod split — issue #182.
