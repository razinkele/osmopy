# Baltic cod-predation mortality (M2) vs the WGSAM SMS key run — first comparison, 2026-10-03

**Issue #136.** First run of `scripts/validate_outputs_vs_ices.py --sms-m2` on the production
Baltic config (`data/baltic/baltic_all-parameters.csv`, 15 yr, fixed seeds, Python engine, output
prefix `osm`), trailing 5-yr model window against the SMS 2018–2022 window. **Report-only; nothing
is gated on this.** Method, provenance and caveats: `data/baltic/reference/ices_snapshots/README.md`
§ WGSAM SMS and the `sms_m2` block of `index.json`.

## WGSAM SMS cod-predation mortality (M2) — report-only, not gated

SMS M2 = annual instantaneous predation mortality BY COD on the prey stock, at age, weighted here across ages by biomass at age from the same key run (effectively ages 1+). Model = cod-ATTRIBUTED M2 on a biomass basis: annual tonnes of the prey eaten by the cod stocks (predatorPressure) over the prey's mean biomass (young-of-year excluded by output.cutoff.age). The two bases differ only through age structure. The model's TOTAL predation rate (all predators, per stage) is shown beside it so the non-cod residual is visible. SMS domain: ICES SD 25-32 excl. Gulf of Riga; the model grid also holds the western basin.

Model window: last 5 years of run · SMS window: 2018-2022

| prey | SMS stock | SMS M2 mean [min, max] | model cod M2 mean [min, max] | model/SMS | by cod stock | model total predation (Juvenil / Adult) | note |
|---|---|---:|---:|---:|---|---|---|
| herring | `her.27.25-2932` | 0.069 [0.060, 0.083] | 0.035 [0.032, 0.036] | 0.51 | cod_west 0.004, cod_east 0.031 | 0.612 / 0.567 | — |
| sprat | `spr.27.22-32` | 0.087 [0.079, 0.094] | 0.277 [0.252, 0.299] | 3.19 | cod_west 0.005, cod_east 0.272 | 3.126 / 0.603 | — |

## Reading it

- **Herring.** Cod-attributed M2 in the model is 0.035 yr⁻¹ against
  the SMS 0.069 (ratio 0.51) — half, with
  cod_east carrying almost all of it. But the model's **total** predation on herring is
  0.57 yr⁻¹ (adults), so cod is only
  6% of what eats herring in the model. The residual is the headline: the
  predatorPressure table shows **pikeperch** eating ~680 t of herring per step against ~1 t for each
  cod stock (the known percid overshoot, `docs/baltic_percid_overshoot_conclusion_2026-07-05.md`),
  and **grey seal is not in that table at all** — it has no column in the accessibility matrix and
  therefore eats every prey at full access (CLAUDE.md gotcha; `docs/baltic_c3_bioen_stage1_2026-09-05.md`
  §9), yet background predators are absent from `predatorPressure`, so its share is invisible here.
- **Sprat.** Cod-attributed M2 is 0.277 yr⁻¹, **3.2×**
  the SMS 0.087, again almost entirely cod_east. Total predation on juvenile
  sprat is 3.13 yr⁻¹ — an order of magnitude
  above anything SMS attributes to cod.
- **What this does and does not say.** The SMS figure is cod-only on an SD 25–32 stock; the model's cod
  stocks sit at ~2.5 Mt of herring and ~0.85 Mt of sprat biomass (SSB table above: herring 3.3× the ICES
  envelope, sprat 0.85×), so the cod rates are rates on a differently sized prey pool. The biomass-basis
  vs numbers-at-age basis difference is small next to these gaps. The comparison is doing its job:
  the predation engine's cod term is within a factor 2–3 of SMS on both stocks, and the discrepancy
  the validator surfaces is **who else is eating** — percids and the un-columned seal — not cod.

## Next

1. Fix the missing GreySeal column in `data/baltic/predation-accessibility.csv` — a recalibration
   decision (one cell moved cod_west 15 → 75 cm), not a validator change; this comparison should be
   re-run after it.
2. If the non-cod residual needs attributing, extend predatorPressure (or a sibling output) to
   background predators via `aggregate_diet_all_predators`, which already exists for diagnostics.
3. The SAG config validator is stale after the cod split (cod biomass row silently skipped) —
   tracked separately.
