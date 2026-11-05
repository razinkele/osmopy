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
  The model's **total** predation rate on adult herring from `mortalityRate` is 0.57 yr⁻¹ — a
  stage rate on a different basis (numbers, adults only), so no cod share is read off the pair;
  the same-basis budget below is what attributes the predation.
- **Sprat.** Cod-attributed M2 is 0.277 yr⁻¹, **3.2×** the SMS
  0.087, again almost entirely cod_east. Total predation on juvenile sprat is
  3.13 yr⁻¹, an order of magnitude above anything SMS
  attributes to cod.
- **The residual is pikeperch.** On the SAME basis as the cod figure — each focal predator's annual
  consumption of the prey over the prey's mean biomass — the shares are: herring — pikeperch 0.530,
  cod_east 0.031, perch 0.020, cod_west 0.004 (focal total 0.585 yr⁻¹); sprat — pikeperch 0.298,
  cod_east 0.272, perch 0.007, cod_west 0.005 (focal total 0.582). As a consistency check only (the
  bases differ), those focal totals sit close to the adult-stage `mortalityRate` predation rates
  (0.567 and 0.603 yr⁻¹), which suggests the focal food web accounts for most of the predation on
  ≥0.5-yr fish, and
  **pikeperch, at 1.35 Mt in this run (the known percid overshoot,
  `docs/baltic_percid_overshoot_conclusion_2026-07-05.md`), is the predator the SMS does not have.**
  The background predators — grey seal (no accessibility column, hence full access; CLAUDE.md gotcha)
  and cormorant — are absent from `predatorPressure` by construction (*no longer since #198 —
  closed with them in the 2026-11-05 addendum below: together 2.5% of the herring budget and 7% of
  sprat's*); the consistency check above
  leaves little room for them on ≥0.5-yr fish, but it is not a stage-matched budget. The juvenile sprat
  term (3.1 yr⁻¹) is not budgeted by this table and is where egg/YOY predation by clupeids, percids
  and the background species would sit.
- **What this does and does not say.** The two rates are not equivalent: they differ through age
  structure (biomass basis vs numbers at age), through domain (SMS: SD 25–32 excl. Gulf of Riga; the
  grid also holds the western basin) and through the prey pool each is taken on (SSB table above:
  herring 3.3× the ICES envelope, sprat 0.85×). Each of those shifts the ratio on its own, so the
  comparison is indicative, not a like-for-like skill score. The comparison does its job: the predation engine's cod term is within a
  factor 2–3 of SMS on both stocks, and what the validator surfaces is the percid predation the SMS
  food web lacks — a calibration finding already on record, now with a number against an external
  multispecies benchmark.

## Next

1. Re-run after any percid or seal-matrix recalibration — this comparison is the cheap external check
   for both (the seal column fix is a recalibration decision: one cell moved cod_west 15 → 75 cm).
2. ~~If the JUVENILE term needs attributing, extend predatorPressure to background predators.~~
   Done in #198; budget closed in the addendum below. predatorPressure is tonnes, so it still
   cannot attribute egg/YOY predation by NUMBERS — that needs an age-resolved instrument.
3. The SAG config validator is stale after the cod split — issue #182.

## Addendum 2026-11-05 — the budget with the background predators

`predatorPressure` gained `GreySeal` and `Cormorant` columns in #198, so the same-basis budget above
can now include them. Fresh run: production config at master `a862a064` (which includes the sprat
F change 0.1637 → 0.369 of 2026-10-29, so these are not the October run's numbers), 15 yr, seed 0,
determinism keys on, diet output on; same trailing 5-yr window.

Each entry is that predator's annual consumption of the prey over the prey's mean biomass (yr⁻¹):

| prey | pikeperch | cod_east | Cormorant | GreySeal | perch | cod_west | all 11 predators | adult predation rate (`mortalityRate`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| herring | 0.560 | 0.033 | 0.010 | 0.006 | 0.020 | 0.004 | 0.632 | 0.566 |
| sprat | 0.297 | 0.292 | 0.028 | 0.018 | 0.008 | 0.005 | 0.648 | 0.620 |

- **The background predators are minor on ≥0.5-yr herring and sprat.** Seal + cormorant take
  0.016 yr⁻¹ of herring (2.5% of the budget) and 0.046 of sprat (7%). The October reading stands:
  the residual against SMS is **pikeperch**, not the seal.
- **The all-predator budget slightly exceeds the adult stage rate** (112% herring, 105% sprat). The
  two are on different bases (biomass vs numbers; the numerator counts every eaten tonne, including
  young-of-year, while the denominator excludes them by `output.cutoff.age`), so this is the same
  consistency check as above, not a closure to the percent.
- **Cod against SMS on this run:** herring 0.036 vs 0.069 (0.53×), sprat 0.298 vs 0.087 (3.43×) —
  unchanged in kind from October.
- **Juvenile sprat total predation is now 0.97 yr⁻¹** (October: 3.13). The sprat F change is the
  only production-config change in between, but the October run's seed was not recorded, so seed
  differences are not ruled out. The cause of that drop is not investigated here.
