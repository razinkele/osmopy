# Baltic herring and sprat F vs ICES — evidence and recommendation

**Date:** 2026-10-29
**Question:** `validate_baltic_vs_ices_sag.py` reports model F at 1.83× ICES for herring and
0.44× for sprat (`docs/baltic_ices_validation_2026-10-03.md`), outside the script's soft
[0.5×, 1.5×] band. Re-tune F, or widen the band?
**Short answer:** neither, for herring — its F comparison is invalid, and the model already
lands the right catch. For sprat, re-tune F toward ICES: it doubles catch toward the ICES level
and keeps every species' envelope status unchanged. **The sprat change was adopted the same day
and certified 5/5 assessed at 50 yr × 5 seeds** (see the final section).

| | herring | sprat |
|---|---|---|
| **Model yield vs ICES catches, 2018–2022 mean** (3 seeds) | **294 kt vs 303 kt (0.97×)** | **123 kt vs 298 kt (0.41×)** |
| Configured base F (`fisheries.rate.base`) | 0.380 | 0.164 |
| Exploitation rate, yield / B(age ≥ 1), final decade, seed 0 | 0.362 | 0.162 |
| ICES F as the validator computes it (SSB-weighted, 2018–2022) | 0.208 | 0.369 |
| …which stocks that ICES F comes from | Gulf of Riga, Bothnian, western — **not** central Baltic | `spr.27.22-32` (the whole stock) |
| ICES Fbar age range | 3–6 / 3–7 | 3–5 |
| Model biomass at age 2+ (final decade, seed 0) | ~50 kt | ~377 kt |
| Cod-attributed M2 vs WGSAM SMS (`docs/baltic_sms_m2_validation_2026-10-03.md`) | 0.51× | 3.2× |
| **Recommendation** | **Do not re-tune; fix the comparison** | **Re-tune F to ~0.37 (ICES)** |

## Where the current values came from

Both are free parameters of the phase-13 differential-evolution fit (commit `4f79b879`,
2026-07-24; `data/baltic/calibration_results/phase13_equilibrium.json`, 1365 evaluations). Its
objective was the biomass envelopes only, with F bounded in [0.003, 1.0]. Nothing anchored F or
catch to ICES, and a biomass-only objective cannot separate F from the other mortality terms
fitted alongside it — herring's adult additional mortality landed at 2.25 yr⁻¹, 0.91 of the way to
its upper bound. The F values are whatever the fit needed, not estimates of F.

## Herring: the model catches the right amount; the F comparison is invalid

1. **Catch matches.** Model herring yield, final-decade mean, is 291 / 301 / 289 kt (seeds 0/1/2;
   mean 294 kt). The ICES catches of the four herring stocks the model covers sum to 303 kt
   (2018–2022 mean; central Baltic 166 kt, Bothnian 82 kt, Gulf of Riga 33 kt, western 22 kt). Ratio
   **0.97**. All four stocks report catches in tonnes, including `her.27.25-2932`, whose SSB is an
   index. The model removes the right tonnage; only the F *label* disagrees.
2. **The F comparator is the wrong stock.** The model's single herring group covers the whole domain
   (10–30° E, 54–66° N). The validator's ICES F excludes `her.27.25-2932` (central Baltic, the
   largest stock and more than half the catch) because its SAG SSB is an index. Its F is relative
   too: SAG assessment key 18982 (2024, SS3) gives FMSY = 1, Fpa = 1.2, and the 2025 advice
   ([asd.ices.dk/viewAdvice/3892](https://asd.ices.dk/viewAdvice/3892)) reports only F/FMSY and
   SSB/MSY Btrigger. **ICES publishes no absolute F for this stock**, so the 0.208 comparator is
   ~80% Bothnian herring (SSB ~580 kt of ~750 kt in the weighting). The snapshot README's statement
   that "F is dimensionless across all stocks" is wrong for this stock: its F is a ratio to FMSY.
3. **The model has almost no fish at the ages ICES measures F on.** Final-decade herring biomass at
   age 1+ is 0.80 Mt and at age 2+ only ~50 kt (seed 0, `biomassByAge`), against 2.54 Mt for
   `biomass()` (≥ 0.5 yr). ICES Fbar is over ages 3–6 / 3–7. With additional mortality of 2.25 yr⁻¹
   (≈ 89% removed per year from that cause alone) herring is a near-annual population in the model,
   so an F over ages 3–6 has no model counterpart. (Model age structure is a separate realism
   question; it does not bear on F.)
4. **Herring F is load-bearing elsewhere.** The certified herring baseline underlies C1 (thermal
   knob), the published B2 RCP×load table, and stickleback's regulation (herring egg predation).
   Moving herring F to chase an invalid comparator would invalidate those numbers — and would push
   catch below ICES.

**Recommendation:** leave herring F alone. Change the validator instead — report herring F without
a verdict (as is already done for index-unit biomass), with the reason: no absolute ICES F exists
for the stock the model mainly represents, and the age ranges do not correspond. A catch comparison
is the meaningful check for herring and passes at 0.97×. Correct the README sentence.

## Sprat: a real gap, and closing it costs little

The comparison is sound: one ICES stock (`spr.27.22-32`), tonnes units, a model age structure that
has the Fbar ages (age 2+ ≈ 377 kt). The exploitation rate matches the configured F (0.162 vs
0.164). The headline gap is **catch: 123 kt against ICES 298 kt (0.41×)**.

**Intervention test** — sprat `fisheries.rate.base.fsh2` 0.164 → 0.369 (ICES 2018–2022
SSB-weighted F), everything else the committed production config, 40 yr, seeds 0/1/2,
final-decade mean biomass (age ≥ 0.5 yr):

| species | envelope (kt) | baseline s0/s1/s2 (kt) | sprat F = 0.369 s0/s1/s2 (kt) | mean change | in envelope, base / arm |
|---|---|---|---|---:|:---:|
| cod_west | [4, 25] | 13 / 14 / 13 | 13 / 13 / 14 | +2.7% | Y / Y |
| cod_east | [60, 85] | 92 / 95 / 95 | 94 / 92 / 94 | −0.9% | n / n |
| herring | [800, 3000] | 2544 / 2567 / 2499 | 2528 / 2509 / 2565 | −0.1% | Y / Y |
| **sprat** | [800, 2500] | 944 / 952 / 929 | **855 / 833 / 851** | **−10.2%** | Y / Y |
| flounder | [20, 100] | 32 / 32 / 34 | 32 / 31 / 32 | −2.7% | Y / Y |
| perch | [8, 50] | 41 / 42 / 43 | 43 / 40 / 44 | +0.4% | Y / Y |
| pikeperch | [4, 25] | 1432 / 1369 / 1449 | 1373 / 1455 / 1354 | −1.6% | n / n |
| smelt | [20, 120] | 675 / 683 / 680 | 694 / 692 / 691 | +1.9% | n / n |
| stickleback | [50, 500] | 77 / 81 / 84 | 89 / 82 / 86 | +6.4% | Y / Y |

- **Catch roughly doubles toward ICES:** 241 / 233 / 239 kt (mean 238 kt, **0.80×** ICES).
- **No species changes envelope status.** Sprat biomass is strongly buffered: 2.25× the F costs
  10% of the biomass. Other species move within a few percent (stickleback +6%).
- **The trade-off, stated plainly:** the arm's sprat biomass (833–855 kt, age ≥ 0.5 yr — a wider
  measure than SSB) falls *below* the ICES SSB window of 892–1127 kt that the validation report
  cites, while the baseline (929–952 kt) sits inside it. The certification envelope is the operative
  test and it passes, but the change trades ~10% biomass realism for ~2× catch realism.
- **Floor margin is thin.** The worst seed sits at 833 kt, 4% above the 800 kt floor. Certification
  takes the worst case across seeds, so a different seed set or horizon could tip it.
- **Why catch stops at 0.80×:** the model's selectivity is knife-edge at age 1 (`a50 = 1.0`), while
  ICES Fbar is over ages 3–5 with partial selection at ages 1–2. Applying 0.369 from age 1 already
  fishes younger fish harder than ICES does, so the age-range mismatch exists for sprat too —
  smaller than herring's, because the ages are present. `a50` is a second lever if a later fit wants
  to close the remaining gap; it is not needed for this decision.
- The fit's low sprat F matches the SMS result that the model's cod predation on sprat is 3.2× the
  SMS estimate (excess predation means less fishing is needed to hold biomass in the envelope). But
  the intervention shows the envelope does **not** force the low F — the fit landed there because
  nothing anchored F or catch, not because a higher F is infeasible.

**Recommendation:** set sprat F to the ICES value (0.369, or ICES FMSY 0.34 if a management
reference is preferred over the 2018–2022 realised F). Re-certify with
`scripts/baltic_stability_certify.py` at the production horizon and seed set before committing.
Expect sprat near 850 kt and stickleback a few percent higher. If certification fails on the sprat
floor, the next lever is cod_east predation on sprat (3.2× SMS), not a lower F.

## Adopted 2026-10-29: `fisheries.rate.base.fsh2` 0.1637 → 0.369 — certification

`scripts/baltic_stability_certify.py`, production settings (50 yr, seeds 42/123/7/999/2024,
config-default seeding), run on the baseline and on the change back to back. Final-decade mean
range across seeds:

| species | baseline | sprat F = 0.369 | verdict, base / new |
|---|---|---|:---:|
| cod_west | 11.9–13.8 kt | 12.3–14.3 kt | PASS / PASS |
| cod_east | 64.0–66.4 kt | 64.2–66.3 kt | PASS / PASS |
| herring | 2.49–2.61 Mt | 2.47–2.61 Mt | PASS / PASS |
| **sprat** | 1013–1036 kt | **905–918 kt** (−10.9%) | PASS / PASS |
| flounder | 31.8–34.0 kt | 32.6–33.6 kt | PASS / PASS |
| perch (indicative) | 42.6–44.8 kt | 41.9–42.5 kt | PASS / PASS |
| pikeperch (indicative) | 1.38–1.46 Mt | 1.35–1.50 Mt | persists / persists |
| smelt (indicative) | 672–694 kt | 683–687 kt | persists / persists |
| stickleback (indicative) | 77.1–85.0 kt | 79.2–87.9 kt | PASS / PASS |

**Assessed 5/5 on both; no verdict changes.** At the certification horizon two of the caveats above
ease: the worst sprat seed sits 13% above the 800 kt floor (not 4%), and sprat's 905–918 kt now
falls *inside* the ICES SSB window (892–1127 kt), so the biomass-realism trade-off seen in the
40-yr runs does not appear at 50 yr. The 40-yr `cod_east` excursion above its ceiling is likewise
a horizon effect: at 50 yr it is in envelope (64–66 kt) in both arms.

Follow-on: `tests/test_tutorial_3species.py::test_headless_fallback_produces_equilibrium` asserted
a max/min equilibrium spread ≥ 10 across sprat, stickleback and total cod, justified by a stale
premise ("sprat ~5M t, cod ~1.5K t"). Measured on the tutorial config (seed 42) the spread is
sprat/stickleback: 11.90 before, 9.51 after. The threshold is now 5 — per-species magnitudes are
already pinned by `_PYRAMID_BOUNDS`, which still pass. `data/baltic/reference/f_byyear_sp2.csv`
(the opt-in, closed F1 hindcast series) still embeds the old base F verbatim; regenerate it with
`scripts/build_baltic_f_byyear.py` before any hindcast re-run.

## Instrument notes (found during this analysis)

- **`scripts/compute_mortality_balance.py` returns F = 0 and M = 0 for every Python-engine run.**
  `osmose/validation/fisheries.py` looks up Java 4.3 labels (`F`, `Mpred`, stages
  `Pre-recruits`/`Recruits`); the Python engine writes `Fishing`/`Predation`/`Additional` and stages
  `Eggs`/`Juvenil`/`Adult` (`osmose/engine/output.py:411`). Every lookup misses and the report prints
  zeros rather than an error.
- **The `Adult` stage is maturity-based, not age-based.** In the Python engine's `mortalityRate`
  output, final-decade adult F reads 0.224 (herring) and 0.100 (sprat) — about 0.6× the configured
  rate — because mature age-0 fish are below the age-1 selectivity cut and dilute the stage. It is
  not evidence that the model under-applies F.
- **`mortalityRate` rows are per saving interval** (`output.recordfrequency.ndt = 24` ⇒ one row per
  year). `annual_rate(..., steps_per_year=24)` on these files sums 24 *years*.
- **`biomassByAge` bin 0 spans [0, 1) yr and contains egg schools** (CLAUDE.md), so its age-0 value
  is not a standing stock of fish. Only age-1+ bins are used above. Whether `biomassByAge` is an
  interval mean or a save-step snapshot was not checked; the yield / B(age ≥ 1) exploitation rate is
  a cross-check on order of magnitude, not a precise F.
- **Baseline `cod_east` is above its envelope on all three seeds** (92–95 kt vs 85 kt ceiling) at
  this 40-yr horizon. Resolved by the certification below: at the 50-yr production horizon it is in
  envelope (64–66 kt).

## Reproduction

Runs: committed config via `osmose_demo("baltic")`, `simulation.time.nyear = 40`,
`PythonEngine().run(cfg, out_dir, seed)`, seeds 0/1/2, arm override
`fisheries.rate.base.fsh2 = 0.369`. ~190 s per run. Final decade = last 10 annual rows of
`osm_biomass_Simu0.csv` / `osm_yield_Simu0.csv` (both have a one-line preamble);
`osm_biomassByAge_<sp>_Simu0.csv` has none. ICES catches: `catches` field of the committed
`data/baltic/reference/ices_snapshots/*.assessment.json`, 2018–2022.
