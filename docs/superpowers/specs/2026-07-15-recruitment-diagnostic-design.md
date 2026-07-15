# Recruitment diagnostic — model vs ICES R (Spec 2, re-scoped) — design

**Date:** 2026-07-15
**Status:** Draft (brainstorming → spec)
**Branch:** `feat/recruitment-diagnostic`

## Motivation

Spec 2 of the ICES-strengthening sequence was originally "add a recruitment *objective*." Investigation
found recruitment is a poor *optimization* target:

- **Deterministically entangled with a target we already have.** Model recruitment `R = SR(SSB)`, and
  both the SR parameters and SSB are exactly what the calibration tunes — so a recruitment target mostly
  re-constrains the same SR the biomass target already shapes, adding little *independent* information.
- **Too noisy to pin.** ICES recruitment swings ~5× year-to-year (sprat 24–113 M over 2018–2022), so any
  defensible (wide) band barely constrains, while a tight band picks a fight with biomass on a near-random
  quantity.
- **Only 2 species.** sprat and herring have clean quantitative ICES R; eastern cod reports R as a relative
  *index* (western as absolute — not summable), and flounder has no recruitment at all.

But the real question underneath is a **validation** one, not an optimization one: *is the calibrated SR
producing roughly realistic recruitment?* Today nothing surfaces that. So Spec 2 is re-scoped to a
**diagnostic** — diagnose before optimizing. If it shows R is systematically off despite in-band biomass,
*that* evidence justifies a real objective later; if R looks fine, we've confirmed the objective isn't
needed. This mirrors the project's re-scoping discipline (HOLAS-3, fisheries validator, percid: target →
diagnostic when the quantity can't cleanly optimize).

## Goal

Add a **"Recruitment (model vs ICES)"** section to `scripts/evaluate_calibration_vs_ices.py` — which already
runs a calibrated Baltic parameter set and compares mean biomass to the ICES envelope — reporting, for
**sprat + herring**, the model's age-matched recruitment vs the ICES R geomean, with an order-of-magnitude
sanity verdict.

## Non-goals (YAGNI)

- **No calibration objective / loss change.** This does NOT add a `reference_point_type=recruitment` target,
  does NOT touch `losses.py` / `_ObjectiveWrapper` / `biomass_targets.csv` / the DE loop. Purely a report
  section. (The Spec 1 dispatch could carry a recruitment objective *later* if the diagnostic warrants it.)
- **No precise match.** The verdict is an order-of-magnitude check, not an RMSE — absolute model-whole-domain
  R vs summed-stock ICES R alignment is inherently loose.
- **cod + flounder get no recruitment comparison** (index/absent) — printed explicitly as "no clean ICES R",
  not silently dropped.
- No new standalone script; no UI change; no optimizer change.

## Design

### 1. Model recruitment (per species, one evaluate run)

`evaluate_calibration_vs_ices.py::evaluate()` runs the model and reads `OsmoseResults`. Add:
- **Enable `abundance_by_age`** for that run by setting `base_config["output.abundance.byage.enabled"] = "true"`
  (it is OFF by default — `config.py:1797` — and there is no dedicated recruitment output). One extra output;
  negligible for a single diagnostic run (not per-eval).
- **Model R for a species = mean of `results.abundance_by_age(species)` at the ICES recruitment-age bin**,
  over the same trailing window `evaluate()` already uses for mean biomass. sprat → age 1, herring → age 0.
  `abundance_by_age()` returns a per-age frame; select the column/bin whose age equals the species' ICES
  `recruitment_age`. (Implementation must verify the model's age binning is in whole years starting at 0 so
  age-0/age-1 map directly; if the youngest bin isn't 0, document the offset used.)

### 2. ICES recruitment reference (geomean, 2018–2022)

From `data/baltic/reference/ices_snapshots/`:
- Per species, per year in **2018–2022**: **sum** ICES `recruitment` across the species' mapped stocks
  (`index.json::model_species_to_ices_stocks`; herring = 4 stocks, all numeric age-0; sprat = 1 stock,
  age-1). Skip empty values.
- **ICES R reference = geometric mean** of those per-year summed recruitments (ICES's own convention for R
  reference points). Geomean, not arithmetic mean, because R is log-normally noisy.
- `recruitment_age` comes from each stock's `*.reference_points.json` (sprat 1, herring 0). Assert the mapped
  stocks agree on a single recruitment age per species (they do: herring all 0, sprat 1); if a species'
  stocks disagree, treat it as "no clean ICES R".

### 3. The report

A small table appended to the existing report (and to the JSON result if `evaluate()` returns one):

```
Recruitment (model vs ICES R geomean, 2018-2022)
  species    model_R        ices_R_geomean   ratio   verdict
  sprat      <n>            <n>              <r>x    OK|FLAG
  herring    <n>            <n>              <r>x    OK|FLAG
  cod        —              —               —       no clean ICES R (eastern index)
  flounder   —              —               —       no clean ICES R (none reported)
```

- **`ratio = model_R / ices_R_geomean`**; **verdict = OK if `1/3 ≤ ratio ≤ 3`, else FLAG** (loose band by
  design — the question is "roughly realistic?", and absolute scale alignment is uncertain).
- If model R can't be computed (abundance-by-age missing/empty, or the age bin absent) the row shows `—`
  with a reason, never a crash.

### 4. Boundaries

- `_ices_recruitment_geomean(snapshot_dir, species) -> float | None` — pure, unit-testable (sum stocks over
  window → geomean; None if no clean numeric R).
- `_model_recruitment(results, species, recruitment_age, window) -> float | None` — pure over a results-like
  object; None if the age bin is unavailable.
- `_recruitment_verdict(model_R, ices_R) -> (ratio, verdict)` — pure ratio + threshold.
- The report/print glue lives in `evaluate_calibration_vs_ices.py` alongside the existing biomass section.

## Testing strategy

1. **`_ices_recruitment_geomean`**: against the in-repo snapshot, sprat = geomean of `spr.27.22-32`
   recruitment 2018–2022; herring = geomean of the per-year SUM across its 4 stocks; cod/flounder → None
   (index/absent). Assert the sprat value equals an independently-computed geomean.
2. **`_model_recruitment`**: feed a synthetic per-age abundance frame with known age-0/age-1 columns; assert
   it selects the recruitment-age bin and means over the window; returns None when the bin is absent.
3. **`_recruitment_verdict`**: ratio math + threshold — ratio 1.0 → OK; 0.4 → OK (≥ 1/3); 0.2 → FLAG; 5 → FLAG.
4. **Report integration**: run `evaluate()` (or its report builder) with a small stubbed result and assert the
   Recruitment section lists sprat/herring with ratios and prints cod/flounder as "no clean ICES R".
5. **No regression**: the existing biomass-envelope evaluation and its catch-row exclusion (Spec 1) are
   unchanged; `evaluate_calibration_vs_ices.py` still runs.

## Verification

- Run `scripts/evaluate_calibration_vs_ices.py --mode bh <a calibrated params file>`; the Recruitment section
  prints sprat + herring model R, ICES R geomean, ratio, verdict, and the two "no clean ICES R" rows.
- Spot-check: the ICES R geomeans are plausible (sprat ~tens of millions; herring ~tens of millions summed);
  the model R is a positive number of the same broad order (or a FLAG that tells us the SR is off — either is
  a useful diagnostic outcome).
- Unit tests green; `ruff` clean.

## Rollback

Additive and revertible: three pure helpers + a report section in `evaluate_calibration_vs_ices.py`, one
config flag flip inside the diagnostic's own run, plus tests. No engine, optimizer, loss, calibration-loop,
config-format, or data change. Reverting removes the section.
