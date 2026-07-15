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

- **No calibration objective / loss change, and no change to DE-loop *behavior*.** This does NOT add a
  `reference_point_type=recruitment` target and does NOT touch `losses.py` / `_ObjectiveWrapper` /
  `biomass_targets.csv`. It DOES add a gated recruitment-stat block to `run_simulation` (which the DE loop
  shares), but that block only runs when `output.abundance.byage.enabled` is set — which the DE loop's
  `base_config` never sets — so **every DE evaluation is byte-for-byte unaffected** (no abundance-by-age
  written, no extra work). (The Spec 1 dispatch could carry a recruitment objective *later* if the diagnostic
  warrants it.)
- **No precise match.** The verdict is an order-of-magnitude check, not an RMSE — absolute model-whole-domain
  R vs summed-stock ICES R alignment is inherently loose.
- **cod + flounder get no recruitment comparison** (index/absent) — printed explicitly as "no clean ICES R",
  not silently dropped.
- No new standalone script; no UI change; no optimizer change.

## Design

### 1. Model recruitment (extracted in `run_simulation`, not `evaluate`)

**Architecture correction (in-loop review):** `evaluate()` does NOT hold an `OsmoseResults`. It calls
`calibrate_baltic.run_simulation()` (`scripts/calibrate_baltic.py:206–281`), which opens `OsmoseResults`
inside a `with tempfile.TemporaryDirectory()`, extracts flat `dict[str, float]` stats (`{sp}_mean`,
`{sp}_yield_mean`, …), calls `results.close()`, and returns only scalars — the results object and its output
dir are gone by the time `evaluate()` sees `stats`. So the model recruitment stat must be extracted **inside
`run_simulation`**, mirroring the existing `{sp}_yield_mean` block:

- **Enable `abundance_by_age` for this run** by setting `output.abundance.byage.enabled = "true"` in the config
  `run_simulation` builds. It is OFF by default (`config.py:1797`) and there is no dedicated recruitment
  output. This is **inert for the DE calibration loop**: `_ObjectiveWrapper` builds its own `base_config` that
  never sets this flag, so DE evals produce no abundance-by-age and pay no cost. The flag is set only on the
  diagnostic's `evaluate()` path (which passes it through to `run_simulation`).
- **`abundance_by_age()` is LONG-format** `[time, species, bin, value]` where `bin` is a **string** age label
  (`"0"`, `"1"`, … — whole-year bins starting at 0; `output.py:182`), NOT a wide per-age frame. So model R for
  a species = **mean of `value` over rows where `bin == str(recruitment_age)`**, over the same trailing window
  `run_simulation` already uses for `{sp}_mean` (`n_eval_years`). Compare `bin` as a **string** (both the frame
  bin and the ICES `recruitment_age` are strings) to avoid a silent no-match. Emit `{sp}_recruitment_mean`
  into `species_stats`.
- **`run_simulation` stays snapshot-agnostic — the ages are passed IN, not looked up.** Give `run_simulation` a
  new optional parameter `recruitment_ages: dict[str, str] | None = None` (species → age-bin string). Only the
  diagnostic's `evaluate()` path provides it (resolving each species' age via `_species_recruitment_age`, §2,
  and setting the config flag); `run_simulation` then extracts `{sp}_recruitment_mean` for each `(sp, age)`
  pair. When `None` (the DE loop — `_ObjectiveWrapper` passes nothing), the block is skipped entirely. This
  keeps the dependency direction correct (`evaluate_calibration_vs_ices.py` already imports `run_simulation`;
  nothing imports back into `calibrate_baltic.py`) and avoids a second hardcoded age derivation.
- **Guard the disabled/empty path:** when abundance-by-age isn't written (flag off, or an older run),
  `OsmoseResults.abundance_by_age()` (non-strict) returns a **bare `pd.DataFrame()` with zero columns** — so
  `df["bin"]` would `KeyError`. Guard with `if not df.empty and "bin" in df.columns` (and a `try/except` like
  the yield read); otherwise leave `{sp}_recruitment_mean` unset (the report then shows `—`).

**Load-bearing science caveat (in-loop review — document, don't silently compare):** `abundance_by_age` is a
**within-year MEAN across the `n_dt_per_year` sub-annual steps** (`simulate.py` averages the distribution over
the recording window), whereas ICES R is a **start-of-year cohort-strength** estimate. Age-0 fish have the
highest mortality and the age-0 bin also absorbs the year's spawning pulses, so **for age-0 species (herring)
the model mean reads systematically ~0.4–0.6× low vs ICES R, independent of SR calibration**; the age-1 case
(sprat) is far less affected. The report MUST carry this caveat and interpret a low **herring** ratio
cautiously — it is a measurement-basis artifact, weaker evidence of SR miscalibration than the same ratio for
sprat. (The [1/3, 3] band mostly absorbs a 0.4–0.6× pull, so a well-calibrated herring should still read OK.)

### 2. ICES recruitment reference (geomean, 2018–2022)

From `data/baltic/reference/ices_snapshots/`. **Reuse the existing snapshot plumbing in
`scripts/validate_baltic_vs_ices_sag.py`** — `_load_manifest`/`_load_assessment`/`_series_by_year` and its
`WINDOW_YEARS = range(2018, 2023)` + full-coverage-only-years pattern (the SSB-envelope code already sums a
quantity per year across mapped stocks over that window and keeps only years all stocks cover). Import those
helpers rather than re-deriving JSON parsing; only the recruitment-specific aggregation is new.

- **`_species_recruitment_age(snapshot_dir, species) -> str | None`** — single source of truth for the age:
  reads each mapped stock's `*.reference_points.json` `recruitment_age`; returns the common value if the mapped
  stocks agree (herring all `"0"`, sprat `"1"`), else `None` = "no clean ICES R". BOTH the ICES aggregation
  (below) and the model-side extraction (§1) consume this — no second hardcoded derivation.
- Per species, per year in **2018–2022**: **sum** ICES `recruitment` across the species' mapped stocks
  (`index.json::model_species_to_ices_stocks`; herring = 4 stocks, all numeric age-0; sprat = 1 stock, age-1),
  keeping only years all mapped stocks report a value.
- **`_ices_recruitment_geomean(...) -> float | None`** = geometric mean of those per-year summed recruitments
  (ICES's own convention for R reference points; geomean because R is log-normally noisy). Returns `None` for
  species with no clean numeric R (`_species_recruitment_age` is None) — cod (eastern index + age mismatch),
  flounder (empty). Also return the per-year **min/max** for the report (so the reference's noisiness is
  visible — sprat swings ~4.6×, herring ~2.7× over the window).
- **Summability is an assumption, not a verified tag:** the snapshot's `units_by_stock` records only SSB units,
  not recruitment units. The 4 herring stocks' recruitments are all absolute counts on a self-consistent scale
  (verified: same order of magnitude, no unit break; central `her.27.25-2932` reports absolute R even though
  its SSB is an index), so summing is sound — document this as an inferred assumption in the helper.

### 3. The report

A small table appended to the existing report; `evaluate()` unconditionally returns a dict, so also add the
recruitment rows to that returned dict for downstream use.

```
Recruitment (model vs ICES R geomean, 2018-2022)
  species    age  model_R   ICES_R_geomean [min–max]   ratio   verdict
  sprat      1    <n>       <n> [<min>–<max>]          <r>x    OK|FLAG
  herring    0    <n>       <n> [<min>–<max>]          <r>x    OK|FLAG (age-0: model reads ~0.4–0.6× low; see note)
  cod        —    —         —                          —       no clean ICES R (eastern index + age mismatch 0 vs 1)
  flounder   —    —         —                          —       no clean ICES R (none reported)
```

- **`ratio = model_R / ices_R_geomean`**; **verdict = OK if `1/3 ≤ ratio ≤ 3`, else FLAG** (loose band by
  design — the question is "roughly realistic?", and absolute scale alignment is uncertain).
- Print the ICES geomean **with its per-year min–max** so the reference's ~3–5× interannual noise is visible.
- **Age-0 caveat in the report text:** a low **herring** ratio (age 0) is expected to some degree from the
  annual-mean-vs-cohort-census artifact (§1) — weaker evidence of SR miscalibration than a sprat (age 1) FLAG.
- If model R can't be computed (abundance-by-age off/empty, or the age bin absent) the row shows `—` with a
  reason, never a crash.

### 4. Boundaries

- **`_species_recruitment_age(snapshot_dir, species) -> str | None`** — lives in
  `evaluate_calibration_vs_ices.py` (alongside `_ices_recruitment_geomean`), the single source of the
  recruitment age (§2). `evaluate()` calls it once per species to build the `recruitment_ages` dict it passes
  to `run_simulation`; the ICES aggregation uses the same helper. `run_simulation` never calls it (it receives
  the resolved ages as a parameter — §1), so `calibrate_baltic.py` gains no dependency on the snapshot code.
- **`_ices_recruitment_geomean(snapshot_dir, species) -> tuple[float, float, float] | None`** — pure,
  unit-testable: `(geomean, min, max)` of the per-year summed recruitment over 2018–2022; `None` if no clean
  numeric R. Reuses `validate_baltic_vs_ices_sag.py`'s snapshot helpers.
- **Model recruitment** is emitted as `{sp}_recruitment_mean` by `run_simulation` (§1) — a scalar in the
  `species_stats` dict `evaluate()` already receives. (There is no `results` object at the report site, so
  there is NO `_model_recruitment(results, …)` helper as the earlier draft assumed; the age-bin filtering is a
  handful of lines inside `run_simulation`'s existing per-species loop, unit-tested via the `run_simulation`
  yield-stat test pattern with a fake `abundance_by_age()` long frame.)
- **`_recruitment_verdict(model_R, ices_R_geomean) -> tuple[float, str]`** — pure ratio + threshold.
- **`_format_recruitment_section(rows) -> str`** — a pure formatter (mirrors the existing `_print_report`
  pattern) fed an already-built list of row dicts; **it never runs the engine**, so it is fully unit-testable
  without a Baltic run (avoids the CI-fragile real-engine trap, [[feedback-ci-fragile-emergent-tests]]).

## Testing strategy

1. **`_species_recruitment_age` + `_ices_recruitment_geomean`** (pure, against the in-repo snapshot):
   sprat age `"1"`, herring `"0"`; cod → None (stocks disagree: index + age 0 vs 1), flounder → None (empty).
   sprat geomean = independently-computed geomean of `spr.27.22-32` R 2018–2022; herring = geomean of the
   per-year SUM across its 4 stocks; assert the returned `(geomean, min, max)` matches.
2. **`run_simulation` recruitment stat** (mirror the existing yield-stat test): call `run_simulation` with
   `recruitment_ages={"sprat": "1", "herring": "0"}` and a fake `OsmoseResults` whose `abundance_by_age()`
   returns a **long** frame `[time, species, bin, value]` (string bins `"0"`/`"1"`, >`n_eval_years` rows);
   assert `{sp}_recruitment_mean` = mean of `value` at `bin == recruitment_ages[sp]` over the trailing window.
   Also assert that with `recruitment_ages=None` (the DE-loop default) NO `{sp}_recruitment_mean` is emitted. Two guard cases: (a) `abundance_by_age()` returns a **bare `pd.DataFrame()`** (no
   columns — the disabled-output path) → the stat is left unset, no `KeyError`; (b) the frame has columns but
   no matching bin → unset.
3. **`_recruitment_verdict`**: ratio + threshold — 1.0 → OK; 0.4 → OK; 0.2 → FLAG; 5 → FLAG; **exact
   boundaries** 1/3 → OK and 3.0 → OK (inclusive), just-outside 0.33 → FLAG.
4. **`_format_recruitment_section`** (pure formatter, NO engine run): feed a stubbed list of row dicts
   (sprat/herring with model_R/geomean/ratio, cod/flounder as "no clean ICES R") and assert the rendered
   section contains the ratios, the age-0 herring caveat text, and the two "no clean ICES R" reasons. This is
   the only "integration"-flavoured test and it never touches `PythonEngine`/`run_simulation` (the real Baltic
   run stays a manual Verification smoke, per [[feedback-ci-fragile-emergent-tests]]).
5. **No regression**: the existing biomass-envelope evaluation + its Spec-1 catch-row exclusion are unchanged;
   `evaluate_calibration_vs_ices.py` still runs; DE evals produce no abundance-by-age (flag unset).

## Verification

- Run `scripts/evaluate_calibration_vs_ices.py --mode bh <a calibrated params file>`; the Recruitment section
  prints sprat + herring model R, ICES R geomean, ratio, verdict, and the two "no clean ICES R" rows.
- Spot-check: the ICES R geomeans are plausible (sprat ~tens of millions; herring ~tens of millions summed);
  the model R is a positive number of the same broad order (or a FLAG that tells us the SR is off — either is
  a useful diagnostic outcome).
- Unit tests green; `ruff` clean.

## Rollback

Additive and revertible: the ICES/verdict/format helpers + a report section in
`evaluate_calibration_vs_ices.py`, a small **gated** recruitment-stat block in `calibrate_baltic.run_simulation`
(inert unless `output.abundance.byage.enabled` is set — which only the diagnostic path does, never the DE
loop), plus tests. No optimizer, loss, `biomass_targets.csv`, engine, or config-format change; DE-loop
behavior is byte-identical. Reverting removes the section and the gated block.
