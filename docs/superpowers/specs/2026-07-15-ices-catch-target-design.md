# Generalized ICES calibration targets + catch objective (Spec 1) — design

**Date:** 2026-07-15
**Status:** Draft (brainstorming → spec)
**Branch:** `feat/ices-catch-target`

## Motivation

The Baltic OSMOSE calibration scores each species against a single hand-set equilibrium
**biomass** band (`data/baltic/reference/biomass_targets.csv`) and currently lands only ~2–3/8
species in-band. Meanwhile the full ICES assessment **time series** are already in the repo
(`data/baltic/reference/ices_snapshots/*.assessment.json` — 51 yearly records per stock:
`ssb, recruitment, f, catches, landings, low_ssb, high_ssb`) plus `*.reference_points.json`
(`blim, bpa, fmsy, msy_btrigger, …`). The calibration uses almost none of it.

This spec adds the first additional quantitative signal — **catch/landings** — and generalizes
the target→objective machinery so later specs (recruitment, SSB-trajectory) can plug in without
re-plumbing.

**Why catch is the right first signal:** the model outputs `yield` in **tonnes**
(`OsmoseResults.yield_biomass()`, `osmose/results.py:429`), and ICES **landings are in tonnes for
every assessed stock — even the index-SSB ones** (e.g. eastern cod `cod.27.24-32`, 2022 landings
1146 t). So it's directly comparable, with **no circularity** (unlike recruitment, driven by the
SR relationship being tuned) and **no hindcast/transient wall** (unlike SSB-trajectory; the prior
interannual-RV hindcast was an honest-negative). It constrains *fishing realism*, not just
standing stock.

## Goal

1. **Generalize** the calibration target→loss dispatch so a target's `reference_point_type`
   selects which **model quantity** it is scored against — enabling **multiple targets per
   species** (a biomass target *and* a catch target).
2. Add a **catch objective** (model yield tonnes vs ICES landings tonnes) for the 4 ICES-assessed
   species (cod, herring, sprat, flounder), weighted **0.5** (secondary to biomass).
3. **Refine biomass bands from ICES data where SSB is in tonnes** (sprat cleanly; tonnes-unit
   herring subunits); keep the documented hand-set value for **index-SSB** species.

## Non-goals (YAGNI)

- **Recruitment objective** — separate Spec 2 (SR-circularity + magnitude scaling).
- **SSB trajectory / hindcast matching** — separate Spec 3, and a *de-risking spike first* (the
  interannual-RV hindcast was an honest-negative on exactly this).
- **F (fishing mortality) objective** — F is a model **input** (the fishing scenario), not a
  prediction; calibrating to it is calibrating to your own input. Never a valid target.
- The 4 data-poor species (perch, pikeperch, smelt, stickleback) have **no ICES stock**
  (`index.json` maps them to `[]`) → they stay biomass-only, unchanged.
- No new optimizer; no change to NSGA-II/surrogate/UI; no engine change.

## Design

### Units nuance (load-bearing — drives what's derivable)

From `data/baltic/reference/ices_snapshots/index.json` `units_by_stock`:

- **Landings/catches: tonnes for ALL assessed stocks** (measured, independent of assessment type)
  → a **catch target is valid for all 4** assessed species.
- **SSB: `index` (relative) for eastern cod (`cod.27.24-32`), central herring (`her.27.25-2932`),
  flounder (`fle.27.2223`)**; `tonnes` for western cod (`cod.27.22-24`), sprat (`spr.27.22-32`),
  and herring subunits `her.27.28 / .3031 / .20-24`. So an **absolute** biomass band can only be
  derived from `tonnes`-unit SSB — sprat cleanly, plus the tonnes herring subunits. For species
  whose dominant stock is index-SSB (eastern cod, central herring, flounder), **keep the existing
  hand-set biomass value** — do NOT fabricate an absolute band from a relative index.

**Honest value split:** the **catch objective is the star** (clean, tonnes-comparable, all 4
assessed species). The **biomass-band refinement is modest** (only sprat + tonnes herring
subunits get a genuine ICES-derived band).

### Species → stock aggregation (`index.json::model_species_to_ices_stocks`)

- cod → `cod.27.24-32` + `cod.27.22-24`
- herring → `her.27.25-2932` + `her.27.28` + `her.27.3031` + `her.27.20-24`
- sprat → `spr.27.22-32`
- flounder → `fle.27.2223`

Per-species ICES landings = **sum of stock landings** (tonnes) — sums cleanly across stocks
regardless of SSB assessment type — comparable to the model's whole-domain per-species yield.

### Methodology (locked)

- **Window:** 2018–2022 — this is the **ICES real-years window for deriving the target** (matches
  the existing `biomass_targets.csv` documented window). It is distinct from the **simulation
  equilibrium window** over which the *model* quantity (`{sp}_mean`, `{sp}_yield_mean`) is averaged
  — that stays whatever `calibrate_baltic.py` already uses for biomass. Model transient vs real
  decadal years are intentionally not year-matched (that's the Spec-3 trajectory question).
- **Catch band:** `target = mean(landings)`, `[lower, upper] = [mean − k·std, mean + k·std]`
  with **k = 1.5**, `std`/`mean` over the window's complete years, **floored positive**:
  `lower = max(mean − k·std, min_landings_in_window)` (a natural positive floor from the data,
  so `banded_log_ratio_loss`'s `lower > 0` precondition always holds).
- **Weight:** catch targets get **0.5** (biomass stays primary). Existing biomass weights
  (1.0/0.5/0.2) unchanged.

### A. Target model — multiple rows per species (`osmose/calibration/targets.py`)

The `BiomassTarget` dataclass + `load_targets` CSV loader already carry `reference_point_type`
(default `"biomass"`). The only change: **allow duplicate species rows** distinguished by type.
`load_targets` already appends every data row to a list (no dedup), so this works as-is —
verify and add a test. `reference_point_type ∈ {biomass, ssb, catch}`. (Keep the dataclass name
`BiomassTarget` to minimize churn; it is already type-agnostic in all fields but the name.)

### B. Generalized loss dispatch (`osmose/calibration/losses.py::make_banded_objective`)

Today it builds `target_dict = {t.species: t for t in targets}` (one target per species) and
reads `{sp}_mean` from `species_stats`. Change to:

- Iterate over **all targets** (not one per species).
- For each target, select the model quantity from `species_stats` via a small map:
  `_QUANTITY_KEY = {"biomass": "_mean", "ssb": "_mean", "catch": "_yield_mean"}` →
  `species_stats[f"{t.species}{_QUANTITY_KEY[t.reference_point_type]}"]`.
- Contribute `t.weight · banded_log_ratio_loss(quantity, t.lower, t.upper)` to `total_error`
  and to `worst_error`.
- **Stability penalties (CV/trend) stay attached to the biomass/ssb target only** (a catch
  target does not add its own CV/trend penalty) — preserves current behaviour exactly.
- **Missing model quantity** (e.g. a catch target but `{sp}_yield_mean` absent): same
  `100.0`-penalty path the current code uses for a missing `{sp}_mean`.
- **Unknown `reference_point_type`:** raise `ValueError` with the offending value (fail loud at
  objective-construction time, not silently).

**Backward-compatibility (exact):** a biomass-only target list (one `biomass`/`ssb` row per
species) must produce a **bit-identical** loss to today. Guaranteed by the mapping (`biomass →
_mean`) + stability penalties unchanged + iteration order preserved.

### C. Surface per-species yield in the run summary (`scripts/calibrate_baltic.py`)

The species-stats builder (`scripts/calibrate_baltic.py:260–281`) computes `{sp}_mean/_cv/_trend`
from the run's biomass. Add `{sp}_yield_mean` = time-mean of `results.yield_biomass()[sp]`
(tonnes) over the same equilibrium window used for `{sp}_mean`. `OsmoseResults.yield_biomass()`
(`osmose/results.py:429`) returns a wide `[time, <species…>]` frame; take the per-species column
mean. If yield output is absent/empty for the run, leave `{sp}_yield_mean` unset (the objective's
missing-quantity path then applies its penalty).

### D. Derive the ICES target data (`scripts/derive_ices_targets.py` → `biomass_targets.csv`)

A one-shot, re-runnable derivation script (mirrors `_pull_ices_snapshots.py`'s provenance style):

- Reads `data/baltic/reference/ices_snapshots/index.json` + `*.assessment.json`.
- For each of the 4 assessed species: sum stock **landings** over 2018–2022 (complete years),
  compute `mean/std/min` → emit a **`catch` row** (`target_tonnes=mean`,
  `lower=max(mean−1.5·std, min)`, `upper=mean+1.5·std`, `weight=0.5`,
  `reference_point_type=catch`, `source=…`, `notes=…`).
- For `tonnes`-SSB stocks (sprat; tonnes herring subunits): recompute the **biomass band** from
  the SSB series (mean over window, band from `low_ssb`/`high_ssb` or mean±1.5·std) and update
  the existing biomass row; **leave index-SSB species' biomass rows hand-set** (with a note).
- Writes `biomass_targets.csv` in place (additive catch rows + refined sprat biomass row),
  bumping `#! version` and `#! last_updated`, preserving all comment/provenance lines.

The script is **committed and run once**; the CSV it produces is the artifact the calibration
loads. (Don't wire it into the engine or CI.)

## Testing strategy

1. **`load_targets` multiple-rows-per-species** (`targets.py`): a CSV with a `biomass` and a
   `catch` row for the same species loads as two `BiomassTarget`s with the right types.
2. **Generalized dispatch backward-compat** (`losses.py`): a biomass-only target list yields a
   loss **bit-identical** (`==`, not approx) to the pre-change `make_banded_objective` on the same
   `species_stats` — the parity guard for the whole refactor.
3. **Dispatch with a catch target**: given `species_stats` with `{sp}_mean` and `{sp}_yield_mean`,
   a species with both a biomass and a catch target contributes `w_bio·bandloss(mean) +
   0.5·bandloss(yield_mean)`; assert the exact sum. Unknown `reference_point_type` → `ValueError`.
   Missing `{sp}_yield_mean` for a catch target → the 100.0-penalty path.
4. **Yield in species_stats** (`calibrate_baltic.py` builder): a small synthetic `OsmoseResults`
   (or fixture) with a known `yield_biomass()` frame → `{sp}_yield_mean` equals the column mean
   over the window.
5. **Derivation script** (`derive_ices_targets.py`): against the in-repo snapshot, per-species
   landings are summed across the mapped stocks over 2018–2022; a `catch` row is emitted for all
   4 assessed species; index-SSB species keep their hand-set biomass row; the emitted band
   satisfies `0 < lower ≤ target ≤ upper`.
6. **Calibration-integration smoke** (not CI-heavy): loading the regenerated CSV, a run's
   objective now includes the catch term (the loss differs from biomass-only on a run whose yield
   is off-target). No full NSGA-II — call the objective on a canned `species_stats`.
7. **No regression:** existing calibration tests green; the shipped `biomass_targets.csv` still
   loads; data-poor species (perch/pikeperch/smelt/stickleback) unaffected.

## Verification

- Unit + integration tests green; `ruff` + type-check clean; `derive_ices_targets.py` runs clean
  against the in-repo snapshot and its output diff on `biomass_targets.csv` is reviewable
  (additive catch rows + refined sprat biomass row + version bump).
- Sanity: a short DE/NSGA-II calibration run loads the new targets and reports both biomass and
  catch residuals per species (the existing residuals accessor surfaces the new terms).
- The catch band for each species is a plausible tonnes range vs the ICES landings it was derived
  from (spot-check sprat ≈ 2018–2022 mean landings).

## Rollback

Additive and revertible: one generalized `losses.py` dispatch (biomass-only path bit-identical),
one new `{sp}_yield_mean` stat, one new derivation script, and additive `catch` rows in
`biomass_targets.csv` (+ a refined sprat biomass row). No engine, optimizer, UI, or config-format
change. Reverting the CSV to biomass-only restores today's behaviour exactly.
