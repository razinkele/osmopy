---
name: Engine reproduction.py background-species fix
description: 2026-04-25 — fixed sex_ratio[:n_focal] slicing in reproduction.py so OSMOSE Python engine actually supports background species. Baltic-OSMOSE now runs grey seal + cormorant end-to-end.
type: feedback
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
`osmose/engine/processes/reproduction.py:62` did `config.sex_ratio * config.relative_fecundity * ssb`. With background species configured, `_merge_focal_background` (config.py:701-702) pads the per-species arrays to length `n_focal + n_bkg`, but `ssb` is built only for the focal species. Broadcast failed.

**Why:** The pathway has been latent in the Python engine since background-species support was added — EEC's config technically declares `species.type.sp55=background` but a separate parsing-pipeline issue prevents it from activating end-to-end (`parse_background_species` returns 0 species for EEC). Baltic-OSMOSE's 2026-04-24 grey-seal + cormorant additions were the first config to fully exercise the path, surfacing the bug.

**How to apply:** the fix is a 2-line change at reproduction.py:62 — slice both arrays:
```python
n_eggs = (
    config.sex_ratio[:n_sp]
    * config.relative_fecundity[:n_sp]
    * ssb
    * season_factor
    * TONNES_TO_GRAMS
)
```

Background species don't reproduce in OSMOSE; their entries past `n_sp` are zeros from the merge step. The slice is correct semantically (no behavior change for focal species) and has zero performance cost.

**Regression test:** `tests/test_reproduction_background_compat.py` runs a 1-year Baltic sim with the two predators activated, asserts `result.returncode == 0`. Skipped if the predator artifacts (`baltic_param-background.csv` + `baltic_predator_biomass.nc`) are absent.

**What this unlocks:**
- The Tier 1 plan's T3-T5 (grey seal + cormorant background species) is no longer abandoned. The artifacts on the feature branch can be cherry-picked or re-derived.
- Future Baltic configs can experiment with top-down predator pressure realistically.
- Future EEC work can fix the upstream `parse_background_species` issue so its sp55 background actually activates.

**What it does NOT solve:**
- Calibrated predator biomass × ingestion rate ≈ 78,500 t/yr total predation vs ~30M t total focal biomass = 0.26% removal. Biologically realistic but materially small. Adding predators alone doesn't fix the Baltic calibration gap (still 1/8 in ICES range with phase 2 params + active predators tested 2026-04-25).
- The 7+/8 in-range goal still needs a re-calibration (DE will find different parameters now that predator pressure is real, however small).

**Master state after fix (2026-04-25):**
- commit `37bc1d1` — engine fix + regression test + predator artifacts
- commit `f54bb26` — `osmose.configuration.background` line activated in `baltic_all-parameters.csv`
- Smoke verified: EEC 1y OK (parity), Baltic 1y w/ predators OK, Baltic 50y w/ phase 2 params + predators = 1/8 in range (cod, same as before).

**Recommended next session:**
- Re-run joint calibration (`scripts/calibrate_baltic.py --phase 12 --maxiter 15 --popsize 24 --popsize-mult 2 --years 50`) with predators now active. Cost ~8h. Compare to phase 12 result on 2026-04-24 (which had predators deactivated).
- Or: scale up predator standing biomass artificially (×3-5) to test whether stronger top-down pressure can pull more species into ICES range. This is biologically dubious but methodologically informative.
- Or: revisit the calibration target structure entirely (per `project_phase12_calibration.md` recommendation #2 — accept Baltic-OSMOSE as a qualitative tool rather than a target-matching tool).
