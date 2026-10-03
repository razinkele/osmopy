# ICES SAG Snapshots

Frozen JSON copies of ICES Stock Assessment Graphs (SAG) data used to
validate Baltic OSMOSE calibration inputs (biomass targets, fishing
mortality rates). Taken via the `ices` MCP server (same payload shape as
`get_stock_assessment` / `get_reference_points`):

- `{stock}.assessment.json` — list of year-row dicts with lowercase keys
  (`year`, `ssb`, `recruitment`, `f`, `catches`, `landings`, `discards`,
  `low_ssb`, `high_ssb`).
- `{stock}.reference_points.json` — dict with `flim`, `fpa`, `fmsy`,
  `blim`, `bpa`, `msy_btrigger`, `f_age_range`, `recruitment_age`,
  optionally `note`.

Advice year: **2024** for seven stocks; **2022** for `cod.27.22-24`
(western Baltic cod is category-3 in 2024 — no SSB/F time series
published, so the last full assessment is used). Covers the 2018–2022
window used for calibration targets in `biomass_targets.csv`.

Pull helper: `scripts/_pull_ices_snapshots.py` (one-shot — not re-run by
CI). Hits the ICES SAG REST API directly, applies the same flattening
the MCP server does, and writes one file per stock.

## `index.json`

- `advice_year` — primary advice year (2024).
- `created` — date of the snapshot pull.
- `model_species_to_ices_stocks` — manifest mapping OSMOSE model species
  to ICES stock keys. Empty list = coastal / data-limited species with
  no SAG assessment; the validator tolerates these.
- `units_by_stock` — `"tonnes"` or `"index"` per stock. See "Unit
  caveat" below — the ICES API's `StockSizeUnits` field is unreliable,
  so this is derived from the `Blim` magnitude (Blim < 100 → index).
- `advice_year_by_stock` — per-stock advice year (overrides the
  top-level `advice_year` for stocks on an alternate cycle, e.g.
  `cod.27.22-24` uses 2022).

## Recruitment units (`recruitment_index_stocks`)

The SSB unit does **not** imply the recruitment unit. `her.27.25-2932` has an index
SSB but reports recruitment as absolute numbers (6–29 million in 2018–2022), while
`cod.27.24-32` reports recruitment on the same relative scale as its SSB (0.4–1.6).
`index.json['recruitment_index_stocks']` lists the stocks whose R is relative;
`scripts/evaluate_calibration_vs_ices.py` skips those in the recruitment comparison
(the model's R is a count) rather than inferring the unit from `units_by_stock`.

## Baltic flounder notes

ICES SAG publishes a **single** Baltic flounder stock: `fle.27.2223`
(Subdivisions 22–23, western Baltic). There is **no** `fle.27.24-32`
assessment in SAG — eastern Baltic flounder is data-limited and falls
under WKBALTIC/WKBFLAT benchmark notes, not routine advice. The plan
originally listed both; only `fle.27.2223` is pulled.

Baltic flounder is also assessed on a **biennial cycle** (even years):
2022, 2024, 2026, … Refreshing odd-year advice (2023, 2025) will not
bring a new flounder snapshot.

## Unit caveat (important)

Three of the eight stocks report SSB as a **relative biomass index**
(dimensionless, scaled to O(1)) rather than absolute tonnes:
`cod.27.24-32`, `her.27.25-2932`, `fle.27.2223`. The ICES SAG API
mislabels these as `"tonnes"` in the per-row metadata — the true unit
is inferred from the `Blim` magnitude. The validator skips biomass
envelope comparisons for index-unit stocks (nothing to compare tonnes
against an index) but still uses them for F-rate comparisons (F is
dimensionless across all stocks).

## WGSAM SMS cod-predation mortality (M2) — `wgsam_sms_baltic_2025.*` (issue #136)

Two files snapshot the ICES WGSAM **Eastern Baltic Sea SMS key run 2025** (the SMS model,
Lewy and Vinther 2004 as cited in the stock annex — an ICES CM paper, not resolved here;
stock assessor Morten Vinther, DTU Aqua), pulled at a pinned commit from the
public repository `ices-eg/wg_WGSAM`, folder `Baltic-2025-keyRun`, commit `f690d4ff`
(2025-10-08, "2025 baltic keyrun"). The WGSAM 2025 report was **not yet in the ICES library**
when this was taken, so the repository artefact is the primary source; replace the citation
when the report appears. Pull helper: `scripts/_pull_wgsam_sms_m2.py` (one-shot).

- `wgsam_sms_baltic_2025.m2_annual.csv` — verbatim `M2_annu_.csv`: `Year, scenario, Species,
  variable, Age, value`. M2 = annual instantaneous **cod**-predation mortality (per year) on
  Herring and Sprat at ages 0–8, 1974–2024, for BOTH the `2022 Key run` and the `2025 key run`
  (the revision between them is informative; the validator uses `scenario_used` from
  `index.json`). The annual file ends at 2024; the projection year (2025) is absent from it
  and marked `-1` only in the quarterly `summary.out`, so the loader's `value < 0` guard is
  defensive.
- `wgsam_sms_baltic_2025.weights.csv` — derived from the key run's `summary.out`, quarter-1 rows:
  stock numbers `N`, mean weight `west` and biomass `BIO` at age, used to weight M2 across ages.
  Age 0 has `N = 0` in quarter 1, so a biomass-weighted mean is effectively ages 1+.

What the key run is (stock annex, `StockAnnex/Baltic/StockAnnex_ICES_EB_SMS_2022_Configuration.pdf`):
ICES Subdivisions 25–32 excluding the Gulf of Riga; **cod is the only predator**, treated since the
2019 key run as an *external* predator whose numbers and size distribution come from the ICES SS3
assessment (SMS no longer estimates cod internally); prey are central Baltic herring
(`her.27.25-2932`, SD 25–29+32 excl. Gulf of Riga) and sprat (`spr.27.22-32`). The `index.json`
block `sms_m2` carries this provenance plus the SMS→model species map (`Herring → herring`,
`Sprat → sprat`).

How it is compared (`osmose/validation/ices.py`, `scripts/validate_outputs_vs_ices.py --sms-m2`):
SMS M2 is cod-only, at age, numbers basis; OSMOSE's `mortalityRate` Predation cause lumps every
predator (both cods, percids, the background seal and cormorant). So the model quantity is
**cod-attributed** M2 on a biomass basis — annual tonnes of the prey eaten by `cod_west` + `cod_east`
(`predatorPressure`, per-step mean × steps per year) over the prey's mean biomass (young-of-year
excluded by `output.cutoff.age`) — against SMS M2 biomass-weighted across ages. The two rates differ
through age structure, domain (SD 25–32 vs the whole grid) and the prey pool each is taken on, so
they are indicative, not equivalent. The model's total predation rate per stage (a different
basis) is reported beside it as context. **Report-only; not gated.** Recent SMS values are low (eastern cod collapsed): ~0.08–0.11 yr⁻¹ for both stocks
over 2019–2024, against historical peaks of ~0.5 (herring) and ~0.85 (sprat) in the 1980s.

## How to Refresh

Snapshots freeze the 2024 ICES advice. When ICES publishes a new advice year
(typically every May/June), refresh via:

1. Start a Claude Code session in `osmose-python/` so the `ices` MCP server
   loads (CWD-sensitive — launching from the parent directory silently drops
   the server).
2. Update `scripts/_pull_ices_snapshots.py`: bump the `year` in each tuple
   of the `STOCKS` list. Cod `cod.27.22-24` is category-3 in odd years — keep
   its year at the most recent even-year advice (2022, 2024, 2026, …) until
   ICES resumes full assessment. Baltic flounder `fle.27.2223` is on the same
   biennial cycle.
3. Run the helper:

   ```bash
   .venv/bin/python scripts/_pull_ices_snapshots.py
   ```

   It hits the ICES SAG REST API directly, applies the same flattening the
   MCP server does (lowercase keys matching `get_stock_assessment` /
   `get_reference_points` output), and rewrites every `{stock}.assessment.json`
   and `{stock}.reference_points.json` plus `index.json` (`advice_year_by_stock`
   and `units_by_stock` are derived automatically — the latter from Blim
   magnitude since the ICES `StockSizeUnits` field is unreliable).
4. Update `index.json`'s top-level `advice_year` and `created` date by hand
   (the helper doesn't touch them).
5. **Update hardcoded constants in `scripts/validate_baltic_vs_ices_sag.py`:**
   - `WINDOW_YEARS` (currently `range(2018, 2023)`) — shift to the last five
     years covered by the new advice (`range(advice_year - 6, advice_year - 1)`).
   - `REPORT_MD` filename (contains `2026-04-18`) — update the date stub to
     the refresh date.
   - The `"2024 advice"` label in the validator docstring and report header
     should match the new advice year.
6. Update the `advice_year` assertion in
   `tests/test_baltic_ices_validation.py::test_manifest_exists_and_is_readable`.
7. Run `.venv/bin/python scripts/validate_baltic_vs_ices_sag.py --report` and
   review the refreshed `docs/baltic_ices_validation_<date>.md` for new drift.
8. Run `.venv/bin/python -m pytest tests/test_baltic_ices_validation.py -v`.
   If the drift fence trips for a species that now has genuine drift, decide:
   broaden the calibration target in `baltic_param-fishing.csv` /
   `biomass_targets.csv` (separate calibration-tuning plan, not this
   validation plan), or — for deliberate modeling choices — add the species
   to `F_KNOWN_EXCEPTIONS` / `B_KNOWN_EXCEPTIONS` with a pointer to the
   findings doc.
