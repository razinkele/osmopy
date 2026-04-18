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
