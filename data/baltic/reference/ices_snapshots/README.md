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
