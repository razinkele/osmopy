# ICES SAG Snapshots

Frozen JSON copies of ICES Stock Assessment Graphs (SAG) data used to
validate Baltic OSMOSE calibration inputs (biomass targets, fishing
mortality rates). Taken via the `ices` MCP server:

- `get_stock_assessment(stock_key, year)` → `{stock}.assessment.json`
- `get_reference_points(stock_key, year)` → `{stock}.reference_points.json`

Advice year: 2024 (most recent full Baltic advice cycle that includes
flounder; 2023 did not publish a Baltic flounder assessment). Covers
the 2018–2022 window used for calibration targets in
`biomass_targets.csv`.

See `index.json` for the model-species → stock-key mapping and the
"How to Refresh" section below for regenerating snapshots when ICES
publishes a new advice year. Refresh is MCP-driven (Tasks 1–3 of the
validation plan) — the offline validator `scripts/validate_baltic_vs_ices_sag.py`
only reads snapshots; it does not re-query ICES.

## Baltic flounder notes

ICES SAG publishes a **single** Baltic flounder stock: `fle.27.2223`
(Subdivisions 22–23, western Baltic). There is **no** `fle.27.24-32`
assessment in SAG — eastern Baltic flounder is data-limited and falls
under WKBALTIC/WKBFLAT benchmark notes, not routine advice. The plan
originally listed both; only `fle.27.2223` is pulled.

Baltic flounder is also assessed on a **biennial cycle** (even years):
2022, 2024, 2026, … Refreshing odd-year advice (2023, 2025) will not
bring a new flounder snapshot.
