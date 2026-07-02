---
name: Baltic grid NC orientation fix
description: baltic_grid.nc mask had 300 spurious cells (mirrored N↔S); rebuilt 2026-04-17 from the union of active movement/fishing-map cells (912→612 cells), all 26 CSVs updated in tandem
type: project
originSessionId: 986fc12f-b4f1-48df-8546-7357efd837b6
---
`data/baltic/baltic_grid.nc` had an internal inconsistency: the `mask` data array was stored south-first (row 0 = southern Baltic, matching the CSV convention used throughout OSMOSE) while the `latitude` coordinate was declared descending 66→54 (labeling row 0 as 66°N = north). Fixed 2026-04-16 by flipping the mask along latitude, keeping lat descending. Backup preserved at `data/baltic/baltic_grid.nc.bak`.

**Why:** User reported "Baltic grid is malformed while fishing distribution is right, align all layers to fishing distribution." Fishing overlay (`load_csv_overlay`) applies `np.flipud(data)` and positions cells at NC `lat[r]`, so with the descending Baltic lat it rendered geographically correctly. The NC base grid (`build_netcdf_grid_layers`) used mask as-is and placed southern-Baltic mask data at 66°N — upside-down. The fix also silently corrects a latent engine-side misalignment: `Grid.from_netcdf` reads NC mask as-is while `_load_spatial_csv` flips CSV maps, so pre-fix the engine had `ocean_mask` south-first vs movement maps north-first.

**How to apply:**
- OSMOSE CSV spatial files are stored **south-first** (row 0 = southernmost lat). The engine's `_load_spatial_csv` and UI's `load_mask`/`load_csv_overlay` all apply `np.flipud` to normalize to north-first memory.
- NC grid files (BoB, EEC, minimal) are all **ascending lat** and self-consistent (mask row r corresponds to lat[r]). Baltic is now the only descending-lat grid but is self-consistent post-fix.
- If adding a new NC grid: either match the ascending-lat convention of other grids, OR store mask matching whatever lat direction is chosen (mask[r] and lat[r] must refer to the same cell).
- The UI's `load_csv_overlay` at `ui/pages/grid_helpers.py:687` implicitly assumes NC lat is descending when combined with the CSV `flipud` — this is a latent fragility that may bite if a config mixes ascending NC lat with CSV overlays. Not worth fixing preemptively, but keep in mind if fishing/map overlays look upside down for a non-Baltic config.

## Deep audit + follow-up fix 2026-04-17

**Audit found residual coordinate convention mismatch:** `baltic_grid.nc` labeled lat at bbox edges (66.0, step=12/39=0.3077), while `baltic_ltl_biomass.nc` labeled lat at cell centers (65.85, step=12/40=0.30). Engine was unaffected (integer indexing) but UI overlaid the grid ~17 km off from the LTL biomass layer.

**Fix applied 2026-04-17:** `scripts/relabel_baltic_grid_nc.py` regenerates `baltic_grid.nc` with cell-center lat/lon matching LTL and the config `grid.upleft/lowright` bbox. Mask content unchanged (912 ocean cells, bit-exact match to prior post-flip mask). Pre-relabel snapshot at `data/baltic/baltic_grid.nc.pre-relabel.bak`. All 2411 tests pass.

**Current state:**
- lat = [65.85, 65.55, …, 54.15], step = 0.30 — matches LTL exactly
- lon = [10.2, 10.6, …, 29.8], step = 0.40 — matches LTL exactly
- UI half-step extent: 54.0–66.0 N, 10.0–30.0 E — matches config bbox exactly

**How to apply:**
- The 360 mask cells without LTL biomass are **intentional** (NC `history` attr: "Mask expanded to include Gulf of Bothnia and other cells with biomass data") — do not treat as missing data.
- Other NC grids (BoB, EEC, minimal) still use the old span/(N−1) convention. Not fixed because none have companion forcing NCs to misalign with; if one is added, apply the same relabel pattern.
- To regenerate from scratch: `.venv/bin/python scripts/relabel_baltic_grid_nc.py` (idempotent — produces the current state).

## Mask content rebuild 2026-04-17 (follow-up fix)

User reported that fishing-distribution overlay looked geographically correct but the ocean/land mask still showed land where real Baltic coastline excludes it. Investigation revealed the mask had ~300 cells that were nearly mirrored from the southern Baltic pattern (cols 0-15 at rows 0-10 showed a "Danish straits-like" footprint at lat 62-66°N, where real geography is northern Norway/Sweden mountains). All 26 movement/fishing CSVs had value=0 (ocean-with-zero) at those 300 cells — no species used them — so they were visually noise but dead weight.

**Fix:** `scripts/rebuild_baltic_mask.py` sets the mask to the union of cells where ANY of the 26 CSVs has value > 0 (612 cells). All 26 CSVs get 0→-99 at the 300 removed cells for internal consistency. Backups created with suffix `.pre-mask-rebuild.bak`. All 2411 tests pass; Baltic 1-year engine smoke run completes cleanly.

**Current state (post-rebuild):**
- NC mask: 612 cells. Geographic shape matches real Baltic: narrow Bothnian Bay at 65°N (3 cells at lon 22-24°E), widening Gulf of Bothnia, Gulf of Finland branch east at 60°N (reaching lon 29°E), wide southern Baltic at 55°N, Kiel Bay at 54°N.
- LTL footprint overlap: 551 of 554 LTL cells still covered (3 coastal cells missing — negligible).
- NC and all 26 CSVs are now self-consistent: same 612-cell ocean footprint.

**How to apply:**
- If regenerating any Baltic CSV or the mask, use `scripts/rebuild_baltic_mask.py` to keep all 27 files consistent.
- If a new fish species is added with a movement map that extends to new cells, run the rebuild script — the mask expands automatically to include those cells.
- Simulation behavior unchanged: no fish previously lived in the 300 removed cells (value=0 in every map), so removing them from ocean_mask is simulation-equivalent.

## ICES BITS validation + fix 2026-04-17

Cross-checked the rebuilt mask against the Baltic International Trawl Survey (BITS) via the ICES MCP server at `/home/razinka/ices-mcp-server/`. Pulled 1,982 haul positions (2021-2023 Q1+Q4) and ~25-27k per-species CPUE-per-length records for cod, herring, sprat, flounder.

**Results (before fix):**
- 98.5% of hauls inside the mask; 24 hauls on mask-land at 4 cells (Öresund ×3 + Gdańsk Bay)
- 15 cells where BITS had positive catches for cod AND herring AND sprat AND flounder but the model had them as absent

**Fix (`scripts/apply_ices_validation_fixes.py`, idempotent):**
- Opened 4 mask-land cells: (r,c) = (35,0), (35,1), (35,5), (37,20) → 612 → 616 ocean cells
- Extended cod adult+juvenile maps to all 15 BITS-documented cells; cod spawning to 4 Eastern-Baltic cells only (lon ≥ 18°E) to preserve the single-stock spawning-footprint assumption
- Extended herring/sprat/flounder adult+juvenile maps to the same 15 cells (their spawning maps untouched — spawning biology is species-specific)

**Post-fix state:** 0 hauls on mask-land, 0 BITS-only cells for all 4 species. 2411 tests pass, 1-year Baltic engine smoke run OK.

**Side-fix:** `/home/razinka/ices-mcp-server/ices/datras.py` had a bug — `get_hauls` called the non-existent `getHH` endpoint; patched to `getHHdata`. The ICES MCP `get_survey_hauls` tool now works.

**How to apply:**
- To re-validate against fresh BITS data: `.venv/bin/python scripts/validate_baltic_vs_ices.py`
- To re-apply fixes (idempotent): `.venv/bin/python scripts/apply_ices_validation_fixes.py`
- When adding new Baltic species, run the validation script to see if BITS coverage implies further map extension.
- Baltic stickleback/perch/pikeperch/whitefish aren't in BITS target species list; validate them against relevant freshwater/coastal survey data if needed.
