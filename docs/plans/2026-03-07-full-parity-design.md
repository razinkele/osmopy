# Full Parity Design: osmose-python vs OSMOSE R Package

**Date:** 2026-03-07
**Scope:** Bridge all gaps between osmose-python and the OSMOSE R package (273 functions, 48 source files)
**Current state:** 181 parameters, 7 output types, 3 chart types, single-phase calibration, 239 tests

---

## Gap Inventory

### What Exists (Strengths)

| Area | Coverage |
|------|----------|
| Schema/parameters (181 across 15 categories) | Good foundation |
| Config I/O (roundtrip read/write) | Complete |
| Java runner (async with progress) | Complete |
| Basic results (7 output types) | Partial |
| Calibration (NSGA-II + GP surrogate + Sobol) | Partial |
| Scenarios (save/load/fork/compare/bulk) | Complete |
| UI (10 pages, Nautical Observatory theme) | Good |
| Tests (239, 100% coverage) | Excellent |

### What's Missing

- 15+ output type parsers (all ByAge/BySize/ByTL structured outputs)
- 11+ chart types (stacked area, histograms, log-log spectra, config plots)
- Multi-replicate ensemble analysis (CI bands, summary stats)
- Multi-phase sequential calibration, more objective functions, more algorithms
- Config visualization (growth curves, predation ranges, food web)
- Config validation (completeness, file references, consistency)
- Grid creation and spatial utilities
- Automated reporting (HTML/PDF)
- Demo generation and version migration

---

## Phase 5: Output Completeness

### New output parsers for `osmose/results.py`

All ByAge/BySize/ByTL outputs share a uniform 2D CSV format: rows = time steps, columns = bins (age classes or size bins). A single `_read_2d_output(pattern, index_col)` helper covers all variants.

| Method | Description | Format |
|--------|-------------|--------|
| `biomass_by_age(species)` | Biomass structured by age class | 2D CSV (time x age) |
| `biomass_by_size(species)` | Biomass by size bin | 2D CSV (time x size) |
| `biomass_by_tl(species)` | Biomass by trophic level bin | 2D CSV (time x TL) |
| `abundance_by_age(species)` | Abundance by age class | 2D CSV |
| `abundance_by_size(species)` | Abundance by size bin | 2D CSV |
| `abundance_by_tl(species)` | Abundance by TL bin | 2D CSV |
| `yield_by_age(species)` | Catch biomass by age | 2D CSV |
| `yield_by_size(species)` | Catch biomass by size | 2D CSV |
| `yield_abundance(species)` | Catch in numbers (1D time series) | CSV |
| `yield_n_by_age(species)` | Catch numbers by age | 2D CSV |
| `yield_n_by_size(species)` | Catch numbers by size | 2D CSV |
| `diet_by_age(species)` | Diet composition by predator age | 3D CSV |
| `diet_by_size(species)` | Diet composition by predator size | 3D CSV |
| `mortality_rate(species)` | Mortality by source (predation/starvation/fishing/natural) | CSV |
| `size_spectrum()` | Community size spectrum | CSV (log size x log abundance) |
| `mean_size_by_age(species)` | Mean body size per age class | 2D CSV |
| `mean_tl_by_size(species)` | Mean TL per size bin | 2D CSV |
| `mean_tl_by_age(species)` | Mean TL per age class | 2D CSV |
| `spatial_abundance(filename)` | Spatial abundance grid | NetCDF |
| `spatial_size(filename)` | Spatial mean size grid | NetCDF |
| `spatial_yield(filename)` | Spatial catch grid | NetCDF |
| `spatial_ltl(filename)` | Spatial LTL distribution | NetCDF |

### Internal refactor

Extract `_read_2d_output(type_name, species)` as a generic helper:
1. Glob for `{prefix}_{type_name}_{species}.csv` (or all species if None)
2. Parse 2D CSV: first column = time, remaining columns = bin labels
3. Return DataFrame with multi-index (time, species) and bin columns
4. Handle missing files gracefully (empty DataFrame)

### Tests
~30 new tests covering all parsers, species filtering, empty files, malformed data.

---

## Phase 6: Visualization Parity

### New chart functions

R has 14 plot methods with 4 display modes. New Plotly chart functions:

| Chart Function | Description | Plotly Type |
|----------------|-------------|-------------|
| `make_stacked_area(df, title)` | Age/size/TL contribution over time | `area` with stackgroup |
| `make_size_histogram(df, time_idx)` | Size distribution at a time snapshot | `bar` grouped by species |
| `make_tl_breakdown(df)` | Trophic level contribution bar | `bar` stacked |
| `make_mortality_stacked(df)` | Mortality by source over time | `area` stacked |
| `make_size_spectrum(df)` | Log-log size spectrum + regression | `scatter` log axes |
| `make_yield_composition(df, by)` | Catch composition by age or size | `bar` stacked |
| `make_ci_timeseries(mean, lower, upper)` | Time series with CI bands | `scatter` + fill |
| `make_growth_curves(config, registry)` | Von Bertalanffy L(t) per species | `line` multi-trace |
| `make_predation_ranges(config)` | Predator-prey size ratio ranges | `bar` horizontal |
| `make_reproduction_season(config)` | Monthly spawning proportion | `bar` grouped |
| `make_food_web(config)` | Network graph from accessibility matrix | `scatter` + annotations |

### Display modes

Add a "Display mode" selector to Results UI:
- **Time series** (default): line chart over time
- **Summary**: bar chart with mean +/- CI across replicates
- **Stacked**: proportional contribution (for ByAge/BySize outputs)

### Results UI updates
- Expand output type dropdown with all Phase 5 types
- Add display mode selector
- Add download chart button (PNG/SVG via plotly `write_image`)
- Config visualization tab (growth curves, predation, reproduction) on Setup page

### Tests
~20 new tests for chart generation functions.

---

## Phase 7: Multi-Replicate Analysis

### New module: `osmose/analysis.py`

OSMOSE is stochastic; real workflows run 5-50 replicates and average.

| Function | Purpose |
|----------|---------|
| `ensemble_stats(results_list, output_type)` | Mean, std, 95% CI across replicates |
| `summary(results)` | Statistical summary table (species x metric) |
| `shannon_diversity(biomass_df)` | Shannon-Wiener index per timestep |
| `mean_tl_catch(yield_df, tl_df)` | Mean trophic level of catch |
| `size_spectrum_slope(spectrum_df)` | Log-log regression slope + intercept |
| `compare_runs(results_a, results_b)` | Relative difference per output |

### Runner changes

New method on `OsmoseRunner`:
```
async def run_ensemble(config_path, output_dir, n_replicates, java_opts, on_progress)
```
- Runs n simulations with different random seeds (`-Psimulation.random.seed=i`)
- Creates `output/rep_0/`, `output/rep_1/`, etc.
- Parallel replicates via asyncio gather or ThreadPoolExecutor
- Returns list of RunResult

### Results UI changes
- "Load Ensemble" mode: read all `rep_*/` subdirectories
- Show mean +/- CI bands on time series charts
- Summary statistics panel (table with mean/std/min/max per species)

### Tests
~15 new tests.

---

## Phase 8: Calibration Depth

### Multi-phase sequential calibration

R's calibrar calibrates parameter subsets in 3-4 phases. New class:

```python
@dataclass
class CalibrationPhase:
    name: str
    free_params: list[FreeParameter]
    algorithm: str  # "NSGA-II", "L-BFGS-B", "Nelder-Mead", "DE"
    n_generations: int
    n_replicates: int  # average objective over this many runs
```

New class `MultiPhaseCalibrator`:
- Takes list of CalibrationPhase objects
- Runs phases sequentially; output of phase N becomes fixed params for phase N+1
- Supports warm restart from checkpoint JSON

### New objective functions in `objectives.py`

| Function | Purpose |
|----------|---------|
| `yield_rmse(sim, obs, species)` | RMSE on catch biomass |
| `catch_at_size_distance(sim, obs)` | Distance on size composition |
| `catch_at_age_distance(sim, obs)` | Distance on age composition |
| `size_at_age_rmse(sim, obs)` | RMSE on mean size per age |
| `weighted_multi_objective(objectives, weights)` | Weighted sum combiner |

### Additional optimization algorithms

Add scipy.optimize wrappers:
- `L-BFGS-B` (gradient-based, good for smooth objectives)
- `Nelder-Mead` (simplex, no gradients needed)
- `differential_evolution` (global, population-based)

### Stochastic objectives

Average objective values across n_replicates per candidate evaluation. This requires Phase 7's `run_ensemble`.

### `configureCalibration(config)` function

Extract calibration parameters from config (R equivalent):
- Returns dict with `guess`, `min`, `max`, `phase` per free parameter
- Auto-identifies natural mortality, accessibility coefficients as calibration targets

### Calibration UI updates
- Phase editor (add/remove phases, assign params + algorithm per phase)
- Number of replicates per evaluation
- More reference data uploads (yield CSV, catch-at-size CSV, size-at-age CSV)
- Export/import calibration results (JSON checkpoint)

### Tests
~20 new tests.

---

## Phase 9: Config Visualization & Validation

### New module: `osmose/config/validator.py`

| Function | Purpose |
|----------|---------|
| `validate_config(config, registry)` | Check all required params present, types valid, bounds respected |
| `check_file_references(config, base_dir)` | Verify all FILE_PATH params point to existing files |
| `check_species_consistency(config)` | Verify nspecies matches indexed param count |
| `list_missing_params(config, registry)` | Show which required params are absent |
| `list_unknown_params(config, registry)` | Show params not in registry (typos, deprecated) |

### Config visualization (Setup page additions)

- Growth curves tab: Von Bertalanffy L(t) for all species overlaid
- Predation tab: size ratio range bars per species
- Reproduction tab: monthly spawning proportion per species
- Food web tab: network diagram from accessibility matrix

### Validation UI

- "Validate Config" button on Advanced page
- Shows validation report: errors (red), warnings (amber), info (blue)
- Blocks Run if critical errors found (optional)

### Tests
~10 new tests.

---

## Phase 10: Grid & Spatial Utilities

### New module: `osmose/grid.py`

| Function | Purpose |
|----------|---------|
| `create_grid_netcdf(lat_bounds, lon_bounds, resolution, mask, output)` | Generate OSMOSE NetCDF grid |
| `create_grid_csv(nrows, ncols, mask_values)` | Generate CSV grid mask |
| `csv_maps_to_netcdf(csv_dir, output_nc)` | Migrate CSV maps to NetCDF (R's `update_maps()`) |
| `visualize_grid(grid_file)` | Show grid with land/sea mask |
| `overlay_distribution(grid, map_file, species)` | Species distribution on grid |

### Grid UI enhancements

- Interactive grid preview: click cells to toggle land/sea (if CSV mask)
- Upload coastline data to auto-generate mask
- Distribution map preview per species/season
- Grid resolution calculator (auto-suggest nrows/ncols from lat/lon bounds)

### Tests
~8 new tests.

---

## Phase 11: Reporting & Export

### New module: `osmose/reporting.py`

| Function | Purpose |
|----------|---------|
| `generate_report(results, config, output_path, format)` | Full HTML or PDF report |
| `summary_table(results)` | Species x metric DataFrame |
| `export_charts(results, output_dir, format)` | Batch export all charts |

### Report contents (matching R's `report.osmose()`)

1. Configuration summary (species table, grid info, simulation settings)
2. Biomass time series per species (with CI if replicates)
3. Abundance and yield time series
4. Mortality breakdown (stacked by source)
5. Diet composition heatmaps
6. Spatial distribution maps (if available)
7. Size spectra analysis
8. Ecological indicators (Shannon diversity, mean TL)
9. Calibration results (if available)

### Implementation

- Jinja2 HTML template with embedded Plotly charts (interactive)
- Optional PDF via weasyprint
- Standalone HTML file (no server needed, shareable)

### UI

- "Generate Report" button on Results page
- Format selector (HTML / PDF)
- Download link after generation

### Tests
~5 new tests.

---

## Phase 12: Demo Generation & Version Migration

### New functions

| Function | R Equivalent | Purpose |
|----------|-------------|---------|
| `osmose_demo(scenario, output_dir)` | `osmose_demo()` | Generate demo configs |
| `initialize_population(config, method)` | `initialize_osmose()` | Generate initial population files |
| `migrate_config(config, target_version)` | `update_osmose()` | Upgrade config between OSMOSE versions |
| `migrate_ltl(config)` | `update_ltl()` | Update LTL NetCDF format |

### Demo scenarios to bundle

- **Bay of Biscay** (existing 3-species, expand with movement maps and forcing)
- **EEC 4.3.0** (English Channel, R package default demo)

### Version migration

Map parameter name changes between OSMOSE versions (3.x → 4.x):
- Key renames (e.g., `simulation.nplankton` → `simulation.nresource`)
- Format changes (CSV maps → NetCDF)
- New required parameters with defaults

### UI

- "Load Demo" dropdown on Setup page
- "Migrate Config" button on Advanced page with version selector

### Tests
~8 new tests.

---

## Phase Dependency Graph

```
Phase 5 (Outputs) ──→ Phase 6 (Charts) ──→ Phase 7 (Replicates)
                                                    │
                                                    ↓
                                              Phase 8 (Calibration)

Phase 9 (Validation)     ── independent
Phase 10 (Grid)          ── independent
Phase 11 (Reporting)     ── depends on Phase 5, 6, 7
Phase 12 (Demo/Migration) ── independent
```

## Estimated Scope

| Phase | Focus | Est. LOC | New Tests | Priority |
|-------|-------|----------|-----------|----------|
| 5 | Output completeness | ~400 | ~30 | Critical |
| 6 | Visualization parity | ~600 | ~20 | High |
| 7 | Multi-replicate analysis | ~300 | ~15 | High |
| 8 | Calibration depth | ~500 | ~20 | High |
| 9 | Config viz & validation | ~250 | ~10 | Medium |
| 10 | Grid & spatial utilities | ~200 | ~8 | Medium |
| 11 | Reporting & export | ~350 | ~5 | Medium |
| 12 | Demo & migration | ~200 | ~8 | Low |
| **Total** | | **~2,800** | **~116** | |
