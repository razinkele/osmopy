# End-to-End Results Workflow — Design Spec

## Goal

Enhance the Results page so users can export data as CSV, view ensemble simulation results with confidence intervals, and compare runs side-by-side.

## Context

The Results page (`ui/pages/results.py`) currently supports 16 output types with species filtering, diet heatmaps, and animated spatial maps. The backend (`osmose/results.py`) has 22 output readers. However, there is no way to download data, no aggregation of ensemble replicates, and no cross-run comparison. These three gaps are the focus of this spec.

## Architecture

Three independent features added to the existing Results page. No structural changes to the page layout — only additions (one button, one toggle, one new tab).

**Dependencies:** `osmose/results.py`, `osmose/history.py`, `osmose/plotting.py` (all existing). New module: `osmose/ensemble.py`.

---

## Feature 1: CSV Data Export

### What

A "Download CSV" button in the Results sidebar that exports the currently displayed data (output type + species filter) as a CSV file.

### Backend

Add a public convenience method to `OsmoseResults`:

```python
def export_dataframe(self, output_type: str, species: str | None = None) -> pd.DataFrame:
    """Return the DataFrame for any supported output type."""
```

This delegates to the existing `_read_species_output()` or `_read_2d_output()` based on a type map:

| Output type | Method | Value column |
|---|---|---|
| `biomass` | `_read_species_output("biomass")` | biomass |
| `abundance` | `_read_species_output("abundance")` | abundance |
| `yield` | `_read_species_output("yield")` | yield |
| `mortality` | `_read_species_output("mortality")` | mortality |
| `trophic` | `_read_species_output("meanTL")` | meanTL |
| `yield_n` | `_read_species_output("yieldN")` | yieldN |
| `mortality_rate` | `_read_species_output("mortalityRate")` | value |
| `biomass_by_age` | `_read_2d_output("biomassByAge")` | value |
| `biomass_by_size` | `_read_2d_output("biomassBySize")` | value |
| `biomass_by_tl` | `_read_2d_output("biomassByTL")` | value |
| `abundance_by_age` | `_read_2d_output("abundanceByAge")` | value |
| `abundance_by_size` | `_read_2d_output("abundanceBySize")` | value |
| `yield_by_age` | `_read_2d_output("yieldByAge")` | value |
| `yield_by_size` | `_read_2d_output("yieldBySize")` | value |
| `diet` | `diet_matrix()` | wide format (prey columns) |
| `size_spectrum` | `size_spectrum()` | abundance (species filter ignored) |

### UI

Add to the Results sidebar (below the species filter):

```python
ui.download_button("download_results_csv", "Download CSV", class_="btn-outline-primary w-100")
```

Server handler uses `@render.download`:

```python
@render.download(filename=lambda: f"osmose_{input.result_type()}_{input.result_species()}.csv")
def download_results_csv():
    # ... get current DataFrame, yield as CSV bytes
```

### Edge cases

- Empty results (no data loaded): button disabled via `ui.update_action_button` after load
- Large DataFrames: write to temp file, return path (matches existing pattern in `advanced.py` and `scenarios.py`)
- Species filter on `size_spectrum`: ignored (size_spectrum() has no species parameter)
- Species="all": filename uses `osmose_biomass.csv` (omit "all" suffix)

---

## Feature 2: Ensemble Results with CI Bands

### What

When simulation output contains replicate subdirectories (`rep_0/`, `rep_1/`, ...), the Results page can toggle into "Ensemble view" showing mean + 95% confidence interval bands instead of individual time series.

### Backend — `osmose/ensemble.py`

New module with one primary function:

```python
def aggregate_replicates(
    rep_dirs: list[Path],
    output_type: str,
    species: str | None = None,
) -> dict[str, list]:
    """Aggregate replicate outputs into mean + CI.

    Returns:
        {"time": [...], "mean": [...], "lower": [...], "upper": [...]}
    """
```

Implementation:
1. For each `rep_dir`, create `OsmoseResults(rep_dir)` and call `export_dataframe(output_type, species)`
2. Align all DataFrames by inner join on time column (truncate to the shortest replicate's time range)
3. At each time step, compute `mean`, 2.5th percentile (`lower`), 97.5th percentile (`upper`) across replicates using `np.nanpercentile`
4. Return dict suitable for `make_ci_timeseries()`

Replicate naming convention: `rep_0/`, `rep_1/`, ... (standard OSMOSE `run_ensemble()` output from `osmose/runner.py`).

Handles 1D output types only (biomass, abundance, yield, mortality, trophic, yieldN, mortalityRate). For 2D types (byAge, bySize), ensemble mode is not supported — the toggle is hidden.

### UI Changes

Add to the Results sidebar (below species filter, above download button):

```python
ui.input_switch("ensemble_mode", "Ensemble view", value=False)
```

The ensemble toggle is rendered conditionally using `ui.output_ui()`:

```python
ui.output_ui("ensemble_toggle")  # in sidebar

@render.ui
def ensemble_toggle():
    if rep_dirs.get():  # non-empty list of rep dirs
        return ui.input_switch("ensemble_mode", "Ensemble view", value=True)
    return ui.div()  # hidden when no replicates
```

Server logic:
- On `btn_load_results`: detect `rep_*/` subdirectories in output dir via `sorted(out_dir.glob("rep_*"))`. If found, store in `rep_dirs: reactive.Value[list[Path]]`.
- When ensemble mode is on and a 1D output type is selected: call `aggregate_replicates()`, render with `make_ci_timeseries()` from `osmose/plotting.py`.
- When ensemble mode is off: render as today (single time series).

### Edge cases

- Single replicate: CI bands collapse to a single line (valid, just no spread)
- Missing output in some replicates: skip replicates that don't have the requested output type, show warning badge with count
- 2D output types: ensemble toggle hidden, chart renders as normal stacked area

---

## Feature 3: Run Comparison Tab

### What

A new tab on the Results page ("Compare Runs") that lets users select 2+ previous runs from history and compare their summary metrics side-by-side, with a config diff table.

### Backend

Modify `osmose/history.py`:
- Add `compare_runs_multi(timestamps: list[str]) -> list[dict]` — N-way config diff. Returns list of `{"key": str, "values": list[str | None]}` for each parameter that differs across any of the selected runs. This replaces the 2-run-only `compare_runs()` for the comparison UI.

Uses existing:
- `RunHistory.list_runs() -> list[RunRecord]` — discovers saved runs
- `RunHistory.load_run(timestamp) -> RunRecord` — loads a specific run
- `make_run_comparison(records) -> go.Figure` — grouped bar chart

**RunRecord.summary population:** The Run page already does a best-effort history save after successful runs. The summary dict is populated from `OsmoseResults` at save time with keys: `biomass` (total final biomass), `yield` (total final yield), `abundance` (total final abundance). If summary is empty (old runs), the comparison chart shows "No summary data" placeholder.

### UI Changes

Add a new tab to the bottom layout row of `results_ui()`:

```python
ui.navset_card_tab(
    ui.nav_panel("Diet Composition Matrix", ...),  # existing
    ui.nav_panel("Spatial Distribution", ...),      # existing
    ui.nav_panel(
        "Compare Runs",
        ui.input_selectize(
            "compare_runs_select",
            "Select runs to compare",
            choices={},
            multiple=True,
        ),
        ui.input_select(
            "compare_metric",
            "Metric",
            choices={"biomass": "Biomass", "yield": "Yield", "abundance": "Abundance"},
        ),
        output_widget("comparison_chart"),
        ui.output_ui("config_diff_table"),
    ),
)
```

Server logic:
- On `btn_load_results`: populate `compare_runs_select` choices from `RunHistory(state.output_dir).list_runs()`, using timestamps as keys and formatted labels as values.
- `comparison_chart`: when 2+ runs selected, load their `RunRecord`s, call `make_run_comparison()`.
- `config_diff_table`: call `RunHistory.compare_runs_multi()`, render as an HTML table with columns: Parameter, Run 1, Run 2, ..., showing only parameters that differ.

### Edge cases

- No run history: tab shows "No previous runs found. Run a simulation first."
- Single run selected: chart shows single-run bars (valid but not very useful — show hint to select more)
- Runs with different species counts: comparison table handles mismatched keys gracefully

---

## Files

| Action | File | Purpose |
|--------|------|---------|
| Create | `osmose/ensemble.py` | Replicate aggregation (mean + CI) |
| Create | `tests/test_ensemble.py` | Tests for aggregation logic |
| Modify | `osmose/results.py` | Add `export_dataframe()` method |
| Modify | `osmose/history.py` | Add `compare_runs_multi()` for N-way diffs |
| Modify | `ui/pages/results.py` | Download button, ensemble toggle, Compare Runs tab |
| Modify | `tests/test_results.py` | Tests for `export_dataframe()` |
| Modify | `tests/test_history.py` | Tests for `compare_runs_multi()` |

## Not In Scope

- New calibration objectives
- Sensitivity analysis methods
- Results page restructuring or layout changes beyond the additions described
- Batch/automated report generation from ensemble data
