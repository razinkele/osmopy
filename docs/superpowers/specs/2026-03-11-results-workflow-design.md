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

This delegates to the existing `_read_species_output()` or `_read_2d_output()` based on a type map. Special cases: `diet` returns `diet_matrix()`, `size_spectrum` returns `size_spectrum()`.

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
- Large DataFrames: streaming via `yield` in the download handler (Shiny handles this natively)

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
2. Align all DataFrames by time column
3. At each time step, compute `mean`, 2.5th percentile (`lower`), 97.5th percentile (`upper`) across replicates
4. Return dict suitable for `make_ci_timeseries()`

Handles 1D output types only (biomass, abundance, yield, mortality, trophic, yieldN, mortalityRate). For 2D types (byAge, bySize), ensemble mode is not supported — the toggle is hidden.

### UI Changes

Add to the Results sidebar (below species filter, above download button):

```python
ui.input_switch("ensemble_mode", "Ensemble view", value=False)
```

Server logic:
- On `btn_load_results`: detect `rep_*/` subdirectories in output dir. If found, store `rep_dirs` list in a `reactive.Value` and auto-enable ensemble toggle. If not found, hide the toggle.
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

No new backend code. Uses existing:
- `RunHistory.list_runs() -> list[RunRecord]` — discovers saved runs
- `RunHistory.load_run(timestamp) -> RunRecord` — loads a specific run
- `RunHistory.compare_runs(records) -> dict` — returns config diffs
- `make_run_comparison(records) -> go.Figure` — grouped bar chart

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
- `config_diff_table`: call `RunHistory.compare_runs()`, render as an HTML table showing only parameters that differ between selected runs.

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
| Modify | `ui/pages/results.py` | Download button, ensemble toggle, Compare Runs tab |
| Modify | `tests/test_results.py` | Tests for `export_dataframe()` |

## Not In Scope

- New calibration objectives
- Sensitivity analysis methods
- Results page restructuring or layout changes beyond the additions described
- Batch/automated report generation from ensemble data
