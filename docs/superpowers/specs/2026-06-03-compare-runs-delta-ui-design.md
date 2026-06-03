# Compare Runs tab — output-delta section — Design

**Date:** 2026-06-03
**Status:** Approved direction (brainstormed; reconnaissance-grounded).
**Builds on:** result delta-tracking (shipped `6d8b2f5..dfcb422`): `osmose.analysis.run_delta`,
`format_delta_report`, `osmose.plotting.make_run_delta_chart`.

## Motivation

The Compare Runs tab (`ui/pages/results.py`) already shows a **config** diff + a summary bar
chart for selected runs, but nothing shows what the runs' **outputs** did differently. The
just-shipped `run_delta` answers exactly that from the CLI; this surfaces it interactively —
pick two runs, see the per-species ranked output delta inline.

## Verified context (reconnaissance)

- Compare Runs tab lives in `results_ui()` / `results_server()` (`ui/pages/results.py`), inside a
  `ui.nav_panel("Compare Runs", ...)`. No `app.py` change needed to extend it.
- Existing controls: `ui.input_selectize("compare_runs_select", multiple=True)` (populated from
  `RunHistory(history_dir).list_runs()`), `ui.input_select("compare_metric", ...)` with choices
  `{biomass, yield, abundance}` — **exactly `run_delta`'s metrics**.
- Existing outputs: `output_widget("comparison_chart")` (`@render_plotly`) + `ui.output_ui(
  "config_diff_table")` (`@render.ui` HTML). The plotly pattern is
  `from shinywidgets import output_widget, render_plotly`; the table pattern is `@render.ui`
  returning `ui.tags.*` / `ui.markdown(...)`.
- `RunRecord` (`osmose/history.py`) stores `timestamp`, `config_snapshot`, `output_dir` (str),
  `summary` — **no `prefix`**. Reconstruct a run as `OsmoseResults(Path(rec.output_dir),
  strict=False)` (default prefix `"osm"`), matching how the page already builds results.
- Render fns read inputs directly (no `@reactive.calc`); guard with `len(selected)` checks.
- Testing surface: `tests/test_ui_results.py` unit-tests only the page's PURE helpers
  (`make_timeseries_chart`, `make_diet_heatmap`); render decorators are not unit-tested (project
  convention). `run_delta`/`format_delta_report`/`make_run_delta_chart` are already tested in
  `tests/test_analysis_delta.py`.
- **Pre-existing quirk (inherited, NOT fixed here):** `run.py` saves history to `data/history`
  while the Results page reads `out_dir.parent/.osmose_history`; the run selector may not
  populate in some setups. The delta section works wherever the selector populates (same as the
  existing config-diff). Reconciling the path is a separate follow-on.

## Architecture (inside `ui/pages/results.py` only)

### Pure helper (testable)

```python
def _delta_for_selected(records, metric: str, window_years: int):
    """records: list of exactly 2 RunRecord (baseline, variant). Reconstruct each run and
    return run_delta's list[SpeciesDelta]. Raises ValueError if not exactly 2 records."""
    if len(records) != 2:
        raise ValueError("need exactly 2 runs")
    from osmose.results import OsmoseResults
    from osmose.analysis import run_delta
    base = OsmoseResults(Path(records[0].output_dir), strict=False)
    var = OsmoseResults(Path(records[1].output_dir), strict=False)
    return run_delta(base, var, metric=metric, window_years=window_years)
```
First-selected = baseline, second = variant (documented in the prompt text).

### UI additions (`results_ui()`, Compare Runs panel)

- Add `ui.input_slider("compare_window_years", "Window (years)", min=1, max=30, value=10)`
  beside `compare_metric`. **Reuse `compare_metric`** (no new metric control).
- After `ui.output_ui("config_diff_table")`: add `output_widget("run_delta_chart")` +
  `ui.output_ui("run_delta_table")`.

### Server additions (`results_server()`)

- `@render_plotly def run_delta_chart()`: read `compare_runs_select`, `compare_metric`,
  `compare_window_years`. If `len(selected) != 2` → return an empty `go.Figure` titled "Select
  exactly 2 runs". Else load the 2 `RunRecord`s (via the same `RunHistory` the page already uses),
  call `_delta_for_selected`, return `make_run_delta_chart(deltas, metric=metric)` with the page's
  template helper applied (as the existing `comparison_chart` does). On a missing/unreadable
  output dir, return a `go.Figure` with an error title (mirror the existing guard style).
- `@render.ui def run_delta_table()`: same selection/guard; <2 or >2 → `ui.div("Select exactly 2
  runs to see the output delta (1st = baseline, 2nd = variant)", style=<empty-style>)`. Else
  `ui.markdown(format_delta_report(deltas, metric=metric, window_years=window_years))`. Wrap the
  load in try/except → on error, `ui.div("Could not load run outputs: <msg>")` (no silent failure).

### Tests (`tests/test_ui_results.py`)

- `_delta_for_selected` with two tiny synthetic run-output dirs (write a wide biomass CSV per dir
  via `OsmoseResults(...).biomass().to_csv` shape, or a minimal `_wide`-style CSV + a fake
  RunRecord with `output_dir`): assert it returns ranked `SpeciesDelta`s and that swapping the two
  records flips the sign of the deltas (baseline/variant order matters).
- `_delta_for_selected` with != 2 records raises `ValueError`.
- (Render decorators not unit-tested, per the page's convention; a manual UI run-through covers
  the wiring — see Delivery.)

## Scope / YAGNI

- **Pairwise only** (exactly 2 runs); config-diff above stays N-way.
- **No `top_n` control** — show all species (≤15 for Baltic/EEC).
- **Reuse `compare_metric`**; only `compare_window_years` is new.
- **No `app.py` change**; no history-dir fix (pre-existing follow-on).
- **No new plotting/analysis code** — reuse the shipped `run_delta`/`make_run_delta_chart`/
  `format_delta_report` unchanged.

## Honest limitations

- Inherits the run-selector population quirk (history-dir mismatch) above.
- UI render functions are verified by a manual run-through (browser), not unit tests — matching
  the page's existing convention; the testable logic is the extracted `_delta_for_selected` helper.

## Delivery

Single PR: the slider + 2 outputs + 2 render fns + the `_delta_for_selected` helper + its unit
tests + a manual UI run-through (launch the app, select 2 runs, confirm the delta chart+table
render). No engine changes, no calibration runs.
