# Results Workflow Enhancement — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CSV data export, ensemble CI band visualization, and run comparison to the Results page.

**Architecture:** Three independent features layered onto the existing Results page. Feature 1 (export) adds `export_dataframe()` to `OsmoseResults` and a download button. Feature 2 (ensemble) creates `osmose/ensemble.py` for replicate aggregation and an ensemble toggle in the UI. Feature 3 (comparison) extends `RunHistory` with N-way config diffs and adds a "Compare Runs" tab.

**Tech Stack:** Python 3.12, Shiny (reactive, @render.download), pandas, numpy, plotly, pytest

**Spec:** `docs/superpowers/specs/2026-03-11-results-workflow-design.md`

---

## Chunk 1: CSV Data Export

### Task 1: Add `export_dataframe()` to `OsmoseResults`

**Files:**
- Modify: `osmose/results.py:257` (before `close()` method)
- Test: `tests/test_results.py`

**Context:** `OsmoseResults` already has `_read_species_output()` for 1D outputs and `_read_2d_output()` for binned outputs, plus `diet_matrix()` and `size_spectrum()` as special cases. We need a single public method that maps an output type string to the correct reader.

- [ ] **Step 1: Write failing tests for `export_dataframe()`**

Add to `tests/test_results.py`:

```python
# ---------------------------------------------------------------------------
# Tests for export_dataframe
# ---------------------------------------------------------------------------


class TestExportDataframe:
    def test_biomass(self, output_dir):
        r = OsmoseResults(output_dir)
        df = r.export_dataframe("biomass")
        assert not df.empty
        assert "species" in df.columns
        assert "biomass" in df.columns

    def test_biomass_species_filter(self, output_dir):
        r = OsmoseResults(output_dir)
        df = r.export_dataframe("biomass", species="Anchovy")
        assert set(df["species"].unique()) == {"Anchovy"}

    def test_abundance(self, output_dir):
        r = OsmoseResults(output_dir)
        df = r.export_dataframe("abundance")
        assert not df.empty

    def test_yield(self, output_dir):
        r = OsmoseResults(output_dir)
        df = r.export_dataframe("yield")
        assert not df.empty

    def test_biomass_by_age_2d(self, output_dir_2d):
        r = OsmoseResults(output_dir_2d)
        df = r.export_dataframe("biomass_by_age")
        assert not df.empty
        assert list(df.columns) == ["time", "species", "bin", "value"]

    def test_diet(self, output_dir):
        r = OsmoseResults(output_dir)
        df = r.export_dataframe("diet")
        # diet_matrix returns empty when no files — that's fine
        assert isinstance(df, pd.DataFrame)

    def test_size_spectrum_ignores_species(self, output_dir_2d):
        r = OsmoseResults(output_dir_2d)
        df = r.export_dataframe("size_spectrum", species="Anchovy")
        assert "species" not in df.columns

    def test_trophic(self, output_dir):
        r = OsmoseResults(output_dir)
        df = r.export_dataframe("trophic")
        assert isinstance(df, pd.DataFrame)

    def test_unknown_type_returns_empty(self, output_dir):
        r = OsmoseResults(output_dir)
        df = r.export_dataframe("nonexistent_type")
        assert df.empty

    def test_yield_n(self, output_dir_2d):
        r = OsmoseResults(output_dir_2d)
        df = r.export_dataframe("yield_n")
        assert not df.empty

    def test_mortality_rate(self, output_dir_2d):
        r = OsmoseResults(output_dir_2d)
        df = r.export_dataframe("mortality_rate")
        assert not df.empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_results.py::TestExportDataframe -v`
Expected: FAIL — `AttributeError: 'OsmoseResults' object has no attribute 'export_dataframe'`

- [ ] **Step 3: Implement `export_dataframe()`**

Add to `osmose/results.py`, just before the `close()` method (around line 257):

```python
    # Type-to-method mapping for export_dataframe
    _EXPORT_MAP: dict[str, tuple[str, str]] = {
        # 1D types: (internal_output_type, method_type)
        "biomass": ("biomass", "1d"),
        "abundance": ("abundance", "1d"),
        "yield": ("yield", "1d"),
        "mortality": ("mortality", "1d"),
        "trophic": ("meanTL", "1d"),
        "yield_n": ("yieldN", "1d"),
        "mortality_rate": ("mortalityRate", "1d"),
        # 2D types
        "biomass_by_age": ("biomassByAge", "2d"),
        "biomass_by_size": ("biomassBySize", "2d"),
        "biomass_by_tl": ("biomassByTL", "2d"),
        "abundance_by_age": ("abundanceByAge", "2d"),
        "abundance_by_size": ("abundanceBySize", "2d"),
        "yield_by_age": ("yieldByAge", "2d"),
        "yield_by_size": ("yieldBySize", "2d"),
        # Special types
        "diet": ("dietMatrix", "special_diet"),
        "size_spectrum": ("sizeSpectrum", "special_spectrum"),
    }

    def export_dataframe(self, output_type: str, species: str | None = None) -> pd.DataFrame:
        """Return the DataFrame for any supported output type.

        Args:
            output_type: One of the keys from the Results page dropdown
                (e.g., 'biomass', 'biomass_by_age', 'diet', 'size_spectrum').
            species: Optional species filter. Ignored for size_spectrum.

        Returns:
            DataFrame with the requested data, or empty DataFrame if unknown type.
        """
        entry = self._EXPORT_MAP.get(output_type)
        if entry is None:
            return pd.DataFrame()

        internal_type, method_type = entry

        if method_type == "1d":
            return self._read_species_output(internal_type, species)
        elif method_type == "2d":
            return self._read_2d_output(internal_type, species)
        elif method_type == "special_diet":
            return self._read_species_output(internal_type, species)
        elif method_type == "special_spectrum":
            return self.size_spectrum()

        return pd.DataFrame()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_results.py::TestExportDataframe -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/results.py tests/test_results.py
git commit -m "feat: add export_dataframe() to OsmoseResults for unified data export"
```

---

### Task 2: Add CSV download button to Results page

**Files:**
- Modify: `ui/pages/results.py:106-179` (UI definition) and `ui/pages/results.py:187+` (server)

**Context:** The Results page has a sidebar with output controls (output dir, load button, species filter, output type selector). We add a download button below the species filter. The `@render.download` decorator writes a temp CSV file and returns its path — this matches the existing pattern in `ui/pages/advanced.py` and `ui/pages/scenarios.py`.

- [ ] **Step 1: Add download button to `results_ui()`**

In `ui/pages/results.py`, first fix the shiny import on line 9 to include `render`:

```python
from shiny import reactive, render, ui
```

Then add the download button inside the sidebar card, after the `result_type` select (after line 145):

```python
ui.hr(),
ui.download_button(
    "download_results_csv", "Download CSV", class_="btn-outline-primary w-100"
),
```

- [ ] **Step 2: Add download handler to `results_server()`**

Add to `results_server()` in `ui/pages/results.py`, after the existing render functions:

```python
    @render.download(
        filename=lambda: f"osmose_{input.result_type()}"
        + (f"_{input.result_species()}" if input.result_species() != "all" else "")
        + ".csv"
    )
    def download_results_csv():
        from osmose.results import OsmoseResults
        import tempfile

        out_dir = Path(input.output_dir())
        if not out_dir.is_dir():
            return

        res = OsmoseResults(out_dir)
        sp = input.result_species()
        species = sp if sp != "all" else None
        df = res.export_dataframe(input.result_type(), species=species)

        if df.empty:
            return

        tmp_dir = Path(tempfile.mkdtemp(prefix="osmose_export_"))
        csv_path = tmp_dir / "export.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)
```

- [ ] **Step 3: Run full test suite to verify no regressions**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: All pass (436+)

- [ ] **Step 4: Commit**

```bash
git add ui/pages/results.py
git commit -m "feat: add CSV download button to Results page"
```

---

## Chunk 2: Ensemble CI Bands

### Task 3: Create `osmose/ensemble.py` with `aggregate_replicates()`

**Files:**
- Create: `osmose/ensemble.py`
- Create: `tests/test_ensemble.py`

**Context:** `OsmoseRunner.run_ensemble()` (in `osmose/runner.py:148`) writes replicates to `rep_0/`, `rep_1/`, etc. Each subdirectory has the same CSV structure as a normal run. We aggregate across replicates to produce mean + 95% CI for 1D output types.

The 1D output types (suitable for ensemble aggregation) are: `biomass`, `abundance`, `yield`, `mortality`, `trophic`, `yield_n`, `mortality_rate`. These produce DataFrames with columns like `[time, species, <value_col>]`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_ensemble.py`:

```python
"""Tests for osmose.ensemble — replicate aggregation."""

import numpy as np
import pandas as pd
import pytest

from osmose.ensemble import aggregate_replicates, ENSEMBLE_OUTPUT_TYPES


@pytest.fixture
def rep_dirs(tmp_path):
    """Create 3 fake replicate directories with biomass CSVs."""
    for i in range(3):
        rep_dir = tmp_path / f"rep_{i}"
        rep_dir.mkdir()
        for sp in ["Anchovy", "Sardine"]:
            df = pd.DataFrame({
                "time": range(5),
                "biomass": np.random.rand(5) * 1000 + i * 100,
            })
            df.to_csv(rep_dir / f"osm_biomass_{sp}.csv", index=False)
        # Also create abundance
        df = pd.DataFrame({
            "time": range(5),
            "abundance": np.random.randint(100, 1000, 5),
        })
        df.to_csv(rep_dir / f"osm_abundance_Anchovy.csv", index=False)
    return [tmp_path / f"rep_{i}" for i in range(3)]


class TestAggregateReplicates:
    def test_basic_aggregation(self, rep_dirs):
        result = aggregate_replicates(rep_dirs, "biomass")
        assert "time" in result
        assert "mean" in result
        assert "lower" in result
        assert "upper" in result
        assert len(result["time"]) == 5
        assert len(result["mean"]) == 5

    def test_mean_between_bounds(self, rep_dirs):
        result = aggregate_replicates(rep_dirs, "biomass")
        for i in range(len(result["time"])):
            assert result["lower"][i] <= result["mean"][i] <= result["upper"][i]

    def test_species_filter(self, rep_dirs):
        result = aggregate_replicates(rep_dirs, "biomass", species="Anchovy")
        assert len(result["time"]) == 5

    def test_empty_rep_dirs(self):
        result = aggregate_replicates([], "biomass")
        assert result["time"] == []
        assert result["mean"] == []

    def test_single_replicate(self, rep_dirs):
        result = aggregate_replicates(rep_dirs[:1], "biomass")
        assert len(result["time"]) == 5
        # With 1 rep, lower == mean == upper
        for i in range(5):
            assert result["lower"][i] == result["mean"][i] == result["upper"][i]

    def test_missing_output_in_some_reps(self, rep_dirs):
        # rep_0 has abundance, but delete from rep_1 and rep_2
        import os
        for i in [1, 2]:
            for f in (rep_dirs[i]).glob("osm_abundance_*.csv"):
                os.remove(f)
        # Should still work with just rep_0
        result = aggregate_replicates(rep_dirs, "abundance")
        assert len(result["time"]) > 0

    def test_different_time_lengths_uses_inner_join(self, tmp_path):
        """Replicates with different time ranges use inner join (shortest)."""
        for i in range(2):
            rep_dir = tmp_path / f"rep_{i}"
            rep_dir.mkdir()
            n_steps = 5 if i == 0 else 3  # rep_0 has 5, rep_1 has 3
            df = pd.DataFrame({
                "time": range(n_steps),
                "biomass": [100.0] * n_steps,
            })
            df.to_csv(rep_dir / "osm_biomass_Anchovy.csv", index=False)
        dirs = [tmp_path / f"rep_{i}" for i in range(2)]
        result = aggregate_replicates(dirs, "biomass")
        # Inner join: only time steps 0,1,2 are common
        assert len(result["time"]) == 3


class TestEnsembleOutputTypes:
    def test_1d_types_listed(self):
        assert "biomass" in ENSEMBLE_OUTPUT_TYPES
        assert "abundance" in ENSEMBLE_OUTPUT_TYPES
        assert "yield" in ENSEMBLE_OUTPUT_TYPES
        assert "trophic" in ENSEMBLE_OUTPUT_TYPES

    def test_2d_types_not_listed(self):
        assert "biomass_by_age" not in ENSEMBLE_OUTPUT_TYPES
        assert "diet" not in ENSEMBLE_OUTPUT_TYPES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ensemble.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.ensemble'`

- [ ] **Step 3: Implement `osmose/ensemble.py`**

Create `osmose/ensemble.py`:

```python
"""Ensemble replicate aggregation for OSMOSE simulation outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from osmose.results import OsmoseResults

# Output types suitable for ensemble aggregation (1D time series only).
ENSEMBLE_OUTPUT_TYPES = frozenset({
    "biomass",
    "abundance",
    "yield",
    "mortality",
    "trophic",
    "yield_n",
    "mortality_rate",
})

# Map output_type to the value column name in the resulting DataFrame.
_VALUE_COL: dict[str, str] = {
    "biomass": "biomass",
    "abundance": "abundance",
    "yield": "yield",
    "mortality": "mortality",
    "trophic": "meanTL",
    "yield_n": "yieldN",
    "mortality_rate": "mortalityRate",
}


def aggregate_replicates(
    rep_dirs: list[Path],
    output_type: str,
    species: str | None = None,
) -> dict[str, list]:
    """Aggregate replicate outputs into mean + 95% CI.

    Reads each replicate's output for the given type, aligns by inner join
    on time column, and computes mean + 2.5th/97.5th percentiles.

    Args:
        rep_dirs: Paths to replicate output directories (rep_0/, rep_1/, ...).
        output_type: One of ENSEMBLE_OUTPUT_TYPES (e.g., 'biomass').
        species: Optional species filter.

    Returns:
        {"time": [...], "mean": [...], "lower": [...], "upper": [...]}
    """
    empty = {"time": [], "mean": [], "lower": [], "upper": []}
    if not rep_dirs:
        return empty

    value_col = _VALUE_COL.get(output_type)
    if value_col is None:
        return empty

    # Collect per-replicate time series
    series_list: list[pd.DataFrame] = []
    for rep_dir in rep_dirs:
        res = OsmoseResults(rep_dir)
        df = res.export_dataframe(output_type, species=species)
        if df.empty or "time" not in df.columns:
            continue
        # Sum across species at each time step to get total
        if "species" in df.columns and value_col in df.columns:
            agg = df.groupby("time")[value_col].sum().reset_index()
        elif value_col in df.columns:
            agg = df[["time", value_col]].copy()
        else:
            # Try first numeric column that isn't time
            numeric_cols = df.select_dtypes(include="number").columns
            non_time = [c for c in numeric_cols if c != "time"]
            if not non_time:
                continue
            agg = df[["time", non_time[0]]].copy()
            agg = agg.rename(columns={non_time[0]: value_col})
        series_list.append(agg)

    if not series_list:
        return empty

    # Inner join on time (truncate to common time steps)
    common_times = set(series_list[0]["time"])
    for s in series_list[1:]:
        common_times &= set(s["time"])
    if not common_times:
        return empty

    sorted_times = sorted(common_times)

    # Build matrix: rows = time steps, cols = replicates
    matrix = np.empty((len(sorted_times), len(series_list)))
    for j, s in enumerate(series_list):
        s_indexed = s.set_index("time")
        for i, t in enumerate(sorted_times):
            matrix[i, j] = s_indexed.loc[t, value_col]

    mean = np.nanmean(matrix, axis=1)
    lower = np.nanpercentile(matrix, 2.5, axis=1)
    upper = np.nanpercentile(matrix, 97.5, axis=1)

    return {
        "time": [float(t) for t in sorted_times],
        "mean": mean.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ensemble.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/ensemble.py tests/test_ensemble.py
git commit -m "feat: add ensemble replicate aggregation with mean + 95% CI"
```

---

### Task 4: Add ensemble toggle to Results page UI

**Files:**
- Modify: `ui/pages/results.py:106-179` (UI) and `ui/pages/results.py:187+` (server)

**Context:** The Results sidebar already has output controls. We add a conditional ensemble toggle that appears when the output directory contains `rep_*/` subdirectories. When ensemble mode is on and a 1D output type is selected, the chart renders CI bands instead of individual lines.

1D output types for ensemble: `biomass`, `abundance`, `yield`, `mortality`, `trophic`, `yield_n`, `mortality_rate`.

- [ ] **Step 1: Add `rep_dirs` reactive value and detection logic**

In `results_server()`, add a new reactive value after `spatial_ds` (line 190), before the `@reactive.effect` on line 192:

```python
    rep_dirs: reactive.Value[list[Path]] = reactive.Value([])
```

In the `_load_results()` effect, after loading results data and before the notification (around line 243), add:

```python
        # Detect ensemble replicate directories
        reps = sorted(out_dir.glob("rep_*"))
        rep_dirs.set([r for r in reps if r.is_dir()])
```

- [ ] **Step 2: Add conditional ensemble toggle to UI**

In `results_ui()`, add inside the sidebar card, after the species filter select and before the download button:

```python
ui.output_ui("ensemble_toggle"),
```

In `results_server()`, add:

```python
    @render.ui
    def ensemble_toggle():
        dirs = rep_dirs.get()
        if dirs:
            return ui.input_switch("ensemble_mode", f"Ensemble view ({len(dirs)} replicates)", value=True)
        return ui.div()
```

- [ ] **Step 3: Wire ensemble rendering into `results_chart()`**

In the `results_chart()` render function, add ensemble branch at the top, before the existing chart logic (after getting `rtype`, `species_filter`, `tmpl`):

```python
        # Ensemble mode: show CI bands for 1D types
        from osmose.ensemble import ENSEMBLE_OUTPUT_TYPES
        ensemble_on = False
        try:
            ensemble_on = bool(input.ensemble_mode()) and bool(rep_dirs.get())
        except Exception:
            pass

        if ensemble_on and rtype in ENSEMBLE_OUTPUT_TYPES:
            from osmose.ensemble import aggregate_replicates
            from osmose.plotting import make_ci_timeseries

            sp = species_filter if species_filter != "all" else None
            agg = aggregate_replicates(rep_dirs.get(), rtype, species=sp)
            if agg["time"]:
                title = title_map.get(rtype, rtype.title())
                fig = make_ci_timeseries(
                    agg["time"], agg["mean"], agg["lower"], agg["upper"],
                    title=f"{title} (ensemble)", y_label=col_map.get(rtype, rtype),
                )
                fig.update_layout(template=tmpl)
                return fig
```

- [ ] **Step 4: Run full test suite to verify no regressions**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add ui/pages/results.py
git commit -m "feat: add ensemble toggle with CI band rendering to Results page"
```

---

## Chunk 3: Run Comparison

### Task 5: Add `compare_runs_multi()` to `RunHistory`

**Files:**
- Modify: `osmose/history.py:59` (after existing `compare_runs`)
- Test: `tests/test_history.py`

**Context:** The existing `compare_runs(ts_a, ts_b)` only compares 2 runs. We need N-way comparison for the comparison tab. The new method loads N runs and returns diffs where any value differs.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_history.py`:

```python
def test_compare_runs_multi_two_runs(tmp_path):
    history = RunHistory(tmp_path)
    r1 = RunRecord(config_snapshot={"a": "1", "b": "2"}, duration_sec=10, output_dir="", summary={})
    r2 = RunRecord(config_snapshot={"a": "1", "b": "3", "c": "4"}, duration_sec=20, output_dir="", summary={})
    history.save(r1)
    history.save(r2)
    records = history.list_runs()
    timestamps = [r.timestamp for r in records]
    diffs = history.compare_runs_multi(timestamps)
    # 'a' is the same, 'b' and 'c' differ
    diff_keys = {d["key"] for d in diffs}
    assert "b" in diff_keys
    assert "c" in diff_keys
    assert "a" not in diff_keys


def test_compare_runs_multi_three_runs(tmp_path):
    import time as time_mod

    history = RunHistory(tmp_path)
    r1 = RunRecord(config_snapshot={"a": "1", "b": "2"}, duration_sec=10, output_dir="", summary={})
    history.save(r1)
    time_mod.sleep(0.01)
    r2 = RunRecord(config_snapshot={"a": "1", "b": "3"}, duration_sec=20, output_dir="", summary={})
    history.save(r2)
    time_mod.sleep(0.01)
    r3 = RunRecord(config_snapshot={"a": "1", "b": "2", "d": "5"}, duration_sec=30, output_dir="", summary={})
    history.save(r3)
    records = history.list_runs()
    timestamps = [r.timestamp for r in records]
    diffs = history.compare_runs_multi(timestamps)
    diff_keys = {d["key"] for d in diffs}
    assert "b" in diff_keys  # differs across runs
    assert "d" in diff_keys  # only in r3
    assert "a" not in diff_keys  # same in all


def test_compare_runs_multi_values_list(tmp_path):
    import time as time_mod

    history = RunHistory(tmp_path)
    r1 = RunRecord(config_snapshot={"x": "1"}, duration_sec=10, output_dir="", summary={})
    history.save(r1)
    time_mod.sleep(0.01)
    r2 = RunRecord(config_snapshot={"x": "2"}, duration_sec=20, output_dir="", summary={})
    history.save(r2)
    records = history.list_runs()
    timestamps = [r.timestamp for r in records]
    diffs = history.compare_runs_multi(timestamps)
    assert len(diffs) == 1
    assert diffs[0]["key"] == "x"
    assert len(diffs[0]["values"]) == 2
    # Verify values are actually different
    assert set(diffs[0]["values"]) == {"1", "2"}


def test_compare_runs_multi_empty(tmp_path):
    history = RunHistory(tmp_path)
    diffs = history.compare_runs_multi([])
    assert diffs == []


def test_compare_runs_multi_single_run(tmp_path):
    history = RunHistory(tmp_path)
    r1 = RunRecord(config_snapshot={"a": "1"}, duration_sec=10, output_dir="", summary={})
    history.save(r1)
    records = history.list_runs()
    diffs = history.compare_runs_multi([records[0].timestamp])
    assert diffs == []  # Nothing to compare
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_history.py::test_compare_runs_multi_two_runs -v`
Expected: FAIL — `AttributeError: 'RunHistory' object has no attribute 'compare_runs_multi'`

- [ ] **Step 3: Implement `compare_runs_multi()`**

Add to `osmose/history.py`, after the existing `compare_runs()` method (after line 70):

```python
    def compare_runs_multi(self, timestamps: list[str]) -> list[dict]:
        """Compare N runs by config snapshot, return parameters that differ.

        Args:
            timestamps: List of run timestamps to compare.

        Returns:
            List of {"key": str, "values": list[str | None]} for each
            parameter that differs across any of the selected runs.
        """
        if len(timestamps) < 2:
            return []

        records = [self.load_run(ts) for ts in timestamps]
        all_keys: set[str] = set()
        for r in records:
            all_keys |= set(r.config_snapshot)

        diffs = []
        for key in sorted(all_keys):
            values = [r.config_snapshot.get(key) for r in records]
            if len(set(values)) > 1:
                diffs.append({"key": key, "values": values})

        return diffs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_history.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/history.py tests/test_history.py
git commit -m "feat: add compare_runs_multi() for N-way config diff"
```

---

### Task 6: Add "Compare Runs" tab to Results page

**Files:**
- Modify: `ui/pages/results.py:154-179` (bottom layout row) and `ui/pages/results.py:187+` (server)

**Context:** The Results page has a bottom row with two cards (Diet and Spatial). We wrap them in a `navset_card_tab` and add a third tab for run comparison. The comparison tab uses `RunHistory` to list previous runs, `make_run_comparison()` for the chart, and `compare_runs_multi()` for the config diff table.

- [ ] **Step 1: Restructure bottom row into tabbed layout**

In `results_ui()`, replace the bottom `ui.layout_columns(...)` block (lines 154-178) with:

```python
        ui.navset_card_tab(
            ui.nav_panel(
                "Diet Composition",
                output_widget("diet_chart"),
            ),
            ui.nav_panel(
                "Spatial Distribution",
                ui.input_slider(
                    "spatial_time_idx",
                    "Time step",
                    min=0,
                    max=1,
                    value=0,
                    step=1,
                    animate=ui.AnimationOptions(
                        interval=1000,
                        loop=True,
                        play_button="Play",
                        pause_button="Pause",
                    ),
                ),
                output_widget("spatial_chart"),
            ),
            ui.nav_panel(
                "Compare Runs",
                ui.layout_columns(
                    ui.div(
                        ui.input_selectize(
                            "compare_runs_select",
                            "Select runs to compare",
                            choices={},
                            multiple=True,
                        ),
                        ui.input_select(
                            "compare_metric",
                            "Metric",
                            choices={
                                "biomass": "Biomass",
                                "yield": "Yield",
                                "abundance": "Abundance",
                            },
                        ),
                    ),
                    col_widths=[12],
                ),
                output_widget("comparison_chart"),
                ui.output_ui("config_diff_table"),
            ),
        ),
```

- [ ] **Step 2: Populate run history choices on load**

In `results_server()`, in the `_load_results()` effect, after the notification (around line 244), add:

```python
        # Populate run comparison choices from history
        from osmose.history import RunHistory
        history_dir = out_dir.parent / ".osmose_history"
        if history_dir.is_dir():
            history = RunHistory(history_dir)
            runs = history.list_runs()
            choices = {r.timestamp: f"{r.timestamp[:19]} ({r.duration_sec:.0f}s)" for r in runs}
            ui.update_selectize("compare_runs_select", choices=choices)
```

- [ ] **Step 3: Add comparison chart and diff table renderers**

Add to `results_server()`:

```python
    @render_plotly
    def comparison_chart():
        tmpl = _tpl(input)
        selected = input.compare_runs_select()
        if not selected or len(selected) < 1:
            return go.Figure().update_layout(
                title="Select runs to compare", template=tmpl
            )

        from osmose.history import RunHistory
        from osmose.plotting import make_run_comparison

        out_dir = Path(input.output_dir())
        history_dir = out_dir.parent / ".osmose_history"
        if not history_dir.is_dir():
            return go.Figure().update_layout(
                title="No run history found", template=tmpl
            )

        history = RunHistory(history_dir)
        records = [history.load_run(ts) for ts in selected]
        metric = input.compare_metric()
        fig = make_run_comparison(records, metrics=[metric])
        fig.update_layout(template=tmpl)
        return fig

    @render.ui
    def config_diff_table():
        selected = input.compare_runs_select()
        if not selected or len(selected) < 2:
            return ui.div("Select 2+ runs to see config differences.", style="color: #999; padding: 1rem;")

        from osmose.history import RunHistory

        out_dir = Path(input.output_dir())
        history_dir = out_dir.parent / ".osmose_history"
        if not history_dir.is_dir():
            return ui.div("No run history found.")

        history = RunHistory(history_dir)
        diffs = history.compare_runs_multi(list(selected))

        if not diffs:
            return ui.div("No config differences found.", style="color: #999; padding: 1rem;")

        # Build table header: Parameter | Run 1 | Run 2 | ...
        headers = [ui.tags.th("Parameter")]
        for i in range(len(selected)):
            headers.append(ui.tags.th(f"Run {i + 1}"))

        rows = []
        for diff in diffs:
            cells = [ui.tags.td(diff["key"], style="font-family: monospace; font-size: 12px;")]
            for val in diff["values"]:
                cells.append(ui.tags.td(str(val) if val is not None else "—"))
            rows.append(ui.tags.tr(*cells))

        return ui.tags.table(
            ui.tags.thead(ui.tags.tr(*headers)),
            ui.tags.tbody(*rows),
            class_="table table-sm table-striped",
            style="font-size: 13px;",
        )
```

- [ ] **Step 4: Run full test suite to verify no regressions**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add ui/pages/results.py
git commit -m "feat: add Compare Runs tab with grouped bar chart and config diff table"
```

---

### Task 7: Final integration verification

**Files:**
- None modified — verification only

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 2: Run linter**

Run: `.venv/bin/ruff check osmose/ensemble.py osmose/results.py osmose/history.py ui/pages/results.py tests/test_ensemble.py tests/test_results.py tests/test_history.py`
Expected: No errors

- [ ] **Step 3: Run formatter**

Run: `.venv/bin/ruff format osmose/ensemble.py osmose/results.py osmose/history.py ui/pages/results.py tests/test_ensemble.py tests/test_results.py tests/test_history.py`

- [ ] **Step 4: Commit any formatting fixes**

```bash
git add -u
git commit -m "style: format results workflow code"
```
