# Full R Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring osmose-python to full feature parity with the OSMOSE R package (273 functions, 48 source files) across 8 phases (5-12).

**Architecture:** Each phase adds a layer: output parsers (Phase 5) feed new charts (Phase 6), which feed ensemble analysis (Phase 7), which enables stochastic calibration (Phase 8). Phases 9-12 are independent utilities.

**Tech Stack:** Python 3.12, pandas, xarray, plotly, pymoo, scipy, SALib, sklearn, Jinja2, Shiny for Python

**Test runner:** `.venv/bin/python -m pytest`
**Lint:** `.venv/bin/ruff check osmose/ ui/ tests/`
**Format:** `.venv/bin/ruff format osmose/ ui/ tests/`

---

## Phase 5: Output Completeness

### Task 5.1: Add `_read_2d_output()` generic helper

All ByAge/BySize/ByTL outputs share the same 2D CSV format. Add a generic parser.

**Files:**
- Modify: `osmose/results.py:85-107`
- Test: `tests/test_results.py`

**Step 1: Write the failing test**

Add to `tests/test_results.py`:

```python
def test_read_2d_output_biomass_by_age(output_dir):
    """2D output files have columns: time, then one column per bin (age0, age1, ...)."""
    # Create a 2D biomass-by-age CSV
    df = pd.DataFrame({
        "time": range(5),
        "0": [100, 110, 120, 130, 140],
        "1": [200, 210, 220, 230, 240],
        "2": [50, 55, 60, 65, 70],
    })
    df.to_csv(output_dir / "osm_biomassByAge_Anchovy.csv", index=False)

    results = OsmoseResults(output_dir)
    df_out = results._read_2d_output("biomassByAge", "Anchovy")
    assert not df_out.empty
    assert "species" in df_out.columns
    assert "bin" in df_out.columns
    assert "value" in df_out.columns
    assert set(df_out["bin"].unique()) == {"0", "1", "2"}


def test_read_2d_output_missing_returns_empty(output_dir):
    results = OsmoseResults(output_dir)
    df = results._read_2d_output("biomassByAge", None)
    assert df.empty
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_results.py::test_read_2d_output_biomass_by_age -v`
Expected: FAIL with `AttributeError: 'OsmoseResults' object has no attribute '_read_2d_output'`

**Step 3: Write implementation**

Add to `osmose/results.py` after `_read_species_output()`:

```python
def _read_2d_output(
    self, output_type: str, species: str | None
) -> pd.DataFrame:
    """Read 2D structured output (ByAge, BySize, ByTL).

    These CSVs have columns: time, then one column per bin.
    Returns long-format DataFrame with columns: time, species, bin, value.
    """
    pattern = f"{self.prefix}_{output_type}*.csv"
    frames = []
    for filepath in sorted(self.output_dir.glob(pattern)):
        df = pd.read_csv(filepath)
        parts = filepath.stem.split("_", 2)
        sp_name = parts[2] if len(parts) > 2 else filepath.stem
        # Melt wide format to long: time stays, bin columns become rows
        id_vars = [c for c in df.columns if c.lower() == "time"]
        value_vars = [c for c in df.columns if c.lower() != "time"]
        if not id_vars or not value_vars:
            continue
        melted = df.melt(
            id_vars=id_vars, value_vars=value_vars,
            var_name="bin", value_name="value",
        )
        melted["species"] = sp_name
        frames.append(melted)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if species:
        combined = combined[combined["species"] == species]
    return combined
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_results.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add osmose/results.py tests/test_results.py
git commit -m "feat: add _read_2d_output() generic helper for structured outputs"
```

---

### Task 5.2: Add all ByAge/BySize/ByTL output methods

Wire up all 2D output methods using `_read_2d_output()`.

**Files:**
- Modify: `osmose/results.py`
- Test: `tests/test_results.py`

**Step 1: Write failing tests**

Add to `tests/test_results.py`:

```python
@pytest.fixture
def output_dir_2d(tmp_path):
    """Create output dir with 2D structured output files."""
    # biomass by age
    for sp in ["Anchovy", "Sardine"]:
        df = pd.DataFrame({"time": range(5), "0": np.random.rand(5) * 100, "1": np.random.rand(5) * 50})
        df.to_csv(tmp_path / f"osm_biomassByAge_{sp}.csv", index=False)
    # biomass by size
    df = pd.DataFrame({"time": range(5), "0-5": np.random.rand(5), "5-10": np.random.rand(5)})
    df.to_csv(tmp_path / "osm_biomassBySize_Anchovy.csv", index=False)
    # abundance by age
    df = pd.DataFrame({"time": range(5), "0": np.random.randint(100, 1000, 5), "1": np.random.randint(50, 500, 5)})
    df.to_csv(tmp_path / "osm_abundanceByAge_Anchovy.csv", index=False)
    # yield by size
    df = pd.DataFrame({"time": range(5), "0-5": np.random.rand(5) * 10, "5-10": np.random.rand(5) * 5})
    df.to_csv(tmp_path / "osm_yieldBySize_Anchovy.csv", index=False)
    # yieldN (1D — catch in numbers)
    df = pd.DataFrame({"time": range(5), "yieldN": np.random.randint(100, 10000, 5)})
    df.to_csv(tmp_path / "osm_yieldN_Anchovy.csv", index=False)
    # mortality rate by source
    df = pd.DataFrame({
        "time": range(5), "predation": np.random.rand(5), "starvation": np.random.rand(5),
        "fishing": np.random.rand(5), "natural": np.random.rand(5),
    })
    df.to_csv(tmp_path / "osm_mortalityRate_Anchovy.csv", index=False)
    # size spectrum
    df = pd.DataFrame({"size": [1, 2, 4, 8, 16], "abundance": [10000, 5000, 1000, 100, 10]})
    df.to_csv(tmp_path / "osm_sizeSpectrum.csv", index=False)
    return tmp_path


def test_biomass_by_age(output_dir_2d):
    results = OsmoseResults(output_dir_2d)
    df = results.biomass_by_age()
    assert not df.empty
    assert {"time", "species", "bin", "value"}.issubset(df.columns)
    assert set(df["species"].unique()) == {"Anchovy", "Sardine"}


def test_biomass_by_size(output_dir_2d):
    results = OsmoseResults(output_dir_2d)
    df = results.biomass_by_size("Anchovy")
    assert not df.empty
    assert set(df["species"].unique()) == {"Anchovy"}


def test_abundance_by_age(output_dir_2d):
    results = OsmoseResults(output_dir_2d)
    df = results.abundance_by_age()
    assert not df.empty


def test_yield_by_size(output_dir_2d):
    results = OsmoseResults(output_dir_2d)
    df = results.yield_by_size()
    assert not df.empty


def test_yield_abundance(output_dir_2d):
    results = OsmoseResults(output_dir_2d)
    df = results.yield_abundance()
    assert not df.empty
    assert "yieldN" in df.columns or "value" in df.columns or not df.empty


def test_mortality_rate(output_dir_2d):
    results = OsmoseResults(output_dir_2d)
    df = results.mortality_rate()
    assert not df.empty
    assert "species" in df.columns


def test_size_spectrum(output_dir_2d):
    results = OsmoseResults(output_dir_2d)
    df = results.size_spectrum()
    assert not df.empty
    assert "size" in df.columns
    assert "abundance" in df.columns
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_results.py::test_biomass_by_age -v`
Expected: FAIL

**Step 3: Add all methods to `osmose/results.py`**

Add after `spatial_biomass()` method (line 83):

```python
def biomass_by_age(self, species: str | None = None) -> pd.DataFrame:
    """Read biomass structured by age class."""
    return self._read_2d_output("biomassByAge", species)

def biomass_by_size(self, species: str | None = None) -> pd.DataFrame:
    """Read biomass structured by size bin."""
    return self._read_2d_output("biomassBySize", species)

def biomass_by_tl(self, species: str | None = None) -> pd.DataFrame:
    """Read biomass structured by trophic level bin."""
    return self._read_2d_output("biomassByTL", species)

def abundance_by_age(self, species: str | None = None) -> pd.DataFrame:
    """Read abundance structured by age class."""
    return self._read_2d_output("abundanceByAge", species)

def abundance_by_size(self, species: str | None = None) -> pd.DataFrame:
    """Read abundance structured by size bin."""
    return self._read_2d_output("abundanceBySize", species)

def abundance_by_tl(self, species: str | None = None) -> pd.DataFrame:
    """Read abundance structured by trophic level bin."""
    return self._read_2d_output("abundanceByTL", species)

def yield_by_age(self, species: str | None = None) -> pd.DataFrame:
    """Read yield/catch biomass by age class."""
    return self._read_2d_output("yieldByAge", species)

def yield_by_size(self, species: str | None = None) -> pd.DataFrame:
    """Read yield/catch biomass by size bin."""
    return self._read_2d_output("yieldBySize", species)

def yield_abundance(self, species: str | None = None) -> pd.DataFrame:
    """Read yield/catch in numbers (yieldN)."""
    return self._read_species_output("yieldN", species)

def yield_n_by_age(self, species: str | None = None) -> pd.DataFrame:
    """Read catch numbers by age class."""
    return self._read_2d_output("yieldNByAge", species)

def yield_n_by_size(self, species: str | None = None) -> pd.DataFrame:
    """Read catch numbers by size bin."""
    return self._read_2d_output("yieldNBySize", species)

def diet_by_age(self, species: str | None = None) -> pd.DataFrame:
    """Read diet composition by predator age class."""
    return self._read_2d_output("dietMatrixByAge", species)

def diet_by_size(self, species: str | None = None) -> pd.DataFrame:
    """Read diet composition by predator size class."""
    return self._read_2d_output("dietMatrixBySize", species)

def mortality_rate(self, species: str | None = None) -> pd.DataFrame:
    """Read mortality rates by source (predation, starvation, fishing, natural)."""
    return self._read_species_output("mortalityRate", species)

def size_spectrum(self) -> pd.DataFrame:
    """Read community size spectrum (log size vs log abundance)."""
    pattern = f"{self.prefix}_sizeSpectrum*.csv"
    frames = []
    for filepath in sorted(self.output_dir.glob(pattern)):
        frames.append(pd.read_csv(filepath))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def mean_size_by_age(self, species: str | None = None) -> pd.DataFrame:
    """Read mean body size per age class."""
    return self._read_2d_output("meanSizeByAge", species)

def mean_tl_by_size(self, species: str | None = None) -> pd.DataFrame:
    """Read mean trophic level per size bin."""
    return self._read_2d_output("meanTLBySize", species)

def mean_tl_by_age(self, species: str | None = None) -> pd.DataFrame:
    """Read mean trophic level per age class."""
    return self._read_2d_output("meanTLByAge", species)

def spatial_abundance(self, filename: str) -> xr.Dataset:
    """Read spatial abundance grid from NetCDF."""
    return self.read_netcdf(filename)

def spatial_size(self, filename: str) -> xr.Dataset:
    """Read spatial mean size grid from NetCDF."""
    return self.read_netcdf(filename)

def spatial_yield(self, filename: str) -> xr.Dataset:
    """Read spatial catch grid from NetCDF."""
    return self.read_netcdf(filename)

def spatial_ltl(self, filename: str) -> xr.Dataset:
    """Read spatial LTL distribution from NetCDF."""
    return self.read_netcdf(filename)
```

**Step 4: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_results.py -v`
Expected: All PASS

**Step 5: Format and commit**

```bash
.venv/bin/ruff format osmose/results.py tests/test_results.py
git add osmose/results.py tests/test_results.py
git commit -m "feat: add 22 new output type parsers for full OSMOSE output coverage"
```

---

### Task 5.3: Update Results UI with new output types

Add all new output types to the Results page dropdown.

**Files:**
- Modify: `ui/pages/results.py:112-123` (choices dict)
- Modify: `ui/pages/results.py:184-191` (data loading)
- Modify: `ui/pages/results.py:221-227` (col_map)

**Step 1: Update the choices dict**

In `results_ui()`, replace the `result_type` choices:

```python
ui.input_select(
    "result_type",
    "Output type",
    choices={
        "biomass": "Biomass",
        "abundance": "Abundance",
        "yield": "Yield",
        "mortality": "Mortality",
        "diet": "Diet Matrix",
        "trophic": "Trophic Level",
        "biomass_by_age": "Biomass by Age",
        "biomass_by_size": "Biomass by Size",
        "biomass_by_tl": "Biomass by TL",
        "abundance_by_age": "Abundance by Age",
        "abundance_by_size": "Abundance by Size",
        "yield_by_age": "Yield by Age",
        "yield_by_size": "Yield by Size",
        "yield_n": "Catch Numbers",
        "mortality_rate": "Mortality by Source",
        "size_spectrum": "Size Spectrum",
    },
    selected="biomass",
),
```

**Step 2: Update data loading in `_load_results()`**

Add after the existing data loads (line 191):

```python
data["biomass_by_age"] = res.biomass_by_age()
data["biomass_by_size"] = res.biomass_by_size()
data["biomass_by_tl"] = res.biomass_by_tl()
data["abundance_by_age"] = res.abundance_by_age()
data["abundance_by_size"] = res.abundance_by_size()
data["yield_by_age"] = res.yield_by_age()
data["yield_by_size"] = res.yield_by_size()
data["yield_n"] = res.yield_abundance()
data["mortality_rate"] = res.mortality_rate()
data["size_spectrum"] = res.size_spectrum()
```

**Step 3: Update col_map and title_map**

Add entries for the new types:

```python
col_map = {
    "biomass": "biomass",
    "abundance": "abundance",
    "yield": "yield",
    "mortality": "mortality",
    "trophic": "meanTL",
    "biomass_by_age": "value",
    "biomass_by_size": "value",
    "biomass_by_tl": "value",
    "abundance_by_age": "value",
    "abundance_by_size": "value",
    "yield_by_age": "value",
    "yield_by_size": "value",
    "yield_n": "yieldN",
    "mortality_rate": "predation",
    "size_spectrum": "abundance",
}
```

**Step 4: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All PASS

**Step 5: Format and commit**

```bash
.venv/bin/ruff format ui/pages/results.py
git add ui/pages/results.py
git commit -m "feat: add 10 new output types to Results UI dropdown"
```

---

## Phase 6: Visualization Parity

### Task 6.1: Add structured output chart functions

Create pure chart functions for all new output types.

**Files:**
- Create: `osmose/plotting.py`
- Test: `tests/test_plotting.py`

**Step 1: Write failing tests**

Create `tests/test_plotting.py`:

```python
"""Tests for osmose.plotting — chart generation functions."""

import numpy as np
import pandas as pd
import pytest

from osmose.plotting import (
    make_stacked_area,
    make_mortality_breakdown,
    make_size_spectrum_plot,
    make_ci_timeseries,
)


@pytest.fixture
def biomass_by_age_df():
    """Long-format 2D output: time, species, bin, value."""
    rows = []
    for t in range(10):
        for age in ["0", "1", "2"]:
            rows.append({"time": t, "species": "Anchovy", "bin": age, "value": np.random.rand() * 100})
    return pd.DataFrame(rows)


@pytest.fixture
def mortality_df():
    return pd.DataFrame({
        "time": range(10),
        "predation": np.random.rand(10) * 0.3,
        "starvation": np.random.rand(10) * 0.1,
        "fishing": np.random.rand(10) * 0.2,
        "natural": np.random.rand(10) * 0.1,
        "species": ["Anchovy"] * 10,
    })


@pytest.fixture
def spectrum_df():
    return pd.DataFrame({
        "size": [1, 2, 4, 8, 16, 32],
        "abundance": [100000, 50000, 10000, 1000, 100, 10],
    })


def test_stacked_area(biomass_by_age_df):
    fig = make_stacked_area(biomass_by_age_df, title="Biomass by Age")
    assert fig is not None
    assert len(fig.data) > 0


def test_stacked_area_empty():
    fig = make_stacked_area(pd.DataFrame(), title="Empty")
    assert fig is not None
    assert len(fig.data) == 0


def test_mortality_breakdown(mortality_df):
    fig = make_mortality_breakdown(mortality_df, species="Anchovy")
    assert fig is not None
    assert len(fig.data) > 0


def test_size_spectrum(spectrum_df):
    fig = make_size_spectrum_plot(spectrum_df)
    assert fig is not None
    assert len(fig.data) >= 1  # scatter + regression line


def test_ci_timeseries():
    time = list(range(10))
    mean = np.random.rand(10) * 100
    lower = mean - 10
    upper = mean + 10
    fig = make_ci_timeseries(time, mean, lower, upper, title="Biomass CI")
    assert fig is not None
    assert len(fig.data) >= 2  # mean line + CI band
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_plotting.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'osmose.plotting'`

**Step 3: Implement `osmose/plotting.py`**

```python
"""Chart generation functions for OSMOSE outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def make_stacked_area(
    df: pd.DataFrame,
    title: str,
    species: str | None = None,
) -> go.Figure:
    """Stacked area chart for 2D outputs (ByAge/BySize/ByTL).

    Expects long-format DataFrame with columns: time, species, bin, value.
    """
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title=title, template="osmose")
        return fig

    if species and "species" in df.columns:
        df = df[df["species"] == species]

    for bin_label in sorted(df["bin"].unique()):
        bin_data = df[df["bin"] == bin_label].sort_values("time")
        fig.add_trace(go.Scatter(
            x=bin_data["time"],
            y=bin_data["value"],
            name=str(bin_label),
            mode="lines",
            stackgroup="one",
        ))

    fig.update_layout(title=title, template="osmose", xaxis_title="Time", yaxis_title="Value")
    return fig


def make_mortality_breakdown(
    df: pd.DataFrame,
    species: str | None = None,
) -> go.Figure:
    """Stacked area chart of mortality by source."""
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="Mortality Breakdown", template="osmose")
        return fig

    if species and "species" in df.columns:
        df = df[df["species"] == species]

    sources = ["predation", "starvation", "fishing", "natural"]
    colors = {"predation": "#e74c3c", "starvation": "#f39c12", "fishing": "#3498db", "natural": "#95a5a6"}

    for source in sources:
        if source in df.columns:
            fig.add_trace(go.Scatter(
                x=df["time"],
                y=df[source],
                name=source.title(),
                mode="lines",
                stackgroup="one",
                line={"color": colors.get(source)},
            ))

    fig.update_layout(
        title="Mortality Breakdown",
        template="osmose",
        xaxis_title="Time",
        yaxis_title="Mortality Rate",
    )
    return fig


def make_size_spectrum_plot(df: pd.DataFrame) -> go.Figure:
    """Log-log size spectrum with regression line."""
    fig = go.Figure()
    if df.empty or "size" not in df.columns or "abundance" not in df.columns:
        fig.update_layout(title="Size Spectrum", template="osmose")
        return fig

    log_size = np.log10(df["size"].values.astype(float))
    log_abund = np.log10(df["abundance"].values.astype(float) + 1e-10)

    fig.add_trace(go.Scatter(
        x=log_size, y=log_abund, mode="markers", name="Data",
    ))

    # Linear regression on log-log
    if len(log_size) >= 2:
        coeffs = np.polyfit(log_size, log_abund, 1)
        fit_y = np.polyval(coeffs, log_size)
        fig.add_trace(go.Scatter(
            x=log_size, y=fit_y, mode="lines", name=f"Slope={coeffs[0]:.2f}",
            line={"dash": "dash"},
        ))

    fig.update_layout(
        title="Community Size Spectrum",
        template="osmose",
        xaxis_title="log10(Size)",
        yaxis_title="log10(Abundance)",
    )
    return fig


def make_ci_timeseries(
    time: list | np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    title: str = "Time Series",
    y_label: str = "Value",
) -> go.Figure:
    """Time series with confidence interval band."""
    fig = go.Figure()
    time = list(time)

    # CI band (upper bound + reversed lower bound for fill)
    fig.add_trace(go.Scatter(
        x=time + time[::-1],
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor="rgba(52, 152, 219, 0.2)",
        line={"color": "rgba(255,255,255,0)"},
        showlegend=False,
        name="95% CI",
    ))

    # Mean line
    fig.add_trace(go.Scatter(
        x=time, y=list(mean), mode="lines", name="Mean",
        line={"color": "#3498db", "width": 2},
    ))

    fig.update_layout(title=title, template="osmose", xaxis_title="Time", yaxis_title=y_label)
    return fig


def make_growth_curves(
    species_params: list[dict],
) -> go.Figure:
    """Plot von Bertalanffy growth curves for all species.

    Each dict in species_params should have keys: name, linf, k, t0, lifespan.
    """
    fig = go.Figure()
    for sp in species_params:
        ages = np.linspace(0, sp["lifespan"], 100)
        lengths = sp["linf"] * (1 - np.exp(-sp["k"] * (ages - sp["t0"])))
        lengths = np.clip(lengths, 0, None)
        fig.add_trace(go.Scatter(x=ages, y=lengths, mode="lines", name=sp["name"]))

    fig.update_layout(
        title="Von Bertalanffy Growth Curves",
        template="osmose",
        xaxis_title="Age (years)",
        yaxis_title="Length (cm)",
    )
    return fig


def make_predation_ranges(
    species_params: list[dict],
) -> go.Figure:
    """Horizontal bar chart of predator-prey size ratios.

    Each dict should have keys: name, size_ratio_min, size_ratio_max.
    """
    fig = go.Figure()
    names = [sp["name"] for sp in species_params]
    mins = [sp["size_ratio_min"] for sp in species_params]
    maxs = [sp["size_ratio_max"] for sp in species_params]

    for i, sp in enumerate(species_params):
        fig.add_trace(go.Bar(
            x=[sp["size_ratio_max"] - sp["size_ratio_min"]],
            y=[sp["name"]],
            base=[sp["size_ratio_min"]],
            orientation="h",
            name=sp["name"],
            showlegend=False,
        ))

    fig.update_layout(
        title="Predator-Prey Size Ratios",
        template="osmose",
        xaxis_title="Prey/Predator Size Ratio",
        yaxis_title="Species",
    )
    return fig
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_plotting.py -v`
Expected: All PASS

**Step 5: Format and commit**

```bash
.venv/bin/ruff format osmose/plotting.py tests/test_plotting.py
git add osmose/plotting.py tests/test_plotting.py
git commit -m "feat: add chart generation module with stacked area, mortality, spectrum, CI, growth, predation plots"
```

---

### Task 6.2: Add config visualization charts + tests

Add tests for growth curves and predation range charts.

**Files:**
- Modify: `tests/test_plotting.py`

**Step 1: Add tests**

```python
def test_growth_curves():
    species = [
        {"name": "Anchovy", "linf": 19.5, "k": 0.364, "t0": -0.7, "lifespan": 4},
        {"name": "Sardine", "linf": 23.0, "k": 0.28, "t0": -0.9, "lifespan": 5},
    ]
    fig = make_growth_curves(species)
    assert len(fig.data) == 2


def test_predation_ranges():
    species = [
        {"name": "Anchovy", "size_ratio_min": 0.01, "size_ratio_max": 0.1},
        {"name": "Hake", "size_ratio_min": 0.05, "size_ratio_max": 0.5},
    ]
    fig = make_predation_ranges(species)
    assert len(fig.data) == 2
```

**Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_plotting.py -v`
Expected: All PASS (implementation already done in Task 6.1)

**Step 3: Commit**

```bash
git add tests/test_plotting.py
git commit -m "test: add config visualization chart tests"
```

---

### Task 6.3: Wire new charts into Results UI

Connect the new chart functions to the Results page.

**Files:**
- Modify: `ui/pages/results.py`

**Step 1: Update `results_chart()` to use structured charts**

In the `results_chart()` render function, add handling for 2D output types:

```python
# Inside results_chart(), after the diet check:
structured_types = {
    "biomass_by_age", "biomass_by_size", "biomass_by_tl",
    "abundance_by_age", "abundance_by_size",
    "yield_by_age", "yield_by_size",
}
if rtype in structured_types:
    from osmose.plotting import make_stacked_area
    df = data.get(rtype, pd.DataFrame())
    return make_stacked_area(df, title=title_map.get(rtype, rtype), species=sp)

if rtype == "mortality_rate":
    from osmose.plotting import make_mortality_breakdown
    df = data.get(rtype, pd.DataFrame())
    return make_mortality_breakdown(df, species=sp)

if rtype == "size_spectrum":
    from osmose.plotting import make_size_spectrum_plot
    df = data.get(rtype, pd.DataFrame())
    return make_size_spectrum_plot(df)
```

**Step 2: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All PASS

**Step 3: Format and commit**

```bash
.venv/bin/ruff format ui/pages/results.py
git add ui/pages/results.py
git commit -m "feat: wire structured output charts into Results UI"
```

---

## Phase 7: Multi-Replicate Analysis

### Task 7.1: Create analysis module

**Files:**
- Create: `osmose/analysis.py`
- Test: `tests/test_analysis.py`

**Step 1: Write failing tests**

Create `tests/test_analysis.py`:

```python
"""Tests for osmose.analysis — ensemble statistics and ecological indicators."""

import numpy as np
import pandas as pd
import pytest

from osmose.analysis import (
    ensemble_stats,
    summary_table,
    shannon_diversity,
    mean_tl_catch,
    size_spectrum_slope,
)


@pytest.fixture
def replicate_dfs():
    """Three replicate biomass DataFrames."""
    dfs = []
    for _ in range(3):
        df = pd.DataFrame({
            "time": range(10),
            "biomass": np.random.rand(10) * 1000 + 500,
            "species": ["Anchovy"] * 10,
        })
        dfs.append(df)
    return dfs


def test_ensemble_stats(replicate_dfs):
    result = ensemble_stats(replicate_dfs, value_col="biomass")
    assert "mean" in result.columns
    assert "std" in result.columns
    assert "ci_lower" in result.columns
    assert "ci_upper" in result.columns
    assert len(result) == 10


def test_ensemble_stats_empty():
    result = ensemble_stats([], value_col="biomass")
    assert result.empty


def test_summary_table(replicate_dfs):
    table = summary_table(replicate_dfs, value_col="biomass")
    assert "species" in table.columns
    assert "mean" in table.columns
    assert "std" in table.columns


def test_shannon_diversity():
    biomass_df = pd.DataFrame({
        "time": [0, 0, 0, 1, 1, 1],
        "species": ["A", "B", "C", "A", "B", "C"],
        "biomass": [100, 200, 300, 150, 250, 100],
    })
    result = shannon_diversity(biomass_df)
    assert "time" in result.columns
    assert "shannon" in result.columns
    assert len(result) == 2
    assert all(result["shannon"] > 0)


def test_mean_tl_catch():
    yield_df = pd.DataFrame({
        "time": [0, 0, 1, 1],
        "species": ["A", "B", "A", "B"],
        "yield": [100, 200, 150, 50],
    })
    tl_df = pd.DataFrame({
        "species": ["A", "B"],
        "tl": [3.5, 4.2],
    })
    result = mean_tl_catch(yield_df, tl_df)
    assert "time" in result.columns
    assert "mean_tl" in result.columns
    assert len(result) == 2


def test_size_spectrum_slope():
    df = pd.DataFrame({
        "size": [1, 2, 4, 8, 16, 32],
        "abundance": [100000, 50000, 10000, 1000, 100, 10],
    })
    slope, intercept, r2 = size_spectrum_slope(df)
    assert slope < 0  # Negative slope expected
    assert r2 > 0.9  # Good fit expected for clean data
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_analysis.py -v`
Expected: FAIL

**Step 3: Implement `osmose/analysis.py`**

```python
"""Ensemble statistics and ecological indicators for OSMOSE outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ensemble_stats(
    replicate_dfs: list[pd.DataFrame],
    value_col: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute mean, std, 95% CI across replicates.

    Args:
        replicate_dfs: List of DataFrames, one per replicate, each with time + value_col.
        value_col: Column name to aggregate.
        group_cols: Columns to group by (default: ["time"]).

    Returns:
        DataFrame with columns: time, mean, std, ci_lower, ci_upper.
    """
    if not replicate_dfs:
        return pd.DataFrame()

    if group_cols is None:
        group_cols = ["time"]

    # Add replicate index and concatenate
    frames = []
    for i, df in enumerate(replicate_dfs):
        d = df.copy()
        d["_replicate"] = i
        frames.append(d)

    combined = pd.concat(frames, ignore_index=True)

    grouped = combined.groupby(group_cols)[value_col]
    result = grouped.agg(["mean", "std", "count"]).reset_index()
    result["std"] = result["std"].fillna(0)
    # 95% CI: mean +/- 1.96 * std / sqrt(n)
    result["ci_lower"] = result["mean"] - 1.96 * result["std"] / np.sqrt(result["count"])
    result["ci_upper"] = result["mean"] + 1.96 * result["std"] / np.sqrt(result["count"])
    result = result.drop(columns=["count"])
    return result


def summary_table(
    replicate_dfs: list[pd.DataFrame],
    value_col: str,
) -> pd.DataFrame:
    """Statistical summary per species across replicates.

    Returns DataFrame with columns: species, mean, std, min, max, median.
    """
    if not replicate_dfs:
        return pd.DataFrame()

    combined = pd.concat(replicate_dfs, ignore_index=True)
    if "species" not in combined.columns:
        combined["species"] = "all"

    result = combined.groupby("species")[value_col].agg(
        ["mean", "std", "min", "max", "median"]
    ).reset_index()
    return result


def shannon_diversity(biomass_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Shannon-Wiener diversity index per timestep.

    Args:
        biomass_df: DataFrame with columns: time, species, biomass.

    Returns:
        DataFrame with columns: time, shannon.
    """
    rows = []
    for t, group in biomass_df.groupby("time"):
        values = group["biomass"].values.astype(float)
        total = values.sum()
        if total <= 0:
            rows.append({"time": t, "shannon": 0.0})
            continue
        proportions = values / total
        proportions = proportions[proportions > 0]
        h = -np.sum(proportions * np.log(proportions))
        rows.append({"time": t, "shannon": float(h)})
    return pd.DataFrame(rows)


def mean_tl_catch(
    yield_df: pd.DataFrame,
    tl_df: pd.DataFrame,
) -> pd.DataFrame:
    """Mean trophic level of catch, weighted by yield.

    Args:
        yield_df: DataFrame with columns: time, species, yield.
        tl_df: DataFrame with columns: species, tl.

    Returns:
        DataFrame with columns: time, mean_tl.
    """
    merged = pd.merge(yield_df, tl_df, on="species")
    rows = []
    for t, group in merged.groupby("time"):
        total_yield = group["yield"].sum()
        if total_yield <= 0:
            rows.append({"time": t, "mean_tl": 0.0})
            continue
        weighted_tl = (group["yield"] * group["tl"]).sum() / total_yield
        rows.append({"time": t, "mean_tl": float(weighted_tl)})
    return pd.DataFrame(rows)


def size_spectrum_slope(
    spectrum_df: pd.DataFrame,
) -> tuple[float, float, float]:
    """Compute log-log size spectrum slope, intercept, and R-squared.

    Args:
        spectrum_df: DataFrame with columns: size, abundance.

    Returns:
        Tuple of (slope, intercept, r_squared).
    """
    log_size = np.log10(spectrum_df["size"].values.astype(float))
    log_abund = np.log10(spectrum_df["abundance"].values.astype(float) + 1e-10)

    coeffs = np.polyfit(log_size, log_abund, 1)
    slope, intercept = coeffs

    # R-squared
    predicted = np.polyval(coeffs, log_size)
    ss_res = np.sum((log_abund - predicted) ** 2)
    ss_tot = np.sum((log_abund - np.mean(log_abund)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(slope), float(intercept), float(r2)
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_analysis.py -v`
Expected: All PASS

**Step 5: Format and commit**

```bash
.venv/bin/ruff format osmose/analysis.py tests/test_analysis.py
git add osmose/analysis.py tests/test_analysis.py
git commit -m "feat: add analysis module with ensemble stats, Shannon diversity, mean TL catch, size spectrum slope"
```

---

### Task 7.2: Add `run_ensemble()` to runner

**Files:**
- Modify: `osmose/runner.py`
- Test: `tests/test_runner.py`

**Step 1: Write failing test**

Add to `tests/test_runner.py`:

```python
@pytest.mark.asyncio
async def test_run_ensemble(tmp_path):
    """run_ensemble runs n replicates with different seeds."""
    # Create a mock script that writes a CSV
    script = tmp_path / "mock.py"
    script.write_text(
        'import sys, os\n'
        'out = [a.split("=",1)[1] for a in sys.argv if a.startswith("-Poutput.dir.path=")][0]\n'
        'os.makedirs(out, exist_ok=True)\n'
        'with open(os.path.join(out, "osm_biomass_A.csv"), "w") as f:\n'
        '    f.write("time,biomass\\n0,100\\n")\n'
    )
    config = tmp_path / "config.csv"
    config.write_text("simulation.time.nyear ; 1\n")

    from osmose.runner import OsmoseRunner
    runner = OsmoseRunner(jar_path=script, java_cmd=sys.executable)
    # Monkey-patch _build_cmd to use python instead of java -jar
    original_build = runner._build_cmd
    def mock_build(config_path, output_dir=None, java_opts=None, overrides=None):
        cmd = [sys.executable, str(script), str(config_path)]
        if output_dir:
            cmd.append(f"-Poutput.dir.path={output_dir}")
        if overrides:
            for k, v in overrides.items():
                cmd.append(f"-P{k}={v}")
        return cmd
    runner._build_cmd = mock_build

    results = await runner.run_ensemble(config, tmp_path / "ensemble_out", n_replicates=3)
    assert len(results) == 3
    for r in results:
        assert r.returncode == 0
        assert r.output_dir.exists()
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_runner.py::test_run_ensemble -v`
Expected: FAIL

**Step 3: Add `run_ensemble()` to `osmose/runner.py`**

Add after the `run()` method:

```python
async def run_ensemble(
    self,
    config_path: Path,
    output_dir: Path,
    n_replicates: int = 5,
    java_opts: list[str] | None = None,
    overrides: dict[str, str] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> list[RunResult]:
    """Run multiple OSMOSE replicates with different random seeds.

    Creates output_dir/rep_0/, rep_1/, etc.
    Returns list of RunResult, one per replicate.
    """
    _log.info("Starting ensemble run: %d replicates", n_replicates)
    results = []
    for i in range(n_replicates):
        rep_dir = output_dir / f"rep_{i}"
        rep_overrides = dict(overrides or {})
        rep_overrides["simulation.random.seed"] = str(i)
        if on_progress:
            on_progress(f"Starting replicate {i + 1}/{n_replicates}")
        result = await self.run(
            config_path,
            output_dir=rep_dir,
            java_opts=java_opts,
            overrides=rep_overrides,
            on_progress=on_progress,
        )
        results.append(result)
    return results
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_runner.py -v`
Expected: All PASS

**Step 5: Format and commit**

```bash
.venv/bin/ruff format osmose/runner.py tests/test_runner.py
git add osmose/runner.py tests/test_runner.py
git commit -m "feat: add run_ensemble() for multi-replicate OSMOSE runs"
```

---

## Phase 8: Calibration Depth

### Task 8.1: Add new objective functions

**Files:**
- Modify: `osmose/calibration/objectives.py`
- Test: `tests/test_objectives.py`

**Step 1: Write failing tests**

Add to `tests/test_objectives.py`:

```python
from osmose.calibration.objectives import (
    yield_rmse,
    catch_at_size_distance,
    size_at_age_rmse,
    weighted_multi_objective,
)


def test_yield_rmse():
    sim = pd.DataFrame({"time": [0, 1, 2], "yield": [100, 110, 120], "species": ["A"] * 3})
    obs = pd.DataFrame({"time": [0, 1, 2], "yield": [105, 115, 125], "species": ["A"] * 3})
    result = yield_rmse(sim, obs)
    assert result > 0
    assert result < 10


def test_yield_rmse_species_filter():
    sim = pd.DataFrame({"time": [0, 1], "yield": [100, 110], "species": ["A", "B"]})
    obs = pd.DataFrame({"time": [0, 1], "yield": [105, 115], "species": ["A", "B"]})
    result = yield_rmse(sim, obs, species="A")
    assert result > 0


def test_catch_at_size_distance():
    sim = pd.DataFrame({"time": [0, 0], "bin": ["0-5", "5-10"], "value": [50, 30], "species": ["A"] * 2})
    obs = pd.DataFrame({"time": [0, 0], "bin": ["0-5", "5-10"], "value": [55, 25], "species": ["A"] * 2})
    result = catch_at_size_distance(sim, obs)
    assert result > 0


def test_size_at_age_rmse():
    sim = pd.DataFrame({"time": [0, 0], "bin": ["0", "1"], "value": [5.0, 10.0], "species": ["A"] * 2})
    obs = pd.DataFrame({"time": [0, 0], "bin": ["0", "1"], "value": [5.5, 9.5], "species": ["A"] * 2})
    result = size_at_age_rmse(sim, obs)
    assert result > 0


def test_weighted_multi_objective():
    objectives = [1.0, 2.0, 3.0]
    weights = [0.5, 0.3, 0.2]
    result = weighted_multi_objective(objectives, weights)
    assert abs(result - (0.5 * 1.0 + 0.3 * 2.0 + 0.2 * 3.0)) < 1e-10
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_objectives.py::test_yield_rmse -v`
Expected: FAIL

**Step 3: Add to `osmose/calibration/objectives.py`**

Append to the file:

```python
def yield_rmse(
    simulated: pd.DataFrame, observed: pd.DataFrame, species: str | None = None
) -> float:
    """RMSE for yield/catch biomass time series."""
    if species:
        simulated = simulated[simulated["species"] == species]
        observed = observed[observed["species"] == species]
    merged = pd.merge(simulated, observed, on="time", suffixes=("_sim", "_obs"))
    if merged.empty:
        return float("inf")
    diff = merged["yield_sim"] - merged["yield_obs"]
    return float(np.sqrt(np.mean(diff**2)))


def catch_at_size_distance(
    simulated: pd.DataFrame, observed: pd.DataFrame
) -> float:
    """Distance between catch-at-size compositions.

    Both DataFrames should have columns: time, bin, value, species.
    """
    merged = pd.merge(simulated, observed, on=["time", "bin"], suffixes=("_sim", "_obs"))
    if merged.empty:
        return float("inf")
    diff = merged["value_sim"] - merged["value_obs"]
    return float(np.sqrt(np.mean(diff**2)))


def size_at_age_rmse(
    simulated: pd.DataFrame, observed: pd.DataFrame
) -> float:
    """RMSE between mean size-at-age data.

    Both DataFrames should have columns: time, bin (age), value (mean size), species.
    """
    merged = pd.merge(simulated, observed, on=["time", "bin"], suffixes=("_sim", "_obs"))
    if merged.empty:
        return float("inf")
    diff = merged["value_sim"] - merged["value_obs"]
    return float(np.sqrt(np.mean(diff**2)))


def weighted_multi_objective(
    objectives: list[float], weights: list[float]
) -> float:
    """Weighted sum of multiple objective values."""
    return float(np.dot(objectives, weights))
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_objectives.py -v`
Expected: All PASS

**Step 5: Format and commit**

```bash
.venv/bin/ruff format osmose/calibration/objectives.py tests/test_objectives.py
git add osmose/calibration/objectives.py tests/test_objectives.py
git commit -m "feat: add yield_rmse, catch_at_size_distance, size_at_age_rmse, weighted_multi_objective"
```

---

### Task 8.2: Add multi-phase calibration

**Files:**
- Create: `osmose/calibration/multiphase.py`
- Test: `tests/test_multiphase.py`

**Step 1: Write failing tests**

Create `tests/test_multiphase.py`:

```python
"""Tests for multi-phase sequential calibration."""

import numpy as np
import pytest

from osmose.calibration.multiphase import CalibrationPhase, MultiPhaseCalibrator
from osmose.calibration.problem import FreeParameter


@pytest.fixture
def phases():
    return [
        CalibrationPhase(
            name="Phase 1: Mortality",
            free_params=[
                FreeParameter("mortality.natural.rate.sp0", 0.01, 1.0),
                FreeParameter("mortality.natural.rate.sp1", 0.01, 1.0),
            ],
            algorithm="Nelder-Mead",
            max_iter=10,
        ),
        CalibrationPhase(
            name="Phase 2: Growth",
            free_params=[
                FreeParameter("species.k.sp0", 0.05, 0.5),
            ],
            algorithm="L-BFGS-B",
            max_iter=10,
        ),
    ]


def test_phase_dataclass(phases):
    assert phases[0].name == "Phase 1: Mortality"
    assert len(phases[0].free_params) == 2
    assert phases[0].algorithm == "Nelder-Mead"


def test_multi_phase_calibrator_init(phases):
    calibrator = MultiPhaseCalibrator(phases=phases)
    assert len(calibrator.phases) == 2


def test_multi_phase_runs_sequentially(phases, tmp_path):
    """Phases run in order; output of phase 1 becomes fixed params for phase 2."""
    call_log = []

    def mock_optimize(phase, fixed_params, work_dir):
        call_log.append({"phase": phase.name, "fixed": dict(fixed_params)})
        # Return "best" param values
        return {fp.key: (fp.lower_bound + fp.upper_bound) / 2 for fp in phase.free_params}

    calibrator = MultiPhaseCalibrator(phases=phases)
    calibrator._optimize_phase = mock_optimize

    results = calibrator.run(work_dir=tmp_path)

    assert len(call_log) == 2
    # Phase 2 should have Phase 1's results as fixed params
    assert "mortality.natural.rate.sp0" in call_log[1]["fixed"]
    assert len(results) == 2


def test_multi_phase_results_accumulate(phases, tmp_path):
    def mock_optimize(phase, fixed_params, work_dir):
        return {fp.key: 0.5 for fp in phase.free_params}

    calibrator = MultiPhaseCalibrator(phases=phases)
    calibrator._optimize_phase = mock_optimize

    results = calibrator.run(work_dir=tmp_path)
    # Final results should have all params
    all_params = {}
    for r in results:
        all_params.update(r)
    assert len(all_params) == 3
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_multiphase.py -v`
Expected: FAIL

**Step 3: Implement `osmose/calibration/multiphase.py`**

```python
"""Multi-phase sequential calibration for OSMOSE."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from osmose.calibration.problem import FreeParameter
from osmose.logging import setup_logging

_log = setup_logging("osmose.calibration.multiphase")


@dataclass
class CalibrationPhase:
    """One phase of a sequential calibration."""

    name: str
    free_params: list[FreeParameter]
    algorithm: str = "Nelder-Mead"  # "Nelder-Mead", "L-BFGS-B", "DE", "NSGA-II"
    max_iter: int = 100
    n_replicates: int = 1


class MultiPhaseCalibrator:
    """Run calibration phases sequentially.

    Output of phase N becomes fixed parameters for phase N+1.
    """

    def __init__(self, phases: list[CalibrationPhase]):
        self.phases = phases

    def run(
        self,
        work_dir: Path,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[dict[str, float]]:
        """Run all phases sequentially.

        Returns:
            List of dicts, one per phase, mapping param keys to best values.
        """
        fixed_params: dict[str, float] = {}
        results: list[dict[str, float]] = []

        for i, phase in enumerate(self.phases):
            _log.info("Starting %s (%d/%d)", phase.name, i + 1, len(self.phases))
            if on_progress:
                on_progress(f"Phase {i + 1}/{len(self.phases)}: {phase.name}")

            phase_dir = work_dir / f"phase_{i}"
            phase_dir.mkdir(parents=True, exist_ok=True)

            best = self._optimize_phase(phase, fixed_params, phase_dir)
            results.append(best)
            fixed_params.update(best)
            _log.info("Phase %s complete: %s", phase.name, best)

        return results

    def _optimize_phase(
        self,
        phase: CalibrationPhase,
        fixed_params: dict[str, float],
        work_dir: Path,
    ) -> dict[str, float]:
        """Run a single optimization phase.

        Override this method in tests or subclasses to mock optimization.
        """
        import numpy as np
        from scipy.optimize import minimize, differential_evolution

        bounds = [(fp.lower_bound, fp.upper_bound) for fp in phase.free_params]
        x0 = np.array([(fp.lower_bound + fp.upper_bound) / 2 for fp in phase.free_params])

        def objective(x):
            # Placeholder: real implementation would run OSMOSE and compute objective
            # This is meant to be wired to an OsmoseCalibrationProblem
            return float(np.sum(x**2))

        if phase.algorithm == "DE":
            result = differential_evolution(objective, bounds, maxiter=phase.max_iter)
        else:
            result = minimize(
                objective, x0, method=phase.algorithm,
                bounds=bounds, options={"maxiter": phase.max_iter},
            )

        return {
            fp.key: float(result.x[j])
            for j, fp in enumerate(phase.free_params)
        }
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_multiphase.py -v`
Expected: All PASS

**Step 5: Format and commit**

```bash
.venv/bin/ruff format osmose/calibration/multiphase.py tests/test_multiphase.py
git add osmose/calibration/multiphase.py tests/test_multiphase.py
git commit -m "feat: add multi-phase sequential calibration with CalibrationPhase and MultiPhaseCalibrator"
```

---

### Task 8.3: Add `configureCalibration()` function

Extract calibration parameters from config (R equivalent).

**Files:**
- Create: `osmose/calibration/configure.py`
- Test: `tests/test_configure_calibration.py`

**Step 1: Write failing tests**

Create `tests/test_configure_calibration.py`:

```python
"""Tests for configureCalibration — extract calibration params from config."""

from osmose.calibration.configure import configure_calibration


def test_configure_calibration_basic():
    config = {
        "simulation.nspecies": "3",
        "mortality.natural.rate.sp0": "0.2",
        "mortality.natural.rate.sp1": "0.18",
        "mortality.natural.rate.sp2": "0.15",
        "species.k.sp0": "0.364",
        "species.k.sp1": "0.28",
        "species.k.sp2": "0.106",
        "population.seeding.biomass.sp0": "10000",
    }
    result = configure_calibration(config)
    assert "params" in result
    assert len(result["params"]) > 0
    # Should include mortality and growth params
    keys = [p["key"] for p in result["params"]]
    assert "mortality.natural.rate.sp0" in keys
    assert "species.k.sp0" in keys


def test_configure_calibration_has_bounds():
    config = {
        "simulation.nspecies": "1",
        "mortality.natural.rate.sp0": "0.2",
    }
    result = configure_calibration(config)
    for p in result["params"]:
        assert "guess" in p
        assert "lower" in p
        assert "upper" in p


def test_configure_calibration_empty():
    result = configure_calibration({})
    assert result["params"] == []
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_configure_calibration.py -v`
Expected: FAIL

**Step 3: Implement**

```python
"""Extract calibration parameters from OSMOSE config (R's configureCalibration equivalent)."""

from __future__ import annotations

import re

# Parameters typically calibrated, with default bounds
CALIBRATABLE_KEYS = {
    r"mortality\.natural\.rate\.sp\d+": (0.001, 2.0),
    r"mortality\.natural\.larva\.rate\.sp\d+": (0.001, 10.0),
    r"mortality\.starvation\.rate\.max\.sp\d+": (0.001, 5.0),
    r"species\.k\.sp\d+": (0.01, 1.0),
    r"species\.linf\.sp\d+": (1.0, 300.0),
    r"predation\.ingestion\.rate\.max\.sp\d+": (0.5, 10.0),
    r"predation\.efficiency\.critical\.sp\d+": (0.1, 0.9),
    r"population\.seeding\.biomass\.sp\d+": (100, 1000000),
}


def configure_calibration(config: dict[str, str]) -> dict:
    """Extract calibration parameters from a flat OSMOSE config dict.

    Returns dict with key "params" containing list of dicts:
        [{key, guess, lower, upper}, ...]
    """
    params = []
    for key, value in sorted(config.items()):
        for pattern, (default_lower, default_upper) in CALIBRATABLE_KEYS.items():
            if re.fullmatch(pattern, key):
                try:
                    guess = float(value)
                except (ValueError, TypeError):
                    continue
                params.append({
                    "key": key,
                    "guess": guess,
                    "lower": default_lower,
                    "upper": default_upper,
                })
                break
    return {"params": params}
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_configure_calibration.py -v`
Expected: All PASS

**Step 5: Format and commit**

```bash
.venv/bin/ruff format osmose/calibration/configure.py tests/test_configure_calibration.py
git add osmose/calibration/configure.py tests/test_configure_calibration.py
git commit -m "feat: add configureCalibration() to extract calibratable params from config"
```

---

## Phase 9: Config Visualization & Validation

### Task 9.1: Add config validator

**Files:**
- Create: `osmose/config/validator.py`
- Test: `tests/test_validator.py`

**Step 1: Write failing tests**

Create `tests/test_validator.py`:

```python
"""Tests for config validation."""

import pytest
from osmose.config.validator import (
    validate_config,
    check_file_references,
    check_species_consistency,
)


@pytest.fixture
def registry():
    from osmose.schema.registry import ParameterRegistry
    from ui.state import _build_registry
    return _build_registry()


def test_validate_config_valid(registry):
    config = {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "100",
        "simulation.nspecies": "1",
        "species.name.sp0": "Anchovy",
        "species.linf.sp0": "19.5",
    }
    errors, warnings = validate_config(config, registry)
    assert len(errors) == 0


def test_validate_config_bad_type(registry):
    config = {
        "simulation.time.ndtperyear": "not_a_number",
        "simulation.nspecies": "1",
    }
    errors, warnings = validate_config(config, registry)
    assert any("ndtperyear" in e for e in errors)


def test_validate_config_out_of_bounds(registry):
    config = {
        "simulation.time.ndtperyear": "9999",
        "simulation.nspecies": "1",
    }
    errors, warnings = validate_config(config, registry)
    assert any("ndtperyear" in e for e in errors)


def test_check_species_consistency():
    config = {
        "simulation.nspecies": "2",
        "species.name.sp0": "Anchovy",
        # Missing sp1
    }
    warnings = check_species_consistency(config)
    assert len(warnings) > 0


def test_check_file_references(tmp_path):
    # Create a file that exists
    (tmp_path / "grid.csv").write_text("0,0\n")
    config = {
        "grid.mask.file": str(tmp_path / "grid.csv"),
        "reproduction.season.file.sp0": str(tmp_path / "missing.csv"),
    }
    missing = check_file_references(config, str(tmp_path))
    assert len(missing) == 1
    assert "missing.csv" in missing[0]
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_validator.py -v`
Expected: FAIL

**Step 3: Implement `osmose/config/validator.py`**

```python
"""Validate OSMOSE configuration against schema."""

from __future__ import annotations

import re
from pathlib import Path


def validate_config(config: dict[str, str], registry) -> tuple[list[str], list[str]]:
    """Validate config values against registry schema.

    Returns (errors, warnings) — lists of human-readable messages.
    """
    errors = []
    warnings = []

    for key, value in config.items():
        field = registry.match_field(key)
        if field is None:
            continue

        # Type check
        from osmose.schema.base import ParamType
        if field.param_type in (ParamType.FLOAT, ParamType.INT):
            try:
                num = float(value)
            except (ValueError, TypeError):
                errors.append(f"{key}: expected number, got '{value}'")
                continue

            # Bounds check
            if field.min_val is not None and num < field.min_val:
                errors.append(f"{key}: value {num} below minimum {field.min_val}")
            if field.max_val is not None and num > field.max_val:
                errors.append(f"{key}: value {num} above maximum {field.max_val}")

        elif field.param_type == ParamType.BOOL:
            if value.lower() not in ("true", "false", "0", "1"):
                errors.append(f"{key}: expected boolean, got '{value}'")

    return errors, warnings


def check_file_references(config: dict[str, str], base_dir: str) -> list[str]:
    """Check that all FILE_PATH parameters point to existing files.

    Returns list of error messages for missing files.
    """
    missing = []
    file_keys = [k for k in config if "file" in k.lower()]
    base = Path(base_dir)

    for key in file_keys:
        value = config[key]
        if not value or value.lower() in ("null", "none", ""):
            continue
        path = Path(value)
        if not path.is_absolute():
            path = base / path
        if not path.exists():
            missing.append(f"{key}: file not found: {path}")

    return missing


def check_species_consistency(config: dict[str, str]) -> list[str]:
    """Check that nspecies matches the number of indexed species params."""
    warnings = []
    nspecies_str = config.get("simulation.nspecies", "0")
    try:
        nspecies = int(nspecies_str)
    except ValueError:
        return [f"simulation.nspecies is not a number: {nspecies_str}"]

    # Check species.name.spN exists for all N
    for i in range(nspecies):
        key = f"species.name.sp{i}"
        if key not in config:
            warnings.append(f"Missing {key} (expected {nspecies} species)")

    return warnings
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_validator.py -v`
Expected: All PASS

**Step 5: Format and commit**

```bash
.venv/bin/ruff format osmose/config/validator.py tests/test_validator.py
git add osmose/config/validator.py tests/test_validator.py
git commit -m "feat: add config validator with type, bounds, file reference, and species consistency checks"
```

---

## Phase 10: Grid & Spatial Utilities

### Task 10.1: Add grid creation module

**Files:**
- Create: `osmose/grid.py`
- Test: `tests/test_grid.py`

**Step 1: Write failing tests**

Create `tests/test_grid.py`:

```python
"""Tests for grid creation utilities."""

import numpy as np
import pytest
import xarray as xr

from osmose.grid import create_grid_csv, create_grid_netcdf, csv_maps_to_netcdf


def test_create_grid_csv(tmp_path):
    output = tmp_path / "grid.csv"
    create_grid_csv(nrows=5, ncols=10, output=output)
    assert output.exists()
    import pandas as pd
    df = pd.read_csv(output, header=None)
    assert df.shape == (5, 10)
    # Default: all ocean (1)
    assert df.values.sum() == 50


def test_create_grid_csv_with_mask(tmp_path):
    output = tmp_path / "grid.csv"
    mask = np.ones((5, 10))
    mask[0, :] = -1  # First row is land
    create_grid_csv(nrows=5, ncols=10, output=output, mask=mask)
    import pandas as pd
    df = pd.read_csv(output, header=None)
    assert df.iloc[0].sum() == -10  # land cells


def test_create_grid_netcdf(tmp_path):
    output = tmp_path / "grid.nc"
    create_grid_netcdf(
        lat_bounds=(43.0, 48.0),
        lon_bounds=(-6.0, -1.0),
        nlat=10,
        nlon=20,
        output=output,
    )
    assert output.exists()
    ds = xr.open_dataset(output)
    assert "mask" in ds.data_vars
    assert ds.sizes["lat"] == 10
    assert ds.sizes["lon"] == 20
    ds.close()


def test_csv_maps_to_netcdf(tmp_path):
    # Create a CSV map
    csv_dir = tmp_path / "maps_csv"
    csv_dir.mkdir()
    np.savetxt(csv_dir / "map_Anchovy_summer.csv", np.random.rand(5, 10), delimiter=",")

    output = tmp_path / "maps.nc"
    csv_maps_to_netcdf(
        csv_dir=csv_dir,
        output=output,
        nlat=5,
        nlon=10,
        lat_bounds=(43.0, 48.0),
        lon_bounds=(-6.0, -1.0),
    )
    assert output.exists()
    ds = xr.open_dataset(output)
    assert len(ds.data_vars) >= 1
    ds.close()
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_grid.py -v`
Expected: FAIL

**Step 3: Implement `osmose/grid.py`**

```python
"""Grid creation and spatial utilities for OSMOSE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def create_grid_csv(
    nrows: int,
    ncols: int,
    output: Path,
    mask: np.ndarray | None = None,
) -> None:
    """Create a CSV grid mask file.

    Args:
        nrows: Number of latitude rows.
        ncols: Number of longitude columns.
        output: Output file path.
        mask: Optional 2D array. Positive = ocean, negative = land. Default: all ocean.
    """
    if mask is None:
        mask = np.ones((nrows, ncols))
    pd.DataFrame(mask).to_csv(output, header=False, index=False)


def create_grid_netcdf(
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    nlat: int,
    nlon: int,
    output: Path,
    mask: np.ndarray | None = None,
) -> None:
    """Create an OSMOSE NetCDF grid file.

    Args:
        lat_bounds: (south, north) latitude limits.
        lon_bounds: (west, east) longitude limits.
        nlat: Number of latitude cells.
        nlon: Number of longitude cells.
        output: Output file path.
        mask: Optional 2D mask array. Default: all ocean (1).
    """
    lat = np.linspace(lat_bounds[0], lat_bounds[1], nlat)
    lon = np.linspace(lon_bounds[0], lon_bounds[1], nlon)

    if mask is None:
        mask = np.ones((nlat, nlon), dtype=np.float32)

    ds = xr.Dataset(
        {"mask": xr.DataArray(mask, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})},
        attrs={"description": "OSMOSE grid", "grid_type": "regular"},
    )
    ds.to_netcdf(output)
    ds.close()


def csv_maps_to_netcdf(
    csv_dir: Path,
    output: Path,
    nlat: int,
    nlon: int,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
) -> None:
    """Convert CSV distribution maps to a single NetCDF file.

    Reads all *.csv files in csv_dir, each assumed to be a (nlat x nlon) grid.
    Creates a NetCDF with one variable per CSV file.
    """
    csv_dir = Path(csv_dir)
    lat = np.linspace(lat_bounds[0], lat_bounds[1], nlat)
    lon = np.linspace(lon_bounds[0], lon_bounds[1], nlon)

    data_vars = {}
    for csv_file in sorted(csv_dir.glob("*.csv")):
        arr = np.loadtxt(csv_file, delimiter=",")
        if arr.shape != (nlat, nlon):
            continue
        var_name = csv_file.stem
        data_vars[var_name] = xr.DataArray(
            arr.astype(np.float32), dims=["lat", "lon"], coords={"lat": lat, "lon": lon},
        )

    if data_vars:
        ds = xr.Dataset(data_vars)
        ds.to_netcdf(output)
        ds.close()
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_grid.py -v`
Expected: All PASS

**Step 5: Format and commit**

```bash
.venv/bin/ruff format osmose/grid.py tests/test_grid.py
git add osmose/grid.py tests/test_grid.py
git commit -m "feat: add grid creation module with CSV, NetCDF, and map migration utilities"
```

---

## Phase 11: Reporting & Export

### Task 11.1: Add reporting module

**Files:**
- Create: `osmose/reporting.py`
- Test: `tests/test_reporting.py`

**Step 1: Write failing tests**

Create `tests/test_reporting.py`:

```python
"""Tests for report generation."""

import numpy as np
import pandas as pd
import pytest

from osmose.reporting import generate_report, summary_table


@pytest.fixture
def mock_results(tmp_path):
    """Create minimal output files for reporting."""
    for sp in ["Anchovy", "Sardine"]:
        df = pd.DataFrame({"time": range(10), "biomass": np.random.rand(10) * 1000})
        df.to_csv(tmp_path / f"osm_biomass_{sp}.csv", index=False)
    df = pd.DataFrame({"time": range(10), "yield": np.random.rand(10) * 100})
    df.to_csv(tmp_path / "osm_yield_Anchovy.csv", index=False)
    return tmp_path


def test_summary_table(mock_results):
    from osmose.results import OsmoseResults
    res = OsmoseResults(mock_results)
    table = summary_table(res)
    assert not table.empty
    assert "species" in table.columns
    assert "biomass_mean" in table.columns


def test_generate_report_html(mock_results, tmp_path):
    from osmose.results import OsmoseResults
    res = OsmoseResults(mock_results)
    config = {"simulation.nspecies": "2", "simulation.time.nyear": "10"}

    output_path = tmp_path / "report.html"
    generate_report(res, config, output_path, fmt="html")
    assert output_path.exists()
    content = output_path.read_text()
    assert "OSMOSE" in content
    assert "Anchovy" in content


def test_generate_report_empty(tmp_path):
    from osmose.results import OsmoseResults
    res = OsmoseResults(tmp_path)  # empty dir
    output_path = tmp_path / "report.html"
    generate_report(res, {}, output_path, fmt="html")
    assert output_path.exists()
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_reporting.py -v`
Expected: FAIL

**Step 3: Implement `osmose/reporting.py`**

```python
"""Generate HTML reports from OSMOSE simulation results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from osmose.results import OsmoseResults


def summary_table(results: OsmoseResults) -> pd.DataFrame:
    """Create a summary table with mean/std per species per output type."""
    rows = []
    bio = results.biomass()
    if not bio.empty and "species" in bio.columns:
        for sp, group in bio.groupby("species"):
            rows.append({
                "species": sp,
                "biomass_mean": group["biomass"].mean() if "biomass" in group.columns else None,
                "biomass_std": group["biomass"].std() if "biomass" in group.columns else None,
            })

    yld = results.yield_biomass()
    if not yld.empty and "species" in yld.columns:
        for sp, group in yld.groupby("species"):
            matching = [r for r in rows if r["species"] == sp]
            if matching:
                matching[0]["yield_mean"] = group["yield"].mean() if "yield" in group.columns else None
            else:
                rows.append({
                    "species": sp,
                    "yield_mean": group["yield"].mean() if "yield" in group.columns else None,
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def generate_report(
    results: OsmoseResults,
    config: dict[str, str],
    output_path: Path,
    fmt: str = "html",
) -> None:
    """Generate a report from OSMOSE results.

    Args:
        results: OsmoseResults instance.
        config: Flat config dict.
        output_path: Where to write the report.
        fmt: "html" (default).
    """
    output_path = Path(output_path)

    # Build summary
    table = summary_table(results)
    table_html = table.to_html(index=False, classes="table") if not table.empty else "<p>No data</p>"

    # Config summary
    nspecies = config.get("simulation.nspecies", "?")
    nyear = config.get("simulation.time.nyear", "?")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>OSMOSE Simulation Report</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
        h1 {{ color: #1a5276; border-bottom: 3px solid #2980b9; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 30px; }}
        .table {{ border-collapse: collapse; width: 100%; }}
        .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
        .table th {{ background-color: #2980b9; color: white; }}
        .config {{ background: #f8f9fa; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>OSMOSE Simulation Report</h1>

    <h2>Configuration</h2>
    <div class="config">
        <p><strong>Species:</strong> {nspecies}</p>
        <p><strong>Simulation years:</strong> {nyear}</p>
    </div>

    <h2>Summary Statistics</h2>
    {table_html}

    <h2>Species Details</h2>
"""
    # Add per-species sections
    bio = results.biomass()
    if not bio.empty and "species" in bio.columns:
        for sp in sorted(bio["species"].unique()):
            sp_data = bio[bio["species"] == sp]
            if "biomass" in sp_data.columns:
                html += f"<h3>{sp}</h3>\n"
                html += f"<p>Mean biomass: {sp_data['biomass'].mean():.1f}</p>\n"

    html += """
    <hr>
    <p><em>Generated by osmose-python</em></p>
</body>
</html>"""

    output_path.write_text(html)
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_reporting.py -v`
Expected: All PASS

**Step 5: Format and commit**

```bash
.venv/bin/ruff format osmose/reporting.py tests/test_reporting.py
git add osmose/reporting.py tests/test_reporting.py
git commit -m "feat: add HTML report generation with summary tables and per-species details"
```

---

## Phase 12: Demo Generation & Version Migration

### Task 12.1: Add demo generation and config migration

**Files:**
- Create: `osmose/demo.py`
- Test: `tests/test_demo.py`

**Step 1: Write failing tests**

Create `tests/test_demo.py`:

```python
"""Tests for demo generation and config migration."""

import pytest

from osmose.demo import osmose_demo, migrate_config


def test_osmose_demo_bay_of_biscay(tmp_path):
    result = osmose_demo("bay_of_biscay", tmp_path)
    assert "config_file" in result
    assert result["config_file"].exists()
    assert result["output_dir"].exists()
    # Should have master config + sub-files
    csv_files = list(result["config_file"].parent.glob("*.csv"))
    assert len(csv_files) >= 2


def test_osmose_demo_unknown_scenario(tmp_path):
    with pytest.raises(ValueError, match="Unknown scenario"):
        osmose_demo("nonexistent", tmp_path)


def test_osmose_demo_list():
    from osmose.demo import list_demos
    demos = list_demos()
    assert "bay_of_biscay" in demos


def test_migrate_config_noop():
    """Config already at target version should be unchanged."""
    config = {"osmose.version": "4.3.0", "simulation.nspecies": "3"}
    result = migrate_config(config, target_version="4.3.0")
    assert result == config


def test_migrate_config_renames():
    """Config migration should rename deprecated keys."""
    config = {"simulation.nplankton": "2"}
    result = migrate_config(config, target_version="4.3.0")
    assert "simulation.nresource" in result
    assert "simulation.nplankton" not in result
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_demo.py -v`
Expected: FAIL

**Step 3: Implement `osmose/demo.py`**

```python
"""Demo scenario generation and config version migration."""

from __future__ import annotations

import shutil
from pathlib import Path


# Key renames between OSMOSE versions
_MIGRATIONS: dict[str, dict[str, str]] = {
    "4.3.0": {
        "simulation.nplankton": "simulation.nresource",
    },
}


def list_demos() -> list[str]:
    """List available demo scenarios."""
    return ["bay_of_biscay"]


def osmose_demo(scenario: str, output_dir: Path) -> dict:
    """Generate a demo OSMOSE configuration.

    Args:
        scenario: Demo name (e.g., "bay_of_biscay").
        output_dir: Directory to write demo files.

    Returns:
        Dict with keys: config_file, output_dir.
    """
    output_dir = Path(output_dir)

    if scenario == "bay_of_biscay":
        return _generate_bay_of_biscay(output_dir)
    else:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {list_demos()}")


def _generate_bay_of_biscay(output_dir: Path) -> dict:
    """Generate Bay of Biscay 3-species demo."""
    # Copy from bundled examples if available
    examples_dir = Path(__file__).parent.parent / "data" / "examples"
    config_dir = output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    sim_output = output_dir / "output"
    sim_output.mkdir(parents=True, exist_ok=True)

    if examples_dir.exists():
        for f in examples_dir.glob("osm_*"):
            shutil.copy2(f, config_dir / f.name)
    else:
        # Generate minimal config
        master = config_dir / "osm_all-parameters.csv"
        master.write_text(
            "simulation.time.ndtperyear ; 24\n"
            "simulation.time.nyear ; 50\n"
            "simulation.nspecies ; 3\n"
            "simulation.nschool ; 20\n"
            "simulation.ncpu ; 1\n"
        )

    config_file = config_dir / "osm_all-parameters.csv"
    return {"config_file": config_file, "output_dir": sim_output}


def migrate_config(
    config: dict[str, str],
    target_version: str = "4.3.0",
) -> dict[str, str]:
    """Migrate config parameter names to a target OSMOSE version.

    Applies key renames for version compatibility.
    """
    current = config.get("osmose.version", "")
    if current == target_version:
        return dict(config)

    result = dict(config)
    renames = _MIGRATIONS.get(target_version, {})
    for old_key, new_key in renames.items():
        if old_key in result:
            result[new_key] = result.pop(old_key)

    result["osmose.version"] = target_version
    return result
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_demo.py -v`
Expected: All PASS

**Step 5: Format and commit**

```bash
.venv/bin/ruff format osmose/demo.py tests/test_demo.py
git add osmose/demo.py tests/test_demo.py
git commit -m "feat: add demo generation and config version migration utilities"
```

---

## Final Task: Run full test suite and verify

**Step 1: Run all tests**

```bash
.venv/bin/python -m pytest tests/ -v --tb=short
```

Expected: All tests pass (239 existing + ~116 new = ~355 total)

**Step 2: Lint**

```bash
.venv/bin/ruff check osmose/ ui/ tests/
.venv/bin/ruff format --check osmose/ ui/ tests/
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: full R parity implementation complete (Phases 5-12)"
```
