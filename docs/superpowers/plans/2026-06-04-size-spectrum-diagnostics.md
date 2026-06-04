# Community size-spectrum diagnostics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute community-level size-spectrum indicators (spectrum curve, log-log slope, Large-Fish Indicator, mean size, peak) from OSMOSE `*DistribBySize` community output — a pure-analysis feature that reuses existing helpers.

**Architecture:** New `osmose/size_spectrum.py` reads the `{prefix}_{metric}DistribBySize*.csv` community file **directly** (via the preamble-safe `osmose.results._read_output_csv`), sums species per `(Time, Size)`, windows by Time, and derives indicators — reusing `osmose.analysis.size_spectrum_slope` for the log-log fit and `osmose.plotting.make_size_spectrum_plot` for the spectrum chart. One new trend chart + a CLI. No engine / `OsmoseResults` change.

**Tech Stack:** Python 3.12, numpy, pandas, plotly; pytest, ruff. Run tests with `.venv/bin/python -m pytest`. Scripts run with `PYTHONPATH=/home/razinka/osmose/osmose-python`.

**Reference spec:** `docs/superpowers/specs/2026-06-04-size-spectrum-diagnostics-design.md` (reviewed clean over 2 in-loop rounds; the corrected chain was prototyped on real EEC data).

---

## Verified facts (audit — use exactly; confirmed by execution in round-2 review)

- Real community file: `data/eec_full/output/Indicators/eec_biomassDistribBySize_Simu0.csv` (and
  `eec_abundanceDistribBySize_Simu0.csv`). 1-line title preamble, then header
  `"Time","Size",<14 species>`, rows keyed by `(Time, Size)`, `Size` = lower edge (cm), equal
  width **10 cm**, 21 bins (0–200), 70 timesteps. `data/baltic/output/Indicators/baltic_*DistribBySize_Simu0.csv`
  are **0 bytes**.
- `osmose.results._read_output_csv(path: Path) -> pd.DataFrame` strips the preamble and returns
  wide `Time, Size, <species>` with **`Size` as float64** (confirmed). On a 0-byte file it raises
  `pandas.errors.EmptyDataError`.
- `osmose.analysis.size_spectrum_slope(df) -> (slope, intercept, r_squared)`: log-log OLS over df
  columns **`size`, `abundance`** (the "value" axis); filters to `size>0 & abundance>0`; **raises
  `ValueError` if < 2 positive pairs**. (Reuse it; wrap the raise → `None`.)
- `osmose.plotting.make_size_spectrum_plot(df) -> go.Figure`: log-log scatter + regression line
  over df columns **`size`, `abundance`**; returns `_empty_figure` if empty. (Reuse for the
  spectrum chart.) plotting.py exposes `TEMPLATE`, `ensure_templates()` (already called at import),
  `_empty_figure(title)`, `_require_columns(df, *cols, context=)`.
- EEC round-2 numbers (for the fixture test ballpark): all-bins biomass slope ≈ **−1.90** (R²0.67);
  min_size 10cm → ≈ **−2.77** (R²0.84); LFI@40 ≈ **0.073**; value-weighted mean midpoint ≈
  **20.8 cm**; peak at the 10 cm bin.
- CLI pattern (`scripts/compare_runs.py`): argparse `RawDescriptionHelpFormatter` + `description=__doc__`;
  `parser.error(...)` for bad args (exit 2); `dataclasses.asdict` for `--json`; `--plot <prefix>`
  writing `{prefix}_*.html` via `.write_html`; `main(argv=None) -> int` returning 0; `sys.exit(main())`.
- CI lints `osmose/ ui/ tests/` (NOT `scripts/`) with `ruff check` + `ruff format --check`. So
  `osmose/size_spectrum.py`, `osmose/plotting.py`, `tests/test_size_spectrum.py` must be ruff-clean.

## File Structure

- Create: `osmose/size_spectrum.py` — reader/reshape/window helpers, `SizeSpectrum`,
  `compute_size_spectrum`, `size_spectrum_timeseries`, `format_size_spectrum_report`,
  `spectrum_plot_df`.
- Modify: `osmose/plotting.py` — add `make_size_indicator_timeseries`.
- Create: `scripts/compute_size_spectrum.py` — CLI.
- Create: `tests/test_size_spectrum.py` — unit + EEC-fixture + plotting-smoke tests.
- Modify: `CHANGELOG.md` — Unreleased/Added note.

---

## Task 1: Reader + reshape + window helpers

**Files:**
- Create: `osmose/size_spectrum.py`
- Test: `tests/test_size_spectrum.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_size_spectrum.py`:
```python
from __future__ import annotations

import pandas as pd
import pytest

from osmose.size_spectrum import (
    _community_long,
    _infer_bin_width,
    _read_community_by_size,
    _window_by_time,
)


def _write_community_csv(path, rows):
    """rows: list of (Time, Size, sp1, sp2). Clean header, NO preamble."""
    pd.DataFrame(rows, columns=["Time", "Size", "sp1", "sp2"]).to_csv(path, index=False)


def test_read_community_by_size_finds_and_reads(tmp_path):
    d = tmp_path / "output" / "Indicators"
    d.mkdir(parents=True)
    _write_community_csv(d / "osm_biomassDistribBySize_Simu0.csv",
                         [(1.0, 0.0, 2.0, 3.0), (1.0, 10.0, 1.0, 1.0)])
    wide = _read_community_by_size(tmp_path / "output", "biomassDistribBySize", "osm")
    assert list(wide.columns) == ["Time", "Size", "sp1", "sp2"]
    assert len(wide) == 2


def test_read_community_by_size_missing_raises(tmp_path):
    (tmp_path / "output").mkdir()
    with pytest.raises(FileNotFoundError):
        _read_community_by_size(tmp_path / "output", "biomassDistribBySize", "osm")


def test_read_community_by_size_empty_file_raises(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    (d / "osm_biomassDistribBySize_Simu0.csv").write_text("")  # 0-content
    with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
        _read_community_by_size(d, "biomassDistribBySize", "osm")


def test_community_long_sums_species():
    wide = pd.DataFrame(
        {"Time": [1.0, 1.0], "Size": [0.0, 10.0], "sp1": [2.0, 1.0], "sp2": [3.0, 4.0]}
    )
    long = _community_long(wide)
    assert list(long.columns) == ["time", "size", "value"]
    # bin 0: 2+3=5 ; bin 10: 1+4=5
    assert long.loc[long["size"] == 0.0, "value"].iloc[0] == 5.0
    assert long.loc[long["size"] == 10.0, "value"].iloc[0] == 5.0
    assert long["size"].dtype == float


def test_window_by_time_selects_years_not_rows():
    df = pd.DataFrame({"time": [1.0, 1.0, 2.0, 3.0], "size": [0, 10, 0, 0], "value": [1, 1, 1, 1]})
    # last 1 year -> only time > 3-1=2 -> time==3 (1 row), NOT the last 1 row by count
    w = _window_by_time(df, "time", 1)
    assert sorted(w["time"].unique()) == [3.0]


def test_window_by_time_rejects_nonpositive():
    df = pd.DataFrame({"time": [1.0], "size": [0.0], "value": [1.0]})
    with pytest.raises(ValueError):
        _window_by_time(df, "time", 0)


def test_infer_bin_width():
    assert _infer_bin_width([0.0, 10.0, 20.0, 30.0]) == 10.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_size_spectrum.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'osmose.size_spectrum'`).

- [ ] **Step 3: Implement the helpers**

Create `osmose/size_spectrum.py`:
```python
"""Community size-spectrum diagnostics from OSMOSE *DistribBySize output.

Reads the community size-distribution file directly (the OsmoseResults *_by_size
accessors target a different, per-species layout), sums species per (Time, Size),
windows by Time, and derives community indicators: the size spectrum, its log-log
slope (reusing osmose.analysis.size_spectrum_slope), the Large-Fish Indicator, the
value-weighted mean size, and the modal (peak) bin.

This is a length-biomass (or length-abundance) spectrum over linear cm bins,
reported for trend/comparison — NOT the canonical Sheldon normalized-by-body-mass
exponent. The slope is sensitive to the small-bin cutoff; use `min_size_cm` to fit
the descending limb above the recruitment peak (see `peak_size_cm`).
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pandas as pd

from osmose.results import _read_output_csv


def _read_community_by_size(output_dir: Path, output_type: str, prefix: str) -> pd.DataFrame:
    """Locate + read the {prefix}_{output_type}*.csv community file (wide Time,Size,<species>).

    rglob handles the Indicators/ subdir. Raises FileNotFoundError if absent;
    pandas.errors.EmptyDataError propagates for a 0-byte file.
    """
    matches = sorted(Path(output_dir).rglob(f"{prefix}_{output_type}*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No '{prefix}_{output_type}*.csv' under {output_dir}; the run did not persist "
            f"by-size output (enable the by-size flag and re-run)."
        )
    return _read_output_csv(matches[0])


def _community_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Wide Time,Size,<species> -> long time,size,value (species summed per (time,size))."""
    species_cols = [c for c in wide.columns if c not in ("Time", "Size")]
    out = pd.DataFrame(
        {
            "time": wide["Time"].astype(float),
            "size": wide["Size"].astype(float),
            "value": wide[species_cols].sum(axis=1).astype(float),
        }
    )
    return out


def _window_by_time(df: pd.DataFrame, time_col: str, window_years: int) -> pd.DataFrame:
    """Keep rows whose time is within the trailing `window_years` (by Time-years, not rows)."""
    if window_years < 1:
        raise ValueError("window_years must be >= 1")
    tmax = float(df[time_col].max())
    return df[df[time_col] > tmax - window_years]


def _infer_bin_width(edges: list[float]) -> float:
    """Median consecutive diff of sorted unique edges (handles the common equal-width case)."""
    uniq = sorted(set(edges))
    if len(uniq) < 2:
        return 1.0
    diffs = [b - a for a, b in zip(uniq[:-1], uniq[1:])]
    return float(statistics.median(diffs))
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_size_spectrum.py -q`
Expected: PASS (7 tests).
Run: `.venv/bin/ruff check osmose/size_spectrum.py tests/test_size_spectrum.py && .venv/bin/ruff format --check osmose/size_spectrum.py tests/test_size_spectrum.py` (clean; if format flags, run `.venv/bin/ruff format <file>` and re-test).

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add osmose/size_spectrum.py tests/test_size_spectrum.py
git -C /home/razinka/osmose/osmose-python commit -m "feat(size-spectrum): community by-size reader + reshape/window helpers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `SizeSpectrum` + `compute_size_spectrum` + indicator helpers

**Files:**
- Modify: `osmose/size_spectrum.py`
- Test: `tests/test_size_spectrum.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_size_spectrum.py`:
```python
from pathlib import Path

from osmose.size_spectrum import (
    SizeSpectrum,
    _large_fish_indicator,
    _mean_size,
    compute_size_spectrum,
    spectrum_plot_df,
)


def test_large_fish_indicator():
    # edges 0,10,40,50 ; values 1,1,1,1 ; threshold 40 -> bins 40,50 -> 2/4
    assert _large_fish_indicator([0.0, 10.0, 40.0, 50.0], [1.0, 1.0, 1.0, 1.0], 40.0) == 0.5
    assert _large_fish_indicator([0.0], [0.0], 40.0) == 0.0  # zero total


def test_mean_size():
    # midpoints 5,15 ; values 1,3 -> (5*1+15*3)/4 = 12.5
    assert _mean_size([5.0, 15.0], [1.0, 3.0]) == 12.5


def test_compute_size_spectrum_known_powerlaw(tmp_path):
    # community value = 1e6 * midpoint^-2  -> log-log slope == -2 exactly
    d = tmp_path / "output"
    d.mkdir()
    rows = []
    for edge in (0.0, 10.0, 20.0, 30.0, 40.0):
        mid = edge + 5.0
        v = 1.0e6 * mid ** -2.0
        rows.append((1.0, edge, v / 2, v / 2))  # split across 2 species
    pd.DataFrame(rows, columns=["Time", "Size", "sp1", "sp2"]).to_csv(
        d / "osm_biomassDistribBySize_Simu0.csv", index=False
    )
    spec = compute_size_spectrum(d, metric="biomass", prefix="osm", window_years=1)
    assert isinstance(spec, SizeSpectrum)
    assert spec.metric == "biomass"
    assert spec.bin_edges == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert spec.slope is not None and abs(spec.slope - (-2.0)) < 1e-6
    assert spec.r_squared is not None and spec.r_squared > 0.999
    assert spec.n_bins_fit == 5
    assert spec.peak_size_cm == 5.0  # smallest midpoint has the largest value


def test_compute_size_spectrum_lfi_and_min_size(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    # equal value in every bin -> LFI@40 = (#bins edge>=40)/(#bins)
    rows = [(1.0, e, 5.0, 5.0) for e in (0.0, 10.0, 20.0, 30.0, 40.0)]
    pd.DataFrame(rows, columns=["Time", "Size", "sp1", "sp2"]).to_csv(
        d / "osm_biomassDistribBySize_Simu0.csv", index=False
    )
    spec = compute_size_spectrum(d, prefix="osm", window_years=1, lfi_threshold_cm=40.0)
    assert spec.lfi == pytest.approx(1 / 5)
    # min_size_cm filter drops bins below cutoff from the fit
    spec2 = compute_size_spectrum(d, prefix="osm", window_years=1, min_size_cm=20.0)
    assert spec2.min_size_cm == 20.0
    assert spec2.n_bins_fit == 3  # bins 20,30,40


def test_compute_size_spectrum_single_bin_slope_none(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    pd.DataFrame([(1.0, 0.0, 1.0, 1.0)], columns=["Time", "Size", "sp1", "sp2"]).to_csv(
        d / "osm_biomassDistribBySize_Simu0.csv", index=False
    )
    spec = compute_size_spectrum(d, prefix="osm", window_years=1)
    assert spec.slope is None and spec.intercept is None and spec.r_squared is None
    assert spec.n_bins_fit < 2


def test_compute_size_spectrum_eec_real():
    spec = compute_size_spectrum(
        Path("data/eec_full/output"), metric="biomass", prefix="eec", window_years=10
    )
    assert spec.slope is not None and spec.slope < 0
    assert 0.0 <= spec.lfi <= 1.0
    assert 0.0 < spec.mean_size_cm < 210.0
    assert spec.peak_size_cm < 50.0  # peak in a small bin
    ab = compute_size_spectrum(
        Path("data/eec_full/output"), metric="abundance", prefix="eec", window_years=10
    )
    assert ab.values != spec.values  # biomass vs abundance differ


def test_spectrum_plot_df_shape(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    rows = [(1.0, e, 1.0, 1.0) for e in (0.0, 10.0, 20.0)]
    pd.DataFrame(rows, columns=["Time", "Size", "sp1", "sp2"]).to_csv(
        d / "osm_biomassDistribBySize_Simu0.csv", index=False
    )
    spec = compute_size_spectrum(d, prefix="osm", window_years=1)
    pdf = spectrum_plot_df(spec)
    assert list(pdf.columns) == ["size", "abundance"]
    assert len(pdf) == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_size_spectrum.py -k "compute_size_spectrum or large_fish or mean_size or spectrum_plot_df" -q`
Expected: FAIL (`ImportError: cannot import name 'SizeSpectrum'`).

- [ ] **Step 3: Implement**

Add to `osmose/size_spectrum.py` — imports at top (extend the existing import block):
```python
from dataclasses import dataclass

from osmose.analysis import size_spectrum_slope
```
Then add the dataclass + helpers + the main function (append after the private helpers):
```python
@dataclass(frozen=True)
class SizeSpectrum:
    metric: str
    bin_edges: list[float]
    bin_midpoints: list[float]
    values: list[float]
    peak_size_cm: float
    slope: float | None
    intercept: float | None
    r_squared: float | None
    n_bins_fit: int
    min_size_cm: float | None
    lfi: float
    lfi_threshold_cm: float
    mean_size_cm: float
    window_years: int
    n_timesteps_used: int
    note: str


def _large_fish_indicator(edges: list[float], values: list[float], threshold: float) -> float:
    total = sum(values)
    if total <= 0:
        return 0.0
    large = sum(v for e, v in zip(edges, values) if e >= threshold)
    return float(large / total)


def _mean_size(midpoints: list[float], values: list[float]) -> float:
    total = sum(values)
    if total <= 0:
        return float("nan")
    return float(sum(m * v for m, v in zip(midpoints, values)) / total)


def _fit_slope(midpoints, values, min_size_cm):
    """Reuse analysis.size_spectrum_slope; apply min_size_cm filter; ValueError -> None."""
    df = pd.DataFrame({"size": midpoints, "abundance": values})
    if min_size_cm is not None:
        df = df[df["size"] >= min_size_cm]
    n_fit = int(((df["size"] > 0) & (df["abundance"] > 0)).sum())
    try:
        slope, intercept, r2 = size_spectrum_slope(df)
        return slope, intercept, r2, n_fit
    except ValueError:
        return None, None, None, n_fit


def compute_size_spectrum(
    output_dir,
    *,
    metric: str = "biomass",
    prefix: str = "osm",
    window_years: int = 10,
    lfi_threshold_cm: float = 40.0,
    min_size_cm: float | None = None,
) -> SizeSpectrum:
    if metric not in ("biomass", "abundance"):
        raise ValueError("metric must be 'biomass' or 'abundance'")
    output_type = f"{metric}DistribBySize"
    wide = _read_community_by_size(Path(output_dir), output_type, prefix)
    long = _community_long(wide)

    notes: list[str] = []
    tmax = float(long["time"].max())
    tmin = float(long["time"].min())
    if tmax - tmin + 1 < window_years:
        notes.append(f"run spans < {window_years} yr; used the available {tmax - tmin + 1:.0f}.")
    windowed = _window_by_time(long, "time", window_years)
    n_steps = int(windowed["time"].nunique())

    per_bin = windowed.groupby("size")["value"].mean().sort_index()
    edges = [float(x) for x in per_bin.index]
    values = [float(x) for x in per_bin.values]
    width = _infer_bin_width(edges)
    midpoints = [e + width / 2.0 for e in edges]

    if values and max(values) > 0:
        peak = midpoints[max(range(len(values)), key=lambda i: values[i])]
    else:
        peak = float("nan")

    slope, intercept, r2, n_fit = _fit_slope(midpoints, values, min_size_cm)
    if n_fit < 2:
        notes.append("fewer than 2 positive bins in the fit window; slope undefined.")
    lfi = _large_fish_indicator(edges, values, lfi_threshold_cm)
    mean_size = _mean_size(midpoints, values)

    return SizeSpectrum(
        metric=metric,
        bin_edges=edges,
        bin_midpoints=midpoints,
        values=values,
        peak_size_cm=float(peak),
        slope=slope,
        intercept=intercept,
        r_squared=r2,
        n_bins_fit=n_fit,
        min_size_cm=min_size_cm,
        lfi=lfi,
        lfi_threshold_cm=lfi_threshold_cm,
        mean_size_cm=mean_size,
        window_years=window_years,
        n_timesteps_used=n_steps,
        note=" ".join(notes),
    )


def spectrum_plot_df(spec: SizeSpectrum) -> pd.DataFrame:
    """Build the {size, abundance} df that plotting.make_size_spectrum_plot expects."""
    return pd.DataFrame({"size": spec.bin_midpoints, "abundance": spec.values})
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_size_spectrum.py -q`
Expected: PASS (all Task-1 + Task-2 tests; the EEC test exercises real data).
Run: `.venv/bin/ruff check osmose/size_spectrum.py tests/test_size_spectrum.py && .venv/bin/ruff format --check osmose/size_spectrum.py tests/test_size_spectrum.py`.

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add osmose/size_spectrum.py tests/test_size_spectrum.py
git -C /home/razinka/osmose/osmose-python commit -m "feat(size-spectrum): compute_size_spectrum + LFI/slope/mean-size indicators

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Time-series indicators + markdown report

**Files:**
- Modify: `osmose/size_spectrum.py`
- Test: `tests/test_size_spectrum.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_size_spectrum.py`:
```python
from osmose.size_spectrum import format_size_spectrum_report, size_spectrum_timeseries


def test_size_spectrum_timeseries_columns(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    rows = []
    for t in (1.0, 2.0):
        for e in (0.0, 10.0, 20.0, 40.0):
            rows.append((t, e, 1.0, 1.0))
    pd.DataFrame(rows, columns=["Time", "Size", "sp1", "sp2"]).to_csv(
        d / "osm_biomassDistribBySize_Simu0.csv", index=False
    )
    ts = size_spectrum_timeseries(d, prefix="osm", lfi_threshold_cm=40.0)
    assert list(ts.columns) == ["time", "slope", "lfi", "mean_size_cm"]
    assert sorted(ts["time"].unique()) == [1.0, 2.0]
    assert (ts["lfi"] == pytest.approx(1 / 4)).all()


def test_format_report_contains_key_fields(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    rows = [(1.0, e, 1.0, 1.0) for e in (0.0, 10.0, 20.0, 40.0)]
    pd.DataFrame(rows, columns=["Time", "Size", "sp1", "sp2"]).to_csv(
        d / "osm_biomassDistribBySize_Simu0.csv", index=False
    )
    spec = compute_size_spectrum(d, prefix="osm", window_years=1)
    md = format_size_spectrum_report(spec)
    assert "size spectrum" in md.lower()
    assert "Large-Fish Indicator" in md
    assert "trend/comparison" in md  # the honesty caveat
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_size_spectrum.py -k "timeseries or format_report" -q`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement**

Append to `osmose/size_spectrum.py`:
```python
def size_spectrum_timeseries(
    output_dir,
    *,
    metric: str = "biomass",
    prefix: str = "osm",
    lfi_threshold_cm: float = 40.0,
    min_size_cm: float | None = None,
) -> pd.DataFrame:
    """Per-timestep community slope / LFI / mean size (for trend lines)."""
    wide = _read_community_by_size(Path(output_dir), f"{metric}DistribBySize", prefix)
    long = _community_long(wide)
    out_rows = []
    for t, g in long.groupby("time"):
        per_bin = g.groupby("size")["value"].sum().sort_index()
        edges = [float(x) for x in per_bin.index]
        values = [float(x) for x in per_bin.values]
        width = _infer_bin_width(edges)
        midpoints = [e + width / 2.0 for e in edges]
        slope, _intercept, _r2, _n = _fit_slope(midpoints, values, min_size_cm)
        out_rows.append(
            {
                "time": float(t),
                "slope": slope,
                "lfi": _large_fish_indicator(edges, values, lfi_threshold_cm),
                "mean_size_cm": _mean_size(midpoints, values),
            }
        )
    return pd.DataFrame(out_rows, columns=["time", "slope", "lfi", "mean_size_cm"])


def format_size_spectrum_report(spec: SizeSpectrum) -> str:
    """Markdown summary of a SizeSpectrum (honest about the slope's interpretation)."""
    weighting = (
        "abundance-weighted mean length"
        if spec.metric == "abundance"
        else "biomass-weighted size centroid"
    )
    slope_txt = (
        f"{spec.slope:.3f} (intercept {spec.intercept:.3f}, R²={spec.r_squared:.3f}, "
        f"n_bins_fit={spec.n_bins_fit})"
        if spec.slope is not None
        else f"undefined (n_bins_fit={spec.n_bins_fit})"
    )
    cutoff = f"{spec.min_size_cm:.0f} cm" if spec.min_size_cm is not None else "none (all bins)"
    lines = [
        f"# OSMOSE community size spectrum — {spec.metric}",
        "",
        "A length–"
        + spec.metric
        + " spectrum over linear cm bins, reported for **trend/comparison** — "
        "not the canonical Sheldon normalized-by-body-mass exponent.",
        "",
        f"- Window: last {spec.window_years} yr ({spec.n_timesteps_used} timesteps)",
        f"- Spectrum slope: {slope_txt}",
        f"- Fit cutoff (min_size_cm): {cutoff}; peak (modal) bin midpoint: "
        f"{spec.peak_size_cm:.1f} cm",
        f"- Large-Fish Indicator (≥ {spec.lfi_threshold_cm:.0f} cm): {spec.lfi:.3f}",
        f"- Mean size ({weighting}): {spec.mean_size_cm:.2f} cm",
    ]
    if spec.note:
        lines += ["", f"_Note: {spec.note}_"]
    lines += ["", "| size (cm, midpoint) | value |", "|---|---|"]
    lines += [f"| {m:.1f} | {v:.6g} |" for m, v in zip(spec.bin_midpoints, spec.values)]
    return "\n".join(lines)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_size_spectrum.py -q`
Expected: PASS (all).
Run: `.venv/bin/ruff check osmose/size_spectrum.py tests/test_size_spectrum.py && .venv/bin/ruff format --check osmose/size_spectrum.py tests/test_size_spectrum.py`.

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add osmose/size_spectrum.py tests/test_size_spectrum.py
git -C /home/razinka/osmose/osmose-python commit -m "feat(size-spectrum): per-timestep indicator series + markdown report

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Trend chart (`make_size_indicator_timeseries`)

**Files:**
- Modify: `osmose/plotting.py`
- Test: `tests/test_size_spectrum.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_size_spectrum.py`:
```python
def test_plotting_reuse_and_new_chart():
    import plotly.graph_objects as go

    from osmose.plotting import make_size_indicator_timeseries, make_size_spectrum_plot

    pdf = pd.DataFrame({"size": [5.0, 15.0, 25.0], "abundance": [100.0, 30.0, 5.0]})
    fig = make_size_spectrum_plot(pdf)  # reused, unchanged
    assert isinstance(fig, go.Figure)

    ts = pd.DataFrame(
        {"time": [1.0, 2.0], "slope": [-2.0, -2.1], "lfi": [0.1, 0.12], "mean_size_cm": [20.0, 21.0]}
    )
    fig2 = make_size_indicator_timeseries(ts)
    assert isinstance(fig2, go.Figure)
    assert len(fig2.data) == 3  # slope, lfi, mean_size traces


def test_size_indicator_timeseries_empty():
    import plotly.graph_objects as go

    from osmose.plotting import make_size_indicator_timeseries

    fig = make_size_indicator_timeseries(pd.DataFrame(columns=["time", "slope", "lfi", "mean_size_cm"]))
    assert isinstance(fig, go.Figure)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_size_spectrum.py -k "plotting_reuse or indicator_timeseries_empty" -v`
Expected: FAIL (`ImportError: cannot import name 'make_size_indicator_timeseries'`).

- [ ] **Step 3: Implement**

Add to `osmose/plotting.py` (after `make_size_spectrum_plot`, which ends before `make_ci_timeseries` at ~:198). Use the existing `TEMPLATE` / `_empty_figure` / `_require_columns`:
```python
def make_size_indicator_timeseries(df: pd.DataFrame) -> go.Figure:
    """Community size-indicators over time: slope, LFI, mean size (3 traces)."""
    title = "Size Indicators Over Time"
    if df.empty:
        return _empty_figure(title)
    _require_columns(df, "time", "slope", "lfi", "mean_size_cm", context="make_size_indicator_timeseries")
    fig = go.Figure()
    for col in ("slope", "lfi", "mean_size_cm"):
        fig.add_trace(go.Scatter(x=df["time"], y=df[col], mode="lines", name=col))
    fig.update_layout(title=dict(text=title), xaxis_title="time", template=TEMPLATE)
    return fig
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_size_spectrum.py -q`
Expected: PASS (all).
Run: `.venv/bin/ruff check osmose/plotting.py tests/test_size_spectrum.py && .venv/bin/ruff format --check osmose/plotting.py tests/test_size_spectrum.py`.

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add osmose/plotting.py tests/test_size_spectrum.py
git -C /home/razinka/osmose/osmose-python commit -m "feat(plotting): make_size_indicator_timeseries trend chart

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: CLI (`scripts/compute_size_spectrum.py`)

**Files:**
- Create: `scripts/compute_size_spectrum.py`

- [ ] **Step 1: Write the CLI**

Create `scripts/compute_size_spectrum.py`:
```python
#!/usr/bin/env python3
"""Compute community size-spectrum diagnostics for a finished OSMOSE run.

Reads the {prefix}_{metric}DistribBySize community output and reports the size
spectrum, its log-log slope (length-biomass spectrum, trend/comparison only — not
the Sheldon exponent), the Large-Fish Indicator, mean size, and the peak bin.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/compute_size_spectrum.py \\
        --results-dir <dir> [--metric biomass|abundance] [--prefix osm] \\
        [--window-years 10] [--lfi-threshold-cm 40] [--min-size-cm N] \\
        [--report out.md] [--json out.json] [--plot out_prefix]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--metric", type=str, default="biomass", choices=["biomass", "abundance"])
    p.add_argument("--prefix", type=str, default="osm")
    p.add_argument("--window-years", type=int, default=10)
    p.add_argument("--lfi-threshold-cm", type=float, default=40.0)
    p.add_argument("--min-size-cm", type=float, default=None)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--plot", type=str, default=None)
    args = p.parse_args(argv)
    if args.window_years < 1:
        p.error("--window-years must be >= 1")

    from osmose import size_spectrum as ss

    try:
        spec = ss.compute_size_spectrum(
            args.results_dir,
            metric=args.metric,
            prefix=args.prefix,
            window_years=args.window_years,
            lfi_threshold_cm=args.lfi_threshold_cm,
            min_size_cm=args.min_size_cm,
        )
    except (FileNotFoundError, __import__("pandas").errors.EmptyDataError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    report = ss.format_size_spectrum_report(spec)
    if args.report:
        args.report.write_text(report)
    print(report)
    if args.json:
        args.json.write_text(json.dumps(asdict(spec), indent=2))
    if args.plot:
        from osmose import plotting

        plotting.make_size_spectrum_plot(ss.spectrum_plot_df(spec)).write_html(
            f"{args.plot}_size_spectrum.html"
        )
        ts = ss.size_spectrum_timeseries(
            args.results_dir,
            metric=args.metric,
            prefix=args.prefix,
            lfi_threshold_cm=args.lfi_threshold_cm,
            min_size_cm=args.min_size_cm,
        )
        plotting.make_size_indicator_timeseries(ts).write_html(f"{args.plot}_size_indicators.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-run the CLI on real EEC data**

Run: `PYTHONPATH=/home/razinka/osmose/osmose-python .venv/bin/python scripts/compute_size_spectrum.py --results-dir data/eec_full/output --prefix eec --window-years 10`
Expected: prints a markdown report with a negative slope, LFI ≈ 0.07, mean ≈ 20.8 cm. Exit 0.

Run (missing-output path): `PYTHONPATH=/home/razinka/osmose/osmose-python .venv/bin/python scripts/compute_size_spectrum.py --results-dir data/baltic/output --prefix baltic`
Expected: stderr "error: ..." (0-byte file → EmptyDataError handled) and exit 1.

- [ ] **Step 3: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add scripts/compute_size_spectrum.py
git -C /home/razinka/osmose/osmose-python commit -m "feat(size-spectrum): compute_size_spectrum.py CLI

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Docs + full verification + lint

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: CHANGELOG note**

In `CHANGELOG.md`, under `## [Unreleased]`, add an `### Added` entry (create the `### Added`
subsection if absent, matching the file's Keep-a-Changelog style):
```markdown
- **analysis (size spectrum):** community size-spectrum diagnostics from `*DistribBySize`
  output — `osmose.size_spectrum.compute_size_spectrum` (spectrum curve, log-log slope with a
  `min_size_cm` cutoff, Large-Fish Indicator, mean size, peak bin) + `size_spectrum_timeseries`,
  a `scripts/compute_size_spectrum.py` CLI, and a `make_size_indicator_timeseries` trend chart.
  A length–biomass spectrum for trend/comparison (not the Sheldon exponent). Validated on the
  EEC config; works on any run that emits by-size output.
```

- [ ] **Step 2: Full verification**

Run: `.venv/bin/python -m pytest tests/test_size_spectrum.py -v` (report count; all pass).
Run: `.venv/bin/python -m pytest tests/ -k "size_spectrum or analysis or plotting" -q` (report pass/fail; classify any failure pre-existing vs caused — if unsure, say so).
Run: `.venv/bin/ruff check osmose/ tests/ && .venv/bin/ruff format --check osmose/ tests/` (clean on touched files; if a file YOU touched is flagged by format, run `.venv/bin/ruff format <file>` and re-test; if untouched files are flagged, leave + note).

- [ ] **Step 3: Commit + finish**

```bash
git -C /home/razinka/osmose/osmose-python add CHANGELOG.md
git -C /home/razinka/osmose/osmose-python commit -m "docs(changelog): community size-spectrum diagnostics

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

Then use superpowers:requesting-code-review then superpowers:finishing-a-development-branch.

---

## Self-Review (plan author)

**Spec coverage:** reader-direct of `*DistribBySize` (no OsmoseResults change) → T1
(`_read_community_by_size`); reshape/sum-species/window-by-Time → T1 (`_community_long`,
`_window_by_time`, `_infer_bin_width`); `SizeSpectrum` + compute + reuse `size_spectrum_slope` +
`min_size_cm` cutoff + LFI + mean size + peak → T2; per-timestep series + honest report (slope
caveat, mean-size weighting label) → T3; reuse `make_size_spectrum_plot` + new
`make_size_indicator_timeseries` → T4; CLI mirroring compare_runs (exit 1 on missing/empty,
`--plot` prefix, asdict JSON) → T5; EEC validation (real-data test + CLI smoke) → T2/T5;
EmptyDataError handling → T1 (test) + T5 (CLI catch); docs → T6; out-of-scope (Baltic re-run,
Sheldon-by-weight, UI tab, per-guild) → not in plan, per spec. ✅

**Placeholder scan:** no TBD/TODO; every code step has complete code; commands have expected
output. The `__import__("pandas").errors.EmptyDataError` in the CLI is deliberate (avoids an
extra top-level import in a scripts/ file that's lint-exempt anyway) — it is concrete, not a
placeholder. ✅

**Type consistency:** `compute_size_spectrum(output_dir, *, metric, prefix, window_years,
lfi_threshold_cm, min_size_cm) -> SizeSpectrum` used identically in T2 tests, T5 CLI; `SizeSpectrum`
fields referenced in T3 report + T5 `asdict` match the T2 dataclass; `spectrum_plot_df(spec) ->
df[size,abundance]` feeds `make_size_spectrum_plot` (T4/T5); `size_spectrum_timeseries(...) ->
df[time,slope,lfi,mean_size_cm]` feeds `make_size_indicator_timeseries` (T3→T4/T5); `_fit_slope`
returns `(slope|None, intercept|None, r2|None, n_fit)` consistently in T2/T3. The reused
`size_spectrum_slope` contract (df cols `size`,`abundance`, raises `ValueError` <2) is wrapped in
`_fit_slope`. ✅
