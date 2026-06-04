# Community size-spectrum diagnostics — Design

**Date:** 2026-06-04
**Status:** Approved direction (brainstormed; codebase-grounded). New feature (science extension).

## Motivation

OSMOSE emits per-species size-distribution outputs but the repo has no community-level
size-structure diagnostics. Community size spectra and the Large-Fish Indicator (LFI) are
recognised ecosystem/food-web indicators (Sheldon spectrum; OSPAR/HELCOM LFI), and OSMOSE is a
size-structured model — so these are a natural, scientifically substantive analysis layer. This
is a pure-analysis feature mirroring the shipped `osmose/analysis.py` (run-delta) and
`osmose/validation/fisheries.py` patterns: read existing output, compute indicators, report.

## Verified context (audit)

- `OsmoseResults` (`osmose/results.py`) already exposes the needed readers:
  `biomass_by_size(species=None)` → `_read_2d_output("biomassBySize")` and
  `abundance_by_size` → `abundanceBySize`. `_read_2d_output` returns **long-form**
  `time, species, bin, value` (it melts the wide on-disk file).
- On-disk format (real): `data/eec_full/output/Indicators/eec_biomassDistribBySize_Simu0.csv`
  has a 1-line title preamble, then header `"Time","Size",<species...>`, then rows keyed by
  `(Time, Size)` where `Size` is the **lower edge in cm** of an equal-width bin (10 cm in the
  shipped configs; preamble: "Class i designates interval [i,i+1["). The size-bin scheme is
  **common across all species** (one `Size` column), so summing species per `(Time, Size)`
  yields the community spectrum. EEC: 21 bins (0–200 cm), 14 species, 70 timesteps.
- **Data caveat:** `data/baltic/output/Indicators/baltic_*DistribBySize_Simu0.csv` are **empty
  (0 bytes)** — the Baltic committed run did not emit by-size output. So **EEC is the test/demo
  substrate**; using this on Baltic requires enabling the by-size output flag and re-running
  (documented limitation, not a blocker — the feature is config-agnostic).
- Pattern to mirror: `osmose/analysis.py` (`run_delta`, `_trailing_window`,
  `_per_species_window_mean`, `format_delta_report`) + `osmose/plotting.py::make_run_delta_chart`
  + `scripts/compare_runs.py`. The "window by the `Time` column in years, never by row count
  (sub-annual-cadence-safe)" lesson from delta-tracking applies here too.
- CI lint runs `ruff check` + `ruff format --check` on `osmose/ ui/ tests/` (NOT `scripts/`).

## Architecture

A focused module + plotting helpers + a CLI. No engine change, no UI (deferred follow-on).

### 1. `osmose/size_spectrum.py`

```python
@dataclass(frozen=True)
class SizeSpectrum:
    metric: str                      # "biomass" | "abundance"
    bin_edges: list[float]           # lower edges (cm), ascending
    bin_midpoints: list[float]       # edge + width/2
    values: list[float]              # community value per bin (summed over species, window-mean)
    slope: float | None              # OLS slope of log10(value) vs log10(midpoint); None if <2 positive bins
    intercept: float | None
    r_squared: float | None
    lfi: float                       # biomass/abundance fraction in bins with edge >= threshold
    lfi_threshold_cm: float
    mean_size_cm: float              # value-weighted mean of midpoints
    window_years: int
    n_timesteps_used: int
    note: str                        # warnings (short window, dropped bins, etc.)
```

Functions:
- `compute_size_spectrum(results, *, metric="biomass", window_years=10, lfi_threshold_cm=40.0) -> SizeSpectrum`
  — read the long-form `(time, species, bin, value)`; window by `Time` (trailing `window_years`
  in Time-units, robust to sub-annual cadence — reuse the `_trailing_window` approach); sum
  across species per `(time, bin)`; mean over the window per bin; derive slope / LFI / mean size.
- `size_spectrum_timeseries(results, *, metric="biomass", lfi_threshold_cm=40.0) -> pd.DataFrame`
  — per-timestep `time, slope, lfi, mean_size_cm` (community-summed per step) for trend lines.
- `format_size_spectrum_report(spec: SizeSpectrum) -> str` — markdown summary (metric, window,
  slope±/R², LFI, mean size, the per-bin table, and the `note`).

Private helpers:
- `_infer_bin_width(edges)` — median consecutive diff of sorted unique edges (handles the common
  equal-width case; falls back gracefully if irregular).
- `_fit_spectrum_slope(midpoints, values) -> (slope|None, intercept|None, r2|None)` — OLS of
  `log10(value)` vs `log10(midpoint)` over bins with `value > 0` and `midpoint > 0`; `None` if
  fewer than 2 such bins.
- `_large_fish_indicator(edges, values, threshold) -> float` — `sum(values[edge>=threshold]) /
  sum(values)`; `0.0` if total is 0.
- `_mean_size(midpoints, values) -> float` — `sum(values*midpoints)/sum(values)`; `nan`-safe.

### 2. `osmose/plotting.py`

- `make_size_spectrum_chart(spec: SizeSpectrum)` — log-log scatter of community value vs
  midpoint + the fitted slope line (when slope is not None); axis labels from `metric`. Uses the
  project Plotly theme like the existing `make_run_delta_chart`.
- `make_size_indicator_timeseries(df: pd.DataFrame)` — LFI / slope / mean-size over time
  (the `size_spectrum_timeseries` output).

### 3. `scripts/compute_size_spectrum.py`

CLI mirroring `scripts/compare_runs.py`: `--results-dir` (required), `--metric {biomass,abundance}`
(default biomass), `--window-years` (default 10), `--lfi-threshold-cm` (default 40.0), `--prefix`
(default osm), `--report <path.md>`, `--json <path.json>`, `--chart <path.html>`. Exit codes
mirror the existing CLIs: 2 for a missing/invalid results dir, 1 when the by-size output is
absent/empty (with a message naming the output flag), 0 otherwise.

## Data flow

1. `OsmoseResults(results_dir, strict=False)` → `biomass_by_size()` / `abundance_by_size()`
   (long `time, species, bin, value`).
2. Window by `Time` (last `window_years`) → group by `bin`, sum over species, mean over the
   window → community value per bin.
3. `_fit_spectrum_slope`, `_large_fish_indicator`, `_mean_size` → `SizeSpectrum`.
4. CLI/markdown/JSON/chart out.

## Key computations (precise)

- **Bin midpoint** = `edge + width/2`, `width = _infer_bin_width(edges)`.
- **Slope** = OLS of `log10(value)` on `log10(midpoint)` over bins with `value > 0`; report
  `slope`, `intercept`, `r_squared`. (A more negative slope = relatively fewer large fish.)
- **LFI** = `Σ value[edge ≥ lfi_threshold_cm] / Σ value`. Uses the bin **lower edge** so "≥ 40 cm"
  means the `[40,50)` bin and above. Default threshold 40 cm (OSPAR North Sea convention; the CLI
  flag lets Baltic studies pick a lower cut).
- **Mean size** = `Σ value·midpoint / Σ value` (metric-weighted: biomass-weighted where the
  biomass concentrates, abundance-weighted mean individual size for the abundance metric).

## Error handling

- Missing/empty by-size output → `FileNotFoundError`-style error: "No `biomassBySize` output in
  <dir> — enable the by-size output flag and re-run" (mirrors the empty-Baltic case). The CLI
  maps this to exit 1.
- `window_years` longer than the run → use the available span; record in `note`.
- Fewer than 2 positive bins → `slope/intercept/r_squared = None`; `note` says why.
- Threshold above all bins → `lfi = 0.0` (+ note); total value 0 → `lfi = 0.0`, `mean_size = nan`.
- Irregular bin widths → `_infer_bin_width` uses the median diff; `note` flags irregularity.

## Testing (`tests/test_size_spectrum.py`)

- **Synthetic wide CSV** (`Time, Size, sp1, sp2`) written via the same fixture technique as the
  delta tests (avoid the title-preamble crash — generate with a clean header, or via
  `OsmoseResults.from_outputs`). Construct a **known power law** across bins so the fitted slope
  is a known value (assert within tolerance); construct biomass split around 40 cm so **LFI** is
  known; assert **mean size** and **community aggregation across species**.
- **Window-by-Time**: a sub-annual-cadence frame (multiple rows per Time-year) → assert the
  window selects by Time-years, not row count.
- **Edge cases**: single positive bin → `slope is None`; threshold above all bins → `lfi == 0`;
  empty/missing output → the expected error.
- **Real EEC fixture**: load `data/eec_full/output` → assert `slope < 0`, `0 ≤ lfi ≤ 1`,
  `mean_size_cm` within the bin range, and that biomass-vs-abundance bases differ.
- **Plotting smoke**: `make_size_spectrum_chart` / `make_size_indicator_timeseries` render
  without error and contain the expected traces.
- (Pure functions → fully unit-tested; no UI in scope.)

## Scope / YAGNI

- **In:** the 4 indicators (spectrum curve, slope+intercept+R², LFI, mean size), biomass &
  abundance bases, trailing window, the module + 2 plot helpers + the CLI + tests.
- **Out:** per-guild / functional-group spectra; mean trophic level (`meanTLBySize`); a Shiny UI
  tab (deferred follow-on, as the delta UI was); size→weight allometric (normalised-by-weight
  Sheldon) conversion — we use biomass and size-in-cm directly; enabling Baltic's by-size output
  (separate, documented); any engine/results.py change (the readers already exist).

## Honest limitations

- Validated/demoed on **EEC** — Baltic's committed by-size output is empty, so applying this to
  Baltic needs the by-size flag enabled + a re-run.
- It is a **length–biomass spectrum** (cm bins), not a normalised-abundance-by-body-mass Sheldon
  spectrum; the slope is interpretable for trend/comparison, not as the canonical −1 Sheldon
  exponent. Documented in the report header.
- Assumes (and is exact for) **equal-width bins** (shipped configs use 10 cm); irregular widths
  are handled via median-width inference with a `note`, but not rigorously density-normalised.

## Delivery

Single PR: `osmose/size_spectrum.py`, `osmose/plotting.py` (2 helpers),
`scripts/compute_size_spectrum.py`, `tests/test_size_spectrum.py`, a docs/CHANGELOG note. No
engine changes, no calibration runs.
