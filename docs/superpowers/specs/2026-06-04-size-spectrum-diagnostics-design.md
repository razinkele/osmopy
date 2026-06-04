# Community size-spectrum diagnostics — Design

**Date:** 2026-06-04
**Status:** Approved direction (brainstormed; **revised after in-loop review** — round 1 found a
blocker + methodology gaps + existing code to reuse; all folded in). New feature (science
extension).

## Motivation

OSMOSE emits per-(Time,Size) community size-distribution output but the repo has no
community-level size-structure indicators. Community size spectra and the Large-Fish Indicator
(LFI) are recognised ecosystem/food-web indicators, and OSMOSE is a size-structured model — so
these are a natural, substantive analysis layer. Pure-analysis feature mirroring the shipped
`osmose/analysis.py` (run-delta) and `osmose/validation/fisheries.py` patterns: read existing
output, compute indicators, report. **Heavily reuses existing helpers** (see below) rather than
re-implementing.

## Verified context (audit — corrected against EXECUTED reader behaviour)

Round-1 review caught that the obvious reader path does **not** work on the real files. The
corrected facts:

- The community by-size output is `{prefix}_biomassDistribBySize_Simu0.csv` /
  `{prefix}_abundanceDistribBySize_Simu0.csv`, in the **`Indicators/` subdir**. Layout: a 1-line
  title preamble, then header `"Time","Size",<species...>`, then rows keyed by `(Time, Size)`
  with **species as wide columns** and `Size` = bin **lower edge in cm**. EEC: 21 bins
  (0–200 cm, equal width 10 cm), 14 species, 70 timesteps. Summing the species columns per
  `(Time, Size)` gives the community spectrum.
- **`OsmoseResults.biomass_by_size()` / `abundance_by_size()` do NOT read these files** — they
  glob `*biomassBySize*` (not `*biomassDistribBySize*`), only search root + `Mortality/Bioen`
  (not `Indicators/`), and `_read_2d_output` assumes `Time + bin-columns` (one file per species),
  so on a `Time,Size,<species>` community file it melts species names into the `bin` column and
  emits `species="all"`. **So this feature must read the `*DistribBySize` file directly**, not via
  those accessors. It reuses `osmose.results._read_output_csv` (which correctly strips the
  preamble) + `pathlib` globbing; this is a small read helper inside the new module — **no change
  to `OsmoseResults`** (keeps blast radius minimal; the existing accessors are left alone).
  **Coupling note:** `_read_output_csv` is a private (`_`-prefixed) cross-module helper and
  `results.py` exposes no public preamble-safe free reader; we depend on it deliberately (the only
  alternative is duplicating `_detect_preamble_lines` + the read). Acceptable given no public
  equivalent; if `results.py` refactors its reader this import must be updated.
- **Empty/0-byte file** (the committed Baltic case) → `_read_output_csv` raises
  `pandas.errors.EmptyDataError`, NOT `FileNotFoundError`. Must catch both (+ a present-but-empty
  frame).
- **`Size` after melt/read is fine as numeric** when we read the wide file directly with
  `_read_output_csv` (the `Size` column parses as float); we still coerce defensively.
- **Reuse, do NOT duplicate** — these already exist and the round-1 review flagged the
  collisions:
  - `osmose/analysis.py::size_spectrum_slope(df) -> (slope, intercept, r_squared)` — log-log OLS
    over `df` columns `size`, `abundance` (generic "value"); **raises `ValueError` if < 2 positive
    pairs**. We call it (wrapping the raise → `None`) instead of re-implementing the fit.
  - `osmose/plotting.py::make_size_spectrum_plot(df) -> go.Figure` — log-log scatter + regression
    line + slope annotation over `df` columns `size`, `abundance`. We reuse it for the spectrum
    chart (feeding a community-spectrum df), NOT a new duplicate chart.
- `osmose/analysis.py::_trailing_window(df, time_col, window_years)` windows by the Time column in
  years (the delta lesson). It is **private and lacks a `window_years < 1` guard** (that guard
  lives in `_per_species_window_mean`). To avoid a cross-module private import, the new module
  **replicates** a tiny `_window_by_time(df, time_col, window_years)` WITH the `>= 1` guard.
- Baltic by-size output is empty only because the committed run didn't persist it — the flag
  `output.biomass.bysize.enabled;true` is **already on** (`data/baltic/baltic_param-output.csv`).
  So Baltic usage needs a **re-run**, not a config edit. **Out of scope** (consistent with how
  run-delta / fisheries diagnostics are BYO-run-output; validated here on the clean committed EEC
  data). The current Baltic config's overshoots (cod ×17–48, percid ×100) would also make a
  committed Baltic spectrum a misleading showcase — another reason to keep it EEC-validated.
- CI lint runs `ruff check` + `ruff format --check` on `osmose/ ui/ tests/` (NOT `scripts/`).

## Architecture

A focused module that reads the community file + reuses the existing slope/chart helpers, plus a
CLI. No engine change, no `OsmoseResults` change, no UI (deferred follow-on).

### 1. `osmose/size_spectrum.py`

```python
@dataclass(frozen=True)
class SizeSpectrum:
    metric: str                      # "biomass" | "abundance"
    bin_edges: list[float]           # lower edges (cm), ascending
    bin_midpoints: list[float]       # edge + width/2
    values: list[float]              # community value per bin (summed over species, window-mean)
    peak_size_cm: float              # midpoint of the modal (max-value) bin — helps choose min_size_cm
    slope: float | None              # via analysis.size_spectrum_slope on bins >= min_size_cm; None if <2 positive
    intercept: float | None
    r_squared: float | None
    n_bins_fit: int                  # bins actually used in the slope fit
    min_size_cm: float | None        # the fit cutoff applied (None = all positive bins)
    lfi: float                       # value fraction in bins with edge >= lfi_threshold_cm
    lfi_threshold_cm: float
    mean_size_cm: float              # value-weighted mean midpoint (weighting = `metric`; labeled in report)
    window_years: int
    n_timesteps_used: int
    note: str
```

Functions:
- `compute_size_spectrum(output_dir, *, metric="biomass", prefix="osm", window_years=10, lfi_threshold_cm=40.0, min_size_cm=None) -> SizeSpectrum`
  — locate + read the `{prefix}_{metric}DistribBySize*.csv` community file; reshape to long
  `(time, size, value)` by summing species columns per `(Time, Size)`; window by Time
  (`_window_by_time`); mean over the window per bin → community spectrum; derive peak / slope
  (reusing `size_spectrum_slope` after the `min_size_cm` filter) / LFI / mean size.
- `size_spectrum_timeseries(output_dir, *, metric="biomass", prefix="osm", lfi_threshold_cm=40.0, min_size_cm=None) -> pd.DataFrame`
  — per-timestep `time, slope, lfi, mean_size_cm` (community-summed per step) for trend lines.
- `format_size_spectrum_report(spec) -> str` — markdown: metric + weighting note, window, peak,
  slope±/R²/n_bins_fit (with the explicit caveat below), LFI, mean size, the per-bin table, `note`.

Private helpers:
- `_read_community_by_size(output_dir, output_type, prefix) -> pd.DataFrame` — rglob
  `{prefix}_{output_type}*.csv` under `output_dir`; read via `osmose.results._read_output_csv`
  (preamble-safe); return wide `Time, Size, <species>`. Raise a clear `FileNotFoundError` if no
  file; let/translate `EmptyDataError` and present-but-empty into the same "no by-size output"
  error.
- `_community_long(wide) -> pd.DataFrame` — melt species columns → `time, size, value`
  (sum is done by groupby downstream); coerce `size` to float.
- `_window_by_time(df, time_col, window_years)` — `window_years < 1` → raise `ValueError`;
  else `df[df[time_col] > df[time_col].max() - window_years]`.
- `_large_fish_indicator(edges, values, threshold) -> float` — `Σ values[edge>=threshold] /
  Σ values`; `0.0` if total 0.
- `_mean_size(midpoints, values) -> float` — `Σ values·midpoints / Σ values`; `nan` if total 0.
- `_infer_bin_width(edges) -> float` — median consecutive diff of sorted unique edges.
- For the slope: build a 2-col df `{"size": midpoints, "abundance": values}` (the generic
  contract `size_spectrum_slope` expects — "abundance" is just the value axis), filter to
  `size >= min_size_cm` when set, then `try: slope,intercept,r2 = size_spectrum_slope(df)` /
  `except ValueError: slope=intercept=r2=None`.

### 2. `osmose/plotting.py`

- **Reuse** `make_size_spectrum_plot(df[size, abundance])` for the spectrum chart (the CLI builds
  the `{size, abundance}` community df from a `SizeSpectrum` and passes it). No new spectrum chart.
- **Add** `make_size_indicator_timeseries(df) -> go.Figure` — LFI / slope / mean-size over time
  (the `size_spectrum_timeseries` output), following the `TEMPLATE`/`update_layout(template=...)`
  convention used by the existing charts. (This trend view is genuinely new; the spectrum curve is
  not.)

### 3. `scripts/compute_size_spectrum.py`

CLI mirroring `scripts/compare_runs.py` conventions (verified against the real script):
`argparse` with `RawDescriptionHelpFormatter` + `description=__doc__`; `--results-dir` (required),
`--metric {biomass,abundance}` (default biomass), `--prefix` (default `osm`), `--window-years`
(default 10), `--lfi-threshold-cm` (default 40.0), `--min-size-cm` (default None),
`--report <path.md>`, `--json` (via `dataclasses.asdict`), `--plot <prefix>` (writes
`{prefix}_size_spectrum.html` + `{prefix}_size_indicators.html` via `.write_html`, matching
compare_runs' `--plot` prefix semantics — NOT a `--chart <path>`). Invalid args → `parser.error`
(argparse exit 2). `main()` returns `0` on success and `1` with a stderr message when the by-size
output is absent/empty. (compare_runs itself only returns 0; the `1` for missing-output is a
deliberate small addition, not claimed as mirrored.)

## Data flow

1. `_read_community_by_size(dir, f"{metric}DistribBySize", prefix)` → wide `Time, Size, <species>`.
2. `_community_long` → `time, size, value` (species summed per (time,size) via groupby).
3. `_window_by_time` (last `window_years`) → groupby `size`, mean over window → community value
   per bin; `size`→midpoint.
4. peak / `size_spectrum_slope` (post `min_size_cm` filter) / `_large_fish_indicator` /
   `_mean_size` → `SizeSpectrum`.
5. CLI → markdown / JSON / `make_size_spectrum_plot` + `make_size_indicator_timeseries`.

## Key computations (precise + honest)

- **Bin midpoint** = `edge + width/2`, `width = _infer_bin_width(edges)`.
- **Slope** = log-log OLS of community value vs midpoint over bins with `value > 0` AND (if set)
  `midpoint >= min_size_cm`, via the existing `size_spectrum_slope`. **Caveat (from review):** the
  community spectrum is typically NON-monotonic — small bins (recruits / small pelagics) dominate
  and an all-bins fit reflects the recruitment peak, not adult structure (EEC: all-bins slope
  ≈ −1.9 R²0.67 vs above-10cm ≈ −2.8 R²0.84). So `min_size_cm` exists and the report names
  `peak_size_cm`, `n_bins_fit`, and R² so the user fits the descending limb consciously. This is a
  **length–biomass spectrum slope for trend/comparison only**, NOT the canonical Sheldon
  normalized-by-body-mass exponent. The report says so. We make **no** "biomass vs abundance slopes
  differ by ~1" claim — for length-binned raw spectra the gap reflects allometry (≈3 on EEC,
  B∝N·L³), not the Sheldon −1.
- **LFI** = `Σ value[edge ≥ lfi_threshold_cm] / Σ value` (biomass-fraction, the OSPAR definition).
  Default threshold 40 cm (OSPAR North Sea); CLI-tunable. Report notes LFI can be small on
  small-pelagic-dominated systems (EEC LFI@40 ≈ 0.07) — informative, not degenerate.
- **Mean size** = value-weighted mean of midpoints; the report **states the weighting**: for
  `metric="abundance"` it is the canonical abundance-weighted mean length; for `metric="biomass"`
  it is the biomass-weighted size centroid (labeled as such, not as "mean fish size").

## Error handling

- No `*DistribBySize` file found, or file empty (`EmptyDataError` / 0 rows) → clear error:
  "No `{metric}DistribBySize` output in <dir>; the run did not persist by-size output (enable +
  re-run)". CLI → exit 1.
- `window_years < 1` → `ValueError` (and CLI `parser.error`). `window_years` longer than the run →
  use the available span; `note`.
- < 2 positive bins after the `min_size_cm` filter → `slope/intercept/r_squared = None`,
  `n_bins_fit < 2`; `note`.
- Threshold above all bins → `lfi = 0.0` (+ note); total value 0 → `lfi = 0.0`, `mean_size = nan`.
- Irregular bin widths → `_infer_bin_width` median; `note` flags it. (Preamble text says 1-cm
  classes but the real increment is 10 cm — trust `_infer_bin_width`, not the preamble.)

## Testing (`tests/test_size_spectrum.py`)

- **Synthetic wide community CSV** `Time, Size, sp1, sp2` written with a **clean header (no title
  preamble)** so `_read_output_csv` (0 preamble lines) reads it — NOT the delta `biomass_wide_sample`
  fixture (that is a 1D file with no `Size` column). Construct a known power law so the fitted slope
  is known (assert within tol, reusing `size_spectrum_slope`); biomass split around 40 cm so LFI is
  known; assert community aggregation across species, peak, mean size.
- **Window-by-Time** on a sub-annual-cadence frame (multiple rows per Time-year) → asserts windowing
  by Time-years, not row count; `window_years < 1` raises.
- **min_size_cm** filter changes the fitted slope + `n_bins_fit` as expected (drops small bins).
- **Edge cases**: < 2 positive bins → `slope is None`; threshold above all bins → `lfi == 0`;
  missing file → error; **0-byte file** → the "no by-size output" error (asserts `EmptyDataError`
  is handled, not leaked).
- **Real EEC fixture**: `compute_size_spectrum("data/eec_full/output", prefix="eec")` → assert
  slope `< 0`, `0 ≤ lfi ≤ 1`, `mean_size_cm` within bin range, peak in a small bin, and that
  biomass vs abundance bases differ.
- **Plotting smoke**: `make_size_spectrum_plot` on a community df + `make_size_indicator_timeseries`
  render without error and contain the expected traces.

## Scope / YAGNI

- **In:** the 4 indicators (spectrum curve via reused chart, slope+intercept+R² via reused fit with
  a `min_size_cm` cutoff, LFI, mean size + peak), biomass & abundance bases, trailing window, the
  module + 1 new trend chart + the CLI + tests.
- **Out:** per-guild / functional-group spectra; mean trophic level; a Shiny UI tab (deferred
  follow-on); a normalized-by-body-mass Sheldon spectrum; **any `OsmoseResults`/engine change** (we
  read the community file directly); **Baltic re-run / committed Baltic by-size output** (BYO-run-
  output, flag already on, EEC validates correctness; documented).

## Honest limitations

- Validated on **EEC** committed data; using it on Baltic means pointing it at a Baltic re-run's
  output (flag already on). A Baltic spectrum on the current overshoot-heavy config would partly
  reflect calibration artifacts.
- It is a **length–biomass (or length–abundance) spectrum** over linear cm bins, reported for
  trend/comparison — not the canonical Sheldon exponent. The slope is sensitive to the small-bin
  cutoff; `min_size_cm` + `peak_size_cm` + `n_bins_fit`/R² are surfaced so the user fits
  deliberately.
- Assumes equal-width bins (shipped configs: 10 cm); irregular widths get median-width inference
  with a `note`, not rigorous density normalisation.

## Delivery

Single PR: `osmose/size_spectrum.py`, `osmose/plotting.py` (1 new trend chart; reuse the existing
spectrum chart), `scripts/compute_size_spectrum.py`, `tests/test_size_spectrum.py`, a
docs/CHANGELOG note. No engine changes, no `OsmoseResults` change, no calibration runs.
