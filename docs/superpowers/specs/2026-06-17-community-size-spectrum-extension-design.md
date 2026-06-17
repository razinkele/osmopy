# Community size-spectrum extension — Design

**Date:** 2026-06-17
**Status:** Approved (brainstorming), pending implementation plan

## Goal

Extend the existing community diagnostics with the **canonical Sheldon (body-mass) normalized
biomass spectrum** and a suite of **community-level ecosystem-state indicators**. Today
`osmose/size_spectrum.py` computes only a *length*-biomass/abundance spectrum over linear cm bins
and explicitly disclaims being "the canonical Sheldon normalized-by-body-mass exponent". This adds
the real thing plus Mean Trophic Level / Marine Trophic Index, community totals, size diversity, and
the Warwick ABC W-statistic.

## Background — what already exists

- `osmose/size_spectrum.py` — `compute_size_spectrum`, `size_spectrum_timeseries`,
  `format_size_spectrum_report`. Reads the wide `{prefix}_{metric}DistribBySize*.csv` community file
  (columns `Time, Size, <species…>`), sums species per `(Time, Size)`, windows by trailing years,
  and derives the length spectrum, its log-log slope, the Large-Fish Indicator (LFI), mean size, and
  the modal bin. **`Size` is a LENGTH (cm) axis.**
- `osmose/analysis.py::size_spectrum_slope(df[size, abundance]) -> (slope, intercept, r2)` — log-log
  OLS fit, raises `ValueError` on < 2 positive points. **Reused** for the NBSS fit.
- `osmose/results.py::OsmoseResults(output_dir, prefix="osm")` — `.biomass(species=None)` and
  `.abundance(species=None)` return per-species 1D time series (`Time` + one column per species);
  `meanTL` is exposed via `_read_species_output("meanTL", …)` and the `"trophic"` accessor.
- Config is a flat `dict[str, str]`. Relevant keys: `simulation.nspecies`, `species.name.sp{i}`,
  `species.length2weight.condition.factor.sp{i}` (= **a**), `species.length2weight.allometric.power.sp{i}`
  (= **b**) — i.e. `W = a · L^b`.
- Diagnostics UI: `ui/pages/results.py` registers a `"size_spectrum"` rtype and renders it via
  `osmose.plotting.make_size_spectrum_plot`. The page holds the loaded config in session.

## Architecture

New module **`osmose/community_metrics.py`** (not an extension of `size_spectrum.py`, which is
already cohesive and reads a single output; the new metrics read different outputs and would bloat
it). It **imports and reuses** `size_spectrum._read_community_by_size`, `_window_by_time`,
`_infer_bin_width`, and `analysis.size_spectrum_slope`, and constructs `OsmoseResults` for per-species
series. Three independently-testable compute units + a thin orchestrator + a markdown formatter.

### Unit 1 — Sheldon mass spectrum + spectrum-derived metrics

```
compute_sheldon_spectrum(
    output_dir, config: dict[str, str], *,
    metric: str = "biomass", prefix: str = "osm",
    window_years: int = 10, n_octaves_min: int = 2,
) -> SheldonSpectrum
```

Algorithm:
1. Read the wide `{metric}DistribBySize` file; window by trailing `window_years`. **Do NOT sum
   species** — keep per-species columns.
2. Build a species→(a, b) map from `config` (iterate `i` in `range(simulation.nspecies)`; key by
   `species.name.sp{i}`). A column whose species is absent from the config, or has a missing /
   non-positive `a` or `b`, is **dropped with a recorded note**.
3. For each species column, convert its length-bin midpoints to mass `W = a · L^b`, and accumulate
   that column's mean-over-window value into **equal log₂ (octave) mass bins** (bin index
   `floor(log2(W / W_ref))`, `W_ref` = smallest positive mass). Octave binning is the Sheldon
   convention.
4. **NBSS** = biomass per bin divided by the bin's mass width (linear width of the octave). Fit
   `log10(NBSS)` vs `log10(mass-bin-midpoint)` via `size_spectrum_slope` → slope, intercept, R².
   **Canonical expectation:** a *normalized-biomass* NBSS slope ≈ **−1** (and a *normalized-abundance*
   NBSS slope ≈ **−2**); the un-normalized biomass-per-octave is what is ≈ flat (slope ≈ 0). A flatter
   (less negative) slope than canonical indicates relative loss of large individuals (fishing-down).
   These are loose reference values for an exploited fish community, not a strict cross-ecosystem
   Sheldon continuum.
5. **Size diversity** = Shannon evenness `H/ln(S_bins)` over the **raw per-octave summed biomass
   shares** (the `binned[k]` biomass values *before* dividing by bin width — NOT the width-normalized
   NBSS density, which would make the metric depend on the arbitrary `w_ref` octave alignment).
6. **Community totals** are sourced from `OsmoseResults` per-species window-mean series (NOT the
   single `{metric}DistribBySize` file, which carries only one metric): **total biomass** = Σ over
   species of window-mean biomass, **total abundance** = Σ over species of window-mean abundance,
   **mean individual body mass** = total biomass / total abundance (units: biomass-unit per
   individual). This keeps mean body mass consistent regardless of `metric` and reuses the same
   per-species series ABC reads.

`SheldonSpectrum` (frozen dataclass): `metric`, `mass_bin_edges`, `mass_bin_midpoints`,
`nbss_values`, `slope`, `intercept`, `r_squared`, `n_bins_fit`, `size_diversity`,
`total_biomass`, `total_abundance`, `mean_body_mass`, `window_years`, `n_timesteps_used`,
`dropped_species: list[str]`, `note: str`.

### Unit 2 — Trophic indicators

```
compute_trophic_indicators(
    output_dir, *, prefix: str = "osm", window_years: int = 10,
    mti_tl_cutoff: float = 3.25,
) -> TrophicIndicators
```

- Read per-species `meanTL` and per-species `biomass`; window by trailing years; take each
  species' window-mean TL and window-mean biomass.
- **Mean Trophic Level (MTL)** = biomass-weighted mean of species' mean TL.
- **Marine Trophic Index (MTI)** = biomass-weighted mean TL over only species whose mean TL
  ≥ `mti_tl_cutoff` (Pauly & Watson's threshold; default 3.25). NB: this is a **biomass-weighted
  standing-stock analogue** of the canonical *catch-based* Pauly & Watson MTI; the formatter and
  docstring must label it as such so it is not mistaken for the fisheries landings index.
- `TrophicIndicators` dataclass: `mtl`, `mti`, `mti_tl_cutoff`, `n_species`,
  `n_species_above_cutoff`, `window_years`, `note`.

### Unit 3 — Abundance-Biomass Comparison (ABC) / W-statistic

```
compute_abc(output_dir, *, prefix: str = "osm", window_years: int = 10) -> ABCResult
```

- Per-species window-mean total biomass `B_i` and abundance `A_i` (from `OsmoseResults`).
- Rank species **separately** by biomass and by abundance (descending); build cumulative %
  dominance curves `k = 1…S` for each.
- **W-statistic** = `Σ_i (B_i − A_i) / (50 · (S − 1))` where `B_i`, `A_i` are the cumulative %
  dominance values at rank `i` (Warwick 1986). W > 0 ⇒ biomass-dominated (undisturbed);
  W < 0 ⇒ abundance-dominated (disturbed).
- `ABCResult` dataclass: `w_statistic`, `ranks: list[int]`, `cum_biomass_pct: list[float]`,
  `cum_abundance_pct: list[float]`, `n_species`, `window_years`, `note`.

### Orchestrator + formatter

```
community_report(
    output_dir, config: dict[str, str] | None = None, *,
    prefix: str = "osm", window_years: int = 10, metric: str = "biomass",
) -> CommunityDiagnostics
format_community_report(diag: CommunityDiagnostics) -> str   # markdown
```

`CommunityDiagnostics` bundles `sheldon: SheldonSpectrum | None`, `trophic: TrophicIndicators | None`,
`abc: ABCResult | None`, and a top-level `notes: list[str]`. Each field is `None` when its inputs are
unavailable (see degradation). The markdown formatter renders whichever sections are present,
honest about units and interpretation (mirrors `format_size_spectrum_report`'s tone).

### Plot helpers (in `osmose/plotting.py`)

- `make_sheldon_spectrum_plot(spec: SheldonSpectrum)` — log-log NBSS scatter + fitted line.
- `make_abc_plot(abc: ABCResult)` — the two cumulative dominance curves vs species rank.

## Data flow

`output_dir (+ config)` → readers (`_read_community_by_size`, `OsmoseResults`) → 3 compute units →
`CommunityDiagnostics` bundle → (a) `format_community_report` markdown, (b) plot DataFrames/figs.

## UI surface

Extend the existing Diagnostics **"Size Spectrum"** entry into a **"Community Diagnostics"** group on
`ui/pages/results.py`:
- Keep the existing **Length spectrum** plot.
- Add a **Sheldon (mass) spectrum** log-log plot (new rtype `"sheldon_spectrum"` →
  `make_sheldon_spectrum_plot`).
- Add an **ABC dominance-curve** plot (new rtype `"abc_curve"` → `make_abc_plot`).
- Add a **metrics summary panel** rendering `format_community_report` (MTL, MTI, W, size
  diversity, totals, plus any degradation notes).
The page passes the session config (for `a,b`) into `community_report`. When no config is loaded the
mass-dependent panels show the graceful note instead of a chart.

## Error handling / graceful degradation

Each unit fails soft and records a note rather than raising past the orchestrator:
- **`config is None` or no usable `a,b`** → `sheldon = None` (skip mass spectrum, size diversity,
  totals); length metrics, trophic, and ABC still computed. Top-level note explains.
- **A species with missing/non-positive `a` or `b`** → dropped from the mass spectrum, listed in
  `SheldonSpectrum.dropped_species`.
- **`meanTL` output absent** → `trophic = None` with a note.
- **`< window_years` of data** → use the available span and note (existing `size_spectrum` pattern).
- **`S < 2` species** → W undefined; `abc.w_statistic = nan`, note. NBSS fit with `< 2` positive
  bins → slope `None`, note.
- **Missing `{metric}DistribBySize` file** → `_read_community_by_size` raises `FileNotFoundError`;
  the orchestrator catches it, sets `sheldon = None`, and records a note (the UI already renders
  reader errors as friendly messages).

## Testing

Per-unit unit tests on synthetic fixtures with **known** answers:
- **Sheldon:** a constructed wide `DistribBySize` + known `a,b` such that masses fall in known
  octaves; assert bin assignment, and that a deliberately power-law mass distribution recovers the
  expected NBSS slope within tolerance. Assert size diversity (uniform shares → evenness 1.0) and
  totals/mean-mass arithmetic.
- **Trophic:** synthetic `meanTL` + `biomass` → hand-computed MTL and MTI (with a species below the
  cutoff excluded from MTI).
- **ABC:** an even community (W ≈ 0), a biomass-dominated community (W > 0), an abundance-dominated
  community (W < 0).
- **Degradation:** `config=None`; species with zero `b`; missing `meanTL`; `S < 2`; missing by-size
  file. Each asserts the corresponding field is `None`/`nan` and a note is present — no exception.
- **Real-data smoke** (guarded by `tests/_data_guards.require_eec_output`): run `community_report`
  against `data/eec_full/output` and assert the NBSS slope and W land in plausible ranges and the
  report renders.

Plot helpers get light tests (figure builds, has the expected number of traces) consistent with the
existing `plotting` tests.

## Out of scope (YAGNI)

- Per-timestep time series of the new metrics (the existing `size_spectrum_timeseries` covers length
  trends; mass-spectrum trends can be a later addition).
- Cross-run / scenario comparison UI (the Scenario Diff page is a separate, already-shipped surface).
- Java-engine-specific output handling beyond what the shared `DistribBySize`/`meanTL` readers
  already do.
