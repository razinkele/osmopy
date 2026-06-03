# Fisheries stock-status diagnostics — Design

**Date:** 2026-06-03
**Status:** Approved direction (brainstormed; reconnaissance-grounded).
**Precedent:** PR #46 ICES output validator (`osmose/validation/ices.py` + `scripts/validate_outputs_vs_ices.py`).

## Motivation

OSMOSE produces per-species biomass, fishing yield, and cause-resolved mortality, but no
standard single-species **stock-status diagnostics** — the F/M, B/Bmsy, F/Fmsy, and
Kobe-quadrant view fisheries scientists use to read a stock at a glance. PR #46 already loads
ICES reference points (`Fmsy`, `Blim`, `Bpa`, `MSY Btrigger`) into `IcesSnapshot.reference_points`
but leaves them **unused**. This feature wires that existing data into a diagnostics module:
no engine changes, no calibration runs, no new data — it computes from a finished run.

## Verified context (reconnaissance)

- **Biomass:** `results.biomass(species=None)` → long-form `time, species, value` (tonnes), `osmose/results.py:339`.
- **Fishing mortality F:** `results.mortality(species)` returns the `mortalityRate-{sp}` CSV with an `F` column (per life stage, per timestep, instantaneous). Annual F = sum of the `F` column (Recruits stage) over `n_dt_per_year`. (`osmose/engine/output.py:216`.) Fallback: `yield_biomass / (biomass·0.5)` Baranov approximation.
- **Natural mortality M:** same CSV — `Mpred + Mstarv + Madd` columns, summed per year.
- **Reference points:** `IcesSnapshot.reference_points[stock]` already holds `fmsy`, `blim`, `bpa`, `msy_btrigger` (loaded by `osmose/validation/ices.py:108`). Populated for sprat (`fmsy=0.34`, `msy_btrigger=541000`), herring sub-stocks, western cod (`fmsy=0.26`), western flounder. **NULL** for eastern Baltic cod (index-unit, not tonnes) and the coastal species (perch/pikeperch/smelt/stickleback — no ICES tonnes assessment).
- **Reusable helpers in `ices.py`:** `load_snapshot()`, `IcesSnapshot`, `model_biomass_window_mean(results, species, window_years)`, the species→stock mapping, `format_markdown_report()`.
- **Plotting:** pure Plotly via `osmose/plotting.py` + the "osmose" template (`osmose/plotly_theme.py`). No matplotlib in library code.
- **OSMOSE has no native MSY/Fmsy** (it's an IBM multispecies model) → B/Bmsy and F/Fmsy must use the external ICES reference points.

## Architecture (mirrors PR #46: library + CLI)

### `osmose/validation/fisheries.py` (new library module)

```python
@dataclass(frozen=True)
class FisheriesStatus:
    species: str
    biomass_t: float            # trailing-window mean model biomass (tonnes)
    fishing_mortality: float    # trailing-window mean annual F (yr^-1)
    natural_mortality: float    # trailing-window mean annual M = Mpred+Mstarv+Madd (yr^-1)
    f_over_m: float             # F / M  (always computable)
    # Reference-point ratios — None when no ICES reference point exists for the species:
    b_over_bmsy: float | None   # biomass / msy_btrigger (Bmsy proxy)
    f_over_fmsy: float | None   # F / Fmsy
    kobe_quadrant: str | None   # "green" | "orange" | "yellow" | "red" | None
    has_reference_points: bool
    note: str                   # e.g. "no ICES reference point (coastal species)"
```

- `compute_fisheries_status(results, snapshot, window_years=10) -> list[FisheriesStatus]`
  - For each model species: B via `model_biomass_window_mean`; F via new `model_fishing_mortality`; M via new `model_natural_mortality`; F/M always.
  - Map species → ICES stock(s) via `ices.py`'s existing mapping. **Reference-point rule (explicit, to avoid mixed-unit ambiguity):** compute B/Bmsy + F/Fmsy + the Kobe quadrant ONLY when *every* contributing ICES sub-stock has tonnes-unit `msy_btrigger` AND non-null `fmsy`. Then Bmsy proxy = sum of sub-stock `msy_btrigger`, Fmsy = biomass-weighted (or simple, if single) mean of sub-stock `fmsy`. If ANY contributing sub-stock is index-unit or has null `fmsy` (e.g. model **cod** = western-cod tonnes + eastern-cod index → mixed → excluded), set `has_reference_points=False`, ratios `None`, and a `note` naming why. This keeps every reference-point ratio defensible rather than silently mixing units.
- `model_fishing_mortality(results, species, window_years) -> float` — read `results.mortality(species)`, take the Recruits-stage `F` column, sum per year, mean over the trailing window.
- `model_natural_mortality(results, species, window_years) -> float` — same CSV, `Mpred+Mstarv+Madd` summed per year, windowed mean.
- `kobe_quadrant(b_over_bmsy, f_over_fmsy) -> str` — green (B≥1, F≤1), red (B<1, F>1), orange (B≥1, F>1 = overfishing but not overfished), yellow (B<1, F≤1 = overfished but not overfishing). Standard Kobe convention.
- `format_fisheries_report(statuses) -> str` — markdown table (species | B | F | M | F/M | B/Bmsy | F/Fmsy | quadrant | note).

**Bmsy proxy rationale (documented):** ICES does not publish Bmsy directly; `MSY Btrigger` is its operational biomass reference (the trigger below which advice reduces F). B/`MSY Btrigger` is the standard available proxy and is labelled as a proxy in output.

### `osmose/plotting.py` (extend)

- `make_kobe_plot(statuses) -> go.Figure` — scatter of B/Bmsy (x) vs F/Fmsy (y) for species with reference points; four shaded quadrants (green SE, red NW, orange NE, yellow SW), reference lines at x=1/y=1, point labels = species. Uses the "osmose" template.
- `make_fm_ratio_bars(statuses) -> go.Figure` — F/M bar chart for all species (F/M is always available), with a reference line at F/M=1.

### `scripts/compute_fisheries_diagnostics.py` (new CLI — mirrors `validate_outputs_vs_ices.py`)

Args: `--results-dir`, `--snapshots-dir`, `--prefix`, `--window` (default 10), `--report` (markdown to stdout/file), `--json` (FisheriesStatus list), `--plot` (write Kobe + F/M figures as HTML and/or PNG). Loads results + snapshot, calls `compute_fisheries_status`, prints the markdown table, optionally writes JSON + plots.

### `tests/test_validation_fisheries.py` (new)

- `kobe_quadrant` classification for the 4 quadrants + the on-the-line edge cases.
- `compute_fisheries_status` on a **synthetic** `OsmoseResults`-like fixture + a synthetic `IcesSnapshot`: assert F (summed from a known F-column series), M, F/M, B/Bmsy, F/Fmsy values, and quadrant.
- **Coverage-gap path:** a species with no reference point → `has_reference_points=False`, ratios `None`, quadrant `None`, F/M still computed, explanatory note present.
- Markdown report renders without error and includes the gap note.
- (If a tiny real Baltic run fixture is cheap, one integration assertion that the CLI path runs; else keep unit-level with synthetic fixtures.)

## Scope / YAGNI

- **Shiny page DEFERRED** — library + CLI + plot first, exactly like PR #46. A `ui/pages/fisheries_status.py` is a trivial follow-on once the lib exists; not in this scope.
- **No model-derived Fmsy** (yield-per-recruit / surplus-production estimation) — out of scope; we use ICES external reference points only.
- **No new reference-point data fetching** — the Baltic snapshots already carry the points. (A future non-Baltic config would need its own snapshots; not this scope.)

## Honest limitations (carried into output + docs)

- Reference-point ratios cover only the ~4 ICES tonnes-unit Baltic stocks; the coastal species and eastern cod show F/M + biomass only, explicitly labelled. This is the same percid/eastern-cod coverage gap documented elsewhere — surfaced honestly, not hidden.
- `B/Bmsy` uses `MSY Btrigger` as a proxy for `Bmsy`; labelled as such.
- Model F is the realized annual F from the mortality output; for rate-input fishing it equals the configured rate, for catch-input it is derived — either way read from the recorded `F` column.

## Delivery

Single PR: lib + plotting + CLI + tests + a short doc note (in the config/feature reference) pointing at the CLI. No engine changes; no calibration runs.
