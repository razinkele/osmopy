# Result delta-tracking — Design

**Date:** 2026-06-03
**Status:** Approved direction (brainstormed; reconnaissance-grounded).
**Precedent:** PR #46 ICES validator + the F/M diagnostics feature (lib + CLI + plot).

## Motivation

OSMOSE can already diff two runs' **configs** (`scenarios.py`, `history.py`, the UI "Compare Runs"
tab), but nothing diffs their **outputs**. After a config tweak — the workflow underneath this
whole session's calibration/diagnostics work — a user must eyeball biomass curves to see what
moved. This feature computes **per-species output deltas** between a baseline and a variant run
and ranks species by how much they changed, so the effect of a change is instantly legible.

## Verified context (reconnaissance)

- **No output-delta exists.** `scenarios.py::ScenarioManager.compare` and
  `history.py::RunHistory.compare_runs` are CONFIG-key diffs; `plotting.py::make_run_comparison`
  is a grouped bar over stored scalar summaries (`RunRecord.summary`), not freshly-loaded runs.
- **Accessors** (`osmose/results.py`): `biomass()`, `yield_biomass()`, `abundance()`. **Shape
  verified by execution (NOT the docstring, which is wrong):** the EEC disk output is **WIDE** —
  `Time` + one column per species + a constant `species="all"` artifact column. `biomass(species=
  "cod")` returns **0 rows** (the `species=` filter matches the constant "all" column, not a real
  per-species row). So per-species values are the **columns**, and the delta normalizer must read
  the wide frame and take per-species column means — NOT use `biomass(species=...)` and NOT expect
  a `value` column. (A long-form `time, species, value` shape may also occur for some outputs;
  the normalizer detects and handles both.) This is the load-bearing detail the design hinges on.
- **`osmose/analysis.py` already exists** as the home for run-level ecological indicators
  (`summary_table`, `shannon_diversity`, `mean_tl_catch`, `size_spectrum_slope`) — `run_delta`
  belongs here, same module/imports/pattern.
- **`reporting.py::report_summary_table`** already computes per-species `biomass_mean/std`; the
  same trailing-window mean idea is reused (do NOT depend on it — analysis.py computes its own to
  support metric switching).
- **Two runs** load as two `OsmoseResults(output_dir, prefix="osm", strict=False)` instances.
- Spatial/per-cell output is xarray via `spatial_biomass(filename)`, gated on
  `output.spatial.enabled` (usually false) — **out of scope** (see Deferred).

## Architecture (library + CLI + plot)

### `osmose/analysis.py` (extend)

```python
@dataclass(frozen=True)
class SpeciesDelta:
    species: str
    baseline_mean: float
    variant_mean: float
    abs_delta: float            # variant_mean - baseline_mean
    pct_delta: float | None     # abs_delta / baseline_mean; None when baseline_mean == 0
    from_zero: bool             # baseline_mean == 0 and variant_mean > 0 (recovered/new)
```

- `run_delta(baseline, variant, *, metric="biomass", window_years=10, top_n=None) -> list[SpeciesDelta]`
  - `metric ∈ {"biomass","yield","abundance"}` → maps to `results.biomass` / `results.yield_biomass`
    / `results.abundance`.
  - For each species present in EITHER run: compute the trailing-`window_years` mean of `metric`
    in each run (a private `_species_window_mean(results, metric, species, window_years)` reusing
    the long-form accessor + trailing-window logic; species absent from a run → mean 0.0).
  - `abs_delta = variant_mean - baseline_mean`; `pct_delta = abs_delta/baseline_mean` if
    `baseline_mean != 0` else `None`; `from_zero = baseline_mean == 0 and variant_mean > 0`.
  - Sort by sort key = `abs(pct_delta)` when defined, else `+inf` (from-zero species rank at top as
    "biggest relative change"); ties broken by `abs(abs_delta)`. Return the top-`top_n` (all if None).
  - The species set is the UNION of both runs' species (so a species that appeared/vanished is
    captured); a species missing from a run contributes mean 0.0 there.
- `format_delta_report(deltas, *, metric="biomass", window_years=10) -> str` — markdown table
  (species | baseline | variant | Δ | Δ% | note), `Δ%` shown as "—(from 0)" when `from_zero`,
  sorted as above, with a one-line header naming metric + window.

### `osmose/plotting.py` (extend)

- `make_run_delta_chart(deltas, *, metric="biomass") -> go.Figure` — horizontal diverging bar of
  `pct_delta` per species (species on y, sorted by magnitude; positive=green, negative=red);
  from-zero species shown with a distinct marker/annotation (no finite bar). "osmose" template.
  Distinct from `make_run_comparison` (which is grouped scalars over `RunRecord`s).

### `scripts/compare_runs.py` (new CLI)

Args: `--baseline <dir>`, `--variant <dir>`, `--prefix` (default osm; or `--baseline-prefix`/
`--variant-prefix` if they differ — keep simple: one `--prefix` applied to both, plus optional
overrides), `--metric` (biomass|yield|abundance, default biomass), `--window-years` (default 10),
`--top-n` (default None=all), `--report`, `--json`, `--plot`. Loads both runs
`OsmoseResults(dir, prefix=..., strict=False)`, calls `run_delta`, prints the markdown table,
optionally writes JSON + the delta chart HTML.

### `tests/test_analysis_delta.py` (new)

Synthetic two-run fixtures via a `_FakeResults` stub (matching the `biomass()/yield_biomass()/
abundance()` long-form return: `time, species, <value>`):
- known per-species series → assert `baseline_mean`, `variant_mean`, `abs_delta`, `pct_delta`,
  and the RANKING order (biggest |Δ%| first).
- `metric="yield"` / `"abundance"` route to the right accessor.
- `top_n` truncates to the N biggest movers.
- **zero-baseline:** a species with baseline 0, variant > 0 → `pct_delta is None`, `from_zero True`,
  ranks at the top; report renders "— (from 0)" not inf.
- species present in only ONE run → mean 0 in the other, captured in the union.
- `format_delta_report` renders without error and includes the from-zero note.
- `make_run_delta_chart` builds (smoke: a Figure with one bar per finite-Δ% species).

## Deferred / out of scope (YAGNI)

- **Per-period (per-year) delta ranking** ("which year moved most") — a clean follow-on; v1 is
  per-species over a window.
- **Per-cell spatial deltas** — needs `output.spatial.enabled=true` (usually off) + xarray
  arithmetic; deferred.
- **Per-age/size-bin deltas** — deferred.
- **UI "Compare Runs" tab extension** — lib + CLI + plot first; the tab (`ui/pages/results.py`) is
  the natural later home (recon mapped the exact insertion point). Not in v1.

## Honest limitations

- `pct_delta` is undefined for a zero-baseline species (extinct→recovered); reported as `from_zero`
  with the absolute delta, not infinity.
- Delta is a windowed-mean comparison; it does not test significance (single run per side, no
  multi-seed band) — it answers "what moved", not "is the move beyond noise". (A multi-seed band
  is a possible follow-on, mirroring the FR diagnostic.)

## Delivery

Single PR: `run_delta` + `format_delta_report` in `osmose/analysis.py`, `make_run_delta_chart` in
`osmose/plotting.py`, `scripts/compare_runs.py`, tests, a short doc note. No engine changes, no
calibration runs.
