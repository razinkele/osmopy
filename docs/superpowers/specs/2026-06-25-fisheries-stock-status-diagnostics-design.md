# Fisheries stock-status diagnostics (Kobe / B-ref / F/Fmsy) — design

> Status: design (awaiting review) · 2026-06-25
> Unblocks the follow-up explicitly deferred by the 2026-06-03 fisheries-diagnostics
> spec (`docs/superpowers/specs/2026-06-03-fisheries-diagnostics-design.md`), whose blocker
> was reference-point coverage. Resolved here by **user-supplied reference points with ICES
> auto-fill**, so the diagnostics work on any config (Baltic / EEC / BoB), not just sprat.

## 1. Why

The Python layer can already read per-species biomass, realized fishing mortality (from the
`mortalityRate` CSV), and catch (yield biomass + the new `yieldN`). What it cannot do is place
a stock on the standard **Kobe** quadrant (`B/Bref` vs `F/Fmsy`) — the headline fisheries
management diagnostic. The 2026-06-03 spec built the F/M (fishing-vs-natural mortality)
library but **deferred** Kobe / `B/Bmsy` / `F/Fmsy` because ICES reference points alone cover
only one fully-eligible Baltic stock (sprat) and carry two methodological traps (below). This
feature adds the stock-status layer and gives fisheries diagnostics a proper Shiny home,
including the existing-but-never-surfaced F/M bars.

## 2. Scope decisions (from brainstorming)

- **Reference points: user-supplied + ICES auto-fill.** A per-species reference table, editable
  in the UI, pre-filled from ICES snapshots where a tonnes-unit stock exists, user-entered
  otherwise. Works on any config; provenance is tracked and shown.
- **Surface: a new dedicated `Fisheries` Shiny page** that bundles the Kobe plot, `B/Bref` and
  `F/Fmsy` time-series, **and** the existing `compute_mortality_balance` F/M bars (first time
  surfaced).

## 3. Methodological basis (approved — these are the scientifically-honest calls)

- **B-axis reference (`Bref`).** ICES does **not** publish `Bmsy` directly; it publishes MSY
  Btrigger / Blim / Bpa, and **MSY Btrigger ≈ Bpa sits *below* Bmsy** — using it as "Bmsy"
  paints overfished stocks green. So: `Bref` = **user-supplied `Bmsy` when provided**; otherwise
  fall back to **MSY Btrigger, and the axis is labelled `B / MSY-Btrigger` (NOT `B/Bmsy`)** with
  a caveat. A per-species `b_ref_kind ∈ {"bmsy", "msy_btrigger"}` records which is in use and
  drives the axis label.
- **`F/Fmsy`.** `Fmsy` always comes from the real ICES (or user) value. The model's realized F
  is **OSMOSE exploited-stage fishing mortality** (the `('F', <exploited stage>)` annualised rate
  the F/M library already computes), which is comparable to but **not identical to** the ICES
  age-ranged `Fbar` that `Fmsy` is defined on — labelled as such in the UI.
- **B = total biomass (v1).** ICES `Bmsy` is SSB-based; the engine outputs total biomass cheaply
  while SSB would need mature-biomass-by-age filtering. v1 uses **total biomass with a clear
  `B = total biomass` label**; SSB-based B is a deferred refinement (§9).
- **Kobe quadrants:** green `B/Bref ≥ 1 ∧ F/Fmsy ≤ 1`; red `B/Bref < 1 ∧ F/Fmsy > 1`; the two
  off-diagonal quadrants yellow/orange (overfished-but-recovering vs healthy-but-overfishing).

## 4. Components (each isolated, testable)

1. **`osmose/fisheries_reference.py`** (new) — reference-point resolution.
   - `@dataclass ReferencePoint`: `species: str`, `fmsy: float | None`, `bmsy: float | None`,
     `msy_btrigger: float | None`, `blim: float | None`, `b_ref_kind: str`, `source: str`
     (`"ices:<stock>"` | `"user"` | `"mixed"`). Property `b_ref` returns `bmsy` if set else
     `msy_btrigger`; `b_ref_label` returns `"Bmsy"` or `"MSY-Btrigger"`.
   - `load_reference_points(ref_dir, species_list, *, ices_snapshot_dir=None) ->
     dict[str, ReferencePoint]`. `ref_dir` is the directory holding the sidecar JSON — the UI
     passes the **selected run's output dir**, so save/load is self-contained relative to the run
     being viewed. Reads optional `ref_dir/fisheries_reference_points.json`
     (`{species: {fmsy?, bmsy?, blim?}}`); for each species with no/partial values and an ICES
     tonnes-unit stock, auto-fills `fmsy`/`msy_btrigger`/`blim` via the **existing**
     `osmose.validation.ices.load_snapshot` (`reference_points[stock]`, `float()`-coerced — they
     are JSON strings). Picks the ICES stock per species from
     `manifest["model_species_to_ices_stocks"]`; if multiple, the first tonnes-unit one (reuse
     the validator's unit logic). Species with no reference point at all → `ReferencePoint` with
     all-None numerics (excluded from Kobe downstream).
   - `save_reference_points(ref_dir, refs)` — writes the JSON back to `ref_dir` (the run output
     dir) for the UI "save" action.
   - The page resolves `ices_snapshot_dir` to the bundled `data/<ecosystem>/reference/ices_snapshots`
     when it exists (Baltic ships one), else `None` (auto-fill becomes a no-op — EEC/BoB rely on
     user entry).

2. **`osmose/validation/stock_status.py`** (new) — pure computation (no I/O beyond an
   `OsmoseResults`).
   - `@dataclass StockStatus`: `species`, `years: list[int]`, `b_over_bref: list[float]`,
     `f_over_fmsy: list[float]`, `b_ref_label: str`, `latest_quadrant: str`, `caveats: list[str]`.
   - `compute_stock_status(results: OsmoseResults, refs: dict[str, ReferencePoint], *,
     steps_per_year: int, species_list=None) -> list[StockStatus]`. Per species: realized annual
     F from the `mortalityRate` CSV via the existing `fisheries.read_mortality` +
     `annual_rate`-style aggregation on the exploited stage; annual total biomass from
     `results.biomass(species)`; `B/Bref` and `F/Fmsy` per year (None where the ref is missing).
     A species with no `fmsy` AND no `b_ref` is returned with empty series + a caveat (so the UI
     can list it as "no reference points"). Reuses `fisheries.py` helpers — does NOT duplicate the
     F extraction.

3. **`osmose/plotting.py`** (extend) — `make_kobe_plot(statuses, *, year=None) -> go.Figure`
   (Plotly: 4 colored quadrant rectangles, one marker per species at the selected year, a faint
   per-species trajectory line across years, latest year emphasised, axis labels driven by
   `b_ref_label`); `make_ratio_timeseries(statuses, which) -> go.Figure` for `B/Bref` & `F/Fmsy`
   trends; `make_fm_ratio_bars(balances) -> go.Figure` for the existing F/M (per the 2026-06-03
   spec's unbuilt `make_fm_ratio_bars`). Use the project "osmose" Plotly template.

4. **`ui/pages/fisheries.py`** (new Shiny page) — `fisheries_ui()` + `fisheries_server()`.
   - Run selection reusing the existing `_safe_output_dir` validation pattern (factor the shared
     helper out of `ui/pages/results.py` if not already shared).
   - An **editable reference-point table** (`bmsy`, `fmsy` per species), pre-filled by
     `load_reference_points`, with a provenance column and a "Save reference points" action →
     `save_reference_points`. Empty cells → that species is excluded from Kobe with a visible note.
   - Plots: Kobe (with a year slider), `B/Bref` & `F/Fmsy` time-series, F/M bars. A methodology
     caveat panel (B = total biomass; exploited-stage F vs Fbar; `Bref` kind per species).
   - Register in `app.py` (nav entry + server wiring), following the existing page pattern.

## 5. Data flow

run dir → `OsmoseResults` → (`results.biomass`, `mortalityRate` CSV) ⨉
`load_reference_points(config_dir, species, ices_snapshot_dir)` →
`compute_stock_status` → per-species `StockStatus` (B/Bref, F/Fmsy, quadrant) →
`make_kobe_plot` / `make_ratio_timeseries` / `make_fm_ratio_bars` in the Fisheries page.

## 6. Reference-point file format

`<run-output-dir>/fisheries_reference_points.json` (optional; absent ⇒ pure ICES auto-fill / empty):
```json
{
  "sprat":  { "fmsy": 0.34, "bmsy": null, "blim": 459000 },
  "cod":    { "fmsy": 0.31, "bmsy": 120000 }
}
```
Only `fmsy`/`bmsy`/`blim` are user-editable; `msy_btrigger`, `b_ref_kind`, `source` are derived.
`bmsy` present ⇒ `b_ref_kind="bmsy"`; else if `msy_btrigger` available ⇒ `"msy_btrigger"`.

## 7. Error handling / edge cases

- **No reference points for a species** → excluded from Kobe; listed in a "no reference points"
  note (not an error).
- **`Fmsy = 0` or `Bref = 0`** → ratio undefined → None (guard divide-by-zero), species shown as
  unplottable with a caveat.
- **No ICES snapshot for the config** (EEC/BoB) → auto-fill is a no-op; everything comes from the
  user file / inline edits. The page still works.
- **Run lacks a `mortalityRate` CSV for a species** (F unavailable) → F/Fmsy None, B/Bref still
  shown; caveat noted.
- **Reference-point JSON malformed** → surfaced as a clear validation message, page degrades to
  inline entry.

## 8. Testing

- `fisheries_reference`: ICES auto-fill resolves `fmsy`/`msy_btrigger` for sprat from a snapshot
  fixture (`float()` coercion of the JSON strings); user file overrides ICES; `b_ref_kind` flips
  to `"bmsy"` when the user supplies `bmsy`; species with no stock → all-None. Round-trip
  save/load.
- `stock_status`: on a synthetic `OsmoseResults` (or a tiny fixture run), `B/Bref` and `F/Fmsy`
  computed correctly against hand-figured numbers; the four Kobe quadrants classified correctly
  at chosen points; divide-by-zero guards; a no-reference species returns empty series + caveat;
  F extraction matches the existing F/M library on shared inputs (no drift).
- `plotting`: `make_kobe_plot` returns a Figure with the expected quadrant shapes + axis label
  switching on `b_ref_label` (`Bmsy` vs `MSY-Btrigger`); markers at the right coordinates.
- UI: page renders for a run with refs (Kobe populated) and for a no-ICES config (inline-only);
  the existing Shiny test pattern.
- No engine/dynamics change → the EEC/BoB parity suites are untouched (this is read-only
  analysis over existing outputs).

## 9. Out of scope (deferred refinements)

- **SSB-based B** (mature-biomass-by-age) for `B/Bref` — v1 uses total biomass with a label.
- **Age-ranged `Fbar`** matching the ICES `f_age_range` — v1 uses exploited-stage F with a caveat.
- **Mixed-unit stock aggregation** (cod/herring index+tonnes) — the reference picks a single
  tonnes-unit stock per species (or the user supplies the value); no index/tonnes blending.
- **B/Bmsy targets inside the engine config schema** — kept in the sidecar JSON, not the schema.
- Per-fishery (vs per-species) stock status; uncertainty envelopes on the Kobe point.
