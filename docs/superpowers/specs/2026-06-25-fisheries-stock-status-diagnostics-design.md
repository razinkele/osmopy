# Fisheries stock-status diagnostics (Kobe / B/Bref / F/Fmsy) — design

> Status: design (revised after in-loop multi-lens review) · 2026-06-25
> Unblocks the follow-up deferred by the 2026-06-03 fisheries-diagnostics spec
> (`docs/superpowers/specs/2026-06-03-fisheries-diagnostics-design.md`). The 2026-06-03 blocker
> was reference-point coverage *and* the methodological traps of a naïve Kobe. This revision
> resolves both: **user-supplied reference points with ICES auto-fill**, an **SSB-based B-axis**
> (matching ICES reference biomass), and **stock-set-summed** ICES references (not a single
> sub-stock). Verified against the bundled Baltic ICES fixtures.

## 1. Why

The Python layer can read per-species biomass, biomass-by-age, realized fishing mortality (from
the `mortalityRate` CSV), and catch. It cannot place a stock on the standard **Kobe** quadrant
(`B/Bref` vs `F/Fmsy`) — the headline fisheries-management diagnostic. The 2026-06-03 spec built
the F/M (fishing-vs-natural mortality) library but **deferred** Kobe / `B/Bref` / `F/Fmsy` for two
reasons: ICES reference points alone cover only one fully-eligible Baltic stock (sprat), and a
naïve Kobe carries unit traps (below). This feature adds a scientifically-honest stock-status
layer and gives fisheries diagnostics a proper Shiny home, including the existing F/M bars.

## 2. Scope decisions (from brainstorming + review)

- **Reference points: user-supplied + ICES auto-fill.** A per-species reference table, editable
  in the UI, pre-filled from ICES snapshots where tonnes-unit stocks exist, user-entered
  otherwise. Works on any config (Baltic / EEC / BoB); provenance is tracked and shown.
- **B-axis is SSB-based and stock-set-summed** (the rigorous option chosen at review): model B is
  **spawning-stock biomass** (mature-class biomass), and the ICES reference is the **sum of
  MSY-Btrigger across the species' full tonnes-unit stock set** — not a single sub-stock. This
  matches the maturity scope (SSB vs SSB) and the stock-set scope (whole species). A residual
  **spatial-domain caveat** remains (the modelled domain need not equal the union of ICES stock
  areas) and is surfaced, not silently ignored.
- **Surface: a new dedicated `Fisheries` Shiny page** bundling the Kobe plot, `B/Bref` & `F/Fmsy`
  time-series, and the **existing** F/M bars (first time surfaced).

## 3. Methodological basis (the scientifically-honest calls)

- **B = SSB (spawning-stock biomass), not total biomass.** ICES MSY-Btrigger / Bpa / Blim are all
  SSB-based (confirmed: every `*.reference_points.json` + assessment carries `ssb`; sprat
  `msy_btrigger = 541000` is an SSB level). Comparing model *total* biomass to an SSB reference is
  systematically too-green. So model B = **Σ mature-class biomass**, derived from
  `results.biomass_by_age(species)` keeping age classes `≥` the species `maturity_age` from the
  EngineConfig (fallback to `biomass_by_size` ≥ `maturity_size` if a config uses size-at-maturity;
  if neither is configured or the by-age/by-size output is absent in the run, the B-axis is
  **unavailable** for that species with a caveat — graceful, not an error).
- **B-axis reference (`Bref`).** ICES does **not** publish `Bmsy` directly (only MSY-Btrigger /
  Blim / Bpa). `Bref` = **user-supplied `Bmsy` when provided**; otherwise **Σ MSY-Btrigger over the
  species' tonnes-unit ICES stock set** (the same set `compare_outputs_to_ices` aggregates for its
  SSB envelope — reuse that selection). MSY-Btrigger ≈ Bpa sits *below* Bmsy, so when the fallback
  is in use the axis is labelled **`B / Σ MSY-Btrigger` (NOT `B/Bmsy`)** with a caveat that it
  understates the true Bmsy target. `b_ref_kind ∈ {"bmsy", "msy_btrigger_sum"}` drives the label.
- **`F/Fmsy`.** `Fmsy` comes from the real ICES (or user) value. The model's realized F is the
  **single exploited-stage** OSMOSE fishing mortality (the stage with `F > _FISHED_TOL`), NOT the
  cross-stage F-sum the F/M bars use — a multi-stage F-sum would over-state F/Fmsy. It is an
  annual instantaneous rate (matches Fmsy's basis) but is **not** the ICES age-ranged `Fbar` that
  `Fmsy` is defined on — labelled as such. For a multi-stock species, `Fmsy` is taken from the
  primary tonnes-unit stock with a caveat (Fmsy is not summable across stocks the way a biomass
  level is).
- **Kobe quadrants:** green `B/Bref ≥ 1 ∧ F/Fmsy ≤ 1`; red `B/Bref < 1 ∧ F/Fmsy > 1`; the two
  off-diagonal quadrants yellow/orange (overfished-but-rebuilding vs healthy-but-overfishing).
  A point needs **both** ratios defined; partial-reference species (only one axis) are excluded
  from the scatter but shown on the corresponding single-axis time-series.

## 4. Components (each isolated, testable)

1. **`osmose/fisheries_reference.py`** (new) — reference-point resolution.
   - `@dataclass ReferencePoint`: `species`, `fmsy: float | None`, `bmsy: float | None`,
     `msy_btrigger_sum: float | None`, `blim_sum: float | None`, `fmsy_stock: str | None`,
     `b_ref_kind: str` (`"bmsy"` | `"msy_btrigger_sum"` | `"none"`), `source: str`
     (`"ices:<stocks>"` | `"user"` | `"mixed"`), `caveats: list[str]`. Property `b_ref` returns
     `bmsy` if set else `msy_btrigger_sum`; `b_ref_label` returns `"Bmsy"` or `"Σ MSY-Btrigger"`.
   - `load_reference_points(ref_dir, species_list, *, ices_snapshot_dir=None) ->
     dict[str, ReferencePoint]`. `ref_dir` is an **ecosystem-scoped, stable** directory
     (`data/<ecosystem>/`), NOT a per-run output dir — see §6. Reads optional
     `ref_dir/fisheries_reference_points.json` (`{species: {fmsy?, bmsy?, blim?}}`). For each
     species, auto-fills from `osmose.validation.ices.load_snapshot` (`ices_snapshot_dir`):
     pick the species' **tonnes-unit** stocks via `manifest["model_species_to_ices_stocks"]` +
     the snapshot's per-stock `units` (the same tonnes/index split `compare_outputs_to_ices` uses);
     `msy_btrigger_sum` = Σ `float(reference_points[s]["msy_btrigger"])` over those stocks,
     `blim_sum` = Σ `blim`, `fmsy` = the primary stock's `float(fmsy)` (JSON strings → `float()`).
     User-file values override ICES. A species with no user value and no tonnes-unit stock →
     `b_ref_kind="none"` (excluded from the Kobe B-axis). Multi-stock species get a caveat
     (Bref summed over stocks; Fmsy from one stock).
   - `save_reference_points(ref_dir, refs)` — writes the user-editable fields back to
     `ref_dir/fisheries_reference_points.json`. Only `fmsy`/`bmsy`/`blim` round-trip; derived
     fields are recomputed on load.

2. **`osmose/validation/stock_status.py`** (new) — pure computation.
   - `@dataclass StockStatus`: `species`, `years: list[int]`, `b_over_bref: list[float | None]`,
     `f_over_fmsy: list[float | None]`, `b_ref_label: str`, `latest_quadrant: str | None`,
     `caveats: list[str]`.
   - `compute_stock_status(results, refs, config, *, species_list=None) -> list[StockStatus]`.
     Per species:
     - **F (per-year series):** reuse `fisheries.read_mortality` + `_STAGES` + the `F > _FISHED_TOL`
       exploited-stage rule, and a NEW `fisheries.annual_series(per_step, steps_per_year) ->
       np.ndarray` factored out of `annual_rate` (which currently collapses to a windowed scalar —
       `annual_rate` is refactored to call `annual_series` then window-mean, no behaviour change).
       F on the single exploited stage per year.
     - **B (SSB per-year series):** from `results.biomass_by_age(species)`, which the 2D reader
       returns **long-form** (`time, species, bin, value`); `bin` is the integer **age-class in
       years** (`"0","1",…`). SSB(t) = Σ `value` over bins `≥ floor(maturity_age_years)`, where
       `maturity_age_years = config.maturity_age_dt[sp] / config.n_dt_per_year`; reduce to one value
       per year by **year-mean** (matching `annual_series`'s per-year basis). (If a config uses
       size-at-maturity, use `biomass_by_size` with bins `≥ config.maturity_size[sp]`.)
     - `B/Bref` and `F/Fmsy` per year, intersecting the two integer-year indices; `None` where a
       reference or input is missing; divide-by-zero guarded (Bref/Fmsy `≤ 0 → None`).
       `latest_quadrant` from the most-recent year where both ratios are defined.
   - `steps_per_year` (= `config.n_dt_per_year`), `maturity_age_dt`, and species come from the
     `EngineConfig`; the output `prefix` comes from the page / `OsmoseResults.prefix` (it is **not**
     an `EngineConfig` field). `compute_stock_status` takes them as explicit arguments — it never
     infers `steps_per_year`.

3. **`osmose/plotting.py`** (extend) — add `make_kobe_plot(statuses, *, year=None) -> go.Figure`
   (4 coloured quadrant rectangles, one marker per species at the selected year, a faint per-species
   trajectory line, latest year emphasised, axis labels from `b_ref_label`) and
   `make_ratio_timeseries(statuses, which) -> go.Figure` (`"b"` / `"f"`). Both confirmed **absent**
   today. `make_fm_ratio_bars` **already exists** (`osmose/plotting.py:337`, tested) — the Fisheries
   page **reuses** it; no F/M plotting work. Use the project "osmose" Plotly template.

4. **`ui/pages/fisheries.py`** (new Shiny page) — `fisheries_ui()` + `fisheries_server()`.
   - Run selection reusing the `_safe_output_dir` validation pattern (factor the shared helper out
     of `ui/pages/results.py` if not already shared). The page obtains the run's `EngineConfig` from
     app state (for `steps_per_year` = `n_dt_per_year`, `maturity_age_dt`, species names) and the
     `prefix` from the selected `OsmoseResults`;
     if no config is in state, expose a `steps_per_year` input and derive species via
     `fisheries.discover_species(output_dir, prefix)` — SSB requires the config's maturity, so the
     B-axis is unavailable without it (caveat).
   - An **editable reference-point table** (`bmsy`, `fmsy` per species), pre-filled by
     `load_reference_points`, with a provenance column and a "Save reference points" action →
     `save_reference_points` (ecosystem-scoped path). Empty cells → that axis omitted, with a note.
   - Plots: Kobe (year slider), `B/Bref` & `F/Fmsy` time-series, F/M bars (reused). A methodology
     caveat panel (B = SSB; Bref kind per species; exploited-stage F vs Fbar; spatial-domain caveat).
   - Register in `app.py` (nav entry + server wiring), following the existing page pattern.

## 5. Data flow

run output dir → `OsmoseResults` → (`biomass_by_age` for SSB, `mortalityRate` CSV for exploited-stage
F) ⨉ `EngineConfig` (`n_dt_per_year`, `maturity_age_dt`, species) + `OsmoseResults.prefix` ⨉
`load_reference_points(ecosystem_dir, species, ices_snapshot_dir)` →
`compute_stock_status` → per-species `StockStatus` (B/Bref, F/Fmsy, quadrant) →
`make_kobe_plot` / `make_ratio_timeseries` / `make_fm_ratio_bars` on the Fisheries page.
(All module paths are `osmose.validation.fisheries`, `osmose.validation.stock_status`,
`osmose.fisheries_reference`, `osmose.plotting`.)

## 6. Reference-point storage

`data/<ecosystem>/fisheries_reference_points.json` — an **ecosystem-scoped, stable** location
(NOT a per-run output dir; runs use fresh `mkdtemp` dirs, so a run-dir sidecar would be lost on the
next run). The page derives `<ecosystem>` from the run's config/data dir; if it can't, it falls
back to a user-config dir keyed by config name. Format (only `fmsy`/`bmsy`/`blim` are user-editable;
the rest are derived on load):
```json
{
  "sprat": { "fmsy": 0.34, "bmsy": null, "blim": 459000 },
  "cod":   { "fmsy": 0.31, "bmsy": 120000 }
}
```
`bmsy` present ⇒ `b_ref_kind="bmsy"`; else if a tonnes-unit stock set exists ⇒ `"msy_btrigger_sum"`;
else `"none"`.

## 7. Error handling / edge cases

- **Reader shapes differ by dimensionality.** The **1D** `biomass(species)` returns WIDE-form
  (capital `Time` + per-species-name value columns, constant `species="all"`) — `biomass(species=
  "cod")` returns **0 rows** on a real run; select `df["cod"]` keyed on `Time`. The **2D**
  `biomass_by_age(species)` returns **long-form** (`time, species, bin, value`) via `_read_2d_output`
  — SSB uses this 2D long-form. Do **not** reuse `ices.model_biomass_window_mean`'s `df["value"]`
  logic against the 1D reader (its only tests use a long-form mock; it raises on a real 1D result).
- **By-age output absent / no maturity configured** → B-axis unavailable for that species; shown on
  the F/Fmsy axis only; caveat "SSB unavailable (enable biomass-by-age output)".
- **Partial reference** (Fmsy only, or Bref only) → excluded from the Kobe scatter (needs both
  axes), still plotted on the available single-axis time-series; caveat distinguishes
  "Fmsy only" / "Bref only" from "no reference points".
- **`Fmsy ≤ 0` or `Bref ≤ 0`** → ratio `None` (divide-by-zero guard); species unplottable, caveat.
- **No ICES snapshot for the config** (EEC/BoB) → auto-fill is a no-op; values come from the user
  file / inline edits. Page still works.
- **Multi-stock species** (cod = west+east, herring = 3 stocks) → `Bref` summed over the tonnes-unit
  set; `Fmsy` from the primary stock; both carry a caveat. Never silently pick one sub-stock's
  Btrigger as the whole-species Bref.
- **Reference-point JSON malformed** → clear validation message; page degrades to inline entry.
- **Residual spatial-domain caveat** (model domain vs union of ICES stock areas) always shown when
  any ICES-derived reference is in use.

## 8. Testing

- `fisheries_reference`: ICES auto-fill sums `msy_btrigger`/`blim` over a species' tonnes-unit
  stock set from a snapshot fixture (`float()` coercion of JSON strings); single-stock sprat vs
  multi-stock herring (3 stocks summed); user `bmsy` flips `b_ref_kind` to `"bmsy"`; species with no
  tonnes stock → `"none"`. Round-trip save/load (user fields only).
- `stock_status`: on a fixture run, `annual_series` returns the per-year array (and `annual_rate`
  still returns its windowed scalar — no behaviour change); SSB = Σ mature age-class biomass on the
  **long-form** (`time, species, bin, value`) by-age output (hand-figured); `B/Bref` & `F/Fmsy`
  correct; the four Kobe quadrants
  classified; divide-by-zero guards; a no-B-axis species returns F-only series + caveat; F uses the
  single exploited stage (not the cross-stage sum).
- `plotting`: `make_kobe_plot` returns a Figure with the quadrant shapes + axis label switching on
  `b_ref_label` (`Bmsy` vs `Σ MSY-Btrigger`); markers at the right coordinates.
- UI: page renders for a Baltic run (ICES-filled) and a no-ICES config (inline-only); the
  ecosystem-scoped sidecar **survives a re-run** (saved under `data/<ecosystem>/`, not the run dir).
- No engine/dynamics change → EEC/BoB parity suites untouched (read-only analysis over outputs).

## 9. Out of scope (deferred refinements)

- **Age-ranged `Fbar`** matching each stock's `f_age_range` — v1 uses single-exploited-stage F with
  a caveat.
- **Spatial-domain reconciliation** (clipping the modelled domain to ICES stock areas) — v1 carries
  the caveat instead.
- **B/Bref reference points in the engine config schema** — kept in the ecosystem sidecar JSON.
- **Per-fishery (vs per-species) stock status**; uncertainty envelopes on the Kobe point;
  index-unit stock handling (index stocks remain excluded, as in the existing validator).
