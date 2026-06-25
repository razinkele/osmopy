# Fisheries stock-status diagnostics (indicative Kobe / B/Bmsy / F/Fmsy) — design

> Status: design (revised after a literature-grounded methodological review) · 2026-06-25
> Unblocks the follow-up deferred by the 2026-06-03 fisheries-diagnostics spec. Two in-loop
> reviews shaped it: a code-grounding workflow (15 findings) and a **deep scientific-literature +
> ICES-data review** (scite + ICES MCP, ~25 papers + real Baltic reference points). The literature
> review reframed the feature from an *authoritative* single-stock Kobe to an **indicative**
> diagnostic and simplified the B-axis (see §3, §10).

## 1. Why

The Python layer can read per-species biomass-by-age, realized fishing mortality (from the
`mortalityRate` CSV), and catch. It cannot place a stock on the **Kobe** quadrant (`B/Bmsy` vs
`F/Fmsy`) — the headline fisheries-management diagnostic. The 2026-06-03 spec built the F/M
(fishing-vs-natural mortality) library but deferred Kobe / `B/Bmsy` / `F/Fmsy` for reference-point
coverage and methodological reasons. This feature adds an **indicative**, scientifically-honest
stock-status layer and gives fisheries diagnostics a proper Shiny home, including the existing
F/M bars.

## 2. Framing (the headline decision from the literature review)

This page is an **indicative, relative diagnostic — NOT a formal stock assessment.** The literature
gives no precedent for placing an OSMOSE "species" on a single-stock Kobe plot scored against
borrowed ICES reference points; published OSMOSE/EwE practice derives reference points *internally*
from the model, and treats model-vs-assessment biomass as a calibration target compared only
*relatively* (Travers-Trolet et al., 2020; Mackinson et al., 2018; Bănaru et al., 2019). Therefore:
- Quadrant shading is **soft/indicative**; the page carries a visible "indicative — relative to the
  supplied reference points, not a formal assessment" disclaimer.
- The **F-axis is the transferable one** (F/Fmsy is effort-linked); ICES **auto-fills `Fmsy` only**.
  The **B-axis is user-owned**: `Bmsy` is supplied by the user on a model-consistent (whole-domain
  SSB) basis. We do **not** auto-fill a B-reference from ICES — ICES publishes no `Bmsy`, and
  summing `MSY-Btrigger` across a species' sub-stocks masks depleted components (Eero et al., 2014;
  Forrest et al., 2023). That summed-Btrigger fallback is **removed** vs the prior draft.
- **Data-limited stocks are first-class:** a species with no `Fmsy` simply has no F-axis; a species
  with no user `Bmsy` has no B-axis. Nothing is fabricated. (Real example: eastern Baltic cod
  `cod.27.24-32` returns `null` Fmsy and `null` MSY-Btrigger.)
- **Surface:** a new dedicated `Fisheries` Shiny page bundling the indicative Kobe, the `B/Bmsy` &
  `F/Fmsy` time-series, and the existing F/M bars (first time surfaced). It links to (does not
  rebuild) the existing community size-spectrum / mean-trophic-level pages.

## 3. Methodological basis (the scientifically-honest calls)

- **B = SSB (spawning-stock biomass).** `B/Bmsy` is conventionally SSB-relative. Model B = Σ
  mature-class biomass from `results.biomass_by_age(species)` keeping age classes `≥` the species
  `maturity_age` (config) — NOT total biomass (which is always `>` SSB and would bias status green).
  Falls back to `biomass_by_size ≥ maturity_size` if a config uses size-at-maturity; if neither is
  configured nor the by-age/by-size output is present, the **B-axis is unavailable** for that species
  with a caveat (graceful).
- **`Bmsy` is user-supplied only** (no ICES auto-fill). The user provides `Bmsy` on the same
  whole-domain SSB basis as the model B. If absent → no B-axis for that species. `b_ref_kind ∈
  {"bmsy_user", "none"}`; the axis is labelled **`SSB / Bmsy [user]`**.
- **`F/Fmsy` (the ICES-fillable, indicative axis).** `Fmsy` comes from ICES (auto-filled) or the
  user. The model F is the **single exploited-stage** OSMOSE fishing mortality (the stage with
  `F > _FISHED_TOL`), an annual instantaneous rate matching Fmsy's basis — but **not** the ICES
  age-ranged `Fbar` that `Fmsy` is defined on (real Fbar windows: sprat 3–5, cod 4–6, herring 3–6),
  and the model's exploited-stage selectivity need not match that window. So F/Fmsy is **indicative
  / ordinal**, labelled `F / Fmsy [ICES, indicative]` with the selectivity caveat (Vasilakopoulos
  et al., 2020). Do NOT use the cross-stage F-sum the F/M bars use. For a multi-stock species, take
  `Fmsy` from the primary tonnes-unit stock with a caveat (F is not summable across stocks).
- **Reference-point provenance & vintage.** Each species records `fmsy` source (`ices:<stock>@<year>`
  vs `user`) and the assessment year (ICES reference points move — Silvar-Viladomiu et al., 2021).
- **Kobe quadrants (indicative):** green `B/Bmsy ≥ 1 ∧ F/Fmsy ≤ 1`; red `B/Bmsy < 1 ∧ F/Fmsy > 1`;
  the off-diagonals yellow/orange — rendered as **soft shading**, not a hard verdict. A point needs
  **both** ratios; partial-reference species (one axis) are excluded from the scatter but shown on
  the available single-axis time-series.

## 4. Components (each isolated, testable)

1. **`osmose/fisheries_reference.py`** (new) — reference-point resolution (simplified: no
   cross-stock aggregation).
   - `@dataclass ReferencePoint`: `species`, `fmsy: float | None`, `bmsy: float | None`,
     `fmsy_stock: str | None`, `fmsy_year: int | None`, `b_ref_kind: str`
     (`"bmsy_user"` | `"none"`), `source: str` (`"ices:<stock>@<year>"` | `"user"` | `"mixed"`),
     `caveats: list[str]`. Properties: `b_ref` → `bmsy`; `has_b_axis` → `bmsy is not None`;
     `has_f_axis` → `fmsy is not None`; `b_ref_label` → `"Bmsy [user]"`.
   - `load_reference_points(ref_dir, species_list, *, ices_snapshot_dir=None) ->
     dict[str, ReferencePoint]`. `ref_dir` is an **ecosystem-scoped, stable** directory
     (`data/<ecosystem>/`), NOT a per-run output dir (runs use fresh `mkdtemp` dirs). Reads optional
     `ref_dir/fisheries_reference_points.json` (`{species: {fmsy?, bmsy?}}`). For each species with
     no user `fmsy`, **auto-fills `fmsy` only** from `osmose.validation.ices.load_snapshot`: the
     primary tonnes-unit stock (via `manifest["model_species_to_ices_stocks"]` + the snapshot's
     `units_by_stock`), `float()`-coerced (ICES values are JSON strings), with `fmsy_stock`/
     `fmsy_year` recorded and a multi-stock caveat where applicable. `bmsy` is **never** auto-filled.
   - `save_reference_points(ref_dir, refs)` — writes back the user-editable fields (`fmsy`, `bmsy`).

2. **`osmose/validation/stock_status.py`** (new) — pure computation.
   - `@dataclass StockStatus`: `species`, `years: list[int]`, `b_over_bmsy: list[float | None]`,
     `f_over_fmsy: list[float | None]`, `b_ref_label: str`, `latest_quadrant: str | None`,
     `caveats: list[str]`.
   - `compute_stock_status(results, refs, config, *, species_list=None) -> list[StockStatus]`.
     Per species:
     - **F (per-year series):** reuse `fisheries.read_mortality` + `_STAGES` + the `F > _FISHED_TOL`
       single-exploited-stage rule, and a NEW `fisheries.annual_series(per_step, steps_per_year) ->
       np.ndarray` factored out of `annual_rate` (which keeps its windowed-scalar behaviour by
       calling `annual_series` then window-mean — no behaviour change).
     - **B (SSB per-year series):** from `results.biomass_by_age(species)`, which the 2D reader
       returns **long-form** (`time, species, bin, value`); `bin` is the **age-class in years**
       (note: it arrives as a *string* `"0","1",…` — `pd.to_numeric` before comparing).
       SSB(t) = Σ `value` over bins `≥ floor(maturity_age_years)`, where
       `maturity_age_years = config.maturity_age_dt[sp] / config.n_dt_per_year`; reduce to one value
       per year by **year-mean**. (Size-at-maturity configs use `biomass_by_size`.)
     - `B/Bmsy` (only if `has_b_axis`) and `F/Fmsy` (only if `has_f_axis`) per year, intersecting the
       integer-year indices; `None` where a reference/input is missing; divide-by-zero guarded
       (`bmsy`/`fmsy` `≤ 0 → None`). `latest_quadrant` from the most-recent year with both ratios.
   - `steps_per_year` (= `config.n_dt_per_year`), `maturity_age_dt`, species come from the
     `EngineConfig`; the output `prefix` from the page / `OsmoseResults.prefix`. All passed
     explicitly — never inferred.

3. **`osmose/plotting.py`** (extend) — add `make_kobe_plot(statuses, *, year=None) -> go.Figure`
   (four **soft-shaded** quadrants, one marker per species at the selected year, faint per-species
   trajectory, latest year emphasised, axis labels from the reference labels, an "indicative"
   annotation) and `make_ratio_timeseries(statuses, which) -> go.Figure`. Both confirmed absent
   today. `make_fm_ratio_bars` **already exists** (`osmose/plotting.py:337`) — **reused**, not built.

4. **`ui/pages/fisheries.py`** (new Shiny page) — `fisheries_ui()` + `fisheries_server()`.
   - Run selection reusing the `_safe_output_dir` pattern (factor the shared helper out of
     `ui/pages/results.py` if not already shared). Obtains the run's `EngineConfig` from app state
     (for `steps_per_year`, `maturity_age_dt`, species) and `prefix` from the `OsmoseResults`; SSB
     needs the config's maturity, so the B-axis is unavailable without it (caveat).
   - An **editable reference-point table** (`bmsy`, `fmsy` per species) pre-filled by
     `load_reference_points` (only `fmsy` is ICES-auto-filled), a provenance + assessment-year
     column, and a "Save" action → `save_reference_points` (ecosystem-scoped path). Empty `bmsy` →
     that species has no B-axis, noted.
   - Plots: indicative Kobe (year slider), `B/Bmsy` & `F/Fmsy` time-series, F/M bars (reused). A
     prominent **methodology/disclaimer panel**: indicative-not-assessment; B = SSB; Bmsy user-owned;
     F = single-exploited-stage vs ICES Fbar (selectivity caveat); reference-point provenance/year;
     a link to the existing community size-spectrum / MTL pages.
   - Register in `app.py` (nav entry + server wiring), following the existing page pattern.

## 5. Data flow

run output dir → `OsmoseResults` → (`biomass_by_age` for SSB, `mortalityRate` CSV for
exploited-stage F) ⨉ `EngineConfig` (`n_dt_per_year`, `maturity_age_dt`, species) +
`OsmoseResults.prefix` ⨉ `load_reference_points(ecosystem_dir, species, ices_snapshot_dir)`
(Fmsy auto-fill only) → `compute_stock_status` → per-species `StockStatus` →
`make_kobe_plot` / `make_ratio_timeseries` / `make_fm_ratio_bars` on the Fisheries page.
(Module paths: `osmose.validation.fisheries`, `osmose.validation.stock_status`,
`osmose.fisheries_reference`, `osmose.plotting`.)

## 6. Reference-point storage

`data/<ecosystem>/fisheries_reference_points.json` — an **ecosystem-scoped, stable** location (NOT
a per-run output dir). The page derives `<ecosystem>` from the run's config/data dir; failing that,
a user-config dir keyed by config name. Format (only `fmsy`/`bmsy` user-editable; `fmsy` may be
ICES-auto-filled and overridden, `bmsy` is always user-entered):
```json
{
  "sprat": { "fmsy": 0.34, "bmsy": 600000 },
  "cod":   { "fmsy": 0.31 }
}
```
`bmsy` present ⇒ `b_ref_kind="bmsy_user"`; absent ⇒ `"none"` (no B-axis).

## 7. Error handling / edge cases

- **Reader shapes differ by dimensionality.** 1D `biomass(species)` is WIDE-form (capital `Time` +
  per-species columns, constant `species="all"`) — `biomass(species="cod")` returns **0 rows**;
  select `df["cod"]` keyed on `Time`. 2D `biomass_by_age(species)` is **long-form** (`time, species,
  bin, value`) via `_read_2d_output` — SSB uses this. Do not reuse `ices.model_biomass_window_mean`'s
  `df["value"]` logic against the 1D reader.
- **Data-limited / missing references** → no F-axis if `fmsy` is `null` (e.g. eastern Baltic cod);
  no B-axis if no user `bmsy`; species excluded from the Kobe scatter, shown on whichever single
  axis it has; distinct caveats ("no Fmsy" / "no Bmsy supplied").
- **By-age output absent / no maturity configured** → B-axis unavailable; caveat "SSB unavailable
  (enable biomass-by-age output)".
- **`Fmsy ≤ 0` or `Bmsy ≤ 0`** → ratio `None` (divide-by-zero guard).
- **No ICES snapshot** (EEC/BoB) → Fmsy auto-fill is a no-op; all values come from the user file /
  inline edits. Page still works.
- **Multi-stock species** → `Fmsy` from the primary tonnes-unit stock with a caveat (F not summed).
- **Reference-point JSON malformed** → clear validation message; page degrades to inline entry.
- **Indicative disclaimer** always shown; a **spatial-scope caveat** (model domain vs ICES stock
  area) shown when any ICES-derived `Fmsy` is in use.

## 8. Testing

- `fisheries_reference`: ICES auto-fill resolves `fmsy` (+ stock + year) for sprat from a snapshot
  fixture (`float()` coercion); a data-limited stock with `null` fmsy → `has_f_axis == False`; `bmsy`
  is **never** auto-filled (only user-supplied); user file overrides ICES `fmsy`; round-trip
  save/load of user fields.
- `stock_status`: `annual_series` returns the per-year array (and `annual_rate` still returns its
  windowed scalar — no behaviour change); SSB = Σ mature age-class biomass on the **long-form**
  (`time, species, bin, value`) by-age output (hand-figured); `B/Bmsy` only when a user `bmsy` is
  set; `F/Fmsy` only when `fmsy` set; the Kobe quadrants classified; divide-by-zero guards;
  single-exploited-stage F (not the cross-stage sum); a no-B-axis / data-limited species returns the
  available single axis + caveat.
- `plotting`: `make_kobe_plot` returns a Figure with soft-shaded quadrants + the indicative
  annotation; markers at the right coordinates; partial-reference species omitted from the scatter.
- UI: page renders for a Baltic run (Fmsy auto-filled, Bmsy user-entered) and a no-ICES config
  (inline-only); the disclaimer + provenance panel present; the ecosystem-scoped sidecar **survives
  a re-run** (saved under `data/<ecosystem>/`, not the run dir).
- No engine/dynamics change → EEC/BoB parity suites untouched (read-only analysis over outputs).

## 9. Out of scope (deferred)

- **Model-internal reference points** (Fmsy & Blim≈0.2·B0 from an OSMOSE yield-vs-F sweep) — the
  literature's preferred approach (Travers-Trolet et al., 2020; Mackinson et al., 2018) and the
  natural **v2**: it adds a reference-point *source* without changing the page. A separate feature
  (needs multi-run sweep orchestration + equilibrium detection).
- **Per-stock spatial resolution** (clipping model biomass to each ICES stock area; per-stock
  small-multiples Kobe; min-component masking guards) — needs per-stock-area grid masks.
- **Cross-stock reference aggregation** — deliberately removed (masking risk).
- **Age-ranged `Fbar`** matching each stock's `f_age_range`; **SSB fecundity refinements** beyond
  mature-biomass; per-fishery status; uncertainty envelopes.
- **Community/ecosystem indicators** (LFI, MTL, size spectrum) — already shipped elsewhere; the page
  links to them rather than rebuilding.

## 10. Scientific basis & provenance

Design choices grounded in a scite + ICES-MCP literature review (2026-06-25):
- Kobe conventions; SSB as the B basis; ICES publishes no Bmsy and `MSY-Btrigger = Bpa` (verified in
  the real Baltic fixtures: sprat `541000 = 541000`) sitting below Bmsy — so it must not be used as a
  Bmsy proxy (Silvar-Viladomiu et al., 2021; ICES guidance).
- Summing reference points across a species' sub-stocks masks depleted components — *the* documented
  failure mode for Baltic cod and herring (Eero et al., 2014; Forrest et al., 2023). → B-aggregate
  removed.
- No precedent for OSMOSE-on-a-single-stock-Kobe; OSMOSE/EwE derive reference points internally and
  compare biomass *relatively* (Travers-Trolet et al., 2020; Mackinson et al., 2018; Briton et al.,
  2019; Bănaru et al., 2019). → indicative framing; F-axis transferable, B-axis user-owned.
- Model exploited-stage F vs ICES `Fbar` is selectivity-conditional (Vasilakopoulos et al., 2020). →
  F/Fmsy indicative with a selectivity caveat.

(Full citations with DOIs are in the review transcript; key DOIs: 10.1111/faf.12591, 10.1093/icesjms/fsu060,
10.1139/cjfas-2022-0168, 10.3389/fmars.2020.568232, 10.1371/journal.pone.0190015, 10.1016/j.ecolmodel.2019.03.005,
10.1111/faf.12451.)
