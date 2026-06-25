# Fisheries stock-status diagnostics (indicative Kobe / B/Bmsy / F/Fmsy) — design

> Status: design (revised after THREE in-loop reviews) · 2026-06-25
> Reviews: (1) code-grounding workflow (15 findings); (2) deep scite + ICES-MCP literature
> review (reframed to *indicative*, dropped summed-Btrigger); (3) multi-angle workflow —
> UX/product, Shiny-impl, codebase-integration, residual-science, adversarial (22 findings,
> incl. a CRITICAL aggregation-cadence bug and a structural SSB-reconstruction limit). This
> revision folds all three. **Key change this round: the model's own SSB is now an explicit,
> parity-safe engine output** (wiring the dormant `output.ssb.enabled`), because SSB cannot be
> faithfully reconstructed from the marginal by-age/by-size outputs.

## 1. Why

The Python layer can read per-species biomass, fishing mortality (from the `mortalityRate` CSV),
and catch, but cannot place a stock on the **Kobe** quadrant (`B/Bmsy` vs `F/Fmsy`). The
2026-06-03 spec built the F/M (fishing-vs-natural mortality) library but deferred Kobe. This
feature adds an **indicative**, scientifically-honest stock-status layer and a proper Shiny home
for fisheries diagnostics (including the existing F/M bars, surfaced for the first time).

## 2. Framing (from the literature review — unchanged)

Indicative, relative diagnostic — **NOT a formal stock assessment.** No precedent exists for an
OSMOSE species on a single-stock Kobe with borrowed ICES reference points; OSMOSE/EwE derive
reference points internally and compare biomass *relatively* (Travers-Trolet 2020; Mackinson
2018; Bănaru 2019). So: soft/indicative quadrant shading + a disclaimer; **F-axis ICES-fillable,
B-axis user-owned**; data-limited stocks first-class; no cross-stock reference summing (masking —
Eero 2014, Forrest 2023). Model-internal reference points (a yield-vs-F sweep) are the deferred v2.

## 3. Page information hierarchy (review finding R3-1 — deliver value with zero input)

Because `Bmsy` is never auto-filled, the **Kobe scatter is empty out-of-the-box** for every
bundled config. The page must NOT lead with an empty plot. Layout, top to bottom:
1. **F/M bars** (`make_fm_ratio_bars`, already built) — zero-config, never surfaced before. The
   immediate win.
2. **F/Fmsy time-series** — populated for species with an ICES (or user) `Fmsy` (Baltic: cod,
   herring, sprat).
3. **Kobe scatter** — a gated panel that renders only once ≥1 species has BOTH ratios; its
   empty-state is an explicit call-to-action ("Enter a `Bmsy` for ≥1 species in the table to
   populate the Kobe quadrant"), never a blank plot.
4. **B/Bmsy time-series** — for species with a user `Bmsy`.
5. The editable reference-point table + the methodology/disclaimer panel.

## 4. Model SSB as an engine output (resolves the SSB-reconstruction cluster R3-12/17/20)

The engine already computes the correct per-species SSB every step in `reproduction.py` using the
**joint** maturity rule `length ≥ maturity_size AND age_dt ≥ maturity_age_dt AND abundance > 0`,
then discards it. The marginal `biomass_by_age` / `biomass_by_size` outputs **cannot** reproduce
that joint mask (a species with both size- and age-at-maturity needs per-school age+size jointly,
which binned marginals don't preserve), and `maturity_age_dt` defaults to `0` when
`species.maturity.age` is unset — so an age-only reconstruction silently yields `SSB = total
biomass`. Therefore SSB becomes a first-class output (the same parity-safe pattern as the
just-shipped yieldN/meanSize):

- **`osmose/engine/config.py`** — parse `output.ssb.enabled` → `output_ssb: bool = False` (the key
  is already in the validation allowlist) and `output.ssb.netcdf.enabled` → `output_ssb_netcdf`.
- **`osmose/engine/simulate.py`** — `StepOutput.ssb: NDArray | None`; `_collect_ssb(state, config)`
  reusing the exact reproduction maturity conjunction; gated on `output_ssb or output_ssb_netcdf`;
  subdt-accumulated as a **mean** across the record window (like a biomass mean), wired into both
  accumulation branches.
- **`osmose/engine/output.py`** — `_write_ssb_csv` → `{prefix}_SSB_Simu0.csv` + `_build_ssb_dataframe`
  → `{"SSB": df}` (wide Time + focal-species, mirrors meanTL); NetCDF want/data_var on
  `["time","focal_species"]`; register in `write_outputs`.
- **`osmose/results.py`** — add `"SSB"` to `_CROSS_SPECIES_OUTPUT_TYPES` + `_build_dataframes_from_outputs`;
  `results.ssb(species)` reader (`_read_species_output("SSB")`, wide-form).
- The stock-status B-axis reads `results.ssb(species)` directly. If a run did not enable
  `output.ssb.enabled`, the **B-axis is unavailable** with a caveat ("enable SSB output"); no
  reconstruction is attempted.

This is output-only and parity-safe (the EEC/BoB suites assert biomass, not SSB).

## 5. Methodological basis

- **B = model SSB** (§4), compared to a **user-supplied `Bmsy`** (no ICES auto-fill — ICES
  publishes no Bmsy, and summing Btrigger masks components). `b_ref_kind ∈ {"bmsy_user","none"}`;
  axis labelled `SSB / Bmsy [user]`.
- **`F/Fmsy` (ICES-fillable, indicative).** `Fmsy` from ICES (auto-filled) or user. Model F = the
  **exploited-stage** OSMOSE fishing mortality, an annual instantaneous rate. It is **not** the ICES
  age-ranged `Fbar` (real windows sprat 3–5, cod 4–6, herring 3–6) → F/Fmsy is indicative/ordinal,
  with a selectivity caveat (Vasilakopoulos 2020).
  - **Exploited-stage disambiguation (R3-14/21):** among the fished stages
    (`F > _FISHED_TOL`, excluding `Eggs`), select the stage with the **largest annual F**; emit a
    caveat when >1 stage exceeds the tolerance ("F measured on `<stage>`; other fished stages
    present"). Do NOT use the F/M bars' cross-stage F-sum.
  - **Multi-stock `Fmsy` (R3-13/19):** "primary" stock is defined **deterministically** as the
    tonnes-unit stock with the **largest `msy_btrigger`** (a stock-size proxy; ties broken by most
    recent `advice_year`). Record `fmsy_stock` + `fmsy_year` in provenance and emit a caveat when
    the species maps to >1 tonnes-unit stock (e.g. Baltic herring → 3 tonnes stocks, Fmsy 0.218–0.31).
- **Annual aggregation cadence (R3-11/18 — was a CRITICAL bug).** The saved series are written
  every `output.recordfrequency.ndt` steps, NOT every step, and the engine **sums** mortality and
  **means** biomass over each record window. So the per-year reshape uses
  **`saved_steps_per_year = config.n_dt_per_year // config.output_record_frequency`** (asserted to
  divide evenly; caveat otherwise) — NOT `n_dt_per_year`. For the Baltic flagship
  (`recordfrequency.ndt = ndtPerYear = 24`) `saved_steps_per_year = 1`: each saved row is already
  one year, so the annual F is the saved value directly and SSB is the saved annual-mean.
- **Year alignment (R3-15).** Both axes are labelled by **absolute simulation year** taken from the
  readers' `Time` column (`year = int(floor(time))`), and the Kobe/ratio intersect on those year
  labels — never on positional array index.
- **B-axis basis caveat (R3-16).** Whenever a user `Bmsy` is present, show a spatial/SSB-basis
  caveat AND display the model's whole-domain mean SSB next to the `Bmsy` input so the user can
  sanity-check that their value is on the same basis (whole-domain SSB tonnes), not an ICES-area SSB.
- **Kobe quadrants (indicative):** green `B/Bmsy ≥ 1 ∧ F/Fmsy ≤ 1`; red `< 1 ∧ > 1`; off-diagonals
  yellow/orange — soft shading, not a verdict. A point needs both ratios; single-axis species go to
  the time-series, not the scatter.

## 6. Components

0. **Engine SSB output** — §4.
1. **`osmose/validation/fisheries_reference.py`** (new; placed under `validation/` for cohesion with
   `fisheries.py`/`ices.py`/`stock_status.py` — R3-9).
   - `@dataclass ReferencePoint`: `species`, `fmsy`, `bmsy`, `fmsy_stock`, `fmsy_year`,
     `b_ref_kind` (`"bmsy_user"`|`"none"`), `source`, `caveats: list[str]`; properties `has_b_axis`,
     `has_f_axis`, `b_ref_label`.
   - `load_reference_points(ref_dir, species_list, *, ices_snapshot_dir=None)`. `ref_dir` =
     **`data/<ecosystem>/reference/`** (R3-6 — alongside `biomass_targets.csv` + `ices_snapshots/`).
     Reads optional `ref_dir/fisheries_reference_points.json` (`{species: {fmsy?, bmsy?}}`);
     **auto-fills `fmsy` only** from the deterministic primary tonnes-unit stock (§5). **Validates
     every JSON key against `species_list`** and returns unmatched keys for a UI warning (R3-22).
     `bmsy` never auto-filled.
   - `save_reference_points(ref_dir, refs)` — writes user fields back.
   - **Contrast with `osmose/calibration/targets.py` (R3-8):** that store (`biomass_targets.csv`,
     `BiomassTarget`) holds *total-biomass calibration* targets; this sidecar holds *SSB-relative
     status* reference points (`Bmsy`/`Fmsy`). Different quantity, different consumer — a second file
     is warranted; documented here so they're not conflated.
2. **`osmose/validation/stock_status.py`** (new).
   - `@dataclass StockStatus`: `species`, `years`, `b_over_bmsy`, `f_over_fmsy`, `b_ref_label`,
     `latest_quadrant`, `takeaway: str | None`, `caveats`.
   - `compute_stock_status(results, refs, config, *, species_list=None)`. F per year from
     `read_mortality` + the exploited-stage rule (§5) + a NEW
     `fisheries.annual_series(per_step, saved_steps_per_year, year_labels) -> dict[int,float]`
     factored from `annual_rate` (which keeps its windowed-scalar behaviour); B per year from
     `results.ssb(species)` (annual-labelled). `B/Bmsy` (if `has_b_axis`) and `F/Fmsy` (if
     `has_f_axis`) per absolute year, divide-by-zero guarded; a one-line `takeaway` per populated
     latest point (R3-3, e.g. "Indicative: F above Fmsy and SSB below your Bmsy"). `steps_per_year`
     is computed as in §5; never `n_dt_per_year`.
3. **`osmose/plotting.py`** (extend) — `make_kobe_plot` (soft-shaded quadrants + indicative
   annotation) and `make_ratio_timeseries`; both new. `make_fm_ratio_bars` already exists
   (`plotting.py:337`) — reused.
4. **`ui/pages/fisheries.py`** (new page) — `fisheries_ui()`/`fisheries_server()`.
   - Builds the run config via **`EngineConfig.from_dict(state.config.get())`** (R3-5 — app state
     holds a flat `config: reactive.Value[dict]` + `output_dir`/`config_dir`, NOT an `EngineConfig`);
     `prefix` from the `OsmoseResults`. Run selection reuses `_safe_output_dir`.
   - **Ecosystem derivation (R3-10):** a small shared helper `ecosystem_of(config_dir)` (in
     `osmose/validation/`) maps the run's `config_dir` basename to `<ecosystem>`; reused by the page
     and `load_reference_points`. Falls back to a user-config dir keyed by config name.
   - Layout per §3. Editable reference-point table (per-species `bmsy`/`fmsy` numeric inputs — the
     concrete widget, since no existing page uses a data-frame editor (R3-5); provenance +
     assessment-year column; each `bmsy` input shows the basis label and the model's current mean SSB
     for scale-checking (R3-2/16); an explicit "enter ≥1 Bmsy to populate the Kobe" hint). A
     **"Save" action** that shows its target ("saved to `data/<ecosystem>/reference/` — shared across
     `<ecosystem>` runs"; R3-4) + a load-time note for unmatched keys.
   - One concise global **disclaimer banner** (R3-3) separate from the per-point takeaways.
   - Links to the **Size Spectrum / Mean Trophic Level views on the Results page** (result-type
     selections, not standalone pages — R3-7).
   - Register in `app.py` (nav entry + server wiring).

## 7. Data flow

run dir → `OsmoseResults` (`ssb` for B, `mortalityRate` for exploited-stage F) ⨉
`EngineConfig.from_dict(state.config)` (`n_dt_per_year`, `output_record_frequency`,
`maturity_*`, species) + `OsmoseResults.prefix` ⨉
`load_reference_points(data/<ecosystem>/reference/, species, ices_snapshot_dir)` (Fmsy only) →
`compute_stock_status` (annual aggregation at `saved_steps_per_year`, absolute-year labels) →
`make_fm_ratio_bars` / `make_ratio_timeseries` / `make_kobe_plot`.
Modules: `osmose.validation.{fisheries, stock_status, fisheries_reference}`, `osmose.plotting`.

## 8. Reference-point storage

`data/<ecosystem>/reference/fisheries_reference_points.json` (R3-6). Only `fmsy`/`bmsy` user-editable
(`fmsy` may be ICES-auto-filled then overridden; `bmsy` always user-entered):
```json
{ "sprat": { "fmsy": 0.34, "bmsy": 600000 }, "cod": { "fmsy": 0.31 } }
```
`bmsy` present ⇒ `b_ref_kind="bmsy_user"`; absent ⇒ `"none"`.

## 9. Error handling / edge cases

- **Reader shapes:** 1D `biomass`/`ssb` are WIDE (Time + per-species cols); 2D readers are LONG.
  Select the species column keyed on `Time`.
- **SSB output disabled in the run** → no B-axis, caveat ("enable `output.ssb.enabled`").
- **Data-limited references** → no F-axis if `fmsy` null (eastern Baltic cod); no B-axis if no user
  `bmsy`; species excluded from the Kobe scatter, shown on the available single axis.
- **`Fmsy ≤ 0` / `Bmsy ≤ 0`** → ratio `None` (guard).
- **`output_record_frequency` does not divide `n_dt_per_year` evenly** → caveat, best-effort annual
  reshape with a warning.
- **No ICES snapshot** (EEC/BoB) → no Fmsy auto-fill; all values manual; page still works.
- **Multi-tonnes-stock species** → deterministic primary stock (§5) recorded + caveat.
- **Unmatched reference-JSON keys** (typo/casing) → explicit UI warning (R3-22), not silent.
- **Malformed JSON** → validation message; degrade to inline entry.
- Indicative disclaimer always shown; spatial/basis caveat whenever a user `Bmsy` or ICES `Fmsy` is in use.

## 10. Testing

- **Engine SSB output:** `_collect_ssb` matches the reproduction conjunction on a hand-built state
  (size-only, age-only, and both-criteria species — the case reconstruction can't do); CSV +
  in-memory + reader round-trip; gating (None when disabled); subdt mean. EEC/BoB parity suites
  stay green (output-only).
- **stock_status:** `annual_series` at `saved_steps_per_year` (incl. the Baltic
  `recordfreq=ndtPerYear ⇒ saved_steps_per_year=1` case — each saved row already annual);
  absolute-year alignment (no positional off-by-one); exploited-stage tie-break (largest-F stage,
  Eggs excluded, caveat when >1); `B/Bmsy` only with a user `bmsy`; `F/Fmsy` only with `fmsy`;
  divide-by-zero; data-limited single-axis + caveat; the `takeaway` line.
- **fisheries_reference:** Fmsy auto-fill from the deterministic primary stock (largest Btrigger;
  herring multi-stock → caveat + recorded `fmsy_stock`); data-limited `null` fmsy → `has_f_axis`
  False; `bmsy` never auto-filled; unmatched-key warning; round-trip save/load.
- **plotting:** soft-shaded Kobe + indicative annotation; partial-reference species omitted from the
  scatter; markers at the right coords.
- **UI:** empty-state CTA renders (no blank Kobe); F/M bars + F/Fmsy lead the layout; save-target +
  unmatched-key notes shown; sidecar under `data/<ecosystem>/reference/` survives a re-run.

## 11. Out of scope (deferred)

- **Model-internal reference points** (Fmsy & Blim≈0.2·B0 from a yield-vs-F sweep) — the literature's
  preferred approach and the natural v2.
- **Per-stock spatial resolution** (clipping model biomass to each ICES stock area; per-stock
  small-multiples; min-component masking guards) — needs per-stock-area grid masks.
- **Cross-stock reference aggregation** (removed — masking).
- **Age-ranged `Fbar`** matching `f_age_range`; fecundity-based SSB refinements; per-fishery status;
  uncertainty envelopes; SSB-derived Bmsy *suggestions* (deliberately not auto-populated — would
  fabricate a reference, contradicting §2).
- **Community indicators** (LFI/MTL/size spectrum) — already shipped; the page links to them.

## 12. Scientific basis & provenance

Grounded in a scite + ICES-MCP literature review (2026-06-25): Kobe conventions; SSB as the B basis;
ICES publishes no Bmsy and `MSY-Btrigger = Bpa` < Bmsy (Silvar-Viladomiu 2021); summing references
masks depleted components (Eero 2014; Forrest 2023); no OSMOSE-on-single-stock-Kobe precedent —
internal/relative practice (Travers-Trolet 2020; Mackinson 2018; Briton 2019; Bănaru 2019); model F
vs ICES `Fbar` is selectivity-conditional (Vasilakopoulos 2020). Key DOIs: 10.1111/faf.12591,
10.1093/icesjms/fsu060, 10.1139/cjfas-2022-0168, 10.3389/fmars.2020.568232, 10.1371/journal.pone.0190015,
10.1016/j.ecolmodel.2019.03.005, 10.1111/faf.12451.
