# Model-internal fishery reference points (Fmsy / Bmsy / Blim) — design

> Status: design (awaiting review) · 2026-06-25
> The deferred **v2** of the fisheries stock-status feature (`docs/superpowers/specs/2026-06-25-fisheries-stock-status-diagnostics-design.md`).
> Derives per-species reference points from the model's OWN yield-vs-F response — the
> established OSMOSE/EwE practice (Travers-Trolet 2020; Mackinson 2018) — so the Kobe page
> auto-populates both axes with no user-entered Bmsy and no ICES dependency.

## 1. Why

The shipped fisheries page is *indicative* and requires a **user-supplied Bmsy** per species
(ICES publishes none); out-of-the-box the Kobe scatter is empty. The literature's preferred fix
is to compute reference points **internally**: run the model across a grid of fishing mortalities,
read equilibrium yield and spawning biomass, and take Fmsy = the F that maximises equilibrium
yield, Bmsy = the equilibrium SSB at Fmsy, and Blim = 0.2·B0 (B0 = unfished SSB). These are
internally consistent with the model's own dynamics (single-stock ICES reference points are not
simultaneously achievable in an interacting system). This feature adds that sweep as a library +
CLI; the existing page reads the result.

## 2. Scope decisions (from brainstorming)

- **Surface = library + CLI** (not a UI button). A function + a script run the sweep (minutes,
  offline) and write a sidecar JSON; the fisheries page auto-reads it. A UI "compute" button is a
  deferred follow-up.
- **Per-species "conditional" Fmsy** (the defensible multispecies method): sweep ONE species' F
  while holding every other species at its **baseline** config F. Fmsy is therefore conditional on
  the current fishing of the rest of the system — surfaced as a caveat. (Cost: `n_species × |f_grid|`
  runs + replicates, on the calibration process-pool.)
- **The model provides the FULL set** per species: `fmsy`, `bmsy` (SSB at Fmsy), `b0` (unfished
  SSB), `blim` (= 0.2·b0) — fully populating both Kobe axes.

## 3. Methodology

- **Per-species sweep.** For species *i*: build configs at each F in an **absolute** grid `f_grid`
  (default `np.linspace(0.0, 2.0, 11)` = `0.0, 0.2, …, 2.0` /yr, 11 points; configurable), overriding
  only species *i*'s fishing rate
  (`mortality.fishing.rate.sp{i}`) and leaving all other species at their baseline config value.
  An absolute grid (not a multiplier of the baseline) is required because a species' baseline F may
  be 0. The `F=0` point doubles as the B0 (unfished SSB) reference.
- **Equilibrium.** Each run executes `n_years` (default = `max(config nyear, 50)`, so the trailing
  window has ≥40 spin-up years before it); the equilibrium yield/SSB is the **trailing-window
  mean** of the per-year series (the codebase's existing convention — `annual_by_year(..., how="mean")`
  over the last `window_years`, default 10). The engine is stochastic (PCG64), so each grid point is
  the mean over `replicates` runs (default a small number, e.g. 1–3; configurable) with distinct seeds.
- **Reference points.** `fmsy` = the grid F maximising equilibrium yield (parabolic-refine the peak
  from the three points around the max for sub-grid precision); `bmsy` = equilibrium SSB at that F
  (interpolated); `b0` = equilibrium SSB at F=0; `blim = 0.2 · b0`. If the yield maximum is at the
  **last grid point** (no interior peak), emit a boundary warning and mark the species' Fmsy
  unreliable (the grid did not bracket the peak — `f_grid` should be extended).
- **Caveats** recorded per species: conditional-Fmsy (depends on others' baseline F); ecosystem-state
  dependence; model-internal (not an external assessment) — consistent with the page's indicative
  framing. Species with no fishing in the baseline config (and thus no meaningful yield curve) are
  reported with a caveat and no Fmsy.

## 4. Components (each isolated, testable)

1. **`osmose/validation/fmsy_sweep.py`** (new) — the sweep core (pure compute over the engine; no UI).
   - `@dataclass SweepPoint`: `species`, `f`, `yield_eq`, `ssb_eq` (per grid point, per replicate-mean).
   - `@dataclass ModelReferencePoint`: `species`, `fmsy: float|None`, `bmsy: float|None`,
     `b0: float|None`, `blim: float|None`, `fmsy_at_boundary: bool`, `caveats: list[str]`,
     `curve: list[SweepPoint]` (for plotting/debug).
   - `run_yield_f_sweep(base_config, grid, species_list, config, *, n_years, replicates, window_years,
     max_workers, seed0) -> dict[str, list[SweepPoint]]`: builds the per-(species, F, replicate) config
     overrides, runs them via `PythonEngine.run_in_memory` on a `ThreadPoolExecutor`/`ProcessPoolExecutor`
     (reuse the calibration backend choice), reads equilibrium yield (`results.fishery_yield`) + SSB
     (`results.ssb`), averages replicates. Pure given a base config dict; deterministic per seed.
   - `derive_reference_points(curves) -> dict[str, ModelReferencePoint]`: the Fmsy/Bmsy/B0/Blim +
     boundary logic above. Separated from the runner so it is unit-testable on synthetic curves.
   - `compute_model_reference_points(base_config, *, grid=None, n_years=None, replicates=1,
     window_years=10, max_workers=None) -> dict[str, ModelReferencePoint]`: the top-level convenience
     (build config → sweep → derive). `grid=None` picks a sensible default absolute grid.
2. **`scripts/compute_model_reference_points.py`** (new CLI) — `--config <path>` (a base config CSV/dir),
   `--grid`/`--n-years`/`--replicates`/`--workers` overrides, `--out` (default
   `data/<ecosystem>/reference/fisheries_model_reference_points.json`). Reads the config via the
   existing `OsmoseConfigReader`, calls `compute_model_reference_points`, writes the sidecar (Fmsy,
   Bmsy, B0, Blim, fmsy_at_boundary, caveats per species) + a one-line progress log per run.
3. **`osmose/validation/fisheries_reference.py`** (extend) — read the model sidecar.
   - `ReferencePoint` gains nothing structurally beyond the existing `source` string; add a model
     branch to `load_reference_points`: read `ref_dir/fisheries_model_reference_points.json`, and for
     each species fill `fmsy`/`bmsy` from the model when present, tagging `source="model"` and
     `b_ref_kind="bmsy_model"`. **Precedence: user > model > ICES** — user JSON overrides model;
     model overrides ICES-auto-filled Fmsy; ICES is the fallback only where neither user nor model
     has a value. `b_ref_label` returns `"Bmsy [model]"` vs `"Bmsy [user]"` per the kind. Model-derived
     values are NOT written back by `save_reference_points` (they live in their own sidecar, regenerated
     by the CLI).
4. **`ui/pages/fisheries.py`** (minor) — surface the provenance: the reference-point table shows a
   `source` column ("model" / "ICES" / "user"); with a model sidecar present the Kobe auto-populates
   both axes (model Bmsy → B-axis). A short note tells the user a model sidecar exists / how it was
   produced. No new background-job machinery (the CLI produces the sidecar).

## 5. Data flow

`compute_model_reference_points(base_config)`:
base config → for each (species *i*, F in grid, replicate): override `mortality.fishing.rate.sp{i}`
→ `PythonEngine.run_in_memory` (pool) → trailing-window-mean equilibrium yield + SSB → average
replicates → per-species yield-vs-F curve → `derive_reference_points` (Fmsy/Bmsy/B0/Blim) →
sidecar JSON. Then, independently, the page: `load_reference_points` merges user > model > ICES →
`compute_stock_status` → Kobe (now populated by model Bmsy + Fmsy).

## 6. Sidecar format

`data/<ecosystem>/reference/fisheries_model_reference_points.json`:
```json
{
  "_meta": { "grid": [0.0, 0.1, "…"], "n_years": 50, "replicates": 1, "window_years": 10 },
  "cod":   { "fmsy": 0.31, "bmsy": 118000, "b0": 410000, "blim": 82000, "fmsy_at_boundary": false },
  "sprat": { "fmsy": 0.62, "bmsy": 540000, "b0": 1800000, "blim": 360000, "fmsy_at_boundary": false }
}
```
Distinct from the user sidecar (`fisheries_reference_points.json`) and never written by the UI.

## 7. Error handling / edge cases

- **No interior yield peak** (max at the last grid F) → `fmsy_at_boundary=True`, Fmsy still reported
  but flagged unreliable + a caveat ("extend the F grid"); the page shows the badge.
- **Species unfished in the baseline config** (its yield curve is ~flat/zero because selectivity or
  catchability zero it out) → no Fmsy, caveat.
- **A sweep run fails / returns empty outputs** → that grid point is dropped with a warning; if too
  few points remain to find a peak, the species gets no Fmsy + a caveat (the sweep does not crash).
- **B0 ≤ 0** (degenerate F=0 SSB) → no Blim, caveat.
- **No model sidecar present** → the page behaves exactly as today (user/ICES only) — this feature is
  purely additive.
- **`grid` not bracketing 0** → the runner inserts F=0 (needed for B0); documented.
- Reuse the engine's existing config validation; the only overridden key is `mortality.fishing.rate.sp{i}`.

## 8. Testing

- `derive_reference_points` (pure, fast): on a synthetic monotone-rise-then-fall yield curve, Fmsy is
  the interior max (with parabolic sub-grid refinement); Bmsy = SSB at that F; B0/Blim from the F=0
  point; a monotone-increasing curve → `fmsy_at_boundary=True` + caveat; a zero-yield curve → no Fmsy.
- `run_yield_f_sweep` (integration, a TINY fast config, few species, small grid, 1 replicate): produces
  per-species curves; the F=0 run yields the largest SSB (unfished); higher F lowers SSB; the override
  only touches the swept species (others' baseline F unchanged — assert via the produced configs).
- `fisheries_reference`: model sidecar loads; **precedence** user > model > ICES (a species with a
  user Bmsy keeps it; a species with only a model entry gets `source="model"`, `b_ref_kind="bmsy_model"`,
  `b_ref_label="Bmsy [model]"`; ICES Fmsy used only where neither). `save_reference_points` never writes
  model values.
- CLI: writes a valid sidecar for a tiny config that `load_reference_points` then reads; `_meta` present.
- UI: `build_fisheries_view` with a model sidecar → Kobe `kobe_ready` True with no user input; source
  column reflects "model".
- No engine/dynamics change → EEC/BoB parity suites untouched (the sweep only RUNS the engine with
  varied fishing-rate config, exactly as calibration already does).

## 9. Out of scope (deferred)

- **UI "Compute Fmsy/Blim" button** + background-job progress (the deferred richer surface; the CLI is v1).
- **Whole-system F-multiplier sweep** (cheaper, scales all species together) — the per-species method is
  the defensible default; a multi-species-MSY mode is a future option.
- **Mixed-fishery / multi-gear F** decomposition; selectivity-pattern sweeps; per-fishery reference points.
- **Climate/environment-conditioned reference points** (Fmsy shifts with ecosystem state — Travers-Trolet
  2020); the sweep is run under the config's fixed forcing.
- **Auto-extending the F grid** when Fmsy hits the boundary (v1 warns; the user re-runs with a wider grid).

## 10. Scientific basis

Model-internal reference points are the established ecosystem-model practice precisely because
single-stock reference points are not simultaneously achievable in an interacting system and shift
with ecosystem/climate state (Travers-Trolet et al., 2020, 10.3389/fmars.2020.568232; Mackinson et
al., 2018, 10.1371/journal.pone.0190015; Briton et al., 2019). Fmsy as the yield-maximising F and
Blim ≈ 0.2·B0 follow Mackinson et al.'s OSMOSE/EwE methodology. The per-species conditional reading
(others held at baseline) is the standard single-species-in-ecosystem reference; its dependence on the
assumed fishing of the rest of the system is a known limitation, surfaced as a caveat.
