# Model-internal fishery reference points (Fmsy / Bmsy / Blim) — design

> Status: design (revised after in-loop workflow review) · 2026-06-25
> The deferred **v2** of the fisheries stock-status feature. Derives per-species reference points
> from the model's OWN yield-vs-F response (Travers-Trolet 2020; Mackinson 2018) so the Kobe page
> auto-populates both axes. The first review found the original draft's central mechanism broken
> three ways (wrong fishing knob, wrong yield reader, SSB output disabled) and the cost badly
> understated — all corrected below.

## 1. Why

The shipped fisheries page is *indicative* and needs a **user-supplied Bmsy** per species; the Kobe
scatter is empty out-of-the-box. The literature's fix: compute reference points **internally** by
sweeping fishing mortality, reading equilibrium yield + spawning biomass, and taking Fmsy = the F
maximising equilibrium yield, Bmsy = the equilibrium SSB at Fmsy, Blim = 0.2·B0. This is an
**offline, parallel, cached-per-config batch** (tens of minutes to a few hours — §2 cost) exposed
as a library + CLI; the page reads the resulting sidecar.

## 2. Scope decisions

- **Surface = library + CLI** (no UI button). The CLI runs the sweep once per config and writes a
  model sidecar; the page auto-reads it. A "Compute" UI button is deferred.
- **Per-species conditional Fmsy:** sweep ONE species' fishing rate while holding all others at their
  **baseline** config rate — Fmsy is conditional on the rest of the system's baseline fishing (a
  surfaced caveat; the assembled Kobe is NOT a mutual-MSY equilibrium).
- **Full reference set** per species: `fmsy`, `bmsy` (SSB at Fmsy), `b0` (conditional unfished SSB =
  species *i* at F=0, others at baseline), `blim` (= 0.2·b0).
- **Cost is real (re-baselined from the review's measurement: ~4.4 s/sim-year on baltic).** The sweep
  is `n_species × |grid| × replicates` runs of `n_years` each. With the tuned defaults below
  (grid 7, n_years 30, replicates 3) baltic (8 sp) ≈ 168 runs × ~130 s ≈ 6 h serial / **~30–45 min on
  an 8-core ProcessPool**; eec_full (14 sp, slower) is larger. It is a deliberate **one-time offline
  computation per config**, cached in the sidecar and reused — acceptable like a calibration run, but
  the CLI prints an up-front time estimate and the defaults are tuned for tractability, not speed.

## 3. Methodology

- **Which fishing knob to override (THE crux — was wrong).** `EngineConfig.from_dict` branches: when
  `module.multispecies.fisheries.enabled==true` AND `simulation.nfisheries>0` (the **v4 fisheries
  mode** — BOTH bundled configs: eec_full nfisheries=14, baltic=8), the legacy
  `mortality.fishing.rate.sp{i}` is **never read**; F comes from `fisheries.rate.base.fsh{j}` with the
  species→fishery map built from the catchability matrix (`config.py:296-315`). So the runner must
  **detect the mode** and override the active knob:
  - **fisheries mode:** map species *i* → its fishery *j* (the first catchability column > 0 for that
    species), override `fisheries.rate.base.fsh{j}`. Guard **shared fisheries**: if fishery *j* lands
    >1 species, overriding it moves F for all of them — in that case sweep at the fishery level (and
    record which species share it) or skip with a caveat. (Both bundled configs are 1:1 species↔fishery
    — verified — so per-species overrides are clean there.)
  - **legacy mode** (`nfisheries==0`): override `mortality.fishing.rate.sp{i}`.
  - The runner asserts the override actually changed `EngineConfig.fishing_rate[i]` before running (a
    no-op override = a silent flat curve = garbage Fmsy — the original bug).
- **Outputs the sweep must force on.** A default `run_in_memory` does NOT produce SSB (it is gated on
  `output.ssb.enabled`; the in-memory `"yield"` output is unconditional, but the flag is still set so
  the disk/CLI path matches). Each sweep config override sets `output.ssb.enabled=true` **and**
  `output.yield.biomass.enabled=true` **and** the mortality output needed for the realized-F basis
  (below); other heavy/spatial outputs are disabled for speed.
- **Reading equilibrium yield + SSB (readers were wrong).** Per-species **yield = `results.yield_biomass(species)`** (the in-memory `"yield"` output, tonnes — NOT `results.fishery_yield`, which reads a Java-only `fisheryYieldBiomass` and raises in-memory); **SSB = `results.ssb(species)`**. Equilibrium = the **trailing-window mean** (`annual_by_year(..., how="mean")`, last `window_years`, default 10) of each run, averaged over `replicates`.
- **Realized-F basis (so Fmsy matches the page).** The grid overrides the *nominal* fishing rate, but
  selectivity makes the realized population F lower; the page's model F is the **realized
  exploited-stage F** (from `mortalityRate`). To keep `F/Fmsy` consistent, the sweep indexes each
  run's equilibrium yield by its **realized exploited-stage annual F** (read from that run's
  `mortalityRate`, the same extraction `stock_status.py` uses), and reports Fmsy on that realized
  basis — not the nominal grid value.
- **Equilibrium convergence (50yr may not suffice after an F step).** `n_years` default
  `max(config nyear, 30)`; the runner checks convergence — compare the last window's mean to the prior
  window's; if yield or SSB still trends beyond a tolerance, flag the grid point as `not_converged`
  (and the species' Fmsy as lower-confidence). Predation-release/trophic transients on the held-baseline
  species are the reason this matters.
- **Deriving Fmsy (single-peak is NOT guaranteed).** In a multispecies model, fishing species *i* down
  can release prey/predators → a non-monotone or multi-peaked yield-vs-F curve. `derive_reference_points`
  therefore: (a) finds the global yield max; (b) counts interior local maxima (gradient sign changes) —
  if >1, flags `multi_peak`; (c) reports the **grid-argmax point** — Fmsy = that point's realized F,
  `bmsy` = that point's SSB (sub-grid parabolic refinement + Bmsy interpolation **deferred to §9**:
  with the realized-F basis, refining nominal F then reporting realized F is incoherent, and the
  7-point grid makes it a minor precision nicety);
  (d) max-at-first-point (F=0, yield monotonically decreasing — over-fished at baseline) → no valid
  Fmsy + caveat; (e) max-at-last-point → `fmsy_at_boundary` + caveat (grid didn't bracket the peak).
  `b0` = SSB at F=0; `blim = 0.2·b0` (require `b0 > 0`).
- **Parallelization (don't oversubscribe).** A single engine run already saturates cores via the
  `@njit(parallel=True)` mortality kernel. So each pool worker sets `numba.set_num_threads(1)`
  (per-run single-threaded) and the sweep parallelizes ACROSS runs on a **ProcessPool** sized to
  `cpu_count` — reusing the calibration process backend's pattern.
- **Replicates.** Default `3` (the PCG64 engine is stochastic; 1 replicate puts Fmsy on a noise-jittered
  grid point and makes the sidecar non-reproducible). Guidance: ~5–10 for noisy small-pelagic/
  recruitment-driven species. Replicate-mean the yield + SSB before deriving.

## 4. Components

1. **`osmose/validation/fmsy_sweep.py`** (new) — the sweep core (pure compute over the engine).
   - `@dataclass SweepPoint`: `species`, `f_nominal`, `f_realized`, `yield_eq`, `ssb_eq`, `not_converged: bool`.
   - `@dataclass ModelReferencePoint`: `species`, `fmsy`, `bmsy`, `b0`, `blim`, `fmsy_at_boundary: bool`,
     `multi_peak: bool`, `caveats: list[str]`, `curve: list[SweepPoint]`.
   - `_fishing_override(config, species_idx) -> tuple[str, float]`: detect mode, return the active key
     to override for species *i* (`fisheries.rate.base.fsh{j}` or `mortality.fishing.rate.sp{i}`) + its
     baseline value; raises on a shared-fishery species (handled by the caller's caveat path).
   - `run_yield_f_sweep(base_config, config, species_list, *, grid, n_years, replicates, window_years,
     max_workers, seed0) -> dict[str, list[SweepPoint]]`: builds the per-(species,F,replicate) override
     configs (active fishing key + forced `output.ssb.enabled`/`output.yield.biomass.enabled`/mortality
     + reduced n_years), runs them on a ProcessPool (numba single-threaded per worker), reads
     `yield_biomass`/`ssb`/realized-F, asserts the override took effect, replicate-means, returns curves.
   - `derive_reference_points(curves) -> dict[str, ModelReferencePoint]`: the peak/boundary/multi-peak/
     B0/Blim logic above. **Pure** — unit-tested on synthetic curves with no engine.
   - `compute_model_reference_points(base_config, *, grid=None, n_years=None, replicates=3,
     window_years=10, max_workers=None) -> dict[str, ModelReferencePoint]`: top-level (build config →
     sweep → derive). `grid=None` → default `np.linspace(0.0, 2.0, 7)`.
2. **`scripts/compute_model_reference_points.py`** (new CLI) — `--config`, grid/n-years/replicates/
   workers overrides, `--out` (default `data/<ecosystem>/reference/fisheries_model_reference_points.json`).
   Reads config via `OsmoseConfigReader`, **prints an up-front run-count + time estimate**, runs the
   sweep with a per-run progress log, writes the sidecar.
3. **`osmose/validation/fisheries_reference.py`** (extend) — read the model sidecar.
   `load_reference_points` gains a model branch: read `ref_dir/fisheries_model_reference_points.json`;
   fill `fmsy`/`bmsy` from the model where present, `source="model"`, `b_ref_kind="bmsy_model"`.
   **Precedence: user > model > ICES.** `b_ref_label` becomes conditional — `"Bmsy [user]"` /
   `"Bmsy [model]"` per `b_ref_kind` (this also fixes the parent feature's always-"[user]" minor).
   `save_reference_points` never writes model values.
4. **`ui/pages/fisheries.py`** (minor) — the reference-point table shows a `source` column
   (model/ICES/user); with a model sidecar the Kobe auto-populates both axes. A note states the
   reference points are model-internal + conditional, and how to (re)generate them via the CLI.

## 5. Data flow

`compute_model_reference_points(base_config)`: for each (species *i*, F in grid, replicate) → override
the **active fishing key** for *i* + force SSB/yield/mortality outputs → `PythonEngine.run_in_memory`
(ProcessPool, numba 1 thread) → trailing-window-mean equilibrium `yield_biomass` + `ssb` + realized
exploited-stage F → replicate-mean → per-species yield-vs-(realized F) curve → `derive_reference_points`
→ sidecar JSON. Then the page: `load_reference_points` merges user > model > ICES → `compute_stock_status`
→ Kobe (populated by model Bmsy + Fmsy).

## 6. Sidecar format

`data/<ecosystem>/reference/fisheries_model_reference_points.json`:
```json
{
  "_meta": { "grid": [0.0, 0.33, 0.67, 1.0, 1.33, 1.67, 2.0], "n_years": 30,
             "replicates": 3, "window_years": 10, "f_basis": "realized_exploited_stage" },
  "cod":   { "fmsy": 0.29, "bmsy": 118000, "b0": 410000, "blim": 82000,
             "fmsy_at_boundary": false, "multi_peak": false },
  "sprat": { "fmsy": 0.55, "bmsy": 540000, "b0": 1800000, "blim": 360000,
             "fmsy_at_boundary": false, "multi_peak": true }
}
```
Fmsy/Bmsy in realized-F / SSB-tonnes basis. Distinct from the user sidecar; never UI-written.

## 7. Error handling / edge cases

- **Fishing-mode mismatch (the original critical bug):** detect fisheries vs legacy mode and override
  the active key; **assert `fishing_rate[i]` actually changed** before the run, else raise (never a
  silent flat curve).
- **Shared fishery** (one fishery, >1 species): sweeping it moves all its species — record the shared
  set + caveat (or sweep at fishery level). Bundled configs are 1:1 so this is the non-default path.
- **SSB/yield outputs off:** forced on per-run; assert `results.ssb()`/`results.yield_biomass()`
  non-empty, else raise with a clear message.
- **Multi-peak / plateau yield curve:** `multi_peak=True` + caveat; report the global max but warn the
  reference point is ambiguous.
- **No interior peak:** max at first point → no Fmsy + caveat; max at last point → `fmsy_at_boundary`
  + caveat (extend grid).
- **Not converged in `n_years`:** grid point flagged `not_converged`; if the Fmsy point is unconverged,
  lower-confidence caveat.
- **B0 ≤ 0 / failed run:** no Blim / drop the point with a warning; too few points to find a peak → no
  Fmsy + caveat. The sweep never crashes the whole batch on one bad run.
- **Selectivity basis:** Fmsy reported on the realized exploited-stage F basis (matches the page's F),
  not the nominal grid value.
- **No model sidecar present:** the page behaves exactly as today (user/ICES) — purely additive.

## 8. Testing

- `derive_reference_points` (pure, fast): synthetic single-peak curve → interior Fmsy (grid-argmax realized F,
  vertex clamped to bracket); monotone-increasing → `fmsy_at_boundary`; monotone-decreasing → no Fmsy +
  caveat; two-peak curve → `multi_peak`; B0/Blim from the F=0 point; B0≤0 → no Blim.
- `_fishing_override` (the no-op trap): for a **fisheries-mode** config (build a tiny one with
  `nfisheries>0`) it returns a `fisheries.rate.base.fsh{j}` key, and applying it CHANGES
  `EngineConfig.from_dict(...).fishing_rate[i]` (assert before/after); for a **legacy-mode** config it
  returns `mortality.fishing.rate.sp{i}` and that changes `fishing_rate[i]`. This is the regression
  guard against the original critical bug.
- `run_yield_f_sweep` (integration, a TINY fast config, few species, small grid, 1 replicate): produces
  curves; the F=0 run has the largest SSB; higher F lowers SSB; `results.yield_biomass`/`results.ssb`
  are non-empty (the forced-output + correct-reader guard); the override touches only the swept species.
- `fisheries_reference`: model sidecar loads; **precedence** user > model > ICES; `b_ref_kind="bmsy_model"`,
  `b_ref_label="Bmsy [model]"`; `save` never writes model values.
- CLI: writes a valid sidecar (with `_meta`) for a tiny config that `load_reference_points` then reads.
- UI: `build_fisheries_view` with a model sidecar → `kobe_ready` True with no user input; source column "model".
- No engine/dynamics change → EEC/BoB parity suites untouched (the sweep only RUNS the engine with
  varied fishing config + output flags, as calibration already does).

## 9. Out of scope (deferred)

- UI "Compute Fmsy/Blim" button + background-job progress (the CLI is v1).
- Whole-system F-multiplier (mutual-MSY) sweep; per-fishery / multi-gear reference points; shared-fishery
  per-species decomposition beyond the caveat path.
- Climate/environment-conditioned reference points (Fmsy shifts with ecosystem state).
- Auto-extending the grid on a boundary Fmsy (v1 warns; user re-runs wider).
- **Sub-grid parabolic Fmsy refinement + Bmsy interpolation** (v1 reports the grid-argmax point's
  realized F + SSB; incoherent with the realized-F basis and a minor precision gain on a 7-point grid).

## 10. Scientific basis & caveats

Model-internal reference points are the established ecosystem-model practice — single-stock reference
points are not simultaneously achievable in an interacting system and shift with ecosystem/climate state
(Travers-Trolet et al., 2020, 10.3389/fmars.2020.568232; Mackinson et al., 2018, 10.1371/journal.pone.0190015;
Briton et al., 2019). Fmsy = yield-maximising F and Blim ≈ 0.2·B0 follow Mackinson et al. Surfaced caveats:
per-species **conditional** (others at baseline F — not mutual-MSY); `b0` is conditional-unfished, not
ecosystem-unfished; multi-peak/boundary/non-converged flags; realized-F basis. The assembled Kobe is an
indicative, internally-consistent snapshot, not a multi-species optimum.
