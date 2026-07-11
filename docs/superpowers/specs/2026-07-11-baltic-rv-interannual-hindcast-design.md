# Baltic cod interannual reproductive-volume hindcast — design spec

**Date:** 2026-07-11
**Status:** approved (design), pending implementation plan
**Related:** SP1 spatial egg-survival (`osmose/forcing/reproductive_volume.py`, merged PR #97), the RV-vs-overshoot diagnostic (`scripts/baltic_rv_overshoot_diagnostic.py`, `docs/diagnostics/baltic_rv_fraction.csv`), `[[project-baltic-salinity-spawning-followup]]`, `docs/baltic_habitat_followup_2026-07-02.md`.

## Goal

Two deliverables:

1. **Reusable interannual reproductive-volume forcing** — extend the existing (stationary, 24-step *climatology*) cod egg-survival gate to a real **1993–2021 interannual** RV field driven by the o₂+so reanalysis already on disk. This is the capability that unlocks environment-driven cod studies (hindcasts and, later, climate/deoxygenation projections).
2. **An honest A/B skill test** — does forcing cod egg-survival with the *real* interannual RV history move the cod trajectory measurably **toward** observed ICES SSB, relative to the stationary-climatology baseline, and reproduce the known MBI (Major Baltic Inflow) recovery pulses (post-2003, post-2014)?

## The honest framing (load-bearing — from a feasibility spike)

A 30-year run of the **deployed** Baltic config (no RV forcing) shows cod is an **intrinsic boom-bust**: peak ~3,580 t at yr6, monotonic decline to ~5 t by yr29; the community collapses to herring+sprat dominance by ~yr15 (percids/flounder/smelt gone by yr12–17). **Cod collapses on its own, with zero environmental forcing.**

Therefore a naive "does the RV collapse *explain* the observed cod decline?" is unanswerable — the config collapses cod regardless. The tractable, honest experiment is a **controlled A/B skill delta**:

- **Null (baseline):** deployed cod with the **stationary climatology** field → same mechanism, no interannual structure (a gate-off arm is also run as a pure-intrinsic reference).
- **Treatment:** the same config with the real interannual RV field.
- **Target:** observed ICES eastern-Baltic cod SSB.
- **Metric:** does the treatment track the target *better* than the null (correlation / skill delta), and reproduce MBI-driven features — **not** "RV explains cod."

Per the user's two scoping decisions: **soft gate** (build the forcing infrastructure regardless of Phase 0's offline result — it has standalone projection value) and **A/B vs stationary baseline, no recalibration** (report skill delta; do not tune larval-M to fit).

## Non-goals (YAGNI)

- **Do not** change the deployed `data/baltic/` config defaults. The gate ships **off**, byte-identical (like SP1). No preset, no default flip.
- **Do not** recalibrate larval mortality to fit the hindcast (the A/B-skill-delta decision).
- **Cod only.** Percids/flounder/etc. are the accepted structural residual (coarse-grid habitat limit, `[[project_percid_overshoot_diagnostic]]`); they collapse intrinsically and are excluded from the cod skill assessment.
- **Do not** re-run the SP1 climatology overshoot test (already NEGATIVE — spatial egg-survival is not the overshoot lever). This is a *non-stationary hindcast*, a different question.

## What already exists (verified) — and what the gap actually is

- **Mechanism:** `osmose/engine/processes/natural.py::larva_mortality` multiplies egg survival by `clip(RV(cell)/RV_ref, 0, 1)` for enabled, non-seeded, located eggs (via `config.rv_spatial_field.get_grid(step)`).
- **Loader:** `osmose/engine/config.py::_load_rv_spatial` — already validates `tlen % ndtperyear == 0` with the explicit comment `# NetCDF time length (24 climatology / 696 interannual)`, resolves `RV_ref` from the `ref` config or the field attr, fails on grid mismatch / NaN-at-spawning / no-species-enabled.
- **Time indexing:** `PhysicalData.get_grid(step)` returns `self._data[step % shape[0]]`. **A chronologically-ordered 696-step field therefore indexes 1:1 over a 29-year (696-step) run with no engine change.** The engine was built anticipating the interannual case.
- **Data:** 54 GB of monthly 1993–2021 depth-resolved `so` (PHY) + `o2` (BGC) reanalysis in `data/cmems_cache/cmems_downloads/` (per-year files). Offline RV series already reconstructed in `docs/diagnostics/baltic_rv_fraction.csv`.
- **Builder:** `osmose/forcing/reproductive_volume.py::build_rv_field` produces the **climatology** — it averages the per-year 24-step fields (`rv = np.mean(np.stack(per_year_24, axis=0), axis=0)`). The **gap is only here**: there is no builder that *stacks* the years chronologically instead of averaging them, and no interannual field file.

## Architecture

### Phase 0 — offline interpretation (informative, NON-blocking)

`scripts/rv_cod_offline_correlation.py`: load `docs/diagnostics/baltic_rv_fraction.csv` (real 1993–2021 RV series), pull observed ICES **eastern-Baltic cod (cod.27.24–32)** recruitment + SSB via the ICES MCP (`get_stock_assessment` / `get_reference_points`), align by year, and report lagged correlation (RV → recruitment, allowing a spawning→recruitment lag). Writes a short results doc + a figure. **Does not gate** Phases 1–3 (soft-gate decision) but frames how much environmental signal is present.
*Data caveat:* the cod.27.24–32 assessment was downgraded to data-limited (~2019) after the collapse; SSB/R post-2014 are uncertain. Phase 0 confirms what ICES actually exposes and picks the most defensible observed series; if the analytical SSB is unusable, fall back to survey-based indices (e.g. BITS CPUE via ICES) and say so.

### Phase 1 — interannual RV field builder

Add `build_rv_field_interannual(phy_years, bgc_years, grid, *, sal_thresh=11.0, o2_thresh=89.3, ocean_mask, spawning_mask, start_year)` to `osmose/forcing/reproductive_volume.py`. It reuses `_rv_year`/`regrid`/`resample_to_24` exactly as `build_rv_field` does, but **concatenates** the per-year 24-step fields chronologically instead of averaging: `rv = np.concatenate(per_year_24, axis=0)` → shape `(nyear*24, nlat, nlon)`, north-first, land→NaN. `RV_ref` = mean over RV>0 spawning cells across **all** steps. Carries a `start_year` attr (1993) — **harness metadata only** for mapping sim-step → calendar year; the engine indexes purely by `step % tlen` and never reads it. The existing `build_rv_field` is left **byte-identical** (the climatology arm of the A/B and SP1's shipped behavior both depend on it).

**Shared-`RV_ref` caveat (load-bearing for the A/B):** the climatology and interannual fields do NOT auto-share a reference — each computes `RV_ref` on its own array, and averaging (climatology) vs concatenating (interannual) changes the RV>0 cell set, so the two attrs differ. To keep the A/B a clean isolation of *temporal structure* (identical mean suppression, only interannual variability differing), the **Phase-3 harness forces one shared `RV_ref` on both arms** via the `reproduction.rv.spatial.ref` config key (`_load_rv_spatial` prefers a positive `ref` over the field attr). Use the interannual field's `RV_ref` as the shared value.

**Provenance:** the interannual field is built from the gitignored local 54 GB `so`+`o2` cache (`data/cmems_cache/cmems_downloads/`, downloaded via `scripts/download_baltic_rv_forcing.py`) — same as SP1's climatology. The committed ~6 MB NetCDF is the reproducible artifact; rebuilding it requires the local cache (not a bare clone). The build script documents this.

**Tie-back validation (de-risk):** the built field's **spawning-cell-mean RV series** (per step, over the cod_spawning cells) must track the already-validated offline series in `docs/diagnostics/baltic_rv_fraction.csv` (which was independently validated against MBI history). The build script asserts a strong positive correlation between the two annual series — if the new engine field doesn't reproduce the offline signal, it is silently wrong and the build fails. This anchors the new per-cell field to the validated basin-scale signal.

Build script: extend `scripts/build_baltic_rv_field.py` with an `--interannual` flag → reads the 29 per-year `so`+`o2` files in chronological order, calls `build_rv_field_interannual`, writes `data/baltic/forcing/baltic_rv_field_interannual.nc` (696, 40, 50). At float32 this is ~6 MB — **commit it** (consistent with the other committed Baltic forcing NetCDFs: `baltic_ltl_biomass.nc`, `baltic_salinity_bottom_climatology.nc`, the SP1 `baltic_rv_field.nc`); it is the reusable deliverable, not a scratch artifact. Orientation verified north-first (reuse SP1's `np.flipud`/mask conventions and the map-orientation gotcha from the RV diagnostic).

### Phase 2 — engine safety guard + config wiring (minimal)

The engine already indexes correctly; the only real risk is **silent temporal wrap**: if a run's `nyear*ndtperyear` exceeds the interannual field length, `step % tlen` silently repeats 1993 instead of failing. Add a fail-fast at **loader time** in `_load_rv_spatial` (it already holds `cfg`, hence `simulation.time.nyear` and `ndtperyear`): when the field is interannual (`tlen > ndtperyear`) and `nyear*ndtperyear > tlen`, raise (the run would wrap past the forcing period). `nyear*ndtperyear == tlen` is fine (exact 1:1). This fails at config load, not mid-run. **Byte-identical when the gate is off** (`_load_rv_spatial` returns early → no new path; parity gate over the existing engine suite).

Config wiring for the hindcast lives in a **harness/overlay**, not in `data/baltic/`: point `reproduction.rv.spatial.field.file` at the interannual NC, `reproduction.rv.spatial.species.enabled.sp0=true`, `simulation.time.nyear=29`. Deployed defaults untouched.

### Phase 3 — A/B hindcast harness + validation

`scripts/baltic_rv_hindcast.py`: for a seed ensemble, run the deployed cod config over 1993–2021, identical except the RV field. **Primary A/B isolates the interannual signal:**
- **Arm A (null):** the **stationary climatology** field (`baltic_rv_field.nc`), with `reproduction.rv.spatial.ref` set to the **shared** `RV_ref` (see the shared-`RV_ref` caveat in Phase 1) — same mechanism and reference as arm B, but *no* interannual structure.
- **Arm B (treatment):** the **interannual** field (`baltic_rv_field_interannual.nc`), same shared `RV_ref`.
Because both arms share the mechanism, the harness-forced shared `RV_ref`, spin-up, and seeding transient, their **difference isolates the interannual variability** (including its nonlinear rectification through `clip`) and the skill *delta* is robust to the intrinsic boom-bust (it cancels). A third **gate-off** arm is run as the pure-intrinsic reference (no RV mechanism at all), for context only.

Extract modeled cod SSB per year (`OsmoseResults`), map sim-year-0 → 1993. **The usable window is constrained by the intrinsic collapse:** the yr1–6 boom is a seeding transient (shared by all arms), and cod is near-zero by ~sim-yr20, so the skill window is roughly **sim-yr6–15 (≈ 1999–2008)** where cod has dynamic range. Consequence for the feature test: the **2004 MBI pulse (≈ sim-yr11) is testable**, but the **2016 MBI (≈ sim-yr23) is NOT** — cod has already collapsed there and cannot respond; report that as a stated limitation, not a null result. Compare to the Phase-0 observed cod series over the usable window:
- Skill delta: `corr(B, observed) − corr(A, observed)`, across the ensemble (report CI over seeds).
- Feature test: does arm B (but not arm A, which has no interannual signal) show a recruitment/biomass uptick lagging the **2004** MBI-driven RV pulse?
Writes `docs/diagnostics/baltic_rv_hindcast.md` + figure. Honest verdict: the skill delta with seed CI, the intrinsic boom-bust acknowledged as the null. A null/negative result (interannual RV adds no skill over the climatology) is a legitimate outcome and must be reported as such.

## Testing

- **CI-safe unit tests:** `build_rv_field_interannual` shape/ordering (concatenate not average; `(nyear*24, ny, nx)`; chronological — year k's block equals that year's standalone RV), `RV_ref` convention, land→NaN; the Phase-2 wrap guard raises when `nyear*ndt > tlen` and is inert otherwise; the climatology `build_rv_field` remains byte-identical (regression). Engine **parity**: gate-off byte-identical vs current (the existing RV parity gate).
- **NOT a CI gate:** the emergent hindcast outcome (skill delta, biomass trajectories) — non-reproducible across CI cores (`[[feedback-ci-fragile-emergent-tests]]`). Phases 0 and 3 are local, documented runs.

## Success criteria

1. `build_rv_field_interannual` produces a correct chronologically-ordered 696-step field; the climatology builder is unchanged (byte-identical).
2. The engine runs the interannual field over 29 years with correct 1:1 year alignment, and **fail-fasts** rather than silently wrapping if a run exceeds the forcing period; gate-off parity byte-identical.
3. Phase 0 reports the offline RV↔observed-cod correlation with the honest data caveat.
4. Phase 3 reports an A/B skill delta (with seed-ensemble CI) and the MBI-pulse feature test, framed as skill-vs-null — not overclaiming that RV explains the cod collapse.
5. Deployed `data/baltic/` defaults byte-unchanged; the gate ships off.
