# Salinity/oxygen forcing + spatial cod egg-survival (SP1) — design

**Date:** 2026-07-02
**Status:** DRAFT — revised after in-loop review (round 1); awaiting user review. Judgment calls flagged in §12.
**Sub-project:** SP1 of the "salinity/oxygen as first-class environmental state" program. SP2 (bioenergetics O2 coupling) and SP3+ (broader coupling) are separate later specs that reuse SP1's forcing layer.
**Supersedes (behaviourally):** the scalar reproductive-volume recruitment gate (`osmose/engine/processes/recruitment_gate.py`, PR #96), which did not stabilise the cod overshoot in either mode. This replaces the basin-integrated scalar with a physically-correct, per-cell spatial mechanism.

## 1. Motivation

The OSMOSE Baltic model has **no salinity or oxygen state**, so cod recruitment is blind to the deep-basin reproductive volume (RV) that governs eastern-Baltic-cod year-class strength. Beyond the (failed) stabilisation attempt of PR #96, the model is **physically incomplete**: it cannot represent that cod eggs — which are pelagic and float at their neutral-buoyancy depth (the saline layer, S ≥ ~11 PSU), **not** on the anoxic seafloor — survive only where a water layer is simultaneously saline enough (≥ ~11 PSU) and oxygenated enough (≥ ~2 mL/L). This sub-project adds that environmental state and a **spatial, per-cell** cod egg-survival mechanism driven by it. **The goal is model correctness**; whether it damps the overshoot is a measured outcome, not the success criterion.

## 2. Goals and non-goals

**Goals**
- A reusable engine load path for time-varying per-cell environmental NetCDF fields (wire the unused `PhysicalData.from_netcdf`), demonstrated end-to-end on the RV field.
- A forcing generator: depth-resolved CMEMS `so`/`o2` → a per-cell **RV field** = summed thickness of the water column where S ≥ 11 PSU AND O2 ≥ 2 mL/L co-occur.
- A **spatial cod egg-survival term** in the `larva_mortality` pre-pass, driven by the RV field, **mean-anchored** so it redistributes egg survival spatially without silently lowering mean recruitment (see §5).
- **Inert by default** — feature off ⇒ engine output bit-identical.

**Non-goals (SP1)**
- No bioenergetics O2 coupling (`f_o2`) — SP2 (Baltic runs bioen off anyway).
- No wiring of generic salinity/O2 fields into any consumer — SP3+. SP1 wires the loader only for the RV field.
- No claim of stabilisation; correctness is the goal, stabilisation measured.
- No depth dimension in the engine — the vertical is collapsed to per-cell fields at forcing-generation time.
- No horizontal egg drift — eggs are gated by their origin cell's RV (existing engine constraint: egg stage is exactly one timestep).

## 3. Design overview

Three units, dependency-ordered:

1. **Forcing generation** (`osmose/forcing/`): read depth-resolved CMEMS `so`+`o2`; for each cell and step compute the RV thickness (§4). Emit a NetCDF (`time, latitude(desc), longitude`, var `reproductive_volume`), north-first (matches the engine grid; no flip), with the suggested `RV_ref` written to attrs (§5). This is offline (one heavy read of the 26 GB data; compact output).

2. **Engine load path** (`osmose/engine/config.py`): a `_load_rv_spatial(...)` helper (mirroring `_load_rv_gate`) that calls `PhysicalData.from_netcdf` for the RV field and stores the resulting `PhysicalData` (or `None` when off) as a plain `EngineConfig` field with a `= None` default, plus the per-species enable mask. This follows the **`rv_gate` config pattern** (load in `from_dict`, store on config), NOT the `temp_data` pattern (threaded param) — no process-signature churn. New `reproduction.rv.spatial.*` keys auto-validate via the AST walk of `config.py`.

3. **Spatial egg-survival consumer** (`osmose/engine/processes/natural.py`): extend `larva_mortality` (the pre-pass at `mortality.py:1829`, where `is_egg` eggs already have valid cells because movement ran first) to read `config.rv_spatial_field.get_grid(step)` at each egg's cell and apply the graded survival factor to enabled species.

### The correctness crux (why per-cell RV, not seafloor bottom)

Baltic cod eggs float within the saline layer (S ≥ ~11 PSU) at a neutral-buoyancy depth that varies by batch/female condition — not at one fixed isohaline, and never on the anoxic deep-basin floor. Reading true-bottom O2 (as the scalar diagnostic did) would wrongly kill eggs in the very basins where cod spawn. The RV field scans the **whole column** for viable water, collapsing the depth-dependent biology into a per-cell 2-D field. This is the classic Baltic-cod reproductive-volume definition (Nissling & Westin 1997; Wieland et al. 1994; Hinrichsen et al. 2008) and the core correctness improvement over the scalar gate.

## 4. The reproductive-volume field

For each cell `(y,x)` and time `t`, from depth-resolved CMEMS `so(z)`, `o2(z)`:
- `viable(z) = (so(z) ≥ 11 PSU) AND (o2(z) ≥ 89.3 mmol/m³)` (2 mL/L × 44.66; verified).
- `RV(y,x,t) = Σ over ALL viable z of layer_thickness(z)` — the **total** thickness (m) of viable water, summing every viable depth interval (not just one contiguous band). Rationale: egg neutral-buoyancy depth varies, and after inflows the profile can have more than one viable band; the sum degrades to the single-band case for the common stratified profile and correctly handles the rest. `0` where no viable water.
- **Land/NaN:** the generator masks land/invalid to NaN **before** any aggregation (do NOT route through `grid.get_var`, which does `NaN→0` and would read land as 0 PSU); land cells are written as `NaN` (the consumer never indexes them — eggs only occupy ocean spawning cells, and the consumer guards `cell ≥ 0` and `NaN→0` survival).
- Regridded to 40×50 (nearest-neighbour), north-first.

**Climatology (default):** compute RV **per year** from each year's depth profiles, then **average the per-year RV fields** (mean-of-RV, preserving the threshold nonlinearity — NOT RV-of-mean-fields), then resample to 24 biweekly steps. Interannual (696-step) is the same generator with no year-averaging; the loader cycles by file length either way (see §6 note).

## 5. Egg-survival function (mean-anchored)

Applied in `larva_mortality` to enabled species (cod sp0), **multiplicatively and mean-anchored**:

- `s_cell = clip(RV(cell,step) / RV_ref, 0, 1)` — per-cell egg-survival fraction.
- **`RV_ref` — one definition (used everywhere):** the **mean of RV over the `cod_spawning` cells whose RV > 0, across all 24 climatology steps**. The builder computes it and writes it to the RV NetCDF variable's `RV_ref` attr; `_load_rv_spatial` reads that attr when `reproduction.rv.spatial.ref ≤ 0` (the default), else uses the config value; a missing attr with `ref ≤ 0` is a load error. Anchoring here makes `s_cell` centre near 1 over the cells where eggs actually are, so the term **redistributes** survival spatially (viable cells ~1, poor cells < 1) rather than uniformly lowering it.
- Application: `surviving_eggs[i] = eggs[i] × s_cell(cell_i)` for egg schools `i` of enabled species — multiplies **only** by `s_cell`; the existing `larva_mortality` constant rate is applied by the same pass exactly as today (no re-application of baseline). Non-egg schools and non-enabled species: unchanged.
- **Calibration interaction (measured in SP1, corrected in a follow-on).** The phase-13 constant `mortality.additional.larva.rate.sp0` was fit as a static long-term average that *already absorbs* the mean RV effect. Even mean-anchored, the [0,1] clip truncates the high tail, so enabling `s_cell` lowers the realised mean somewhat (`mean(s)` over spawning cells is ≤ 1). **SP1 does NOT recalibrate** — it ships the correct spatial mechanism and **measures** the mean shift honestly. Restoring the mean by re-fitting `mortality.additional.larva.rate.sp0` (a run-in-the-loop 1-D calibration needing new tooling) is **deferred to follow-on sub-project SP1b** (§13). This split keeps SP1 a single, TDD-able build plan and does not gate it on a calibration loop.

**Consumer guards:** index the RV grid only for schools with `cell_x ≥ 0` and not `is_out` (movement can, in edge cases, leave a school unplaced or out-of-domain); such schools get `s_cell = 1` (unaffected).

**Seeding-skip (concrete plumbing):** the `seeded_this_step` flag in `reproduction.py:123` is local and one step earlier, so it is NOT visible in `larva_mortality`. SP1 adds a boolean field `from_seeding` on `SchoolState`, set `True` when an egg school is created from seeded (not real-SSB) biomass in `reproduction.py`, and the spatial term skips schools with `from_seeding == True`. (This new state field defaults `False`, is carried through `SchoolState` like other per-school arrays, and does not affect the inert-off path.)

## 6. Configuration keys (schema-driven, lowercase dot-separated)

| key | type | default | meaning |
|---|---|---|---|
| `reproduction.rv.spatial.enabled` | bool | `false` | master switch |
| `reproduction.rv.spatial.field.file` | path | `""` | RV NetCDF forcing file |
| `reproduction.rv.spatial.field.varname` | str | `reproductive_volume` | variable name in the NetCDF |
| `reproduction.rv.spatial.ref` | float | `-1` → field attr | `RV_ref` (m); `≤0` means "use the builder's attr value" |
| `reproduction.rv.spatial.species.enabled.sp{idx}` | bool | `false` | per-species enable (cod on) |

`reproduction.rv.spatial.ref` default `-1` (≤0) means "read the `RV_ref` attr the builder wrote to the RV NetCDF" (§5); `_load_rv_spatial` opens the file to read that attr (`PhysicalData.from_netcdf` returns only the array), erroring if `ref ≤ 0` and no attr.

Note: there is **no `nsteps.year` key** — `PhysicalData.get_grid` cycles by the file's own time length (`step % n_time`), so a 24-step file cycles yearly and a 696-step file spans 29 years automatically. The loader asserts the field's time length is a positive multiple of the run's `ndtperyear` (else raise: a mis-stepped field would silently misalign). For SP2/SP3, only **`oxygen.*`** NetCDF keys are already Java-allowlisted (`config_validation.py:221-225`); **no `salinity.*` keys exist yet** (they'd be added by SP2/SP3). Either way these are **reserved, not loader-wired, in SP1**.

Fail-fast on load (master on): missing/unreadable file; field grid ≠ engine grid (40×50); time length not a positive multiple of `ndtperyear`; NaN in a `cod_spawning`-mapped cell (the loader reads `cod_spawning.csv` via `_load_spatial_csv` to get the mask — §7); `ref ≤ 0` with no `RV_ref` attr; no species enabled.

## 7. Components and files

**New**
- `osmose/forcing/reproductive_volume.py`: `build_rv_field(phy_ds, bgc_ds, grid, *, sal_thresh, o2_thresh, ocean_mask, spawning_mask) -> xr.Dataset`. A **new depth-integration routine** (not a generalisation of the diagnostic's `bottom_slice`, which only reads the deepest level and uses 2-D `o2b`): it consumes **full depth-resolved** `so(z)` and `o2(z)`, sums viable-layer thickness per cell (§4), regrids, averages per-year RV to a 24-step climatology, computes the suggested `RV_ref` = mean RV over the **RV>0** `spawning_mask` cells (the one definition, §5), and writes it to the `reproductive_volume` variable's `RV_ref` attr. CLI emits `data/baltic/forcing/baltic_rv_field.nc`.

**Modified**
- `osmose/engine/config.py`: `_load_rv_spatial(cfg, n_species) -> tuple[PhysicalData|None, NDArray[bool]|None]` (mirrors `_load_rv_gate`). It: opens the RV NetCDF via `PhysicalData.from_netcdf(varname=...)`; reads the `RV_ref` **attr** from the file (a separate `xr.open_dataset` — `from_netcdf` returns only the array) when `reproduction.rv.spatial.ref ≤ 0`, erroring if the attr is absent; builds the per-species enable mask; and for the on-load NaN validation reads `data/baltic/maps/cod_spawning.csv` **directly via the existing `_load_spatial_csv` helper** (the `MovementMapSet` is not built until `simulate.py`, so the loader derives the spawning-cell mask itself). The expected 40×50 grid shape for the field-grid check is taken from that loaded `cod_spawning` mask's shape (there is no grid-dims config key to compare against). Two new `EngineConfig` fields, **both defaulted** so `_minimal_config` needs no change: `rv_spatial_field: PhysicalData | None = None`, `rv_spatial_enabled: NDArray[np.bool_] | None = None`. Both placed after the existing defaulted-field block (dataclass ordering). Called in `from_dict`.
- `osmose/engine/state.py`: add `from_seeding: NDArray[np.bool_]` to `SchoolState` (frozen dataclass; `replace`/`append`/`compact` iterate `fields()` generically, so only `create()` needs an explicit initializer, default all-`False`).
- `osmose/engine/processes/reproduction.py`: set `from_seeding=True` on egg schools created for a species whose `seeded_this_step[sp]` is true (the flag is per-species and in scope at the egg-creation loop).
- `osmose/engine/processes/natural.py`: `larva_mortality` applies mean-anchored `s_cell` to enabled species' egg schools with `cell_x ≥ 0` AND `not is_out` AND `from_seeding == False`; any excluded school or NaN-RV cell → `s_cell = 1` (unaffected).
- `scripts/baltic_rv_overshoot_diagnostic.py` (or a sibling): extend to build the RV field, map it, compute the basin-contrast + within-basin-CV metrics, and record the mean-shift; writes to `docs/diagnostics/` (mirrors the `rv_gate_effect.md` precedent).
- `osmose/schema/species.py`: add the 5 `reproduction.rv.spatial.*` `OsmoseField`s to `SPECIES_FIELDS` (mirroring the `reproduction.rv.gate.*` fields added in PR #96) — bool `enabled`, file-path `field.file`, str `field.varname`, float `ref`, indexed bool `species.enabled.sp{idx}`. (Schema registration + the AST walk of `config.py` both cover validation.)
- `tests/test_engine_config_validation.py`: no change — both new `EngineConfig` fields have `= None` defaults.

## 8. Data flow

```
CMEMS so/o2 depth-resolved (26 GB) → build_rv_field (per-cell viable-thickness sum;
   per-year RV → 24-step climatology; RV_ref = mean over RV>0 cod_spawning cells → attr)
 → data/baltic/forcing/baltic_rv_field.nc  (var reproductive_volume, north-first, RV_ref attr)
config (reproduction.rv.spatial.*) → EngineConfig._load_rv_spatial → PhysicalData + enable mask
larva_mortality(step): for enabled egg schools with cell≥0 AND not is_out AND not from_seeding:
   s = clip(RV_grid[cell]/RV_ref, 0, 1); eggs *= s
 → re-run diagnostic: RV map + within-basin heterogeneity + measured mean-shift + cod response
   (docs/diagnostics/).  [mean-restoring larval-M recalibration = follow-on SP1b, §13]
```

## 9. Testing

**Unit**
- `build_rv_field`: synthetic profiles — (a) fresh oxygenated surface + saline anoxic bottom + a viable mid-layer → thickness = the mid-layer; (b) **two separated viable bands → sum of both**; (c) fully-fresh → 0; (d) fully-viable → full depth; (e) land column → NaN (not 0). Verify per-year-then-average ≠ average-then-RV on a two-year toy input.
- Load path: `from_netcdf` → `(time,40,50)` grid; `get_grid(step)` cycles by file length; **wrong-grid file raises**; missing file raises; time-length-not-multiple-of-ndtperyear raises; disabled → field `None`.
- Egg-survival: fabricated RV grid + cod egg schools at known cells → cod eggs scaled by `clip(RV/RV_ref,0,1)`; non-cod / non-egg / `from_seeding` / `cell<0` / `is_out` schools unchanged; NaN-RV cell → `s=1`. (The `mean(s) ∈ [0.6,1.0]` mean-anchor assertion is under Correctness below.)

**Regression / parity**
- Feature off (default): Baltic/EEC/BoB bit-identical (the `is None` guard adds no computation; the new `from_seeding` state defaults False and is inert unless read). Cross-engine/parity green. Determinism under `simulation.rng.fixed`.

**Correctness demonstration (concrete thresholds; provisional values set from the shipped field, recorded in the test)**
- **Basin contrast (assertion):** `mean(RV over cod_spawning cells) ≥ 3 × mean(RV over the fresh-coastal set)`, where the **fresh-coastal set** is defined concretely as the ocean cells whose climatological surface salinity < 7 PSU (the Gulf-of-Bothnia / coastal band); the builder writes this mask (or the diagnostic constructs it from the salinity field) so the test has a defined reference. Ratio ≥ 3 is the provisional go value; the actual ratio is recorded.
- **Within-basin heterogeneity (go/no-go assertion):** the coefficient of variation of RV *across `cod_spawning` cells* (mean over the 24 steps, shipped regridded climatology) is **≥ 0.20** — i.e. the field is not near-uniform within the basins. If it is below 0.20, the spatial machinery adds nothing over the failed scalar gate; this is the explicit go/no-go signal (record the value even on a no-go).
- **Mean-anchor check (assertion):** `mean(s_cell) over RV>0 cod_spawning cells ∈ [0.6, 1.0]` — confirms RV_ref centres the survival factor near 1 (the upper bound is the clip ceiling; the 0.6 floor bounds the residual shift SP1b will correct).
- **Mean-shift measurement (recorded, not pass/fail):** report run-window mean cod recruitment with the term on vs off — the residual that SP1b's recalibration will restore.

## 10. Success criteria

1. **Correctness:** RV represented as a per-cell field; cod egg survival spatially gated by it; the basin-contrast assertion passes AND within-basin heterogeneity ≥ 0.20 (the spatial resolution does real work — the go/no-go for the whole approach).
2. **Mean-anchored:** `mean(s_cell)` over spawning cells ∈ [0.6, 1.0]; the residual mean-recruitment shift (term on vs off) is measured and recorded. (Restoring the mean to within ±10% via larval-M recalibration is **SP1b**, not an SP1 gate.)
3. **Inert by default:** parity suite green; bit-identical off.
4. All unit tests pass; ruff + pyright clean on changed files.
5. RV field regenerable from CMEMS; mechanism documented. Effect on the overshoot measured and recorded honestly (secondary).

## 11. Risks and flags

- **Within-basin uniformity risk:** if the 24-step regridded climatology washes out sub-basin RV structure, the mechanism is spatially decorative (functionally the failed scalar). §9's within-basin CV test is the go/no-go; if it fails, reconsider interannual forcing or finer handling before investing further.
- **Calibration double-count:** mean-anchored `RV_ref` (§5) minimises the mean shift; the residual is measured in SP1 and corrected by SP1b's larval-M recalibration (§13). If SP1b is skipped, cod recruitment runs somewhat below the historical fit — the same mean-shift direction as the scalar gate, but bounded by the mean-anchor (mean s ≥ 0.6), not the full ~0.4× the scalar raw_cap imposed.
- **No horizontal egg drift AND one-step egg residence:** eggs are gated by their **origin cell's** RV at a **single** step; real eggs both develop over ~2 weeks and drift horizontally between viable/non-viable water (Hinrichsen drift studies). Both are structural approximations inside the existing one-timestep-egg engine convention; deferred to a future refinement.
- **Threshold non-stationarity:** the 11 PSU buoyancy threshold is optimistic for the post-2015, lower-condition eastern-Baltic-cod population (denser eggs need higher salinity to float); the config default matches the pre-collapse parameters the model otherwise uses. Config-overridable; note for a future sensitivity check.
- **Depth-resolved CMEMS cost:** the full-column read of 26 GB is one-time/offline in the builder, not per-run.

## 12. Judgment calls made while the user was away (please confirm)

1. **Egg environment = whole-column viable-thickness sum** (not seafloor bottom, not a single band) — physically correct; science-review-validated.
2. **Mean-anchored** spatial survival (RV_ref = mean of RV over RV>0 cod_spawning cells) to minimise the mean shift; the mean-**restoring** larval-M recalibration is split out to follow-on **SP1b** (keeps SP1 a single build plan). (Alternative: raw clip with global 90th-pctile RV_ref — rejected as re-introducing the double-count.)
3. **24-step climatology** (per-year RV then averaged), interannual optional.
4. **Cod-only** consumer; generic salinity/O2 fields deferred (loader reserved, not wired) to SP2/SP3.
5. **Correctness, not stabilisation**, is the success metric; the within-basin-heterogeneity go/no-go decides whether the spatial machinery earns its complexity.

## 13. Out of scope (follow-on sub-projects)

- **SP1b (immediate follow-on): mean-restoring recalibration.** Re-fit the scalar `mortality.additional.larva.rate.sp0` (a 1-D, run-in-the-loop calibration — monotone in recruitment, read from the `Recruits`/biomass output) so run-window mean cod recruitment returns within ±10% of the pre-feature baseline with the spatial term on. Needs a small calibrate-one-scalar harness (none exists today). Depends on SP1.
- SP2: wire the O2 field into the `f_o2` bioenergetics dose-response.
- SP3+: salinity/O2 in movement/habitat; egg survival for other species; multi-step egg residence + horizontal egg drift.
