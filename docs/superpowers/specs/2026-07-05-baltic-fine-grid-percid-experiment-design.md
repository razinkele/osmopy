# Baltic finer-grid percid-habitat deciding experiment — design (v2)

**Date:** 2026-07-05 (v2 — revised after a 5-lens adversarial workflow review; 1 critical + 8 important
findings folded in, see §12).
**Status:** Design (approved in brainstorming; revised post-review) — awaiting user review before writing-plans.
**Topic:** An alternative finer-grid (4×) Baltic OSMOSE config whose perch/pikeperch habitat is derived
from *real* high-resolution bathymetry (EMODnet) + a spawning-stage salinity gate, run as a 3-rung
attribution ladder to decide whether resolving real coastal percid habitat cures the structural percid
overshoot.

---

## 1. Motivation & context

The Baltic percid overshoot (×38–96 vs ICES/HELCOM) is **structural**: eight recruitment- and
mortality-side levers are ruled out (`docs/baltic_percid_overshoot_conclusion_2026-07-05.md`). Two
independent probes this session — the thermal recruitment gate (heavy recruitment cut → perch −3.9%) and
a cannibalism A/B (heavy mortality increase → perch −1.4%) — show perch is limited by neither recruitment
nor mortality, i.e. by the **standing stock its coarse-grid habitat supports** (carrying capacity, which
in OSMOSE is emergent from spatial prey availability in occupied cells).

The SP-B grid spike (`3b3cde5`, never merged) tried a 2× grid but **block-upsampled everything** — added
cells while preserving the blocky coarse coastline and the exact percid footprints, so it tested "more
cells, same habitat" (which failed) and could **not** test "a real finer grid with genuine estuary/lagoon
habitat detail." Its banked recommendation was this experiment. This v2 additionally derives habitat from
a **real high-resolution bathymetry source** rather than the CMEMS validity mask, so the littoral fringe
is genuinely resolved, not synthesised.

**Engineering facts (verified against the current tree during review):** the grid is parameterised via
`GridSpec.from_config` (`osmose/maps/builder.py`); OSMOSE aligns forcing NetCDFs by **cell index, not
coordinate** (matching shapes suffice); CMEMS BAL (~2 km) auto-regrids for salinity/thetao/LTL. **The
SP-B upsample/staging scripts do NOT exist in the tree or git history** — the block-replicate upsampler
(component [C]) and its tests are **built from scratch**, not reused.

## 2. Goals & non-goals

**Goals**
- An **alternative** 4× Baltic config (`nlon=200, nlat=160`, same extent), built from scripts, leaving
  `data/baltic/` untouched.
- Perch/pikeperch maps derived from **real EMODnet bathymetry** (sub-grid shallow-littoral fraction per
  cell) with a **spawning-stage salinity gate**, at 4×.
- A **3-rung attribution ladder** (coarse baseline → 4×-upsampled → 4×-real-percid) separating the
  resolution effect from the habitat-detail effect.
- A clear **GO / NO-GO gate** deciding whether to invest in the full high-res Baltic model.

**Non-goals**
- No recalibration of species parameters (the ladder isolates the habitat effect; recalibration is
  deferred to the full build, gated on a GO).
- No re-derivation of the other 6 species' habitat maps (block-upsampled — the target is percid-specific).
  The resulting predator/prey footprint asymmetry is a documented caveat (§11 R5), conservative for a
  GO/NO-GO screen (biases percids downward → toward GO, cannot manufacture a false NO-GO).
- No change to shipped `data/baltic/`, the engine, or any merged feature.

## 3. Architecture overview

Five independently-testable units + a new config tree. Pure builders (unit-tested on synthetic grids)
produce the spatial inputs; the ladder runner assembles and runs the three configs.

```
EMODnet bathymetry (~115 m)  ─▶ [A] grid+bathy builder ─▶ mask (200×160) + per-cell shallow-fraction
CMEMS BAL (~2 km: so, thetao, LTL, predators)                     │
   │                              ┌──────────────────────────────┴──────────────────────────┐
   ├─▶ [B] percid habitat builder (shallow-fraction + spawning salinity gate) ─▶ REAL perch/pikeperch maps
   ├─▶ [C] upsampler (block_replicate for maps; block_conserve for biomass) ─▶ other-species maps + UPSAMPLED percid maps
   └─▶ [D] forcing regridder (CONSERVING for absolute-biomass fields) ─▶ 4× LTL / predator / salinity / thetao NetCDFs
                                                                                 │
   [E] ladder runner: 3 configs, MULTIPLE seeds, existing params ─▶ overshoot table (+stability)
```

## 4. Component detail

### 4.1 [A] Grid + bathymetry builder (`scripts/build_baltic_fine_grid.py` + pure core)
- Same lat/lon extent as coarse Baltic, `nlon=200, nlat=160` (4×; ~5.6 km lon × ~8.3 km lat).
- **Bathymetry from EMODnet Bathymetry** (~115 m Baltic DTM, freely available), NOT CMEMS bottom-level.
  For each 4× grid cell compute, from the ~115 m sub-cells it contains: (i) `ocean` iff any sub-cell is
  wet; (ii) **`shallow_fraction`** = fraction of the cell's wet sub-cells with depth `< D_sp` (per-species
  littoral depth). This sub-grid fraction captures sub-km lagoons/flads that a binary 4× threshold or the
  CMEMS validity mask would miss (review-critical fix). Emits `data/baltic-fine/baltic_fine_grid.nc` +
  mask CSV + a per-species `shallow_fraction` array consumed by [B].
- **Fallback if EMODnet ingest proves impractical:** GEBCO 2024 (~450 m) — coarser but still ≫ CMEMS and
  covers the lagoons. The choice is isolated in [A]; [B] onward is source-agnostic.

### 4.2 [B] Percid habitat builder (`osmose/forcing/percid_habitat.py` pure core + script)
- **Adult / juvenile maps:** occupancy weight = `shallow_fraction` (from [A]) — i.e. the realistic
  fraction of each cell that is shallow littoral, **depth-dominant**. A *relaxed* adult salinity ceiling
  is applied only to exclude fully-marine cells (perch `< 12 PSU`, pikeperch `< 14 PSU` — physiological
  adult limits per Christensen et al. 2019, NOT the early-life figures). Juvenile uses the same rule with
  its own (slightly shallower) `D_sp` (§7) — juvenile/adult differentiation preserved.
- **Spawning maps:** the salinity gate lives HERE (where it is physiologically defensible): spawning
  weight = `shallow_fraction(depth < D_spawn)` AND `bottom_salinity < S_spawn` (perch larvae fail
  >~9.6 ppt → `S_spawn=6`; pikeperch eggs restricted <~5 PSU → `S_spawn=5`). Bottom salinity = the
  **annual-mean** of the re-derived 4× bottom-salinity climatology (time-collapse made explicit).
- **Land/normalisation:** land cells use the existing **−99 sentinel** (not 0); occupancy normalisation
  follows the existing map convention.
- **Vacuity guard:** assert each percid habitat is non-empty AND materially smaller than the upsampled
  footprint. The bar is set from the mechanism, not an arbitrary 70%: the guard requires the real habitat
  area be **≤ ~40%** of the upsampled footprint (a meaningful area cut) — and the runner reports the
  actual area ratio so the biomass response can be read against it (§5).

### 4.3 [C] Upsampler (`scripts/baltic_grid_upsample.py` — BUILT NEW, not reused)
- **Occupancy maps** (movement maps, values in {−99, 0, weight}): `block_replicate` to 4× (each coarse
  cell → 16 fine cells, same value). Used for the 6 non-percid species (both rungs) and the block-upsampled
  percid set (control rung 2).
- Unit-tested from scratch (SP-B's tests do not exist).

### 4.4 [D] Forcing regridder (reuse `osmose/forcing/` primitives + a conservation wrapper)
- **CRITICAL correctness rule:** distinguish field types on regrid to 4×:
  - **Absolute-biomass fields** (`baltic_ltl_biomass.nc`, `baltic_predator_biomass.nc` — tonnes/cell):
    **CONSERVE total** — each coarse cell's biomass is *split* across its 16 sub-cells (÷16 for a uniform
    split, or area-weighted), NOT replicated. Replication would inflate total system biomass ×16 and
    silently confound the entire ladder.
  - **Intensive fields** (bottom-salinity, thetao — per-cell concentrations): regrid/replicate as-is.
- A test asserts total biomass is conserved (sum over fine grid == sum over coarse) for every
  absolute-biomass field. All outputs shaped `(…, 160, 200)`.

### 4.5 [E] Ladder runner (`scripts/baltic_fine_grid_ladder.py`)
- Three configs, each run over **≥5 seeds** at **nyear=30** (perch is chaotic/near-extinction-prone —
  a single seed / 15 yr cannot separate a stable lower CC from stochastic collapse; the runner reports
  per-seed spread + late-window CV so a boom/bust collapse is not misread as a "cure"):
  1. **coarse baseline** — existing `data/baltic/` (known ×38–96).
  2. **4×-upsampled** — 4× grid; ALL species (incl. percids) block-upsampled.
  3. **4×-real-percid** — identical to (2) except perch/pikeperch use the real EMODnet maps.
- Reports per-species overshoot ratio (mean ± seed spread) + in-envelope count, foregrounding
  perch/pikeperch and the real-habitat **area ratio** (§4.2), plus the high-weight guardrail
  (cod/herring/sprat).

## 5. Attribution ladder & success gate

- **Resolution effect = rung 1 → 2** (expected ≈ no cure, per SP-B).
- **Habitat-detail effect = rung 2 → 3** (the untested lever).
- **GO** (build the full model): rung-3 percid overshoot drops **substantially** toward range (target:
  into single digits, or ≥3/8 species in-envelope), the drop is **stable across seeds** (not a
  stochastic collapse), **AND** the real-habitat area cut is large enough to plausibly explain it (read
  against the area ratio). The high-weight cod/herring/sprat freed-prey worsening is *reported* but — per
  the 2026-06-03 finding that suppressing percids necessarily frees prey — is treated as an **expected
  cost to quantify in the recalibrated production build**, not a veto (correcting v1's unsatisfiable
  "no high-weight worsening" clause).
- **NO-GO / structural** (accept the overshoot, pivot): rung-3 ≈ rung-2 (real habitat doesn't move
  percids beyond what mere area reduction would), or the only "drop" is a stochastic collapse.

## 6. Config tree & keys

New tree `data/baltic-fine/` mirroring `data/baltic/` with `grid.nlon=200, grid.nlat=160`,
`grid.mask.file`/`grid.netcdf.file` repointed, movement-map + forcing keys repointed to the 4× files.
Two variants differ only in the perch/pikeperch `movement.file.map*` targets (upsampled vs real). No new
OSMOSE engine keys — habitat thresholds are builder CLI args.

## 7. Percid habitat thresholds (builder CLI args with defaults)

Per species and life stage; salinity gate on spawning only (§4.2). See
`docs/baltic_percid_low_salinity_refuge_literature_review_2026-07-04.md`.

| Species | adult `D` | juvenile `D` | spawning `D_spawn` | spawning `S_spawn` | adult salinity ceiling |
|---|---|---|---|---|---|
| perch (sp4) | < 12 m | < 8 m | < 6 m | < 6 PSU | < 12 PSU |
| pikeperch (sp5) | < 18 m | < 12 m | < 8 m | < 5 PSU | < 14 PSU |

Defaults; CLI-exposed so a threshold sweep is cheap if rung-3 is borderline. **Acknowledged proxy
limitation:** depth + salinity omit turbidity, wave exposure and vegetation — drivers the literature
(esp. for turbidity-loving pikeperch) rates as important. This experiment tests the *depth+salinity+
bathymetry* habitat proxy specifically; a null result is evidence against that proxy, not against all
possible habitat refinement.

## 8. Error handling (fail-fast)

- EMODnet/CMEMS coverage missing over the extent → loud error naming the source.
- **All mask + forcing + map shapes == (160, 200)** — assert on assembly (index alignment).
- Percid habitat empty OR area ratio > guard bound (§4.2) → raise (vacuous test).
- **Orientation:** percid maps + forcing must match the **movement-map CSV convention (`np.flipud`)** —
  NOT the NetCDF (no-flip) convention (review fix); one-cell sanity check against a known shallow bay.
- Every absolute-biomass forcing: assert total conserved after regrid (§4.4).

## 9. Testing strategy

- **Grid+bathy builder unit:** synthetic EMODnet-like DTM → correct ocean mask + per-cell shallow_fraction
  (incl. a sub-km "lagoon" that a binary 4× threshold would miss but the fraction captures).
- **Percid habitat unit:** synthetic shallow_fraction + salinity → correct adult/juvenile (depth-dominant,
  relaxed adult salinity) and spawning (depth+salinity gate) maps; land = −99; vacuity guard fires.
- **Upsampler unit** (new): block_replicate correctness on {−99, 0, weight} maps.
- **Forcing regrid unit:** total biomass conserved for absolute fields; intensive fields unchanged; all
  outputs (…,160,200).
- **Integration:** each of the 3 configs loads via `EngineConfig.from_dict` and runs a short sim; runner
  emits an overshoot table naming perch, pikeperch, area ratio, seed spread, and the high-weight guardrail.

## 10. Compute & feasibility

4× ≈ 16× cells; the real cost driver is **school count** (fixed total schools spread over more cells →
also affects per-cell super-individual granularity — a documented caveat, §11 R6) and the predation
cell-loop. A 30 yr run over ≥5 seeds × 3 rungs is the budget — expect tens of minutes to a few hours
total on the Python engine; acceptable for a one-off deciding experiment. If per-run cost is prohibitive,
`nyear` and seed count are the tuning levers (with the stability caveat).

## 11. Risks & caveats

- **R1 — Still may be NO-GO (structural).** If rung-3 ≈ rung-2, the depth+salinity+bathymetry habitat
  proxy is not the lever; the conclusion doc's "structural, accept it" stands. Informative, accepted.
- **R2 — Params not recalibrated (deliberate).** Isolates the habitat effect; a real GO needs a
  finer-grid recalibration (deferred).
- **R3 — Threshold + source sensitivity.** Thresholds are CLI args (sweep if borderline); EMODnet vs GEBCO
  fallback isolated in [A]; the vacuity guard prevents a silently-fat "real" habitat faking a NO-GO.
- **R4 — Ratio-to-envelope across grids.** Overshoot compared as dimensionless ratios, not absolute
  biomass, to stay grid-fair.
- **R5 — Predator/prey footprint asymmetry.** Only percids get real maps; prey stay upsampled →
  overlap-mismatch could bias percids downward. Conservative for a GO/NO-GO screen (cannot fake a NO-GO);
  fully co-refined in the production build if GO.
- **R6 — Super-individual granularity.** Fixed school count over 16× cells changes schools-per-cell;
  reported and mitigated by multi-seed runs; a full build would rescale school count.
- **R7 — Turbidity/exposure omitted.** The proxy is depth+salinity+bathymetry (see §7); pikeperch's
  turbidity dependence is unmodelled, so its "real" habitat may be mis-placed toward clear/exposed cells.
  Read pikeperch results with this caveat; perch (vegetated-littoral, better captured) is the cleaner test.

## 12. Review changelog (v1 → v2)

Folded in from the 5-lens adversarial workflow review (1 critical + 8 important, 18 total verified):
- **[critical]** absolute-biomass forcings now CONSERVED on regrid, not replicated (§4.4) — was a ×16
  inflation.
- **[important]** salinity gate moved from adults to spawning; adult salinity relaxed to physiological
  limits (§4.2/§7); juvenile thresholds specified; real EMODnet bathymetry + sub-grid shallow-fraction
  replaces the CMEMS validity mask (§4.1); ≥5 seeds × nyear=30 replaces single-seed/15 yr (§4.5);
  upsampler corrected to build-not-reuse (§1/§4.3).
- **[minor]** vacuity guard tightened + tied to the mechanism (§4.2); salinity time-collapse made explicit
  (annual mean, §4.2); land −99 convention (§4.2); orientation is CSV-flip not NetCDF (§8); super-individual
  and turbidity-proxy caveats added (§11); GO gate's high-weight clause corrected from a veto to a reported
  cost (§5).
