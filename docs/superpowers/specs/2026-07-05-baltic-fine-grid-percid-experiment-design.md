# Baltic finer-grid percid-habitat deciding experiment — design

**Date:** 2026-07-05
**Status:** Design (approved in brainstorming) — awaiting user review before writing-plans.
**Topic:** An alternative finer-grid (4×) Baltic OSMOSE config whose perch/pikeperch habitat is
derived from *real* bathymetry + bottom-salinity (not block-upsampled), run as a 3-rung attribution
ladder to decide whether resolving real coastal percid habitat cures the structural percid overshoot.

---

## 1. Motivation & context

The Baltic percid overshoot (×38–96 vs ICES/HELCOM) is **structural**: eight recruitment- and
mortality-side levers are ruled out (`docs/baltic_percid_overshoot_conclusion_2026-07-05.md`). Two
independent probes this session — the thermal recruitment gate (heavy recruitment cut → perch −3.9%)
and a cannibalism A/B (heavy mortality increase → perch −1.4%) — show perch is limited by neither
recruitment nor mortality, i.e. by the **standing stock the coarse-grid habitat supports** (carrying
capacity).

The SP-B grid spike (`3b3cde5`, NO-GO) tried a 2× grid but **block-upsampled everything** — it added
cells while preserving the blocky coarse coastline and the exact percid footprints, so it tested "more
cells, same habitat" (which failed) and explicitly **could not** test "a real finer grid with genuine
estuary/lagoon habitat detail." Its banked recommendation was precisely this experiment: build a real
bathymetry mask + real finer percid habitat and ask whether *habitat detail* (not cell count) cures the
overshoot. This spec is that deciding experiment.

**Reusable engineering facts (from SP-B):** the grid is fully parameterised (`GridSpec.from_config`);
OSMOSE aligns forcing NetCDFs by **cell index, not coordinate** (matching shapes suffice); CMEMS BAL
source is ~2 km (≫ finer than the grid), so bathymetry/salinity/thetao/LTL regrid to any finer grid;
`scripts/baltic_grid_upsample.py` (block_replicate / block_conserve_total) + `baltic_grid_spike_stage.py`
are reusable. The one manual blocker SP-B named — the 4 coastal species have no ICES point source — is
solved here for perch/pikeperch by deriving habitat from depth + salinity.

## 2. Goals & non-goals

**Goals**
- An **alternative** 4× Baltic config (`nlon=200, nlat=160`, same extent) built entirely from scripts,
  leaving `data/baltic/` untouched.
- Perch/pikeperch movement maps derived from **real** per-cell bottom depth AND bottom salinity (thin
  littoral fringe), at 4× resolution.
- A **3-rung attribution ladder** (coarse baseline → 4×-upsampled → 4×-real-percid) that cleanly
  separates the resolution effect from the habitat-detail effect.
- A clear **GO / NO-GO gate** that decides whether to invest in the full high-res Baltic model.

**Non-goals**
- No recalibration of species parameters (the ladder runs with existing coarse-calibrated params — that
  is the clean isolation; recalibration is deferred to the full build, gated on a GO).
- No re-derivation of the other 6 species' habitat maps (they are block-upsampled — only perch/pikeperch
  get real habitat, since they are the overshoot target and the hypothesis is percid-specific).
- No change to the shipped `data/baltic/` config, the engine, or any merged feature.
- Not the full production high-res model (this is the deciding experiment that gates it).

## 3. Architecture overview

Five independently-testable units + a new config tree. Pure builders (scriptable, unit-tested on small
synthetic grids) produce the spatial inputs; the ladder runner assembles and runs the three configs.

```
CMEMS BAL (~2 km: depth, so, thetao, LTL) 
   │
   ├─▶ [A] grid builder ─────▶ baltic_fine_grid.nc + mask CSV (200×160) + per-cell bottom depth
   │                                 │
   │        ┌────────────────────────┴─────────────────────────────┐
   ├─▶ [B] percid habitat builder (depth<D_sp AND salinity<S_sp) ─▶ REAL perch/pikeperch maps
   ├─▶ [C] upsampler (block_replicate) ─────────────────────────▶ other-species maps + UPSAMPLED percid maps
   └─▶ [D] forcing regridder ───────────────────────────────────▶ 4× LTL / predator / salinity / thetao NetCDFs
                                                                          │
                          [E] ladder runner: 3 configs, fixed seeds, existing params ─▶ overshoot table
```

## 4. Component detail

### 4.1 [A] Grid builder (`scripts/build_baltic_fine_grid.py` + pure core in `osmose/forcing/`)
- Same lat/lon extent as coarse Baltic (`grid.upleft 66N/10E`, `grid.lowright 54N/30E`), at `nlon=200,
  nlat=160` (4×).
- **Bathymetry:** per-cell bottom depth = deepest CMEMS depth level with valid data over that cell
  (reuse the deepest-valid-level logic from `scripts/build_baltic_salinity_forcing.py`).
- **Ocean/land mask** at 4× from CMEMS validity (a cell is ocean iff it has ≥1 valid CMEMS level).
- Outputs `data/baltic-fine/grid/baltic_fine_mask.csv` (`;`-CSV, `np.flipud` convention) +
  `data/baltic-fine/baltic_fine_grid.nc` (mask/lat/lon) + a bathymetry array consumed by [B].

### 4.2 [B] Percid habitat builder (`osmose/forcing/percid_habitat.py` pure core + script)
- **Habitat rule (the scientific crux):** a cell is perch/pikeperch habitat iff
  `bottom_depth < D_sp` AND `bottom_salinity < S_sp`, with per-species thresholds (§7). Bathymetry from
  [A]; bottom salinity from a 4×-re-derived climatology (reuse the salinity builder at the fine grid).
- Produces the per-life-stage maps the config needs: `perch_adult`, `perch_juvenile`, `perch_spawning`,
  `pikeperch_adult`, `pikeperch_juvenile`, `pikeperch_spawning`. Spawning maps use a tighter
  (shallower/fresher) sub-threshold (§7). Values are occupancy weights (1.0 in habitat cells, 0 else;
  normalisation follows the existing map convention).
- **Vacuity guard:** assert each percid habitat is non-empty AND its ocean-cell count is materially
  smaller (e.g. ≤ 70%) than the block-upsampled coarse footprint — otherwise the experiment is vacuous.

### 4.3 [C] Upsampler (reuse `scripts/baltic_grid_upsample.py`)
- Block-replicate the 6 non-percid species' coarse maps (cod/herring/sprat/flounder/smelt/stickleback ×
  life-stages) to 4×. Used in **both** 4× rungs.
- Also produce a block-replicated perch/pikeperch set (fat footprint, no detail) for the **control**
  rung (2).

### 4.4 [D] Forcing regridder (reuse `osmose/forcing/` primitives)
- Regrid to 200×160: LTL biomass (`baltic_ltl_biomass.nc`), predator biomass
  (`baltic_predator_biomass.nc`), bottom-salinity climatology, thetao (if used). Auto-regrids from CMEMS
  ~2 km. Align by cell index; all outputs shaped `(…, 160, 200)`.

### 4.5 [E] Ladder runner (`scripts/baltic_fine_grid_ladder.py`)
- Assembles three configs and runs each deterministically (fixed `movement.randomseed.fixed` +
  `stochastic.mortality.randomseed.fixed`, ≥1 seed, `nyear=15`):
  1. **coarse baseline** — the existing `data/baltic/` config (known ×38–96).
  2. **4×-upsampled** — 4× grid; ALL species (incl. percids) block-upsampled.
  3. **4×-real-percid** — identical to (2) except perch/pikeperch use the real depth+salinity maps.
- Reports per-species overshoot ratio + in-envelope count, foregrounding perch/pikeperch, plus the
  high-weight guardrail (cod/herring/sprat). Rungs 2↔3 differ only in the percid maps.

## 5. The attribution ladder & success gate

- **Resolution effect = rung 1 → 2** (expected ≈ no cure, per SP-B).
- **Habitat-detail effect = rung 2 → 3** (the untested lever).
- **GO** (build the full model): rung-3 percid overshoot drops to **single digits** (or ≥3/8 species
  in-envelope) **AND** cod/herring/sprat do not materially worsen (SP-B's freed-prey guardrail: e.g. no
  high-weight species' overshoot rises > ~20%).
- **NO-GO / structural** (accept the overshoot, pivot): rung-3 ≈ rung-2 (real habitat doesn't move
  percids), OR the only improvement comes with high-weight degradation.

## 6. Config tree & keys

New tree `data/baltic-fine/` mirroring `data/baltic/` but with: `grid.nlon=200`, `grid.nlat=160`,
`grid.mask.file=grid/baltic_fine_mask.csv`, `grid.netcdf.file=baltic_fine_grid.nc`; movement-map keys
repointed to the 4× maps; forcing file keys repointed to the 4× NetCDFs. Two variants differ only in the
perch/pikeperch `movement.file.map*` targets (upsampled vs real). No new engine config keys — habitat
thresholds live in the builder script (CLI args with defaults), not the OSMOSE config.

## 7. Percid habitat thresholds (literature-grounded; builder CLI args with defaults)

Per-species, reflecting perch as the shallower/fresher littoral spawner and pikeperch as the
turbid-lagoon species (see `docs/baltic_percid_low_salinity_refuge_literature_review_2026-07-04.md`):

| Species | `D_sp` (depth) | `S_sp` (salinity) | Spawning sub-threshold |
|---|---|---|---|
| perch (sp4) | < 10 m | < 6 PSU | depth < 6 m |
| pikeperch (sp5) | < 15 m | < 7 PSU | depth < 8 m |

Defaults; exposed as CLI args so a threshold sweep is cheap if rung-3 is borderline. Values chosen so
the resulting habitat is a thin littoral fringe (materially smaller than the coarse footprint —
enforced by the §4.2 vacuity guard).

## 8. Error handling (fail-fast)

- CMEMS coverage missing over the extent → loud `FileNotFoundError`/`ValueError` naming the product.
- **All mask + forcing + map shapes must equal (160, 200)** — assert on assembly (OSMOSE aligns by
  index; a silent shape mismatch would corrupt the run).
- Percid habitat empty OR not materially smaller than the upsampled footprint → raise (vacuous test).
- Orientation: percid maps and forcing must share the `np.flipud`/`target_coords` convention — one-cell
  sanity check against a known shallow coastal location, as in the salinity builder.

## 9. Testing strategy

- **Grid builder unit:** synthetic CMEMS-like array → correct 4× bottom-depth + mask.
- **Percid habitat unit:** synthetic depth+salinity grids → exactly the cells satisfying both thresholds;
  spawning sub-threshold tighter; vacuity guard fires when habitat is empty/too-large.
- **Upsampler:** reuse SP-B's block-replicate tests.
- **Forcing shapes:** every regridded NetCDF is (…,160,200).
- **Integration:** each of the 3 configs loads via `EngineConfig.from_dict` and runs a short sim;
  ladder runner emits an overshoot table naming perch, pikeperch, and the high-weight guardrail.

## 10. Compute & feasibility

4× ≈ 16× cells (~6k ocean of 32k) → expect ~16–30× slower than coarse; a 15 yr run in
minutes-to-tens-of-minutes on the Python engine; ladder = 3 runs × seed(s). Feasible for a deciding
experiment. (The predation cell-loop is the cost driver — see the parallel-cell-loop note in perf memory.)

## 11. Risks & caveats

- **R1 — Still may be NO-GO (structural).** If rung-3 ≈ rung-2, real habitat detail is not the lever and
  the conclusion doc's "structural, accept it" stands. This is an accepted, informative outcome.
- **R2 — Params not recalibrated (deliberate).** The ladder holds params fixed to isolate the habitat
  effect; a real GO would still need a finer-grid recalibration (deferred). A rung-3 that *reduces*
  percids without recalibration is a strong signal; a rung-3 that worsens high-weight stocks is
  confounded by the freed-prey effect and read cautiously.
- **R3 — Threshold sensitivity.** The depth/salinity thresholds drive the habitat area; they are CLI
  args so a quick sweep can bracket the result if rung-3 is borderline. The §4.2 vacuity guard prevents
  a silently-fat "real" habitat that would fake a NO-GO.
- **R4 — Coarse baseline comparability.** Rung 1 is the existing coarse config; overshoot ratios across
  grids are compared as ratios-to-envelope (dimensionless), not absolute biomass, to stay grid-fair.
