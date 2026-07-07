# Southern Benguela example bundling — design

**Date:** 2026-07-06
**Status:** In-loop review (Rounds 1–2 applied) — confirming before writing-plans
**Branch:** `feat/benguela-example`

## Goal

Add **`benguela`** (Southern Benguela, source config `osmose-ben_v4.3_Florance`) as a bundled
osmopy demo that loads and runs **stably and non-degenerate** on the osmopy Python engine, registered
in the demo registry (`osmose/demo.py`) with tests, at the rigor of the Bay-of-Biscay (BoB) migration.

Southern Benguela is a well-known Eastern Boundary Upwelling System (EBUS): 10 focal species
(euphausiids, anchovy, sardine, redeye, horse mackerel, mesopelagic, silver kob, snoek, shallow-
and deep-water hake) driven by 4 ROMS-derived plankton resource groups (sphy/lphy/szoo/lzoo). It
broadens osmopy's bundled example set beyond the current NE-Atlantic / Baltic configs with a
scientifically distinct upwelling ecosystem.

## Scope

**In scope**
- Bundle a runnable, **stable**, unfished Southern Benguela config under `data/benguela/`.
- Supply non-zero analytic seeding biomass (osmopy's supported warm-up path) so the populations
  bootstrap — the source config seeds from a NetCDF restart file osmopy cannot read.
- **Merge** the 4 single-variable ROMS forcing NetCDFs into one multi-variable file (osmopy loads a
  single resource NetCDF).
- Convert the per-species NetCDF movement maps to the CSV + `movement.steps.mapN` form osmopy reads.
- Synthesize the R-driver master (`osmose-ben.R`) into osmopy's CSV master + includes convention,
  with fishing cleanly and completely disabled.
- Smoke / stability / determinism / registry tests + ruff + pyright green.

**Out of scope (explicit follow-ups)**
- The fleet-based fishery (v1 is unfished; see "Fishing").
- Native 4.4.x conversion and cross-engine Java parity — this is a Python-engine example. osmopy
  auto-migrates the 4.3.3 keys to 4.4.0 internally on read (`_MIGRATION_CHAIN`; verified — 7
  deprecated keys migrate correctly), which is sufficient to run; no native rewrite or parity harness.
- Calibration / realism tuning beyond what's needed for a **stable, bounded** run — like BoB and EEC
  this is an uncalibrated example; the bar is "runs, stable, non-degenerate," not "matches
  observations."

## Confirmed decisions (brainstorming)

1. **Initialization** — supply analytic seeding (osmopy's supported path), starting from the
   authors' ready-made `osmose-ben_seeding.R` values and/or restart-file aggregation. *Not* building
   NetCDF restart-file support into the engine.
2. **Movement maps** — convert the NetCDF maps to osmopy CSV + `movement.steps.mapN` (full spatial
   fidelity), *not* uniform/random distribution.
3. **Engine scope** — Python-engine example only; no native-4.4.x / Java-parity work.
4. **Fishing** — disabled for v1 (and its dangling file keys stripped, see below); the fleet fishery
   is a follow-up.

## Spike findings + Round-1 review (de-risking evidence)

Run against the source clone `osmose-ben_v4.3_Florance/` on the osmopy Python engine, then verified
by three rotating reviewers against the code (file:line). **The config has multiple independent
blockers and a real dynamical-stability risk — not just the seeding gap the first spike saw.**

**Parses & grid:**
- `OsmoseConfigReader().read("osmose-ben.R")` → 800 keys, `simulation.nspecies=10`,
  `simulation.nresource=4`, no `simulation.nbackground` (= 0 background species).
- `grid-mask.nc` is `y=62, x=56`; each ROMS forcing NetCDF is `time=24, ny=62, nx=56`. Forcing steps
  (24) = `simulation.time.ndtperyear` (24). **Grid and forcing dims match** — no resample, no regrid.

**Blocker 1 — seeding (CONFIRMED).** The config seeds its entire population from a NetCDF restart
file with analytic seeding off:
```
population.initialization.file = ben-initial_conditions.nc   # 6582 schools
population.seeding.biomass.sp0..9 = 0
population.seeding.year.max = 0
```
osmopy's only seeding path is analytic warm-up (`osmose/engine/processes/reproduction.py:122-129`;
`seeding_biomass` from `population.seeding.biomass.spN` at `config.py:522-524`, `seeding_max_step`
from `population.seeding.year.max` at `config.py:528-533`). It does **not** read
`population.initialization.file` (grep: zero hits; logged as an unknown key). With analytic seeding
zeroed, nothing populates → **all-zero run**. Verified sufficient: `seeding.biomass.spN > 0` +
`year.max > 0` is enough — no hidden gating key (no relativebiomass flag, no required season file).
- **Recovery:** the source clone already ships `osmose-ben_seeding.R` — an analytic-seeding variant
  with `population.seeding.year.max = 30` and non-zero per-species seeding
  (`sp0=3,129,213 … sp9=591,907` tonnes). The restart file `ben-initial_conditions.nc`
  (`nschool=6582`; vars species/abundance/weight/age/…) also yields per-species standing stock via
  `abundance × weight` (v4.3 species ordering: sp0=euphausiids…sp9=deepwaterhake). Use
  `osmose-ben_seeding.R` as the primary source, cross-checked against the restart aggregation.

**Blocker 2 — resource forcing must be MERGED (CRITICAL; two reviewers + engine run).** Benguela
wires each resource to its **own** single-variable file (`species.file.sp300=…-sphy….nc`, `sp301=…
-lphy….nc`, …). osmopy's `resources.py::_load_config_species_type` (lines 109-174) sets `nc_file`
from the **first** resource only and opens a **single** dataset; `update()` looks up each resource
**by name** in that one dataset (`resources.py:216-239`) and falls to `biomass=0` otherwise. Verified
live: `sphy` sum≈68947, `lphy/szoo/lzoo` = 0.0. So **3 of 4 plankton groups are silently zero** if
the files are copied as-is (the native `species.biomass.file.spN` keys do NOT rescue this — only the
config *writer* emits them; the reader never consumes them). **Fix:** merge the 4 ROMS files into one
multi-variable NetCDF (vars sphy/lphy/szoo/lzoo, `time=24, ny=62, nx=56`) and point all four
`species.file.sp300-303` at it (EEC convention — `eec_ltlbiomassTons.nc` holds all resource vars).

**Blocker 3 — fishing keys must be STRIPPED, not just disabled (CRITICAL).** `fisheries.enabled=FALSE`
correctly migrates to `module.multispecies.fisheries.enabled=FALSE` and gates `_parse_fisheries`
(`config.py:1984`). But `_load_fishing_seasonality` (`config.py:2170`) and `_load_discard_rates`
(`config.py:2173`) run **unconditionally** in `EngineConfig.from_dict`, reading
`fisheries.catchability.file`, `fisheries.seasonality.file.fshN`, and `fisheries.discards.file` via
`_require_file`, which **raises FileNotFoundError** if the path is missing. Since v1 does not bundle
`input/fisheries/`, the synthesis step must **strip** `fisheries.catchability.file`,
`fisheries.discards.file`, and `fisheries.seasonality.file.fsh1..9`, and set
`simulation.fishing.mortality.enabled=FALSE` (and ideally `simulation.nfisheries=0`) for a truly
clean unfished run. (`fisheries.movement.file.map0` is read unconditionally too, but Benguela uses
`map1..9`, so it's safe.)

**Blocker 4 — movement maps are NetCDF, loader wants CSV (viable to convert).** 10 per-species
NetCDF maps (`input/maps/<species>.nc`), each with 3 life-stage vars (`stage0/1/2`) over
`time=24, y=62, x=56`, wired by 27 map indices (`map0`–`map26`) × 6 fields each
(`species/variable/nsteps.year/initialage/lastage/file`) = 162 `movement.*.mapN` key lines. osmopy's map loader
(`osmose/engine/movement_maps.py`) reads CSV grids and **does** support time-varying/seasonal maps:
each `movement.*.mapN` carries its own age range (`initialage/lastage`), **step subset**
(`movement.steps.mapN`, semicolon list, default all), and year range, each pointing at one static CSV
(`movement.file.mapN`); the loader auto-dedups identical CSV paths (`movement_maps.py:140-197`). This
is precedented in production by EEC (`eec_param-movement.csv:34-43`: whiting map11 steps 0-11 vs
map12 steps 12-23). So the conversion — one CSV per distinct stage×step slice, wired via
`movement.steps.mapN` — is directly expressible.

**Stability risk (the core challenge).** Running `osmose-ben_seeding.R` as-is on the Python engine
(seeding present, but forcing unmerged and maps unloaded) does not merely fail to seed — the
populations **explode** to physically absurd values (anchovy 8.5×10⁵ → ~1.2×10²² tonnes over 10 yr,
~40×/yr). This is driven by the broken spatial/forcing structure (map-less fish spread across the
whole grid, dissolving predator–prey overlap; 3/4 plankton empty). **Fixing blockers 1–4 is
necessary but may not be sufficient for a stable run.** The plan must include a stability-validation
step (below) with a decision gate, and the smoke test must assert **bounded, non-exploding** biomass,
not merely finite-and-positive (the exploding run *passes* a naive `>0` check).

## Architecture & data layout

Bundled under `data/benguela/`. The other demos are **flat** (param CSVs + grid/forcing beside the
master), but Benguela keeps the source's `input/` subtree so most master path keys
(`grid.netcdf.file = input/grid-mask.nc`, `reproduction.season.file.spN = input/reproduction/…`,
`predation.accessibility.file = input/…`) stay valid unchanged. Two key families **are** rewritten:
(a) `species.file.sp300-303` is repointed from the 4 separate ROMS files to the single merged
`input/roms_climatological_merged.nc` (Blocker 2); (b) the `movement.*.mapN` keys are repointed from
NetCDF to the converted CSVs (Blocker 4). All copied-as-is files retain their `input/` prefix in both
the bundle and the master.

```
data/benguela/
  benguela_all-parameters.csv        # synthesized osmopy master (;-separated, with includes)
  input/
    grid-mask.nc                     # 62×56 grid (copied as-is)
    roms_climatological_merged.nc    # 4 ROMS vars merged into ONE multi-var file (Blocker 2)
    predation-accessibility-25mars2015.csv
    *.csv                            # species / predation param CSVs
    reproduction/
      reproduction-seasonality-sp{0..9}.csv   # subdir preserved (read via _require_file)
  maps/
    <species>_stage<N>[_s<slice>].csv   # converted presence grids (deduped over steps)
```

`_generate_benguela(output_dir)` copies `data/benguela/` into `output_dir/config` and returns
`{config_file, output_dir}`, where `config_file` is **hardcoded** to
`config_dir / "benguela_all-parameters.csv"` (the generators hardcode the master filename — they do
not glob) and `output_dir` is the `output_dir/"output"` sim dir.

## Components (build pipeline)

Each build step is a small, tested, one-time script under `scripts/` writing committed artifacts into
`data/benguela/`. Scripts are reproducible against the source clone; the committed data is what ships.
(`scripts/` is outside the ruff/pyright scope, per project convention.)

### 1. Seeding — `scripts/derive_benguela_seeding.py`
- **Source:** `osmose-ben_seeding.R` (primary, authoritative), cross-checked against
  `ben-initial_conditions.nc` aggregation (`abundance × weight` per v4.3 species id).
- **Produces:** `population.seeding.biomass.sp0..9` (tonnes) + `population.seeding.year.max` (warm-up
  window; `osmose-ben_seeding.R` uses 30 — the plan confirms/adjusts against the chosen horizon).
- **Config edits:** set the seeding keys; drop `population.initialization.file` and
  `osmose.configuration.initialization` (osmopy cannot read them).

### 2. Resource-forcing merge — `scripts/merge_benguela_forcing.py`
- **Reads:** the 4 `roms_climatological-{sphy,lphy,szoo,lzoo}_…nc` files.
- **Produces:** `data/benguela/input/roms_climatological_merged.nc` — one dataset with 4 variables
  (`sphy/lphy/szoo/lzoo`, the var names MUST equal the resource names `species.name.sp300-303` — the
  lookup is `rsc.name in forcing_data`), dims `time=24, ny=62, nx=56`, values copied verbatim (no
  interpolation). Source units are already "tons per cell"; Benguela sets no
  `species.multiplier/offset.sp30x` — no unit conversion needed.
- **Config edits:** repoint `species.file.sp300..303` all at the merged file.
- **Gate:** after wiring, `ResourceState.update(0)` yields non-zero biomass for **all 4** groups —
  checked via **`np.nansum`** (or ocean-mask cells), NOT a plain `.sum()`/`>0`: the ROMS forcing is
  NaN over land (~54% of cells, aligned to `grid-mask.nc`, exactly like EEC's shipped
  `eec_ltlbiomassTons.nc`), so a naive `.sum()` returns NaN and would spuriously fail.

### 3. Movement-map conversion — `scripts/convert_benguela_maps.py`
- **Reads:** the 10 `input/maps/<species>.nc` files + the 162 `movement.*.mapN` key lines (27 map
  indices; discover indices from the config, don't hardcode a count).
- **Produces:** osmopy CSV presence grids (`ny × nx`, 0..1) under `data/benguela/maps/`, plus a
  rewritten movement-map key block. Each emitted map index MUST carry the full field set:
  **`movement.species.mapN`** (the only key binding an index to a species — `movement_maps.py:128-132`;
  omitting it orphans the species from movement → schools flagged `is_out`, reproducing the explosion
  failure mode) plus `movement.file.mapN`, `movement.steps.mapN`, and
  `movement.initialage/lastage.mapN`. Deduplicate identical stage×step slices by content (emit one CSV,
  point multiple indices at its path). Verify land/ocean orientation against `grid-mask.nc` so fish
  aren't placed on land.

### 4. Master-config synthesis — `scripts/build_benguela_config.py`
- **Reads:** `osmose-ben.R` + `.osm`/`.csv` includes.
- **Produces:** `data/benguela/benguela_all-parameters.csv` (osmopy `;`-master + includes) with the
  edits from steps 1–3 applied, plus:
  - **Fishing off (Blocker 3):** `fisheries.enabled=FALSE`, `simulation.fishing.mortality.enabled
    =FALSE`, `simulation.nfisheries=0` (load-bearing — `n_fisheries=0` short-circuits both the
    `_load_fishing_seasonality` loop and `_parse_fisheries`), and strip `fisheries.catchability.file`
    + `fisheries.discards.file` (read unconditionally). Stripping `fisheries.seasonality.file.fsh1..9`
    is defense-in-depth (already unreachable at `nfisheries=0`) but keeps the master clean. Verified:
    `EngineConfig.from_dict` then loads with `input/fisheries/` entirely absent.
  - `output.file.prefix = benguela`.
  - `simulation.time.nyear` = the empirically-pinned safe horizon (step 5).
  - Keeps 4.3.3 keys (osmopy migrates on read); preserves `input/` and `input/reproduction/` paths.

### 5. Stability validation + horizon pin — `scripts/validate_benguela_stability.py` (spike → gate)
- Run the fully-wired config (steps 1–4 applied) long (e.g. 50–100 yr, fixed seed). Check whether the
  10 species stay **bounded and positive** (no explosion, no collapse to zero).
- **Diagnostics (not black-box pass/fail):** report per species the number of steps analytic seeding
  actually fired (`reproduction.py:122-129` re-injects the seed as SSB on EVERY step with SSB==0 while
  `step < year.max×ndt`, not a one-time pulse) and the step at which natural SSB first exceeds the
  seed — so instability can be attributed (seeding re-injection vs food-web dynamics).
- **Seeding-window decision:** `osmose-ben_seeding.R` uses `year.max=30`. If the pinned horizon lands
  Baltic-like (~15 yr), `30 > 15` means seeding stays live for the entire demo run, not just a warm-up.
  Decide deliberately whether to trim `year.max` to (well below) the pinned `nyear`; record the choice.
- If stable: pin `simulation.time.nyear` safely below any late-horizon breakdown, with an inline
  comment (mirroring the Baltic `nyear=15` pin at `demo.py:201-203`).
- **Decision gate:** if no bounded horizon exists even with blockers 1–4 fixed, escalate — options
  are light stabilization (e.g. revisit the unfished decision / seeding magnitude / accessibility) or
  accept a short demo horizon; do not ship an exploding demo.

### 6. Demo-registry wiring — `osmose/demo.py`
- Add `benguela` to `list_demos()`; add `_generate_benguela` (copy-from-bundle + `_bundled_data_dir`
  loud fallback, LTL-style stub using `simulation.nresource ; 4`) and register it in `generators`.
- Add a complete `DEMO_INFO["benguela"]` entry (all 6 fields — title "Southern Benguela", region
  "SE Atlantic upwelling (Benguela)", species "10 focal species", resources "4 ROMS plankton groups",
  engine, summary). `engine`: use a value consistent with the existing labels (this field is
  display-only — `ui/pages/grid.py:80`).
- **Enforce Python-only (real guard, not just docs):** `DEMO_INFO["engine"]` is cosmetic; the actual
  UI Java-engine gate is `java_engine_block_reason()` (`osmose/runner.py:17-55`), which only blocks on
  `simulation.nbackground > 0`. Benguela has 0 background, so nothing currently stops a user selecting
  the Java engine for this Python-only demo. Extend `java_engine_block_reason` to also block Benguela
  (keyed off a config marker or the demo/prefix), with a test, so the Python-only scope is enforced,
  not merely documented.

### 7. Gates / tests — `tests/`
- **Smoke / stability (load-bearing):** generate the demo, run a fixed-seed Python-engine run at the
  pinned horizon; assert for all 10 species biomass is **finite and positive across the whole
  timeseries** (no mid-run NaN/collapse) **and bounded** by a concrete ceiling:
  `biomass[t] ≤ K × seeding_biomass[species]` with **K = 1000** (an order of magnitude looser than the
  ~100× Baltic overshoot the repo already tolerates), plus an absolute backstop of **10⁹ tonnes** per
  species (the observed explosion hit ~10²²). Also assert **resource biomass > 0 for all 4 groups**
  via `np.nansum`/ocean-mask (NOT `.sum()` — forcing is NaN over land; catches Blocker 2 regressions).
  Template: `tests/test_bob_440_smoke.py`, `tests/test_demo.py`.
- **Determinism:** two fixed-seed runs identical.
- **Registry / files-copied:** `benguela` in `list_demos()`; `demo_info("benguela")` complete;
  `osmose_demo("benguela", tmp)` writes a valid master + copies grid/forcing/maps/reproduction
  (pattern: `test_osmose_demo_eec_full_has_netcdf_and_maps`, `test_osmose_demo_eec_copies_support_dirs`).
- **Auto-parametrized suites:** adding `benguela` extends `test_demo.py:275`
  (`test_demo_info_covers_all_demos_with_full_fields`) and `tests/test_ui_load_scenarios.py:195`
  (`test_all_demos_produce_unique_configs`) — the config must read cleanly and DEMO_INFO must be
  complete or these go red. (They don't run the engine, so they won't catch runtime blockers.)
- **Java-guard test:** assert `java_engine_block_reason` blocks a Benguela run (Component 6).
- **Housekeeping:** `ruff format`/`ruff check` and `pyright` clean over `osmose/ ui/ tests/`.

## Data flow

```
osmose-ben_v4.3_Florance/  (source clone, scratchpad)
   │  derive_benguela_seeding.py   → seeding.biomass.spN + year.max   (from osmose-ben_seeding.R)
   │  merge_benguela_forcing.py    → roms_climatological_merged.nc + repoint species.file.sp300-303
   │  convert_benguela_maps.py     → maps/*.csv + movement.file/steps.mapN block
   │  build_benguela_config.py     → benguela_all-parameters.csv (fishing stripped, input/ preserved)
   │  validate_benguela_stability.py → pinned simulation.time.nyear
   ▼
data/benguela/  (committed bundle)
   │  osmose.demo._generate_benguela()  → copies into output_dir/config
   ▼
PythonEngine().run_in_memory(seed=…)  → stable, bounded biomass for 10 species; 4 resources non-zero
```

## Fishing (v1 = unfished; follow-up)

Benguela's fishery is fleet-based and additionally uses a NetCDF fleet-movement map (`mapFleets.nc`)
with the same NetCDF-vs-CSV incompatibility as the species maps. v1 disables **and strips** fishing
(Blocker 3). A follow-up can port the fleet fishery (catchability/discards/seasonality CSVs +
`mapFleets.nc` → CSV) the way EEC's fleets are wired, and add a fished smoke test. **Risk note:** if
the stability spike (step 5) finds the unfished config cannot be bounded but a fished one can, the
unfished-v1 decision is revisited at that gate.

## Error handling & edge cases

- **Missing data bundle:** `_generate_benguela` uses the existing `_bundled_data_dir` loud-fallback
  (warns, writes a minimal non-runnable LTL-style stub), matching the other generators.
- **Size-ratio warnings:** the engine emits benign `Swapping size ratios for spN` UserWarnings on
  load (pre-existing in the source; the engine swaps them). Leave as-is.
- **Map orientation:** converted CSVs must match osmopy's grid orientation; verify a converted map
  against `grid-mask.nc` ocean cells.
- **Seeding vs restart mismatch:** `osmose-ben_seeding.R` values differ from raw restart aggregation
  (SSB vs total stock); the plan documents which is used and why.
- **Dedup correctness:** the map converter compares full slices for exact equality — never collapse
  genuinely distinct seasonal patterns.

## Success criteria

1. `osmose_demo("benguela", tmp)` generates a config that runs on the Python engine with **bounded,
   finite, positive biomass for all 10 species across the whole pinned horizon**, and **non-zero
   biomass for all 4 resource groups**.
2. `benguela` is listed in `list_demos()` with a complete `DEMO_INFO` entry (UI model picker).
3. Runs are deterministic under a fixed seed.
4. All new + auto-parametrized tests pass; ruff + pyright clean; the full suite stays green.
5. The bundle is self-contained (~1.6 MB) and needs no external files or network.

## Known limitations

- **Mesopelagic (sp5) decline.** At the pinned 15-yr horizon, mesopelagic biomass decays ~2.3×/year
  and ends near-extinct (~0.00066 t vs a seed of 1,439,984 t; final/seed ≈ 4.6e-10) — functionally
  collapsed, though still `> 0`. The other 9 species end healthy (0.10×–2.4× seed). This is accepted
  as an uncalibrated-example artifact (calibration is explicitly out of scope, see "Scope" above),
  documented rather than fixed — mirroring the Baltic precedent (`demo.py` nyear=15 pin comment,
  "collapses to herring+sprat... calibration limit, not an engine bug"). The smoke test's
  `v[-1] > 0` check alone can't certify non-degeneracy, so `test_benguela_smoke_bounded_and_positive`
  also asserts at least 9 of 10 species stay above a `1e-3 × seed` floor, so a regression collapsing
  a second species is caught.

## Risks

- **Stability (primary).** The config explodes on osmopy today; blockers 1–4 are necessary but not
  proven sufficient. Mitigated by the step-5 validation gate with explicit escalation options. If
  unresolvable within example-appropriate effort, the fallback is a short pinned horizon (Baltic
  precedent) or deferral — surfaced to the user at that gate, not silently shipped.
- **Map conversion volume.** 81 map declarations × up-to-24 steps; dedup keeps CSV count manageable,
  but orientation and step-subset wiring must be verified against EEC's precedent.

## Follow-ups (not this task)

- Port the fleet fishery (fished Benguela).
- Optional native-4.4.x conversion + cross-engine Java parity.
- Optional calibration if the uncalibrated dynamics need more than a pinned horizon.
- Optional engine enhancement: teach `resources.py` to read one NetCDF per resource index (would
  remove the merge step for future one-file-per-resource configs) — out of scope here (no engine
  change).

## References

- Source config: `osmose-ben_v4.3_Florance/` (`osmose-ben.R`, `osmose-ben_seeding.R`, ROMS forcing,
  v4.3.3). Restart: `input/ben-initial_conditions.nc`.
- Demo registry: `osmose/demo.py` (`list_demos`, `DEMO_INFO`, `_generate_*` hardcoded master name,
  `_bundled_data_dir`, `_MIGRATION_CHAIN`).
- Seeding path: `osmose/engine/processes/reproduction.py:122-129`; parse `config.py:522-533`.
- Resource forcing: `osmose/engine/resources.py:109-174, 216-239` (single-file load).
- Fishing loaders (unconditional): `osmose/engine/config.py:2170, 2173`; `_require_file` `config.py:125+`.
- Movement maps: `osmose/engine/movement_maps.py:140-197`; EEC precedent `data/eec_full/eec_param-movement.csv:34-43`.
- Test templates: `tests/test_bob_440_smoke.py`, `tests/test_demo.py`, `tests/test_ui_load_scenarios.py`.
- Prior art (rigor + gotchas): BoB 4.4.1 migration —
  `docs/superpowers/specs/2026-07-06-bob-440-migration-phase3-ices-design.md`.
