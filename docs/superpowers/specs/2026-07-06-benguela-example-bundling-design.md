# Southern Benguela example bundling — design

**Date:** 2026-07-06
**Status:** Approved (brainstorming) — ready for writing-plans
**Branch:** `feat/benguela-example`

## Goal

Add **`benguela`** (Southern Benguela, source config `osmose-ben_v4.3_Florance`) as a bundled
osmopy demo that loads and runs **non-degenerate** on the osmopy Python engine, registered in the
demo registry (`osmose/demo.py`) with tests, at the rigor of the Bay-of-Biscay (BoB) migration.

Southern Benguela is a well-known Eastern Boundary Upwelling System (EBUS): 10 focal species
(euphausiids, anchovy, sardine, redeye, horse mackerel, mesopelagic, silver kob, snoek, shallow-
and deep-water hake) driven by 4 ROMS-derived plankton resource groups. It broadens osmopy's
bundled example set beyond the current NE-Atlantic / Baltic configs with a scientifically distinct
upwelling ecosystem.

## Scope

**In scope**
- Bundle a runnable, unfished Southern Benguela config under `data/benguela/`.
- Derive analytic seeding biomass from the config's NetCDF restart file so the osmopy Python
  engine can bootstrap the populations.
- Convert the per-species NetCDF movement maps to the CSV format osmopy reads.
- Synthesize the R-driver master (`osmose-ben.R`) into osmopy's CSV master + includes convention.
- Wire the demo into `list_demos()`, `DEMO_INFO`, and the `_generate_*` dispatch.
- Smoke / determinism / registry tests + ruff + pyright green.

**Out of scope (explicit follow-ups)**
- The fleet-based fishery (see "Fishing" below) — v1 is unfished.
- Native 4.4.x conversion and cross-engine Java parity — this is a Python-engine example. osmopy
  auto-migrates the 4.3.3 keys to 4.4.0 internally on read (`_MIGRATION_CHAIN`), which is
  sufficient to run; no native rewrite or parity harness.
- Calibration / realism tuning — like BoB and EEC, this is an uncalibrated example config; the bar
  is "runs and is non-degenerate," not "matches observations."

## Confirmed decisions (brainstorming)

1. **Initialization** — derive analytic seeding from the restart file (osmopy's supported path).
   *Not* building NetCDF restart-file support into the engine.
2. **Movement maps** — convert the NetCDF maps to osmopy CSV (full spatial fidelity), *not*
   uniform/random distribution.
3. **Engine scope** — Python-engine example only; no native-4.4.x / Java-parity work.
4. **Fishing** — disabled for v1 (`fisheries.enabled=FALSE`); the fleet fishery is a follow-up.

## Spike findings (de-risking evidence)

Run against the source clone `osmose-ben_v4.3_Florance/` on the osmopy Python engine:

- **Parses:** `OsmoseConfigReader().read("osmose-ben.R")` → 800 keys, `simulation.nspecies=10`,
  `simulation.nresource=4`, no `simulation.nbackground` key (= 0 background species).
- **Grid ↔ forcing match exactly:** `grid-mask.nc` is `y=62, x=56`; each ROMS forcing NetCDF
  (`roms_climatological-{sphy,lphy,szoo,lzoo}_benguela_15days_2000_2009.nc`) is
  `time=24, ny=62, nx=56`. Forcing steps (24) = `simulation.time.ndtperyear` (24). **No resample,
  no regrid** — unlike BoB.
- **The blocker:** the run produces **all-zero biomass from the first timestep**. Cause: the config
  seeds its entire population from a NetCDF restart file with analytic seeding switched off —
  ```
  population.initialization.file = ben-initial_conditions.nc   # 6582 schools
  population.seeding.biomass.sp0..9 = 0
  population.seeding.year.max = 0
  ```
  osmopy's only seeding path is analytic `population.seeding.biomass.spN` (warm-up SSB bootstrap,
  `osmose/engine/simulate.py:538`, requires `seeding_biomass[sp] > 0`); it does **not** read
  `population.initialization.file`. With analytic seeding zeroed, nothing ever populates.
- **The blocker is recoverable:** `ben-initial_conditions.nc` (dims `nschool=6582`; vars `species`,
  `abundance`, `weight`, `age`, `length`, `trophiclevel`, `x`, `y`) carries per-school biomass.
  Aggregating `abundance × weight` per species yields sensible standing stocks for all 10 species
  (anchovy ≈ 2.6 Mt down to hakes ≈ 0.07–0.4 Mt), so the NetCDF-restart seeding can be converted to
  osmopy analytic seeding.
- **Movement maps:** 10 per-species NetCDF files (`input/maps/<species>.nc`), each with 3 life-stage
  variables (`stage0/1/2`) over `time=24, y=62, x=56`, wired by 81 `movement.*.mapN` keys. osmopy's
  map loader (`osmose/engine/movement_maps.py`) reads CSV grids (cf. `data/eec_full/maps/*.csv`), so
  all 10 NetCDF maps fail to load today (`utf-8 codec can't decode byte 0x89` — the NetCDF magic
  byte). Converting each NetCDF map variable to CSV resolves this.
- **Fishing:** fleet-based (`fisheries.enabled=TRUE`; `catchability.csv`, `discards.csv`, per-species
  seasonality CSVs, and `fisheries.movement.netcdf.enabled=TRUE` + `mapFleets.nc`). Disabled in v1.

## Architecture & data layout

Bundled under `data/benguela/`, mirroring the other demos (`data/eec_full/`, `data/baltic/`):

```
data/benguela/
  benguela_all-parameters.csv     # synthesized osmopy master (;-separated, with includes)
  input/
    grid-mask.nc                  # 62×56 grid (copied as-is)
    roms_climatological-*.nc      # 4 native 24-step forcing files (copied as-is)
    *.csv                         # species / predation / reproduction param CSVs
  maps/
    <species>_stage<N>[_s<step>].csv   # converted presence grids (deduped over time)
```

Generation follows the established pattern: `_generate_benguela(output_dir)` copies
`data/benguela/` into `output_dir/config` and returns `{config_file, output_dir}`, where
`config_file` is `benguela_all-parameters.csv` (the demo tooling globs `*all-parameters*.csv`).

## Components (build pipeline)

Each build step is a small, tested, one-time script under `scripts/` that writes committed
artifacts into `data/benguela/`. The scripts are reproducible (re-runnable against the source clone)
but the committed data is what ships.

### 1. Seeding-biomass derivation — `scripts/derive_benguela_seeding.py`
- **Reads:** `ben-initial_conditions.nc`.
- **Produces:** per-species seeding biomass (tonnes) for `population.seeding.biomass.sp0..9`.
- **Method:** aggregate school biomass (`abundance × weight`, grams → tonnes) per `species` id. The
  plan resolves total-standing-stock vs SSB (mature schools via the maturity threshold) — SSB is the
  more faithful proxy for osmopy's egg-bootstrap seeding, but total stock is acceptable for an
  example; the plan picks one and documents it.
- **Config edits (in the synthesized master):** set `population.seeding.biomass.spN` to the derived
  values; set `population.seeding.year.max` to a positive warm-up window (e.g. a few years) so the
  engine bootstraps eggs from that SSB; drop `population.initialization.file` /
  `osmose.configuration.initialization` (osmopy cannot read them).

### 2. Movement-map conversion — `scripts/convert_benguela_maps.py`
- **Reads:** the 10 `input/maps/<species>.nc` files and the 81 `movement.*.mapN` declarations
  (species, variable=stage, `initialAge`/`lastAge`, `file`).
- **Produces:** osmopy CSV presence grids (`ny × nx`, values 0..1) under `data/benguela/maps/`, and
  a rewritten movement-map key block pointing at the CSVs.
- **Method:** for each map declaration, read the referenced NetCDF variable `(time, y, x)`,
  **deduplicate identical time-slices** (many OSMOSE maps are seasonally constant), emit one CSV per
  distinct pattern, and wire osmopy's movement-map metadata (species, age range, seasonality/steps)
  so each life-stage gets its time-appropriate grid. Land/ocean orientation follows osmopy's map
  convention (verify against `grid-mask.nc`).

### 3. Master-config synthesis — `scripts/build_benguela_config.py`
- **Reads:** `osmose-ben.R` + its `.osm`/`.csv` includes.
- **Produces:** `data/benguela/benguela_all-parameters.csv` (osmopy `;`-separated master +
  `osmose.configuration.*` includes) with the edits from steps 1–2 applied, `fisheries.enabled=FALSE`
  (Fishing decision), `output.file.prefix = benguela`, and a demo-appropriate
  `simulation.time.nyear`. Keeps 4.3.3 keys (osmopy migrates them on read).
- **Note:** this flattens the R driver into the CSV convention; it does not re-derive parameters.

### 4. Demo-registry wiring — `osmose/demo.py`
- Add `benguela` to `list_demos()`.
- Add a `DEMO_INFO["benguela"]` entry: title "Southern Benguela", region "SE Atlantic upwelling
  (Benguela)", species "10 focal species", resources "4 ROMS plankton groups", engine "Python",
  a one-line summary.
- Add `_generate_benguela(output_dir)` (copy-from-bundle pattern, with the `_bundled_data_dir`
  fallback stub) and register it in the `generators` dispatch.

### 5. Gates / tests — `tests/`
- **Smoke (the load-bearing gate):** generate the demo, run a short fixed-seed Python-engine run,
  assert biomass is **finite and > 0 for all 10 species** (the exact assertion that fails today).
- **Determinism:** two fixed-seed runs produce identical output.
- **Registry:** `benguela` appears in `list_demos()`; `demo_info("benguela")` returns the metadata;
  `osmose_demo("benguela", tmp)` writes a valid master that reads back.
- **Housekeeping:** `ruff format`/`ruff check` and `pyright` clean over `osmose/ ui/ tests/`.
  (`scripts/` is outside the ruff/pyright scope, per project convention.)

## Data flow

```
osmose-ben_v4.3_Florance/  (source clone, scratchpad)
   │  scripts/derive_benguela_seeding.py     → seeding.biomass.spN
   │  scripts/convert_benguela_maps.py        → maps/*.csv + movement key block
   │  scripts/build_benguela_config.py        → benguela_all-parameters.csv (+ copied input/)
   ▼
data/benguela/  (committed bundle)
   │  osmose.demo._generate_benguela()  → copies into output_dir/config
   ▼
PythonEngine().run_in_memory(seed=…)  → non-degenerate biomass for 10 species
```

## Fishing (v1 = unfished; follow-up)

Benguela's fishery is fleet-based and additionally uses a NetCDF fleet-movement map (`mapFleets.nc`)
with the same NetCDF-vs-CSV incompatibility as the species maps. For v1 we set
`fisheries.enabled=FALSE`, producing a clean unfished ecosystem run. A follow-up can port the fleet
fishery (catchability/discards/seasonality CSVs + `mapFleets.nc` → CSV) the way EEC's fleets are
wired, and add a fished smoke test.

## Error handling & edge cases

- **Missing data bundle:** `_generate_benguela` uses the existing `_bundled_data_dir` fallback that
  warns loudly and writes a minimal (non-runnable) stub, matching the other generators — never a
  silent 5-line stub.
- **Size-ratio warnings:** the engine emits `Swapping size ratios for spN` UserWarnings on load
  (predator/prey min>max in the source config). These are pre-existing, benign (the engine swaps
  them), and shared with the source — not introduced here; leave as-is.
- **Map land/ocean orientation:** the CSV maps must match osmopy's grid orientation. Verify a
  converted map against `grid-mask.nc` (ocean cells) so fish aren't placed on land.
- **Seeding magnitude:** derived biomass is a bootstrap seed, not a calibration target; the warm-up
  window lets populations settle. The smoke gate only requires non-degeneracy, not specific levels.
- **Deduplication correctness:** the map converter must not collapse genuinely distinct seasonal
  patterns; dedup compares full slices for exact equality.

## Success criteria

1. `osmose_demo("benguela", tmp)` generates a config that runs on the Python engine with
   **finite, positive biomass for all 10 species** over a multi-year fixed-seed run.
2. `benguela` is listed in `list_demos()` with a populated `DEMO_INFO` entry (UI model picker).
3. Runs are deterministic under a fixed seed.
4. All new tests pass; ruff + pyright clean; the full suite stays green.
5. The bundle is self-contained (~1.6 MB) and needs no external files or network.

## Follow-ups (not this task)

- Port the fleet fishery (fished Benguela).
- Optional native-4.4.x conversion + cross-engine Java parity, if a Java-side Benguela is wanted.
- Optional light calibration if the uncalibrated dynamics prove too unstable for a useful demo
  horizon (mirrors the Baltic `nyear` horizon note).

## References

- Source config: `osmose-ben_v4.3_Florance/` (Southern Benguela, ROMS-forced, v4.3.3).
- Demo registry: `osmose/demo.py` (`list_demos`, `DEMO_INFO`, `_generate_*`, `_bundled_data_dir`,
  `_MIGRATION_CHAIN`).
- Seeding path: `osmose/engine/simulate.py:538` (warm-up SSB bootstrap).
- Map format precedent: `data/eec_full/maps/*.csv`; loader `osmose/engine/movement_maps.py`.
- Prior art (rigor + gotchas): BoB 4.4.1 migration —
  `docs/superpowers/specs/2026-07-06-bob-440-migration-phase3-ices-design.md`.
