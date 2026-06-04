# Climate-driven temperature forcing (CMEMS → baltic_ev) — Design

**Date:** 2026-06-04
**Status:** ⚠️ **CLOSED via diagnostic — NOT built.** See
`docs/baltic_temperature_forcing_diagnostic_2026-06-04.md`.

> ## Outcome (post in-loop review + Arrhenius diagnostic)
>
> An in-loop review (3 executing reviewers) confirmed the engine mechanics but proved — by
> reproducing `compute_energy_budget` — that **temperature has ~zero effect on `baltic_ev`**:
> `phit.enabled=false`, so temperature enters only via Arrhenius maintenance, which at
> `c_m=0.001` is ~1e-12 of intake (`dw` bit-identical 4 °C↔18 °C). The constant-vs-CMEMS demo
> would show noise, not a climate signal.
>
> A follow-up diagnostic tested whether the maintenance Arrhenius was missing a reference-temp
> normalization (a latent engine bug). **Refuted by the Java source:** `TempFunction.java:204`
> (`get_Arrhenius`) and `EnergyBudget.java:205` (`getMaintenance`) are byte-for-byte the Python
> formulas — bare Arrhenius by design (MTE-standard, `c_m` carries scale). **No engine bug.**
>
> **Decision (user-approved):** close the line. Climate temperature forcing is blocked behind a
> calibration prerequisite (a `phiT`-enabled / maintenance-calibrated bioen config), not code —
> a recalibration project, out of scope. No engine change, no CMEMS download, no inert loader.
> The design below is retained as provenance; actionable-later notes live in the diagnostic doc.
>
> ---

**Original status:** Approved direction (brainstormed; codebase-grounded by execution). New feature
(science extension #2). **Engine change → parity-gated.**

## Motivation

OSMOSE can be driven by real climate data. The LTL (plankton) half is **already done** —
`data/baltic/baltic_param-ltl.csv` + `baltic_ltl_biomass.nc` were generated from CMEMS via the
`mcp_servers/copernicus` MCP (2026-04-16) and are consumed by `osmose/engine/resources.py`. The
missing half is **temperature**: the engine's bioenergetics (Arrhenius maintenance) can use a
spatial/seasonal temperature field, but the Python engine currently only reads a *constant*
`temperature.value` — the spatial NetCDF path is dormant. This feature activates it and drives the
bioenergetics-enabled `baltic_ev` config with real CMEMS temperature.

## Verified context (audit — confirmed by reading/running the code)

- **The consumer exists:** `data/baltic_ev/baltic_ev_param-simulation.csv:7` sets
  `simulation.bioen.enabled;true`. `baltic` (non-ev) has bioen OFF → temperature does nothing
  there, so **`baltic_ev` is the target**, not `baltic`.
- **The consumption side already handles spatial temperature:** `simulate.py:358-363` — for
  non-constant `temp_data` it does `temp_grid = temp_data.get_grid(step);
  temp_c_arr = temp_grid[state.cell_y, state.cell_x]`, feeding per-school temperature into the
  Arrhenius maintenance (`temp_function.arrhenius`). No change needed there.
- **The gap is purely loading:** `simulate.py:1287-1292` — `if config.bioen_enabled: temp_val =
  config.raw_config.get("temperature.value", ""); if temp_val: temp_data =
  PhysicalData.from_constant(...)`. It **never reads `temperature.filename`** and never calls the
  existing `PhysicalData.from_netcdf`.
- **`PhysicalData.from_netcdf(path, varname="temp", nsteps_year=12, factor=1.0, offset=0.0)`**
  (`physical_data.py:33`) reads `ds[varname].values`, promotes 2D→3D, applies `factor*(x+offset)`,
  returns `(time, ny, nx)`. `get_grid(step)` returns `data[step % nslices]`; `get_value` uses
  `step % nslices` too. So a 24-slice file with `ndtPerYear=24` maps `step % 24` → biweekly. Good.
- **varname gotcha:** `from_netcdf`'s default varname is `"temp"`, but the MCP's
  `generate_osmose_physics` writes the variable as **`"temperature"`** (file `baltic_temperature.nc`).
  So the config MUST set `temperature.varname;temperature`.
- **Path resolution:** `osmose/engine/path_resolution.resolve_data_path(file_key, config_dir=...)`
  is the resolver LTL uses (`resources.py:176-179`), with `config_dir =
  raw_config["_osmose.config.dir"]`. simulate.py already has that value
  (`simulate.py:1250,1315`). Temperature resolves the same way.
- **Allowlist already permits the keys:** `config_validation.py:224-227` already lists
  `temperature.filename`, `temperature.nsteps.year`, `temperature.varname` (added as
  "Java-side knobs Python ignores"). So **no allowlist/parser change** — only the now-outdated
  comment (d) at `config_validation.py:133-134` ("Python uses temperature.value scalars") needs
  updating. `simulate.py` reads `temperature.filename` straight from `raw_config` (no new
  `EngineConfig` field needed).
- **Grid:** `baltic_ev` is `grid.nlon;50`, `grid.nlat;40` → ny=40, nx=50; `ndtPerYear;24`. The MCP
  emits `(24, 40, 50)` lat-descending to match `grid.nc`. Alignment confirmed.
- **CMEMS data:** the cache (`data/cmems_cache/`) has the BGC source (for LTL) but **NO physics
  source** — generating `baltic_temperature.nc` needs a one-time CMEMS *physics* download
  (`thetao`, dataset `cmems_mod_bal_phy_my_P1M-m`). Creds present (`.env`, 3 keys); the MCP tools
  (`mcp__copernicus-marine__download_field`, `generate_osmose_physics`) are available.
- Engine change → CI/parity: the 12/12 Java parity suite must stay bit-exact on the default
  (no-`temperature.filename`) path. CI lints `osmose/ ui/ tests/`.

## Architecture

Five pieces: a minimal engine loader change (parity-gated), a one-shot committed forcing file, a
config wiring, a demo, and tests.

### 1. Engine — `osmose/engine/simulate.py` (the core, parity-gated)

Replace the constant-only temperature load (~:1287-1292) with NetCDF-first / constant-fallback:
```python
    temp_data = None
    if config.bioen_enabled:
        temp_file = config.raw_config.get("temperature.filename", "").strip()
        if temp_file:
            from osmose.engine.path_resolution import resolve_data_path

            cfg_dir = config.raw_config.get("_osmose.config.dir", "")
            temp_path = resolve_data_path(temp_file, config_dir=cfg_dir)
            if temp_path is None:
                raise FileNotFoundError(f"temperature.filename not found: {temp_file}")
            temp_data = PhysicalData.from_netcdf(
                temp_path,
                varname=config.raw_config.get("temperature.varname", "temperature"),
                nsteps_year=int(config.raw_config.get("temperature.nsteps.year", "24")),
                factor=float(config.raw_config.get("temperature.factor", "1.0")),
                offset=float(config.raw_config.get("temperature.offset", "0.0")),
            )
        else:
            temp_val = config.raw_config.get("temperature.value", "")
            if temp_val:
                temp_data = PhysicalData.from_constant(float(temp_val))
```
**Default path unchanged:** with no `temperature.filename`, behaviour is byte-identical to today
(constant or None). Update the stale `config_validation.py:133-134` comment (d) to note Python now
reads `temperature.filename` too. (Oxygen NetCDF left as-is — out of scope.)

### 2. Forcing data — one-shot, committed

`scripts/_pull_baltic_temperature.py` (mirrors the snapshot-puller pattern): downloads CMEMS
Baltic physics `thetao` (via `mcp_servers.copernicus.server` functions / the MCP) and runs
`generate_osmose_physics` → `data/baltic_ev/baltic_temperature.nc` (`(24, 40, 50)`, var
`temperature`, °C, lat-descending, NaN on land). Run once; commit the `.nc`; document refresh.
(The generation needs CMEMS creds + network; the committed file makes the engine/tests/demo
self-contained thereafter.)

### 3. Config wiring — `data/baltic_ev/`

Add to the bioenergetics/physics config block:
`temperature.filename;baltic_temperature.nc`, `temperature.varname;temperature`,
`temperature.nsteps.year;24`.

### 4. Demo — `scripts/run_temperature_forcing_demo.py`

Mirrors `scripts/run_dsvm_demo.py`: run `baltic_ev` twice (constant `temperature.value` vs the
CMEMS `temperature.filename`), then compare with the just-shipped tools
(`osmose.analysis.run_delta`, `osmose.size_spectrum.compute_size_spectrum`) and print the
per-species / size-structure differences attributable to spatial-seasonal temperature.

## Data flow

One-shot: CMEMS `thetao` → `generate_osmose_physics` → committed `baltic_temperature.nc`.
Per run: `simulate` (bioen on) → resolve+load `temperature.filename` via `from_netcdf` →
`get_grid(step % 24)` per step → per-school `temp_c` → Arrhenius maintenance → bioenergetic
growth/mortality.

## Parity (gating)

Use the **migration-check skill**: confirm the 12/12 Java parity configs stay **bit-exact** (none
set `temperature.filename`, so they take the unchanged constant/None path). The NetCDF path is new
opt-in behaviour with no Java reference to match (the Python engine is the reference for it). A
focused engine test covers the new path numerically.

## Error handling

- `temperature.filename` set but unresolvable → `FileNotFoundError` (explicit, at load).
- `varname` absent in the NetCDF → `from_netcdf` raises a `KeyError`-style error from xarray;
  wrap/test for a clear message.
- NaN at an ocean cell a school occupies → would poison Arrhenius. The committed forcing must have
  valid values at all model ocean cells; add a generation-time + test-time check (no NaN where the
  grid mask is ocean). Land NaN is fine (no schools there).
- Slice count ≠ `nsteps.year` → `step % nslices` still works but is documented; the demo/test use
  24==24.
- `bioen_enabled` False → temperature ignored entirely (so `baltic` is unaffected — by design).

## Testing

- **Engine load + spatial effect** (`tests/test_engine_temperature_forcing.py`): a synthetic
  `temperature.nc` (small grid, 24 steps, a known spatial gradient) + a minimal bioen-enabled
  config → assert `temp_data` is non-constant, `get_grid(step)` returns the right slice, and two
  cells at different temperatures yield different Arrhenius maintenance (the constant path gives
  uniform). Also assert grid shape == (grid.ny, grid.nx).
- **Default-path unchanged**: a bioen config with `temperature.value` and no `filename` →
  `temp_data.is_constant` (no behavioural change); and a non-bioen config → `temp_data is None`.
- **Config parse**: the three `temperature.*` keys load via the reader and pass
  `EngineConfig.from_dict` warn-mode clean.
- **No-NaN-at-ocean** check on the committed `baltic_temperature.nc` (skip if the file is absent,
  so the suite passes pre-generation).
- **Parity** (migration-check): default configs bit-exact.
- **Demo smoke**: the demo script runs end-to-end on a short horizon (or is import-smoke-tested if
  a full bioen run is too slow for CI).

## Scope / YAGNI

- **In:** the engine NetCDF-temperature loader (parity-gated), the one-shot CMEMS forcing +
  committed `baltic_temperature.nc`, the `baltic_ev` wiring, the demo, the tests, docs.
- **Out:** oxygen NetCDF forcing (analogous — deferred); LTL changes (already done); applying to
  `baltic` (bioen off); recalibrating `baltic_ev`; a Java-engine NetCDF-temp path; the
  `temperature.value` constant path (kept as the fallback).

## Honest limitations

- Forcing generation needs CMEMS creds + a one-time download (committed thereafter; refresh =
  re-run the puller).
- Temperature only affects **bioen-enabled** configs, so the visible payoff is on `baltic_ev`, not
  the main `baltic` calibration.
- The demo shows a **mechanism** (spatial-seasonal temperature now drives the bioenergetics), not a
  recalibrated climate projection — `baltic_ev` is not re-tuned for the new forcing.

## Delivery

Single PR: `osmose/engine/simulate.py` (loader) + `config_validation.py` (comment),
`scripts/_pull_baltic_temperature.py`, `data/baltic_ev/baltic_temperature.nc` + config keys,
`scripts/run_temperature_forcing_demo.py`, `tests/test_engine_temperature_forcing.py`, a
docs/CHANGELOG note. Parity verified via migration-check. No recalibration.
