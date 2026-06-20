# CMEMS→OSMOSE Forcing — Conversion Core (Sub-project A) — Design

**Date:** 2026-06-20
**Status:** Approved (brainstorming complete)

## Context & scope

The Copernicus MCP server (`mcp_servers/copernicus/server.py`) already converts CMEMS Baltic data into OSMOSE forcing — `generate_osmose_ltl` (6-group plankton/benthos resource forcing) and `generate_osmose_physics` (temperature/salinity). But that logic is **trapped inside MCP tool functions**: it was run once by hand (the Baltic `baltic_param-ltl.csv` header records *"Generated: 2026-04-16 via … generate_osmose_ltl()"*), it is hardcoded to the Baltic 50×40 grid, and it is not importable, reproducible, or CI-tested.

This sub-project (**A**) makes that conversion a first-class OSMOPY capability: a pure, importable, grid-general `osmose/forcing/` package plus a convert-only CLI. It is the foundation for three follow-up sub-projects, each with its own spec/plan/cycle:

- **B — live CMEMS download** (credential-gated; blocked on the owed credential rotation).
- **C — config scaffolding/repointing** (wire generated forcing into any config).
- **D — "CMEMS Forcing" Shiny page** (interactive driver + preview).

**This spec is A only.** No new science — a faithful extraction. The science heuristics and coefficients are preserved exactly; only the grid is generalized and the code is made pure/testable.

## Goals

1. A pure `osmose/forcing/` package (numpy/xarray/scipy only) that converts a **downloaded** CMEMS BGC NetCDF → 6-group LTL forcing, and a CMEMS PHY NetCDF → temperature/salinity forcing, for **any config's grid**.
2. The MCP `generate_osmose_*` tools refactored to thin wrappers over this core (single source of truth, no drift; unchanged tool behavior).
3. A convert-only CLI `scripts/convert_cmems_forcing.py` (bring-your-own downloaded NetCDF → forcing NetCDF).
4. Full CI coverage with synthetic source datasets — no credentials, no network.

## Non-goals (explicit — deferred to B/C/D or out of scope)

- Live CMEMS download (Sub-project B).
- Writing/scaffolding/repointing config keys (Sub-project C).
- Any Shiny UI (Sub-project D).
- The CMEMS dataset catalog / `list_datasets` (stays in the MCP server).
- Any change to the science: C:wet ratios, seasonal community-split fractions, Mode-A/Mode-B selection, depth integration, regridding method — all extracted verbatim.

## Architecture

```
downloaded CMEMS NetCDF (BGC | PHY)  +  target config
                       │
                       ▼
            GridSpec.from_config(cfg)          (reused: osmose/maps/builder.py)
                       │
        ┌──────────────┴───────────────┐
        ▼                              ▼
  bgc_to_ltl(ds, grid, …)        phy_to_physics(ds, grid, …)
   (regrid → resample_to_24 → science → apply_land_mask)
        │                              │
        ▼                              ▼
   write_ltl(ds, path)           write_physics(dsets, out_dir)
        └──────────────┬───────────────┘
                       ▼
            OSMOSE forcing NetCDF(s)
```

- **Purity / clean-CI constraint:** `osmose/forcing/` imports ONLY `numpy`, `xarray`, `scipy` (all declared core deps), and `osmose.maps.builder.GridSpec`. It MUST NOT import `copernicusmarine`, `fastmcp`, or `dotenv`. This is what lets the package and its tests run in the clean `[dev]` CI venv (per the clean-venv reproduction gotcha — heavy MCP deps are absent there).
- **Dependency direction:** the MCP server imports FROM `osmose.forcing`, never the reverse.
- **Grid generality:** the target grid is derived from a config via `GridSpec.from_config(cfg)`; the hardcoded `OSMOSE_GRID` constant is gone from the conversion path. The Baltic case is just `GridSpec.from_config(<baltic cfg>)`.

## Components

### `osmose/forcing/grid.py` — grid-parameterized helpers
Generalized versions of the MCP helpers, each taking a `GridSpec` instead of closing over a module constant:

- `target_coords(grid: GridSpec) -> tuple[np.ndarray, np.ndarray]` — `(lat[nlat], lon[nlon])` cell centers, **latitude descending (north→south)** to match `grid.nc`. Equivalent to the current `_make_target_coords` but reading bounds from `grid`.
- `regrid(data_3d: np.ndarray, src_lat, src_lon, grid: GridSpec) -> np.ndarray` — nearest-neighbor `(time, src_lat, src_lon) → (time, nlat, nlon)` (verbatim algorithm).
- `resample_to_24(data: np.ndarray) -> np.ndarray` — linear interp to 24 biweekly steps; identity if already 24.
- `cell_volume_m3(grid: GridSpec, depth_m: float) -> float` — area×depth using the grid's `dx`/`dy` and a cos(latitude) factor at the grid's mid-latitude (was hardcoded cos(60°); now derived from the grid centroid).
- `get_coords(ds) -> (lat, lon)` and `get_var(ds, name) -> np.ndarray | None` — dataset accessors (NaN→0, promote 2D→3D), verbatim.
- `load_ocean_mask(grid_file: Path | None) -> np.ndarray | None` — load `mask` (bool, True=ocean) from a grid NetCDF; `None` if absent/unreadable.
- `apply_land_mask(groups: dict[str, np.ndarray], ocean_mask: np.ndarray) -> None` — set land cells to NaN in place; silent no-op on shape mismatch (current behavior, now warns via logging).

### `osmose/forcing/ltl.py` — BGC → 6-group LTL
- `@dataclass LtlParams` holding **all** science constants as documented defaults, with the two modes' coefficients kept distinct (they genuinely differ):
  - Mode A (direct biomass): `PHYTO_C_TO_WET=0.012`, `ZOO_C_TO_WET=0.12`, Mode-A `diatom_frac` 12-month array `[0.40,0.60,0.75,0.80,0.70,0.40,0.25,0.20,0.25,0.35,0.40,0.40]`, zoo size split `micro=0.40, meso=0.45, macro=0.15`, benthos `nppv*0.05/3.0` (or `zoo*0.3` fallback when `nppv` absent).
  - Mode B (chl-derived): `chl_to_biomass_factor=50.0`, `nppv≈chl*5` fallback, Mode-B `diatom_frac` array `[0.3,0.5,0.7,0.8,0.7,0.5,0.3,0.2,0.2,0.3,0.3,0.3]`, NPP-derived zoo divisors `micro=0.30/50, meso=0.10/15, macro=0.03/8`, benthos `0.05/3`.
  - Non-12-step inputs fall back to `diatom_frac=0.5` (current behavior). All values clamped non-negative before masking. Defaults reproduce today's output exactly.
- `bgc_to_ltl(ds: xr.Dataset, grid: GridSpec, *, year: int = 0, depth_integrate_m: float = 50.0, params: LtlParams = LtlParams(), ocean_mask: np.ndarray | None = None) -> xr.Dataset`
  - Selects year, depth-integrates, regrids, resamples to 24.
  - **Mode A** when `phyc` & `zooc` present (forecast products): direct carbon biomass → tonnes wet; phyto split into Diatoms/Dinoflagellates by seasonal `diatom_frac`; zoo split into Micro/Meso/Macro; Benthos derived as today.
  - **Mode B** when only `chl`(+`nppv`) present (reanalysis): chl-derived estimate with seasonal community splits.
  - Raises `ValueError` if neither pathway's variables are present.
  - Applies `ocean_mask` (if given) as the final step.
  - Returns an `xr.Dataset` with one variable per group (`Diatoms`, `Dinoflagellates`, `Microzooplankton`, `Mesozooplankton`, `Macrozooplankton`, `Benthos`), dims `(time=24, latitude=nlat, longitude=nlon)`, plus `time`/`latitude`/`longitude` coords and a `mode` attribute. (Structure mirrors the current output so the existing config reader and UI consume it unchanged.)

### `osmose/forcing/physics.py` — PHY → temperature/salinity
- `phy_to_physics(ds: xr.Dataset, grid: GridSpec, *, year: int = 0, depth_surface_m: float = 10.0) -> dict[str, xr.Dataset]`
  - Year-select, surface-depth-select (nearest), regrid, resample to 24.
  - Returns `{"temperature": <ds>, "salinity": <ds>}` for whichever of `thetao`/`so` are present (a missing variable is omitted with a logged note, not an error — matches current behavior).

### `osmose/forcing/io.py` — NetCDF writers
- `write_ltl(ds: xr.Dataset, path: Path) -> Path` — write the LTL dataset (latitude descending, attrs incl. `source`, `title`, conventions note).
- `write_physics(dsets: dict[str, xr.Dataset], out_dir: Path, prefix: str = "baltic") -> list[Path]` — write `f"{prefix}_temperature.nc"` / `f"{prefix}_salinity.nc"`. (Prefix is a parameter so non-Baltic grids can name outputs sensibly; default keeps Baltic filenames.)

### `osmose/forcing/__init__.py`
Public re-exports: `GridSpec`-free entry points `bgc_to_ltl`, `phy_to_physics`, `LtlParams`, `write_ltl`, `write_physics`, and the grid helpers.

### `scripts/convert_cmems_forcing.py` — convert-only CLI
- Args: `--source <bgc|phy .nc>` (required), `--config <dir|master file>` (required — to build the target grid), `--kind {ltl,physics}` (required), `--out <path>` (LTL: file; physics: dir; default derived), `--year` (0=all), `--depth-integrate` (LTL), `--depth-surface` (physics), `--prefix` (physics filenames).
- Loads the config (existing reader), builds `GridSpec.from_config`, loads the grid's ocean mask if available, opens the source NetCDF, calls the core, writes outputs, prints a summary.
- Catches `ValueError`/`OSError` from the core, prints a friendly message, exits non-zero.
- Lives outside the app; importable-free (uses the public `osmose.forcing` API).

### `mcp_servers/copernicus/server.py` — refactor (delegation)
- `generate_osmose_ltl` / `generate_osmose_physics` lose their inline conversion bodies and instead: build a Baltic `GridSpec` (from the bundled Baltic config, or a small Baltic-constant helper), call `osmose.forcing.bgc_to_ltl` / `phy_to_physics` + `write_*`, and format the same human-readable summary string they return today.
- The MCP-server-local helpers (`_make_target_coords`, `_regrid`, `_resample_to_24`, `_cell_volume_m3`, `_get_coords`, `_get_var`, `_load_baltic_ocean_mask`, `_apply_land_mask`) are removed (now provided by `osmose.forcing.grid`).
- `download_field`, `list_datasets`, `check_credentials`, `DATASETS`, `BALTIC_BBOX`, `_require_creds`, `_login` are unchanged.

## Error handling

| Condition | Behavior |
|---|---|
| Source file missing | `bgc_to_ltl`/`phy_to_physics` raise `FileNotFoundError`/`ValueError`; CLI reports + exit 1 |
| BGC lacks both (phyc+zooc) and chl | `bgc_to_ltl` raises `ValueError` with guidance on required variables |
| PHY lacks both thetao & so | `phy_to_physics` returns `{}`; CLI reports "no physics variables found" + exit 1 |
| Ocean mask absent | masking skipped, logged at WARNING (current behavior) |
| Mask vs grid shape mismatch | masking skipped, logged at WARNING (current behavior) |
| Config can't be loaded / grid keys missing | CLI reports the config error + exit 1 |

No silent failures introduced; pure functions raise, the CLI is the only place that catches and formats for humans.

## Testing (CI — no creds, no network)

Synthetic source datasets are constructed in-test with xarray (small src grids, a handful of monthly steps), so everything runs in the clean `[dev]` venv.

**`tests/test_forcing_grid.py`**
- `target_coords` for a known `GridSpec` returns the expected cell centers, latitude descending.
- `regrid` maps a synthetic src cell to the nearest target cell (assert a known value lands where expected).
- `resample_to_24` is identity on 24-step input; interpolates a 12-step input to shape `(24, …)`.
- `cell_volume_m3` positive and scales linearly with depth.
- `apply_land_mask` NaNs land cells; shape-mismatch is a no-op.
- **grid-generality:** a `10×8` `GridSpec` produces `(…, 8, 10)` outputs.

**`tests/test_forcing_ltl.py`**
- Mode A: synthetic BGC with `phyc`,`zooc`,`chl`,`si` → dataset with the 6 named groups, dims `(24, nlat, nlon)`, all values finite & ≥ 0 on ocean cells, `mode` attr == direct.
- Mode B: synthetic BGC with only `chl`,`nppv` → 6 groups, `mode` attr == chl-derived.
- Neither pathway present → `ValueError`.
- Land cells NaN when an ocean mask is passed.

**`tests/test_forcing_physics.py`**
- Synthetic PHY with `thetao`,`so` → `{"temperature","salinity"}`, dims `(24, nlat, nlon)`, plausible ranges.
- PHY missing `so` → only `temperature`; missing both → `{}`.

**`tests/test_forcing_io.py`**
- `write_ltl` / `write_physics` round-trip: write → reopen → variables/dims/values match; latitude coordinate descending.

**`tests/test_forcing_mcp_parity.py`** (anti-drift)
- The refactored MCP wrapper and the core produce identical group set/structure/values for the Baltic `GridSpec` on a synthetic source — guards the extraction. Guarded with `find_spec` so it skips cleanly if MCP deps (`fastmcp`/`copernicusmarine`) are absent in the clean venv (the wrapper module imports them); the core-side equivalence is still covered by the Mode-A/B tests regardless.

## Files

- **Create:** `osmose/forcing/__init__.py`, `osmose/forcing/grid.py`, `osmose/forcing/ltl.py`, `osmose/forcing/physics.py`, `osmose/forcing/io.py`; `scripts/convert_cmems_forcing.py`; `tests/test_forcing_grid.py`, `tests/test_forcing_ltl.py`, `tests/test_forcing_physics.py`, `tests/test_forcing_io.py`, `tests/test_forcing_mcp_parity.py`.
- **Modify:** `mcp_servers/copernicus/server.py` (delegate the two `generate_*` tools; drop the now-shared helpers).

## Reused infrastructure

`osmose.maps.builder.GridSpec` (target grid from config), the existing config reader (CLI loads the target config), `osmose.logging` (WARNING for skipped masking), and core deps `numpy`/`xarray`/`scipy`.
