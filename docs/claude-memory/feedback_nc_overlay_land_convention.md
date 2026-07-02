---
name: NC overlay land convention inconsistency
description: EEC LTL writes NaN on land; Baltic LTL writes 0.0. load_netcdf_overlay now accepts an ocean_mask to normalise this.
type: feedback
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
LTL / forcing NetCDF generators in this repo use **two different sentinel conventions** for land cells:

- `data/eec_full/eec_ltlbiomassTons.nc` — **NaN on land** (EEC generator)
- `data/baltic/baltic_ltl_biomass.nc` — **0.0 on land** (Baltic CMEMS generator at `mcp_servers/copernicus/server.py`)

**Why:** prior to 2026-04-21, `load_netcdf_overlay` filtered only NaN, which let the Baltic zeros render over land. Shipped fix on 2026-04-21:

1. UI safety net — `load_netcdf_overlay(..., ocean_mask=bool_array)`. Caller supplies the grid mask (`mask >= 0` for CSV mask, `mask > 0` for NcGrid mask) and land cells are skipped regardless of value. `grid.py:update_grid_map` passes this automatically.
2. Generator aligned with EEC — `mcp_servers/copernicus/server.py:generate_osmose_ltl()` now calls `_apply_land_mask()` to stamp NaN on land using `baltic_grid.nc`'s mask. Future generator runs produce canonical output; the committed `data/baltic/baltic_ltl_biomass.nc` still has 0.0 on land but the UI handles it.

**How to apply:** when adding a new LTL / forcing NC generator, write NaN on land (matches EEC, matches xarray semantics). Tests: `TestNetcdfOverlayOceanMask` in `tests/test_overlay_display.py`; `tests/test_copernicus_ltl_mask.py` for the generator helper.
