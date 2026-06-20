# CMEMS→OSMOSE Forcing Conversion Core (Sub-project A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the LTL + physics conversion logic out of `mcp_servers/copernicus/server.py` into a pure, importable, grid-general `osmose/forcing/` package; expose it via a convert-only CLI; refactor the MCP tools to delegate to it.

**Architecture:** `osmose/forcing/` depends only on numpy/xarray/scipy + `osmose.maps.builder.GridSpec` (no copernicusmarine/fastmcp → runs in clean CI). Target grid comes from `GridSpec.from_config`. The MCP `generate_osmose_*` tools become thin wrappers; a `scripts/convert_cmems_forcing.py` CLI wraps the core for bring-your-own downloaded files.

**Tech Stack:** Python 3.12, numpy, xarray, scipy; pytest. No new dependencies.

---

## File Structure

- **Create:** `osmose/forcing/__init__.py`, `osmose/forcing/grid.py`, `osmose/forcing/ltl.py`, `osmose/forcing/physics.py`, `osmose/forcing/io.py`
- **Create:** `scripts/convert_cmems_forcing.py`
- **Create:** `tests/test_forcing_grid.py`, `tests/test_forcing_ltl.py`, `tests/test_forcing_physics.py`, `tests/test_forcing_io.py`, `tests/test_forcing_cli.py`, `tests/test_forcing_mcp_parity.py`
- **Modify:** `mcp_servers/copernicus/server.py` (delegate the two `generate_*` tools; drop the now-shared helpers)

Each module has one responsibility: `grid` = regrid/resample/mask geometry; `ltl` = BGC→6 groups; `physics` = PHY→temp/salinity; `io` = NetCDF writers.

Run tests with `.venv/bin/python -m pytest`. Lint with `.venv/bin/ruff check` + `.venv/bin/ruff format --check`.

---

## Task 1: Grid helpers (`osmose/forcing/grid.py`)

**Files:**
- Create: `osmose/forcing/__init__.py` (empty for now — populated in Task 4), `osmose/forcing/grid.py`
- Test: `tests/test_forcing_grid.py`

> **Why an empty `__init__.py` now:** `pyproject.toml` uses `setuptools` `packages.find` (NOT `find_namespace`) with `include = ["osmose*"]`, which only discovers packages that contain `__init__.py`. Without one, `osmose/forcing/` would import fine in editable/dev installs but be silently dropped from a built wheel/Docker image. Create it as a regular package from the first task.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forcing_grid.py
import numpy as np
import pytest

from osmose.forcing.grid import (
    apply_land_mask,
    cell_volume_m3,
    get_var,
    regrid,
    resample_to_24,
    target_coords,
)
from osmose.maps.builder import GridSpec

BALTIC = GridSpec(nlon=50, nlat=40, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30)
SMALL = GridSpec(nlon=10, nlat=8, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30)


def test_target_coords_descending_lat_ascending_lon():
    lat, lon = target_coords(BALTIC)
    assert lat.shape == (40,) and lon.shape == (50,)
    assert lat[0] > lat[-1]  # north -> south
    assert lon[0] < lon[-1]  # west -> east
    # cell centers, not edges
    assert lat[0] < 66 and lat[-1] > 54


def test_regrid_picks_nearest_src_cell():
    # NON-constant field so the test distinguishes correct nearest-neighbour
    # selection (and lat/lon orientation) from a degenerate/transposed impl.
    src_lat = np.array([66.0, 60.0, 54.0])  # descending
    src_lon = np.array([10.0, 20.0, 30.0])  # ascending
    # value encodes (lat_index, lon_index): 10*li + ci
    data = np.array([[[10 * li + ci for ci in range(3)] for li in range(3)]], dtype=float)
    out = regrid(data, src_lat, src_lon, SMALL)
    assert out.shape == (1, 8, 10)
    # northern-most target row (high lat) must map to src lat index 0; southern to 2
    assert out[0, 0, 0] == 0.0  # NW corner -> src (lat0, lon0)
    assert out[0, -1, -1] == 22.0  # SE corner -> src (lat2, lon2)
    assert out[0, 0, -1] == 2.0  # NE corner -> src (lat0, lon2)


def test_regrid_warns_when_target_exceeds_source_extent(caplog):
    import logging

    # source covers only a small central patch; SMALL grid spans the full Baltic box
    src_lat = np.array([60.0, 59.0])
    src_lon = np.array([19.0, 20.0])
    data = np.ones((1, 2, 2)) * 5.0
    with caplog.at_level(logging.WARNING):
        out = regrid(data, src_lat, src_lon, SMALL)
    assert out.shape == (1, 8, 10)
    assert any("beyond source coverage" in r.message for r in caplog.records)
    # out-of-coverage cells are nearest-edge filled (still the source value here)
    assert np.allclose(out, 5.0)


def test_resample_to_24_identity_and_interp():
    data24 = np.ones((24, 2, 2))
    assert resample_to_24(data24) is data24 or np.allclose(resample_to_24(data24), data24)
    data12 = np.arange(12)[:, None, None] * np.ones((12, 2, 2))
    out = resample_to_24(data12)
    assert out.shape == (24, 2, 2)
    assert out.min() >= 0 and out.max() <= 11


def test_cell_volume_positive_and_scales_with_depth():
    v10 = cell_volume_m3(BALTIC, 10.0)
    v50 = cell_volume_m3(BALTIC, 50.0)
    assert v10 > 0
    assert np.isclose(v50, 5 * v10)


def test_apply_land_mask_nans_land_and_noops_on_mismatch():
    groups = {"A": np.ones((24, 8, 10))}
    mask = np.ones((8, 10), dtype=bool)
    mask[0, 0] = False  # one land cell
    apply_land_mask(groups, mask)
    assert np.isnan(groups["A"][0, 0, 0])
    assert not np.isnan(groups["A"][0, 1, 1])
    # shape mismatch -> no-op (no raise)
    g2 = {"B": np.ones((24, 4, 4))}
    apply_land_mask(g2, mask)
    assert not np.any(np.isnan(g2["B"]))


def test_get_var_promotes_2d_and_fills_nan():
    import xarray as xr

    ds = xr.Dataset({"x": (["latitude", "longitude"], np.array([[1.0, np.nan], [3.0, 4.0]]))})
    arr = get_var(ds, "x")
    assert arr.shape == (1, 2, 2)
    assert arr[0, 0, 1] == 0.0  # NaN -> 0
    assert get_var(ds, "missing") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_forcing_grid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.forcing'`

- [ ] **Step 3: Write the implementation**

```python
# osmose/forcing/grid.py
"""Grid geometry helpers for CMEMS->OSMOSE forcing conversion.

Pure: numpy/xarray + osmose.maps.builder.GridSpec only. No CMEMS/MCP deps,
so this module (and its tests) run in the clean CI venv.

regrid/resample use O(nlat*nlon) Python loops (verbatim from the MCP source),
intended for OSMOSE-scale config grids (~1e3-1e4 cells, coarse by construction).
A much finer grid would want a vectorized scipy.spatial.cKDTree / np.searchsorted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from osmose.logging import setup_logging
from osmose.maps.builder import GridSpec

_log = setup_logging("osmose.forcing")


def target_coords(grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """Cell-center (lat[nlat], lon[nlon]); latitude descending (north->south)."""
    rows = np.arange(grid.nlat)
    cols = np.arange(grid.nlon)
    lat = grid.upleft_lat - (rows + 0.5) * grid.dy
    lon = grid.upleft_lon + (cols + 0.5) * grid.dx
    return lat, lon


def regrid(
    data_3d: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray, grid: GridSpec
) -> np.ndarray:
    """Nearest-neighbor regrid (time, src_lat, src_lon) -> (time, nlat, nlon).

    Warns once when the target grid extends beyond the source data extent:
    out-of-coverage cells are silently filled by nearest-EDGE extrapolation
    (argmin always returns an in-bounds index). The MCP path was implicitly
    safe (download bbox always covered the grid); the grid-general + BYO-file
    surface this package adds removes that guarantee, so surface it loudly.
    """
    tlat, tlon = target_coords(grid)
    if (
        tlat.min() < src_lat.min()
        or tlat.max() > src_lat.max()
        or tlon.min() < src_lon.min()
        or tlon.max() > src_lon.max()
    ):
        _log.warning(
            "target grid (lat %.2f-%.2f lon %.2f-%.2f) extends beyond source "
            "coverage (lat %.2f-%.2f lon %.2f-%.2f); out-of-coverage cells are "
            "filled by nearest-edge extrapolation",
            tlat.min(), tlat.max(), tlon.min(), tlon.max(),
            src_lat.min(), src_lat.max(), src_lon.min(), src_lon.max(),
        )
    nlat, nlon = len(tlat), len(tlon)
    nt = data_3d.shape[0]
    out = np.zeros((nt, nlat, nlon), dtype=np.float64)
    for j in range(nlat):
        lat_idx = int(np.argmin(np.abs(src_lat - tlat[j])))
        for i in range(nlon):
            lon_idx = int(np.argmin(np.abs(src_lon - tlon[i])))
            out[:, j, i] = data_3d[:, lat_idx, lon_idx]
    return out


def resample_to_24(data: np.ndarray) -> np.ndarray:
    """Linear-interpolate (time, lat, lon) to 24 biweekly steps; identity if already 24."""
    nt, nlat, nlon = data.shape
    if nt == 24:
        return data
    out = np.zeros((24, nlat, nlon), dtype=np.float64)
    xp = np.linspace(0, 1, nt)
    x = np.linspace(0, 1, 24)
    for j in range(nlat):
        for i in range(nlon):
            out[:, j, i] = np.interp(x, xp, data[:, j, i])
    return out


def cell_volume_m3(grid: GridSpec, depth_m: float) -> float:
    """Approximate cell volume (m^3) using the grid's mid-latitude cos factor."""
    mid_lat = (grid.upleft_lat + grid.lowright_lat) / 2.0
    cos_lat = np.cos(np.radians(mid_lat))
    area = (abs(grid.dy) * 111320) * (abs(grid.dx) * 111320 * cos_lat)
    return float(area * depth_m)


def get_coords(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Extract lat/lon arrays, tolerating 'latitude'/'lat' and 'longitude'/'lon'."""
    lat = ds.latitude.values if "latitude" in ds.coords else ds.lat.values
    lon = ds.longitude.values if "longitude" in ds.coords else ds.lon.values
    return lat, lon


def get_var(ds: xr.Dataset, name: str) -> np.ndarray | None:
    """Variable as 3D (time, lat, lon), NaN->0; None if absent."""
    if name not in ds:
        return None
    arr = np.nan_to_num(ds[name].values, nan=0.0)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    return arr


def load_ocean_mask(grid_file: Path | None) -> np.ndarray | None:
    """Load a bool (nlat, nlon) ocean mask (True=ocean) from a grid NetCDF, or None."""
    if grid_file is None or not Path(grid_file).exists():
        return None
    try:
        with xr.open_dataset(grid_file) as gds:
            return gds["mask"].values.astype(bool)
    except (OSError, KeyError):
        return None


def apply_land_mask(groups: dict[str, np.ndarray], ocean_mask: np.ndarray) -> None:
    """Set land cells to NaN in every (time, lat, lon) array, in place.

    Shape mismatch is a logged no-op so conversion still runs with a stale grid file.
    """
    for arr in groups.values():
        if arr.shape[1:] != ocean_mask.shape:
            _log.warning(
                "ocean mask %s does not match data grid %s; skipping land mask",
                ocean_mask.shape,
                arr.shape[1:],
            )
            return
        arr[:, ~ocean_mask] = np.nan
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_forcing_grid.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

First create the empty package marker, then commit both:

```bash
touch osmose/forcing/__init__.py
git add osmose/forcing/__init__.py osmose/forcing/grid.py tests/test_forcing_grid.py
git commit -m "feat(forcing): grid-parameterized regrid/resample/mask helpers"
```

---

## Task 2: LTL conversion (`osmose/forcing/ltl.py`)

**Files:**
- Create: `osmose/forcing/ltl.py`
- Test: `tests/test_forcing_ltl.py`

This ports `generate_osmose_ltl`'s Mode A / Mode B math verbatim, parameterized by `GridSpec` and `LtlParams`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forcing_ltl.py
import numpy as np
import pytest
import xarray as xr

from osmose.forcing.ltl import bgc_to_ltl
from osmose.maps.builder import GridSpec

SMALL = GridSpec(nlon=10, nlat=8, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30)
GROUPS = ["Diatoms", "Dinoflagellates", "Microzooplankton", "Mesozooplankton", "Macrozooplankton", "Benthos"]


def _src(vars_):
    lat = np.linspace(66, 54, 4)
    lon = np.linspace(10, 30, 5)
    nt = 12
    data = {k: (["time", "latitude", "longitude"], np.abs(np.ones((nt, 4, 5)) * v)) for k, v in vars_.items()}
    return xr.Dataset(data, coords={"time": np.arange(nt), "latitude": lat, "longitude": lon})


def test_mode_a_direct_biomass():
    ds = _src({"phyc": 10.0, "zooc": 5.0, "chl": 2.0, "nppv": 100.0})
    out = bgc_to_ltl(ds, SMALL)
    assert set(GROUPS).issubset(out.data_vars)
    for g in GROUPS:
        assert out[g].shape == (24, 8, 10)
        vals = out[g].values
        assert np.all(np.nan_to_num(vals) >= 0)
        assert np.all(np.isfinite(np.nan_to_num(vals)))
    assert "direct" in out.attrs["mode"].lower()


def test_mode_b_chl_derived():
    ds = _src({"chl": 2.0, "nppv": 100.0})
    out = bgc_to_ltl(ds, SMALL)
    assert set(GROUPS).issubset(out.data_vars)
    assert "chl" in out.attrs["mode"].lower() or "b" in out.attrs["mode"].lower()


def test_missing_all_inputs_raises():
    ds = _src({"o2": 200.0})
    with pytest.raises(ValueError, match="phyc|chl"):
        bgc_to_ltl(ds, SMALL)


def test_land_mask_applied():
    ds = _src({"phyc": 10.0, "zooc": 5.0})
    mask = np.ones((8, 10), dtype=bool)
    mask[0, 0] = False
    out = bgc_to_ltl(ds, SMALL, ocean_mask=mask)
    assert np.isnan(out["Diatoms"].values[0, 0, 0])


def test_depth_slice_empty_raises():
    # all source depth levels are deeper than depth_integrate_m -> empty slice -> raise
    # (not a silent all-zero forcing). Also covers a descending depth axis via sortby.
    lat = np.linspace(66, 54, 4)
    lon = np.linspace(10, 30, 5)
    ds = xr.Dataset(
        {"phyc": (["time", "depth", "latitude", "longitude"], np.ones((12, 1, 4, 5)) * 10.0),
         "zooc": (["time", "depth", "latitude", "longitude"], np.ones((12, 1, 4, 5)) * 5.0)},
        coords={"time": np.arange(12), "depth": [100.0], "latitude": lat, "longitude": lon},
    )
    with pytest.raises(ValueError, match="depth"):
        bgc_to_ltl(ds, SMALL, depth_integrate_m=50.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_forcing_ltl.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# osmose/forcing/ltl.py
"""BGC NetCDF -> OSMOSE 6-group LTL forcing. Pure port of the MCP logic.

IMPORTANT: the default coefficients in LtlParams are BALTIC-CALIBRATED (carried
verbatim from the MCP source). The seasonal diatom_frac arrays encode Northern-
Hemisphere Baltic phytoplankton succession and assume Jan-start MONTHLY input
(index 0 = January). The C:wet ratios were calibrated against Baltic standing
stock. The conversion regrids to ANY config grid, but these coefficients are NOT
validated for other seas / hemispheres — non-Baltic use needs explicit params.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from osmose.forcing.grid import (
    apply_land_mask,
    cell_volume_m3,
    get_coords,
    get_var,
    regrid,
    resample_to_24,
    target_coords,
)
from osmose.maps.builder import GridSpec

GROUP_NAMES = [
    "Diatoms",
    "Dinoflagellates",
    "Microzooplankton",
    "Mesozooplankton",
    "Macrozooplankton",
    "Benthos",
]


@dataclass(frozen=True)
class LtlParams:
    """LTL conversion coefficients. Defaults are BALTIC-calibrated (see module docstring)."""

    # Mode A (direct phyc/zooc biomass)
    phyto_c_to_wet: float = 0.012  # gC/mmol * 1:1 wet:C (Baltic standing-stock calibrated)
    zoo_c_to_wet: float = 0.12  # gC/mmol * 10:1 wet:C (crustacean zooplankton)
    # Baltic phytoplankton succession: spring diatom bloom, summer cyano/dino (NH, Jan-start)
    diatom_frac_a: tuple[float, ...] = (
        0.40, 0.60, 0.75, 0.80, 0.70, 0.40, 0.25, 0.20, 0.25, 0.35, 0.40, 0.40,
    )
    micro_frac: float = 0.40  # Baltic zoo size split: ~40% micro
    meso_frac: float = 0.45  # ~45% meso (copepods)
    macro_frac: float = 0.15  # ~15% macro (mysids, krill)
    # Mode B (chl-derived)
    chl_to_biomass_factor: float = 50.0
    diatom_frac_b: tuple[float, ...] = (
        0.3, 0.5, 0.7, 0.8, 0.7, 0.5, 0.3, 0.2, 0.2, 0.3, 0.3, 0.3,
    )
    micro_npp_div: float = 50.0
    meso_npp_div: float = 15.0
    macro_npp_div: float = 8.0
    micro_npp_frac: float = 0.30
    meso_npp_frac: float = 0.10
    macro_npp_frac: float = 0.03
    # Benthos: shared across modes — reproduces server.py:478 (Mode A) / :512 (Mode B)
    # as npp_tonnes * benthos_npp_frac / benthos_npp_div = npp_tonnes * 0.05 / 3.0.
    benthos_npp_frac: float = 0.05
    benthos_npp_div: float = 3.0
    benthos_zoo_frac: float = 0.3  # Mode-A fallback when nppv absent


def _seasonal(frac_tuple: tuple[float, ...], n_steps: int, nlat: int, nlon: int) -> np.ndarray:
    if n_steps == 12:
        f = np.array(frac_tuple)[:, np.newaxis, np.newaxis] * np.ones((1, nlat, nlon))
    else:
        f = np.ones((n_steps, nlat, nlon)) * 0.5
    return f


def bgc_to_ltl(
    ds: xr.Dataset,
    grid: GridSpec,
    *,
    year: int = 0,
    depth_integrate_m: float = 50.0,
    params: LtlParams = LtlParams(),
    ocean_mask: np.ndarray | None = None,
) -> xr.Dataset:
    """Convert CMEMS biogeochemistry into OSMOSE 6-group LTL forcing.

    Mode A (phyc+zooc present): direct carbon biomass. Mode B (chl only): chl-derived.
    Raises ValueError if neither pathway's variables are present.
    """
    tlat, tlon = target_coords(grid)
    nlat, nlon = len(tlat), len(tlon)
    cell_vol = cell_volume_m3(grid, depth_integrate_m)

    work = ds
    if year > 0 and "time" in work.dims:
        work = work.sel(time=work.time.dt.year == year)
    if "depth" in work.dims:
        # sortby so the slice works regardless of depth-axis order; raise (not
        # silently produce an all-zero field) if no levels fall in the range.
        work = work.sortby("depth")
        sliced = work.sel(depth=slice(0, depth_integrate_m))
        if sliced.sizes.get("depth", 0) == 0:
            raise ValueError(
                f"no source depth levels within [0, {depth_integrate_m}] m; "
                f"source depth range is [{float(work.depth.min())}, {float(work.depth.max())}]"
            )
        work = sliced.mean(dim="depth", skipna=True)

    src_lat, src_lon = get_coords(work)
    has_phyc = "phyc" in work
    has_zooc = "zooc" in work
    mode = "A (direct biomass)" if (has_phyc and has_zooc) else "B (chl-derived)"

    if has_phyc and has_zooc:
        phyc = get_var(work, "phyc")
        zooc = get_var(work, "zooc")
        nppv = get_var(work, "nppv")
        phyto_tonnes = regrid(phyc, src_lat, src_lon, grid) * params.phyto_c_to_wet * cell_vol / 1e6
        zoo_tonnes = regrid(zooc, src_lat, src_lon, grid) * params.zoo_c_to_wet * cell_vol / 1e6

        n_steps = phyto_tonnes.shape[0]
        diatom_frac = _seasonal(params.diatom_frac_a, n_steps, nlat, nlon)
        diatoms = phyto_tonnes * diatom_frac
        dinoflagellates = phyto_tonnes * (1.0 - diatom_frac)

        microzoo = zoo_tonnes * params.micro_frac
        mesozoo = zoo_tonnes * params.meso_frac
        macrozoo = zoo_tonnes * params.macro_frac

        if nppv is not None:
            npp_tonnes = regrid(nppv, src_lat, src_lon, grid) * cell_vol / 1e9 * 365
            benthos = npp_tonnes * params.benthos_npp_frac / params.benthos_npp_div
        else:
            benthos = zoo_tonnes * params.benthos_zoo_frac
    else:
        chl = get_var(work, "chl")
        nppv = get_var(work, "nppv")
        if chl is None:
            raise ValueError(
                "BGC source has neither phyc/zooc nor chl. Provide phyc,zooc[,nppv,si] "
                "(forecast) or chl,nppv (reanalysis)."
            )
        if nppv is None:
            nppv = chl * 5.0

        chl_grid = regrid(chl, src_lat, src_lon, grid)
        nppv_grid = regrid(nppv, src_lat, src_lon, grid)
        phyto_tonnes = chl_grid * params.chl_to_biomass_factor * cell_vol / 1e9

        n_steps = chl_grid.shape[0]
        diatom_frac = _seasonal(params.diatom_frac_b, n_steps, nlat, nlon)
        diatoms = phyto_tonnes * diatom_frac
        dinoflagellates = phyto_tonnes * (1.0 - diatom_frac)

        npp_tonnes = nppv_grid * cell_vol / 1e9 * 365
        microzoo = npp_tonnes * params.micro_npp_frac / params.micro_npp_div
        mesozoo = npp_tonnes * params.meso_npp_frac / params.meso_npp_div
        macrozoo = npp_tonnes * params.macro_npp_frac / params.macro_npp_div
        benthos = npp_tonnes * params.benthos_npp_frac / params.benthos_npp_div

    groups = {
        "Diatoms": resample_to_24(diatoms),
        "Dinoflagellates": resample_to_24(dinoflagellates),
        "Microzooplankton": resample_to_24(microzoo),
        "Mesozooplankton": resample_to_24(mesozoo),
        "Macrozooplankton": resample_to_24(macrozoo),
        "Benthos": resample_to_24(benthos),
    }
    for arr in groups.values():
        arr[arr < 0] = 0.0

    if ocean_mask is not None:
        apply_land_mask(groups, ocean_mask)

    return xr.Dataset(
        {name: (["time", "latitude", "longitude"], data) for name, data in groups.items()},
        coords={"time": np.arange(24), "latitude": tlat, "longitude": tlon},
        attrs={
            "title": "OSMOSE LTL Forcing (from CMEMS)",
            "mode": mode,
            "description": "6 lower trophic level groups, 24 biweekly timesteps",
            "depth_integration_m": depth_integrate_m,
            "calibration": (
                "Baltic Sea (CMEMS BAL products); coefficients (C:wet, seasonal "
                "splits) not validated for other seas/hemispheres"
            ),
            "seasonal_split_assumption": (
                "diatom_frac mapped positionally as Jan-start monthly, Northern-"
                "Hemisphere phenology; n_steps!=12 uses a flat 0.5 split"
            ),
            "conventions": "Latitude descending (north to south) to match grid.nc; NaN on land",
        },
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_forcing_ltl.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/forcing/ltl.py tests/test_forcing_ltl.py
git commit -m "feat(forcing): bgc_to_ltl 6-group conversion (Mode A/B), grid-general"
```

---

## Task 3: Physics conversion (`osmose/forcing/physics.py`)

**Files:**
- Create: `osmose/forcing/physics.py`
- Test: `tests/test_forcing_physics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forcing_physics.py
import numpy as np
import xarray as xr

from osmose.forcing.physics import phy_to_physics
from osmose.maps.builder import GridSpec

SMALL = GridSpec(nlon=10, nlat=8, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30)


def _src(vars_):
    lat = np.linspace(66, 54, 4)
    lon = np.linspace(10, 30, 5)
    data = {k: (["time", "latitude", "longitude"], np.ones((12, 4, 5)) * v) for k, v in vars_.items()}
    return xr.Dataset(data, coords={"time": np.arange(12), "latitude": lat, "longitude": lon})


def test_both_vars_present():
    out = phy_to_physics(_src({"thetao": 8.0, "so": 7.5}), SMALL)
    assert set(out) == {"temperature", "salinity"}
    assert out["temperature"]["temperature"].shape == (24, 8, 10)
    assert np.allclose(out["temperature"]["temperature"].values, 8.0)
    assert np.allclose(out["salinity"]["salinity"].values, 7.5)


def test_missing_salinity_omitted():
    out = phy_to_physics(_src({"thetao": 8.0}), SMALL)
    assert set(out) == {"temperature"}


def test_missing_both_returns_empty():
    out = phy_to_physics(_src({"o2": 200.0}), SMALL)
    assert out == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_forcing_physics.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# osmose/forcing/physics.py
"""PHY NetCDF -> OSMOSE temperature/salinity forcing. Pure port of the MCP logic."""

from __future__ import annotations

import numpy as np
import xarray as xr

from osmose.forcing.grid import get_coords, get_var, regrid, resample_to_24, target_coords
from osmose.logging import setup_logging
from osmose.maps.builder import GridSpec

_log = setup_logging("osmose.forcing")

_VARS = [("thetao", "temperature", "degC"), ("so", "salinity", "PSU")]


def phy_to_physics(
    ds: xr.Dataset,
    grid: GridSpec,
    *,
    year: int = 0,
    depth_surface_m: float = 10.0,
) -> dict[str, xr.Dataset]:
    """Convert CMEMS physics into OSMOSE temperature/salinity forcing datasets.

    Returns a dict keyed by 'temperature'/'salinity' for whichever source
    variables are present; an empty dict if neither thetao nor so exists.
    """
    tlat, tlon = target_coords(grid)

    work = ds
    if year > 0 and "time" in work.dims:
        work = work.sel(time=work.time.dt.year == year)
    if "depth" in work.dims:
        work = work.sel(depth=depth_surface_m, method="nearest")

    src_lat, src_lon = get_coords(work)
    out: dict[str, xr.Dataset] = {}
    for src_name, osmose_name, units in _VARS:
        data = get_var(work, src_name)
        if data is None:
            _log.info("physics: %s not found in source, skipped", src_name)
            continue
        regridded = resample_to_24(regrid(data, src_lat, src_lon, grid))
        out[osmose_name] = xr.Dataset(
            {osmose_name: (["time", "latitude", "longitude"], regridded)},
            coords={"time": np.arange(24), "latitude": tlat, "longitude": tlon},
            attrs={
                "title": f"OSMOSE {osmose_name.title()} Forcing (from CMEMS)",
                "units": units,
                "depth_m": depth_surface_m,
                "conventions": "Latitude descending (north to south) to match grid.nc",
            },
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_forcing_physics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/forcing/physics.py tests/test_forcing_physics.py
git commit -m "feat(forcing): phy_to_physics temperature/salinity conversion"
```

---

## Task 4: NetCDF writers + package exports (`osmose/forcing/io.py`, `__init__.py`)

**Files:**
- Create: `osmose/forcing/io.py`
- Modify: `osmose/forcing/__init__.py` (populate the empty marker created in Task 1)
- Test: `tests/test_forcing_io.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forcing_io.py
import numpy as np
import pytest
import xarray as xr

from osmose.forcing import bgc_to_ltl, phy_to_physics, write_ltl, write_physics
from osmose.maps.builder import GridSpec

SMALL = GridSpec(nlon=10, nlat=8, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30)


def _bgc():
    lat = np.linspace(66, 54, 4)
    lon = np.linspace(10, 30, 5)
    return xr.Dataset(
        {"phyc": (["time", "latitude", "longitude"], np.ones((12, 4, 5)) * 10.0),
         "zooc": (["time", "latitude", "longitude"], np.ones((12, 4, 5)) * 5.0)},
        coords={"time": np.arange(12), "latitude": lat, "longitude": lon},
    )


def test_write_ltl_roundtrip(tmp_path):
    ds = bgc_to_ltl(_bgc(), SMALL)
    path = write_ltl(ds, tmp_path / "ltl.nc")
    assert path.exists()
    reopened = xr.open_dataset(path)
    assert "Diatoms" in reopened.data_vars
    assert reopened["Diatoms"].shape == (24, 8, 10)
    assert float(reopened.latitude[0]) > float(reopened.latitude[-1])  # descending
    reopened.close()


def test_write_ltl_refuses_clobber(tmp_path):
    ds = bgc_to_ltl(_bgc(), SMALL)
    path = tmp_path / "ltl.nc"
    write_ltl(ds, path)
    with pytest.raises(FileExistsError):
        write_ltl(ds, path)  # default overwrite=False
    write_ltl(ds, path, overwrite=True)  # explicit overwrite OK


def test_write_physics_roundtrip(tmp_path):
    src = xr.Dataset(
        {"thetao": (["time", "latitude", "longitude"], np.ones((12, 4, 5)) * 8.0)},
        coords={"time": np.arange(12), "latitude": np.linspace(66, 54, 4), "longitude": np.linspace(10, 30, 5)},
    )
    dsets = phy_to_physics(src, SMALL)
    paths = write_physics(dsets, tmp_path, prefix="test")
    assert (tmp_path / "test_temperature.nc").exists()
    # write_physics returns a {name: path} mapping
    assert paths["temperature"].name == "test_temperature.nc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_forcing_io.py -v`
Expected: FAIL — `ImportError` (no `write_ltl` in package)

- [ ] **Step 3: Write the implementation**

```python
# osmose/forcing/io.py
"""NetCDF writers for OSMOSE forcing datasets."""

from __future__ import annotations

from pathlib import Path

import xarray as xr


def write_ltl(ds: xr.Dataset, path: Path | str, *, overwrite: bool = False) -> Path:
    """Write an LTL forcing dataset to NetCDF; returns the path.

    Refuses to clobber an existing file unless overwrite=True (the convert CLI
    is a new write surface; the in-tree Baltic forcing asset must not be
    silently destroyed). The MCP wrapper passes overwrite=True to preserve its
    always-regenerate behavior.
    """
    out = Path(path)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass overwrite=True (CLI: --force) to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(str(out))
    return out


def write_physics(
    dsets: dict[str, xr.Dataset],
    out_dir: Path | str,
    prefix: str = "baltic",
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write each physics dataset to f'{prefix}_{name}.nc'; returns {name: path}.

    Returns a name->path mapping (not a bare list) so consumers get the
    explicit pairing. Refuses to clobber unless overwrite=True.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, ds in dsets.items():
        fpath = out / f"{prefix}_{name}.nc"
        if fpath.exists() and not overwrite:
            raise FileExistsError(
                f"{fpath} exists; pass overwrite=True (CLI: --force) to replace it"
            )
        ds.to_netcdf(str(fpath))
        paths[name] = fpath
    return paths
```

```python
# osmose/forcing/__init__.py  (replace the empty marker from Task 1 with this)
"""Pure CMEMS->OSMOSE forcing conversion (grid-general, browser/MCP-free)."""

from osmose.forcing.grid import (
    apply_land_mask,
    cell_volume_m3,
    load_ocean_mask,
    regrid,
    resample_to_24,
    target_coords,
)
from osmose.forcing.io import write_ltl, write_physics
from osmose.forcing.ltl import GROUP_NAMES, LtlParams, bgc_to_ltl
from osmose.forcing.physics import phy_to_physics

__all__ = [
    "GROUP_NAMES",
    "LtlParams",
    "apply_land_mask",
    "bgc_to_ltl",
    "cell_volume_m3",
    "load_ocean_mask",
    "phy_to_physics",
    "regrid",
    "resample_to_24",
    "target_coords",
    "write_ltl",
    "write_physics",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_forcing_io.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/forcing/io.py osmose/forcing/__init__.py tests/test_forcing_io.py
git commit -m "feat(forcing): NetCDF writers + package public API"
```

---

## Task 5: Convert-only CLI (`scripts/convert_cmems_forcing.py`)

**Files:**
- Create: `scripts/convert_cmems_forcing.py`
- Test: `tests/test_forcing_cli.py`

The CLI loads a config (to build the target grid), opens a downloaded source NetCDF, calls the core, writes outputs. Conversion logic stays in the importable `_run(...)` function so the test exercises it without subprocess.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forcing_cli.py
import numpy as np
import xarray as xr

from scripts.convert_cmems_forcing import _run


def _write_bgc(path):
    ds = xr.Dataset(
        {"phyc": (["time", "latitude", "longitude"], np.ones((12, 4, 5)) * 10.0),
         "zooc": (["time", "latitude", "longitude"], np.ones((12, 4, 5)) * 5.0)},
        coords={"time": np.arange(12), "latitude": np.linspace(66, 54, 4), "longitude": np.linspace(10, 30, 5)},
    )
    ds.to_netcdf(str(path))


def _grid_cfg():
    return {
        "grid.nlon": "10", "grid.nlat": "8",
        "grid.upleft.lat": "66", "grid.upleft.lon": "10",
        "grid.lowright.lat": "54", "grid.lowright.lon": "30",
    }


def test_cli_run_ltl(tmp_path, monkeypatch):
    src = tmp_path / "bgc.nc"
    _write_bgc(src)
    out = tmp_path / "ltl.nc"
    # _run takes a pre-resolved config dict + grid_file to stay unit-testable
    rc = _run(source=str(src), config=_grid_cfg(), kind="ltl", out=str(out), grid_file=None)
    assert rc == 0
    assert out.exists()
    reopened = xr.open_dataset(out)
    assert "Diatoms" in reopened.data_vars
    reopened.close()


def test_cli_run_missing_vars_returns_nonzero(tmp_path):
    src = tmp_path / "bad.nc"
    xr.Dataset(
        {"o2": (["time", "latitude", "longitude"], np.ones((12, 4, 5)))},
        coords={"time": np.arange(12), "latitude": np.linspace(66, 54, 4), "longitude": np.linspace(10, 30, 5)},
    ).to_netcdf(str(src))
    rc = _run(source=str(src), config=_grid_cfg(), kind="ltl", out=str(tmp_path / "x.nc"), grid_file=None)
    assert rc != 0


def test_cli_resolves_config_directory(tmp_path):
    # A directory containing a single *all-parameters*.csv master resolves cleanly.
    cfgdir = tmp_path / "cfg"
    cfgdir.mkdir()
    (cfgdir / "x_all-parameters.csv").write_text(
        "grid.nlon ; 10\ngrid.nlat ; 8\ngrid.upleft.lat ; 66\ngrid.upleft.lon ; 10\n"
        "grid.lowright.lat ; 54\ngrid.lowright.lon ; 30\n"
    )
    src = tmp_path / "bgc.nc"
    _write_bgc(src)
    rc = _run(source=str(src), config=str(cfgdir), kind="ltl", out=str(tmp_path / "ltl.nc"), grid_file=None)
    assert rc == 0


def test_cli_refuses_clobber_without_force(tmp_path):
    src = tmp_path / "bgc.nc"
    _write_bgc(src)
    out = tmp_path / "ltl.nc"
    assert _run(source=str(src), config=_grid_cfg(), kind="ltl", out=str(out), grid_file=None) == 0
    # second run without force -> FileExistsError caught -> nonzero
    assert _run(source=str(src), config=_grid_cfg(), kind="ltl", out=str(out), grid_file=None) != 0
    # with force -> succeeds
    assert _run(source=str(src), config=_grid_cfg(), kind="ltl", out=str(out), grid_file=None, force=True) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_forcing_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.convert_cmems_forcing'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/convert_cmems_forcing.py
"""Convert a downloaded CMEMS NetCDF into OSMOSE forcing (convert-only, no download).

Usage:
  convert_cmems_forcing.py --source bgc.nc --config data/baltic --kind ltl --out ltl.nc
  convert_cmems_forcing.py --source phy.nc --config data/baltic --kind physics --out out_dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

from osmose.config.reader import OsmoseConfigReader
from osmose.forcing import (
    bgc_to_ltl,
    load_ocean_mask,
    phy_to_physics,
    write_ltl,
    write_physics,
)
from osmose.logging import setup_logging
from osmose.maps.builder import GridSpec

_log = setup_logging("osmose.forcing.cli")


def _load_config(config: str | dict) -> dict:
    """Accept a pre-resolved dict (tests) or a path to a config DIR or master file."""
    if isinstance(config, dict):
        return config
    p = Path(config)
    if p.is_dir():
        masters = sorted(p.glob("*all-parameters*.csv"))
        if len(masters) != 1:
            raise ValueError(
                f"expected exactly one *all-parameters*.csv in {p}, found {len(masters)}: "
                f"{[m.name for m in masters]}; pass the master file directly"
            )
        p = masters[0]
    reader = OsmoseConfigReader()
    return reader.read(p)  # read(master_file: Path) -> dict[str, str]


def _run(
    *,
    source: str,
    config: str | dict,
    kind: str,
    out: str,
    grid_file: str | None = None,
    year: int = 0,
    depth_integrate_m: float = 50.0,
    depth_surface_m: float = 10.0,
    prefix: str = "baltic",
    force: bool = False,
) -> int:
    """Core CLI logic; returns a process exit code."""
    try:
        cfg = _load_config(config)
        grid = GridSpec.from_config(cfg)
        # A produces 24 biweekly steps; warn (don't fail) if the config wants otherwise,
        # so a future non-24 target is visible rather than silently mismatched.
        ndt = cfg.get("simulation.time.ndtPerYear")
        if ndt is not None and str(ndt).strip() not in ("", "24"):
            _log.warning(
                "config simulation.time.ndtPerYear=%s but forcing is generated with 24 "
                "biweekly steps (sub-project A is 24-step only)",
                ndt,
            )
        src = Path(source)
        if not src.exists():
            _log.error("source file not found: %s", source)
            return 1
        mask = load_ocean_mask(Path(grid_file)) if grid_file else None
        ds = xr.open_dataset(src)
        try:
            if kind == "ltl":
                result = bgc_to_ltl(
                    ds, grid, year=year, depth_integrate_m=depth_integrate_m, ocean_mask=mask
                )
                path = write_ltl(result, out, overwrite=force)
                _log.info("wrote LTL forcing: %s", path)
                for g in result.data_vars:
                    _log.info("  %s: total=%.0f t", g, float(result[g].sum(skipna=True)))
            elif kind == "physics":
                dsets = phy_to_physics(ds, grid, year=year, depth_surface_m=depth_surface_m)
                if not dsets:
                    _log.error("no physics variables (thetao/so) found in source")
                    return 1
                paths = write_physics(dsets, out, prefix=prefix, overwrite=force)
                _log.info(
                    "wrote physics forcing: %s",
                    ", ".join(str(p) for p in paths.values()),
                )
            else:
                _log.error("unknown --kind %r (use 'ltl' or 'physics')", kind)
                return 2
        finally:
            ds.close()
    except (ValueError, OSError, KeyError) as exc:  # FileExistsError ⊂ OSError
        _log.error("conversion failed: %s", exc)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert a downloaded CMEMS file to OSMOSE forcing")
    p.add_argument("--source", required=True, help="downloaded CMEMS NetCDF (BGC or PHY)")
    p.add_argument(
        "--config",
        required=True,
        help="config directory or master config file (for the target grid)",
    )
    p.add_argument("--kind", required=True, choices=["ltl", "physics"])
    p.add_argument("--out", required=True, help="output file (ltl) or directory (physics)")
    p.add_argument("--grid-file", default=None, help="grid NetCDF for the ocean mask (optional)")
    p.add_argument("--year", type=int, default=0, help="year to extract (0 = all available)")
    p.add_argument(
        "--depth-integrate",
        type=float,
        default=50.0,
        help="LTL integration depth (m); default 50 is Baltic-tuned, set per region",
    )
    p.add_argument(
        "--depth-surface",
        type=float,
        default=10.0,
        help="physics surface-layer depth (m); default 10 is Baltic-tuned",
    )
    p.add_argument("--prefix", default="baltic", help="physics output filename prefix")
    p.add_argument("--force", action="store_true", help="overwrite existing output file(s)")
    a = p.parse_args(argv)
    return _run(
        source=a.source,
        config=a.config,
        kind=a.kind,
        out=a.out,
        grid_file=a.grid_file,
        year=a.year,
        depth_integrate_m=a.depth_integrate,
        depth_surface_m=a.depth_surface,
        prefix=a.prefix,
        force=a.force,
    )


if __name__ == "__main__":
    sys.exit(main())
```

NOTE: the reader API is confirmed — `OsmoseConfigReader().read(master_file: Path) -> dict[str, str]` (see `osmose/cli.py:21-22`, `osmose/config/reader.py:80`). `--config` is a master file path. The dict-passthrough branch keeps the unit tests independent of file I/O.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_forcing_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/convert_cmems_forcing.py tests/test_forcing_cli.py
git commit -m "feat(forcing): convert-only CLI (bring-your-own downloaded file)"
```

---

## Task 6: Refactor the MCP server to delegate + anti-drift parity test

**Files:**
- Modify: `mcp_servers/copernicus/server.py`
- Modify: `tests/test_copernicus_ltl_mask.py` (repoint — it imports the deleted helpers)
- Test: `tests/test_forcing_mcp_parity.py`

The MCP `generate_osmose_ltl` / `generate_osmose_physics` lose their inline math and call the core with a Baltic `GridSpec`. The MCP-local helpers (`_make_target_coords`, `_regrid`, `_resample_to_24`, `_cell_volume_m3`, `_get_coords`, `_get_var`, `_load_baltic_ocean_mask`, `_apply_land_mask`) and the `OSMOSE_GRID` constant are removed.

**Existing-test breakage (must fix in this task):** `tests/test_copernicus_ltl_mask.py` imports `_apply_land_mask` / `_load_baltic_ocean_mask` from the MCP server inside its test bodies (after a module-level `pytest.importorskip("mcp_servers.copernicus.server")`). In the dev venv the MCP deps are present so `importorskip` does NOT skip — deleting the helpers turns those 3 tests into ERRORS, failing Task 7's full suite. Repoint that test to the new home: change its imports to `from osmose.forcing.grid import apply_land_mask, load_ocean_mask` (drop the leading underscores) and keep its real-grid assertion `load_ocean_mask(data/baltic/baltic_grid.nc)` returns shape `(40, 50)` with 616 ocean cells, so the real-grid regression coverage survives the extraction. Since the test now uses pure-core symbols, replace the `importorskip("mcp_servers.copernicus.server")` with a plain import of `osmose.forcing.grid` (no skip needed — it runs in the clean venv too).

- [ ] **Step 1: Write the failing parity test**

```python
# tests/test_forcing_mcp_parity.py
import importlib.util

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from osmose.forcing import bgc_to_ltl, load_ocean_mask
from osmose.maps.builder import GridSpec

# The MCP wrapper module imports fastmcp/copernicusmarine, absent in the clean venv.
_HAS_MCP = importlib.util.find_spec("fastmcp") is not None and (
    importlib.util.find_spec("copernicusmarine") is not None
)
BALTIC = GridSpec(nlon=50, nlat=40, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30)
_BALTIC_GRID_NC = Path("data/baltic/baltic_grid.nc")


def _bgc(tmp_path):
    lat = np.linspace(66, 54, 6)
    lon = np.linspace(10, 30, 7)
    ds = xr.Dataset(
        {"phyc": (["time", "latitude", "longitude"], np.abs(np.random.default_rng(0).random((12, 6, 7))) * 10),
         "zooc": (["time", "latitude", "longitude"], np.abs(np.random.default_rng(1).random((12, 6, 7))) * 5),
         "nppv": (["time", "latitude", "longitude"], np.abs(np.random.default_rng(2).random((12, 6, 7))) * 100)},
        coords={"time": np.arange(12), "latitude": lat, "longitude": lon},
    )
    path = tmp_path / "bgc.nc"
    ds.to_netcdf(str(path))
    return path


@pytest.mark.skipif(not _HAS_MCP, reason="MCP deps (fastmcp/copernicusmarine) not installed")
def test_mcp_wrapper_matches_core(tmp_path):
    import mcp_servers.copernicus.server as srv

    src = _bgc(tmp_path)
    out_file = tmp_path / "mcp_ltl.nc"
    srv.generate_osmose_ltl(source_bgc_file=str(src), output_file=str(out_file))
    mcp_ds = xr.open_dataset(out_file)

    # Reuse the wrapper's OWN Baltic grid so the two can never silently disagree.
    # (BALTIC above must equal it; cell_volume_m3's mid-lat = (66+54)/2 = 60.0, which
    # is exactly the old hardcoded cos(radians(60)) — so this Baltic-only parity check
    # cannot detect the cos-factor GENERALIZATION, only Baltic-grid value drift.)
    assert srv._baltic_grid() == BALTIC
    # The MCP wrapper applies the Baltic ocean mask (land -> NaN). Pass the SAME
    # mask to the core call so both sides NaN identical land cells; otherwise the
    # core leaves land cells as real biomass and nan_to_num(0) != real value fails.
    mask = load_ocean_mask(_BALTIC_GRID_NC)
    with xr.open_dataset(src) as core_src:
        core_ds = bgc_to_ltl(core_src, srv._baltic_grid(), ocean_mask=mask)
    for g in ["Diatoms", "Dinoflagellates", "Microzooplankton", "Mesozooplankton", "Macrozooplankton", "Benthos"]:
        assert np.allclose(np.nan_to_num(mcp_ds[g].values), np.nan_to_num(core_ds[g].values), rtol=1e-6)
    mcp_ds.close()
```

- [ ] **Step 2: Run test to verify it fails (or skips if MCP absent)**

Run: `.venv/bin/python -m pytest tests/test_forcing_mcp_parity.py -v`
Expected: SKIP in the clean venv (MCP deps absent), OR FAIL in the dev venv until the refactor lands. If it ERRORs on import of the server (old helpers vs new), that's the pre-refactor state.

- [ ] **Step 3: Refactor the server**

Replace the body of `generate_osmose_ltl` (keep the `@mcp.tool()` signature) with a delegating implementation:

```python
@mcp.tool()
def generate_osmose_ltl(
    source_bgc_file: Annotated[str, "Path to downloaded BGC NetCDF file"],
    output_file: Annotated[str, "Output path for OSMOSE-compatible LTL NetCDF"] = "",
    year: Annotated[int, "Year to extract (0 = use all available)"] = 0,
    depth_integrate_m: Annotated[float, "Depth range to integrate over (meters)"] = 50.0,
    chl_to_biomass_factor: Annotated[float, "C:Chl ratio for fallback mode"] = 50.0,
) -> str:
    """Convert CMEMS biogeochemistry data into OSMOSE 6-group LTL forcing (Baltic grid)."""
    from osmose.forcing import LtlParams, bgc_to_ltl, load_ocean_mask, write_ltl

    src = Path(source_bgc_file)
    if not src.exists():
        return f"Error: Source file not found: {source_bgc_file}"
    grid = _baltic_grid()
    grid_nc = Path(__file__).resolve().parents[2] / "data" / "baltic" / "baltic_grid.nc"
    mask = load_ocean_mask(grid_nc)
    ds = xr.open_dataset(src)
    try:
        result = bgc_to_ltl(
            ds, grid, year=year, depth_integrate_m=depth_integrate_m,
            params=LtlParams(chl_to_biomass_factor=chl_to_biomass_factor), ocean_mask=mask,
        )
    except ValueError as exc:
        ds.close()
        return f"Error: {exc}"
    result.attrs["source"] = str(source_bgc_file)
    if not output_file:
        output_file = str(Path(DEFAULT_OUTPUT_DIR) / "baltic_ltl_biomass_cmems.nc")
    # overwrite=True preserves the MCP's pre-existing always-regenerate behavior.
    path = write_ltl(result, output_file, overwrite=True)
    lines = [f"Generated OSMOSE LTL forcing: {path}", f"Mode: {result.attrs['mode']}",
             f"Grid: {grid.nlat} x {grid.nlon}, 24 biweekly steps"]
    for g in result.data_vars:
        lines.append(f"  {g}: total={float(result[g].sum(skipna=True)):.0f} t, "
                     f"max/cell={float(result[g].max(skipna=True)):.1f} t")
    ds.close()
    result.close()
    return "\n".join(lines)
```

Replace `generate_osmose_physics` with the delegating body below — spelled out (not "similarly") so the original per-variable `range X - Y units` summary is preserved (server.py:623). `write_physics` now returns a `{name: path}` dict:

```python
@mcp.tool()
def generate_osmose_physics(
    source_phy_file: Annotated[str, "Path to downloaded PHY NetCDF file (with thetao, so)"],
    output_dir: Annotated[str, "Output directory for OSMOSE physics NetCDF files"] = "",
    year: Annotated[int, "Year to extract (0 = use all available)"] = 0,
    depth_surface_m: Annotated[float, "Depth for surface fields (meters)"] = 10.0,
) -> str:
    """Convert downloaded CMEMS physics into OSMOSE temperature/salinity forcing (Baltic grid)."""
    from osmose.forcing import phy_to_physics, write_physics

    src = Path(source_phy_file)
    if not src.exists():
        return f"Error: Source file not found: {source_phy_file}"
    grid = _baltic_grid()
    ds = xr.open_dataset(src)
    try:
        dsets = phy_to_physics(ds, grid, year=year, depth_surface_m=depth_surface_m)
    finally:
        ds.close()
    if not dsets:
        return "Error: no physics variables (thetao/so) found in source"
    if not output_dir:
        output_dir = DEFAULT_OUTPUT_DIR
    paths = write_physics(dsets, output_dir, overwrite=True)  # name -> Path
    results = []
    for name, fds in dsets.items():
        arr = fds[name].values
        units = fds.attrs.get("units", "")
        results.append(
            f"  {name}: {paths[name]} (range {np.nanmin(arr):.2f} - {np.nanmax(arr):.2f} {units})"
        )
    return "Generated OSMOSE physics forcing:\n" + "\n".join(results)
```

Add a Baltic-grid helper near the top (after `BALTIC_BBOX`):

```python
def _baltic_grid():
    from osmose.maps.builder import GridSpec

    return GridSpec(nlon=50, nlat=40, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30)
```

Delete the now-unused helpers `_make_target_coords`, `_regrid`, `_resample_to_24`, `_cell_volume_m3`, `_get_coords`, `_get_var`, `_load_baltic_ocean_mask`, `_apply_land_mask`, and the `OSMOSE_GRID` constant. Keep `download_field`, `list_datasets`, `check_credentials`, `DATASETS`, `BALTIC_BBOX`, `_require_creds`, `_login`.

- [ ] **Step 4: Verify**

Run (dev venv, where MCP deps exist): `.venv/bin/python -m pytest tests/test_forcing_mcp_parity.py -v`
Expected: PASS (wrapper output matches the core).
Also: `.venv/bin/python -c "import ast; ast.parse(open('mcp_servers/copernicus/server.py').read())"` to confirm the file still parses, and grep that the deleted helpers are gone:
`grep -nE "OSMOSE_GRID|_make_target_coords|_apply_land_mask" mcp_servers/copernicus/server.py` → no matches.

- [ ] **Step 5: Commit**

```bash
git add mcp_servers/copernicus/server.py tests/test_forcing_mcp_parity.py tests/test_copernicus_ltl_mask.py
git commit -m "refactor(mcp): delegate generate_osmose_* to osmose.forcing core"
```

---

## Task 7: Full suite + lint/format/pyright

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `.venv/bin/python -m pytest -q -n auto`
Expected: all pass (the 6 new forcing test files + the existing suite, including the **repointed** `tests/test_copernicus_ltl_mask.py` from Task 6). The MCP parity test passes in the dev venv (MCP deps present). Note: do NOT add `scripts/__init__.py` — `from scripts.convert_cmems_forcing import _run` already resolves via pytest's rootdir (same pattern as the existing `from scripts.calibrate_baltic import ...`); adding one would diverge from repo convention.

- [ ] **Step 2: Lint + format (matches CI "lint" job — BOTH)**

CI lints only `osmose/ ui/ tests/` (`.github/workflows/ci.yml`), NOT all of `scripts/` — and `scripts/` has ~31 pre-existing ruff errors + 16 unformatted legacy files. So lint CI's targets PLUS only the one new script file; do NOT run ruff over all of `scripts/` (it would fail on unrelated legacy and reformat stray files).

Run: `.venv/bin/ruff check osmose/ ui/ tests/ scripts/convert_cmems_forcing.py && .venv/bin/ruff format --check osmose/ ui/ tests/ scripts/convert_cmems_forcing.py`
Expected: clean. If format fails, run `.venv/bin/ruff format` on the SAME targets and re-commit.

- [ ] **Step 3: Pyright on the new code**

Run: `.venv/bin/python -m pyright --pythonpath .venv/bin/python osmose/forcing/ scripts/convert_cmems_forcing.py`
Expected: 0 errors. (Per the CI-pyright-reproduction gotcha, don't run bare `pyright`.)

- [ ] **Step 4: Clean-venv guard (the load-bearing CI invariant)**

Confirm the core imports with NO MCP deps: `osmose/forcing/` and its tests (except the find_spec-guarded parity test) must run without `fastmcp`/`copernicusmarine`. Grep proves the dependency direction:
Run: `grep -rnE "fastmcp|copernicusmarine|dotenv" osmose/forcing/`
Expected: no matches.

- [ ] **Step 5: Document the new package in the architecture tree**

Add one line to the `CLAUDE.md` architecture tree (the `osmose/` block, alongside `config/`, `calibration/`), matching the existing prose-tree style:

```
  forcing/       # Pure CMEMS->OSMOSE forcing conversion (grid-general, LTL + physics)
```

- [ ] **Step 6: Commit any fixups**

```bash
git add -A
git commit -m "chore(forcing): lint/format/pyright fixups + CLAUDE.md tree" || echo "nothing to commit"
```

---

## Deep-review hardening notes (doc-only, applied)

These came out of the multi-angle workflow review; they are documentation/scoping clarifications, not code behavior:

- **Physics output targets the Java engine, NOT the default Python engine.** The Python engine reads only the scalar `temperature.value` (under bioen) and has no salinity input (`PhysicalData.from_netcdf` is currently unwired; no salinity consumer). `phy_to_physics` faithfully emits the gridded temperature/salinity NetCDFs the MCP already produced, but sub-project C must not assume wiring `temperature.varname`/salinity into a *Python-engine* config does anything — that needs an engine-side loader first. Sub-project D's physics view should be export/preview-only until then.
- **LTL group-name contract:** the OSMOSE engine resolves resource forcing by *species name* (`species.name.spN`), so consumers (sub-project C) must write the six `GROUP_NAMES` (exported from `osmose.forcing`) into the resource species-name keys. This is the load-bearing "consumed unchanged" contract for LTL (already true for the bundled Baltic config, sp8–sp13).
- **CLI is dev-tree-only:** `scripts/convert_cmems_forcing.py` is not packaged in the wheel (`packages.find include=["osmose*"]`) and has no `[project.scripts]` entry point — runnable via `python scripts/convert_cmems_forcing.py` from a source checkout (same as `scripts/calibrate_baltic.py`). If a shipped console command is later wanted, move the module under `osmose/` and add an entry point.
- **Temporal resolution:** sub-project A produces 24 biweekly steps only (faithful to source); the CLI warns when `simulation.time.ndtPerYear != 24`. A non-24 target would need a future keyword-only `n_out_steps` arg (backward-compatible to add later — does not force rework now).

## Self-Review notes (applied)

- **Spec coverage:** grid helpers (Task 1), `bgc_to_ltl`+`LtlParams` (Task 2), `phy_to_physics` (Task 3), io+`__init__` (Task 4), convert-only CLI (Task 5), MCP delegation + parity (Task 6), gates incl. clean-venv guard (Task 7). All spec components covered.
- **Type consistency:** `bgc_to_ltl(ds, grid, *, year, depth_integrate_m, params, ocean_mask)`, `phy_to_physics(ds, grid, *, year, depth_surface_m)`, `write_ltl(ds, path)`, `write_physics(dsets, out_dir, prefix)`, `GROUP_NAMES`/`LtlParams` — identical across the module defs, `__init__` exports, the CLI, the MCP wrapper, and the tests.
- **Purity invariant:** `osmose/forcing/` imports only numpy/xarray + `osmose.maps.builder`/`osmose.config.reader`/`osmose.logging` (all core). The MCP module imports FROM the core. Task 7 Step 4 enforces it.
- **Faithfulness:** Mode-A/B math, coefficients, and output structure (one var per group, `(24, lat, lon)`, `mode` attr) are ported verbatim; the parity test guards against drift.
- **Out of scope (per spec):** live download (B), config scaffolding/repointing (C), Shiny page (D), dataset catalog.
- **Open detail flagged for execution:** the exact `OsmoseConfigReader` read call in Task 5 — adapt `_load_config` to return a flat `dict[str, str]`; the dict-passthrough keeps tests decoupled from that API.
