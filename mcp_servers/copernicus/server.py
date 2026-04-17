#!/usr/bin/env python3
"""Copernicus Marine MCP Server for OSMOSE Baltic forcing data.

Provides tools to browse, download, and convert Copernicus Marine Service
(CMEMS) data into OSMOSE-compatible NetCDF forcing files for the Baltic Sea.

Key CMEMS datasets used:
  - cmems_mod_bal_phy_my_P1M-m  (temperature, salinity — multiyear monthly)
  - cmems_mod_bal_bgc_my_P1M-m  (chlorophyll, NPP, O2, nutrients — multiyear monthly)
  - cmems_mod_bal_phy_anfc_P1M-m  (physics analysis/forecast monthly)
  - cmems_mod_bal_bgc_anfc_P1M-m  (biogeochemistry analysis/forecast monthly)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import copernicusmarine as cm
import numpy as np
import xarray as xr
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CMEMS_USER: str | None = os.environ.get("CMEMS_USERNAME")
CMEMS_PASS: str | None = os.environ.get("CMEMS_PASSWORD")


def _require_creds() -> tuple[str, str]:
    """Return non-null (username, password) or raise with operator guidance."""
    if not CMEMS_USER or not CMEMS_PASS:
        raise RuntimeError(
            "CMEMS_USERNAME and CMEMS_PASSWORD environment variables must be set. "
            "See mcp_servers/copernicus/README.md."
        )
    return CMEMS_USER, CMEMS_PASS


# Baltic Sea bounding box (matches OSMOSE grid: 10-30E, 54-66N)
BALTIC_BBOX = {
    "minimum_longitude": 9.5,
    "maximum_longitude": 30.5,
    "minimum_latitude": 53.5,
    "maximum_latitude": 66.5,
}

# Known Baltic datasets
DATASETS = {
    "phy_monthly_reanalysis": {
        "dataset_id": "cmems_mod_bal_phy_my_P1M-m",
        "product_id": "BALTICSEA_MULTIYEAR_PHY_003_011",
        "variables": {
            "thetao": "Sea water temperature (degC)",
            "so": "Sea water salinity (PSU)",
            "bottomT": "Bottom temperature (degC)",
            "mlotst": "Mixed layer thickness (m)",
            "uo": "Eastward current velocity (m/s)",
            "vo": "Northward current velocity (m/s)",
        },
        "description": "Baltic Sea Physics Multiyear Reanalysis — monthly means",
    },
    "bgc_monthly_reanalysis": {
        "dataset_id": "cmems_mod_bal_bgc_my_P1M-m",
        "product_id": "BALTICSEA_MULTIYEAR_BGC_003_012",
        "variables": {
            "chl": "Chlorophyll-a concentration (mg/m3)",
            "nppv": "Net primary production (mgC/m3/day)",
            "o2": "Dissolved oxygen (mmol/m3)",
            "no3": "Nitrate concentration (mmol/m3)",
            "po4": "Phosphate concentration (mmol/m3)",
            "nh4": "Ammonium concentration (mmol/m3)",
            "ph": "Sea water pH",
            "zsd": "Secchi depth (m)",
        },
        "description": "Baltic Sea Biogeochemistry Multiyear Reanalysis — monthly means",
    },
    "phy_monthly_forecast": {
        "dataset_id": "cmems_mod_bal_phy_anfc_P1M-m",
        "product_id": "BALTICSEA_ANALYSISFORECAST_PHY_003_006",
        "variables": {
            "thetao": "Sea water temperature (degC)",
            "so": "Sea water salinity (PSU)",
            "bottomT": "Bottom temperature (degC)",
            "mlotst": "Mixed layer thickness (m)",
            "uo": "Eastward current velocity (m/s)",
            "vo": "Northward current velocity (m/s)",
        },
        "description": "Baltic Sea Physics Analysis/Forecast — monthly means",
    },
    "bgc_monthly_forecast": {
        "dataset_id": "cmems_mod_bal_bgc_anfc_P1M-m",
        "product_id": "BALTICSEA_ANALYSISFORECAST_BGC_003_007",
        "variables": {
            "chl": "Chlorophyll-a concentration (mg/m3)",
            "phyc": "Phytoplankton carbon biomass (mmolC/m3) — DIRECT biomass, preferred for LTL",
            "zooc": "Zooplankton carbon biomass (mmolC/m3) — DIRECT biomass, preferred for LTL",
            "nppv": "Net primary production (mgC/m3/day)",
            "o2": "Dissolved oxygen (mmol/m3)",
            "o2b": "Bottom dissolved oxygen (mmol/m3)",
            "h2s": "Hydrogen sulfide concentration (mmol/m3) — anoxia indicator",
            "no3": "Nitrate concentration (mmol/m3)",
            "po4": "Phosphate concentration (mmol/m3)",
            "si": "Silicate concentration (mmol/m3) — diatom indicator",
            "nh4": "Ammonium concentration (mmol/m3)",
            "pH": "Sea water pH",
            "dissic": "Dissolved inorganic carbon (mmol/m3)",
            "kd": "Light attenuation coefficient (1/m)",
        },
        "description": (
            "Baltic Sea Biogeochemistry Analysis/Forecast — monthly means. "
            "PREFERRED for OSMOSE LTL: has phyc (phytoplankton C) and zooc (zooplankton C) "
            "as direct biomass variables, plus silicate for diatom identification."
        ),
    },
    "bgc_daily_forecast": {
        "dataset_id": "cmems_mod_bal_bgc_anfc_P1D-m",
        "product_id": "BALTICSEA_ANALYSISFORECAST_BGC_003_007",
        "variables": {
            "chl": "Chlorophyll-a concentration (mg/m3)",
            "phyc": "Phytoplankton carbon biomass (mmolC/m3)",
            "zooc": "Zooplankton carbon biomass (mmolC/m3)",
            "nppv": "Net primary production (mgC/m3/day)",
            "o2": "Dissolved oxygen (mmol/m3)",
            "o2b": "Bottom dissolved oxygen (mmol/m3)",
            "h2s": "Hydrogen sulfide (mmol/m3)",
            "no3": "Nitrate (mmol/m3)",
            "po4": "Phosphate (mmol/m3)",
            "si": "Silicate (mmol/m3)",
            "nh4": "Ammonium (mmol/m3)",
            "pH": "Sea water pH",
            "dissic": "Dissolved inorganic carbon (mmol/m3)",
            "kd": "Light attenuation (1/m)",
        },
        "description": "Baltic Sea BGC Analysis/Forecast — daily means. Higher temporal resolution.",
    },
}

# OSMOSE grid parameters (must match baltic_param-grid.csv)
OSMOSE_GRID = {
    "nlon": 50,
    "nlat": 40,
    "lon_min": 10.0,
    "lon_max": 30.0,
    "lat_min": 54.0,
    "lat_max": 66.0,
}

# Default output directory
DEFAULT_OUTPUT_DIR = str(Path.home() / "osmose" / "osmose-python" / "data" / "cmems_cache" / "cmems_downloads")

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Copernicus Marine Baltic",
    instructions=(
        "Access Copernicus Marine Service data for Baltic Sea ecosystem modeling. "
        "Download monthly temperature, salinity, and biogeochemistry fields, "
        "and convert them to OSMOSE-compatible NetCDF forcing files."
    ),
)


def _login() -> None:
    """Ensure CMEMS credentials are configured."""
    user, password = _require_creds()
    cm.login(username=user, password=password, force_overwrite=True)


# ---------------------------------------------------------------------------
# Tool 1: List available datasets
# ---------------------------------------------------------------------------
@mcp.tool()
def list_datasets() -> str:
    """List available Copernicus Marine Baltic Sea datasets and their variables.

    Returns a structured summary of physics and biogeochemistry datasets
    with all downloadable variables.
    """
    lines = ["# Available Baltic Sea CMEMS Datasets\n"]
    for key, info in DATASETS.items():
        lines.append(f"## {key}")
        lines.append(f"**Dataset ID:** `{info['dataset_id']}`")
        lines.append(f"**Product:** `{info['product_id']}`")
        lines.append(f"**Description:** {info['description']}")
        lines.append("**Variables:**")
        for var, desc in info["variables"].items():
            lines.append(f"  - `{var}`: {desc}")
        lines.append("")
    lines.append("## Bounding box (Baltic OSMOSE grid)")
    lines.append(f"  Lon: {BALTIC_BBOX['minimum_longitude']}–{BALTIC_BBOX['maximum_longitude']}E")
    lines.append(f"  Lat: {BALTIC_BBOX['minimum_latitude']}–{BALTIC_BBOX['maximum_latitude']}N")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2: Download a field
# ---------------------------------------------------------------------------
@mcp.tool()
def download_field(
    dataset: Annotated[str, "Dataset key: phy_monthly_reanalysis, bgc_monthly_reanalysis, phy_monthly_forecast, bgc_monthly_forecast"],
    variables: Annotated[list[str], "Variable short names to download, e.g. ['thetao', 'so']"],
    start_date: Annotated[str, "Start date YYYY-MM-DD or YYYY-MM"],
    end_date: Annotated[str, "End date YYYY-MM-DD or YYYY-MM"],
    depth_min: Annotated[float, "Minimum depth in meters (0 = surface)"] = 0.0,
    depth_max: Annotated[float, "Maximum depth in meters (e.g. 200 for full water column)"] = 200.0,
    output_dir: Annotated[str, "Output directory path"] = DEFAULT_OUTPUT_DIR,
) -> str:
    """Download Copernicus Marine data for the Baltic Sea OSMOSE domain.

    Downloads selected variables from the specified dataset, subsetted to the
    Baltic OSMOSE grid bounding box and requested time/depth range.
    Returns the path to the downloaded NetCDF file.
    """
    if dataset not in DATASETS:
        return f"Error: Unknown dataset '{dataset}'. Use list_datasets() to see options."

    ds_info = DATASETS[dataset]
    invalid = [v for v in variables if v not in ds_info["variables"]]
    if invalid:
        available = ", ".join(ds_info["variables"].keys())
        return f"Error: Unknown variables {invalid}. Available: {available}"

    _login()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    var_str = "_".join(variables)
    filename = f"baltic_{dataset}_{var_str}_{start_date}_{end_date}.nc"

    try:
        _result = cm.subset(
            dataset_id=ds_info["dataset_id"],
            variables=variables,
            start_datetime=f"{start_date}T00:00:00",
            end_datetime=f"{end_date}T23:59:59",
            minimum_longitude=BALTIC_BBOX["minimum_longitude"],
            maximum_longitude=BALTIC_BBOX["maximum_longitude"],
            minimum_latitude=BALTIC_BBOX["minimum_latitude"],
            maximum_latitude=BALTIC_BBOX["maximum_latitude"],
            minimum_depth=depth_min,
            maximum_depth=depth_max,
            output_directory=str(out_path),
            output_filename=filename,
            overwrite_output_data=True,
            disable_progress_bar=False,
        )

        fpath = out_path / filename
        if fpath.exists():
            ds = xr.open_dataset(fpath)
            summary = []
            summary.append(f"Downloaded: {fpath}")
            summary.append(f"Variables: {list(ds.data_vars)}")
            summary.append(f"Time steps: {len(ds.time) if 'time' in ds.dims else 'N/A'}")
            summary.append(f"Lat range: {float(ds.latitude.min()):.2f} - {float(ds.latitude.max()):.2f}")
            summary.append(f"Lon range: {float(ds.longitude.min()):.2f} - {float(ds.longitude.max()):.2f}")
            if "depth" in ds.dims:
                summary.append(f"Depth range: {float(ds.depth.min()):.1f} - {float(ds.depth.max()):.1f} m")
            for var in ds.data_vars:
                d = ds[var]
                summary.append(f"  {var}: shape={d.shape}, range=[{float(d.min()):.4f}, {float(d.max()):.4f}]")
            ds.close()
            return "\n".join(summary)
        else:
            return f"Download completed but file not found at {fpath}. Check output_directory."

    except Exception as e:
        return f"Download failed: {e}"


# ---------------------------------------------------------------------------
# Shared regridding helpers
# ---------------------------------------------------------------------------
def _make_target_coords() -> tuple[np.ndarray, np.ndarray]:
    """Return (target_lat, target_lon) cell centers for the OSMOSE 50x40 grid."""
    g = OSMOSE_GRID
    nlat, nlon = g["nlat"], g["nlon"]
    lat = np.linspace(
        g["lat_max"] - (g["lat_max"] - g["lat_min"]) / nlat / 2,
        g["lat_min"] + (g["lat_max"] - g["lat_min"]) / nlat / 2,
        nlat,
    )
    lon = np.linspace(
        g["lon_min"] + (g["lon_max"] - g["lon_min"]) / nlon / 2,
        g["lon_max"] - (g["lon_max"] - g["lon_min"]) / nlon / 2,
        nlon,
    )
    return lat, lon


def _regrid(data_3d: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray) -> np.ndarray:
    """Nearest-neighbor regrid (time, src_lat, src_lon) -> (time, nlat, nlon)."""
    target_lat, target_lon = _make_target_coords()
    nlat, nlon = len(target_lat), len(target_lon)
    nt = data_3d.shape[0]
    result = np.zeros((nt, nlat, nlon), dtype=np.float64)
    for j in range(nlat):
        lat_idx = int(np.argmin(np.abs(src_lat - target_lat[j])))
        for i in range(nlon):
            lon_idx = int(np.argmin(np.abs(src_lon - target_lon[i])))
            result[:, j, i] = data_3d[:, lat_idx, lon_idx]
    return result


def _resample_to_24(data: np.ndarray) -> np.ndarray:
    """Interpolate (time, lat, lon) to 24 biweekly timesteps."""
    nt, nlat, nlon = data.shape
    if nt == 24:
        return data
    out = np.zeros((24, nlat, nlon), dtype=np.float64)
    for j in range(nlat):
        for i in range(nlon):
            out[:, j, i] = np.interp(
                np.linspace(0, 1, 24), np.linspace(0, 1, nt), data[:, j, i]
            )
    return out


def _cell_volume_m3(depth_m: float) -> float:
    """Approximate cell volume in m3 for the OSMOSE Baltic grid."""
    g = OSMOSE_GRID
    dlat = (g["lat_max"] - g["lat_min"]) / g["nlat"]
    dlon = (g["lon_max"] - g["lon_min"]) / g["nlon"]
    cos_lat = np.cos(np.radians(60.0))
    area = (dlat * 111320) * (dlon * 111320 * cos_lat)
    return area * depth_m


def _get_coords(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Extract lat/lon coordinate arrays from a dataset."""
    lat = ds.latitude.values if "latitude" in ds.coords else ds.lat.values
    lon = ds.longitude.values if "longitude" in ds.coords else ds.lon.values
    return lat, lon


def _get_var(ds: xr.Dataset, name: str) -> np.ndarray | None:
    """Get a variable as 3D array (time, lat, lon), NaN filled to 0."""
    if name not in ds:
        return None
    arr = np.nan_to_num(ds[name].values, nan=0.0)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    return arr


# ---------------------------------------------------------------------------
# Tool 3: Generate OSMOSE LTL forcing NetCDF
# ---------------------------------------------------------------------------
@mcp.tool()
def generate_osmose_ltl(
    source_bgc_file: Annotated[str, "Path to downloaded BGC NetCDF file"],
    output_file: Annotated[str, "Output path for OSMOSE-compatible LTL NetCDF"] = "",
    year: Annotated[int, "Year to extract (0 = use all available)"] = 0,
    depth_integrate_m: Annotated[float, "Depth range to integrate over (meters)"] = 50.0,
    chl_to_biomass_factor: Annotated[float, "C:Chl ratio for fallback mode (only used if phyc missing)"] = 50.0,
) -> str:
    """Convert CMEMS biogeochemistry data into OSMOSE 6-group LTL forcing.

    Two modes depending on source data:

    **Mode A (preferred): Direct biomass from forecast product**
    When `phyc` and `zooc` are present (bgc_monthly_forecast / bgc_daily_forecast),
    uses phytoplankton and zooplankton carbon biomass directly. Silicate (`si`) is
    used to identify diatom-dominated vs dinoflagellate-dominated regimes.

    **Mode B (fallback): Estimated from reanalysis**
    When only `chl` and `nppv` are available (bgc_monthly_reanalysis), estimates
    plankton biomass from chlorophyll using C:Chl ratio and seasonal community splits.

    Both modes produce 6 LTL groups on the OSMOSE 50x40 grid with 24 biweekly steps:
      Diatoms, Dinoflagellates, Microzooplankton, Mesozooplankton, Macrozooplankton, Benthos
    """
    src = Path(source_bgc_file)
    if not src.exists():
        return f"Error: Source file not found: {source_bgc_file}"

    g = OSMOSE_GRID
    nlat, nlon = g["nlat"], g["nlon"]
    target_lat, target_lon = _make_target_coords()
    cell_vol = _cell_volume_m3(depth_integrate_m)

    ds = xr.open_dataset(src)
    if year > 0 and "time" in ds.dims:
        ds = ds.sel(time=ds.time.dt.year == year)
    if "depth" in ds.dims:
        ds = ds.sel(depth=slice(0, depth_integrate_m)).mean(dim="depth", skipna=True)

    src_lat, src_lon = _get_coords(ds)
    has_phyc = "phyc" in ds
    has_zooc = "zooc" in ds
    mode = "A (direct biomass)" if (has_phyc and has_zooc) else "B (chl-derived)"

    if has_phyc and has_zooc:
        # ---- Mode A: Direct phyto/zoo carbon from forecast ----
        phyc = _get_var(ds, "phyc")   # mmolC/m3
        zooc = _get_var(ds, "zooc")   # mmolC/m3
        chl = _get_var(ds, "chl")     # mg/m3 (for diagnostics)
        nppv = _get_var(ds, "nppv")   # mgC/m3/day

        # Convert mmolC/m3 -> tonnes wet weight per cell
        # 1 mmolC = 0.012 gC
        # Phytoplankton: ERGOM phyc includes all autotrophic C; calibrated
        #   against Baltic standing stock ~2 Mt wet. Use C:wet = 1:1 (carbon-equiv).
        # Zooplankton: use C:wet = 1:10 (standard for crustacean zooplankton).
        PHYTO_C_TO_WET = 0.012 * 1.0   # gC/mmol * 1:1 wet:C
        ZOO_C_TO_WET = 0.012 * 10.0    # gC/mmol * 10:1 wet:C
        phyto_tonnes = _regrid(phyc, src_lat, src_lon) * PHYTO_C_TO_WET * cell_vol / 1e6
        zoo_tonnes = _regrid(zooc, src_lat, src_lon) * ZOO_C_TO_WET * cell_vol / 1e6

        # Split phytoplankton into diatoms vs dinoflagellates/cyanobacteria.
        # Baltic silicate is non-limiting (~60+ mmol/m3 year-round), so a
        # Michaelis-Menten approach doesn't work. Instead use a seasonal
        # community split based on Baltic phytoplankton succession:
        #   Spring (Feb-May): diatom-dominated bloom (70-80%)
        #   Summer (Jun-Sep): cyanobacteria + dinoflagellate bloom (70-80%)
        #   Autumn/Winter: mixed, low biomass (40-50% diatom)
        n_steps = phyto_tonnes.shape[0]
        if n_steps == 12:
            #               Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
            diatom_frac = [0.40, 0.60, 0.75, 0.80, 0.70, 0.40, 0.25, 0.20, 0.25, 0.35, 0.40, 0.40]
            diatom_frac = np.array(diatom_frac)[:, np.newaxis, np.newaxis] * np.ones((1, nlat, nlon))
        else:
            diatom_frac = np.ones((n_steps, nlat, nlon)) * 0.5

        diatoms = phyto_tonnes * diatom_frac
        dinoflagellates = phyto_tonnes * (1.0 - diatom_frac)

        # Split zooplankton into micro/meso/macro by size fraction
        # Baltic zooplankton: ~40% micro, ~45% meso (copepods), ~15% macro (mysids, krill)
        microzoo = zoo_tonnes * 0.40
        mesozoo = zoo_tonnes * 0.45
        macrozoo = zoo_tonnes * 0.15

        # Benthos: estimate from NPP bottom flux if available, else from zoo
        if nppv is not None:
            npp_grid = _regrid(nppv, src_lat, src_lon)
            npp_tonnes = npp_grid * cell_vol / 1e9 * 365
            benthos = npp_tonnes * 0.05 / 3.0
        else:
            benthos = zoo_tonnes * 0.3  # rough benthic fraction

    else:
        # ---- Mode B: Fallback from chl/nppv (reanalysis) ----
        chl = _get_var(ds, "chl")
        nppv = _get_var(ds, "nppv")
        if chl is None:
            ds.close()
            return (
                "Error: Neither phyc/zooc nor chl found. Download with "
                "variables=['phyc','zooc','chl','nppv','si'] from bgc_monthly_forecast, "
                "or ['chl','nppv'] from bgc_monthly_reanalysis."
            )
        if nppv is None:
            nppv = chl * 5.0

        chl_grid = _regrid(chl, src_lat, src_lon)
        nppv_grid = _regrid(nppv, src_lat, src_lon)

        phyto_tonnes = chl_grid * chl_to_biomass_factor * cell_vol / 1e9

        n_steps = chl_grid.shape[0]
        diatom_frac = np.ones(n_steps) * 0.5
        if n_steps == 12:
            diatom_frac = np.array([0.3, 0.5, 0.7, 0.8, 0.7, 0.5, 0.3, 0.2, 0.2, 0.3, 0.3, 0.3])
        diatoms = phyto_tonnes * diatom_frac[:, np.newaxis, np.newaxis]
        dinoflagellates = phyto_tonnes * (1.0 - diatom_frac[:, np.newaxis, np.newaxis])

        npp_tonnes = nppv_grid * cell_vol / 1e9 * 365
        microzoo = npp_tonnes * 0.30 / 50
        mesozoo = npp_tonnes * 0.10 / 15
        macrozoo = npp_tonnes * 0.03 / 8
        benthos = npp_tonnes * 0.05 / 3

    ds.close()

    # Resample all to 24 biweekly steps and ensure non-negative
    groups = {
        "Diatoms": _resample_to_24(diatoms),
        "Dinoflagellates": _resample_to_24(dinoflagellates),
        "Microzooplankton": _resample_to_24(microzoo),
        "Mesozooplankton": _resample_to_24(mesozoo),
        "Macrozooplankton": _resample_to_24(macrozoo),
        "Benthos": _resample_to_24(benthos),
    }
    for arr in groups.values():
        arr[arr < 0] = 0.0

    out_ds = xr.Dataset(
        {name: (["time", "latitude", "longitude"], data) for name, data in groups.items()},
        coords={"time": np.arange(24), "latitude": target_lat, "longitude": target_lon},
        attrs={
            "title": "Baltic Sea OSMOSE LTL Forcing (from CMEMS)",
            "source": str(source_bgc_file),
            "mode": mode,
            "description": "6 lower trophic level groups, 24 biweekly timesteps",
            "depth_integration_m": depth_integrate_m,
            "conventions": "Latitude descending (north to south) to match grid.nc",
        },
    )

    if not output_file:
        output_file = str(Path(DEFAULT_OUTPUT_DIR) / "baltic_ltl_biomass_cmems.nc")
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(str(out_path))

    lines = [f"Generated OSMOSE LTL forcing: {out_path}", f"Mode: {mode}"]
    lines.append(f"Grid: {nlat} x {nlon}, 24 biweekly steps")
    for name, data in groups.items():
        lines.append(f"  {name}: total={np.sum(data):.0f} t, max/cell={np.max(data):.1f} t")
    out_ds.close()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4: Generate OSMOSE temperature/salinity forcing
# ---------------------------------------------------------------------------
@mcp.tool()
def generate_osmose_physics(
    source_phy_file: Annotated[str, "Path to downloaded PHY NetCDF file (with thetao, so)"],
    output_dir: Annotated[str, "Output directory for OSMOSE physics NetCDF files"] = "",
    year: Annotated[int, "Year to extract (0 = use all available)"] = 0,
    depth_surface_m: Annotated[float, "Depth for surface fields (meters)"] = 10.0,
) -> str:
    """Convert downloaded CMEMS physics data into OSMOSE-compatible forcing.

    Creates temperature and salinity NetCDF files regridded to the OSMOSE
    50x40 Baltic grid with 24 biweekly timesteps.
    These files can be used with the OSMOSE bioenergetics module.
    """
    src = Path(source_phy_file)
    if not src.exists():
        return f"Error: Source file not found: {source_phy_file}"

    target_lat, target_lon = _make_target_coords()

    ds = xr.open_dataset(src)
    if year > 0 and "time" in ds.dims:
        ds = ds.sel(time=ds.time.dt.year == year)
    if "depth" in ds.dims:
        ds = ds.sel(depth=depth_surface_m, method="nearest")

    src_lat, src_lon = _get_coords(ds)
    results = []

    if not output_dir:
        output_dir = DEFAULT_OUTPUT_DIR
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for var_name, osmose_name, units in [
        ("thetao", "temperature", "degC"),
        ("so", "salinity", "PSU"),
    ]:
        data = _get_var(ds, var_name)
        if data is None:
            results.append(f"  {var_name}: not found in source file, skipped")
            continue

        regridded = _resample_to_24(_regrid(data, src_lat, src_lon))

        fpath = out_path / f"baltic_{osmose_name}.nc"
        out_ds = xr.Dataset(
            {osmose_name: (["time", "latitude", "longitude"], regridded)},
            coords={"time": np.arange(24), "latitude": target_lat, "longitude": target_lon},
            attrs={
                "title": f"Baltic Sea OSMOSE {osmose_name.title()} Forcing (from CMEMS)",
                "source": str(source_phy_file),
                "units": units,
                "depth_m": depth_surface_m,
                "conventions": "Latitude descending (north to south) to match grid.nc",
            },
        )
        out_ds.to_netcdf(str(fpath))
        out_ds.close()
        results.append(f"  {osmose_name}: {fpath} (range {regridded.min():.2f} - {regridded.max():.2f} {units})")

    ds.close()
    return "Generated OSMOSE physics forcing:\n" + "\n".join(results)


# ---------------------------------------------------------------------------
# Tool 5: Quick status check
# ---------------------------------------------------------------------------
@mcp.tool()
def check_credentials() -> str:
    """Test Copernicus Marine Service login credentials."""
    user, password = _require_creds()
    try:
        result = cm.login(
            username=user,
            password=password,
            check_credentials_valid=True,
        )
        if result:
            return "Login successful. Credentials are valid."
        else:
            return (
                "Login returned False — credentials may be invalid. "
                "Check username/password or visit https://data.marine.copernicus.eu "
                "to verify your account."
            )
    except Exception as e:
        return f"Login failed: {e}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
