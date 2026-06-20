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
from dotenv import load_dotenv
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Load .env from the project root (osmose-python/) so the MCP server can be
# started without the parent shell having the vars exported.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

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

def _baltic_grid():
    from osmose.maps.builder import GridSpec

    return GridSpec(nlon=50, nlat=40, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30)


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
# Tool 3: Generate OSMOSE LTL forcing NetCDF
# ---------------------------------------------------------------------------
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
