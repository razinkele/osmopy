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
        0.40,
        0.60,
        0.75,
        0.80,
        0.70,
        0.40,
        0.25,
        0.20,
        0.25,
        0.35,
        0.40,
        0.40,
    )
    micro_frac: float = 0.40  # Baltic zoo size split: ~40% micro
    meso_frac: float = 0.45  # ~45% meso (copepods)
    macro_frac: float = 0.15  # ~15% macro (mysids, krill)
    # Mode B (chl-derived)
    chl_to_biomass_factor: float = 50.0
    diatom_frac_b: tuple[float, ...] = (
        0.3,
        0.5,
        0.7,
        0.8,
        0.7,
        0.5,
        0.3,
        0.2,
        0.2,
        0.3,
        0.3,
        0.3,
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
        assert phyc is not None and zooc is not None  # guaranteed by has_phyc/has_zooc
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
