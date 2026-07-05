from pathlib import Path
import numpy as np
import xarray as xr
from osmose.forcing.conserve_regrid import split_conserve
from osmose.forcing.grid import target_coords, regrid, resample_to_24, load_ocean_mask
from scripts.build_baltic_fine_grid import FINE, OUT

SRC = Path("data/baltic")
ABS_BIOMASS = ["baltic_ltl_biomass.nc", "baltic_predator_biomass.nc"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tlat, tlon = target_coords(FINE)
    for fname in ABS_BIOMASS:
        ds = xr.open_dataset(SRC / fname)
        out = {}
        nt = 24
        for name, da in ds.data_vars.items():
            coarse = np.asarray(da.values)
            fine = split_conserve(coarse, 4)
            nt = fine.shape[0]
            assert np.isclose(fine.sum(), coarse.sum()), f"{fname}:{name} biomass not conserved"
            assert fine.shape[-2:] == (160, 200), f"{fname}:{name} shape {fine.shape}"
            out[name] = (["time", "latitude", "longitude"], fine)
        xr.Dataset(
            out, coords={"time": np.arange(nt), "latitude": tlat, "longitude": tlon}
        ).to_netcdf(OUT / fname)
        print(f"conserved-regrid {fname}")
    # salinity: import accumulate_climatology from the salinity builder, regrid to FINE, gap-fill, guard
    from scripts.build_baltic_salinity_forcing import accumulate_climatology, fill_ocean_nan

    so_files = sorted(
        (Path("data") / "cmems_cache" / "cmems_downloads").glob(
            "baltic_phy_monthly_reanalysis_so_*.nc"
        )
    )
    clim, src_lat, src_lon = accumulate_climatology(
        [str(p) for p in so_files]
    )  # (12, src_lat, src_lon)
    field24 = resample_to_24(regrid(clim, src_lat, src_lon, FINE))  # (24,160,200), NaN in gaps
    mask = load_ocean_mask(OUT / "baltic_fine_grid.nc")
    assert mask is not None, "fine ocean mask missing — run build_baltic_fine_grid first"
    field24 = fill_ocean_nan(field24, mask)
    assert not bool(np.isnan(field24[np.broadcast_to(mask, field24.shape)]).any()), (
        "ocean NaN after fill"
    )
    xr.Dataset(
        {"salinity": (["time", "latitude", "longitude"], field24)},
        coords={"time": np.arange(24), "latitude": tlat, "longitude": tlon},
    ).to_netcdf(OUT / "baltic_salinity_bottom_climatology.nc")
    print("salinity climatology (gap-filled) written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
