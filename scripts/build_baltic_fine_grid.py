# scripts/build_baltic_fine_grid.py
from __future__ import annotations
import io
from pathlib import Path

import numpy as np
import rasterio
import requests
import xarray as xr

from osmose.forcing.bathymetry import shallow_fraction
from osmose.forcing.grid import target_coords
from osmose.maps.builder import GridSpec

WCS = "https://ows.emodnet-bathymetry.eu/wcs"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "baltic-fine"
FINE = GridSpec(
    nlon=200, nlat=160, upleft_lat=66.0, upleft_lon=10.0, lowright_lat=54.0, lowright_lon=30.0
)


def fetch_emodnet(lat0, lat1, lon0, lon1):
    """Return (elev, lat_hi desc, lon_hi asc) for a bbox (lat0<lat1, lon0<lon1)."""
    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageId": "emodnet__mean",
        "subset": [f"Lat({lat0},{lat1})", f"Long({lon0},{lon1})"],
        "format": "image/tiff",
    }
    r = requests.get(WCS, params=params, timeout=180)
    r.raise_for_status()
    with rasterio.open(io.BytesIO(r.content)) as ds:
        elev = ds.read(1).astype(np.float64)
        b = ds.bounds
        return elev, np.linspace(b.top, b.bottom, ds.height), np.linspace(b.left, b.right, ds.width)


def build_shallow_fraction(depth_max_m: float):
    """Per-4x-cell shallow-fraction + ocean over the whole extent. Tiled in
    row-disjoint lat strips x lon blocks; per-tile cells are disjoint, so ratios
    are exact per tile — assign directly, no cross-tile accumulation."""
    frac = np.zeros((FINE.nlat, FINE.nlon))
    ocean = np.zeros((FINE.nlat, FINE.nlon), bool)
    lat_edges = np.linspace(FINE.upleft_lat, FINE.lowright_lat, FINE.nlat + 1)  # desc
    lon_edges = np.linspace(FINE.upleft_lon, FINE.lowright_lon, FINE.nlon + 1)  # asc
    RSTRIP, CSTRIP = 8, 50  # 4x-cells per fetch (keep GeoTIFF < ~40M px)
    for r0 in range(0, FINE.nlat, RSTRIP):
        r1 = min(r0 + RSTRIP, FINE.nlat)
        for c0 in range(0, FINE.nlon, CSTRIP):
            c1 = min(c0 + CSTRIP, FINE.nlon)
            elev, lat_hi, lon_hi = fetch_emodnet(
                lat_edges[r1], lat_edges[r0], lon_edges[c0], lon_edges[c1]
            )
            f, oc = shallow_fraction(elev, lat_hi, lon_hi, FINE, depth_max_m)
            frac[r0:r1, c0:c1] = f[r0:r1, c0:c1]
            ocean[r0:r1, c0:c1] = oc[r0:r1, c0:c1]
    return frac, ocean


def _cell_of(lat, lon):
    r = int(
        np.clip(
            np.searchsorted(
                -np.linspace(FINE.upleft_lat, FINE.lowright_lat, FINE.nlat + 1), -lat, "right"
            )
            - 1,
            0,
            FINE.nlat - 1,
        )
    )
    c = int(
        np.clip(
            np.searchsorted(
                np.linspace(FINE.upleft_lon, FINE.lowright_lon, FINE.nlon + 1), lon, "right"
            )
            - 1,
            0,
            FINE.nlon - 1,
        )
    )
    return r, c


def main() -> int:
    _, ocean = build_shallow_fraction(depth_max_m=1e9)
    r, c = _cell_of(55.3, 21.1)  # Curonian Lagoon — must be ocean (orientation sanity, spec §8)
    assert ocean[r, c], "orientation check failed: Curonian Lagoon not ocean"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "grid").mkdir(exist_ok=True)
    mask = np.where(ocean, 0, -99).astype(int)
    np.savetxt(OUT / "grid" / "baltic_fine_mask.csv", np.flipud(mask), fmt="%d", delimiter=";")
    tlat, tlon = target_coords(FINE)
    xr.Dataset(
        {"mask": (["latitude", "longitude"], ocean.astype("int8"))},
        coords={"latitude": tlat, "longitude": tlon},
    ).to_netcdf(OUT / "baltic_fine_grid.nc")
    print(f"fine grid: ocean cells = {int(ocean.sum())} of {ocean.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
