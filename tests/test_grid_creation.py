"""Tests for grid creation utilities."""

import numpy as np
import xarray as xr

from osmose.grid import create_grid_csv, create_grid_netcdf, csv_maps_to_netcdf


def test_create_grid_csv(tmp_path):
    output = tmp_path / "grid.csv"
    create_grid_csv(nrows=5, ncols=10, output=output)
    assert output.exists()
    import pandas as pd

    df = pd.read_csv(output, header=None)
    assert df.shape == (5, 10)
    # Default: all ocean (1)
    assert df.values.sum() == 50


def test_create_grid_csv_with_mask(tmp_path):
    output = tmp_path / "grid.csv"
    mask = np.ones((5, 10))
    mask[0, :] = -1  # First row is land
    create_grid_csv(nrows=5, ncols=10, output=output, mask=mask)
    import pandas as pd

    df = pd.read_csv(output, header=None)
    assert df.iloc[0].sum() == -10  # land cells


def test_create_grid_netcdf(tmp_path):
    output = tmp_path / "grid.nc"
    create_grid_netcdf(
        lat_bounds=(43.0, 48.0),
        lon_bounds=(-6.0, -1.0),
        nlat=10,
        nlon=20,
        output=output,
    )
    assert output.exists()
    ds = xr.open_dataset(output)
    assert "mask" in ds.data_vars
    assert ds.sizes["lat"] == 10
    assert ds.sizes["lon"] == 20
    ds.close()


def test_csv_maps_to_netcdf(tmp_path):
    # Create a CSV map
    csv_dir = tmp_path / "maps_csv"
    csv_dir.mkdir()
    np.savetxt(csv_dir / "map_Anchovy_summer.csv", np.random.rand(5, 10), delimiter=",")

    output = tmp_path / "maps.nc"
    csv_maps_to_netcdf(
        csv_dir=csv_dir,
        output=output,
        nlat=5,
        nlon=10,
        lat_bounds=(43.0, 48.0),
        lon_bounds=(-6.0, -1.0),
    )
    assert output.exists()
    ds = xr.open_dataset(output)
    assert len(ds.data_vars) >= 1
    ds.close()
