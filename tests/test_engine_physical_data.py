"""Tests for PhysicalData loader."""

import numpy as np
import pytest
from osmose.engine.physical_data import PhysicalData


class TestPhysicalDataConstant:
    def test_constant_value_all_cells(self):
        pd = PhysicalData.from_constant(value=15.0)
        assert pd.get_value(step=0, cell_y=0, cell_x=0) == pytest.approx(15.0)
        assert pd.is_constant is True

    def test_constant_with_factor_offset(self):
        pd = PhysicalData.from_constant(value=288.15, factor=1.0, offset=-273.15)
        assert pd.get_value(0, 0, 0) == pytest.approx(15.0)

    def test_get_scalar(self):
        pd = PhysicalData.from_constant(value=15.0)
        assert pd.get_scalar() == 15.0

    def test_constant_any_step(self):
        pd = PhysicalData.from_constant(value=20.0)
        assert pd.get_value(step=999, cell_y=5, cell_x=3) == pytest.approx(20.0)

    def test_get_scalar_raises_for_netcdf(self, tmp_path):
        import xarray as xr

        data = np.ones((2, 2, 2))
        ds = xr.Dataset({"temp": (["time", "y", "x"], data)})
        path = tmp_path / "temp.nc"
        ds.to_netcdf(path)
        pd = PhysicalData.from_netcdf(path, varname="temp", nsteps_year=2)
        with pytest.raises(ValueError):
            pd.get_scalar()

    def test_get_grid_raises_for_constant(self):
        pd = PhysicalData.from_constant(value=15.0)
        with pytest.raises(ValueError):
            pd.get_grid(0)


class TestPhysicalDataNetCDF:
    def test_netcdf_load(self, tmp_path):
        """Create a small NetCDF, load it, verify values."""
        import xarray as xr

        data = np.random.rand(12, 3, 4)  # 12 months, 3y, 4x
        ds = xr.Dataset({"temp": (["time", "y", "x"], data)})
        path = tmp_path / "temp.nc"
        ds.to_netcdf(path)
        pd = PhysicalData.from_netcdf(path, varname="temp", nsteps_year=12)
        assert pd.is_constant is False
        assert pd.get_value(0, 1, 2) == pytest.approx(data[0, 1, 2])

    def test_netcdf_cyclic(self, tmp_path):
        """Step beyond data length wraps around."""
        import xarray as xr

        data = np.ones((6, 2, 2)) * np.arange(6)[:, None, None]
        ds = xr.Dataset({"temp": (["time", "y", "x"], data)})
        path = tmp_path / "temp.nc"
        ds.to_netcdf(path)
        pd = PhysicalData.from_netcdf(path, nsteps_year=6)
        assert pd.get_value(6, 0, 0) == pd.get_value(0, 0, 0)  # wraps

    def test_netcdf_factor_offset(self, tmp_path):
        """Factor and offset are applied on load."""
        import xarray as xr

        data = np.full((1, 2, 2), 273.15)
        ds = xr.Dataset({"temp": (["time", "y", "x"], data)})
        path = tmp_path / "temp.nc"
        ds.to_netcdf(path)
        pd = PhysicalData.from_netcdf(
            path, varname="temp", nsteps_year=1, factor=1.0, offset=-273.15
        )
        assert pd.get_value(0, 0, 0) == pytest.approx(0.0)

    def test_netcdf_2d_input(self, tmp_path):
        """2D array (y, x) is promoted to (1, y, x)."""
        import xarray as xr

        data = np.random.rand(3, 4)
        ds = xr.Dataset({"temp": (["y", "x"], data)})
        path = tmp_path / "temp.nc"
        ds.to_netcdf(path)
        pd = PhysicalData.from_netcdf(path, varname="temp", nsteps_year=1)
        assert pd.is_constant is False
        assert pd.get_value(0, 1, 2) == pytest.approx(data[1, 2])

    def test_get_grid(self, tmp_path):
        """get_grid returns (ny, nx) slice."""
        import xarray as xr

        data = np.random.rand(3, 4, 5)
        ds = xr.Dataset({"temp": (["time", "y", "x"], data)})
        path = tmp_path / "temp.nc"
        ds.to_netcdf(path)
        pd = PhysicalData.from_netcdf(path, varname="temp", nsteps_year=3)
        grid = pd.get_grid(1)
        assert grid.shape == (4, 5)
        np.testing.assert_array_almost_equal(grid, data[1])
