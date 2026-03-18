"""Tests for the engine Grid class and ResourceState."""

import numpy as np
import pytest
import xarray as xr

from osmose.engine.grid import Grid
from osmose.engine.resources import ResourceState


@pytest.fixture
def simple_grid_ds(tmp_path):
    """Create a simple 4x5 NetCDF grid file and return its path."""
    ny, nx = 4, 5
    lat = np.linspace(43.0, 48.0, ny)
    lon = np.linspace(-5.0, 0.0, nx)
    mask = np.ones((ny, nx), dtype=np.float32)
    mask[0, 0] = -1  # one land cell
    ds = xr.Dataset(
        {"mask": (["latitude", "longitude"], mask)},
        coords={"latitude": lat, "longitude": lon},
    )
    path = tmp_path / "grid.nc"
    ds.to_netcdf(path)
    return path


class TestGrid:
    def test_from_netcdf(self, simple_grid_ds):
        grid = Grid.from_netcdf(simple_grid_ds)
        assert grid.ny == 4
        assert grid.nx == 5
        assert grid.n_cells == 20

    def test_ocean_mask(self, simple_grid_ds):
        grid = Grid.from_netcdf(simple_grid_ds)
        assert grid.ocean_mask.shape == (4, 5)
        assert grid.ocean_mask[0, 0] is np.bool_(False)
        assert grid.ocean_mask[1, 1] is np.bool_(True)

    def test_n_ocean_cells(self, simple_grid_ds):
        grid = Grid.from_netcdf(simple_grid_ds)
        assert grid.n_ocean_cells == 19

    def test_from_dimensions(self):
        grid = Grid.from_dimensions(ny=10, nx=8)
        assert grid.ny == 10
        assert grid.nx == 8
        assert grid.n_cells == 80
        assert grid.n_ocean_cells == 80

    def test_cell_to_coords(self, simple_grid_ds):
        grid = Grid.from_netcdf(simple_grid_ds)
        y, x = grid.cell_to_yx(0)
        assert y == 0 and x == 0
        y, x = grid.cell_to_yx(6)
        assert y == 1 and x == 1

    def test_yx_to_cell(self, simple_grid_ds):
        grid = Grid.from_netcdf(simple_grid_ds)
        assert grid.yx_to_cell(0, 0) == 0
        assert grid.yx_to_cell(1, 1) == 6

    def test_neighbors_corner(self):
        grid = Grid.from_dimensions(ny=3, nx=3)
        nbrs = grid.neighbors(0, 0)
        assert (0, 1) in nbrs
        assert (1, 0) in nbrs
        assert (1, 1) in nbrs
        assert len(nbrs) == 3

    def test_neighbors_center(self):
        grid = Grid.from_dimensions(ny=3, nx=3)
        nbrs = grid.neighbors(1, 1)
        assert len(nbrs) == 8


class TestResourceState:
    def test_create_placeholder(self):
        grid = Grid.from_dimensions(ny=4, nx=5)
        rs = ResourceState(config={}, grid=grid)
        assert rs.grid is grid

    def test_update_is_noop(self):
        grid = Grid.from_dimensions(ny=4, nx=5)
        rs = ResourceState(config={}, grid=grid)
        rs.update(step=0)
