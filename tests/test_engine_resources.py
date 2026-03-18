"""Tests for resource species (LTL) system."""

import numpy as np
import pytest
import xarray as xr

from osmose.engine.grid import Grid
from osmose.engine.resources import ResourceSpeciesInfo, ResourceState


class TestResourceSpeciesInfo:
    def test_dataclass_fields(self):
        info = ResourceSpeciesInfo(
            name="Phyto", size_min=0.001, size_max=0.01, trophic_level=1.0, accessibility=0.05
        )
        assert info.name == "Phyto"
        assert info.size_min == 0.001
        assert info.size_max == 0.01
        assert info.trophic_level == 1.0
        assert info.accessibility == 0.05


class TestResourceState:
    def test_no_resources(self):
        grid = Grid.from_dimensions(ny=5, nx=5)
        rs = ResourceState(config={}, grid=grid)
        assert rs.n_resources == 0
        assert len(rs.species) == 0
        rs.update(step=0)
        # Should not error

    def test_uniform_biomass(self):
        grid = Grid.from_dimensions(ny=5, nx=5)
        config = {
            "simulation.nresource": "1",
            "simulation.time.ndtperyear": "24",
            "ltl.name.rsc0": "Plankton",
            "ltl.size.min.rsc0": "0.01",
            "ltl.size.max.rsc0": "0.1",
            "ltl.tl.rsc0": "1.0",
            "ltl.accessibility2fish.rsc0": "0.05",
            "ltl.biomass.total.rsc0": "1000.0",
        }
        rs = ResourceState(config=config, grid=grid)
        assert rs.n_resources == 1
        assert rs.species[0].name == "Plankton"
        rs.update(step=0)
        # Uniform: 1000 / 25 cells * 0.05 accessibility = 2.0 per cell
        expected = 1000.0 / 25 * 0.05
        np.testing.assert_allclose(rs.biomass[0, 0], expected, rtol=1e-10)

    def test_uniform_biomass_all_cells_equal(self):
        grid = Grid.from_dimensions(ny=3, nx=4)
        config = {
            "simulation.nresource": "1",
            "simulation.time.ndtperyear": "24",
            "ltl.name.rsc0": "Phyto",
            "ltl.size.min.rsc0": "0.001",
            "ltl.size.max.rsc0": "0.01",
            "ltl.tl.rsc0": "1.0",
            "ltl.accessibility2fish.rsc0": "0.10",
            "ltl.biomass.total.rsc0": "120.0",
        }
        rs = ResourceState(config=config, grid=grid)
        rs.update(step=0)
        expected = 120.0 / 12 * 0.10
        for cell in range(12):
            np.testing.assert_allclose(rs.biomass[0, cell], expected, rtol=1e-10)

    def test_netcdf_forcing(self, tmp_path):
        """Test loading from NetCDF file."""
        grid = Grid.from_dimensions(ny=4, nx=4)
        # Create a simple NetCDF
        data = np.ones((12, 4, 4), dtype=np.float32) * 10.0
        ds = xr.Dataset({"TestPlankton": (["time", "lat", "lon"], data)})
        nc_path = tmp_path / "test_ltl.nc"
        ds.to_netcdf(nc_path)
        ds.close()

        config = {
            "simulation.nresource": "1",
            "simulation.time.ndtperyear": "24",
            "ltl.name.rsc0": "TestPlankton",
            "ltl.size.min.rsc0": "0.01",
            "ltl.size.max.rsc0": "0.1",
            "ltl.tl.rsc0": "1.0",
            "ltl.accessibility2fish.rsc0": "0.10",
            "ltl.netcdf.file": str(nc_path),
        }
        rs = ResourceState(config=config, grid=grid)
        rs.update(step=0)
        # 10.0 * 0.10 accessibility = 1.0 per cell
        np.testing.assert_allclose(rs.biomass[0, 0], 1.0, rtol=1e-6)
        rs.close()

    def test_netcdf_regrid(self, tmp_path):
        """Test regridding from larger forcing grid to model grid."""
        grid = Grid.from_dimensions(ny=4, nx=4)
        # Forcing on 8x8 grid, model on 4x4
        data = np.ones((6, 8, 8), dtype=np.float32) * 5.0
        ds = xr.Dataset({"Phyto": (["time", "lat", "lon"], data)})
        nc_path = tmp_path / "test_regrid.nc"
        ds.to_netcdf(nc_path)
        ds.close()

        config = {
            "simulation.nresource": "1",
            "simulation.time.ndtperyear": "24",
            "ltl.name.rsc0": "Phyto",
            "ltl.size.min.rsc0": "0.01",
            "ltl.size.max.rsc0": "0.1",
            "ltl.tl.rsc0": "1.0",
            "ltl.accessibility2fish.rsc0": "0.20",
            "ltl.netcdf.file": str(nc_path),
        }
        rs = ResourceState(config=config, grid=grid)
        rs.update(step=0)
        # 5.0 * 0.20 = 1.0 per cell
        np.testing.assert_allclose(rs.biomass[0, 0], 1.0, rtol=1e-6)
        rs.close()

    def test_get_cell_biomass(self):
        grid = Grid.from_dimensions(ny=5, nx=5)
        config = {
            "simulation.nresource": "1",
            "simulation.time.ndtperyear": "24",
            "ltl.name.rsc0": "Plankton",
            "ltl.size.min.rsc0": "0.01",
            "ltl.size.max.rsc0": "0.1",
            "ltl.tl.rsc0": "1.0",
            "ltl.accessibility2fish.rsc0": "0.05",
            "ltl.biomass.total.rsc0": "1000.0",
        }
        rs = ResourceState(config=config, grid=grid)
        rs.update(step=0)
        bio = rs.get_cell_biomass(0, 2, 3)
        assert bio > 0
        expected = 1000.0 / 25 * 0.05
        np.testing.assert_allclose(bio, expected, rtol=1e-10)

    def test_get_cell_biomass_out_of_range(self):
        grid = Grid.from_dimensions(ny=5, nx=5)
        rs = ResourceState(config={}, grid=grid)
        assert rs.get_cell_biomass(0, 0, 0) == 0.0
        assert rs.get_cell_biomass(99, 0, 0) == 0.0

    def test_species_info_multiple(self):
        grid = Grid.from_dimensions(ny=5, nx=5)
        config = {
            "simulation.nresource": "2",
            "simulation.time.ndtperyear": "24",
            "ltl.name.rsc0": "SmallPhyto",
            "ltl.size.min.rsc0": "0.0002",
            "ltl.size.max.rsc0": "0.002",
            "ltl.tl.rsc0": "1.0",
            "ltl.accessibility2fish.rsc0": "0.01",
            "ltl.name.rsc1": "LargeZoo",
            "ltl.size.min.rsc1": "0.2",
            "ltl.size.max.rsc1": "2.0",
            "ltl.tl.rsc1": "2.5",
            "ltl.accessibility2fish.rsc1": "0.10",
        }
        rs = ResourceState(config=config, grid=grid)
        assert len(rs.species) == 2
        assert rs.species[0].name == "SmallPhyto"
        assert rs.species[0].size_min == pytest.approx(0.0002)
        assert rs.species[1].name == "LargeZoo"
        assert rs.species[1].size_max == pytest.approx(2.0)
        assert rs.species[1].trophic_level == pytest.approx(2.5)

    def test_biomass_shape(self):
        grid = Grid.from_dimensions(ny=3, nx=4)
        config = {
            "simulation.nresource": "3",
            "simulation.time.ndtperyear": "24",
            "ltl.name.rsc0": "A",
            "ltl.size.min.rsc0": "0.01",
            "ltl.size.max.rsc0": "0.1",
            "ltl.tl.rsc0": "1.0",
            "ltl.accessibility2fish.rsc0": "0.01",
            "ltl.name.rsc1": "B",
            "ltl.size.min.rsc1": "0.01",
            "ltl.size.max.rsc1": "0.1",
            "ltl.tl.rsc1": "1.0",
            "ltl.accessibility2fish.rsc1": "0.01",
            "ltl.name.rsc2": "C",
            "ltl.size.min.rsc2": "0.01",
            "ltl.size.max.rsc2": "0.1",
            "ltl.tl.rsc2": "1.0",
            "ltl.accessibility2fish.rsc2": "0.01",
        }
        rs = ResourceState(config=config, grid=grid)
        assert rs.biomass.shape == (3, 12)

    def test_close_idempotent(self):
        grid = Grid.from_dimensions(ny=2, nx=2)
        rs = ResourceState(config={}, grid=grid)
        rs.close()
        rs.close()  # Should not error

    def test_timestep_mapping(self, tmp_path):
        """Different timesteps map to different forcing indices."""
        grid = Grid.from_dimensions(ny=2, nx=2)
        # 4 forcing steps, distinct values
        data = np.zeros((4, 2, 2), dtype=np.float32)
        for t in range(4):
            data[t, :, :] = (t + 1) * 10.0
        ds = xr.Dataset({"Phyto": (["time", "lat", "lon"], data)})
        nc_path = tmp_path / "timestep_test.nc"
        ds.to_netcdf(nc_path)
        ds.close()

        config = {
            "simulation.nresource": "1",
            "simulation.time.ndtperyear": "4",
            "ltl.name.rsc0": "Phyto",
            "ltl.size.min.rsc0": "0.01",
            "ltl.size.max.rsc0": "0.1",
            "ltl.tl.rsc0": "1.0",
            "ltl.accessibility2fish.rsc0": "1.0",
            "ltl.netcdf.file": str(nc_path),
        }
        rs = ResourceState(config=config, grid=grid)

        rs.update(step=0)
        val_step0 = rs.biomass[0, 0]
        rs.update(step=2)
        val_step2 = rs.biomass[0, 0]

        # Step 0 -> forcing idx 0 (value 10), step 2 -> forcing idx 2 (value 30)
        np.testing.assert_allclose(val_step0, 10.0, rtol=1e-6)
        np.testing.assert_allclose(val_step2, 30.0, rtol=1e-6)
        rs.close()
