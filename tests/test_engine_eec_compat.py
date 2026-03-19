"""Tests for EEC configuration compatibility fixes.

Covers:
1. Resource species loaded from species.type.sp{N} = resource pattern
2. Spawning season CSV found when in a subdirectory
3. Grid loads from EEC-style NetCDF (y/x dimensions, 2D lat/lon)
4. Egg weight override from config
"""

import numpy as np
import pytest
import xarray as xr

from osmose.engine.config import EngineConfig, _resolve_file
from osmose.engine.grid import Grid
from osmose.engine.resources import ResourceState
from osmose.engine.simulate import initialize


# ---------------------------------------------------------------------------
# Issue 1: Resource species key pattern mismatch
# ---------------------------------------------------------------------------


class TestResourceSpeciesTypePattern:
    def test_species_type_resource_discovery(self):
        """Resources loaded via species.type.sp{N} = resource keys."""
        grid = Grid.from_dimensions(ny=3, nx=3)
        config = {
            "simulation.nresource": "2",
            "simulation.time.ndtperyear": "24",
            "species.type.sp14": "resource",
            "species.type.sp15": "resource",
            "species.name.sp14": "Dinoflagellates",
            "species.name.sp15": "Diatoms",
            "species.size.min.sp14": "0.0002",
            "species.size.max.sp14": "0.02",
            "species.accessibility2fish.sp14": "0.08",
            "species.trophic.level.sp14": "1.0",
            "species.size.min.sp15": "0.001",
            "species.size.max.sp15": "0.05",
            "species.accessibility2fish.sp15": "0.10",
            "species.trophic.level.sp15": "2.0",
        }
        rs = ResourceState(config=config, grid=grid)
        assert rs.n_resources == 2
        assert rs.species[0].name == "Dinoflagellates"
        assert rs.species[0].size_min == pytest.approx(0.0002)
        assert rs.species[0].accessibility == pytest.approx(0.08)
        assert rs.species[1].name == "Diatoms"
        assert rs.species[1].trophic_level == pytest.approx(2.0)

    def test_species_type_resource_sorted_by_index(self):
        """Resource species are sorted by file index."""
        grid = Grid.from_dimensions(ny=2, nx=2)
        config = {
            "simulation.nresource": "2",
            "simulation.time.ndtperyear": "24",
            "species.type.sp20": "resource",
            "species.type.sp15": "resource",
            "species.name.sp15": "First",
            "species.name.sp20": "Second",
        }
        rs = ResourceState(config=config, grid=grid)
        assert rs.species[0].name == "First"
        assert rs.species[1].name == "Second"

    def test_species_type_resource_with_netcdf(self, tmp_path):
        """Resource species loaded from species.type pattern with NetCDF forcing."""
        grid = Grid.from_dimensions(ny=3, nx=3)
        data = np.ones((6, 3, 3), dtype=np.float32) * 5.0
        ds = xr.Dataset({"Plankton": (["time", "lat", "lon"], data)})
        nc_path = tmp_path / "ltl.nc"
        ds.to_netcdf(nc_path)
        ds.close()

        config = {
            "simulation.nresource": "1",
            "simulation.time.ndtperyear": "24",
            "species.type.sp10": "resource",
            "species.name.sp10": "Plankton",
            "species.size.min.sp10": "0.01",
            "species.size.max.sp10": "0.1",
            "species.accessibility2fish.sp10": "0.5",
            "species.file.sp10": str(nc_path),
        }
        rs = ResourceState(config=config, grid=grid)
        assert rs.n_resources == 1
        rs.update(step=0)
        # 5.0 * 0.5 = 2.5
        np.testing.assert_allclose(rs.biomass[0, 0], 2.5, rtol=1e-6)
        rs.close()

    def test_legacy_ltl_pattern_still_works(self):
        """Ensure legacy ltl.*.rsc{i} keys still work."""
        grid = Grid.from_dimensions(ny=3, nx=3)
        config = {
            "simulation.nresource": "1",
            "simulation.time.ndtperyear": "24",
            "ltl.name.rsc0": "Phyto",
            "ltl.size.min.rsc0": "0.001",
            "ltl.size.max.rsc0": "0.01",
            "ltl.tl.rsc0": "1.0",
            "ltl.accessibility2fish.rsc0": "0.05",
            "ltl.biomass.total.rsc0": "900.0",
        }
        rs = ResourceState(config=config, grid=grid)
        assert rs.n_resources == 1
        assert rs.species[0].name == "Phyto"
        rs.update(step=0)
        expected = 900.0 / 9 * 0.05
        np.testing.assert_allclose(rs.biomass[0, 0], expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Issue 2: Spawning season path resolution
# ---------------------------------------------------------------------------


class TestPathResolution:
    def test_resolve_file_in_subdirectory(self, tmp_path, monkeypatch):
        """_resolve_file finds files in data/*/ subdirectories."""
        monkeypatch.chdir(tmp_path)
        subdir = tmp_path / "data" / "eec_full" / "reproduction"
        subdir.mkdir(parents=True)
        csv_file = subdir / "seasonality.csv"
        csv_file.write_text("Time;Value\n0;0.5\n1;0.5\n")

        result = _resolve_file("reproduction/seasonality.csv")
        assert result is not None
        assert result.exists()

    def test_resolve_file_not_found(self, tmp_path, monkeypatch):
        """_resolve_file returns None for missing files."""
        monkeypatch.chdir(tmp_path)
        result = _resolve_file("nonexistent/file.csv")
        assert result is None

    def test_spawning_season_from_subdir(self, tmp_path, monkeypatch):
        """Spawning season CSV loads from data/*/reproduction/ subdirectory."""
        monkeypatch.chdir(tmp_path)
        subdir = tmp_path / "data" / "mymodel"
        repro_dir = subdir / "reproduction"
        repro_dir.mkdir(parents=True)

        # Write a spawning season CSV with 12 timesteps
        rows = ["Time;Season"] + [f"{i};{0.5 if i < 6 else 0.0}" for i in range(12)]
        csv_path = repro_dir / "seasonality-sp0.csv"
        csv_path.write_text("\n".join(rows))

        cfg = _make_minimal_config(n_dt=12)
        cfg["reproduction.season.file.sp0"] = "reproduction/seasonality-sp0.csv"

        ec = EngineConfig.from_dict(cfg)
        assert ec.spawning_season is not None
        assert ec.spawning_season.shape == (1, 12)
        # First 6 values should be 0.5, last 6 should be 0.0
        np.testing.assert_allclose(ec.spawning_season[0, :6], 0.5)
        np.testing.assert_allclose(ec.spawning_season[0, 6:], 0.0)


# ---------------------------------------------------------------------------
# Issue 3: Grid NetCDF flexible dimension names
# ---------------------------------------------------------------------------


class TestGridFlexibleDimNames:
    def test_grid_yx_dimensions(self, tmp_path):
        """Grid loads from NetCDF with y/x dimension names."""
        ny, nx = 5, 6
        lat_2d = np.tile(np.linspace(48.0, 51.0, ny).reshape(-1, 1), (1, nx))
        lon_2d = np.tile(np.linspace(-5.0, 0.0, nx).reshape(1, -1), (ny, 1))
        mask = np.ones((ny, nx), dtype=np.float32)
        mask[0, 0] = -1

        ds = xr.Dataset(
            {
                "mask": (["y", "x"], mask),
                "lat": (["y", "x"], lat_2d),
                "lon": (["y", "x"], lon_2d),
            },
            coords={"y": np.arange(ny), "x": np.arange(nx)},
        )
        path = tmp_path / "eec_grid.nc"
        ds.to_netcdf(path)

        grid = Grid.from_netcdf(path)
        assert grid.ny == 5
        assert grid.nx == 6
        assert grid.n_ocean_cells == 29
        # lat should be 1D (extracted from 2D)
        assert grid.lat.ndim == 1
        assert len(grid.lat) == 5
        assert grid.lon.ndim == 1
        assert len(grid.lon) == 6

    def test_grid_lat_lon_dimensions(self, tmp_path):
        """Grid loads from NetCDF with lat/lon dimension names (short form)."""
        ny, nx = 3, 4
        lat = np.linspace(43.0, 46.0, ny)
        lon = np.linspace(-3.0, 1.0, nx)
        mask = np.ones((ny, nx), dtype=np.float32)

        ds = xr.Dataset(
            {"mask": (["lat", "lon"], mask)},
            coords={"lat": lat, "lon": lon},
        )
        path = tmp_path / "grid_latlon.nc"
        ds.to_netcdf(path)

        grid = Grid.from_netcdf(path)
        assert grid.ny == 3
        assert grid.nx == 4
        np.testing.assert_allclose(grid.lat, lat)
        np.testing.assert_allclose(grid.lon, lon)

    def test_grid_latitude_longitude_dimensions(self, tmp_path):
        """Grid loads from NetCDF with full latitude/longitude dimension names."""
        ny, nx = 4, 5
        lat = np.linspace(43.0, 48.0, ny)
        lon = np.linspace(-5.0, 0.0, nx)
        mask = np.ones((ny, nx), dtype=np.float32)

        ds = xr.Dataset(
            {"mask": (["latitude", "longitude"], mask)},
            coords={"latitude": lat, "longitude": lon},
        )
        path = tmp_path / "grid_std.nc"
        ds.to_netcdf(path)

        grid = Grid.from_netcdf(path)
        assert grid.ny == 4
        assert grid.nx == 5
        np.testing.assert_allclose(grid.lat, lat)
        np.testing.assert_allclose(grid.lon, lon)

    def test_grid_explicit_dim_override(self, tmp_path):
        """Explicit lat_dim/lon_dim parameters still work."""
        ny, nx = 3, 3
        lat = np.linspace(40.0, 42.0, ny)
        lon = np.linspace(0.0, 2.0, nx)
        mask = np.ones((ny, nx), dtype=np.float32)

        ds = xr.Dataset(
            {"mask": (["my_lat", "my_lon"], mask)},
            coords={"my_lat": lat, "my_lon": lon},
        )
        path = tmp_path / "grid_custom.nc"
        ds.to_netcdf(path)

        grid = Grid.from_netcdf(path, lat_dim="my_lat", lon_dim="my_lon")
        assert grid.ny == 3
        assert grid.nx == 3
        np.testing.assert_allclose(grid.lat, lat)


# ---------------------------------------------------------------------------
# Issue 4: Egg weight override from config
# ---------------------------------------------------------------------------


class TestEggWeightOverride:
    def test_egg_weight_override_parsed(self):
        """species.egg.weight.sp{i} parsed into egg_weight_override."""
        cfg = _make_minimal_config()
        cfg["species.egg.weight.sp0"] = "0.001"
        ec = EngineConfig.from_dict(cfg)
        assert ec.egg_weight_override is not None
        assert ec.egg_weight_override[0] == pytest.approx(0.001)

    def test_egg_weight_override_none_when_absent(self):
        """egg_weight_override is None when no species.egg.weight keys present."""
        cfg = _make_minimal_config()
        ec = EngineConfig.from_dict(cfg)
        assert ec.egg_weight_override is None

    def test_egg_weight_override_partial(self):
        """Partial override: some species have it, others get NaN."""
        cfg = _make_minimal_config(n_sp=2)
        cfg["species.egg.weight.sp0"] = "0.001"
        # sp1 has no override
        ec = EngineConfig.from_dict(cfg)
        assert ec.egg_weight_override is not None
        assert ec.egg_weight_override[0] == pytest.approx(0.001)
        assert np.isnan(ec.egg_weight_override[1])

    def test_initialize_uses_egg_weight_override(self):
        """initialize() uses egg_weight_override instead of allometry."""
        cfg = _make_minimal_config()
        cfg["species.egg.weight.sp0"] = "0.005"
        cfg["population.seeding.biomass.sp0"] = "100.0"
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        state = initialize(ec, grid, rng)
        # All schools of sp0 should have weight = 0.005
        np.testing.assert_allclose(state.weight, 0.005)


# ---------------------------------------------------------------------------
# Helper to build minimal config dicts for testing
# ---------------------------------------------------------------------------


def _make_minimal_config(n_sp: int = 1, n_dt: int = 24) -> dict[str, str]:
    """Build a minimal valid config dict for n_sp focal species."""
    cfg: dict[str, str] = {
        "simulation.time.ndtperyear": str(n_dt),
        "simulation.time.nyear": "1",
        "simulation.nspecies": str(n_sp),
        "mortality.subdt": "10",
    }
    for i in range(n_sp):
        cfg.update(
            {
                f"simulation.nschool.sp{i}": "10",
                f"species.name.sp{i}": f"Species{i}",
                f"species.linf.sp{i}": "20.0",
                f"species.k.sp{i}": "0.3",
                f"species.t0.sp{i}": "-0.1",
                f"species.egg.size.sp{i}": "0.1",
                f"species.length2weight.condition.factor.sp{i}": "0.006",
                f"species.length2weight.allometric.power.sp{i}": "3.0",
                f"species.lifespan.sp{i}": "3",
                f"species.vonbertalanffy.threshold.age.sp{i}": "1.0",
                f"predation.ingestion.rate.max.sp{i}": "3.5",
                f"predation.efficiency.critical.sp{i}": "0.57",
            }
        )
    return cfg
