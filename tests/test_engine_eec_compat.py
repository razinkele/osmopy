"""Tests for EEC configuration compatibility fixes.

Covers:
1. Resource species loaded from species.type.sp{N} = resource pattern
2. Spawning season CSV found when in a subdirectory
3. Grid loads from EEC-style NetCDF (y/x dimensions, 2D lat/lon)
4. Egg weight override from config
5. Maturity age check (Fix 1.1)
6. Spatial fishing distribution (Fix 1.2)
7. Lmax cap (Fix 1.3)
8. Resource multiplier/offset (Fix 1.4)
9. Time-varying accessibility (Fix 1.5)
10. Accessibility cap at 0.99 (Fix 1.6)
11. Trophic level computation (Fix 1.7)
"""

import numpy as np
import pytest
import xarray as xr

from osmose.engine.config import EngineConfig, _resolve_file
from osmose.engine.grid import Grid
from osmose.engine.processes.growth import growth
from osmose.engine.processes.mortality import (
    _apply_fishing_for_school,
    mortality,
)
from osmose.engine.processes.reproduction import reproduction
from osmose.engine.resources import ResourceState
from osmose.engine.state import SchoolState


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
        # Config value 0.001 grams → stored as 0.001 * 1e-6 = 1e-9 tonnes
        assert ec.egg_weight_override[0] == pytest.approx(0.001 * 1e-6)

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
        # Config value 0.001 grams → stored as 0.001 * 1e-6 = 1e-9 tonnes
        assert ec.egg_weight_override[0] == pytest.approx(0.001 * 1e-6)
        assert np.isnan(ec.egg_weight_override[1])


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


# ---------------------------------------------------------------------------
# Fix 1.1: Maturity age check
# ---------------------------------------------------------------------------


class TestMaturityAgeCheck:
    """Schools below maturity age excluded from SSB."""

    def test_below_maturity_age_excluded_from_ssb(self):
        cfg_dict = _make_minimal_config()
        cfg_dict["species.maturity.age.sp0"] = "2.0"
        cfg_dict["species.maturity.size.sp0"] = "5.0"
        cfg_dict["population.seeding.biomass.sp0"] = "0.0"
        cfg = EngineConfig.from_dict(cfg_dict)

        # School with adequate size but below maturity age
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            length=np.array([10.0]),  # above maturity size
            weight=np.array([6e-3]),
            age_dt=np.array([24], dtype=np.int32),  # 1 year, below 2-year maturity
            abundance=np.array([1000.0]),
            biomass=np.array([6.0]),
        )
        rng = np.random.default_rng(42)
        new_state = reproduction(state, cfg, step=0, rng=rng)
        # No new schools should be created (original school + age increment only)
        assert len(new_state) == 1

    def test_above_maturity_age_included_in_ssb(self):
        cfg_dict = _make_minimal_config()
        cfg_dict["species.maturity.age.sp0"] = "1.0"
        cfg_dict["species.maturity.size.sp0"] = "5.0"
        cfg_dict["population.seeding.biomass.sp0"] = "0.0"
        cfg_dict["species.sexratio.sp0"] = "0.5"
        cfg_dict["species.relativefecundity.sp0"] = "500.0"
        cfg = EngineConfig.from_dict(cfg_dict)

        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            length=np.array([10.0]),
            weight=np.array([6e-3]),
            age_dt=np.array([48], dtype=np.int32),  # 2 years, above 1-year maturity
            abundance=np.array([1000.0]),
            biomass=np.array([6.0]),
        )
        rng = np.random.default_rng(42)
        new_state = reproduction(state, cfg, step=0, rng=rng)
        assert len(new_state) > 1

    def test_maturity_age_default_zero(self):
        cfg_dict = _make_minimal_config()
        cfg_dict["species.maturity.size.sp0"] = "5.0"
        cfg = EngineConfig.from_dict(cfg_dict)
        assert cfg.maturity_age_dt[0] == 0


# ---------------------------------------------------------------------------
# Fix 1.2: Spatial fishing distribution
# ---------------------------------------------------------------------------


class TestSpatialFishing:
    """Cell with 0 fishing map value -> no fishing."""

    def test_zero_map_cell_no_fishing(self, tmp_path):
        map_data = np.ones((4, 4))
        map_data[2, 1] = 0.0  # row 2, col 1 in the grid
        map_file = tmp_path / "fishing_map.csv"
        # Write without flipud; _load_spatial_csv does flipud on load
        np.savetxt(map_file, map_data, delimiter=";", fmt="%.1f")

        cfg_dict = _make_minimal_config()
        cfg_dict["_osmose.config.dir"] = str(tmp_path)
        cfg_dict["mortality.fishing.rate.sp0"] = "0.5"
        cfg_dict["mortality.fishing.spatial.distrib.file.sp0"] = str(map_file)
        cfg_dict["simulation.fishing.mortality.enabled"] = "true"
        cfg = EngineConfig.from_dict(cfg_dict)

        # After flipud, row 2 of file becomes row 1 of grid
        assert cfg.fishing_spatial_maps[0] is not None
        # Find the zero cell in the loaded map
        loaded_map = cfg.fishing_spatial_maps[0]
        zero_y, zero_x = np.where(loaded_map == 0.0)
        assert len(zero_y) == 1
        zy, zx = int(zero_y[0]), int(zero_x[0])

        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            biomass=np.array([6.0]),
            length=np.array([15.0]),
            weight=np.array([6e-3]),
            age_dt=np.array([48], dtype=np.int32),
            cell_y=np.array([zy], dtype=np.int32),
            cell_x=np.array([zx], dtype=np.int32),
        )
        _apply_fishing_for_school(0, state, cfg, n_subdt=1, inst_abd=state.abundance.copy())
        assert state.n_dead[0, 3] == 0.0  # FISHING = 3

    def test_nonzero_map_cell_has_fishing(self, tmp_path):
        map_data = np.ones((4, 4)) * 0.5
        map_file = tmp_path / "fishing_map.csv"
        np.savetxt(map_file, map_data, delimiter=";", fmt="%.1f")

        cfg_dict = _make_minimal_config()
        cfg_dict["_osmose.config.dir"] = str(tmp_path)
        cfg_dict["mortality.fishing.rate.sp0"] = "0.5"
        cfg_dict["mortality.fishing.spatial.distrib.file.sp0"] = str(map_file)
        cfg_dict["simulation.fishing.mortality.enabled"] = "true"
        cfg = EngineConfig.from_dict(cfg_dict)

        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            biomass=np.array([6.0]),
            length=np.array([15.0]),
            weight=np.array([6e-3]),
            age_dt=np.array([48], dtype=np.int32),
            cell_y=np.array([1], dtype=np.int32),
            cell_x=np.array([2], dtype=np.int32),
        )
        _apply_fishing_for_school(0, state, cfg, n_subdt=1, inst_abd=state.abundance.copy())
        assert state.n_dead[0, 3] > 0.0


# ---------------------------------------------------------------------------
# Fix 1.3: Lmax cap
# ---------------------------------------------------------------------------


class TestLmaxCap:
    """Growth capped at lmax, not linf."""

    def test_growth_capped_at_lmax_not_linf(self):
        cfg_dict = _make_minimal_config()
        cfg_dict["species.linf.sp0"] = "30.0"
        cfg_dict["species.lmax.sp0"] = "25.0"
        cfg = EngineConfig.from_dict(cfg_dict)

        assert cfg.lmax[0] == 25.0
        assert cfg.linf[0] == 30.0

        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            length=np.array([24.5]),
            weight=np.array([0.006 * 24.5**3 * 1e-6]),
            age_dt=np.array([96], dtype=np.int32),
            abundance=np.array([100.0]),
            biomass=np.array([100.0 * 0.006 * 24.5**3 * 1e-6]),
            pred_success_rate=np.array([1.0]),
        )
        new_state = growth(state, cfg, np.random.default_rng(42))
        assert new_state.length[0] <= 25.0

    def test_lmax_defaults_to_linf(self):
        cfg_dict = _make_minimal_config()
        cfg = EngineConfig.from_dict(cfg_dict)
        assert cfg.lmax[0] == cfg.linf[0]


# ---------------------------------------------------------------------------
# Fix 1.4: Resource multiplier and offset
# ---------------------------------------------------------------------------


class TestResourceMultiplierOffset:
    """Resource biomass scaled by multiplier."""

    def test_multiplier_parsed(self):
        grid = Grid.from_dimensions(ny=2, nx=2)
        config = {
            "simulation.nresource": "1",
            "simulation.time.ndtperyear": "24",
            "species.type.sp1": "resource",
            "species.name.sp1": "Plankton",
            "species.size.min.sp1": "0.01",
            "species.size.max.sp1": "0.1",
            "species.trophic.level.sp1": "1.0",
            "species.accessibility2fish.sp1": "0.5",
            "species.multiplier.sp1": "2.0",
            "species.offset.sp1": "5.0",
        }
        rs = ResourceState(config=config, grid=grid)
        assert rs.species[0].multiplier == 2.0
        assert rs.species[0].offset == 5.0


# ---------------------------------------------------------------------------
# Fix 1.5: Time-varying accessibility
# ---------------------------------------------------------------------------


class TestTimeVaryingAccessibility:
    """Different step -> different accessibility."""

    def test_time_varying_accessibility(self, tmp_path):
        grid = Grid.from_dimensions(ny=2, nx=2)

        access_file = tmp_path / "access_ts.csv"
        access_file.write_text("step;value\n0;0.1\n1;0.3\n2;0.5\n3;0.7\n")

        data = np.ones((4, 2, 2), dtype=np.float32) * 100.0
        ds = xr.Dataset({"Plankton": (["time", "lat", "lon"], data)})
        nc_path = tmp_path / "forcing.nc"
        ds.to_netcdf(nc_path)
        ds.close()

        config = {
            "simulation.nresource": "1",
            "simulation.time.ndtperyear": "4",
            "species.type.sp1": "resource",
            "species.name.sp1": "Plankton",
            "species.size.min.sp1": "0.01",
            "species.size.max.sp1": "0.1",
            "species.trophic.level.sp1": "1.0",
            "species.accessibility2fish.sp1": "0.5",
            "species.accessibility2fish.file.sp1": str(access_file),
            "species.file.sp1": str(nc_path),
        }
        rs = ResourceState(config=config, grid=grid)

        rs.update(step=0)
        val_step0 = rs.biomass[0, 0]

        rs.update(step=2)
        val_step2 = rs.biomass[0, 0]

        assert val_step0 != val_step2
        # multiplier=1.0 (default), step 0 access=0.1, step 2 access=0.5
        np.testing.assert_allclose(val_step0, 100.0 * 0.1, rtol=1e-6)
        np.testing.assert_allclose(val_step2, 100.0 * 0.5, rtol=1e-6)
        rs.close()


# ---------------------------------------------------------------------------
# Fix 1.6: Accessibility cap at 0.99
# ---------------------------------------------------------------------------


class TestAccessibilityCap:
    """Accessibility capped at 0.99."""

    def test_ltl_accessibility_capped(self):
        grid = Grid.from_dimensions(ny=2, nx=2)
        config = {
            "simulation.nresource": "1",
            "simulation.time.ndtperyear": "24",
            "ltl.name.rsc0": "Phyto",
            "ltl.size.min.rsc0": "0.01",
            "ltl.size.max.rsc0": "0.1",
            "ltl.tl.rsc0": "1.0",
            "ltl.accessibility2fish.rsc0": "1.0",
        }
        rs = ResourceState(config=config, grid=grid)
        assert rs.species[0].accessibility == 0.99

    def test_species_type_accessibility_capped(self):
        grid = Grid.from_dimensions(ny=2, nx=2)
        config = {
            "simulation.nresource": "1",
            "simulation.time.ndtperyear": "24",
            "species.type.sp1": "resource",
            "species.name.sp1": "Phyto",
            "species.size.min.sp1": "0.01",
            "species.size.max.sp1": "0.1",
            "species.trophic.level.sp1": "1.0",
            "species.accessibility2fish.sp1": "1.0",
        }
        rs = ResourceState(config=config, grid=grid)
        assert rs.species[0].accessibility == 0.99

    def test_below_cap_unchanged(self):
        grid = Grid.from_dimensions(ny=2, nx=2)
        config = {
            "simulation.nresource": "1",
            "simulation.time.ndtperyear": "24",
            "ltl.name.rsc0": "Phyto",
            "ltl.size.min.rsc0": "0.01",
            "ltl.size.max.rsc0": "0.1",
            "ltl.tl.rsc0": "1.0",
            "ltl.accessibility2fish.rsc0": "0.5",
        }
        rs = ResourceState(config=config, grid=grid)
        assert rs.species[0].accessibility == 0.5


# ---------------------------------------------------------------------------
# Fix 1.7: Trophic level computation
# ---------------------------------------------------------------------------


class TestTrophicLevel:
    """Predator eating prey of known TL -> correct weighted TL."""

    def test_trophic_level_from_predation(self):
        cfg_dict = _make_minimal_config(n_sp=2)
        cfg_dict["mortality.subdt"] = "1"
        cfg_dict["predation.ingestion.rate.max.sp0"] = "10.0"
        cfg_dict["predation.ingestion.rate.max.sp1"] = "0.0"
        cfg_dict["predation.predprey.sizeratio.min.sp0"] = "1.5"
        cfg_dict["predation.predprey.sizeratio.max.sp0"] = "5.0"
        cfg_dict["predation.predprey.sizeratio.min.sp1"] = "100.0"
        cfg_dict["predation.predprey.sizeratio.max.sp1"] = "200.0"
        cfg = EngineConfig.from_dict(cfg_dict)

        grid = Grid.from_dimensions(ny=2, nx=2)
        resources = ResourceState(config=cfg_dict, grid=grid)

        # Predator (large) and prey (small) in same cell
        state = SchoolState.create(n_schools=2, species_id=np.array([0, 1], dtype=np.int32))
        state = state.replace(
            length=np.array([20.0, 8.0]),  # ratio = 2.5, in [1.5, 5.0)
            weight=np.array([0.048, 0.003]),
            age_dt=np.array([48, 48], dtype=np.int32),
            abundance=np.array([100.0, 5000.0]),
            biomass=np.array([4.8, 15.0]),
            trophic_level=np.array([1.0, 2.5]),
            cell_x=np.array([0, 0], dtype=np.int32),
            cell_y=np.array([0, 0], dtype=np.int32),
            first_feeding_age_dt=np.array([1, 1], dtype=np.int32),
        )

        new_state = mortality(state, resources, cfg, np.random.default_rng(42), grid)

        # Predator should have TL > 1 if it ate anything
        if new_state.preyed_biomass[0] > 0:
            assert new_state.trophic_level[0] > 1.0
            # TL = 1 + weighted_prey_TL. Prey TL = 2.5, so predator TL ~ 3.5
            np.testing.assert_allclose(new_state.trophic_level[0], 3.5, atol=0.5)
