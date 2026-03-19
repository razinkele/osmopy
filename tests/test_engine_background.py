"""Tests for BackgroundSpeciesInfo and EngineConfig background species support."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from osmose.engine.background import (
    BackgroundSpeciesInfo,
    BackgroundState,
    parse_background_species,
)
from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.state import SchoolState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_base_config() -> dict[str, str]:
    """Minimal focal species config (1 species) for combining with background keys."""
    return {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.type.sp0": "focal",
        "species.name.sp0": "Anchovy",
        "species.linf.sp0": "20.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
    }


def _make_bkg_config(file_idx: int = 10) -> dict[str, str]:
    """Config keys for one background species at sp{file_idx}."""
    return {
        f"species.type.sp{file_idx}": "background",
        f"species.name.sp{file_idx}": "BkgSpecies",
        f"species.nclass.sp{file_idx}": "2",
        f"species.length.sp{file_idx}": "10;30",
        f"species.size.proportion.sp{file_idx}": "0.3;0.7",
        f"species.trophic.level.sp{file_idx}": "2;3",
        f"species.age.sp{file_idx}": "1;3",
        f"species.length2weight.condition.factor.sp{file_idx}": "0.00308",
        f"species.length2weight.allometric.power.sp{file_idx}": "3.029",
        f"predation.predprey.sizeratio.max.sp{file_idx}": "3",
        f"predation.predprey.sizeratio.min.sp{file_idx}": "50",
        f"predation.ingestion.rate.max.sp{file_idx}": "3.5",
        f"species.biomass.total.sp{file_idx}": "1000.0",
        "simulation.nbackground": "1",
    }


# ---------------------------------------------------------------------------
# Tests for parse_background_species
# ---------------------------------------------------------------------------


class TestParseBackgroundSpecies:
    def test_parse_single_species(self):
        """Parse a single background species and check basic fields."""
        cfg = {**_make_base_config(), **_make_bkg_config(file_idx=10)}
        result = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        assert len(result) == 1
        bkg = result[0]
        assert isinstance(bkg, BackgroundSpeciesInfo)
        assert bkg.file_index == 10
        assert bkg.species_index == 0  # first background species, 0-based among backgrounds
        assert bkg.n_class == 2

    def test_proportions_sum_to_one(self):
        """size proportions must sum to 1.0."""
        cfg = {**_make_base_config(), **_make_bkg_config(file_idx=10)}
        result = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        bkg = result[0]
        assert abs(sum(bkg.proportions) - 1.0) < 1e-9

    def test_ages_dt_truncate_first(self):
        """ages_dt uses truncate-first: int(age_float) * n_dt_per_year."""
        cfg = {**_make_base_config(), **_make_bkg_config(file_idx=10)}
        # ages are "1;3" in the fixture, n_dt_per_year=24
        result = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        bkg = result[0]
        assert bkg.ages_dt == [int(1) * 24, int(3) * 24]
        assert bkg.ages_dt == [24, 72]

    def test_multiple_species_sorted_numerically(self):
        """Multiple background species are sorted by file index, not lexicographically."""
        cfg = {**_make_base_config()}
        # Add two background species at indices 2 and 10
        cfg.update(_make_bkg_config(file_idx=2))
        cfg.update(_make_bkg_config(file_idx=10))
        cfg["simulation.nbackground"] = "2"
        result = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        assert len(result) == 2
        assert result[0].file_index == 2
        assert result[1].file_index == 10
        # species_index should be assigned in sorted order
        assert result[0].species_index == 0
        assert result[1].species_index == 1

    def test_name_stripping(self):
        """Underscores and hyphens are stripped from species names."""
        cfg = {**_make_base_config(), **_make_bkg_config(file_idx=10)}
        cfg["species.name.sp10"] = "_Bkg-Species_"
        result = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        assert result[0].name == "BkgSpecies"

    def test_multiplier_offset_defaults(self):
        """multiplier defaults to 1.0 and offset defaults to 0.0 when not in config."""
        cfg = {**_make_base_config(), **_make_bkg_config(file_idx=10)}
        result = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        bkg = result[0]
        assert bkg.multiplier == pytest.approx(1.0)
        assert bkg.offset == pytest.approx(0.0)

    def test_multiplier_offset_from_config(self):
        """multiplier and offset are read from config when present."""
        cfg = {**_make_base_config(), **_make_bkg_config(file_idx=10)}
        cfg["species.biomass.multiplier.sp10"] = "2.5"
        cfg["species.biomass.offset.sp10"] = "100.0"
        result = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        bkg = result[0]
        assert bkg.multiplier == pytest.approx(2.5)
        assert bkg.offset == pytest.approx(100.0)

    def test_forcing_nsteps_per_species_override(self):
        """forcing_nsteps_year is read from per-species key when present."""
        cfg = {**_make_base_config(), **_make_bkg_config(file_idx=10)}
        cfg["species.biomass.nsteps.year.sp10"] = "12"
        result = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        assert result[0].forcing_nsteps_year == 12

    def test_forcing_nsteps_global_fallback(self):
        """forcing_nsteps_year falls back to global key, then n_dt_per_year."""
        cfg = {**_make_base_config(), **_make_bkg_config(file_idx=10)}
        # No per-species or global key — should default to n_dt_per_year
        result = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        assert result[0].forcing_nsteps_year == 24

        # With global fallback key
        cfg["simulation.nsteps.year"] = "12"
        result2 = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        assert result2[0].forcing_nsteps_year == 12

    def test_no_background_species(self):
        """Returns empty list when no background species exist."""
        cfg = _make_base_config()
        result = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        assert result == []

    def test_lengths_and_trophic_parsed(self):
        """Lengths and trophic levels are parsed as float lists."""
        cfg = {**_make_base_config(), **_make_bkg_config(file_idx=10)}
        result = parse_background_species(cfg, n_focal=1, n_dt_per_year=24)
        bkg = result[0]
        assert bkg.lengths == pytest.approx([10.0, 30.0])
        assert bkg.trophic_levels == pytest.approx([2.0, 3.0])


# ---------------------------------------------------------------------------
# Tests for EngineConfig background integration
# ---------------------------------------------------------------------------


class TestEngineConfigBackground:
    def test_n_background_and_file_indices(self):
        """EngineConfig.from_dict sets n_background and background_file_indices."""
        cfg = {**_make_base_config(), **_make_bkg_config(file_idx=10)}
        ec = EngineConfig.from_dict(cfg)
        assert ec.n_background == 1
        assert ec.background_file_indices == [10]

    def test_all_species_names_includes_background(self):
        """all_species_names includes focal + background species names."""
        cfg = {**_make_base_config(), **_make_bkg_config(file_idx=10)}
        ec = EngineConfig.from_dict(cfg)
        assert "Anchovy" in ec.all_species_names
        assert "BkgSpecies" in ec.all_species_names
        assert len(ec.all_species_names) == 2

    def test_no_background_backward_compat(self):
        """EngineConfig.from_dict works with no background species (backward compat)."""
        cfg = _make_base_config()
        ec = EngineConfig.from_dict(cfg)
        assert ec.n_background == 0
        assert ec.background_file_indices == []
        assert ec.all_species_names == ec.species_names


# ---------------------------------------------------------------------------
# Helper to build BackgroundState from fixtures
# ---------------------------------------------------------------------------


def _make_bkg_state(extra_cfg: dict | None = None) -> tuple[BackgroundState, EngineConfig, Grid]:
    """Build a BackgroundState with 1 focal + 1 background species on a 3x3 all-ocean grid."""
    cfg = {**_make_base_config(), **_make_bkg_config(file_idx=10)}
    if extra_cfg:
        cfg.update(extra_cfg)
    ec = EngineConfig.from_dict(cfg)
    grid = Grid.from_dimensions(ny=3, nx=3)
    state = BackgroundState(config=cfg, grid=grid, engine_config=ec)
    return state, ec, grid


# ---------------------------------------------------------------------------
# Tests for BackgroundState uniform forcing
# ---------------------------------------------------------------------------


class TestBackgroundStateUniform:
    def test_get_schools_returns_schoolstate(self):
        """get_schools() returns a SchoolState instance."""
        bkg, ec, grid = _make_bkg_state()
        result = bkg.get_schools(step=0)
        assert isinstance(result, SchoolState)

    def test_school_count_equals_nclass_times_ocean_cells(self):
        """1 species * 2 classes * 9 ocean cells = 18 schools."""
        bkg, ec, grid = _make_bkg_state()
        result = bkg.get_schools(step=0)
        # 1 background species, 2 classes, 3x3 all-ocean grid = 9 cells
        assert len(result) == 1 * 2 * 9

    def test_is_background_flag(self):
        """All schools have is_background == True."""
        bkg, ec, grid = _make_bkg_state()
        result = bkg.get_schools(step=0)
        assert result.is_background.all()

    def test_species_id_offset(self):
        """species_id == n_focal + bkg_idx == 1 + 0 == 1 for all schools."""
        bkg, ec, grid = _make_bkg_state()
        result = bkg.get_schools(step=0)
        assert (result.species_id == 1).all()

    def test_first_feeding_age_dt_is_negative_one(self):
        """first_feeding_age_dt must be -1 (Java convention: always eligible)."""
        bkg, ec, grid = _make_bkg_state()
        result = bkg.get_schools(step=0)
        assert (result.first_feeding_age_dt == -1).all()

    def test_biomass_from_uniform_forcing(self):
        """Biomass per school matches uniform distribution.

        total=1000, 9 ocean cells → per_cell=1000/9
        class 0 proportion=0.3: biomass = (1000/9) * 0.3
        class 1 proportion=0.7: biomass = (1000/9) * 0.7
        """
        bkg, ec, grid = _make_bkg_state()
        result = bkg.get_schools(step=0)
        per_cell = 1000.0 / 9
        # First 9 schools = class 0, next 9 = class 1
        expected_cls0 = per_cell * 0.3
        expected_cls1 = per_cell * 0.7
        assert result.biomass[:9] == pytest.approx(expected_cls0)
        assert result.biomass[9:] == pytest.approx(expected_cls1)

    def test_abundance_consistent_with_biomass(self):
        """abundance == biomass / weight for every school."""
        bkg, ec, grid = _make_bkg_state()
        result = bkg.get_schools(step=0)
        # Avoid division-by-zero on zero-weight schools (none expected here)
        nonzero = result.weight > 0
        expected = result.biomass[nonzero] / result.weight[nonzero]
        assert result.abundance[nonzero] == pytest.approx(expected)

    def test_weight_from_allometry(self):
        """weight = condition_factor * length^allometric_power.

        c=0.00308, b=3.029, lengths=[10, 30]
        w0 = 0.00308 * 10^3.029
        w1 = 0.00308 * 30^3.029
        """
        bkg, ec, grid = _make_bkg_state()
        result = bkg.get_schools(step=0)
        c, b = 0.00308, 3.029
        w0 = c * (10.0**b)
        w1 = c * (30.0**b)
        # First 9 schools are class 0, next 9 are class 1
        assert result.weight[:9] == pytest.approx(w0)
        assert result.weight[9:] == pytest.approx(w1)

    def test_uniform_with_multiplier_and_offset(self):
        """per_cell = multiplier * (total / n_ocean + offset).

        multiplier=2.0, offset=10.0, total=1000, 9 cells
        per_cell = 2.0 * (1000/9 + 10.0)
        """
        extra = {
            "species.biomass.multiplier.sp10": "2.0",
            "species.biomass.offset.sp10": "10.0",
        }
        bkg, ec, grid = _make_bkg_state(extra_cfg=extra)
        result = bkg.get_schools(step=0)
        per_cell = 2.0 * (1000.0 / 9 + 10.0)
        # class 0 proportion=0.3
        expected_cls0 = per_cell * 0.3
        assert result.biomass[:9] == pytest.approx(expected_cls0)

    def test_land_cells_excluded(self):
        """Land cells are excluded: mask with 1 land cell → 8 ocean cells → 16 schools."""
        cfg = {**_make_base_config(), **_make_bkg_config(file_idx=10)}
        ec = EngineConfig.from_dict(cfg)
        mask = np.ones((3, 3), dtype=np.bool_)
        mask[1, 1] = False  # one land cell
        grid = Grid(ny=3, nx=3, ocean_mask=mask)
        bkg = BackgroundState(config=cfg, grid=grid, engine_config=ec)
        result = bkg.get_schools(step=0)
        # 2 classes * 8 ocean cells = 16
        assert len(result) == 16
        # No school should be at (y=1, x=1)
        land_mask = (result.cell_y == 1) & (result.cell_x == 1)
        assert not land_mask.any()


# ---------------------------------------------------------------------------
# Tests for BackgroundState NetCDF forcing
# ---------------------------------------------------------------------------


class TestBackgroundStateNetCDF:
    def _create_forcing_nc(self, tmp_path, ny=3, nx=3, n_steps=12):
        """Create a synthetic NetCDF forcing file and return its path."""
        data = np.random.default_rng(42).uniform(10, 100, size=(n_steps, ny, nx))
        ds = xr.Dataset(
            {"BkgSpecies": (("time", "latitude", "longitude"), data)},
            coords={
                "time": np.arange(n_steps),
                "latitude": np.linspace(45, 47, ny),
                "longitude": np.linspace(-5, -3, nx),
            },
        )
        path = tmp_path / "bkg_forcing.nc"
        ds.to_netcdf(path)
        return path

    def _make_netcdf_cfg(self, nc_path):
        """Build config for NetCDF-mode background species (no uniform total key)."""
        cfg = {**_make_base_config(), **_make_bkg_config(10)}
        # Remove uniform mode key → triggers NetCDF mode
        del cfg["species.biomass.total.sp10"]
        cfg["species.file.sp10"] = str(nc_path)
        cfg["species.biomass.nsteps.year.sp10"] = "12"
        return cfg

    def test_netcdf_loading(self, tmp_path):
        """Loading a NetCDF file produces positive biomass from get_schools."""
        nc_path = self._create_forcing_nc(tmp_path)
        cfg = self._make_netcdf_cfg(nc_path)
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        bkg = BackgroundState(config=cfg, grid=grid, engine_config=ec)

        result = bkg.get_schools(step=0)
        assert isinstance(result, SchoolState)
        # NetCDF data is random uniform [10, 100] — all biomass should be > 0
        assert (result.biomass > 0).all()

    def test_netcdf_temporal_mapping(self, tmp_path):
        """Steps 0 and 1 map to same forcing index when n_steps=12 and n_dt_per_year=24."""
        nc_path = self._create_forcing_nc(tmp_path, n_steps=12)
        cfg = self._make_netcdf_cfg(nc_path)
        # n_dt_per_year=24, forcing_nsteps_year=12 → step_in_year * 12 / 24
        # step 0 → int(0 * 12/24) = 0
        # step 1 → int(1 * 12/24) = 0   (same forcing index)
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        bkg = BackgroundState(config=cfg, grid=grid, engine_config=ec)

        result_step0 = bkg.get_schools(step=0)
        result_step1 = bkg.get_schools(step=1)
        np.testing.assert_array_equal(result_step0.biomass, result_step1.biomass)

    def test_netcdf_multiplier_applied(self, tmp_path):
        """multiplier=2.0 produces exactly 2× biomass compared to multiplier=1.0."""
        nc_path = self._create_forcing_nc(tmp_path)

        # Base config: multiplier=1.0 (default)
        cfg_base = self._make_netcdf_cfg(nc_path)
        ec_base = EngineConfig.from_dict(cfg_base)
        grid = Grid.from_dimensions(ny=3, nx=3)
        bkg_base = BackgroundState(config=cfg_base, grid=grid, engine_config=ec_base)
        result_base = bkg_base.get_schools(step=0)

        # Config with multiplier=2.0
        cfg_2x = {**cfg_base, "species.biomass.multiplier.sp10": "2.0"}
        ec_2x = EngineConfig.from_dict(cfg_2x)
        bkg_2x = BackgroundState(config=cfg_2x, grid=grid, engine_config=ec_2x)
        result_2x = bkg_2x.get_schools(step=0)

        np.testing.assert_allclose(result_2x.biomass, result_base.biomass * 2.0)
