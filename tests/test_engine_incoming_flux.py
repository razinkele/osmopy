"""Tests for the incoming flux (external biomass injection) process."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.incoming_flux import (
    IncomingFluxState,
    _age_to_length,
    _length_to_age,
    _parse_flux_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_CONFIG: dict[str, str] = {
    "simulation.time.ndtperyear": "24",
    "simulation.time.nyear": "1",
    "simulation.nspecies": "1",
    "simulation.nschool.sp0": "10",
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


def _write_csv(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content).strip() + "\n")
    return p


def _make_age_csv(tmp_path: Path, n_rows: int = 3) -> Path:
    """Create a 2-class byAge CSV: boundaries 0;1 (years), n_rows of data."""
    header = "0;1"
    rows = [f"{100.0 * (r + 1)};{200.0 * (r + 1)}" for r in range(n_rows)]
    return _write_csv(tmp_path, "flux_age.csv", header + "\n" + "\n".join(rows))


def _make_size_csv(tmp_path: Path, n_rows: int = 3) -> Path:
    """Create a 2-class bySize CSV: boundaries 5;10 (cm), n_rows of data."""
    header = "5;10"
    rows = [f"{50.0 * (r + 1)};{80.0 * (r + 1)}" for r in range(n_rows)]
    return _write_csv(tmp_path, "flux_size.csv", header + "\n" + "\n".join(rows))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConfigDisabled:
    def test_disabled_by_default(self):
        """IncomingFluxState with no enabled key returns None."""
        cfg = dict(_BASE_CONFIG)
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        flux = IncomingFluxState(config=cfg, engine_config=ec, grid=grid)
        assert not flux.enabled
        assert flux.get_incoming_schools(0, np.random.default_rng(0)) is None

    def test_disabled_explicit(self):
        cfg = dict(_BASE_CONFIG)
        cfg["simulation.incoming.flux.enabled"] = "false"
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        flux = IncomingFluxState(config=cfg, engine_config=ec, grid=grid)
        assert not flux.enabled


class TestCsvParsing:
    def test_parse_age_csv(self, tmp_path):
        """Parse a 2-class age CSV with 3 timesteps."""
        csv_path = _make_age_csv(tmp_path, n_rows=3)
        boundaries, data = _parse_flux_csv(csv_path)
        assert len(boundaries) == 2
        np.testing.assert_array_equal(boundaries, [0.0, 1.0])
        assert data.shape == (3, 2)
        assert data[0, 0] == 100.0
        assert data[2, 1] == 600.0

    def test_parse_size_csv(self, tmp_path):
        csv_path = _make_size_csv(tmp_path, n_rows=2)
        boundaries, data = _parse_flux_csv(csv_path)
        assert len(boundaries) == 2
        np.testing.assert_array_equal(boundaries, [5.0, 10.0])
        assert data.shape == (2, 2)


class TestVonBertalanffy:
    def test_age_to_length(self):
        """VB: age=1yr with Linf=20, k=0.3, t0=-0.1 -> known length."""
        length = _age_to_length(1.0, linf=20.0, k=0.3, t0=-0.1)
        expected = 20.0 * (1.0 - np.exp(-0.3 * (1.0 - (-0.1))))
        np.testing.assert_allclose(length, expected, rtol=1e-10)

    def test_length_to_age(self):
        """Inverse VB: roundtrip age -> length -> age."""
        age_orig = 2.0
        length = _age_to_length(age_orig, linf=20.0, k=0.3, t0=-0.1)
        age_back = _length_to_age(length, linf=20.0, k=0.3, t0=-0.1)
        np.testing.assert_allclose(age_back, age_orig, rtol=1e-10)

    def test_length_to_age_clamps(self):
        """Inverse VB clamps when length >= Linf."""
        age = _length_to_age(20.0, linf=20.0, k=0.3, t0=-0.1)
        # Should not raise; uses ratio = 0.999
        assert np.isfinite(age)


class TestByAgeMidpoints:
    def test_midpoints_and_length(self, tmp_path):
        """byAge: midpoint of class [0, 1] = 0.5yr, last class [1, lifespan=3] = 2.0yr."""
        csv_path = _make_age_csv(tmp_path)
        cfg = dict(_BASE_CONFIG)
        cfg["simulation.incoming.flux.enabled"] = "true"
        cfg["flux.incoming.bydt.byage.file.sp0"] = str(csv_path)
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        flux = IncomingFluxState(config=cfg, engine_config=ec, grid=grid)

        assert len(flux._fluxes) == 1
        sf = flux._fluxes[0]

        # Class 0: midpoint(0, 1) = 0.5
        expected_age0 = 0.5
        expected_len0 = _age_to_length(expected_age0, 20.0, 0.3, -0.1)
        np.testing.assert_allclose(sf.lengths[0], expected_len0, rtol=1e-10)

        # Class 1: midpoint(1, lifespan=3) = 2.0
        expected_age1 = 2.0
        expected_len1 = _age_to_length(expected_age1, 20.0, 0.3, -0.1)
        np.testing.assert_allclose(sf.lengths[1], expected_len1, rtol=1e-10)

        # age_dt
        assert sf.ages_dt[0] == round(0.5 * 24)  # 12
        assert sf.ages_dt[1] == round(2.0 * 24)  # 48


class TestBySizeMidpoints:
    def test_midpoints_and_age(self, tmp_path):
        """bySize: midpoint of class [5, 10] = 7.5cm, last class [10, Linf=20] = 15cm."""
        csv_path = _make_size_csv(tmp_path)
        cfg = dict(_BASE_CONFIG)
        cfg["simulation.incoming.flux.enabled"] = "true"
        cfg["flux.incoming.bydt.bysize.file.sp0"] = str(csv_path)
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        flux = IncomingFluxState(config=cfg, engine_config=ec, grid=grid)

        assert len(flux._fluxes) == 1
        sf = flux._fluxes[0]

        # Class 0: midpoint(5, 10) = 7.5
        np.testing.assert_allclose(sf.lengths[0], 7.5, rtol=1e-10)

        # Class 1: midpoint(10, Linf=20) = 15.0
        np.testing.assert_allclose(sf.lengths[1], 15.0, rtol=1e-10)

        # age_dt should be inverse VB
        expected_age0 = _length_to_age(7.5, 20.0, 0.3, -0.1)
        assert sf.ages_dt[0] == round(expected_age0 * 24)


class TestSchoolCreation:
    def test_correct_abundance_weight_length(self, tmp_path):
        """Schools have correct abundance = biomass*1e6/weight."""
        csv_path = _write_csv(
            tmp_path, "flux.csv", "0;1\n100.0;0.0"
        )  # 100 tonnes in class 0, 0 in class 1
        cfg = dict(_BASE_CONFIG)
        cfg["simulation.incoming.flux.enabled"] = "true"
        cfg["flux.incoming.bydt.byage.file.sp0"] = str(csv_path)
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        flux = IncomingFluxState(config=cfg, engine_config=ec, grid=grid)
        rng = np.random.default_rng(42)

        schools = flux.get_incoming_schools(0, rng)
        assert schools is not None

        sf = flux._fluxes[0]
        expected_weight = sf.weights[0]
        expected_abund_total = 100.0 / expected_weight  # both in tonnes

        # Should be split into n_schools (10) schools
        n_schools = int(ec.n_schools[0])
        assert len(schools) == n_schools
        np.testing.assert_allclose(
            schools.abundance.sum(), expected_abund_total, rtol=1e-10
        )
        np.testing.assert_allclose(schools.biomass.sum(), 100.0, rtol=1e-10)
        assert (schools.length == sf.lengths[0]).all()
        assert (schools.weight == expected_weight).all()
        assert (schools.species_id == 0).all()
        assert not schools.is_egg.any()


class TestTimestepLooping:
    def test_wraps_around(self, tmp_path):
        """step >= n_rows wraps via modulo."""
        csv_path = _write_csv(tmp_path, "flux.csv", "0\n10.0\n20.0")
        cfg = dict(_BASE_CONFIG)
        cfg["simulation.incoming.flux.enabled"] = "true"
        cfg["flux.incoming.bydt.byage.file.sp0"] = str(csv_path)
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        flux = IncomingFluxState(config=cfg, engine_config=ec, grid=grid)
        rng = np.random.default_rng(42)

        # step=0 -> row 0 (10.0 tonnes), step=2 -> row 0 again (wrap)
        s0 = flux.get_incoming_schools(0, rng)
        s2 = flux.get_incoming_schools(2, rng)
        assert s0 is not None
        assert s2 is not None
        np.testing.assert_allclose(s0.biomass.sum(), s2.biomass.sum(), rtol=1e-10)


class TestZeroBiomassSkipped:
    def test_zero_class_skipped(self, tmp_path):
        """Classes with 0 biomass produce no schools."""
        csv_path = _write_csv(tmp_path, "flux.csv", "0;1\n0.0;0.0")
        cfg = dict(_BASE_CONFIG)
        cfg["simulation.incoming.flux.enabled"] = "true"
        cfg["flux.incoming.bydt.byage.file.sp0"] = str(csv_path)
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        flux = IncomingFluxState(config=cfg, engine_config=ec, grid=grid)
        rng = np.random.default_rng(42)

        schools = flux.get_incoming_schools(0, rng)
        assert schools is None


class TestOceanCellPlacement:
    def test_schools_placed_on_ocean_cells(self, tmp_path):
        """Schools should be placed only on ocean cells."""
        csv_path = _write_csv(tmp_path, "flux.csv", "0\n1000.0")
        cfg = dict(_BASE_CONFIG)
        cfg["simulation.incoming.flux.enabled"] = "true"
        cfg["flux.incoming.bydt.byage.file.sp0"] = str(csv_path)
        ec = EngineConfig.from_dict(cfg)

        # Grid with some land cells
        mask = np.zeros((4, 4), dtype=np.bool_)
        mask[0, 0] = True  # only one ocean cell
        mask[1, 2] = True
        grid = Grid(ny=4, nx=4, ocean_mask=mask)

        flux = IncomingFluxState(config=cfg, engine_config=ec, grid=grid)
        rng = np.random.default_rng(42)
        schools = flux.get_incoming_schools(0, rng)
        assert schools is not None

        # All schools should be on ocean cells
        for i in range(len(schools)):
            y, x = int(schools.cell_y[i]), int(schools.cell_x[i])
            assert mask[y, x], f"School {i} placed on land at ({y}, {x})"


class TestIntegration:
    def test_simulate_with_flux(self, tmp_path):
        """Full simulation with incoming flux produces non-zero biomass."""
        csv_path = _write_csv(tmp_path, "flux.csv", "0;1\n500.0;300.0")
        cfg = dict(_BASE_CONFIG)
        cfg["simulation.incoming.flux.enabled"] = "true"
        cfg["flux.incoming.bydt.byage.file.sp0"] = str(csv_path)
        cfg["population.seeding.biomass.sp0"] = "0.0"  # no seeding

        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)

        from osmose.engine.simulate import simulate

        outputs = simulate(ec, grid, rng)
        assert len(outputs) == 24

        # At least one timestep should have positive biomass from flux
        total_biomass = sum(o.biomass.sum() for o in outputs)
        assert total_biomass > 0, "Incoming flux should inject biomass into the simulation"


class TestMutuallyExclusive:
    def test_both_age_and_size_raises(self, tmp_path):
        """Having both byAge and bySize files for same species raises ValueError."""
        age_csv = _make_age_csv(tmp_path)
        size_csv = _make_size_csv(tmp_path)
        cfg = dict(_BASE_CONFIG)
        cfg["simulation.incoming.flux.enabled"] = "true"
        cfg["flux.incoming.bydt.byage.file.sp0"] = str(age_csv)
        cfg["flux.incoming.bydt.bysize.file.sp0"] = str(size_csv)
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)

        with pytest.raises(ValueError, match="both byAge and bySize"):
            IncomingFluxState(config=cfg, engine_config=ec, grid=grid)
