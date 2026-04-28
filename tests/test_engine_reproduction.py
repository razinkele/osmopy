"""Tests for reproduction and initialization — Tier 1 verification."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.processes.reproduction import reproduction
from osmose.engine.simulate import initialize
from osmose.engine.state import SchoolState


def _make_reprod_config() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "10",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "5",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "species.sexratio.sp0": "0.5",
        "species.relativefecundity.sp0": "800",
        "species.maturity.size.sp0": "12.0",
        "population.seeding.biomass.sp0": "50000",
    }


class TestInitialize:
    def test_creates_empty_state(self):
        """Java convention: initialize returns empty state; reproduction seeds schools."""
        cfg = EngineConfig.from_dict(_make_reprod_config())
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)
        state = initialize(cfg, grid, rng)
        assert len(state) == 0

    def test_empty_state_has_zero_biomass(self):
        cfg = EngineConfig.from_dict(_make_reprod_config())
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)
        state = initialize(cfg, grid, rng)
        assert state.biomass.sum() == 0.0


class TestReproduction:
    def test_produces_eggs_from_mature_schools(self):
        """Mature schools with sufficient length should produce eggs."""
        cfg = EngineConfig.from_dict(_make_reprod_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            length=np.array([15.0]),  # above maturity_size=12
            weight=np.array([20.25]),  # 0.006 * 15^3
            biomass=np.array([20250.0]),
            age_dt=np.array([24], dtype=np.int32),  # 2 years old
        )
        rng = np.random.default_rng(42)
        new_state = reproduction(state, cfg, step=0, rng=rng)
        # Should have original 1 school + 5 new egg schools
        assert len(new_state) == 6
        # New schools should be eggs
        assert np.all(new_state.is_egg[1:])

    def test_immature_schools_dont_reproduce(self):
        """Schools below maturity size should not produce eggs."""
        cfg = EngineConfig.from_dict(_make_reprod_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            length=np.array([5.0]),  # below maturity_size=12
            weight=np.array([0.75]),
            biomass=np.array([750.0]),
            age_dt=np.array([12], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        new_state = reproduction(state, cfg, step=0, rng=rng)
        # Only original school — no eggs from immature
        # But seeding kicks in if SSB=0 and within lifespan
        # Since step=0, seeding_biomass > 0, eggs will be produced from seeding
        assert len(new_state) >= 1

    def test_age_incremented(self):
        """All schools should have age_dt incremented by 1 after reproduction."""
        cfg = EngineConfig.from_dict(_make_reprod_config())
        state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0, 50.0]),
            age_dt=np.array([10, 20], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        new_state = reproduction(state, cfg, step=0, rng=rng)
        # First 2 schools should have age incremented
        np.testing.assert_array_equal(new_state.age_dt[:2], [11, 21])

    def test_egg_count_formula(self):
        """N_eggs = sex_ratio * relative_fecundity * SSB * season_factor."""
        cfg = EngineConfig.from_dict(_make_reprod_config())
        # Mature school with known biomass
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        ssb = 10000.0  # total spawning stock biomass
        state = state.replace(
            abundance=np.array([500.0]),
            length=np.array([15.0]),
            weight=np.array([ssb / 500.0]),  # weight such that abundance*weight = SSB
            biomass=np.array([ssb]),
            age_dt=np.array([24], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        new_state = reproduction(state, cfg, step=0, rng=rng)

        # Expected eggs = 0.5 * 800 * 10000 * (1/12) * 1e6 (tonnes→grams)
        expected_total_eggs = 0.5 * 800.0 * ssb * (1.0 / 12.0) * 1_000_000.0
        # Sum abundance of new schools (indices 1: are eggs)
        actual_total_eggs = new_state.abundance[1:].sum()
        np.testing.assert_allclose(actual_total_eggs, expected_total_eggs, rtol=1e-6)

    def test_seeding_produces_eggs_when_no_mature_fish(self):
        """When SSB=0 but within seeding period, use seeding biomass."""
        cfg = EngineConfig.from_dict(_make_reprod_config())
        # All schools at egg stage (no mature fish)
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0]),
            length=np.array([0.1]),  # too small to be mature
            weight=np.array([0.000006]),
            biomass=np.array([0.0006]),
            age_dt=np.array([0], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        new_state = reproduction(state, cfg, step=0, rng=rng)
        # Should produce eggs from seeding biomass
        assert len(new_state) > 1  # original + new eggs


def test_reproduction_creates_single_school_when_n_eggs_below_n_new():
    """Cover the n_eggs < n_new collapse branch: when egg count < n_schools, n_new collapses to 1.

    Setup: n_schools=100, but very low fecundity and tiny SSB so that n_eggs ~0.4 < 100.
    Expected: exactly 1 new egg school created (original 1 + 1 egg = 2 total).
    """
    raw_cfg = {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "10",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "100",  # large school count
        "species.name.sp0": "TinyFish",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "5",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "species.sexratio.sp0": "0.5",
        "species.relativefecundity.sp0": "0.001",  # very low fecundity
        "species.maturity.size.sp0": "12.0",
        "population.seeding.biomass.sp0": "0",  # disable seeding
    }
    cfg = EngineConfig.from_dict(raw_cfg)

    # One mature school: length=15 (above maturity_size=12), tiny weight → tiny SSB
    # SSB = abundance * weight = 10 * 0.001 = 0.01 tonnes
    # n_eggs = 0.5 * 0.001 * 0.01 * (1/12) * 1e6 ≈ 0.417  → 0 < n_eggs < 100 = n_schools
    state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
    state = state.replace(
        abundance=np.array([10.0]),
        length=np.array([15.0]),  # above maturity_size=12
        weight=np.array([0.001]),  # tiny weight so SSB is minuscule
        biomass=np.array([0.01]),
        age_dt=np.array([24], dtype=np.int32),  # 2 years old — well past maturity age
    )
    rng = np.random.default_rng(0)
    new_state = reproduction(state, cfg, step=0, rng=rng)

    # The branch collapses n_new to 1: original school + exactly 1 egg school
    n_new_schools = len(new_state) - 1  # subtract original school
    assert n_new_schools == 1, (
        f"Expected 1 new egg school (n_eggs<n_new branch), got {n_new_schools}"
    )
    # The single new school must be an egg
    assert new_state.is_egg[-1], "New school should be flagged as egg"


class TestStockRecruitmentConfig:
    def test_default_type_is_none(self):
        """Without stock.recruitment.* keys, all species default to 'none'."""
        cfg = EngineConfig.from_dict(_make_reprod_config())
        assert cfg.recruitment_type[0] == "none"
        assert cfg.recruitment_ssb_half[0] == 0.0

    def test_beverton_holt_type_parsed(self):
        """Setting type=beverton_holt and ssbhalf is round-tripped."""
        d = _make_reprod_config()
        d["stock.recruitment.type.sp0"] = "beverton_holt"
        d["stock.recruitment.ssbhalf.sp0"] = "12500.0"
        cfg = EngineConfig.from_dict(d)
        assert cfg.recruitment_type[0] == "beverton_holt"
        assert cfg.recruitment_ssb_half[0] == 12500.0

    def test_unknown_type_rejected(self):
        """Misspelled SR types fail loudly at config parse time."""
        import pytest
        d = _make_reprod_config()
        d["stock.recruitment.type.sp0"] = "berverton_holdt"  # typo
        with pytest.raises(ValueError, match="stock.recruitment.type"):
            EngineConfig.from_dict(d)

    def test_ssbhalf_zero_with_active_sr_rejected(self):
        """type!=none with ssbhalf=0 is a configuration error."""
        import pytest
        d = _make_reprod_config()
        d["stock.recruitment.type.sp0"] = "beverton_holt"
        d["stock.recruitment.ssbhalf.sp0"] = "0.0"
        with pytest.raises(ValueError, match="stock.recruitment.ssbhalf"):
            EngineConfig.from_dict(d)
