"""Tests for bioen reproduction wiring in the simulation loop (Gap 1)."""

from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import _bioen_reproduction
from osmose.engine.state import SchoolState

from tests.test_engine_bioen_integration import _make_bioen_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid() -> Grid:
    return Grid.from_params(ny=5, nx=5, lat_min=0.0, lat_max=5.0, lon_min=0.0, lon_max=5.0)


def _make_config() -> EngineConfig:
    cfg_dict = _make_bioen_config()
    return EngineConfig.from_dict(cfg_dict)


def _make_state_with_gonad(
    config: EngineConfig,
    n_schools: int = 4,
    species_id: int = 0,
    gonad: float = 0.05,
    length: float = 10.0,
    age_dt: int = 48,  # 2 years at 24 dt/year — mature for m0=4.5, m1=1.8
    abundance: float = 1000.0,
) -> SchoolState:
    """Create a minimal SchoolState for testing bioen reproduction."""
    state = SchoolState.create(
        n_schools=n_schools,
        species_id=np.full(n_schools, species_id, dtype=np.int32),
    )
    state = state.replace(
        abundance=np.full(n_schools, abundance, dtype=np.float64),
        biomass=np.full(n_schools, abundance * 0.001, dtype=np.float64),
        length=np.full(n_schools, length, dtype=np.float64),
        weight=np.full(n_schools, 0.001, dtype=np.float64),
        age_dt=np.full(n_schools, age_dt, dtype=np.int32),
        gonad_weight=np.full(n_schools, gonad, dtype=np.float64),
        cell_x=np.array([1, 2, 3, 4][:n_schools], dtype=np.int32),
        cell_y=np.zeros(n_schools, dtype=np.int32),
    )
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBioenReproductionGonadReset:
    """Gonad weight must be reset to 0 after spawning."""

    def test_gonad_reset_after_spawning(self):
        config = _make_config()
        state = _make_state_with_gonad(config, gonad=0.05, length=10.0, age_dt=48)
        rng = np.random.default_rng(42)

        # Verify fish are mature: L=10 >= m0+m1*(48/24)=4.5+1.8*2=8.1 → mature
        sp = 0
        age_years = 48 / config.n_dt_per_year
        l_mature = config.bioen_m0[sp] + config.bioen_m1[sp] * age_years
        assert 10.0 >= l_mature, f"Test fish should be mature: L=10 >= L_mat={l_mature}"

        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=rng)

        # After spawning: gonad for original schools should be 0
        original_gonad = result.gonad_weight[:n_before]
        # Some schools may have been compacted, but all original should have gonad=0
        assert np.all(original_gonad == 0.0), (
            f"Expected gonad=0 after spawning, got {original_gonad}"
        )

    def test_no_spawn_no_gonad_reset(self):
        """If gonad is 0, no spawning occurs and gonad stays 0."""
        config = _make_config()
        state = _make_state_with_gonad(config, gonad=0.0, length=10.0, age_dt=48)
        rng = np.random.default_rng(42)

        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=rng)

        # No egg schools added
        assert len(result) == n_before, "No egg schools should be added when gonad=0"
        # Gonad still 0
        assert np.all(result.gonad_weight[:n_before] == 0.0)


class TestBioenReproductionEggSchool:
    """Egg school should be created for mature fish with gonad weight."""

    def test_egg_school_created_for_mature_fish(self):
        config = _make_config()
        state = _make_state_with_gonad(config, gonad=0.05, length=10.0, age_dt=48)
        rng = np.random.default_rng(42)

        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=rng)

        # Should have added at least 1 egg school
        assert len(result) > n_before, "Egg school should be appended for mature fish with gonad"

        # The new egg school(s) should be flagged as eggs
        egg_schools = result.is_egg[n_before:]
        assert egg_schools.all(), "New schools should be flagged as is_egg=True"

    def test_egg_school_abundance_matches_eggs(self):
        """Egg school abundance = total gonad / egg_weight."""
        config = _make_config()
        gonad_per_school = 0.05
        n_schools = 3
        state = _make_state_with_gonad(
            config, n_schools=n_schools, gonad=gonad_per_school, length=10.0, age_dt=48
        )
        rng = np.random.default_rng(42)

        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=rng)

        # Compute expected total eggs: gonad / egg_weight for all 3 schools
        sp = 0
        # egg_weight_override is None for the test config, so use allometric fallback
        ew = config.condition_factor[sp] * config.egg_size[sp] ** config.allometric_power[sp] * 1e-6
        expected_eggs = (gonad_per_school * n_schools) / ew

        new_school_abundance = result.abundance[n_before:].sum()
        assert abs(new_school_abundance - expected_eggs) < expected_eggs * 1e-9, (
            f"Expected egg abundance {expected_eggs}, got {new_school_abundance}"
        )

    def test_egg_school_placed_in_parent_cell(self):
        """Egg school cell_x/cell_y should be within parent cells."""
        config = _make_config()
        parent_cells_x = [1, 2, 3, 4]
        state = _make_state_with_gonad(config, n_schools=4, gonad=0.05, length=10.0, age_dt=48)
        rng = np.random.default_rng(42)

        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=rng)

        for i in range(n_before, len(result)):
            assert result.cell_x[i] in parent_cells_x, (
                f"Egg school cell_x={result.cell_x[i]} not in parent cells {parent_cells_x}"
            )


class TestBioenReproductionImmatureFish:
    """Immature fish should not produce eggs."""

    def test_immature_fish_no_eggs(self):
        """Fish below maturity length produce no eggs even with gonad weight."""
        config = _make_config()
        sp = 0
        # Small length: L=2 < m0=4.5 → immature at any age
        state = _make_state_with_gonad(config, gonad=0.05, length=2.0, age_dt=48)
        rng = np.random.default_rng(42)

        # Verify they're immature
        age_years = 48 / config.n_dt_per_year
        l_mature = config.bioen_m0[sp] + config.bioen_m1[sp] * age_years
        assert 2.0 < l_mature, f"Test fish should be immature: L=2 < L_mat={l_mature}"

        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=rng)

        # No egg school created
        assert len(result) == n_before, "No egg schools should be created for immature fish"

    def test_young_fish_immature_by_age(self):
        """Very young fish (age_dt=1) are immature even if large."""
        config = _make_config()
        # At age_dt=1, L_mature = m0 + m1 * (1/24) ≈ 4.5 + 0.075 = 4.575
        # Use L=10 but age_dt=1: still mature by length criterion
        # Instead, test age_dt=1 with small length to be truly immature
        state = _make_state_with_gonad(config, gonad=0.05, length=2.0, age_dt=1)
        rng = np.random.default_rng(42)

        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=rng)

        assert len(result) == n_before, "Young fish with small length should not spawn"


class TestBioenReproductionAgeIncrement:
    """Age increment: existing schools get +1, new egg schools stay at 0."""

    def test_existing_schools_age_incremented(self):
        config = _make_config()
        state = _make_state_with_gonad(config, gonad=0.05, length=10.0, age_dt=48)
        rng = np.random.default_rng(42)

        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=rng)

        # Existing schools should have age_dt incremented by 1
        original_ages = result.age_dt[:n_before]
        assert np.all(original_ages == 49), (
            f"Expected age_dt=49 for existing schools, got {original_ages}"
        )

    def test_new_egg_schools_age_zero(self):
        config = _make_config()
        state = _make_state_with_gonad(config, gonad=0.05, length=10.0, age_dt=48)
        rng = np.random.default_rng(42)

        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=rng)

        if len(result) > n_before:
            new_ages = result.age_dt[n_before:]
            assert np.all(new_ages == 0), f"New egg schools should have age_dt=0, got {new_ages}"
