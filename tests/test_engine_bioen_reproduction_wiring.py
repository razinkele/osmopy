"""Tests for bioen reproduction wiring in the simulation loop (Gap 1).

Rewritten for the Java-parity contract (task 5). What changed against the v1 assertions,
and why (all verified against `BioenReproductionProcess.java` 4.3.3):

* Gonad is DECREMENTED by `gonad*season`, not flushed to zero every spawning step.
* Eggs scale with school ABUNDANCE (`* school.getInstantaneousAbundance()`).
* `n_schools[sp]` egg schools are created, all UNLOCATED (`School.java:204-207` builds
  reproduction schools at x = y = -1) — not one school placed in a random parent cell.
* Seeding fires when no MATURE school exists (`SSB[cpt] == 0.`), not when the gonad sum
  happens to be zero.
* Egg schools are created at `computeLength(eggWeight)` (`Species.java:327`).
"""

from __future__ import annotations

import numpy as np
import pytest

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


def _make_config(**overrides: str) -> EngineConfig:
    cfg_dict = _make_bioen_config()
    cfg_dict.update(overrides)
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


def _season(config: EngineConfig, sp: int = 0, step: int = 0) -> float:
    if config.spawning_season is None:
        return 1.0 / config.n_dt_per_year
    return float(config.spawning_season[sp, step % config.spawning_season.shape[1]])


def _egg_weight(config: EngineConfig, sp: int = 0) -> float:
    if config.egg_weight_override is not None and not np.isnan(config.egg_weight_override[sp]):
        return float(config.egg_weight_override[sp])
    return float(
        config.condition_factor[sp] * config.egg_size[sp] ** config.allometric_power[sp] * 1e-6
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBioenReproductionGonadDecrement:
    """Java removes `gonad*season` from the gonad — a partial decrement, not a flush."""

    def test_gonad_decremented_by_season_fraction_not_zeroed(self):
        config = _make_config()
        gonad0 = 0.05
        state = _make_state_with_gonad(config, gonad=gonad0, length=10.0, age_dt=48)
        rng = np.random.default_rng(42)

        # Verify fish are mature: L=10 >= m0+m1*(48/24)=4.5+1.8*2=8.1 → mature
        sp = 0
        age_years = 48 / config.n_dt_per_year
        l_mature = config.bioen_m0[sp] + config.bioen_m1[sp] * age_years
        assert 10.0 >= l_mature, f"Test fish should be mature: L=10 >= L_mat={l_mature}"

        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=rng)

        season = _season(config, sp, 0)
        expected = gonad0 * (1.0 - season)
        original_gonad = result.gonad_weight[:n_before]
        assert np.allclose(original_gonad, expected), (
            f"Expected gonad={expected} after releasing a {season} share, got {original_gonad}"
        )
        # The parent pool must survive: create_offspring_genotypes selects parents by
        # gonad_weight > 0 and raises when a spawning species has none.
        assert np.all(original_gonad > 0.0)

    def test_no_spawn_no_gonad_change(self):
        """If gonad is 0, no spawning occurs and gonad stays 0."""
        config = _make_config()
        state = _make_state_with_gonad(config, gonad=0.0, length=10.0, age_dt=48)
        rng = np.random.default_rng(42)

        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=rng)

        assert len(result) == n_before, "No egg schools should be added when gonad=0"
        assert np.all(result.gonad_weight[:n_before] == 0.0)

    def test_only_mature_schools_lose_gonad(self):
        config = _make_config()
        state = _make_state_with_gonad(config, n_schools=4, gonad=0.05, age_dt=48)
        # Schools 2 and 3 are too short to be mature (m0 = 4.5).
        state = state.replace(length=np.array([10.0, 10.0, 2.0, 2.0]))
        rng = np.random.default_rng(0)

        result = _bioen_reproduction(state, config, step=0, rng=rng)
        season = _season(config)
        np.testing.assert_allclose(result.gonad_weight[:2], 0.05 * (1.0 - season))
        np.testing.assert_allclose(result.gonad_weight[2:4], 0.05)


class TestBioenReproductionEggSchools:
    """`create_reproduction_schools`: `nSchool` unlocated schools of `nEgg/nSchool`."""

    def test_n_schools_unlocated_egg_schools_created(self):
        config = _make_config()
        state = _make_state_with_gonad(config, gonad=0.05, length=10.0, age_dt=48)
        rng = np.random.default_rng(42)

        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=rng)

        n_new = len(result) - n_before
        assert n_new == int(config.n_schools[0]), (
            f"Expected {int(config.n_schools[0])} egg schools (Java nSchool), got {n_new}"
        )
        assert result.is_egg[n_before:].all(), "New schools should be flagged as is_egg=True"
        assert np.all(result.cell_x[n_before:] == -1), "Java lays reproduction schools unlocated"
        assert np.all(result.cell_y[n_before:] == -1)
        assert np.all(result.abundance[n_before:] == result.abundance[n_before])

    def test_egg_abundance_matches_java_formula(self):
        """total nEgg = sum_schools(gonad*season * sexRatio / eggWeight * N)."""
        config = _make_config()
        gonad_per_school = 0.05
        n_schools = 3
        abundance = 1000.0
        state = _make_state_with_gonad(
            config,
            n_schools=n_schools,
            gonad=gonad_per_school,
            length=10.0,
            age_dt=48,
            abundance=abundance,
        )
        rng = np.random.default_rng(42)

        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=rng)

        sp = 0
        season = _season(config, sp, 0)
        ew = _egg_weight(config, sp)
        w_egg = gonad_per_school * season
        expected = w_egg * float(config.sex_ratio[sp]) / ew * abundance * n_schools

        got = result.abundance[n_before:].sum()
        assert got == pytest.approx(expected, rel=1e-9), f"Expected {expected}, got {got}"

    def test_egg_count_scales_with_parent_abundance(self):
        """The headline v1 bug: eggs were independent of N."""
        config = _make_config()
        rng = np.random.default_rng(0)

        lo = _make_state_with_gonad(config, gonad=0.05, age_dt=48, abundance=1_000.0)
        hi = _make_state_with_gonad(config, gonad=0.05, age_dt=48, abundance=10_000.0)
        n_before = len(lo)
        out_lo = _bioen_reproduction(lo, config, step=0, rng=rng)
        out_hi = _bioen_reproduction(hi, config, step=0, rng=np.random.default_rng(0))

        eggs_lo = out_lo.abundance[n_before:].sum()
        eggs_hi = out_hi.abundance[n_before:].sum()
        assert eggs_hi / eggs_lo == pytest.approx(10.0)

    def test_egg_schools_carry_egg_weight_and_bioen_egg_length(self):
        """Weight = configured egg weight; length = computeLength(eggWeight)."""
        config = _make_config()
        state = _make_state_with_gonad(config, gonad=0.05, length=10.0, age_dt=48)
        rng = np.random.default_rng(1)

        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=rng)

        sp = 0
        ew = _egg_weight(config, sp)
        expected_len = (ew * 1e6 / float(config.condition_factor[sp])) ** (
            1.0 / float(config.allometric_power[sp])
        )
        assert np.all(result.weight[n_before:] == pytest.approx(ew))
        assert np.all(result.length[n_before:] == pytest.approx(expected_len))
        # With no egg-weight override the bioen length reduces to config.egg_size.
        assert expected_len == pytest.approx(float(config.egg_size[sp]))

    def test_egg_length_follows_egg_weight_override(self):
        """With an explicit egg weight, bioen length must follow it — NOT egg_size."""
        ew = 5.0e-9  # tonnes
        config = _make_config(**{"species.egg.weight.sp0": str(ew * 1e6)})
        if config.egg_weight_override is None or np.isnan(config.egg_weight_override[0]):
            pytest.skip("config reader does not expose species.egg.weight as an override here")
        state = _make_state_with_gonad(config, gonad=0.05, length=10.0, age_dt=48)
        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=np.random.default_rng(1))

        expected_len = (
            float(config.egg_weight_override[0]) * 1e6 / config.condition_factor[0]
        ) ** (1.0 / config.allometric_power[0])
        assert np.all(result.length[n_before:] == pytest.approx(expected_len))
        assert expected_len != pytest.approx(float(config.egg_size[0]))


class TestBioenReproductionSeeding:
    """Java seeds on SSB == 0 (no MATURE school), not on "the gonad sum is zero"."""

    def test_seeding_fires_when_no_mature_school(self):
        config = _make_config(**{"population.seeding.biomass.sp0": "100.0"})
        # length 2.0 < m0 = 4.5 -> nothing mature -> SSB == 0
        state = _make_state_with_gonad(config, gonad=0.0, length=2.0, age_dt=48)
        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=np.random.default_rng(2))

        n_new = len(result) - n_before
        assert n_new == int(config.n_schools[0])
        season = _season(config)
        expected = (
            float(config.sex_ratio[0]) * float(config.relative_fecundity[0]) * 100.0 * season * 1e6
        )
        assert result.abundance[n_before:].sum() == pytest.approx(expected, rel=1e-9)
        assert np.all(result.from_seeding[n_before:]), "seeded eggs must be tagged"

    def test_seeding_does_not_fire_when_mature_schools_exist_with_empty_gonads(self):
        """v1 keyed seeding off `gonad.sum() == 0`, so a healthy stock between spawning
        windows was re-seeded every step. Java keys it off SSB."""
        config = _make_config(**{"population.seeding.biomass.sp0": "100.0"})
        state = _make_state_with_gonad(config, gonad=0.0, length=10.0, age_dt=48)
        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=np.random.default_rng(2))
        assert len(result) == n_before, "mature schools present -> SSB != 0 -> no seeding"

    def test_empty_population_bootstraps(self):
        """A bioen run whose population starts empty must still seed every species: the
        gonad-release branch alone can never create the first cohort."""
        config = _make_config(
            **{
                "population.seeding.biomass.sp0": "100.0",
                "population.seeding.biomass.sp1": "50.0",
            }
        )
        state = SchoolState.create(n_schools=0, species_id=np.zeros(0, dtype=np.int32))
        result = _bioen_reproduction(state, config, step=0, rng=np.random.default_rng(0))

        counts = np.bincount(result.species_id, minlength=2)
        assert counts[0] == int(config.n_schools[0])
        assert counts[1] == int(config.n_schools[1])
        assert np.all(result.from_seeding)
        assert np.all(result.cell_x == -1)

    def test_egg_schools_are_excluded_from_ssb_when_m0_defaults_to_zero(self):
        """Covers the `~state.is_egg` guard in `_bioen_reproduction`'s species mask.

        `bioen_m0` falls back to 0.0 when `species.maturity.m0.sp{i}` is absent
        (`config.py:2509`). With m0 = m1 = 0 every school of positive length reads as
        mature — egg schools included. An egg entering SSB would keep SSB != 0 forever,
        so a collapsed stock could never be rescued by the seeding bootstrap.
        """
        cfg_dict = _make_bioen_config()
        for k in [k for k in cfg_dict if "maturity.m0" in k or "maturity.m1" in k]:
            del cfg_dict[k]
        cfg_dict["population.seeding.biomass.sp0"] = "100.0"
        config = EngineConfig.from_dict(cfg_dict)
        assert float(config.bioen_m0[0]) == 0.0
        assert float(config.bioen_m1[0]) == 0.0

        # The only school of species 0 is an egg -> the stock is collapsed, SSB == 0.
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1e6]),
            length=np.array([float(config.egg_size[0])]),
            weight=np.array([1e-11]),
            age_dt=np.array([0], dtype=np.int32),
            is_egg=np.array([True]),
            first_feeding_age_dt=np.array([1], dtype=np.int32),
        )
        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=np.random.default_rng(3))

        n_new = len(result) - n_before
        assert n_new == int(config.n_schools[0]), (
            "seeding must fire: the only school present is an egg, so SSB == 0 "
            f"(got {n_new} new schools — the egg was counted as a mature spawner)"
        )
        assert np.all(result.from_seeding[n_before:])

    def test_seeding_stops_after_the_seeding_window(self):
        config = _make_config(
            **{"population.seeding.biomass.sp0": "100.0", "population.seeding.year.max": "1"}
        )
        state = _make_state_with_gonad(config, gonad=0.0, length=2.0, age_dt=48)
        n_before = len(state)
        late = int(config.seeding_max_step[0]) + 1
        result = _bioen_reproduction(state, config, step=late, rng=np.random.default_rng(2))
        assert len(result) == n_before


class TestBioenReproductionMaturity:
    """LMRN maturity `L >= m0 + m1*age_years` (moved here from bioen_egg_production)."""

    def test_immature_fish_no_eggs(self):
        """Fish below maturity length produce no eggs even with gonad weight."""
        config = _make_config()
        sp = 0
        state = _make_state_with_gonad(config, gonad=0.05, length=2.0, age_dt=48)
        rng = np.random.default_rng(42)

        age_years = 48 / config.n_dt_per_year
        l_mature = config.bioen_m0[sp] + config.bioen_m1[sp] * age_years
        assert 2.0 < l_mature, f"Test fish should be immature: L=2 < L_mat={l_mature}"

        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=rng)

        assert len(result) == n_before, "No egg schools should be created for immature fish"

    def test_young_fish_immature_by_age(self):
        """Very young fish (age_dt=1) are immature even if large."""
        config = _make_config()
        state = _make_state_with_gonad(config, gonad=0.05, length=2.0, age_dt=1)
        rng = np.random.default_rng(42)

        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=rng)

        assert len(result) == n_before, "Young fish with small length should not spawn"

    def test_m1_slope_raises_the_threshold_with_age(self):
        """m1 > 0 makes older fish need to be longer. L=9 is mature at 2 yr
        (4.5 + 1.8*2 = 8.1) but not at 3 yr (4.5 + 1.8*3 = 9.9)."""
        config = _make_config()
        young = _make_state_with_gonad(config, gonad=0.05, length=9.0, age_dt=48)
        old = _make_state_with_gonad(config, gonad=0.05, length=9.0, age_dt=72)
        n_before = len(young)
        out_young = _bioen_reproduction(young, config, step=0, rng=np.random.default_rng(0))
        out_old = _bioen_reproduction(old, config, step=0, rng=np.random.default_rng(0))
        assert len(out_young) > n_before
        assert len(out_old) == n_before

    def test_per_school_m0_override_changes_who_spawns(self):
        """Genetic trait override: a per-school m0 array selects which schools mature.

        Moved from `test_genetics_bioen_integration.py::TestBioenReproductionOverride`,
        which tested this through the removed `bioen_egg_production`.
        """
        config = _make_config()
        state = _make_state_with_gonad(config, n_schools=3, gonad=0.05, length=12.0, age_dt=24)
        n_before = len(state)

        # Scalar (config) m0 = 4.5 -> all three mature.
        out_all = _bioen_reproduction(state, config, step=0, rng=np.random.default_rng(0))
        eggs_all = out_all.abundance[n_before:].sum()
        assert eggs_all > 0

        # Per-school m0: school 2 needs 20 cm and stays immature -> 2/3 of the eggs.
        overrides = {"bioen_m0": np.array([4.5, 4.5, 20.0])}
        out_two = _bioen_reproduction(
            state, config, step=0, rng=np.random.default_rng(0), trait_overrides=overrides
        )
        eggs_two = out_two.abundance[n_before:].sum()
        assert eggs_two == pytest.approx(eggs_all * 2.0 / 3.0, rel=1e-9)
        # The immature school keeps its gonad.
        assert out_two.gonad_weight[2] == pytest.approx(0.05)


class TestBioenReproductionAgeIncrement:
    """Age increment: existing schools get +1, new egg schools stay at 0."""

    def test_existing_schools_age_incremented(self):
        config = _make_config()
        state = _make_state_with_gonad(config, gonad=0.05, length=10.0, age_dt=48)
        rng = np.random.default_rng(42)

        n_before = len(state)
        result = _bioen_reproduction(state, config, step=0, rng=rng)

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

        assert len(result) > n_before
        new_ages = result.age_dt[n_before:]
        assert np.all(new_ages == 0), f"New egg schools should have age_dt=0, got {new_ages}"

    def test_eggs_hatch_when_age_reaches_first_feeding(self):
        """`is_egg` must clear, or starvation/feeding-stage/growth stay blocked forever."""
        config = _make_config()
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1e6]),
            weight=np.array([1e-11]),
            length=np.array([0.1]),
            age_dt=np.array([0], dtype=np.int32),
            is_egg=np.array([True]),
            first_feeding_age_dt=np.array([1], dtype=np.int32),
        )
        result = _bioen_reproduction(state, config, step=0, rng=np.random.default_rng(0))
        assert result.age_dt[0] == 1
        assert not result.is_egg[0]
