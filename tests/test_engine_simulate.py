"""Tests for the simulation loop skeleton."""

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import StepOutput, simulate


@pytest.fixture
def minimal_config() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "10",
        "species.name.sp0": "TestFish",
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


class TestSimulate:
    def test_simulate_returns_outputs(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert len(outputs) == 12

    def test_step_output_has_biomass(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert isinstance(outputs[0], StepOutput)
        assert outputs[0].step == 0
        assert outputs[0].biomass.shape == (1,)

    def test_simulate_correct_step_count(self, minimal_config):
        minimal_config["simulation.time.nyear"] = "2"
        cfg = EngineConfig.from_dict(minimal_config)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert len(outputs) == 24
        assert outputs[-1].step == 23

    def test_aging_mortality_kills_old_schools(self):
        """Aging mortality should kill a school at lifespan - 1."""
        from osmose.engine.processes.natural import aging_mortality as _aging
        from osmose.engine.state import SchoolState

        cfg_dict = {
            "simulation.time.ndtperyear": "24",
            "simulation.time.nyear": "1",
            "simulation.nspecies": "1",
            "simulation.nschool.sp0": "1",
            "species.name.sp0": "TestFish",
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
        cfg = EngineConfig.from_dict(cfg_dict)
        # Create school at age 71 dt (lifespan=72, threshold=71)
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0]),
            age_dt=np.array([71], dtype=np.int32),
        )
        new_state = _aging(state, cfg)
        assert new_state.abundance[0] == 0.0

    def test_schools_move_during_simulation(self):
        cfg_dict = {
            "simulation.time.ndtperyear": "12",
            "simulation.time.nyear": "1",
            "simulation.nspecies": "1",
            "simulation.nschool.sp0": "10",
            "species.name.sp0": "TestFish",
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
            "movement.distribution.method.sp0": "random",
            "movement.randomwalk.range.sp0": "2",
            "population.seeding.biomass.sp0": "50000",
        }
        cfg = EngineConfig.from_dict(cfg_dict)
        grid = Grid.from_dimensions(ny=10, nx=10)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert len(outputs) == 12


def test_fishing_yield_not_double_scaled(minimal_config):
    """Regression: yield was 1e6 too small due to double 1e-6 conversion."""
    from osmose.engine.simulate import _collect_outputs
    from osmose.engine.state import SchoolState

    cfg = EngineConfig.from_dict(minimal_config)

    state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
    # weight=0.001 tonnes (= 1 kg), n_dead[FISHING]=100 -> yield = 100 * 0.001 = 0.1 tonnes
    state = state.replace(
        weight=np.array([0.001]),
        abundance=np.array([100.0]),
        n_dead=np.array([[0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0]]),
    )

    output = _collect_outputs(state, cfg, step=0)
    total_yield = output.yield_by_species.sum()
    assert total_yield > 0.01, f"Yield {total_yield} is suspiciously small (double 1e-6 bug?)"
    assert abs(total_yield - 0.1) < 1e-10, f"Expected 0.1 tonnes, got {total_yield}"


def test_age_distribution_uses_year_bins(minimal_config):
    """Regression: duplicate block used timestep bins instead of year bins."""
    cfg_dict = dict(minimal_config)
    cfg_dict["output.biomass.byage.enabled"] = "true"
    cfg_dict["output.abundance.byage.enabled"] = "true"

    cfg = EngineConfig.from_dict(cfg_dict)
    grid = Grid.from_dimensions(ny=3, nx=3)
    rng = np.random.default_rng(42)
    outputs = simulate(cfg, grid, rng)

    # With lifespan=3 years and year-based bins, we expect max_age_yr+1 = 4 bins
    # The duplicate block would produce lifespan_dt+1 = 37 bins (3*12+1)
    for out in outputs:
        if out.biomass_by_age is not None:
            n_bins = len(out.biomass_by_age[0])
            assert n_bins <= 5, (
                f"Got {n_bins} age bins — expected ~4 (year-based), "
                f"not ~37 (timestep-based). Duplicate block bug?"
            )
            break


def test_average_step_outputs_preserves_distributions():
    """Distribution dicts must not be silently dropped during averaging."""
    from osmose.engine.simulate import _average_step_outputs, StepOutput

    dist = {0: np.array([1.0, 2.0, 3.0])}
    so = StepOutput(
        step=0,
        biomass=np.array([100.0]),
        abundance=np.array([50.0]),
        mortality_by_cause=np.zeros((1, 6)),
        biomass_by_age=dist,
        abundance_by_age=dist,
        biomass_by_size=dist,
        abundance_by_size=dist,
    )
    result = _average_step_outputs([so], freq=1, record_step=0)
    assert result.biomass_by_age is not None
    assert result.abundance_by_age is not None
    assert result.biomass_by_size is not None
    assert result.abundance_by_size is not None


def test_simulate_output_step0_include(minimal_config):
    """output.step0.include=true prepends a step=-1 snapshot to the output list."""
    cfg_dict = dict(minimal_config)
    cfg_dict["output.step0.include"] = "true"
    # Pin record frequency explicitly so this test is not coupled to the default.
    cfg_dict["output.recordfrequency.ndt"] = "1"
    cfg = EngineConfig.from_dict(cfg_dict)
    grid = Grid.from_dimensions(ny=3, nx=3)
    rng = np.random.default_rng(42)

    outputs = simulate(cfg, grid, rng)

    # First element must be the step=-1 snapshot
    assert outputs[0].step == -1, (
        f"Expected first output to be step=-1 snapshot, got step={outputs[0].step}"
    )
    # With record_freq=1 and n_dt_per_year*n_years=12, the normal run produces 12
    # regular outputs on top of the snapshot — total 13.
    assert len(outputs) == 13, f"Expected 13 outputs (12 + step0), got {len(outputs)}"


def test_simulate_partial_flush_non_divisible_record_freq(minimal_config):
    """A record frequency that doesn't divide n_steps must still flush the tail."""
    cfg_dict = dict(minimal_config)
    # 12 total steps, recording every 7 -> 1 full window (steps 0-6) + 1 partial window (7-11)
    cfg_dict["output.recordfrequency.ndt"] = "7"
    cfg = EngineConfig.from_dict(cfg_dict)
    grid = Grid.from_dimensions(ny=3, nx=3)
    rng = np.random.default_rng(42)

    outputs = simulate(cfg, grid, rng)

    # 12 / 7 = 1 remainder 5 -> at least 2 outputs (1 full, 1 partial flush)
    assert len(outputs) >= 2, f"Expected ≥2 outputs from partial flush, got {len(outputs)}"
    # Regression guard: the old buggy behavior was to drop the partial entirely,
    # producing exactly 1 output. Assert strictly more than 1.
    assert len(outputs) > 1, "Partial flush regression: tail accumulation was dropped"
