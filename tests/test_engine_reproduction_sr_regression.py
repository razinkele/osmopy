"""Regression: stock.recruitment.type=none reproduces the linear formula exactly."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.processes.reproduction import reproduction
from osmose.engine.state import SchoolState


def _make_cfg() -> EngineConfig:
    return EngineConfig.from_dict(
        {
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
    )


def _make_state(biomass_t: float, n_schools: int = 1) -> SchoolState:
    state = SchoolState.create(n_schools=n_schools, species_id=np.zeros(n_schools, dtype=np.int32))
    return state.replace(
        abundance=np.full(n_schools, 1000.0, dtype=np.float64),
        length=np.full(n_schools, 15.0, dtype=np.float64),
        weight=np.full(n_schools, biomass_t / 1000.0, dtype=np.float64),
        biomass=np.full(n_schools, biomass_t, dtype=np.float64),
        age_dt=np.full(n_schools, 24, dtype=np.int32),
    )


def test_type_none_matches_linear_formula_across_ssb_sweep():
    """For five SSB levels at step=0, type=none returns sex_ratio*fec*SSB*season*1e6 exactly."""
    cfg = _make_cfg()
    sex_ratio = 0.5
    rel_fec = 800.0
    season = 1.0 / cfg.n_dt_per_year  # uniform when no spawning_season CSV is configured

    biomasses_t = [10000.0, 50000.0, 100000.0, 250000.0, 1_000_000.0]
    for bm in biomasses_t:
        state = _make_state(bm)
        out = reproduction(state, cfg, step=0, rng=np.random.default_rng(42))
        eggs = out.abundance[out.is_egg].sum()
        # Reproduction uses weight*abundance for SSB (not the biomass field).
        # Recompute the same way for the manual expected value.
        ssb_t = (state.abundance * state.weight).sum()
        expected = sex_ratio * rel_fec * ssb_t * season * 1_000_000.0
        np.testing.assert_allclose(eggs, expected, rtol=1e-9)


def test_type_none_matches_linear_formula_across_step_phase():
    """Sample three steps within a year; the season factor must drop out of the byte-equality."""
    cfg = _make_cfg()
    sex_ratio = 0.5
    rel_fec = 800.0
    season = 1.0 / cfg.n_dt_per_year  # uniform; not phase-dependent without a spawning CSV
    state = _make_state(100_000.0)
    ssb_t = (state.abundance * state.weight).sum()
    expected = sex_ratio * rel_fec * ssb_t * season * 1_000_000.0

    for step in (0, 6, 11):
        out = reproduction(state, cfg, step=step, rng=np.random.default_rng(42))
        eggs = out.abundance[out.is_egg].sum()
        np.testing.assert_allclose(eggs, expected, rtol=1e-9)
