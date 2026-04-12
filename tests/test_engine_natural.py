"""Tests for out_mortality — natural.py targeted regression coverage.

Deep review v3 I-7: pin the is_out=True branch of out_mortality.
"""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.processes.natural import out_mortality
from osmose.engine.state import MortalityCause, SchoolState


def test_out_mortality_applies_rate_when_is_out_true():
    """A school with is_out=True must lose abundance at out_mortality_rate / n_dt_per_year.

    Deep review v3 I-7: the is_out=True branch of out_mortality had no coverage.
    """
    cfg_dict = {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "1",
        "species.name.sp0": "Migrator",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "1",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "mortality.out.rate.sp0": "2.4",
    }
    cfg = EngineConfig.from_dict(cfg_dict)

    state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
    initial_abundance = 1000.0
    state = state.replace(
        abundance=np.array([initial_abundance]),
        is_out=np.array([True]),
        weight=np.array([0.01]),
        biomass=np.array([10.0]),
        length=np.array([20.0]),
        age_dt=np.array([10], dtype=np.int32),
        first_feeding_age_dt=np.array([0], dtype=np.int32),
    )

    result = out_mortality(state, cfg)

    expected_survival = np.exp(-2.4 / 24)
    expected_abundance = initial_abundance * expected_survival
    np.testing.assert_allclose(
        result.abundance[0],
        expected_abundance,
        rtol=1e-10,
        err_msg="out_mortality did not apply the expected M_out/n_dt_per_year rate",
    )
    # n_dead should record the lost individuals under MortalityCause.OUT
    expected_dead = initial_abundance * (1 - expected_survival)
    np.testing.assert_allclose(
        result.n_dead[0, MortalityCause.OUT],
        expected_dead,
        rtol=1e-10,
        err_msg="out_mortality did not record dead individuals in n_dead[OUT]",
    )
