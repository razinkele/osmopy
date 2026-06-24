"""Egg-retention: predation must only see the RELEASED egg fraction, not the full
egg cohort. Drives _apply_predation_for_school directly with the verified harness
from tests/test_engine_functional_response.py:551. Deep review 2026-06-22 (HIGH-1)."""

from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.processes.mortality import _apply_predation_for_school
from osmose.engine.resources import ResourceState
from osmose.engine.state import SchoolState

_CFG = {
    "simulation.time.ndtperyear": "24",
    "simulation.time.nyear": "1",
    "simulation.nspecies": "2",
    "simulation.nschool.sp0": "1",
    "simulation.nschool.sp1": "1",
    "species.name.sp0": "Egg",
    "species.name.sp1": "Predator",
    "species.linf.sp0": "15.0",
    "species.linf.sp1": "50.0",
    "species.k.sp0": "0.5",
    "species.k.sp1": "0.2",
    "species.t0.sp0": "-0.1",
    "species.t0.sp1": "-0.1",
    "species.egg.size.sp0": "0.1",
    "species.egg.size.sp1": "0.1",
    "species.length2weight.condition.factor.sp0": "0.006",
    "species.length2weight.condition.factor.sp1": "0.006",
    "species.length2weight.allometric.power.sp0": "3.0",
    "species.length2weight.allometric.power.sp1": "3.0",
    "species.lifespan.sp0": "5",
    "species.lifespan.sp1": "10",
    "species.vonbertalanffy.threshold.age.sp0": "1.0",
    "species.vonbertalanffy.threshold.age.sp1": "1.0",
    "mortality.subdt": "10",
    "predation.ingestion.rate.max.sp0": "3.5",
    "predation.ingestion.rate.max.sp1": "3.5",
    "predation.efficiency.critical.sp0": "0.57",
    "predation.efficiency.critical.sp1": "0.57",
    # NOTE: the parser keys are ALL-LOWERCASE (config.py:646-647); camelCase
    # sizeRatio keys are silently ignored -> defaults. Use lowercase + the real
    # operating window so the test exercises the guard, not a default fallback.
    "predation.predprey.sizeratio.min.sp0": "1.0",
    "predation.predprey.sizeratio.min.sp1": "1.0",
    "predation.predprey.sizeratio.max.sp0": "3.5",
    "predation.predprey.sizeratio.max.sp1": "3.5",
    "mortality.additional.rate.sp0": "0.0",
    "mortality.additional.rate.sp1": "0.0",
    "mortality.starvation.rate.max.sp0": "0.0",
    "mortality.starvation.rate.max.sp1": "0.0",
    "simulation.fishing.mortality.enabled": "false",
}


def _eaten_eggs(egg_retained_frac: float) -> float:
    n_subdt = 10
    cfg = EngineConfig.from_dict(dict(_CFG))
    grid = Grid.from_dimensions(ny=1, nx=1)
    rs = ResourceState(config=cfg.raw_config, grid=grid)
    state = SchoolState.create(n_schools=2, species_id=np.array([1, 0], dtype=np.int32))
    pred_w = 0.006 * 30**3 * 1e-6
    prey_w = 0.006 * 10**3 * 1e-6
    pred_abundance = 100.0
    pred_biomass = pred_abundance * pred_w
    max_eatable = pred_biomass * 3.5 / (24 * n_subdt)
    prey_abundance = (2.0 * max_eatable) / prey_w  # r=2: prey plentiful, predator appetite-bound
    state = state.replace(
        abundance=np.array([pred_abundance, prey_abundance]),
        length=np.array([30.0, 10.0]),  # predator/prey length ratio 3.0, within [1.0, 3.5)
        weight=np.array([pred_w, prey_w]),
        biomass=np.array([pred_biomass, prey_abundance * prey_w]),
        age_dt=np.array([48, 24], dtype=np.int32),
        cell_x=np.array([0, 0], dtype=np.int32),
        cell_y=np.array([0, 0], dtype=np.int32),
        feeding_stage=np.array([0, 0], dtype=np.int32),
        is_egg=np.array([False, True]),
        egg_retained=np.array([0.0, egg_retained_frac * prey_abundance]),
    )
    rng = np.random.default_rng(42)
    cell_indices = np.array([0, 1], dtype=np.int32)
    _apply_predation_for_school(
        0,
        cell_indices,
        state,
        cfg,
        rs,
        0,
        0,
        rng,
        n_subdt,
        None,
        False,
        False,
        None,
        None,
        inst_abd=state.abundance.copy(),
    )
    return float(state.preyed_biomass[0])  # eaten biomass by the predator


def test_fully_retained_eggs_are_not_eaten():
    # egg_retained == full abundance -> eatable (inst_abd - egg_retained) == 0.
    assert _eaten_eggs(egg_retained_frac=1.0) == 0.0


def test_released_eggs_are_eaten():
    # Baseline: with nothing retained the predator eats eggs (proves the harness bites).
    assert _eaten_eggs(egg_retained_frac=0.0) > 0.0
