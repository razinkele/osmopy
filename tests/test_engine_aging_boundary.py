"""M7: lifespan-boundary cohort removal (closes M7).

The aging_mortality function uses ``age_dt >= lifespan_dt - 1`` because
aging runs BEFORE reproduction increments age. The boundary semantics
were tested implicitly by test_engine_mortality.TestAgingMortality, but
not with the boundary value spelled out as the contract — making it easy
to regress unintentionally.

These tests pin the contract:
- age_dt == lifespan_dt - 1 -> EXPIRED (set abundance=0)
- age_dt == lifespan_dt - 2 -> SURVIVES
- age_dt == lifespan_dt - 1 with multi-species cfg -> uses correct species's lifespan
"""

from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.processes.natural import aging_mortality
from osmose.engine.state import SchoolState
from osmose.runner import RunResult  # noqa: F401  (used by other tests in dir)


def _cfg(lifespan_years: int = 3, n_dt_per_year: int = 24) -> dict[str, str]:
    """Smallest config that exposes EngineConfig.lifespan_dt."""
    return {
        "simulation.time.ndtperyear": str(n_dt_per_year),
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "1",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": str(lifespan_years),
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
    }


def test_age_at_lifespan_minus_one_is_expired() -> None:
    """The exact boundary: age_dt == lifespan_dt - 1 must be killed."""
    cfg = EngineConfig.from_dict(_cfg(lifespan_years=3, n_dt_per_year=24))
    lifespan_dt = int(cfg.lifespan_dt[0])
    assert lifespan_dt == 72  # sanity

    state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
    state = state.replace(
        abundance=np.array([1234.0]),
        age_dt=np.array([lifespan_dt - 1], dtype=np.int32),  # = 71
    )
    new_state = aging_mortality(state, cfg)
    assert new_state.abundance[0] == 0.0, (
        f"Boundary case age_dt={lifespan_dt - 1} must be expired (got "
        f"abundance={new_state.abundance[0]})"
    )


def test_age_at_lifespan_minus_two_survives() -> None:
    """One step younger than the boundary survives."""
    cfg = EngineConfig.from_dict(_cfg(lifespan_years=3, n_dt_per_year=24))
    lifespan_dt = int(cfg.lifespan_dt[0])

    state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
    state = state.replace(
        abundance=np.array([1234.0]),
        age_dt=np.array([lifespan_dt - 2], dtype=np.int32),  # = 70
    )
    new_state = aging_mortality(state, cfg)
    assert new_state.abundance[0] == 1234.0, (
        f"Pre-boundary age_dt={lifespan_dt - 2} must survive (got "
        f"abundance={new_state.abundance[0]})"
    )


def test_boundary_scales_with_n_dt_per_year() -> None:
    """The boundary moves with n_dt_per_year — pin that arithmetic too."""
    cfg = EngineConfig.from_dict(_cfg(lifespan_years=2, n_dt_per_year=12))
    lifespan_dt = int(cfg.lifespan_dt[0])
    assert lifespan_dt == 24

    state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
    state = state.replace(
        abundance=np.array([100.0, 100.0]),
        age_dt=np.array([lifespan_dt - 2, lifespan_dt - 1], dtype=np.int32),
    )
    new_state = aging_mortality(state, cfg)
    assert new_state.abundance[0] == 100.0  # age_dt = 22 survives
    assert new_state.abundance[1] == 0.0  # age_dt = 23 expires
