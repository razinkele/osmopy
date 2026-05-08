"""H8 (final extension): NaN/Inf propagation through fishing_mortality.

Companion to:
- tests/test_numerical_propagation.py (mortality processes)
- tests/test_numerical_propagation_extensions.py (reproduction / accessibility / starvation)

The fishing process has more configuration surface (per-fishery selectivity,
seasonal modulation, MPA, discards) than the simpler mortality kernels —
so this file uses the existing test_engine_fishing.py scaffold to spin up
a minimal fishing-enabled config and probe each input column for NaN
behaviour.
"""

from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.processes.fishing import fishing_mortality
from osmose.engine.state import MortalityCause, SchoolState


def _cfg() -> EngineConfig:
    """Minimal fishing-enabled config (mirrors test_engine_fishing.py)."""
    return EngineConfig.from_dict(
        {
            "simulation.time.ndtperyear": "24",
            "simulation.time.nyear": "1",
            "simulation.nspecies": "1",
            "simulation.nschool.sp0": "5",
            "species.name.sp0": "TestFish",
            "species.linf.sp0": "20.0",
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
            "simulation.fishing.mortality.enabled": "true",
            "fishing.rate.sp0": "0.5",
        }
    )


def test_fishing_propagates_nan_in_abundance() -> None:
    """A NaN in state.abundance flows through into n_dead and new_abundance.

    The fishing kernel computes `n_dead = state.abundance * (1 - exp(-D))`
    — any NaN in abundance hits the multiplication and propagates. A
    clean parallel school is unaffected.
    """
    cfg = _cfg()
    state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
    state = state.replace(
        abundance=np.array([np.nan, 1000.0]),
        weight=np.array([6.0, 6.0]),
        age_dt=np.array([24, 24], dtype=np.int32),
    )
    new = fishing_mortality(state, cfg, n_subdt=10)
    assert np.isnan(new.abundance[0]), "NaN should propagate through corrupted school"
    assert np.isfinite(new.abundance[1]), "clean parallel school should remain finite"


def test_fishing_propagates_nan_in_weight_into_yield() -> None:
    """Yield is computed as `n_dead * weight` — NaN weight propagates into yield."""
    cfg = _cfg()
    state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
    state = state.replace(
        abundance=np.array([1000.0]),
        weight=np.array([np.nan]),  # corrupted weight
        age_dt=np.array([24], dtype=np.int32),
    )
    new = fishing_mortality(state, cfg, n_subdt=10)
    # Mortality computation finishes (n_dead is finite) but biomass derivation hits NaN.
    assert np.isfinite(new.n_dead[0, MortalityCause.FISHING])
    assert np.isnan(new.biomass[0])


def test_fishing_clean_input_finite_output() -> None:
    """Sanity floor: clean input never produces NaN or Inf."""
    cfg = _cfg()
    state = SchoolState.create(n_schools=3, species_id=np.zeros(3, dtype=np.int32))
    state = state.replace(
        abundance=np.array([1000.0, 500.0, 2000.0]),
        weight=np.array([6.0, 5.0, 7.0]),
        age_dt=np.array([24, 36, 48], dtype=np.int32),
    )
    new = fishing_mortality(state, cfg, n_subdt=10)
    assert np.isfinite(new.abundance).all()
    assert np.isfinite(new.biomass).all()
    assert np.isfinite(new.n_dead[:, MortalityCause.FISHING]).all()


def test_fishing_disabled_returns_state_unchanged_even_with_nan() -> None:
    """When fishing is disabled, the function early-returns — NaN in any
    input column is irrelevant. Confirms the disabled-path isn't a hidden
    NaN amplifier."""
    cfg_dict = {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "20.0",
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
        "simulation.fishing.mortality.enabled": "false",
        "fishing.rate.sp0": "0.5",
    }
    cfg = EngineConfig.from_dict(cfg_dict)
    state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
    state = state.replace(
        abundance=np.array([np.nan]),
        weight=np.array([np.nan]),
        age_dt=np.array([24], dtype=np.int32),
    )
    new = fishing_mortality(state, cfg, n_subdt=10)
    # Disabled-path returns state unchanged — NaN is preserved as-is, no amplification.
    assert np.isnan(new.abundance[0])
    assert np.isnan(new.weight[0])
    # No fishing-mortality-cause was added (n_dead column zero).
    assert new.n_dead[0, MortalityCause.FISHING] == 0.0


def test_fishing_zero_abundance_no_nan_no_inf() -> None:
    """Edge case: a degenerate school with abundance=0 must not hit divide-by-zero
    or exp-overflow paths. A reasonable engine clamps n_dead = 0."""
    cfg = _cfg()
    state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
    state = state.replace(
        abundance=np.array([0.0]),
        weight=np.array([6.0]),
        age_dt=np.array([24], dtype=np.int32),
    )
    new = fishing_mortality(state, cfg, n_subdt=10)
    assert np.isfinite(new.abundance[0])
    assert new.n_dead[0, MortalityCause.FISHING] == 0.0
