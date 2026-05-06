"""H8: NaN/Inf propagation through the engine's process functions.

Existing tests (`test_numerical_guards.py`, `test_numerical_edges.py`)
cover bioenergetic helpers (f_o2, phi_t) and growth (von Bertalanffy,
size-overlap). H8 from
`docs/plans/2026-05-05-deep-review-remediation-plan.md` extends the
audit to the mortality / aging / larva-mortality / out-mortality
processes that were not previously covered.

The contract these tests pin: when a NaN sneaks into an input array
(via a corrupted config, an upstream divide-by-zero, or a calibration
candidate at the boundary of the parameter space), the process either
(a) propagates the NaN cleanly (caller can detect and fail loudly), or
(b) is unaffected because the NaN is in a column the process does not
read.

What these tests do NOT assert: that a NaN is silently clamped to 0 or
ignored. The plan calls clamping a future enhancement; for now, the
audit just documents the current behaviour.
"""

from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.processes.natural import (
    additional_mortality,
    aging_mortality,
    larva_mortality,
    out_mortality,
)
from osmose.engine.state import SchoolState


def _cfg(lifespan_years: int = 3, n_dt_per_year: int = 24) -> EngineConfig:
    """Smallest EngineConfig that exposes lifespan_dt + mortality rates."""
    return EngineConfig.from_dict(
        {
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
            "mortality.additional.rate.sp0": "0.2",
        }
    )


# ---------------------------------------------------------------------------
# additional_mortality
# ---------------------------------------------------------------------------


def test_additional_mortality_propagates_nan_in_abundance() -> None:
    """A NaN in state.abundance flows through into n_dead and new_abundance."""
    cfg = _cfg()
    state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
    state = state.replace(
        abundance=np.array([np.nan, 100.0]),
        age_dt=np.array([10, 10], dtype=np.int32),
    )
    new = additional_mortality(state, cfg, n_subdt=1, step=0)
    # NaN propagates through the corrupted school; the clean school is unaffected.
    assert np.isnan(new.abundance[0])
    assert not np.isnan(new.abundance[1])


def test_additional_mortality_does_not_introduce_nan_from_clean_input() -> None:
    """Sanity floor: clean input never produces NaN."""
    cfg = _cfg()
    state = SchoolState.create(n_schools=3, species_id=np.zeros(3, dtype=np.int32))
    state = state.replace(
        abundance=np.array([100.0, 50.0, 200.0]),
        age_dt=np.array([5, 10, 20], dtype=np.int32),
    )
    new = additional_mortality(state, cfg, n_subdt=1, step=0)
    assert np.isfinite(new.abundance).all(), (
        f"clean input produced non-finite output: {new.abundance}"
    )


# ---------------------------------------------------------------------------
# aging_mortality
# ---------------------------------------------------------------------------


def test_aging_mortality_propagates_nan_abundance() -> None:
    """NaN abundance survives the age-cull (function only zeros expired schools)."""
    cfg = _cfg()
    state = SchoolState.create(n_schools=3, species_id=np.zeros(3, dtype=np.int32))
    lifespan_dt = int(cfg.lifespan_dt[0])
    state = state.replace(
        abundance=np.array([np.nan, 100.0, 50.0]),
        # last school is at the lifespan boundary -> killed regardless of input.
        age_dt=np.array([10, 10, lifespan_dt - 1], dtype=np.int32),
    )
    new = aging_mortality(state, cfg)
    assert np.isnan(new.abundance[0])
    assert new.abundance[1] == 100.0
    assert new.abundance[2] == 0.0


def test_aging_mortality_finite_inputs_finite_outputs() -> None:
    cfg = _cfg()
    state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
    state = state.replace(
        abundance=np.array([100.0, 100.0]),
        age_dt=np.array([10, 20], dtype=np.int32),
    )
    new = aging_mortality(state, cfg)
    assert np.isfinite(new.abundance).all()


# ---------------------------------------------------------------------------
# larva_mortality
# ---------------------------------------------------------------------------


def test_larva_mortality_only_touches_eggs() -> None:
    """A NaN on a non-egg school is preserved; the engine doesn't read it."""
    cfg = _cfg()
    state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
    state = state.replace(
        abundance=np.array([100.0, np.nan]),
        weight=np.array([0.001, 6.0]),
        is_egg=np.array([True, False]),  # only school 0 is an egg
    )
    new = larva_mortality(state, cfg, step=0)
    # School 0 (egg, finite): mortality is 0 with default config; abundance unchanged.
    assert np.isfinite(new.abundance[0])
    # School 1 (non-egg, NaN): ignored by the egg-mask path; NaN passes through
    # without being multiplied by anything.
    assert np.isnan(new.abundance[1])


def test_larva_mortality_with_no_eggs_short_circuits() -> None:
    """No eggs -> early-return preserves all input arrays exactly."""
    cfg = _cfg()
    state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
    state = state.replace(
        abundance=np.array([np.nan, 100.0]),
        is_egg=np.array([False, False]),
    )
    new = larva_mortality(state, cfg, step=0)
    # Returned object is the same — no array manipulation happened.
    assert new is state


# ---------------------------------------------------------------------------
# out_mortality
# ---------------------------------------------------------------------------


def test_out_mortality_only_touches_out_schools() -> None:
    """A NaN abundance on an in-domain school is preserved (the function
    only adjusts schools with is_out=True)."""
    cfg = _cfg()
    state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
    state = state.replace(
        abundance=np.array([np.nan, 100.0]),
        is_out=np.array([False, True]),  # only school 1 is "out"
    )
    new = out_mortality(state, cfg)
    # School 0: in-domain, NaN passed through unchanged.
    assert np.isnan(new.abundance[0])
    # School 1: out, mortality applied with finite rate -> finite result.
    assert np.isfinite(new.abundance[1])


# ---------------------------------------------------------------------------
# Cross-process invariants
# ---------------------------------------------------------------------------


def test_clean_pipeline_stays_finite() -> None:
    """End-to-end sanity: clean input through (additional, aging, out) stays finite."""
    cfg = _cfg()
    state = SchoolState.create(n_schools=4, species_id=np.zeros(4, dtype=np.int32))
    state = state.replace(
        abundance=np.array([1000.0, 1000.0, 1000.0, 1000.0]),
        age_dt=np.array([5, 10, 20, 30], dtype=np.int32),
        is_egg=np.array([False, False, False, False]),
        is_out=np.array([False, False, False, False]),
    )
    s = additional_mortality(state, cfg, n_subdt=1, step=0)
    s = aging_mortality(s, cfg)
    s = out_mortality(s, cfg)
    assert np.isfinite(s.abundance).all(), (
        f"clean pipeline produced non-finite abundance: {s.abundance}"
    )
