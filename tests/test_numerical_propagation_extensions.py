"""H8 extensions: NaN/Inf propagation through reproduction / starvation /
accessibility process functions.

Companion to `test_numerical_propagation.py` (mortality processes).
The same contract applies: each function either propagates NaN cleanly
so a caller can detect and fail loudly, or is unaffected because the
NaN sits in a column the function does not read.
"""

from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.processes.dynamic_accessibility import compute_prey_density_scale
from osmose.engine.processes.reproduction import apply_stock_recruitment
from osmose.engine.processes.starvation import update_starvation_rate
from osmose.engine.state import SchoolState


# ---------------------------------------------------------------------------
# reproduction.apply_stock_recruitment
# ---------------------------------------------------------------------------


def test_apply_stock_recruitment_none_type_unaffected_by_nan_ssb_half() -> None:
    """type='none' short-circuits before reading ssb_half — NaN there is irrelevant."""
    linear = np.array([100.0, 200.0])
    ssb = np.array([1.0, 1.0])
    ssb_half = np.array([np.nan, np.nan])  # NaN
    out = apply_stock_recruitment(linear, ssb, ssb_half, ["none", "none"])
    np.testing.assert_array_equal(out, linear)  # unchanged


def test_apply_stock_recruitment_bh_propagates_nan_in_ssb_half() -> None:
    """type='beverton_holt' divides by ssb_half — NaN propagates."""
    linear = np.array([100.0])
    ssb = np.array([1.0])
    ssb_half = np.array([np.nan])
    out = apply_stock_recruitment(linear, ssb, ssb_half, ["beverton_holt"])
    assert np.isnan(out[0])


def test_apply_stock_recruitment_zero_ssb_short_circuits_for_nan() -> None:
    """SSB <= 0 short-circuits before division; NaN ssb_half doesn't matter."""
    linear = np.array([100.0])
    ssb = np.array([0.0])  # short-circuit branch
    ssb_half = np.array([np.nan])
    out = apply_stock_recruitment(linear, ssb, ssb_half, ["beverton_holt"])
    assert out[0] == 100.0  # unchanged (linear regime)


def test_apply_stock_recruitment_ricker_propagates_nan() -> None:
    linear = np.array([100.0])
    ssb = np.array([np.nan])  # NaN in ssb
    ssb_half = np.array([10.0])
    out = apply_stock_recruitment(linear, ssb, ssb_half, ["ricker"])
    # ssb<=0.0 check on NaN: NaN compared with 0.0 is False, so falls through to ricker
    # exp(-NaN/finite) = exp(NaN) = NaN
    assert np.isnan(out[0])


def test_apply_stock_recruitment_clean_input_clean_output() -> None:
    linear = np.array([100.0, 200.0, 300.0])
    ssb = np.array([10.0, 20.0, 30.0])
    ssb_half = np.array([100.0, 200.0, 300.0])
    out = apply_stock_recruitment(
        linear, ssb, ssb_half, ["none", "beverton_holt", "ricker"]
    )
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# dynamic_accessibility.compute_prey_density_scale
# ---------------------------------------------------------------------------


def test_compute_prey_density_scale_propagates_nan_prey_biomass() -> None:
    """NaN prey_biomass propagates through the ratio computation."""
    prey = np.array([np.nan, 100.0])
    ref = np.array([100.0, 100.0])
    scale = compute_prey_density_scale(prey, ref, exponent=1.0, floor=0.05)
    # NaN propagates; np.clip(NaN, ...) returns NaN
    assert np.isnan(scale[0])
    assert np.isfinite(scale[1])


def test_compute_prey_density_scale_zero_reference_floored() -> None:
    """The safe_ref = max(ref, 1e-10) guard prevents div-by-zero.

    M5a-related: documents that the function is robust to ref=0 (a likely
    config-corruption sentinel) without producing inf or NaN.
    """
    prey = np.array([1.0])
    ref = np.array([0.0])  # would otherwise divide by 0
    scale = compute_prey_density_scale(prey, ref, exponent=1.0, floor=0.05)
    # ratio = 1 / 1e-10 = huge; clipped to 1.0
    assert scale[0] == 1.0


def test_compute_prey_density_scale_inf_prey_biomass_clamped() -> None:
    """Inf prey biomass clamps to 1.0 — the floor/ceiling guard contains it."""
    prey = np.array([np.inf])
    ref = np.array([100.0])
    scale = compute_prey_density_scale(prey, ref, exponent=1.0, floor=0.05)
    assert scale[0] == 1.0


def test_compute_prey_density_scale_clean_input_in_bounds() -> None:
    prey = np.array([10.0, 50.0, 100.0, 200.0])
    ref = np.array([100.0, 100.0, 100.0, 100.0])
    scale = compute_prey_density_scale(prey, ref, exponent=1.0, floor=0.05)
    assert (scale >= 0.05).all()
    assert (scale <= 1.0).all()


# ---------------------------------------------------------------------------
# starvation.update_starvation_rate
# ---------------------------------------------------------------------------


def _starvation_cfg() -> EngineConfig:
    return EngineConfig.from_dict(
        {
            "simulation.time.ndtperyear": "12",
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
            "species.lifespan.sp0": "3",
            "species.vonbertalanffy.threshold.age.sp0": "1.0",
            "mortality.subdt": "10",
            "predation.ingestion.rate.max.sp0": "3.5",
            "predation.efficiency.critical.sp0": "0.57",
            "mortality.starvation.rate.max.sp0": "1.0",
        }
    )


def test_update_starvation_rate_clamps_nan_pred_success_rate() -> None:
    """NaN in pred_success_rate is clamped to 0 (no starvation) by the np.where path.

    Documents (defensive) current behaviour: `NaN <= csr` evaluates to False
    in NumPy, so the np.where(...) call hits the else=0.0 branch. The
    subsequent `np.maximum(0.0, 0.0) = 0.0` finalises it. This is a
    benign clamp — NaN treated as "high success rate, no starvation" —
    rather than NaN propagation. Future maintainers should not "fix" this
    to propagate without considering the downstream mortality consumers.
    """
    cfg = _starvation_cfg()
    state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
    state = state.replace(
        pred_success_rate=np.array([np.nan, 0.5]),
    )
    new = update_starvation_rate(state, cfg)
    # School 0 (NaN input): silently clamped to 0 — no starvation.
    assert new.starvation_rate[0] == 0.0
    # School 1 (clean input): finite rate.
    assert np.isfinite(new.starvation_rate[1])


def test_update_starvation_rate_clean_input_finite_output() -> None:
    cfg = _starvation_cfg()
    state = SchoolState.create(n_schools=3, species_id=np.zeros(3, dtype=np.int32))
    state = state.replace(
        pred_success_rate=np.array([0.1, 0.5, 0.9]),
    )
    new = update_starvation_rate(state, cfg)
    assert np.isfinite(new.starvation_rate).all()
    assert (new.starvation_rate >= 0.0).all()  # no negative rates
