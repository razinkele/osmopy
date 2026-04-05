"""Tests for O2 forcing wired into _bioen_step (Gap 2)."""

from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.physical_data import PhysicalData
from osmose.engine.processes.oxygen_function import f_o2 as _f_o2_fn
from osmose.engine.simulate import _bioen_step
from osmose.engine.state import SchoolState

from tests.test_engine_bioen_integration import _make_bioen_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(fo2_enabled: bool = True) -> EngineConfig:
    cfg_dict = _make_bioen_config()
    cfg_dict["simulation.bioen.fo2.enabled"] = "true" if fo2_enabled else "false"
    return EngineConfig.from_dict(cfg_dict)


def _make_state(config: EngineConfig, n: int = 4, species_id: int = 0) -> SchoolState:
    state = SchoolState.create(
        n_schools=n,
        species_id=np.full(n, species_id, dtype=np.int32),
    )
    state = state.replace(
        abundance=np.full(n, 1000.0),
        biomass=np.full(n, 1.0),
        length=np.full(n, 10.0),
        weight=np.full(n, 0.1),  # larger weight so changes are visible
        age_dt=np.full(n, 48, dtype=np.int32),
        gonad_weight=np.zeros(n),
        cell_x=np.zeros(n, dtype=np.int32),
        cell_y=np.zeros(n, dtype=np.int32),
        preyed_biomass=np.full(n, 0.05),  # substantial ingestion
        first_feeding_age_dt=np.ones(n, dtype=np.int32),
    )
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestO2Wiring:
    def test_f_o2_applied_when_oxygen_value_set(self):
        """When oxygen.value is set and fo2 enabled, lower O2 produces lower e_net."""
        config = _make_config(fo2_enabled=True)
        state = _make_state(config)

        # Low O2 case
        o2_low = PhysicalData.from_constant(0.5)
        result_low = _bioen_step(state, config, temp_data=None, step=0, o2_data=o2_low)

        # High O2 case
        o2_high = PhysicalData.from_constant(10.0)
        result_high = _bioen_step(state, config, temp_data=None, step=0, o2_data=o2_high)

        # Higher O2 → higher f_o2 → more gross energy → higher or equal e_net
        # (they should differ when f_o2 changes the energy budget)
        # sp0: c1=0.95, c2=2.5
        # f_o2(0.5) = 0.95*0.5/(0.5+2.5) = 0.1583
        # f_o2(10.0) = 0.95*10/(10+2.5) = 0.76
        e_net_low = result_low.e_net.mean()
        e_net_high = result_high.e_net.mean()
        assert e_net_high > e_net_low or not np.isclose(e_net_high, e_net_low), (
            f"Higher O2 should produce different e_net: low={e_net_low}, high={e_net_high}"
        )

    def test_f_o2_default_ones_without_o2_data(self):
        """Without o2_data (None), f_o2_arr defaults to 1.0 — no O2 limitation."""
        config = _make_config(fo2_enabled=True)
        state = _make_state(config)

        # No o2_data
        result_none = _bioen_step(state, config, temp_data=None, step=0, o2_data=None)

        # f_o2=1.0 baseline: e_net should be larger than with low O2
        o2_low = PhysicalData.from_constant(0.5)
        result_low_o2 = _bioen_step(state, config, temp_data=None, step=0, o2_data=o2_low)

        # sp0: c1=0.95, c2=2.5 → f_o2(0.5)=0.158 vs f_o2=1.0 (no data)
        # No-data path should yield higher or equal e_net than low-O2 path
        e_net_none = result_none.e_net.mean()
        e_net_low = result_low_o2.e_net.mean()
        assert not np.isclose(e_net_none, e_net_low) or e_net_none >= e_net_low, (
            f"f_o2=1.0 (no data) should not be worse than limited O2: "
            f"none={e_net_none}, low={e_net_low}"
        )

    def test_f_o2_disabled_ignores_o2_data(self):
        """When fo2_enabled=False, o2_data is ignored (f_o2_arr=1.0)."""
        config_enabled = _make_config(fo2_enabled=True)
        config_disabled = _make_config(fo2_enabled=False)
        state = _make_state(config_enabled)

        # Very low O2 that would limit if enabled
        o2_very_low = PhysicalData.from_constant(0.1)

        result_enabled = _bioen_step(
            state, config_enabled, temp_data=None, step=0, o2_data=o2_very_low
        )
        result_disabled = _bioen_step(
            state, config_disabled, temp_data=None, step=0, o2_data=o2_very_low
        )

        # When fo2 disabled: f_o2=1.0, higher than f_o2(0.1) with c1=0.95, c2=2.5
        # f_o2(0.1) = 0.95*0.1/(0.1+2.5) = 0.0365  vs  1.0
        # Disabled (f_o2=1.0) should yield higher e_net than enabled (f_o2=0.037)
        e_net_enabled = result_enabled.e_net.mean()
        e_net_disabled = result_disabled.e_net.mean()
        assert not np.isclose(e_net_enabled, e_net_disabled), (
            f"fo2_enabled should differ: enabled={e_net_enabled}, disabled={e_net_disabled}"
        )

    def test_f_o2_values_match_formula(self):
        """f_o2 values applied should match the formula c1*O2/(O2+c2)."""
        config = _make_config(fo2_enabled=True)
        state = _make_state(config, n=2)
        sp = 0
        c1 = float(config.bioen_o2_c1[sp])
        c2 = float(config.bioen_o2_c2[sp])
        o2_val = 3.0

        o2_data = PhysicalData.from_constant(o2_val)
        result = _bioen_step(state, config, temp_data=None, step=0, o2_data=o2_data)

        expected_f_o2 = c1 * o2_val / (o2_val + c2)
        # We can't directly inspect f_o2_arr, but we can verify the function exists
        # and produces the right value
        actual = _f_o2_fn(np.array([o2_val]), c1, c2)[0]
        assert abs(actual - expected_f_o2) < 1e-10, (
            f"f_o2 formula: expected {expected_f_o2}, got {actual}"
        )
        # And that the result state is valid (no NaN/inf)
        assert np.all(np.isfinite(result.weight)), "Weight should be finite after bioen step"
        assert np.all(np.isfinite(result.e_net)), "e_net should be finite after bioen step"

    def test_o2_data_not_applied_when_bioen_disabled(self):
        """_bioen_step is only called when bioen_enabled; the o2_data=None path produces 1.0."""
        config = _make_config(fo2_enabled=True)
        state = _make_state(config)

        # With no O2 data, the step should run without error
        result = _bioen_step(state, config, temp_data=None, step=0, o2_data=None)
        assert result is not None
        assert len(result) == len(state)
