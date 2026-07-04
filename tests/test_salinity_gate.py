import numpy as np
import pytest

from osmose.engine.processes.salinity_gate import salinity_weight, salinity_weighted_map
from osmose.schema import build_registry


def test_salinity_weight_ramp_scalar():
    assert salinity_weight(2.0, 3.0, 6.0) == 0.0  # below low
    assert salinity_weight(3.0, 3.0, 6.0) == 0.0  # at low
    assert salinity_weight(4.5, 3.0, 6.0) == pytest.approx(0.5)  # mid
    assert salinity_weight(6.0, 3.0, 6.0) == 1.0  # at high
    assert salinity_weight(8.0, 3.0, 6.0) == 1.0  # above high


def test_salinity_weight_array():
    S = np.array([2.0, 4.5, 8.0])
    np.testing.assert_allclose(salinity_weight(S, 3.0, 6.0), [0.0, 0.5, 1.0])


def test_salinity_weight_bad_thresholds_raise():
    with pytest.raises(ValueError):
        salinity_weight(5.0, 6.0, 6.0)  # s_high <= s_low


def test_weighted_map_zeros_low_keeps_high():
    m = np.ones((2, 3))
    w = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
    out = salinity_weighted_map(m, w)
    np.testing.assert_allclose(out, w)  # 1 * w == w
    assert out is not m  # gated -> new array


def test_weighted_map_all_zero_guard_returns_original():
    m = np.ones((2, 2))
    w = np.zeros((2, 2))
    out = salinity_weighted_map(m, w)
    assert out is m  # identity: guard fell back to original


def test_salinity_gate_keys_registered():
    keys = {f.key_pattern for f in build_registry().all_fields()}
    assert "movement.salinity.gate.enabled" in keys
    assert "movement.salinity.gate.species.enabled.sp{idx}" in keys
    assert "movement.salinity.gate.s.low" in keys
    assert "movement.salinity.gate.s.high" in keys
    assert "movement.salinity.field.constant" in keys
    assert "movement.salinity.field.file" in keys
    assert "movement.salinity.field.varname" in keys
