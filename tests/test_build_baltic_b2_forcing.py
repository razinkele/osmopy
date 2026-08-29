"""Builder unit tests (B2 spec, Task 2): pure offset/predicted-dK functions on synthetic
fixtures (task-2-brief.md, verbatim)."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "build_baltic_b2_forcing",
    Path(__file__).resolve().parent.parent / "scripts" / "build_baltic_b2_forcing.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def _field():
    o2 = np.full((24, 4, 4), 200.0)
    o2[:, 0, 0] = 30.0  # hypoxic wet cell
    o2[:, 3, 3] = np.nan  # land
    wet = np.ones((4, 4), dtype=bool)
    wet[3, 3] = False
    return o2, wet


def test_offset_wet_only_and_floor():
    o2, wet = _field()
    out = m.offset_o2(o2, wet, -50.0)
    assert out[0, 0, 0] == 0.0  # floored, not negative
    assert out[0, 1, 1] == 150.0
    assert np.isnan(out[0, 3, 3])  # land untouched
    out2 = m.offset_o2(o2, wet, 26.8)
    assert out2[0, 0, 0] == 30.0 + 26.8
    assert np.isnan(out2[0, 3, 3])


def test_offset_zero_is_identity():
    o2, wet = _field()
    out = m.offset_o2(o2, wet, 0.0)
    assert np.array_equal(out[:, wet], o2[:, wet])  # exact, bit-level on wet cells


def test_predicted_k_change_signs_and_zero():
    o2, wet = _field()
    k = np.ones((4, 4))
    assert m.predicted_k_change(o2, wet, k, 0.0) == 0.0
    assert m.predicted_k_change(o2, wet, k, 26.8) > 0.0
    assert m.predicted_k_change(o2, wet, k, -8.9) < 0.0


def test_predicted_k_change_uses_real_hill():
    o2, wet = _field()
    k = np.zeros((4, 4))
    k[0, 0] = 1.0  # all weight on the hypoxic cell
    from osmose.engine.processes.oxygen_function import f_o2_hill

    expect = (
        f_o2_hill(np.array([56.8]), 60.0, 3.0)[0] / f_o2_hill(np.array([30.0]), 60.0, 3.0)[0] - 1.0
    )
    got = m.predicted_k_change(o2, wet, k, 26.8)
    assert abs(got - expect) < 1e-12
