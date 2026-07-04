import numpy as np
import pytest

from osmose.engine.config import _load_salinity_gate
from osmose.engine.movement_maps import MovementMapSet
from osmose.engine.processes.movement import _map_move_school
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


def test_salinity_gate_off_returns_defaults():
    enabled, mask, lo, hi, field = _load_salinity_gate({}, 3)
    assert enabled is False and mask is None and field is None
    assert (lo, hi) == (3.0, 6.0)


def _on_cfg(**extra):
    cfg = {
        "movement.salinity.gate.enabled": "true",
        "movement.salinity.field.constant": "8.0",
        "movement.salinity.gate.species.enabled.sp0": "true",
    }
    cfg.update(extra)
    return cfg


def test_salinity_gate_on_constant_field():
    enabled, mask, lo, hi, field = _load_salinity_gate(_on_cfg(), 3)
    assert enabled is True
    assert list(mask) == [True, False, False]
    assert (lo, hi) == (3.0, 6.0)
    assert field is not None and field.is_constant and field.get_scalar() == 8.0


def test_salinity_gate_custom_thresholds():
    _, _, lo, hi, _ = _load_salinity_gate(
        _on_cfg(**{"movement.salinity.gate.s.low": "4.0", "movement.salinity.gate.s.high": "7.0"}),
        3,
    )
    assert (lo, hi) == (4.0, 7.0)


def test_salinity_gate_bad_thresholds_raise():
    with pytest.raises(ValueError, match="s.high|s_high"):
        _load_salinity_gate(_on_cfg(**{"movement.salinity.gate.s.high": "3.0"}), 3)


def test_salinity_gate_no_species_raises():
    cfg = {"movement.salinity.gate.enabled": "true", "movement.salinity.field.constant": "8.0"}
    with pytest.raises(ValueError, match="no species"):
        _load_salinity_gate(cfg, 3)


def test_salinity_gate_no_field_raises():
    cfg = {
        "movement.salinity.gate.enabled": "true",
        "movement.salinity.gate.species.enabled.sp0": "true",
    }
    with pytest.raises(ValueError, match="salinity field|field"):
        _load_salinity_gate(cfg, 3)


def _uniform_map_set(ny, nx):
    """A MovementMapSet whose single presence map is 1.0 over all cells."""
    ms = MovementMapSet.__new__(MovementMapSet)
    ms.maps = [np.ones((ny, nx), dtype=np.float64)]
    # shape (lifespan_dt, n_total_steps); index 0 for ALL age/step so get_map is
    # valid for age_dt 0 AND 1 (the random-walk test uses age_dt=1, step=1).
    ms.index_maps = np.zeros((10, 1000), dtype=np.int32)
    ms.max_proba = np.array([0.0])  # presence/absence -> uniform accept
    ms.n_maps = 1
    return ms


def _draw_columns(gate_grid, n=4000):
    ny, nx = 5, 6
    ms = _uniform_map_set(ny, nx)
    ocean = np.ones((ny, nx), dtype=np.bool_)
    rng = np.random.default_rng(0)
    cols = np.zeros(nx, dtype=np.int64)
    for _ in range(n):
        x, y, out = _map_move_school(
            0, -1, -1, ny, nx, ocean, ms, 1, 0, rng, salinity_weight_grid=gate_grid
        )
        assert not out
        cols[x] += 1
    return cols


def test_placement_excludes_low_and_grades_mid_vs_high():
    ny, nx = 5, 6
    # three salinity bands by column: cols 0-1 = 2 psu, 2-3 = 4.5 psu, 4-5 = 8 psu
    S = np.zeros((ny, nx))
    S[:, 0:2] = 2.0
    S[:, 2:4] = 4.5
    S[:, 4:6] = 8.0
    w = salinity_weight(S, 3.0, 6.0)  # weights 0 / 0.5 / 1.0
    cols = _draw_columns(w)
    assert cols[0] == 0 and cols[1] == 0  # excluded (weight 0)
    high = cols[4] + cols[5]
    mid = cols[2] + cols[3]
    assert mid > 0 and high > 0
    assert high / mid == pytest.approx(2.0, rel=0.15)  # graded ~2x (Step 3a path)


def test_placement_ungated_is_uniform():
    cols = _draw_columns(None)
    # No gate: all 6 columns populated roughly equally
    assert (cols > 0).all()


def test_random_walk_weighted(monkeypatch):
    # Force Step 3b (same-map, located school): weighted selection ~2x high vs mid.
    ny, nx = 5, 6
    ms = _uniform_map_set(ny, nx)
    ocean = np.ones((ny, nx), dtype=np.bool_)
    S = np.zeros((ny, nx))
    S[:, 2:4] = 4.5
    S[:, 4:6] = 8.0
    w = salinity_weight(S, 3.0, 6.0)
    rng = np.random.default_rng(1)
    cols = np.zeros(nx, dtype=np.int64)
    # start located at (cx=3, cy=2), walk_range large enough to reach cols 2-5
    for _ in range(4000):
        x, y, out = _map_move_school(1, 3, 2, ny, nx, ocean, ms, 5, 1, rng, salinity_weight_grid=w)
        cols[x] += 1
    high = cols[4] + cols[5]
    mid = cols[2] + cols[3]
    assert cols[0] == 0 and cols[1] == 0
    assert high / mid == pytest.approx(2.0, rel=0.2)


from types import SimpleNamespace  # noqa: E402

from osmose.engine.physical_data import PhysicalData  # noqa: E402
from osmose.engine.processes.movement import _movement_salinity_weight  # noqa: E402


def _cfg_grid(enabled, field):
    cfg = SimpleNamespace(
        salinity_gate_enabled=enabled,
        salinity_field=field,
        salinity_gate_s_low=3.0,
        salinity_gate_s_high=6.0,
    )
    grid = SimpleNamespace(ny=5, nx=6)
    return cfg, grid


def test_movement_weight_off_returns_none():
    cfg, grid = _cfg_grid(False, None)
    assert _movement_salinity_weight(cfg, grid, 0) is None


def test_movement_weight_enabled_but_no_field_returns_none():
    cfg, grid = _cfg_grid(True, None)
    assert _movement_salinity_weight(cfg, grid, 0) is None


def test_movement_weight_constant_high_all_ones():
    cfg, grid = _cfg_grid(True, PhysicalData.from_constant(8.0))
    w = _movement_salinity_weight(cfg, grid, 0)
    assert w.shape == (5, 6)
    np.testing.assert_array_equal(w, np.ones((5, 6)))


def test_movement_weight_constant_low_all_zeros():
    cfg, grid = _cfg_grid(True, PhysicalData.from_constant(2.0))
    np.testing.assert_array_equal(_movement_salinity_weight(cfg, grid, 0), np.zeros((5, 6)))


from osmose.config import OsmoseConfigReader  # noqa: E402
from osmose.engine import PythonEngine  # noqa: E402


def test_gate_off_is_bit_identical():
    cfg = OsmoseConfigReader().read("data/eec_full/eec_all-parameters.csv")
    cfg["simulation.time.nyear"] = "2"
    cfg["simulation.rng.fixed"] = "true"
    cfg["movement.randomseed.fixed"] = "true"
    cfg["stochastic.mortality.randomseed.fixed"] = "true"
    base = PythonEngine().run_in_memory(dict(cfg), seed=0).biomass()
    cfg["movement.salinity.gate.enabled"] = "false"
    off = PythonEngine().run_in_memory(dict(cfg), seed=0).biomass()
    np.testing.assert_array_equal(base.to_numpy(), off.to_numpy())


def test_gate_enabled_warns_on_numba_path():
    """Enabling the gate on a config that uses map-based (Numba) movement must
    warn loudly that the gate has no effect there (prototype: Python path only).

    eec_full's sp0 (lesserSpottedDogfish) uses movement.distribution.method=maps
    and the engine runs with Numba available (flat_map_data populated), so this
    exercises the real Numba branch in `movement()`, not a mock.
    """
    cfg = OsmoseConfigReader().read("data/eec_full/eec_all-parameters.csv")
    cfg["simulation.time.nyear"] = "1"
    cfg["simulation.rng.fixed"] = "true"
    cfg["movement.randomseed.fixed"] = "true"
    cfg["stochastic.mortality.randomseed.fixed"] = "true"
    cfg["movement.salinity.gate.enabled"] = "true"
    cfg["movement.salinity.field.constant"] = "8.0"
    cfg["movement.salinity.gate.species.enabled.sp0"] = "true"
    with pytest.warns(RuntimeWarning, match="Numba movement path"):
        PythonEngine().run_in_memory(dict(cfg), seed=0).biomass()
