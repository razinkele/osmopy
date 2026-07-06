import numpy as np
from osmose.forcing.percid_habitat import percid_stage_map, vacuity_ok


def test_binary_threshold_and_land():
    frac = np.array([[0.5, 0.1], [0.9, 0.0]])
    ocean = np.array([[True, True], [True, False]])
    sal = np.full((2, 2), 4.0)
    m = percid_stage_map(frac, ocean, sal, tau=0.25, sal_ceiling=12.0)
    assert m[1, 1] == -99.0  # land
    assert m[0, 0] == 1.0  # frac 0.5 >= 0.25
    assert m[0, 1] == 0.0  # frac 0.1 < 0.25
    assert set(np.unique(m)) <= {-99.0, 0.0, 1.0}  # strictly binary (+ land)


def test_adult_salinity_ceiling():
    m = percid_stage_map(
        np.array([[0.8, 0.8]]),
        np.array([[True, True]]),
        np.array([[5.0, 20.0]]),
        tau=0.25,
        sal_ceiling=12.0,
    )
    assert m[0, 0] == 1.0 and m[0, 1] == 0.0  # 20 PSU excluded


def test_spawning_gate_tighter():
    m = percid_stage_map(
        np.array([[0.8, 0.8]]),
        np.array([[True, True]]),
        np.array([[4.0, 5.5]]),
        tau=0.25,
        sal_gate=5.0,
    )
    assert m[0, 0] == 1.0 and m[0, 1] == 0.0  # 5.5 >= 5.0 gate


def test_nan_salinity_is_excluded_not_kept():
    # gap-fill should remove NaN, but guard anyway: NaN must NOT pass as habitat
    m = percid_stage_map(
        np.array([[0.8]]), np.array([[True]]), np.array([[np.nan]]), tau=0.25, sal_ceiling=12.0
    )
    assert m[0, 0] == 0.0


def test_vacuity_guard_area_vs_upsampled_percid_footprint():
    real = np.array([[0.0, 1.0, 0.0, 0.0]])
    up = np.array([[1.0, 1.0, 1.0, 0.0]])  # 3-cell percid footprint
    assert vacuity_ok(real, up, max_ratio=0.4)  # 1/3 = 0.33
    assert not vacuity_ok(up, up, max_ratio=0.4)  # 3/3
    assert not vacuity_ok(np.zeros((1, 4)), up)  # empty
