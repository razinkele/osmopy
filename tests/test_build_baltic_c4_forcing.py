"""Builder unit tests (C4 spec 2026-08-30, Task 2): pure offset/ramp/instrument functions on
synthetic fixtures, plus an orientation pin on a real production map (task-2-brief.md,
verbatim bullet list)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "build_baltic_c4_forcing",
    Path(__file__).resolve().parent.parent / "scripts" / "build_baltic_c4_forcing.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


# ---------------------------------------------------------------------------
# offset_salinity: wet-only, NaN-untouched, floor-at-0, zero-identity
# ---------------------------------------------------------------------------


def _sal_field():
    sal = np.full((24, 4, 4), 20.0)
    sal[:, 0, 0] = 1.5  # near-floor wet cell
    sal[:, 3, 3] = np.nan  # land
    wet = np.ones((4, 4), dtype=bool)
    wet[3, 3] = False
    return sal, wet


def test_offset_wet_only_and_floor():
    sal, wet = _sal_field()
    out = m.offset_salinity(sal, wet, -2.0)
    assert out[0, 0, 0] == 0.0  # floored, not negative
    assert out[0, 1, 1] == 18.0
    assert np.isnan(out[0, 3, 3])  # land untouched
    out2 = m.offset_salinity(sal, wet, 3.2)
    assert out2[0, 0, 0] == pytest.approx(1.5 + 3.2)
    assert np.isnan(out2[0, 3, 3])


def test_offset_zero_is_identity():
    sal, wet = _sal_field()
    out = m.offset_salinity(sal, wet, 0.0)
    assert np.array_equal(out[:, wet], sal[:, wet])  # exact, bit-level on wet cells


# ---------------------------------------------------------------------------
# ramp_w: production ramp values, NaN-safe
# ---------------------------------------------------------------------------


def test_ramp_w_values():
    w = m.ramp_w(np.array([3.0, 4.5, 6.0, np.nan]))
    assert w[0] == pytest.approx(0.0)
    assert w[1] == pytest.approx(0.5)
    assert w[2] == pytest.approx(1.0)
    assert np.isnan(w[3])


def test_ramp_w_clips_beyond_bounds():
    w = m.ramp_w(np.array([0.0, 9.0]))
    assert w[0] == pytest.approx(0.0)
    assert w[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TV distance / excluded_fraction: vacuity case (saturation fixture)
# ---------------------------------------------------------------------------


def test_tv_and_exclusions_zero_on_saturation_fixture():
    # All cells >= 6 PSU at both base and arm -> w=1 everywhere -> TV=0, exclusions=0.
    map_grid = np.array([[1.0, 0.5], [0.0, -99.0]])
    sal = np.full((3, 2, 2), 10.0)
    wet = np.array([[True, True], [True, True]])
    offset = m.offset_salinity(sal, wet, -1.0)  # still >= 6 everywhere
    w_base = m.ramp_w(sal)
    w_arm = m.ramp_w(offset)
    assert m.tv_distance(map_grid, w_base, w_arm) == pytest.approx(0.0)
    assert m.excluded_fraction(map_grid, w_base, w_arm) == pytest.approx(0.0)
    assert m.mean_dw(map_grid, w_base, w_arm) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TV distance: mixed fixture, hand-computed
# ---------------------------------------------------------------------------


def test_tv_distance_hand_computed():
    # support cells: (0,0) map=1.0, (0,1) map=0.5; (1,0)=0 excluded; (1,1)=-99 land excluded.
    map_grid = np.array([[1.0, 0.5], [0.0, -99.0]])
    # 2 frames, constant weights per frame (keeps the hand computation exam single-valued).
    w_base = np.zeros((2, 2, 2))
    w_base[:, 0, 0] = 1.0
    w_base[:, 0, 1] = 1.0
    w_arm = np.zeros((2, 2, 2))
    w_arm[:, 0, 0] = 0.6
    w_arm[:, 0, 1] = 1.0

    # base: vals=[1.0, 0.5], sum=1.5, p=[2/3, 1/3]
    # arm:  vals=[0.6, 0.5], sum=1.1, q=[6/11, 5/11]
    # TV = 0.5*(|2/3-6/11| + |1/3-5/11|) = 0.5*(4/33+4/33) = 4/33
    expected = 4.0 / 33.0
    assert m.tv_distance(map_grid, w_base, w_arm) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# excluded_fraction: hand-computed via the full offset/ramp pipeline (dS=-3)
# ---------------------------------------------------------------------------


def test_excluded_fraction_hand_computed_dS_minus_3():
    # cell A=4.0 (base w=1/3>0, arm w=0 -> newly excluded)
    # cell B=7.0 (base w=1, arm w=1/3 -> stays open)
    # cell C=2.0 (base w=0 already -> not "newly" excluded)
    # cell D=NaN (land, non-wet)
    sal = np.empty((2, 2, 2))
    sal[:, 0, 0] = 4.0
    sal[:, 0, 1] = 7.0
    sal[:, 1, 0] = 2.0
    sal[:, 1, 1] = np.nan
    wet = np.array([[True, True], [True, False]])
    map_grid = np.array([[1.0, 1.0], [1.0, -99.0]])  # support = A, B, C

    offset = m.offset_salinity(sal, wet, -3.0)
    w_base = m.ramp_w(sal)
    w_arm = m.ramp_w(offset)

    # 1 of 3 support cells newly excluded, identical both frames -> 2/6 = 1/3
    assert m.excluded_fraction(map_grid, w_base, w_arm) == pytest.approx(1.0 / 3.0)
    # mean_dw: A: 0-1/3=-1/3; B: 1/3-1=-2/3; C: 0-0=0 -> mean = (-1/3-2/3+0)/3 = -1/3
    assert m.mean_dw(map_grid, w_base, w_arm) == pytest.approx(-1.0 / 3.0)


# ---------------------------------------------------------------------------
# All-zero event detection
# ---------------------------------------------------------------------------


def test_all_zero_event_single_cell_map_at_4psu_dS_minus_3():
    # Single support cell at 4 PSU baseline; dS=-3 -> 1 PSU -> w_arm=0 (below s_low=3):
    # the map*w_arm sum is 0 for every frame -> nan TV + a recorded all-zero event.
    sal = np.full((2, 2, 2), np.nan)
    sal[:, 0, 0] = 4.0
    wet = np.array([[True, False], [False, False]])
    map_grid = np.array([[1.0, -99.0], [-99.0, -99.0]])

    offset = m.offset_salinity(sal, wet, -3.0)
    w_base = m.ramp_w(sal)
    w_arm = m.ramp_w(offset)

    assert np.isnan(m.tv_distance(map_grid, w_base, w_arm))
    assert m.all_zero_frames(map_grid, w_arm) == [0, 1]
    assert m.all_zero_frames(map_grid, w_base) == []  # base w=1/3>0: not all-zero


# ---------------------------------------------------------------------------
# prey_overlap_shift: wiring sanity (no real numeric expectation pinned by the spec)
# ---------------------------------------------------------------------------


def test_prey_overlap_shift_zero_when_nothing_changes():
    map_grid = np.array([[1.0, 0.5], [0.0, -99.0]])
    prey_map = np.array([[1.0, 0.0], [0.0, 0.0]])
    w = np.zeros((2, 2, 2))
    w[:, 0, 0] = 0.7
    w[:, 0, 1] = 0.9
    assert m.prey_overlap_shift(map_grid, w, w, prey_map) == pytest.approx(0.0)


def test_prey_overlap_shift_nonzero_when_mass_moves_into_prey_domain():
    map_grid = np.array([[1.0, 1.0], [0.0, -99.0]])
    prey_map = np.array([[1.0, 0.0], [0.0, 0.0]])  # prey occupies cell (0,0) only
    w_base = np.zeros((1, 2, 2))
    w_base[:, 0, 0] = 0.5
    w_base[:, 0, 1] = 0.5
    w_arm = np.zeros((1, 2, 2))
    w_arm[:, 0, 0] = 1.0  # arm shifts mass toward the prey cell
    w_arm[:, 0, 1] = 0.5
    # base: p=[0.5,0.5]; arm: q=[1/1.5, 0.5/1.5]=[2/3,1/3]; prey mass base=0.5, arm=2/3
    expected = 2.0 / 3.0 - 0.5
    assert m.prey_overlap_shift(map_grid, w_base, w_arm, prey_map) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Orientation pin: a real cod map, loaded via the engine's own loader, must have zero
# map-positive cells on land (grid mask) -- catches a naive upside-down read.
# ---------------------------------------------------------------------------


def test_orientation_pin_real_cod_map_vs_grid_mask():
    grid_mask = m.load_grid_mask(m.DEFAULT_GRID_PATH)
    ny, nx = grid_mask.shape
    map_grid = m.load_stage_map("cod_west", "adult", ny, nx, str(m.MAPS_CONFIG_DIR))
    on_land_and_positive = (map_grid > 0.0) & ~grid_mask
    assert int(on_land_and_positive.sum()) == 0


# ---------------------------------------------------------------------------
# ZERO_ARM_DEF single source of truth (B2 precedent)
# ---------------------------------------------------------------------------


def test_zero_arm_def_is_the_single_source_of_truth():
    assert m.ZERO_ARM_DEF == {"name": "zero", "dS_PSU": 0.0}
