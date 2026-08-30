"""Builder unit tests (C4 spec 2026-08-30, Task 2): pure offset/ramp/instrument functions on
synthetic fixtures, plus an orientation pin on a real production map (task-2-brief.md,
verbatim bullet list)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

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
# write_arm_dir: prey_overlap_shift is computed PER NAMED STAGE (adult AND juvenile), not
# adult-only (review finding: the max-TV argument that justifies "adult" as the (i)/(iii)
# headline row doesn't bear on this instrument -- it measures where redistributed mass
# lands, not how much w changes).
# ---------------------------------------------------------------------------


def test_write_arm_dir_prey_overlap_computed_per_named_stage(tmp_path, monkeypatch):
    # 1x4 synthetic grid, all wet, salinity varies per cell (a spatially-uniform ramp would
    # trivially cancel out of every normalized ratio and hide a stage-dependent bug).
    ny, nx = 1, 4
    sal_row = np.array([5.0, 4.0, 4.5, 7.0])
    sal = np.broadcast_to(sal_row, (24, ny, nx)).astype(np.float64).copy()
    sal_path = tmp_path / "sal.nc"
    xr.Dataset({"salinity": (("time", "latitude", "longitude"), sal)}).to_netcdf(sal_path)
    grid_path = tmp_path / "grid.nc"
    xr.Dataset({"mask": (("latitude", "longitude"), np.ones((ny, nx), dtype=np.int32))}).to_netcdf(
        grid_path
    )

    # adult support = cells {0,1}; juvenile support = cells {1,2}; prey present at cell 1
    # only. Both stages' support overlaps the prey cell, but the normalization denominator
    # differs (adult sums over {0,1}, juvenile over {1,2}), so the SAME w_base/w_arm must
    # yield DIFFERENT shift values for the two named stages -- proving the instrument is
    # genuinely evaluated per stage, not collapsed onto a single hardcoded one.
    adult_map = np.array([[1.0, 1.0, 0.0, 0.0]])
    juvenile_map = np.array([[0.0, 1.0, 1.0, 0.0]])
    prey_map = np.array([[0.0, 1.0, 0.0, 0.0]])

    def fake_load_species_maps(species, ny_, nx_, config_dir=None):
        return {"juvenile": juvenile_map, "adult": adult_map, "spawning": adult_map}

    def fake_load_prey_union_map(species, ny_, nx_, config_dir=None):
        return prey_map

    monkeypatch.setattr(m, "load_species_maps", fake_load_species_maps)
    monkeypatch.setattr(m, "load_prey_union_map", fake_load_prey_union_map)

    result = m.write_arm_dir(
        {"name": "test_arm", "dS_PSU": -1.0}, tmp_path / "out", sal_path, grid_path
    )

    prey_overlap = result["instruments"]["prey_overlap"]["cod_west"]
    assert set(prey_overlap.keys()) == {"adult", "juvenile"}
    # hand-computed: w_base=[2/3,1/3,1/2,1.0], w_arm=[1/3,0,1/6,1.0] (S_LOW=3, S_HIGH=6).
    # adult: p=[2/3,1/3], q=[1,0] over support {0,1} -> mass at prey cell1: 1/3 -> 0 = -1/3.
    # juvenile: p=[2/5,3/5], q=[0,1] over support {1,2} -> mass at prey cell1: 2/5 -> 0 = -2/5.
    assert prey_overlap["adult"]["stickleback"] == pytest.approx(-1.0 / 3.0)
    assert prey_overlap["juvenile"]["stickleback"] == pytest.approx(-2.0 / 5.0)
    assert prey_overlap["adult"]["stickleback"] != pytest.approx(
        prey_overlap["juvenile"]["stickleback"]
    )


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
