"""Tests for osmose.live_movement (snapshot + transport for the live movement view)."""

from __future__ import annotations

import queue
import types

import numpy as np

from osmose.engine.grid import Grid
from osmose.live_movement import (
    MovementSnapshot,
    build_snapshot,
    make_step_observer,
    resolve_grid_latlon,
)


def _state(species_id, cell_x, cell_y, biomass, is_out=None, length=None, age_dt=None, is_egg=None):
    """A lightweight stand-in exposing only the fields build_snapshot reads."""
    n = len(species_id)
    return types.SimpleNamespace(
        species_id=np.array(species_id, dtype=np.int32),
        cell_x=np.array(cell_x, dtype=np.int32),
        cell_y=np.array(cell_y, dtype=np.int32),
        biomass=np.array(biomass, dtype=np.float64),
        is_out=np.array(is_out if is_out is not None else [False] * n, dtype=bool),
        length=np.array(length if length is not None else [10.0] * n, dtype=np.float64),
        age_dt=np.array(age_dt if age_dt is not None else [12] * n, dtype=np.int32),
        is_egg=np.array(is_egg if is_egg is not None else [False] * n, dtype=bool),
    )


def _config(
    n_species=2,
    n_steps=12,
    names=("cod", "sprat"),
    maturity_size=(5.0, 5.0),
    maturity_age_dt=(6, 6),
    n_dt_per_year=12,
):
    return types.SimpleNamespace(
        n_species=n_species,
        n_steps=n_steps,
        species_names=list(names),
        maturity_size=np.array(maturity_size, dtype=np.float64),
        maturity_age_dt=np.array(maturity_age_dt, dtype=np.int32),
        n_dt_per_year=n_dt_per_year,
    )


class _StubMap:
    def __init__(self, grid):
        self._grid = grid

    def get_map(self, age_dt, step):
        return self._grid


def test_resolve_grid_latlon_uses_grid_arrays_when_present():
    g = Grid.from_dimensions(ny=3, nx=4)
    g.lat = np.array([54.0, 54.5, 55.0])
    g.lon = np.array([10.0, 10.5, 11.0, 11.5])
    lat, lon = resolve_grid_latlon(g)
    np.testing.assert_allclose(lat, [54.0, 54.5, 55.0])
    np.testing.assert_allclose(lon, [10.0, 10.5, 11.0, 11.5])
    assert lat.dtype == np.float64 and lon.dtype == np.float64


def test_resolve_grid_latlon_falls_back_to_float_indices():
    g = Grid.from_dimensions(ny=3, nx=4)  # lat/lon are None
    lat, lon = resolve_grid_latlon(g)
    np.testing.assert_array_equal(lat, np.arange(3, dtype=np.float64))
    np.testing.assert_array_equal(lon, np.arange(4, dtype=np.float64))
    assert lat.dtype == np.float64 and lon.dtype == np.float64


def test_build_snapshot_maps_cells_to_lonlat_and_carries_biomass():
    g = Grid.from_dimensions(ny=3, nx=4)
    g.lat = np.array([54.0, 54.5, 55.0])
    g.lon = np.array([10.0, 10.5, 11.0, 11.5])
    # two focal schools: cod at (cy=1,cx=2), sprat at (cy=0,cx=0)
    st = _state(species_id=[0, 1], cell_x=[2, 0], cell_y=[1, 0], biomass=[5.0, 3.0])
    snap = build_snapshot(0, st, g, _config())
    assert isinstance(snap, MovementSnapshot)
    assert snap.step == 0 and snap.n_steps == 12 and snap.status == "running"
    assert snap.species == ["cod", "sprat"]
    np.testing.assert_allclose(snap.lon, [11.0, 10.0])
    np.testing.assert_allclose(snap.lat, [54.5, 54.0])
    np.testing.assert_allclose(snap.biomass, [5.0, 3.0])
    np.testing.assert_array_equal(snap.sp_id, [0, 1])
    assert (snap.lon_min, snap.lon_max) == (10.0, 11.5)
    assert (snap.lat_min, snap.lat_max) == (54.0, 55.0)
    assert snap.lon_step == 0.5 and snap.lat_step == 0.5  # grid cell spacing
    assert snap.truncated is False and snap.n_total == 2


def test_build_snapshot_excludes_unlocated_eggs_background_dead_and_out():
    g = Grid.from_dimensions(ny=2, nx=2)
    st = _state(
        species_id=[0, 0, 0, 2, 0],  # idx3 sp_id=2 is background (>= n_species=2)
        cell_x=[0, -1, 1, 0, 1],  # idx1 cell_x=-1 (unlocated egg)
        cell_y=[0, 0, 1, 0, 1],
        biomass=[5.0, 9.0, 0.0, 9.0, 7.0],  # idx2 biomass=0 (dead)
        is_out=[False, False, False, False, True],  # idx4 out of domain
    )
    snap = build_snapshot(3, st, g, _config())
    # only idx0 survives the focal+located+living+in-domain mask
    np.testing.assert_array_equal(snap.sp_id, [0])
    np.testing.assert_allclose(snap.lon, [0.0])
    np.testing.assert_allclose(snap.lat, [0.0])
    assert snap.n_total == 1


def test_build_snapshot_dot_cap_samples_deterministically():
    g = Grid.from_dimensions(ny=10, nx=10)
    n = 100
    st = _state(
        species_id=[0] * n,
        cell_x=[i % 10 for i in range(n)],
        cell_y=[i // 10 for i in range(n)],
        biomass=[1.0] * n,
    )
    snap = build_snapshot(0, st, g, _config(), dot_cap=10)
    assert snap.truncated is True
    assert snap.n_total == 100
    assert snap.sp_id.size == 10
    # deterministic: same inputs → same sample
    snap2 = build_snapshot(0, st, g, _config(), dot_cap=10)
    np.testing.assert_array_equal(snap.lon, snap2.lon)


def test_build_snapshot_empty_selection_is_safe():
    g = Grid.from_dimensions(ny=2, nx=2)
    st = _state(species_id=[0], cell_x=[0], cell_y=[0], biomass=[0.0])  # dead
    snap = build_snapshot(0, st, g, _config())
    assert snap.sp_id.size == 0 and snap.lon.size == 0 and snap.n_total == 0


def test_build_snapshot_status_passthrough():
    g = Grid.from_dimensions(ny=2, nx=2)
    st = _state(species_id=[0], cell_x=[0], cell_y=[0], biomass=[1.0])
    snap = build_snapshot(11, st, g, _config(), status="done")
    assert snap.status == "done"


def test_make_step_observer_emits_first_and_final_and_throttles():
    g = Grid.from_dimensions(ny=2, nx=2)
    st = _state(species_id=[0], cell_x=[0], cell_y=[0], biomass=[1.0])
    cfg = _config(n_steps=5)
    clock = {"t": 0.0}
    obs = make_step_observer(q := queue.Queue(maxsize=10), throttle_s=1.0, now=lambda: clock["t"])
    obs(0, st, g, cfg)  # step 0 always emits
    clock["t"] = 0.1
    obs(1, st, g, cfg)  # throttled out (< 1.0s since last)
    clock["t"] = 2.0
    obs(2, st, g, cfg)  # emits (>= 1.0s)
    clock["t"] = 2.1
    obs(4, st, g, cfg)  # final step always emits
    steps = []
    while not q.empty():
        steps.append(q.get_nowait().step)
    assert steps == [0, 2, 4]


def test_make_step_observer_drops_oldest_keeps_newest_when_full():
    g = Grid.from_dimensions(ny=2, nx=2)
    st = _state(species_id=[0], cell_x=[0], cell_y=[0], biomass=[1.0])
    cfg = _config(n_steps=100)
    q = queue.Queue(maxsize=1)
    obs = make_step_observer(q, throttle_s=0.0, now=lambda: 0.0)
    obs(0, st, g, cfg)
    obs(50, st, g, cfg)  # queue was full → drop oldest, keep newest
    assert q.qsize() == 1
    assert q.get_nowait().step == 50


def test_make_step_observer_swallows_build_errors():
    cfg = _config()
    bad_grid = object()  # build_snapshot will raise on this
    q = queue.Queue(maxsize=2)
    obs = make_step_observer(q, throttle_s=0.0, now=lambda: 0.0)
    obs(0, object(), bad_grid, cfg)  # must not raise
    assert q.empty()


def test_build_snapshot_assigns_life_stage():
    from osmose.live_movement import build_snapshot, STAGE_LABELS

    g = Grid.from_dimensions(ny=3, nx=3)
    # 3 cod schools: egg (is_egg), juvenile (small/young, immature), adult (mature)
    st = _state(
        species_id=[0, 0, 0],
        cell_x=[0, 1, 2],
        cell_y=[0, 1, 2],
        biomass=[1.0, 1.0, 1.0],
        length=[0.1, 2.0, 50.0],
        age_dt=[0, 2, 30],
        is_egg=[True, False, False],
    )
    snap = build_snapshot(0, st, g, _config(maturity_size=(5.0, 5.0), maturity_age_dt=(6, 6)))
    assert list(snap.stage) == [0, 1, 2]  # egg/larva, juvenile, adult
    assert STAGE_LABELS == {0: "Egg/larva", 1: "Juvenile", 2: "Adult"}


def test_build_snapshot_stage_sliced_under_dot_cap():
    from osmose.live_movement import build_snapshot

    g = Grid.from_dimensions(ny=3, nx=3)
    n = 30
    st = _state(
        species_id=[0] * n,
        cell_x=[1] * n,
        cell_y=[1] * n,
        biomass=[1.0] * n,
        length=[50.0] * n,
        age_dt=[30] * n,
        is_egg=[False] * n,
    )
    snap = build_snapshot(0, st, g, _config(), dot_cap=10)
    assert snap.stage.size == snap.sp_id.size == 10  # sampled in lockstep
    assert set(snap.stage.tolist()) == {2}  # all adult


def test_unlocated_egg_placed_on_spawning_map():
    from osmose.live_movement import build_snapshot

    g = Grid.from_dimensions(ny=3, nx=3)
    # one located adult (idx0) + one unlocated egg of sp0 (idx1, cell=-1, is_egg)
    st = _state(
        species_id=[0, 0],
        cell_x=[1, -1],
        cell_y=[1, -1],
        biomass=[5.0, 0.1],
        is_egg=[False, True],
        length=[50.0, 0.1],
        age_dt=[30, 0],
    )
    m = np.zeros((3, 3), dtype=np.float64)
    m[2, 0] = 1.0  # only (row=2, col=0) is a spawning cell
    snap = build_snapshot(0, st, g, _config(n_species=1), map_sets={0: _StubMap(m)})
    # the egg is now in the snapshot at the spawning cell with stage 0
    assert 0 in snap.stage.tolist()  # egg/larva present
    egg_i = snap.stage.tolist().index(0)
    lat_arr, lon_arr = (np.arange(3.0), np.arange(3.0))
    assert snap.lon[egg_i] == lon_arr[0] and snap.lat[egg_i] == lat_arr[2]  # placed at (col0,row2)
    # within-build deterministic
    snap2 = build_snapshot(0, st, g, _config(n_species=1), map_sets={0: _StubMap(m)})
    assert snap2.lon.tolist() == snap.lon.tolist()


def test_egg_placement_probability_weighted():
    from osmose.live_movement import build_snapshot

    g = Grid.from_dimensions(ny=1, nx=3)
    n = 60
    st = _state(
        species_id=[0] * n,
        cell_x=[-1] * n,
        cell_y=[-1] * n,
        biomass=[0.1] * n,
        is_egg=[True] * n,
        length=[0.1] * n,
        age_dt=[0] * n,
    )
    m = np.array([[0.01, 0.01, 5.0]], dtype=np.float64)  # cell (0,2) dominates
    snap = build_snapshot(0, st, g, _config(n_species=1), map_sets={0: _StubMap(m)})
    # most eggs land on the high-proba cell (col=2)
    from collections import Counter

    modal_col = Counter(snap.lon.tolist()).most_common(1)[0][0]
    assert modal_col == 2.0


def test_egg_random_fallback_no_map():
    from osmose.live_movement import build_snapshot

    g = Grid.from_dimensions(ny=2, nx=2)
    g.ocean_mask = np.ones((2, 2), dtype=bool)
    st = _state(
        species_id=[0],
        cell_x=[-1],
        cell_y=[-1],
        biomass=[0.1],
        is_egg=[True],
        length=[0.1],
        age_dt=[0],
    )
    snap = build_snapshot(0, st, g, _config(n_species=1), map_sets=None)
    assert snap.stage.tolist() == [0]  # placed on an ocean cell, stage 0


def test_no_unlocated_eggs_off_season():
    from osmose.live_movement import build_snapshot

    g = Grid.from_dimensions(ny=2, nx=2)
    # only a located adult, no unlocated eggs -> nothing to place, no stage-0
    st = _state(
        species_id=[0],
        cell_x=[0],
        cell_y=[0],
        biomass=[5.0],
        is_egg=[False],
        length=[50.0],
        age_dt=[30],
    )
    snap = build_snapshot(0, st, g, _config(n_species=1), map_sets={0: _StubMap(np.ones((2, 2)))})
    assert 0 not in snap.stage.tolist()


def test_date_label_one_based_year():
    from osmose.live_movement import build_snapshot

    g = Grid.from_dimensions(ny=2, nx=2)
    st = _state(
        species_id=[0],
        cell_x=[0],
        cell_y=[0],
        biomass=[5.0],
        is_egg=[False],
        length=[50.0],
        age_dt=[30],
    )
    cfg = _config(n_species=1, n_dt_per_year=24)
    assert build_snapshot(0, st, g, cfg).date_label == "Y1 · 01 Jan"
    assert build_snapshot(24, st, g, cfg).date_label.startswith("Y2 · 01 Jan")
    assert build_snapshot(12, st, g, cfg).date_label.startswith("Y1 · ")  # mid-year
