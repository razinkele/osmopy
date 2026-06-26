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
):
    return types.SimpleNamespace(
        n_species=n_species,
        n_steps=n_steps,
        species_names=list(names),
        maturity_size=np.array(maturity_size, dtype=np.float64),
        maturity_age_dt=np.array(maturity_age_dt, dtype=np.int32),
    )


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
