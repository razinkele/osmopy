"""Tests for ui.pages.live_movement_render (deck.gl layer builders)."""

from __future__ import annotations

import numpy as np

from osmose.live_movement import MovementSnapshot
from ui.pages.live_movement_render import (
    choose_live_layer,
    dots_layer_from_points,
    heatmap_layer_from_points,
    species_color,
)


def _snap(
    sp_id, lon, lat, biomass, species=("cod", "sprat"), lon_step=1.0, lat_step=1.0, stage=None
):
    lo, la = list(lon), list(lat)
    return MovementSnapshot(
        step=0,
        n_steps=12,
        status="running",
        species=list(species),
        sp_id=np.array(sp_id, dtype=np.int32),
        lon=np.array(lon, dtype=np.float64),
        lat=np.array(lat, dtype=np.float64),
        biomass=np.array(biomass, dtype=np.float64),
        stage=np.array(stage if stage is not None else [1] * len(sp_id), dtype=np.int8),
        truncated=False,
        n_total=len(sp_id),
        lon_min=float(min(lo)) if lo else 0.0,
        lon_max=float(max(lo)) if lo else 0.0,
        lat_min=float(min(la)) if la else 0.0,
        lat_max=float(max(la)) if la else 0.0,
        lon_step=lon_step,
        lat_step=lat_step,
    )


def test_species_color_distinct_and_deterministic():
    c0, c1 = species_color(0), species_color(1)
    assert c0 != c1
    assert species_color(0) == c0  # deterministic
    assert len(c0) == 4 and all(0 <= v <= 255 for v in c0)


def test_heatmap_layer_structure():
    snap = _snap([0, 1], [10.0, 11.0], [54.0, 55.0], [5.0, 3.0])
    layer = heatmap_layer_from_points(snap, None)
    assert layer["id"] == "live_movement_heatmap"
    assert layer["getPosition"] == "@@=d.position"
    assert layer["getWeight"] == "@@=d.weight"
    assert len(layer["data"]) == 2
    assert layer["data"][0]["position"] == [10.0, 54.0]
    assert layer["data"][0]["weight"] == 5.0
    assert isinstance(layer["colorRange"], list) and len(layer["colorRange"]) >= 2


def test_species_filter_reduces_rows():
    snap = _snap([0, 1, 0], [10.0, 11.0, 12.0], [54.0, 55.0, 56.0], [1.0, 2.0, 3.0])
    layer = heatmap_layer_from_points(snap, "cod")  # sp_id 0
    assert len(layer["data"]) == 2  # only the two cod rows


def test_dots_layer_structure_and_jitter_bounded_deterministic():
    snap = _snap([0, 0], [10.0, 10.0], [54.0, 54.0], [4.0, 9.0])  # same cell
    layer = dots_layer_from_points(snap, None)
    assert layer["id"] == "live_movement_dots"
    assert layer["getFillColor"] == "@@=d.fill"
    assert layer["getRadius"] == "@@=d.radius"
    assert layer["pickable"] is True
    rows = layer["data"]
    assert len(rows) == 2
    # two schools in one cell get distinct jittered positions (deterministic)
    assert rows[0]["position"] != rows[1]["position"]
    layer2 = dots_layer_from_points(snap, None)
    assert [r["position"] for r in layer2["data"]] == [r["position"] for r in rows]
    # fill colored by species
    assert rows[0]["fill"] == list(species_color(0))


def test_empty_snapshot_yields_empty_layer():
    snap = _snap([], [], [], [])
    h = heatmap_layer_from_points(snap, None)
    d = dots_layer_from_points(snap, None)
    assert h["data"] == [] and d["data"] == []
    # deck.gl reconciles layers by id; a HeatmapLayer and a ScatterplotLayer sharing one id
    # crashes on the class swap (shaderInputs undefined) -> blank map. They MUST differ.
    assert h["id"] == "live_movement_heatmap" and d["id"] == "live_movement_dots"
    assert h["id"] != d["id"]


def _choose_snap(n, n_species=8):
    rng = np.random.default_rng(0)
    sp = rng.integers(0, n_species, n).astype(np.int32)
    sp[: n // 2] = 0  # half are species 0 ("cod")
    return MovementSnapshot(
        step=1,
        n_steps=10,
        status="running",
        species=[
            "cod",
            "herring",
            "sprat",
            "flounder",
            "perch",
            "pikeperch",
            "smelt",
            "stickleback",
        ][:n_species],
        sp_id=sp,
        lon=rng.uniform(10, 30, n).astype(np.float64),
        lat=rng.uniform(54, 66, n).astype(np.float64),
        biomass=rng.uniform(1e-3, 1e3, n).astype(np.float64),
        stage=np.ones(n, dtype=np.int8),  # default all juvenile
        truncated=False,
        n_total=n,
        lon_min=10.0,
        lon_max=30.0,
        lat_min=54.0,
        lat_max=66.0,
        lon_step=0.4,
        lat_step=0.3,
    )


def test_dots_below_threshold_renders_dots():
    layer, note = choose_live_layer(_choose_snap(200), None, "dots", dots_max=1500)
    assert layer["type"] == "ScatterplotLayer"
    assert note is None


def test_dots_above_threshold_falls_back_to_heatmap():
    # 2000 points, all species (filter None) -> > 1500 -> heatmap + note
    layer, note = choose_live_layer(_choose_snap(2000), None, "dots", dots_max=1500)
    assert layer["type"] == "HeatmapLayer"
    assert note is not None and "heatmap" in note.lower()


def test_heatmap_mode_always_heatmap():
    layer, note = choose_live_layer(_choose_snap(3000), None, "heatmap", dots_max=1500)
    assert layer["type"] == "HeatmapLayer"
    assert note is None


def test_filter_reduces_count_so_dots_stays_dots():
    # 2400 pts but only ~half are cod (~1200 < 1500) -> dots kept for the cod filter
    layer, note = choose_live_layer(_choose_snap(2400), "cod", "dots", dots_max=1500)
    assert layer["type"] == "ScatterplotLayer"
    assert note is None


def test_dot_cap_default_is_2000():
    # The LIVE path's cap is make_step_observer's default, which is what reaches
    # build_snapshot at runtime (it passes dot_cap through). Assert THAT default.
    import inspect

    from osmose.live_movement import make_step_observer

    assert inspect.signature(make_step_observer).parameters["dot_cap"].default == 2000


def test_filter_mask_stage_and_species_compose():
    from ui.pages.live_movement_render import _filter_mask

    snap = _snap(
        sp_id=[0, 0, 1], lon=[0, 1, 2], lat=[0, 1, 2], biomass=[1, 1, 1], stage=[1, 2, 2]
    )  # cod-juv, cod-adult, sprat-adult
    # species cod + stage adult -> only the 2nd school
    m = _filter_mask(snap, "cod", 2)
    assert list(m) == [False, True, False]
    # stage only (adult) -> schools 2 and 3
    assert list(_filter_mask(snap, None, 2)) == [False, True, True]
    # no filters -> all
    assert list(_filter_mask(snap, None, None)) == [True, True, True]


def test_choose_live_layer_fallback_uses_composed_count():
    from ui.pages.live_movement_render import choose_live_layer

    # 5 cod schools, only 1 adult; dots_max=2 -> species-only count (5) would fall back to
    # heatmap, but the composed (stage=adult) count is 1 -> must STAY in dots.
    snap = _snap(
        sp_id=[0, 0, 0, 0, 0],
        lon=[0, 1, 2, 3, 4],
        lat=[0, 1, 2, 3, 4],
        biomass=[1, 1, 1, 1, 1],
        stage=[1, 1, 1, 1, 2],
    )
    layer, note = choose_live_layer(snap, "cod", "dots", dots_max=2, stage_filter=2)
    assert note is None  # stayed in dots (composed count = 1 <= 2)
    assert layer["type"] == "ScatterplotLayer"
