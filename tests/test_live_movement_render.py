"""Tests for ui.pages.live_movement_render (deck.gl layer builders)."""

from __future__ import annotations

import numpy as np

from osmose.live_movement import MovementSnapshot
from ui.pages.live_movement_render import (
    dots_layer_from_points,
    heatmap_layer_from_points,
    species_color,
)


def _snap(sp_id, lon, lat, biomass, species=("cod", "sprat"), lon_step=1.0, lat_step=1.0):
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
    assert layer["id"] == "live_movement"
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
    assert layer["id"] == "live_movement"
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
    assert h["id"] == "live_movement" and d["id"] == "live_movement"
