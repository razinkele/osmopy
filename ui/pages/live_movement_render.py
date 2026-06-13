"""deck.gl layer builders for the live movement view.

A new module rather than an addition to ui/pages/grid_helpers.py: that file is
Plotly/config-page-scoped (~1254 LOC) and builds Plotly figures, not deck.gl layers. The
only in-repo deck.gl layer code is inline at ui/pages/spatial_results.py:570-598 — this
module follows that convention (positional id, camelCase props, "@@=d.field" accessors,
row-dict data).
"""

from __future__ import annotations

import numpy as np
from shiny_deckgl import (  # type: ignore[import-untyped]
    PALETTE_THERMAL,
    color_range,
    heatmap_layer,
    scatterplot_layer,
)

from osmose.live_movement import MovementSnapshot

_LAYER_ID = "live_movement"

# Deterministic categorical RGBA palette (NOT shiny_deckgl.SPECIES_COLORS — that is a
# 3-entry seal palette unusable for fish).
_SPECIES_PALETTE: list[list[int]] = [
    [31, 119, 180, 200],
    [255, 127, 14, 200],
    [44, 160, 44, 200],
    [214, 39, 40, 200],
    [148, 103, 189, 200],
    [140, 86, 75, 200],
    [227, 119, 194, 200],
    [127, 127, 127, 200],
    [188, 189, 34, 200],
    [23, 190, 207, 200],
]


def species_color(sp_id: int) -> list[int]:
    """Deterministic RGBA for a species index (cycles the palette)."""
    return list(_SPECIES_PALETTE[int(sp_id) % len(_SPECIES_PALETTE)])


def _filter_mask(snap: MovementSnapshot, species_filter: str | None) -> np.ndarray:
    if species_filter is None:
        return np.ones(snap.sp_id.size, dtype=bool)
    try:
        target = snap.species.index(species_filter)  # name -> sp_id (species in sp_id order)
    except ValueError:
        return np.zeros(snap.sp_id.size, dtype=bool)
    return snap.sp_id == target


def _points_to_rows(snap: MovementSnapshot, species_filter: str | None) -> list[dict]:
    """Base rows: position + weight + fill. Heatmap ignores fill (one builder, both modes)."""
    m = _filter_mask(snap, species_filter)
    sp_id, lon, lat, bm = snap.sp_id[m], snap.lon[m], snap.lat[m], snap.biomass[m]
    return [
        {"position": [float(lo), float(la)], "weight": float(b), "fill": species_color(s)}
        for s, lo, la, b in zip(sp_id, lon, lat, bm)
    ]


def heatmap_layer_from_points(snap: MovementSnapshot, species_filter: str | None) -> dict:
    """Native deck.gl HeatmapLayer weighted by biomass, from un-jittered cell centers."""
    return heatmap_layer(
        _LAYER_ID,
        data=_points_to_rows(snap, species_filter),
        getPosition="@@=d.position",
        getWeight="@@=d.weight",
        colorRange=color_range(palette=PALETTE_THERMAL),
    )


def dots_layer_from_points(snap: MovementSnapshot, species_filter: str | None) -> dict:
    """ScatterplotLayer: one dot per school, colored by species, biomass-sized.

    Deterministic per-school in-cell jitter (seeded by row index, no RNG) bounded to ±¼ of
    the grid cell spacing carried in the snapshot (``lon_step``/``lat_step``) — so
    overlapping schools in one cell spread out even when every school is in the same cell
    (a per-occupied-coord estimate would collapse to 0 there). ``*_step == 0`` (a 1-cell
    grid) → no jitter; ``radiusMinPixels`` still separates dots visually.
    """
    m = _filter_mask(snap, species_filter)
    sp_id, lon, lat, bm = snap.sp_id[m], snap.lon[m], snap.lat[m], snap.biomass[m]
    jx = snap.lon_step * 0.25
    jy = snap.lat_step * 0.25
    bmax = float(bm.max()) if bm.size and bm.max() > 0 else 1.0
    rows = []
    for i, (s, lo, la, b) in enumerate(zip(sp_id, lon, lat, bm)):
        # Deterministic offsets in [-1, 1] from the row index (no RNG, reproducible).
        ox = ((i * 2654435761) % 1000 / 500.0 - 1.0) * jx
        oy = ((i * 40503) % 1000 / 500.0 - 1.0) * jy
        rows.append(
            {
                "position": [float(lo) + ox, float(la) + oy],
                "fill": species_color(s),
                "radius": 3.0 + 12.0 * float(np.sqrt(max(b, 0.0) / bmax)),
            }
        )
    return scatterplot_layer(
        _LAYER_ID,
        data=rows,
        getPosition="@@=d.position",
        getFillColor="@@=d.fill",
        getRadius="@@=d.radius",
        radiusUnits="pixels",
        radiusMinPixels=2,
        pickable=True,
    )
