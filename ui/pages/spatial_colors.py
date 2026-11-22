"""Value -> colour binning for the Spatial Results deck.gl map.

Uses shiny_deckgl's palettes and ``color_range``; the bin edges are computed here (not inside
``color_bins``/``color_quantiles``) so the legend can label each bin with the exact range it
covers. Quantile bins keep a single hotspot cell from flattening the rest of the map into one
colour, which equal-width (linear) bins do on skewed biomass fields.
"""

from __future__ import annotations

import numpy as np
from shiny_deckgl import (  # type: ignore[import-untyped]
    PALETTE_CHLOROPHYLL,
    PALETTE_OCEAN,
    PALETTE_THERMAL,
    PALETTE_VIRIDIS,
    color_range,
)

PALETTES: dict[str, list[list[int]]] = {
    "viridis": PALETTE_VIRIDIS,
    "ocean": PALETTE_OCEAN,
    "chlorophyll": PALETTE_CHLOROPHYLL,
    "thermal": PALETTE_THERMAL,
}
PALETTE_LABELS = {
    "viridis": "Viridis",
    "ocean": "Ocean",
    "chlorophyll": "Chlorophyll",
    "thermal": "Thermal",
}
SCALE_LABELS = {"quantile": "Quantile bins", "linear": "Equal-width bins"}
N_BINS = 7
ALPHA = 190


def bin_edges(values: np.ndarray, scale: str, n_bins: int = N_BINS) -> np.ndarray:
    """Monotone, de-duplicated bin edges over the finite values (len >= 2)."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.array([0.0, 1.0])
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return np.array([lo, lo + 1.0])
    if scale == "quantile":
        edges = np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1))
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
    edges = np.unique(edges)  # ties in skewed data collapse quantile edges
    return edges if edges.size >= 2 else np.array([lo, hi])


def bin_colors(
    values: np.ndarray, palette: str = "viridis", scale: str = "quantile", n_bins: int = N_BINS
) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    """Return (bin index per value, RGBA colour per bin, edges).

    The bin index is -1 for non-finite values. The last edge is inclusive.
    """
    edges = bin_edges(np.asarray(values, dtype=float), scale, n_bins)
    k = edges.size - 1
    colors = [[*c[:3], ALPHA] for c in color_range(k, PALETTES.get(palette, PALETTE_VIRIDIS))]
    vals = np.asarray(values, dtype=float)
    idx = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, k - 1)
    idx = np.where(np.isfinite(vals), idx, -1)
    return idx, colors, edges


def legend_entries(layer_id: str, colors: list[list[int]], edges: np.ndarray) -> list[dict]:
    """One legend row per bin, labelled with the range it covers."""
    return [
        {
            "layer_id": layer_id,
            "label": f"{edges[i]:.3g} – {edges[i + 1]:.3g}",
            "color": colors[i][:3],
            "shape": "rect",
        }
        for i in range(len(colors))
    ]
