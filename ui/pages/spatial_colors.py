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
    values: np.ndarray,
    palette: str = "viridis",
    scale: str = "quantile",
    n_bins: int = N_BINS,
    *,
    edges: np.ndarray | None = None,
) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    """Return (bin index per value, RGBA colour per bin, edges).

    Pass ``edges`` (from :func:`bin_edges` over the WHOLE time series) when animating: edges
    recomputed per frame would give every frame the same colour spread, hiding change over time.
    Values outside the edges clamp to the end bins. The bin index is -1 for non-finite values.
    """
    if edges is None:
        edges = bin_edges(np.asarray(values, dtype=float), scale, n_bins)
    k = edges.size - 1
    colors = [[*c[:3], ALPHA] for c in color_range(k, PALETTES.get(palette, PALETTE_VIRIDIS))]
    vals = np.asarray(values, dtype=float)
    idx = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, k - 1)
    idx = np.where(np.isfinite(vals), idx, -1)
    return idx, colors, edges


def series_edges(slices, scale: str, n_bins: int = N_BINS, max_steps: int = 120) -> np.ndarray:
    """Bin edges over a time series of 2-D slices (zeros/non-finite excluded, as on the map).

    ``slices`` is a callable ``t -> 2-D array`` plus its length, given as ``(fn, n_time)``; at
    most ``max_steps`` evenly spaced steps are read, so long monthly runs stay cheap.
    """
    return bin_edges(series_values(slices, max_steps), scale, n_bins)


def series_values(slices, max_steps: int = 120) -> np.ndarray:
    """Drawn (finite, non-zero) values of up to ``max_steps`` evenly spaced frames."""
    fn, n_time = slices
    steps = np.unique(np.linspace(0, max(n_time - 1, 0), min(n_time, max_steps)).astype(int))
    parts = []
    for t in steps:
        a = np.asarray(fn(int(t)), dtype=float).ravel()
        parts.append(a[np.isfinite(a) & (a != 0)])
    return np.concatenate(parts) if parts else np.array([])


def percentile_range(values: np.ndarray, lo_pct: float, hi_pct: float) -> list[float]:
    """Value range for a percentile window on the raw (tie-preserving) distribution.

    Full window = no filter. True percentiles, NOT de-duplicated bin edges: on a constant or
    tied field de-duplicated edges shift percentile positions and hide every cell.
    """
    if (lo_pct <= 0 and hi_pct >= 100) or values.size == 0:
        return [-1e300, 1e300]
    lo, hi = np.percentile(values, [lo_pct, hi_pct])
    return [float(lo), float(hi)]


def legend_entries(layer_id: str, colors: list[list[int]], edges: np.ndarray) -> list[dict]:
    """One legend row per bin, labelled with the range it covers.

    The end bins are open-ended: values outside the (sampled) edges clamp into them, so a
    closed label would misstate their range.
    """
    k = len(colors)

    def label(i: int) -> str:
        if k == 1:
            return "all values"
        if i == 0:
            return f"< {edges[1]:.3g}"
        if i == k - 1:
            return f"\u2265 {edges[i]:.3g}"
        return f"{edges[i]:.3g} \u2013 {edges[i + 1]:.3g}"

    return [
        {"layer_id": layer_id, "label": label(i), "color": colors[i][:3], "shape": "rect"}
        for i in range(k)
    ]
