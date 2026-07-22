"""Reduce one OsmoseResults to a per-species UQ stat dict.

UQ-scoped: emits distinct ``{sp}_biomass_mean`` / ``{sp}_ssb_mean`` /
``{sp}_yield_mean`` keys (see ``keying.py``); does not touch
``losses.quantity_key`` so NSGA/DE scoring is unaffected. Values are
linear-scale trailing-window means — the natural-log and seed-mean transforms
happen later in ``design.py`` (Phase 1). SSB extraction is net-new relative to
``scripts/calibrate_baltic.py`` (which computes only mean/yield/cv/trend).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from osmose.results import OsmoseResults

# (getter-attribute-name, output-key-suffix) for each UQ output stat.
_UQ_OUTPUTS = (
    ("biomass", "_biomass_mean"),
    ("ssb", "_ssb_mean"),
    ("yield_biomass", "_yield_mean"),
)


def _read_frame(getter: Callable[[], pd.DataFrame]) -> pd.DataFrame | None:
    """Call a results getter, returning None if its output is absent/empty."""
    try:
        frame = getter()
    except Exception:  # noqa: BLE001 -- output not enabled/absent for this run: skip its stats
        return None
    if frame is None or frame.empty:
        return None
    return frame


def _trailing_mean(frame: pd.DataFrame | None, species: str, n_eval_years: int) -> float | None:
    """Mean of a species column over the last ``n_eval_years`` rows, or None."""
    if frame is None or species not in frame.columns:
        return None
    vals = frame[species].to_numpy(dtype=float)
    window = vals[-n_eval_years:] if len(vals) > n_eval_years else vals
    if window.size == 0:
        return None
    return float(np.mean(window))


def compute_uq_stats(
    results: OsmoseResults,
    species_names: Sequence[str],
    n_eval_years: int = 10,
) -> dict[str, float]:
    """Per-species linear-scale trailing-window means keyed for the UQ emulator.

    ``results`` must expose ``biomass()``, ``ssb()`` and ``yield_biomass()``,
    each returning a wide DataFrame (``Time`` + one column per species) or
    raising when that output is not enabled — the ``OsmoseResults`` contract,
    though only those three methods are used (any duck-typed object works, which
    is how the tests pass a stub). Outputs that are absent, empty, or lack a
    species column are silently skipped — their keys are omitted rather than
    raising, so a run without SSB/yield output still yields biomass stats.
    """
    frames = {name: _read_frame(getattr(results, name)) for name, _ in _UQ_OUTPUTS}
    stats: dict[str, float] = {}
    for species in species_names:
        for name, suffix in _UQ_OUTPUTS:
            mean = _trailing_mean(frames[name], species, n_eval_years)
            if mean is not None:
                stats[f"{species}{suffix}"] = mean
    return stats
