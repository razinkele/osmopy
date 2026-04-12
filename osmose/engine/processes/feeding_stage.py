"""Feeding stage computation for schools.

Each school's feeding stage is determined by comparing a metric value
(age, size, weight, or trophic level) against species-specific thresholds.
The stage index counts how many thresholds the value meets or exceeds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from osmose.engine.config import EngineConfig
    from osmose.engine.state import SchoolState


def compute_feeding_stages(
    state: SchoolState,
    config: EngineConfig,
) -> NDArray[np.int32]:
    """Compute the feeding stage for every school.

    Parameters
    ----------
    state:
        Current school state (species_id, age_dt, length, weight, trophic_level).
    config:
        Engine config with feeding_stage_thresholds, feeding_stage_metric,
        n_dt_per_year, and n_species.

    Returns
    -------
    NDArray[np.int32]
        Stage index per school (0-based).
    """
    n = len(state.species_id)
    stages = np.zeros(n, dtype=np.int32)

    if n == 0:
        return stages

    n_total = config.n_species + config.n_background
    thresholds = config.feeding_stage_thresholds
    metrics = config.feeding_stage_metric

    for sp_idx in range(n_total):
        sp_thresholds = thresholds[sp_idx]
        if not sp_thresholds:
            # No thresholds → single stage (0)
            continue

        mask = state.species_id == sp_idx
        if not np.any(mask):
            continue

        metric = metrics[sp_idx]
        if metric == "size":
            values = state.length[mask]
        elif metric == "age":
            values = state.age_dt[mask].astype(np.float64) / config.n_dt_per_year
        elif metric == "weight":
            values = state.weight[mask] * 1e6  # tonnes → grams
        elif metric == "tl":
            values = state.trophic_level[mask]
        else:
            raise ValueError(f"Unrecognized feeding stage metric: {metric!r}")

        # Count thresholds exceeded (>= comparison) using vectorized searchsorted
        sorted_thr = np.sort(sp_thresholds)
        stages[mask] = np.searchsorted(sorted_thr, values, side="right").astype(np.int32)

    return stages
