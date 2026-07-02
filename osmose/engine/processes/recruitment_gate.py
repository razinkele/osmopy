"""Reproductive-volume recruitment gate — pure per-step factor helper.

Engine-state-free: reads only precomputed EngineConfig fields (see
osmose/engine/config.py:_load_rv_gate). Returns a per-species egg multiplier,
constant within a model year, 1.0 for species with the gate disabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from osmose.engine.config import EngineConfig


def rv_gate_factor(config: "EngineConfig", step: int) -> NDArray[np.float64]:
    """Per-species egg-production multiplier for this timestep.

    1.0 for every species when the gate is off or the species is disabled;
    otherwise the mode factor for the current model year's series index.
    """
    out = np.ones(config.n_species, dtype=np.float64)
    factor = config.rv_gate_factor_by_index
    if factor is None:
        return out
    n_years = factor.shape[0]
    year = step // config.n_dt_per_year
    idx = (config.rv_gate_offset + year) % n_years
    out[config.rv_gate_enabled] = factor[idx]
    return out
