"""Per-species deterministic RNG factory."""
from __future__ import annotations

import numpy as np


def build_rng(seed: int, n_species: int, fixed: bool) -> list[np.random.Generator]:
    """Create RNG instances for each species.

    When fixed=False: all species share a single Generator.
    When fixed=True: each species gets an independent Generator from SeedSequence.
    """
    if not fixed:
        shared = np.random.default_rng(seed)
        return [shared] * n_species
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(n_species)
    return [np.random.default_rng(child) for child in children]
