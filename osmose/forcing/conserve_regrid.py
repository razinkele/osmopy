"""Conservative regrid for ABSOLUTE-biomass forcing (tonnes/cell): split each
coarse cell's mass equally across its factor**2 sub-cells so the global total is
preserved. block_replicate here would inflate total system biomass factor**2x."""

from __future__ import annotations
import numpy as np


def split_conserve(field, factor: int):
    up = np.repeat(np.repeat(np.asarray(field, dtype=np.float64), factor, axis=-2), factor, axis=-1)
    return up / (factor * factor)
