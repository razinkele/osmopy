"""Block-replicate an (nlat,nlon) OCCUPANCY array to (nlat*f,nlon*f). For movement
maps / masks / fishing-distribution maps ({-99,0,weight}) — NOT absolute biomass
(use osmose/forcing/conserve_regrid for those)."""

from __future__ import annotations
import numpy as np


def block_replicate(arr, factor: int):
    a = np.asarray(arr, dtype=np.float64)
    return np.repeat(np.repeat(a, factor, axis=0), factor, axis=1)
