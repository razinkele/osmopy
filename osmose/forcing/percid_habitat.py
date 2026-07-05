"""Real perch/pikeperch habitat as a BINARY thin-littoral set: a cell is habitat
iff a meaningful fraction (>= tau) of it is shallow littoral AND salinity permits.
Binary (not fractional) so the habitat SET/AREA genuinely shrinks and is read
unambiguously by the engine's map semantics. Salinity gate on spawning; relaxed
ceiling on adult/juvenile. Land -> -99."""

from __future__ import annotations
import numpy as np


def percid_stage_map(frac, ocean, salinity, tau, land_value=-99.0, sal_ceiling=None, sal_gate=None):
    frac = np.asarray(frac, float)
    ocean = np.asarray(ocean, bool)
    salinity = np.asarray(salinity, float)
    thr = sal_ceiling if sal_ceiling is not None else sal_gate
    sal_ok = np.isfinite(salinity) & (salinity < thr)  # NaN -> False (excluded)
    habitat = ocean & (frac >= tau) & sal_ok
    out = np.where(ocean, 0.0, land_value)
    out[habitat] = 1.0
    return out


def vacuity_ok(real_map, upsampled_percid_footprint, max_ratio=0.4):
    real = int(np.sum(np.asarray(real_map) == 1.0))
    up = int(np.sum(np.asarray(upsampled_percid_footprint) > 0))
    if real == 0 or up == 0:
        return False
    return real / up <= max_ratio
