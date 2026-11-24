"""Java's output distribution classes (``OutputDistribution``), for the outputs that follow it.

Thresholds are ``min + i * incr`` for ``i in 0 .. ceil((max - min) / incr)``, computed in float32
exactly as Java's ``float`` arithmetic does; a value's class is the number of thresholds at or below
it, so there are ``len(thresholds) + 1`` classes and class 0 is "below min". ``min == 0`` is
corrected to ``incr`` (Java warns and does the same). The three keys
``output.distrib.by{kind}.{min,max,incr}`` are used only when all three are set (Java
``distribExist``); otherwise Java's defaults apply.

The engine's older by-age/by-size distributions (``biomass_by_age``/``biomass_by_size``) predate
this module and bin differently (``len(edges)`` classes, below-min folded into class 0); they are
left as they are here (a separate parity question). The families added on 2026-11-24 --
biomass by TL, mean size by age, mean TL by size -- use Java's rule.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Java OutputDistribution defaults (AGE_DEFAULT_MIN ... TL_DEFAULT_INCR)
DEFAULTS: dict[str, tuple[float, float, float]] = {
    "age": (1.0, 25.0, 1.0),
    "size": (10.0, 200.0, 10.0),
    "tl": (1.0, 6.0, 0.1),
}


#: Upper bound on thresholds per distribution (a config error beyond it, not an allocation).
MAX_CLASSES = 10_000


def java_thresholds(cfg: dict[str, str], kind: str) -> NDArray[np.float32]:
    keys = [f"output.distrib.by{kind}.{p}" for p in ("min", "max", "incr")]
    if all(str(cfg.get(k, "")).strip() for k in keys):
        mn, mx, incr = (np.float32(float(cfg[k])) for k in keys)
    else:
        mn, mx, incr = (np.float32(v) for v in DEFAULTS[kind])
    if not all(np.isfinite(v) for v in (mn, mx, incr)) or incr <= 0 or mn < 0 or mx < mn:
        raise ValueError(
            f"output.distrib.by{kind}: need finite min >= 0, max >= min and incr > 0 "
            f"(got {mn}, {mx}, {incr})"
        )
    if mn == 0:
        mn = incr
    n = int(np.ceil(np.float32(max(mx - mn, 0)) / incr)) + 1
    if n > MAX_CLASSES:
        raise ValueError(f"output.distrib.by{kind}: {n} classes, more than {MAX_CLASSES}")
    return (mn + np.arange(n, dtype=np.float32) * incr).astype(np.float32)


def java_classes(values: NDArray[np.float64], thresholds: NDArray[np.float32]) -> NDArray[np.intp]:
    """Class index per value: the count of thresholds <= value (Java getClass), 0..len."""
    return np.searchsorted(
        thresholds.astype(np.float64), np.asarray(values, dtype=np.float64), side="right"
    )


def class_labels(thresholds: NDArray[np.float32]) -> list[str]:
    """Lower bound of each class, as Java writes the first column (class 0 -> 0)."""
    return ["0"] + [f"{float(t):g}" for t in thresholds]
